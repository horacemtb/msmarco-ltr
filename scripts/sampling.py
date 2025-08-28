import json
import random
from typing import Dict, Set, List, Tuple, Optional
import pandas as pd
from tqdm import tqdm


def sample_global_random_doc(searcher, exclude: Set[str], max_tries: int = 100, seed: int = 79) -> Optional[str]:
    """
    Get an easy negative - random doc from the corpus excluding the given set
    """
    ndocs = searcher.num_docs
    for i in range(max_tries):
        local_seed = seed + i
        local_random = random.Random(local_seed)
        internal = local_random.randint(0, ndocs - 1)
        d = searcher.doc(internal)
        if d is None:
            continue
        docid = d.docid()
        if docid not in exclude:
            return docid
    return None


def rank_map(run: List[Tuple[str, float]]) -> Dict[str, int]:
    """"
    Look up rank (bm25 or impact) that will be used as a feature for a particular query-doc pair
    """
    return {docid: i+1 for i, (docid, _) in enumerate(run)}  # rank starting from 1


def fetch_texts_bm25(searcher_bm25, docids: List[str]) -> Dict[str, str]:
    """
    Extract document text by docid
    Returns: 
        {docid: text}
    """
    out = {}
    for docid in docids:
        try:
            doc = searcher_bm25.doc(docid)
            if not doc:
                out[docid] = ""
                continue
            data = json.loads(doc.raw())
            out[docid] = data.get("contents", "") or ""
        except:
            out[docid] = ""
    return out


def sample_negatives_for_query(
    qid: str,
    positives: Set[str],
    imp_run: List[Tuple[str, float]],
    bm25_run: List[Tuple[str, float]],
    bm25_searcher_for_random = None,
    negative_types: Dict[str, int] = {"imp_hard": 1, "imp_med": 1, "bm25_hard": 1, "easy": 1},
    seed: int = 79) -> Dict[str, List[str]]:
    """
    Sample negatives:
        hard negative: one random doc from top-10 retrieved candidates by Impact Searcher
        hard negative: one random doc from top-10 retrieved candidates by BM25 Searcher
        medium negative: one random doc from top-11-100 retrieved candidates by Impact Searcher
        easy negative: one random doc from the rest of the corpus
    """
    query_seed = seed + int(qid)
    local_random = random.Random(query_seed)

    negs = {k: [] for k in negative_types.keys()}

    # Impact Searcher: hard and medium negatives
    imp_nonpos = [d for d, _ in imp_run if d not in positives]
    hard_bucket = imp_nonpos[:10]
    med_bucket = imp_nonpos[10:100]
    if negative_types.get("imp_hard", 0) and hard_bucket:
        negs["imp_hard"] = local_random.sample(hard_bucket, min(negative_types["imp_hard"], len(hard_bucket)))
    if negative_types.get("imp_med", 0) and med_bucket:
        negs["imp_med"] = local_random.sample(med_bucket, min(negative_types["imp_med"], len(med_bucket)))

    # BM25 Searcher: hard negative
    bm_nonpos = [d for d, _ in bm25_run if d not in positives and d not in negs["imp_hard"] and d not in negs["imp_med"]]
    bm_hard = bm_nonpos[:10]
    if negative_types.get("bm25_hard", 0) and bm_hard:
        negs["bm25_hard"] = local_random.sample(bm_hard, min(negative_types["bm25_hard"], len(bm_hard)))

    # easy negative
    if negative_types.get("easy", 0) and bm25_searcher_for_random is not None:
        exclude = set(positives) | set(imp_nonpos) | set(bm_nonpos)
        for i in range(negative_types["easy"]):
            easy_seed = query_seed+i
            res = sample_global_random_doc(bm25_searcher_for_random, exclude, seed=easy_seed)
            if res is None:
                break
            negs["easy"].append(res)
            exclude.add(res)
    return negs


def build_pairs_df(
    qid_to_query,
    qrels: Dict[str, Set[str]],
    runs_imp: Dict[str, List[Tuple[str, float]]],
    runs_bm25: Dict[str, List[Tuple[str, float]]],
    bm25_searcher,
    negative_types = {"imp_hard": 1, "imp_med": 1, "bm25_hard": 1, "easy": 1},
    fetch_texts: bool = True) -> pd.DataFrame:
    """
    Generate a dataset in the form of a pandas dataframe that contains:
        qid, pid, label (1 for any ground truth passages, 0 for sampled negatives)
        query_text, passage_text
        imp_rank, imp_score, bm25_rank, bm25_score
        passage type (pos, imp_hard, imp_med, bm25_hard, easy)
    """
    rows = []
    all_docids_needed = set()

    for qid, qtext in tqdm(qid_to_query.items(), desc="Sampling negs", unit="q"):
        if qid not in qrels or not qrels[qid]:
            continue
        pos = qrels[qid]

        imp_run = runs_imp.get(qid, [])
        bm_run = runs_bm25.get(qid, [])

        # ranks for features
        imp_ranks = rank_map(imp_run)
        bm_ranks = rank_map(bm_run)

        # positives (label = 1)
        for pid in pos:
            rows.append({
                "qid": qid, "pid": pid, "label": 1, "pas_type": "pos",
                "imp_rank": imp_ranks.get(pid, None),
                "imp_score": dict(imp_run).get(pid, None) if imp_run else None,
                "bm25_rank": bm_ranks.get(pid, None),
                "bm25_score": dict(bm_run).get(pid, None) if bm_run else None,
                "query_text": qtext,
                "passage_text": None
            })
            all_docids_needed.add(pid)

        # negatives (label = 0)
        negs = sample_negatives_for_query(qid, pos, imp_run, bm_run, bm25_searcher_for_random=bm25_searcher, negative_types=negative_types)
        for bucket, docids in negs.items():
            for pid in docids:
                rows.append({
                    "qid": qid, "pid": pid, "label": 0, "pas_type": bucket,
                    "imp_rank": imp_ranks.get(pid, None),
                    "imp_score": dict(imp_run).get(pid, None) if imp_run else None,
                    "bm25_rank": bm_ranks.get(pid, None),
                    "bm25_score": dict(bm_run).get(pid, None) if bm_run else None,
                    "query_text": qtext,
                    "passage_text": None
                })
                all_docids_needed.add(pid)

    df = pd.DataFrame(rows)

    # fetch passage texts
    if fetch_texts and not df.empty:
        texts = fetch_texts_bm25(bm25_searcher, list(all_docids_needed))
        df["passage_text"] = df["pid"].map(lambda p: texts.get(p, ""))

    # check types
    for c in ["imp_rank", "bm25_rank"]:
        if c in df.columns:
            df[c] = df[c].astype("float").where(df[c].notna(), None)

    return df


def build_test_pairs_df(
    qid_to_query,
    qrels: Dict[str, Set[str]],
    runs_imp: Dict[str, List[Tuple[str, float]]],
    runs_bm25: Dict[str, List[Tuple[str, float]]],
    bm25_searcher,
    fetch_texts: bool = True) -> pd.DataFrame:
    """
    Generate a dataset in the form of a pandas dataframe that contains:
        qid, pid, label (1 for any ground truth passages, 0 for negatives)
        query_text, passage_text
        imp_rank, imp_score, bm25_rank, bm25_score
    """
    rows = []
    all_docids_needed = set()

    for qid, qtext in tqdm(qid_to_query.items(), desc="Constructing dataset", unit="q"):
        if qid not in qrels or not qrels[qid]:
            continue

        pos_set = qrels[qid]

        imp_run = runs_imp.get(qid, [])
        bm_run = runs_bm25.get(qid, [])

        # ranks for features
        imp_ranks = rank_map(imp_run)
        bm_ranks = rank_map(bm_run)

        for pid, _ in imp_run:
            rows.append({
                "qid": qid, "pid": pid, "label": int(pid in pos_set),
                "imp_rank": imp_ranks.get(pid, None),
                "imp_score": dict(imp_run).get(pid, None) if imp_run else None,
                "bm25_rank": bm_ranks.get(pid, None),
                "bm25_score": dict(bm_run).get(pid, None) if bm_run else None,
                "query_text": qtext,
                "passage_text": None
            })

            all_docids_needed.add(pid)

    df = pd.DataFrame(rows)

    # fetch passage texts
    if fetch_texts and not df.empty:
        texts = fetch_texts_bm25(bm25_searcher, list(all_docids_needed))
        df["passage_text"] = df["pid"].map(lambda p: texts.get(p, ""))

    # check types
    for c in ["imp_rank", "bm25_rank"]:
        if c in df.columns:
            df[c] = df[c].astype("float").where(df[c].notna(), None)

    return df
