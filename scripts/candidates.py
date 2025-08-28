from typing import List, Dict, Tuple, Set, Iterable
from collections import defaultdict
from tqdm import tqdm


def _chunks(seq: List[str], size: int) -> Iterable[List[str]]:
    for i in range(0, len(seq), size):
        yield seq[i:i+size]


def retrieve_topk(searcher,
                  qid_to_query: Dict[str, str],
                  k: int = 100,
                  threads: int = 8,
                  batch_size: int = 1000,
                  desc: str = "Retrieving") -> Dict[str, List[Tuple[str, float]]]:
    """Returns: runs[qid] = [(docid, score), ...] length <= k."""
    qids = [q for q in qid_to_query.keys()]
    texts = [qid_to_query[q] for q in qids]
    runs: Dict[str, List[Tuple[str, float]]] = {}

    # batch_search if available
    if hasattr(searcher, "batch_search"):
        pbar = tqdm(total=len(qids), desc=desc, unit="q")

        for idx_chunk in _chunks(list(range(len(qids))), batch_size):
            sub_qids = [qids[i] for i in idx_chunk]
            sub_texts = [texts[i] for i in idx_chunk]
            results = searcher.batch_search(sub_texts, sub_qids, k, threads)
            for qid, hits in results.items():
                runs[qid] = [(h.docid, float(h.score)) for h in hits]
            pbar.update(len(sub_qids))
        pbar.close()
        return runs

    # per-query search
    for qid, qtext in tqdm(qid_to_query.items()):
        hits = searcher.search(qtext, k=k)
        runs[qid] = [(h.docid, float(h.score)) for h in hits]
    return runs


def rrf_fuse_runs(
    runs_list: List[Dict[str, List[Tuple[str, float]]]],
    K: int = 100,  # how many to keep per query
    k: int = 60,   # RRF constant; according to Microsoft, "experimentally observed to perform best if it's set to a small value like 60".
) -> Dict[str, List[Tuple[str, float]]]:

    fused: Dict[str, List[Tuple[str, float]]] = {}
    all_qids = set().union(*[r.keys() for r in runs_list])
    for qid in all_qids:
        scores = defaultdict(float)
        for runs in runs_list:
            ranked = runs.get(qid, [])
            for rank, (docid, _) in enumerate(ranked, start=1):
                scores[docid] += 1.0 / (k + rank)
        fused_list = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:K]
        fused[qid] = [(docid, fused_score) for docid, fused_score in fused_list]
    return fused


def hit_rate_recall_at_k(
    runs: Dict[str, List[Tuple[str, float]]],
    qrels: Dict[str, Set[str]],
    k: int = 100) -> Tuple[float, float]:
    """Calculates two metrics in one pass:

    1) hit rate: proportion of queries with >= 1 relevant in top-k
    2) recall: relevant retrieved in top-k / total relevant

    Args:
        runs: {qid: [(doc_id, score), ...]}
        qrels: {qid: {relevant_doc_ids}}
        k: threshold to calculate metrics at

    Returns:
        Tuple of (hit_rate, recall)
    """
    if not runs or not qrels:
        return (0.0, 0.0)

    hits = retrieved = total = 0
    evaluated_queries = 0

    for qid, ranked in runs.items():
        if qid not in qrels:
            continue

        pos_docs = qrels[qid]
        top_docids = {d for d, _ in ranked[:k]}
        intersection = top_docids & pos_docs

        hits += bool(intersection)
        retrieved += len(intersection)
        total += len(pos_docs)
        evaluated_queries += 1

    hit_rate = hits / evaluated_queries if evaluated_queries else 0.0
    recall = retrieved / total if total else 0.0

    return (round(hit_rate, 4), round(recall, 4))


def print_metrics_at_k(
    runs: Dict[str, List[Tuple[str, float]]],
    qrels: Dict[str, Set[str]],
    K: int,
    model: str):

    model_metrics = hit_rate_recall_at_k(runs, qrels, K)
    hit_rate, recall = model_metrics[0], model_metrics[1]
    print(f"{model} hit rate at k={K}: {hit_rate}, {model} recall at k={K}: {recall}")
