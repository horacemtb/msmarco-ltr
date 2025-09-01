import numpy as np
import pandas as pd
import re, json, hashlib
from typing import Dict, List, Tuple, Set
from collections import defaultdict
from tqdm import tqdm

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from FlagEmbedding import FlagModel

import gensim.downloader as api

import matplotlib.pyplot as plt
plt.rcParams['figure.dpi'] = 600


TOKEN_RE = re.compile(r"[a-z0-9]+")
VEC_CACHE = {}


def add_retrieval_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Simple rank-based features
    """
    df = df.copy()

    for c in ["imp_rank", "bm25_rank", "imp_score", "bm25_score"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # presence flags (in top-K or not)
    df["in_imp_topk"] = df["imp_rank"].notna().astype(np.int8)
    df["in_bm25_topk"] = df["bm25_rank"].notna().astype(np.int8)

    # reciprocal ranks (NaN -> 0)
    df["imp_rr"] = 0.0
    m = df["imp_rank"].notna()
    df.loc[m, "imp_rr"] = 1.0 / df.loc[m, "imp_rank"]

    df["bm25_rr"] = 0.0
    m = df["bm25_rank"].notna()
    df.loc[m, "bm25_rr"] = 1.0 / df.loc[m, "bm25_rank"]

    return df


def encode_bge(encoder: FlagModel, texts: List[str], is_query: bool) -> np.ndarray:
    """
    Encode query and passage sequences using pre-trained BAAI/bge model
    """
    if is_query:
        texts = [("query: " + (t or "")) for t in texts]
    else:
        texts = [("passage: " + (t or "")) for t in texts]
    embs = encoder.encode(texts, batch_size=256)
    return np.asarray(embs, dtype=np.float32)


def build_bge_maps(encoder: FlagModel, df: pd.DataFrame):
    """
    Store unique encoded sequences in dictionaries for fast mapping
    """
    unique_q = df[["qid", "query_text"]].drop_duplicates().fillna("")
    unique_p = df[["pid", "passage_text"]].drop_duplicates().fillna("")
    Q = encode_bge(encoder, unique_q["query_text"].tolist(), is_query=True)
    P = encode_bge(encoder, unique_p["passage_text"].tolist(), is_query=False)
    qid2row = {qid: i for i, qid in enumerate(unique_q["qid"].tolist())}
    pid2row = {pid: i for i, pid in enumerate(unique_p["pid"].tolist())}
    return unique_q, unique_p, Q, P, qid2row, pid2row


def add_dense_cosine(df: pd.DataFrame, Q, P, qrow, prow, name="bge_cosine") -> pd.DataFrame:
    """
    Calculate cosine similarity between query-passage pairs based on BGE embeddings
    """
    vals = np.zeros(len(df), dtype=np.float32)
    for i, (qid, pid) in enumerate(zip(df["qid"].values, df["pid"].values)):
        iq = qrow.get(qid, None)
        ip = prow.get(pid, None)
        if iq is None or ip is None:
            vals[i] = 0.0
        else:
            vals[i] = float(np.dot(Q[iq], P[ip]))
    df[name] = vals
    return df


def build_tfidf_maps(df: pd.DataFrame, vec: TfidfVectorizer):
    """
    Store unique encoded sequences in dictionaries for fast mapping
    """
    unique_q = df[["qid", "query_text"]].drop_duplicates().fillna("")
    unique_p = df[["pid", "passage_text"]].drop_duplicates().fillna("")
    Q = vec.transform(unique_q["query_text"].tolist())
    P = vec.transform(unique_p["passage_text"].tolist())
    qid2row = {qid: i for i, qid in enumerate(unique_q["qid"].tolist())}
    pid2row = {pid: i for i, pid in enumerate(unique_p["pid"].tolist())}
    return unique_q, unique_p, Q, P, qid2row, pid2row


def add_tfidf_cosine(df: pd.DataFrame, Q, P, qrow, prow, name="tfidf_cosine") -> pd.DataFrame:
    """
    Calculate cosine similarity between query-passage pairs based on tf-idf
    """
    vals = np.zeros(len(df), dtype=np.float32)
    for i, (qid, pid) in enumerate(zip(df["qid"].values, df["pid"].values)):
        iq = qrow.get(qid, None)
        ip = prow.get(pid, None)
        if iq is None or ip is None:
            vals[i] = 0.0
        else:
            vals[i] = (Q[iq].multiply(P[ip])).sum()
    df[name] = vals
    return df


def tokenize_(s: str) -> List[str]:
    return TOKEN_RE.findall((s or "").lower())


def add_overlap_length_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Simple rules like sequence lengths, token overlap, etc. to describe queries and passages
    """
    df = df.copy()
    q_tok = df["query_text"].map(tokenize_)
    p_tok = df["passage_text"].map(tokenize_)

    q_set = q_tok.map(set)
    p_set = p_tok.map(set)

    overlap = [len(a & b) for a, b in zip(q_set, p_set)]

    # compute stats
    q_len_w = q_tok.map(len)
    p_len_w = p_tok.map(len)
    q_len_u = q_set.map(len)
    p_len_u = p_set.map(len)

    df["len_q_words"] = q_len_w
    df["len_p_words"] = p_len_w
    df["len_q_unique"] = q_len_u
    df["len_p_unique"] = p_len_u

    df["overlap_unique"] = overlap
    df["overlap_unique_ratio_q"] = np.where(q_len_u > 0, overlap / q_len_u, 0.0).astype(np.float32)
    df["overlap_unique_ratio_p"] = np.where(p_len_u > 0, overlap / p_len_u, 0.0).astype(np.float32)

    # Jaccard over unique tokens
    denom = (q_len_u + p_len_u - overlap).replace(0, np.nan)
    df["jaccard_unique"] = (overlap / denom).fillna(0.0).astype(np.float32)

    # length ratios
    df["len_ratio_wp"] = np.where(p_len_w > 0, q_len_w / p_len_w, 0.0).astype(np.float32)

    return df


def hashed_vec(token: str, dim: int) -> np.ndarray:
    """
    Build a look-up table for unknown tokens based on randomly initialized vectors
    """
    key = ("#h#", token)
    if key in VEC_CACHE:
        return VEC_CACHE[key]
    rs = np.random.RandomState(int(hashlib.md5(token.encode()).hexdigest()[:8], 16))
    v = rs.normal(0, 1, size=dim).astype(np.float32)
    n = np.linalg.norm(v)
    if n > 0:
        v /= n
    VEC_CACHE[key] = v
    return v


def clear_vec_cache():
    VEC_CACHE.clear()


def token_vec(token: str, kv) -> np.ndarray:
    """
    Get token vector from pre-trained FastText or GloVe model, or use a random vector for oov token
    """
    try:
        v = kv.get_vector(token)
        v = v.astype(np.float32)
        n = np.linalg.norm(v)
        return v / n if n > 0 else v
    except KeyError:
        return hashed_vec(token, kv.vector_size)


def gaussian_kernels(K=11, sigma=0.1):
    """
    Initialize parameters (mu, sigma) for K Gaussian kernels
    """
    mus = np.linspace(-1.0, 1.0, K, dtype=np.float32)
    sig = np.full(K, sigma, dtype=np.float32)
    sig[-1] = 1e-3
    return mus, sig


def add_knrm_pretrained(df, kv, K=11, max_q=16, max_p=128, prefix="knrm_ft"):
    """
    Calculate token-based cosine similarity using pre-trained FastText or GloVe model, apply Gaussian kernels over similarity matrix
    """
    df = df.copy()
    mu, sig = gaussian_kernels(K=K, sigma=0.1)
    feats = np.zeros((len(df), K), dtype=np.float32)

    for i, (q, p) in tqdm(enumerate(zip(df["query_text"].values, df["passage_text"].values))):
        q_tokens = tokenize_(q)[:max_q]
        p_tokens = tokenize_(p)[:max_p]
        if not q_tokens or not p_tokens:
            continue
        Q = np.stack([token_vec(t, kv) for t in q_tokens], axis=0)
        P = np.stack([token_vec(t, kv) for t in p_tokens], axis=0)
        S = np.clip(Q @ P.T, -1.0, 1.0) # cosine similarity scores

        for k in range(K):
            K_matrix = np.exp(-(S - mu[k])**2 / (2.0 * (sig[k]**2))) # [query_len, pas_len]
            sum_j = K_matrix.sum(axis=1)
            feats[i, k] = np.log1p(sum_j).sum()

    for k in range(K):
        df[f"{prefix}_k{k}"] = feats[:, k]
    return df


def build_idf_map_from_train(pairs_train) -> dict:
    """
    Token -> idf score dictionary
    """
    corpus = pd.concat([pairs_train["query_text"], pairs_train["passage_text"]]).fillna("").tolist()
    vec = TfidfVectorizer(lowercase=True, analyzer="word", ngram_range=(1, 1), min_df=2)
    vec.fit(corpus)
    vocab = vec.vocabulary_
    idfs = vec.idf_
    return {tok: idfs[idx] for tok, idx in vocab.items()}


def add_maxsim_features(df, kv, idf_map=None,  max_q=16, max_p=128, prefix="maxsim"):
    """
    Calculate similarity features based on best per-token matches between queries and passages:
        - for each token in query find best (max) similarity match among passage tokens;
        - take mean of the similarities
        - calculate idf weighted means of max similarities
    """
    df = df.copy()
    means, idf_means = [], []

    for q, p in tqdm(zip(df["query_text"].values, df["passage_text"].values)):
        q_tokens = tokenize_(q)[:max_q]
        p_tokens = tokenize_(p)[:max_p]
        if not q_tokens or not p_tokens:
            means.append(0.0)
            idf_means.append(0.0)
            continue
        Q = np.stack([token_vec(t, kv) for t in q_tokens], axis=0)
        P = np.stack([token_vec(t, kv) for t in p_tokens], axis=0)
        S = np.clip(Q @ P.T, -1.0, 1.0)
        smax = S.max(axis=1) # per-query-token best match

        means.append(float(smax.mean()))

        if idf_map:
            idfs = np.array([idf_map.get(t, 1.0) for t in q_tokens], dtype=np.float32)
            w = idfs.sum()
            idf_weighted_mean = float((smax * idfs).sum() / w) if w > 0 else 0.0
            idf_means.append(idf_weighted_mean)
        else:
            idf_means.append(0.0)

    df[f"{prefix}_mean"] = np.array(means, dtype=np.float32)
    df[f"{prefix}_idf_mean"] = np.array(idf_means, dtype=np.float32)
    return df


def add_centroid_cosine(df, kv, max_q=16, max_p=128, name="centroid_cosine"):
    """
    Calculate cosine similarity between query and passage centroids
    """
    df = df.copy()
    vals = np.zeros(len(df), dtype=np.float32)

    for i, (q, p) in tqdm(enumerate(zip(df["query_text"].values, df["passage_text"].values))):
        q_tokens = tokenize_(q)[:max_q]
        p_tokens = tokenize_(p)[:max_p]
        if not q_tokens or not p_tokens:
            vals[i] = 0.0
            continue
        Q_M = np.stack([token_vec(t, kv) for t in q_tokens], axis=0).mean(axis=0)
        P_M = np.stack([token_vec(t, kv) for t in p_tokens], axis=0).mean(axis=0)
        # calc cosine similarity
        nq = np.linalg.norm(Q_M)
        np_norm = np.linalg.norm(P_M)
        vals[i] = float(Q_M.dot(P_M) / (nq * np_norm)) if (nq > 0 and np_norm > 0) else 0.0

    df[name] = vals
    return df


def plot_knrm_distributions(df: pd.DataFrame,
                            kernel_prefix_filter: str,
                            title_prefix: str = "Distribution of"):
    """
    Build one violin plot per kernel column to compare pas_type categories.
    """
    if "pas_type" not in df.columns:
        raise ValueError("df must contain a pas_type col for grouping")

    ker_cols = [c for c in df.columns if c.startswith(kernel_prefix_filter)]
    if not ker_cols:
        raise ValueError("No kernel cols detected")

    # order kernels by k-index
    ker_cols = sorted(ker_cols, key=lambda x: int(re.findall(r"k(\d+)$", x)[0]))

    # order categories
    cats = df["pas_type"].dropna().astype(str).unique().tolist()
    cats = ["pos"] + [c for c in cats if c != "pos"]

    for col in ker_cols:

        data = [df.loc[df["pas_type"] == cat, col].astype(float).dropna().values for cat in cats]

        plt.figure(figsize=(8, 5))
        parts = plt.violinplot(data, showmeans=True, showmedians=False, showextrema=False)
        plt.xticks(range(1, len(cats) + 1), cats, rotation=20, ha='right')
        plt.title(f"{title_prefix} {col} by pas_type")
        plt.ylabel("kernel score")
        plt.grid(True, axis='y', alpha=0.3)
        plt.tight_layout()
        plt.show()


def kernel_summary_table(df: pd.DataFrame, kernel_prefix_filter: str) -> pd.DataFrame:
    """
    Summary table that shows mean value of each kernel per pas_type
    """
    ker_cols = [c for c in df.columns if c.startswith(kernel_prefix_filter)]
    cats = df["pas_type"].dropna().astype(str).unique().tolist()
    rows = []
    for c in sorted(ker_cols, key=lambda x: int(re.findall(r"k(\d+)$", x)[0])):
        for cat in ["pos"] + [x for x in cats if x != "pos"]:
            vals = df.loc[df["pas_type"] == cat, c].astype(float).dropna()
            rows.append({"kernel": c, "pas_type": cat, "mean": float(vals.mean()) if len(vals) else np.nan, "n": int(len(vals))})
    return pd.DataFrame(rows)


def add_char_ngram_jaccard(df, n=3, name="char_jac"):
    """
    Calculate ngram overlap between query and passage
    """
    def count_grams(s):
        s = (s or "").lower()
        return set([s[i:i+n] for i in range(max(0, len(s)-n+1))])

    query_text = df["query_text"].map(count_grams)
    passage_text = df["passage_text"].map(count_grams)
    inter = [len(a & b) for a, b in zip(query_text, passage_text)]
    union = [len(a | b) for a, b in zip(query_text, passage_text)]
    df[name] = (np.array(inter) / np.maximum(1, np.array(union))).astype(np.float32)
    return df
