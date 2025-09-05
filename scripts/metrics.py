import math
from typing import List, Dict
import numpy as np
import pandas as pd


def dcg_at_k(gains: List[int], k: int) -> float:
    return sum(g / math.log2(i+2) for i, g in enumerate(gains[:k]))


def ndcg_for_group(labels: np.ndarray, preds: np.ndarray, k: int) -> float:
    """
    Calculate NDCG@K for a single query group
    """
    order = np.argsort(-preds)
    gains = labels[order]
    dcg = dcg_at_k(gains.tolist(), k)
    ideal = dcg_at_k(sorted(labels, reverse=True), k)
    return (dcg / ideal) if ideal > 0 else 0.0


def ndcg_at_ks(qids: np.ndarray, labels: np.ndarray, preds: np.ndarray, ks=(10, 20)) -> Dict[int, float]:
    """
    Calculate average NDCG@K scores across multiple queries for various K values
    
    Args:
        qids: query id
        labels: ground truth relevance labels
        preds: predicted scores from Catboost ranker
        ks: tuple of K values to compute NDCG@K
        
    Returns:
        Dictionary mapping each K to its average NDCG@K score across all queries
    """
    out = {}
    df = pd.DataFrame({"qid": qids, "label": labels, "pred": preds})
    by_query = df.groupby("qid", sort=False)
    for K in ks:
        s = 0.0
        n = 0
        for _, g in by_query:
            s += ndcg_for_group(g["label"].values, g["pred"].values, K)
            n += 1
        out[K] = s / n if n else 0.0
    return out


def mrr_for_group(labels: np.ndarray, preds: np.ndarray, k: int) -> float:
    """
    Calculate MRR@K for a single query group
    """
    order = np.argsort(-preds)[:k]
    topk_labels = labels[order]
    hits = np.where(topk_labels > 0)[0]
    if hits.size == 0:
        return 0.0
    rank = int(hits[0]) + 1
    return 1.0 / rank


def mrr_at_k(qids: np.ndarray, labels: np.ndarray, preds: np.ndarray, ks=(10, 20)) -> Dict[int, float]:
    """
    Calculate average MRR@K scores across multiple queries for various K values
    
    Args:
        qids: query id
        labels: ground truth relevance labels
        preds: predicted scores from Catboost ranker
        ks: tuple of K values to compute MRR@K
        
    Returns:
        Dictionary mapping each K to its average MRR@K score across all queries
    """
    out = {}
    df = pd.DataFrame({"qid": qids, "label": labels, "pred": preds})
    by_query = df.groupby("qid", sort=False)
    for K in ks:
        s = 0.0
        n = 0
        for _, g in by_query:
            s += mrr_for_group(g["label"].values, g["pred"].values, K)
            n += 1
        out[K] = s / n if n else 0.0
    return out
