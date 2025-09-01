import pandas as pd
import numpy as np
import math
from typing import List, Dict, Tuple
from catboost import CatBoostRanker, Pool
import optuna


def make_group_ids(df: pd.DataFrame) -> np.ndarray:
    """Map string qid -> integer group_id while keeping row order"""
    qids = df["qid"].values
    uniq, inv = np.unique(qids, return_inverse=True)
    return inv.astype(np.int64)


def make_pool(df: pd.DataFrame, feature_cols: List) -> Pool:
    """
    Creates a Catboost Pool object from a dataframe

    Args:
        df: processed dataframe containing features and labels
        feature_cols: list of column names to use as features
    """
    X = df[feature_cols]
    y = df["label"].values
    group_id = make_group_ids(df)
    return Pool(data=X, label=y, group_id=group_id, feature_names=feature_cols)


def objective(trial: optuna.Trial, val: pd.DataFrame, train_pool: Pool, val_pool: Pool, ndcg_at_k_for_tuning: int, random_state: int) -> float:
    """
    Optuna objective for hyperparameter optimization of Catboost ranker using NDCG@K metric on validation data.
    Designed for ranking tasks with query-grouped data. Uses GPU acceleration and early stopping.
    """
    loss = trial.suggest_categorical("loss_function", ["QuerySoftMax", "YetiRank", "YetiRankPairwise", "PairLogitPairwise"])

    params = {
        "loss_function": loss,
        "eval_metric": "NDCG:top={}".format(ndcg_at_k_for_tuning),
        "random_seed": random_state,
        "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.2, log=True),
        "depth": trial.suggest_int("depth", 5, 8),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-3, 10.0, log=True),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 20, 200),
        "random_strength": trial.suggest_float("random_strength", 0.0, 2.0),
        "bootstrap_type": trial.suggest_categorical("bootstrap_type", ["Bayesian", "Bernoulli"]),
        "task_type": "GPU",
        "verbose": False,
        "iterations": 5000,
        "use_best_model": True
        }

    if params["bootstrap_type"] == "Bernoulli":
        params["subsample"] = trial.suggest_float("subsample", 0.5, 1.0)
    elif params["bootstrap_type"] == "Bayesian":
        params["bagging_temperature"] = trial.suggest_float("bagging_temperature", 0.0, 10.0)

    model = CatBoostRanker(**params)
    model.fit(train_pool, eval_set=val_pool, early_stopping_rounds=100)

    # predict on validation and compute NDCG@K
    val_preds = model.predict(val_pool)
    ndcgs = ndcg_at_ks(val["qid"].values, val["label"].values, val_preds, ks=(ndcg_at_k_for_tuning,))
    score = ndcgs[ndcg_at_k_for_tuning]
    return score


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
