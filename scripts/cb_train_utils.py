import pandas as pd
import numpy as np
import math
from typing import List, Dict, Tuple
from catboost import CatBoostRanker, Pool
import optuna
from metrics import ndcg_at_ks


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
