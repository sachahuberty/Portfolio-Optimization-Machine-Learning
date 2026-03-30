"""
Phase 4a – Combinatorial Purged Cross-Validation (CPCV)
========================================================
Standard k-fold is *forbidden* in financial ML because random
partitioning leaks future information into the training set.
Even expanding-window time-series split produces too few test
paths to estimate the distribution of out-of-sample performance.

CPCV partitions the time axis into N contiguous groups and
evaluates every combination of k test groups, yielding C(N,k)
unique train/test splits.  Two additional safeguards are applied:

  * **Purging** – remove any training observation whose triple-
    barrier label window overlaps with the test period.
  * **Embargoing** – after each test-group boundary, delete an
    additional buffer of training observations so the model
    cannot learn from the immediate aftermath of test events.
"""

from __future__ import annotations

from itertools import combinations
from typing import Generator, List, Optional, Tuple

import numpy as np
import pandas as pd


def _build_time_groups(
    dates: np.ndarray,
    n_groups: int,
) -> List[Tuple[int, int]]:
    """Split the sorted date index into N contiguous groups.
    Returns list of (start_idx, end_idx) pairs (inclusive)."""
    n = len(dates)
    boundaries = np.linspace(0, n, n_groups + 1, dtype=int)
    return [(boundaries[i], boundaries[i + 1] - 1) for i in range(n_groups)]


def purged_embargo_split(
    dates: pd.Series,
    t_barrier: pd.Series,
    groups: List[Tuple[int, int]],
    test_group_indices: Tuple[int, ...],
    embargo_days: int = 5,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Given the time groups and the chosen test-group indices,
    return purged+embargoed train indices and test indices.

    Parameters
    ----------
    dates             : Series of entry dates aligned by position.
    t_barrier         : Series of barrier-touch dates aligned by position.
    groups            : Output of _build_time_groups().
    test_group_indices: Which of the N groups form the test set.
    embargo_days      : Number of calendar days to embargo after
                        each test-group boundary.
    """
    test_mask = np.zeros(len(dates), dtype=bool)
    for gi in test_group_indices:
        s, e = groups[gi]
        test_mask[s : e + 1] = True

    test_indices = np.where(test_mask)[0]
    if len(test_indices) == 0:
        return np.array([], dtype=int), np.array([], dtype=int)

    test_start_dates = []
    test_end_dates = []
    for gi in test_group_indices:
        s, e = groups[gi]
        test_start_dates.append(dates.iloc[s])
        test_end_dates.append(dates.iloc[e])

    train_mask = ~test_mask.copy()
    date_vals = dates.values
    tbar_vals = t_barrier.values

    for i in range(len(dates)):
        if not train_mask[i]:
            continue

        # Purging: if the training observation's barrier-touch date
        # falls within or after any test group's start, remove it.
        for ts, te in zip(test_start_dates, test_end_dates):
            ts_np = np.datetime64(ts)
            te_np = np.datetime64(te)
            if not pd.isna(tbar_vals[i]):
                tbar_np = np.datetime64(tbar_vals[i])
                if tbar_np >= ts_np:
                    train_mask[i] = False
                    break

        # Embargoing: remove training observations that fall within
        # the embargo window after each test group's end.
        if train_mask[i]:
            for te in test_end_dates:
                te_np = np.datetime64(te)
                embargo_end = te_np + np.timedelta64(embargo_days, "D")
                if te_np < np.datetime64(date_vals[i]) <= embargo_end:
                    train_mask[i] = False
                    break

    train_indices = np.where(train_mask)[0]
    return train_indices, test_indices


def cpcv_splits(
    dates: pd.Series,
    t_barrier: pd.Series,
    n_groups: int = 10,
    k_test_groups: int = 2,
    embargo_days: int = 5,
) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
    """
    Generator yielding (train_idx, test_idx) for every CPCV fold.

    Produces C(n_groups, k_test_groups) unique splits.
    """
    sorted_order = dates.argsort()
    dates_sorted = dates.iloc[sorted_order].reset_index(drop=True)
    tbar_sorted = t_barrier.iloc[sorted_order].reset_index(drop=True)
    groups = _build_time_groups(dates_sorted.values, n_groups)

    for test_combo in combinations(range(n_groups), k_test_groups):
        train_idx, test_idx = purged_embargo_split(
            dates_sorted, tbar_sorted, groups, test_combo, embargo_days,
        )
        if len(train_idx) == 0 or len(test_idx) == 0:
            continue
        # Map back to original indices
        yield sorted_order[train_idx], sorted_order[test_idx]


def evaluate_cpcv(
    model_factory,
    X: pd.DataFrame,
    y: pd.Series,
    dates: pd.Series,
    t_barrier: pd.Series,
    n_groups: int = 10,
    k_test_groups: int = 2,
    embargo_days: int = 5,
) -> pd.DataFrame:
    """
    Train+evaluate a model across all CPCV folds and collect
    per-fold metrics.

    Parameters
    ----------
    model_factory : Callable that returns a fresh, untrained model
                    with .fit(X, y) and .predict_proba(X) methods.
    X, y          : Feature matrix and labels.
    dates         : Entry dates for purging alignment.
    t_barrier     : Barrier-touch dates for purging.

    Returns
    -------
    DataFrame with one row per fold containing accuracy, log-loss,
    AUC, and fold metadata.
    """
    from sklearn.metrics import accuracy_score, log_loss, roc_auc_score

    records = []
    for fold_id, (train_idx, test_idx) in enumerate(
        cpcv_splits(dates, t_barrier, n_groups, k_test_groups, embargo_days)
    ):
        model = model_factory()
        X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
        X_te, y_te = X.iloc[test_idx], y.iloc[test_idx]

        if y_tr.nunique() < 2 or y_te.nunique() < 2:
            continue

        model.fit(X_tr, y_tr)
        proba = model.predict_proba(X_te)
        preds = model.predict(X_te)

        pos_col = 1 if proba.shape[1] > 1 else 0
        records.append({
            "fold": fold_id,
            "n_train": len(train_idx),
            "n_test": len(test_idx),
            "accuracy": accuracy_score(y_te, preds),
            "logloss": log_loss(y_te, proba, labels=[0, 1]),
            "auc": roc_auc_score(y_te, proba[:, pos_col]),
        })

    return pd.DataFrame(records)
