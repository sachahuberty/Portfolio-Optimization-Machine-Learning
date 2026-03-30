"""
Phase 4b – Two-Stage Modeling with Meta-Labeling
==================================================
Stage 1: Primary directional model (XGBoost) predicts Long / Short.
Stage 2: Meta-model  (Random Forest) predicts whether the primary
         model's signal will be correct, outputting a calibrated
         probability used directly for bet sizing.

Decoupling direction from conviction produces better-calibrated
position sizes and acts as a false-positive filter: low meta-
probability signals are zeroed out, preventing the portfolio
from taking noise trades.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from src.modeling.cpcv_pipeline import cpcv_splits


# ═══════════════════════════════════════════════════════════════
# 1. Primary Directional Model
# ═══════════════════════════════════════════════════════════════

def build_primary_model(cfg: dict, early_stopping: bool = True) -> XGBClassifier:
    """Instantiate XGBoost with the aggressive regularisation
    protocol mandated for financial time-series.

    Set early_stopping=False for CPCV evaluation folds where no
    separate validation set is available.
    """
    params = cfg["modeling"]["primary"]["params"].copy()
    n_est = params.pop("n_estimators", 1500)
    early = params.pop("early_stopping_rounds", 50)
    kw = dict(n_estimators=n_est, use_label_encoder=False, random_state=42, **params)
    if early_stopping:
        kw["early_stopping_rounds"] = early
    return XGBClassifier(**kw)


def train_primary_model(
    model: XGBClassifier,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: Optional[pd.DataFrame] = None,
    y_val: Optional[pd.Series] = None,
) -> XGBClassifier:
    """Fit the primary model, using the validation set for early stopping."""
    fit_params = {}
    if X_val is not None and y_val is not None:
        fit_params["eval_set"] = [(X_val, y_val)]
        fit_params["verbose"] = False
    model.fit(X_train, y_train, **fit_params)
    return model


# ═══════════════════════════════════════════════════════════════
# 2. Meta-Label Target Construction
# ═══════════════════════════════════════════════════════════════

def construct_meta_labels(
    primary_preds: np.ndarray,
    true_labels: pd.Series,
) -> pd.Series:
    """
    Build the meta-labeling target:
      1  if the primary model's prediction matched the true outcome,
      0  otherwise.
    """
    correct = (primary_preds == true_labels.values).astype(int)
    return pd.Series(correct, index=true_labels.index, name="meta_target")


# ═══════════════════════════════════════════════════════════════
# 3. Meta-Model (Random Forest)
# ═══════════════════════════════════════════════════════════════

def build_meta_model(cfg: dict) -> RandomForestClassifier:
    """Instantiate the meta-labeling Random Forest."""
    params = cfg["modeling"]["meta"]["params"].copy()
    return RandomForestClassifier(random_state=42, **params)


def train_meta_model(
    meta_model: RandomForestClassifier,
    X_train: pd.DataFrame,
    meta_target_train: pd.Series,
) -> RandomForestClassifier:
    """Fit the meta-model on the features + meta-label target."""
    meta_model.fit(X_train, meta_target_train)
    return meta_model


# ═══════════════════════════════════════════════════════════════
# 4. Bet Sizing
# ═══════════════════════════════════════════════════════════════

def compute_bet_sizes(
    primary_direction: np.ndarray,
    meta_probability: np.ndarray,
    cfg: dict,
) -> np.ndarray:
    """
    Convert the two-stage outputs into position sizes.

    Methods
    -------
    meta_prob   : size = direction * meta_probability
    half_kelly  : size = direction * 0.5 * (2*p - 1)   (bounded risk)
    equal       : size = direction * 1.0  (ignores meta)
    """
    method = cfg["modeling"]["bet_sizing"]["method"]
    min_prob = cfg["modeling"]["bet_sizing"]["min_meta_probability"]
    max_wt = cfg["modeling"]["bet_sizing"]["max_position_weight"]

    mask = meta_probability >= min_prob

    if method == "meta_prob":
        raw_size = primary_direction * meta_probability
    elif method == "half_kelly":
        edge = np.clip(2 * meta_probability - 1, 0, None)
        raw_size = primary_direction * 0.5 * edge
    else:  # equal
        raw_size = primary_direction.astype(float)

    raw_size[~mask] = 0.0
    raw_size = np.clip(raw_size, -max_wt, max_wt)
    return raw_size


# ═══════════════════════════════════════════════════════════════
# 5. Full Two-Stage Training Orchestrator
# ═══════════════════════════════════════════════════════════════

def _out_of_fold_primary_predictions(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    dates_train: pd.Series,
    t_barrier_train: pd.Series,
    cfg: dict,
    n_folds: int = 5,
) -> np.ndarray:
    """
    Generate out-of-fold (OOF) primary-model predictions on the
    training set.  This prevents meta-label leakage: each training
    observation's meta-target is derived from a model that never
    saw that observation.
    """
    # -1 sentinel: fold 0 has no history to train on, so it gets no prediction.
    oof_preds = np.full(len(y_train), -1, dtype=int)
    sorted_idx = dates_train.argsort().values
    fold_size = len(sorted_idx) // n_folds

    # Start at k=1: fold 0 has no prior data, so skip it.
    # tr_idx uses only past folds — never future ones.
    for k in range(1, n_folds):
        val_start = k * fold_size
        val_end = (k + 1) * fold_size if k < n_folds - 1 else len(sorted_idx)
        val_idx = sorted_idx[val_start:val_end]
        tr_idx = sorted_idx[:val_start]  # past only

        model = build_primary_model(cfg, early_stopping=False)
        model.fit(X_train.iloc[tr_idx], y_train.iloc[tr_idx])
        oof_preds[val_idx] = model.predict(X_train.iloc[val_idx])

    return oof_preds


def run_two_stage_pipeline(
    X: pd.DataFrame,
    y: pd.Series,
    dates: pd.Series,
    t_barrier: pd.Series,
    cfg: dict,
) -> dict:
    """
    Leakage-free two-stage pipeline:

    Fix #1: Early stopping uses a validation slice carved from training
            data (last 10% of training period), NOT the test set.
    Fix #4: Meta-label targets are built from out-of-fold primary
            predictions, not in-sample predictions.
    """
    np.random.seed(42)
    from src.modeling.cpcv_pipeline import evaluate_cpcv

    feature_cols = [c for c in X.columns if X[c].dtype in ("float64", "float32", "int64")]
    X_clean = X[feature_cols].copy()
    X_clean = X_clean.fillna(0)

    # ── CPCV evaluation (unchanged — already leakage-free) ──
    print("[Phase 4] Running CPCV evaluation …")
    val_cfg = cfg["validation"]

    def _primary_factory():
        return build_primary_model(cfg, early_stopping=False)

    cpcv_results = evaluate_cpcv(
        _primary_factory, X_clean, y, dates, t_barrier,
        n_groups=val_cfg["n_groups"],
        k_test_groups=val_cfg["k_test_groups"],
        embargo_days=val_cfg["embargo_days"],
    )
    print(f"  CPCV folds evaluated: {len(cpcv_results)}")
    if not cpcv_results.empty:
        print(f"  Mean OOS accuracy: {cpcv_results['accuracy'].mean():.4f}")
        print(f"  Mean OOS AUC:      {cpcv_results['auc'].mean():.4f}")

    # ── Three-way temporal split ──
    # train_model: [0%, val_start)   — fit primary + OOF for meta
    # validation:  [val_start, test_start) — tune bet sizing (leakage-free)
    # test:        [test_start, ...)  — final evaluation
    test_start = cfg.get("universe", {}).get("test_start_date")
    if test_start:
        test_split = pd.Timestamp(test_start)
    else:
        sorted_dates = dates.sort_values()
        test_split = sorted_dates.iloc[int(len(sorted_dates) * 0.8)]

    test_mask = dates > test_split
    pre_test = dates <= test_split

    # Validation = last 12.5% of pre-test data (~70/12.5/17.5 split)
    pre_test_dates = dates[pre_test].sort_values()
    val_split = pre_test_dates.iloc[int(len(pre_test_dates) * 0.85)]
    train_model_mask = dates <= val_split
    val_mask = (dates > val_split) & (dates <= test_split)

    X_tr, y_tr = X_clean[train_model_mask], y[train_model_mask]
    X_val, y_val = X_clean[val_mask], y[val_mask]
    X_te, y_te = X_clean[test_mask], y[test_mask]
    dates_tr = dates[train_model_mask]
    dates_val = dates[val_mask]
    t_barrier_tr = t_barrier[train_model_mask]

    print(f"  Three-way split:")
    print(f"    Train model: {len(X_tr)} obs (→ {val_split.date()})")
    print(f"    Validation:  {len(X_val)} obs ({val_split.date()} → {test_split.date()})")
    print(f"    Test:        {len(X_te)} obs ({test_split.date()} →)")

    # ── FIX #1: Early stopping on last 10% of model-training data ──
    print("[Phase 4] Training primary directional model (XGBoost) …")
    tr_sorted = dates_tr.sort_values()
    es_split = tr_sorted.iloc[int(len(tr_sorted) * 0.9)]
    tr_fit_mask = dates_tr <= es_split
    tr_es_mask = dates_tr > es_split

    X_tr_fit, y_tr_fit = X_tr[tr_fit_mask.values], y_tr[tr_fit_mask.values]
    X_tr_es, y_tr_es = X_tr[tr_es_mask.values], y_tr[tr_es_mask.values]

    primary = build_primary_model(cfg, early_stopping=True)
    primary = train_primary_model(primary, X_tr_fit, y_tr_fit, X_tr_es, y_tr_es)
    print(f"  Primary model: trained on {len(X_tr_fit)} obs, "
          f"early-stop on {len(X_tr_es)} obs")

    primary_preds_test = primary.predict(X_te)
    primary_proba_test = primary.predict_proba(X_te)

    # ── FIX #4: Out-of-fold primary predictions for meta-label targets ──
    print("[Phase 4] Generating out-of-fold predictions for meta-labels …")
    oof_primary_preds = _out_of_fold_primary_predictions(
        X_tr, y_tr, dates_tr, t_barrier_tr, cfg, n_folds=5,
    )
    oof_valid = oof_primary_preds != -1
    meta_target_train = construct_meta_labels(
        oof_primary_preds[oof_valid],
        y_tr.iloc[oof_valid],
    )
    print(f"  OOF meta-target balance: "
          f"{(meta_target_train == 1).sum()} correct / "
          f"{(meta_target_train == 0).sum()} incorrect")
    print(f"  (fold 0 excluded: {(~oof_valid).sum()} obs had no prior training data)")

    # Stage 2: Meta-Model (trained on model-training data only)
    print("[Phase 4] Training meta-model (Random Forest) …")
    meta_model = build_meta_model(cfg)
    meta_model = train_meta_model(meta_model, X_tr.iloc[oof_valid], meta_target_train)

    # ── Predictions on test set ──
    meta_proba_test = meta_model.predict_proba(X_te)
    meta_prob_correct = (
        meta_proba_test[:, 1] if meta_proba_test.shape[1] > 1
        else meta_proba_test[:, 0]
    )

    direction = np.where(primary_preds_test == 1, 1, -1).astype(float)
    bet_sizes = compute_bet_sizes(direction, meta_prob_correct, cfg)

    # ── Predictions on validation set (for leakage-free bet sizing sweep) ──
    val_preds = primary.predict(X_val)
    val_meta_proba = meta_model.predict_proba(X_val)
    val_meta_prob = (
        val_meta_proba[:, 1] if val_meta_proba.shape[1] > 1
        else val_meta_proba[:, 0]
    )
    val_direction = np.where(val_preds == 1, 1, -1).astype(float)

    print(f"  Non-zero positions (test): {np.count_nonzero(bet_sizes)} / {len(bet_sizes)}")

    return {
        "primary_model": primary,
        "meta_model": meta_model,
        "cpcv_results": cpcv_results,
        "X_test": X_te,
        "y_test": y_te,
        "dates_test": dates[test_mask],
        "primary_preds": primary_preds_test,
        "primary_proba": primary_proba_test,
        "meta_probability": meta_prob_correct,
        "direction": direction,
        "bet_sizes": bet_sizes,
        "feature_cols": feature_cols,
        "train_mask": train_model_mask,
        "val_mask": val_mask,
        "test_mask": test_mask,
        # Validation set outputs for bet sizing sweep
        "val_direction": val_direction,
        "val_meta_probability": val_meta_prob,
    }
