"""
Phase 5 – Feature Neutralization (Risk Orthogonalisation)
==========================================================
Even a perfectly trained model will load heavily on market beta
and sector momentum – the dominant variance sources in equities.
A signal that says "buy tech, sell energy" is a macro bet, not alpha.

Feature neutralization geometrically projects out the linear
component of known risk exposures from the model's raw predictions,
leaving only the orthogonal residual: pure stock-specific signal.

The projection uses the Moore-Penrose pseudo-inverse:
    s_neutralized = s − proportion · E @ pinv(E) @ s
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd
from scipy import stats as sp_stats


def neutralize_predictions(
    predictions: np.ndarray,
    exposures: np.ndarray,
    proportion: float = 1.0,
) -> np.ndarray:
    """
    Remove the linear component of *exposures* from *predictions*
    and re-standardise.

    Parameters
    ----------
    predictions : 1-D array of raw model scores (one per stock).
    exposures   : 2-D array (n_stocks, n_risk_factors).
    proportion  : 0–1 aggressiveness knob.  1.0 = full neutralization.

    Returns
    -------
    Neutralized, standardized prediction vector.
    """
    s = predictions.copy().astype(float)
    E = exposures.astype(float)

    mask = np.isfinite(s) & np.all(np.isfinite(E), axis=1)
    s_clean = s[mask]
    E_clean = E[mask]

    pinv_E = np.linalg.pinv(E_clean)
    projected = E_clean @ (pinv_E @ s_clean)
    s_clean = s_clean - proportion * projected

    std = s_clean.std(ddof=0)
    if std > 0:
        s_clean = s_clean / std

    out = np.full_like(s, np.nan)
    out[mask] = s_clean
    return out


def build_exposure_matrix(
    df: pd.DataFrame,
    exposure_cols: List[str],
) -> np.ndarray:
    """
    Construct the risk-exposure matrix E from a DataFrame.

    For categorical columns (e.g. sector), one-hot encode them.
    For numeric columns (e.g. log_market_cap, beta), pass through.
    """
    parts = []
    for col in exposure_cols:
        if col not in df.columns:
            continue
        if df[col].dtype == object or df[col].dtype.name == "category":
            dummies = pd.get_dummies(df[col], prefix=col, drop_first=True).astype(float)
            parts.append(dummies.values)
        else:
            vals = pd.to_numeric(df[col], errors="coerce").fillna(0).values.reshape(-1, 1)
            parts.append(vals)
    if not parts:
        return np.zeros((len(df), 1))
    return np.hstack(parts)


def validate_neutralization(
    neutralized: np.ndarray,
    exposures: np.ndarray,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """
    Regress neutralized predictions against each risk factor column.
    All coefficients should be statistically zero (p > alpha).
    """
    mask = np.isfinite(neutralized) & np.all(np.isfinite(exposures), axis=1)
    s = neutralized[mask]
    E = exposures[mask]

    records = []
    for j in range(E.shape[1]):
        col = E[:, j]
        if np.std(col) == 0:
            records.append({
                "factor_idx": j, "slope": 0.0, "p_value": 1.0,
                "r_squared": 0.0, "neutralized": True,
            })
            continue
        slope, intercept, r, p, se = sp_stats.linregress(col, s)
        records.append({
            "factor_idx": j,
            "slope": slope,
            "p_value": p,
            "r_squared": r ** 2,
            "neutralized": p > alpha,
        })
    return pd.DataFrame(records)


def run_neutralization(
    df: pd.DataFrame,
    prediction_col: str,
    cfg: dict,
) -> tuple[np.ndarray, pd.DataFrame]:
    """
    Leakage-free neutralization: project out risk exposures
    independently per rebalance date (cross-sectional), so future
    dates never influence the pseudo-inverse at earlier dates.

    FIX #3: Previously computed pinv(E) over the full test window,
    leaking future exposure structure into earlier scores.
    """
    risk_cfg = cfg["risk"]["neutralization"]
    if not risk_cfg.get("enabled", True):
        return df[prediction_col].values, pd.DataFrame()

    exposure_cols = risk_cfg["exposures"]
    proportion = risk_cfg["proportion"]

    neutralized = np.full(len(df), np.nan)

    if "date" in df.columns:
        # Per-date cross-sectional neutralisation (leakage-free)
        for dt, grp in df.groupby("date"):
            idx = grp.index
            E_date = build_exposure_matrix(grp, exposure_cols)
            s_date = grp[prediction_col].values.astype(float)
            neutralized[idx] = neutralize_predictions(s_date, E_date, proportion)
    else:
        # Fallback: single cross-section (no temporal dimension)
        E = build_exposure_matrix(df, exposure_cols)
        s = df[prediction_col].values.astype(float)
        neutralized = neutralize_predictions(s, E, proportion)

    # Validation on the aggregated result
    E_full = build_exposure_matrix(df, exposure_cols)
    validation = validate_neutralization(neutralized, E_full)

    n_failed = (~validation["neutralized"]).sum()
    if n_failed > 0:
        print(f"  WARNING: {n_failed} risk factors not fully neutralized")
    else:
        print("  All risk factors successfully neutralized (p > 0.05)")

    return neutralized, validation
