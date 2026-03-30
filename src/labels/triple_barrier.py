"""
Phase 3 – Triple Barrier Labeling
===================================
Volatility-adaptive label construction that integrates risk-management
parameters directly into the learning target.  Each label encodes
whether a trade hit take-profit, stop-loss, or timed out – forcing
the model to learn price *paths*, not arbitrary endpoints.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


# ═══════════════════════════════════════════════════════════════
# 1. EWMA Volatility Estimation
# ═══════════════════════════════════════════════════════════════

def ewma_volatility(
    prices: pd.Series,
    span: int = 50,
) -> pd.Series:
    """
    Exponentially Weighted Moving Average of absolute daily log
    returns – the per-asset, per-day volatility estimate used to
    set dynamic barrier widths.

    The EWMA gives higher weight to recent observations, making the
    barrier widths responsive to the current volatility regime rather
    than a stale trailing average.
    """
    log_ret = np.log(prices / prices.shift(1))
    return log_ret.abs().ewm(span=span, min_periods=span).mean()


# ═══════════════════════════════════════════════════════════════
# 2. Triple Barrier Label Assignment
# ═══════════════════════════════════════════════════════════════

def _apply_triple_barrier_single(
    prices: np.ndarray,
    entry_idx: int,
    upper_mult: float,
    lower_mult: float,
    daily_vol: float,
    vertical_bars: int,
) -> dict:
    """
    Walk the price path forward from *entry_idx* and determine which
    of the three barriers is touched first.

    Parameters
    ----------
    prices       : 1-D array of adjusted close prices.
    entry_idx    : Index into *prices* for the entry day.
    upper_mult   : Multiplier for the take-profit barrier.
    lower_mult   : Multiplier for the stop-loss barrier.
    daily_vol    : EWMA daily volatility at entry date.
    vertical_bars: Maximum holding period in trading days.

    Returns
    -------
    dict with keys: label, touch_idx, return_at_touch, barrier_type.
    """
    entry_price = prices[entry_idx]
    horizon = min(entry_idx + vertical_bars, len(prices) - 1)

    upper_barrier = entry_price * (1 + upper_mult * daily_vol * np.sqrt(vertical_bars))
    lower_barrier = entry_price * (1 - lower_mult * daily_vol * np.sqrt(vertical_bars))

    for t in range(entry_idx + 1, horizon + 1):
        p = prices[t]

        if p >= upper_barrier:
            return {
                "label": 1,
                "touch_idx": t,
                "return_at_touch": (p - entry_price) / entry_price,
                "barrier_type": "upper",
            }
        if p <= lower_barrier:
            return {
                "label": -1,
                "touch_idx": t,
                "return_at_touch": (p - entry_price) / entry_price,
                "barrier_type": "lower",
            }

    # Vertical barrier hit
    final_price = prices[horizon]
    final_ret = (final_price - entry_price) / entry_price
    return {
        "label": int(np.sign(final_ret)) if final_ret != 0 else 0,
        "touch_idx": horizon,
        "return_at_touch": final_ret,
        "barrier_type": "vertical",
    }


def compute_triple_barrier_labels(
    pricing: pd.DataFrame,
    cfg: dict,
    entry_freq: str = "ME",
) -> pd.DataFrame:
    """
    For every (ticker, entry_date), compute the Triple Barrier label.

    Parameters
    ----------
    pricing    : Long-form pricing DataFrame with [date, ticker, adj_close].
    cfg        : Full config dict (uses cfg['labels'] block).
    entry_freq : How often to generate entry points.  'ME' = end of each
                 calendar month.  'W-FRI' = weekly.

    Returns
    -------
    DataFrame with columns:
        ticker, entry_date, label, t_barrier (barrier touch date),
        return_at_touch, barrier_type, ewma_vol, upper_barrier, lower_barrier
    """
    lcfg = cfg["labels"]
    upper_mult = lcfg["upper_barrier_mult"]
    lower_mult = lcfg["lower_barrier_mult"]
    vert_days = lcfg["vertical_barrier_days"]
    span = lcfg["ewma_span"]
    discard_vert = lcfg.get("discard_vertical", False)
    min_ret = lcfg.get("min_return_threshold", 0.0)

    results = []

    for ticker, grp in pricing.groupby("ticker"):
        grp = grp.sort_values("date").reset_index(drop=True)
        prices = grp["adj_close"].values
        dates = grp["date"].values

        if len(grp) < span + vert_days + 10:
            continue

        vol_series = ewma_volatility(grp["adj_close"], span=span)

        entry_dates = grp.set_index("date").resample(entry_freq).last().dropna().index
        entry_positions = grp[grp["date"].isin(entry_dates)].index.tolist()

        for idx in entry_positions:
            if idx < span or idx + vert_days >= len(prices):
                continue

            dvol = vol_series.iloc[idx]
            if np.isnan(dvol) or dvol <= 0:
                continue

            res = _apply_triple_barrier_single(
                prices, idx, upper_mult, lower_mult, float(dvol), vert_days,
            )

            if discard_vert and res["barrier_type"] == "vertical":
                continue
            if abs(res["return_at_touch"]) < min_ret and res["barrier_type"] == "vertical":
                continue

            entry_price = prices[idx]
            results.append({
                "ticker": ticker,
                "entry_date": dates[idx],
                "label": res["label"],
                "t_barrier": dates[res["touch_idx"]],
                "return_at_touch": res["return_at_touch"],
                "barrier_type": res["barrier_type"],
                "ewma_vol": dvol,
                "upper_barrier": entry_price * (1 + upper_mult * dvol * np.sqrt(vert_days)),
                "lower_barrier": entry_price * (1 - lower_mult * dvol * np.sqrt(vert_days)),
                "entry_price": entry_price,
            })

    labels_df = pd.DataFrame(results)
    if not labels_df.empty:
        labels_df["entry_date"] = pd.to_datetime(labels_df["entry_date"])
        labels_df["t_barrier"] = pd.to_datetime(labels_df["t_barrier"])

        # Convert to binary (1 / 0) for compatibility with standard classifiers
        labels_df["label_binary"] = (labels_df["label"] == 1).astype(int)

    return labels_df


# ═══════════════════════════════════════════════════════════════
# 3. Label Diagnostics
# ═══════════════════════════════════════════════════════════════

def label_summary(labels_df: pd.DataFrame) -> pd.DataFrame:
    """Return distribution of labels by barrier type."""
    if labels_df.empty:
        return pd.DataFrame()
    return labels_df.groupby(["barrier_type", "label"]).size().unstack(fill_value=0)


def holding_period_stats(labels_df: pd.DataFrame) -> pd.DataFrame:
    """Average holding period (days) by barrier type."""
    if labels_df.empty:
        return pd.DataFrame()
    labels_df = labels_df.copy()
    labels_df["holding_days"] = (
        labels_df["t_barrier"] - labels_df["entry_date"]
    ).dt.days
    return labels_df.groupby("barrier_type")["holding_days"].describe()


# ═══════════════════════════════════════════════════════════════
# 4. Convenience Orchestrator
# ═══════════════════════════════════════════════════════════════

def run_labeling_pipeline(
    pricing: pd.DataFrame,
    cfg: dict,
) -> pd.DataFrame:
    """Execute the full labeling phase."""
    print("[Phase 3] Computing Triple Barrier labels …")
    labels = compute_triple_barrier_labels(pricing, cfg)
    print(f"  Labels: {len(labels)} observations")
    if not labels.empty:
        print(f"  Distribution:\n{label_summary(labels)}")
        print(f"  Holding periods:\n{holding_period_stats(labels)}")
    return labels
