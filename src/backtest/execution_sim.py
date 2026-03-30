"""
Phase 6 – Execution-Aware Backtest
====================================
Rank-weighted long/short portfolio construction with position limits,
staggered execution, realistic transaction costs, and market-impact
modeling.  The goal is to measure the strategy's performance under
conditions that approximate institutional execution reality — not the
frictionless fantasy of academic backtests.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd


# ═══════════════════════════════════════════════════════════════
# 1. Portfolio Construction
# ═══════════════════════════════════════════════════════════════

def rank_weighted_portfolio(
    scores: pd.Series,
    sectors: pd.Series,
    cfg: dict,
) -> pd.Series:
    """
    Build a long-only (or long/short) portfolio from conviction scores.

    Parameters
    ----------
    scores  : Series indexed by ticker with neutralized conviction scores.
    sectors : Series indexed by ticker with sector labels.
    cfg     : portfolio block from config.

    Returns
    -------
    Series of target weights (positive = long, negative = short).
    """
    pcfg = cfg["portfolio"]
    max_name = pcfg["max_single_name_pct"]
    max_sector = pcfg["max_sector_pct"]
    long_only = pcfg.get("long_only", False)

    scores = scores.dropna()
    if scores.empty:
        return pd.Series(dtype=float)

    n = len(scores)
    ranked = scores.rank(pct=True)

    if pcfg["long_leg"] == "top_decile":
        long_mask = ranked >= 0.9
    elif pcfg["long_leg"] == "top_vigintile":
        long_mask = ranked >= 0.95
    else:
        long_mask = ranked >= 0.8

    weights = pd.Series(0.0, index=scores.index)

    if not long_only:
        if pcfg["short_leg"] == "bottom_decile":
            short_mask = ranked <= 0.1
        else:
            short_mask = ranked <= 0.2

    # Weight by conviction within each leg
    if pcfg["weighting"] == "conviction":
        long_scores = scores[long_mask]
        if len(long_scores) > 0:
            weights[long_mask] = long_scores / long_scores.abs().sum()
        if not long_only:
            short_scores = scores[short_mask]
            if len(short_scores) > 0:
                weights[short_mask] = short_scores / short_scores.abs().sum()
    else:
        n_long = long_mask.sum()
        if n_long > 0:
            weights[long_mask] = 1.0 / n_long
        if not long_only:
            n_short = short_mask.sum()
            if n_short > 0:
                weights[short_mask] = -1.0 / n_short

    # Normalise: long weights sum to 1
    long_sum = weights[weights > 0].sum()
    if long_sum > 0:
        weights[weights > 0] /= long_sum

    if not long_only:
        short_sum = weights[weights < 0].abs().sum()
        if short_sum > 0:
            weights[weights < 0] /= -short_sum

    # Position caps
    weights = weights.clip(-max_name, max_name)

    # Sector caps
    if sectors is not None and not sectors.empty:
        aligned_sectors = sectors.reindex(weights.index)
        for side_mask in ([weights > 0] if long_only else [weights > 0, weights < 0]):
            side_weights = weights[side_mask]
            if side_weights.empty:
                continue
            sector_exposure = side_weights.abs().groupby(aligned_sectors).sum()
            for sect in sector_exposure[sector_exposure > max_sector].index:
                mask = (aligned_sectors == sect) & side_mask
                scale = max_sector / sector_exposure[sect]
                weights[mask] *= scale

    # Re-normalise long weights to sum to 1
    long_sum = weights[weights > 0].sum()
    if long_sum > 0:
        weights[weights > 0] /= long_sum
    if not long_only:
        short_sum = weights[weights < 0].abs().sum()
        if short_sum > 0:
            weights[weights < 0] /= -short_sum

    return weights


# ═══════════════════════════════════════════════════════════════
# 2. Transaction Cost & Market Impact Model
# ═══════════════════════════════════════════════════════════════

def transaction_costs(
    old_weights: pd.Series,
    new_weights: pd.Series,
    cost_bps: float = 7.5,
) -> float:
    """Proportional transaction costs (both sides)."""
    turnover = (new_weights - old_weights).abs().sum()
    return turnover * cost_bps / 10_000


def market_impact(
    trade_weights: pd.Series,
    daily_vol: pd.Series,
    adv: pd.Series,
    portfolio_nav: float = 1e8,
) -> float:
    """
    Square-root market impact model:
        impact_i = sigma_i * sqrt(|trade_i| / ADV_i)
    Aggregate as the weight-averaged impact.
    """
    trade_dollars = trade_weights.abs() * portfolio_nav
    aligned_adv = adv.reindex(trade_weights.index).replace(0, np.nan)
    aligned_vol = daily_vol.reindex(trade_weights.index).fillna(0)

    impact_per_name = aligned_vol * np.sqrt(trade_dollars / aligned_adv.fillna(1e12))
    return (impact_per_name * trade_weights.abs()).sum()


# ═══════════════════════════════════════════════════════════════
# 3. Walk-Forward Backtest Engine
# ═══════════════════════════════════════════════════════════════

def run_backtest(
    monthly_scores: pd.DataFrame,
    pricing: pd.DataFrame,
    sectors: pd.Series,
    cfg: dict,
) -> Dict[str, pd.DataFrame]:
    """
    Execute the full walk-forward backtest with staggered execution.

    Parameters
    ----------
    monthly_scores : DataFrame with [date, ticker, score] per rebalance.
    pricing        : Long-form daily pricing for return calculation.
    sectors        : Series ticker -> sector for caps.
    cfg            : Full config dict.

    Returns
    -------
    dict with keys:
        portfolio_returns : Series of daily portfolio returns.
        weights_history   : DataFrame of weights per rebalance.
        metrics           : Summary performance metrics.
        turnover_history  : Series of monthly turnover rates.
    """
    ecfg = cfg["execution"]
    stagger = ecfg["stagger_days"]
    cost_bps = ecfg["cost_bps_per_side"]

    rebalance_dates = sorted(monthly_scores["date"].unique())
    prev_weights = pd.Series(dtype=float)

    daily_returns_list = []
    weights_history = []
    turnover_list = []

    price_pivot = pricing.pivot(index="date", columns="ticker", values="adj_close")
    ret_pivot = price_pivot.pct_change()

    for i, reb_date in enumerate(rebalance_dates):
        snapshot = monthly_scores[monthly_scores["date"] == reb_date]
        score_series = snapshot.set_index("ticker")["score"]

        sector_aligned = sectors.reindex(score_series.index)
        target_weights = rank_weighted_portfolio(score_series, sector_aligned, cfg)

        if target_weights.empty:
            continue

        # Staggered execution: blend old->new weights over stagger_days.
        # FIX #2: Start at reb_loc + 1 so signals computed from close on
        # reb_date do NOT earn that same day's return (t+1 execution).
        if reb_date not in ret_pivot.index:
            continue

        reb_loc = ret_pivot.index.get_loc(reb_date)
        exec_start = reb_loc + 1  # t+1 execution lag
        next_reb_loc = (
            ret_pivot.index.get_loc(rebalance_dates[i + 1])
            if i + 1 < len(rebalance_dates) and rebalance_dates[i + 1] in ret_pivot.index
            else len(ret_pivot)
        )

        for day_offset in range(exec_start, next_reb_loc):
            date = ret_pivot.index[day_offset]
            day_ret = ret_pivot.iloc[day_offset]

            days_since_reb = day_offset - exec_start
            if days_since_reb < stagger and not prev_weights.empty:
                blend = (days_since_reb + 1) / stagger
                active = target_weights.reindex(
                    target_weights.index.union(prev_weights.index), fill_value=0
                )
                old = prev_weights.reindex(active.index, fill_value=0)
                current_weights = old * (1 - blend) + active * blend
            else:
                current_weights = target_weights

            port_ret = (current_weights * day_ret.reindex(current_weights.index, fill_value=0)).sum()
            daily_returns_list.append({"date": date, "return": port_ret})

        # Costs
        cost = transaction_costs(prev_weights.reindex(target_weights.index, fill_value=0),
                                 target_weights, cost_bps)
        turnover = (target_weights - prev_weights.reindex(target_weights.index, fill_value=0)).abs().sum()
        turnover_list.append({"date": reb_date, "turnover": turnover, "cost": cost})

        weights_history.append(
            target_weights.to_frame("weight").assign(date=reb_date).reset_index()
        )
        prev_weights = target_weights.copy()

    portfolio_returns = pd.DataFrame(daily_returns_list)
    if portfolio_returns.empty:
        return {"portfolio_returns": pd.Series(dtype=float),
                "weights_history": pd.DataFrame(),
                "metrics": pd.DataFrame(),
                "turnover_history": pd.DataFrame()}

    portfolio_returns = portfolio_returns.set_index("date")["return"]

    # Subtract transaction costs (spread evenly across the month)
    cost_df = pd.DataFrame(turnover_list)
    if not cost_df.empty:
        for _, row in cost_df.iterrows():
            month_mask = (portfolio_returns.index >= row["date"]) & (
                portfolio_returns.index < row["date"] + pd.DateOffset(months=1)
            )
            n_days = month_mask.sum()
            if n_days > 0:
                portfolio_returns[month_mask] -= row["cost"] / n_days

    metrics = compute_performance_metrics(portfolio_returns, cfg)
    wh = pd.concat(weights_history, ignore_index=True) if weights_history else pd.DataFrame()
    th = pd.DataFrame(turnover_list) if turnover_list else pd.DataFrame()

    return {
        "portfolio_returns": portfolio_returns,
        "weights_history": wh,
        "metrics": metrics,
        "turnover_history": th,
    }


# ═══════════════════════════════════════════════════════════════
# 4. Performance Metrics
# ═══════════════════════════════════════════════════════════════

def compute_performance_metrics(
    returns: pd.Series,
    cfg: dict,
    risk_free: float = 0.0,
) -> pd.Series:
    """
    Comprehensive performance metrics for the portfolio return stream.
    """
    if returns.empty:
        return pd.Series(dtype=float)

    ann_factor = 252
    total_ret = (1 + returns).prod() - 1
    ann_ret = (1 + total_ret) ** (ann_factor / len(returns)) - 1
    ann_vol = returns.std() * np.sqrt(ann_factor)
    sharpe = (ann_ret - risk_free) / ann_vol if ann_vol > 0 else 0

    downside = returns[returns < 0].std() * np.sqrt(ann_factor)
    sortino = (ann_ret - risk_free) / downside if downside > 0 else 0

    cumulative = (1 + returns).cumprod()
    rolling_max = cumulative.cummax()
    drawdowns = (cumulative - rolling_max) / rolling_max
    max_dd = drawdowns.min()
    calmar = ann_ret / abs(max_dd) if max_dd != 0 else 0

    hit_rate = (returns > 0).mean()

    monthly_ret = returns.resample("ME").sum() if hasattr(returns.index, "freq") or True else returns
    months_positive = (monthly_ret > 0).mean() if len(monthly_ret) > 0 else 0

    return pd.Series({
        "total_return": total_ret,
        "annualized_return": ann_ret,
        "annualized_volatility": ann_vol,
        "sharpe_ratio": sharpe,
        "sortino_ratio": sortino,
        "max_drawdown": max_dd,
        "calmar_ratio": calmar,
        "daily_hit_rate": hit_rate,
        "monthly_positive_pct": months_positive,
        "n_days": len(returns),
    })


def regime_decomposition(
    returns: pd.Series,
    market_returns: pd.Series,
    cfg: dict,
) -> pd.DataFrame:
    """
    Split performance metrics into bull / bear / sideways regimes.

    Regimes are defined by trailing 6-month cumulative market return.
    """
    thresholds = cfg["reporting"]["regime_thresholds"]
    bull_thresh = thresholds["bull_6m_return"]
    bear_thresh = thresholds["bear_6m_return"]

    market_6m = market_returns.rolling(126).sum()
    aligned = pd.DataFrame({
        "port_ret": returns,
        "mkt_6m": market_6m.reindex(returns.index),
    }).dropna()

    regimes = pd.cut(
        aligned["mkt_6m"],
        bins=[-np.inf, bear_thresh, bull_thresh, np.inf],
        labels=["bear", "sideways", "bull"],
    )

    results = {}
    for regime in ["bear", "sideways", "bull"]:
        mask = regimes == regime
        if mask.sum() < 10:
            continue
        results[regime] = compute_performance_metrics(
            aligned["port_ret"][mask], cfg,
        )

    return pd.DataFrame(results)
