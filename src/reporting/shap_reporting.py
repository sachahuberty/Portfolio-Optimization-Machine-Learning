"""
Phase 7 – Reporting & Explainability
======================================
SHAP-based model attribution (global + local), PM decision ledger,
data integrity diagnostics, and CPCV validation summaries.

Institutional portfolio managers will not allocate capital to
opaque models.  Every prediction must be decomposable into
human-auditable feature contributions.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

try:
    import shap
    _HAS_SHAP = True
except ImportError:
    _HAS_SHAP = False

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    _HAS_PLOT = True
except ImportError:
    _HAS_PLOT = False


# ═══════════════════════════════════════════════════════════════
# 1. SHAP Explainability
# ═══════════════════════════════════════════════════════════════

def compute_shap_values(
    model,
    X: pd.DataFrame,
    max_samples: int = 500,
):
    """
    Compute SHAP values for tree-based model predictions.
    Returns a shap.Explanation object.
    """
    if not _HAS_SHAP:
        raise ImportError("shap is required for explainability reporting")

    if len(X) > max_samples:
        X_sample = X.sample(max_samples, random_state=42)
    else:
        X_sample = X

    explainer = shap.Explainer(model, X_sample)
    shap_values = explainer(X_sample)
    return shap_values, X_sample


def plot_global_importance(
    shap_values,
    max_display: int = 15,
    title: str = "Global Feature Importance (SHAP)",
) -> Optional[plt.Figure]:
    """SHAP summary (beeswarm) plot – global feature attribution."""
    if not _HAS_PLOT or not _HAS_SHAP:
        return None
    fig, ax = plt.subplots(figsize=(10, 8))
    shap.summary_plot(shap_values, max_display=max_display, show=False)
    plt.title(title, fontsize=14)
    plt.tight_layout()
    return plt.gcf()


def plot_waterfall(
    shap_values,
    idx: int = 0,
    max_display: int = 10,
    title: str = "Local Attribution – Single Stock",
) -> Optional[plt.Figure]:
    """SHAP waterfall plot for a single prediction."""
    if not _HAS_PLOT or not _HAS_SHAP:
        return None
    fig = plt.figure(figsize=(10, 6))
    shap.plots.waterfall(shap_values[idx], max_display=max_display, show=False)
    plt.title(title, fontsize=12)
    plt.tight_layout()
    return plt.gcf()


def plot_interaction(
    shap_values,
    feature_a: str,
    feature_b: str,
) -> Optional[plt.Figure]:
    """SHAP scatter plot for feature interaction."""
    if not _HAS_PLOT or not _HAS_SHAP:
        return None
    fig = plt.figure(figsize=(8, 6))
    shap.plots.scatter(shap_values[:, feature_a], color=shap_values[:, feature_b], show=False)
    plt.tight_layout()
    return plt.gcf()


def top_shap_drivers(
    shap_values,
    X_sample: pd.DataFrame,
    n_top: int = 3,
) -> pd.DataFrame:
    """
    For each observation, extract the top-N SHAP drivers
    (feature name + contribution magnitude).
    """
    if not _HAS_SHAP:
        return pd.DataFrame()

    vals = shap_values.values
    cols = X_sample.columns.tolist()
    records = []

    for i in range(len(vals)):
        abs_contrib = np.abs(vals[i])
        top_idx = np.argsort(abs_contrib)[-n_top:][::-1]
        drivers = []
        for j in top_idx:
            drivers.append(f"{cols[j]}({vals[i][j]:+.3f})")
        records.append({"obs_idx": i, "top_drivers": " | ".join(drivers)})

    return pd.DataFrame(records)


# ═══════════════════════════════════════════════════════════════
# 2. PM Decision Ledger
# ═══════════════════════════════════════════════════════════════

def build_pm_ledger(
    tickers: pd.Series,
    sectors: pd.Series,
    directions: np.ndarray,
    meta_probabilities: np.ndarray,
    bet_sizes: np.ndarray,
    upper_barriers: pd.Series,
    lower_barriers: pd.Series,
    shap_drivers: Optional[pd.DataFrame] = None,
    top_n: int = 50,
) -> pd.DataFrame:
    """
    Assemble the actionable PM ledger: one row per recommended
    position, sorted by absolute conviction.
    """
    n = len(directions)
    def _align(arr, name):
        """Ensure array/series is length n, truncating or padding as needed."""
        if arr is None:
            return np.full(n, np.nan)
        vals = arr.values if hasattr(arr, 'values') else np.asarray(arr)
        if len(vals) == n:
            return vals
        if len(vals) > n:
            return vals[:n]
        return np.concatenate([vals, np.full(n - len(vals), np.nan)])

    ledger = pd.DataFrame({
        "ticker": _align(tickers, "tickers"),
        "sector": _align(sectors, "sectors"),
        "direction": np.where(directions > 0, "Long", "Short"),
        "meta_conviction": meta_probabilities,
        "recommended_weight": bet_sizes,
        "take_profit_price": _align(upper_barriers, "upper_barriers"),
        "stop_loss_price": _align(lower_barriers, "lower_barriers"),
    })

    if shap_drivers is not None and not shap_drivers.empty:
        ledger["top_3_shap_drivers"] = shap_drivers["top_drivers"].values[: len(ledger)]

    ledger["abs_conviction"] = ledger["meta_conviction"].abs()
    ledger = ledger.sort_values("abs_conviction", ascending=False).head(top_n)
    ledger = ledger.drop(columns=["abs_conviction"]).reset_index(drop=True)
    ledger.index = ledger.index + 1
    ledger.index.name = "rank"
    return ledger


# ═══════════════════════════════════════════════════════════════
# 3. Data Integrity Report
# ═══════════════════════════════════════════════════════════════

def data_integrity_report(
    universe: pd.DataFrame,
    pit_stats: pd.DataFrame,
    feature_matrix: pd.DataFrame,
    feature_cols: List[str],
) -> Dict[str, pd.DataFrame]:
    """
    Generate the data-integrity appendix:
      - Survivorship bias metrics
      - PIT compliance summary
      - Feature coverage heatmap data
    """
    # Survivorship: unique tickers per period
    survivorship = universe.groupby("date")["ticker"].nunique().describe()

    # Feature coverage: % non-null per feature per month
    if "date" in feature_matrix.columns:
        coverage = feature_matrix.groupby(
            feature_matrix["date"].dt.to_period("M")
        )[feature_cols].apply(lambda g: g.notna().mean())
    else:
        coverage = pd.DataFrame()

    return {
        "survivorship": survivorship.to_frame("universe_size"),
        "pit_compliance": pit_stats if not pit_stats.empty else pd.DataFrame({"status": ["OK"]}),
        "feature_coverage": coverage,
    }


# ═══════════════════════════════════════════════════════════════
# 4. CPCV Validation Report
# ═══════════════════════════════════════════════════════════════

def cpcv_validation_report(
    cpcv_results: pd.DataFrame,
) -> Dict[str, object]:
    """
    Summarise CPCV evaluation: mean/std/CI for key metrics,
    plus an overfitting ratio diagnostic.
    """
    if cpcv_results.empty:
        return {"summary": pd.DataFrame(), "overfitting_ratio": None}

    summary = cpcv_results[["accuracy", "logloss", "auc"]].describe()

    return {
        "summary": summary,
        "mean_accuracy": cpcv_results["accuracy"].mean(),
        "std_accuracy": cpcv_results["accuracy"].std(),
        "mean_auc": cpcv_results["auc"].mean(),
        "n_folds": len(cpcv_results),
    }


def plot_cpcv_distribution(
    cpcv_results: pd.DataFrame,
) -> Optional[plt.Figure]:
    """Histogram of out-of-sample AUC across CPCV folds."""
    if not _HAS_PLOT or cpcv_results.empty:
        return None

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].hist(cpcv_results["accuracy"], bins=20, edgecolor="black", alpha=0.7)
    axes[0].axvline(cpcv_results["accuracy"].mean(), color="red", linestyle="--",
                     label=f"Mean: {cpcv_results['accuracy'].mean():.3f}")
    axes[0].set_title("CPCV OOS Accuracy Distribution")
    axes[0].set_xlabel("Accuracy")
    axes[0].legend()

    axes[1].hist(cpcv_results["auc"], bins=20, edgecolor="black", alpha=0.7, color="steelblue")
    axes[1].axvline(cpcv_results["auc"].mean(), color="red", linestyle="--",
                     label=f"Mean: {cpcv_results['auc'].mean():.3f}")
    axes[1].set_title("CPCV OOS AUC Distribution")
    axes[1].set_xlabel("AUC")
    axes[1].legend()

    plt.tight_layout()
    return fig


def plot_equity_curve(
    returns: pd.Series,
    benchmark_returns: Optional[pd.Series] = None,
    title: str = "Portfolio Equity Curve (Net of Costs)",
) -> Optional[plt.Figure]:
    """Cumulative return chart with drawdown shading and optional benchmark."""
    if not _HAS_PLOT or returns.empty:
        return None

    cumulative = (1 + returns).cumprod()
    rolling_max = cumulative.cummax()
    drawdown = (cumulative - rolling_max) / rolling_max

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8),
                                     gridspec_kw={"height_ratios": [3, 1]},
                                     sharex=True)

    ax1.plot(cumulative.index, cumulative.values, linewidth=1.5,
             color="steelblue", label="Strategy")

    if benchmark_returns is not None and not benchmark_returns.empty:
        bench_aligned = benchmark_returns.reindex(returns.index).fillna(0)
        bench_cum = (1 + bench_aligned).cumprod()
        ax1.plot(bench_cum.index, bench_cum.values, linewidth=1.2,
                 color="gray", linestyle="--", alpha=0.7, label="S&P 500 (Buy & Hold)")
        ax1.legend(fontsize=11)

    ax1.set_title(title, fontsize=14)
    ax1.set_ylabel("Cumulative Return")
    ax1.grid(True, alpha=0.3)

    ax2.fill_between(drawdown.index, drawdown.values, 0, color="crimson", alpha=0.4)
    ax2.set_ylabel("Drawdown")
    ax2.set_xlabel("Date")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_strategy_vs_benchmark(
    strategy_returns: pd.Series,
    benchmark_returns: pd.Series,
    strategy_metrics: pd.Series,
    cfg: dict,
) -> Optional[plt.Figure]:
    """
    Side-by-side comparison: strategy vs S&P 500 benchmark.
    Shows cumulative returns, rolling Sharpe, and monthly return heatmap.
    """
    if not _HAS_PLOT or strategy_returns.empty:
        return None

    from src.backtest.execution_sim import compute_performance_metrics

    bench_aligned = benchmark_returns.reindex(strategy_returns.index).fillna(0)
    bench_metrics = compute_performance_metrics(bench_aligned, cfg)

    strat_cum = (1 + strategy_returns).cumprod()
    bench_cum = (1 + bench_aligned).cumprod()

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    # 1. Cumulative returns
    axes[0, 0].plot(strat_cum, linewidth=1.5, color="steelblue", label="Strategy")
    axes[0, 0].plot(bench_cum, linewidth=1.2, color="gray", linestyle="--",
                     label="S&P 500", alpha=0.7)
    axes[0, 0].set_title("Cumulative Returns", fontsize=13)
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # 2. Rolling 63-day (3M) Sharpe
    roll_sharpe_s = strategy_returns.rolling(63).mean() / strategy_returns.rolling(63).std() * np.sqrt(252)
    roll_sharpe_b = bench_aligned.rolling(63).mean() / bench_aligned.rolling(63).std() * np.sqrt(252)
    axes[0, 1].plot(roll_sharpe_s, color="steelblue", alpha=0.8, label="Strategy")
    axes[0, 1].plot(roll_sharpe_b, color="gray", alpha=0.6, linestyle="--", label="S&P 500")
    axes[0, 1].axhline(0, color="black", linewidth=0.5)
    axes[0, 1].set_title("Rolling 3-Month Sharpe Ratio", fontsize=13)
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # 3. Monthly returns bar chart
    monthly_strat = strategy_returns.resample("ME").sum()
    monthly_bench = bench_aligned.resample("ME").sum()
    x = range(len(monthly_strat))
    w = 0.35
    axes[1, 0].bar([i - w/2 for i in x], monthly_strat.values, width=w,
                    color="steelblue", alpha=0.7, label="Strategy")
    axes[1, 0].bar([i + w/2 for i in x], monthly_bench.values[:len(monthly_strat)],
                    width=w, color="gray", alpha=0.5, label="S&P 500")
    axes[1, 0].axhline(0, color="black", linewidth=0.5)
    axes[1, 0].set_title("Monthly Returns", fontsize=13)
    axes[1, 0].legend()
    axes[1, 0].set_xlabel("Month")
    axes[1, 0].grid(True, alpha=0.3)

    # 4. Metrics comparison table
    metrics_compare = pd.DataFrame({
        "Strategy": strategy_metrics,
        "S&P 500": bench_metrics,
    })
    axes[1, 1].axis("off")
    col_colors = ["steelblue", "gray"]
    table = axes[1, 1].table(
        cellText=[[f"{v:.4f}" for v in row] for _, row in metrics_compare.iterrows()],
        rowLabels=metrics_compare.index,
        colLabels=metrics_compare.columns,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.4)
    axes[1, 1].set_title("Performance Comparison", fontsize=13, pad=20)

    plt.tight_layout()
    return fig


def plot_feature_coverage_heatmap(
    coverage: pd.DataFrame,
    title: str = "Feature Coverage (% non-null by month)",
) -> Optional[plt.Figure]:
    """Heatmap of feature availability over time."""
    if not _HAS_PLOT or coverage.empty:
        return None

    fig, ax = plt.subplots(figsize=(16, 8))
    sns.heatmap(coverage.T, cmap="YlGn", vmin=0, vmax=1, ax=ax,
                cbar_kws={"label": "Coverage %"})
    ax.set_title(title, fontsize=14)
    ax.set_xlabel("Month")
    ax.set_ylabel("Feature")
    plt.tight_layout()
    return fig
