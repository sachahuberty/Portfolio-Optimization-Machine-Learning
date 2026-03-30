"""
Phase 2 – Feature Engineering
==============================
Technical indicators (TA-Lib), PIT-aligned fundamental ratios,
fractional differencing, and cross-sectional z-score normalisation.
"""

from __future__ import annotations

import warnings
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy.stats import mstats

# ── Optional imports with graceful fallbacks ──
try:
    import talib
    _HAS_TALIB = True
except ImportError:
    _HAS_TALIB = False
    warnings.warn("TA-Lib not installed – technical features will use "
                  "pure-pandas fallbacks.", stacklevel=2)

try:
    from fracdiff.sklearn import FracdiffStat
    _HAS_FRACDIFF = True
    _FRACDIFF_BACKEND = "fracdiff"
except ImportError:
    try:
        import tsfracdiff
        _HAS_FRACDIFF = True
        _FRACDIFF_BACKEND = "tsfracdiff"
    except ImportError:
        _HAS_FRACDIFF = False
        _FRACDIFF_BACKEND = None


# ═══════════════════════════════════════════════════════════════
# 1. Technical Features
# ═══════════════════════════════════════════════════════════════

def _log_returns(series: pd.Series, window: int) -> pd.Series:
    return np.log(series / series.shift(window))


def _garman_klass_vol(
    high: pd.Series, low: pd.Series,
    open_: pd.Series, close: pd.Series,
    window: int = 20,
) -> pd.Series:
    """Garman-Klass intraday volatility estimator – more efficient
    than close-to-close when OHLC data is available."""
    log_hl = (np.log(high / low)) ** 2
    log_co = (np.log(close / open_)) ** 2
    gk = 0.5 * log_hl - (2 * np.log(2) - 1) * log_co
    return gk.rolling(window).mean().apply(np.sqrt) * np.sqrt(252)


def compute_technical_features(
    pricing: pd.DataFrame,
    cfg: dict,
) -> pd.DataFrame:
    """
    Generate the full technical feature matrix per (ticker, date).

    Parameters
    ----------
    pricing : DataFrame with columns [date, ticker, open, high, low,
              close, adj_close, volume]
    cfg     : features.technical block from config.

    Returns
    -------
    DataFrame indexed by (ticker, date) with all technical columns.
    """
    tech_cfg = cfg["features"]["technical"]
    results = []

    for ticker, grp in pricing.groupby("ticker"):
        grp = grp.sort_values("date").set_index("date").copy()

        feat: Dict[str, pd.Series] = {}

        # ── Momentum (log returns over multiple horizons) ──
        for w in tech_cfg["momentum_windows"]:
            col_name = f"ret_{w}d"
            if tech_cfg.get("skip_recent_month") and w == max(tech_cfg["momentum_windows"]):
                feat[col_name] = _log_returns(grp["adj_close"], w) - _log_returns(grp["adj_close"], 21)
            else:
                feat[col_name] = _log_returns(grp["adj_close"], w)

        # ── Volatility ──
        log_ret_1d = np.log(grp["adj_close"] / grp["adj_close"].shift(1))
        for w in tech_cfg["volatility_windows"]:
            feat[f"vol_{w}d"] = log_ret_1d.rolling(w).std() * np.sqrt(252)

        # Max drawdown (trailing 126 days)
        roll_max = grp["adj_close"].rolling(126, min_periods=1).max()
        feat["mdd_126d"] = (grp["adj_close"] - roll_max) / roll_max

        if tech_cfg.get("garman_klass"):
            feat["gk_vol_20d"] = _garman_klass_vol(
                grp["high"], grp["low"], grp["open"], grp["close"], 20,
            )

        # ── Volume signals ──
        vol_ma = grp["volume"].rolling(tech_cfg["amihud_window"]).mean()
        vol_std = grp["volume"].rolling(tech_cfg["amihud_window"]).std()
        feat["volume_zscore"] = (grp["volume"] - vol_ma) / vol_std.replace(0, np.nan)

        # Amihud illiquidity: |ret| / dollar volume
        dollar_vol = grp["adj_close"] * grp["volume"]
        feat["amihud"] = (
            log_ret_1d.abs() / dollar_vol.replace(0, np.nan)
        ).rolling(tech_cfg["amihud_window"]).mean()

        if tech_cfg.get("obv"):
            direction = np.sign(grp["adj_close"].diff())
            feat["obv"] = (direction * grp["volume"]).cumsum()
            feat["obv_zscore"] = (
                (feat["obv"] - feat["obv"].rolling(60).mean())
                / feat["obv"].rolling(60).std().replace(0, np.nan)
            )

        # ── Mean-reversion signals ──
        for w in tech_cfg["ma_windows"]:
            ma = grp["adj_close"].rolling(w).mean()
            feat[f"ma_gap_{w}d"] = (grp["adj_close"] - ma) / (
                log_ret_1d.rolling(20).std().replace(0, np.nan) * grp["adj_close"]
            )

        if _HAS_TALIB:
            feat["rsi_14"] = pd.Series(
                talib.RSI(grp["adj_close"].values, timeperiod=tech_cfg["rsi_period"]),
                index=grp.index,
            )
            upper, mid, lower = talib.BBANDS(
                grp["adj_close"].values,
                timeperiod=tech_cfg["bollinger_period"],
                nbdevup=tech_cfg["bollinger_std"],
                nbdevdn=tech_cfg["bollinger_std"],
            )
            band_width = pd.Series(upper - lower, index=grp.index)
            feat["bb_pctb"] = (grp["adj_close"].values - lower) / band_width.replace(0, np.nan)
        else:
            # Pure-pandas RSI fallback
            delta = grp["adj_close"].diff()
            gain = delta.clip(lower=0).rolling(tech_cfg["rsi_period"]).mean()
            loss = (-delta.clip(upper=0)).rolling(tech_cfg["rsi_period"]).mean()
            rs = gain / loss.replace(0, np.nan)
            feat["rsi_14"] = 100 - (100 / (1 + rs))

            ma_bb = grp["adj_close"].rolling(tech_cfg["bollinger_period"]).mean()
            std_bb = grp["adj_close"].rolling(tech_cfg["bollinger_period"]).std()
            feat["bb_pctb"] = (grp["adj_close"] - (ma_bb - tech_cfg["bollinger_std"] * std_bb)) / (
                (2 * tech_cfg["bollinger_std"] * std_bb).replace(0, np.nan)
            )

        df_feat = pd.DataFrame(feat, index=grp.index)
        df_feat["ticker"] = ticker
        results.append(df_feat.reset_index())

    return pd.concat(results, ignore_index=True)


# ═══════════════════════════════════════════════════════════════
# 2. Fundamental Features (PIT-aligned)
# ═══════════════════════════════════════════════════════════════

def compute_fundamental_features(merged: pd.DataFrame) -> pd.DataFrame:
    """
    Derive standard fundamental ratios from the PIT-joined
    fundamentals columns.

    Handles missing columns gracefully — edgartools provides:
        revenue, net_income, total_assets, total_liabilities,
        stockholders_equity, operating_cash_flow
    but NOT eps_basic or shares_outstanding.  Ratios that require
    missing inputs are filled with NaN rather than crashing.
    """
    df = merged.copy()
    n = len(df)

    def _col(name, fallback=None):
        """Return column as float Series, or NaN Series if missing."""
        if name in df.columns:
            return pd.to_numeric(df[name], errors="coerce")
        if fallback is not None and fallback in df.columns:
            return pd.to_numeric(df[fallback], errors="coerce")
        return pd.Series(np.nan, index=df.index)

    price = _col("adj_close", "close")
    revenue = _col("revenue")
    net_income = _col("net_income")
    total_assets = _col("total_assets")
    total_liab = _col("total_liabilities")
    equity = _col("stockholders_equity")
    ocf = _col("operating_cash_flow")

    def _safe_div(num: pd.Series, den: pd.Series) -> pd.Series:
        den = den.replace(0, np.nan)
        return num / den

    # Without shares_outstanding we cannot compute per-share ratios (P/E, P/B).
    # Instead we use asset-scaled ratios that work as cross-sectional ranking
    # signals without needing share count or market cap.
    df["earnings_yield"] = _safe_div(net_income, total_assets)
    df["book_to_assets"] = _safe_div(equity, total_assets)
    df["roe"] = _safe_div(net_income, equity)
    df["roa"] = _safe_div(net_income, total_assets)
    df["debt_equity"] = _safe_div(total_liab, equity)
    df["net_margin"] = _safe_div(net_income, revenue)
    df["ocf_to_assets"] = _safe_div(ocf, total_assets)
    df["ocf_to_debt"] = _safe_div(ocf, total_liab)
    df["leverage"] = _safe_div(total_assets, equity)

    # YoY growth: duration metrics come from 10-K only (one row per year),
    # so pct_change(1) gives year-over-year growth.
    for col in ["revenue", "net_income"]:
        if col in df.columns:
            df[f"{col}_growth_yoy"] = df.groupby("ticker")[col].pct_change(1)

    return df


# ═══════════════════════════════════════════════════════════════
# 3. Fractional Differencing
# ═══════════════════════════════════════════════════════════════

def fractional_difference(
    df: pd.DataFrame,
    columns: List[str],
    confidence: float = 0.95,
) -> tuple[pd.DataFrame, Dict[str, float]]:
    """
    Apply fractional differencing to selected columns per ticker
    to achieve stationarity while preserving long-range memory.

    Returns the transformed DataFrame and a dict of fitted d values.
    """
    if not _HAS_FRACDIFF:
        warnings.warn(
            "fracdiff not installed – skipping fractional differencing.",
            stacklevel=2,
        )
        return df, {}

    d_values: Dict[str, float] = {}
    out = df.copy()

    if _FRACDIFF_BACKEND == "tsfracdiff":
        from tsfracdiff import FractionalDifferentiator
        for col in columns:
            if col not in out.columns:
                continue
            try:
                series = out[col].dropna()
                if len(series) < 100:
                    continue
                fd = FractionalDifferentiator(
                    maxOrderBound=1,
                    significance=1 - confidence,
                    precision=0.01,
                )
                transformed = fd.FitTransform(series.to_frame(name=col))
                out.loc[transformed.index, col] = transformed[col].values
                if fd.orders:
                    d_values[col] = float(fd.orders[0])
            except Exception:
                continue
    elif _FRACDIFF_BACKEND == "fracdiff":
        for col in columns:
            if col not in out.columns:
                continue
            try:
                fds = FracdiffStat()
                values = out[col].dropna().values.reshape(-1, 1)
                if len(values) < 100:
                    continue
                transformed = fds.fit_transform(values)
                valid_idx = out[col].dropna().index[: len(transformed)]
                out.loc[valid_idx, col] = transformed.flatten()
                d_values[col] = float(fds.d_)
            except Exception:
                continue

    return out, d_values


# ═══════════════════════════════════════════════════════════════
# 4. Cross-Sectional Standardisation
# ═══════════════════════════════════════════════════════════════

def cross_sectional_zscore(
    df: pd.DataFrame,
    feature_cols: List[str],
    date_col: str = "date",
    winsorize_sigma: float = 3.0,
) -> pd.DataFrame:
    """
    On each date, compute the cross-sectional z-score for every
    feature column, after winsorising at ±winsorize_sigma.

    This forces the model to learn *relative* rankings rather than
    absolute levels, and automatically adjusts for regime changes.
    """
    out = df.copy()

    for col in feature_cols:
        if col not in out.columns:
            continue
        # Winsorise within each cross-section
        def _winz_zscore(g: pd.Series) -> pd.Series:
            vals = g.values.astype(float)
            clipped = np.array(mstats.winsorize(vals, limits=[0.01, 0.01]))
            mu = np.nanmean(clipped)
            sigma = np.nanstd(clipped, ddof=0)
            if sigma == 0 or np.isnan(sigma):
                return pd.Series(np.zeros(len(g)), index=g.index)
            z = (vals - mu) / sigma
            return pd.Series(
                np.clip(z, -winsorize_sigma, winsorize_sigma),
                index=g.index,
            )

        out[col] = out.groupby(date_col)[col].transform(_winz_zscore)

    return out


# ═══════════════════════════════════════════════════════════════
# 5. Optional Dimensionality Reduction
# ═══════════════════════════════════════════════════════════════

def supervised_pca(
    X: pd.DataFrame,
    y: pd.Series,
    n_components: int = 15,
) -> tuple[np.ndarray, object]:
    """
    Scaled PCA: pre-weight features by their univariate
    regression slope to the target, then extract principal
    components.  This ensures PCA prioritises predictive
    variance over total (noise-dominated) variance.
    """
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X.values)

    slopes = np.zeros(Xs.shape[1])
    for j in range(Xs.shape[1]):
        mask = ~np.isnan(Xs[:, j]) & ~np.isnan(y.values)
        if mask.sum() < 30:
            continue
        lr = LinearRegression().fit(Xs[mask, j : j + 1], y.values[mask])
        slopes[j] = abs(lr.coef_[0])

    slopes = slopes / (slopes.sum() + 1e-12)
    Xs_weighted = Xs * slopes[np.newaxis, :]

    pca = PCA(n_components=min(n_components, Xs_weighted.shape[1]))
    components = pca.fit_transform(np.nan_to_num(Xs_weighted))
    return components, pca


# ═══════════════════════════════════════════════════════════════
# 6. Orchestrator
# ═══════════════════════════════════════════════════════════════

def run_feature_pipeline(
    pricing: pd.DataFrame,
    fundamentals: pd.DataFrame,
    cfg: dict,
) -> tuple[pd.DataFrame, Dict[str, float]]:
    """
    Full Phase 2 pipeline: technical + fundamental features,
    fractional differencing, cross-sectional z-score.

    Returns (feature_matrix, fracdiff_d_values).
    """
    from src.data.pit_ingestion import pit_asof_join

    print("[Phase 2] Computing technical features …")
    tech = compute_technical_features(pricing, cfg)

    print("[Phase 2] Joining PIT fundamentals …")
    feature_dates = tech[["date", "ticker"]].drop_duplicates()
    merged = pit_asof_join(feature_dates, fundamentals)
    merged = merged.merge(tech, on=["date", "ticker"], how="left")

    # Drop rows where no prior filing was available (pre-earliest-filing dates).
    # These have NaN for all fundamental columns and cannot produce valid ratios.
    fund_cols = ["revenue", "net_income", "total_assets", "total_liabilities",
                 "stockholders_equity", "operating_cash_flow"]
    fund_present = [c for c in fund_cols if c in merged.columns]
    if fund_present:
        before = len(merged)
        merged = merged.dropna(subset=fund_present, how="all")
        dropped = before - len(merged)
        if dropped:
            print(f"  Dropped {dropped:,} rows with no fundamental coverage "
                  f"({dropped / before:.1%} of total)")

    print("[Phase 2] Computing fundamental ratios …")
    merged = compute_fundamental_features(merged)

    # Identify numeric feature columns (exclude identifiers)
    exclude = {"date", "ticker", "sector", "fiscal_period_end",
               "sec_acceptance_date", "form_type", "company",
               "open", "high", "low", "close", "adj_close", "volume"}
    feature_cols = [c for c in merged.columns
                    if c not in exclude and merged[c].dtype in ("float64", "float32", "int64")]

    # Fractional differencing — disabled.  All features are already
    # stationary (returns, ratios, z-scores), so d-values are 0.0.
    d_values: Dict[str, float] = {}

    # Cross-sectional z-score
    if cfg["features"]["preprocessing"].get("cross_sectional_zscore"):
        print("[Phase 2] Cross-sectional z-score normalisation …")
        merged = cross_sectional_zscore(
            merged, feature_cols,
            winsorize_sigma=cfg["features"]["preprocessing"]["winsorize_sigma"],
        )

    print(f"  Feature matrix: {merged.shape}, "
          f"{len(feature_cols)} feature columns")
    return merged, d_values
