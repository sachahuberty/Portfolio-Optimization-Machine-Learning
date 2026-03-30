"""
Bet sizing parameter sweep — fast, no retraining needed.

Reuses the existing pipeline results from a notebook run and only
varies the bet sizing + portfolio construction parameters.

Usage:
    cd ML2_FINAL_PROJECT_FINANCE_CRACKS
    python scripts/param_sweep.py
"""
import sys
import os
import warnings

warnings.filterwarnings("ignore")

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

import copy
import itertools
import numpy as np
import pandas as pd
import yaml

from src.modeling.meta_labeling import compute_bet_sizes
from src.risk.neutralization import run_neutralization
from src.backtest.execution_sim import run_backtest

# ── Load config ──
with open("config/default.yaml") as f:
    BASE_CFG = yaml.safe_load(f)

# ── Sweep grid ──
GRID = {
    "method": ["half_kelly", "meta_prob", "equal"],
    "min_meta_probability": [0.49, 0.51, 0.53, 0.55, 0.57],
    "long_leg": ["top_decile", "top_vigintile"],
    "max_single_name_pct": [0.05, 0.10, 0.15],
}


def main():
    print("Loading cached pipeline data ...")
    pricing = pd.read_parquet("data/cache/pricing.parquet")
    universe = pd.read_parquet("data/cache/universe.parquet")
    features = pd.read_parquet("data/cache/features.parquet")
    labels = pd.read_parquet("data/cache/labels.parquet")
    sectors = universe.drop_duplicates("ticker").set_index("ticker")["sector"]

    # ── Rebuild model data (same as notebook) ──
    model_data = features.merge(
        labels[["ticker", "entry_date", "label_binary", "t_barrier"]],
        left_on=["ticker", "date"],
        right_on=["ticker", "entry_date"],
        how="inner",
    )

    y = model_data["label_binary"]
    dates = model_data["date"]

    feature_cols = [
        c for c in model_data.columns
        if model_data[c].dtype in ("float64", "float32", "int64")
        and c not in (
            "label_binary", "label", "entry_price",
            "return_at_touch", "ewma_vol",
            "upper_barrier", "lower_barrier",
        )
    ]
    X = model_data[feature_cols].fillna(0)

    # ── Train/test split ──
    sorted_dates = dates.sort_values()
    split_date = sorted_dates.iloc[int(len(sorted_dates) * 0.8)]
    train_mask = dates <= split_date
    test_mask = dates > split_date

    X_tr, y_tr = X[train_mask], y[train_mask]
    X_te = X[test_mask]
    dates_tr = dates[train_mask]
    t_barrier_tr = model_data["t_barrier"][train_mask]

    # ── Train models once ──
    print("Training primary + meta models (one time) ...")
    from src.modeling.meta_labeling import (
        build_primary_model, train_primary_model,
        build_meta_model, train_meta_model,
        construct_meta_labels, _out_of_fold_primary_predictions,
    )

    tr_sorted = dates_tr.sort_values()
    val_split = tr_sorted.iloc[int(len(tr_sorted) * 0.9)]
    tr_fit = dates_tr <= val_split
    tr_val = dates_tr > val_split

    primary = build_primary_model(BASE_CFG, early_stopping=True)
    primary = train_primary_model(
        primary,
        X_tr[tr_fit.values], y_tr[tr_fit.values],
        X_tr[tr_val.values], y_tr[tr_val.values],
    )
    primary_preds_test = primary.predict(X_te)
    direction = np.where(primary_preds_test == 1, 1, -1).astype(float)

    # OOF for meta
    oof_preds = _out_of_fold_primary_predictions(
        X_tr, y_tr, dates_tr, t_barrier_tr, BASE_CFG, n_folds=5,
    )
    oof_valid = oof_preds != -1
    meta_target = construct_meta_labels(
        oof_preds[oof_valid], y_tr.iloc[oof_valid],
    )
    meta_model = build_meta_model(BASE_CFG)
    meta_model = train_meta_model(
        meta_model, X_tr.iloc[oof_valid], meta_target,
    )
    meta_proba = meta_model.predict_proba(X_te)
    meta_prob_correct = (
        meta_proba[:, 1] if meta_proba.shape[1] > 1
        else meta_proba[:, 0]
    )

    # ── Prepare test_df for neutralization (once) ──
    test_df_base = model_data[test_mask].copy().reset_index(drop=True)
    sector_map = universe.drop_duplicates("ticker").set_index("ticker")["sector"]
    if "sector" not in test_df_base.columns:
        test_df_base["sector"] = (
            test_df_base["ticker"].map(sector_map).fillna("Unknown")
        )
    price_col = (
        "adj_close" if "adj_close" in test_df_base.columns else "close"
    )
    if "volume" in test_df_base.columns:
        test_df_base["log_market_cap"] = np.log1p(
            test_df_base[price_col].fillna(0)
            * test_df_base["volume"].fillna(0)
        )
    else:
        test_df_base["log_market_cap"] = 0.0
    test_df_base["beta"] = (
        test_df_base["vol_20d"]
        if "vol_20d" in test_df_base.columns
        else 0.0
    )

    print("Models trained. Starting sweep ...\n")

    # ── Sweep ──
    combos = list(itertools.product(
        GRID["method"],
        GRID["min_meta_probability"],
        GRID["long_leg"],
        GRID["max_single_name_pct"],
    ))

    print(f"Total combinations: {len(combos)}\n")
    results = []

    for i, (method, min_prob, long_leg, max_name) in enumerate(combos):
        cfg = copy.deepcopy(BASE_CFG)
        cfg["modeling"]["bet_sizing"]["method"] = method
        cfg["modeling"]["bet_sizing"]["min_meta_probability"] = min_prob
        cfg["portfolio"]["long_leg"] = long_leg
        cfg["portfolio"]["max_single_name_pct"] = max_name

        bet_sizes = compute_bet_sizes(direction, meta_prob_correct, cfg)

        test_df = test_df_base.copy()
        test_df["raw_score"] = bet_sizes
        neutralized, _ = run_neutralization(test_df, "raw_score", cfg)
        test_df["neutralized_score"] = neutralized

        monthly_scores = test_df[["date", "ticker", "neutralized_score"]].copy()
        monthly_scores.rename(
            columns={"neutralized_score": "score"}, inplace=True,
        )
        monthly_scores = monthly_scores.dropna(subset=["score"])

        bt = run_backtest(monthly_scores, pricing, sectors, cfg)

        if bt["portfolio_returns"].empty:
            continue

        m = bt["metrics"]
        non_zero = np.count_nonzero(bet_sizes)

        row = {
            "method": method,
            "min_meta_prob": min_prob,
            "long_leg": long_leg,
            "max_name_pct": max_name,
            "sharpe": m.get("sharpe_ratio", np.nan),
            "total_return": m.get("total_return", np.nan),
            "ann_return": m.get("annualized_return", np.nan),
            "ann_vol": m.get("annualized_volatility", np.nan),
            "sortino": m.get("sortino_ratio", np.nan),
            "max_dd": m.get("max_drawdown", np.nan),
            "calmar": m.get("calmar_ratio", np.nan),
            "hit_rate": m.get("daily_hit_rate", np.nan),
            "n_positions": non_zero,
        }
        results.append(row)
        print(
            f"  [{i+1}/{len(combos)}] "
            f"{method:10s} prob={min_prob:.2f} "
            f"leg={long_leg:15s} cap={max_name:.2f} → "
            f"Sharpe={row['sharpe']:+.3f}  "
            f"Return={row['total_return']:+.1%}  "
            f"DD={row['max_dd']:+.1%}"
        )

    # ── Results ──
    df = pd.DataFrame(results)
    out_path = "data/sweep_results.csv"
    df.to_csv(out_path, index=False)

    print(f"\n{'=' * 70}")
    print(f"DONE — {len(df)} runs saved to {out_path}")
    print(f"{'=' * 70}")

    if not df.empty:
        print("\n── Top 10 by Sharpe ──")
        top = df.nlargest(10, "sharpe")
        print(top.to_string(index=False))

        print("\n── Avg Sharpe by parameter ──")
        for col in ["method", "min_meta_prob", "long_leg", "max_name_pct"]:
            grp = df.groupby(col)["sharpe"].mean().sort_values(ascending=False)
            print(f"\n  {col}:")
            for val, s in grp.items():
                print(f"    {val}: {s:.4f}")


if __name__ == "__main__":
    main()
