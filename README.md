# Advanced Quantitative Framework for Cross-Sectional Stock Selection

**Algorithmic efficacy, risk neutralization, and execution-aware backtesting** — a full pipeline from bias-aware data ingestion to portfolio simulation.

This repository is the codebase for a Machine Learning II capstone: a cross-sectional equity engine that answers *which names to overweight or underweight next month* using fundamentals, technicals, and macro context, with safeguards appropriate for non-stationary markets.

---

## What this project does

| Layer | Role |
|--------|------|
| **Data** | Historical universe (e.g. S&P 500), prices via `yfinance`, point-in-time fundamentals via `edgartools`, macro series via FRED |
| **Features** | Cross-sectional preprocessing (winsorization, z-scores, optional fractional differencing), technical and fundamental factor sets |
| **Labels** | **Triple-barrier** labels — path-aware, volatility-scaled outcomes instead of fixed horizons |
| **Models** | Primary classifier (default: **XGBoost**) + **meta-labeling** (default: Random Forest) for direction vs. sizing |
| **Validation** | **Combinatorial Purged Cross-Validation (CPCV)** — purged/embargoed splits to limit leakage in time-series panels |
| **Risk** | Cross-sectional **neutralization** (e.g. sector, size, beta) so predictions are not confounded with simple exposures |
| **Backtest** | Costs, **staggered execution**, and optional market-impact style constraints — not frictionless same-bar fills |

The main narrative, methodology, and charts live in the notebook; reusable logic is in `src/`.

---

## Repository layout

```
groupassigenement/
├── config/
│   └── default.yaml          # Single source of truth: universe, features, models, CPCV, costs
├── data/
│   └── cache/                # Generated parquet/json (gitignored — see below)
├── notebooks/
│   └── quant_cross_sectional_framework.ipynb
├── src/
│   ├── data/                 # Config loading, PIT ingestion
│   ├── features/             # Feature store & transforms
│   ├── labels/               # Triple-barrier labeling
│   ├── modeling/             # CPCV + meta-labeling
│   ├── risk/                 # Neutralization
│   ├── backtest/             # Execution simulation
│   └── reporting/            # SHAP and diagnostics
└── requirements.txt
```

---

## Quick start

### 1. Environment

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. API keys (optional but recommended)

- **FRED** — set `FRED_API_KEY` in the environment, or put the key in `config/default.yaml` under `data.fred_api_key` for local runs (do **not** commit secrets).

### 3. Run the project

Open `notebooks/quant_cross_sectional_framework.ipynb` and run top-to-bottom. The notebook adds the project root to `sys.path` whether you start Jupyter from the repo root or from `notebooks/`.

All tunable parameters (universe dates, barrier multipliers, XGBoost hyperparameters, CPCV groups, transaction costs, etc.) are driven from **`config/default.yaml`**.

---

## Data cache (why the repo stays small)

Intermediate artifacts — merged panels, features, labels, model inputs — are written under **`data/cache/`** as parquet/json. That directory is **`.gitignore`d** because it is large and reproducible from code + config.

After cloning:

1. Ensure dependencies and config are set.
2. Run the notebook (or your own script calling the same `src` modules) to rebuild `data/cache/`.

There is nothing proprietary in omitting the cache: it is derived data, not source code.

---

## Configuration at a glance

`config/default.yaml` groups:

- **Universe** — index, date range, liquidity/size floors  
- **Features** — technical windows, fundamental ratios, preprocessing and optional dimensionality reduction  
- **Labels** — triple-barrier parameters and volatility settings  
- **Modeling** — primary + meta algorithms and hyperparameters  
- **Validation** — CPCV `n_groups`, test fold count, purge/embargo  
- **Risk** — neutralization exposures and strength  
- **Execution** — bps costs, impact model, rebalance staggering  

Change one YAML and keep runs comparable across machines.

---

## Dependencies — notes

- **Core stack**: NumPy, Pandas, scikit-learn, XGBoost, SHAP, PyYAML, Jupyter.  
- **Optional / platform-sensitive** packages (TA-Lib, `fracdiff`, commercial `mlfinlab`) are commented in `requirements.txt`. CPCV is implemented in-repo so a paid `mlfinlab` license is not required.

---

## Academic / integrity context

The companion notebook includes an explicit **leakage audit** mindset (PIT fundamentals, purged CV, barrier-aligned horizons). When you present results, tie claims to the config hash and the CPCV protocol you actually ran.

---

## Author

Course project — **Machine Learning II** — cross-sectional equity ML with institutional-style controls.
