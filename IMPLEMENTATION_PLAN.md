# Implementation Plan — ML2 Group Assignment Enhancements

> **Instructions for Claude Code**: Read this entire file first, then implement each task in order. Each task specifies the exact file to modify and where to insert code. Ask before proceeding if anything is ambiguous. Do NOT modify existing working logic — only ADD new cells/sections.

---

## Context

This is a quantitative cross-sectional stock selection project for an ML2 course. The codebase has:
- `notebooks/quant_cross_sectional_framework.ipynb` — main orchestrator notebook
- `src/data/pit_ingestion.py` — data ingestion
- `src/features/feature_store.py` — feature engineering  
- `src/labels/triple_barrier.py` — labeling
- `src/modeling/cpcv_pipeline.py` — CPCV validation
- `src/modeling/meta_labeling.py` — two-stage XGBoost + Random Forest
- `src/risk/neutralization.py` — risk neutralization
- `src/backtest/execution_sim.py` — backtest engine
- `src/reporting/shap_reporting.py` — SHAP and reporting
- `config/default.yaml` — all hyperparameters

The notebook runs top-to-bottom with a cache-or-compute pattern (parquet files in `data/cache/`).

---

## Task 1: Add EDA Section to the Notebook

**Where**: Insert NEW cells in the notebook AFTER Cell 10 (survivorship check) and BEFORE Cell 12 (Phase 2 markdown).

**What to add** (each bullet = one new code cell):

### Cell: Correlation Heatmap
```python
# ═══ Exploratory Data Analysis ═══
# Correlation heatmap to understand feature relationships
# (This also motivates the PCA step later)

# Merge pricing with a sample date for cross-sectional snapshot
sample_date = pricing['date'].dropna().unique()[len(pricing['date'].unique())//2]
snapshot = pricing[pricing['date'] == sample_date].copy()

# Compute basic features for EDA (before full feature pipeline)
import matplotlib.pyplot as plt
import seaborn as sns

# Use the tech features if already cached, otherwise compute on pricing
if 'tech_features' in dir() and not tech_features.empty:
    eda_data = tech_features
else:
    from src.features.feature_store import compute_technical_features
    eda_data = compute_technical_features(pricing, cfg)

eda_cols = [c for c in eda_data.columns if c not in ['date', 'ticker'] and eda_data[c].dtype in ('float64', 'float32', 'int64')]
sample = eda_data.dropna(subset=eda_cols[:5]).sample(min(5000, len(eda_data)), random_state=42)

fig, ax = plt.subplots(figsize=(14, 10))
corr = sample[eda_cols].corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, cmap='RdBu_r', center=0, vmin=-1, vmax=1,
            annot=False, square=True, ax=ax, cbar_kws={'shrink': 0.8})
ax.set_title('Cross-Sectional Feature Correlation Matrix', fontsize=14)
plt.tight_layout()
plt.show()

print(f'\nHighly correlated pairs (|r| > 0.7):')
high_corr = []
for i in range(len(corr.columns)):
    for j in range(i+1, len(corr.columns)):
        if abs(corr.iloc[i, j]) > 0.7:
            high_corr.append((corr.columns[i], corr.columns[j], corr.iloc[i, j]))
for a, b, r in sorted(high_corr, key=lambda x: -abs(x[2])):
    print(f'  {a} <-> {b}: r={r:.3f}')
```

### Cell: Feature Distributions
```python
# Feature distributions (box plots for cross-sectional spread)
fig, axes = plt.subplots(3, 4, figsize=(16, 10))
axes = axes.flatten()
plot_cols = eda_cols[:12]  # First 12 features

for i, col in enumerate(plot_cols):
    if i >= len(axes):
        break
    sample[col].dropna().hist(bins=50, ax=axes[i], color='steelblue', alpha=0.7, edgecolor='white')
    axes[i].set_title(col, fontsize=10)
    axes[i].axvline(0, color='red', linestyle='--', alpha=0.5)

for j in range(len(plot_cols), len(axes)):
    axes[j].set_visible(False)

plt.suptitle('Feature Distributions (sampled cross-section)', fontsize=14)
plt.tight_layout()
plt.show()
```

### Cell: Missing Data Analysis
```python
# Missing data analysis
missing_pct = eda_data[eda_cols].isnull().mean().sort_values(ascending=False) * 100

fig, ax = plt.subplots(figsize=(12, 5))
missing_pct.plot.bar(ax=ax, color='coral', edgecolor='white')
ax.set_ylabel('% Missing')
ax.set_title('Feature Missing Data Percentage')
ax.axhline(y=50, color='red', linestyle='--', alpha=0.5, label='50% threshold')
ax.legend()
plt.tight_layout()
plt.show()

print(f'Features with >50% missing: {(missing_pct > 50).sum()}')
print(f'Features with 0% missing: {(missing_pct == 0).sum()}')
```

### Cell: Class Balance (add after labels are computed, after Cell 21)
```python
# Class balance visualization
if not labels.empty:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    labels['label_binary'].value_counts().plot.bar(
        ax=axes[0], color=['#d32f2f', '#388e3c'], edgecolor='white')
    axes[0].set_title('Binary Label Distribution')
    axes[0].set_xticklabels(['Short (0)', 'Long (1)'], rotation=0)
    for i, v in enumerate(labels['label_binary'].value_counts().values):
        axes[0].text(i, v + 200, f'{v:,} ({v/len(labels)*100:.1f}%)', ha='center')
    
    # Label balance over time
    monthly_balance = labels.groupby(labels['entry_date'].dt.to_period('Y'))['label_binary'].mean()
    monthly_balance.plot(ax=axes[1], marker='o', color='steelblue')
    axes[1].axhline(0.5, color='red', linestyle='--', alpha=0.5)
    axes[1].set_title('Positive Label Rate Over Time')
    axes[1].set_ylabel('% Positive (Long)')
    
    plt.tight_layout()
    plt.show()
```

---

## Task 2: Implement PCA / Dimensionality Reduction

**Where**: Insert NEW cells in the notebook AFTER Cell 17 (z-score normalization) and BEFORE Cell 19 (Phase 3 markdown).

### Cell: PCA Exploration (scree plot, eigenvalues)
```python
# ═══ Dimensionality Reduction: PCA Analysis ═══
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

print('═' * 60)
print('PCA / Factor Analysis')
print('═' * 60)

# Prepare feature matrix for PCA
pca_feature_cols = [c for c in feature_cols if merged[c].notna().mean() > 0.5]  # Only features with >50% coverage
X_pca_raw = merged[pca_feature_cols].fillna(0).values

scaler_pca = StandardScaler()
X_pca_scaled = scaler_pca.fit_transform(X_pca_raw)

# Fit PCA with ALL components to inspect variance structure
pca_full = PCA()
pca_full.fit(X_pca_scaled)

eigenvalues = pca_full.explained_variance_
cum_var = np.cumsum(pca_full.explained_variance_ratio_)

# Eigenvalue table
eigen_df = pd.DataFrame({
    'Component': range(1, len(eigenvalues) + 1),
    'Eigenvalue': eigenvalues,
    '% Variance': pca_full.explained_variance_ratio_ * 100,
    'Cumulative %': cum_var * 100
})
print('\nEigenvalue Table (top 15 components):')
display(eigen_df.head(15))

# Scree plot
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(range(1, len(eigenvalues) + 1), eigenvalues, 'bo-', markersize=4)
axes[0].axhline(y=1.0, color='red', linestyle='--', label='Kaiser criterion (eigenvalue=1)')
axes[0].set_xlabel('Component Number')
axes[0].set_ylabel('Eigenvalue')
axes[0].set_title('Scree Plot')
axes[0].legend()
axes[0].set_xlim(0, min(20, len(eigenvalues)))

axes[1].plot(range(1, len(cum_var) + 1), cum_var * 100, 'go-', markersize=4)
axes[1].axhline(y=85, color='red', linestyle='--', label='85% threshold')
axes[1].set_xlabel('Number of Components')
axes[1].set_ylabel('Cumulative Variance Explained (%)')
axes[1].set_title('Cumulative Explained Variance')
axes[1].legend()
axes[1].set_xlim(0, min(20, len(cum_var)))

plt.tight_layout()
plt.show()

# Kaiser criterion
n_kaiser = int((eigenvalues > 1).sum())
print(f'\nKaiser criterion: {n_kaiser} components with eigenvalue > 1')
print(f'These {n_kaiser} components explain {cum_var[n_kaiser - 1] * 100:.1f}% of total variance')
```

### Cell: Component Matrix and Communalities
```python
# Select number of components
N_COMPONENTS = n_kaiser
print(f'Selected {N_COMPONENTS} components based on Kaiser criterion')

pca_selected = PCA(n_components=N_COMPONENTS)
pca_selected.fit(X_pca_scaled)

# Component Matrix (factor loadings)
loadings = pd.DataFrame(
    pca_selected.components_.T,
    index=pca_feature_cols,
    columns=[f'PC{i+1}' for i in range(N_COMPONENTS)]
)

print('\n── Component Matrix (Factor Loadings) ──')
print('(Correlation between original variables and extracted components)')
display(loadings.style.background_gradient(cmap='RdBu_r', vmin=-1, vmax=1).format('{:.3f}'))

# Communalities (% of each variable's variance explained by the components)
communalities = pd.Series(
    (loadings ** 2).sum(axis=1),
    name='Communality'
).sort_values(ascending=False)

print('\n── Variable Communalities ──')
print('(Higher = more of the variable\'s information is captured by the components)')
display(communalities.to_frame().style.background_gradient(cmap='YlGn', vmin=0, vmax=1).format('{:.3f}'))

low_communality = communalities[communalities < 0.5]
if len(low_communality) > 0:
    print(f'\n⚠ {len(low_communality)} variables have communality < 0.5 (poorly represented):')
    for var, val in low_communality.items():
        print(f'  {var}: {val:.3f}')

# Component Plot (if 2+ components)
if N_COMPONENTS >= 2:
    fig, ax = plt.subplots(figsize=(10, 8))
    for i, feat in enumerate(pca_feature_cols):
        ax.annotate(feat, (loadings.iloc[i, 0], loadings.iloc[i, 1]), fontsize=8)
        ax.scatter(loadings.iloc[i, 0], loadings.iloc[i, 1], c='steelblue', s=30)
    ax.axhline(0, color='gray', linewidth=0.5)
    ax.axvline(0, color='gray', linewidth=0.5)
    circle = plt.Circle((0, 0), 1, fill=False, color='gray', linestyle='--')
    ax.add_patch(circle)
    ax.set_xlabel(f'PC1 ({pca_selected.explained_variance_ratio_[0]*100:.1f}%)')
    ax.set_ylabel(f'PC2 ({pca_selected.explained_variance_ratio_[1]*100:.1f}%)')
    ax.set_title('Component Plot (Factor Loadings)')
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.set_aspect('equal')
    plt.tight_layout()
    plt.show()
```

### Modify Cell 25: Use PCA features in training (leakage-free)

In the modeling section (Cell 25), AFTER the train/test split but BEFORE training, add a PCA option. **Do NOT replace the existing code** — add a flag to compare both approaches:

```python
# ═══ PCA Feature Transformation (leakage-free) ═══
# Fit scaler + PCA on training data ONLY, then transform both splits

from sklearn.decomposition import PCA as PCA_model
from sklearn.preprocessing import StandardScaler as SS

USE_PCA_FEATURES = False  # Set True to train on PCA features instead of raw

if USE_PCA_FEATURES:
    pca_feat_cols = [c for c in feature_cols if X[c].notna().mean() > 0.5]
    
    scaler_tr = SS()
    X_tr_scaled = scaler_tr.fit_transform(X_tr[pca_feat_cols].fillna(0))
    X_te_scaled = scaler_tr.transform(X_te[pca_feat_cols].fillna(0))
    
    pca_tr = PCA_model(n_components=N_COMPONENTS)
    X_tr_pca = pca_tr.fit_transform(X_tr_scaled)
    X_te_pca = pca_tr.transform(X_te_scaled)
    
    pca_col_names = [f'PC{i+1}' for i in range(N_COMPONENTS)]
    X_tr = pd.DataFrame(X_tr_pca, columns=pca_col_names, index=X_tr.index)
    X_te = pd.DataFrame(X_te_pca, columns=pca_col_names, index=X_te.index)
    
    print(f'Using PCA features: {N_COMPONENTS} components')
    print(f'Variance explained: {sum(pca_tr.explained_variance_ratio_)*100:.1f}%')
else:
    print('Using original features (no PCA transformation)')
```

---

## Task 3: Implement Grid Search with CPCV

**Where**: Create a new utility file `src/modeling/grid_search.py`, then add cells in the notebook BEFORE the main `run_two_stage_pipeline` call (Cell 25).

### File: `src/modeling/grid_search.py` (NEW FILE)
```python
"""
Grid Search with CPCV-aware cross-validation.
Uses the project's purged/embargoed splits instead of random k-fold.
"""
from __future__ import annotations
from math import comb
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier

from src.modeling.cpcv_pipeline import cpcv_splits


class CPCVSplitter:
    """Sklearn-compatible CV splitter using CPCV with purging and embargo."""

    def __init__(
        self,
        dates: pd.Series,
        t_barrier: pd.Series,
        n_groups: int = 6,
        k_test_groups: int = 2,
        embargo_days: int = 5,
    ):
        self.dates = dates
        self.t_barrier = t_barrier
        self.n_groups = n_groups
        self.k_test_groups = k_test_groups
        self.embargo_days = embargo_days

    def split(self, X, y=None, groups=None):
        for train_idx, test_idx in cpcv_splits(
            self.dates, self.t_barrier,
            self.n_groups, self.k_test_groups,
            self.embargo_days,
        ):
            yield train_idx, test_idx

    def get_n_splits(self, X=None, y=None, groups=None):
        return comb(self.n_groups, self.k_test_groups)


def run_grid_search(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    dates_train: pd.Series,
    t_barrier_train: pd.Series,
    cfg: dict,
    param_grid: Optional[dict] = None,
    n_groups: int = 6,
    k_test_groups: int = 2,
    scoring: str = "roc_auc",
) -> dict:
    """
    Run GridSearchCV for XGBoost using CPCV splits.
    
    Returns dict with best_params, best_score, cv_results DataFrame, and fitted model.
    """
    if param_grid is None:
        param_grid = {
            "n_estimators": [500, 1000, 1500],
            "max_depth": [3, 4, 5],
            "learning_rate": [0.01, 0.02, 0.05],
            "min_child_weight": [50, 75, 100],
        }

    cpcv_cv = CPCVSplitter(
        dates=dates_train,
        t_barrier=t_barrier_train,
        n_groups=n_groups,
        k_test_groups=k_test_groups,
        embargo_days=cfg["validation"]["embargo_days"],
    )

    base_model = XGBClassifier(
        use_label_encoder=False,
        eval_metric="logloss",
        reg_alpha=cfg["modeling"]["primary"]["params"].get("reg_alpha", 0.1),
        reg_lambda=cfg["modeling"]["primary"]["params"].get("reg_lambda", 1.0),
        subsample=cfg["modeling"]["primary"]["params"].get("subsample", 0.8),
        colsample_bytree=cfg["modeling"]["primary"]["params"].get("colsample_bytree", 0.8),
    )

    grid = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        cv=cpcv_cv,
        scoring=scoring,
        n_jobs=-1,
        verbose=1,
        refit=True,
    )

    X_clean = X_train.fillna(0)
    grid.fit(X_clean, y_train)

    return {
        "best_params": grid.best_params_,
        "best_score": grid.best_score_,
        "cv_results": pd.DataFrame(grid.cv_results_),
        "best_model": grid.best_estimator_,
        "grid_search": grid,
    }
```

### Notebook cell: Run Grid Search (insert BEFORE Cell 25)

```python
# ═══ Hyperparameter Tuning via Grid Search (CPCV-aware) ═══
from src.modeling.grid_search import run_grid_search

if not model_data.empty and len(model_data) > 100:
    print('═' * 60)
    print('Grid Search with CPCV Cross-Validation')
    print('═' * 60)
    
    X_gs = model_data[feature_cols].copy()
    y_gs = model_data['label_binary'].copy()
    dates_gs = model_data['date'].copy()
    t_barrier_gs = model_data['t_barrier'].copy()
    
    # Use the training portion only (first 80% by date)
    sorted_dates = dates_gs.sort_values()
    split_date = sorted_dates.iloc[int(len(sorted_dates) * 0.8)]
    train_mask_gs = dates_gs <= split_date
    
    # Reduced grid for speed (expand if you have time)
    param_grid = {
        'n_estimators': [500, 1000, 1500],
        'max_depth': [3, 4, 5],
        'learning_rate': [0.01, 0.02, 0.05],
    }
    
    print(f'Parameter grid: {len(param_grid["n_estimators"])} x '
          f'{len(param_grid["max_depth"])} x '
          f'{len(param_grid["learning_rate"])} = '
          f'{np.prod([len(v) for v in param_grid.values()])} combinations')
    
    gs_results = run_grid_search(
        X_train=X_gs[train_mask_gs],
        y_train=y_gs[train_mask_gs],
        dates_train=dates_gs[train_mask_gs],
        t_barrier_train=t_barrier_gs[train_mask_gs],
        cfg=cfg,
        param_grid=param_grid,
        n_groups=6,  # Fewer groups = fewer folds = faster
        k_test_groups=2,
        scoring='roc_auc',
    )
    
    print(f'\n✓ Best Parameters: {gs_results["best_params"]}')
    print(f'✓ Best CV AUC: {gs_results["best_score"]:.4f}')
    
    # Show top 10 parameter combinations
    top_results = gs_results['cv_results'].nsmallest(10, 'rank_test_score')[
        ['params', 'mean_test_score', 'std_test_score', 'rank_test_score']
    ]
    print('\nTop 10 parameter combinations:')
    display(top_results)
```

### Notebook cell: Grid Search Heatmap Visualization
```python
# Grid Search Results Heatmap
if 'gs_results' in dir() and gs_results is not None:
    results_df = gs_results['cv_results']
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Heatmap: max_depth vs learning_rate
    pivot1 = results_df.pivot_table(
        values='mean_test_score',
        index='param_max_depth',
        columns='param_learning_rate',
        aggfunc='max'
    )
    sns.heatmap(pivot1, annot=True, fmt='.4f', cmap='YlOrRd', ax=axes[0])
    axes[0].set_title('Best AUC: max_depth vs learning_rate')
    
    # Heatmap: n_estimators vs learning_rate
    pivot2 = results_df.pivot_table(
        values='mean_test_score',
        index='param_n_estimators',
        columns='param_learning_rate',
        aggfunc='max'
    )
    sns.heatmap(pivot2, annot=True, fmt='.4f', cmap='YlOrRd', ax=axes[1])
    axes[1].set_title('Best AUC: n_estimators vs learning_rate')
    
    plt.suptitle('Grid Search Results (CPCV Cross-Validation)', fontsize=14)
    plt.tight_layout()
    plt.show()
```

### Notebook cell: Update config with best params (add after grid search, before run_two_stage_pipeline)
```python
# Apply best hyperparameters to config for the production run
if 'gs_results' in dir() and gs_results is not None:
    best = gs_results['best_params']
    print('Updating config with grid search best parameters:')
    for k, v in best.items():
        print(f'  {k}: {cfg["modeling"]["primary"]["params"].get(k, "N/A")} → {v}')
        cfg['modeling']['primary']['params'][k] = v
```

---

## Task 4: Add Model Comparison and Classification Metrics

**Where**: Insert NEW cells AFTER the modeling pipeline results (after Cell 25), BEFORE neutralization (Cell 28).

### Cell: Confusion Matrix and Classification Report
```python
# ═══ Classification Metrics ═══
from sklearn.metrics import (confusion_matrix, classification_report,
                             ConfusionMatrixDisplay, roc_curve, auc,
                             precision_score, recall_score, f1_score)

if pipeline_results is not None:
    y_te = pipeline_results['y_test']
    preds = pipeline_results['primary_preds']
    proba = pipeline_results['primary_proba']
    
    # Confusion Matrix
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    cm = confusion_matrix(y_te, preds)
    ConfusionMatrixDisplay(cm, display_labels=['Short (0)', 'Long (1)']).plot(ax=axes[0], cmap='Blues')
    axes[0].set_title('Primary Model — Confusion Matrix')
    
    # ROC Curve
    pos_col = 1 if proba.shape[1] > 1 else 0
    fpr, tpr, thresholds = roc_curve(y_te, proba[:, pos_col])
    roc_auc = auc(fpr, tpr)
    axes[1].plot(fpr, tpr, color='steelblue', lw=2, label=f'XGBoost (AUC = {roc_auc:.3f})')
    axes[1].plot([0, 1], [0, 1], 'k--', lw=1)
    axes[1].set_xlabel('False Positive Rate')
    axes[1].set_ylabel('True Positive Rate')
    axes[1].set_title('ROC Curve')
    axes[1].legend(loc='lower right')
    
    plt.tight_layout()
    plt.show()
    
    # Classification Report
    print('\n── Classification Report ──')
    print(classification_report(y_te, preds, target_names=['Short', 'Long']))
```

### Cell: Multi-Model Comparison
```python
# ═══ Model Comparison: XGBoost vs Random Forest vs Naive Bayes ═══
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, roc_auc_score

if pipeline_results is not None:
    train_mask = pipeline_results['train_mask']
    test_mask = pipeline_results['test_mask']
    
    X_all = model_data[feature_cols].fillna(0)
    y_all = model_data['label_binary']
    
    X_tr_cmp = X_all[train_mask]
    X_te_cmp = X_all[test_mask]
    y_tr_cmp = y_all[train_mask]
    y_te_cmp = y_all[test_mask]
    
    models = {
        'XGBoost (tuned)': pipeline_results['primary_model'],
        'Random Forest': RandomForestClassifier(
            n_estimators=300, max_depth=10, min_samples_leaf=20,
            max_features='sqrt', random_state=42, n_jobs=-1),
        'Naive Bayes': GaussianNB(),
    }
    
    comparison = []
    for name, model in models.items():
        if name != 'XGBoost (tuned)':
            model.fit(X_tr_cmp, y_tr_cmp)
        
        preds = model.predict(X_te_cmp)
        proba = model.predict_proba(X_te_cmp)
        pos_col = 1 if proba.shape[1] > 1 else 0
        
        comparison.append({
            'Model': name,
            'Accuracy': accuracy_score(y_te_cmp, preds),
            'AUC': roc_auc_score(y_te_cmp, proba[:, pos_col]),
            'Precision': precision_score(y_te_cmp, preds, zero_division=0),
            'Recall': recall_score(y_te_cmp, preds, zero_division=0),
            'F1-Score': f1_score(y_te_cmp, preds, zero_division=0),
        })
    
    comp_df = pd.DataFrame(comparison).set_index('Model')
    print('── Model Comparison (Out-of-Sample) ──')
    display(comp_df.style.highlight_max(axis=0, color='lightgreen').format('{:.4f}'))
```

---

## Task 5: Fix the OOF Leakage in meta_labeling.py

**File**: `src/modeling/meta_labeling.py`  
**Function**: `_out_of_fold_primary_predictions()`

Replace the current symmetric k-fold with expanding-window (walk-forward) splits:

```python
def _out_of_fold_primary_predictions(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    dates_train: pd.Series,
    t_barrier_train: pd.Series,
    cfg: dict,
    n_folds: int = 5,
) -> np.ndarray:
    """
    Generate out-of-fold primary-model predictions using WALK-FORWARD
    splits (not symmetric k-fold) to prevent future data leaking into
    early fold predictions.
    
    Each fold's model is trained ONLY on data before the validation window.
    """
    oof_preds = np.full(len(y_train), -1, dtype=int)
    sorted_idx = dates_train.argsort().values
    fold_size = len(sorted_idx) // n_folds

    for k in range(n_folds):
        val_start = k * fold_size
        val_end = (k + 1) * fold_size if k < n_folds - 1 else len(sorted_idx)
        val_idx = sorted_idx[val_start:val_end]

        # WALK-FORWARD: train only on data BEFORE the validation window
        tr_idx = sorted_idx[:val_start]

        if len(tr_idx) < 100:
            # First fold: not enough prior data, use a small expanding window
            # Skip this fold's predictions (leave as -1, will be handled below)
            continue

        model = build_primary_model(cfg, early_stopping=False)
        model.fit(X_train.iloc[tr_idx], y_train.iloc[tr_idx])
        oof_preds[val_idx] = model.predict(X_train.iloc[val_idx])

    # Handle any unfilled predictions (from skipped first fold)
    unfilled = oof_preds == -1
    if unfilled.any():
        # Use the first available model to fill (trained on fold 0+1 data)
        first_valid_fold = 1
        tr_fill = sorted_idx[:first_valid_fold * fold_size]
        if len(tr_fill) >= 100:
            model_fill = build_primary_model(cfg, early_stopping=False)
            model_fill.fit(X_train.iloc[tr_fill], y_train.iloc[tr_fill])
            oof_preds[unfilled] = model_fill.predict(X_train.iloc[np.where(unfilled)[0]])
        else:
            # Last resort: assign majority class
            majority = int(y_train.mode().iloc[0])
            oof_preds[unfilled] = majority

    return oof_preds
```

---

## Implementation Order

1. Task 5 (fix leakage) — smallest change, biggest integrity impact
2. Task 1 (EDA) — visual, easy, addresses missing rubric criterion
3. Task 2 (PCA) — core course material, high grade impact
4. Task 3 (Grid Search) — core course material, high grade impact
5. Task 4 (metrics + model comparison) — standard ML evaluation

After all tasks: re-run the notebook top-to-bottom (delete cache files first: `rm data/cache/*.parquet data/cache/*.json`) to ensure everything works end-to-end.
