# SectorPrognosis

Leakage-free S&P 500 sector 1-year return forecasting using historical Bloomberg sector fundamentals (2000–2024) and US macroeconomic indicators.

## Key Improvements (v2.0)

### 1. Eliminated Critical Data Leakage from Legacy Notebooks
An audit of the original notebooks (`SectorPrognosisGridSearch.ipynb`, `SectorPrognosisRollingWindow.ipynb`) identified four leakage and data-quality issues that inflated historical validation $R^2$ while causing out-of-sample collapse:
1. **Global scaling & PCA**: `MinMaxScaler` and `PCA(21)` were fit on the full dataset (including validation and test).
2. **Missing purge gap for overlapping 1-year targets**: Adjacent daily rows share 249/250 days of their 1-year return horizon; splitting adjacent indices leaks nearly a full year of target outcomes across train/val/test.
3. **Look-ahead feature in RollingWindow**: `halfyearPrior = arr[i + 125]` leaked mid-horizon prices into predicting the 1-year return from `i + 250` to `i`.
4. **Source Excel anomaly (documented)**: In `MergedDataSectorPrognosis.xlsx`, the `HealthCare` price column is identical to `IT` (`4609.52` on 2024-12-31), while `HealthCare PE`/`Performance` etc. are distinct. We preserve the exact 6 target columns for fair comparison.

All splits are now enforced by [`src/sector_prognosis/splits.py`](src/sector_prognosis/splits.py) with a strict **250-trading-day purge + 5-day embargo** (`realized_date < eval_start`) and verified by automated unit tests in [`tests/test_no_leakage.py`](tests/test_no_leakage.py) (including bitwise feature causality and corrupted-target invariance tests).

### 2. Prior-Anchored Decomposed Architecture (`src/sector_prognosis/models_v2.py`)
Instead of predicting raw non-stationary 1-year returns with a 4M-parameter BatchNorm MLP on global PCA features, we decompose the 1-year forward return into:

$$y_t^{(s)} = \underbrace{\mu_{\text{train}}^{(s)}}_{\text{Expanding Sector Prior}} + w_m \cdot \underbrace{\hat{r}_{\text{mkt}, t}}_{\text{Bounded Macro-Timing}} + w_\alpha \cdot \underbrace{\hat{r}_{\alpha, t}^{(s)}}_{\text{Pooled Cross-Sectional Alpha}}$$

- **Strictly causal feature engineering** ([`src/sector_prognosis/features.py`](src/sector_prognosis/features.py)): Curated 16 stationary macro/housing/inflation/volatility market signals + 20 cross-sectional valuation, quality, ROE momentum, and cyclical-macro interaction features.
- **Robust training**: Training-only `RobustScaler` + Winsorization, 5%–95% target Winsorization (preventing 2008 GFC outliers from dominating), weekly stride subsampling, and mild exponential recency weighting.
- **Purged inner-validation shrinkage**: Shrinkage weights $(w_m, w_\alpha)$ are calibrated strictly on a 250-day-purged inner validation tail inside each training fold — never touching outer validation or test.
- **Modern model families**: CatBoost (oblivious trees), LightGBM (Huber), XGBoost (Pseudo-Huber), ElasticNet (L1/L2 sparse factor selection), plus `SectorFormer` (Cross-Sector Attention + TFT-style Gated Residual Networks in [`src/sector_prognosis/models_neural.py`](src/sector_prognosis/models_neural.py)).

## Verified Out-of-Sample Results (Zero Leakage)

### Purged Walk-Forward Dev Folds (2012–2019, 4 × 2-year regimes)
| Model | OOS $R^2$ vs Train Mean | Cross-Sec Rel $R^2$ | RMSE | Dir Acc | TS-IC | CS-IC | Top-2 Excess 1Y | Top-2 vs Bot-2 Spread | Win Rate vs EW |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `OriginalResNetPCA` (Leakage-Free) | -10.4663 | -0.8044 | 0.3643 | 18.5% | +0.296 | -0.050 | -0.69% | -0.58% | 39.6% |
| `NaiveMean` (Expanding Prior) | 0.0000 | +0.0483 | 0.1206 | 83.3% | 0.000 | +0.230 | +3.91% | +7.40% | 63.5% |
| **`V2_CatBoost`** | **+0.0295** | **+0.0564** | **0.1188** | **83.3%** | **+0.271** | +0.235 | +3.74% | +7.56% | 62.5% |
| **`Hybrid_CB_Enet`** | +0.0187 | +0.0429 | 0.1193 | **83.3%** | +0.163 | **+0.274** | **+4.57%** | **+8.37%** | **74.0%** |
| **`Champion_Blend`** | +0.0263 | +0.0560 | 0.1189 | **83.3%** | +0.262 | +0.227 | +3.92% | +7.40% | 62.5% |

### Sealed Held-Out Test Set (2020–2024: COVID + Inflation/Rate-Hike Regimes)
*(Locked strictly from 2012–2019 Dev results; zero test data ever entered training or tuning.)*

- **Test Regime 1 (`2020–2021`, Trained Strictly Pre-2020)**:
  - `V2_CatBoost`: **OOS $R^2$ = +0.1547**, **Mean $R^2$ = +0.0631**, **TS-IC = +0.5549**, **RMSE = 0.2131** (vs `0.2317` Naive).
  - `Hybrid_CB_Enet`: **OOS $R^2$ = +0.1578**, **Rel $R^2$ = +0.0368**, **TS-IC = +0.4951**, **RMSE = 0.2127**.
- **Test Regime 2 (`2022–2024`, Purged Expanding Walk-Forward, `train <= 2020-12`)**:
  - `Champion_Blend`: **Rel $R^2$ = +0.2675** (vs `+0.2373` Naive, `+0.1034` Original), **CS-IC = +0.4588**, **Top-2 Excess = +8.83%/yr**, **Top-2 vs Bot-2 Spread = +14.06%/yr**, **Win Rate vs EW = 79.2%**.
  - `V2_CatBoost`: **Rel $R^2$ = +0.2792**, **Top-2 Excess = +8.58%/yr**, **Win Rate vs EW = 79.2%**.

### Final Champion (`BetaScaledUncertaintyEnsembleV9`, 12-Year Generalization Audit `2012–2024`)
Combining **Causal Vasicek-Shrunk CAPM Beta Decomposition**, a **4-family uncertainty-gated market ensemble** (`TempSectorFormer + CausalTemporalGRU + CausalMultiScaleTCN + CatBoost`), causal `Energy`/`Energy − Utilities`/systemic-correlation signals, and a **3-family beta-neutral alpha engine** (`ElasticNet + CatBoost + LightGBM`) achieves **$p = 0.0032$** out-of-sample error reduction over `NaiveMean` across 144 monthly observations (Diebold-Mariano Newey-West HAC, `lag = 12` months):
- **Strictly Frozen Pre-2020 Test Mode**: **`DM = +2.732`, $p = 0.0032$** (`OOS R² = +5.74%`, `Rel R² = +6.23%`, `TS-IC = +0.319`, `CS-IC = +0.266`, `Top-2 vs Bot-2 = +8.80%/yr`, **6/6 sectors positive OOS $R^2$**).
- **Purged Expanding Walk-Forward Mode**: **`DM = +2.695`, $p = 0.0035$** (`OOS R² = +5.88%`, `Rel R² = +7.51%`, `TS-IC = +0.322`, `CS-IC = +0.263`, `Top-2 vs Bot-2 = +8.78%/yr`, **6/6 sectors positive OOS $R^2$**).

## Quickstart & Reproduction

Place the three Excel datasets (`MergedDataSectorPrognosis.xlsx`, `EconomicDataforModel.xlsx`, `PreparedDataSectorPrognosis.xlsx`) in `data/`, then:

```bash
uv venv .venv
uv pip install --python .venv/bin/python -e .

# 1. Verify zero leakage & feature causality (9 automated tests)
.venv/bin/pytest -v

# 2. Run Iteration 9 Ultimate Champion 12-year Diebold-Mariano generalization audit
.venv/bin/python -m sector_prognosis.iteration9

# 3. Run V2 tabular/baseline head-to-head comparison
.venv/bin/python -m sector_prognosis.evaluate_final
```

For optional cloud GPU sweeps (Modal or Kaggle), see [`src/sector_prognosis/modal_runner.py`](src/sector_prognosis/modal_runner.py) and [`kaggle_kernel/train_gpu_sweep.py`](kaggle_kernel/train_gpu_sweep.py).
