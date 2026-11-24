# SectorPrognosis — Alternative ML Paradigms & Iterative Scaling (v1 → v4)

This folder explores radically different ML paradigms for predicting 1-year S&P 500 sector performance from coupled sector fundamentals and macroeconomic indicators, under **strictly causal, never-mixed / never-interwoven** evaluation.

## Zero-Leakage Protocol (`stationary_pipeline.py`)

Every walk-forward fold enforces hard `assert` guarantees (`assert_strict_purged_split`):
- **Contiguous, non-overlapping blocks**: `Train < [250d purge] < Val < [250d purge] < Test`
- **Zero target overlap**: 250-trading-day purge gaps prevent overlapping 1-year returns from leaking across splits.
- **Strictly causal stationarity**: trending macro levels (`GDPChained`, `CPI`, `CoreCPI`, `CorePCE`, `PPI`, `Jobless Claims`, `HomeSales`) are converted to backward-looking YoY growth + 3M acceleration and normalized with a strictly causal trailing 5-year rolling z-score (using only data $\le t$).
- **Train/Val-only fitting**: all PCAs, scalers, regularization strengths, checkpoints, and blend weights are fit strictly on Train/Val; Test is evaluated once per fold.

## 4-Fold Causal Walk-Forward Progression (`2012–2024`, 4 Unseen Regimes)

| Iteration / Model | RMSE ↓ | Mean $R^2$ ↑ | Dir Acc ↑ | TS IC ↑ | Rot IC ↑ | L/S Spread ↑ | +L/S Folds |
|---|---:|---:|---:|---:|---:|---:|---:|
| `Baseline_Ridge` (raw levels) | 0.2644 | -6.089 | 44.8% | +0.473 | +0.039 | +3.01% | 2/4 |
| `Stationary_Ridge` (v3 causal) | 0.2120 | -2.783 | 46.1% | +0.238 | -0.038 | +1.59% | 2/4 |
| `RoughPath_v1` (10-PC minimal) | 0.1683 | -0.823 | 65.7% | +0.323 | +0.045 | +3.77% | 2/4 |
| `RoughPath_v2` (multi-scale) | 0.1609 | -0.617 | 73.6% | +0.216 | +0.041 | +3.42% | 2/4 |
| `RoughPath_v3` (stationary + cross-Lévy) | 0.1822 | -1.377 | 64.0% | +0.127 | +0.124 | +6.16% | 3/4 |
| **`RoughPath_v4_Decomp`** *(best rotation)* | **0.1633** | **-0.661** | **73.5%** | +0.021 | **+0.172** | **+6.99%** | **3/4** |
| `CoupledOsc_v1` (303 params) | 0.1653 | -1.112 | 62.7% | **+0.358** | +0.056 | +3.50% | 3/4 |
| **`DuffingOsc_v3_Ens`** *(best physics)* | **0.1642** | -0.914 | 64.8% | +0.293 | +0.009 | +0.48% | 3/4 |
| `DuffingOsc_v4_LowRank` *(Fold 4 $R^2=+0.104$)* | 0.1695 | -1.347 | 63.6% | +0.230 | -0.115 | -1.68% | 1/4 |
| **`Champion_ValBlend` (RPv4 + Duffv4)** | **0.1528** | **-0.552** | **74.1%** | +0.245 | +0.093 | +3.70% | 2/4 |

## Reproducing the Runs

```bash
python experiments/run_benchmark.py           # Initial 8-model paradigm survey
python experiments/run_three_new.py           # Minimal tests: EBM, Rough-Path, Coupled Oscillator
python experiments/run_winners_walkforward.py # 4-fold causal walk-forward (v1/v2)
python experiments/run_v3_iteration.py        # 4-fold causal walk-forward (v3 stationary)
python experiments/run_v4_iteration.py        # 4-fold causal walk-forward (v4 champions)
```
