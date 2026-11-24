"""
Leak-free, structured data pipeline for SectorPrognosis experiments.

Fixes key methodological issues in the original notebooks:
1. Chronological ascending order (oldest -> newest).
2. Strict train-only feature scaling (RobustScaler + clipping).
3. 250-trading-day purge gaps between Train / Val / Test so overlapping
   1-year forward return targets never leak across splits.
4. Structured tensor outputs:
   - X_sec:   (N, T_steps, 6, F_sec)  per-sector temporal trajectories
   - X_macro: (N, T_steps, F_macro)   macroeconomic temporal trajectories
   - Y:       (N, 6)                  forward 250-day returns
   - Y_multi: (N, 4, 6)               intermediate returns at +63, +126, +189, +250 days
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler

SECTORS: List[str] = [
    "HealthCare",
    "Materials",
    "Industrials",
    "ConsumerStaples",
    "ConsumerDiscretionary",
    "Utilities",
]

# Common fundamental feature suffixes shared across all 6 sectors (Utilities lacks PriceToFCF)
COMMON_SECTOR_FEATURES: List[str] = [
    "Performance",
    "Dividends",
    "PE",
    "PriceToBook",
    "ROE",
    "PriceToSales",
    "BestPE",
    "DebtToEquity",
    "DebtToEBITDA",
    "GrossMargin",
    "EBITDAMargin",
    "ProfitMargin",
    "EV/EBITDA",
]

DEFAULT_DATA_DIR = "/tmp/sector_data"
DEFAULT_CACHE_PATH = "/tmp/sector_data/processed_cache.npz"


@dataclass
class SectorDataset:
    dates: np.ndarray            # (N,) datetime64[D]
    X_sec: np.ndarray            # (N, T_steps, 6, F_sec)
    X_macro: np.ndarray          # (N, T_steps, F_macro)
    X_flat: np.ndarray           # (N, D_flat) flattened latest + lagged summary for baselines
    Y: np.ndarray                # (N, 6) forward 250-day return
    Y_multi: np.ndarray          # (N, 4, 6) forward returns at [63, 126, 189, 250] days
    future_idx: np.ndarray       # (N, 4) indices into raw sequence for world model targets
    train_idx: np.ndarray
    val_idx: np.ndarray
    test_idx: np.ndarray
    macro_names: List[str]
    sector_feature_names: List[str]


def _load_raw_dataframes(data_dir: str = DEFAULT_DATA_DIR) -> Tuple[pd.DataFrame, List[str]]:
    merged_path = os.path.join(data_dir, "MergedDataSectorPrognosis.xlsx")
    econ_path = os.path.join(data_dir, "EconomicDataforModel.xlsx")

    df = pd.read_excel(merged_path)
    econ = pd.read_excel(econ_path, sheet_name="FullData")

    df["Date"] = pd.to_datetime(df["Date"])
    econ["Date"] = pd.to_datetime(econ["Date"])

    macro_cols = [c for c in econ.columns if c != "Date"]

    # Keep only the 6 core sectors + Date
    keep_cols = ["Date"] + SECTORS.copy()
    for s in SECTORS:
        for feat in COMMON_SECTOR_FEATURES:
            col = f"{s} {feat}"
            if col in df.columns:
                keep_cols.append(col)

    df = df[keep_cols].copy()
    merged = pd.merge(df, econ, on="Date", how="inner")
    merged = merged.dropna().sort_values("Date", ascending=True).reset_index(drop=True)
    return merged, macro_cols


def load_sector_dataset(
    data_dir: str = DEFAULT_DATA_DIR,
    cache_path: str = DEFAULT_CACHE_PATH,
    horizon: int = 250,
    lookback: int = 250,
    n_steps: int = 16,
    purge_gap: int = 250,
    force_rebuild: bool = False,
) -> SectorDataset:
    """Load, structure, split (with purge gaps), and train-only scale the dataset."""

    if os.path.exists(cache_path) and not force_rebuild:
        cached = np.load(cache_path, allow_pickle=True)
        raw_dates = cached["raw_dates"]
        prices = cached["prices"]
        sec_raw = cached["sec_raw"]
        macro_raw = cached["macro_raw"]
        macro_names = list(cached["macro_names"])
        sec_feat_names = list(cached["sec_feat_names"])
    else:
        merged, macro_cols = _load_raw_dataframes(data_dir)
        raw_dates = merged["Date"].values.astype("datetime64[D]")
        prices = merged[SECTORS].values.astype(np.float32)  # (T_total, 6)

        # Build (T_total, 6, F_sec) where features include common fundamentals + log-price momentum
        T_total = len(merged)
        sec_feat_names = COMMON_SECTOR_FEATURES + ["Ret21d", "Ret63d", "Vol21d"]
        F_sec = len(sec_feat_names)
        sec_raw = np.zeros((T_total, len(SECTORS), F_sec), dtype=np.float32)

        for si, s in enumerate(SECTORS):
            for fi, feat in enumerate(COMMON_SECTOR_FEATURES):
                col = f"{s} {feat}"
                sec_raw[:, si, fi] = merged[col].values.astype(np.float32)

            # Add trailing 1-month return, 3-month return, and 1-month realized vol (strictly backward-looking)
            p = prices[:, si]
            log_p = np.log(np.maximum(p, 1e-6))
            ret1 = np.diff(log_p, prepend=log_p[0])
            ret21 = np.zeros_like(p)
            ret21[21:] = (p[21:] - p[:-21]) / np.maximum(p[:-21], 1e-6)
            ret63 = np.zeros_like(p)
            ret63[63:] = (p[63:] - p[:-63]) / np.maximum(p[:-63], 1e-6)
            vol21 = pd.Series(ret1).rolling(21, min_periods=1).std().fillna(0.0).values.astype(np.float32)

            sec_raw[:, si, -3] = ret21
            sec_raw[:, si, -2] = ret63
            sec_raw[:, si, -1] = vol21

        # Add cyclical month encoding (sin, cos) to macro features
        months = pd.to_datetime(raw_dates).month.values.astype(np.float32)
        m_sin = np.sin(2 * np.pi * months / 12.0)[:, None]
        m_cos = np.cos(2 * np.pi * months / 12.0)[:, None]
        macro_raw = np.concatenate(
            [merged[macro_cols].values.astype(np.float32), m_sin, m_cos], axis=1
        )
        macro_names = macro_cols + ["MonthSin", "MonthCos"]

        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        np.savez_compressed(
            cache_path,
            raw_dates=raw_dates,
            prices=prices,
            sec_raw=sec_raw,
            macro_raw=macro_raw,
            macro_names=np.array(macro_names, dtype=object),
            sec_feat_names=np.array(sec_feat_names, dtype=object),
        )

    T_total = len(raw_dates)
    # Valid sample indices t: need [t - lookback, t] for input and [t, t + horizon] for target
    valid_t = np.arange(lookback, T_total - horizon)
    N = len(valid_t)

    # Purged chronological split:
    # Reserve two purge gaps of `purge_gap` days so target horizons NEVER overlap
    usable = N - 2 * purge_gap
    n_test = int(usable * 0.15)
    n_val = int(usable * 0.15)
    n_train = usable - n_val - n_test

    train_idx = np.arange(0, n_train)
    val_start = n_train + purge_gap
    val_idx = np.arange(val_start, val_start + n_val)
    test_start = val_start + n_val + purge_gap
    test_idx = np.arange(test_start, N)

    # Fit scalers STRICTLY on raw indices up to the end of train_idx (no future info!)
    train_raw_end = valid_t[train_idx[-1]] + 1

    # Scale each sector's features using train-only RobustScaler
    sec_scaled = np.zeros_like(sec_raw, dtype=np.float32)
    for si in range(len(SECTORS)):
        scaler = RobustScaler()
        scaler.fit(sec_raw[:train_raw_end, si, :])
        sec_scaled[:, si, :] = np.clip(scaler.transform(sec_raw[:, si, :]), -5.0, 5.0)

    macro_scaler = RobustScaler()
    macro_scaler.fit(macro_raw[:train_raw_end, :])
    macro_scaled = np.clip(macro_scaler.transform(macro_raw), -5.0, 5.0).astype(np.float32)

    # Build lookback sequences sampled evenly over [t - lookback, t]
    step_offsets = np.linspace(-lookback, 0, n_steps, dtype=int)  # 16 points over trailing year
    multi_horizons = [63, 126, 189, horizon]

    F_sec = sec_scaled.shape[2]
    F_macro = macro_scaled.shape[1]

    X_sec = np.zeros((N, n_steps, len(SECTORS), F_sec), dtype=np.float32)
    X_macro = np.zeros((N, n_steps, F_macro), dtype=np.float32)
    Y = np.zeros((N, len(SECTORS)), dtype=np.float32)
    Y_multi = np.zeros((N, len(multi_horizons), len(SECTORS)), dtype=np.float32)
    future_idx = np.zeros((N, len(multi_horizons)), dtype=np.int64)

    for i, t in enumerate(valid_t):
        idxs = t + step_offsets
        X_sec[i] = sec_scaled[idxs]
        X_macro[i] = macro_scaled[idxs]
        p0 = prices[t]
        for hi, h in enumerate(multi_horizons):
            Y_multi[i, hi] = (prices[t + h] - p0) / np.maximum(p0, 1e-6)
            future_idx[i, hi] = min(i + h, N - 1)
        Y[i] = Y_multi[i, -1]

    # Flattened multi-lag snapshot for MLP / Ridge baselines (5 lags matching original repo spirit)
    lag_indices = [0, 4, 8, 12, 15]
    sec_flat = X_sec[:, lag_indices, :, :].reshape(N, -1)
    mac_flat = X_macro[:, lag_indices, :].reshape(N, -1)
    X_flat = np.concatenate([sec_flat, mac_flat], axis=1).astype(np.float32)

    return SectorDataset(
        dates=raw_dates[valid_t],
        X_sec=X_sec,
        X_macro=X_macro,
        X_flat=X_flat,
        Y=Y,
        Y_multi=Y_multi,
        future_idx=future_idx,
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
        macro_names=macro_names,
        sector_feature_names=sec_feat_names,
    )


if __name__ == "__main__":
    ds = load_sector_dataset(force_rebuild=True)
    print(f"Total samples: {len(ds.dates)}")
    print(f"X_sec:   {ds.X_sec.shape}, X_macro: {ds.X_macro.shape}, X_flat: {ds.X_flat.shape}")
    print(f"Y:       {ds.Y.shape}, Y_multi: {ds.Y_multi.shape}")
    print(f"Train: {len(ds.train_idx)} ({ds.dates[ds.train_idx[0]]} .. {ds.dates[ds.train_idx[-1]]})")
    print(f"Val:   {len(ds.val_idx)}   ({ds.dates[ds.val_idx[0]]} .. {ds.dates[ds.val_idx[-1]]})")
    print(f"Test:  {len(ds.test_idx)}  ({ds.dates[ds.test_idx[0]]} .. {ds.dates[ds.test_idx[-1]]})")
    print(f"Any NaNs? X_sec={np.isnan(ds.X_sec).any()}, X_macro={np.isnan(ds.X_macro).any()}, Y={np.isnan(ds.Y).any()}")
