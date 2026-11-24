"""
Strictly Causal, Stationary Data Pipeline for SectorPrognosis (v3).

Solves the multi-decade non-stationarity problem (e.g., raw CPI / GDP / HomeSales
drifting monotonically upward from 2001 to 2024) while guaranteeing ZERO leakage
and ZERO train/test mixing or interweaving:

1. Converts trending macro level series (GDP, CPI, CoreCPI, CorePCE, PPI,
   Total Jobless Claims, Existing/New Home Sales) into strictly backward-looking:
     - 1-year (250d) log-growth / YoY rate
     - 3-month (63d) acceleration (change in growth rate)
2. Keeps naturally bounded macro indicators (UnemploymentRate, ISMManufacturing,
   ISMService, EmploymentRatio, JoblessClaimsYoY) as level + 63d change + 250d change.
3. Normalizes every feature using a STRICTLY CAUSAL trailing 5-year (1250-day)
   expanding/rolling z-score using ONLY data <= t (so 2022 is measured relative
   to 2017-2022, never peeking into the future and never leaking across splits!).
4. Enforces hard assertions on 250-trading-day purge gaps between contiguous
   Train, Val, and Test blocks so targets NEVER overlap or interweave.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd

from experiments.data_pipeline import DEFAULT_CACHE_PATH, SECTORS, load_sector_dataset

# Macro columns that trend monotonically as raw levels -> transform to growth + acceleration
TRENDING_MACRO_COLS = {
    "GDPChained",
    "CPI",
    "CoreCPI",
    "CorePCE",
    "PPI",
    "Total Jobless Claims",
    "ExistingHomeSales",
    "NewHomeSales",
}


def causal_rolling_zscore(arr: np.ndarray, window: int = 1250, min_periods: int = 126) -> np.ndarray:
    """
    Strictly causal rolling z-score: output[t] uses ONLY arr[max(0, t-window+1) : t+1].
    Zero future data ever enters output[t].
    """
    df = pd.DataFrame(arr)
    roll_mean = df.rolling(window=window, min_periods=min_periods).mean()
    roll_std = df.rolling(window=window, min_periods=min_periods).std().replace(0.0, 1.0)
    # For the very first `min_periods` rows, fall back to expanding mean/std (still strictly <= t)
    exp_mean = df.expanding(min_periods=2).mean()
    exp_std = df.expanding(min_periods=2).std().replace(0.0, 1.0)
    mean = roll_mean.fillna(exp_mean).fillna(0.0)
    std = roll_std.fillna(exp_std).fillna(1.0)
    z = ((df - mean) / std).fillna(0.0).values.astype(np.float32)
    return np.clip(z, -4.0, 4.0)


def build_stationary_macro(macro_raw: np.ndarray, macro_names: List[str]) -> Tuple[np.ndarray, List[str]]:
    """Transform raw macro matrix into strictly stationary backward-looking features."""
    T, M = macro_raw.shape
    feats = []
    names = []

    for j, name in enumerate(macro_names):
        col = macro_raw[:, j].astype(np.float64)
        if name in ("MonthSin", "MonthCos"):
            feats.append(col.astype(np.float32))
            names.append(name)
        elif name in TRENDING_MACRO_COLS:
            # 1-year YoY percentage change (strictly backward-looking)
            yoy = np.zeros(T, dtype=np.float32)
            denom = np.maximum(np.abs(col[:-250]), 1e-6)
            yoy[250:] = (col[250:] - col[:-250]) / denom
            # 3-month acceleration of YoY rate
            acc = np.zeros(T, dtype=np.float32)
            acc[63:] = yoy[63:] - yoy[:-63]
            feats.extend([yoy, acc])
            names.extend([f"{name}_YoY", f"{name}_Acc3m"])
        else:
            # Bounded rate/index: keep level + 3m change + 1y change
            d3m = np.zeros(T, dtype=np.float32)
            d3m[63:] = (col[63:] - col[:-63]).astype(np.float32)
            d1y = np.zeros(T, dtype=np.float32)
            d1y[250:] = (col[250:] - col[:-250]).astype(np.float32)
            feats.extend([col.astype(np.float32), d3m, d1y])
            names.extend([name, f"{name}_d3m", f"{name}_d1y"])

    mat = np.stack(feats, axis=1)
    # Apply strictly causal trailing 5-year z-score to all non-cyclical columns
    out = np.zeros_like(mat, dtype=np.float32)
    non_cyc = [i for i, n in enumerate(names) if n not in ("MonthSin", "MonthCos")]
    out[:, non_cyc] = causal_rolling_zscore(mat[:, non_cyc], window=1250)
    for i, n in enumerate(names):
        if n in ("MonthSin", "MonthCos"):
            out[:, i] = mat[:, i]
    return out, names


@dataclass
class StationarySeries:
    dates: np.ndarray        # (N,) datetime64[D]
    X_sec: np.ndarray        # (N, 16, 6, F_sec)  strictly causal rolling-z stationary
    X_macro: np.ndarray      # (N, 16, F_macro)   strictly causal rolling-z stationary
    X_flat: np.ndarray       # (N, D_flat)
    Y: np.ndarray            # (N, 6)             forward 250-day returns
    Y_multi: np.ndarray      # (N, 4, 6)          forward returns at +63, +126, +189, +250d
    macro_names: List[str]


def load_stationary_dataset(
    lookback: int = 250,
    horizon: int = 250,
    n_steps: int = 16,
) -> StationarySeries:
    """Build strictly causal, stationary multi-sector & macro dataset."""
    if not os.path.exists(DEFAULT_CACHE_PATH):
        load_sector_dataset()

    cached = np.load(DEFAULT_CACHE_PATH, allow_pickle=True)
    raw_dates = cached["raw_dates"]
    prices = cached["prices"]
    sec_raw = cached["sec_raw"]          # (T_total, 6, 16)
    macro_raw = cached["macro_raw"]      # (T_total, 15)
    macro_names = list(cached["macro_names"])

    T_total, S, Fs = sec_raw.shape

    # Strictly causal 5-year rolling z-score per sector (zero future info!)
    sec_stat = np.zeros_like(sec_raw, dtype=np.float32)
    for si in range(S):
        sec_stat[:, si, :] = causal_rolling_zscore(sec_raw[:, si, :], window=1250)

    # Strictly causal stationary macro features
    macro_stat, new_macro_names = build_stationary_macro(macro_raw, macro_names)
    Fm = macro_stat.shape[1]

    # Drop the first 250 days of warm-up (needed for YoY macro transforms) + lookback
    start_t = max(lookback, 250)
    valid_t = np.arange(start_t, T_total - horizon)
    N = len(valid_t)

    step_offsets = np.linspace(-lookback, 0, n_steps, dtype=int)
    multi_h = [63, 126, 189, horizon]

    X_sec = np.zeros((N, n_steps, S, Fs), dtype=np.float32)
    X_macro = np.zeros((N, n_steps, Fm), dtype=np.float32)
    Y_multi = np.zeros((N, 4, S), dtype=np.float32)

    for i, t in enumerate(valid_t):
        idxs = t + step_offsets
        X_sec[i] = sec_stat[idxs]
        X_macro[i] = macro_stat[idxs]
        p0 = prices[t]
        for hi, h in enumerate(multi_h):
            Y_multi[i, hi] = (prices[t + h] - p0) / np.maximum(p0, 1e-6)

    lag_idx = [0, 4, 8, 12, 15]
    X_flat = np.concatenate(
        [X_sec[:, lag_idx].reshape(N, -1), X_macro[:, lag_idx].reshape(N, -1)], axis=1
    ).astype(np.float32)

    return StationarySeries(
        dates=raw_dates[valid_t],
        X_sec=X_sec,
        X_macro=X_macro,
        X_flat=X_flat,
        Y=Y_multi[:, -1, :],
        Y_multi=Y_multi,
        macro_names=new_macro_names,
    )


def assert_strict_purged_split(
    tr: np.ndarray, va: np.ndarray, te: np.ndarray, purge_gap: int = 250
):
    """Hard guarantee: Train, Val, and Test are strictly ordered, contiguous, non-overlapping,
    non-interwoven, and separated by at least `purge_gap` trading days."""
    assert tr[-1] + purge_gap <= va[0], f"Leakage! tr_end={tr[-1]}, va_start={va[0]}"
    assert va[-1] + purge_gap <= te[0], f"Leakage! va_end={va[-1]}, te_start={te[0]}"
    assert len(set(tr).intersection(set(va))) == 0
    assert len(set(tr).intersection(set(te))) == 0
    assert len(set(va).intersection(set(te))) == 0
