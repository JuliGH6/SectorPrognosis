"""
Purged Expanding-Window Walk-Forward Evaluation for Top Paradigms.

Unlike SectorPrognosisRollingWindow.ipynb (which did not enforce a 250-day purge gap
between training and test windows), this script evaluates models across multiple
historical out-of-sample regimes with:
  - Strict 250-day purge gaps between Train / Val / Test
  - Strictly causal, per-fold train-only RobustScaler normalization

Usage:
    python experiments/walk_forward.py
"""

from __future__ import annotations

import os
import sys
import numpy as np
import torch
from sklearn.preprocessing import RobustScaler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.data_pipeline import DEFAULT_CACHE_PATH, SECTORS, load_sector_dataset
from experiments.metrics import evaluate_predictions
from experiments.models import (
    LiquidReservoirForecaster,
    SectorRotationTransformer,
    SpikingMacroStressNet,
)
from experiments.run_benchmark import set_seed, train_standard_torch_model


def _build_fold_tensors(
    sec_raw: np.ndarray,
    macro_raw: np.ndarray,
    prices: np.ndarray,
    valid_t: np.ndarray,
    tr_end: int,
    step_offsets: np.ndarray,
    horizon: int = 250,
):
    """Fit scalers strictly up to valid_t[tr_end - 1] and build windowed tensors."""
    train_raw_end = int(valid_t[tr_end - 1]) + 1
    sec_scaled = np.zeros_like(sec_raw, dtype=np.float32)
    for si in range(sec_raw.shape[1]):
        sc = RobustScaler().fit(sec_raw[:train_raw_end, si, :])
        sec_scaled[:, si, :] = np.clip(sc.transform(sec_raw[:, si, :]), -5.0, 5.0)

    msc = RobustScaler().fit(macro_raw[:train_raw_end, :])
    macro_scaled = np.clip(msc.transform(macro_raw), -5.0, 5.0).astype(np.float32)

    N = len(valid_t)
    X_sec = np.zeros((N, len(step_offsets), sec_raw.shape[1], sec_raw.shape[2]), dtype=np.float32)
    X_macro = np.zeros((N, len(step_offsets), macro_raw.shape[1]), dtype=np.float32)
    Y = np.zeros((N, sec_raw.shape[1]), dtype=np.float32)

    for i, t in enumerate(valid_t):
        idxs = t + step_offsets
        X_sec[i] = sec_scaled[idxs]
        X_macro[i] = macro_scaled[idxs]
        Y[i] = (prices[t + horizon] - prices[t]) / np.maximum(prices[t], 1e-6)

    return X_sec, X_macro, Y


def run_walk_forward(n_folds: int = 4, purge_gap: int = 250, lookback: int = 250, horizon: int = 250):
    set_seed(42)
    torch.set_num_threads(2)

    if not os.path.exists(DEFAULT_CACHE_PATH):
        load_sector_dataset()

    cached = np.load(DEFAULT_CACHE_PATH, allow_pickle=True)
    raw_dates = cached["raw_dates"]
    prices = cached["prices"]
    sec_raw = cached["sec_raw"]
    macro_raw = cached["macro_raw"]

    valid_t = np.arange(lookback, len(raw_dates) - horizon)
    dates = raw_dates[valid_t]
    step_offsets = np.linspace(-lookback, 0, 16, dtype=int)
    N = len(valid_t)
    S, Fs, Fm = sec_raw.shape[1], sec_raw.shape[2], macro_raw.shape[1]

    first_test_start = int(N * 0.45)
    fold_len = (N - first_test_start) // n_folds

    print(f"Running {n_folds}-Fold Causal Purged Walk-Forward (purge_gap={purge_gap}d, fold_len={fold_len}d)")
    agg = {name: [] for name in ["Liquid_ESN", "Spiking_SNN", "CrossSector_Attn"]}

    for fold in range(n_folds):
        te_start = first_test_start + fold * fold_len
        te_end = min(te_start + fold_len, N)
        val_end = te_start - purge_gap
        val_start = max(val_end - 300, 0)
        tr_end = val_start - purge_gap
        if tr_end < 500:
            continue

        tr = np.arange(0, tr_end)
        va = np.arange(val_start, val_end)
        te = np.arange(te_start, te_end)

        X_sec, X_macro, Y = _build_fold_tensors(sec_raw, macro_raw, prices, valid_t, tr_end, step_offsets, horizon)

        print(
            f"\n--- Fold {fold+1}: Train {dates[tr[0]]}..{dates[tr[-1]]} | "
            f"Test {dates[te[0]]}..{dates[te[-1]]} ---"
        )

        # 1) Liquid Reservoir ESN
        esn = LiquidReservoirForecaster(reservoir_size=384, ridge_alpha=2000.0, seed=42 + fold)
        esn.fit(X_sec[tr], X_macro[tr], Y[tr])
        m_esn = evaluate_predictions(Y[te], esn.predict(X_sec[te], X_macro[te]), SECTORS)
        agg["Liquid_ESN"].append(m_esn)
        print(f"  Liquid_ESN       | RMSE={m_esn['rmse']:.4f} | Dir={m_esn['dir_acc']*100:5.1f}% | RotIC={m_esn['cs_ic']:+.3f} | L/S={m_esn['ls_spread']*100:+.2f}%")

        # 2) Spiking SNN
        X_sec_t = torch.from_numpy(X_sec)
        X_mac_t = torch.from_numpy(X_macro)
        Y_t = torch.from_numpy(Y)

        set_seed(42 + fold)
        snn = SpikingMacroStressNet(n_sectors=S, f_sec=Fs, f_macro=Fm, hidden_dim=96)
        snn = train_standard_torch_model(
            snn, X_sec_t[tr], X_mac_t[tr], Y_t[tr], X_sec_t[va], X_mac_t[va], Y_t[va],
            epochs=35, lr=2e-3,
        )
        with torch.no_grad():
            pred_snn = snn(X_sec_t[te], X_mac_t[te]).numpy()
        m_snn = evaluate_predictions(Y[te], pred_snn, SECTORS)
        agg["Spiking_SNN"].append(m_snn)
        print(f"  Spiking_SNN      | RMSE={m_snn['rmse']:.4f} | Dir={m_snn['dir_acc']*100:5.1f}% | RotIC={m_snn['cs_ic']:+.3f} | L/S={m_snn['ls_spread']*100:+.2f}%")

        # 3) Cross-Sector Transformer
        set_seed(42 + fold)
        xf = SectorRotationTransformer(n_sectors=S, f_sec=Fs, f_macro=Fm, d_model=48, n_heads=4, n_layers=2)
        xf = train_standard_torch_model(
            xf, X_sec_t[tr], X_mac_t[tr], Y_t[tr], X_sec_t[va], X_mac_t[va], Y_t[va],
            epochs=35, lr=1.5e-3, custom_loss_fn=SectorRotationTransformer.rotation_loss,
        )
        with torch.no_grad():
            pred_xf = xf(X_sec_t[te], X_mac_t[te]).numpy()
        m_xf = evaluate_predictions(Y[te], pred_xf, SECTORS)
        agg["CrossSector_Attn"].append(m_xf)
        print(f"  CrossSector_Attn | RMSE={m_xf['rmse']:.4f} | Dir={m_xf['dir_acc']*100:5.1f}% | RotIC={m_xf['cs_ic']:+.3f} | L/S={m_xf['ls_spread']*100:+.2f}%")

    print("\n=== Walk-Forward Average Across Folds ===")
    for name, ms in agg.items():
        if not ms:
            continue
        print(
            f"  {name:<18} | RMSE={np.mean([m['rmse'] for m in ms]):.4f} | "
            f"Dir={np.mean([m['dir_acc'] for m in ms])*100:5.1f}% | "
            f"RotIC={np.mean([m['cs_ic'] for m in ms]):+.3f} | "
            f"L/S={np.mean([m['ls_spread'] for m in ms])*100:+.2f}%"
        )


if __name__ == "__main__":
    run_walk_forward()
