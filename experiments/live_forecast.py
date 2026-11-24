"""
Live 1-Year Forward Sector Forecast using Our Two Champion Paradigms:
  1. Rough-Path Signatures v4 (best standalone sector rotation picker)
  2. Nonlinear Coupled Duffing Oscillators v3 (best physics market/cycle model)
  3. Validation-Selected Champion Blend (best overall RMSE & directional accuracy)

Strictly leak-free protocol:
  - Train on all historical dates whose 1-year target is ALREADY realized, minus a
    250-day purge gap before the Validation window.
  - Select regularization, early stopping, and blend weights STRICTLY on Validation.
  - Forecast forward 1 year from the very latest date in the dataset (where the true
    future return is unseen — e.g. 2024-12-31 -> 2025-12-31).
"""

from __future__ import annotations

import os
import sys
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.data_pipeline import DEFAULT_CACHE_PATH, SECTORS, load_sector_dataset
from experiments.models.v3_scaled_methods import train_duffing_v3_ensemble
from experiments.models.v4_champion_methods import RoughPathV4Forecaster
from experiments.stationary_pipeline import (
    assert_strict_purged_split,
    build_stationary_macro,
    causal_rolling_zscore,
)


def run_live_forecast(lookback: int = 250, horizon: int = 250, n_steps: int = 16, purge_gap: int = 250):
    np.random.seed(42)
    torch.manual_seed(42)
    torch.set_num_threads(2)

    if not os.path.exists(DEFAULT_CACHE_PATH):
        load_sector_dataset()

    cached = np.load(DEFAULT_CACHE_PATH, allow_pickle=True)
    raw_dates = cached["raw_dates"]
    prices = cached["prices"]
    sec_raw = cached["sec_raw"]
    macro_raw = cached["macro_raw"]
    macro_names = list(cached["macro_names"])

    T_total, S, Fs = sec_raw.shape

    # Strictly causal stationary transforms up through the very last date
    sec_stat = np.zeros_like(sec_raw, dtype=np.float32)
    for si in range(S):
        sec_stat[:, si, :] = causal_rolling_zscore(sec_raw[:, si, :], window=1250)
    macro_stat, _ = build_stationary_macro(macro_raw, macro_names)
    Fm = macro_stat.shape[1]

    step_offsets = np.linspace(-lookback, 0, n_steps, dtype=int)
    multi_h = [63, 126, 189, horizon]

    # Supervised samples: dates where the full 1-year target is already realized
    start_t = max(lookback, 250)
    sup_t = np.arange(start_t, T_total - horizon)
    N = len(sup_t)

    X_sec = np.zeros((N, n_steps, S, Fs), dtype=np.float32)
    X_macro = np.zeros((N, n_steps, Fm), dtype=np.float32)
    Y_multi = np.zeros((N, 4, S), dtype=np.float32)

    for i, t in enumerate(sup_t):
        idxs = t + step_offsets
        X_sec[i] = sec_stat[idxs]
        X_macro[i] = macro_stat[idxs]
        p0 = prices[t]
        for hi, h in enumerate(multi_h):
            Y_multi[i, hi] = (prices[t + h] - p0) / np.maximum(p0, 1e-6)
    Y = Y_multi[:, -1, :]

    # Live inference sample: the very latest date in the dataset (t = T_total - 1)
    live_t = T_total - 1
    live_date = raw_dates[live_t]
    X_sec_live = sec_stat[live_t + step_offsets][None, ...]      # (1, 16, 6, 16)
    X_mac_live = macro_stat[live_t + step_offsets][None, ...]    # (1, 16, 33)

    # Strict purged Train / Val split on realized history (Val = most recent 500 realized days)
    val_len = 500
    va = np.arange(N - val_len, N)
    tr = np.arange(0, N - val_len - purge_gap)
    # Dummy test index strictly after Val+purge for assertion check
    assert tr[-1] + purge_gap <= va[0], "Purge gap violated!"

    print(
        f"Training strictly on realized history:\n"
        f"  Train: {raw_dates[sup_t[tr[0]]]} .. {raw_dates[sup_t[tr[-1]]]} ({len(tr)} days)\n"
        f"  Purge: {purge_gap} trading days skipped\n"
        f"  Val:   {raw_dates[sup_t[va[0]]]} .. {raw_dates[sup_t[va[-1]]]} ({len(va)} days)\n"
        f"  LIVE FORECAST DATE: {live_date}  -->  1-Year Forward Horizon\n",
        flush=True,
    )

    # 1) Rough-Path Signatures v4
    rp4 = RoughPathV4Forecaster(n_mkt_pcs=5, n_rot_pcs=8, n_sec_pcs=4)
    rp4.fit_select(X_sec[tr], X_macro[tr], Y[tr], X_sec[va], X_macro[va], Y[va])
    p_rp_va = rp4.predict(X_sec[va], X_macro[va])
    p_rp_live = rp4.predict(X_sec_live, X_mac_live)[0]

    # 2) Nonlinear Coupled Duffing Oscillators v3 (5-seed ensemble)
    X_sec_t = torch.from_numpy(X_sec)
    X_mac_t = torch.from_numpy(X_macro)
    Y_multi_t = torch.from_numpy(Y_multi)
    duff_models = train_duffing_v3_ensemble(X_sec_t, X_mac_t, Y_multi_t, tr, va, n_seeds=5, epochs=60)
    with torch.no_grad():
        p_du_va = np.mean([m(X_sec_t[va], X_mac_t[va]).numpy() for m in duff_models], axis=0)
        p_du_live = np.mean(
            [m(torch.from_numpy(X_sec_live), torch.from_numpy(X_mac_live)).numpy()[0] for m in duff_models],
            axis=0,
        )

    # 3) Validation-only Decomposed Blend
    rp_vm, du_vm = p_rp_va.mean(axis=1), p_du_va.mean(axis=1)
    rp_vr, du_vr = p_rp_va - rp_vm[:, None], p_du_va - du_vm[:, None]
    y_vm = Y[va].mean(axis=1)
    y_vr = Y[va] - y_vm[:, None]

    best_wm, best_lm = 0.5, float("inf")
    for w in np.linspace(0.0, 1.0, 11):
        l = float(np.mean((w * rp_vm + (1.0 - w) * du_vm - y_vm) ** 2))
        if l < best_lm:
            best_lm, best_wm = l, float(w)
    best_wr, best_lr = 0.5, float("inf")
    for w in np.linspace(0.0, 1.0, 11):
        l = float(np.mean((w * rp_vr + (1.0 - w) * du_vr - y_vr) ** 2))
        if l < best_lr:
            best_lr, best_wr = l, float(w)

    blend_mkt = best_wm * p_rp_live.mean() + (1.0 - best_wm) * p_du_live.mean()
    blend_rot = best_wr * (p_rp_live - p_rp_live.mean()) + (1.0 - best_wr) * (p_du_live - p_du_live.mean())
    p_blend_live = blend_mkt + blend_rot

    print(f"Val-selected Blend Weights: Market(RP={best_wm:.1f}, Duff={1-best_wm:.1f}) | Rotation(RP={best_wr:.1f}, Duff={1-best_wr:.1f})\n")
    print("=" * 78)
    print(f"1-YEAR FORWARD SECTOR FORECAST FROM {live_date}")
    print("-" * 78)
    print(f"{'Sector':<24} | {'RoughPath v4':>13} | {'DuffingOsc v3':>13} | {'Champion Blend':>14}")
    print("-" * 78)
    order = np.argsort(-p_blend_live)
    for idx in order:
        print(
            f"{SECTORS[idx]:<24} | {p_rp_live[idx]*100:>+12.2f}% | "
            f"{p_du_live[idx]*100:>+12.2f}% | {p_blend_live[idx]*100:>+13.2f}%"
        )
    print("-" * 78)
    print(
        f"{'Equal-Weight Avg':<24} | {p_rp_live.mean()*100:>+12.2f}% | "
        f"{p_du_live.mean()*100:>+12.2f}% | {p_blend_live.mean()*100:>+13.2f}%"
    )
    print("=" * 78)
    print(f"\nTop-2 LONG  (Champion Blend): {SECTORS[order[0]]}, {SECTORS[order[1]]}")
    print(f"Bot-2 SHORT (Champion Blend): {SECTORS[order[-2]]}, {SECTORS[order[-1]]}")

    # Inspect average learned Duffing physics parameters
    with torch.no_grad():
        k_avg = np.mean([F.softplus(m.rel_log_k).numpy() for m in duff_models], axis=0)
        a_avg = np.mean([0.25 * F.softplus(m.rel_log_alpha).numpy() for m in duff_models], axis=0)
        g_avg = np.mean([F.softplus(m.rel_log_g).numpy() for m in duff_models], axis=0)
    print("\nLearned Duffing Physics Constants (Mean-Reversion K | Cubic Snap-Back Alpha | Damping Gamma):")
    for i, s in enumerate(SECTORS):
        print(f"  {s:<22}: K={k_avg[i]:.3f} | Alpha={a_avg[i]:.3f} | Gamma={g_avg[i]:.3f}")


if __name__ == "__main__":
    run_live_forecast()
