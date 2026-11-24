"""
4-Fold Strictly Causal Purged Walk-Forward Validation of Our Two Breakout Winners
(v1 minimal vs v2 scaled-up vs Signature-Oscillator Hybrid vs Ridge baseline).

Every fold enforces:
  - Strictly causal train-only RobustScaler normalization
  - 250-trading-day purge gaps between Train / Val / Test (zero target overlap)
  - Evaluation across 4 distinct historical market regimes spanning 2011 -> 2024!
"""

from __future__ import annotations

import copy
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import Ridge
from sklearn.preprocessing import RobustScaler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.data_pipeline import DEFAULT_CACHE_PATH, SECTORS, load_sector_dataset
from experiments.metrics import evaluate_predictions
from experiments.models.coupled_oscillator import CoupledOscillatorNet
from experiments.models.path_signatures import PathSignatureForecaster
from experiments.models.scaled_winners import (
    MultiScalePathSignatureForecaster,
    RegimeCoupledOscillatorNet,
)


def set_seed(seed: int = 42):
    np.random.seed(seed)
    torch.manual_seed(seed)


def _build_causal_fold(
    sec_raw: np.ndarray,
    macro_raw: np.ndarray,
    prices: np.ndarray,
    valid_t: np.ndarray,
    tr_end: int,
    step_offsets: np.ndarray,
    horizon: int = 250,
):
    train_raw_end = int(valid_t[tr_end - 1]) + 1
    sec_scaled = np.zeros_like(sec_raw, dtype=np.float32)
    for si in range(sec_raw.shape[1]):
        sc = RobustScaler().fit(sec_raw[:train_raw_end, si, :])
        sec_scaled[:, si, :] = np.clip(sc.transform(sec_raw[:, si, :]), -5.0, 5.0)

    msc = RobustScaler().fit(macro_raw[:train_raw_end, :])
    macro_scaled = np.clip(msc.transform(macro_raw), -5.0, 5.0).astype(np.float32)

    N = len(valid_t)
    multi_h = [63, 126, 189, horizon]
    X_sec = np.zeros((N, len(step_offsets), sec_raw.shape[1], sec_raw.shape[2]), dtype=np.float32)
    X_macro = np.zeros((N, len(step_offsets), macro_raw.shape[1]), dtype=np.float32)
    Y_multi = np.zeros((N, 4, sec_raw.shape[1]), dtype=np.float32)

    for i, t in enumerate(valid_t):
        idxs = t + step_offsets
        X_sec[i] = sec_scaled[idxs]
        X_macro[i] = macro_scaled[idxs]
        p0 = prices[t]
        for hi, h in enumerate(multi_h):
            Y_multi[i, hi] = (prices[t + h] - p0) / np.maximum(p0, 1e-6)

    lag_idx = [0, 4, 8, 12, 15]
    X_flat = np.concatenate(
        [X_sec[:, lag_idx].reshape(N, -1), X_macro[:, lag_idx].reshape(N, -1)], axis=1
    ).astype(np.float32)
    return X_sec, X_macro, X_flat, Y_multi[:, -1, :], Y_multi


def train_osc_v1(X_sec_t, X_mac_t, Y_t, tr, va, S, Fs, Fm, seed=42):
    set_seed(seed)
    model = CoupledOscillatorNet(n_sectors=S, f_sec=Fs, f_macro=Fm, n_steps=8)
    opt = torch.optim.AdamW(model.parameters(), lr=5e-3, weight_decay=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=55)
    best_v, best_st = float("inf"), copy.deepcopy(model.state_dict())
    for _ in range(55):
        model.train()
        perm = torch.randperm(len(tr))
        for s0 in range(0, len(tr), 128):
            idx = tr[perm[s0 : s0 + 128].numpy()]
            pred = model(X_sec_t[idx], X_mac_t[idx])
            abs_l = F.smooth_l1_loss(pred, Y_t[idx], beta=0.1)
            rel_l = F.smooth_l1_loss(
                pred - pred.mean(dim=-1, keepdim=True),
                Y_t[idx] - Y_t[idx].mean(dim=-1, keepdim=True),
                beta=0.05,
            )
            loss = abs_l + 0.4 * rel_l
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()
        sched.step()
        model.eval()
        with torch.no_grad():
            vl = F.mse_loss(model(X_sec_t[va], X_mac_t[va]), Y_t[va]).item()
            if vl < best_v:
                best_v, best_st = vl, copy.deepcopy(model.state_dict())
    model.load_state_dict(best_st)
    model.eval()
    return model


def train_osc_v2(X_sec_t, X_mac_t, Y_multi_t, tr, va, S, Fs, Fm, extra_t=None, seed=42):
    set_seed(seed)
    ef_dim = extra_t.shape[1] if extra_t is not None else 0
    model = RegimeCoupledOscillatorNet(
        n_sectors=S, f_sec=Fs, f_macro=Fm, n_modes=2, n_steps=8, extra_force_dim=ef_dim
    )
    opt = torch.optim.AdamW(model.parameters(), lr=4e-3, weight_decay=2e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=55)
    best_v, best_st = float("inf"), copy.deepcopy(model.state_dict())
    for _ in range(55):
        model.train()
        perm = torch.randperm(len(tr))
        for s0 in range(0, len(tr), 128):
            idx = tr[perm[s0 : s0 + 128].numpy()]
            ef = extra_t[idx] if extra_t is not None else None
            traj = model.integrate_trajectory(X_sec_t[idx], X_mac_t[idx], ef)
            loss = RegimeCoupledOscillatorNet.multi_horizon_loss(traj, Y_multi_t[idx])
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()
        sched.step()
        model.eval()
        with torch.no_grad():
            ef_va = extra_t[va] if extra_t is not None else None
            vl = F.mse_loss(model(X_sec_t[va], X_mac_t[va], ef_va), Y_multi_t[va, -1, :]).item()
            if vl < best_v:
                best_v, best_st = vl, copy.deepcopy(model.state_dict())
    model.load_state_dict(best_st)
    model.eval()
    return model


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

    # 4 distinct out-of-sample test folds spanning the second half (~2012..2024)
    first_test_start = int(N * 0.48)
    fold_len = (N - first_test_start) // n_folds

    model_names = [
        "0_Baseline_Ridge",
        "1_RoughPath_v1",
        "2_RoughPath_v2_MultiScale",
        "3_CoupledOsc_v1_303p",
        "4_CoupledOsc_v2_Regime",
        "5_SigOsc_Hybrid",
        "6_Ensemble_RPv2+Oscv2",
    ]
    agg = {m: [] for m in model_names}

    print(f"Running {n_folds}-Fold Causal Purged Walk-Forward (purge_gap={purge_gap}d, fold_len={fold_len}d)", flush=True)

    for fold in range(n_folds):
        te_start = first_test_start + fold * fold_len
        te_end = min(te_start + fold_len, N)
        val_end = te_start - purge_gap
        val_start = max(val_end - 350, 0)
        tr_end = val_start - purge_gap
        if tr_end < 600:
            continue

        tr = np.arange(0, tr_end)
        va = np.arange(val_start, val_end)
        te = np.arange(te_start, te_end)

        X_sec, X_macro, X_flat, Y, Y_multi = _build_causal_fold(
            sec_raw, macro_raw, prices, valid_t, tr_end, step_offsets, horizon
        )
        X_sec_t = torch.from_numpy(X_sec)
        X_mac_t = torch.from_numpy(X_macro)
        Y_t = torch.from_numpy(Y)
        Y_multi_t = torch.from_numpy(Y_multi)
        Y_te = Y[te]

        print(
            f"\n=== Fold {fold+1}: Train {dates[tr[0]]}..{dates[tr[-1]]} ({len(tr)}d) | "
            f"Test {dates[te[0]]}..{dates[te[-1]]} ({len(te)}d) ===",
            flush=True,
        )

        # 0) Baseline Ridge
        best_a, best_v = 5000.0, float("inf")
        for a in [500.0, 2000.0, 10000.0, 30000.0]:
            r = Ridge(alpha=a).fit(X_flat[tr], Y[tr])
            vl = float(np.mean((r.predict(X_flat[va]) - Y[va]) ** 2))
            if vl < best_v:
                best_v, best_a = vl, a
        pred_ridge = Ridge(alpha=best_a).fit(X_flat[tr], Y[tr]).predict(X_flat[te])

        # 1) RoughPath v1 (minimal)
        best_a, best_v = 10000.0, float("inf")
        for a in [1000.0, 5000.0, 20000.0]:
            m = PathSignatureForecaster(n_components=10, ridge_alpha=a).fit(X_sec[tr], X_macro[tr], Y[tr])
            vl = float(np.mean((m.predict(X_sec[va], X_macro[va]) - Y[va]) ** 2))
            if vl < best_v:
                best_v, best_a = vl, a
        rp1 = PathSignatureForecaster(n_components=10, ridge_alpha=best_a).fit(X_sec[tr], X_macro[tr], Y[tr])
        pred_rp1 = rp1.predict(X_sec[te], X_macro[te])

        # 2) RoughPath v2 (Multi-Scale + Time-Augmented + Cross-Sector Relative Signatures)
        best_a, best_v = 20000.0, float("inf")
        for a in [2000.0, 10000.0, 30000.0, 80000.0]:
            m = MultiScalePathSignatureForecaster(n_macro_pcs=8, n_rel_pcs=6, ridge_alpha=a).fit(
                X_sec[tr], X_macro[tr], Y[tr]
            )
            vl = float(np.mean((m.predict(X_sec[va], X_macro[va]) - Y[va]) ** 2))
            if vl < best_v:
                best_v, best_a = vl, a
        rp2 = MultiScalePathSignatureForecaster(n_macro_pcs=8, n_rel_pcs=6, ridge_alpha=best_a).fit(
            X_sec[tr], X_macro[tr], Y[tr]
        )
        pred_rp2 = rp2.predict(X_sec[te], X_macro[te])

        # 3) Coupled Oscillator v1 (303 params)
        osc1 = train_osc_v1(X_sec_t, X_mac_t, Y_t, tr, va, S, Fs, Fm, seed=42 + fold)
        with torch.no_grad():
            pred_osc1 = osc1(X_sec_t[te], X_mac_t[te]).numpy()

        # 4) Coupled Oscillator v2 (2-mode + macro-gated coupling + multi-horizon)
        osc2 = train_osc_v2(X_sec_t, X_mac_t, Y_multi_t, tr, va, S, Fs, Fm, extra_t=None, seed=42 + fold)
        with torch.no_grad():
            pred_osc2 = osc2(X_sec_t[te], X_mac_t[te]).numpy()

        # 5) Signature-Forced Coupled Oscillator Hybrid
        sig_feats_all = torch.from_numpy(rp2.transform_for_hybrid(X_sec, X_macro))
        hyb = train_osc_v2(X_sec_t, X_mac_t, Y_multi_t, tr, va, S, Fs, Fm, extra_t=sig_feats_all, seed=42 + fold)
        with torch.no_grad():
            pred_hyb = hyb(X_sec_t[te], X_mac_t[te], sig_feats_all[te]).numpy()

        # 6) Simple 50/50 Ensemble of RoughPath v2 + Coupled Oscillator v2
        pred_ens = 0.5 * pred_rp2 + 0.5 * pred_osc2

        preds_map = {
            "0_Baseline_Ridge": pred_ridge,
            "1_RoughPath_v1": pred_rp1,
            "2_RoughPath_v2_MultiScale": pred_rp2,
            "3_CoupledOsc_v1_303p": pred_osc1,
            "4_CoupledOsc_v2_Regime": pred_osc2,
            "5_SigOsc_Hybrid": pred_hyb,
            "6_Ensemble_RPv2+Oscv2": pred_ens,
        }

        for name, p in preds_map.items():
            m = evaluate_predictions(Y_te, p, SECTORS)
            agg[name].append(m)
            print(
                f"  {name:<26} | RMSE={m['rmse']:.4f} | R2={m['mean_r2']:>6.3f} | "
                f"Dir={m['dir_acc']*100:5.1f}% | TS_IC={m['ts_ic']:+.3f} | "
                f"RotIC={m['cs_ic']:+.3f} | L/S={m['ls_spread']*100:+.2f}%",
                flush=True,
            )

    print("\n" + "=" * 105)
    print("4-FOLD CAUSAL WALK-FORWARD AVERAGE (2012–2024 Across 4 Unseen Regimes)")
    print("-" * 105)
    print(f"{'Model':<26} | {'RMSE':>6} | {'MeanR2':>7} | {'DirAcc':>7} | {'TS_IC':>6} | {'Rot_IC':>6} | {'L/S Sprd':>8} | {'Folds+LS':>8}")
    print("-" * 105)
    for name in model_names:
        ms = agg[name]
        pos_ls = sum(1 for m in ms if m["ls_spread"] > 0)
        print(
            f"{name:<26} | {np.mean([m['rmse'] for m in ms]):>6.4f} | "
            f"{np.mean([m['mean_r2'] for m in ms]):>7.3f} | "
            f"{np.mean([m['dir_acc'] for m in ms])*100:>6.1f}% | "
            f"{np.mean([m['ts_ic'] for m in ms]):>+6.3f} | "
            f"{np.mean([m['cs_ic'] for m in ms]):>+6.3f} | "
            f"{np.mean([m['ls_spread'] for m in ms])*100:>+7.2f}% | "
            f"   {pos_ls}/{len(ms)}"
        )
    print("=" * 105)


if __name__ == "__main__":
    run_walk_forward()
