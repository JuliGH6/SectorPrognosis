"""
Iterative v3 Benchmark: Strictly Stationary, Causal, Never-Mixed / Never-Interwoven
4-Fold Purged Walk-Forward Evaluation of Scaled-Up Rough-Path Signatures & Coupled
Duffing Oscillators.

Guarantees:
  - Every fold uses contiguous blocks: Train < [250d purge] < Val < [250d purge] < Test
  - Hard assertions verify zero overlap, zero interweaving, and >=250d purge gaps
  - All rolling z-scores are strictly causal (only data <= t)
  - All PCAs, scalers, ridge alphas, ensemble weights, and early-stopping checkpoints
    are fit/selected STRICTLY on Train/Val; Test is evaluated strictly once.
"""

from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import torch
from sklearn.linear_model import Ridge

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.data_pipeline import SECTORS
from experiments.metrics import evaluate_predictions
from experiments.models.coupled_oscillator import CoupledOscillatorNet
from experiments.models.v3_scaled_methods import (
    RoughPathV3Forecaster,
    train_duffing_v3_ensemble,
)
from experiments.run_winners_walkforward import train_osc_v1
from experiments.stationary_pipeline import (
    assert_strict_purged_split,
    load_stationary_dataset,
)


def run_v3_walkforward(n_folds: int = 4, purge_gap: int = 250):
    np.random.seed(42)
    torch.manual_seed(42)
    torch.set_num_threads(2)

    print("Building strictly causal, stationary dataset (YoY macro + trailing 5Y rolling z-scores)...", flush=True)
    ds = load_stationary_dataset()
    N, T, S, Fs = ds.X_sec.shape
    Fm = ds.X_macro.shape[2]
    print(f"Ready: N={N}, T={T}, Sectors={S}, F_sec={Fs}, F_macro={Fm} ({ds.dates[0]} .. {ds.dates[-1]})", flush=True)

    X_sec_t = torch.from_numpy(ds.X_sec)
    X_mac_t = torch.from_numpy(ds.X_macro)
    Y_t = torch.from_numpy(ds.Y)
    Y_multi_t = torch.from_numpy(ds.Y_multi)

    # Exact same 4 unseen contiguous test regimes (~2012..2024)
    first_test_start = int(N * 0.48)
    fold_len = (N - first_test_start) // n_folds

    model_names = [
        "0_Stationary_Ridge",
        "1_RoughPath_v3_Decomp",
        "2_CoupledOsc_v1_Stat",
        "3_DuffingOsc_v3_Ens",
        "4_ValBlend_RPv3+Duffv3",
    ]
    agg = {m: [] for m in model_names}
    fold_details = []

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

        # HARD ZERO-LEAKAGE / ZERO-INTERWEAVING ASSERTION
        assert_strict_purged_split(tr, va, te, purge_gap=purge_gap)

        print(
            f"\n=== Fold {fold+1} (STRICT PURGED): "
            f"Train {ds.dates[tr[0]]}..{ds.dates[tr[-1]]} ({len(tr)}d) | "
            f"Val {ds.dates[va[0]]}..{ds.dates[va[-1]]} ({len(va)}d) | "
            f"Test {ds.dates[te[0]]}..{ds.dates[te[-1]]} ({len(te)}d) ===",
            flush=True,
        )

        # 0) Stationary Ridge Baseline
        best_a, best_v = 5000.0, float("inf")
        for a in [500.0, 2000.0, 10000.0, 30000.0, 80000.0]:
            r = Ridge(alpha=a).fit(ds.X_flat[tr], ds.Y[tr])
            vl = float(np.mean((r.predict(ds.X_flat[va]) - ds.Y[va]) ** 2))
            if vl < best_v:
                best_v, best_a = vl, a
        ridge = Ridge(alpha=best_a).fit(ds.X_flat[tr], ds.Y[tr])
        pred_ridge = ridge.predict(ds.X_flat[te])

        # 1) RoughPath v3 (Decomposed Market + Rotation Multi-Scale Enriched Signatures + Cross-Levy)
        rp3 = RoughPathV3Forecaster(n_mkt_pcs=8, n_rot_pcs=8, use_huber_blend=True)
        rp3.fit_select(ds.X_sec[tr], ds.X_macro[tr], ds.Y[tr], ds.X_sec[va], ds.X_macro[va], ds.Y[va])
        pred_rp3_va = rp3.predict(ds.X_sec[va], ds.X_macro[va])
        pred_rp3_te = rp3.predict(ds.X_sec[te], ds.X_macro[te])

        # 2) Coupled Oscillator v1 on Stationary Features (clean ablation!)
        osc1 = train_osc_v1(X_sec_t, X_mac_t, Y_t, tr, va, S, Fs, Fm, seed=42 + fold)
        with torch.no_grad():
            pred_osc1_te = osc1(X_sec_t[te], X_mac_t[te]).numpy()

        # 3) Duffing Oscillator v3 (Decomposed Nonlinear Market + Zero-Sum Coupled Sector Springs, 3-seed)
        duff_models = train_duffing_v3_ensemble(X_sec_t, X_mac_t, Y_multi_t, tr, va, n_seeds=3, epochs=60)
        with torch.no_grad():
            pred_duff_va = np.mean([m(X_sec_t[va], X_mac_t[va]).numpy() for m in duff_models], axis=0)
            pred_duff_te = np.mean([m(X_sec_t[te], X_mac_t[te]).numpy() for m in duff_models], axis=0)

        # 4) Validation-Selected Decomposed Blend of RoughPath v3 + Duffing v3 (NEVER touches Test!)
        rp_va_m, du_va_m = pred_rp3_va.mean(axis=1), pred_duff_va.mean(axis=1)
        rp_va_r, du_va_r = pred_rp3_va - rp_va_m[:, None], pred_duff_va - du_va_m[:, None]
        y_va_m = ds.Y[va].mean(axis=1)
        y_va_r = ds.Y[va] - y_va_m[:, None]

        best_wm, best_lm = 0.5, float("inf")
        for w in np.linspace(0.0, 1.0, 11):
            l = float(np.mean((w * rp_va_m + (1.0 - w) * du_va_m - y_va_m) ** 2))
            if l < best_lm:
                best_lm, best_wm = l, float(w)

        best_wr, best_lr = 0.5, float("inf")
        for w in np.linspace(0.0, 1.0, 11):
            l = float(np.mean((w * rp_va_r + (1.0 - w) * du_va_r - y_va_r) ** 2))
            if l < best_lr:
                best_lr, best_wr = l, float(w)

        rp_te_m, du_te_m = pred_rp3_te.mean(axis=1), pred_duff_te.mean(axis=1)
        rp_te_r, du_te_r = pred_rp3_te - rp_te_m[:, None], pred_duff_te - du_te_m[:, None]
        pred_blend_te = (best_wm * rp_te_m + (1.0 - best_wm) * du_te_m)[:, None] + (
            best_wr * rp_te_r + (1.0 - best_wr) * du_te_r
        )

        preds_map = {
            "0_Stationary_Ridge": pred_ridge,
            "1_RoughPath_v3_Decomp": pred_rp3_te,
            "2_CoupledOsc_v1_Stat": pred_osc1_te,
            "3_DuffingOsc_v3_Ens": pred_duff_te,
            "4_ValBlend_RPv3+Duffv3": pred_blend_te,
        }

        fold_res = {"fold": fold + 1, "test_range": f"{ds.dates[te[0]]}..{ds.dates[te[-1]]}", "blend_w": (best_wm, best_wr)}
        for name, p in preds_map.items():
            m = evaluate_predictions(ds.Y[te], p, SECTORS)
            agg[name].append(m)
            fold_res[name] = {k: v for k, v in m.items() if k != "per_sector_r2"}
            print(
                f"  {name:<24} | RMSE={m['rmse']:.4f} | R2={m['mean_r2']:>6.3f} | "
                f"Dir={m['dir_acc']*100:5.1f}% | TS_IC={m['ts_ic']:+.3f} | "
                f"RotIC={m['cs_ic']:+.3f} | L/S={m['ls_spread']*100:+.2f}%",
                flush=True,
            )
        fold_details.append(fold_res)

    print("\n" + "=" * 105)
    print("v3 STATIONARY 4-FOLD STRICT CAUSAL WALK-FORWARD AVERAGE (2012–2024)")
    print("-" * 105)
    print(f"{'Model':<24} | {'RMSE':>6} | {'MeanR2':>7} | {'DirAcc':>7} | {'TS_IC':>6} | {'Rot_IC':>6} | {'L/S Sprd':>8} | {'Folds+LS':>8}")
    print("-" * 105)
    summary = {}
    for name in model_names:
        ms = agg[name]
        pos_ls = sum(1 for m in ms if m["ls_spread"] > 0)
        row = {
            "rmse": float(np.mean([m["rmse"] for m in ms])),
            "mean_r2": float(np.mean([m["mean_r2"] for m in ms])),
            "dir_acc": float(np.mean([m["dir_acc"] for m in ms])),
            "ts_ic": float(np.mean([m["ts_ic"] for m in ms])),
            "cs_ic": float(np.mean([m["cs_ic"] for m in ms])),
            "ls_spread": float(np.mean([m["ls_spread"] for m in ms])),
            "pos_ls_folds": f"{pos_ls}/{len(ms)}",
        }
        summary[name] = row
        print(
            f"{name:<24} | {row['rmse']:>6.4f} | {row['mean_r2']:>7.3f} | "
            f"{row['dir_acc']*100:>6.1f}% | {row['ts_ic']:>+6.3f} | "
            f"{row['cs_ic']:>+6.3f} | {row['ls_spread']*100:>+7.2f}% | "
            f"   {row['pos_ls_folds']}"
        )
    print("=" * 105)

    with open("/tmp/sector_data/v3_walkforward_results.json", "w") as f:
        json.dump({"summary": summary, "folds": fold_details}, f, indent=2)


if __name__ == "__main__":
    run_v3_walkforward()
