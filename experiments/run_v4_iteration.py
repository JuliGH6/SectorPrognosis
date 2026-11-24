"""
Iteration v4: Full Strictly Causal, Never-Mixed / Never-Interwoven 4-Fold Purged
Walk-Forward Benchmark for RoughPathV4Forecaster & DuffingOscillatorV4Net.
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.data_pipeline import SECTORS
from experiments.metrics import evaluate_predictions
from experiments.models.v4_champion_methods import (
    RoughPathV4Forecaster,
    train_duffing_v4_ensemble,
)
from experiments.stationary_pipeline import (
    assert_strict_purged_split,
    load_stationary_dataset,
)


def run_v4(n_folds: int = 4, purge_gap: int = 250):
    np.random.seed(42)
    torch.manual_seed(42)
    torch.set_num_threads(2)

    ds = load_stationary_dataset()
    N, T, S, Fs = ds.X_sec.shape
    Fm = ds.X_macro.shape[2]

    X_sec_t = torch.from_numpy(ds.X_sec)
    X_mac_t = torch.from_numpy(ds.X_macro)
    Y_multi_t = torch.from_numpy(ds.Y_multi)

    first_test_start = int(N * 0.48)
    fold_len = (N - first_test_start) // n_folds

    model_names = [
        "1_RoughPath_v4_Decomp",
        "2_DuffingOsc_v4_LowRank",
        "3_Champion_50_50_Fusion",
        "4_Champion_ValBlend",
    ]
    agg = {m: [] for m in model_names}

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
            f"Train {ds.dates[tr[0]]}..{ds.dates[tr[-1]]} | "
            f"Val {ds.dates[va[0]]}..{ds.dates[va[-1]]} | "
            f"Test {ds.dates[te[0]]}..{ds.dates[te[-1]]} ===",
            flush=True,
        )

        # 1) RoughPath v4
        rp4 = RoughPathV4Forecaster(n_mkt_pcs=5, n_rot_pcs=8, n_sec_pcs=4)
        rp4.fit_select(ds.X_sec[tr], ds.X_macro[tr], ds.Y[tr], ds.X_sec[va], ds.X_macro[va], ds.Y[va])
        p_rp_va = rp4.predict(ds.X_sec[va], ds.X_macro[va])
        p_rp_te = rp4.predict(ds.X_sec[te], ds.X_macro[te])

        # 2) Duffing Oscillator v4 (4-seed low-rank ensemble)
        duff_models = train_duffing_v4_ensemble(X_sec_t, X_mac_t, Y_multi_t, tr, va, n_seeds=4, epochs=60)
        with torch.no_grad():
            p_du_va = np.mean([m(X_sec_t[va], X_mac_t[va]).numpy() for m in duff_models], axis=0)
            p_du_te = np.mean([m(X_sec_t[te], X_mac_t[te]).numpy() for m in duff_models], axis=0)

        # 3) Clean 50/50 Orthogonal Fusion
        p_5050_te = 0.5 * p_rp_te + 0.5 * p_du_te

        # 4) Validation-Selected Decomposed Blend (fit strictly on Val, never Test)
        rp_vm, du_vm = p_rp_va.mean(axis=1), p_du_va.mean(axis=1)
        rp_vr, du_vr = p_rp_va - rp_vm[:, None], p_du_va - du_vm[:, None]
        y_vm = ds.Y[va].mean(axis=1)
        y_vr = ds.Y[va] - y_vm[:, None]

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

        rp_tm, du_tm = p_rp_te.mean(axis=1), p_du_te.mean(axis=1)
        rp_tr, du_tr = p_rp_te - rp_tm[:, None], p_du_te - du_tm[:, None]
        p_valb_te = (best_wm * rp_tm + (1.0 - best_wm) * du_tm)[:, None] + (
            best_wr * rp_tr + (1.0 - best_wr) * du_tr
        )

        for name, p in [
            ("1_RoughPath_v4_Decomp", p_rp_te),
            ("2_DuffingOsc_v4_LowRank", p_du_te),
            ("3_Champion_50_50_Fusion", p_5050_te),
            ("4_Champion_ValBlend", p_valb_te),
        ]:
            m = evaluate_predictions(ds.Y[te], p, SECTORS)
            agg[name].append(m)
            print(
                f"  {name:<25} | RMSE={m['rmse']:.4f} | R2={m['mean_r2']:>6.3f} | "
                f"Dir={m['dir_acc']*100:5.1f}% | TS_IC={m['ts_ic']:+.3f} | "
                f"RotIC={m['cs_ic']:+.3f} | L/S={m['ls_spread']*100:+.2f}%",
                flush=True,
            )

    print("\n" + "=" * 105)
    print("v4 CHAMPION 4-FOLD STRICT CAUSAL WALK-FORWARD AVERAGE (2012–2024)")
    print("-" * 105)
    print(f"{'Model':<25} | {'RMSE':>6} | {'MeanR2':>7} | {'DirAcc':>7} | {'TS_IC':>6} | {'Rot_IC':>6} | {'L/S Sprd':>8} | {'Folds+LS':>8}")
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
            f"{name:<25} | {row['rmse']:>6.4f} | {row['mean_r2']:>7.3f} | "
            f"{row['dir_acc']*100:>6.1f}% | {row['ts_ic']:>+6.3f} | "
            f"{row['cs_ic']:>+6.3f} | {row['ls_spread']*100:>+7.2f}% | "
            f"   {row['pos_ls_folds']}"
        )
    print("=" * 105)

    with open("/tmp/sector_data/v4_walkforward_results.json", "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    run_v4()
