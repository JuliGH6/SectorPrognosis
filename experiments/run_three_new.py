"""
Fast Minimal Test of 3 Innovative New Approaches on SectorPrognosis (Leak-Free):
  1. Conditional Energy-Based Model (Macro-Ising + Neural EBM with Langevin Inference)
  2. Rough-Path Depth-2 Signature & Signed Levy-Area Forecaster (Stochastic Analysis)
  3. Physics-Informed Coupled Damped Harmonic Oscillators (Velocity Verlet, ~1.5k params)

Evaluated on the EXACT same purged split (250-day purge gaps, Test = 2021-2024) as
run_benchmark.py for direct apples-to-apples comparison.
"""

from __future__ import annotations

import copy
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.data_pipeline import SECTORS, load_sector_dataset
from experiments.metrics import evaluate_predictions
from experiments.models.energy_based_model import SectorEnergyModel
from experiments.models.path_signatures import PathSignatureForecaster
from experiments.models.coupled_oscillator import CoupledOscillatorNet


def set_seed(seed: int = 42):
    np.random.seed(seed)
    torch.manual_seed(seed)


def run_three():
    set_seed(42)
    torch.set_num_threads(2)

    ds = load_sector_dataset()
    N, T, S, Fs = ds.X_sec.shape
    Fm = ds.X_macro.shape[2]
    tr, va, te = ds.train_idx, ds.val_idx, ds.test_idx

    X_sec_t = torch.from_numpy(ds.X_sec)
    X_mac_t = torch.from_numpy(ds.X_macro)
    Y_t = torch.from_numpy(ds.Y)
    Y_te_np = ds.Y[te]

    results = {}
    diagnostics = {}

    # ------------------------------------------------------------------
    # 1. Conditional Energy-Based Model (EBM + Langevin Inference)
    # ------------------------------------------------------------------
    set_seed(42)
    t0 = time.time()
    ebm = SectorEnergyModel(n_sectors=S, f_sec=Fs, f_macro=Fm, d_model=40, n_heads=4)
    opt = torch.optim.AdamW(ebm.parameters(), lr=2e-3, weight_decay=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=30)

    best_v, best_st = float("inf"), copy.deepcopy(ebm.state_dict())
    for ep in range(30):
        ebm.train()
        perm = torch.randperm(len(tr))
        for s0 in range(0, len(tr), 128):
            idx = tr[perm[s0 : s0 + 128].numpy()]
            loss = ebm.compute_loss(X_sec_t[idx], X_mac_t[idx], Y_t[idx])
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(ebm.parameters(), 2.0)
            opt.step()
        sched.step()

        if (ep + 1) % 5 == 0 or ep == 29:
            ebm.eval()
            val_pred, _, _ = ebm.predict_equilibrium(X_sec_t[va], X_mac_t[va], n_chains=12, n_steps=20)
            vl = F.mse_loss(val_pred, Y_t[va]).item()
            if vl < best_v:
                best_v, best_st = vl, copy.deepcopy(ebm.state_dict())

    ebm.load_state_dict(best_st)
    ebm.eval()
    pred_ebm_t, std_ebm_t, min_e_t = ebm.predict_equilibrium(X_sec_t[te], X_mac_t[te], n_chains=24, n_steps=30)
    pred_ebm, std_ebm, min_e = pred_ebm_t.numpy(), std_ebm_t.numpy(), min_e_t.numpy()
    results["New1_EnergyBased_EBM"] = evaluate_predictions(Y_te_np, pred_ebm, SECTORS, uncertainty=std_ebm)
    results["New1_EnergyBased_EBM"]["seconds"] = round(time.time() - t0, 2)

    # Check if minimum energy E*(x) correlates with forecast error (anomaly detection!)
    med_e = np.median(min_e)
    low_e_rmse = float(np.sqrt(np.mean((Y_te_np[min_e <= med_e] - pred_ebm[min_e <= med_e]) ** 2)))
    high_e_rmse = float(np.sqrt(np.mean((Y_te_np[min_e > med_e] - pred_ebm[min_e > med_e]) ** 2)))
    diagnostics["ebm"] = {
        "n_params": sum(p.numel() for p in ebm.parameters()),
        "low_energy_rmse": round(low_e_rmse, 4),
        "high_energy_rmse": round(high_e_rmse, 4),
        "low_chain_std_rmse": round(results["New1_EnergyBased_EBM"]["low_unc_rmse"], 4),
        "high_chain_std_rmse": round(results["New1_EnergyBased_EBM"]["high_unc_rmse"], 4),
    }
    print(f"[Done] New1_EnergyBased_EBM in {results['New1_EnergyBased_EBM']['seconds']}s", flush=True)

    # ------------------------------------------------------------------
    # 2. Rough-Path Depth-2 Signatures & Signed Levy Areas
    # ------------------------------------------------------------------
    t0 = time.time()
    best_cfg, best_v = (14, 1000.0), float("inf")
    for nc in [10, 14, 18]:
        for alpha in [200.0, 1000.0, 5000.0, 20000.0]:
            sig_try = PathSignatureForecaster(n_components=nc, ridge_alpha=alpha)
            sig_try.fit(ds.X_sec[tr], ds.X_macro[tr], ds.Y[tr])
            vl = float(np.mean((sig_try.predict(ds.X_sec[va], ds.X_macro[va]) - ds.Y[va]) ** 2))
            if vl < best_v:
                best_v, best_cfg = vl, (nc, alpha)

    sig = PathSignatureForecaster(n_components=best_cfg[0], ridge_alpha=best_cfg[1])
    sig.fit(ds.X_sec[tr], ds.X_macro[tr], ds.Y[tr])
    pred_sig = sig.predict(ds.X_sec[te], ds.X_macro[te])
    results["New2_RoughPath_LevySig"] = evaluate_predictions(Y_te_np, pred_sig, SECTORS)
    results["New2_RoughPath_LevySig"]["seconds"] = round(time.time() - t0, 2)
    diagnostics["path_signatures"] = {
        "best_pca_dims": best_cfg[0],
        "best_ridge_alpha": best_cfg[1],
        "total_sig_features": best_cfg[0] * 2 + best_cfg[0] * (best_cfg[0] - 1) // 2,
    }
    print(f"[Done] New2_RoughPath_LevySig (d={best_cfg[0]}, alpha={best_cfg[1]}) in {results['New2_RoughPath_LevySig']['seconds']}s", flush=True)

    # ------------------------------------------------------------------
    # 3. Physics-Informed Coupled Damped Harmonic Oscillators (~300 params!)
    # ------------------------------------------------------------------
    set_seed(42)
    t0 = time.time()
    osc = CoupledOscillatorNet(n_sectors=S, f_sec=Fs, f_macro=Fm, n_steps=8)
    opt = torch.optim.AdamW(osc.parameters(), lr=5e-3, weight_decay=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=60)
    best_v, best_st = float("inf"), copy.deepcopy(osc.state_dict())

    for ep in range(60):
        osc.train()
        perm = torch.randperm(len(tr))
        for s0 in range(0, len(tr), 128):
            idx = tr[perm[s0 : s0 + 128].numpy()]
            pred = osc(X_sec_t[idx], X_mac_t[idx])
            # Smooth L1 + light rotation spread loss
            abs_l = F.smooth_l1_loss(pred, Y_t[idx], beta=0.1)
            rel_l = F.smooth_l1_loss(
                pred - pred.mean(dim=-1, keepdim=True),
                Y_t[idx] - Y_t[idx].mean(dim=-1, keepdim=True),
                beta=0.05,
            )
            loss = abs_l + 0.4 * rel_l
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(osc.parameters(), 2.0)
            opt.step()
        sched.step()

        osc.eval()
        with torch.no_grad():
            vl = F.mse_loss(osc(X_sec_t[va], X_mac_t[va]), Y_t[va]).item()
            if vl < best_v:
                best_v, best_st = vl, copy.deepcopy(osc.state_dict())

    osc.load_state_dict(best_st)
    osc.eval()
    with torch.no_grad():
        pred_osc = osc(X_sec_t[te], X_mac_t[te]).numpy()
    results["New3_Coupled_Oscillator"] = evaluate_predictions(Y_te_np, pred_osc, SECTORS)
    results["New3_Coupled_Oscillator"]["seconds"] = round(time.time() - t0, 2)
    diagnostics["coupled_oscillator"] = {
        "n_params": sum(p.numel() for p in osc.parameters()),
        **osc.get_physics_summary(SECTORS),
    }
    print(f"[Done] New3_Coupled_Oscillator ({diagnostics['coupled_oscillator']['n_params']} params!) in {results['New3_Coupled_Oscillator']['seconds']}s", flush=True)

    # Print Table
    print("\n" + "=" * 102)
    print(f"{'Model':<26} | {'RMSE':>6} | {'MeanR2':>7} | {'DirAcc':>7} | {'TS_IC':>6} | {'Rot_IC':>6} | {'L/S Sprd':>8} | {'Time':>5}")
    print("-" * 102)
    for name, m in results.items():
        print(
            f"{name:<26} | {m['rmse']:>6.4f} | {m['mean_r2']:>7.3f} | "
            f"{m['dir_acc']*100:>6.1f}% | {m['ts_ic']:>6.3f} | {m['cs_ic']:>6.3f} | "
            f"{m['ls_spread']*100:>+7.2f}% | {m['seconds']:>4.1f}s"
        )
    print("=" * 102)
    print("\nDiagnostics:")
    print(json.dumps(diagnostics, indent=2))


if __name__ == "__main__":
    run_three()
