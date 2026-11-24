"""
Leak-Free Benchmark Harness for Novel ML Paradigms on SectorPrognosis.

Evaluates 2 clean baselines + 6 radically different ML paradigms on the EXACT same
purged chronological splits (250-day purge gaps between Train / Val / Test,
train-only scaling):

Baselines:
  0. Ridge Regression (linear multi-lag baseline)
  1. Residual MLP (original repo's architecture, evaluated leak-free)

Novel Paradigms:
  2. Spiking Neural Network (Delta-Encoded Multi-Timescale LIF + Surrogate Gradients)
  3. Latent Economic World Model (JEPA + RK2 Neural-ODE Imagination Rollout)
  4. Conditional Flow Matching (Generative Rectified-Flow Diffusion ODE + Uncertainty)
  5. Bidirectional Cross-Sector Axial Attention Transformer (Shared-Weight Sector Tokens)
  6. 2D Gramian Angular Field + Recurrence Vision CNN (Market Photographs)
  7. Liquid Reservoir Computing (Multi-Scale Leaky Echo State Network)
"""

from __future__ import annotations

import copy
import json
import os
import sys
import time
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import Ridge

# Ensure repo root is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.data_pipeline import SECTORS, load_sector_dataset
from experiments.metrics import evaluate_predictions
from experiments.models import (
    EconomicWorldModel,
    LiquidReservoirForecaster,
    MarketVisionCNN,
    SectorFlowMatcher,
    SectorRotationTransformer,
    SpikingMacroStressNet,
)


def set_seed(seed: int = 42):
    np.random.seed(seed)
    torch.manual_seed(seed)


class ResidualMLPBaseline(nn.Module):
    """Leak-free version of the original repo's Residual MLP architecture."""

    def __init__(self, in_dim: int, hidden_dim: int = 128, n_layers: int = 3, out_dim: int = 6):
        super().__init__()
        self.in_proj = nn.Linear(in_dim, hidden_dim)
        self.blocks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    nn.Dropout(0.3),
                    nn.ReLU(),
                )
                for _ in range(n_layers)
            ]
        )
        self.head = nn.Linear(hidden_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.in_proj(x)
        for blk in self.blocks:
            h = h + blk(h)
        return self.head(h)


def train_standard_torch_model(
    model: nn.Module,
    X_sec_tr: torch.Tensor,
    X_mac_tr: torch.Tensor,
    Y_tr: torch.Tensor,
    X_sec_val: torch.Tensor,
    X_mac_val: torch.Tensor,
    Y_val: torch.Tensor,
    epochs: int = 45,
    batch_size: int = 128,
    lr: float = 1.5e-3,
    weight_decay: float = 1e-3,
    custom_loss_fn=None,
) -> nn.Module:
    """Train a PyTorch model with AdamW + CosineAnnealing and Val-loss early selection."""
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    best_val = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    N = X_sec_tr.shape[0]

    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(N)
        for start in range(0, N, batch_size):
            idx = perm[start : start + batch_size]
            xb_s, xb_m, yb = X_sec_tr[idx], X_mac_tr[idx], Y_tr[idx]
            pred = model(xb_s, xb_m)
            loss = custom_loss_fn(pred, yb) if custom_loss_fn else F.smooth_l1_loss(pred, yb, beta=0.1)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()
        sched.step()

        model.eval()
        with torch.no_grad():
            val_pred = model(X_sec_val, X_mac_val)
            val_loss = F.mse_loss(val_pred, Y_val).item()
            if val_loss < best_val:
                best_val = val_loss
                best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    model.eval()
    return model


def train_world_model(
    model: EconomicWorldModel,
    ds,
    X_sec_all: torch.Tensor,
    X_mac_all: torch.Tensor,
    Y_multi_all: torch.Tensor,
    epochs: int = 45,
    batch_size: int = 128,
    lr: float = 1.5e-3,
) -> EconomicWorldModel:
    """Train World Model using strictly in-train future states for JEPA latent rollout."""
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    # Strictly restrict training indices so even the +250d future latent stays inside train_idx!
    max_train_i = int(ds.train_idx[-1])
    valid_tr = ds.train_idx[ds.future_idx[ds.train_idx, -1] <= max_train_i]
    fut_idx_t = torch.from_numpy(ds.future_idx)

    X_sec_val = X_sec_all[ds.val_idx]
    X_mac_val = X_mac_all[ds.val_idx]
    Y_val = Y_multi_all[ds.val_idx, -1, :]

    best_val = float("inf")
    best_state = copy.deepcopy(model.state_dict())

    for epoch in range(epochs):
        model.train()
        perm = valid_tr[np.random.permutation(len(valid_tr))]
        for start in range(0, len(perm), batch_size):
            idx = perm[start : start + batch_size]
            f_idx = fut_idx_t[idx]  # (B, 4)
            loss = model.compute_loss(
                X_sec_all[idx],
                X_mac_all[idx],
                Y_multi_all[idx],
                X_sec_all[f_idx],
                X_mac_all[f_idx],
            )
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()
            model.update_target_encoder(ema_decay=0.985)
        sched.step()

        model.eval()
        with torch.no_grad():
            val_pred = model(X_sec_val, X_mac_val)
            val_loss = F.mse_loss(val_pred, Y_val).item()
            if val_loss < best_val:
                best_val = val_loss
                best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    model.eval()
    return model


def train_flow_matcher(
    model: SectorFlowMatcher,
    X_sec_tr: torch.Tensor,
    X_mac_tr: torch.Tensor,
    Y_tr: torch.Tensor,
    X_sec_val: torch.Tensor,
    X_mac_val: torch.Tensor,
    Y_val: torch.Tensor,
    epochs: int = 55,
    batch_size: int = 128,
    lr: float = 2e-3,
) -> SectorFlowMatcher:
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    best_val = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    N = X_sec_tr.shape[0]

    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(N)
        for start in range(0, N, batch_size):
            idx = perm[start : start + batch_size]
            loss = model.compute_loss(X_sec_tr[idx], X_mac_tr[idx], Y_tr[idx])
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()
        sched.step()

        if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
            model.eval()
            val_mean, _, _ = model.sample_futures(X_sec_val, X_mac_val, n_samples=12, n_steps=8)
            val_loss = F.mse_loss(val_mean, Y_val).item()
            if val_loss < best_val:
                best_val = val_loss
                best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    model.eval()
    return model


def run_all():
    set_seed(42)
    torch.set_num_threads(2)

    print("Loading leak-free SectorPrognosis dataset...")
    ds = load_sector_dataset()
    N, T, S, Fs = ds.X_sec.shape
    Fm = ds.X_macro.shape[2]

    print(
        f"Dataset ready: N={N}, T={T}, Sectors={S}, F_sec={Fs}, F_macro={Fm}\n"
        f"  Train: {len(ds.train_idx)} ({ds.dates[ds.train_idx[0]]} .. {ds.dates[ds.train_idx[-1]]})\n"
        f"  Val:   {len(ds.val_idx)}   ({ds.dates[ds.val_idx[0]]} .. {ds.dates[ds.val_idx[-1]]})\n"
        f"  Test:  {len(ds.test_idx)}  ({ds.dates[ds.test_idx[0]]} .. {ds.dates[ds.test_idx[-1]]})\n"
    )

    X_sec_t = torch.from_numpy(ds.X_sec)
    X_mac_t = torch.from_numpy(ds.X_macro)
    X_flat_t = torch.from_numpy(ds.X_flat)
    Y_t = torch.from_numpy(ds.Y)
    Y_multi_t = torch.from_numpy(ds.Y_multi)

    tr, va, te = ds.train_idx, ds.val_idx, ds.test_idx
    Y_te_np = ds.Y[te]

    results: Dict[str, dict] = {}
    diagnostics: Dict[str, dict] = {}

    # ------------------------------------------------------------------
    # Baseline 0: Ridge Regression
    # ------------------------------------------------------------------
    t0 = time.time()
    best_alpha, best_v = 100.0, float("inf")
    for alpha in [10.0, 100.0, 500.0, 2000.0, 10000.0]:
        reg = Ridge(alpha=alpha).fit(ds.X_flat[tr], ds.Y[tr])
        vl = float(np.mean((reg.predict(ds.X_flat[va]) - ds.Y[va]) ** 2))
        if vl < best_v:
            best_v, best_alpha = vl, alpha
    ridge = Ridge(alpha=best_alpha).fit(ds.X_flat[tr], ds.Y[tr])
    pred_ridge = ridge.predict(ds.X_flat[te])
    results["0_Baseline_Ridge"] = evaluate_predictions(Y_te_np, pred_ridge, SECTORS)
    results["0_Baseline_Ridge"]["seconds"] = round(time.time() - t0, 2)
    print(f"[Done] 0_Baseline_Ridge (alpha={best_alpha}) in {results['0_Baseline_Ridge']['seconds']}s")

    # ------------------------------------------------------------------
    # Baseline 1: Residual MLP (Original Repo Style, Leak-Free)
    # ------------------------------------------------------------------
    set_seed(42)
    t0 = time.time()
    mlp = ResidualMLPBaseline(ds.X_flat.shape[1], hidden_dim=128, n_layers=3, out_dim=S)
    opt = torch.optim.AdamW(mlp.parameters(), lr=1e-3, weight_decay=1e-3)
    best_v, best_st = float("inf"), copy.deepcopy(mlp.state_dict())
    for ep in range(45):
        mlp.train()
        perm = torch.randperm(len(tr))
        for s0 in range(0, len(tr), 128):
            idx = tr[perm[s0 : s0 + 128].numpy()]
            loss = F.mse_loss(mlp(X_flat_t[idx]), Y_t[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
        mlp.eval()
        with torch.no_grad():
            vl = F.mse_loss(mlp(X_flat_t[va]), Y_t[va]).item()
            if vl < best_v:
                best_v, best_st = vl, copy.deepcopy(mlp.state_dict())
    mlp.load_state_dict(best_st)
    mlp.eval()
    with torch.no_grad():
        pred_mlp = mlp(X_flat_t[te]).numpy()
    results["1_Baseline_ResMLP"] = evaluate_predictions(Y_te_np, pred_mlp, SECTORS)
    results["1_Baseline_ResMLP"]["seconds"] = round(time.time() - t0, 2)
    print(f"[Done] 1_Baseline_ResMLP in {results['1_Baseline_ResMLP']['seconds']}s")

    # ------------------------------------------------------------------
    # Paradigm 1: Spiking Neural Network (Coupled Multi-Timescale LIF)
    # ------------------------------------------------------------------
    set_seed(42)
    t0 = time.time()
    snn = SpikingMacroStressNet(n_sectors=S, f_sec=Fs, f_macro=Fm, hidden_dim=96)
    snn = train_standard_torch_model(
        snn, X_sec_t[tr], X_mac_t[tr], Y_t[tr], X_sec_t[va], X_mac_t[va], Y_t[va],
        epochs=45, lr=2e-3, weight_decay=1e-3,
    )
    with torch.no_grad():
        pred_snn = snn(X_sec_t[te], X_mac_t[te]).numpy()
        spike_rate = snn.get_spike_rate(X_sec_t[te], X_mac_t[te])
        betas = torch.sigmoid(snn.lif1.beta_logit).numpy()
    results["2_Spiking_LIF_SNN"] = evaluate_predictions(Y_te_np, pred_snn, SECTORS)
    results["2_Spiking_LIF_SNN"]["seconds"] = round(time.time() - t0, 2)
    diagnostics["snn"] = {
        "mean_spike_rate": round(spike_rate, 4),
        "beta_min": round(float(betas.min()), 3),
        "beta_median": round(float(np.median(betas)), 3),
        "beta_max": round(float(betas.max()), 3),
    }
    print(f"[Done] 2_Spiking_LIF_SNN (spike_rate={spike_rate:.3f}) in {results['2_Spiking_LIF_SNN']['seconds']}s")

    # ------------------------------------------------------------------
    # Paradigm 2: Latent Economic World Model (JEPA + Neural ODE)
    # ------------------------------------------------------------------
    set_seed(42)
    t0 = time.time()
    wm = EconomicWorldModel(n_sectors=S, f_sec=Fs, f_macro=Fm, latent_dim=32)
    wm = train_world_model(wm, ds, X_sec_t, X_mac_t, Y_multi_t, epochs=45, lr=1.5e-3)
    with torch.no_grad():
        _, _, ret_traj_te = wm.imagine(X_sec_t[te], X_mac_t[te], steps=4)
        ret_traj_np = ret_traj_te.numpy()
        pred_wm = ret_traj_np[:, -1, :]
    results["3_Economic_WorldModel"] = evaluate_predictions(Y_te_np, pred_wm, SECTORS)
    results["3_Economic_WorldModel"]["seconds"] = round(time.time() - t0, 2)
    # Track imagined quarterly horizon directional accuracy
    q_dir = {}
    for qi, qname in enumerate(["+3m", "+6m", "+9m", "+12m"]):
        q_dir[qname] = round(float(np.mean((ret_traj_np[:, qi] > 0) == (ds.Y_multi[te, qi] > 0))), 4)
    diagnostics["world_model"] = {"quarterly_dir_acc": q_dir}
    print(f"[Done] 3_Economic_WorldModel (Q-DirAcc={q_dir}) in {results['3_Economic_WorldModel']['seconds']}s")

    # ------------------------------------------------------------------
    # Paradigm 3: Conditional Flow Matching (Generative Diffusion-ODE)
    # ------------------------------------------------------------------
    set_seed(42)
    t0 = time.time()
    fm = SectorFlowMatcher(n_sectors=S, f_sec=Fs, f_macro=Fm, cond_dim=64, hidden_dim=96)
    fm = train_flow_matcher(
        fm, X_sec_t[tr], X_mac_t[tr], Y_t[tr], X_sec_t[va], X_mac_t[va], Y_t[va],
        epochs=55, lr=2e-3,
    )
    pred_fm_t, std_fm_t, _ = fm.sample_futures(X_sec_t[te], X_mac_t[te], n_samples=32, n_steps=12)
    pred_fm, std_fm = pred_fm_t.numpy(), std_fm_t.numpy()
    results["4_FlowMatching_DiffODE"] = evaluate_predictions(Y_te_np, pred_fm, SECTORS, uncertainty=std_fm)
    results["4_FlowMatching_DiffODE"]["seconds"] = round(time.time() - t0, 2)
    diagnostics["flow_matching"] = {
        "mean_pred_std": round(float(std_fm.mean()), 4),
        "low_unc_rmse": round(results["4_FlowMatching_DiffODE"]["low_unc_rmse"], 4),
        "high_unc_rmse": round(results["4_FlowMatching_DiffODE"]["high_unc_rmse"], 4),
        "low_unc_ls_spread": round(results["4_FlowMatching_DiffODE"]["low_unc_ls_spread"], 4),
        "high_unc_ls_spread": round(results["4_FlowMatching_DiffODE"]["high_unc_ls_spread"], 4),
    }
    print(f"[Done] 4_FlowMatching_DiffODE in {results['4_FlowMatching_DiffODE']['seconds']}s")

    # ------------------------------------------------------------------
    # Paradigm 4: Bidirectional Cross-Sector Axial Attention Transformer
    # ------------------------------------------------------------------
    set_seed(42)
    t0 = time.time()
    xformer = SectorRotationTransformer(n_sectors=S, f_sec=Fs, f_macro=Fm, d_model=48, n_heads=4, n_layers=2)
    xformer = train_standard_torch_model(
        xformer, X_sec_t[tr], X_mac_t[tr], Y_t[tr], X_sec_t[va], X_mac_t[va], Y_t[va],
        epochs=45, lr=1.5e-3, weight_decay=2e-3,
        custom_loss_fn=SectorRotationTransformer.rotation_loss,
    )
    with torch.no_grad():
        pred_xf_t, attn_w = xformer(X_sec_t[te], X_mac_t[te], return_attn=True)
        pred_xf = pred_xf_t.numpy()
        mean_attn = attn_w.mean(dim=0).numpy()  # (7, 7)
    results["5_CrossSector_Attn"] = evaluate_predictions(Y_te_np, pred_xf, SECTORS)
    results["5_CrossSector_Attn"]["seconds"] = round(time.time() - t0, 2)
    token_names = SECTORS + ["Macro"]
    # For each sector, which other token does it attend to most?
    top_attn = {}
    for i, sname in enumerate(SECTORS):
        row = mean_attn[i].copy()
        row[i] = -1.0  # exclude self
        j = int(np.argmax(row))
        top_attn[sname] = f"{token_names[j]} ({row[j]:.2f})"
    diagnostics["cross_sector_attn"] = {"top_cross_token_attention": top_attn}
    print(f"[Done] 5_CrossSector_Attn in {results['5_CrossSector_Attn']['seconds']}s")

    # ------------------------------------------------------------------
    # Paradigm 5: 2D Gramian Angular Field + Recurrence Vision CNN
    # ------------------------------------------------------------------
    set_seed(42)
    t0 = time.time()
    vcnn = MarketVisionCNN(n_sectors=S, f_sec=Fs, f_macro=Fm, n_channels=12)
    vcnn = train_standard_torch_model(
        vcnn, X_sec_t[tr], X_mac_t[tr], Y_t[tr], X_sec_t[va], X_mac_t[va], Y_t[va],
        epochs=45, lr=1.5e-3, weight_decay=2e-3,
    )
    with torch.no_grad():
        pred_vcnn = vcnn(X_sec_t[te], X_mac_t[te]).numpy()
    results["6_GAF_MarketVisionCNN"] = evaluate_predictions(Y_te_np, pred_vcnn, SECTORS)
    results["6_GAF_MarketVisionCNN"]["seconds"] = round(time.time() - t0, 2)
    print(f"[Done] 6_GAF_MarketVisionCNN in {results['6_GAF_MarketVisionCNN']['seconds']}s")

    # ------------------------------------------------------------------
    # Paradigm 6: Liquid Reservoir Computing (Echo State Network)
    # ------------------------------------------------------------------
    t0 = time.time()
    best_a, best_v = 500.0, float("inf")
    for alpha in [100.0, 500.0, 2000.0, 8000.0]:
        esn_try = LiquidReservoirForecaster(reservoir_size=384, ridge_alpha=alpha, seed=42)
        esn_try.fit(ds.X_sec[tr], ds.X_macro[tr], ds.Y[tr])
        vl = float(np.mean((esn_try.predict(ds.X_sec[va], ds.X_macro[va]) - ds.Y[va]) ** 2))
        if vl < best_v:
            best_v, best_a = vl, alpha
    esn = LiquidReservoirForecaster(reservoir_size=384, ridge_alpha=best_a, seed=42)
    esn.fit(ds.X_sec[tr], ds.X_macro[tr], ds.Y[tr])
    pred_esn = esn.predict(ds.X_sec[te], ds.X_macro[te])
    results["7_Liquid_Reservoir_ESN"] = evaluate_predictions(Y_te_np, pred_esn, SECTORS)
    results["7_Liquid_Reservoir_ESN"]["seconds"] = round(time.time() - t0, 2)
    print(f"[Done] 7_Liquid_Reservoir_ESN (alpha={best_a}) in {results['7_Liquid_Reservoir_ESN']['seconds']}s")

    # ------------------------------------------------------------------
    # Print Comparison Table
    # ------------------------------------------------------------------
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

    print("\nParadigm-Specific Diagnostics:")
    print(json.dumps(diagnostics, indent=2))

    out_path = "/tmp/sector_data/benchmark_results.json"
    with open(out_path, "w") as f:
        json.dump({"results": results, "diagnostics": diagnostics}, f, indent=2)
    print(f"\nSaved full results to {out_path}")


if __name__ == "__main__":
    run_all()
