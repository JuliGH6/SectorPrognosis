"""
Scaled-Up v3 Implementations of Both Breakout Paradigms:
  1. RoughPathV3Forecaster:
     - Decomposes 1-year sector returns into:
         (a) Market Common Mode (equal-weighted 6-sector mean return)
         (b) Zero-Sum Cross-Sector Rotation Spreads (sector minus market mean)
     - Computes multi-scale (1Y + 1Q) time-augmented Depth-2 Signatures, Signed Levy Areas,
       Quadratic Path Energy (sum dX^2), and Path Asymmetry (sum dX^3).
     - Adds explicit Macro-to-Sector Cross-Levy Areas (did Macro move BEFORE Sector?).
     - Uses separate validation-tuned Ridge + Huber-resistant readouts for Market vs Rotation!

  2. DuffingOscillatorV3Net:
     - Physics-informed Nonlinear Coupled Duffing Oscillators with explicit
       Market Common-Mode + Zero-Sum Cross-Sector Spring Decomposition:
         d^2 q / dt^2 = -K*(q - q_eq) - Alpha*(q - q_eq)^3 - Gamma*v + C(macro)*q + F_ext
     - Cubic Duffing term (-Alpha * dq^3) captures nonlinear snap-back from valuation extremes!
     - Macro-gated cross-sector coupling matrix C(macro).
     - Symplectic Velocity Verlet integration over 8 steps supervised at +3m, +6m, +9m, +12m.
"""

from __future__ import annotations

import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.decomposition import PCA
from sklearn.linear_model import HuberRegressor, Ridge
from sklearn.preprocessing import StandardScaler


def compute_enriched_signatures(paths: np.ndarray) -> np.ndarray:
    """
    Compute Depth-2 Signature + Signed Levy Areas + Path Quadratic Variation + Path Skewness.
    Args:
        paths: (N, T, D)
    Returns:
        feats: (N, 4*D + D*(D-1)//2)
    """
    N, T, D = paths.shape
    dX = np.diff(paths, axis=1)  # (N, T-1, D)

    latest = paths[:, -1, :]
    lvl1 = dX.sum(axis=1)                                 # Net change
    quad_var = np.sqrt(np.mean(dX * dX, axis=1) + 1e-8)   # Path energy / volatility
    skew_var = np.cbrt(np.mean(dX * dX * dX, axis=1))     # Crash vs melt-up asymmetry

    prefix = np.zeros_like(dX)
    prefix[:, 1:, :] = np.cumsum(dX[:, :-1, :], axis=1)
    S2 = np.einsum("nti,ntj->nij", prefix, dX)
    levy = 0.5 * (S2 - np.swapaxes(S2, 1, 2))
    iu = np.triu_indices(D, k=1)
    levy_flat = levy[:, iu[0], iu[1]]

    return np.concatenate([latest, lvl1, quad_var, skew_var, levy_flat], axis=1).astype(np.float32)


def _time_augment(paths: np.ndarray) -> np.ndarray:
    N, T, _ = paths.shape
    t_ch = np.broadcast_to(np.linspace(-1.0, 1.0, T, dtype=np.float32)[None, :, None], (N, T, 1))
    return np.concatenate([paths, t_ch], axis=-1)


class RoughPathV3Forecaster:
    """
    Decomposed Multi-Scale Enriched Rough-Path Signature Forecaster.
    Fits all PCAs, scalers, and readouts STRICTLY on Train; selects regularization STRICTLY on Val.
    """

    def __init__(self, n_mkt_pcs: int = 8, n_rot_pcs: int = 8, use_huber_blend: bool = True):
        self.n_mkt_pcs = n_mkt_pcs
        self.n_rot_pcs = n_rot_pcs
        self.use_huber_blend = use_huber_blend

    def _build_views(self, X_sec: np.ndarray, X_macro: np.ndarray):
        # X_sec: (N, T, 6, Fs), X_macro: (N, T, Fm)
        N, T, S, Fs = X_sec.shape
        # Market view: macro + cross-sector average fundamentals + cross-sector dispersion
        sec_mean = X_sec.mean(axis=2)   # (N, T, Fs)
        sec_std = X_sec.std(axis=2)     # (N, T, Fs)
        mkt_raw = np.concatenate([X_macro, sec_mean, sec_std], axis=-1)

        # Rotation view: sector-relative fundamentals + macro context
        sec_rel = (X_sec - X_sec.mean(axis=2, keepdims=True)).reshape(N, T, S * Fs)
        return mkt_raw, sec_rel, X_macro

    def _fit_transform_features(
        self, X_sec_tr: np.ndarray, X_mac_tr: np.ndarray
    ):
        mkt_raw, rot_raw, mac_raw = self._build_views(X_sec_tr, X_mac_tr)
        N, T, Dm = mkt_raw.shape
        Dr = rot_raw.shape[-1]

        self.pca_mkt = PCA(n_components=self.n_mkt_pcs, random_state=42).fit(mkt_raw.reshape(N * T, Dm))
        self.pca_rot = PCA(n_components=self.n_rot_pcs, random_state=42).fit(rot_raw.reshape(N * T, Dr))
        self.pca_mac_cross = PCA(n_components=4, random_state=42).fit(mac_raw.reshape(N * T, mac_raw.shape[-1]))

        phi_mkt, phi_rot = self._transform_features(X_sec_tr, X_mac_tr)
        self.sc_mkt = StandardScaler().fit(phi_mkt)
        self.sc_rot = StandardScaler().fit(phi_rot)
        return self.sc_mkt.transform(phi_mkt), self.sc_rot.transform(phi_rot)

    def _transform_features(self, X_sec: np.ndarray, X_macro: np.ndarray):
        mkt_raw, rot_raw, mac_raw = self._build_views(X_sec, X_macro)
        N, T, Dm = mkt_raw.shape
        Dr = rot_raw.shape[-1]

        p_mkt = self.pca_mkt.transform(mkt_raw.reshape(N * T, Dm)).reshape(N, T, self.n_mkt_pcs)
        p_rot = self.pca_rot.transform(rot_raw.reshape(N * T, Dr)).reshape(N, T, self.n_rot_pcs)
        p_mac = self.pca_mac_cross.transform(mac_raw.reshape(N * T, mac_raw.shape[-1])).reshape(N, T, 4)

        # Market features: 1Y time-augmented + 1Q enriched signatures
        phi_mkt = np.concatenate(
            [
                compute_enriched_signatures(_time_augment(p_mkt)),
                compute_enriched_signatures(p_mkt[:, -5:, :]),
            ],
            axis=1,
        )

        # Rotation features: 1Y relative signatures + 1Q relative signatures + Macro-to-Relative Cross-Levy Areas!
        # Cross-Levy between 4 Macro PCs and 6 top Relative Sector PCs
        dM = np.diff(p_mac, axis=1)          # (N, T-1, 4)
        dR = np.diff(p_rot[:, :, :6], axis=1)  # (N, T-1, 6)
        preM = np.zeros_like(dM); preM[:, 1:, :] = np.cumsum(dM[:, :-1, :], axis=1)
        preR = np.zeros_like(dR); preR[:, 1:, :] = np.cumsum(dR[:, :-1, :], axis=1)
        cross_levy = 0.5 * (
            np.einsum("nti,ntj->nij", preM, dR) - np.einsum("ntj,nti->nij", preR, dM)
        ).reshape(N, -1)

        phi_rot = np.concatenate(
            [
                compute_enriched_signatures(_time_augment(p_rot)),
                compute_enriched_signatures(p_rot[:, -5:, :]),
                cross_levy.astype(np.float32),
            ],
            axis=1,
        )
        return phi_mkt, phi_rot

    def fit_select(
        self,
        X_sec_tr: np.ndarray,
        X_mac_tr: np.ndarray,
        Y_tr: np.ndarray,
        X_sec_va: np.ndarray,
        X_mac_va: np.ndarray,
        Y_va: np.ndarray,
    ):
        """Fit strictly on Train, tune separate Market and Rotation regularization strictly on Val."""
        Z_mkt_tr, Z_rot_tr = self._fit_transform_features(X_sec_tr, X_mac_tr)
        phi_mkt_va, phi_rot_va = self._transform_features(X_sec_va, X_mac_va)
        Z_mkt_va = self.sc_mkt.transform(phi_mkt_va)
        Z_rot_va = self.sc_rot.transform(phi_rot_va)

        y_mkt_tr = Y_tr.mean(axis=1)
        y_mkt_va = Y_va.mean(axis=1)
        y_rot_tr = Y_tr - y_mkt_tr[:, None]
        y_rot_va = Y_va - y_mkt_va[:, None]

        # 1) Select Market Common-Mode Ridge alpha on Val
        best_am, best_vm = 10000.0, float("inf")
        for a in [500.0, 2000.0, 8000.0, 25000.0, 80000.0]:
            reg = Ridge(alpha=a).fit(Z_mkt_tr, y_mkt_tr)
            vl = float(np.mean((reg.predict(Z_mkt_va) - y_mkt_va) ** 2))
            if vl < best_vm:
                best_vm, best_am = vl, a
        self.mkt_ridge = Ridge(alpha=best_am).fit(Z_mkt_tr, y_mkt_tr)

        # Optional Huber robust head blended only if it improves Val MSE
        self.mkt_huber = None
        self.huber_w = 0.0
        if self.use_huber_blend:
            hub = HuberRegressor(alpha=best_am * 0.25, max_iter=300).fit(Z_mkt_tr, y_mkt_tr)
            p_r = self.mkt_ridge.predict(Z_mkt_va)
            p_h = hub.predict(Z_mkt_va)
            for w in [0.0, 0.25, 0.5]:
                vl = float(np.mean(((1 - w) * p_r + w * p_h - y_mkt_va) ** 2))
                if vl < best_vm:
                    best_vm, self.huber_w, self.mkt_huber = vl, w, hub

        # 2) Select Cross-Sector Rotation Ridge alpha on Val (optimizing relative spread MSE)
        best_ar, best_vr = 10000.0, float("inf")
        for a in [500.0, 2000.0, 8000.0, 25000.0, 80000.0]:
            reg = Ridge(alpha=a).fit(Z_rot_tr, y_rot_tr)
            p = reg.predict(Z_rot_va)
            p = p - p.mean(axis=1, keepdims=True)
            vl = float(np.mean((p - y_rot_va) ** 2))
            if vl < best_vr:
                best_vr, best_ar = vl, a
        self.rot_ridge = Ridge(alpha=best_ar).fit(Z_rot_tr, y_rot_tr)
        self.best_alphas = (best_am, best_ar, self.huber_w)
        return self

    def predict_decomposed(self, X_sec: np.ndarray, X_macro: np.ndarray):
        phi_mkt, phi_rot = self._transform_features(X_sec, X_macro)
        Z_mkt = self.sc_mkt.transform(phi_mkt)
        Z_rot = self.sc_rot.transform(phi_rot)

        p_mkt = self.mkt_ridge.predict(Z_mkt)
        if self.mkt_huber is not None and self.huber_w > 0:
            p_mkt = (1.0 - self.huber_w) * p_mkt + self.huber_w * self.mkt_huber.predict(Z_mkt)

        p_rot = self.rot_ridge.predict(Z_rot)
        p_rot = p_rot - p_rot.mean(axis=1, keepdims=True)
        return p_mkt.astype(np.float32), p_rot.astype(np.float32)

    def predict(self, X_sec: np.ndarray, X_macro: np.ndarray) -> np.ndarray:
        p_mkt, p_rot = self.predict_decomposed(X_sec, X_macro)
        return (p_mkt[:, None] + p_rot).astype(np.float32)


class DuffingOscillatorV3Net(nn.Module):
    """
    Physics-Informed Nonlinear Coupled Duffing Oscillator Network (v3).
    Explicitly separates:
      1) 1D Nonlinear Market Duffing Oscillator (driven by stationary macro + avg fundamentals)
      2) 6D Zero-Sum Coupled Sector Duffing Oscillators (driven by relative fundamentals +
         macro-gated cross-sector spring coupling C(macro))
    """

    def __init__(self, n_sectors: int = 6, f_sec: int = 16, f_macro: int = 33, n_steps: int = 8):
        super().__init__()
        self.n_sectors = n_sectors
        self.n_steps = n_steps

        # --- Market Common-Mode 1D Nonlinear Duffing Oscillator ---
        self.mkt_log_k = nn.Parameter(torch.tensor(0.2))      # linear spring stiffness
        self.mkt_log_alpha = nn.Parameter(torch.tensor(-1.5)) # cubic Duffing snap-back
        self.mkt_log_g = nn.Parameter(torch.tensor(0.5))      # damping
        self.mkt_q0 = nn.Linear(f_sec, 1)
        self.mkt_v0 = nn.Linear(f_sec, 1)
        self.mkt_qeq = nn.Linear(f_sec, 1)
        self.mkt_force = nn.Linear(f_macro, 1)
        self.mkt_read_q = nn.Parameter(torch.tensor(0.15))
        self.mkt_read_v = nn.Parameter(torch.tensor(0.08))
        self.mkt_drift = nn.Parameter(torch.tensor(0.02))     # quarterly drift

        # --- 6D Zero-Sum Cross-Sector Coupled Duffing Oscillators ---
        self.rel_log_k = nn.Parameter(torch.full((n_sectors,), 0.2))
        self.rel_log_alpha = nn.Parameter(torch.full((n_sectors,), -1.5))
        self.rel_log_g = nn.Parameter(torch.full((n_sectors,), 0.5))

        # Macro-gated symmetric cross-sector coupling
        self.C_base = nn.Parameter(torch.zeros(n_sectors, n_sectors))
        self.C_stress = nn.Parameter(torch.zeros(n_sectors, n_sectors))
        self.regime_gate = nn.Linear(f_macro, 1)

        # Shared per-sector relative coordinate projections (6x statistical sharing!)
        self.rel_q0 = nn.Linear(f_sec, 1)
        self.rel_v0 = nn.Linear(f_sec, 1)
        self.rel_qeq = nn.Linear(f_sec, 1)
        self.rel_macro_force = nn.Linear(f_macro, n_sectors)

        self.rel_read_q = nn.Parameter(torch.ones(n_sectors) * 0.15)
        self.rel_read_v = nn.Parameter(torch.ones(n_sectors) * 0.08)

    def _sym_c(self, M: torch.Tensor) -> torch.Tensor:
        C = 0.5 * (M + M.T)
        C = C - torch.diag(torch.diag(C))
        return 0.25 * torch.tanh(C)

    def _mkt_accel(self, q, v, q_eq, f_ext):
        k = F.softplus(self.mkt_log_k)
        a3 = 0.25 * F.softplus(self.mkt_log_alpha)
        g = F.softplus(self.mkt_log_g)
        dq = torch.clamp(q - q_eq, -3.0, 3.0)
        return -k * dq - a3 * (dq ** 3) - g * v + f_ext

    def _rel_accel(self, q, v, q_eq, f_ext, gate):
        k = F.softplus(self.rel_log_k)
        a3 = 0.25 * F.softplus(self.rel_log_alpha)
        g = F.softplus(self.rel_log_g)
        Cb = self._sym_c(self.C_base)
        Cs = self._sym_c(self.C_stress)
        dq = torch.clamp(q - q_eq, -3.0, 3.0)
        coup = (1.0 - gate) * (q @ Cb) + gate * (q @ Cs)
        acc = -k * dq - a3 * (dq ** 3) - g * v + coup + f_ext
        # Keep relative acceleration zero-sum across the 6 sectors!
        return acc - acc.mean(dim=-1, keepdim=True)

    def integrate_decomposed(self, x_sec: torch.Tensor, x_macro: torch.Tensor):
        """Return quarterly (mkt_traj: (B, 4), rel_traj: (B, 4, 6)) via Velocity Verlet."""
        sec_now = x_sec[:, -1, :, :]
        sec_vel = x_sec[:, -1, :, :] - x_sec[:, -5, :, :]  # 1-quarter fundamental velocity
        mac_now = x_macro[:, -1, :]

        # --- 1) Market Common Mode ---
        sec_avg_now = sec_now.mean(dim=1)  # (B, Fs)
        sec_avg_vel = sec_vel.mean(dim=1)
        qm0 = self.mkt_q0(sec_avg_now).squeeze(-1)
        vm = self.mkt_v0(sec_avg_vel).squeeze(-1)
        qm_eq = self.mkt_qeq(sec_avg_now).squeeze(-1)
        fm_ext = self.mkt_force(mac_now).squeeze(-1)

        qm = qm0
        dt = 1.0 / self.n_steps
        am = self._mkt_accel(qm, vm, qm_eq, fm_ext)

        # --- 2) Cross-Sector Relative Mode (zero-centered inputs) ---
        sec_rel_now = sec_now - sec_now.mean(dim=1, keepdim=True)
        sec_rel_vel = sec_vel - sec_vel.mean(dim=1, keepdim=True)
        gate = torch.sigmoid(self.regime_gate(mac_now))  # (B, 1)

        qr0 = self.rel_q0(sec_rel_now).squeeze(-1)
        qr0 = qr0 - qr0.mean(dim=-1, keepdim=True)
        vr = self.rel_v0(sec_rel_vel).squeeze(-1)
        vr = vr - vr.mean(dim=-1, keepdim=True)
        qr_eq = self.rel_qeq(sec_rel_now).squeeze(-1)
        qr_eq = qr_eq - qr_eq.mean(dim=-1, keepdim=True)
        fr_ext = self.rel_macro_force(mac_now)
        fr_ext = fr_ext - fr_ext.mean(dim=-1, keepdim=True)

        qr = qr0
        ar = self._rel_accel(qr, vr, qr_eq, fr_ext, gate)

        mkt_qs, rel_qs = [], []
        for step in range(1, self.n_steps + 1):
            # Market Verlet step
            qm = qm + vm * dt + 0.5 * am * (dt * dt)
            am_new = self._mkt_accel(qm, vm, qm_eq, fm_ext)
            vm = vm + 0.5 * (am + am_new) * dt
            am = am_new

            # Relative Sector Verlet step
            qr = qr + vr * dt + 0.5 * ar * (dt * dt)
            ar_new = self._rel_accel(qr, vr, qr_eq, fr_ext, gate)
            vr = vr + 0.5 * (ar + ar_new) * dt
            ar = ar_new

            if step % (self.n_steps // 4) == 0:
                qi = step // (self.n_steps // 4)
                mkt_ret = self.mkt_read_q * (qm - qm0) + self.mkt_read_v * vm + self.mkt_drift * qi
                rel_ret = self.rel_read_q * (qr - qr0) + self.rel_read_v * vr
                rel_ret = rel_ret - rel_ret.mean(dim=-1, keepdim=True)
                mkt_qs.append(mkt_ret)
                rel_qs.append(rel_ret)

        return torch.stack(mkt_qs, dim=1), torch.stack(rel_qs, dim=1)

    def forward(self, x_sec: torch.Tensor, x_macro: torch.Tensor) -> torch.Tensor:
        mkt_q, rel_q = self.integrate_decomposed(x_sec, x_macro)
        return mkt_q[:, -1:].expand_as(rel_q[:, -1, :]) + rel_q[:, -1, :]

    @staticmethod
    def decomposed_loss(mkt_q: torch.Tensor, rel_q: torch.Tensor, y_multi: torch.Tensor) -> torch.Tensor:
        y_mkt = y_multi.mean(dim=-1)                        # (B, 4)
        y_rel = y_multi - y_mkt.unsqueeze(-1)               # (B, 4, 6)
        w = torch.tensor([0.15, 0.25, 0.30, 1.0], device=y_multi.device)
        mkt_loss = (F.smooth_l1_loss(mkt_q, y_mkt, beta=0.1, reduction="none") * w.unsqueeze(0)).mean()
        rel_loss = (F.smooth_l1_loss(rel_q, y_rel, beta=0.05, reduction="none") * w.view(1, 4, 1)).mean()
        return mkt_loss + 1.2 * rel_loss


def train_duffing_v3_ensemble(
    X_sec_t: torch.Tensor,
    X_mac_t: torch.Tensor,
    Y_multi_t: torch.Tensor,
    tr: np.ndarray,
    va: np.ndarray,
    n_seeds: int = 3,
    epochs: int = 60,
):
    """Train a 3-seed DuffingOscillatorV3 ensemble strictly on Train, early-stopped on Val."""
    S, Fs, Fm = X_sec_t.shape[2], X_sec_t.shape[3], X_mac_t.shape[2]
    models = []
    for s_idx in range(n_seeds):
        torch.manual_seed(100 + s_idx)
        np.random.seed(100 + s_idx)
        model = DuffingOscillatorV3Net(n_sectors=S, f_sec=Fs, f_macro=Fm, n_steps=8)
        opt = torch.optim.AdamW(model.parameters(), lr=4e-3, weight_decay=3e-3)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

        best_v = float("inf")
        best_st = copy.deepcopy(model.state_dict())

        for _ in range(epochs):
            model.train()
            perm = torch.randperm(len(tr))
            for s0 in range(0, len(tr), 128):
                idx = tr[perm[s0 : s0 + 128].numpy()]
                mkt_q, rel_q = model.integrate_decomposed(X_sec_t[idx], X_mac_t[idx])
                loss = DuffingOscillatorV3Net.decomposed_loss(mkt_q, rel_q, Y_multi_t[idx])
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
                opt.step()
            sched.step()

            model.eval()
            with torch.no_grad():
                pred_va = model(X_sec_t[va], X_mac_t[va])
                vl = F.mse_loss(pred_va, Y_multi_t[va, -1, :]).item()
                if vl < best_v:
                    best_v, best_st = vl, copy.deepcopy(model.state_dict())

        model.load_state_dict(best_st)
        model.eval()
        models.append(model)
    return models
