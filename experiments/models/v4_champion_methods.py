"""
Iteration v4: Targeted Scaling & Refinement of Both Methods Based on v3 Diagnostics.

1. RoughPathV4Forecaster:
   - Fixes v3's Market-head variance by using a tight 5-PC Depth-2 Signature + Levy Area
     market head with shrinkage toward historical training drift.
   - Upgrades the Rotation head by combining:
       (a) Macro-to-Sector Cross-Levy Areas (v3's breakout rotation driver)
       (b) Weight-Shared Per-Sector Fundamental Signatures (all 6 sectors share ONE
           signature-to-relative-return scorer -> 6x statistical efficiency!).

2. DuffingOscillatorV4Net:
   - Upgrades v3's Duffing Oscillator with:
       (a) Low-rank 4-factor Macro Forcing bottleneck (Growth/Inflation/Labor/Housing)
           preventing macro-to-sector overfitting on early folds.
       (b) Mild causal recency weighting within Train (half-life ~8y, strictly in-train).
       (c) Cosine-similarity cross-sectional ranking term alongside Smooth-L1 trajectory loss.
"""

from __future__ import annotations

import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from .path_signatures import compute_depth2_signatures
from .v3_scaled_methods import compute_enriched_signatures, _time_augment


class RoughPathV4Forecaster:
    """
    v4 Decomposed Rough-Path Forecaster:
      - Tight 5-PC Market Signature Head (low variance, high directional accuracy)
      - Hybrid Cross-Levy + Weight-Shared Per-Sector Signature Rotation Head
    All transforms & hyperparameters fit strictly on Train and selected strictly on Val.
    """

    def __init__(self, n_mkt_pcs: int = 5, n_rot_pcs: int = 8, n_sec_pcs: int = 4):
        self.n_mkt_pcs = n_mkt_pcs
        self.n_rot_pcs = n_rot_pcs
        self.n_sec_pcs = n_sec_pcs

    def _build_views(self, X_sec: np.ndarray, X_macro: np.ndarray):
        N, T, S, Fs = X_sec.shape
        sec_mean = X_sec.mean(axis=2)
        mkt_raw = np.concatenate([X_macro, sec_mean], axis=-1)
        sec_rel = X_sec - X_sec.mean(axis=2, keepdims=True)  # (N, T, 6, Fs)
        return mkt_raw, sec_rel, X_macro

    def _fit_transforms(self, X_sec_tr: np.ndarray, X_mac_tr: np.ndarray):
        mkt_raw, sec_rel, mac_raw = self._build_views(X_sec_tr, X_mac_tr)
        N, T, S, Fs = sec_rel.shape

        # 1) Tight Market PCA (5 dims)
        self.pca_mkt = PCA(n_components=self.n_mkt_pcs, random_state=42).fit(
            mkt_raw.reshape(N * T, mkt_raw.shape[-1])
        )
        # 2) Global Relative Sector PCA (8 dims) + Macro Cross-Levy PCA (4 dims)
        self.pca_rot = PCA(n_components=self.n_rot_pcs, random_state=42).fit(
            sec_rel.reshape(N * T, S * Fs)
        )
        self.pca_mac = PCA(n_components=4, random_state=42).fit(
            mac_raw.reshape(N * T, mac_raw.shape[-1])
        )
        # 3) Weight-Shared Per-Sector Relative PCA (fit across all 6 sectors pooled!)
        self.pca_sec_shared = PCA(n_components=self.n_sec_pcs, random_state=42).fit(
            sec_rel.reshape(N * T * S, Fs)
        )

        fm, fr, fs = self._extract_features(X_sec_tr, X_mac_tr)
        self.sc_mkt = StandardScaler().fit(fm)
        self.sc_rot = StandardScaler().fit(fr)
        self.sc_sec = StandardScaler().fit(fs.reshape(N * S, -1))
        return (
            self.sc_mkt.transform(fm),
            self.sc_rot.transform(fr),
            self.sc_sec.transform(fs.reshape(N * S, -1)).reshape(N, S, -1),
        )

    def _extract_features(self, X_sec: np.ndarray, X_macro: np.ndarray):
        mkt_raw, sec_rel, mac_raw = self._build_views(X_sec, X_macro)
        N, T, S, Fs = sec_rel.shape

        # Market: pure Depth-2 + Levy Areas on 5 PCs (1Y time-aug + 1Q)
        p_mkt = self.pca_mkt.transform(mkt_raw.reshape(N * T, mkt_raw.shape[-1])).reshape(N, T, self.n_mkt_pcs)
        phi_mkt = np.concatenate(
            [
                compute_depth2_signatures(_time_augment(p_mkt)),
                compute_depth2_signatures(p_mkt[:, -5:, :]),
            ],
            axis=1,
        )

        # Global Rotation + Macro-to-Sector Cross-Levy Areas
        p_rot = self.pca_rot.transform(sec_rel.reshape(N * T, S * Fs)).reshape(N, T, self.n_rot_pcs)
        p_mac = self.pca_mac.transform(mac_raw.reshape(N * T, mac_raw.shape[-1])).reshape(N, T, 4)

        dM = np.diff(p_mac, axis=1)
        dR = np.diff(p_rot[:, :, :6], axis=1)
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

        # Weight-Shared Per-Sector Relative Path Signatures: (N, 6, F_sig_sec)
        p_sec = self.pca_sec_shared.transform(sec_rel.reshape(N * T * S, Fs)).reshape(N, T, S, self.n_sec_pcs)
        p_sec_flat = p_sec.transpose(0, 2, 1, 3).reshape(N * S, T, self.n_sec_pcs)
        phi_sec = np.concatenate(
            [
                compute_depth2_signatures(_time_augment(p_sec_flat)),
                compute_depth2_signatures(p_sec_flat[:, -5:, :]),
            ],
            axis=1,
        ).reshape(N, S, -1)

        return phi_mkt, phi_rot, phi_sec

    def fit_select(
        self,
        X_sec_tr: np.ndarray,
        X_mac_tr: np.ndarray,
        Y_tr: np.ndarray,
        X_sec_va: np.ndarray,
        X_mac_va: np.ndarray,
        Y_va: np.ndarray,
    ):
        N_tr, _, S, _ = X_sec_tr.shape
        N_va = X_sec_va.shape[0]
        Zm_tr, Zr_tr, Zs_tr = self._fit_transforms(X_sec_tr, X_mac_tr)
        fm_va, fr_va, fs_va = self._extract_features(X_sec_va, X_mac_va)
        Zm_va = self.sc_mkt.transform(fm_va)
        Zr_va = self.sc_rot.transform(fr_va)
        Zs_va = self.sc_sec.transform(fs_va.reshape(N_va * S, -1)).reshape(N_va, S, -1)

        y_mkt_tr = Y_tr.mean(axis=1)
        y_mkt_va = Y_va.mean(axis=1)
        y_rot_tr = Y_tr - y_mkt_tr[:, None]
        y_rot_va = Y_va - y_mkt_va[:, None]
        self.train_mkt_mean = float(y_mkt_tr.mean())

        # 1) Market head: Ridge + optimal drift shrinkage selected strictly on Val
        best_am, best_shrink, best_vm = 20000.0, 0.0, float("inf")
        for a in [1000.0, 5000.0, 20000.0, 60000.0, 150000.0]:
            reg = Ridge(alpha=a).fit(Zm_tr, y_mkt_tr)
            p = reg.predict(Zm_va)
            for sh in [0.0, 0.25, 0.5]:
                ps = (1.0 - sh) * p + sh * self.train_mkt_mean
                vl = float(np.mean((ps - y_mkt_va) ** 2))
                if vl < best_vm:
                    best_vm, best_am, best_shrink = vl, a, sh
        self.mkt_ridge = Ridge(alpha=best_am).fit(Zm_tr, y_mkt_tr)
        self.mkt_shrink = best_shrink

        # 2) Global Cross-Levy Rotation Head
        best_ar, best_vr = 10000.0, float("inf")
        for a in [1000.0, 5000.0, 20000.0, 60000.0]:
            reg = Ridge(alpha=a).fit(Zr_tr, y_rot_tr)
            p = reg.predict(Zr_va)
            p -= p.mean(axis=1, keepdims=True)
            vl = float(np.mean((p - y_rot_va) ** 2))
            if vl < best_vr:
                best_vr, best_ar = vl, a
        self.rot_global = Ridge(alpha=best_ar).fit(Zr_tr, y_rot_tr)
        p_glob_va = self.rot_global.predict(Zr_va)
        p_glob_va -= p_glob_va.mean(axis=1, keepdims=True)

        # 3) Weight-Shared Per-Sector Signature Rotation Head (6x pooled samples!)
        best_as, best_vs = 10000.0, float("inf")
        Zs_tr_flat = Zs_tr.reshape(N_tr * S, -1)
        yr_tr_flat = y_rot_tr.reshape(N_tr * S)
        Zs_va_flat = Zs_va.reshape(N_va * S, -1)
        for a in [1000.0, 5000.0, 20000.0, 60000.0]:
            reg = Ridge(alpha=a).fit(Zs_tr_flat, yr_tr_flat)
            p = reg.predict(Zs_va_flat).reshape(N_va, S)
            p -= p.mean(axis=1, keepdims=True)
            vl = float(np.mean((p - y_rot_va) ** 2))
            if vl < best_vs:
                best_vs, best_as = vl, a
        self.rot_shared = Ridge(alpha=best_as).fit(Zs_tr_flat, yr_tr_flat)
        p_shar_va = self.rot_shared.predict(Zs_va_flat).reshape(N_va, S)
        p_shar_va -= p_shar_va.mean(axis=1, keepdims=True)

        # Select optimal blend between global Cross-Levy and shared per-sector rotation strictly on Val
        best_w, best_vb = 0.5, float("inf")
        for w in np.linspace(0.0, 1.0, 11):
            pb = w * p_glob_va + (1.0 - w) * p_shar_va
            vl = float(np.mean((pb - y_rot_va) ** 2))
            if vl < best_vb:
                best_vb, best_w = vl, float(w)
        self.rot_glob_w = best_w
        return self

    def predict(self, X_sec: np.ndarray, X_macro: np.ndarray) -> np.ndarray:
        N, _, S, _ = X_sec.shape
        fm, fr, fs = self._extract_features(X_sec, X_macro)
        Zm = self.sc_mkt.transform(fm)
        Zr = self.sc_rot.transform(fr)
        Zs = self.sc_sec.transform(fs.reshape(N * S, -1))

        p_mkt = (1.0 - self.mkt_shrink) * self.mkt_ridge.predict(Zm) + self.mkt_shrink * self.train_mkt_mean
        p_glob = self.rot_global.predict(Zr)
        p_glob -= p_glob.mean(axis=1, keepdims=True)
        p_shar = self.rot_shared.predict(Zs).reshape(N, S)
        p_shar -= p_shar.mean(axis=1, keepdims=True)
        p_rot = self.rot_glob_w * p_glob + (1.0 - self.rot_glob_w) * p_shar
        return (p_mkt[:, None] + p_rot).astype(np.float32)


class DuffingOscillatorV4Net(nn.Module):
    """
    v4 Nonlinear Coupled Duffing Oscillator with Low-Rank 4-Factor Macro Forcing Bottleneck.
    Cuts macro-to-sector overfitting while preserving full nonlinear physics interpretability.
    """

    def __init__(self, n_sectors: int = 6, f_sec: int = 16, f_macro: int = 33, n_steps: int = 8):
        super().__init__()
        self.n_sectors = n_sectors
        self.n_steps = n_steps

        # Shared low-rank 4-factor macro bottleneck (Growth / Inflation / Labor / Housing)
        self.macro_bottleneck = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(f_macro, 4),
            nn.Tanh(),
        )

        # --- Market Common-Mode 1D Nonlinear Duffing Oscillator ---
        self.mkt_log_k = nn.Parameter(torch.tensor(0.2))
        self.mkt_log_alpha = nn.Parameter(torch.tensor(-1.5))
        self.mkt_log_g = nn.Parameter(torch.tensor(0.5))
        self.mkt_q0 = nn.Linear(f_sec, 1)
        self.mkt_v0 = nn.Linear(f_sec, 1)
        self.mkt_qeq = nn.Linear(f_sec, 1)
        self.mkt_force = nn.Linear(4, 1)
        self.mkt_read_q = nn.Parameter(torch.tensor(0.15))
        self.mkt_read_v = nn.Parameter(torch.tensor(0.08))
        self.mkt_drift = nn.Parameter(torch.tensor(0.02))

        # --- 6D Zero-Sum Coupled Sector Duffing Oscillators ---
        self.rel_log_k = nn.Parameter(torch.full((n_sectors,), 0.2))
        self.rel_log_alpha = nn.Parameter(torch.full((n_sectors,), -1.5))
        self.rel_log_g = nn.Parameter(torch.full((n_sectors,), 0.5))

        self.C_base = nn.Parameter(torch.zeros(n_sectors, n_sectors))
        self.C_stress = nn.Parameter(torch.zeros(n_sectors, n_sectors))
        self.regime_gate = nn.Linear(4, 1)

        self.rel_q0 = nn.Linear(f_sec, 1)
        self.rel_v0 = nn.Linear(f_sec, 1)
        self.rel_qeq = nn.Linear(f_sec, 1)
        self.rel_macro_force = nn.Linear(4, n_sectors)

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
        return acc - acc.mean(dim=-1, keepdim=True)

    def integrate_decomposed(self, x_sec: torch.Tensor, x_macro: torch.Tensor):
        sec_now = x_sec[:, -1, :, :]
        sec_vel = x_sec[:, -1, :, :] - x_sec[:, -5, :, :]
        mac_4 = self.macro_bottleneck(x_macro[:, -1, :])  # (B, 4)

        sec_avg_now = sec_now.mean(dim=1)
        sec_avg_vel = sec_vel.mean(dim=1)
        qm0 = self.mkt_q0(sec_avg_now).squeeze(-1)
        vm = self.mkt_v0(sec_avg_vel).squeeze(-1)
        qm_eq = self.mkt_qeq(sec_avg_now).squeeze(-1)
        fm_ext = self.mkt_force(mac_4).squeeze(-1)

        qm = qm0
        dt = 1.0 / self.n_steps
        am = self._mkt_accel(qm, vm, qm_eq, fm_ext)

        sec_rel_now = sec_now - sec_now.mean(dim=1, keepdim=True)
        sec_rel_vel = sec_vel - sec_vel.mean(dim=1, keepdim=True)
        gate = torch.sigmoid(self.regime_gate(mac_4))

        qr0 = self.rel_q0(sec_rel_now).squeeze(-1); qr0 = qr0 - qr0.mean(dim=-1, keepdim=True)
        vr = self.rel_v0(sec_rel_vel).squeeze(-1); vr = vr - vr.mean(dim=-1, keepdim=True)
        qr_eq = self.rel_qeq(sec_rel_now).squeeze(-1); qr_eq = qr_eq - qr_eq.mean(dim=-1, keepdim=True)
        fr_ext = self.rel_macro_force(mac_4); fr_ext = fr_ext - fr_ext.mean(dim=-1, keepdim=True)

        qr = qr0
        ar = self._rel_accel(qr, vr, qr_eq, fr_ext, gate)

        mkt_qs, rel_qs = [], []
        for step in range(1, self.n_steps + 1):
            qm = qm + vm * dt + 0.5 * am * (dt * dt)
            am_new = self._mkt_accel(qm, vm, qm_eq, fm_ext)
            vm = vm + 0.5 * (am + am_new) * dt
            am = am_new

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
    def loss_fn(mkt_q: torch.Tensor, rel_q: torch.Tensor, y_multi: torch.Tensor, sample_w: torch.Tensor) -> torch.Tensor:
        y_mkt = y_multi.mean(dim=-1)
        y_rel = y_multi - y_mkt.unsqueeze(-1)
        hw = torch.tensor([0.15, 0.25, 0.30, 1.0], device=y_multi.device)

        mkt_l = (F.smooth_l1_loss(mkt_q, y_mkt, beta=0.1, reduction="none") * hw.unsqueeze(0)).mean(dim=1)
        rel_l = (F.smooth_l1_loss(rel_q, y_rel, beta=0.05, reduction="none") * hw.view(1, 4, 1)).mean(dim=(1, 2))
        # Cosine cross-sectional alignment on 1Y relative spreads (optimizes rotation IC!)
        cos_l = 1.0 - F.cosine_similarity(rel_q[:, -1, :], y_rel[:, -1, :], dim=-1, eps=1e-6)
        return ((mkt_l + 1.2 * rel_l + 0.05 * cos_l) * sample_w).mean()


def train_duffing_v4_ensemble(
    X_sec_t: torch.Tensor,
    X_mac_t: torch.Tensor,
    Y_multi_t: torch.Tensor,
    tr: np.ndarray,
    va: np.ndarray,
    n_seeds: int = 4,
    epochs: int = 60,
):
    """Train a 4-seed low-rank Duffing v4 ensemble strictly on Train with mild recency weighting."""
    S, Fs, Fm = X_sec_t.shape[2], X_sec_t.shape[3], X_mac_t.shape[2]
    # Mild causal recency weight within Train (half-life = 2000 trading days ~ 8 years)
    age = (tr[-1] - tr).astype(np.float32)
    w_tr = np.exp(-np.log(2.0) * age / 2000.0)
    w_tr = torch.from_numpy((w_tr / w_tr.mean()).astype(np.float32))

    models = []
    for s_idx in range(n_seeds):
        torch.manual_seed(200 + s_idx)
        np.random.seed(200 + s_idx)
        model = DuffingOscillatorV4Net(n_sectors=S, f_sec=Fs, f_macro=Fm, n_steps=8)
        opt = torch.optim.AdamW(model.parameters(), lr=4e-3, weight_decay=5e-3)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

        best_v = float("inf")
        best_st = copy.deepcopy(model.state_dict())

        for _ in range(epochs):
            model.train()
            perm = torch.randperm(len(tr))
            for s0 in range(0, len(tr), 128):
                loc = perm[s0 : s0 + 128]
                idx = tr[loc.numpy()]
                mkt_q, rel_q = model.integrate_decomposed(X_sec_t[idx], X_mac_t[idx])
                loss = DuffingOscillatorV4Net.loss_fn(mkt_q, rel_q, Y_multi_t[idx], w_tr[loc])
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
