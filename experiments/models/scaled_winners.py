"""
Scaled-Up Implementations of Our Two Breakout Paradigms + Their Hybrid Fusion:
  1. MultiScalePathSignatureForecaster (RoughPath v2):
     - Multi-window Depth-2 Signatures & Signed Levy Areas: full-year (16 steps) + recent quarter (5 steps)
     - Time-augmented paths (captures *when* in the year a move happened)
     - Dedicated Cross-Sector Relative Signatures (each sector minus cross-sector mean -> explicit rotation lead-lag)
  2. RegimeCoupledOscillatorNet (Coupled Oscillator v2):
     - 2D mechanical modes per sector (e.g. valuation spring + momentum/earnings spring)
     - Macro-modulated cross-sector spring coupling C(x_macro)
     - Multi-horizon Velocity Verlet trajectory supervision (+3m, +6m, +9m, +12m)
  3. SignatureOscillatorHybrid:
     - Uses Rough-Path Signed Levy-Area event-order features as the external driving force
       inside the Coupled Damped Harmonic Oscillator physics integrator!
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from .path_signatures import compute_depth2_signatures


def _time_augment(paths: np.ndarray) -> np.ndarray:
    """Append normalized time channel t in [-1, 1] so Depth-2 signatures capture timing."""
    N, T, _ = paths.shape
    t_ch = np.linspace(-1.0, 1.0, T, dtype=np.float32)[None, :, None]
    t_ch = np.broadcast_to(t_ch, (N, T, 1))
    return np.concatenate([paths, t_ch], axis=-1)


class MultiScalePathSignatureForecaster:
    """
    RoughPath v2: Multi-scale (1Y + 3M) time-augmented + cross-sector relative
    Depth-2 Signatures & Signed Levy Areas with closed-form Ridge readout.
    """

    def __init__(self, n_macro_pcs: int = 8, n_rel_pcs: int = 6, ridge_alpha: float = 15000.0):
        self.n_macro_pcs = n_macro_pcs
        self.n_rel_pcs = n_rel_pcs
        self.ridge_alpha = ridge_alpha

    def _split_views(self, X_sec: np.ndarray, X_macro: np.ndarray):
        N, T, S, Fs = X_sec.shape
        # Global joint state (macro + all sectors)
        joint = np.concatenate([X_sec.reshape(N, T, S * Fs), X_macro], axis=-1)
        # Cross-sector relative state (each sector minus cross-sector mean -> pure rotation!)
        sec_rel = (X_sec - X_sec.mean(axis=2, keepdims=True)).reshape(N, T, S * Fs)
        return joint, sec_rel

    def _extract_features(self, X_sec: np.ndarray, X_macro: np.ndarray, fit: bool = False) -> np.ndarray:
        joint, sec_rel = self._split_views(X_sec, X_macro)
        N, T, Dj = joint.shape
        Dr = sec_rel.shape[-1]

        if fit:
            self.pca_joint = PCA(n_components=self.n_macro_pcs, random_state=42).fit(joint.reshape(N * T, Dj))
            self.pca_rel = PCA(n_components=self.n_rel_pcs, random_state=42).fit(sec_rel.reshape(N * T, Dr))

        p_joint = self.pca_joint.transform(joint.reshape(N * T, Dj)).reshape(N, T, self.n_macro_pcs)
        p_rel = self.pca_rel.transform(sec_rel.reshape(N * T, Dr)).reshape(N, T, self.n_rel_pcs)

        # Full-year (16 steps) time-augmented signatures
        sig_joint_1y = compute_depth2_signatures(_time_augment(p_joint))
        sig_rel_1y = compute_depth2_signatures(_time_augment(p_rel))

        # Short-horizon (last 5 steps ~ 1 quarter) signatures for recent inflection order
        sig_joint_1q = compute_depth2_signatures(p_joint[:, -5:, :])
        sig_rel_1q = compute_depth2_signatures(p_rel[:, -5:, :])

        return np.concatenate([sig_joint_1y, sig_rel_1y, sig_joint_1q, sig_rel_1q], axis=1).astype(np.float32)

    def fit(self, X_sec: np.ndarray, X_macro: np.ndarray, Y: np.ndarray):
        Phi = self._extract_features(X_sec, X_macro, fit=True)
        self.scaler = StandardScaler().fit(Phi)
        self.readout = Ridge(alpha=self.ridge_alpha).fit(self.scaler.transform(Phi), Y)
        return self

    def predict(self, X_sec: np.ndarray, X_macro: np.ndarray) -> np.ndarray:
        Phi = self._extract_features(X_sec, X_macro, fit=False)
        return self.readout.predict(self.scaler.transform(Phi)).astype(np.float32)

    def transform_for_hybrid(self, X_sec: np.ndarray, X_macro: np.ndarray) -> np.ndarray:
        """Return standardized signature features for downstream physics forcing."""
        Phi = self._extract_features(X_sec, X_macro, fit=False)
        return np.clip(self.scaler.transform(Phi), -5.0, 5.0).astype(np.float32)


class RegimeCoupledOscillatorNet(nn.Module):
    """
    Coupled Oscillator v2:
      - 2 mechanical modes per sector (e.g. valuation spring + earnings/momentum spring)
      - Macro-gated cross-sector spring coupling C(x_macro) = C_base + g(macro) * C_stress
      - 8 Velocity Verlet steps returning quarterly snapshots (+3m, +6m, +9m, +12m) for
        multi-horizon trajectory supervision!
    """

    def __init__(
        self,
        n_sectors: int = 6,
        f_sec: int = 16,
        f_macro: int = 15,
        n_modes: int = 2,
        n_steps: int = 8,
        extra_force_dim: int = 0,
    ):
        super().__init__()
        self.n_sectors = n_sectors
        self.n_modes = n_modes
        self.n_steps = n_steps
        self.extra_force_dim = extra_force_dim

        # Per-sector, per-mode spring stiffness K > 0 and damping Gamma > 0
        self.log_stiffness = nn.Parameter(torch.full((n_sectors, n_modes), 0.2))
        self.log_damping = nn.Parameter(torch.full((n_sectors, n_modes), 0.5))

        # Two symmetric cross-sector coupling matrices blended by a macro regime gate
        self.C_base = nn.Parameter(torch.zeros(n_sectors, n_sectors))
        self.C_stress = nn.Parameter(torch.zeros(n_sectors, n_sectors))
        self.regime_gate = nn.Linear(f_macro * 2, 1)

        # Shared per-sector linear maps to 2D mechanical coordinates
        self.q0_proj = nn.Linear(f_sec, n_modes)
        self.v0_proj = nn.Linear(f_sec, n_modes)
        self.q_eq_proj = nn.Linear(f_sec, n_modes)

        # External forcing from macro (+ optional Rough-Path Signature features!)
        force_in = f_macro * 2 + extra_force_dim
        self.force_drop = nn.Dropout(0.2)
        self.ext_force = nn.Linear(force_in, n_sectors * n_modes)

        # Readout from (q - q0, v) across the 2 modes -> scalar return per sector
        self.readout_w = nn.Parameter(torch.ones(n_sectors, n_modes * 2) * 0.1)
        self.bias = nn.Parameter(torch.full((n_sectors,), 0.02))  # quarterly drift scale

    def _sym_coupling(self, M: torch.Tensor) -> torch.Tensor:
        C = 0.5 * (M + M.T)
        C = C - torch.diag(torch.diag(C))
        return 0.2 * torch.tanh(C)

    def _acceleration(
        self,
        q: torch.Tensor,
        v: torch.Tensor,
        q_eq: torch.Tensor,
        F_ext: torch.Tensor,
        gate: torch.Tensor,
    ) -> torch.Tensor:
        # q, v, q_eq, F_ext: (B, S, M), gate: (B, 1, 1)
        K = F.softplus(self.log_stiffness).unsqueeze(0)   # (1, S, M)
        G = F.softplus(self.log_damping).unsqueeze(0)     # (1, S, M)
        Cb = self._sym_coupling(self.C_base)              # (S, S)
        Cs = self._sym_coupling(self.C_stress)            # (S, S)

        # Coupled spring force across sectors
        coup_b = torch.einsum("bsm,st->btm", q, Cb)
        coup_s = torch.einsum("bsm,st->btm", q, Cs)
        coup = (1.0 - gate) * coup_b + gate * coup_s
        return -K * (q - q_eq) - G * v + coup + F_ext

    def integrate_trajectory(
        self,
        x_sec: torch.Tensor,
        x_macro: torch.Tensor,
        extra_force: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Integrate 8 Verlet steps and return quarterly predictions (B, 4, 6) at steps 2, 4, 6, 8."""
        B = x_sec.shape[0]
        sec_now = x_sec[:, -1, :, :]
        sec_delta = x_sec[:, -1, :, :] - x_sec[:, 0, :, :]

        mac_now = x_macro[:, -1, :]
        mac_delta = x_macro[:, -1, :] - x_macro[:, 0, :]
        mac_feat = torch.cat([mac_now, mac_delta], dim=-1)  # (B, 2*F_macro)

        gate = torch.sigmoid(self.regime_gate(mac_feat)).unsqueeze(-1)  # (B, 1, 1)

        if extra_force is not None:
            f_in = torch.cat([mac_feat, self.force_drop(extra_force)], dim=-1)
        else:
            f_in = mac_feat

        q0 = self.q0_proj(sec_now)          # (B, S, M)
        v = self.v0_proj(sec_delta)         # (B, S, M)
        q_eq = self.q_eq_proj(sec_now)      # (B, S, M)
        F_ext = self.ext_force(f_in).reshape(B, self.n_sectors, self.n_modes)

        q = q0
        dt = 1.0 / self.n_steps
        a = self._acceleration(q, v, q_eq, F_ext, gate)

        quarterly = []
        for step in range(1, self.n_steps + 1):
            q = q + v * dt + 0.5 * a * (dt * dt)
            a_new = self._acceleration(q, v, q_eq, F_ext, gate)
            v = v + 0.5 * (a + a_new) * dt
            a = a_new

            if step % (self.n_steps // 4) == 0:
                q_idx = step // (self.n_steps // 4)  # 1, 2, 3, 4
                state = torch.cat([q - q0, v], dim=-1)  # (B, S, 2*M)
                ret = (state * self.readout_w.unsqueeze(0)).sum(dim=-1) + self.bias * q_idx
                quarterly.append(ret)

        return torch.stack(quarterly, dim=1)  # (B, 4, 6)

    def forward(
        self,
        x_sec: torch.Tensor,
        x_macro: torch.Tensor,
        extra_force: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.integrate_trajectory(x_sec, x_macro, extra_force)[:, -1, :]

    @staticmethod
    def multi_horizon_loss(traj_pred: torch.Tensor, y_multi: torch.Tensor) -> torch.Tensor:
        """Multi-horizon Smooth-L1 + cross-sectional rotation spread loss on 1Y horizon."""
        weights = torch.tensor([0.15, 0.25, 0.30, 1.0], device=traj_pred.device).view(1, 4, 1)
        abs_l = (F.smooth_l1_loss(traj_pred, y_multi, beta=0.1, reduction="none") * weights).mean()
        p1y = traj_pred[:, -1, :]
        y1y = y_multi[:, -1, :]
        rel_l = F.smooth_l1_loss(
            p1y - p1y.mean(dim=-1, keepdim=True),
            y1y - y1y.mean(dim=-1, keepdim=True),
            beta=0.05,
        )
        return abs_l + 0.5 * rel_l
