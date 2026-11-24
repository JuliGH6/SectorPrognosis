"""
Paradigm 2: Latent Economic World Model (JEPA + Neural-ODE Imagination Rollout)
-------------------------------------------------------------------------------
Instead of jumping blindly from t -> t+250 in one black-box regression, this model
learns an internal simulator of the macro-sector economy:

1. Context Encoder E_theta(x_t) -> z_t in R^d: compresses the trailing 1-year macro
   and 6-sector state into a compact latent economic manifold.
2. Continuous-Time Dynamics Operator (RK2 Midpoint Neural ODE step): rolls the latent
   economy forward quarter-by-quarter:
       z_t -> z_hat_{t+1q} -> z_hat_{t+2q} -> z_hat_{t+3q} -> z_hat_{t+4q}
3. Self-Supervised JEPA Latent Consistency + VICReg Anti-Collapse:
   During training, a stop-grad EMA target encoder embeds the *actual* future economic
   states at +63d, +126d, +189d, +250d (strictly within the training set!). The world
   model is trained so its imagined trajectory matches the real future latent states,
   while VICReg variance/covariance terms prevent representation collapse.
4. Multi-Horizon Trajectory Decoder: decodes intermediate quarterly returns (+3m, +6m,
   +9m, +12m) directly from the imagined latent trajectory!
"""

from __future__ import annotations

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F


class StateEncoder(nn.Module):
    """Encodes a trailing 1-year macro+sector window into latent economic state z_t."""

    def __init__(self, in_dim: int, latent_dim: int = 32, dropout: float = 0.2):
        super().__init__()
        # Lightweight 1D conv over time + pooling
        self.conv = nn.Sequential(
            nn.Conv1d(in_dim, 64, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.proj = nn.Sequential(
            nn.LayerNorm(128),
            nn.Dropout(dropout),
            nn.Linear(128, latent_dim),
        )

    def forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        # x_seq: (B, T, D)
        h = self.conv(x_seq.transpose(1, 2))  # (B, 64, T)
        # Combine latest state and mean temporal context
        h_last = h[:, :, -1]
        h_mean = h.mean(dim=-1)
        return self.proj(torch.cat([h_last, h_mean], dim=-1))


class LatentDynamicsODE(nn.Module):
    """RK2 (Midpoint) Neural ODE transition step advancing latent economic state by 1 quarter."""

    def __init__(self, latent_dim: int = 32, hidden_dim: int = 64):
        super().__init__()
        self.f = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, latent_dim),
        )
        # Small initialization so initial dynamics start near identity
        nn.init.normal_(self.f[-1].weight, std=0.02)
        nn.init.zeros_(self.f[-1].bias)

    def step(self, z: torch.Tensor, dt: float = 0.25) -> torch.Tensor:
        # RK2 midpoint integration
        k1 = self.f(z)
        z_mid = z + 0.5 * dt * k1
        k2 = self.f(z_mid)
        return z + dt * k2


def vicreg_loss(z: torch.Tensor, sim_weight: float = 0.0, var_weight: float = 1.0, cov_weight: float = 0.2) -> torch.Tensor:
    """Variance + Covariance regularization to prevent latent manifold collapse."""
    # z: (B, D)
    B, D = z.shape
    z_centered = z - z.mean(dim=0, keepdim=True)
    std = torch.sqrt(z_centered.var(dim=0) + 1e-4)
    var_loss = F.relu(1.0 - std).mean()

    cov = (z_centered.T @ z_centered) / max(B - 1, 1)
    off_diag = cov - torch.diag(torch.diag(cov))
    cov_loss = (off_diag.pow(2).sum()) / D
    return var_weight * var_loss + cov_weight * cov_loss


class EconomicWorldModel(nn.Module):
    """JEPA-style Latent Economic World Model with 4-quarter Neural-ODE imagination rollout."""

    def __init__(
        self,
        n_sectors: int = 6,
        f_sec: int = 16,
        f_macro: int = 15,
        latent_dim: int = 32,
        dropout: float = 0.2,
    ):
        super().__init__()
        in_dim = n_sectors * f_sec + f_macro
        self.encoder = StateEncoder(in_dim, latent_dim=latent_dim, dropout=dropout)
        # EMA target encoder (stop-grad) for JEPA latent supervision
        self.target_encoder = copy.deepcopy(self.encoder)
        for p in self.target_encoder.parameters():
            p.requires_grad = False

        self.dynamics = LatentDynamicsODE(latent_dim=latent_dim, hidden_dim=64)

        # Shared step return decoder: maps (z_0, z_hat_k) -> cumulative sector returns at step k
        self.step_decoder = nn.Sequential(
            nn.LayerNorm(latent_dim * 2),
            nn.Linear(latent_dim * 2, 48),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(48, n_sectors),
        )

    @torch.no_grad()
    def update_target_encoder(self, ema_decay: float = 0.99):
        for p_online, p_target in zip(self.encoder.parameters(), self.target_encoder.parameters()):
            p_target.data.mul_(ema_decay).add_(p_online.data, alpha=1.0 - ema_decay)

    def _pack(self, x_sec: torch.Tensor, x_macro: torch.Tensor) -> torch.Tensor:
        B, T, S, Fs = x_sec.shape
        return torch.cat([x_sec.reshape(B, T, S * Fs), x_macro], dim=-1)

    def imagine(self, x_sec: torch.Tensor, x_macro: torch.Tensor, steps: int = 4):
        """Encode current state and roll forward `steps` quarters in latent space."""
        z0 = self.encoder(self._pack(x_sec, x_macro))
        z_traj = []
        ret_traj = []
        z = z0
        for _ in range(steps):
            z = self.dynamics.step(z, dt=0.25)
            z_traj.append(z)
            ret_traj.append(self.step_decoder(torch.cat([z0, z], dim=-1)))
        return z0, torch.stack(z_traj, dim=1), torch.stack(ret_traj, dim=1)

    def forward(self, x_sec: torch.Tensor, x_macro: torch.Tensor) -> torch.Tensor:
        """Predict 1-year (4th quarter) sector returns from the imagined trajectory."""
        _, _, ret_traj = self.imagine(x_sec, x_macro, steps=4)
        return ret_traj[:, -1, :]

    def compute_loss(
        self,
        x_sec: torch.Tensor,
        x_macro: torch.Tensor,
        y_multi: torch.Tensor,
        fut_x_sec: torch.Tensor,
        fut_x_macro: torch.Tensor,
        jepa_weight: float = 0.3,
        vicreg_weight: float = 0.05,
    ) -> torch.Tensor:
        """
        Joint loss:
        1. Multi-horizon return supervision (+3m, +6m, +9m, +12m) with extra weight on +12m.
        2. Self-supervised JEPA latent rollout consistency against EMA target encoder.
        3. VICReg anti-collapse regularization on latent manifold.
        """
        z0, z_hat_traj, ret_traj = self.imagine(x_sec, x_macro, steps=4)

        # Multi-horizon Huber loss (gives intermediate trajectory supervision)
        weights = torch.tensor([0.15, 0.25, 0.30, 1.0], device=x_sec.device).view(1, 4, 1)
        ret_loss = (F.smooth_l1_loss(ret_traj, y_multi, beta=0.1, reduction="none") * weights).mean()

        # Target latents for +3m, +6m, +9m, +12m (stop-grad)
        B, K = fut_x_sec.shape[0], fut_x_sec.shape[1]
        with torch.no_grad():
            fut_flat_sec = fut_x_sec.reshape(B * K, *fut_x_sec.shape[2:])
            fut_flat_mac = fut_x_macro.reshape(B * K, *fut_x_macro.shape[2:])
            z_target = self.target_encoder(self._pack(fut_flat_sec, fut_flat_mac)).reshape(B, K, -1)

        # Cosine + MSE latent matching
        jepa_loss = F.mse_loss(F.normalize(z_hat_traj, dim=-1), F.normalize(z_target, dim=-1))
        reg_loss = vicreg_loss(z0)

        return ret_loss + jepa_weight * jepa_loss + vicreg_weight * reg_loss
