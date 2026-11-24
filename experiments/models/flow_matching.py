"""
Paradigm 3: Conditional Flow Matching (Generative Diffusion-ODE)
----------------------------------------------------------------
Instead of predicting a single point estimate E[y|x] with MSE (which blurs
multimodal macro regimes like 'soft landing' vs 'recession'), we train a
Conditional Optimal-Transport Flow Matching (Rectified Flow) model over the
joint 6-sector 1-year return vector y_1 in R^6.

Training:
  Sample y_0 ~ N(0, I_6), t ~ Uniform(0, 1)
  Straight interpolation: y_t = (1 - t) * y_0 + t * y_1
  Target velocity:        u_t = y_1 - y_0
  Minimize || v_theta(y_t, t, c(x)) - (y_1 - y_0) ||^2

Inference:
  Sample K=32 Gaussian particles y_0^(k) ~ N(0, I_6) and integrate the learned
  ODE dy/dt = v_theta(y, t, c(x)) from t=0 to t=1 using midpoint (RK2) steps.
  This yields:
    - Mean forecast across K simulated futures
    - Calibrated epistemic/aleatoric predictive std per sector (uncertainty!)
    - Joint cross-sector return covariance matrix per date!
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalTimeEmb(nn.Module):
    def __init__(self, dim: int = 32):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # t: (B, 1) in [0, 1]
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(1000.0) * torch.arange(half, device=t.device, dtype=t.dtype) / max(half - 1, 1)
        )
        args = t * freqs.unsqueeze(0) * 2.0 * math.pi
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


class ContextEncoder(nn.Module):
    """Encodes the macro + sector time series into a conditioning vector c."""

    def __init__(self, in_dim: int, cond_dim: int = 64, dropout: float = 0.25):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_dim, 64, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.SiLU(),
        )
        self.proj = nn.Sequential(
            nn.LayerNorm(128),
            nn.Dropout(dropout),
            nn.Linear(128, cond_dim),
            nn.SiLU(),
        )

    def forward(self, x_sec: torch.Tensor, x_macro: torch.Tensor) -> torch.Tensor:
        B, T, S, Fs = x_sec.shape
        x = torch.cat([x_sec.reshape(B, T, S * Fs), x_macro], dim=-1)
        h = self.net(x.transpose(1, 2))
        return self.proj(torch.cat([h[:, :, -1], h.mean(dim=-1)], dim=-1))


class VelocityMLP(nn.Module):
    """AdaLN-inspired velocity network v_theta(y_t, t | c) -> R^6."""

    def __init__(self, y_dim: int = 6, cond_dim: int = 64, hidden_dim: int = 96, time_dim: int = 32):
        super().__init__()
        self.time_emb = SinusoidalTimeEmb(time_dim)
        self.cond_proj = nn.Linear(cond_dim + time_dim, hidden_dim)
        self.y_proj = nn.Linear(y_dim, hidden_dim)

        self.block1 = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.block2 = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.out = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, y_dim),
        )

    def forward(self, y_t: torch.Tensor, t: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        te = self.time_emb(t)
        h_cond = self.cond_proj(torch.cat([c, te], dim=-1))
        h = self.y_proj(y_t) + h_cond
        h = h + self.block1(h)
        h = h + self.block2(h + h_cond)
        return self.out(h)


class SectorFlowMatcher(nn.Module):
    """Conditional Optimal-Transport Flow Matching for joint 6-sector return distributions."""

    def __init__(
        self,
        n_sectors: int = 6,
        f_sec: int = 16,
        f_macro: int = 15,
        cond_dim: int = 64,
        hidden_dim: int = 96,
        noise_scale: float = 0.25,
    ):
        super().__init__()
        self.n_sectors = n_sectors
        self.noise_scale = noise_scale
        in_dim = n_sectors * f_sec + f_macro
        self.encoder = ContextEncoder(in_dim, cond_dim=cond_dim)
        self.velocity = VelocityMLP(y_dim=n_sectors, cond_dim=cond_dim, hidden_dim=hidden_dim)

    def compute_loss(self, x_sec: torch.Tensor, x_macro: torch.Tensor, y1: torch.Tensor) -> torch.Tensor:
        B = y1.shape[0]
        device = y1.device
        c = self.encoder(x_sec, x_macro)

        # Sample base noise y0 ~ N(0, noise_scale^2 * I) matching typical return scale
        y0 = torch.randn_like(y1) * self.noise_scale
        t = torch.rand(B, 1, device=device)

        y_t = (1.0 - t) * y0 + t * y1
        target_v = y1 - y0
        pred_v = self.velocity(y_t, t, c)
        return F.mse_loss(pred_v, target_v)

    @torch.no_grad()
    def sample_futures(
        self,
        x_sec: torch.Tensor,
        x_macro: torch.Tensor,
        n_samples: int = 32,
        n_steps: int = 12,
        seed: int = 42,
    ):
        """
        Integrate ODE from t=0 -> 1 for K=n_samples particles per input.
        Returns:
            mean_pred: (B, 6)
            std_pred:  (B, 6)  predictive uncertainty per sector
            samples:   (K, B, 6) full joint distribution samples
        """
        B = x_sec.shape[0]
        device = x_sec.device
        c = self.encoder(x_sec, x_macro)  # (B, cond_dim)
        c_rep = c.unsqueeze(0).expand(n_samples, B, -1).reshape(n_samples * B, -1)

        gen = torch.Generator(device=device)
        gen.manual_seed(seed)
        y = torch.randn(n_samples * B, self.n_sectors, device=device, generator=gen) * self.noise_scale

        dt = 1.0 / n_steps
        for step in range(n_steps):
            t0 = torch.full((n_samples * B, 1), step * dt, device=device)
            t_mid = torch.full((n_samples * B, 1), (step + 0.5) * dt, device=device)
            # Midpoint RK2 ODE integration
            v1 = self.velocity(y, t0, c_rep)
            y_mid = y + 0.5 * dt * v1
            v2 = self.velocity(y_mid, t_mid, c_rep)
            y = y + dt * v2

        samples = y.reshape(n_samples, B, self.n_sectors)
        return samples.mean(dim=0), samples.std(dim=0), samples
