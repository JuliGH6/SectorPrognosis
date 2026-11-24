"""
Paradigm 5: 2D Gramian Angular Field (GAF) + Recurrence Vision CNN
------------------------------------------------------------------
Turns each trading day's 1-year macro & sector history into an actual multi-channel
2D 'market photograph' (C, 16, 16) and processes it with a lightweight 2D Vision CNN!

How the 2D Market Image is constructed (all differentiable, zero hand-tuned thresholds):
  Given 16 time steps over the trailing year:
  1. Project each of the 6 sectors + Macro into compact angular phase series via tanh -> arccos.
  2. Build Gramian Angular Summation Fields (GASF): G_ij = cos(phi_i + phi_j) in [-1, 1]^(16x16),
     which preserves exact temporal order and turns trends/cycles/inflections into crisp
     2D visual textures (diagonal stripes, checkerboards, regime-break corners).
  3. Build a 2D Temporal Self-Similarity / Recurrence Matrix across the full 97-dim macro+sector
     state: R_ij = exp(-||x_i - x_j||^2 / d), capturing repeating cycles and structural breaks.
  4. Feed the resulting 16x16 multi-channel image into a 2D Residual Vision CNN!
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MarketImageEncoder(nn.Module):
    """Differentiably renders (x_sec, x_macro) into a multi-channel 16x16 2D image."""

    def __init__(self, n_sectors: int = 6, f_sec: int = 16, f_macro: int = 15, n_channels: int = 12):
        super().__init__()
        in_dim = n_sectors * f_sec + f_macro
        # Learn 10 orthogonal-ish 1D phase projections (6 sector-focused + 4 macro/cross-sector)
        self.phase_proj = nn.Linear(in_dim, n_channels - 2)
        self.n_channels = n_channels

    def forward(self, x_sec: torch.Tensor, x_macro: torch.Tensor) -> torch.Tensor:
        # x_sec: (B, T, 6, F_sec), x_macro: (B, T, F_macro)
        B, T, S, Fs = x_sec.shape
        x = torch.cat([x_sec.reshape(B, T, S * Fs), x_macro], dim=-1)  # (B, T, D)

        # 1) Learnable Gramian Angular Summation Fields (GASF) & Difference Fields (GADF)
        # Map projected signals to (-0.98, 0.98) -> cos(phi) = u, sin(phi) = sqrt(1 - u^2)
        u = 0.98 * torch.tanh(self.phase_proj(x))  # (B, T, C_gaf)
        v = torch.sqrt(torch.clamp(1.0 - u * u, min=1e-4))

        # GASF_ij = u_i * u_j - v_i * v_j  (exact trigonometric identity for cos(phi_i + phi_j)!)
        u_t = u.permute(0, 2, 1)  # (B, C_gaf, T)
        v_t = v.permute(0, 2, 1)
        gasf = torch.einsum("bci,bcj->bcij", u_t, u_t) - torch.einsum("bci,bcj->bcij", v_t, v_t)

        # 2) Full-state 2D Recurrence Plot (R_ij = exp(-dist^2 / D))
        diff = x.unsqueeze(2) - x.unsqueeze(1)  # (B, T, T, D)
        rec_plot = torch.exp(-diff.pow(2).mean(dim=-1) * 0.5).unsqueeze(1)  # (B, 1, T, T)

        # 3) Cross-sector dispersion recurrence (how spread out the 6 sectors are at time i vs j)
        sec_disp = x_sec.std(dim=2).mean(dim=-1)  # (B, T)
        disp_mat = (sec_disp.unsqueeze(2) - sec_disp.unsqueeze(1)).unsqueeze(1)  # (B, 1, T, T)

        return torch.cat([gasf, rec_plot, disp_mat], dim=1)  # (B, C, T, T)


class MarketVisionCNN(nn.Module):
    """2D Vision CNN operating on differentiable 16x16 Market Photographs (GASF + Recurrence)."""

    def __init__(
        self,
        n_sectors: int = 6,
        f_sec: int = 16,
        f_macro: int = 15,
        n_channels: int = 12,
        dropout: float = 0.25,
    ):
        super().__init__()
        self.imager = MarketImageEncoder(n_sectors, f_sec, f_macro, n_channels=n_channels)
        self.conv = nn.Sequential(
            nn.Conv2d(n_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.Conv2d(32, 48, kernel_size=3, stride=2, padding=1),  # 16x16 -> 8x8
            nn.BatchNorm2d(48),
            nn.GELU(),
            nn.Dropout2d(dropout * 0.5),
            nn.Conv2d(48, 64, kernel_size=3, stride=2, padding=1),  # 8x8 -> 4x4
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Linear(32, n_sectors),
        )

    def forward(self, x_sec: torch.Tensor, x_macro: torch.Tensor) -> torch.Tensor:
        img = self.imager(x_sec, x_macro)
        feat = self.conv(img)
        return self.head(feat)
