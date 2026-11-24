"""
Simple Prototype: Physics-Informed Coupled Damped Harmonic Oscillator Network
-----------------------------------------------------------------------------
Models the 6 S&P 500 sectors as 6 coupled masses on damped springs driven by an
external macroeconomic force:

    d^2 q / dt^2 = -K * (q - q_eq(x)) - Gamma * v + C_couple * q + F_macro(x)

Where:
  - q in R^6: sector valuation/performance displacement from equilibrium
  - v = dq/dt in R^6: sector momentum velocity (initialized from trailing 1m/3m returns)
  - K > 0: learnable per-sector mean-reversion spring stiffness
  - Gamma > 0: learnable per-sector momentum damping
  - C_couple (6x6 symmetric, zero-diag): learnable cross-sector spring coupling
  - q_eq(x), F_macro(x): simple linear maps from current fundamentals & macro state

Integrated forward 1 year (4 quarters) via symplectic Velocity Verlet!
Ultra-compact (~1,500 parameters total) and directly physically interpretable.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class CoupledOscillatorNet(nn.Module):
    """Minimal 6-Sector Coupled Damped Harmonic Oscillator with Velocity Verlet integration."""

    def __init__(self, n_sectors: int = 6, f_sec: int = 16, f_macro: int = 15, n_steps: int = 8):
        super().__init__()
        self.n_sectors = n_sectors
        self.n_steps = n_steps

        # Physical constants (constrained positive via softplus)
        self.log_stiffness = nn.Parameter(torch.full((n_sectors,), 0.2))  # K_s > 0 (mean reversion)
        self.log_damping = nn.Parameter(torch.full((n_sectors,), 0.5))    # Gamma_s > 0 (friction)

        # Cross-sector spring coupling matrix (made symmetric with zero diagonal)
        self.coupling_raw = nn.Parameter(torch.zeros(n_sectors, n_sectors))

        # Simple linear maps for initial displacement q0, equilibrium q_eq, and macro forcing F
        self.q0_proj = nn.Linear(f_sec, 1)         # shared across sectors!
        self.v0_proj = nn.Linear(f_sec, 1)         # shared across sectors!
        self.q_eq_proj = nn.Linear(f_sec, 1)       # fundamental fair-value anchor
        self.macro_force = nn.Linear(f_macro * 2, n_sectors)  # [macro_level, macro_1y_change] -> 6 forces

        # Small linear readout from final (displacement_change, final_velocity) -> 1Y return
        self.scale_q = nn.Parameter(torch.ones(n_sectors) * 0.2)
        self.scale_v = nn.Parameter(torch.ones(n_sectors) * 0.1)
        self.bias = nn.Parameter(torch.full((n_sectors,), 0.08))  # ~8% long-run equity drift

    def _acceleration(self, q: torch.Tensor, v: torch.Tensor, q_eq: torch.Tensor, F_ext: torch.Tensor):
        K = F.softplus(self.log_stiffness)     # (6,)
        G = F.softplus(self.log_damping)       # (6,)
        C = 0.5 * (self.coupling_raw + self.coupling_raw.T)
        C = C - torch.diag(torch.diag(C))      # zero diagonal
        C = 0.2 * torch.tanh(C)                # bounded coupling for numerical stability
        return -K * (q - q_eq) - G * v + (q @ C) + F_ext

    def forward(self, x_sec: torch.Tensor, x_macro: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_sec:   (B, T, 6, F_sec)
            x_macro: (B, T, F_macro)
        """
        sec_now = x_sec[:, -1, :, :]                          # (B, 6, F_sec)
        sec_delta = x_sec[:, -1, :, :] - x_sec[:, 0, :, :]    # (B, 6, F_sec) 1Y fundamental velocity

        mac_now = x_macro[:, -1, :]
        mac_delta = x_macro[:, -1, :] - x_macro[:, 0, :]
        mac_feat = torch.cat([mac_now, mac_delta], dim=-1)    # (B, 2*F_macro)

        # Initial mechanical state of the 6 sectors
        q0 = self.q0_proj(sec_now).squeeze(-1)                # (B, 6)
        v = self.v0_proj(sec_delta).squeeze(-1)               # (B, 6)
        q_eq = self.q_eq_proj(sec_now).squeeze(-1)            # (B, 6)
        F_ext = self.macro_force(mac_feat)                    # (B, 6)

        q = q0
        dt = 1.0 / self.n_steps
        a = self._acceleration(q, v, q_eq, F_ext)

        # Symplectic Velocity Verlet integration over 1 year
        for _ in range(self.n_steps):
            q = q + v * dt + 0.5 * a * (dt * dt)
            a_new = self._acceleration(q, v, q_eq, F_ext)
            v = v + 0.5 * (a + a_new) * dt
            a = a_new

        # 1-year return = displacement travel (q - q0) + momentum impulse + drift
        return self.scale_q * (q - q0) + self.scale_v * v + self.bias

    @torch.no_grad()
    def get_physics_summary(self, sector_names):
        K = F.softplus(self.log_stiffness).cpu().numpy()
        G = F.softplus(self.log_damping).cpu().numpy()
        C = 0.5 * (self.coupling_raw + self.coupling_raw.T)
        C = (0.2 * torch.tanh(C - torch.diag(torch.diag(C)))).cpu().numpy()
        return {
            "stiffness_K": {s: round(float(k), 3) for s, k in zip(sector_names, K)},
            "damping_Gamma": {s: round(float(g), 3) for s, g in zip(sector_names, G)},
            "max_coupling_pair": (
                sector_names[int(divmod(int(abs(C).argmax()), len(sector_names))[0])],
                sector_names[int(divmod(int(abs(C).argmax()), len(sector_names))[1])],
                round(float(C.flat[int(abs(C).argmax())]), 3),
            ),
        }
