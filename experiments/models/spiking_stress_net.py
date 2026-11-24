"""
Paradigm 1: Neuromorphic Spiking Neural Network (SNN)
-----------------------------------------------------
Delta-Encoded Multi-Timescale Coupled Leaky-Integrate-and-Fire (LIF) Network.

Why this is a genuinely different (and realistic!) approach for macro/sector forecasting:
1. Event-driven delta encoding: Markets & macro regimes respond to *changes* (surprises,
   inflections, rate-of-change) as much as levels. We feed both levels and temporal deltas
   as excitatory/inhibitory synaptic currents over the 1-year lookback window.
2. Heterogeneous learnable membrane decay (beta_i in (0.3, 0.98)): Some LIF neurons have
   short memory (fast market shocks), others integrate slowly over quarters (cumulative
   inflation / tightening / valuation stress) until crossing their firing threshold theta_i!
3. Cross-sector synaptic coupling: Spikes fired in one sector or macro population propagate
   through a learned synaptic matrix into other sectors, modeling contagion & rotation.
4. End-to-end training via fast-sigmoid surrogate gradients (no external SNN library needed).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FastSigmoidSpike(torch.autograd.Function):
    """Heaviside spike function with smooth fast-sigmoid surrogate gradient."""

    @staticmethod
    def forward(ctx, membrane_minus_thresh: torch.Tensor, slope: float = 10.0) -> torch.Tensor:
        ctx.save_for_backward(membrane_minus_thresh)
        ctx.slope = slope
        return (membrane_minus_thresh > 0.0).float()

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (x,) = ctx.saved_tensors
        slope = ctx.slope
        # Derivative of fast sigmoid: 1 / (1 + slope * |x|)^2
        surrogate = 1.0 / (1.0 + slope * x.abs()).pow(2)
        return grad_output * surrogate, None


def spike_fn(x: torch.Tensor, slope: float = 10.0) -> torch.Tensor:
    return FastSigmoidSpike.apply(x, slope)


class HeterogeneousLIFLayer(nn.Module):
    """Leaky-Integrate-and-Fire population with learnable per-neuron decay & threshold."""

    def __init__(self, in_dim: int, hidden_dim: int, slope: float = 8.0):
        super().__init__()
        self.fc_in = nn.Linear(in_dim, hidden_dim)
        self.fc_rec = nn.Linear(hidden_dim, hidden_dim, bias=False)
        nn.init.orthogonal_(self.fc_rec.weight, gain=0.3)

        # Initialize heterogeneous time constants spanning fast (~0.4) to slow (~0.95)
        init_beta = torch.linspace(-0.4, 2.8, hidden_dim)
        self.beta_logit = nn.Parameter(init_beta)
        self.threshold = nn.Parameter(torch.full((hidden_dim,), 0.8))
        self.slope = slope

    def forward(self, current_seq: torch.Tensor):
        """
        Args:
            current_seq: (B, T, in_dim)
        Returns:
            spikes: (B, T, hidden_dim)
            membranes: (B, T, hidden_dim)
        """
        B, T, _ = current_seq.shape
        H = self.fc_in.out_features
        device = current_seq.device

        beta = torch.sigmoid(self.beta_logit)  # (H,) in (0, 1)
        thresh = F.softplus(self.threshold) + 0.1  # positive threshold

        u = torch.zeros(B, H, device=device)
        s = torch.zeros(B, H, device=device)

        spikes_list = []
        mem_list = []

        for t in range(T):
            i_t = self.fc_in(current_seq[:, t, :]) + self.fc_rec(s)
            # Soft-reset LIF dynamics: u[t] = beta * u[t-1] + i[t] - s[t-1] * thresh
            u = beta * u + i_t - s * thresh
            s = spike_fn(u - thresh, self.slope)
            spikes_list.append(s)
            mem_list.append(u)

        return torch.stack(spikes_list, dim=1), torch.stack(mem_list, dim=1)


class SpikingMacroStressNet(nn.Module):
    """
    2-layer Coupled Multi-Timescale SNN with delta-event encoding and
    dual firing-rate + final-membrane-potential readout.
    """

    def __init__(
        self,
        n_sectors: int = 6,
        f_sec: int = 16,
        f_macro: int = 15,
        hidden_dim: int = 96,
        dropout: float = 0.25,
    ):
        super().__init__()
        raw_dim = n_sectors * f_sec + f_macro
        # Concatenate raw level + temporal delta (event-driven encoding)
        self.in_norm = nn.LayerNorm(raw_dim * 2)
        self.lif1 = HeterogeneousLIFLayer(raw_dim * 2, hidden_dim)
        self.drop = nn.Dropout(dropout)
        self.lif2 = HeterogeneousLIFLayer(hidden_dim, hidden_dim)

        # Readout combines mean spike rate (cumulative stress) + final membrane potential (latent pressure)
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, 48),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(48, n_sectors),
        )

    def forward(self, x_sec: torch.Tensor, x_macro: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_sec:   (B, T, 6, F_sec)
            x_macro: (B, T, F_macro)
        """
        B, T, S, Fs = x_sec.shape
        x = torch.cat([x_sec.reshape(B, T, S * Fs), x_macro], dim=-1)  # (B, T, D)

        # Event-based delta encoding: dx[t] = x[t] - x[t-1]
        dx = torch.zeros_like(x)
        dx[:, 1:, :] = x[:, 1:, :] - x[:, :-1, :]
        events = self.in_norm(torch.cat([x, dx], dim=-1))

        s1, _ = self.lif1(events)
        s2, u2 = self.lif2(self.drop(s1))

        # Firing rate over time + final accumulated membrane potential
        firing_rate = s2.mean(dim=1)
        final_mem = torch.tanh(u2[:, -1, :])
        feat = torch.cat([firing_rate, final_mem], dim=-1)
        return self.head(feat)

    @torch.no_grad()
    def get_spike_rate(self, x_sec: torch.Tensor, x_macro: torch.Tensor) -> float:
        B, T, S, Fs = x_sec.shape
        x = torch.cat([x_sec.reshape(B, T, S * Fs), x_macro], dim=-1)
        dx = torch.zeros_like(x)
        dx[:, 1:, :] = x[:, 1:, :] - x[:, :-1, :]
        events = self.in_norm(torch.cat([x, dx], dim=-1))
        s1, _ = self.lif1(events)
        s2, _ = self.lif2(s1)
        return float(0.5 * (s1.mean().item() + s2.mean().item()))
