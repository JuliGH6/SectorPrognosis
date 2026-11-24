"""
Paradigm 7: Conditional Energy-Based Model (EBM) with Langevin Inference
------------------------------------------------------------------------
Instead of directly regressing x -> y_hat, this model learns a scalar Energy
Landscape E_theta(x, y) in R that scores how economically compatible ANY candidate
6-sector return vector y in R^6 is with the current macro + sector state x:
  - Low energy  = economically plausible future
  - High energy = economically contradictory future

Architecture:
  1. Weight-shared Cross-Sector + Macro context encoder -> sector tokens h_1..h_6 & macro c.
  2. Decomposed Smooth Energy Function (SiLU everywhere so grad_y E is smooth):
       E_theta(x, y) = E_unary(h, y) + E_ising(c, y) + E_joint(c, y) + 0.5 * alpha * ||y||^2
     where E_ising(c, y) = -y^T J(c) y models context-dependent pairwise cross-sector
     coupling (like a continuous Macro-Ising / Hopfield network).

Training (Hybrid Contrastive InfoNCE + Short-Run Langevin Negative Sampling):
  - Positives: true future 6-sector return y+
  - Negatives:
      (a) In-batch cross-date historical futures (forces EBM to distinguish today's
          regime from other historical regimes),
      (b) Local Gaussian-perturbed futures around y+ (sharpens the local valley),
      (c) Short-run Langevin-sampled "hallucinated" low-energy valleys (Contrastive Divergence).
  - Loss: InfoNCE contrastive loss over {y+, negatives} + IGEBM energy-magnitude regularization.

Inference:
  - Start K=24 random candidate futures y^(0) ~ N(0, 0.25^2 I_6).
  - Run gradient descent / Langevin dynamics on y to minimize E_theta(x, y).
  - Return the Boltzmann-weighted / lowest-energy equilibrium forecast, chain spread
    (epistemic uncertainty), and minimum energy E*(x) (regime anomaly score).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .cross_sector_transformer import TemporalTokenEncoder, BidirectionalCrossSectorBlock


class SectorEnergyModel(nn.Module):
    """Conditional Macro-Ising + Neural Energy-Based Model with Langevin inference."""

    def __init__(
        self,
        n_sectors: int = 6,
        f_sec: int = 16,
        f_macro: int = 15,
        d_model: int = 40,
        n_heads: int = 4,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.n_sectors = n_sectors
        self.sector_encoder = TemporalTokenEncoder(f_sec, d_model=d_model, dropout=dropout)
        self.macro_encoder = TemporalTokenEncoder(f_macro, d_model=d_model, dropout=dropout)
        self.token_id_emb = nn.Parameter(torch.randn(1, n_sectors + 1, d_model) * 0.05)
        self.cross_block = BidirectionalCrossSectorBlock(d_model=d_model, n_heads=n_heads, dropout=dropout)
        self.norm = nn.LayerNorm(d_model)

        # 1) Unary per-sector energy: scores each sector's scalar return y_s given token h_s
        self.unary_net = nn.Sequential(
            nn.Linear(d_model + 2, 32),  # [h_s, y_s, y_s^2]
            nn.SiLU(),
            nn.Linear(32, 1),
        )

        # 2) Context-dependent pairwise Ising coupling matrix J(c) in R^(6x6)
        self.ising_proj = nn.Linear(d_model, n_sectors * n_sectors)

        # 3) Joint nonlinear energy residual over (macro_summary, y, y_demeaned)
        self.joint_net = nn.Sequential(
            nn.Linear(d_model * 2 + n_sectors * 2, 64),
            nn.SiLU(),
            nn.Linear(64, 32),
            nn.SiLU(),
            nn.Linear(32, 1),
        )

        # Learnable quadratic confining potential so energy -> +inf as ||y|| -> inf
        self.log_quad_reg = nn.Parameter(torch.tensor(0.0))

    def encode_context(self, x_sec: torch.Tensor, x_macro: torch.Tensor):
        B, T, S, Fs = x_sec.shape
        sec_in = x_sec.permute(0, 2, 1, 3).reshape(B * S, T, Fs)
        sec_tok = self.sector_encoder(sec_in).reshape(B, S, -1)
        mac_tok = self.macro_encoder(x_macro).unsqueeze(1)
        tokens = torch.cat([sec_tok, mac_tok], dim=1) + self.token_id_emb
        tokens, _ = self.cross_block(tokens)
        tokens = self.norm(tokens)
        h_sec = tokens[:, :S, :]           # (B, 6, D)
        h_mac = tokens[:, S, :]            # (B, D)
        h_mean = h_sec.mean(dim=1)         # (B, D)
        J = self.ising_proj(h_mac).reshape(B, S, S)
        J = 0.5 * (J + J.transpose(1, 2)) * 0.25  # symmetric coupling
        return h_sec, h_mac, h_mean, J

    def energy_from_context(
        self,
        h_sec: torch.Tensor,
        h_mac: torch.Tensor,
        h_mean: torch.Tensor,
        J: torch.Tensor,
        y: torch.Tensor,
    ) -> torch.Tensor:
        """
        Evaluate scalar energy E(x, y) for one or multiple candidate y's.
        Args:
            h_sec:  (B, 6, D)
            h_mac:  (B, D)
            h_mean: (B, D)
            J:      (B, 6, 6)
            y:      (B, M, 6)  where M = number of candidate return vectors per sample
        Returns:
            energy: (B, M)
        """
        B, M, S = y.shape
        D = h_sec.shape[-1]

        # 1) Unary energy per sector
        h_exp = h_sec.unsqueeze(1).expand(B, M, S, D)
        y_feat = torch.stack([y, y * y], dim=-1)  # (B, M, S, 2)
        e_unary = self.unary_net(torch.cat([h_exp, y_feat], dim=-1)).squeeze(-1).sum(dim=-1)  # (B, M)

        # 2) Pairwise Macro-Ising interaction energy: -y^T J y
        e_ising = -torch.einsum("bms,bst,bmt->bm", y, J, y)

        # 3) Joint nonlinear energy (includes cross-sectional demeaned returns for rotation!)
        y_rel = y - y.mean(dim=-1, keepdim=True)
        ctx = torch.cat([h_mac, h_mean], dim=-1).unsqueeze(1).expand(B, M, D * 2)
        e_joint = self.joint_net(torch.cat([ctx, y, y_rel], dim=-1)).squeeze(-1)

        # 4) Smooth confining quadratic prior
        quad = 0.5 * F.softplus(self.log_quad_reg) * (y.pow(2).sum(dim=-1))

        return e_unary + e_ising + e_joint + quad

    def langevin_sample(
        self,
        h_sec: torch.Tensor,
        h_mac: torch.Tensor,
        h_mean: torch.Tensor,
        J: torch.Tensor,
        n_chains: int = 16,
        n_steps: int = 20,
        step_size: float = 0.04,
        noise_scale: float = 0.015,
        init_std: float = 0.25,
        seed: int | None = None,
    ):
        """Run Langevin dynamics / gradient descent on y to find low-energy equilibria."""
        B = h_sec.shape[0]
        device = h_sec.device

        # Detach context so Langevin only differentiates w.r.t. y
        h_sec_d = h_sec.detach()
        h_mac_d = h_mac.detach()
        h_mean_d = h_mean.detach()
        J_d = J.detach()

        if seed is not None:
            gen = torch.Generator(device=device)
            gen.manual_seed(seed)
            y = torch.randn(B, n_chains, self.n_sectors, device=device, generator=gen) * init_std
        else:
            y = torch.randn(B, n_chains, self.n_sectors, device=device) * init_std

        for step in range(n_steps):
            y = y.detach().requires_grad_(True)
            e = self.energy_from_context(h_sec_d, h_mac_d, h_mean_d, J_d, y)
            grad_y = torch.autograd.grad(e.sum(), y, create_graph=False)[0]
            grad_y = torch.clamp(grad_y, -5.0, 5.0)

            # Anneal noise to 0 in the final steps so chains settle cleanly into energy minima
            cur_noise = noise_scale * max(0.0, 1.0 - step / max(n_steps - 4, 1))
            if cur_noise > 0:
                y = y - step_size * grad_y + cur_noise * torch.randn_like(y)
            else:
                y = y - step_size * grad_y
            y = torch.clamp(y, -1.5, 2.5)

        with torch.no_grad():
            final_e = self.energy_from_context(h_sec_d, h_mac_d, h_mean_d, J_d, y)
        return y.detach(), final_e

    def compute_loss(
        self,
        x_sec: torch.Tensor,
        x_macro: torch.Tensor,
        y_pos: torch.Tensor,
        n_local_neg: int = 16,
        n_langevin_neg: int = 8,
    ) -> torch.Tensor:
        """
        Hybrid InfoNCE + Short-Run Langevin Contrastive Divergence Loss.
        Pushes E(x, y_pos) DOWN and E(x, y_neg) UP across:
          1. Local Gaussian perturbations around y_pos (multi-scale: 0.08 and 0.25)
          2. Cross-date in-batch negatives (real futures from other market regimes!)
          3. Hard Langevin-mined false energy valleys
        """
        B, S = y_pos.shape
        device = y_pos.device
        h_sec, h_mac, h_mean, J = self.encode_context(x_sec, x_macro)

        # (a) Local multi-scale negatives around y_pos
        half = n_local_neg // 2
        neg_fine = y_pos.unsqueeze(1) + 0.08 * torch.randn(B, half, S, device=device)
        neg_coarse = y_pos.unsqueeze(1) + 0.25 * torch.randn(B, n_local_neg - half, S, device=device)

        # (b) Cross-date historical negatives (randomly drawn from other dates in batch)
        perm_idx = torch.randint(0, B, (B, 12), device=device)
        neg_hist = y_pos[perm_idx]  # (B, 12, S)

        # (c) Hard Langevin negatives (short-run MCMC finding spurious low-energy valleys)
        neg_mcmc, _ = self.langevin_sample(
            h_sec, h_mac, h_mean, J,
            n_chains=n_langevin_neg, n_steps=8, step_size=0.05, noise_scale=0.03,
        )

        # Concatenate: index 0 is ALWAYS y_pos
        y_all = torch.cat([y_pos.unsqueeze(1), neg_fine, neg_coarse, neg_hist, neg_mcmc], dim=1)
        energies = self.energy_from_context(h_sec, h_mac, h_mean, J, y_all)  # (B, 1 + M_neg)

        # InfoNCE: minimize -log softmax(-E)_0  <=>  cross-entropy with logits = -E, target = 0
        logits = -energies
        labels = torch.zeros(B, dtype=torch.long, device=device)
        nce_loss = F.cross_entropy(logits, labels)

        # IGEBM energy-magnitude regularization (prevents energy drift to +-inf)
        reg_loss = 0.01 * (energies[:, 0].pow(2).mean() + energies[:, 1:].pow(2).mean())

        return nce_loss + reg_loss

    def predict_equilibrium(
        self,
        x_sec: torch.Tensor,
        x_macro: torch.Tensor,
        n_chains: int = 24,
        n_steps: int = 30,
        temperature: float = 0.5,
        seed: int = 42,
    ):
        """
        Inference via Langevin optimization on the learned energy surface.
        Returns:
            pred_mean:  (B, 6)  Boltzmann-weighted equilibrium forecast
            pred_std:   (B, 6)  spread across converged chains (uncertainty / multimodality)
            min_energy: (B,)    minimum achieved energy (regime familiarity / anomaly score)
        """
        with torch.no_grad():
            h_sec, h_mac, h_mean, J = self.encode_context(x_sec, x_macro)
        y_chains, e_chains = self.langevin_sample(
            h_sec, h_mac, h_mean, J,
            n_chains=n_chains, n_steps=n_steps, step_size=0.04, noise_scale=0.015, seed=seed,
        )
        # Boltzmann weights w_k = softmax(-E_k / T) emphasize the deepest valleys
        w = torch.softmax(-e_chains / temperature, dim=1)  # (B, K)
        pred_mean = (w.unsqueeze(-1) * y_chains).sum(dim=1)  # (B, 6)
        pred_std = y_chains.std(dim=1)  # (B, 6)
        min_energy = e_chains.min(dim=1).values  # (B,)
        return pred_mean, pred_std, min_energy
