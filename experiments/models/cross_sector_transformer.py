"""
Paradigm 4: Shared-Weight Bidirectional Cross-Sector Axial Attention Transformer
--------------------------------------------------------------------------------
Fixes the biggest inductive-bias flaw of flattening all sectors into one PCA vector:
The 6 sectors share the *exact same* fundamental semantics (P/E, P/B, ROE, Margins,
Debt, Momentum), and they interact as a coupled economic rotation graph!

Architecture:
1. Weight-Shared Sector Temporal Encoder: a single 1D-CNN + attention-pooling encoder
   embeds EVERY sector's (T, F_sec) trajectory into a token h_s in R^D. This gives 6x
   statistical weight-sharing across sectors!
2. Macro Context Token: a separate temporal encoder embeds the (T, F_macro) macro
   trajectory into a 7th token h_macro in R^D.
3. Learnable Sector Identity Embeddings: added to each token so attention knows
   defensive vs cyclical identities (Utilities/Staples vs Industrials/Materials/Discretionary).
4. Bidirectional Multi-Head Self-Attention across all 7 tokens: every sector explicitly
   attends to every other sector's relative valuation/momentum AND the macro regime!
5. Dual Rotation-Aware Loss: combines absolute Huber return loss with a cross-sectional
   demeaned (spread) loss so the model learns *both* market direction and relative
   sector rotation!
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalTokenEncoder(nn.Module):
    """Multi-scale 1D-CNN encoder turning a (B*, T, F) series into a single token in R^D."""

    def __init__(self, in_dim: int, d_model: int = 48, dropout: float = 0.2):
        super().__init__()
        self.conv_short = nn.Conv1d(in_dim, d_model // 2, kernel_size=3, padding=1)
        self.conv_long = nn.Conv1d(in_dim, d_model // 2, kernel_size=5, padding=2)
        self.norm = nn.LayerNorm(d_model)
        self.attn_pool = nn.Linear(d_model, 1)
        self.out = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B*, T, F)
        xt = x.transpose(1, 2)
        h = torch.cat([F.gelu(self.conv_short(xt)), F.gelu(self.conv_long(xt))], dim=1)
        h = self.norm(h.transpose(1, 2))  # (B*, T, D)

        # Temporal attention pooling + latest step
        w = torch.softmax(self.attn_pool(h), dim=1)  # (B*, T, 1)
        h_pool = (h * w).sum(dim=1)
        h_last = h[:, -1, :]
        return self.out(torch.cat([h_pool, h_last], dim=-1))


class BidirectionalCrossSectorBlock(nn.Module):
    """Pre-LN Bidirectional Multi-Head Self-Attention + SwiGLU-style FFN over sector tokens."""

    def __init__(self, d_model: int = 48, n_heads: int = 4, dropout: float = 0.2):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, tokens: torch.Tensor):
        h = self.norm1(tokens)
        attn_out, attn_weights = self.attn(h, h, h, need_weights=True)
        tokens = tokens + attn_out
        tokens = tokens + self.ffn(self.norm2(tokens))
        return tokens, attn_weights


class SectorRotationTransformer(nn.Module):
    """Shared-weight Sector+Macro Bidirectional Transformer with cross-sector attention."""

    def __init__(
        self,
        n_sectors: int = 6,
        f_sec: int = 16,
        f_macro: int = 15,
        d_model: int = 48,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.25,
    ):
        super().__init__()
        self.n_sectors = n_sectors
        self.sector_encoder = TemporalTokenEncoder(f_sec, d_model=d_model, dropout=dropout)
        self.macro_encoder = TemporalTokenEncoder(f_macro, d_model=d_model, dropout=dropout)

        # 6 sector tokens + 1 macro token
        self.token_id_emb = nn.Parameter(torch.randn(1, n_sectors + 1, d_model) * 0.05)
        self.blocks = nn.ModuleList(
            [BidirectionalCrossSectorBlock(d_model, n_heads, dropout) for _ in range(n_layers)]
        )
        self.final_norm = nn.LayerNorm(d_model)

        # Shared readout head applied to each of the 6 sector tokens (predicts scalar return per sector)
        self.sector_head = nn.Linear(d_model, 1)

    def forward(self, x_sec: torch.Tensor, x_macro: torch.Tensor, return_attn: bool = False):
        # x_sec: (B, T, 6, F_sec), x_macro: (B, T, F_macro)
        B, T, S, Fs = x_sec.shape

        # Encode all 6 sectors using the SAME shared temporal encoder
        sec_in = x_sec.permute(0, 2, 1, 3).reshape(B * S, T, Fs)
        sec_tokens = self.sector_encoder(sec_in).reshape(B, S, -1)  # (B, 6, D)

        # Encode macro trajectory as the 7th token
        mac_token = self.macro_encoder(x_macro).unsqueeze(1)  # (B, 1, D)

        tokens = torch.cat([sec_tokens, mac_token], dim=1) + self.token_id_emb  # (B, 7, D)

        last_attn = None
        for blk in self.blocks:
            tokens, last_attn = blk(tokens)

        tokens = self.final_norm(tokens)
        # Apply shared scalar head to the 6 sector tokens
        preds = self.sector_head(tokens[:, :S, :]).squeeze(-1)  # (B, 6)

        if return_attn:
            return preds, last_attn
        return preds

    @staticmethod
    def rotation_loss(pred: torch.Tensor, target: torch.Tensor, spread_weight: float = 0.6) -> torch.Tensor:
        """Combines absolute return Huber loss + cross-sectional spread (sector rotation) loss."""
        abs_loss = F.smooth_l1_loss(pred, target, beta=0.1)
        pred_rel = pred - pred.mean(dim=-1, keepdim=True)
        targ_rel = target - target.mean(dim=-1, keepdim=True)
        spread_loss = F.smooth_l1_loss(pred_rel, targ_rel, beta=0.05)
        return abs_loss + spread_weight * spread_loss
