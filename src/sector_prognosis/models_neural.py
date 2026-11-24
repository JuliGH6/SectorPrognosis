"""Iteration 3: Modern Deep Learning for Cross-Sector Forecasting (SectorFormer).

Architecture highlights:
- Gated Residual Networks (GRN, Temporal Fusion Transformer style) with LayerNorm & SiLU
- Macro-conditioned FiLM / gating modulation of sector tokens
- Multi-Head Self-Attention across the 6 S&P 500 sectors
- Explicit Target Decomposition head: bounded market residual + zero-sum cross-sectional alpha
- Hybrid Huber + Cross-Sectional Cosine Correlation loss
- Multi-seed bagging + strictly purged inner-validation early stopping
"""

from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import RobustScaler

from . import HORIZON_DAYS, SECTORS
from .data import compute_forward_targets, load_raw_merged_df
from .features import build_all_feature_views
from .metrics import evaluate_predictions, summarize_fold_metrics
from .splits import get_dev_walk_forward_folds, split_inner_train_val


class GatedResidualNetwork(nn.Module):
    """TFT-style Gated Residual Network with LayerNorm, SiLU, and GLU gating."""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, context_dim: int | None = None, dropout: float = 0.25):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.ctx_proj = nn.Linear(context_dim, hidden_dim, bias=False) if context_dim is not None else None
        self.fc2 = nn.Linear(hidden_dim, out_dim * 2)  # for GLU gating
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(out_dim)
        self.skip = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()

    def forward(self, x: torch.Tensor, context: torch.Tensor | None = None) -> torch.Tensor:
        h = self.fc1(x)
        if self.ctx_proj is not None and context is not None:
            h = h + self.ctx_proj(context)
        h = F.silu(h)
        h = self.dropout(h)
        a, b = self.fc2(h).chunk(2, dim=-1)
        gated = a * torch.sigmoid(b)
        return self.norm(self.skip(x) + gated)


class SectorFormer(nn.Module):
    """Cross-Sector Transformer with Macro-Conditioned GRNs & Decomposed Output Heads."""

    def __init__(
        self,
        n_sec_feat: int,
        n_macro_feat: int,
        n_sectors: int = 6,
        d_model: int = 64,
        n_heads: int = 4,
        dropout: float = 0.30,
        max_mkt_delta: float = 0.20,
    ):
        super().__init__()
        self.n_sectors = n_sectors
        self.max_mkt_delta = max_mkt_delta
        self.input_dropout = nn.Dropout(0.15)

        # Learnable sector identity embedding
        self.sector_emb = nn.Embedding(n_sectors, d_model)

        # Macro context encoder
        self.macro_grn = GatedResidualNetwork(n_macro_feat, d_model, d_model, dropout=dropout)

        # Per-sector token encoder conditioned on macro context
        self.sec_grn = GatedResidualNetwork(
            n_sec_feat, d_model, d_model, context_dim=d_model, dropout=dropout
        )

        # Cross-sector Multi-Head Self-Attention block (Pre-LN Transformer)
        self.attn_norm = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, num_heads=n_heads, dropout=dropout, batch_first=True)
        self.ffn_grn = GatedResidualNetwork(d_model, d_model, d_model, context_dim=d_model, dropout=dropout)

        # Cross-sectional alpha head (predicts per-sector relative return)
        self.alpha_head = nn.Linear(d_model, 1)
        # Bounded market-level residual head (predicts delta around training prior)
        self.mkt_head = nn.Sequential(
            nn.Linear(d_model * 2, d_model // 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )

        # Small output init for stability
        nn.init.normal_(self.alpha_head.weight, std=0.02)
        nn.init.zeros_(self.alpha_head.bias)

    def forward(
        self, x_sec: torch.Tensor, x_macro: torch.Tensor, mkt_prior: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x_sec: (B, 6, F_sec)
            x_macro: (B, F_macro)
            mkt_prior: scalar or (B, 1) historical training market mean
        Returns:
            y_pred: (B, 6) total predicted 1Y returns
            alpha_pred: (B, 6) zero-sum cross-sectional alpha
            mkt_pred: (B, 1) market-level return prediction
        """
        if self.training:
            x_sec = self.input_dropout(x_sec) + 0.03 * torch.randn_like(x_sec)
            x_macro = self.input_dropout(x_macro) + 0.03 * torch.randn_like(x_macro)

        B, S, _ = x_sec.shape
        m_ctx = self.macro_grn(x_macro)  # (B, d_model)
        m_ctx_rep = m_ctx.unsqueeze(1).expand(-1, S, -1)  # (B, S, d_model)

        sec_ids = torch.arange(S, device=x_sec.device).unsqueeze(0).expand(B, -1)
        h = self.sec_grn(x_sec, context=m_ctx_rep) + self.sector_emb(sec_ids)  # (B, S, d_model)

        # Cross-sector attention
        h_norm = self.attn_norm(h)
        attn_out, _ = self.attn(h_norm, h_norm, h_norm, need_weights=False)
        h = h + attn_out
        h = self.ffn_grn(h, context=m_ctx_rep)

        # Zero-sum cross-sectional alpha
        raw_alpha = self.alpha_head(h).squeeze(-1)  # (B, S)
        alpha_pred = raw_alpha - raw_alpha.mean(dim=-1, keepdim=True)

        # Bounded market-level residual around prior
        pooled = torch.cat([m_ctx, h.mean(dim=1)], dim=-1)  # (B, 2*d_model)
        mkt_delta = self.max_mkt_delta * torch.tanh(self.mkt_head(pooled))  # (B, 1)
        mkt_pred = mkt_prior + mkt_delta

        y_pred = mkt_pred + alpha_pred
        return y_pred, alpha_pred, mkt_pred


def hybrid_sector_loss(
    y_pred: torch.Tensor,
    alpha_pred: torch.Tensor,
    y_true: torch.Tensor,
    alpha_weight: float = 1.5,
    cosine_weight: float = 0.08,
) -> torch.Tensor:
    """Huber total loss + Huber cross-sectional alpha loss + Cross-sectional cosine correlation."""
    y_alpha_true = y_true - y_true.mean(dim=-1, keepdim=True)
    loss_total = F.huber_loss(y_pred, y_true, delta=0.15)
    loss_alpha = F.huber_loss(alpha_pred, y_alpha_true, delta=0.10)
    cos_sim = F.cosine_similarity(alpha_pred, y_alpha_true, dim=-1, eps=1e-6).mean()
    return loss_total + alpha_weight * loss_alpha - cosine_weight * cos_sim


class SectorFormerEnsemble:
    """Multi-seed bagged SectorFormer with strictly causal scaling & purged early stopping."""

    def __init__(
        self,
        seeds: Tuple[int, ...] = (42, 102, 202),
        d_model: int = 64,
        n_heads: int = 4,
        dropout: float = 0.30,
        lr: float = 1e-3,
        weight_decay: float = 1e-2,
        max_epochs: int = 80,
        batch_size: int = 64,
        train_stride: int = 3,
        max_mkt_delta: float = 0.15,
    ):
        self.seeds = seeds
        self.d_model = d_model
        self.n_heads = n_heads
        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.train_stride = train_stride
        self.max_mkt_delta = max_mkt_delta

    @staticmethod
    def _extract_3d(macro_df, sec_dfs, indices: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        x_m = macro_df.iloc[indices].to_numpy(dtype=np.float32)
        x_s = np.stack(
            [sec_dfs[s].iloc[indices].to_numpy(dtype=np.float32) for s in SECTORS], axis=1
        )  # (N, 6, F_sec)
        return x_s, x_m

    def fit_predict(
        self,
        macro_df,
        sec_dfs,
        y_all: np.ndarray,
        dates,
        realized_dates,
        train_idx: np.ndarray,
        eval_idx: np.ndarray,
    ) -> np.ndarray:
        itr, ival = split_inner_train_val(train_idx, dates, realized_dates, val_fraction=0.15)
        itr_sub = itr[:: self.train_stride] if self.train_stride > 1 else itr

        xs_tr, xm_tr = self._extract_3d(macro_df, sec_dfs, itr_sub)
        xs_val, xm_val = self._extract_3d(macro_df, sec_dfs, ival)
        xs_ev, xm_ev = self._extract_3d(macro_df, sec_dfs, eval_idx)

        # Strictly causal RobustScaler fit ONLY on itr_sub
        m_scaler = RobustScaler()
        xm_tr = np.clip(m_scaler.fit_transform(xm_tr), -4.0, 4.0).astype(np.float32)
        xm_val = np.clip(m_scaler.transform(xm_val), -4.0, 4.0).astype(np.float32)
        xm_ev = np.clip(m_scaler.transform(xm_ev), -4.0, 4.0).astype(np.float32)

        s_scaler = RobustScaler()
        N_tr, S, F_s = xs_tr.shape
        s_scaler.fit(xs_tr.reshape(-1, F_s))
        xs_tr = np.clip(s_scaler.transform(xs_tr.reshape(-1, F_s)), -4.0, 4.0).reshape(N_tr, S, F_s).astype(np.float32)
        xs_val = np.clip(s_scaler.transform(xs_val.reshape(-1, F_s)), -4.0, 4.0).reshape(len(ival), S, F_s).astype(np.float32)
        xs_ev = np.clip(s_scaler.transform(xs_ev.reshape(-1, F_s)), -4.0, 4.0).reshape(len(eval_idx), S, F_s).astype(np.float32)

        y_tr = y_all[itr_sub]
        y_val = y_all[ival]
        mkt_prior_val = float(y_all[train_idx].mean())

        t_xs_tr = torch.from_numpy(xs_tr)
        t_xm_tr = torch.from_numpy(xm_tr)
        t_y_tr = torch.from_numpy(y_tr)
        t_xs_val = torch.from_numpy(xs_val)
        t_xm_val = torch.from_numpy(xm_val)
        t_y_val = torch.from_numpy(y_val)
        t_xs_ev = torch.from_numpy(xs_ev)
        t_xm_ev = torch.from_numpy(xm_ev)
        t_prior = torch.tensor(mkt_prior_val, dtype=torch.float32)

        preds_seeds = []
        for seed in self.seeds:
            torch.manual_seed(seed)
            np.random.seed(seed)
            model = SectorFormer(
                n_sec_feat=F_s,
                n_macro_feat=xm_tr.shape[1],
                n_sectors=S,
                d_model=self.d_model,
                n_heads=self.n_heads,
                dropout=self.dropout,
                max_mkt_delta=self.max_mkt_delta,
            )
            opt = torch.optim.AdamW(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.max_epochs)

            best_vloss = float("inf")
            best_state = None
            bad = 0
            patience = 15

            for _ in range(self.max_epochs):
                model.train()
                perm = torch.randperm(N_tr)
                for i in range(0, N_tr, self.batch_size):
                    idx = perm[i : i + self.batch_size]
                    if len(idx) < 4:
                        continue
                    opt.zero_grad()
                    yp, ap, _ = model(t_xs_tr[idx], t_xm_tr[idx], t_prior)
                    loss = hybrid_sector_loss(yp, ap, t_y_tr[idx])
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    opt.step()
                sched.step()

                model.eval()
                with torch.no_grad():
                    yp_v, ap_v, _ = model(t_xs_val, t_xm_val, t_prior)
                    vloss = hybrid_sector_loss(yp_v, ap_v, t_y_val).item()
                if vloss < best_vloss - 1e-5:
                    best_vloss = vloss
                    best_state = {k: v.clone() for k, v in model.state_dict().items()}
                    bad = 0
                else:
                    bad += 1
                    if bad >= patience:
                        break

            if best_state is not None:
                model.load_state_dict(best_state)
            model.eval()
            with torch.no_grad():
                yp_ev, _, _ = model(t_xs_ev, t_xm_ev, t_prior)
                preds_seeds.append(yp_ev.numpy())

        return np.mean(preds_seeds, axis=0).astype(np.float32)


def run_iteration_3():
    df = load_raw_merged_df("data")
    y_df, realized = compute_forward_targets(df, horizon=HORIZON_DAYS)
    y_all = y_df.to_numpy(dtype=np.float32)

    macro_df, sec_dfs, feat_valid = build_all_feature_views(df)
    valid_mask = feat_valid & (~y_df.isna().any(axis=1)).to_numpy(copy=True)
    folds = get_dev_walk_forward_folds(df["Date"], realized, valid_mask)

    model = SectorFormerEnsemble(seeds=(42, 102, 202), d_model=64, n_heads=4, dropout=0.30)
    fold_rows = []
    for fold in folds:
        y_tr_mean = y_all[fold.train_idx].mean(axis=0)
        preds = model.fit_predict(
            macro_df, sec_dfs, y_all, df["Date"], realized, fold.train_idx, fold.eval_idx
        )
        m = evaluate_predictions(y_all[fold.eval_idx], preds, y_train_mean=y_tr_mean)
        m["fold"] = fold.name
        fold_rows.append(m)

    key_cols = [
        "fold", "mean_r2", "oos_r2_vs_train_mean", "rel_r2",
        "rmse", "dir_acc", "mean_ts_ic", "mean_cs_ic",
        "top2_excess_1y_ret", "top2_vs_bot2_spread", "top2_win_rate_vs_ew",
    ]
    summ = summarize_fold_metrics(fold_rows)
    print("=" * 98)
    print("MODEL: SectorFormerEnsemble (Cross-Sector Attention + GRN + Hybrid Huber/Cosine Loss)")
    print(summ[key_cols].to_string(index=False, float_format=lambda x: f"{x:7.4f}"))


if __name__ == "__main__":
    run_iteration_3()
