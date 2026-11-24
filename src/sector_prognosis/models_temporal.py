"""Iteration 6: Decoupled Temporal Macro Transformer (TempMarketNet) +
Dedicated Cross-Sectional ListNet Ranker (CrossSectorRankNet) with
Empirical-Bayes Shrinkage and Stochastic Weight Averaging (SWA).

Strict zero-leakage guarantees:
- All sequences at decision index t use strictly indices <= t ([-231..0]).
- All scalers, priors, and shrinkage weights are fit strictly on training indices.
- Inner validation uses a strict 250-day purge + 5-day embargo.
"""

from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import RobustScaler

from . import HORIZON_DAYS, SECTORS
from .data import compute_forward_targets, load_raw_merged_df
from .features import build_all_feature_views
from .metrics import evaluate_predictions, summarize_fold_metrics
from .models_v2 import ALPHA_SEC_CORE_COLS, MARKET_CORE_COLS
from .splits import get_dev_walk_forward_folds, split_inner_train_val

SEQ_OFFSETS = np.array([-231, -210, -189, -168, -147, -126, -105, -84, -63, -42, -21, 0], dtype=np.int64)


class TempMarketNet(nn.Module):
    """Multi-Scale Causal 1D-Conv + Pre-LN Transformer for Market-Level 1Y Return Timing."""

    def __init__(self, f_mac: int, d_model: int = 48, n_heads: int = 4, dropout: float = 0.35):
        super().__init__()
        self.in_proj = nn.Sequential(nn.Linear(f_mac, d_model), nn.LayerNorm(d_model), nn.SiLU())
        # Causal depthwise-separable local trend conv (kernel=3 over monthly steps = 3M trend)
        self.conv_local = nn.Conv1d(d_model, d_model, kernel_size=3, padding=0, groups=d_model)
        self.conv_mix = nn.Linear(d_model, d_model)
        self.pos_emb = nn.Parameter(torch.randn(1, len(SEQ_OFFSETS), d_model) * 0.02)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 2,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=1, enable_nested_tensor=False)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model * 2),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model // 2),
            nn.SiLU(),
            nn.Linear(d_model // 2, 1),
        )
        nn.init.normal_(self.head[-1].weight, std=0.01)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, x_mac_seq: torch.Tensor) -> torch.Tensor:
        # x_mac_seq: (B, T=12, F_mac)
        if self.training:
            x_mac_seq = x_mac_seq + 0.05 * torch.randn_like(x_mac_seq)
        h = self.in_proj(x_mac_seq)  # (B, T, D)
        # Strictly causal left-pad for kernel_size=3
        hc = F.pad(h.transpose(1, 2), (2, 0))
        hc = self.conv_mix(F.silu(self.conv_local(hc)).transpose(1, 2))
        h = h + 0.5 * hc + self.pos_emb[:, : h.shape[1], :]
        h = self.encoder(h)
        # Concatenate latest state + temporal mean state
        pooled = torch.cat([h[:, -1, :], h.mean(dim=1)], dim=-1)
        return 0.22 * torch.tanh(self.head(pooled).squeeze(-1))


class CrossSectorRankNet(nn.Module):
    """Dedicated Cross-Sector Attention Network trained with ListNet + Huber Alpha Loss.

    Operates purely on cross-sectional relative sector trajectories (invariant to market beta).
    """

    def __init__(self, f_sec: int, d_model: int = 32, n_heads: int = 4, dropout: float = 0.35):
        super().__init__()
        # Summarize 12-month trajectory per sector with lightweight temporal pooling + latest
        self.step_proj = nn.Sequential(nn.Linear(f_sec, d_model), nn.LayerNorm(d_model), nn.SiLU())
        self.trend_proj = nn.Sequential(nn.Linear(d_model * 2, d_model), nn.LayerNorm(d_model), nn.SiLU())
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 2,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.sec_attn = nn.TransformerEncoder(enc_layer, num_layers=1, enable_nested_tensor=False)
        self.alpha_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )
        nn.init.normal_(self.alpha_head[-1].weight, std=0.01)
        nn.init.zeros_(self.alpha_head[-1].bias)

    def forward(self, x_sec_seq: torch.Tensor) -> torch.Tensor:
        # x_sec_seq: (B, T=12, S=6, F_sec)
        if self.training:
            x_sec_seq = x_sec_seq + 0.05 * torch.randn_like(x_sec_seq)
        h = self.step_proj(x_sec_seq)  # (B, T, S, D)
        # Combine latest snapshot + trajectory slope (latest - mean of first half)
        h_last = h[:, -1, :, :]  # (B, S, D)
        h_slope = h[:, -3:, :, :].mean(dim=1) - h[:, :6, :, :].mean(dim=1)  # (B, S, D)
        hs = self.trend_proj(torch.cat([h_last, h_slope], dim=-1))  # (B, S, D)
        hs = self.sec_attn(hs)  # Cross-sector attention across the 6 sectors!
        raw = self.alpha_head(hs).squeeze(-1)  # (B, S)
        # Exact zero-sum bounded alpha residual
        raw = raw - raw.mean(dim=-1, keepdim=True)
        return 0.18 * torch.tanh(raw)


def _listnet_loss(alpha_pred: torch.Tensor, alpha_true: torch.Tensor, temp: float = 0.08) -> torch.Tensor:
    """ListNet cross-entropy over softmax distributions across the 6 sectors."""
    p_true = F.softmax(alpha_true / temp, dim=-1)
    log_p_pred = F.log_softmax(alpha_pred / temp, dim=-1)
    return -(p_true * log_p_pred).sum(dim=-1).mean()


def _eb_shrinkage(pred_val: np.ndarray, true_val: np.ndarray, prior_strength: float = 0.5, cap: float = 0.65) -> float:
    """Empirical-Bayes / James-Stein shrinkage toward 0 (prevents over-betting on noisy inner val)."""
    num = float(np.sum(pred_val * true_val))
    pred_energy = float(np.sum(pred_val * pred_val))
    target_energy = float(np.sum(true_val * true_val)) + 1e-8
    # Regularize denominator proportional to target variance so low-SNR inner fits shrink smoothly
    denom = pred_energy + prior_strength * 0.15 * target_energy + 1e-8
    return float(np.clip(num / denom, 0.0, cap))


class DecoupledTemporalSectorModel:
    """Decoupled TempMarketNet + CrossSectorRankNet with SWA & Empirical-Bayes Shrinkage."""

    def __init__(
        self,
        seeds: Tuple[int, ...] = (42, 102, 202),
        epochs_mkt: int = 45,
        epochs_alpha: int = 45,
        train_stride: int = 5,
        eb_prior_mkt: float = 0.6,
        eb_prior_alpha: float = 0.6,
        cap_mkt: float = 0.65,
        cap_alpha: float = 0.65,
        device: str | None = None,
    ):
        self.seeds = seeds
        self.epochs_mkt = epochs_mkt
        self.epochs_alpha = epochs_alpha
        self.train_stride = train_stride
        self.eb_prior_mkt = eb_prior_mkt
        self.eb_prior_alpha = eb_prior_alpha
        self.cap_mkt = cap_mkt
        self.cap_alpha = cap_alpha
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))

    def _fit_market_swa(
        self, xm_tr: np.ndarray, ym_tr: np.ndarray, xm_va: np.ndarray, xm_ev: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        if self.device.type == "cpu":
            torch.set_num_threads(1)
        t_tr = torch.from_numpy(xm_tr).to(self.device)
        t_y = torch.from_numpy(ym_tr).to(self.device)
        t_va = torch.from_numpy(xm_va).to(self.device)
        t_ev = torch.from_numpy(xm_ev).to(self.device)

        va_preds, ev_preds = [], []
        swa_start = int(self.epochs_mkt * 0.65)
        for seed in self.seeds:
            torch.manual_seed(seed)
            np.random.seed(seed)
            net = TempMarketNet(xm_tr.shape[-1], d_model=48, n_heads=4, dropout=0.35).to(self.device)
            opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=3e-2)
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.epochs_mkt)

            swa_state = None
            swa_n = 0
            for ep in range(self.epochs_mkt):
                net.train()
                opt.zero_grad()
                pred = net(t_tr)
                loss = F.huber_loss(pred, t_y, delta=0.12)
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                opt.step()
                sched.step()
                if ep >= swa_start:
                    if swa_state is None:
                        swa_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
                    else:
                        for k, v in net.state_dict().items():
                            swa_state[k] += v.detach()
                    swa_n += 1

            if swa_state is not None and swa_n > 0:
                net.load_state_dict({k: v / swa_n for k, v in swa_state.items()})
            net.eval()
            with torch.no_grad():
                va_preds.append(net(t_va).cpu().numpy())
                ev_preds.append(net(t_ev).cpu().numpy())

        return np.mean(va_preds, axis=0).astype(np.float32), np.mean(ev_preds, axis=0).astype(np.float32)

    def _fit_alpha_swa(
        self, xs_tr: np.ndarray, ya_tr: np.ndarray, xs_va: np.ndarray, xs_ev: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        if self.device.type == "cpu":
            torch.set_num_threads(1)
        t_tr = torch.from_numpy(xs_tr).to(self.device)
        t_y = torch.from_numpy(ya_tr).to(self.device)
        t_va = torch.from_numpy(xs_va).to(self.device)
        t_ev = torch.from_numpy(xs_ev).to(self.device)

        va_preds, ev_preds = [], []
        swa_start = int(self.epochs_alpha * 0.65)
        for seed in self.seeds:
            torch.manual_seed(seed)
            np.random.seed(seed)
            net = CrossSectorRankNet(xs_tr.shape[-1], d_model=32, n_heads=4, dropout=0.35).to(self.device)
            opt = torch.optim.AdamW(net.parameters(), lr=2.5e-3, weight_decay=3e-2)
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.epochs_alpha)

            swa_state = None
            swa_n = 0
            for ep in range(self.epochs_alpha):
                net.train()
                opt.zero_grad()
                pred = net(t_tr)
                loss = F.huber_loss(pred, t_y, delta=0.10) + 0.04 * _listnet_loss(pred, t_y, temp=0.08)
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                opt.step()
                sched.step()
                if ep >= swa_start:
                    if swa_state is None:
                        swa_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
                    else:
                        for k, v in net.state_dict().items():
                            swa_state[k] += v.detach()
                    swa_n += 1

            if swa_state is not None and swa_n > 0:
                net.load_state_dict({k: v / swa_n for k, v in swa_state.items()})
            net.eval()
            with torch.no_grad():
                va_preds.append(net(t_va).cpu().numpy())
                ev_preds.append(net(t_ev).cpu().numpy())

        return np.mean(va_preds, axis=0).astype(np.float32), np.mean(ev_preds, axis=0).astype(np.float32)

    def fit_predict(
        self,
        macro_df: pd.DataFrame,
        sec_dfs: Dict[str, pd.DataFrame],
        y_all: np.ndarray,
        dates: pd.Series,
        realized_dates: pd.Series,
        train_idx: np.ndarray,
        eval_idx: np.ndarray,
        return_diagnostics: bool = False,
    ):
        mac_raw = macro_df[MARKET_CORE_COLS].to_numpy(dtype=np.float32)
        sec_raw = np.stack([sec_dfs[s][ALPHA_SEC_CORE_COLS].to_numpy(dtype=np.float32) for s in SECTORS], axis=1)

        # 1. Calibrate Empirical-Bayes shrinkage strictly on purged inner split
        itr, ival = split_inner_train_val(train_idx, dates, realized_dates, val_fraction=0.20)
        itr_sub = itr[:: self.train_stride]

        sp_i = y_all[itr].mean(axis=0).astype(np.float32)
        mp_i = float(sp_i.mean())
        ap_i = sp_i - mp_i

        # Fit scalers strictly on itr_sub
        m_sc = RobustScaler().fit(mac_raw[itr_sub])
        mac_s = np.clip(m_sc.transform(mac_raw), -3.5, 3.5).astype(np.float32)
        s_sc = RobustScaler().fit(sec_raw[itr_sub].reshape(-1, sec_raw.shape[-1]))
        sec_s = np.clip(s_sc.transform(sec_raw.reshape(-1, sec_raw.shape[-1])), -3.5, 3.5).reshape(sec_raw.shape).astype(np.float32)

        xm_tr = mac_s[itr_sub[:, None] + SEQ_OFFSETS[None, :]]
        xm_va = mac_s[ival[:, None] + SEQ_OFFSETS[None, :]]
        xs_tr = sec_s[itr_sub[:, None] + SEQ_OFFSETS[None, :]]
        xs_va = sec_s[ival[:, None] + SEQ_OFFSETS[None, :]]

        ym_tr = np.clip(y_all[itr_sub].mean(axis=1) - mp_i, -0.30, 0.30).astype(np.float32)
        ya_tr = np.clip((y_all[itr_sub] - y_all[itr_sub].mean(axis=1, keepdims=True)) - ap_i[None, :], -0.25, 0.25).astype(np.float32)
        ym_va = (y_all[ival].mean(axis=1) - mp_i).astype(np.float32)
        ya_va = ((y_all[ival] - y_all[ival].mean(axis=1, keepdims=True)) - ap_i[None, :]).astype(np.float32)

        pm_va, _ = self._fit_market_swa(xm_tr, ym_tr, xm_va, xm_va[:2])
        pa_va, _ = self._fit_alpha_swa(xs_tr, ya_tr, xs_va, xs_va[:2])

        w_m = _eb_shrinkage(pm_va, ym_va, prior_strength=self.eb_prior_mkt, cap=self.cap_mkt)
        w_a = _eb_shrinkage(pa_va, ya_va, prior_strength=self.eb_prior_alpha, cap=self.cap_alpha)

        # 2. Refit on full train_idx (still strictly purged from eval_idx!)
        tr_sub = train_idx[:: self.train_stride]
        sp = y_all[train_idx].mean(axis=0).astype(np.float32)
        mp = float(sp.mean())
        ap = sp - mp

        m_sc_f = RobustScaler().fit(mac_raw[tr_sub])
        mac_sf = np.clip(m_sc_f.transform(mac_raw), -3.5, 3.5).astype(np.float32)
        s_sc_f = RobustScaler().fit(sec_raw[tr_sub].reshape(-1, sec_raw.shape[-1]))
        sec_sf = np.clip(s_sc_f.transform(sec_raw.reshape(-1, sec_raw.shape[-1])), -3.5, 3.5).reshape(sec_raw.shape).astype(np.float32)

        xm_trf = mac_sf[tr_sub[:, None] + SEQ_OFFSETS[None, :]]
        xm_ev = mac_sf[eval_idx[:, None] + SEQ_OFFSETS[None, :]]
        xs_trf = sec_sf[tr_sub[:, None] + SEQ_OFFSETS[None, :]]
        xs_ev = sec_sf[eval_idx[:, None] + SEQ_OFFSETS[None, :]]

        ym_trf = np.clip(y_all[tr_sub].mean(axis=1) - mp, -0.30, 0.30).astype(np.float32)
        ya_trf = np.clip((y_all[tr_sub] - y_all[tr_sub].mean(axis=1, keepdims=True)) - ap[None, :], -0.25, 0.25).astype(np.float32)

        _, pm_ev = self._fit_market_swa(xm_trf, ym_trf, xm_trf[:2], xm_ev)
        _, pa_ev = self._fit_alpha_swa(xs_trf, ya_trf, xs_trf[:2], xs_ev)

        preds = (sp[None, :] + w_m * pm_ev[:, None] + w_a * pa_ev).astype(np.float32)
        if return_diagnostics:
            return preds, {"w_m": w_m, "w_a": w_a}
        return preds


def run_iteration_6_dev():
    df = load_raw_merged_df("data")
    y_df, realized = compute_forward_targets(df, horizon=HORIZON_DAYS)
    y_all = y_df.to_numpy(dtype=np.float32)
    macro_df, sec_dfs, feat_valid = build_all_feature_views(df)
    valid_mask = feat_valid & (~y_df.isna().any(axis=1)).to_numpy(copy=True)
    valid_mask[:500] = False  # 250d feature warmup + 231d sequence lookback

    folds = get_dev_walk_forward_folds(df["Date"], realized, valid_mask)
    mdl = DecoupledTemporalSectorModel()

    rows_naive, rows_v6 = [], []
    for f in folds:
        ym = y_all[f.train_idx].mean(axis=0)
        p_n = np.tile(ym, (len(f.eval_idx), 1))
        mn = evaluate_predictions(y_all[f.eval_idx], p_n, y_train_mean=ym)
        mn["fold"] = f.name
        rows_naive.append(mn)

        p6, diag = mdl.fit_predict(
            macro_df, sec_dfs, y_all, df["Date"], realized, f.train_idx, f.eval_idx, return_diagnostics=True
        )
        m6 = evaluate_predictions(y_all[f.eval_idx], p6, y_train_mean=ym)
        m6["fold"] = f.name
        rows_v6.append(m6)
        print(
            f"  {f.name}: w_m={diag['w_m']:.2f}, w_a={diag['w_a']:.2f} | "
            f"oos_r2={m6['oos_r2_vs_train_mean']:+.4f}, rel_r2={m6['rel_r2']:+.4f}, "
            f"ts_ic={m6['mean_ts_ic']:+.4f}, cs_ic={m6['mean_cs_ic']:+.4f}, "
            f"top2_exc={m6['top2_excess_1y_ret']:+.4f}",
            flush=True,
        )

    cols = [
        "fold", "mean_r2", "oos_r2_vs_train_mean", "rel_r2",
        "rmse", "dir_acc", "mean_ts_ic", "mean_cs_ic",
        "top2_excess_1y_ret", "top2_vs_bot2_spread", "top2_win_rate_vs_ew",
    ]
    print("\n=== NaiveMean (Dev 2012-2019) ===")
    print(summarize_fold_metrics(rows_naive)[cols].to_string(index=False, float_format=lambda x: f"{x:7.4f}"))
    print("\n=== Iteration 6: DecoupledTemporalSectorModel (TempMarketNet + CrossSectorRankNet + SWA + EB) ===")
    print(summarize_fold_metrics(rows_v6)[cols].to_string(index=False, float_format=lambda x: f"{x:7.4f}"))


if __name__ == "__main__":
    run_iteration_6_dev()
