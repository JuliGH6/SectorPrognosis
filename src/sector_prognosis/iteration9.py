"""Iteration 9: Causal CAPM Beta-Scaled Decomposition + 4-Family Uncertainty-Gated Temporal Engine
(TempSectorFormer + CausalTemporalGRU + CausalMultiScaleTCN + CatBoost) + Beta-Neutral Alpha.

Key financial & ML innovations (100% self-contained on the repo's dataset, zero external APIs):
1. Causal Vasicek-Shrunk CAPM Beta Decomposition:
       y[t, s] = mu_train[s] + beta_train[s] * r_mkt[t] + alpha_beta_neutral[t, s]
   Eliminates the beta=1.0 distortion (over-predicting Utilities/Staples swings and
   under-predicting Industrials/Materials/Discretionary swings) AND naturally rotates
   into defensives when market timing turns bearish and cyclicals when bullish!
2. 4-Family Market Timing Ensemble with Epistemic Uncertainty Gating:
   - TempSectorFormerBackbone (Self-Attention Transformer)
   - CausalTemporalGRU (Recurrent/State-Space with temporal attention)
   - CausalMultiScaleTCN (Strictly causal dilated 1D-CNN, dilations 1 & 2)
   - Shallow CatBoost (Non-extrapolative oblivious trees)
3. Beta-Neutral 3-Family Cross-Sectional Alpha Engine:
   - Sparse ElasticNet (L1/L2) + shallow CatBoost + Huber LightGBM trained on true
     beta-neutral idiosyncratic returns with Empirical-Bayes shrinkage.
4. Additional strictly causal systemic-risk features:
   - Equal-weight market 250d drawdown (`mkt_ew_dd_250d`)
   - Cross-sector 63d average pairwise correlation (`mkt_avg_corr_63d`)
"""

from pathlib import Path
from typing import Dict, Tuple

import lightgbm as lgb
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from catboost import CatBoostRegressor
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import RobustScaler

from . import HORIZON_DAYS, SECTORS
from .champion import TempSectorFormerBackbone
from .data import compute_forward_targets, load_raw_merged_df
from .features import build_all_feature_views
from .iteration8 import (
    MARKET_V8_COLS,
    CausalTemporalGRU,
    augment_macro_features_v8,
)
from .metrics import evaluate_predictions, summarize_fold_metrics
from .models_temporal import SEQ_OFFSETS, _eb_shrinkage
from .models_v2 import (
    ALPHA_SEC_CORE_COLS,
    _recency_weights,
    _winsorize_train_target,
    extract_curated_alpha_panel,
)
from .splits import get_dev_walk_forward_folds, split_inner_train_val, validate_no_leakage

MARKET_V9_COLS = MARKET_V8_COLS + ["mkt_ew_dd_250d", "mkt_avg_corr_63d"]


def augment_macro_features_v9(df: pd.DataFrame, macro_df: pd.DataFrame, data_dir: str | Path = "data") -> pd.DataFrame:
    """Add strictly causal market drawdown and 63-day cross-sector systemic correlation."""
    out = augment_macro_features_v8(df, macro_df, data_dir=data_dir)
    prices = df[SECTORS].astype(np.float64)
    # Equal-weight normalized index (strictly causal)
    ew_idx = (prices / prices.iloc[0]).mean(axis=1)
    out["mkt_ew_dd_250d"] = (ew_idx / ew_idx.rolling(250, min_periods=250).max() - 1.0).astype(np.float32)

    # Strictly causal 63d average pairwise correlation across the 6 sectors (fast variance-ratio formula!)
    # Avg pairwise corr = (Var(sum z_i) - S) / (S * (S - 1)) where z_i are 63d standardized daily returns
    dret = np.log(prices / prices.shift(1))
    roll_mean = dret.rolling(63, min_periods=63).mean()
    roll_std = dret.rolling(63, min_periods=63).std() + 1e-8
    z = (dret - roll_mean) / roll_std
    # Rolling mean of (sum_s z_s)^2 over 63d estimates Var(sum z_i)
    z_sum_sq = (z.sum(axis=1) ** 2).rolling(63, min_periods=63).mean()
    S = float(len(SECTORS))
    avg_corr = ((z_sum_sq - S) / (S * (S - 1.0))).clip(-0.5, 1.0)
    out["mkt_avg_corr_63d"] = avg_corr.astype(np.float32)
    return out


def compute_causal_sector_betas(y_train: np.ndarray, vasicek_shrink: float = 0.20) -> np.ndarray:
    """Compute strictly causal CAPM market betas on training targets with Vasicek shrinkage toward 1.0.

    Returns:
        betas: shape (6,) positive betas whose mean is exactly 1.0.
    """
    y_mkt = y_train.mean(axis=1)
    var_m = float(np.var(y_mkt)) + 1e-8
    raw_betas = np.array(
        [float(np.cov(y_train[:, j], y_mkt, bias=True)[0, 1]) / var_m for j in range(y_train.shape[1])],
        dtype=np.float32,
    )
    # Vasicek shrinkage toward 1.0 + clip for stability + normalize mean=1.0
    shrunk = (1.0 - vasicek_shrink) * raw_betas + vasicek_shrink * 1.0
    shrunk = np.clip(shrunk, 0.45, 1.55)
    shrunk = shrunk / float(np.mean(shrunk))
    return shrunk.astype(np.float32)


class CausalMultiScaleTCN(nn.Module):
    """Strictly causal dilated 1D-CNN (dilations 1 & 2) with gated residual connections."""

    def __init__(self, f_mac: int, channels: int = 48, dropout: float = 0.35):
        super().__init__()
        self.in_proj = nn.Sequential(nn.Linear(f_mac, channels), nn.LayerNorm(channels), nn.SiLU())
        # Dilation 1 (receptive field 3 months) + Dilation 2 (receptive field 7 months)
        self.conv1 = nn.Conv1d(channels, channels * 2, kernel_size=3, dilation=1, padding=0)
        self.conv2 = nn.Conv1d(channels, channels * 2, kernel_size=3, dilation=2, padding=0)
        self.norm1 = nn.LayerNorm(channels)
        self.norm2 = nn.LayerNorm(channels)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Sequential(
            nn.LayerNorm(channels * 2),
            nn.Dropout(dropout),
            nn.Linear(channels * 2, channels // 2),
            nn.SiLU(),
            nn.Linear(channels // 2, 1),
        )
        nn.init.normal_(self.head[-1].weight, std=0.01)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, x_mac_seq: torch.Tensor) -> torch.Tensor:
        if self.training:
            x_mac_seq = x_mac_seq + 0.04 * torch.randn_like(x_mac_seq)
        h = self.in_proj(x_mac_seq)  # (B, T, C)

        # Block 1: causal left-pad = (k-1)*d = 2
        c1 = self.conv1(F.pad(h.transpose(1, 2), (2, 0))).transpose(1, 2)
        a1, b1 = c1.chunk(2, dim=-1)
        h = self.norm1(h + self.drop(a1 * torch.sigmoid(b1)))

        # Block 2: causal left-pad = (k-1)*d = 4
        c2 = self.conv2(F.pad(h.transpose(1, 2), (4, 0))).transpose(1, 2)
        a2, b2 = c2.chunk(2, dim=-1)
        h = self.norm2(h + self.drop(a2 * torch.sigmoid(b2)))

        pooled = torch.cat([h[:, -1, :], h.mean(dim=1)], dim=-1)
        return 0.20 * torch.tanh(self.head(pooled).squeeze(-1))


class BetaScaledUncertaintyEnsembleV9:
    """Iteration 9: Causal Beta-Scaled 4-Family Uncertainty-Gated Market Engine
    + 3-Family Beta-Neutral Cross-Sectional Alpha Engine."""

    def __init__(
        self,
        seeds: Tuple[int, ...] = (42, 102, 202),
        epochs: int = 35,
        kappa_uncertainty: float = 0.35,
        vasicek_shrink: float = 0.20,
        eb_prior_mkt: float = 0.35,
        eb_prior_alpha: float = 0.45,
        cap_mkt: float = 0.75,
        cap_alpha: float = 0.70,
        market_weights: Tuple[float, float, float, float] = (0.40, 0.20, 0.20, 0.20),  # TF, GRU, TCN, CB
        device: str | None = None,
    ):
        self.seeds = seeds
        self.epochs = epochs
        self.kappa_uncertainty = kappa_uncertainty
        self.vasicek_shrink = vasicek_shrink
        self.eb_prior_mkt = eb_prior_mkt
        self.eb_prior_alpha = eb_prior_alpha
        self.cap_mkt = cap_mkt
        self.cap_alpha = cap_alpha
        self.market_weights = market_weights
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))

    def _fit_neural_market_families(
        self,
        mac_raw: np.ndarray,
        sec_raw: np.ndarray,
        y_all: np.ndarray,
        itr: np.ndarray,
        ival: np.ndarray,
        eval_idx: np.ndarray,
        sec_prior: np.ndarray,
    ):
        if self.device.type == "cpu":
            torch.set_num_threads(1)

        itr_sub = itr[::5]
        mp = float(sec_prior.mean())
        ap = sec_prior - mp

        m_sc = RobustScaler().fit(mac_raw[itr_sub])
        mac_s = np.clip(m_sc.transform(mac_raw), -3.5, 3.5).astype(np.float32)
        s_sc = RobustScaler().fit(sec_raw[itr_sub].reshape(-1, sec_raw.shape[-1]))
        sec_s = np.clip(s_sc.transform(sec_raw.reshape(-1, sec_raw.shape[-1])), -3.5, 3.5).reshape(sec_raw.shape).astype(np.float32)

        xs_tr, xm_tr = sec_s[itr_sub[:, None] + SEQ_OFFSETS[None, :]], mac_s[itr_sub[:, None] + SEQ_OFFSETS[None, :]]
        xs_va, xm_va = sec_s[ival[:, None] + SEQ_OFFSETS[None, :]], mac_s[ival[:, None] + SEQ_OFFSETS[None, :]]
        xs_ev, xm_ev = sec_s[eval_idx[:, None] + SEQ_OFFSETS[None, :]], mac_s[eval_idx[:, None] + SEQ_OFFSETS[None, :]]

        ym_tr = np.clip(y_all[itr_sub].mean(axis=1) - mp, -0.30, 0.30).astype(np.float32)
        ya_tr = np.clip((y_all[itr_sub] - y_all[itr_sub].mean(axis=1, keepdims=True)) - ap[None, :], -0.25, 0.25).astype(np.float32)
        ym_va = (y_all[ival].mean(axis=1) - mp).astype(np.float32)
        ya_va = ((y_all[ival] - y_all[ival].mean(axis=1, keepdims=True)) - ap[None, :]).astype(np.float32)

        txs_tr, txm_tr = torch.from_numpy(xs_tr).to(self.device), torch.from_numpy(xm_tr).to(self.device)
        tym_tr, tya_tr = torch.from_numpy(ym_tr).to(self.device), torch.from_numpy(ya_tr).to(self.device)
        txs_va, txm_va = torch.from_numpy(xs_va).to(self.device), torch.from_numpy(xm_va).to(self.device)
        tym_va, tya_va = torch.from_numpy(ym_va).to(self.device), torch.from_numpy(ya_va).to(self.device)
        txs_ev, txm_ev = torch.from_numpy(xs_ev).to(self.device), torch.from_numpy(xm_ev).to(self.device)

        tf_va, tf_ev, gru_va, gru_ev, tcn_va, tcn_ev = [], [], [], [], [], []
        N, bs = len(txs_tr), 128

        def _train_1d_seq(model, seed_offset):
            torch.manual_seed(seed_offset)
            np.random.seed(seed_offset)
            opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=2.5e-2)
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.epochs)
            best_l, best_st, bad = float("inf"), None, 0
            for _ in range(self.epochs):
                model.train()
                perm = torch.randperm(N, device=self.device)
                for i in range(0, N, bs):
                    idx = perm[i : i + bs]
                    if len(idx) < 4:
                        continue
                    opt.zero_grad()
                    loss = F.huber_loss(model(txm_tr[idx]), tym_tr[idx], delta=0.12)
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    opt.step()
                sched.step()
                model.eval()
                with torch.no_grad():
                    vl = F.huber_loss(model(txm_va), tym_va, delta=0.12).item()
                if vl < best_l - 1e-5:
                    best_l, best_st, bad = vl, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}, 0
                else:
                    bad += 1
                    if bad >= 10:
                        break
            if best_st is not None:
                model.load_state_dict({k: v.to(self.device) for k, v in best_st.items()})
            model.eval()
            with torch.no_grad():
                return model(txm_va).cpu().numpy(), model(txm_ev).cpu().numpy()

        for seed in self.seeds:
            # 1. TempSectorFormer
            torch.manual_seed(seed)
            np.random.seed(seed)
            m_tf = TempSectorFormerBackbone(sec_raw.shape[-1], mac_raw.shape[-1], d_model=48, n_heads=4, dropout=0.35).to(self.device)
            opt = torch.optim.AdamW(m_tf.parameters(), lr=1.5e-3, weight_decay=2e-2)
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.epochs)
            best_l, best_st, bad = float("inf"), None, 0
            for _ in range(self.epochs):
                m_tf.train()
                perm = torch.randperm(N, device=self.device)
                for i in range(0, N, bs):
                    idx = perm[i : i + bs]
                    if len(idx) < 4:
                        continue
                    opt.zero_grad()
                    pm, pa = m_tf(txs_tr[idx], txm_tr[idx])
                    loss = (
                        F.huber_loss(pm, tym_tr[idx], delta=0.12)
                        + 1.5 * F.huber_loss(pa, tya_tr[idx], delta=0.10)
                        - 0.05 * F.cosine_similarity(pa, tya_tr[idx], dim=-1, eps=1e-6).mean()
                    )
                    loss.backward()
                    nn.utils.clip_grad_norm_(m_tf.parameters(), 1.0)
                    opt.step()
                sched.step()
                m_tf.eval()
                with torch.no_grad():
                    pm_v, pa_v = m_tf(txs_va, txm_va)
                    vl = (F.huber_loss(pm_v, tym_va, delta=0.12) + 1.5 * F.huber_loss(pa_v, tya_va, delta=0.10)).item()
                if vl < best_l - 1e-5:
                    best_l, best_st, bad = vl, {k: v.detach().cpu().clone() for k, v in m_tf.state_dict().items()}, 0
                else:
                    bad += 1
                    if bad >= 10:
                        break
            if best_st is not None:
                m_tf.load_state_dict({k: v.to(self.device) for k, v in best_st.items()})
            m_tf.eval()
            with torch.no_grad():
                tf_va.append(m_tf(txs_va, txm_va)[0].cpu().numpy())
                tf_ev.append(m_tf(txs_ev, txm_ev)[0].cpu().numpy())

            # 2. CausalTemporalGRU
            va_g, ev_g = _train_1d_seq(CausalTemporalGRU(mac_raw.shape[-1], hidden_dim=48, dropout=0.35).to(self.device), seed + 777)
            gru_va.append(va_g); gru_ev.append(ev_g)

            # 3. CausalMultiScaleTCN
            va_c, ev_c = _train_1d_seq(CausalMultiScaleTCN(mac_raw.shape[-1], channels=48, dropout=0.35).to(self.device), seed + 1555)
            tcn_va.append(va_c); tcn_ev.append(ev_c)

        return (
            np.mean(tf_va, axis=0).astype(np.float32), np.mean(tf_ev, axis=0).astype(np.float32),
            np.mean(gru_va, axis=0).astype(np.float32), np.mean(gru_ev, axis=0).astype(np.float32),
            np.mean(tcn_va, axis=0).astype(np.float32), np.mean(tcn_ev, axis=0).astype(np.float32),
        )

    def _fit_cb_market(self, mac_raw: np.ndarray, y_all: np.ndarray, tr_idx: np.ndarray, ev_idx: np.ndarray, mp: float) -> np.ndarray:
        tr_sub = tr_idx[::5]
        wd = _recency_weights(len(tr_sub), half_life_steps=10.0 * 50.0)
        ym = _winsorize_train_target(y_all[tr_sub].mean(axis=1) - mp, 0.04, 0.96)
        cb = CatBoostRegressor(
            iterations=200, learning_rate=0.03, depth=3, l2_leaf_reg=30.0,
            subsample=0.7, rsm=0.6, random_seed=42, verbose=0,
        )
        cb.fit(mac_raw[tr_sub], ym, sample_weight=wd)
        return cb.predict(mac_raw[ev_idx]).astype(np.float32)

    def _fit_beta_neutral_alpha(
        self, macro_v9, sec_dfs, y_all, tr_idx, ev_idx, sec_prior: np.ndarray, betas: np.ndarray
    ) -> np.ndarray:
        tr_sub = tr_idx[::5]
        mp = float(sec_prior.mean())
        wd = _recency_weights(len(tr_sub), half_life_steps=10.0 * 50.0)
        wp = np.repeat(wd, len(SECTORS))

        # True beta-neutral idiosyncratic return: (y[t,s] - mu[s]) - beta[s] * (y_mkt[t] - mu_mkt)
        y_mkt_resid = y_all[tr_sub].mean(axis=1) - mp
        ya_2d = (y_all[tr_sub] - sec_prior[None, :]) - y_mkt_resid[:, None] * betas[None, :]
        ya = _winsorize_train_target(ya_2d.reshape(-1), 0.04, 0.96)

        Xa_tr, _ = extract_curated_alpha_panel(macro_v9, sec_dfs, tr_sub)
        Xa_ev, _ = extract_curated_alpha_panel(macro_v9, sec_dfs, ev_idx)

        # 1. Sparse ElasticNet
        sc = RobustScaler()
        Xa_tr_s = np.clip(sc.fit_transform(Xa_tr), -3.0, 3.0)
        Xa_ev_s = np.clip(sc.transform(Xa_ev), -3.0, 3.0)
        en = ElasticNet(alpha=0.012, l1_ratio=0.55, max_iter=2000, random_state=42)
        en.fit(Xa_tr_s, ya, sample_weight=wp)
        pa_en = np.clip(en.predict(Xa_ev_s), -0.25, 0.25).reshape(-1, len(SECTORS))
        pa_en -= pa_en.mean(axis=1, keepdims=True)

        # 2. Shallow CatBoost
        cb = CatBoostRegressor(
            iterations=200, learning_rate=0.03, depth=2, l2_leaf_reg=30.0,
            subsample=0.7, rsm=0.6, random_seed=42, verbose=0,
        )
        cb.fit(Xa_tr, ya, sample_weight=wp)
        pa_cb = cb.predict(Xa_ev).reshape(-1, len(SECTORS))
        pa_cb -= pa_cb.mean(axis=1, keepdims=True)

        # 3. Shallow Huber LightGBM
        lgbm = lgb.LGBMRegressor(
            objective="huber", n_estimators=120, learning_rate=0.03,
            max_depth=2, num_leaves=4, min_child_samples=60,
            subsample=0.7, subsample_freq=1, colsample_bytree=0.6,
            reg_alpha=3.0, reg_lambda=20.0, random_state=42, verbose=-1,
        )
        lgbm.fit(Xa_tr, ya, sample_weight=wp)
        pa_lgb = lgbm.predict(Xa_ev).reshape(-1, len(SECTORS))
        pa_lgb -= pa_lgb.mean(axis=1, keepdims=True)

        return (0.50 * pa_en + 0.30 * pa_cb + 0.20 * pa_lgb).astype(np.float32)

    def fit_predict(
        self,
        macro_v9: pd.DataFrame,
        sec_dfs: Dict[str, pd.DataFrame],
        y_all: np.ndarray,
        dates: pd.Series,
        realized_dates: pd.Series,
        train_idx: np.ndarray,
        eval_idx: np.ndarray,
        return_diagnostics: bool = False,
    ):
        mac_raw = macro_v9[MARKET_V9_COLS].to_numpy(dtype=np.float32)
        sec_raw = np.stack([sec_dfs[s][ALPHA_SEC_CORE_COLS].to_numpy(dtype=np.float32) for s in SECTORS], axis=1)

        itr, ival = split_inner_train_val(train_idx, dates, realized_dates, val_fraction=0.20)
        sec_prior = y_all[train_idx].mean(axis=0).astype(np.float32)
        mp = float(sec_prior.mean())
        sp_i = y_all[itr].mean(axis=0).astype(np.float32)

        # Causal CAPM sector betas fit strictly on training targets
        betas_i = compute_causal_sector_betas(y_all[itr], vasicek_shrink=self.vasicek_shrink)
        betas_f = compute_causal_sector_betas(y_all[train_idx], vasicek_shrink=self.vasicek_shrink)

        # 1. 4-Family Uncertainty-Gated Market Engine
        tf_va, tf_ev, gru_va, gru_ev, tcn_va, tcn_ev = self._fit_neural_market_families(
            mac_raw, sec_raw, y_all, itr, ival, eval_idx, sec_prior
        )
        cb_va = self._fit_cb_market(mac_raw, y_all, itr, ival, float(sp_i.mean()))
        cb_ev = self._fit_cb_market(mac_raw, y_all, train_idx, eval_idx, mp)

        w1, w2, w3, w4 = self.market_weights
        pm_va = w1 * tf_va + w2 * gru_va + w3 * tcn_va + w4 * cb_va
        pm_ev = w1 * tf_ev + w2 * gru_ev + w3 * tcn_ev + w4 * cb_ev

        sig_va = np.std(np.stack([tf_va, gru_va, tcn_va, cb_va], axis=0), axis=0)
        sig_ev = np.std(np.stack([tf_ev, gru_ev, tcn_ev, cb_ev], axis=0), axis=0)
        sig_ref = float(np.median(sig_va)) + 1e-4

        gate_va = 1.0 / (1.0 + self.kappa_uncertainty * (sig_va / sig_ref) ** 2)
        gate_ev = 1.0 / (1.0 + self.kappa_uncertainty * (sig_ev / sig_ref) ** 2)
        pm_va_g = pm_va * gate_va
        pm_ev_g = pm_ev * gate_ev

        ym_va = (y_all[ival].mean(axis=1) - mp).astype(np.float32)
        w_m = _eb_shrinkage(pm_va_g, ym_va, prior_strength=self.eb_prior_mkt, cap=self.cap_mkt)

        # 2. Beta-Neutral Cross-Sectional Alpha Engine
        pa_va = self._fit_beta_neutral_alpha(macro_v9, sec_dfs, y_all, itr, ival, sp_i, betas_i)
        ya_va = (y_all[ival] - sp_i[None, :]) - (y_all[ival].mean(axis=1) - float(sp_i.mean()))[:, None] * betas_i[None, :]
        w_a = _eb_shrinkage(pa_va, ya_va, prior_strength=self.eb_prior_alpha, cap=self.cap_alpha)
        pa_ev = self._fit_beta_neutral_alpha(macro_v9, sec_dfs, y_all, train_idx, eval_idx, sec_prior, betas_f)

        # Beta-scaled reconstruction!
        preds = (
            sec_prior[None, :]
            + w_m * pm_ev_g[:, None] * betas_f[None, :]
            + w_a * pa_ev
        ).astype(np.float32)

        if return_diagnostics:
            return preds, {
                "w_m": w_m,
                "w_a": w_a,
                "mean_gate": float(np.mean(gate_ev)),
                "betas": np.round(betas_f, 2).tolist(),
            }
        return preds


def run_iteration_9_audit(out_dir: str = "outputs/iteration9_audit"):
    import json
    from scipy.stats import spearmanr
    from sklearn.metrics import mean_squared_error
    from .generalization_audit import diebold_mariano_nw
    from .plots import plot_sector_predictions_grid
    from .splits import get_sealed_test_folds

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    df = load_raw_merged_df("data")
    y_df, realized = compute_forward_targets(df, horizon=HORIZON_DAYS)
    y_all = y_df.to_numpy(dtype=np.float32)
    macro_df, sec_dfs, feat_valid = build_all_feature_views(df)
    macro_v9 = augment_macro_features_v9(df, macro_df)

    v9_valid = feat_valid & np.isfinite(macro_v9[MARKET_V9_COLS].to_numpy()).all(axis=1)
    valid_mask = v9_valid & (~y_df.isna().any(axis=1)).to_numpy(copy=True)
    valid_mask[:500] = False

    dev_folds = get_dev_walk_forward_folds(df["Date"], realized, valid_mask)
    test_frozen = get_sealed_test_folds(df["Date"], realized, valid_mask, strictly_frozen_pre2020=True)
    test_expand = get_sealed_test_folds(df["Date"], realized, valid_mask, strictly_frozen_pre2020=False)

    mdl = BetaScaledUncertaintyEnsembleV9(seeds=(42, 102, 202), epochs=35)

    def eval_suite(label: str, folds):
        print("\n" + "=" * 100, flush=True)
        print(f"ITERATION 9 AUDIT: {label}", flush=True)
        print("=" * 100, flush=True)
        rows_n, rows_v9 = [], []
        yt_all, pn_all, p9_all, dt_all = [], [], [], []
        for f in folds:
            validate_no_leakage(df["Date"], realized, f.train_idx, f.eval_idx)
            ym = y_all[f.train_idx].mean(axis=0)
            p_n = np.tile(ym, (len(f.eval_idx), 1))
            p9, diag = mdl.fit_predict(
                macro_v9, sec_dfs, y_all, df["Date"], realized, f.train_idx, f.eval_idx, return_diagnostics=True
            )
            mn = evaluate_predictions(y_all[f.eval_idx], p_n, y_train_mean=ym); mn["fold"] = f.name
            m9 = evaluate_predictions(y_all[f.eval_idx], p9, y_train_mean=ym); m9["fold"] = f.name
            rows_n.append(mn); rows_v9.append(m9)
            print(
                f"  {f.name:28s} (w_m={diag['w_m']:.2f}, w_a={diag['w_a']:.2f}, gate={diag['mean_gate']:.2f}, betas={diag['betas']}) | "
                f"oos_r2={m9['oos_r2_vs_train_mean']:+.4f}, rel_r2={m9['rel_r2']:+.4f}, rmse={m9['rmse']:.4f} (naive {mn['rmse']:.4f}), "
                f"ts_ic={m9['mean_ts_ic']:+.4f}, cs_ic={m9['mean_cs_ic']:+.4f}, top2_exc={m9['top2_excess_1y_ret']:+.4f}",
                flush=True,
            )
            yt_all.append(y_all[f.eval_idx]); pn_all.append(p_n); p9_all.append(p9); dt_all.append(df["Date"].iloc[f.eval_idx])

        cols = [
            "fold", "mean_r2", "oos_r2_vs_train_mean", "rel_r2",
            "rmse", "dir_acc", "mean_ts_ic", "mean_cs_ic",
            "top2_excess_1y_ret", "top2_vs_bot2_spread", "top2_win_rate_vs_ew",
        ]
        print(f"\n--- {label} :: NaiveMean ---", flush=True)
        print(summarize_fold_metrics(rows_n)[cols].to_string(index=False, float_format=lambda x: f"{x:7.4f}"), flush=True)
        print(f"\n--- {label} :: BetaScaledUncertaintyEnsembleV9 (Iter 9) ---", flush=True)
        print(summarize_fold_metrics(rows_v9)[cols].to_string(index=False, float_format=lambda x: f"{x:7.4f}"), flush=True)

        yt = np.concatenate(yt_all, axis=0)
        pn = np.concatenate(pn_all, axis=0)
        p9 = np.concatenate(p9_all, axis=0)
        dm = diebold_mariano_nw(yt - p9, yt - pn, step=21, max_lag=12)
        wins = sum(1 for r in rows_v9 if r["oos_r2_vs_train_mean"] > 0)
        print(
            f"\n  STATISTICAL GENERALIZATION CHECK ({label}):\n"
            f"    Regimes with positive OOS R² vs Naive: {wins}/{len(rows_v9)}\n"
            f"    Diebold-Mariano HAC (lag=12m) vs Naive: DM={dm['dm_hac_stat']:+.3f}, one-sided p={dm['p_value_one_sided']:.4f} (n_months={dm['n_monthly_obs']})",
            flush=True,
        )

        sec_rows = []
        for j, s in enumerate(SECTORS):
            sse_m = np.sum((yt[:, j] - p9[:, j]) ** 2)
            sse_b = np.sum((yt[:, j] - pn[:, j]) ** 2)
            sec_rows.append({
                "sector": s,
                "oos_r2_vs_naive": 1.0 - sse_m / (sse_b + 1e-12),
                "rmse_v9": float(np.sqrt(mean_squared_error(yt[:, j], p9[:, j]))),
                "rmse_naive": float(np.sqrt(mean_squared_error(yt[:, j], pn[:, j]))),
                "ts_ic": float(np.nan_to_num(spearmanr(yt[:, j], p9[:, j])[0])),
            })
        print("\n  Per-Sector Pooled OOS Generalization (Iter 9 vs Naive):", flush=True)
        print(pd.DataFrame(sec_rows).to_string(index=False, float_format=lambda x: f"{x:7.4f}"), flush=True)

        plot_sector_predictions_grid(
            pd.concat(dt_all), yt, p9,
            f"{label}: BetaScaledUncertaintyEnsembleV9 (Iter 9) Out-of-Sample 1Y Sector Forecasts",
            out_path / f"{label.lower()}_v9_forecasts.png",
        )
        return {"dm": dm, "wins": f"{wins}/{len(rows_v9)}", "per_sector": sec_rows, "folds_v9": rows_v9, "folds_naive": rows_n}

    res_dev = eval_suite("DEV_2012_2019", dev_folds)
    res_frozen = eval_suite("ALL_12YR_2012_2024_FROZEN_PRE2020_TEST", dev_folds + test_frozen)
    res_expand = eval_suite("ALL_12YR_2012_2024_EXPANDING_PURGED", dev_folds + test_expand)
    (out_path / "iteration9_audit.json").write_text(
        json.dumps({"dev": res_dev, "frozen": res_frozen, "expand": res_expand}, indent=2, default=str)
    )
    print(f"\nSaved Iteration 9 audit to {out_path.resolve()}", flush=True)


if __name__ == "__main__":
    run_iteration_9_audit()
