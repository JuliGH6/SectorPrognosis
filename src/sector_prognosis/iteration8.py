"""Iteration 8: Multi-Family Temporal + Tree Ensemble with Epistemic Uncertainty Gating
and Causal Commodity/Overextension Regime Features.

Developed and validated strictly on Purged Dev Folds (2012-2019) before any Sealed Test unlock.

New techniques introduced:
1. Auxiliary causal market-regime signals already present in MergedDataSectorPrognosis.xlsx:
   - Energy index 6M/1Y return & drawdown (oil/commodity shock detector)
   - Energy - Utilities 1Y spread (market-implied inflation/rate-regime proxy)
   - 2-Year (500d) equal-weight market cumulative return (multi-year overextension)
   - Strictly causal expanding valuation z-score (expensive vs cheap market regime)
2. Diverse 3-family Market Timing Ensemble:
   - TempSectorFormerBackbone (Self-Attention Transformer)
   - CausalTemporalGRU (Recurrent/State-Space inductive bias with attention pooling)
   - Shallow CatBoost (Non-extrapolative oblivious trees)
3. Epistemic Uncertainty Gating:
   - Dynamically damps market residual predictions when cross-family disagreement spikes.
"""

from pathlib import Path
from typing import Dict, Tuple

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
from .metrics import evaluate_predictions, summarize_fold_metrics
from .models_temporal import SEQ_OFFSETS, _eb_shrinkage
from .models_v2 import (
    ALPHA_SEC_CORE_COLS,
    MARKET_CORE_COLS,
    _recency_weights,
    _winsorize_train_target,
    extract_curated_alpha_panel,
)
from .splits import get_dev_walk_forward_folds, split_inner_train_val, validate_no_leakage

EXTRA_MARKET_COLS = [
    "aux_energy_ret_126d",
    "aux_energy_ret_250d",
    "aux_energy_dd_250d",
    "aux_energy_minus_util_250d",
    "mkt_val_expanding_z",
]

MARKET_V8_COLS = MARKET_CORE_COLS + EXTRA_MARKET_COLS


def load_aux_prices(df_dates: pd.Series, data_dir: str | Path = "data") -> pd.DataFrame:
    """Load Energy and Communications price columns aligned strictly to df_dates (cached)."""
    data_dir = Path(data_dir)
    cache = data_dir / "aux_prices.parquet"
    if cache.exists():
        aux = pd.read_parquet(cache)
        aux["Date"] = pd.to_datetime(aux["Date"])
    else:
        raw = pd.read_excel(data_dir / "MergedDataSectorPrognosis.xlsx", engine="calamine")
        aux = raw[["Date", "Energy", "Communications"]].copy()
        aux["Date"] = pd.to_datetime(aux["Date"])
        aux = aux.sort_values("Date").reset_index(drop=True)
        aux.to_parquet(cache, index=False)

    aligned = pd.merge(pd.DataFrame({"Date": df_dates}), aux, on="Date", how="left")
    assert len(aligned) == len(df_dates) and not aligned[["Energy", "Communications"]].isna().any().any()
    return aligned


def augment_macro_features_v8(df: pd.DataFrame, macro_df: pd.DataFrame, data_dir: str | Path = "data") -> pd.DataFrame:
    """Add strictly causal commodity-shock, 2Y market-overextension, and expanding-valuation features."""
    aux = load_aux_prices(df["Date"], data_dir=data_dir)
    out = macro_df.copy()

    e = aux["Energy"].astype(np.float64)
    u = df["Utilities"].astype(np.float64)
    out["aux_energy_ret_126d"] = e.pct_change(126).astype(np.float32)
    out["aux_energy_ret_250d"] = e.pct_change(250).astype(np.float32)
    out["aux_energy_dd_250d"] = (e / e.rolling(250, min_periods=250).max() - 1.0).astype(np.float32)
    out["aux_energy_minus_util_250d"] = (e.pct_change(250) - u.pct_change(250)).astype(np.float32)

    # 2-year (500d) equal-weight market cumulative return (strictly causal)
    ew_ret_500 = df[SECTORS].pct_change(500).mean(axis=1)
    out["mkt_ew_ret_500d"] = ew_ret_500.astype(np.float32)

    # Strictly causal expanding valuation z-score (min_periods=250)
    v = out["mkt_avg_valuation"].astype(np.float64)
    exp_mean = v.expanding(min_periods=250).mean()
    exp_std = v.expanding(min_periods=250).std() + 1e-6
    out["mkt_val_expanding_z"] = ((v - exp_mean) / exp_std).clip(-4.0, 4.0).astype(np.float32)

    return out


class CausalTemporalGRU(nn.Module):
    """Strictly causal GRU + Temporal Attention Pooling for Market-Level 1Y Timing."""

    def __init__(self, f_mac: int, hidden_dim: int = 48, dropout: float = 0.35):
        super().__init__()
        self.in_proj = nn.Sequential(nn.Linear(f_mac, hidden_dim), nn.LayerNorm(hidden_dim), nn.SiLU())
        self.gru = nn.GRU(hidden_dim, hidden_dim, num_layers=2, batch_first=True, dropout=dropout)
        self.attn_score = nn.Linear(hidden_dim, 1)
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        nn.init.normal_(self.head[-1].weight, std=0.01)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, x_mac_seq: torch.Tensor) -> torch.Tensor:
        if self.training:
            x_mac_seq = x_mac_seq + 0.04 * torch.randn_like(x_mac_seq)
        h = self.in_proj(x_mac_seq)
        out, _ = self.gru(h)  # (B, T, H)
        w = F.softmax(self.attn_score(out), dim=1)  # (B, T, 1)
        ctx = (out * w).sum(dim=1)  # (B, H)
        pooled = torch.cat([out[:, -1, :], ctx], dim=-1)
        return 0.20 * torch.tanh(self.head(pooled).squeeze(-1))


class UncertaintyGatedEnsembleV8:
    """Iteration 8: 3-Family Market Ensemble (TempSectorFormer + CausalGRU + CatBoost)
    with Epistemic Uncertainty Gating + Sparse ElasticNet/CatBoost Cross-Sectional Alpha."""

    def __init__(
        self,
        seeds: Tuple[int, ...] = (42, 102, 202),
        epochs: int = 35,
        kappa_uncertainty: float = 0.35,
        eb_prior_mkt: float = 0.35,
        eb_prior_alpha: float = 0.45,
        cap_mkt: float = 0.70,
        cap_alpha: float = 0.70,
        market_weights: Tuple[float, float, float] = (0.45, 0.30, 0.25),  # TF, GRU, CatBoost
        device: str | None = None,
    ):
        self.seeds = seeds
        self.epochs = epochs
        self.kappa_uncertainty = kappa_uncertainty
        self.eb_prior_mkt = eb_prior_mkt
        self.eb_prior_alpha = eb_prior_alpha
        self.cap_mkt = cap_mkt
        self.cap_alpha = cap_alpha
        self.market_weights = market_weights
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))

    def _fit_tf_and_gru(
        self,
        mac_raw: np.ndarray,
        sec_raw: np.ndarray,
        y_all: np.ndarray,
        itr: np.ndarray,
        ival: np.ndarray,
        eval_idx: np.ndarray,
        sec_prior: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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

        tf_va, tf_ev, gru_va, gru_ev = [], [], [], []
        N, bs = len(txs_tr), 128

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
            torch.manual_seed(seed + 777)
            np.random.seed(seed + 777)
            m_gru = CausalTemporalGRU(mac_raw.shape[-1], hidden_dim=48, dropout=0.35).to(self.device)
            opt_g = torch.optim.AdamW(m_gru.parameters(), lr=2e-3, weight_decay=2.5e-2)
            sched_g = torch.optim.lr_scheduler.CosineAnnealingLR(opt_g, T_max=self.epochs)
            best_lg, best_stg, bad_g = float("inf"), None, 0
            for _ in range(self.epochs):
                m_gru.train()
                perm = torch.randperm(N, device=self.device)
                for i in range(0, N, bs):
                    idx = perm[i : i + bs]
                    if len(idx) < 4:
                        continue
                    opt_g.zero_grad()
                    loss_g = F.huber_loss(m_gru(txm_tr[idx]), tym_tr[idx], delta=0.12)
                    loss_g.backward()
                    nn.utils.clip_grad_norm_(m_gru.parameters(), 1.0)
                    opt_g.step()
                sched_g.step()
                m_gru.eval()
                with torch.no_grad():
                    vlg = F.huber_loss(m_gru(txm_va), tym_va, delta=0.12).item()
                if vlg < best_lg - 1e-5:
                    best_lg, best_stg, bad_g = vlg, {k: v.detach().cpu().clone() for k, v in m_gru.state_dict().items()}, 0
                else:
                    bad_g += 1
                    if bad_g >= 10:
                        break
            if best_stg is not None:
                m_gru.load_state_dict({k: v.to(self.device) for k, v in best_stg.items()})
            m_gru.eval()
            with torch.no_grad():
                gru_va.append(m_gru(txm_va).cpu().numpy())
                gru_ev.append(m_gru(txm_ev).cpu().numpy())

        return (
            np.mean(tf_va, axis=0).astype(np.float32),
            np.mean(tf_ev, axis=0).astype(np.float32),
            np.mean(gru_va, axis=0).astype(np.float32),
            np.mean(gru_ev, axis=0).astype(np.float32),
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

    def _fit_alpha_specialist(self, macro_df, sec_dfs, y_all, tr_idx, ev_idx) -> np.ndarray:
        tr_sub = tr_idx[::5]
        sp = y_all[tr_idx].mean(axis=0).astype(np.float32)
        ap = sp - float(sp.mean())
        wd = _recency_weights(len(tr_sub), half_life_steps=10.0 * 50.0)
        wp = np.repeat(wd, len(SECTORS))
        ya = _winsorize_train_target(
            ((y_all[tr_sub] - y_all[tr_sub].mean(axis=1, keepdims=True)) - ap[None, :]).reshape(-1), 0.04, 0.96
        )
        Xa_tr, _ = extract_curated_alpha_panel(macro_df, sec_dfs, tr_sub)
        Xa_ev, _ = extract_curated_alpha_panel(macro_df, sec_dfs, ev_idx)

        sc = RobustScaler()
        Xa_tr_s = np.clip(sc.fit_transform(Xa_tr), -3.0, 3.0)
        Xa_ev_s = np.clip(sc.transform(Xa_ev), -3.0, 3.0)
        en = ElasticNet(alpha=0.012, l1_ratio=0.55, max_iter=2000, random_state=42)
        en.fit(Xa_tr_s, ya, sample_weight=wp)
        pa_en = np.clip(en.predict(Xa_ev_s), -0.25, 0.25).reshape(-1, len(SECTORS))
        pa_en -= pa_en.mean(axis=1, keepdims=True)

        cb = CatBoostRegressor(
            iterations=200, learning_rate=0.03, depth=2, l2_leaf_reg=30.0,
            subsample=0.7, rsm=0.6, random_seed=42, verbose=0,
        )
        cb.fit(Xa_tr, ya, sample_weight=wp)
        pa_cb = cb.predict(Xa_ev).reshape(-1, len(SECTORS))
        pa_cb -= pa_cb.mean(axis=1, keepdims=True)

        return (0.65 * pa_en + 0.35 * pa_cb).astype(np.float32)

    def fit_predict(
        self,
        macro_v8: pd.DataFrame,
        sec_dfs: Dict[str, pd.DataFrame],
        y_all: np.ndarray,
        dates: pd.Series,
        realized_dates: pd.Series,
        train_idx: np.ndarray,
        eval_idx: np.ndarray,
        return_diagnostics: bool = False,
    ):
        mac_raw = macro_v8[MARKET_V8_COLS].to_numpy(dtype=np.float32)
        sec_raw = np.stack([sec_dfs[s][ALPHA_SEC_CORE_COLS].to_numpy(dtype=np.float32) for s in SECTORS], axis=1)

        itr, ival = split_inner_train_val(train_idx, dates, realized_dates, val_fraction=0.20)
        sec_prior = y_all[train_idx].mean(axis=0).astype(np.float32)
        mp = float(sec_prior.mean())
        sp_i = y_all[itr].mean(axis=0).astype(np.float32)

        # 1. Three diverse market families: TempSectorFormer + CausalGRU + CatBoost
        tf_va, tf_ev, gru_va, gru_ev = self._fit_tf_and_gru(mac_raw, sec_raw, y_all, itr, ival, eval_idx, sec_prior)
        cb_va = self._fit_cb_market(mac_raw, y_all, itr, ival, float(sp_i.mean()))
        cb_ev = self._fit_cb_market(mac_raw, y_all, train_idx, eval_idx, mp)

        w1, w2, w3 = self.market_weights
        pm_va = w1 * tf_va + w2 * gru_va + w3 * cb_va
        pm_ev = w1 * tf_ev + w2 * gru_ev + w3 * cb_ev

        # Epistemic disagreement across the 3 diverse model families
        sig_va = np.std(np.stack([tf_va, gru_va, cb_va], axis=0), axis=0)
        sig_ev = np.std(np.stack([tf_ev, gru_ev, cb_ev], axis=0), axis=0)
        sig_ref = float(np.median(sig_va)) + 1e-4

        if self.kappa_uncertainty > 0:
            gate_va = 1.0 / (1.0 + self.kappa_uncertainty * (sig_va / sig_ref) ** 2)
            gate_ev = 1.0 / (1.0 + self.kappa_uncertainty * (sig_ev / sig_ref) ** 2)
            pm_va_g = pm_va * gate_va
            pm_ev_g = pm_ev * gate_ev
        else:
            pm_va_g, pm_ev_g = pm_va, pm_ev

        ym_va = (y_all[ival].mean(axis=1) - mp).astype(np.float32)
        w_m = _eb_shrinkage(pm_va_g, ym_va, prior_strength=self.eb_prior_mkt, cap=self.cap_mkt)

        # 2. Cross-Sectional Alpha Specialist
        pa_va = self._fit_alpha_specialist(macro_v8, sec_dfs, y_all, itr, ival)
        ya_va = (y_all[ival] - y_all[ival].mean(axis=1, keepdims=True)) - (sp_i - float(sp_i.mean()))[None, :]
        w_a = _eb_shrinkage(pa_va, ya_va, prior_strength=self.eb_prior_alpha, cap=self.cap_alpha)
        pa_ev = self._fit_alpha_specialist(macro_v8, sec_dfs, y_all, train_idx, eval_idx)

        preds = (sec_prior[None, :] + w_m * pm_ev_g[:, None] + w_a * pa_ev).astype(np.float32)
        if return_diagnostics:
            return preds, {
                "w_m": w_m,
                "w_a": w_a,
                "mean_gate": float(np.mean(gate_ev)) if self.kappa_uncertainty > 0 else 1.0,
            }
        return preds


def run_iteration_8_audit(out_dir: str = "outputs/iteration8_audit"):
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
    macro_v8 = augment_macro_features_v8(df, macro_df)

    v8_valid = feat_valid & np.isfinite(macro_v8[MARKET_V8_COLS].to_numpy()).all(axis=1)
    valid_mask = v8_valid & (~y_df.isna().any(axis=1)).to_numpy(copy=True)
    valid_mask[:500] = False  # 250d feature warmup + 231d sequence lookback

    dev_folds = get_dev_walk_forward_folds(df["Date"], realized, valid_mask)
    test_frozen = get_sealed_test_folds(df["Date"], realized, valid_mask, strictly_frozen_pre2020=True)
    test_expand = get_sealed_test_folds(df["Date"], realized, valid_mask, strictly_frozen_pre2020=False)

    mdl = UncertaintyGatedEnsembleV8(
        seeds=(42, 102, 202),
        epochs=35,
        kappa_uncertainty=0.35,
        market_weights=(0.50, 0.25, 0.25),
    )

    def eval_suite(label: str, folds):
        print("\n" + "=" * 100, flush=True)
        print(f"ITERATION 8B AUDIT: {label}", flush=True)
        print("=" * 100, flush=True)
        rows_n, rows_v8 = [], []
        yt_all, pn_all, p8_all, dt_all = [], [], [], []
        for f in folds:
            validate_no_leakage(df["Date"], realized, f.train_idx, f.eval_idx)
            ym = y_all[f.train_idx].mean(axis=0)
            p_n = np.tile(ym, (len(f.eval_idx), 1))
            p8, diag = mdl.fit_predict(
                macro_v8, sec_dfs, y_all, df["Date"], realized, f.train_idx, f.eval_idx, return_diagnostics=True
            )
            mn = evaluate_predictions(y_all[f.eval_idx], p_n, y_train_mean=ym); mn["fold"] = f.name
            m8 = evaluate_predictions(y_all[f.eval_idx], p8, y_train_mean=ym); m8["fold"] = f.name
            rows_n.append(mn); rows_v8.append(m8)
            print(
                f"  {f.name:28s} (w_m={diag['w_m']:.2f}, w_a={diag['w_a']:.2f}, gate={diag['mean_gate']:.2f}) | "
                f"oos_r2={m8['oos_r2_vs_train_mean']:+.4f}, rmse={m8['rmse']:.4f} (naive {mn['rmse']:.4f}), "
                f"ts_ic={m8['mean_ts_ic']:+.4f}, cs_ic={m8['mean_cs_ic']:+.4f}, top2_exc={m8['top2_excess_1y_ret']:+.4f}",
                flush=True,
            )
            yt_all.append(y_all[f.eval_idx]); pn_all.append(p_n); p8_all.append(p8); dt_all.append(df["Date"].iloc[f.eval_idx])

        cols = [
            "fold", "mean_r2", "oos_r2_vs_train_mean", "rel_r2",
            "rmse", "dir_acc", "mean_ts_ic", "mean_cs_ic",
            "top2_excess_1y_ret", "top2_vs_bot2_spread", "top2_win_rate_vs_ew",
        ]
        print(f"\n--- {label} :: NaiveMean ---", flush=True)
        print(summarize_fold_metrics(rows_n)[cols].to_string(index=False, float_format=lambda x: f"{x:7.4f}"), flush=True)
        print(f"\n--- {label} :: UncertaintyGatedEnsembleV8 (Iter 8B) ---", flush=True)
        print(summarize_fold_metrics(rows_v8)[cols].to_string(index=False, float_format=lambda x: f"{x:7.4f}"), flush=True)

        yt = np.concatenate(yt_all, axis=0)
        pn = np.concatenate(pn_all, axis=0)
        p8 = np.concatenate(p8_all, axis=0)
        dm = diebold_mariano_nw(yt - p8, yt - pn, step=21, max_lag=12)
        wins = sum(1 for r in rows_v8 if r["oos_r2_vs_train_mean"] > 0)
        print(
            f"\n  STATISTICAL GENERALIZATION CHECK ({label}):\n"
            f"    Regimes with positive OOS R² vs Naive: {wins}/{len(rows_v8)}\n"
            f"    Diebold-Mariano HAC (lag=12m) vs Naive: DM={dm['dm_hac_stat']:+.3f}, one-sided p={dm['p_value_one_sided']:.4f} (n_months={dm['n_monthly_obs']})",
            flush=True,
        )

        sec_rows = []
        for j, s in enumerate(SECTORS):
            sse_m = np.sum((yt[:, j] - p8[:, j]) ** 2)
            sse_b = np.sum((yt[:, j] - pn[:, j]) ** 2)
            sec_rows.append({
                "sector": s,
                "oos_r2_vs_naive": 1.0 - sse_m / (sse_b + 1e-12),
                "rmse_v8": float(np.sqrt(mean_squared_error(yt[:, j], p8[:, j]))),
                "rmse_naive": float(np.sqrt(mean_squared_error(yt[:, j], pn[:, j]))),
                "ts_ic": float(np.nan_to_num(spearmanr(yt[:, j], p8[:, j])[0])),
            })
        print("\n  Per-Sector Pooled OOS Generalization (Iter 8B vs Naive):", flush=True)
        print(pd.DataFrame(sec_rows).to_string(index=False, float_format=lambda x: f"{x:7.4f}"), flush=True)

        plot_sector_predictions_grid(
            pd.concat(dt_all), yt, p8,
            f"{label}: UncertaintyGatedEnsembleV8 (Iter 8B) Out-of-Sample 1Y Sector Forecasts",
            out_path / f"{label.lower()}_v8_forecasts.png",
        )
        return {"dm": dm, "wins": f"{wins}/{len(rows_v8)}", "per_sector": sec_rows, "folds_v8": rows_v8, "folds_naive": rows_n}

    res_dev = eval_suite("DEV_2012_2019", dev_folds)
    res_frozen = eval_suite("ALL_12YR_2012_2024_FROZEN_PRE2020_TEST", dev_folds + test_frozen)
    res_expand = eval_suite("ALL_12YR_2012_2024_EXPANDING_PURGED", dev_folds + test_expand)
    (out_path / "iteration8_audit.json").write_text(
        json.dumps({"dev": res_dev, "frozen": res_frozen, "expand": res_expand}, indent=2, default=str)
    )
    print(f"\nSaved Iteration 8B audit to {out_path.resolve()}", flush=True)


if __name__ == "__main__":
    run_iteration_8_audit()
