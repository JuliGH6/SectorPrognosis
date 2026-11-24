"""Iteration 7 Champion: HybridTempSectorFormer + 12-Year Statistical Generalization Audit.

Combines the two empirically validated best-in-class specialists from Iterations 1-6:
1. Market Timing Specialist: TempSectorFormer (12-month causal temporal + cross-sector
   Transformer with purged early stopping & Empirical-Bayes shrinkage).
2. Cross-Sectional Alpha Specialist: Sparse ElasticNet (L1/L2) + shallow CatBoost
   with Empirical-Bayes shrinkage around the expanding sector prior.

Evaluates true generalization across ALL 6 non-overlapping 2-year regimes (2012-2024)
with Diebold-Mariano Newey-West HAC (lag=12 months) significance testing.
"""

import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from catboost import CatBoostRegressor
from scipy.stats import norm, spearmanr
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import RobustScaler

from . import HORIZON_DAYS, SECTORS
from .data import compute_forward_targets, load_raw_merged_df
from .features import build_all_feature_views
from .generalization_audit import diebold_mariano_nw
from .metrics import evaluate_predictions, summarize_fold_metrics
from .models_temporal import SEQ_OFFSETS, _eb_shrinkage
from .models_v2 import (
    ALPHA_SEC_CORE_COLS,
    MARKET_CORE_COLS,
    _recency_weights,
    _winsorize_train_target,
    extract_curated_alpha_panel,
)
from .plots import plot_sector_predictions_grid
from .splits import (
    get_dev_walk_forward_folds,
    get_sealed_test_folds,
    split_inner_train_val,
    validate_no_leakage,
)


class TempSectorFormerBackbone(nn.Module):
    """Factorized Temporal + Cross-Sector Attention Transformer for Market + Sector Context."""

    def __init__(self, f_sec: int, f_mac: int, d_model: int = 48, n_heads: int = 4, dropout: float = 0.35):
        super().__init__()
        self.sec_proj = nn.Sequential(nn.Linear(f_sec, d_model), nn.LayerNorm(d_model), nn.SiLU())
        self.mac_proj = nn.Sequential(nn.Linear(f_mac, d_model), nn.LayerNorm(d_model), nn.SiLU())
        self.sec_emb = nn.Embedding(6, d_model)
        self.pos_emb = nn.Parameter(torch.randn(1, len(SEQ_OFFSETS), d_model) * 0.02)

        enc_t = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 2,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True,
        )
        self.temp_enc = nn.TransformerEncoder(enc_t, num_layers=1, enable_nested_tensor=False)

        enc_s = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 2,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True,
        )
        self.sec_enc = nn.TransformerEncoder(enc_s, num_layers=1, enable_nested_tensor=False)

        self.alpha_head = nn.Linear(d_model, 1)
        self.mkt_head = nn.Sequential(
            nn.Linear(d_model * 2, d_model // 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )
        nn.init.normal_(self.alpha_head.weight, std=0.01)
        nn.init.zeros_(self.alpha_head.bias)

    def forward(self, x_sec_seq: torch.Tensor, x_mac_seq: torch.Tensor):
        if self.training:
            x_sec_seq = x_sec_seq + 0.04 * torch.randn_like(x_sec_seq)
            x_mac_seq = x_mac_seq + 0.04 * torch.randn_like(x_mac_seq)

        B, T, S, _ = x_sec_seq.shape
        hm = self.mac_proj(x_mac_seq) + self.pos_emb[:, :T, :]
        hm_t = self.temp_enc(hm)[:, -1, :]

        hs = self.sec_proj(x_sec_seq).permute(0, 2, 1, 3).reshape(B * S, T, -1) + self.pos_emb[:, :T, :]
        hs_t = self.temp_enc(hs)[:, -1, :].reshape(B, S, -1)

        sec_ids = torch.arange(S, device=x_sec_seq.device).unsqueeze(0).expand(B, -1)
        hs_out = self.sec_enc(hs_t + hm_t.unsqueeze(1) + self.sec_emb(sec_ids))

        raw_a = self.alpha_head(hs_out).squeeze(-1)
        alpha_resid = 0.25 * torch.tanh(raw_a - raw_a.mean(dim=-1, keepdim=True))
        mkt_resid = 0.20 * torch.tanh(self.mkt_head(torch.cat([hm_t, hs_out.mean(dim=1)], dim=-1)).squeeze(-1))
        return mkt_resid, alpha_resid


class HybridTempSectorFormer:
    """Iteration 7 Champion: TempSectorFormer Market Timing + ElasticNet/CatBoost Alpha."""

    def __init__(
        self,
        seeds: Tuple[int, ...] = (42, 102, 202),
        epochs: int = 35,
        eb_prior_mkt: float = 0.35,
        eb_prior_alpha: float = 0.45,
        cap_mkt: float = 0.70,
        cap_alpha: float = 0.70,
        alpha_enet_weight: float = 0.65,
        device: str | None = None,
    ):
        self.seeds = seeds
        self.epochs = epochs
        self.eb_prior_mkt = eb_prior_mkt
        self.eb_prior_alpha = eb_prior_alpha
        self.cap_mkt = cap_mkt
        self.cap_alpha = cap_alpha
        self.alpha_enet_weight = alpha_enet_weight
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))

    def _fit_predict_market_tf(
        self,
        mac_raw: np.ndarray,
        sec_raw: np.ndarray,
        y_all: np.ndarray,
        itr: np.ndarray,
        ival: np.ndarray,
        eval_idx: np.ndarray,
        sec_prior: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
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

        pm_va_list, pm_ev_list = [], []
        N, bs = len(txs_tr), 128

        for seed in self.seeds:
            torch.manual_seed(seed)
            np.random.seed(seed)
            model = TempSectorFormerBackbone(sec_raw.shape[-1], mac_raw.shape[-1], d_model=48, n_heads=4, dropout=0.35).to(self.device)
            opt = torch.optim.AdamW(model.parameters(), lr=1.5e-3, weight_decay=2e-2)
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.epochs)
            best_loss, best_state, bad = float("inf"), None, 0

            for _ in range(self.epochs):
                model.train()
                perm = torch.randperm(N, device=self.device)
                for i in range(0, N, bs):
                    idx = perm[i : i + bs]
                    if len(idx) < 4:
                        continue
                    opt.zero_grad()
                    pm, pa = model(txs_tr[idx], txm_tr[idx])
                    loss = (
                        F.huber_loss(pm, tym_tr[idx], delta=0.12)
                        + 1.5 * F.huber_loss(pa, tya_tr[idx], delta=0.10)
                        - 0.05 * F.cosine_similarity(pa, tya_tr[idx], dim=-1, eps=1e-6).mean()
                    )
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    opt.step()
                sched.step()

                model.eval()
                with torch.no_grad():
                    pm_v, pa_v = model(txs_va, txm_va)
                    vloss = (F.huber_loss(pm_v, tym_va, delta=0.12) + 1.5 * F.huber_loss(pa_v, tya_va, delta=0.10)).item()
                if vloss < best_loss - 1e-5:
                    best_loss = vloss
                    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                    bad = 0
                else:
                    bad += 1
                    if bad >= 10:
                        break

            if best_state is not None:
                model.load_state_dict({k: v.to(self.device) for k, v in best_state.items()})
            model.eval()
            with torch.no_grad():
                pm_v, _ = model(txs_va, txm_va)
                pm_e, _ = model(txs_ev, txm_ev)
                pm_va_list.append(pm_v.cpu().numpy())
                pm_ev_list.append(pm_e.cpu().numpy())

        return np.mean(pm_va_list, axis=0).astype(np.float32), np.mean(pm_ev_list, axis=0).astype(np.float32)

    def _fit_predict_alpha_specialist(
        self,
        macro_df: pd.DataFrame,
        sec_dfs: Dict[str, pd.DataFrame],
        y_all: np.ndarray,
        tr_idx: np.ndarray,
        ev_idx: np.ndarray,
    ) -> np.ndarray:
        tr_sub = tr_idx[::5]
        sp = y_all[tr_idx].mean(axis=0).astype(np.float32)
        ap = sp - float(sp.mean())
        wd = _recency_weights(len(tr_sub), half_life_steps=10.0 * 50.0)
        wp = np.repeat(wd, len(SECTORS))

        ya_2d = (y_all[tr_sub] - y_all[tr_sub].mean(axis=1, keepdims=True)) - ap[None, :]
        ya = _winsorize_train_target(ya_2d.reshape(-1), 0.04, 0.96)
        Xa_tr, _ = extract_curated_alpha_panel(macro_df, sec_dfs, tr_sub)
        Xa_ev, _ = extract_curated_alpha_panel(macro_df, sec_dfs, ev_idx)

        # ElasticNet (sparse factor model)
        sc = RobustScaler()
        Xa_tr_s = np.clip(sc.fit_transform(Xa_tr), -3.0, 3.0)
        Xa_ev_s = np.clip(sc.transform(Xa_ev), -3.0, 3.0)
        en = ElasticNet(alpha=0.012, l1_ratio=0.55, max_iter=2000, random_state=42)
        en.fit(Xa_tr_s, ya, sample_weight=wp)
        pa_en = np.clip(en.predict(Xa_ev_s), -0.25, 0.25).reshape(-1, len(SECTORS))
        pa_en -= pa_en.mean(axis=1, keepdims=True)

        # Shallow regularized CatBoost (non-linear interaction)
        cb = CatBoostRegressor(
            iterations=200, learning_rate=0.03, depth=2, l2_leaf_reg=30.0,
            subsample=0.7, rsm=0.6, random_seed=42, verbose=0,
        )
        cb.fit(Xa_tr, ya, sample_weight=wp)
        pa_cb = cb.predict(Xa_ev).reshape(-1, len(SECTORS))
        pa_cb -= pa_cb.mean(axis=1, keepdims=True)

        pa = self.alpha_enet_weight * pa_en + (1.0 - self.alpha_enet_weight) * pa_cb
        return pa.astype(np.float32)

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

        itr, ival = split_inner_train_val(train_idx, dates, realized_dates, val_fraction=0.20)
        sec_prior = y_all[train_idx].mean(axis=0).astype(np.float32)
        sp_i = y_all[itr].mean(axis=0).astype(np.float32)

        # 1. Market Timing via TempSectorFormer + Empirical-Bayes shrinkage
        pm_va, pm_ev = self._fit_predict_market_tf(mac_raw, sec_raw, y_all, itr, ival, eval_idx, sec_prior)
        ym_va = (y_all[ival].mean(axis=1) - float(sec_prior.mean())).astype(np.float32)
        w_m = _eb_shrinkage(pm_va, ym_va, prior_strength=self.eb_prior_mkt, cap=self.cap_mkt)

        # 2. Cross-Sectional Alpha via ElasticNet + CatBoost + Empirical-Bayes shrinkage
        pa_va = self._fit_predict_alpha_specialist(macro_df, sec_dfs, y_all, itr, ival)
        ya_va = (y_all[ival] - y_all[ival].mean(axis=1, keepdims=True)) - (sp_i - float(sp_i.mean()))[None, :]
        w_a = _eb_shrinkage(pa_va, ya_va, prior_strength=self.eb_prior_alpha, cap=self.cap_alpha)
        pa_ev = self._fit_predict_alpha_specialist(macro_df, sec_dfs, y_all, train_idx, eval_idx)

        preds = (sec_prior[None, :] + w_m * pm_ev[:, None] + w_a * pa_ev).astype(np.float32)
        if return_diagnostics:
            return preds, {"w_m": w_m, "w_a": w_a}
        return preds


def run_iteration_7_audit(out_dir: str = "outputs/champion_audit"):
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    df = load_raw_merged_df("data")
    y_df, realized = compute_forward_targets(df, horizon=HORIZON_DAYS)
    y_all = y_df.to_numpy(dtype=np.float32)
    macro_df, sec_dfs, feat_valid = build_all_feature_views(df)
    valid_mask = feat_valid & (~y_df.isna().any(axis=1)).to_numpy(copy=True)
    valid_mask[:500] = False

    dev_folds = get_dev_walk_forward_folds(df["Date"], realized, valid_mask)
    test_frozen = get_sealed_test_folds(df["Date"], realized, valid_mask, strictly_frozen_pre2020=True)
    test_expand = get_sealed_test_folds(df["Date"], realized, valid_mask, strictly_frozen_pre2020=False)

    champ = HybridTempSectorFormer(seeds=(42, 102, 202), epochs=35)

    def eval_suite(label: str, folds):
        print("\n" + "=" * 100, flush=True)
        print(f"ITERATION 7 GENERALIZATION AUDIT: {label}", flush=True)
        print("=" * 100, flush=True)
        rows_n, rows_c = [], []
        yt_all, pn_all, pc_all, dt_all = [], [], [], []
        for f in folds:
            validate_no_leakage(df["Date"], realized, f.train_idx, f.eval_idx)
            ym = y_all[f.train_idx].mean(axis=0)
            p_n = np.tile(ym, (len(f.eval_idx), 1))
            p_c, diag = champ.fit_predict(
                macro_df, sec_dfs, y_all, df["Date"], realized, f.train_idx, f.eval_idx, return_diagnostics=True
            )
            mn = evaluate_predictions(y_all[f.eval_idx], p_n, y_train_mean=ym); mn["fold"] = f.name
            mc = evaluate_predictions(y_all[f.eval_idx], p_c, y_train_mean=ym); mc["fold"] = f.name
            rows_n.append(mn); rows_c.append(mc)
            print(
                f"  {f.name:28s} (w_m={diag['w_m']:.2f}, w_a={diag['w_a']:.2f}) | "
                f"oos_r2={mc['oos_r2_vs_train_mean']:+.4f}, rmse={mc['rmse']:.4f} (naive {mn['rmse']:.4f}), "
                f"ts_ic={mc['mean_ts_ic']:+.4f}, cs_ic={mc['mean_cs_ic']:+.4f}, "
                f"top2_exc={mc['top2_excess_1y_ret']:+.4f} (naive {mn['top2_excess_1y_ret']:+.4f})",
                flush=True,
            )
            yt_all.append(y_all[f.eval_idx]); pn_all.append(p_n); pc_all.append(p_c); dt_all.append(df["Date"].iloc[f.eval_idx])

        cols = [
            "fold", "mean_r2", "oos_r2_vs_train_mean", "rel_r2",
            "rmse", "dir_acc", "mean_ts_ic", "mean_cs_ic",
            "top2_excess_1y_ret", "top2_vs_bot2_spread", "top2_win_rate_vs_ew",
        ]
        print(f"\n--- {label} :: NaiveMean ---", flush=True)
        print(summarize_fold_metrics(rows_n)[cols].to_string(index=False, float_format=lambda x: f"{x:7.4f}"), flush=True)
        print(f"\n--- {label} :: HybridTempSectorFormer (Iter 7 Champion) ---", flush=True)
        print(summarize_fold_metrics(rows_c)[cols].to_string(index=False, float_format=lambda x: f"{x:7.4f}"), flush=True)

        yt = np.concatenate(yt_all, axis=0)
        pn = np.concatenate(pn_all, axis=0)
        pc = np.concatenate(pc_all, axis=0)

        dm = diebold_mariano_nw(yt - pc, yt - pn, step=21, max_lag=12)
        wins_oos = sum(1 for r in rows_c if r["oos_r2_vs_train_mean"] > 0)
        print(
            f"\n  STATISTICAL GENERALIZATION CHECK ({label}):\n"
            f"    Regimes with positive OOS R² vs Naive: {wins_oos}/{len(rows_c)}\n"
            f"    Diebold-Mariano HAC (lag=12m) vs Naive: DM={dm['dm_hac_stat']:+.3f}, one-sided p={dm['p_value_one_sided']:.4f} (n_months={dm['n_monthly_obs']})",
            flush=True,
        )

        # Per-sector 12-year breakdown
        sec_rows = []
        for j, s in enumerate(SECTORS):
            sse_m = np.sum((yt[:, j] - pc[:, j]) ** 2)
            sse_b = np.sum((yt[:, j] - pn[:, j]) ** 2)
            sec_rows.append({
                "sector": s,
                "oos_r2_vs_naive": 1.0 - sse_m / (sse_b + 1e-12),
                "rmse_champ": float(np.sqrt(mean_squared_error(yt[:, j], pc[:, j]))),
                "rmse_naive": float(np.sqrt(mean_squared_error(yt[:, j], pn[:, j]))),
                "ts_ic": float(np.nan_to_num(spearmanr(yt[:, j], pc[:, j])[0])),
            })
        print("\n  Per-Sector Pooled OOS Generalization (HybridTempSectorFormer vs Naive):", flush=True)
        print(pd.DataFrame(sec_rows).to_string(index=False, float_format=lambda x: f"{x:7.4f}"), flush=True)

        plot_sector_predictions_grid(
            pd.concat(dt_all), yt, pc,
            f"{label}: HybridTempSectorFormer (Iter 7) Out-of-Sample 1Y Sector Forecasts",
            out_path / f"{label.lower()}_forecasts.png",
        )
        return {"dm": dm, "regimes_positive_oos_r2": f"{wins_oos}/{len(rows_c)}", "per_sector": sec_rows, "folds_champ": rows_c, "folds_naive": rows_n}

    res_dev = eval_suite("DEV_2012_2019", dev_folds)
    res_all_frozen = eval_suite("ALL_12YR_2012_2024_FROZEN_PRE2020_TEST", dev_folds + test_frozen)
    res_all_expand = eval_suite("ALL_12YR_2012_2024_EXPANDING_PURGED", dev_folds + test_expand)

    (out_path / "champion_audit.json").write_text(
        json.dumps({"dev": res_dev, "all_12yr_frozen": res_all_frozen, "all_12yr_expand": res_all_expand}, indent=2, default=str)
    )
    print(f"\nSaved Iteration 7 champion audit to {out_path.resolve()}", flush=True)


if __name__ == "__main__":
    run_iteration_7_audit()
