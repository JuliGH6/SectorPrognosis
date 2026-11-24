"""Free-Public-Data-Only Ablation (Zero Bloomberg Fundamentals, Zero Paid ISM).

Tests how much of Iteration 9's generalization survives when restricted strictly to
signals that are 100% FREE forever via Yahoo Finance (sector ETF prices) + FRED (public US macro):
- NO Bloomberg PE, PB, PS, BestPE, ROE, Margins, Leverage, EV/EBITDA, Dividends
- NO licensed ISM Manufacturing / ISM Services
- ONLY: Sector prices (returns, vol, drawdown, momentum, Energy spread, systemic corr)
        + Free FRED-equivalent macro (Jobless Claims, Unemployment/Sahm, CPI, CoreCPI,
          CorePCE, PPI, GDP, Housing/Home Sales, Employment Ratio).
"""

from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import RobustScaler
from scipy.stats import spearmanr

from . import HORIZON_DAYS, SECTORS
from .data import compute_forward_targets, load_raw_merged_df
from .features import CYCLICAL_SECTORS, build_all_feature_views
from .generalization_audit import diebold_mariano_nw
from .iteration9 import (
    BetaScaledUncertaintyEnsembleV9,
    augment_macro_features_v9,
    compute_causal_sector_betas,
)
from .metrics import evaluate_predictions, summarize_fold_metrics
from .models_temporal import _eb_shrinkage
from .models_v2 import _recency_weights, _winsorize_train_target
from .portfolio_robustness import simulate_equity_curve
from .splits import (
    get_dev_walk_forward_folds,
    get_sealed_test_folds,
    split_inner_train_val,
    validate_no_leakage,
)

# 100% Free-Public-Data Market Features (Yahoo ETF prices + Free FRED macro only!)
FREE_MARKET_COLS = [
    "macro_ehs_chg250",       # Housing (FRED: HSN1F / HOUST)
    "macro_ehs_chg126",
    "macro_nhs_chg250",
    "macro_cpi",              # FRED: CPIAUCSL
    "macro_core_cpi",         # FRED: CPILFESL
    "macro_cpi_chg126",
    "macro_ppi_minus_cpi",    # FRED: PPIACO - CPIAUCSL
    "macro_sahm_proxy",       # FRED: UNRATE Sahm rule
    "macro_unemp_chg250",     # FRED: UNRATE
    "macro_jc_log_chg250",    # FRED: ICSA
    "macro_emp_ratio_chg250", # FRED: EMRATIO
    "mkt_ew_vol_63d",         # Free ETF prices
    "mkt_ew_vol_250d",
    "mkt_cyc_minus_def_250d",
    "aux_energy_ret_126d",    # Free XLE price
    "aux_energy_ret_250d",
    "aux_energy_dd_250d",
    "aux_energy_minus_util_250d",
    "mkt_ew_dd_250d",
    "mkt_avg_corr_63d",
]

# 100% Free-Public-Data Per-Sector Features (ETF prices only, zero Bloomberg fundamentals!)
FREE_SEC_COLS = [
    "cs_z_ret_63d",
    "cs_z_ret_250d",
    "cs_z_mom_12m_1m",
    "cs_z_vol_63d",
    "cs_z_drawdown_250d",
    "cs_z_dist_sma200",
    "cs_z_sharpe_250d",
    "ret_63d",
    "ret_250d",
    "vol_63d",
    "drawdown_250d",
]

FREE_INTERACT_COLS = [
    "macro_cpi_chg126",
    "macro_ehs_chg250",
    "macro_sahm_proxy",
    "aux_energy_ret_250d",
]


def extract_free_alpha_panel(macro_v9: pd.DataFrame, sec_dfs: dict, indices: np.ndarray) -> np.ndarray:
    n_dates = len(indices)
    sec_stack = np.stack(
        [sec_dfs[s].iloc[indices][FREE_SEC_COLS].to_numpy(dtype=np.float32) for s in SECTORS],
        axis=1,
    )
    mac_inter = macro_v9.iloc[indices][FREE_INTERACT_COLS].to_numpy(dtype=np.float32)
    cyc_sign = np.array([1.0 if s in CYCLICAL_SECTORS else -1.0 for s in SECTORS], dtype=np.float32)[None, :, None]
    inter_stack = np.repeat(mac_inter[:, None, :], len(SECTORS), axis=1) * cyc_sign
    return np.concatenate([sec_stack, inter_stack], axis=2).reshape(n_dates * len(SECTORS), -1).astype(np.float32)


class FreeDataEnsembleV9(BetaScaledUncertaintyEnsembleV9):
    """Zero-Bloomberg, Zero-ISM version of BetaScaledUncertaintyEnsembleV9 (100% free public data)."""

    def _fit_beta_neutral_alpha(self, macro_v9, sec_dfs, y_all, tr_idx, ev_idx, sec_prior, betas):
        tr_sub = tr_idx[::5]
        mp = float(sec_prior.mean())
        wd = _recency_weights(len(tr_sub), half_life_steps=10.0 * 50.0)
        wp = np.repeat(wd, len(SECTORS))
        y_mkt_resid = y_all[tr_sub].mean(axis=1) - mp
        ya_2d = (y_all[tr_sub] - sec_prior[None, :]) - y_mkt_resid[:, None] * betas[None, :]
        ya = _winsorize_train_target(ya_2d.reshape(-1), 0.04, 0.96)

        Xa_tr = extract_free_alpha_panel(macro_v9, sec_dfs, tr_sub)
        Xa_ev = extract_free_alpha_panel(macro_v9, sec_dfs, ev_idx)

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

    def fit_predict(self, macro_v9, sec_dfs, y_all, dates, realized_dates, train_idx, eval_idx, return_diagnostics=False):
        mac_raw = macro_v9[FREE_MARKET_COLS].to_numpy(dtype=np.float32)
        sec_raw = np.stack([sec_dfs[s][FREE_SEC_COLS].to_numpy(dtype=np.float32) for s in SECTORS], axis=1)

        itr, ival = split_inner_train_val(train_idx, dates, realized_dates, val_fraction=0.20)
        sec_prior = y_all[train_idx].mean(axis=0).astype(np.float32)
        mp = float(sec_prior.mean())
        sp_i = y_all[itr].mean(axis=0).astype(np.float32)
        betas_i = compute_causal_sector_betas(y_all[itr], vasicek_shrink=self.vasicek_shrink)
        betas_f = compute_causal_sector_betas(y_all[train_idx], vasicek_shrink=self.vasicek_shrink)

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
        pm_va_g, pm_ev_g = pm_va * gate_va, pm_ev * gate_ev

        ym_va = (y_all[ival].mean(axis=1) - mp).astype(np.float32)
        w_m = _eb_shrinkage(pm_va_g, ym_va, prior_strength=self.eb_prior_mkt, cap=self.cap_mkt)

        pa_va = self._fit_beta_neutral_alpha(macro_v9, sec_dfs, y_all, itr, ival, sp_i, betas_i)
        ya_va = (y_all[ival] - sp_i[None, :]) - (y_all[ival].mean(axis=1) - float(sp_i.mean()))[:, None] * betas_i[None, :]
        w_a = _eb_shrinkage(pa_va, ya_va, prior_strength=self.eb_prior_alpha, cap=self.cap_alpha)
        pa_ev = self._fit_beta_neutral_alpha(macro_v9, sec_dfs, y_all, train_idx, eval_idx, sec_prior, betas_f)

        preds = (sec_prior[None, :] + w_m * pm_ev_g[:, None] * betas_f[None, :] + w_a * pa_ev).astype(np.float32)
        if return_diagnostics:
            return preds, {"w_m": w_m, "w_a": w_a, "mean_gate": float(np.mean(gate_ev))}
        return preds


def run_free_data_ablation():
    df = load_raw_merged_df("data")
    y_df, realized = compute_forward_targets(df, horizon=HORIZON_DAYS)
    y_all = y_df.to_numpy(dtype=np.float32)
    macro_df, sec_dfs, feat_valid = build_all_feature_views(df)
    macro_v9 = augment_macro_features_v9(df, macro_df)

    dates = df["Date"]
    free_valid = feat_valid & np.isfinite(macro_v9[FREE_MARKET_COLS].to_numpy()).all(axis=1)
    train_valid_mask = free_valid & (~y_df.isna().any(axis=1)).to_numpy(copy=True)
    train_valid_mask[:500] = False

    # 1. 12-Year Statistical Generalization Audit (Expanding Purged, 6 regimes)
    dev_folds = get_dev_walk_forward_folds(dates, realized, train_valid_mask)
    test_expand = get_sealed_test_folds(dates, realized, train_valid_mask, strictly_frozen_pre2020=False)
    folds_6 = dev_folds + test_expand

    mdl = FreeDataEnsembleV9(seeds=(42, 102, 202), epochs=35)
    rows_f, yt_all, pn_all, pf_all = [], [], [], []
    print("=== FREE-DATA-ONLY V9 (Zero Bloomberg, Zero ISM) — 12-YEAR GENERALIZATION AUDIT ===", flush=True)
    for f in folds_6:
        validate_no_leakage(dates, realized, f.train_idx, f.eval_idx)
        ym = y_all[f.train_idx].mean(axis=0)
        p_n = np.tile(ym, (len(f.eval_idx), 1))
        pf = mdl.fit_predict(macro_v9, sec_dfs, y_all, dates, realized, f.train_idx, f.eval_idx)
        mf = evaluate_predictions(y_all[f.eval_idx], pf, y_train_mean=ym); mf["fold"] = f.name
        rows_f.append(mf)
        yt_all.append(y_all[f.eval_idx]); pn_all.append(p_n); pf_all.append(pf)

    cols = ["fold", "oos_r2_vs_train_mean", "rel_r2", "rmse", "mean_ts_ic", "mean_cs_ic", "top2_vs_bot2_spread"]
    print(summarize_fold_metrics(rows_f)[cols].to_string(index=False, float_format=lambda x: f"{x:7.4f}"), flush=True)

    yt, pn, pf = np.concatenate(yt_all), np.concatenate(pn_all), np.concatenate(pf_all)
    dm = diebold_mariano_nw(yt - pf, yt - pn, step=21, max_lag=12)
    print(f"\n  Diebold-Mariano HAC (lag=12m) vs Naive: DM={dm['dm_hac_stat']:+.3f}, one-sided p={dm['p_value_one_sided']:.4f}", flush=True)

    # 2. 13-Year S&P 500 + ±6% Nudge Backtest (2012-2024)
    ev_2024 = np.where(free_valid & (dates >= pd.Timestamp("2024-01-01")) & (dates <= pd.Timestamp("2024-12-31")))[0]
    min_24 = int(ev_2024.min())
    tr_24 = np.where(train_valid_mask & (np.arange(len(dates)) <= min_24 - HORIZON_DAYS - 5) & (realized < dates.iloc[min_24]))[0]
    validate_no_leakage(dates, realized, tr_24, ev_2024)
    pf_24 = mdl.fit_predict(macro_v9, sec_dfs, y_all, dates, realized, tr_24, ev_2024)

    bt_idx = np.concatenate([f.eval_idx for f in folds_6[:-1]] + [np.where(free_valid & (dates >= pd.Timestamp("2022-01-01")) & (dates <= pd.Timestamp("2023-12-31")))[0], ev_2024])
    # Reuse clean 2022-2023 slice from folds_6[-1] (which covers 2022-01-03..2024-01-03) + 2024
    # Simpler: directly concatenate non-overlapping 2012..2024 dates using folds_6 + 2024 tail
    idx_list = [f.eval_idx for f in folds_6]
    p_list = pf_all
    tail_mask = ev_2024 > idx_list[-1].max()
    bt_idx = np.concatenate(idx_list + [ev_2024[tail_mask]])
    preds_13 = np.concatenate(p_list + [pf_24[tail_mask]], axis=0)
    bt_dates = dates.iloc[bt_idx].reset_index(drop=True)

    prices_bt = df[SECTORS].iloc[bt_idx].to_numpy(dtype=np.float64)
    daily_rets = np.zeros_like(prices_bt)
    daily_rets[1:] = prices_bt[1:] / prices_bt[:-1] - 1.0
    T = len(bt_dates)
    years_13 = (bt_dates.iloc[-1] - bt_dates.iloc[0]).days / 365.25

    cap_w0 = np.array([0.38, 0.06, 0.18, 0.15, 0.17, 0.06], dtype=np.float64)
    cap_w0 /= cap_w0.sum()
    cap_w = np.zeros((T, 6)); cap_w[0] = cap_w0
    for t in range(1, T):
        d = cap_w[t - 1] * (1.0 + daily_rets[t])
        cap_w[t] = d / d.sum()

    tilt_w = np.zeros((T, 6))
    for t in range(T):
        z = (preds_13[t] - np.mean(preds_13[t])) / (np.std(preds_13[t]) + 1e-8)
        w = np.clip(cap_w[t] + 0.06 * z, 0.0, 1.0)
        tilt_w[t] = w / w.sum()

    eq_base, _ = simulate_equity_curve(daily_rets, cap_w, rebal_every=999999, tc_bps=10.0)
    eq_free, _ = simulate_equity_curve(daily_rets, tilt_w, rebal_every=21, tc_bps=10.0)
    s5 = np.arange(0, T - 1260, 21)
    w5 = float(np.mean((eq_free[s5 + 1260] / eq_free[s5]) > (eq_base[s5 + 1260] / eq_base[s5])) * 100.0)
    exc_dr = (eq_free[1:] / eq_free[:-1] - 1.0) - (eq_base[1:] / eq_base[:-1] - 1.0)
    ir = float((np.mean(exc_dr) * 252.0) / (np.std(exc_dr) * np.sqrt(252.0) + 1e-12))

    print("\n=== FREE-DATA-ONLY 13-YEAR $1,000 S&P 500 + ±6% NUDGE (2012-2024, 10 bps TC) ===", flush=True)
    print(f"  Cap-Weighted S&P 500 (Base)         : ${eq_base[-1]:,.2f}  (CAGR={((eq_base[-1]/1000)**(1/years_13)-1)*100:.2f}%)", flush=True)
    print(f"  S&P 500 + Free-Data V9 Nudge (±6%)  : ${eq_free[-1]:,.2f}  (CAGR={((eq_free[-1]/1000)**(1/years_13)-1)*100:.2f}%, IR={ir:+.2f}, Beats S&P in {w5:.1f}% of 5Y starts)", flush=True)


if __name__ == "__main__":
    run_free_data_ablation()
