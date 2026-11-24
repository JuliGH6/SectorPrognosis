"""Kaggle GPU Sweep for SectorPrognosis v2.0 (Strict Zero-Leakage).

Runs on Kaggle NVIDIA GPU (T4 / P100):
1. Verifies GPU availability and zero-leakage split assertions.
2. Trains TempSectorFormer (Factorized Temporal + Cross-Sector Transformer with
   Prior-Anchored Decomposition, Huber + Cosine IC Loss, and 5-seed bagging on CUDA).
3. Runs Optuna hyperparameter search (35 trials) strictly on Purged Dev Folds (2012-2019)
   for the Prior-Anchored Decomposed Ensemble (CatBoost + LightGBM + ElasticNet).
4. Locks the best Dev config and evaluates ONCE on the Sealed Test Set (2020-2024)
   in both Frozen Pre-2020 and Purged Expanding modes.
5. Saves complete results to /kaggle/working/gpu_sweep_results.json.
"""

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from catboost import CatBoostRegressor
from scipy.stats import spearmanr
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import RobustScaler

optuna.logging.set_verbosity(optuna.logging.WARNING)

SECTORS = [
    "HealthCare",
    "Materials",
    "Industrials",
    "ConsumerStaples",
    "ConsumerDiscretionary",
    "Utilities",
]
CYCLICAL_SECTORS = {"Materials", "Industrials", "ConsumerDiscretionary"}
DEFENSIVE_SECTORS = {"HealthCare", "ConsumerStaples", "Utilities"}
VALUATION_METRICS = ["PE", "PriceToBook", "PriceToSales", "BestPE", "EV/EBITDA"]
QUALITY_METRICS = ["ROE", "GrossMargin", "EBITDAMargin", "ProfitMargin"]
LEVERAGE_METRICS = ["DebtToEquity", "DebtToEBITDA"]
HORIZON_DAYS = 250

MARKET_CORE_COLS = [
    "macro_ehs_chg250",
    "macro_ehs_chg126",
    "macro_nhs_chg250",
    "macro_cpi",
    "macro_core_cpi",
    "macro_cpi_chg126",
    "macro_ppi_minus_cpi",
    "macro_ism_mfg",
    "macro_ism_mfg_chg126",
    "macro_ism_srv",
    "macro_sahm_proxy",
    "macro_unemp_chg250",
    "mkt_ew_vol_63d",
    "mkt_ew_vol_250d",
    "mkt_avg_valuation",
    "mkt_cyc_minus_def_250d",
]

ALPHA_SEC_CORE_COLS = [
    "cs_z_comp_valuation",
    "cs_z_comp_quality",
    "cs_z_comp_leverage",
    "cs_z_comp_garp",
    "cs_z_fund_Dividends",
    "cs_z_fund_ROE",
    "cs_z_fund_ProfitMargin",
    "cs_z_ret_63d",
    "cs_z_ret_250d",
    "cs_z_mom_12m_1m",
    "cs_z_vol_63d",
    "cs_z_drawdown_250d",
    "chg250_val",
    "chg250_qual",
    "chg250_roe",
    "chg63_roe",
]

ALPHA_MACRO_INTERACT_COLS = [
    "macro_ism_mfg_chg126",
    "macro_cpi_chg126",
    "macro_ehs_chg250",
    "macro_sahm_proxy",
]


def _signed_log1p(s: pd.Series) -> pd.Series:
    return np.sign(s) * np.log1p(np.abs(s))


def build_features(df: pd.DataFrame):
    assert df["Date"].is_monotonic_increasing
    out: Dict[str, pd.Series] = {}

    jc_log = np.log1p(df["Total Jobless Claims"].clip(lower=0))
    out["macro_jc_log"] = jc_log
    out["macro_jc_log_chg63"] = jc_log - jc_log.shift(63)
    out["macro_jc_log_chg250"] = jc_log - jc_log.shift(250)
    out["macro_jc_yoy_slog"] = _signed_log1p(df["JoblessClaimsYoY"])

    unemp = df["UnemploymentRate"]
    out["macro_unemp"] = unemp
    out["macro_unemp_chg63"] = unemp - unemp.shift(63)
    out["macro_unemp_chg250"] = unemp - unemp.shift(250)
    out["macro_sahm_proxy"] = unemp - unemp.rolling(250, min_periods=250).min()

    for col, short in [("ISMManufacturing", "ism_mfg"), ("ISMService", "ism_srv")]:
        s = df[col]
        out[f"macro_{short}"] = s - 50.0
        out[f"macro_{short}_chg63"] = s - s.shift(63)
        out[f"macro_{short}_chg126"] = s - s.shift(126)
        out[f"macro_{short}_chg250"] = s - s.shift(250)
    out["macro_ism_spread"] = df["ISMManufacturing"] - df["ISMService"]

    gdp = _signed_log1p(df["GDPChained"])
    out["macro_gdp_slog"] = gdp
    out["macro_gdp_chg126"] = gdp - gdp.shift(126)

    for col, short in [("CPI", "cpi"), ("CoreCPI", "core_cpi"), ("CorePCE", "core_pce"), ("PPI", "ppi")]:
        s = df[col]
        out[f"macro_{short}"] = s
        out[f"macro_{short}_chg63"] = s - s.shift(63)
        out[f"macro_{short}_chg126"] = s - s.shift(126)
    out["macro_ppi_minus_cpi"] = df["PPI"] - df["CPI"]
    out["macro_headline_minus_core_cpi"] = df["CPI"] - df["CoreCPI"]

    for col, short in [("ExistingHomeSales", "ehs"), ("NewHomeSales", "nhs")]:
        s = np.log1p(df[col].clip(lower=0))
        out[f"macro_{short}_log"] = s
        out[f"macro_{short}_chg126"] = s - s.shift(126)
        out[f"macro_{short}_chg250"] = s - s.shift(250)

    emp = df["EmploymentRatio"]
    out["macro_emp_ratio"] = emp
    out["macro_emp_ratio_chg126"] = emp - emp.shift(126)
    out["macro_emp_ratio_chg250"] = emp - emp.shift(250)

    prices = df[SECTORS]
    daily_log_ret = np.log(prices / prices.shift(1))
    ew_daily_ret = daily_log_ret.mean(axis=1)

    for w in (21, 63, 126, 250):
        sec_rets = prices.pct_change(w)
        out[f"mkt_ew_ret_{w}d"] = sec_rets.mean(axis=1)
        out[f"mkt_disp_ret_{w}d"] = sec_rets.std(axis=1)
        out[f"mkt_cyc_minus_def_{w}d"] = (
            sec_rets[list(CYCLICAL_SECTORS)].mean(axis=1)
            - sec_rets[list(DEFENSIVE_SECTORS)].mean(axis=1)
        )

    for w in (21, 63, 250):
        out[f"mkt_ew_vol_{w}d"] = ew_daily_ret.rolling(w, min_periods=w).std() * np.sqrt(252.0)

    val_cols = [f"{s} {m}" for s in SECTORS for m in VALUATION_METRICS if f"{s} {m}" in df.columns]
    out["mkt_avg_valuation"] = df[val_cols].mean(axis=1)
    macro_df = pd.DataFrame(out, index=df.index).astype(np.float32)

    per_sec = {s: {} for s in SECTORS}
    for s in SECTORS:
        p = df[s]
        dlog = np.log(p / p.shift(1))
        d = per_sec[s]
        for w in (21, 63, 126, 250):
            d[f"ret_{w}d"] = p.pct_change(w)
        d["mom_12m_1m"] = d["ret_250d"] - d["ret_21d"]
        for w in (21, 63, 250):
            d[f"vol_{w}d"] = dlog.rolling(w, min_periods=w).std() * np.sqrt(252.0)
        d["drawdown_250d"] = p / p.rolling(250, min_periods=250).max() - 1.0
        for m in ["Dividends", "PE", "ROE", "ProfitMargin"]:
            d[f"fund_{m}"] = df[f"{s} {m}"]
        val_comp = df[[f"{s} {m}" for m in VALUATION_METRICS]].mean(axis=1)
        qual_comp = df[[f"{s} {m}" for m in QUALITY_METRICS]].mean(axis=1)
        lev_comp = df[[f"{s} {m}" for m in LEVERAGE_METRICS]].mean(axis=1)
        d["comp_valuation"] = val_comp
        d["comp_quality"] = qual_comp
        d["comp_leverage"] = lev_comp
        d["comp_garp"] = qual_comp - val_comp
        d["chg250_val"] = val_comp - val_comp.shift(250)
        d["chg250_qual"] = qual_comp - qual_comp.shift(250)
        d["chg250_roe"] = df[f"{s} ROE"] - df[f"{s} ROE"].shift(250)
        d["chg63_roe"] = df[f"{s} ROE"] - df[f"{s} ROE"].shift(63)

    sec_dfs = {s: pd.DataFrame(per_sec[s], index=df.index) for s in SECTORS}
    cs_cols = [
        "comp_valuation", "comp_quality", "comp_leverage", "comp_garp",
        "fund_Dividends", "fund_ROE", "fund_ProfitMargin",
        "ret_63d", "ret_250d", "mom_12m_1m", "vol_63d", "drawdown_250d",
    ]
    for col in cs_cols:
        mat = pd.concat([sec_dfs[s][col] for s in SECTORS], axis=1)
        mat.columns = SECTORS
        rmean = mat.mean(axis=1)
        rstd = mat.std(axis=1) + 1e-6
        for s in SECTORS:
            sec_dfs[s][f"cs_z_{col}"] = (mat[s] - rmean) / rstd

    sec_dfs = {s: sec_dfs[s].astype(np.float32) for s in SECTORS}
    valid = np.isfinite(macro_df[MARKET_CORE_COLS].to_numpy()).all(axis=1)
    for s in SECTORS:
        valid &= np.isfinite(sec_dfs[s][ALPHA_SEC_CORE_COLS].to_numpy()).all(axis=1)
    return macro_df, sec_dfs, valid


@dataclass(frozen=True)
class Fold:
    name: str
    train_idx: np.ndarray
    eval_idx: np.ndarray


def validate_no_leakage(dates, realized, tr, ev):
    assert len(np.intersect1d(tr, ev)) == 0, "LEAKAGE: overlapping indices!"
    assert int(tr.max()) + HORIZON_DAYS < int(ev.min()), "LEAKAGE: horizon purge violated!"
    assert realized.iloc[tr].max() < dates.iloc[ev].min(), "LEAKAGE: realized date >= eval start!"


def make_fold(name, dates, realized, valid_mask, start_s, end_s, cap=None, embargo=5):
    ev = np.where(valid_mask & (dates >= pd.Timestamp(start_s)) & (dates <= pd.Timestamp(end_s)))[0]
    min_ev = int(ev.min())
    min_ev_date = dates.iloc[min_ev]
    idx = np.arange(len(dates))
    tr_mask = valid_mask & (idx <= min_ev - HORIZON_DAYS - embargo) & (realized < min_ev_date)
    if cap is not None:
        tr_mask &= dates <= pd.Timestamp(cap)
    tr = np.where(tr_mask)[0]
    validate_no_leakage(dates, realized, tr, ev)
    return Fold(name, tr, ev)


def split_inner(tr, dates, realized, val_frac=0.20, embargo=5):
    n_val = max(60, int(len(tr) * val_frac))
    ival = tr[-n_val:]
    min_v = int(ival.min())
    min_vd = dates.iloc[min_v]
    itr = tr[(tr <= min_v - HORIZON_DAYS - embargo) & (realized.iloc[tr].to_numpy() < min_vd)]
    validate_no_leakage(dates, realized, itr, ival)
    return itr, ival


def evaluate_predictions(y_true, y_pred, y_train_mean, rebalance_step=21):
    sse_m = np.sum((y_true - y_pred) ** 2)
    sse_b = np.sum((y_true - y_train_mean[None, :]) ** 2)
    oos_r2 = float(1.0 - sse_m / (sse_b + 1e-12))
    yt_rel = y_true - y_true.mean(axis=1, keepdims=True)
    yp_rel = y_pred - y_pred.mean(axis=1, keepdims=True)
    rel_r2 = float(r2_score(yt_rel.ravel(), yp_rel.ravel()))
    ts_ics = [
        0.0 if np.std(y_pred[:, j]) < 1e-6 else float(np.nan_to_num(spearmanr(y_true[:, j], y_pred[:, j])[0]))
        for j in range(6)
    ]
    cs_ics = [
        0.0 if np.std(y_pred[i]) < 1e-6 else float(np.nan_to_num(spearmanr(y_true[i], y_pred[i])[0]))
        for i in range(len(y_true))
    ]
    sub = np.arange(0, len(y_true), rebalance_step)
    excess, spread = [], []
    for i in sub:
        order = np.argsort(y_pred[i])
        t2, b2, ew = y_true[i, order[-2:]].mean(), y_true[i, order[:2]].mean(), y_true[i].mean()
        excess.append(t2 - ew)
        spread.append(t2 - b2)
    return {
        "mean_r2": float(np.mean([r2_score(y_true[:, j], y_pred[:, j]) for j in range(6)])),
        "oos_r2_vs_train_mean": oos_r2,
        "rel_r2": rel_r2,
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "dir_acc": float(np.mean((y_true > 0) == (y_pred > 0))),
        "mean_ts_ic": float(np.mean(ts_ics)),
        "mean_cs_ic": float(np.mean(cs_ics)),
        "top2_excess_1y_ret": float(np.mean(excess)),
        "top2_vs_bot2_spread": float(np.mean(spread)),
        "top2_win_rate_vs_ew": float(np.mean(np.array(excess) > 0)),
    }


def extract_panel(macro_df, sec_dfs, indices):
    n_dates = len(indices)
    sec_stack = np.stack([sec_dfs[s].iloc[indices][ALPHA_SEC_CORE_COLS].to_numpy(dtype=np.float32) for s in SECTORS], axis=1)
    mac = macro_df.iloc[indices][ALPHA_MACRO_INTERACT_COLS].to_numpy(dtype=np.float32)
    cyc = np.array([1.0 if s in CYCLICAL_SECTORS else -1.0 for s in SECTORS], dtype=np.float32)[None, :, None]
    inter = np.repeat(mac[:, None, :], 6, axis=1) * cyc
    return np.concatenate([sec_stack, inter], axis=2).reshape(n_dates * 6, -1).astype(np.float32)


def recency_weights(n, half_life_steps):
    age = np.arange(n - 1, -1, -1, dtype=np.float32)
    w = np.exp(-np.log(2.0) * age / half_life_steps)
    return (w * (n / w.sum())).astype(np.float32)


# ==============================================================================
# GPU MODEL: Factorized Temporal + Cross-Sector Transformer (TempSectorFormer)
# ==============================================================================
class TempSectorFormer(nn.Module):
    """Multi-scale causal temporal sequence + cross-sector attention Transformer."""

    def __init__(self, f_sec: int, f_mac: int, d_model: int = 64, n_heads: int = 4, dropout: float = 0.30):
        super().__init__()
        self.sec_proj = nn.Sequential(nn.Linear(f_sec, d_model), nn.LayerNorm(d_model), nn.SiLU())
        self.mac_proj = nn.Sequential(nn.Linear(f_mac, d_model), nn.LayerNorm(d_model), nn.SiLU())
        self.sec_emb = nn.Embedding(6, d_model)
        self.pos_emb = nn.Parameter(torch.randn(1, 12, d_model) * 0.02)

        enc_t = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 2,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True,
        )
        self.temp_enc = nn.TransformerEncoder(enc_t, num_layers=1)

        enc_s = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 2,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True,
        )
        self.sec_enc = nn.TransformerEncoder(enc_s, num_layers=1)

        self.alpha_head = nn.Linear(d_model, 1)
        self.mkt_head = nn.Sequential(
            nn.Linear(d_model * 2, d_model // 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )
        nn.init.normal_(self.alpha_head.weight, std=0.01)
        nn.init.zeros_(self.alpha_head.bias)

    def forward(self, x_sec_seq, x_mac_seq):
        # x_sec_seq: (B, T=12, S=6, F_sec), x_mac_seq: (B, T=12, F_mac)
        if self.training:
            x_sec_seq = x_sec_seq + 0.04 * torch.randn_like(x_sec_seq)
            x_mac_seq = x_mac_seq + 0.04 * torch.randn_like(x_mac_seq)

        B, T, S, _ = x_sec_seq.shape
        hm = self.mac_proj(x_mac_seq) + self.pos_emb[:, :T, :]  # (B, T, D)
        hm_t = self.temp_enc(hm)[:, -1, :]  # (B, D) latest temporal summary

        # Per-sector temporal encoding
        hs = self.sec_proj(x_sec_seq)  # (B, T, S, D)
        hs = hs.permute(0, 2, 1, 3).reshape(B * S, T, -1) + self.pos_emb[:, :T, :]
        hs_t = self.temp_enc(hs)[:, -1, :].reshape(B, S, -1)  # (B, S, D)

        # Cross-sector attention conditioned on macro context + sector identity
        sec_ids = torch.arange(S, device=x_sec_seq.device).unsqueeze(0).expand(B, -1)
        hs_ctx = hs_t + hm_t.unsqueeze(1) + self.sec_emb(sec_ids)
        hs_out = self.sec_enc(hs_ctx)  # (B, S, D)

        raw_a = self.alpha_head(hs_out).squeeze(-1)  # (B, S)
        alpha_resid = 0.25 * torch.tanh(raw_a - raw_a.mean(dim=-1, keepdim=True))
        mkt_resid = 0.20 * torch.tanh(self.mkt_head(torch.cat([hm_t, hs_out.mean(dim=1)], dim=-1)).squeeze(-1))
        return mkt_resid, alpha_resid


def build_causal_sequences(macro_all: np.ndarray, sec_all: np.ndarray, indices: np.ndarray, offsets: np.ndarray):
    # offsets: e.g. [-231, -210, ..., 0] (12 monthly points, strictly <= t)
    seq_idx = indices[:, None] + offsets[None, :]  # (N, 12)
    return sec_all[seq_idx], macro_all[seq_idx]


def fit_predict_temp_sector_former(
    macro_df, sec_dfs, y_all, dates, realized, train_idx, eval_idx, device, seeds=(42, 102, 202)
):
    offsets = np.array([-231, -210, -189, -168, -147, -126, -105, -84, -63, -42, -21, 0], dtype=np.int64)
    mac_raw = macro_df[MARKET_CORE_COLS].to_numpy(dtype=np.float32)
    sec_raw = np.stack([sec_dfs[s][ALPHA_SEC_CORE_COLS].to_numpy(dtype=np.float32) for s in SECTORS], axis=1)

    itr, ival = split_inner(train_idx, dates, realized, val_frac=0.20)
    itr_sub = itr[::5]
    sec_prior = y_all[train_idx].mean(axis=0).astype(np.float32)
    mkt_prior = float(sec_prior.mean())
    alpha_prior = sec_prior - mkt_prior

    # Strictly causal scaling fit ONLY on rows <= itr.max()
    m_sc = RobustScaler().fit(mac_raw[itr_sub])
    mac_s = np.clip(m_sc.transform(mac_raw), -3.5, 3.5).astype(np.float32)
    s_sc = RobustScaler().fit(sec_raw[itr_sub].reshape(-1, sec_raw.shape[-1]))
    sec_s = np.clip(s_sc.transform(sec_raw.reshape(-1, sec_raw.shape[-1])), -3.5, 3.5).reshape(sec_raw.shape).astype(np.float32)

    xs_tr, xm_tr = build_causal_sequences(mac_s, sec_s, itr_sub, offsets)
    xs_va, xm_va = build_causal_sequences(mac_s, sec_s, ival, offsets)
    xs_ev, xm_ev = build_causal_sequences(mac_s, sec_s, eval_idx, offsets)

    ym_tr = np.clip(y_all[itr_sub].mean(axis=1) - mkt_prior, -0.30, 0.30).astype(np.float32)
    ya_tr = np.clip((y_all[itr_sub] - y_all[itr_sub].mean(axis=1, keepdims=True)) - alpha_prior[None, :], -0.25, 0.25).astype(np.float32)
    ym_va = (y_all[ival].mean(axis=1) - mkt_prior).astype(np.float32)
    ya_va = ((y_all[ival] - y_all[ival].mean(axis=1, keepdims=True)) - alpha_prior[None, :]).astype(np.float32)

    txs_tr = torch.from_numpy(xs_tr).to(device)
    txm_tr = torch.from_numpy(xm_tr).to(device)
    tym_tr = torch.from_numpy(ym_tr).to(device)
    tya_tr = torch.from_numpy(ya_tr).to(device)
    txs_va = torch.from_numpy(xs_va).to(device)
    txm_va = torch.from_numpy(xm_va).to(device)
    tym_va = torch.from_numpy(ym_va).to(device)
    tya_va = torch.from_numpy(ya_va).to(device)
    txs_ev = torch.from_numpy(xs_ev).to(device)
    txm_ev = torch.from_numpy(xm_ev).to(device)

    pm_va_list, pa_va_list, pm_ev_list, pa_ev_list = [], [], [], []
    N = len(txs_tr)
    bs = 128

    for seed in seeds:
        torch.manual_seed(seed)
        np.random.seed(seed)
        model = TempSectorFormer(sec_raw.shape[-1], mac_raw.shape[-1], d_model=48, n_heads=4, dropout=0.35).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=1.5e-3, weight_decay=2e-2)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=35)
        best_loss, best_state, bad = float("inf"), None, 0

        for _ in range(35):
            model.train()
            perm = torch.randperm(N, device=device)
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
                if bad >= 12:
                    break

        if best_state is not None:
            model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
        model.eval()
        with torch.no_grad():
            pm_v, pa_v = model(txs_va, txm_va)
            pm_e, pa_e = model(txs_ev, txm_ev)
            pm_va_list.append(pm_v.cpu().numpy())
            pa_va_list.append(pa_v.cpu().numpy())
            pm_ev_list.append(pm_e.cpu().numpy())
            pa_ev_list.append(pa_e.cpu().numpy())

    pm_va = np.mean(pm_va_list, axis=0)
    pa_va = np.mean(pa_va_list, axis=0)
    pm_ev = np.mean(pm_ev_list, axis=0)
    pa_ev = np.mean(pa_ev_list, axis=0)

    # Purged inner-val shrinkage calibration
    w_m = float(np.clip(np.dot(pm_va, ym_va) / (np.dot(pm_va, pm_va) + 1e-8), 0.0, 0.80))
    w_a = float(np.clip(np.sum(pa_va * ya_va) / (np.sum(pa_va * pa_va) + 1e-8), 0.0, 0.80))
    return (sec_prior[None, :] + w_m * pm_ev[:, None] + w_a * pa_ev).astype(np.float32)


# ==============================================================================
# Optuna-Tunable Prior-Anchored Decomposed Model
# ==============================================================================
def fit_predict_decomposed(macro_df, sec_dfs, y_all, dates, realized, train_idx, eval_idx, params: dict):
    stride = params.get("stride", 5)
    hl_years = params.get("hl_years", 8.0)
    q_win = params.get("q_win", 0.05)
    cb_depth = params.get("cb_depth", 3)
    cb_l2 = params.get("cb_l2", 25.0)
    cb_iters = params.get("cb_iters", 220)
    enet_alpha = params.get("enet_alpha", 0.012)
    enet_l1 = params.get("enet_l1", 0.35)
    alpha_enet_w = params.get("alpha_enet_w", 0.45)
    shrink_cap = params.get("shrink_cap", 0.85)

    def _fit_raw(tr_i, ev_i):
        tr_sub = tr_i[::stride]
        sp = y_all[tr_i].mean(axis=0).astype(np.float32)
        mp = float(sp.mean())
        ap = sp - mp
        wd = recency_weights(len(tr_sub), hl_years * (250.0 / stride))

        ym = y_all[tr_sub].mean(axis=1) - mp
        lo, hi = np.quantile(ym, [q_win, 1.0 - q_win])
        ym = np.clip(ym, lo, hi).astype(np.float32)
        Xm_tr = macro_df.iloc[tr_sub][MARKET_CORE_COLS].to_numpy(dtype=np.float32)
        Xm_ev = macro_df.iloc[ev_i][MARKET_CORE_COLS].to_numpy(dtype=np.float32)

        cb_m = CatBoostRegressor(
            iterations=cb_iters, learning_rate=0.03, depth=cb_depth, l2_leaf_reg=cb_l2,
            subsample=0.7, rsm=0.6, random_seed=42, verbose=0,
        )
        cb_m.fit(Xm_tr, ym, sample_weight=wd)
        pm = cb_m.predict(Xm_ev).astype(np.float32)

        ya = ((y_all[tr_sub] - y_all[tr_sub].mean(axis=1, keepdims=True)) - ap[None, :]).reshape(-1)
        lo_a, hi_a = np.quantile(ya, [q_win, 1.0 - q_win])
        ya = np.clip(ya, lo_a, hi_a).astype(np.float32)
        wp = np.repeat(wd, 6)
        Xa_tr = extract_panel(macro_df, sec_dfs, tr_sub)
        Xa_ev = extract_panel(macro_df, sec_dfs, ev_i)

        cb_a = CatBoostRegressor(
            iterations=cb_iters, learning_rate=0.03, depth=cb_depth, l2_leaf_reg=cb_l2,
            subsample=0.7, rsm=0.6, random_seed=42, verbose=0,
        )
        cb_a.fit(Xa_tr, ya, sample_weight=wp)
        pa_cb = cb_a.predict(Xa_ev).reshape(-1, 6)
        pa_cb -= pa_cb.mean(axis=1, keepdims=True)

        sc = RobustScaler()
        Xa_tr_s = np.clip(sc.fit_transform(Xa_tr), -3.0, 3.0)
        Xa_ev_s = np.clip(sc.transform(Xa_ev), -3.0, 3.0)
        en = ElasticNet(alpha=enet_alpha, l1_ratio=enet_l1, max_iter=2000, random_state=42)
        en.fit(Xa_tr_s, ya, sample_weight=wp)
        pa_en = np.clip(en.predict(Xa_ev_s), -0.30, 0.30).reshape(-1, 6)
        pa_en -= pa_en.mean(axis=1, keepdims=True)

        pa = ((1.0 - alpha_enet_w) * pa_cb + alpha_enet_w * pa_en).astype(np.float32)
        return sp, pm, pa

    itr, ival = split_inner(train_idx, dates, realized, val_frac=0.20)
    sp_i, pm_i, pa_i = _fit_raw(itr, ival)
    yv = y_all[ival]
    ym_v = yv.mean(axis=1) - float(sp_i.mean())
    ya_v = (yv - yv.mean(axis=1, keepdims=True)) - (sp_i - float(sp_i.mean()))[None, :]
    w_m = float(np.clip(np.dot(pm_i, ym_v) / (np.dot(pm_i, pm_i) + 1e-8), 0.0, shrink_cap))
    w_a = float(np.clip(np.sum(pa_i * ya_v) / (np.sum(pa_i * pa_i) + 1e-8), 0.0, shrink_cap))

    sp, pm, pa = _fit_raw(train_idx, eval_idx)
    return (sp[None, :] + w_m * pm[:, None] + w_a * pa).astype(np.float32)


def main():
    t0 = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== Kaggle GPU Sweep Starting | device={device} ===", flush=True)
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)

    found = list(Path("/kaggle/input").rglob("merged_clean.parquet")) if Path("/kaggle/input").exists() else []
    data_path = found[0] if found else Path("data/merged_clean.parquet")
    print(f"Using dataset path: {data_path}", flush=True)
    df = pd.read_parquet(data_path)
    df["Date"] = pd.to_datetime(df["Date"])
    assert df["Date"].is_monotonic_increasing

    prices = df[SECTORS]
    y_df = (prices.shift(-HORIZON_DAYS) - prices) / prices
    realized = df["Date"].shift(-HORIZON_DAYS)
    y_all = y_df.to_numpy(dtype=np.float32)

    macro_df, sec_dfs, feat_valid = build_features(df)
    valid_mask = feat_valid & (~y_df.isna().any(axis=1)).to_numpy(copy=True)
    # Extra warmup for 231-day temporal sequence lookback
    valid_mask[:500] = False

    dev_folds = [
        make_fold("Val_2012_2013", df["Date"], realized, valid_mask, "2012-01-01", "2013-12-31"),
        make_fold("Val_2014_2015", df["Date"], realized, valid_mask, "2014-01-01", "2015-12-31"),
        make_fold("Val_2016_2017", df["Date"], realized, valid_mask, "2016-01-01", "2017-12-31"),
        make_fold("Val_2018_2019", df["Date"], realized, valid_mask, "2018-01-01", "2019-12-31"),
    ]
    test_frozen = [
        make_fold("Test_2020_2021_Frozen", df["Date"], realized, valid_mask, "2020-01-01", "2021-12-31", cap="2018-12-31"),
        make_fold("Test_2022_2024_Frozen", df["Date"], realized, valid_mask, "2022-01-01", "2024-12-31", cap="2018-12-31"),
    ]
    test_expand = [
        make_fold("Test_2020_2021_Expand", df["Date"], realized, valid_mask, "2020-01-01", "2021-12-31"),
        make_fold("Test_2022_2024_Expand", df["Date"], realized, valid_mask, "2022-01-01", "2024-12-31"),
    ]

    # 1. Optuna hyperparameter sweep strictly on Dev Folds (2012-2019)
    print("\n--- Step 1: Optuna Search (35 trials) strictly on Dev Folds 2012-2019 ---", flush=True)

    def objective(trial):
        params = {
            "stride": 5,
            "hl_years": trial.suggest_float("hl_years", 4.0, 14.0),
            "q_win": trial.suggest_float("q_win", 0.02, 0.12),
            "cb_depth": trial.suggest_int("cb_depth", 2, 5),
            "cb_l2": trial.suggest_float("cb_l2", 10.0, 60.0),
            "cb_iters": trial.suggest_int("cb_iters", 150, 350, step=50),
            "enet_alpha": trial.suggest_float("enet_alpha", 0.005, 0.05, log=True),
            "enet_l1": trial.suggest_float("enet_l1", 0.15, 0.70),
            "alpha_enet_w": trial.suggest_float("alpha_enet_w", 0.0, 0.8),
            "shrink_cap": trial.suggest_float("shrink_cap", 0.45, 0.90),
        }
        oos_r2s, cs_ics = [], []
        for f in dev_folds:
            ym = y_all[f.train_idx].mean(axis=0)
            p = fit_predict_decomposed(macro_df, sec_dfs, y_all, df["Date"], realized, f.train_idx, f.eval_idx, params)
            m = evaluate_predictions(y_all[f.eval_idx], p, ym)
            oos_r2s.append(m["oos_r2_vs_train_mean"])
            cs_ics.append(m["mean_cs_ic"])
        # Multi-objective scalar: OOS R² + 0.15 * CS-IC
        return float(np.mean(oos_r2s) + 0.15 * np.mean(cs_ics))

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=20)
    best_params = study.best_params
    print(f"Best Optuna score={study.best_value:.4f} | params={best_params}", flush=True)

    # 2. Evaluate all models on Dev + Sealed Test (Frozen & Expanding)
    def run_suite(suite_name, folds):
        res = {"NaiveMean": [], "TempSectorFormer_GPU": [], "Optuna_Decomposed": [], "SuperEnsemble_GPU": []}
        for f in folds:
            ym = y_all[f.train_idx].mean(axis=0)
            p_naive = np.tile(ym, (len(f.eval_idx), 1))
            p_tf = fit_predict_temp_sector_former(
                macro_df, sec_dfs, y_all, df["Date"], realized, f.train_idx, f.eval_idx, device
            )
            p_opt = fit_predict_decomposed(
                macro_df, sec_dfs, y_all, df["Date"], realized, f.train_idx, f.eval_idx, best_params
            )
            p_sup = 0.70 * p_opt + 0.30 * p_tf
            for k, p in [
                ("NaiveMean", p_naive),
                ("TempSectorFormer_GPU", p_tf),
                ("Optuna_Decomposed", p_opt),
                ("SuperEnsemble_GPU", p_sup),
            ]:
                m = evaluate_predictions(y_all[f.eval_idx], p, ym)
                m["fold"] = f.name
                res[k].append(m)
        summary = {}
        print(f"\n=== {suite_name} ===", flush=True)
        for k, rows in res.items():
            df_r = pd.DataFrame(rows)
            mean_d = df_r.drop(columns=["fold"]).mean().to_dict()
            summary[k] = {"per_fold": rows, "mean": mean_d}
            print(
                f"  {k:22s}: oos_r2={mean_d['oos_r2_vs_train_mean']:+.4f}, rel_r2={mean_d['rel_r2']:+.4f}, "
                f"rmse={mean_d['rmse']:.4f}, ts_ic={mean_d['mean_ts_ic']:+.4f}, cs_ic={mean_d['mean_cs_ic']:+.4f}, "
                f"top2_exc={mean_d['top2_excess_1y_ret']:+.4f}, spread={mean_d['top2_vs_bot2_spread']:+.4f}, "
                f"win={mean_d['top2_win_rate_vs_ew']:.3f}",
                flush=True,
            )
        return summary

    out = {
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "elapsed_sec": None,
        "best_optuna_params": best_params,
        "dev_2012_2019": run_suite("DEV_2012_2019", dev_folds),
        "sealed_test_frozen_2020_2024": run_suite("SEALED_TEST_FROZEN_2020_2024", test_frozen),
        "sealed_test_expanding_2020_2024": run_suite("SEALED_TEST_EXPANDING_2020_2024", test_expand),
    }
    out["elapsed_sec"] = round(time.time() - t0, 1)
    out_path = Path("/kaggle/working/gpu_sweep_results.json") if Path("/kaggle/working").exists() else Path("outputs/gpu_sweep_results.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nDONE in {out['elapsed_sec']}s! Saved {out_path}", flush=True)


if __name__ == "__main__":
    main()
