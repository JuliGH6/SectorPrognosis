"""Iteration 4: Parsimonious Prior-Anchored Decomposed Architecture + Diverse Modern Ensemble.

Key innovations:
1. Expanding Sector Prior Anchor: y[t,s] = mu_train[s] + w_m * r_mkt[t] + w_a * r_alpha[t,s]
2. Curated, stationary, low-collinearity feature sets (16 market features, 20 cross-sectional alpha features)
3. Training-only target Winsorization (5%-95%) + mild exponential recency weighting (half-life ~6y)
4. Strictly purged inner-validation shrinkage calibration (zero leakage!)
5. Diverse model families: Ridge, LightGBM, CatBoost, XGBoost, and Modern LayerNorm-SiLU Tabular ResNet
"""

from typing import Dict, List, Tuple

import lightgbm as lgb
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import xgboost as xgb
from catboost import CatBoostRegressor
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.preprocessing import RobustScaler

from . import HORIZON_DAYS, SECTORS
from .data import compute_forward_targets, load_raw_merged_df
from .features import CYCLICAL_SECTORS, build_all_feature_views
from .metrics import evaluate_predictions, summarize_fold_metrics
from .splits import get_dev_walk_forward_folds, split_inner_train_val

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


def extract_curated_market(macro_df: pd.DataFrame, indices: np.ndarray) -> np.ndarray:
    return macro_df.iloc[indices][MARKET_CORE_COLS].to_numpy(dtype=np.float32)


def extract_curated_alpha_panel(
    macro_df: pd.DataFrame, sec_dfs: Dict[str, pd.DataFrame], indices: np.ndarray
) -> Tuple[np.ndarray, List[str]]:
    """Build compact (len(indices)*6, 20) cross-sectional alpha feature matrix."""
    n_dates = len(indices)
    n_sec = len(SECTORS)
    sec_stack = np.stack(
        [sec_dfs[s].iloc[indices][ALPHA_SEC_CORE_COLS].to_numpy(dtype=np.float32) for s in SECTORS],
        axis=1,
    )  # (N, 6, 16)
    mac_inter = macro_df.iloc[indices][ALPHA_MACRO_INTERACT_COLS].to_numpy(dtype=np.float32)
    cyc_sign = np.array(
        [1.0 if s in CYCLICAL_SECTORS else -1.0 for s in SECTORS], dtype=np.float32
    )[None, :, None]
    inter_stack = np.repeat(mac_inter[:, None, :], n_sec, axis=1) * cyc_sign  # (N, 6, 4)

    panel = np.concatenate([sec_stack, inter_stack], axis=2).reshape(n_dates * n_sec, -1)
    names = ALPHA_SEC_CORE_COLS + [f"cyc_x_{c}" for c in ALPHA_MACRO_INTERACT_COLS]
    return panel.astype(np.float32), names


def _recency_weights(n: int, half_life_steps: float = 300.0) -> np.ndarray:
    """Mild exponential recency weights (sum to n), oldest lowest, newest highest."""
    age = np.arange(n - 1, -1, -1, dtype=np.float32)
    w = np.exp(-np.log(2.0) * age / half_life_steps)
    return (w * (n / w.sum())).astype(np.float32)


def _winsorize_train_target(y: np.ndarray, q_low: float = 0.05, q_high: float = 0.95) -> np.ndarray:
    lo, hi = np.quantile(y, [q_low, q_high])
    return np.clip(y, lo, hi).astype(np.float32)


def _fit_scale_clip(X_tr: np.ndarray, X_ev: np.ndarray, clip: float = 3.5) -> Tuple[np.ndarray, np.ndarray]:
    sc = RobustScaler()
    return (
        np.clip(sc.fit_transform(X_tr), -clip, clip).astype(np.float32),
        np.clip(sc.transform(X_ev), -clip, clip).astype(np.float32),
    )


class CompactTabularResNet(nn.Module):
    """Lightweight LayerNorm + SiLU Residual MLP for noisy financial tabular regression."""

    def __init__(self, in_dim: int, hidden_dim: int = 48, dropout: float = 0.35):
        super().__init__()
        self.in_proj = nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.SiLU())
        self.block1 = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.head = nn.Linear(hidden_dim, 1)
        nn.init.normal_(self.head.weight, std=0.01)
        nn.init.zeros_(self.head.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            x = x + 0.05 * torch.randn_like(x)
        h = self.in_proj(x)
        h = self.norm1(h + self.block1(h))
        return self.head(h).squeeze(-1)


def _fit_predict_resnet(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_ev: np.ndarray,
    sample_weight: np.ndarray,
    seeds: Tuple[int, ...] = (42, 102),
    epochs: int = 35,
    lr: float = 2e-3,
    weight_decay: float = 2e-2,
) -> np.ndarray:
    torch.set_num_threads(1)
    X_tr_s, X_ev_s = _fit_scale_clip(X_tr, X_ev, clip=3.5)
    tX = torch.from_numpy(X_tr_s)
    ty = torch.from_numpy(y_tr.astype(np.float32))
    tw = torch.from_numpy(sample_weight.astype(np.float32))
    tXev = torch.from_numpy(X_ev_s)

    preds = []
    n = len(tX)
    bs = min(128, max(32, n // 8))
    for seed in seeds:
        torch.manual_seed(seed)
        np.random.seed(seed)
        net = CompactTabularResNet(X_tr_s.shape[1], hidden_dim=48, dropout=0.35)
        opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=weight_decay)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
        net.train()
        for _ in range(epochs):
            perm = torch.randperm(n)
            for i in range(0, n, bs):
                idx = perm[i : i + bs]
                if len(idx) < 4:
                    continue
                opt.zero_grad()
                p = net(tX[idx])
                loss = (F.huber_loss(p, ty[idx], delta=0.12, reduction="none") * tw[idx]).mean()
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                opt.step()
            sched.step()
        net.eval()
        with torch.no_grad():
            preds.append(net(tXev).numpy())
    return np.mean(preds, axis=0).astype(np.float32)


def _fit_predict_1d(
    kind: str,
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_ev: np.ndarray,
    sample_weight: np.ndarray,
    seed: int = 42,
) -> np.ndarray:
    if kind == "ridge":
        Xtr_s, Xev_s = _fit_scale_clip(X_tr, X_ev, clip=3.0)
        m = Ridge(alpha=500.0)
        m.fit(Xtr_s, y_tr, sample_weight=sample_weight)
        return np.clip(m.predict(Xev_s), -0.30, 0.30).astype(np.float32)
    elif kind == "enet":
        Xtr_s, Xev_s = _fit_scale_clip(X_tr, X_ev, clip=3.0)
        m = ElasticNet(alpha=0.015, l1_ratio=0.3, max_iter=2000, random_state=seed)
        m.fit(Xtr_s, y_tr, sample_weight=sample_weight)
        return np.clip(m.predict(Xev_s), -0.30, 0.30).astype(np.float32)
    elif kind == "lgbm":
        m = lgb.LGBMRegressor(
            objective="huber",
            n_estimators=120,
            learning_rate=0.03,
            max_depth=3,
            num_leaves=6,
            min_child_samples=50,
            subsample=0.7,
            subsample_freq=1,
            colsample_bytree=0.6,
            reg_alpha=3.0,
            reg_lambda=15.0,
            random_state=seed,
            verbose=-1,
        )
        m.fit(X_tr, y_tr, sample_weight=sample_weight)
        return m.predict(X_ev).astype(np.float32)
    elif kind == "catboost":
        m = CatBoostRegressor(
            iterations=220,
            learning_rate=0.03,
            depth=3,
            l2_leaf_reg=25.0,
            subsample=0.7,
            rsm=0.6,
            loss_function="RMSE",
            random_seed=seed,
            verbose=0,
        )
        m.fit(X_tr, y_tr, sample_weight=sample_weight)
        return m.predict(X_ev).astype(np.float32)
    elif kind == "xgb":
        m = xgb.XGBRegressor(
            n_estimators=120,
            learning_rate=0.03,
            max_depth=3,
            min_child_weight=40,
            subsample=0.7,
            colsample_bytree=0.6,
            reg_alpha=3.0,
            reg_lambda=20.0,
            objective="reg:pseudohubererror",
            random_state=seed,
            n_jobs=2,
        )
        m.fit(X_tr, y_tr, sample_weight=sample_weight)
        return m.predict(X_ev).astype(np.float32)
    elif kind == "resnet":
        return _fit_predict_resnet(X_tr, y_tr, X_ev, sample_weight=sample_weight)
    else:
        raise ValueError(f"Unknown model kind: {kind}")


class PriorAnchoredDecomposedModel:
    """Prior-anchored Market + Cross-Sectional Alpha model with optional inner-val shrinkage."""

    def __init__(
        self,
        market_kind: str = "ridge",
        alpha_kind: str = "ridge",
        market_shrinkage: float | None = None,
        alpha_shrinkage: float | None = None,
        train_stride: int = 5,
        recency_half_life_years: float = 8.0,
        seed: int = 42,
    ):
        self.market_kind = market_kind
        self.alpha_kind = alpha_kind
        self.market_shrinkage = market_shrinkage
        self.alpha_shrinkage = alpha_shrinkage
        self.train_stride = train_stride
        self.recency_half_life_years = recency_half_life_years
        self.seed = seed

    def _fit_raw_components(
        self,
        macro_df: pd.DataFrame,
        sec_dfs: Dict[str, pd.DataFrame],
        y_all: np.ndarray,
        tr_idx: np.ndarray,
        ev_idx: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        tr_sub = tr_idx[:: self.train_stride] if self.train_stride > 1 else tr_idx
        sec_prior = y_all[tr_idx].mean(axis=0).astype(np.float32)  # (6,)
        mkt_prior = float(sec_prior.mean())
        alpha_prior = sec_prior - mkt_prior  # (6,)

        # Recency weights (half-life in subsampled steps)
        steps_per_year = 250.0 / max(1, self.train_stride)
        w_date = _recency_weights(len(tr_sub), half_life_steps=self.recency_half_life_years * steps_per_year)

        # 1. Market residual component
        y_mkt_tr = _winsorize_train_target(y_all[tr_sub].mean(axis=1) - mkt_prior)
        X_mkt_tr = extract_curated_market(macro_df, tr_sub)
        X_mkt_ev = extract_curated_market(macro_df, ev_idx)
        pred_mkt_resid = _fit_predict_1d(self.market_kind, X_mkt_tr, y_mkt_tr, X_mkt_ev, w_date, self.seed)

        # 2. Cross-sectional alpha residual component (around expanding sector alpha prior)
        y_alpha_tr_2d = (y_all[tr_sub] - y_all[tr_sub].mean(axis=1, keepdims=True)) - alpha_prior[None, :]
        y_alpha_tr = _winsorize_train_target(y_alpha_tr_2d.reshape(-1))
        w_panel = np.repeat(w_date, len(SECTORS))
        X_a_tr, _ = extract_curated_alpha_panel(macro_df, sec_dfs, tr_sub)
        X_a_ev, _ = extract_curated_alpha_panel(macro_df, sec_dfs, ev_idx)
        pred_a = _fit_predict_1d(self.alpha_kind, X_a_tr, y_alpha_tr, X_a_ev, w_panel, self.seed)
        pred_a_2d = pred_a.reshape(-1, len(SECTORS))
        pred_a_2d = pred_a_2d - pred_a_2d.mean(axis=1, keepdims=True)

        return sec_prior, pred_mkt_resid, pred_a_2d

    def fit_predict(
        self,
        macro_df: pd.DataFrame,
        sec_dfs: Dict[str, pd.DataFrame],
        y_all: np.ndarray,
        dates: pd.Series,
        realized_dates: pd.Series,
        train_idx: np.ndarray,
        eval_idx: np.ndarray,
    ) -> np.ndarray:
        w_m = self.market_shrinkage
        w_a = self.alpha_shrinkage

        # If shrinkage not fixed, calibrate strictly on purged inner validation!
        if w_m is None or w_a is None:
            itr, ival = split_inner_train_val(train_idx, dates, realized_dates, val_fraction=0.20)
            sec_p_i, pm_i, pa_i = self._fit_raw_components(macro_df, sec_dfs, y_all, itr, ival)
            y_val = y_all[ival]
            y_val_mkt_resid = y_val.mean(axis=1) - float(sec_p_i.mean())
            y_val_a_resid = (y_val - y_val.mean(axis=1, keepdims=True)) - (sec_p_i - float(sec_p_i.mean()))[None, :]

            if w_m is None:
                # Optimal non-negative shrinkage w in [0, 0.85] minimizing MSE on purged inner val
                denom = float(np.dot(pm_i, pm_i)) + 1e-8
                w_m = float(np.clip(np.dot(pm_i, y_val_mkt_resid) / denom, 0.0, 0.85))
            if w_a is None:
                denom = float(np.sum(pa_i * pa_i)) + 1e-8
                w_a = float(np.clip(np.sum(pa_i * y_val_a_resid) / denom, 0.0, 0.85))

        sec_prior, pred_mkt_resid, pred_a_2d = self._fit_raw_components(
            macro_df, sec_dfs, y_all, train_idx, eval_idx
        )
        return (sec_prior[None, :] + w_m * pred_mkt_resid[:, None] + w_a * pred_a_2d).astype(np.float32)


def run_iteration_4():
    df = load_raw_merged_df("data")
    y_df, realized = compute_forward_targets(df, horizon=HORIZON_DAYS)
    y_all = y_df.to_numpy(dtype=np.float32)

    macro_df, sec_dfs, feat_valid = build_all_feature_views(df)
    valid_mask = feat_valid & (~y_df.isna().any(axis=1)).to_numpy(copy=True)
    folds = get_dev_walk_forward_folds(df["Date"], realized, valid_mask)

    specs = {
        "V2_Ridge": PriorAnchoredDecomposedModel("ridge", "ridge"),
        "V2_ElasticNet": PriorAnchoredDecomposedModel("enet", "enet"),
        "V2_LGBM": PriorAnchoredDecomposedModel("lgbm", "lgbm"),
        "V2_CatBoost": PriorAnchoredDecomposedModel("catboost", "catboost"),
        "V2_XGB": PriorAnchoredDecomposedModel("xgb", "xgb"),
        "V2_ResNet": PriorAnchoredDecomposedModel("resnet", "resnet"),
    }

    key_cols = [
        "fold", "mean_r2", "oos_r2_vs_train_mean", "rel_r2",
        "rmse", "dir_acc", "mean_ts_ic", "mean_cs_ic",
        "top2_excess_1y_ret", "top2_vs_bot2_spread", "top2_win_rate_vs_ew",
    ]

    all_preds: Dict[str, List[np.ndarray]] = {k: [] for k in specs}
    summary_rows = []

    for name, mdl in specs.items():
        fold_rows = []
        for fold in folds:
            y_tr_mean = y_all[fold.train_idx].mean(axis=0)
            preds = mdl.fit_predict(
                macro_df, sec_dfs, y_all, df["Date"], realized, fold.train_idx, fold.eval_idx
            )
            all_preds[name].append(preds)
            m = evaluate_predictions(y_all[fold.eval_idx], preds, y_train_mean=y_tr_mean)
            m["fold"] = fold.name
            fold_rows.append(m)
        summ = summarize_fold_metrics(fold_rows)
        print("=" * 98)
        print(f"MODEL: {name}")
        print(summ[key_cols].to_string(index=False, float_format=lambda x: f"{x:7.4f}"))
        mean_dict = summ[summ["fold"] == "MEAN"].iloc[0].to_dict()
        mean_dict["model"] = name
        summary_rows.append(mean_dict)

    # Heterogeneous Ensemble: Equal blend of Ridge + ElasticNet + LGBM + CatBoost + XGB + ResNet
    ens_members = ["V2_Ridge", "V2_ElasticNet", "V2_LGBM", "V2_CatBoost", "V2_XGB", "V2_ResNet"]
    ens_rows = []
    for fi, fold in enumerate(folds):
        y_tr_mean = y_all[fold.train_idx].mean(axis=0)
        p_ens = np.mean([all_preds[k][fi] for k in ens_members], axis=0)
        m = evaluate_predictions(y_all[fold.eval_idx], p_ens, y_train_mean=y_tr_mean)
        m["fold"] = fold.name
        ens_rows.append(m)
    summ_ens = summarize_fold_metrics(ens_rows)
    print("=" * 98)
    print("MODEL: V2_DiverseEnsemble (Ridge + ElasticNet + LGBM + CatBoost + XGB + ResNet)")
    print(summ_ens[key_cols].to_string(index=False, float_format=lambda x: f"{x:7.4f}"))
    mean_ens = summ_ens[summ_ens["fold"] == "MEAN"].iloc[0].to_dict()
    mean_ens["model"] = "V2_DiverseEnsemble"
    summary_rows.append(mean_ens)

    print("\n" + "#" * 98)
    print("ITERATION 4 SUMMARY (MEAN ACROSS 4 PURGED DEV FOLDS 2012-2019):")
    comp = pd.DataFrame(summary_rows)
    cols = ["model"] + [c for c in key_cols if c != "fold"]
    print(comp[cols].to_string(index=False, float_format=lambda x: f"{x:7.4f}"))


if __name__ == "__main__":
    run_iteration_4()
