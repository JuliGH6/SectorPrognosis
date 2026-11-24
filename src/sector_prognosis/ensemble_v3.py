"""Iteration 5: Specialist Decomposed Ensemble (V3) with Multi-Fold Purged Inner Calibration.

Key design choices validated on Purged Dev Folds (2012-2019):
1. Prior-Anchored Decomposition:
       y_hat[t, s] = mu_train[s] + w_m * r_mkt_hat[t] + w_a * r_alpha_hat[t, s]
2. Specialist Market Engine (non-extrapolative bounded learners):
       Ensemble of CatBoost + LightGBM (Huber) + XGBoost (Pseudo-Huber) + ExtraTrees + Full-Batch Tabular ResNet
3. Specialist Cross-Sectional Alpha Engine (sparse factor + non-linear interaction learners):
       Ensemble of ElasticNet (L1/L2) + CatBoost + LightGBM + Full-Batch Tabular ResNet
4. 2-Fold Purged Inner Cross-Validation (strictly inside train_idx with 250d purge)
   to calibrate shrinkage weights (w_m, w_a) without any outer-val or test leakage.
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
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.preprocessing import RobustScaler

from . import HORIZON_DAYS, SECTORS
from .data import compute_forward_targets, load_raw_merged_df
from .features import build_all_feature_views
from .metrics import evaluate_predictions, summarize_fold_metrics
from .models_v2 import (
    CompactTabularResNet,
    _fit_scale_clip,
    _recency_weights,
    _winsorize_train_target,
    extract_curated_alpha_panel,
    extract_curated_market,
)
from .splits import get_dev_walk_forward_folds, validate_no_leakage


def _fit_predict_fast_resnet(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_ev: np.ndarray,
    sample_weight: np.ndarray,
    seeds: Tuple[int, ...] = (42, 102, 202),
    epochs: int = 80,
    lr: float = 5e-3,
    weight_decay: float = 3e-2,
) -> np.ndarray:
    """Full-batch vectorized PyTorch LayerNorm-SiLU Tabular ResNet (<0.3s on CPU!)."""
    torch.set_num_threads(1)
    X_tr_s, X_ev_s = _fit_scale_clip(X_tr, X_ev, clip=3.5)
    tX = torch.from_numpy(X_tr_s)
    ty = torch.from_numpy(y_tr.astype(np.float32))
    tw = torch.from_numpy(sample_weight.astype(np.float32))
    tXev = torch.from_numpy(X_ev_s)

    preds = []
    for seed in seeds:
        torch.manual_seed(seed)
        np.random.seed(seed)
        net = CompactTabularResNet(X_tr_s.shape[1], hidden_dim=48, dropout=0.35)
        opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=weight_decay)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
        net.train()
        for _ in range(epochs):
            opt.zero_grad()
            p = net(tX)
            loss = (F.huber_loss(p, ty, delta=0.12, reduction="none") * tw).mean()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            sched.step()
        net.eval()
        with torch.no_grad():
            preds.append(np.clip(net(tXev).numpy(), -0.30, 0.30))
    return np.mean(preds, axis=0).astype(np.float32)


def _fit_predict_base(
    kind: str,
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_ev: np.ndarray,
    sample_weight: np.ndarray,
    seed: int = 42,
) -> np.ndarray:
    if kind == "enet":
        Xtr_s, Xev_s = _fit_scale_clip(X_tr, X_ev, clip=3.0)
        m = ElasticNet(alpha=0.012, l1_ratio=0.35, max_iter=2000, random_state=seed)
        m.fit(Xtr_s, y_tr, sample_weight=sample_weight)
        return np.clip(m.predict(Xev_s), -0.30, 0.30).astype(np.float32)
    elif kind == "ridge":
        Xtr_s, Xev_s = _fit_scale_clip(X_tr, X_ev, clip=3.0)
        m = Ridge(alpha=800.0)
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
    elif kind == "extratrees":
        m = ExtraTreesRegressor(
            n_estimators=150,
            max_depth=5,
            min_samples_leaf=25,
            max_features=0.6,
            random_state=seed,
            n_jobs=2,
        )
        m.fit(X_tr, y_tr, sample_weight=sample_weight)
        return m.predict(X_ev).astype(np.float32)
    elif kind == "resnet":
        return _fit_predict_fast_resnet(X_tr, y_tr, X_ev, sample_weight=sample_weight)
    else:
        raise ValueError(f"Unknown model kind: {kind}")


def get_purged_inner_splits(
    train_idx: np.ndarray,
    dates: pd.Series,
    realized_dates: pd.Series,
    n_splits: int = 2,
    val_frac: float = 0.18,
    horizon: int = HORIZON_DAYS,
    embargo: int = 5,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Generate expanding purged inner CV splits strictly inside train_idx."""
    n = len(train_idx)
    val_len = max(80, int(n * val_frac))
    splits = []
    for k in range(n_splits):
        end_pos = n - k * val_len
        start_pos = end_pos - val_len
        if start_pos < horizon + embargo + 150:
            break
        ival = train_idx[start_pos:end_pos]
        min_val_i = int(ival.min())
        min_val_date = dates.iloc[min_val_i]
        itr = train_idx[
            (train_idx <= min_val_i - horizon - embargo)
            & (realized_dates.iloc[train_idx].to_numpy() < min_val_date)
        ]
        if len(itr) < 150:
            break
        validate_no_leakage(dates, realized_dates, itr, ival, horizon=horizon)
        splits.append((itr, ival))
    return splits


class SpecialistSectorEnsemble:
    """V3 Specialist Decomposed Ensemble with 2-Fold Purged Inner Calibration."""

    def __init__(
        self,
        market_models: Tuple[str, ...] = ("catboost", "lgbm", "xgb", "extratrees", "resnet"),
        alpha_models: Tuple[str, ...] = ("enet", "catboost", "lgbm", "resnet"),
        train_stride: int = 5,
        recency_half_life_years: float = 8.0,
        seed: int = 42,
    ):
        self.market_models = market_models
        self.alpha_models = alpha_models
        self.train_stride = train_stride
        self.recency_half_life_years = recency_half_life_years
        self.seed = seed

    def _fit_predict_components(
        self,
        macro_df: pd.DataFrame,
        sec_dfs: Dict[str, pd.DataFrame],
        y_all: np.ndarray,
        tr_idx: np.ndarray,
        ev_idx: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        tr_sub = tr_idx[:: self.train_stride] if self.train_stride > 1 else tr_idx
        sec_prior = y_all[tr_idx].mean(axis=0).astype(np.float32)
        mkt_prior = float(sec_prior.mean())
        alpha_prior = sec_prior - mkt_prior

        steps_per_year = 250.0 / max(1, self.train_stride)
        w_date = _recency_weights(len(tr_sub), half_life_steps=self.recency_half_life_years * steps_per_year)

        # 1. Market Timing Engine (equal blend of bounded non-extrapolative learners)
        y_mkt_tr = _winsorize_train_target(y_all[tr_sub].mean(axis=1) - mkt_prior)
        X_mkt_tr = extract_curated_market(macro_df, tr_sub)
        X_mkt_ev = extract_curated_market(macro_df, ev_idx)
        mkt_preds = [
            _fit_predict_base(k, X_mkt_tr, y_mkt_tr, X_mkt_ev, w_date, self.seed)
            for k in self.market_models
        ]
        pred_mkt_resid = np.mean(mkt_preds, axis=0).astype(np.float32)

        # 2. Cross-Sectional Alpha Engine (sparse factor + non-linear learners)
        y_alpha_tr_2d = (y_all[tr_sub] - y_all[tr_sub].mean(axis=1, keepdims=True)) - alpha_prior[None, :]
        y_alpha_tr = _winsorize_train_target(y_alpha_tr_2d.reshape(-1))
        w_panel = np.repeat(w_date, len(SECTORS))
        X_a_tr, _ = extract_curated_alpha_panel(macro_df, sec_dfs, tr_sub)
        X_a_ev, _ = extract_curated_alpha_panel(macro_df, sec_dfs, ev_idx)
        alpha_preds = []
        for k in self.alpha_models:
            pa = _fit_predict_base(k, X_a_tr, y_alpha_tr, X_a_ev, w_panel, self.seed).reshape(-1, len(SECTORS))
            pa = pa - pa.mean(axis=1, keepdims=True)
            alpha_preds.append(pa)
        pred_alpha_resid = np.mean(alpha_preds, axis=0).astype(np.float32)

        return sec_prior, pred_mkt_resid, pred_alpha_resid

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
        # Calibrate shrinkage weights (w_m, w_a) strictly on 2-fold purged inner CV
        inner_splits = get_purged_inner_splits(train_idx, dates, realized_dates, n_splits=2)
        pm_list, ym_list, pa_list, ya_list = [], [], [], []
        for itr, ival in inner_splits:
            sp_i, pm_i, pa_i = self._fit_predict_components(macro_df, sec_dfs, y_all, itr, ival)
            yv = y_all[ival]
            pm_list.append(pm_i)
            ym_list.append(yv.mean(axis=1) - float(sp_i.mean()))
            pa_list.append(pa_i)
            ya_list.append((yv - yv.mean(axis=1, keepdims=True)) - (sp_i - float(sp_i.mean()))[None, :])

        pm_all = np.concatenate(pm_list)
        ym_all = np.concatenate(ym_list)
        pa_all = np.concatenate(pa_list, axis=0)
        ya_all = np.concatenate(ya_list, axis=0)

        w_m = float(np.clip(np.dot(pm_all, ym_all) / (np.dot(pm_all, pm_all) + 1e-8), 0.0, 0.85))
        w_a = float(np.clip(np.sum(pa_all * ya_all) / (np.sum(pa_all * pa_all) + 1e-8), 0.0, 0.85))

        sec_prior, pred_mkt_resid, pred_alpha_resid = self._fit_predict_components(
            macro_df, sec_dfs, y_all, train_idx, eval_idx
        )
        preds = (sec_prior[None, :] + w_m * pred_mkt_resid[:, None] + w_a * pred_alpha_resid).astype(np.float32)

        if return_diagnostics:
            return preds, {"w_m": w_m, "w_a": w_a, "n_inner_splits": len(inner_splits)}
        return preds


def run_iteration_5():
    df = load_raw_merged_df("data")
    y_df, realized = compute_forward_targets(df, horizon=HORIZON_DAYS)
    y_all = y_df.to_numpy(dtype=np.float32)

    macro_df, sec_dfs, feat_valid = build_all_feature_views(df)
    valid_mask = feat_valid & (~y_df.isna().any(axis=1)).to_numpy(copy=True)
    folds = get_dev_walk_forward_folds(df["Date"], realized, valid_mask)

    configs = {
        "V3_FullSpecialistEnsemble": SpecialistSectorEnsemble(
            market_models=("catboost", "lgbm", "xgb", "extratrees", "resnet"),
            alpha_models=("enet", "catboost", "lgbm", "resnet"),
        ),
        "V3_TreePlusEnet": SpecialistSectorEnsemble(
            market_models=("catboost", "lgbm", "extratrees"),
            alpha_models=("enet", "catboost"),
        ),
        "V3_NeuralResNetOnly": SpecialistSectorEnsemble(
            market_models=("resnet",),
            alpha_models=("resnet",),
        ),
    }

    key_cols = [
        "fold", "mean_r2", "oos_r2_vs_train_mean", "rel_r2",
        "rmse", "dir_acc", "mean_ts_ic", "mean_cs_ic",
        "top2_excess_1y_ret", "top2_vs_bot2_spread", "top2_win_rate_vs_ew",
    ]

    summary_rows = []
    for name, mdl in configs.items():
        fold_rows = []
        for fold in folds:
            y_tr_mean = y_all[fold.train_idx].mean(axis=0)
            preds, diag = mdl.fit_predict(
                macro_df, sec_dfs, y_all, df["Date"], realized, fold.train_idx, fold.eval_idx,
                return_diagnostics=True,
            )
            m = evaluate_predictions(y_all[fold.eval_idx], preds, y_train_mean=y_tr_mean)
            m["fold"] = fold.name
            print(f"  [{name}] {fold.name}: w_m={diag['w_m']:.2f}, w_a={diag['w_a']:.2f} -> oos_r2={m['oos_r2_vs_train_mean']:+.4f}, cs_ic={m['mean_cs_ic']:+.4f}, ts_ic={m['mean_ts_ic']:+.4f}", flush=True)
            fold_rows.append(m)
        summ = summarize_fold_metrics(fold_rows)
        print("=" * 98, flush=True)
        print(f"MODEL: {name}", flush=True)
        print(summ[key_cols].to_string(index=False, float_format=lambda x: f"{x:7.4f}"), flush=True)
        mean_dict = summ[summ["fold"] == "MEAN"].iloc[0].to_dict()
        mean_dict["model"] = name
        summary_rows.append(mean_dict)

    print("\n" + "#" * 98, flush=True)
    print("ITERATION 5 SUMMARY (MEAN ACROSS 4 PURGED DEV FOLDS 2012-2019):", flush=True)
    comp = pd.DataFrame(summary_rows)
    cols = ["model"] + [c for c in key_cols if c != "fold"]
    print(comp[cols].to_string(index=False, float_format=lambda x: f"{x:7.4f}"), flush=True)


if __name__ == "__main__":
    run_iteration_5()
