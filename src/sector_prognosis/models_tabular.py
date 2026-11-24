"""Modern tabular & panel models for Decomposed Sector Return Forecasting.

Decomposes 1Y sector return into:
    y[t, s] = y_mkt[t] + alpha[t, s]
where y_mkt[t] = mean_s(y[t, s]) and alpha[t, s] = y[t, s] - y_mkt[t].
All scalers, models, and shrinkage parameters are fit strictly on training indices.
"""

from dataclasses import dataclass
from typing import Dict, Tuple

import lightgbm as lgb
import numpy as np
import xgboost as xgb
from catboost import CatBoostRegressor
from sklearn.linear_model import HuberRegressor, Ridge
from sklearn.preprocessing import RobustScaler

from . import HORIZON_DAYS, SECTORS
from .data import compute_forward_targets, load_raw_merged_df
from .features import build_all_feature_views, make_panel_arrays
from .metrics import evaluate_predictions, summarize_fold_metrics
from .splits import get_dev_walk_forward_folds, split_inner_train_val


def _clip_scale(
    X_tr: np.ndarray, X_ev: np.ndarray, clip_val: float = 4.0
) -> Tuple[np.ndarray, np.ndarray]:
    """Strictly causal RobustScaler + symmetric Winsorization."""
    scaler = RobustScaler()
    X_tr_s = np.clip(scaler.fit_transform(X_tr), -clip_val, clip_val)
    X_ev_s = np.clip(scaler.transform(X_ev), -clip_val, clip_val)
    return X_tr_s.astype(np.float32), X_ev_s.astype(np.float32)


def _subsample_stride(idx: np.ndarray, stride: int = 5) -> np.ndarray:
    """Subsample overlapping daily training rows (e.g. weekly stride=5) to curb overfitting."""
    if stride <= 1:
        return idx
    return idx[::stride]


class DecomposedTabularModel:
    """Decomposed Market-Level + Pooled Cross-Sectional Sector Alpha forecaster."""

    def __init__(
        self,
        alpha_kind: str = "catboost",
        market_kind: str = "catboost",
        market_shrinkage: float = 0.5,
        train_stride: int = 5,
        seed: int = 42,
    ):
        self.alpha_kind = alpha_kind
        self.market_kind = market_kind
        self.market_shrinkage = market_shrinkage
        self.train_stride = train_stride
        self.seed = seed

    def _fit_predict_market(
        self,
        X_mkt_tr: np.ndarray,
        y_mkt_tr: np.ndarray,
        X_mkt_ev: np.ndarray,
    ) -> np.ndarray:
        """Predict market-level 1Y return with shrinkage toward expanding historical mean."""
        prior = float(np.mean(y_mkt_tr))
        y_resid = y_mkt_tr - prior

        if self.market_kind == "prior":
            return np.full(len(X_mkt_ev), prior, dtype=np.float32)

        if self.market_kind == "ridge":
            Xtr_s, Xev_s = _clip_scale(X_mkt_tr, X_mkt_ev, clip_val=3.0)
            m = Ridge(alpha=5000.0)
            m.fit(Xtr_s, y_resid)
            pred_resid = np.clip(m.predict(Xev_s), -0.25, 0.25)
        elif self.market_kind == "lgbm":
            m = lgb.LGBMRegressor(
                objective="huber",
                n_estimators=120,
                learning_rate=0.03,
                max_depth=3,
                num_leaves=6,
                min_child_samples=40,
                subsample=0.7,
                subsample_freq=1,
                colsample_bytree=0.5,
                reg_alpha=5.0,
                reg_lambda=20.0,
                random_state=self.seed,
                verbose=-1,
            )
            m.fit(X_mkt_tr, y_resid)
            pred_resid = m.predict(X_mkt_ev)
        elif self.market_kind == "catboost":
            m = CatBoostRegressor(
                iterations=200,
                learning_rate=0.03,
                depth=4,
                l2_leaf_reg=25.0,
                subsample=0.7,
                rsm=0.5,
                loss_function="RMSE",
                random_seed=self.seed,
                verbose=0,
            )
            m.fit(X_mkt_tr, y_resid)
            pred_resid = m.predict(X_mkt_ev)
        elif self.market_kind == "xgb":
            m = xgb.XGBRegressor(
                n_estimators=120,
                learning_rate=0.03,
                max_depth=3,
                min_child_weight=30,
                subsample=0.7,
                colsample_bytree=0.5,
                reg_alpha=5.0,
                reg_lambda=25.0,
                objective="reg:pseudohubererror",
                random_state=self.seed,
                n_jobs=2,
            )
            m.fit(X_mkt_tr, y_resid)
            pred_resid = m.predict(X_mkt_ev)
        else:
            raise ValueError(f"Unknown market_kind: {self.market_kind}")

        return (prior + self.market_shrinkage * pred_resid).astype(np.float32)

    def _fit_predict_alpha(
        self,
        X_panel_tr: np.ndarray,
        y_alpha_tr: np.ndarray,
        X_panel_ev: np.ndarray,
    ) -> np.ndarray:
        """Predict cross-sectional sector alpha using pooled panel model."""
        if self.alpha_kind == "ridge":
            Xtr_s, Xev_s = _clip_scale(X_panel_tr, X_panel_ev, clip_val=4.0)
            m = Ridge(alpha=5000.0)
            m.fit(Xtr_s, y_alpha_tr)
            pred = m.predict(Xev_s)
        elif self.alpha_kind == "huber":
            Xtr_s, Xev_s = _clip_scale(X_panel_tr, X_panel_ev, clip_val=4.0)
            m = HuberRegressor(alpha=1000.0, max_iter=300)
            m.fit(Xtr_s, y_alpha_tr)
            pred = m.predict(Xev_s)
        elif self.alpha_kind == "lgbm":
            m = lgb.LGBMRegressor(
                objective="huber",
                n_estimators=200,
                learning_rate=0.03,
                max_depth=4,
                num_leaves=10,
                min_child_samples=60,
                subsample=0.7,
                subsample_freq=1,
                colsample_bytree=0.4,
                reg_alpha=5.0,
                reg_lambda=25.0,
                random_state=self.seed,
                verbose=-1,
            )
            m.fit(X_panel_tr, y_alpha_tr)
            pred = m.predict(X_panel_ev)
        elif self.alpha_kind == "catboost":
            m = CatBoostRegressor(
                iterations=350,
                learning_rate=0.03,
                depth=4,
                l2_leaf_reg=25.0,
                subsample=0.7,
                rsm=0.4,
                loss_function="RMSE",
                random_seed=self.seed,
                verbose=0,
            )
            m.fit(X_panel_tr, y_alpha_tr)
            pred = m.predict(X_panel_ev)
        elif self.alpha_kind == "xgb":
            m = xgb.XGBRegressor(
                n_estimators=200,
                learning_rate=0.03,
                max_depth=3,
                min_child_weight=40,
                subsample=0.7,
                colsample_bytree=0.4,
                reg_alpha=5.0,
                reg_lambda=25.0,
                objective="reg:pseudohubererror",
                random_state=self.seed,
                n_jobs=2,
            )
            m.fit(X_panel_tr, y_alpha_tr)
            pred = m.predict(X_panel_ev)
        else:
            raise ValueError(f"Unknown alpha_kind: {self.alpha_kind}")

        # Reshape to (N_ev, 6) and enforce exact zero-sum cross-sectional alpha
        pred_2d = pred.reshape(-1, len(SECTORS)).astype(np.float32)
        pred_2d = pred_2d - pred_2d.mean(axis=1, keepdims=True)
        return pred_2d

    def fit_predict(
        self,
        macro_df,
        sec_dfs,
        y_all: np.ndarray,
        train_idx: np.ndarray,
        eval_idx: np.ndarray,
    ) -> np.ndarray:
        tr_sub = _subsample_stride(train_idx, self.train_stride)

        # Market target & features
        y_mkt_tr = y_all[tr_sub].mean(axis=1)
        X_mkt_tr = macro_df.iloc[tr_sub].to_numpy(dtype=np.float32)
        X_mkt_ev = macro_df.iloc[eval_idx].to_numpy(dtype=np.float32)
        pred_mkt = self._fit_predict_market(X_mkt_tr, y_mkt_tr, X_mkt_ev)  # (N_ev,)

        # Cross-sectional alpha target & panel features
        y_alpha_tr_2d = y_all[tr_sub] - y_mkt_tr[:, None]
        y_alpha_tr = y_alpha_tr_2d.reshape(-1)
        X_panel_tr, _ = make_panel_arrays(macro_df, sec_dfs, tr_sub)
        X_panel_ev, _ = make_panel_arrays(macro_df, sec_dfs, eval_idx)
        pred_alpha = self._fit_predict_alpha(X_panel_tr, y_alpha_tr, X_panel_ev)  # (N_ev, 6)

        return pred_mkt[:, None] + pred_alpha


class DirectPanelLGBM:
    """Ablation: Pooled Panel LightGBM predicting raw 1Y return directly (no target decomposition)."""

    def __init__(self, train_stride: int = 5, seed: int = 42):
        self.train_stride = train_stride
        self.seed = seed

    def fit_predict(self, macro_df, sec_dfs, y_all, train_idx, eval_idx) -> np.ndarray:
        tr_sub = _subsample_stride(train_idx, self.train_stride)
        X_tr, _ = make_panel_arrays(macro_df, sec_dfs, tr_sub)
        X_ev, _ = make_panel_arrays(macro_df, sec_dfs, eval_idx)
        y_tr = y_all[tr_sub].reshape(-1)
        m = lgb.LGBMRegressor(
            objective="huber",
            n_estimators=200,
            learning_rate=0.03,
            max_depth=4,
            num_leaves=10,
            min_child_samples=60,
            subsample=0.7,
            subsample_freq=1,
            colsample_bytree=0.4,
            reg_alpha=5.0,
            reg_lambda=25.0,
            random_state=self.seed,
            verbose=-1,
        )
        m.fit(X_tr, y_tr)
        return m.predict(X_ev).reshape(-1, len(SECTORS)).astype(np.float32)


def run_iteration_1_2():
    df = load_raw_merged_df("data")
    y_df, realized = compute_forward_targets(df, horizon=HORIZON_DAYS)
    y_all = y_df.to_numpy(dtype=np.float32)

    macro_df, sec_dfs, feat_valid = build_all_feature_views(df)
    valid_mask = feat_valid & (~y_df.isna().any(axis=1)).to_numpy(copy=True)
    folds = get_dev_walk_forward_folds(df["Date"], realized, valid_mask)

    models = {
        "Direct_PanelLGBM": DirectPanelLGBM(),
        "Decomp_Ridge_PriorMkt": DecomposedTabularModel("ridge", "prior", 0.0),
        "Decomp_Ridge_RidgeMkt": DecomposedTabularModel("ridge", "ridge", 0.5),
        "Decomp_LGBM_LGBMMkt": DecomposedTabularModel("lgbm", "lgbm", 0.5),
        "Decomp_CatBoost_CBMkt": DecomposedTabularModel("catboost", "catboost", 0.5),
        "Decomp_XGB_XGBMkt": DecomposedTabularModel("xgb", "xgb", 0.5),
        "Decomp_CatBoost_PriorMkt": DecomposedTabularModel("catboost", "prior", 0.0),
    }

    key_cols = [
        "fold", "mean_r2", "oos_r2_vs_train_mean", "rel_r2",
        "rmse", "dir_acc", "mean_ts_ic", "mean_cs_ic",
        "top2_excess_1y_ret", "top2_vs_bot2_spread", "top2_win_rate_vs_ew",
    ]

    summary_rows = []
    for name, mdl in models.items():
        fold_rows = []
        for fold in folds:
            y_tr_mean = y_all[fold.train_idx].mean(axis=0)
            preds = mdl.fit_predict(macro_df, sec_dfs, y_all, fold.train_idx, fold.eval_idx)
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

    print("\n" + "#" * 98)
    print("ITERATION 1 & 2 COMPARISON (MEAN ACROSS 4 PURGED DEV FOLDS 2012-2019):")
    comp = pd.DataFrame(summary_rows)
    cols = ["model"] + [c for c in key_cols if c != "fold"]
    print(comp[cols].to_string(index=False, float_format=lambda x: f"{x:7.4f}"))


if __name__ == "__main__":
    import pandas as pd

    run_iteration_1_2()
