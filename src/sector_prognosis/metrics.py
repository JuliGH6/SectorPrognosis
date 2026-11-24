"""Evaluation metrics and out-of-sample sector rotation backtesting."""

from typing import Dict

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from . import SECTORS


def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_train_mean: np.ndarray | None = None,
    rebalance_step: int = 21,
) -> Dict[str, float]:
    """Compute comprehensive statistical and economic evaluation metrics.

    Args:
        y_true: Shape (N, 6) actual 1-year forward returns.
        y_pred: Shape (N, 6) predicted 1-year forward returns.
        y_train_mean: Optional shape (6,) historical training mean per sector for OOS R² benchmark.
        rebalance_step: Trading days between non-overlapping portfolio rebalances for rotation metrics.
    """
    assert y_true.shape == y_pred.shape and y_true.shape[1] == len(SECTORS)

    per_sector_r2 = {}
    per_sector_ts_ic = {}
    per_sector_dir = {}

    for j, s in enumerate(SECTORS):
        yt, yp = y_true[:, j], y_pred[:, j]
        per_sector_r2[s] = float(r2_score(yt, yp))
        # Spearman IC (guard against constant predictions)
        if np.std(yp) < 1e-6:
            per_sector_ts_ic[s] = 0.0
        else:
            ic, _ = spearmanr(yt, yp)
            per_sector_ts_ic[s] = 0.0 if np.isnan(ic) else float(ic)
        per_sector_dir[s] = float(np.mean((yt > 0) == (yp > 0)))

    # Out-of-sample R² relative to historical training mean (Campbell & Thompson 2008)
    oos_r2_vs_train_mean = np.nan
    if y_train_mean is not None:
        sse_model = np.sum((y_true - y_pred) ** 2)
        sse_bench = np.sum((y_true - y_train_mean[None, :]) ** 2)
        oos_r2_vs_train_mean = float(1.0 - sse_model / (sse_bench + 1e-12))

    # Cross-sectional IC: on each date t, Spearman corr across the 6 sectors
    cs_ics = []
    for i in range(len(y_true)):
        if np.std(y_pred[i]) < 1e-12 or np.std(y_true[i]) < 1e-12:
            cs_ics.append(0.0)
        else:
            c, _ = spearmanr(y_true[i], y_pred[i])
            cs_ics.append(0.0 if np.isnan(c) else float(c))
    mean_cs_ic = float(np.mean(cs_ics))
    cs_ic_ir = float(np.mean(cs_ics) / (np.std(cs_ics) + 1e-12))

    # Cross-sectional relative R² (demean each row across the 6 sectors)
    yt_rel = y_true - y_true.mean(axis=1, keepdims=True)
    yp_rel = y_pred - y_pred.mean(axis=1, keepdims=True)
    rel_r2 = float(r2_score(yt_rel.ravel(), yp_rel.ravel()))

    # Sector rotation economic evaluation (sampled every `rebalance_step` trading days)
    sub_idx = np.arange(0, len(y_true), rebalance_step)
    top2_rets = []
    bot2_rets = []
    ew_rets = []
    spread_rets = []
    for i in sub_idx:
        order = np.argsort(y_pred[i])
        bot2 = y_true[i, order[:2]].mean()
        top2 = y_true[i, order[-2:]].mean()
        ew = y_true[i].mean()
        top2_rets.append(top2)
        bot2_rets.append(bot2)
        ew_rets.append(ew)
        spread_rets.append(top2 - bot2)

    top2_rets = np.array(top2_rets)
    ew_rets = np.array(ew_rets)
    spread_rets = np.array(spread_rets)
    excess_rets = top2_rets - ew_rets

    metrics = {
        "mean_r2": float(np.mean(list(per_sector_r2.values()))),
        "oos_r2_vs_train_mean": oos_r2_vs_train_mean,
        "rel_r2": rel_r2,
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "dir_acc": float(np.mean(list(per_sector_dir.values()))),
        "mean_ts_ic": float(np.mean(list(per_sector_ts_ic.values()))),
        "mean_cs_ic": mean_cs_ic,
        "cs_ic_ir": cs_ic_ir,
        "top2_mean_1y_ret": float(np.mean(top2_rets)),
        "ew_mean_1y_ret": float(np.mean(ew_rets)),
        "top2_excess_1y_ret": float(np.mean(excess_rets)),
        "top2_vs_bot2_spread": float(np.mean(spread_rets)),
        "top2_win_rate_vs_ew": float(np.mean(excess_rets > 0)),
        "top2_vs_bot2_win_rate": float(np.mean(spread_rets > 0)),
    }
    for s in SECTORS:
        metrics[f"r2_{s}"] = per_sector_r2[s]
        metrics[f"ts_ic_{s}"] = per_sector_ts_ic[s]
    return metrics


def summarize_fold_metrics(fold_results: list[dict]) -> pd.DataFrame:
    """Aggregate per-fold metric dicts into a clean summary DataFrame with Mean row."""
    df = pd.DataFrame(fold_results)
    num_cols = df.select_dtypes(include=[np.number]).columns
    mean_row = {c: df[c].mean() for c in num_cols}
    mean_row["fold"] = "MEAN"
    return pd.concat([df, pd.DataFrame([mean_row])], ignore_index=True)
