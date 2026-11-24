"""
Financial & statistical evaluation metrics for SectorPrognosis experiments.

Beyond raw R^2 (which can be dominated by regime shifts in average market return),
we measure:
- RMSE & Mean R^2 (direct comparability with original notebooks)
- Directional Accuracy (sign of 1-year return)
- Time-Series Rank IC (Spearman correlation across time per sector)
- Cross-Sectional Rank IC (sector rotation skill: ranking the 6 sectors on each date)
- Long-Short Rotation Spread (realized 1Y return of Top-2 minus Bottom-2 predicted sectors)
- Optional Uncertainty Calibration (performance on lowest-uncertainty 50% of forecasts)
"""

from __future__ import annotations

from typing import Dict, List, Optional
import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import r2_score


def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sector_names: List[str],
    uncertainty: Optional[np.ndarray] = None,
) -> Dict[str, float | Dict[str, float]]:
    """Compute comprehensive out-of-sample metrics."""
    assert y_true.shape == y_pred.shape
    N, S = y_true.shape

    per_sector_r2 = {}
    ts_ics = []
    for i, name in enumerate(sector_names):
        per_sector_r2[name] = float(r2_score(y_true[:, i], y_pred[:, i]))
        rho, _ = spearmanr(y_true[:, i], y_pred[:, i])
        ts_ics.append(0.0 if np.isnan(rho) else float(rho))

    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    dir_acc = float(np.mean((y_pred > 0) == (y_true > 0)))

    # Cross-sectional Rank IC & Long-Short Top-2 vs Bottom-2 spread
    cs_ics = []
    ls_spreads = []
    for t in range(N):
        rho, _ = spearmanr(y_true[t], y_pred[t])
        if not np.isnan(rho):
            cs_ics.append(float(rho))
        order = np.argsort(y_pred[t])
        bottom2 = np.mean(y_true[t, order[:2]])
        top2 = np.mean(y_true[t, order[-2:]])
        ls_spreads.append(float(top2 - bottom2))

    result: Dict[str, float | Dict[str, float]] = {
        "mean_r2": float(np.mean(list(per_sector_r2.values()))),
        "rmse": rmse,
        "dir_acc": dir_acc,
        "ts_ic": float(np.mean(ts_ics)),
        "cs_ic": float(np.mean(cs_ics)) if cs_ics else 0.0,
        "ls_spread": float(np.mean(ls_spreads)),
        "per_sector_r2": per_sector_r2,
    }

    if uncertainty is not None:
        # Mean uncertainty across sectors per date
        day_unc = np.mean(uncertainty, axis=1)
        med = np.median(day_unc)
        low_unc_mask = day_unc <= med
        high_unc_mask = ~low_unc_mask
        result["low_unc_rmse"] = float(
            np.sqrt(np.mean((y_true[low_unc_mask] - y_pred[low_unc_mask]) ** 2))
        )
        result["high_unc_rmse"] = float(
            np.sqrt(np.mean((y_true[high_unc_mask] - y_pred[high_unc_mask]) ** 2))
        )
        result["low_unc_ls_spread"] = float(np.mean(np.array(ls_spreads)[low_unc_mask]))
        result["high_unc_ls_spread"] = float(np.mean(np.array(ls_spreads)[high_unc_mask]))

    return result
