"""Rigorous Statistical Generalization Audit across 12 Years (6 Non-Overlapping Regimes).

Tests whether the final locked architecture genuinely generalizes out-of-sample using:
1. Per-regime OOS R², RMSE, TS-IC, CS-IC, and Top-2 Excess Return across 2012-2024
2. Both Strictly Frozen Pre-2020 (zero 2020+ training data) and Purged Expanding modes
3. Diebold-Mariano (Newey-West HAC, lag=12 monthly blocks) test of forecast error reduction vs NaiveMean
4. Per-sector generalization breakdown across all 6 S&P 500 sectors
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm, spearmanr
from sklearn.metrics import mean_squared_error

from . import HORIZON_DAYS, SECTORS
from .data import compute_forward_targets, load_raw_merged_df
from .features import build_all_feature_views
from .metrics import evaluate_predictions, summarize_fold_metrics
from .models_temporal import DecoupledTemporalSectorModel
from .models_v2 import PriorAnchoredDecomposedModel
from .plots import plot_sector_predictions_grid
from .splits import get_dev_walk_forward_folds, get_sealed_test_folds, validate_no_leakage


def diebold_mariano_nw(e_model: np.ndarray, e_bench: np.ndarray, step: int = 21, max_lag: int = 12) -> dict:
    """Diebold-Mariano test with Newey-West HAC variance on monthly-sampled squared errors.

    Sampling every 21 trading days (~1 month) with lag=12 (1 full year of overlap) accounts
    properly for the 250-day overlapping return horizon autocorrelation.
    Positive DM stat means `e_model` has lower squared error than `e_bench`.
    """
    # Average squared error across the 6 sectors per date, sampled monthly
    se_m = np.mean(e_model[::step] ** 2, axis=1)
    se_b = np.mean(e_bench[::step] ** 2, axis=1)
    d = se_b - se_m  # positive = model beats benchmark
    T = len(d)
    d_mean = float(np.mean(d))
    d_c = d - d_mean

    # Newey-West Bartlett kernel long-run variance
    gamma0 = float(np.dot(d_c, d_c) / T)
    lrv = gamma0
    L = min(max_lag, T - 1)
    for lag in range(1, L + 1):
        w = 1.0 - lag / (L + 1.0)
        gamma_l = float(np.dot(d_c[lag:], d_c[:-lag]) / T)
        lrv += 2.0 * w * gamma_l

    se = np.sqrt(max(lrv, 1e-12) / T)
    dm_stat = float(d_mean / se)
    p_one_sided = float(1.0 - norm.cdf(dm_stat))
    return {
        "n_monthly_obs": int(T),
        "mean_mse_reduction": d_mean,
        "dm_hac_stat": dm_stat,
        "p_value_one_sided": p_one_sided,
    }


def run_generalization_audit(out_dir: str = "outputs/generalization_audit"):
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

    mdl_v6 = DecoupledTemporalSectorModel(seeds=(42, 102, 202))
    mdl_cb = PriorAnchoredDecomposedModel("catboost", "catboost")

    def eval_6_regimes(label: str, folds):
        print("\n" + "=" * 100, flush=True)
        print(f"GENERALIZATION AUDIT: {label} ({len(folds)} non-overlapping 2-year regimes)", flush=True)
        print("=" * 100, flush=True)
        rows = {"NaiveMean": [], "V6_DecoupledTemporal": [], "V6_Plus_CatBoost": []}
        all_true, all_naive, all_v6, all_blend, all_dates = [], [], [], [], []

        for f in folds:
            validate_no_leakage(df["Date"], realized, f.train_idx, f.eval_idx)
            ym = y_all[f.train_idx].mean(axis=0)
            p_n = np.tile(ym, (len(f.eval_idx), 1))
            p_v6, diag = mdl_v6.fit_predict(
                macro_df, sec_dfs, y_all, df["Date"], realized, f.train_idx, f.eval_idx, return_diagnostics=True
            )
            p_cb = mdl_cb.fit_predict(macro_df, sec_dfs, y_all, df["Date"], realized, f.train_idx, f.eval_idx)
            p_bl = 0.65 * p_v6 + 0.35 * p_cb

            print(f"  {f.name:28s}: w_m={diag['w_m']:.2f}, w_a={diag['w_a']:.2f}", flush=True)
            for k, p in [("NaiveMean", p_n), ("V6_DecoupledTemporal", p_v6), ("V6_Plus_CatBoost", p_bl)]:
                m = evaluate_predictions(y_all[f.eval_idx], p, y_train_mean=ym)
                m["fold"] = f.name
                rows[k].append(m)

            all_dates.append(df["Date"].iloc[f.eval_idx])
            all_true.append(y_all[f.eval_idx])
            all_naive.append(p_n)
            all_v6.append(p_v6)
            all_blend.append(p_bl)

        cols = [
            "fold", "oos_r2_vs_train_mean", "rel_r2", "rmse",
            "dir_acc", "mean_ts_ic", "mean_cs_ic", "top2_excess_1y_ret", "top2_win_rate_vs_ew",
        ]
        for k in ("NaiveMean", "V6_DecoupledTemporal", "V6_Plus_CatBoost"):
            print(f"\n--- {label} :: {k} ---", flush=True)
            print(summarize_fold_metrics(rows[k])[cols].to_string(index=False, float_format=lambda x: f"{x:7.4f}"), flush=True)

        yt = np.concatenate(all_true, axis=0)
        pn = np.concatenate(all_naive, axis=0)
        pv = np.concatenate(all_v6, axis=0)
        pb = np.concatenate(all_blend, axis=0)

        dm_v6 = diebold_mariano_nw(yt - pv, yt - pn)
        dm_bl = diebold_mariano_nw(yt - pb, yt - pn)
        print(f"\n  Diebold-Mariano HAC Test (lag=12m) vs NaiveMean:", flush=True)
        print(f"    V6_DecoupledTemporal: DM={dm_v6['dm_hac_stat']:+.3f}, p={dm_v6['p_value_one_sided']:.4f} (n_months={dm_v6['n_monthly_obs']})", flush=True)
        print(f"    V6_Plus_CatBoost    : DM={dm_bl['dm_hac_stat']:+.3f}, p={dm_bl['p_value_one_sided']:.4f} (n_months={dm_bl['n_monthly_obs']})", flush=True)

        # Per-sector OOS R² vs NaiveMean across the entire 12-year OOS span
        print("\n  Per-Sector 12-Year Pooled OOS Skill (V6_DecoupledTemporal vs NaiveMean):", flush=True)
        sec_stats = []
        for j, s in enumerate(SECTORS):
            sse_m = np.sum((yt[:, j] - pv[:, j]) ** 2)
            sse_b = np.sum((yt[:, j] - pn[:, j]) ** 2)
            oos_r2_s = 1.0 - sse_m / (sse_b + 1e-12)
            ic_s = float(np.nan_to_num(spearmanr(yt[:, j], pv[:, j])[0]))
            sec_stats.append({
                "sector": s,
                "oos_r2_vs_naive": oos_r2_s,
                "rmse_v6": float(np.sqrt(mean_squared_error(yt[:, j], pv[:, j]))),
                "rmse_naive": float(np.sqrt(mean_squared_error(yt[:, j], pn[:, j]))),
                "ts_ic": ic_s,
            })
        print(pd.DataFrame(sec_stats).to_string(index=False, float_format=lambda x: f"{x:7.4f}"), flush=True)

        plot_sector_predictions_grid(
            pd.concat(all_dates),
            yt,
            pv,
            f"{label}: V6 Decoupled Temporal-Cross-Sector Forecasts (2012-2024)",
            out_path / f"{label.lower()}_v6_12yr_forecasts.png",
        )
        return {
            "dm_v6": dm_v6,
            "dm_blend": dm_bl,
            "per_sector_v6": sec_stats,
            "v6_folds": rows["V6_DecoupledTemporal"],
            "blend_folds": rows["V6_Plus_CatBoost"],
            "naive_folds": rows["NaiveMean"],
        }

    res_frozen = eval_6_regimes("ALL_6_REGIMES_2012_2024_FROZEN_PRE2020_TEST", dev_folds + test_frozen)
    res_expand = eval_6_regimes("ALL_6_REGIMES_2012_2024_EXPANDING_PURGED", dev_folds + test_expand)

    (out_path / "generalization_audit.json").write_text(
        json.dumps({"frozen_pre2020_test": res_frozen, "expanding_purged": res_expand}, indent=2, default=str)
    )
    print(f"\nSaved generalization audit to {out_path.resolve()}", flush=True)


if __name__ == "__main__":
    run_generalization_audit()
