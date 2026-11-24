"""Final Evaluation Pipeline: Head-to-head comparison across Dev (2012-2019) and Sealed Test (2020-2024).

Strictly enforces:
- Zero leakage across all splits (validated via validate_no_leakage).
- Final model architectures & hyperparameters are locked strictly from Dev (2012-2019).
- Sealed Test (2020-2024) is evaluated in BOTH:
  (A) Strictly Frozen Pre-2020 Mode (zero 2020+ data ever seen in training!)
  (B) Purged Expanding Walk-Forward Mode (250-day purge gap before each test regime)
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from . import HORIZON_DAYS, SECTORS
from .baseline import build_original_concat_features, train_original_net
from .data import compute_forward_targets, load_raw_merged_df
from .features import build_all_feature_views
from .metrics import evaluate_predictions, summarize_fold_metrics
from .models_v2 import PriorAnchoredDecomposedModel
from .plots import plot_iteration_progression, plot_sector_predictions_grid
from .splits import (
    get_dev_walk_forward_folds,
    get_sealed_test_folds,
    split_inner_train_val,
    validate_no_leakage,
)


def run_final_evaluation(out_dir: str = "outputs") -> None:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    df = load_raw_merged_df("data")
    y_df, realized = compute_forward_targets(df, horizon=HORIZON_DAYS)
    y_all = y_df.to_numpy(dtype=np.float32)

    # Raw feature matrix for OriginalResNetPCA comparison
    feat_cols = [c for c in df.columns if c not in ("Date", *SECTORS)]
    raw_df = df[feat_cols].copy()
    raw_df["Month"] = (df["Date"].dt.month / 12.0).astype(np.float32)
    raw_mat = raw_df.to_numpy(dtype=np.float32)

    macro_df, sec_dfs, feat_valid = build_all_feature_views(df)
    valid_mask = feat_valid & (~y_df.isna().any(axis=1)).to_numpy(copy=True)

    dev_folds = get_dev_walk_forward_folds(df["Date"], realized, valid_mask)
    test_frozen_folds = get_sealed_test_folds(df["Date"], realized, valid_mask, strictly_frozen_pre2020=True)
    test_expand_folds = get_sealed_test_folds(df["Date"], realized, valid_mask, strictly_frozen_pre2020=False)

    # Locked champions selected strictly from Dev (2012-2019)
    mdl_cb = PriorAnchoredDecomposedModel("catboost", "catboost")
    mdl_hybrid = PriorAnchoredDecomposedModel("catboost", "enet")
    mdl_lgbm_cb = PriorAnchoredDecomposedModel("lgbm", "catboost")

    key_cols = [
        "fold", "mean_r2", "oos_r2_vs_train_mean", "rel_r2",
        "rmse", "dir_acc", "mean_ts_ic", "mean_cs_ic",
        "top2_excess_1y_ret", "top2_vs_bot2_spread", "top2_win_rate_vs_ew",
    ]

    def _eval_suite(suite_name: str, folds, plot_prefix: str | None = None):
        print("\n" + "=" * 100, flush=True)
        print(f"EVALUATION SUITE: {suite_name}", flush=True)
        print("=" * 100, flush=True)
        suite_results = {
            "NaiveMean": [],
            "OriginalResNetPCA": [],
            "V2_CatBoost": [],
            "Hybrid_CB_Enet": [],
            "Champion_Blend": [],
        }
        champ_dates, champ_true, champ_pred = [], [], []

        for fold in folds:
            validate_no_leakage(df["Date"], realized, fold.train_idx, fold.eval_idx)
            tr, ev = fold.train_idx, fold.eval_idx
            y_tr_mean = y_all[tr].mean(axis=0)
            print(
                f"  Fold {fold.name}: train=[{fold.train_decision_start.date()}..{fold.train_decision_end.date()}] "
                f"(realized<={fold.train_max_realized_date.date()}) -> eval=[{fold.eval_start.date()}..{fold.eval_end.date()}] (n={len(ev)})",
                flush=True,
            )

            # 1. NaiveMean
            p_naive = np.tile(y_tr_mean, (len(ev), 1))
            m = evaluate_predictions(y_all[ev], p_naive, y_train_mean=y_tr_mean)
            m["fold"] = fold.name
            suite_results["NaiveMean"].append(m)

            # 2. OriginalResNetPCA (leakage-free reproduction)
            itr, ival = split_inner_train_val(tr, df["Date"], realized)
            X_itr = build_original_concat_features(raw_mat, itr, itr)
            X_ival = build_original_concat_features(raw_mat, itr, ival)
            X_ev = build_original_concat_features(raw_mat, itr, ev)
            net = train_original_net(X_itr, y_all[itr], X_ival, y_all[ival])
            with torch.no_grad():
                p_orig = net(torch.from_numpy(X_ev).float()).numpy()
            m = evaluate_predictions(y_all[ev], p_orig, y_train_mean=y_tr_mean)
            m["fold"] = fold.name
            suite_results["OriginalResNetPCA"].append(m)

            # 3. V2_CatBoost
            p_cb = mdl_cb.fit_predict(macro_df, sec_dfs, y_all, df["Date"], realized, tr, ev)
            m = evaluate_predictions(y_all[ev], p_cb, y_train_mean=y_tr_mean)
            m["fold"] = fold.name
            suite_results["V2_CatBoost"].append(m)

            # 4. Hybrid_CB_Enet
            p_hyb = mdl_hybrid.fit_predict(macro_df, sec_dfs, y_all, df["Date"], realized, tr, ev)
            m = evaluate_predictions(y_all[ev], p_hyb, y_train_mean=y_tr_mean)
            m["fold"] = fold.name
            suite_results["Hybrid_CB_Enet"].append(m)

            # 5. Champion_Blend (0.50 CB + 0.35 Hybrid_CB_Enet + 0.15 LGBM_CB)
            p_lgb = mdl_lgbm_cb.fit_predict(macro_df, sec_dfs, y_all, df["Date"], realized, tr, ev)
            p_blend = 0.50 * p_cb + 0.35 * p_hyb + 0.15 * p_lgb
            m = evaluate_predictions(y_all[ev], p_blend, y_train_mean=y_tr_mean)
            m["fold"] = fold.name
            suite_results["Champion_Blend"].append(m)

            champ_dates.append(df["Date"].iloc[ev])
            champ_true.append(y_all[ev])
            champ_pred.append(p_hyb)

        summary_list = []
        for mname, rows in suite_results.items():
            summ = summarize_fold_metrics(rows)
            print(f"\n--- {suite_name} :: {mname} ---", flush=True)
            print(summ[key_cols].to_string(index=False, float_format=lambda x: f"{x:7.4f}"), flush=True)
            mean_row = summ[summ["fold"] == "MEAN"].iloc[0].to_dict()
            mean_row["model"] = mname
            summary_list.append(mean_row)

        comp_df = pd.DataFrame(summary_list)
        if plot_prefix is not None:
            plot_iteration_progression(comp_df, out_path / f"{plot_prefix}_comparison.png")
            plot_sector_predictions_grid(
                pd.concat(champ_dates),
                np.concatenate(champ_true, axis=0),
                np.concatenate(champ_pred, axis=0),
                f"{suite_name}: Hybrid_CB_Enet Out-of-Sample 1Y Sector Forecasts",
                out_path / f"{plot_prefix}_forecasts.png",
            )
        return comp_df

    dev_comp = _eval_suite("DEV_WALK_FORWARD_2012_2019", dev_folds, plot_prefix="dev_2012_2019")
    test_frozen_comp = _eval_suite(
        "SEALED_TEST_FROZEN_PRE2020_2020_2024", test_frozen_folds, plot_prefix="test_frozen_2020_2024"
    )
    test_expand_comp = _eval_suite(
        "SEALED_TEST_EXPANDING_PURGED_2020_2024", test_expand_folds, plot_prefix="test_expanding_2020_2024"
    )

    summary_payload = {
        "dev_2012_2019": dev_comp.to_dict(orient="records"),
        "sealed_test_frozen_pre2020": test_frozen_comp.to_dict(orient="records"),
        "sealed_test_expanding_purged": test_expand_comp.to_dict(orient="records"),
    }
    (out_path / "final_results.json").write_text(json.dumps(summary_payload, indent=2, default=str))
    print(f"\nSaved all plots and metrics to {out_path.resolve()}", flush=True)


if __name__ == "__main__":
    run_final_evaluation()
