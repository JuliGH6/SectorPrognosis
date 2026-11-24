"""Turnkey Live Paper-Trading Evaluation CLI for Newly Exported Data (e.g., 2025-2026).

Usage:
    # Once updated MergedDataSectorPrognosis.xlsx and EconomicDataforModel.xlsx are in data/:
    .venv/bin/python -m sector_prognosis.paper_test --start-date 2025-01-01 --refresh-cache
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from . import HORIZON_DAYS, SECTORS
from .data import compute_forward_targets, load_raw_merged_df
from .features import build_all_feature_views
from .iteration9 import MARKET_V9_COLS, BetaScaledUncertaintyEnsembleV9, augment_macro_features_v9
from .metrics import evaluate_predictions
from .portfolio_robustness import simulate_equity_curve
from .splits import validate_no_leakage


def run_paper_test(start_date: str = "2025-01-01", data_dir: str = "data", refresh_cache: bool = False):
    data_path = Path(data_dir)
    if refresh_cache:
        for f in ("merged_clean.parquet", "aux_prices.parquet"):
            p = data_path / f
            if p.exists():
                p.unlink()

    df = load_raw_merged_df(data_path, use_cache=True)
    dates = df["Date"]
    print(f"Loaded dataset: {dates.min().date()} to {dates.max().date()} ({len(df)} trading days)")

    # Data quality check: warn if HealthCare == IT still exists in raw Excel
    raw_sector_xlsx = data_path / "MergedDataSectorPrognosis.xlsx"
    if raw_sector_xlsx.exists():
        raw_chk = pd.read_excel(raw_sector_xlsx, engine="calamine", usecols=["HealthCare", "IT"])
        if (raw_chk["HealthCare"] == raw_chk["IT"]).all():
            print(
                "WARNING: 'HealthCare' and 'IT' price columns are still identical in MergedDataSectorPrognosis.xlsx!\n"
                "         Replace the 'HealthCare' price column with true S&P 500 Health Care (S5HLTH) for live trading."
            )

    start_ts = pd.Timestamp(start_date)
    if dates.max() < start_ts:
        print(
            f"\nDataset currently ends on {dates.max().date()}, which is before --start-date {start_date}.\n"
            f"Export updated Bloomberg fundamentals (MergedDataSectorPrognosis.xlsx) and macro indicators\n"
            f"(EconomicDataforModel.xlsx) through 2025-2026 into '{data_path}/' and re-run with --refresh-cache."
        )
        return

    y_df, realized = compute_forward_targets(df, horizon=HORIZON_DAYS)
    y_all = y_df.to_numpy(dtype=np.float32)
    macro_df, sec_dfs, feat_valid = build_all_feature_views(df)
    macro_v9 = augment_macro_features_v9(df, macro_df, data_dir=data_path)

    v9_valid = feat_valid & np.isfinite(macro_v9[MARKET_V9_COLS].to_numpy()).all(axis=1)
    bt_idx = np.where(v9_valid & (dates >= start_ts))[0]
    min_bt = int(bt_idx.min())
    min_bt_date = dates.iloc[min_bt]

    train_mask = (
        v9_valid
        & (~y_df.isna().any(axis=1)).to_numpy(copy=True)
        & (np.arange(len(dates)) <= min_bt - HORIZON_DAYS - 5)
        & (realized < min_bt_date)
    )
    train_mask[:500] = False
    train_idx = np.where(train_mask)[0]
    validate_no_leakage(dates, realized, train_idx, bt_idx, horizon=HORIZON_DAYS)

    print(
        f"Training (strictly purged): {dates.iloc[train_idx[0]].date()}..{dates.iloc[train_idx[-1]].date()} "
        f"(outcomes realized <= {realized.iloc[train_idx[-1]].date()}, n={len(train_idx)})"
    )
    print(f"Paper-Trading Window      : {dates.iloc[bt_idx[0]].date()}..{dates.iloc[bt_idx[-1]].date()} (n={len(bt_idx)} days)")

    mdl = BetaScaledUncertaintyEnsembleV9(seeds=(42, 102, 202), epochs=35)
    preds = mdl.fit_predict(macro_v9, sec_dfs, y_all, dates, realized, train_idx, bt_idx)

    # 1. Print latest actionable 1-year sector forecast & suggested S&P 500 ±6% nudge
    latest_date = dates.iloc[bt_idx[-1]].date()
    latest_pred = preds[-1]
    z = (latest_pred - np.mean(latest_pred)) / (np.std(latest_pred) + 1e-8)
    print(f"\n=== Latest Out-of-Sample 1Y Forecast ({latest_date}) ===")
    for s, p_val, z_val in sorted(zip(SECTORS, latest_pred, z), key=lambda x: -x[1]):
        print(f"  {s:22s}: predicted 1Y return = {p_val*100:+6.2f}%  |  active tilt z = {z_val:+.2f}")

    # 2. If >= 250 days have elapsed in the paper window, score realized 1Y forecast accuracy
    realized_bt_mask = ~y_df.iloc[bt_idx].isna().any(axis=1).to_numpy()
    if realized_bt_mask.sum() >= 20:
        ym = y_all[train_idx].mean(axis=0)
        m = evaluate_predictions(y_all[bt_idx[realized_bt_mask]], preds[realized_bt_mask], y_train_mean=ym)
        print(f"\n=== Realized 1Y Forecast Metrics ({realized_bt_mask.sum()} scored days) ===")
        for k in ("oos_r2_vs_train_mean", "rel_r2", "rmse", "mean_ts_ic", "mean_cs_ic", "top2_excess_1y_ret"):
            print(f"  {k:24s}: {m[k]:+.4f}")

    # 3. Daily compounded $1,000 paper-trading simulation (Cap-Weighted S&P 500 vs ±6% Iter 9 Nudge)
    prices_bt = df[SECTORS].iloc[bt_idx].to_numpy(dtype=np.float64)
    daily_rets = np.zeros_like(prices_bt)
    daily_rets[1:] = prices_bt[1:] / prices_bt[:-1] - 1.0
    T = len(bt_idx)

    # Approximate current S&P 500 6-sector cap shares at start of paper window
    cap_w0 = np.array([0.48, 0.04, 0.14, 0.10, 0.19, 0.05], dtype=np.float64)
    cap_w0 /= cap_w0.sum()
    cap_w = np.zeros((T, 6))
    cap_w[0] = cap_w0
    for t in range(1, T):
        d = cap_w[t - 1] * (1.0 + daily_rets[t])
        cap_w[t] = d / d.sum()

    tilt_w = np.zeros((T, 6))
    for t in range(T):
        zt = (preds[t] - np.mean(preds[t])) / (np.std(preds[t]) + 1e-8)
        w = np.clip(cap_w[t] + 0.06 * zt, 0.0, 1.0)
        tilt_w[t] = w / w.sum()

    eq_base, _ = simulate_equity_curve(daily_rets, cap_w, rebal_every=999999, tc_bps=10.0)
    eq_tilt, _ = simulate_equity_curve(daily_rets, tilt_w, rebal_every=21, tc_bps=10.0)
    print(f"\n=== Paper-Trading $1,000 Result ({dates.iloc[bt_idx[0]].date()} to {dates.iloc[bt_idx[-1]].date()}, 10 bps TC) ===")
    print(f"  Cap-Weighted S&P 500 Base       : ${eq_base[-1]:,.2f} ({(eq_base[-1]/1000-1)*100:+.2f}%)")
    print(f"  S&P 500 + Iter 9 Nudge (±6%)    : ${eq_tilt[-1]:,.2f} ({(eq_tilt[-1]/1000-1)*100:+.2f}%)")


def main():
    ap = argparse.ArgumentParser(description="Run zero-leakage live paper-trading test.")
    ap.add_argument("--start-date", default="2025-01-01", help="Paper-trading start date (YYYY-MM-DD)")
    ap.add_argument("--data-dir", default="data", help="Directory with Excel datasets")
    ap.add_argument("--refresh-cache", action="store_true", help="Rebuild parquet cache from updated Excel files")
    args = ap.parse_args()
    run_paper_test(args.start_date, args.data_dir, args.refresh_cache)


if __name__ == "__main__":
    main()
