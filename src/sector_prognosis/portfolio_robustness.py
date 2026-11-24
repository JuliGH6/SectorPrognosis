"""13-Year (2012-2024) Multi-Strategy & Rolling-Start-Date Portfolio Robustness Study.

Eliminates single-entry-date luck by testing:
1. Full 13-year continuous out-of-sample equity curve (2012-01-03 to 2024-12-31)
   built from 7 strictly purged walk-forward models (zero leakage, verified per fold).
2. 96 rolling 5-year start months (2012-01 to 2019-12) and 120 rolling 3-year start months
   (2012-01 to 2021-12) to measure entry-date robustness & win rates.
3. Head-to-head comparison against 5 classic institutional benchmarks (all 10 bps TC, monthly rebal):
   - Equal-Weight 6-Sector ETF
   - Classic 12M-1M Sector Momentum (Softmax)
   - Inverse-Volatility Risk Parity (1 / vol_63d)
   - Value-Only Sector Rotation (-cs_z_comp_valuation)
   - Expanding Historical-Mean Prior (critical ablation: does ML beat static historical winners?)
   - Iter 9: Softmax-Weighted Sector ETFs
   - Iter 9: Risk-Aware Softmax (60%-100% market-timed equity exposure)
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import HORIZON_DAYS, SECTORS
from .data import compute_forward_targets, load_raw_merged_df
from .features import build_all_feature_views
from .iteration9 import MARKET_V9_COLS, BetaScaledUncertaintyEnsembleV9, augment_macro_features_v9
from .splits import make_purged_fold, validate_no_leakage


def _softmax(x: np.ndarray, temp: float = 0.04) -> np.ndarray:
    z = (x - np.max(x)) / temp
    e = np.exp(z)
    return e / np.sum(e)


def simulate_equity_curve(
    daily_rets: np.ndarray,
    weight_series: np.ndarray,
    rebal_every: int = 21,
    tc_bps: float = 10.0,
    initial_capital: float = 1000.0,
) -> tuple[np.ndarray, float]:
    """Simulate daily compounded portfolio with monthly rebalancing & 10 bps turnover cost.

    weight_series[t] is computed using ONLY info available at close of day t and applied to day t+1.
    """
    T = len(daily_rets)
    eq = np.zeros(T, dtype=np.float64)
    eq[0] = initial_capital
    cur_w = weight_series[0].copy()
    eq[0] *= 1.0 - float(np.sum(np.abs(cur_w))) * (tc_bps / 10000.0)
    total_to = 0.0

    for t in range(1, T):
        pret = float(np.dot(cur_w, daily_rets[t]))
        eq[t] = eq[t - 1] * (1.0 + pret)
        if np.sum(cur_w) > 0:
            cur_w = (cur_w * (1.0 + daily_rets[t])) / (1.0 + pret)
        if t % rebal_every == 0:
            target_w = weight_series[t]
            to = float(np.sum(np.abs(target_w - cur_w)))
            total_to += to
            eq[t] *= 1.0 - to * (tc_bps / 10000.0)
            cur_w = target_w.copy()
    return eq, total_to


def run_robustness_study(art_dir: str = "/home/antigravity/.gemini/jetski/brain/v1/default/jetski_remote_agents_3p_devbox/8c52d08b-527c-4ead-ae46-ef8636125548"):
    df = load_raw_merged_df("data")
    y_df, realized = compute_forward_targets(df, horizon=HORIZON_DAYS)
    y_all = y_df.to_numpy(dtype=np.float32)
    macro_df, sec_dfs, feat_valid = build_all_feature_views(df)
    macro_v9 = augment_macro_features_v9(df, macro_df)

    dates = df["Date"]
    v9_valid = feat_valid & np.isfinite(macro_v9[MARKET_V9_COLS].to_numpy()).all(axis=1)
    train_valid_mask = v9_valid & (~y_df.isna().any(axis=1)).to_numpy(copy=True)
    train_valid_mask[:500] = False

    # 7 non-overlapping strictly purged OOS windows covering ALL 13 years (2012-01-03 to 2024-12-31)
    windows = [
        ("OOS_2012_2013", "2012-01-01", "2013-12-31"),
        ("OOS_2014_2015", "2014-01-01", "2015-12-31"),
        ("OOS_2016_2017", "2016-01-01", "2017-12-31"),
        ("OOS_2018_2019", "2018-01-01", "2019-12-31"),
        ("OOS_2020_2021", "2020-01-01", "2021-12-31"),
        ("OOS_2022_2023", "2022-01-01", "2023-12-31"),
        ("OOS_2024", "2024-01-01", "2024-12-31"),
    ]

    mdl = BetaScaledUncertaintyEnsembleV9(seeds=(42, 102, 202), epochs=35)
    eval_indices_all = []
    preds_iter9_all = []
    preds_naive_all = []

    print("=== Generating 13-Year Strictly Purged Out-of-Sample Forecasts (2012-2024) ===", flush=True)
    for wname, start_s, end_s in windows:
        ev_mask = v9_valid & (dates >= pd.Timestamp(start_s)) & (dates <= pd.Timestamp(end_s))
        ev_idx = np.where(ev_mask)[0]
        min_ev = int(ev_idx.min())
        min_ev_date = dates.iloc[min_ev]
        idx_all = np.arange(len(dates))
        tr_mask = train_valid_mask & (idx_all <= min_ev - HORIZON_DAYS - 5) & (realized < min_ev_date)
        tr_idx = np.where(tr_mask)[0]
        validate_no_leakage(dates, realized, tr_idx, ev_idx, horizon=HORIZON_DAYS)
        print(
            f"  {wname:14s}: train<={dates.iloc[tr_idx[-1]].date()} (realized<={realized.iloc[tr_idx[-1]].date()}) -> "
            f"live=[{dates.iloc[ev_idx[0]].date()}..{dates.iloc[ev_idx[-1]].date()}] (n={len(ev_idx)})",
            flush=True,
        )
        p9 = mdl.fit_predict(macro_v9, sec_dfs, y_all, dates, realized, tr_idx, ev_idx)
        ym = y_all[tr_idx].mean(axis=0)
        eval_indices_all.append(ev_idx)
        preds_iter9_all.append(p9)
        preds_naive_all.append(np.tile(ym, (len(ev_idx), 1)))

    bt_idx = np.concatenate(eval_indices_all)
    preds_i9 = np.concatenate(preds_iter9_all, axis=0)
    preds_naive = np.concatenate(preds_naive_all, axis=0)
    bt_dates = dates.iloc[bt_idx].reset_index(drop=True)

    prices_bt = df[SECTORS].iloc[bt_idx].to_numpy(dtype=np.float64)
    daily_rets = np.zeros_like(prices_bt)
    daily_rets[1:] = prices_bt[1:] / prices_bt[:-1] - 1.0
    T = len(bt_dates)

    # Extract strictly causal classic signals on bt_idx
    mom_mat = np.stack([sec_dfs[s]["mom_12m_1m"].iloc[bt_idx].to_numpy() for s in SECTORS], axis=1)
    vol_mat = np.stack([sec_dfs[s]["vol_63d"].iloc[bt_idx].to_numpy() for s in SECTORS], axis=1)
    val_mat = np.stack([-sec_dfs[s]["cs_z_comp_valuation"].iloc[bt_idx].to_numpy() for s in SECTORS], axis=1)

    # Build weight time series (T, 6) for every strategy
    # Note: S&P 500 Cap-Weighted 6-Sector Buy-and-Hold proxy starts at realistic 2012 S&P 500
    # cap shares across [HealthCare(IT), Materials, Industrials, Staples, Discretionary, Utilities]
    # and drifts naturally with market cap (no rebalancing, just like SPY/VOO).
    cap_w_2012 = np.array([0.38, 0.06, 0.18, 0.15, 0.17, 0.06], dtype=np.float64)
    cap_w_2012 /= cap_w_2012.sum()

    W = {}
    W["Cap-Weighted S&P 500 Proxy (B&H)"] = np.tile(cap_w_2012, (T, 1))
    W["Equal-Weight 6-Sector ETF"] = np.full((T, 6), 1.0 / 6.0)
    W["12M-1M Momentum (Softmax)"] = np.array([_softmax(mom_mat[t], temp=0.15) for t in range(T)])
    inv_v = 1.0 / np.clip(vol_mat, 0.05, 2.0)
    W["Inverse-Vol Risk Parity"] = inv_v / inv_v.sum(axis=1, keepdims=True)
    W["Value-Only Rotation"] = np.array([_softmax(val_mat[t], temp=1.5) for t in range(T)])
    W["Naive Historical-Mean Prior"] = np.array([_softmax(preds_naive[t], temp=0.04) for t in range(T)])
    W["Iter 9: Softmax-Weighted"] = np.array([_softmax(preds_i9[t], temp=0.04) for t in range(T)])
    W["Iter 9: Risk-Aware Softmax"] = np.array([
        float(np.clip(0.60 + 4.0 * float(np.mean(preds_i9[t])), 0.60, 1.00)) * _softmax(preds_i9[t], temp=0.04)
        for t in range(T)
    ])

    # 1. Full 13-Year Performance Table (2012-2024)
    equities = {}
    rows_13y = []
    years_13 = (bt_dates.iloc[-1] - bt_dates.iloc[0]).days / 365.25
    for name, w_arr in W.items():
        # Cap-Weighted S&P 500 Proxy is buy-and-hold (drifts naturally with market cap, 0 rebalancing)
        reb = 999999 if "Cap-Weighted" in name else 21
        eq, to = simulate_equity_curve(daily_rets, w_arr, rebal_every=reb, tc_bps=10.0, initial_capital=1000.0)
        equities[name] = eq
        dr = eq[1:] / eq[:-1] - 1.0
        cagr = (eq[-1] / 1000.0) ** (1.0 / years_13) - 1.0
        vol = np.std(dr) * np.sqrt(252.0)
        sharpe = (np.mean(dr) * 252.0) / (vol + 1e-12)
        mdd = float(np.min(eq / np.maximum.accumulate(eq) - 1.0))
        rows_13y.append({
            "Strategy": name,
            "13Y Final ($)": eq[-1],
            "CAGR (%)": cagr * 100,
            "Ann Vol (%)": vol * 100,
            "Sharpe": sharpe,
            "Max DD (%)": mdd * 100,
            "Ann Turnover": to / years_13,
        })

    print("\n=== FULL 13-YEAR OUT-OF-SAMPLE $1,000 BACKTEST (2012-01-03 to 2024-12-31, 10bps TC) ===", flush=True)
    df_13y = pd.DataFrame(rows_13y)
    print(df_13y.to_string(index=False, float_format=lambda x: f"{x:8.2f}"), flush=True)

    # 2. Rolling 5-Year (1260d) and 3-Year (756d) Start-Date Sensitivity Analysis (sampled monthly every 21d)
    def rolling_horizon_table(horizon_days: int, label: str):
        starts = np.arange(0, T - horizon_days, 21)
        cap_eq = equities["Cap-Weighted S&P 500 Proxy (B&H)"]
        cap_ret = (cap_eq[starts + horizon_days] / cap_eq[starts] - 1.0) * 100.0
        ew_eq = equities["Equal-Weight 6-Sector ETF"]
        ew_ret = (ew_eq[starts + horizon_days] / ew_eq[starts] - 1.0) * 100.0
        mom_eq = equities["12M-1M Momentum (Softmax)"]
        mom_ret = (mom_eq[starts + horizon_days] / mom_eq[starts] - 1.0) * 100.0
        naive_eq = equities["Naive Historical-Mean Prior"]
        naive_ret = (naive_eq[starts + horizon_days] / naive_eq[starts] - 1.0) * 100.0

        r_rows = []
        roll_rets = {}
        for name, eq in equities.items():
            ret = (eq[starts + horizon_days] / eq[starts] - 1.0) * 100.0
            roll_rets[name] = ret
            r_rows.append({
                "Strategy": name,
                f"Median {label} ($)": 1000.0 * (1.0 + np.median(ret) / 100.0),
                f"Mean {label} (%)": np.mean(ret),
                f"Worst {label} (%)": np.min(ret),
                "Win% vs CapWt": np.mean(ret > cap_ret) * 100.0,
                "Win% vs EW": np.mean(ret > ew_ret) * 100.0,
                "Win% vs Mom": np.mean(ret > mom_ret) * 100.0,
                "Win% vs Naive": np.mean(ret > naive_ret) * 100.0,
            })
        print(f"\n=== ROLLING {label.upper()} ENTRY-DATE ROBUSTNESS ({len(starts)} distinct start months) ===", flush=True)
        print(pd.DataFrame(r_rows).to_string(index=False, float_format=lambda x: f"{x:8.2f}"), flush=True)
        return starts, roll_rets

    starts_5y, roll_5y = rolling_horizon_table(1260, "5Y")
    starts_3y, roll_3y = rolling_horizon_table(756, "3Y")

    # 3. Generate comprehensive 3-panel robustness chart
    art = Path(art_dir)
    fig, axes = plt.subplots(3, 1, figsize=(13.5, 12.5), gridspec_kw={"height_ratios": [1.8, 1.3, 1.3]})

    # Panel 1: Full 13-Year Log-Scale Growth of $1,000
    ax = axes[0]
    styles = {
        "Cap-Weighted S&P 500 Proxy (B&H)": ("#d62728", "--", 2.2),
        "Equal-Weight 6-Sector ETF": ("#444444", "--", 1.8),
        "12M-1M Momentum (Softmax)": ("#9467bd", "-.", 1.6),
        "Inverse-Vol Risk Parity": ("#8c564b", ":", 1.6),
        "Value-Only Rotation": ("#e377c2", ":", 1.5),
        "Naive Historical-Mean Prior": ("#7f7f7f", "-.", 1.5),
        "Iter 9: Softmax-Weighted": ("#1f77b4", "-", 2.3),
        "Iter 9: Risk-Aware Softmax": ("#2ca02c", "-", 2.3),
    }
    for name, (col, ls, lw) in styles.items():
        eq = equities[name]
        sh = df_13y.loc[df_13y["Strategy"] == name, "Sharpe"].iloc[0]
        ax.plot(bt_dates, eq, label=f"{name}: \${eq[-1]:,.0f} (Sharpe {sh:.2f})", color=col, ls=ls, lw=lw)
    ax.set_yscale("log")
    ax.set_title("13-Year Out-of-Sample Growth of \$1,000 Across Strategies (2012–2024, Monthly Rebal, 10 bps TC, Log Scale)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Portfolio Value (\$)", fontweight="bold")
    ax.legend(loc="upper left", fontsize=8.3, ncol=2, frameon=True)
    ax.grid(True, which="both", alpha=0.25)

    # Panel 2: Rolling 5-Year $1,000 Final Value by Start Date (eliminates entry-date luck!)
    ax = axes[1]
    s_dates_5y = bt_dates.iloc[starts_5y]
    for name in ["Cap-Weighted S&P 500 Proxy (B&H)", "Equal-Weight 6-Sector ETF", "12M-1M Momentum (Softmax)", "Iter 9: Softmax-Weighted", "Iter 9: Risk-Aware Softmax"]:
        col, ls, lw = styles[name]
        val_5y = 1000.0 * (1.0 + roll_5y[name] / 100.0)
        ax.plot(s_dates_5y, val_5y, label=name, color=col, ls=ls, lw=lw)
    ax.set_title(f"Rolling 5-Year Final Value of \$1,000 by Entry Month ({len(starts_5y)} Start Dates: 2012–2019)", fontsize=11.5, fontweight="bold")
    ax.set_ylabel("5-Year Ending Value (\$)", fontweight="bold")
    ax.legend(loc="upper left", fontsize=8.5, ncol=2, frameon=True)
    ax.grid(alpha=0.25)

    # Panel 3: Rolling 3-Year Excess Return vs Cap-Weighted S&P 500 Proxy
    ax = axes[2]
    s_dates_3y = bt_dates.iloc[starts_3y]
    cap_3y = roll_3y["Cap-Weighted S&P 500 Proxy (B&H)"]
    for name in ["Equal-Weight 6-Sector ETF", "12M-1M Momentum (Softmax)", "Naive Historical-Mean Prior", "Iter 9: Softmax-Weighted", "Iter 9: Risk-Aware Softmax"]:
        col, ls, lw = styles[name]
        exc = roll_3y[name] - cap_3y
        win_pct = np.mean(exc > 0) * 100.0
        ax.plot(s_dates_3y, exc, label=f"{name} (Wins {win_pct:.0f}% vs Cap-Wt)", color=col, ls=ls, lw=lw)
    ax.axhline(0.0, color="black", lw=1.0)
    ax.set_title(f"Rolling 3-Year Excess Return vs Cap-Weighted S&P 500 Proxy by Entry Month ({len(starts_3y)} Start Dates: 2012–2021)", fontsize=11.5, fontweight="bold")
    ax.set_ylabel("3Y Excess Return vs Cap-Wt (%)", fontweight="bold")
    ax.set_xlabel("Investment Start Date", fontweight="bold")
    ax.legend(loc="upper left", fontsize=8.5, ncol=2, frameon=True)
    ax.grid(alpha=0.25)

    plt.tight_layout()
    out_img = art / "portfolio_robustness_13yr.png"
    plt.savefig(out_img, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved {out_img}", flush=True)


if __name__ == "__main__":
    run_robustness_study()
