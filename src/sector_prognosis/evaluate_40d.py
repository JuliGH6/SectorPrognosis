"""Automated 40-Day Forecast Evaluator (runs via GitHub Actions after 40 days).

Loads forecasts/latest_40d_forecast.csv and forecasts/latest_summary.json, fetches
actual realized prices from Yahoo Finance, scores day-by-day price & return accuracy,
and writes forecasts/evaluation_report_40d.md for automated GitHub Issue reporting.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import spearmanr

from . import SECTORS
from .live_free_data import LIVE_ETFS


def evaluate_40d_forecast(forecast_dir: str = "forecasts") -> None:
    fdir = Path(forecast_dir)
    summary_path = fdir / "latest_summary.json"
    csv_path = fdir / "latest_40d_forecast.csv"
    if not summary_path.exists() or not csv_path.exists():
        raise FileNotFoundError(
            "No saved forecast found in forecasts/. Run live_free_data.py first."
        )

    meta = json.loads(summary_path.read_text())
    fc_df = pd.read_csv(csv_path)
    as_of = meta["as_of_date"]
    tickers = list(LIVE_ETFS.values())

    raw = yf.download(tickers, start=as_of, auto_adjust=True, progress=False)
    close = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw
    close.index = pd.to_datetime(close.index).tz_localize(None)
    close = close.dropna().sort_index()

    actual_end_date = close.index[-1]
    start_prices = close.iloc[0]
    end_prices = close.iloc[-1]
    n_days_elapsed = len(close) - 1

    # Compare predicted vs actual returns over the elapsed period
    pred_rets, act_rets = [], []
    lines = [
        f"# 40-Day Live Paper-Trading Evaluation Report",
        f"",
        f"- **Forecast Generated On**: `{as_of}`",
        f"- **Evaluated Through**: `{actual_end_date.strftime('%Y-%m-%d')}` (`{n_days_elapsed}` trading days elapsed)",
        f"- **Model**: `FreeDataEnsembleV9` (100% Free Public Data: Yahoo Finance ETFs + FRED Macro)",
        f"",
        f"## Per-Sector Return & Price Comparison",
        f"| Sector (ETF) | Start Price | Predicted End Price | Actual Price | Predicted Ret (%) | Actual Ret (%) | Error (%) | Direction Correct |",
        f"|---|---:|---:|---:|---:|---:|---:|:---:|",
    ]

    # Pick the forecast row corresponding to n_days_elapsed (clamped to 40)
    row_idx = min(max(n_days_elapsed, 1), len(fc_df)) - 1
    fc_row = fc_df.iloc[row_idx]

    dir_hits = 0
    for s in SECTORS:
        sym = LIVE_ETFS[s]
        p0 = float(start_prices[sym])
        p_act = float(end_prices[sym])
        p_pred = float(fc_row[f"{s}_{sym}"])
        r_act = (p_act / p0 - 1.0) * 100.0
        r_pred = (p_pred / p0 - 1.0) * 100.0
        err = r_pred - r_act
        hit = (r_pred > 0) == (r_act > 0)
        dir_hits += int(hit)
        pred_rets.append(r_pred)
        act_rets.append(r_act)
        lines.append(
            f"| {s} (`{sym}`) | ${p0:.2f} | ${p_pred:.2f} | ${p_act:.2f} | {r_pred:+.2f}% | {r_act:+.2f}% | {err:+.2f}% | {'✅' if hit else '❌'} |"
        )

    pred_rets = np.array(pred_rets)
    act_rets = np.array(act_rets)
    mae = float(np.mean(np.abs(pred_rets - act_rets)))
    rmse = float(np.sqrt(np.mean((pred_rets - act_rets) ** 2)))
    cs_ic = float(np.nan_to_num(spearmanr(pred_rets, act_rets)[0]))

    # Simple $1,000 paper trade over the elapsed window: Equal-Weight vs Top-2 vs S&P 500 ±6% Nudge
    ew_ret = float(np.mean(act_rets))
    top2_ret = float(np.mean(act_rets[np.argsort(pred_rets)[-2:]]))
    bot2_ret = float(np.mean(act_rets[np.argsort(pred_rets)[:2]]))

    cap_w = np.array([0.22, 0.05, 0.19, 0.14, 0.34, 0.06], dtype=np.float64)
    cap_w /= cap_w.sum()
    z = (pred_rets - np.mean(pred_rets)) / (np.std(pred_rets) + 1e-8)
    tilt_w = np.clip(cap_w + 0.06 * z, 0.0, 1.0)
    tilt_w /= tilt_w.sum()
    cap_ret = float(np.dot(cap_w, act_rets))
    tilt_ret = float(np.dot(tilt_w, act_rets))

    lines += [
        "",
        "## Summary Metrics",
        f"- **Directional Accuracy**: `{dir_hits}/6` (`{dir_hits/6*100:.1f}%`)",
        f"- **Cross-Sectional Ranking Correlation (CS-IC)**: `{cs_ic:+.3f}`",
        f"- **Return MAE / RMSE**: `{mae:.2f}%` / `{rmse:.2f}%`",
        "",
        "## $1,000 Paper-Trade Snapshot",
        f"- **Equal-Weight 6-Sector ETF**: `${1000*(1+ew_ret/100):,.2f}` (`{ew_ret:+.2f}%`)",
        f"- **Top-2 Predicted Sectors**: `${1000*(1+top2_ret/100):,.2f}` (`{top2_ret:+.2f}%`, spread vs Bottom-2: `{top2_ret-bot2_ret:+.2f}%`)",
        f"- **Cap-Weighted Base**: `${1000*(1+cap_ret/100):,.2f}` (`{cap_ret:+.2f}%`)",
        f"- **Cap-Weighted + ±6% Model Nudge**: `${1000*(1+tilt_ret/100):,.2f}` (`{tilt_ret:+.2f}%`)",
    ]

    report_md = "\n".join(lines) + "\n"
    (fdir / "evaluation_report_40d.md").write_text(report_md)
    print(report_md)


if __name__ == "__main__":
    evaluate_40d_forecast()
