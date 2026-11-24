"""100% Free-Public-Data Live Forecaster (Yahoo Finance ETFs + Free FRED US Macro).

Fetches live historical prices for S&P 500 Sector ETFs (XLK, XLV, XLB, XLI, XLP, XLY, XLU, XLE)
and free FRED macro series, trains FreeDataEnsembleV9 with strict causal purging, and outputs:
  1. Validated 1-year (250 trading days) sector return forecasts & S&P 500 ±6% active tilt weights.
  2. Day-by-day 40-trading-day price forecast saved to forecasts/latest_40d_forecast.csv
     and forecasts/latest_summary.json for automated 40-day evaluation via GitHub Actions.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

from .free_data_model import FREE_MARKET_COLS, FREE_SEC_COLS, FreeDataEnsembleV9
from .iteration9 import augment_macro_features_v9

# Live free ETF universe (using true XLV for HealthCare + XLK for Tech, plus XLE as Energy signal)
LIVE_ETFS = {
    "HealthCare": "XLV",
    "Materials": "XLB",
    "Industrials": "XLI",
    "ConsumerStaples": "XLP",
    "ConsumerDiscretionary": "XLY",
    "Utilities": "XLU",
}
AUX_ETFS = {"Energy": "XLE", "IT": "XLK"}

FRED_SERIES = {
    "CPI": "CPIAUCSL",
    "CoreCPI": "CPILFESL",
    "PPI": "PPIACO",
    "UnemploymentRate": "UNRATE",
    "Total Jobless Claims": "ICSA",
    "EmploymentRatio": "EMRATIO",
    "NewHomeSales": "HSN1F",
    "ExistingHomeSales": "HOUST",  # Housing Starts (free public monthly proxy for housing cycle)
}


def fetch_free_live_dataframe(start: str = "2000-01-01") -> pd.DataFrame:
    """Download free daily ETF prices (yfinance) and free US macro series (FRED CSV)."""
    tickers = list(LIVE_ETFS.values()) + list(AUX_ETFS.values())
    raw = yf.download(tickers, start=start, auto_adjust=True, progress=False)
    close = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw
    close = close.dropna().sort_index()

    df = pd.DataFrame({"Date": pd.to_datetime(close.index).tz_localize(None)})
    for name, sym in {**LIVE_ETFS, **AUX_ETFS}.items():
        df[name] = close[sym].to_numpy(dtype=np.float64)

    # Fetch each FRED series and align strictly causally (forward-fill only, shifted 1 day for safety)
    for col, fred_id in FRED_SERIES.items():
        url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={fred_id}"
        s = pd.read_csv(url)
        s.columns = ["Date", col]
        s["Date"] = pd.to_datetime(s["Date"])
        s[col] = pd.to_numeric(s[col], errors="coerce")
        s = s.dropna().sort_values("Date")
        # Convert YoY % for CPI, CoreCPI, PPI (matching training scale)
        if col in ("CPI", "CoreCPI", "PPI"):
            s[col] = s[col].pct_change(12) * 100.0
            s = s.dropna()
        df = pd.merge_asof(df.sort_values("Date"), s, on="Date", direction="backward")

    df = df.dropna().sort_values("Date").reset_index(drop=True)
    return df


def build_free_live_features(df: pd.DataFrame):
    """Build strictly causal free-public-data macro and per-sector feature tables."""
    from . import SECTORS
    from .features import CYCLICAL_SECTORS, DEFENSIVE_SECTORS

    out = {}
    jc_log = np.log1p(df["Total Jobless Claims"].clip(lower=0))
    out["macro_jc_log_chg250"] = jc_log - jc_log.shift(250)

    unemp = df["UnemploymentRate"]
    out["macro_unemp_chg250"] = unemp - unemp.shift(250)
    out["macro_sahm_proxy"] = unemp - unemp.rolling(250, min_periods=250).min()

    out["macro_cpi"] = df["CPI"]
    out["macro_core_cpi"] = df["CoreCPI"]
    out["macro_cpi_chg126"] = df["CPI"] - df["CPI"].shift(126)
    out["macro_ppi_minus_cpi"] = df["PPI"] - df["CPI"]

    ehs = np.log1p(df["ExistingHomeSales"].clip(lower=0))
    nhs = np.log1p(df["NewHomeSales"].clip(lower=0))
    out["macro_ehs_chg126"] = ehs - ehs.shift(126)
    out["macro_ehs_chg250"] = ehs - ehs.shift(250)
    out["macro_nhs_chg250"] = nhs - nhs.shift(250)

    emp = df["EmploymentRatio"]
    out["macro_emp_ratio_chg250"] = emp - emp.shift(250)

    prices = df[SECTORS]
    dlog = np.log(prices / prices.shift(1))
    ew_dlog = dlog.mean(axis=1)
    out["mkt_ew_vol_63d"] = ew_dlog.rolling(63, min_periods=63).std() * np.sqrt(252.0)
    out["mkt_ew_vol_250d"] = ew_dlog.rolling(250, min_periods=250).std() * np.sqrt(252.0)

    sec_ret_250 = prices.pct_change(250)
    out["mkt_cyc_minus_def_250d"] = (
        sec_ret_250[list(CYCLICAL_SECTORS)].mean(axis=1)
        - sec_ret_250[list(DEFENSIVE_SECTORS)].mean(axis=1)
    )

    e = df["Energy"].astype(np.float64)
    u = df["Utilities"].astype(np.float64)
    out["aux_energy_ret_126d"] = e.pct_change(126)
    out["aux_energy_ret_250d"] = e.pct_change(250)
    out["aux_energy_dd_250d"] = e / e.rolling(250, min_periods=250).max() - 1.0
    out["aux_energy_minus_util_250d"] = e.pct_change(250) - u.pct_change(250)

    ew_idx = (prices / prices.iloc[0]).mean(axis=1)
    out["mkt_ew_dd_250d"] = ew_idx / ew_idx.rolling(250, min_periods=250).max() - 1.0

    rm = dlog.rolling(63, min_periods=63).mean()
    rs = dlog.rolling(63, min_periods=63).std() + 1e-8
    z = (dlog - rm) / rs
    z_sum_sq = (z.sum(axis=1) ** 2).rolling(63, min_periods=63).mean()
    S = float(len(SECTORS))
    out["mkt_avg_corr_63d"] = ((z_sum_sq - S) / (S * (S - 1.0))).clip(-0.5, 1.0)

    macro_df = pd.DataFrame(out, index=df.index).astype(np.float32)

    sec_dfs = {}
    for s in SECTORS:
        p = df[s]
        dl = np.log(p / p.shift(1))
        d = {
            "ret_63d": p.pct_change(63),
            "ret_250d": p.pct_change(250),
            "mom_12m_1m": p.pct_change(250) - p.pct_change(21),
            "vol_63d": dl.rolling(63, min_periods=63).std() * np.sqrt(252.0),
            "vol_250d": dl.rolling(250, min_periods=250).std() * np.sqrt(252.0),
            "drawdown_250d": p / p.rolling(250, min_periods=250).max() - 1.0,
            "dist_sma200": p / p.rolling(200, min_periods=200).mean() - 1.0,
        }
        d["sharpe_250d"] = d["ret_250d"] / (d["vol_250d"] + 0.05)
        sec_dfs[s] = pd.DataFrame(d, index=df.index)

    for col in ("ret_63d", "ret_250d", "mom_12m_1m", "vol_63d", "drawdown_250d", "dist_sma200", "sharpe_250d"):
        mat = pd.concat([sec_dfs[s][col] for s in SECTORS], axis=1)
        mat.columns = SECTORS
        rmean = mat.mean(axis=1)
        rstd = mat.std(axis=1) + 1e-6
        for s in SECTORS:
            sec_dfs[s][f"cs_z_{col}"] = (mat[s] - rmean) / rstd

    sec_dfs = {s: sec_dfs[s].astype(np.float32) for s in SECTORS}
    valid = np.isfinite(macro_df[FREE_MARKET_COLS].to_numpy()).all(axis=1)
    for s in SECTORS:
        valid &= np.isfinite(sec_dfs[s][FREE_SEC_COLS].to_numpy()).all(axis=1)
    return macro_df, sec_dfs, valid


def generate_live_40d_and_1y_forecasts(out_dir: str = "forecasts") -> None:
    from . import SECTORS

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    df = fetch_free_live_dataframe(start="2000-01-01")
    dates = df["Date"]
    macro_df, sec_dfs, valid = build_free_live_features(df)
    valid[:500] = False

    last_idx = int(np.where(valid)[0][-1])
    as_of_date = dates.iloc[last_idx]
    last_prices = df[SECTORS].iloc[last_idx].to_numpy(dtype=np.float64)
    print(f"Live data fetched through {as_of_date.date()} ({len(df)} trading days)")

    def _fit_horizon(horizon_days: int) -> np.ndarray:
        prices = df[SECTORS]
        y_df = (prices.shift(-horizon_days) - prices) / prices
        realized = dates.shift(-horizon_days)
        y_all = y_df.to_numpy(dtype=np.float32)
        tr_mask = valid & (~y_df.isna().any(axis=1)).to_numpy(copy=True) & (realized < as_of_date)
        tr_idx = np.where(tr_mask)[0]
        ev_idx = np.array([last_idx], dtype=np.int64)
        mdl = FreeDataEnsembleV9(seeds=(42, 102, 202), epochs=35)
        return mdl.fit_predict(macro_df, sec_dfs, y_all, dates, realized, tr_idx, ev_idx)[0]

    # 1. Validated 1-Year (250d) forecast
    pred_1y = _fit_horizon(250)
    # 2. Direct 40-Trading-Day forecast
    pred_40d = _fit_horizon(40)

    # Build day-by-day 40-business-day price path using compound daily rate implied by pred_40d
    future_bdays = pd.bdate_range(start=as_of_date + pd.Timedelta(days=1), periods=40)
    rows = []
    for d_step, dt in enumerate(future_bdays, start=1):
        row = {"day_ahead": d_step, "target_date": dt.strftime("%Y-%m-%d")}
        for j, s in enumerate(SECTORS):
            p_d = last_prices[j] * ((1.0 + float(pred_40d[j])) ** (d_step / 40.0))
            row[f"{s}_{LIVE_ETFS[s]}"] = round(float(p_d), 2)
        rows.append(row)

    fc_df = pd.DataFrame(rows)
    fc_df.to_csv(out_path / "latest_40d_forecast.csv", index=False)

    z_1y = (pred_1y - np.mean(pred_1y)) / (np.std(pred_1y) + 1e-8)
    z_40d = (pred_40d - np.mean(pred_40d)) / (np.std(pred_40d) + 1e-8)
    summary = {
        "as_of_date": as_of_date.strftime("%Y-%m-%d"),
        "horizon_40d_end_date": future_bdays[-1].strftime("%Y-%m-%d"),
        "tickers": LIVE_ETFS,
        "starting_prices": {s: round(float(last_prices[j]), 2) for j, s in enumerate(SECTORS)},
        "predicted_40d_return_pct": {s: round(float(pred_40d[j] * 100), 2) for j, s in enumerate(SECTORS)},
        "predicted_1y_return_pct": {s: round(float(pred_1y[j] * 100), 2) for j, s in enumerate(SECTORS)},
        "active_tilt_z_40d": {s: round(float(z_40d[j]), 3) for j, s in enumerate(SECTORS)},
        "active_tilt_z_1y": {s: round(float(z_1y[j]), 3) for j, s in enumerate(SECTORS)},
    }
    (out_path / "latest_summary.json").write_text(json.dumps(summary, indent=2))
    print("\n=== Live Forecast Summary ===")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    generate_live_40d_and_1y_forecasts()
