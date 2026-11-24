"""Strictly causal financial, cross-sectional, and macroeconomic feature engineering.

Every feature at index t depends ONLY on data at indices <= t.
"""

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from . import MACRO_COLS, SECTORS

CYCLICAL_SECTORS = {"Materials", "Industrials", "ConsumerDiscretionary"}
DEFENSIVE_SECTORS = {"HealthCare", "ConsumerStaples", "Utilities"}

VALUATION_METRICS = ["PE", "PriceToBook", "PriceToSales", "BestPE", "EV/EBITDA"]
QUALITY_METRICS = ["ROE", "GrossMargin", "EBITDAMargin", "ProfitMargin"]
LEVERAGE_METRICS = ["DebtToEquity", "DebtToEBITDA"]


def _signed_log1p(s: pd.Series) -> pd.Series:
    return np.sign(s) * np.log1p(np.abs(s))


def build_macro_and_market_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build strictly causal macro regime, macro momentum, and market breadth features."""
    assert df["Date"].is_monotonic_increasing
    out: Dict[str, pd.Series] = {}

    # 1. Stabilized macro levels & multi-horizon changes
    jc_log = np.log1p(df["Total Jobless Claims"].clip(lower=0))
    jc_yoy_slog = _signed_log1p(df["JoblessClaimsYoY"])
    out["macro_jc_log"] = jc_log
    out["macro_jc_log_chg63"] = jc_log - jc_log.shift(63)
    out["macro_jc_log_chg250"] = jc_log - jc_log.shift(250)
    out["macro_jc_yoy_slog"] = jc_yoy_slog

    # Sahm-rule-like recession signal: unemployment rise above 1Y rolling minimum
    unemp = df["UnemploymentRate"]
    out["macro_unemp"] = unemp
    out["macro_unemp_chg63"] = unemp - unemp.shift(63)
    out["macro_unemp_chg250"] = unemp - unemp.shift(250)
    out["macro_sahm_proxy"] = unemp - unemp.rolling(250, min_periods=250).min()

    # Growth & ISM business cycle
    for col, short in [("ISMManufacturing", "ism_mfg"), ("ISMService", "ism_srv")]:
        s = df[col]
        out[f"macro_{short}"] = s - 50.0
        out[f"macro_{short}_chg63"] = s - s.shift(63)
        out[f"macro_{short}_chg126"] = s - s.shift(126)
        out[f"macro_{short}_chg250"] = s - s.shift(250)
    out["macro_ism_spread"] = df["ISMManufacturing"] - df["ISMService"]

    gdp = _signed_log1p(df["GDPChained"])
    out["macro_gdp_slog"] = gdp
    out["macro_gdp_chg126"] = gdp - gdp.shift(126)

    # Inflation & corporate margin pressure
    for col, short in [("CPI", "cpi"), ("CoreCPI", "core_cpi"), ("CorePCE", "core_pce"), ("PPI", "ppi")]:
        s = df[col]
        out[f"macro_{short}"] = s
        out[f"macro_{short}_chg63"] = s - s.shift(63)
        out[f"macro_{short}_chg126"] = s - s.shift(126)
    out["macro_ppi_minus_cpi"] = df["PPI"] - df["CPI"]
    out["macro_headline_minus_core_cpi"] = df["CPI"] - df["CoreCPI"]

    # Housing (leading business cycle indicator) & Employment
    for col, short in [("ExistingHomeSales", "ehs"), ("NewHomeSales", "nhs")]:
        s = np.log1p(df[col].clip(lower=0))
        out[f"macro_{short}_log"] = s
        out[f"macro_{short}_chg126"] = s - s.shift(126)
        out[f"macro_{short}_chg250"] = s - s.shift(250)

    emp = df["EmploymentRatio"]
    out["macro_emp_ratio"] = emp
    out["macro_emp_ratio_chg126"] = emp - emp.shift(126)
    out["macro_emp_ratio_chg250"] = emp - emp.shift(250)

    # 2. Equal-weight market & cyclical/defensive breadth features
    prices = df[SECTORS]
    daily_log_ret = np.log(prices / prices.shift(1))
    ew_daily_ret = daily_log_ret.mean(axis=1)

    for w in (21, 63, 126, 250):
        sec_rets = prices.pct_change(w)
        out[f"mkt_ew_ret_{w}d"] = sec_rets.mean(axis=1)
        out[f"mkt_disp_ret_{w}d"] = sec_rets.std(axis=1)
        cyc_ret = sec_rets[list(CYCLICAL_SECTORS)].mean(axis=1)
        def_ret = sec_rets[list(DEFENSIVE_SECTORS)].mean(axis=1)
        out[f"mkt_cyc_minus_def_{w}d"] = cyc_ret - def_ret

    for w in (21, 63, 250):
        out[f"mkt_ew_vol_{w}d"] = ew_daily_ret.rolling(w, min_periods=w).std() * np.sqrt(252.0)
    out["mkt_vol_ratio_21_250"] = out["mkt_ew_vol_21d"] / (out["mkt_ew_vol_250d"] + 1e-4)

    # Market-wide average valuation, quality, leverage
    val_cols = [f"{s} {m}" for s in SECTORS for m in VALUATION_METRICS if f"{s} {m}" in df.columns]
    qual_cols = [f"{s} {m}" for s in SECTORS for m in QUALITY_METRICS if f"{s} {m}" in df.columns]
    lev_cols = [f"{s} {m}" for s in SECTORS for m in LEVERAGE_METRICS if f"{s} {m}" in df.columns]
    out["mkt_avg_valuation"] = df[val_cols].mean(axis=1)
    out["mkt_avg_quality"] = df[qual_cols].mean(axis=1)
    out["mkt_avg_leverage"] = df[lev_cols].mean(axis=1)
    out["mkt_avg_val_chg126"] = out["mkt_avg_valuation"] - out["mkt_avg_valuation"].shift(126)

    # Cyclical calendar features
    month = df["Date"].dt.month.to_numpy(dtype=np.float32)
    out["cal_month_sin"] = pd.Series(np.sin(2.0 * np.pi * month / 12.0), index=df.index)
    out["cal_month_cos"] = pd.Series(np.cos(2.0 * np.pi * month / 12.0), index=df.index)

    return pd.DataFrame(out, index=df.index).astype(np.float32)


def build_per_sector_features(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Build strictly causal per-sector technical, fundamental, and cross-sectional relative features.

    Returns:
        Dict mapping sector_name -> DataFrame of shape (N_dates, F_sector) with identical column names across sectors.
    """
    assert df["Date"].is_monotonic_increasing
    per_sec: Dict[str, Dict[str, pd.Series]] = {s: {} for s in SECTORS}

    for s in SECTORS:
        p = df[s]
        dlog = np.log(p / p.shift(1))
        d = per_sec[s]

        # Multi-horizon returns & momentum
        for w in (21, 63, 126, 250):
            d[f"ret_{w}d"] = p.pct_change(w)
        d["mom_12m_1m"] = d["ret_250d"] - d["ret_21d"]
        d["mom_6m_1m"] = d["ret_126d"] - d["ret_21d"]

        # Volatility & risk-adjusted momentum
        for w in (21, 63, 250):
            d[f"vol_{w}d"] = dlog.rolling(w, min_periods=w).std() * np.sqrt(252.0)
        d["sharpe_63d"] = d["ret_63d"] / (d["vol_63d"] + 0.05)
        d["sharpe_250d"] = d["ret_250d"] / (d["vol_250d"] + 0.05)
        d["vol_ratio_21_250"] = d["vol_21d"] / (d["vol_250d"] + 1e-4)

        # Drawdown & trend
        roll_max_250 = p.rolling(250, min_periods=250).max()
        d["drawdown_250d"] = p / roll_max_250 - 1.0
        sma_200 = p.rolling(200, min_periods=200).mean()
        d["dist_sma200"] = p / sma_200 - 1.0

        # Raw Bloomberg fundamental levels (common across all 6 sectors)
        common_metrics = [
            "Performance", "Dividends", "PE", "PriceToBook", "ROE",
            "PriceToSales", "BestPE", "DebtToEquity", "DebtToEBITDA",
            "GrossMargin", "EBITDAMargin", "ProfitMargin", "EV/EBITDA",
        ]
        for m in common_metrics:
            col = f"{s} {m}"
            d[f"fund_{m}"] = df[col]

        # Composite fundamental factors
        val_comp = df[[f"{s} {m}" for m in VALUATION_METRICS]].mean(axis=1)
        qual_comp = df[[f"{s} {m}" for m in QUALITY_METRICS]].mean(axis=1)
        lev_comp = df[[f"{s} {m}" for m in LEVERAGE_METRICS]].mean(axis=1)
        d["comp_valuation"] = val_comp
        d["comp_quality"] = qual_comp
        d["comp_leverage"] = lev_comp
        d["comp_garp"] = qual_comp - val_comp

        # Fundamental trends (3M and 1Y changes)
        for name, series in [
            ("val", val_comp),
            ("qual", qual_comp),
            ("lev", lev_comp),
            ("div", df[f"{s} Dividends"]),
            ("pe", df[f"{s} PE"]),
            ("roe", df[f"{s} ROE"]),
            ("pm", df[f"{s} ProfitMargin"]),
        ]:
            d[f"chg63_{name}"] = series - series.shift(63)
            d[f"chg250_{name}"] = series - series.shift(250)

        # Cyclical vs Defensive indicator
        d["is_cyclical"] = pd.Series(1.0 if s in CYCLICAL_SECTORS else 0.0, index=df.index)

    # Convert to DataFrames
    sec_dfs = {s: pd.DataFrame(per_sec[s], index=df.index) for s in SECTORS}

    # Add cross-sectional relative features (computed across the 6 sectors at date t ONLY!)
    cs_base_cols = [
        "ret_21d", "ret_63d", "ret_126d", "ret_250d", "mom_12m_1m",
        "vol_63d", "sharpe_250d", "drawdown_250d", "dist_sma200",
        "fund_Dividends", "fund_PE", "fund_BestPE", "fund_ROE", "fund_ProfitMargin", "fund_EV/EBITDA",
        "comp_valuation", "comp_quality", "comp_leverage", "comp_garp",
        "chg63_val", "chg250_val", "chg250_qual",
    ]
    for col in cs_base_cols:
        mat = pd.concat([sec_dfs[s][col] for s in SECTORS], axis=1)
        mat.columns = SECTORS
        row_mean = mat.mean(axis=1)
        row_std = mat.std(axis=1) + 1e-6
        row_rank = mat.rank(axis=1, pct=True)  # in [1/6, 1.0]
        for s in SECTORS:
            sec_dfs[s][f"cs_diff_{col}"] = mat[s] - row_mean
            sec_dfs[s][f"cs_z_{col}"] = (mat[s] - row_mean) / row_std
            sec_dfs[s][f"cs_rank_{col}"] = row_rank[s]

    return {s: sec_dfs[s].astype(np.float32) for s in SECTORS}


def build_all_feature_views(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame], np.ndarray]:
    """Build macro/market features and per-sector feature tables + valid warmup mask.

    Returns:
        macro_df: DataFrame (N_dates, F_macro)
        sec_dfs: Dict[sector -> DataFrame (N_dates, F_sector)]
        feature_valid_mask: Boolean array (N_dates,) True where all features are finite.
    """
    macro_df = build_macro_and_market_features(df)
    sec_dfs = build_per_sector_features(df)

    valid = np.isfinite(macro_df.to_numpy()).all(axis=1)
    for s in SECTORS:
        valid &= np.isfinite(sec_dfs[s].to_numpy()).all(axis=1)

    return macro_df, sec_dfs, valid


def make_panel_arrays(
    macro_df: pd.DataFrame,
    sec_dfs: Dict[str, pd.DataFrame],
    indices: np.ndarray,
) -> Tuple[np.ndarray, List[str]]:
    """Build pooled panel feature matrix of shape (len(indices) * 6, F_panel)
    ordered as [date_0_sec_0, ..., date_0_sec_5, date_1_sec_0, ...].

    Includes sector-specific features, macro features, macro x cyclical interactions,
    and sector one-hot indicators.
    """
    macro_sub = macro_df.iloc[indices].to_numpy(dtype=np.float32)
    macro_cols = list(macro_df.columns)
    sec_cols = list(sec_dfs[SECTORS[0]].columns)

    # Key macro columns for explicit cyclical interaction
    interact_names = [
        "macro_ism_mfg", "macro_ism_mfg_chg126", "macro_sahm_proxy",
        "macro_cpi", "macro_cpi_chg126", "macro_mkt_vol" if "macro_mkt_vol" in macro_cols else "mkt_ew_vol_63d",
        "mkt_ew_ret_250d",
    ]
    interact_idx = [macro_cols.index(c) for c in interact_names]

    n_dates = len(indices)
    n_sec = len(SECTORS)
    sec_stack = np.stack(
        [sec_dfs[s].iloc[indices].to_numpy(dtype=np.float32) for s in SECTORS], axis=1
    )  # (N_dates, 6, F_sec)

    macro_rep = np.repeat(macro_sub[:, None, :], n_sec, axis=1)  # (N_dates, 6, F_macro)

    # Cyclicality sign (+1 cyclical, -1 defensive) * key macro indicators
    cyc_sign = np.array(
        [1.0 if s in CYCLICAL_SECTORS else -1.0 for s in SECTORS], dtype=np.float32
    )[None, :, None]
    macro_inter = macro_rep[:, :, interact_idx] * cyc_sign  # (N_dates, 6, len(interact_idx))

    # Sector one-hot
    onehot = np.broadcast_to(np.eye(n_sec, dtype=np.float32)[None, :, :], (n_dates, n_sec, n_sec))

    panel_3d = np.concatenate([sec_stack, macro_rep, macro_inter, onehot], axis=2)
    panel_2d = panel_3d.reshape(n_dates * n_sec, -1)

    feature_names = (
        sec_cols
        + macro_cols
        + [f"cyc_x_{c}" for c in interact_names]
        + [f"sec_id_{s}" for s in SECTORS]
    )
    return panel_2d, feature_names
