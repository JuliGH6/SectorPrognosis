"""Data loading and strictly causal target construction."""

from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

from . import HORIZON_DAYS, MACRO_COLS, SECTORS

EXCLUDED_PREFIXES = ("Energy", "Communications", "IT")


def load_raw_merged_df(data_dir: Path | str = "data", use_cache: bool = True) -> pd.DataFrame:
    """Load merged sector + macroeconomic dataframe sorted strictly ascending by Date.

    Uses cached parquet if available for fast iterative experimentation.
    """
    data_dir = Path(data_dir)
    cache_path = data_dir / "merged_clean.parquet"
    if use_cache and cache_path.exists():
        df = pd.read_parquet(cache_path)
        df["Date"] = pd.to_datetime(df["Date"])
        assert df["Date"].is_monotonic_increasing, "Cached dataframe must be sorted ascending!"
        return df

    sector_path = data_dir / "MergedDataSectorPrognosis.xlsx"
    econ_path = data_dir / "EconomicDataforModel.xlsx"
    if not sector_path.exists() or not econ_path.exists():
        raise FileNotFoundError(
            f"Expected Excel files in {data_dir}. Found: {list(data_dir.glob('*'))}"
        )

    df = pd.read_excel(sector_path, engine="calamine")
    econ = pd.read_excel(econ_path, sheet_name="FullData", engine="calamine")

    for prefix in EXCLUDED_PREFIXES:
        df = df.loc[:, ~df.columns.str.startswith(prefix)].copy()

    df["Date"] = pd.to_datetime(df["Date"])
    econ["Date"] = pd.to_datetime(econ["Date"])

    merged = (
        pd.merge(df, econ, on="Date", how="inner")
        .dropna()
        .sort_values("Date", ascending=True)
        .reset_index(drop=True)
    )

    if use_cache:
        merged.to_parquet(cache_path, index=False)

    return merged


def compute_forward_targets(
    df: pd.DataFrame, horizon: int = HORIZON_DAYS
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Compute strictly forward H-day simple returns for each sector.

    At decision index t:
        y[t, s] = (P[t + horizon, s] - P[t, s]) / P[t, s]

    Returns:
        y_df: DataFrame of forward returns indexed identically to df (last `horizon` rows are NaN).
        realized_dates: Series of the date on which each target y[t] is fully realized (Date[t + horizon]).
    """
    assert df["Date"].is_monotonic_increasing, "DataFrame must be sorted chronologically ascending."
    prices = df[SECTORS]
    future_prices = prices.shift(-horizon)
    y_df = (future_prices - prices) / prices
    y_df.columns = SECTORS
    realized_dates = df["Date"].shift(-horizon)
    return y_df, realized_dates


def get_sector_metric_columns(df: pd.DataFrame) -> list[str]:
    """Return all non-Date, non-price, non-macro sector fundamental columns."""
    exclude = {"Date", *SECTORS, *MACRO_COLS}
    return [c for c in df.columns if c not in exclude]
