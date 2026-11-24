"""Leakage-free temporal splitting with strict purging for overlapping return horizons."""

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd

from . import HORIZON_DAYS


@dataclass(frozen=True)
class TemporalFold:
    """A single purged train/eval split with explicit date boundaries and leakage assertions."""

    name: str
    train_idx: np.ndarray
    eval_idx: np.ndarray
    train_decision_start: pd.Timestamp
    train_decision_end: pd.Timestamp
    train_max_realized_date: pd.Timestamp
    eval_start: pd.Timestamp
    eval_end: pd.Timestamp


def validate_no_leakage(
    dates: pd.Series,
    realized_dates: pd.Series,
    train_idx: np.ndarray,
    eval_idx: np.ndarray,
    horizon: int = HORIZON_DAYS,
) -> None:
    """Assert mathematically that zero target/feature leakage exists between train and eval."""
    if len(train_idx) == 0 or len(eval_idx) == 0:
        raise ValueError("Train and eval index arrays must be non-empty.")

    # 1. Disjoint indices
    overlap = np.intersect1d(train_idx, eval_idx)
    if len(overlap) > 0:
        raise AssertionError(f"LEAKAGE: {len(overlap)} overlapping indices between train and eval!")

    # 2. Strict chronological ordering with full horizon purge
    max_train_i = int(train_idx.max())
    min_eval_i = int(eval_idx.min())
    if max_train_i + horizon >= min_eval_i:
        raise AssertionError(
            f"LEAKAGE: max(train_idx) + horizon ({max_train_i} + {horizon} = {max_train_i + horizon}) "
            f">= min(eval_idx) ({min_eval_i}). Overlapping return horizon!"
        )

    # 3. Date-level check: every training label must be realized strictly before eval starts
    max_train_realized = realized_dates.iloc[train_idx].max()
    min_eval_date = dates.iloc[eval_idx].min()
    if pd.isna(max_train_realized) or max_train_realized >= min_eval_date:
        raise AssertionError(
            f"LEAKAGE: max train realized date ({max_train_realized}) >= min eval date ({min_eval_date})!"
        )


def make_purged_fold(
    name: str,
    dates: pd.Series,
    realized_dates: pd.Series,
    valid_mask: np.ndarray,
    eval_start_str: str,
    eval_end_str: str,
    horizon: int = HORIZON_DAYS,
    embargo_days: int = 5,
    max_train_decision_end: str | None = None,
) -> TemporalFold:
    """Create a purged expanding-window fold where all train targets are realized before eval_start.

    Args:
        name: Human-readable fold name.
        dates: Ascending Series of decision dates.
        realized_dates: Series of dates when each decision's target is realized (Date[t + horizon]).
        valid_mask: Boolean mask of rows with valid (non-NaN) features and targets.
        eval_start_str: Inclusive start date for evaluation period (YYYY-MM-DD).
        eval_end_str: Inclusive end date for evaluation period (YYYY-MM-DD).
        horizon: Target forecast horizon in trading days (default 250).
        embargo_days: Extra buffer trading days beyond the purge gap (default 5).
        max_train_decision_end: Optional hard cap on training decision dates (e.g. for strictly frozen pre-2020 test).
    """
    eval_start_ts = pd.Timestamp(eval_start_str)
    eval_end_ts = pd.Timestamp(eval_end_str)

    eval_mask = valid_mask & (dates >= eval_start_ts) & (dates <= eval_end_ts)
    eval_idx = np.where(eval_mask)[0]
    if len(eval_idx) == 0:
        raise ValueError(f"Fold {name}: no valid eval samples in [{eval_start_str}, {eval_end_str}]")

    min_eval_i = int(eval_idx.min())
    min_eval_date = dates.iloc[min_eval_i]

    # Strict purge: index <= min_eval_i - horizon - embargo_days AND realized_date < min_eval_date
    idx_all = np.arange(len(dates))
    train_mask = (
        valid_mask
        & (idx_all <= (min_eval_i - horizon - embargo_days))
        & (realized_dates < min_eval_date)
    )
    if max_train_decision_end is not None:
        train_mask = train_mask & (dates <= pd.Timestamp(max_train_decision_end))

    train_idx = np.where(train_mask)[0]
    validate_no_leakage(dates, realized_dates, train_idx, eval_idx, horizon=horizon)

    return TemporalFold(
        name=name,
        train_idx=train_idx,
        eval_idx=eval_idx,
        train_decision_start=dates.iloc[train_idx].min(),
        train_decision_end=dates.iloc[train_idx].max(),
        train_max_realized_date=realized_dates.iloc[train_idx].max(),
        eval_start=dates.iloc[eval_idx].min(),
        eval_end=dates.iloc[eval_idx].max(),
    )


def get_dev_walk_forward_folds(
    dates: pd.Series,
    realized_dates: pd.Series,
    valid_mask: np.ndarray,
    horizon: int = HORIZON_DAYS,
) -> List[TemporalFold]:
    """4 non-overlapping 2-year validation folds (2012-2019).

    Strictly excludes 2020-2024 (the sealed held-out test set).
    """
    windows = [
        ("Val_2012_2013", "2012-01-01", "2013-12-31"),
        ("Val_2014_2015", "2014-01-01", "2015-12-31"),
        ("Val_2016_2017", "2016-01-01", "2017-12-31"),
        ("Val_2018_2019", "2018-01-01", "2019-12-31"),
    ]
    return [
        make_purged_fold(name, dates, realized_dates, valid_mask, start, end, horizon=horizon)
        for name, start, end in windows
    ]


def get_sealed_test_folds(
    dates: pd.Series,
    realized_dates: pd.Series,
    valid_mask: np.ndarray,
    horizon: int = HORIZON_DAYS,
    strictly_frozen_pre2020: bool = True,
) -> List[TemporalFold]:
    """Sealed out-of-sample test folds covering 2020-2024 (COVID + Rate Hikes + Recovery).

    When strictly_frozen_pre2020=True (default), training is capped strictly at pre-2020
    realized outcomes (decision dates <= 2018-12-31) so NO 2020+ data is ever used for
    training even when evaluating 2022-2024.
    """
    cap = "2018-12-31" if strictly_frozen_pre2020 else None
    suffix = "FrozenPre2020" if strictly_frozen_pre2020 else "Expanding"
    windows = [
        (f"Test_2020_2021_{suffix}", "2020-01-01", "2021-12-31"),
        (f"Test_2022_2024_{suffix}", "2022-01-01", "2024-12-31"),
    ]
    return [
        make_purged_fold(
            name,
            dates,
            realized_dates,
            valid_mask,
            start,
            end,
            horizon=horizon,
            max_train_decision_end=cap,
        )
        for name, start, end in windows
    ]


def split_inner_train_val(
    train_idx: np.ndarray,
    dates: pd.Series,
    realized_dates: pd.Series,
    val_fraction: float = 0.15,
    horizon: int = HORIZON_DAYS,
    embargo_days: int = 5,
) -> Tuple[np.ndarray, np.ndarray]:
    """Carve a strictly purged inner validation set from the end of train_idx (for early stopping)."""
    n = len(train_idx)
    n_val = max(60, int(n * val_fraction))
    inner_val_idx = train_idx[-n_val:]
    min_val_i = int(inner_val_idx.min())
    min_val_date = dates.iloc[min_val_i]

    inner_train_idx = train_idx[
        (train_idx <= (min_val_i - horizon - embargo_days))
        & (realized_dates.iloc[train_idx].values < min_val_date)
    ]
    validate_no_leakage(dates, realized_dates, inner_train_idx, inner_val_idx, horizon=horizon)
    return inner_train_idx, inner_val_idx
