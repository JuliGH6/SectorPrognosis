"""Strict automated tests proving zero train/validation/test leakage."""

import numpy as np
import pandas as pd
import pytest

from sector_prognosis import HORIZON_DAYS, SECTORS
from sector_prognosis.data import compute_forward_targets, load_raw_merged_df
from sector_prognosis.splits import (
    get_dev_walk_forward_folds,
    get_sealed_test_folds,
    split_inner_train_val,
    validate_no_leakage,
)


def test_data_chronological_and_targets_causal():
    df = load_raw_merged_df("data")
    assert df["Date"].is_monotonic_increasing
    y_df, realized = compute_forward_targets(df, horizon=HORIZON_DAYS)
    # Last HORIZON_DAYS rows must have NaN targets
    assert y_df.iloc[-HORIZON_DAYS:].isna().all().all()
    assert realized.iloc[-HORIZON_DAYS:].isna().all()
    # Check exact forward return calculation on an arbitrary row
    i = 500
    for s in SECTORS:
        expected = (df[s].iloc[i + HORIZON_DAYS] - df[s].iloc[i]) / df[s].iloc[i]
        assert np.isclose(y_df[s].iloc[i], expected, rtol=1e-9)
    assert realized.iloc[i] == df["Date"].iloc[i + HORIZON_DAYS]


def test_dev_folds_strictly_purged_and_pre_2020():
    df = load_raw_merged_df("data")
    y_df, realized = compute_forward_targets(df, horizon=HORIZON_DAYS)
    valid_mask = (~y_df.isna().any(axis=1)).to_numpy(copy=True)
    valid_mask[:500] = False  # simulate lookback warmup

    folds = get_dev_walk_forward_folds(df["Date"], realized, valid_mask)
    assert len(folds) == 4
    for f in folds:
        validate_no_leakage(df["Date"], realized, f.train_idx, f.eval_idx)
        # Dev folds must NEVER touch 2020+
        assert f.eval_end < pd.Timestamp("2020-01-01")
        assert f.train_max_realized_date < f.eval_start
        # Inner train/val split must also be strictly purged
        itr, ival = split_inner_train_val(f.train_idx, df["Date"], realized)
        validate_no_leakage(df["Date"], realized, itr, ival)
        assert np.all(np.isin(itr, f.train_idx))
        assert np.all(np.isin(ival, f.train_idx))


def test_sealed_test_folds_strictly_purged():
    df = load_raw_merged_df("data")
    y_df, realized = compute_forward_targets(df, horizon=HORIZON_DAYS)
    valid_mask = (~y_df.isna().any(axis=1)).to_numpy(copy=True)
    valid_mask[:500] = False

    for frozen in (True, False):
        folds = get_sealed_test_folds(
            df["Date"], realized, valid_mask, strictly_frozen_pre2020=frozen
        )
        assert len(folds) == 2
        for f in folds:
            validate_no_leakage(df["Date"], realized, f.train_idx, f.eval_idx)
            assert f.eval_start >= pd.Timestamp("2020-01-01")
            if frozen:
                assert f.train_max_realized_date < pd.Timestamp("2020-01-01")


def test_validate_no_leakage_catches_violations():
    dates = pd.Series(pd.bdate_range("2010-01-01", periods=1000))
    realized = dates.shift(-250)
    # Violation: gap of only 100 days (< 250)
    with pytest.raises(AssertionError, match="LEAKAGE"):
        validate_no_leakage(dates, realized, np.arange(0, 400), np.arange(500, 700), horizon=250)
    # Valid: gap of 255 days
    validate_no_leakage(dates, realized, np.arange(0, 400), np.arange(655, 740), horizon=250)


def test_features_strictly_causal():
    """Perturbing all rows after index k must produce BITWISE IDENTICAL features for all rows <= k."""
    from sector_prognosis.features import build_all_feature_views

    df = load_raw_merged_df("data").iloc[:900].copy()
    macro_orig, sec_orig, _ = build_all_feature_views(df)

    # Corrupt everything strictly after index k=600
    k = 600
    df_perturbed = df.copy()
    rng = np.random.default_rng(123)
    num_cols = [c for c in df_perturbed.columns if c != "Date"]
    for c in num_cols:
        df_perturbed[c] = df_perturbed[c].astype(float)
        df_perturbed.loc[k + 1 :, c] = rng.uniform(10.0, 5000.0, size=len(df_perturbed) - (k + 1))

    macro_pert, sec_pert, _ = build_all_feature_views(df_perturbed)

    np.testing.assert_array_equal(
        macro_orig.iloc[: k + 1].to_numpy(),
        macro_pert.iloc[: k + 1].to_numpy(),
    )
    for s in SECTORS:
        np.testing.assert_array_equal(
            sec_orig[s].iloc[: k + 1].to_numpy(),
            sec_pert[s].iloc[: k + 1].to_numpy(),
        )


def test_model_never_sees_eval_targets():
    """Corrupting y_all[eval_idx] must produce IDENTICAL predictions from PriorAnchoredDecomposedModel."""
    from sector_prognosis.features import build_all_feature_views
    from sector_prognosis.models_v2 import PriorAnchoredDecomposedModel

    df = load_raw_merged_df("data")
    y_df, realized = compute_forward_targets(df, horizon=HORIZON_DAYS)
    y_all = y_df.to_numpy(dtype=np.float32)
    macro_df, sec_dfs, feat_valid = build_all_feature_views(df)
    valid_mask = feat_valid & (~y_df.isna().any(axis=1)).to_numpy(copy=True)
    fold = get_dev_walk_forward_folds(df["Date"], realized, valid_mask)[0]

    mdl = PriorAnchoredDecomposedModel("catboost", "enet", seed=42)
    pred_clean = mdl.fit_predict(
        macro_df, sec_dfs, y_all, df["Date"], realized, fold.train_idx, fold.eval_idx
    )

    y_corrupted = y_all.copy()
    y_corrupted[fold.eval_idx] = 999.0  # Massive corruption in eval targets
    pred_corrupted = mdl.fit_predict(
        macro_df, sec_dfs, y_corrupted, df["Date"], realized, fold.train_idx, fold.eval_idx
    )
    np.testing.assert_allclose(pred_clean, pred_corrupted, rtol=0, atol=1e-7)


def test_temporal_model_never_sees_eval_targets():
    """Corrupting y_all[eval_idx] must produce IDENTICAL predictions from DecoupledTemporalSectorModel."""
    from sector_prognosis.features import build_all_feature_views
    from sector_prognosis.models_temporal import DecoupledTemporalSectorModel

    df = load_raw_merged_df("data")
    y_df, realized = compute_forward_targets(df, horizon=HORIZON_DAYS)
    y_all = y_df.to_numpy(dtype=np.float32)
    macro_df, sec_dfs, feat_valid = build_all_feature_views(df)
    valid_mask = feat_valid & (~y_df.isna().any(axis=1)).to_numpy(copy=True)
    valid_mask[:500] = False
    fold = get_dev_walk_forward_folds(df["Date"], realized, valid_mask)[0]

    mdl = DecoupledTemporalSectorModel(seeds=(42,), epochs_mkt=5, epochs_alpha=5)
    pred_clean = mdl.fit_predict(
        macro_df, sec_dfs, y_all, df["Date"], realized, fold.train_idx, fold.eval_idx
    )

    y_corrupted = y_all.copy()
    y_corrupted[fold.eval_idx] = -999.0
    pred_corrupted = mdl.fit_predict(
        macro_df, sec_dfs, y_corrupted, df["Date"], realized, fold.train_idx, fold.eval_idx
    )
    np.testing.assert_allclose(pred_clean, pred_corrupted, rtol=0, atol=1e-6)


def test_iteration8_strictly_causal_and_isolated():
    """Verify Iteration 8B auxiliary features are bitwise causal and model never sees eval targets."""
    from sector_prognosis.features import build_all_feature_views
    from sector_prognosis.iteration8 import (
        MARKET_V8_COLS,
        UncertaintyGatedEnsembleV8,
        augment_macro_features_v8,
    )

    df = load_raw_merged_df("data")
    y_df, realized = compute_forward_targets(df, horizon=HORIZON_DAYS)
    y_all = y_df.to_numpy(dtype=np.float32)
    macro_df, sec_dfs, feat_valid = build_all_feature_views(df)
    macro_v8 = augment_macro_features_v8(df, macro_df)

    # Causality check on expanding valuation z-score: truncating at k=1000 must match full df at <= 1000
    k = 1000
    macro_trunc, _, _ = build_all_feature_views(df.iloc[: k + 1].copy())
    macro_v8_trunc = augment_macro_features_v8(df.iloc[: k + 1].copy(), macro_trunc)
    np.testing.assert_allclose(
        macro_v8[MARKET_V8_COLS].iloc[500 : k + 1].to_numpy(),
        macro_v8_trunc[MARKET_V8_COLS].iloc[500 : k + 1].to_numpy(),
        rtol=1e-6,
        atol=1e-6,
    )

    # Target isolation check
    v8_valid = feat_valid & np.isfinite(macro_v8[MARKET_V8_COLS].to_numpy()).all(axis=1)
    valid_mask = v8_valid & (~y_df.isna().any(axis=1)).to_numpy(copy=True)
    valid_mask[:500] = False
    fold = get_dev_walk_forward_folds(df["Date"], realized, valid_mask)[0]

    mdl = UncertaintyGatedEnsembleV8(seeds=(42,), epochs=4)
    p_clean = mdl.fit_predict(macro_v8, sec_dfs, y_all, df["Date"], realized, fold.train_idx, fold.eval_idx)
    y_bad = y_all.copy()
    y_bad[fold.eval_idx] = 777.0
    p_bad = mdl.fit_predict(macro_v8, sec_dfs, y_bad, df["Date"], realized, fold.train_idx, fold.eval_idx)
    np.testing.assert_allclose(p_clean, p_bad, rtol=0, atol=1e-6)


def test_iteration9_strictly_causal_and_isolated():
    """Verify Iteration 9 systemic correlation/drawdown features are bitwise causal and model never sees eval targets."""
    from sector_prognosis.features import build_all_feature_views
    from sector_prognosis.iteration9 import (
        MARKET_V9_COLS,
        BetaScaledUncertaintyEnsembleV9,
        augment_macro_features_v9,
    )

    df = load_raw_merged_df("data")
    y_df, realized = compute_forward_targets(df, horizon=HORIZON_DAYS)
    y_all = y_df.to_numpy(dtype=np.float32)
    macro_df, sec_dfs, feat_valid = build_all_feature_views(df)
    macro_v9 = augment_macro_features_v9(df, macro_df)

    # Bitwise causality check on systemic correlation & drawdown
    k = 1000
    macro_trunc, _, _ = build_all_feature_views(df.iloc[: k + 1].copy())
    macro_v9_trunc = augment_macro_features_v9(df.iloc[: k + 1].copy(), macro_trunc)
    np.testing.assert_allclose(
        macro_v9[MARKET_V9_COLS].iloc[500 : k + 1].to_numpy(),
        macro_v9_trunc[MARKET_V9_COLS].iloc[500 : k + 1].to_numpy(),
        rtol=1e-6,
        atol=1e-6,
    )

    # Target isolation check
    v9_valid = feat_valid & np.isfinite(macro_v9[MARKET_V9_COLS].to_numpy()).all(axis=1)
    valid_mask = v9_valid & (~y_df.isna().any(axis=1)).to_numpy(copy=True)
    valid_mask[:500] = False
    fold = get_dev_walk_forward_folds(df["Date"], realized, valid_mask)[0]

    mdl = BetaScaledUncertaintyEnsembleV9(seeds=(42,), epochs=4)
    p_clean = mdl.fit_predict(macro_v9, sec_dfs, y_all, df["Date"], realized, fold.train_idx, fold.eval_idx)
    y_bad = y_all.copy()
    y_bad[fold.eval_idx] = -888.0
    p_bad = mdl.fit_predict(macro_v9, sec_dfs, y_bad, df["Date"], realized, fold.train_idx, fold.eval_idx)
    np.testing.assert_allclose(p_clean, p_bad, rtol=0, atol=1e-6)





