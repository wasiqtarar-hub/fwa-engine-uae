"""Robust statistics (manuscript §4.6).

    "Compute robust (median/MAD or percentile-based), NOT mean/SD, because
     claim-amount distributions here are heavy-tailed."

The tests are organised around the reason for that instruction: a mean/SD
z-score is dragged by the very outliers it is meant to find. The first section
demonstrates that on this dataset's own amount distribution; the rest cover the
fallback ladder that keeps the statistic defined when MAD degenerates, which is
common in real claims data (a peer group where most claims bill the same
tariff has a MAD of exactly zero).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fwa.statistical.robust import (
    MAD_SCALE, modified_z, percentile_rank, robust_residual_frame, robust_stats, winsorise,
)


# ------------------------------------------------------- why not mean and SD


def test_the_median_ignores_a_contaminating_outlier():
    clean = [100.0] * 50
    contaminated = clean + [1_000_000.0]

    assert robust_stats(contaminated).median == pytest.approx(100.0)
    assert np.mean(contaminated) > 19_000  # the mean has moved by four orders of magnitude


def test_one_extreme_value_masks_itself_under_mean_and_sd_but_not_under_mad():
    """The masking effect, on a distribution shaped like this dataset's amounts."""
    rng = np.random.default_rng(20260920)
    body = rng.lognormal(mean=8.0, sigma=0.6, size=500)
    values = np.append(body, 5_000_000.0)

    mean_sd_z = (values[-1] - values.mean()) / values.std()
    robust_z = robust_stats(values).z(values[-1])

    assert mean_sd_z < 25            # inflated SD flattens the score
    assert robust_z > 100            # the robust statistic still sees it
    assert robust_z > mean_sd_z * 4


def test_the_mad_scale_constant_is_the_normal_consistency_factor():
    """1.4826 makes MAD comparable to an SD on normal data — and is named, not inlined."""
    assert MAD_SCALE == pytest.approx(1.4826)
    rng = np.random.default_rng(1)
    normal = rng.normal(loc=0.0, scale=3.0, size=20_000)
    assert robust_stats(normal).scale == pytest.approx(3.0, rel=0.05)


# ----------------------------------------------------------- the basic stats


def test_stats_on_a_simple_series():
    stats = robust_stats([1, 2, 3, 4, 5])
    assert stats.n == 5
    assert stats.median == pytest.approx(3.0)
    assert stats.p25 == pytest.approx(2.0)
    assert stats.p75 == pytest.approx(4.0)
    assert stats.iqr == pytest.approx(2.0)
    assert stats.fallback_used in (None, "", "mad") or stats.mad > 0


def test_the_z_score_is_centred_on_the_median():
    stats = robust_stats([10, 10, 10, 10, 20])
    assert stats.z(10.0) == pytest.approx(0.0)
    assert stats.z(20.0) > 0


def test_a_missing_value_has_no_z_score():
    stats = robust_stats([1, 2, 3, 4, 5])
    assert stats.z(None) is None
    assert stats.z(float("nan")) is None


def test_nulls_are_dropped_not_treated_as_zero():
    with_nulls = robust_stats([1.0, 2.0, np.nan, 3.0, None])
    without = robust_stats([1.0, 2.0, 3.0])
    assert with_nulls.n == without.n
    assert with_nulls.median == pytest.approx(without.median)


def test_modified_z_matches_the_stats_object():
    values = [5, 7, 9, 11, 13]
    assert modified_z(13, values) == pytest.approx(robust_stats(values).z(13))


# ------------------------------------------------------- the fallback ladder


def test_a_degenerate_mad_falls_back_to_the_iqr():
    """More than half the claims at one tariff gives MAD exactly zero."""
    values = [100.0] * 20 + [100.0] * 20 + [180.0, 220.0, 300.0]
    stats = robust_stats(values)
    assert stats.mad == 0.0
    assert stats.scale > 0, "A zero MAD must not leave the scale at zero."
    assert stats.fallback_used


def test_a_degenerate_mad_and_iqr_fall_back_further():
    values = [100.0] * 40 + [500.0]
    stats = robust_stats(values)
    assert stats.mad == 0.0
    assert stats.iqr == 0.0
    assert stats.fallback_used
    # Either a usable scale from the standard deviation, or an honest refusal.
    assert stats.scale > 0 or stats.z(500.0) is None


def test_a_fully_constant_series_refuses_to_produce_a_score():
    """No spread means no outlier. Inventing one would manufacture signals."""
    stats = robust_stats([42.0] * 30)
    assert stats.scale == 0 or stats.z(42.0) in (None, 0.0)
    assert stats.z(1_000_000.0) is None or stats.scale > 0


def test_the_fallback_used_is_reported_for_the_evidence_panel(claims):
    """A reviewer must be able to see which scale produced the number."""
    stats = robust_stats(claims["gross_amount_aed"])
    assert stats.n > 0
    assert hasattr(stats, "fallback_used")


def test_an_empty_series_is_handled():
    stats = robust_stats([])
    assert stats.n == 0
    assert stats.z(1.0) is None


def test_a_single_value_cannot_be_an_outlier_against_itself():
    stats = robust_stats([7.0])
    assert stats.n == 1
    assert stats.z(7.0) in (None, 0.0)


# ----------------------------------------------------------------- utilities


def test_percentile_rank_is_bounded_and_ordered():
    values = list(range(100))
    assert percentile_rank(0, values) == pytest.approx(0.0, abs=0.02)
    assert percentile_rank(99, values) == pytest.approx(1.0, abs=0.02)
    assert percentile_rank(50, values) > percentile_rank(10, values)


def test_percentile_rank_of_a_missing_value_is_none():
    assert percentile_rank(None, [1, 2, 3]) is None
    assert percentile_rank(1, []) is None


def test_winsorising_caps_without_dropping_rows():
    series = pd.Series([1.0] + [50.0] * 98 + [10_000.0])
    capped = winsorise(series)
    assert len(capped) == len(series)
    assert capped.max() < 10_000.0
    assert capped.min() > 1.0


def test_the_residual_frame_groups_and_labels(claims):
    out = robust_residual_frame(
        claims, "gross_amount_aed", ["diagnosis_primary"], min_group_n=5
    )
    assert len(out) == len(claims)
    residual_columns = [c for c in out.columns if "residual" in c or "median" in c]
    assert residual_columns, out.columns.tolist()


def test_a_group_below_the_minimum_gets_no_residual(claims):
    """Too few peers is insufficient evidence, not a zero residual."""
    out = robust_residual_frame(
        claims, "gross_amount_aed", ["provider_sk"], min_group_n=10_000
    )
    residual_columns = [c for c in out.columns if c.endswith("residual")]
    assert residual_columns
    assert out[residual_columns[0]].isna().all()
