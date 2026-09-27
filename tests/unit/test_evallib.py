"""The canonical evaluation library (manuscript §3.7).

    "ONE CANONICAL EVALUATION LIBRARY define[s] null handling, time-zone
     behaviour, rounding, currency conversion, lookback windows and as-of
     semantics consistently across every engine that executes rules."

Each of those six is a decision that changes results, so each gets its own
section below. The recurring theme is that the library refuses to guess: a
missing operand yields ``UNKNOWN`` rather than ``False``, an undefined ratio
yields ``None`` rather than ``0.0``, and an as-of lookup without a service date
raises rather than silently using today.
"""

from __future__ import annotations

import datetime as _dt
from decimal import Decimal

import numpy as np
import pandas as pd
import pytest

from fwa.engine.evallib import (
    LOCAL_TZ, Tri, UNKNOWN, as_of_date, clamp01, compare, is_missing, lookback_window,
    normalise_unit, overlaps, period_bucket, round_money, safe_ratio, to_local, to_utc,
    within_days,
)


# ------------------------------------------------------------- null handling


@pytest.mark.parametrize("value", [None, float("nan"), np.nan, "", "   ", pd.NaT, pd.NA])
def test_these_are_missing(value):
    assert is_missing(value) is True


@pytest.mark.parametrize("value", [0, 0.0, False, "0", "x", [], {}, _dt.date(2026, 1, 1)])
def test_these_are_not_missing(value):
    """Zero, False and the empty list are values, not absences."""
    assert is_missing(value) is False


@pytest.mark.parametrize("op", ["==", "!=", "<", "<=", ">", ">=", "in", "not in"])
def test_every_operator_returns_unknown_on_a_missing_operand(op):
    assert compare(None, op, 1) is UNKNOWN
    assert compare(1, op, None) is UNKNOWN


def test_comparison_returns_three_valued_logic_not_a_bool():
    assert compare(2, ">", 1) is Tri.TRUE
    assert compare(1, ">", 2) is Tri.FALSE
    assert compare(None, ">", 2) is Tri.UNKNOWN


def test_unknown_refuses_to_be_coerced_to_a_bool():
    """The whole point: ``if compare(...)`` must not quietly mean "not UNKNOWN"."""
    with pytest.raises(TypeError) as exc:
        bool(UNKNOWN)
    assert "UNKNOWN is not False" in str(exc.value)


def test_only_an_unambiguous_true_fires_a_control():
    assert Tri.TRUE.fired is True
    assert Tri.FALSE.fired is False
    assert Tri.UNKNOWN.fired is False


def test_membership_operators():
    assert compare("A", "in", ["A", "B"]) is Tri.TRUE
    assert compare("C", "not in", ["A", "B"]) is Tri.TRUE


def test_an_incomparable_pair_is_unknown_not_an_exception():
    assert compare("text", "<", 5) is UNKNOWN


def test_an_unsupported_operator_is_an_error_not_a_guess():
    with pytest.raises(ValueError):
        compare(1, "~=", 1)


# -------------------------------------------------------------------- ratios


def test_an_undefined_ratio_is_none_not_zero():
    """A ratio of ``None`` and a ratio of ``0.0`` are different facts."""
    assert safe_ratio(5, 0) is None
    assert safe_ratio(5, None) is None
    assert safe_ratio(None, 5) is None
    assert safe_ratio(0, 5) == 0.0


def test_a_defined_ratio_is_a_float():
    assert safe_ratio(9, 10) == pytest.approx(0.9)


def test_a_non_numeric_denominator_is_undefined():
    assert safe_ratio(1, "many") is None


@pytest.mark.parametrize("value,expected", [(-1.0, 0.0), (0.4, 0.4), (2.0, 1.0),
                                            (None, 0.0), (float("nan"), 0.0)])
def test_clamp01(value, expected):
    assert clamp01(value) == pytest.approx(expected)


# --------------------------------------------------------------------- money


@pytest.mark.parametrize("amount,expected", [
    (1.005, 1.01),    # half-up, not banker's — banker's would give 1.00
    (2.675, 2.68),
    (0.125, 0.13),
    (-1.005, -1.01),  # "half-up" is away from zero, matching a hand calculation
    (100.0, 100.0),
])
def test_money_rounds_half_up(amount, expected):
    assert round_money(amount) == pytest.approx(expected)


def test_rounding_a_missing_amount_is_missing():
    assert round_money(None) is None


def test_decimal_input_is_accepted():
    assert round_money(Decimal("3.14159")) == pytest.approx(3.14)


def test_rounding_is_configurable_in_places():
    assert round_money(1.23456, places=3) == pytest.approx(1.235)


def test_converting_then_rounding_differs_from_rounding_then_converting(config):
    """Why §3.7 fixes the order: the two answers are not the same.

    Rounding before conversion discards sub-fils precision that the conversion
    then magnifies. Over a 5,000-claim run the difference is not noise, and a
    reviewer reconciling a total against the source would find it.
    """
    service_date = _dt.date(2025, 6, 1)
    amounts = [1234.567, 89.014, 2345.678]

    convert_then_round = sum(
        round_money(config.fx.convert(a, "INR", service_date)) for a in amounts
    )
    round_then_convert = sum(
        config.fx.convert(round_money(a, places=0), "INR", service_date) for a in amounts
    )
    assert convert_then_round != pytest.approx(round_then_convert)


# ---------------------------------------------------------------------- time


def test_a_naive_timestamp_is_read_as_dubai_local():
    utc = to_utc(_dt.datetime(2026, 3, 1, 12, 0))
    assert utc.tzinfo is _dt.timezone.utc
    assert utc.hour == 8  # Asia/Dubai is UTC+4, year-round


def test_an_aware_timestamp_keeps_its_instant():
    aware = _dt.datetime(2026, 3, 1, 12, 0, tzinfo=_dt.timezone.utc)
    assert to_utc(aware) == aware


def test_a_date_becomes_midnight_local_not_midnight_utc():
    """A bare date is a *local* day. Reading it as UTC would shift every claim.

    Midnight on 1 March in Dubai is 20:00 on 28 February in UTC, so a control
    that bucketed on the UTC date would put a month-end claim in the wrong
    month — and month buckets are what peer baselines and change points are
    computed over.
    """
    utc = to_utc(_dt.date(2026, 3, 1))
    assert (utc.year, utc.month, utc.day, utc.hour) == (2026, 2, 28, 20)


def test_an_iso_string_is_accepted():
    assert to_utc("2026-03-01T12:00:00") is not None


def test_a_missing_timestamp_is_none():
    assert to_utc(None) is None
    assert to_local(None) is None


def test_the_local_representation_is_recoverable():
    """§3.6: UTC internally, the Asia/Dubai representation retained."""
    original = _dt.datetime(2026, 3, 1, 12, 0)
    assert to_local(to_utc(original)).replace(tzinfo=None) == original
    assert to_local(to_utc(original)).tzinfo == LOCAL_TZ


def test_a_naive_utc_value_rendered_locally_is_treated_as_utc():
    rendered = to_local(_dt.datetime(2026, 3, 1, 8, 0))
    assert rendered.hour == 12


# -------------------------------------------------------------------- as-of


def test_as_of_requires_a_service_date():
    """There is no no-argument form, and a missing date is fatal."""
    with pytest.raises(ValueError) as exc:
        as_of_date(None)
    assert "§3.6" in str(exc.value)


def test_as_of_accepts_the_shapes_a_claim_carries():
    assert as_of_date(_dt.date(2025, 4, 1)) == _dt.date(2025, 4, 1)
    assert as_of_date(_dt.datetime(2025, 4, 1, 9, 30)) == _dt.date(2025, 4, 1)
    assert as_of_date("2025-04-01") == _dt.date(2025, 4, 1)


# ------------------------------------------------------------------ windows


def test_within_days_is_inclusive_at_the_boundary():
    a, b = _dt.date(2026, 1, 1), _dt.date(2026, 1, 15)
    assert within_days(a, b, 14) is Tri.TRUE
    assert within_days(a, b, 13) is Tri.FALSE


def test_within_days_is_symmetric_and_null_safe():
    a, b = _dt.date(2026, 1, 15), _dt.date(2026, 1, 1)
    assert within_days(a, b, 14) is Tri.TRUE
    assert within_days(a, None, 14) is UNKNOWN


def test_a_lookback_window_never_reaches_forward():
    """The leakage rule of §4.5 expressed as an interval."""
    anchor = _dt.date(2026, 6, 30)
    low, high = lookback_window(anchor, 90)
    assert high == anchor
    assert low == anchor - _dt.timedelta(days=90)
    assert low < high


def test_a_lookback_window_on_a_missing_anchor_is_none():
    assert lookback_window(None, 30) is None


def test_overlap_detects_a_genuine_overlap():
    assert overlaps(_dt.date(2026, 1, 1), _dt.date(2026, 1, 10),
                    _dt.date(2026, 1, 5), _dt.date(2026, 1, 15)) is Tri.TRUE


def test_adjacent_intervals_do_not_overlap():
    assert overlaps(_dt.date(2026, 1, 1), _dt.date(2026, 1, 5),
                    _dt.date(2026, 1, 5), _dt.date(2026, 1, 9)) is Tri.FALSE


def test_the_overlap_tolerance_absorbs_a_same_day_transfer():
    """claims.csv has dates without times, so a one-day touch is ambiguous."""
    args = (_dt.date(2026, 1, 1), _dt.date(2026, 1, 6),
            _dt.date(2026, 1, 5), _dt.date(2026, 1, 10))
    assert overlaps(*args) is Tri.TRUE
    assert overlaps(*args, tolerance_days=1) is Tri.FALSE


def test_overlap_with_a_missing_bound_is_unknown():
    assert overlaps(_dt.date(2026, 1, 1), None,
                    _dt.date(2026, 1, 5), _dt.date(2026, 1, 9)) is UNKNOWN


# ------------------------------------------------------------------- buckets


@pytest.mark.parametrize("freq,expected", [
    ("M", "2026-03"), ("Q", "2026-Q1"), ("Y", "2026"),
])
def test_period_bucket(freq, expected):
    assert period_bucket(_dt.date(2026, 3, 15), freq) == expected


def test_weekly_bucket_uses_iso_weeks():
    assert period_bucket(_dt.date(2026, 3, 15), "W").startswith("2026-W")


def test_a_missing_date_buckets_to_unknown_not_to_an_epoch():
    assert period_bucket(None) == "UNKNOWN"


def test_an_unsupported_frequency_raises():
    with pytest.raises(ValueError):
        period_bucket(_dt.date(2026, 3, 15), "X")


def test_a_pandas_timestamp_is_understood():
    assert period_bucket(pd.Timestamp("2026-03-15")) == "2026-03"


def test_an_unparseable_string_is_unknown_not_an_exception():
    assert period_bucket("not a date") == "UNKNOWN"


# --------------------------------------------------------------------- units


@pytest.mark.parametrize("value,unit,expected", [
    (2, "days", 2.0), (24, "hours", 1.0), (1, "weeks", 7.0),
    (50, "percent", 0.5), (3, "count", 3.0), (0.4, "ratio", 0.4),
])
def test_unit_conversion(value, unit, expected):
    assert normalise_unit(value, unit) == pytest.approx(expected)


def test_an_unknown_unit_raises_rather_than_passing_the_number_through():
    with pytest.raises(ValueError):
        normalise_unit(1, "furlongs")


def test_a_missing_value_converts_to_none():
    assert normalise_unit(None, "days") is None
