"""The canonical evaluation library (manuscript §3.7).

    "Expressions may compile to SQL, Spark or a streaming rule engine, but the
     specification requires that ONE CANONICAL EVALUATION LIBRARY define null
     handling, time-zone behaviour, rounding, currency conversion, lookback
     windows and as-of semantics consistently across every engine that executes
     rules, a requirement this dissertation's architecture treats as a distinct,
     shared platform service rather than something left to each rule author to
     reimplement."                                          — manuscript §3.7

No rule author reimplements any of these. Every implemented control in
:mod:`fwa.engine.controls` reaches these helpers rather than writing its own
comparison, and the governance test suite asserts that the control modules
contain no bare ``==`` on nullable money or date columns.

The semantics chosen here, and why:

**Nulls are not zero, and not false.** :func:`compare` returns
:data:`UNKNOWN` when either operand is missing, and a control that receives
``UNKNOWN`` does not fire — it records a *null-input outcome* instead. In claims
data the absence of a field (a missing authorisation ID, a missing result) is
frequently itself the signal (§4.5), so a missing value must be an explicit,
visible state rather than something coerced into a comparison.

**Time is UTC internally, Asia/Dubai retained.** §3.6 requires exactly this.
:func:`to_utc` normalises; :func:`to_local` renders. The original local
representation is never lost because it is stored alongside, not instead of.

**Money rounds half-up at 2dp, after conversion, never before.** Converting a
rounded figure and rounding a converted figure differ, and the difference
accumulates across a full-file run.

**As-of means service date.** :func:`as_of_date` exists so that no caller can
accidentally pass ``date.today()`` into a reference lookup, which is the single
most common way an effective-dated system silently misapplies policy.
"""

from __future__ import annotations

import datetime as _dt
from decimal import Decimal, ROUND_HALF_UP
from enum import Enum
from typing import Any, Iterable, Sequence
from zoneinfo import ZoneInfo

__all__ = [
    "UNKNOWN",
    "Tri",
    "LOCAL_TZ",
    "is_missing",
    "compare",
    "safe_ratio",
    "round_money",
    "to_utc",
    "to_local",
    "as_of_date",
    "within_days",
    "overlaps",
    "lookback_window",
    "period_bucket",
    "normalise_unit",
    "clamp01",
    "EvaluationOutcome",
]

LOCAL_TZ = ZoneInfo("Asia/Dubai")


class Tri(str, Enum):
    """Three-valued logic. ``UNKNOWN`` is what a null comparison returns."""

    TRUE = "TRUE"
    FALSE = "FALSE"
    UNKNOWN = "UNKNOWN"

    def __bool__(self) -> bool:  # pragma: no cover - guard
        raise TypeError(
            "Refusing to coerce a three-valued result to bool. UNKNOWN is not False. "
            "Handle the null case explicitly — §4.5: 'the absence of a field is "
            "frequently itself the signal'."
        )

    @property
    def fired(self) -> bool:
        """True only for an unambiguous TRUE. UNKNOWN never fires a control."""
        return self is Tri.TRUE


UNKNOWN = Tri.UNKNOWN


# ---------------------------------------------------------------------------
# Null handling
# ---------------------------------------------------------------------------


def is_missing(value: Any) -> bool:
    """Single definition of 'missing' for the whole system."""
    if value is None:
        return True
    if isinstance(value, float) and value != value:  # NaN
        return True
    if isinstance(value, str) and value.strip() == "":
        return True
    # pandas NaT / NA, without importing pandas at module level
    cls = type(value).__name__
    if cls in {"NaTType", "NAType"}:
        return True
    return False


def compare(left: Any, op: str, right: Any) -> Tri:
    """Null-safe comparison. Returns :class:`Tri`, never a bare bool.

    Supported ``op``: ``==``, ``!=``, ``<``, ``<=``, ``>``, ``>=``, ``in``,
    ``not in``.
    """
    if is_missing(left) or is_missing(right):
        return Tri.UNKNOWN
    try:
        if op == "==":
            result = left == right
        elif op == "!=":
            result = left != right
        elif op == "<":
            result = left < right
        elif op == "<=":
            result = left <= right
        elif op == ">":
            result = left > right
        elif op == ">=":
            result = left >= right
        elif op == "in":
            result = left in right
        elif op == "not in":
            result = left not in right
        else:
            raise ValueError(f"Unsupported operator {op!r} in the canonical evaluation library.")
    except TypeError:
        return Tri.UNKNOWN
    return Tri.TRUE if result else Tri.FALSE


def safe_ratio(numerator: Any, denominator: Any) -> float | None:
    """Ratio with an explicit ``None`` for undefined, never a silent zero.

    An approval ratio of ``None`` (because the requested amount was zero or
    missing) is a *different fact* from an approval ratio of ``0.0``, and
    conflating them is how a data-quality problem becomes a false positive.
    """
    if is_missing(numerator) or is_missing(denominator):
        return None
    try:
        d = float(denominator)
    except (TypeError, ValueError):
        return None
    if d == 0:
        return None
    return float(numerator) / d


def clamp01(x: float | None) -> float:
    if x is None or (isinstance(x, float) and x != x):
        return 0.0
    return max(0.0, min(1.0, float(x)))


# ---------------------------------------------------------------------------
# Money
# ---------------------------------------------------------------------------


def round_money(amount: float | Decimal | None, places: int = 2) -> float | None:
    """Banker's-rounding-free, half-up rounding to ``places``.

    Half-up is chosen because it is what claims arithmetic conventionally uses
    and because it is what a reviewer reproduces with a calculator when checking
    a repricing by hand. Applied AFTER currency conversion, never before.
    """
    if is_missing(amount):
        return None
    quant = Decimal(1).scaleb(-places)
    return float(Decimal(str(float(amount))).quantize(quant, rounding=ROUND_HALF_UP))


# ---------------------------------------------------------------------------
# Time
# ---------------------------------------------------------------------------


def to_utc(value: _dt.datetime | _dt.date | str | None) -> _dt.datetime | None:
    """Normalise any timestamp to timezone-aware UTC (§3.6).

    A naive timestamp is interpreted as Asia/Dubai, because the source systems
    this design targets are UAE-local. That interpretation is a decision, so it
    lives here, once, rather than in each rule.
    """
    if is_missing(value):
        return None
    if isinstance(value, str):
        value = _dt.datetime.fromisoformat(value)
    if isinstance(value, _dt.datetime):
        dt = value
    elif isinstance(value, _dt.date):
        dt = _dt.datetime.combine(value, _dt.time.min)
    else:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=LOCAL_TZ)
    return dt.astimezone(_dt.timezone.utc)


def to_local(value: _dt.datetime | None) -> _dt.datetime | None:
    """Render a UTC timestamp back in Asia/Dubai for display.

    §3.6: "UAE timestamps must be normalised to UTC internally while the
    original Asia/Dubai representation is retained."
    """
    if value is None:
        return None
    if value.tzinfo is None:
        value = value.replace(tzinfo=_dt.timezone.utc)
    return value.astimezone(LOCAL_TZ)


def as_of_date(service_time: _dt.datetime | _dt.date | str | None) -> _dt.date:
    """The date every reference lookup must be evaluated at (§3.6).

    Deliberately takes the *service* time and nothing else. There is no
    ``as_of_date()`` no-argument form, because a reference lookup with no
    as-of date is a lookup against today, and a lookup against today silently
    applies 2026 policy to a 2023 claim.
    """
    if is_missing(service_time):
        raise ValueError(
            "An as-of reference lookup requires the service date. Evaluating against "
            "today's reference data rather than the data in force on the service date "
            "silently misapplies policy (§3.6)."
        )
    if isinstance(service_time, str):
        service_time = _dt.datetime.fromisoformat(service_time)
    if isinstance(service_time, _dt.datetime):
        return service_time.date()
    return service_time


def within_days(a: Any, b: Any, days: int) -> Tri:
    """``|a - b| <= days``, null-safe."""
    if is_missing(a) or is_missing(b):
        return Tri.UNKNOWN
    da, db = _coerce_date(a), _coerce_date(b)
    if da is None or db is None:
        return Tri.UNKNOWN
    return Tri.TRUE if abs((da - db).days) <= days else Tri.FALSE


def overlaps(
    start_a: Any, end_a: Any, start_b: Any, end_b: Any, tolerance_days: int = 0
) -> Tri:
    """Interval overlap with a configurable tolerance, null-safe.

    Used by ENT-02-R03 (member impossible presence) and CLN-07 (capacity). The
    tolerance exists because the claim extract carries dates without times, so a
    same-day transfer is indistinguishable from a genuine overlap without it.
    """
    sa, ea, sb, eb = (_coerce_date(x) for x in (start_a, end_a, start_b, end_b))
    if None in (sa, ea, sb, eb):
        return Tri.UNKNOWN
    tol = _dt.timedelta(days=tolerance_days)
    return Tri.TRUE if (sa < eb - tol) and (sb < ea - tol) else Tri.FALSE


def lookback_window(anchor: Any, days: int) -> tuple[_dt.date, _dt.date] | None:
    """The half-open window ``(anchor - days, anchor]``.

    Half-open and backward-looking by definition, so that a lookback can never
    accidentally include data from after the scored event — the leakage rule of
    §4.5 and §6.6 expressed as an interval.
    """
    a = _coerce_date(anchor)
    if a is None:
        return None
    return (a - _dt.timedelta(days=days), a)


def period_bucket(value: Any, freq: str = "M") -> str:
    """Canonical period label used in case fingerprinting (§4.12)."""
    d = _coerce_date(value)
    if d is None:
        return "UNKNOWN"
    if freq == "M":
        return f"{d.year:04d}-{d.month:02d}"
    if freq == "W":
        iso = d.isocalendar()
        return f"{iso[0]:04d}-W{iso[1]:02d}"
    if freq == "Q":
        return f"{d.year:04d}-Q{((d.month - 1) // 3) + 1}"
    if freq == "Y":
        return f"{d.year:04d}"
    raise ValueError(f"Unsupported period frequency {freq!r}")


def _coerce_date(value: Any) -> _dt.date | None:
    if is_missing(value):
        return None
    if isinstance(value, _dt.datetime):
        return value.date()
    if isinstance(value, _dt.date):
        return value
    if isinstance(value, str):
        try:
            return _dt.date.fromisoformat(value[:10])
        except ValueError:
            return None
    # pandas Timestamp
    if hasattr(value, "to_pydatetime"):
        try:
            return value.to_pydatetime().date()
        except Exception:  # pragma: no cover
            return None
    return None


def normalise_unit(value: Any, unit: str) -> float | None:
    """Single place where unit conversions live (days/hours, count/rate)."""
    if is_missing(value):
        return None
    v = float(value)
    factors = {"days": 1.0, "hours": 1 / 24.0, "weeks": 7.0, "count": 1.0, "percent": 0.01, "ratio": 1.0}
    if unit not in factors:
        raise ValueError(f"Unknown unit {unit!r}")
    return v * factors[unit]


# ---------------------------------------------------------------------------
# Outcome of evaluating one control against one row
# ---------------------------------------------------------------------------


class EvaluationOutcome(str, Enum):
    """Why a control did or did not produce a signal.

    ``NULL_INPUT`` and ``EXCLUDED`` are distinct from ``NOT_TRIGGERED`` on
    purpose. The ten-category test suite has a category for each, and the
    Governance page reports them separately, because "this rule did not fire"
    and "this rule could not be evaluated" are very different operational
    facts.
    """

    TRIGGERED = "TRIGGERED"
    NOT_TRIGGERED = "NOT_TRIGGERED"
    NULL_INPUT = "NULL_INPUT"
    EXCLUDED = "EXCLUDED"
    OUT_OF_POPULATION = "OUT_OF_POPULATION"
    NOT_EFFECTIVE = "NOT_EFFECTIVE"
    KILL_SWITCHED = "KILL_SWITCHED"
    INSUFFICIENT_PEER_EVIDENCE = "INSUFFICIENT_PEER_EVIDENCE"
    VOLUME_CEILING_BREACHED = "VOLUME_CEILING_BREACHED"
    NOT_EXECUTABLE_ON_THIS_DATASET = "NOT_EXECUTABLE_ON_THIS_DATASET"
