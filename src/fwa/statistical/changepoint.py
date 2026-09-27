"""Statistical change detection — CUSUM and EWMA (manuscript §4.7).

    "Where the question is not 'is this provider different from peers' but 'has
     this provider's own behaviour changed,' the design applies CUSUM, EWMA or
     Bayesian online/offline change-point detection to weekly or monthly
     normalised provider metrics, requiring a configured MINIMUM NUMBER OF
     HISTORY PERIODS before a change point can be declared... Critically, the
     design requires that KNOWN STRUCTURAL CHANGES (a tariff revision, a new
     contract, a facility or ownership change, a system migration) be explicitly
     SEGMENTED or added as covariates before a change point is attributed to
     provider behaviour, and the system stores the PRE-CHANGE MEAN, POST-CHANGE
     MEAN, CHANGE DATE AND CONFIDENCE for every declared change point so that a
     reviewer can see exactly what changed and when, rather than only a binary
     alert."                                                — manuscript §4.7

All four of those requirements are implemented here:

* ``cfg.min_history_periods`` is enforced before any declaration.
* :class:`SegmentRegistry` lets a policy owner declare a known structural change
  for a provider (or globally), and the detector then treats each segment
  independently instead of attributing the step to provider behaviour.
* Every :class:`ChangePoint` stores pre/post mean, date and confidence.
* CUSUM follows Page (1954): standardised deviations accumulated against a
  reference value ``k``, signalling when the cumulative sum crosses ``h``.

The honest limitation on the claim extract, stated on every signal this produces:
there are **no** contract, tariff or ownership records, so the segment registry
is empty unless a user populates it, and a declared change point cannot be
distinguished from a legitimate business change. That is why CLN-07-R04 and
ANL-01-R02 both carry a segmentation caveat and dispose to an audit rather than
anything stronger.
"""

from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass, asdict, field
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

__all__ = ["ChangePoint", "SegmentRegistry", "cusum", "ewma", "detect_changepoints",
           "monthly_provider_metrics"]


@dataclass(frozen=True)
class ChangePoint:
    entity: str
    metric: str
    method: str                # "CUSUM" | "EWMA"
    change_period: str
    change_date: str
    pre_change_mean: float
    post_change_mean: float
    change_magnitude: float
    direction: str             # "increase" | "decrease"
    statistic: float
    threshold: float
    confidence: float
    periods_of_history: int
    segment_id: str
    declared_segments: tuple[str, ...] = ()
    caveat: str = ""

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["declared_segments"] = list(self.declared_segments)
        return d

    def explain(self) -> str:
        return (
            f"{self.entity}'s {self.metric} moved from {self.pre_change_mean:,.2f} to "
            f"{self.post_change_mean:,.2f} ({self.direction}, "
            f"{abs(self.change_magnitude):,.2f}) at {self.change_period}, detected by "
            f"{self.method} after {self.periods_of_history} periods of history."
            + (f" {self.caveat}" if self.caveat else "")
        )


class SegmentRegistry:
    """Declared structural changes, so a tariff revision is not read as behaviour.

    A segment boundary declared here splits an entity's series; the detector
    runs within each segment and never signals across a boundary. Boundaries are
    owned and reasoned, like every other governed configuration value — they are
    written through the Governance page and land in the audit log.
    """

    def __init__(self) -> None:
        self._boundaries: dict[str, list[tuple[str, str, str]]] = {}   # entity -> [(period, reason, owner)]

    def declare(self, entity: str, period: str, reason: str, owner: str) -> None:
        if not reason.strip():
            raise ValueError(
                "A declared structural change requires a reason. Without one the "
                "segment is indistinguishable from suppressing an inconvenient alert."
            )
        self._boundaries.setdefault(entity, []).append((period, reason, owner))
        self._boundaries[entity].sort()

    def boundaries_for(self, entity: str) -> list[tuple[str, str, str]]:
        return list(self._boundaries.get(entity, [])) + list(self._boundaries.get("*", []))

    def segment_of(self, entity: str, period: str) -> str:
        bounds = [b for b, _, _ in self.boundaries_for(entity)]
        n = sum(1 for b in bounds if period >= b)
        return f"seg{n}"

    def descriptions(self, entity: str) -> tuple[str, ...]:
        return tuple(f"{p}: {r} (declared by {o})" for p, r, o in self.boundaries_for(entity))

    def __len__(self) -> int:
        return sum(len(v) for v in self._boundaries.values())


# ---------------------------------------------------------------------------
# Detectors
# ---------------------------------------------------------------------------


def cusum(values: Sequence[float], k: float, h: float) -> tuple[np.ndarray, np.ndarray, int | None]:
    """Two-sided CUSUM (Page, 1954) on standardised values.

    Returns ``(c_high, c_low, first_crossing_index)``. ``k`` is the reference
    value (allowance) in standardised units — the shift size the chart is tuned
    to detect quickly — and ``h`` is the decision interval.
    """
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return np.array([]), np.array([]), None
    mu = float(np.nanmean(arr))
    sd = float(np.nanstd(arr, ddof=1)) if arr.size > 1 else 0.0
    if sd <= 0:
        return np.zeros_like(arr), np.zeros_like(arr), None
    z = (arr - mu) / sd
    hi = np.zeros_like(z)
    lo = np.zeros_like(z)
    crossing = None
    for i in range(len(z)):
        prev_hi = hi[i - 1] if i else 0.0
        prev_lo = lo[i - 1] if i else 0.0
        hi[i] = max(0.0, prev_hi + z[i] - k)
        lo[i] = min(0.0, prev_lo + z[i] + k)
        if crossing is None and (hi[i] > h or lo[i] < -h):
            crossing = i
    return hi, lo, crossing


def ewma(values: Sequence[float], lam: float, L: float) -> tuple[np.ndarray, np.ndarray, int | None]:
    """EWMA control chart. Returns ``(ewma_series, control_limit, first_exit_index)``."""
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return np.array([]), np.array([]), None
    mu = float(np.nanmean(arr))
    sd = float(np.nanstd(arr, ddof=1)) if arr.size > 1 else 0.0
    if sd <= 0:
        return np.full_like(arr, mu), np.zeros_like(arr), None
    z = np.zeros_like(arr)
    prev = mu
    for i, v in enumerate(arr):
        prev = lam * v + (1 - lam) * prev
        z[i] = prev
    i_idx = np.arange(1, len(arr) + 1)
    limit = L * sd * np.sqrt(lam / (2 - lam) * (1 - (1 - lam) ** (2 * i_idx)))
    exits = np.where(np.abs(z - mu) > limit)[0]
    return z, limit, int(exits[0]) if exits.size else None


def detect_changepoints(
    series: pd.DataFrame,
    *,
    entity_column: str,
    period_column: str,
    value_column: str,
    metric_name: str,
    min_history_periods: int,
    cusum_k: float,
    cusum_h: float,
    ewma_lambda: float,
    ewma_L: float,
    segments: SegmentRegistry | None = None,
    caveat: str = "",
) -> list[ChangePoint]:
    """Run CUSUM and EWMA per entity per declared segment."""
    segments = segments or SegmentRegistry()
    out: list[ChangePoint] = []
    for entity, sub in series.groupby(entity_column, sort=False):
        sub = sub.sort_values(period_column)
        sub = sub.assign(_segment=[segments.segment_of(str(entity), str(p)) for p in sub[period_column]])
        for seg_id, seg in sub.groupby("_segment", sort=True):
            vals = seg[value_column].astype(float).to_numpy()
            periods = seg[period_column].astype(str).tolist()
            n = len(vals)
            if n < min_history_periods:
                continue
            for method, crossing, stat, thresh in _run_both(vals, cusum_k, cusum_h, ewma_lambda, ewma_L):
                if crossing is None or crossing < min_history_periods:
                    # A change point inside the minimum-history window is NOT
                    # declared: §4.7 requires a configured minimum number of
                    # history periods before a change point may be declared.
                    continue
                pre = float(np.nanmean(vals[:crossing]))
                post = float(np.nanmean(vals[crossing:]))
                magnitude = post - pre
                spread = float(np.nanstd(vals[:crossing], ddof=1)) if crossing > 1 else 0.0
                confidence = float(np.clip(abs(magnitude) / (spread + 1e-9) / 4.0, 0.0, 1.0)) if spread > 0 else 0.5
                out.append(
                    ChangePoint(
                        entity=str(entity),
                        metric=metric_name,
                        method=method,
                        change_period=periods[crossing],
                        change_date=_period_to_date(periods[crossing]),
                        pre_change_mean=pre,
                        post_change_mean=post,
                        change_magnitude=magnitude,
                        direction="increase" if magnitude > 0 else "decrease",
                        statistic=float(stat),
                        threshold=float(thresh),
                        confidence=confidence,
                        periods_of_history=n,
                        segment_id=str(seg_id),
                        declared_segments=segments.descriptions(str(entity)),
                        caveat=caveat,
                    )
                )
                break   # one declaration per entity-segment; CUSUM is preferred
    return out


def _run_both(vals, k, h, lam, L):
    hi, lo, c_cross = cusum(vals, k, h)
    stat = float(max(hi.max() if hi.size else 0.0, abs(lo.min()) if lo.size else 0.0))
    yield "CUSUM", c_cross, stat, h
    z, limit, e_cross = ewma(vals, lam, L)
    e_stat = float(np.abs(z - np.nanmean(vals)).max()) if z.size else 0.0
    yield "EWMA", e_cross, e_stat, float(limit.max()) if limit.size else 0.0


def _period_to_date(period: str) -> str:
    try:
        if "-W" in period:
            year, week = period.split("-W")
            return _dt.date.fromisocalendar(int(year), int(week), 1).isoformat()
        parts = period.split("-")
        return _dt.date(int(parts[0]), int(parts[1]), 1).isoformat()
    except Exception:  # pragma: no cover
        return period


def monthly_provider_metrics(claims: pd.DataFrame) -> pd.DataFrame:
    """Monthly normalised provider metrics — the §4.7 input series.

    "Normalised" here means per-claim averages rather than raw totals, so that a
    provider that simply grew does not look like a provider whose behaviour
    changed. Volume itself is kept as a separate metric (``claim_count``)
    because CLN-07-R04 is specifically about throughput.
    """
    work = claims.copy()
    work["period"] = pd.to_datetime(work["service_date"], errors="coerce").dt.to_period("M")
    # A claim with no usable service date has no period, and a period is what a
    # change point is expressed in ("behaviour changed at 2025-04"). Dropping the
    # undated rows here is explicit rather than incidental: pandas would silently
    # drop them at the groupby anyway, and an all-undated frame would then fall
    # through to an empty aggregate that misreports itself as "no change points"
    # instead of "no datable history".
    work = work[work["period"].notna()]
    work["period"] = work["period"].astype(str)
    if work.empty:
        return pd.DataFrame(
            columns=["provider_sk", "period", "claim_count", "mean_gross_aed",
                     "median_gross_aed", "mean_los", "mean_pharmacy_ratio", "approval_ratio"]
        )
    grouped = work.groupby(["provider_sk", "period"], as_index=False)
    out = grouped.agg(
        claim_count=("claim_sk", "count"),
        mean_gross_aed=("gross_amount_aed", "mean"),
        median_gross_aed=("gross_amount_aed", "median"),
        mean_los=("length_of_stay_days", "mean"),
        mean_pharmacy_ratio=("pharmacy_bill_ratio", "mean"),
    )
    approved = work.groupby(["provider_sk", "period"]).apply(
        lambda g: float(
            (g["approved_amount_aed"].sum() / g["gross_amount_aed"].sum())
            if g["gross_amount_aed"].sum() > 0 else np.nan
        ),
        include_groups=False,
    ).rename("approval_ratio").reset_index()
    return out.merge(approved, on=["provider_sk", "period"], how="left")
