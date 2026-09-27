"""Robust statistics for skewed money (manuscript §4.6).

    "For skewed continuous values (billed amounts, cost-to-service ratios, the
     metrics the original proposal specifically targeted), the design uses
     median/median-absolute-deviation (MAD) or percentile-based robust
     statistics RATHER THAN a mean/standard-deviation z-score, since
     claims-amount distributions are heavily right-skewed and a z-score computed
     on such a distribution systematically both under- and over-flags in ways
     that do not track genuine risk."                      — manuscript §4.6

That is the entire justification for this module, and it is worth spelling out
once because it is the kind of choice that looks like a detail and is not. In a
right-skewed amount distribution the mean sits above the median and the standard
deviation is inflated by the same long tail the control is trying to detect.
The result is that ordinary providers near the mode get large negative z-scores
(under-flagging is not the problem) while the threshold needed to flag the tail
is pushed out by the tail's own contribution to sigma — so the very outliers of
interest are partially hidden by their own influence on the statistic. Median
and MAD have a 50% breakdown point: half the sample would have to be
contaminated before they move.

the claim extract confirms the premise — requested amounts run from roughly 5k to
1.86M INR with a mean well above the median. Every amount-based control here
therefore uses :func:`modified_z` and none uses a z-score.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd

__all__ = [
    "RobustStats",
    "robust_stats",
    "modified_z",
    "robust_residual_frame",
    "percentile_rank",
    "winsorise",
    "MAD_SCALE",
]

#: Consistency constant making MAD a consistent estimator of sigma for normal
#: data (1 / Phi^-1(0.75)). Kept explicit so the modified z-score below is
#: comparable in magnitude to a conventional z-score for well-behaved data,
#: which is what makes a threshold like 3.5 interpretable.
MAD_SCALE = 1.4826


@dataclass(frozen=True)
class RobustStats:
    """Median, MAD and derived scale for one group."""

    n: int
    median: float
    mad: float
    scale: float          # MAD_SCALE * mad, with the fallback below applied
    p25: float
    p75: float
    iqr: float
    fallback_used: str    # "" | "iqr" | "std" | "degenerate"

    def z(self, value: float) -> float | None:
        """Modified z-score of ``value`` against this group."""
        if value is None or (isinstance(value, float) and value != value):
            return None
        if self.scale <= 0:
            return None
        return (float(value) - self.median) / self.scale


def robust_stats(values: Sequence[float] | pd.Series | np.ndarray) -> RobustStats:
    """Median/MAD with documented fallbacks.

    MAD is zero whenever more than half the values are identical — common for
    small peer groups and for integer metrics such as length of stay. Dividing
    by zero would produce infinities, and silently substituting the standard
    deviation would reintroduce exactly the non-robust behaviour §4.6 rejects.
    The fallback ladder is therefore explicit and is *recorded* on the result so
    a reviewer can see which scale produced a residual:

    1. ``MAD_SCALE * MAD``    — preferred.
    2. ``IQR / 1.349``        — still robust; used when MAD is 0 but the IQR is not.
    3. ``std``                — last resort, marked ``fallback_used="std"``.
    4. degenerate             — no scale at all; no residual is computed and the
       control reports insufficient evidence rather than a number.
    """
    arr = pd.Series(values).astype(float).dropna().to_numpy()
    n = int(arr.size)
    if n == 0:
        return RobustStats(0, float("nan"), float("nan"), 0.0, float("nan"), float("nan"),
                           float("nan"), "degenerate")
    med = float(np.median(arr))
    mad = float(np.median(np.abs(arr - med)))
    p25, p75 = (float(np.percentile(arr, 25)), float(np.percentile(arr, 75)))
    iqr = p75 - p25
    if mad > 0:
        return RobustStats(n, med, mad, MAD_SCALE * mad, p25, p75, iqr, "")
    if iqr > 0:
        return RobustStats(n, med, mad, iqr / 1.349, p25, p75, iqr, "iqr")
    std = float(np.std(arr, ddof=1)) if n > 1 else 0.0
    if std > 0:
        return RobustStats(n, med, mad, std, p25, p75, iqr, "std")
    return RobustStats(n, med, mad, 0.0, p25, p75, iqr, "degenerate")


def modified_z(value: float, values: Sequence[float]) -> float | None:
    return robust_stats(values).z(value)


def robust_residual_frame(
    df: pd.DataFrame,
    value_column: str,
    group_columns: Sequence[str],
    *,
    min_group_n: int = 1,
    prefix: str | None = None,
) -> pd.DataFrame:
    """Add robust residual columns for ``value_column`` within ``group_columns``.

    Returns the input frame plus ``<prefix>_median``, ``<prefix>_mad``,
    ``<prefix>_scale``, ``<prefix>_residual``, ``<prefix>_n`` and
    ``<prefix>_scale_source``. Groups smaller than ``min_group_n`` produce a
    null residual, not a residual computed on three observations.
    """
    prefix = prefix or value_column
    out = df.copy()
    med = np.full(len(out), np.nan)
    mad = np.full(len(out), np.nan)
    scale = np.full(len(out), np.nan)
    resid = np.full(len(out), np.nan)
    counts = np.zeros(len(out), dtype=float)
    source = np.array([""] * len(out), dtype=object)

    positions = {idx: i for i, idx in enumerate(out.index)}
    for _, sub in out.groupby(list(group_columns), dropna=False, sort=False):
        stats = robust_stats(sub[value_column])
        if stats.n < min_group_n or stats.scale <= 0:
            for idx in sub.index:
                counts[positions[idx]] = stats.n
                source[positions[idx]] = stats.fallback_used or "insufficient"
            continue
        for idx, val in sub[value_column].items():
            i = positions[idx]
            med[i], mad[i], scale[i] = stats.median, stats.mad, stats.scale
            counts[i] = stats.n
            source[i] = stats.fallback_used or "mad"
            resid[i] = stats.z(val) if pd.notna(val) else np.nan

    out[f"{prefix}_median"] = med
    out[f"{prefix}_mad"] = mad
    out[f"{prefix}_scale"] = scale
    out[f"{prefix}_residual"] = resid
    out[f"{prefix}_n"] = counts
    out[f"{prefix}_scale_source"] = source
    return out


def percentile_rank(value: float, values: Sequence[float]) -> float | None:
    """Percentile of ``value`` within ``values``, in [0, 1]."""
    arr = pd.Series(values).astype(float).dropna().to_numpy()
    if arr.size == 0 or value is None or value != value:
        return None
    return float((arr <= float(value)).mean())


def winsorise(values: pd.Series, lower: float = 0.01, upper: float = 0.99) -> pd.Series:
    """Clip to percentile bounds.

    Used only for *display* scaling and for the model feature matrix, never for
    a control's residual: winsorising before a residual would pull the outliers
    the control exists to find back toward the body of the distribution.
    """
    s = values.astype(float)
    lo, hi = s.quantile(lower), s.quantile(upper)
    return s.clip(lower=lo, upper=hi)
