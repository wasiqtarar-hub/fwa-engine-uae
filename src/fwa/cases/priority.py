"""The priority score (manuscript §4.11), term by term.

    priority = 100 * sigmoid(
        1.10 * evidence_strength
      + 0.80 * log1p(exposure_aed / 1000)
      + 0.55 * independent_domain_count
      + 0.45 * validated_prior_history
      + 0.25 * recency
      - 0.60 * data_quality_penalty
      - 0.50 * known_exception_strength
    )

Three properties of this formula are requirements rather than implementation
detail, and all three are enforced here:

**Every input except exposure is normalised to [0,1].** Exposure enters through
``log1p(exposure_aed / 1000)`` precisely so that a large amount raises priority
without a single very large claim dominating the queue.

**Evidence is capped, not summed.** ``evidence_strength`` comes from
:func:`fwa.engine.signals.cap_evidence`, which takes the maximum among signals
resting on the same underlying fact (§3.8). Three rules that all depend on one
missing document contribute once.

**The number is decomposed.** :meth:`PriorityScorer.score` returns every term's
contribution, and the UI renders them — "the formula's whole purpose is that a
reviewer can see how the number was made" (build brief §10). A priority score
shown as a bare number would satisfy the arithmetic and defeat the design.

And the constraint that sits above the formula: **priority orders the queue
only**. §4.11: "A hard-edit disposition is never overridden by a low priority
score: disposition is a policy decision, and priority only orders the review
queue, never substitutes for the disposition rule itself."
"""

from __future__ import annotations

import datetime as _dt
import math
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

from ..engine.signals import Signal, cap_evidence
from ..enums import SignalDomain

__all__ = ["PriorityTerm", "PriorityScore", "PriorityScorer"]


@dataclass(frozen=True)
class PriorityTerm:
    name: str
    label: str
    raw_value: Any
    normalised: float
    weight: float
    contribution: float
    explanation: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "term": self.name,
            "label": self.label,
            "raw_value": self.raw_value,
            "normalised": round(self.normalised, 4),
            "weight": self.weight,
            "contribution": round(self.contribution, 4),
            "explanation": self.explanation,
        }


@dataclass
class PriorityScore:
    value: float                      # 0–100
    logit: float
    terms: list[PriorityTerm] = field(default_factory=list)
    band: str = ""
    band_label: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "priority": round(self.value, 1),
            "band": self.band,
            "band_label": self.band_label,
            "logit": round(self.logit, 4),
            "terms": [t.to_dict() for t in self.terms],
            "formula": (
                "priority = 100 * sigmoid(1.10*evidence_strength + 0.80*log1p(exposure_aed/1000) "
                "+ 0.55*independent_domain_count + 0.45*validated_prior_history + 0.25*recency "
                "- 0.60*data_quality_penalty - 0.50*known_exception_strength)"
            ),
            "boundary_note": (
                "Priority ORDERS THE QUEUE ONLY. It never sets, overrides or softens a "
                "disposition — disposition is a policy decision."
            ),
        }

    def arithmetic_lines(self) -> list[str]:
        """The score, reconstructable by hand. Rendered in the evidence panel."""
        lines = [f"{t.weight:+.2f} × {t.normalised:.3f}  = {t.contribution:+.4f}   {t.label}"
                 for t in self.terms]
        lines.append(f"{'':>24}sum = {self.logit:+.4f}")
        lines.append(f"{'':>18}100 × sigmoid({self.logit:.4f}) = {self.value:.1f}")
        return lines


#: Priority is shown as a LABELLED BAND, not a naked number (build brief §14.1).
_BANDS = (
    (80.0, "P1", "Urgent — review first"),
    (65.0, "P2", "High"),
    (50.0, "P3", "Medium"),
    (35.0, "P4", "Low"),
    (0.0, "P5", "Watch only"),
)


class PriorityScorer:
    def __init__(self, config) -> None:
        self.config = config
        self.weights: dict[str, float] = dict(config.get("priority_weights"))
        self.exposure_scale = float(config.get("priority_exposure_scale_aed"))
        self.recency_half_life = float(config.get("recency_half_life_days"))
        self.max_domains = int(config.get("max_independent_domains"))

    # ------------------------------------------------------------------ score

    def score(
        self,
        signals: Sequence[Signal],
        *,
        exposure_aed: float,
        as_of: _dt.date | None = None,
        validated_prior_history: float = 0.0,
    ) -> PriorityScore:
        as_of = as_of or _dt.date.today()

        # evidence_strength — CAPPED per underlying fact (§3.8)
        evidence_strength = cap_evidence(signals)
        fact_count = len({s.fact_key or s.signal_id for s in signals})

        domains = {s.domain for s in signals}
        domain_norm = min(len(domains) / max(self.max_domains, 1), 1.0)

        recency = self._recency(signals, as_of)
        dq = max((s.data_quality_penalty for s in signals), default=0.0)
        exception = max((s.known_exception_strength for s in signals), default=0.0)

        exposure_term_value = math.log1p(max(exposure_aed, 0.0) / self.exposure_scale)

        terms = [
            PriorityTerm(
                "evidence_strength", "Evidence strength", round(evidence_strength, 4),
                evidence_strength, self.weights["evidence_strength"],
                self.weights["evidence_strength"] * evidence_strength,
                f"{len(signals)} signal(s) resting on {fact_count} distinct underlying fact(s). "
                f"Signals sharing a fact are CAPPED at the strongest, not summed.",
            ),
            PriorityTerm(
                "log_exposure", "Exposure", f"AED {exposure_aed:,.2f}",
                exposure_term_value, self.weights["log_exposure"],
                self.weights["log_exposure"] * exposure_term_value,
                f"log1p(AED {exposure_aed:,.0f} / {self.exposure_scale:,.0f}). Logged so a single "
                f"very large claim cannot dominate the queue.",
            ),
            PriorityTerm(
                "independent_domain_count", "Independent evidence domains",
                sorted(d.value for d in domains), domain_norm,
                self.weights["independent_domain_count"],
                self.weights["independent_domain_count"] * domain_norm,
                f"{len(domains)} of {self.max_domains} domains "
                f"({', '.join(sorted(d.value for d in domains))}). Two findings from different "
                f"domains are stronger than two from the same one.",
            ),
            PriorityTerm(
                "validated_prior_history", "Validated prior history",
                round(validated_prior_history, 4), min(max(validated_prior_history, 0.0), 1.0),
                self.weights["validated_prior_history"],
                self.weights["validated_prior_history"] * min(max(validated_prior_history, 0.0), 1.0),
                "Confirmed prior findings against this subject. Zero until reviewers record "
                "dispositions — it is CONFIRMED history, not a system flag.",
            ),
            PriorityTerm(
                "recency", "Recency", f"{self._age_days(signals, as_of)} day(s) old",
                recency, self.weights["recency"], self.weights["recency"] * recency,
                f"Exponential decay with a {self.recency_half_life:.0f}-day half-life.",
            ),
            PriorityTerm(
                "data_quality_penalty", "Data-quality penalty", round(dq, 4), dq,
                self.weights["data_quality_penalty"], self.weights["data_quality_penalty"] * dq,
                "Subtracted. A finding resting on a proxy, an undated flag or a date without a "
                "time is worth less, and the score says so.",
            ),
            PriorityTerm(
                "known_exception_strength", "Known exception", round(exception, 4), exception,
                self.weights["known_exception_strength"],
                self.weights["known_exception_strength"] * exception,
                "Subtracted. A declared, applicable exclusion lowers priority.",
            ),
        ]

        logit = sum(t.contribution for t in terms)
        value = 100.0 / (1.0 + math.exp(-logit))
        band, label = self._band(value)
        return PriorityScore(value=value, logit=logit, terms=terms, band=band, band_label=label)

    # ---------------------------------------------------------------- helpers

    def _recency(self, signals: Sequence[Signal], as_of: _dt.date) -> float:
        age = self._age_days(signals, as_of)
        if age is None:
            return 0.0
        return float(0.5 ** (age / max(self.recency_half_life, 1.0)))

    @staticmethod
    def _age_days(signals: Sequence[Signal], as_of: _dt.date) -> int | None:
        times = [s.event_time for s in signals if s.event_time is not None]
        if not times:
            return None
        newest = max(times)
        return max((as_of - newest.date()).days, 0)

    @staticmethod
    def _band(value: float) -> tuple[str, str]:
        for cut, band, label in _BANDS:
            if value >= cut:
                return band, label
        return "P5", "Watch only"
