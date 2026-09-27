"""The supervised-modelling prerequisite gate (manuscript §6.4, §4.8).

    "Supervised modelling: explicitly out of scope for now... Doing so before
     outcome capture (review_outcome) is stable and before random-audit data
     exists to correct for investigation-selection bias would produce a model
     that learns to reproduce the blind spots of whatever detection process
     generated its training labels, a well-documented failure mode this
     dissertation treats as DISQUALIFYING, NOT MERELY UNDESIRABLE."
                                                            — manuscript §4.8

The build brief asks for this to be made concrete: *"add
``models/supervised_gate.py`` that checks the five prerequisites... evaluates
them against this dataset, and returns ``PROMOTION_BLOCKED`` with the specific
unmet prerequisites. Surface that verdict in the UI. **Train nothing.**"*

Nothing is trained here. There is no ``fit``, no estimator, no import of any
classifier. This module evaluates five conditions and returns a verdict, and
that is the whole of its behaviour — which is itself the point being made: the
answer to "should we build a supervised model?" is a governance question that
gets a governance answer, not a modelling experiment that gets an AUC.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd
from ..evaluation.prose import count

__all__ = ["SupervisedGate", "SupervisedGateVerdict", "Prerequisite", "PROMOTION_BLOCKED"]

PROMOTION_BLOCKED = "PROMOTION_BLOCKED"
PROMOTION_PERMITTED = "PROMOTION_PERMITTED"


@dataclass
class Prerequisite:
    name: str
    met: bool
    evidence: str
    what_would_satisfy_it: str

    def to_row(self) -> dict[str, Any]:
        return {
            "prerequisite": self.name,
            "met": self.met,
            "evidence": self.evidence,
            "what_would_satisfy_it": self.what_would_satisfy_it,
        }


@dataclass
class SupervisedGateVerdict:
    status: str
    prerequisites: list[Prerequisite] = field(default_factory=list)
    summary: str = ""

    @property
    def unmet(self) -> list[Prerequisite]:
        return [p for p in self.prerequisites if not p.met]

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame([p.to_row() for p in self.prerequisites])


class SupervisedGate:
    """Checks §6.4's five prerequisites. Trains nothing, ever."""

    def evaluate(
        self,
        *,
        review_outcomes: pd.DataFrame | None,
        random_audit_performed: bool,
        temporal_split_available: bool,
        entity_isolation_available: bool,
        calibration_and_stability_available: bool,
        label_source_counts: dict[str, int] | None = None,
    ) -> SupervisedGateVerdict:
        n_outcomes = 0 if review_outcomes is None else len(review_outcomes)
        sources = label_source_counts or {}

        prereqs = [
            Prerequisite(
                name="Stable outcome capture (review_outcome)",
                met=n_outcomes >= 500,
                evidence=(
                    f"{count(n_outcomes, 'confirmed reviewer outcome')} recorded. A supervised model "
                    f"needs a stable, high-volume stream of validated dispositions; this "
                    f"artefact has just begun capturing them."
                ),
                what_would_satisfy_it=(
                    "Several hundred reviewer dispositions with validated categories and "
                    "confirmed amounts, captured consistently over at least two quarters."
                ),
            ),
            Prerequisite(
                name="Random-audit data to correct investigation-selection bias",
                met=bool(random_audit_performed),
                evidence=(
                    "A random-audit SIMULATOR runs in this artefact and estimates a "
                    "false-negative rate from the held-out labels, but a simulated audit over "
                    "labels that were themselves produced by a detection process cannot correct "
                    "for that process's blind spots. "
                    + (
                        "The three ground_truth_source values present here — "
                        + ", ".join(f"{k} ({v})" for k, v in sorted(sources.items()))
                        + " — are exactly the biased ground truth warn about."
                        if sources else ""
                    )
                ),
                what_would_satisfy_it=(
                    "A genuine random audit: a statistically representative sample of claims "
                    "reviewed by humans REGARDLESS of whether the system flagged them."
                ),
            ),
            Prerequisite(
                name="Temporal train/validation/test splits",
                met=bool(temporal_split_available),
                evidence=(
                    "The model layer already performs a strict temporal split on date_of_claim, "
                    "so this prerequisite is satisfied by the existing pipeline."
                    if temporal_split_available else
                    "No temporal split is available."
                ),
                what_would_satisfy_it="A date-ordered split with no future data in training.",
            ),
            Prerequisite(
                name="Entity isolation across splits",
                met=bool(entity_isolation_available),
                evidence=(
                    "Provider and member entities are held out between training and scoring, so "
                    "this prerequisite is satisfied."
                    if entity_isolation_available else
                    "Entities straddle the splits."
                ),
                what_would_satisfy_it=(
                    "No provider, member or episode present in both a training and an "
                    "evaluation split."
                ),
            ),
            Prerequisite(
                name="Calibration and specialty/payer stability",
                met=bool(calibration_and_stability_available),
                evidence=(
                    "Calibration requires confirmed outcomes to calibrate against, and "
                    "specialty stability requires a provider specialty field. the claim extract has "
                    "neither: specialty is proxied by ICD chapter and outcomes are not yet "
                    "captured."
                ),
                what_would_satisfy_it=(
                    "Confirmed outcomes for a calibration curve, plus a real provider specialty "
                    "and payer dimension to check stability across."
                ),
            ),
        ]

        blocked = any(not p.met for p in prereqs)
        verdict = SupervisedGateVerdict(
            status=PROMOTION_BLOCKED if blocked else PROMOTION_PERMITTED,
            prerequisites=prereqs,
        )
        unmet = [p.name for p in prereqs if not p.met]
        verdict.summary = (
            f"{PROMOTION_BLOCKED}. {len(unmet)} of 5 prerequisites are unmet: "
            + "; ".join(unmet)
            + ". No supervised model is trained, and none should be: a propensity model fitted "
              "to these labels would learn to reproduce the blind spots of whatever process "
              "produced them. The promotion gate treats that failure mode as disqualifying, not merely "
              "undesirable."
            if blocked else
            "PROMOTION_PERMITTED. All five prerequisites are met; supervised modelling may "
            "begin, under the same shadow-before-active discipline as every other control."
        )
        return verdict
