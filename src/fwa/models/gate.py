"""The model promotion gate (manuscript §4.8, Table 6.1).

    "Model | Temporal holdout evaluation; calibration check; DEMONSTRATED
     PROSPECTIVE LIFT OVER THE SIMPLE STATISTICAL BASELINE; drift and
     explanation-quality tests"                             — Table 6.1

    "an opaque model is only added to the production signal set once it has been
     shown, PROSPECTIVELY, to catch something the transparent composite does
     not."                                                  — manuscript §4.8

The build brief adds the instruction that matters most here: *"print the verdict
— including when the honest verdict is the transparent composite was not beaten.
That is a legitimate, publishable finding; do not tune until the model wins."*

This gate is therefore written to be **failable**. It has five criteria, each
returning PASS, FAIL or NOT_ASSESSABLE with its evidence, and a model is
promoted only if every criterion passes. ``NOT_ASSESSABLE`` never counts as a
pass — a criterion that cannot be evaluated is a reason not to promote, not a
reason to shrug.

One note on where the outcome labels come from. Lift is measured against an
outcome series supplied by :mod:`fwa.evaluation`, which is the only module
permitted to read the held-out labels. The gate itself never touches them, and
the verdict records that the outcomes used on this dataset are
investigation-derived and selection-biased, so the lift figure is an upper
bound rather than an estimate of production behaviour.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np
import pandas as pd
from ..evaluation.prose import count

__all__ = ["GateCriterion", "GateVerdict", "ModelPromotionGate"]

PASS = "PASS"
FAIL = "FAIL"
NOT_ASSESSABLE = "NOT_ASSESSABLE"


@dataclass
class GateCriterion:
    name: str
    status: str
    evidence: str
    value: float | None = None
    threshold: float | None = None

    def to_row(self) -> dict[str, Any]:
        return {
            "criterion": self.name,
            "status": self.status,
            "value": None if self.value is None else round(float(self.value), 4),
            "threshold": self.threshold,
            "evidence": self.evidence,
        }


@dataclass
class GateVerdict:
    model: str
    promoted: bool
    criteria: list[GateCriterion] = field(default_factory=list)
    summary: str = ""

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame([c.to_row() for c in self.criteria])

    @property
    def headline(self) -> str:
        return f"{self.model}: {'PROMOTED TO ACTIVE' if self.promoted else 'REMAINS IN SHADOW'}"


class ModelPromotionGate:
    def __init__(self, config) -> None:
        self.config = config
        self.min_lift = float(config.get("model_promotion_min_lift"))
        self.max_ece = float(config.get("model_calibration_max_ece"))

    # ------------------------------------------------------------------ run

    def evaluate(
        self,
        *,
        model_name: str,
        model_scores: pd.Series,
        composite_ranking: pd.Series,
        split_plan: dict[str, Any],
        drift_results: Sequence[Any],
        explanation_sample: Sequence[dict[str, Any]],
        outcomes: pd.Series | None = None,
        capacity: int | None = None,
    ) -> GateVerdict:
        criteria: list[GateCriterion] = []

        # ---- 1. temporal holdout ------------------------------------------
        overlap = split_plan.get("provider_overlap", None)
        if not split_plan:
            criteria.append(GateCriterion(
                "Temporal holdout evaluation", NOT_ASSESSABLE,
                "No split plan was recorded for this model."))
        elif overlap == 0 and split_plan.get("train_claims", 0) > 0:
            criteria.append(GateCriterion(
                "Temporal holdout evaluation", PASS,
                f"Trained on {split_plan['train_claims']:,} claims before "
                f"{split_plan['split_date']}; scored {split_plan['score_claims']:,} claims after "
                f"it. Entity isolation holds: {overlap} providers appear in both sets. "
                f"{split_plan.get('coverage_note', '')}"))
        else:
            criteria.append(GateCriterion(
                "Temporal holdout evaluation", FAIL,
                f"{count(overlap, 'provider')} appear in both the training and scoring sets; "
                f"requires entity isolation."))

        # ---- 2. prospective lift over the transparent composite -----------
        criteria.append(self._lift_criterion(model_scores, composite_ranking, outcomes, capacity))

        # ---- 3. calibration -----------------------------------------------
        criteria.append(self._calibration_criterion(model_scores, outcomes))

        # ---- 4. drift ------------------------------------------------------
        breaches = [r for r in drift_results if getattr(r, "status", "") == "BREACH"]
        warns = [r for r in drift_results if getattr(r, "status", "") == "WARN"]
        if breaches:
            criteria.append(GateCriterion(
                "Drift monitoring", FAIL,
                "Drift breach on: " + "; ".join(r.monitor for r in breaches)))
        elif not drift_results:
            criteria.append(GateCriterion("Drift monitoring", NOT_ASSESSABLE, "No drift monitors ran."))
        else:
            criteria.append(GateCriterion(
                "Drift monitoring", PASS,
                f"{count(len(drift_results), 'monitor')} ran; "
                + (f"{count(len(warns), 'warning')}, " if warns else "")
                + "no breaches."))

        # ---- 5. explanation quality ---------------------------------------
        usable = [
            e for e in explanation_sample
            if e.get("feature") not in ("EXPLANATION_UNAVAILABLE", "EXPLANATION_FAILED")
            and e.get("value_display") not in (None, "not available")
        ]
        if not explanation_sample:
            criteria.append(GateCriterion(
                "Explanation quality", NOT_ASSESSABLE, "No explanation sample was supplied."))
        elif len(usable) >= max(1, len(explanation_sample) // 2):
            criteria.append(GateCriterion(
                "Explanation quality", PASS,
                f"{len(usable)} of {len(explanation_sample)} sampled contributions render in "
                f"ORIGINAL UNITS beside a peer value, as the promotion gate requires."))
        else:
            criteria.append(GateCriterion(
                "Explanation quality", FAIL,
                f"Only {len(usable)} of {len(explanation_sample)} contributions render usably. "
                f"the promotion gate requires top contributing features in original units with peer comparison."))

        promoted = all(c.status == PASS for c in criteria)
        verdict = GateVerdict(model=model_name, promoted=promoted, criteria=criteria)
        verdict.summary = self._summarise(verdict)
        return verdict

    # ------------------------------------------------------------ criteria

    def _lift_criterion(
        self,
        model_scores: pd.Series,
        composite_ranking: pd.Series,
        outcomes: pd.Series | None,
        capacity: int | None,
    ) -> GateCriterion:
        if outcomes is None or outcomes.empty:
            return GateCriterion(
                "Prospective lift over ANL-01-R01", NOT_ASSESSABLE,
                "Lift is defined as FUTURE-PERIOD REVIEW YIELD, not in-sample "
                "separation, and review yield needs confirmed reviewer outcomes. None exist "
                "yet in this artefact, so the criterion cannot be assessed — which means the "
                "model cannot be promoted. NOT_ASSESSABLE is not a pass.",
                threshold=self.min_lift,
            )
        k = capacity or max(1, int(len(model_scores) * 0.05))
        common = model_scores.index.intersection(outcomes.index)
        if len(common) < k:
            return GateCriterion(
                "Prospective lift over ANL-01-R01", NOT_ASSESSABLE,
                f"Only {len(common)} scored entities have an outcome; fewer than the review "
                f"capacity of {k}.", threshold=self.min_lift)

        model_top = model_scores.loc[common].nlargest(k).index
        model_yield = float(outcomes.loc[model_top].mean())

        comp = composite_ranking.reindex(common).dropna()
        if comp.empty:
            return GateCriterion(
                "Prospective lift over ANL-01-R01", NOT_ASSESSABLE,
                "The transparent composite produced no ranking over the scored entities, so "
                "there is no benchmark to beat.", threshold=self.min_lift)
        comp_top = comp.nlargest(min(k, len(comp))).index
        comp_yield = float(outcomes.loc[comp_top].mean())

        if comp_yield <= 0:
            lift = float("inf") if model_yield > 0 else 0.0
        else:
            lift = model_yield / comp_yield

        status = PASS if lift >= self.min_lift else FAIL
        return GateCriterion(
            "Prospective lift over ANL-01-R01", status,
            f"At a reviewer capacity of {k}, the model's review yield is {model_yield:.1%} "
            f"against the transparent composite's {comp_yield:.1%} — a lift of {lift:.2f}× "
            f"against a required {self.min_lift:.2f}×. "
            + ("The model beats the transparent composite." if status == PASS else
               "THE TRANSPARENT COMPOSITE WAS NOT BEATEN. This is a legitimate finding and the "
               "model stays in shadow; it is not a reason to retune until the model wins.")
            + " NOTE: the outcomes used are investigation-derived, selection-biased labels "
              ", so this lift is an upper bound, not an estimate of production behaviour.",
            value=lift, threshold=self.min_lift,
        )

    def _calibration_criterion(self, scores: pd.Series, outcomes: pd.Series | None) -> GateCriterion:
        if outcomes is None or outcomes.empty:
            return GateCriterion(
                "Calibration check", NOT_ASSESSABLE,
                "Calibration checks that a stated confidence corresponds to the observed rate "
                "of confirmation; with no confirmed outcomes there is no observed rate.",
                threshold=self.max_ece)
        common = scores.index.intersection(outcomes.index)
        if len(common) < 50:
            return GateCriterion(
                "Calibration check", NOT_ASSESSABLE,
                f"Only {len(common)} scored entities have an outcome; too few to bin.",
                threshold=self.max_ece)
        s = scores.loc[common]
        y = outcomes.loc[common].astype(float)
        # min-max to [0,1]; an unsupervised score is not a probability, and
        # saying so is part of the finding.
        p = (s - s.min()) / max(s.max() - s.min(), 1e-9)
        bins = pd.qcut(p, q=min(10, max(2, len(p) // 25)), duplicates="drop")
        grouped = pd.DataFrame({"p": p, "y": y, "bin": bins}).groupby("bin", observed=True)
        ece = float((grouped["p"].mean() - grouped["y"].mean()).abs().mul(
            grouped.size() / len(p)).sum())
        status = PASS if ece <= self.max_ece else FAIL
        return GateCriterion(
            "Calibration check", status,
            f"Expected calibration error {ece:.3f} against a maximum of {self.max_ece:.3f}. "
            f"An unsupervised anomaly score is NOT a probability; it is min-max rescaled here "
            f"purely so a calibration curve can be drawn, and a poor ECE on an unsupervised "
            f"score is expected rather than surprising.",
            value=ece, threshold=self.max_ece)

    @staticmethod
    def _summarise(verdict: GateVerdict) -> str:
        if verdict.promoted:
            return (
                f"{verdict.model} clears every model-gate criterion and may be "
                f"promoted from shadow to active by a policy owner who did not propose it."
            )
        failed = [c.name for c in verdict.criteria if c.status == FAIL]
        unassessable = [c.name for c in verdict.criteria if c.status == NOT_ASSESSABLE]
        parts = [f"{verdict.model} REMAINS IN SHADOW."]
        if failed:
            parts.append("Failed: " + "; ".join(failed) + ".")
        if unassessable:
            parts.append("Not assessable: " + "; ".join(unassessable) + ".")
        parts.append(
            "The system loses nothing by this: ANL-01-R01, the transparent composite, is the "
            "explainable fallback and continues to run."
        )
        return " ".join(parts)
