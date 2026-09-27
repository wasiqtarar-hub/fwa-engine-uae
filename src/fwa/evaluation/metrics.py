"""Operational metrics (manuscript §6.5) — including the ones that cannot be computed.

    "Where a metric requires operational data this dataset cannot provide
     (appeals, turnaround), IMPLEMENT THE METRIC and render it as
     ``NOT_MEASURABLE_ON_THIS_DATASET`` with the reason — do not fabricate a
     number, and do not quietly drop the metric."         — build brief §11.1

Every metric §6.5 names is implemented here. Some return values; some return
``NOT_MEASURABLE_ON_THIS_DATASET`` with the specific missing input and what
would supply it. Both are results.

**This is the only module permitted to read the held-out labels.** Build brief
§3.3: "Use labels only in ``src/fwa/evaluation/``". Everything upstream loads
features through a :class:`~fwa.features.frame.FeatureFrame`, which cannot
contain them, and the governance test asserts the separation holds.

Two things this module refuses to do, because §6.5 is explicit about both:

* **Gross flagged value is not savings.** ``flagged_value_aed`` and
  ``confirmed_aed`` are separate fields and are never added together. An
  unestablished exposure is excluded from every savings figure by construction,
  not by a filter someone might forget.
* **Headline precision is an upper bound.** Labels here carry three different
  ``ground_truth_source`` values with different reliability, and every precision
  figure is reported stratified by that source with the caveat attached.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
from .prose import count

__all__ = ["Metric", "NOT_MEASURABLE", "MetricSuite"]

NOT_MEASURABLE = "NOT_MEASURABLE_ON_THIS_DATASET"


@dataclass
class Metric:
    name: str
    value: Any
    unit: str
    status: str                    # MEASURED | NOT_MEASURABLE_ON_THIS_DATASET
    definition: str
    note: str = ""
    required_data: str = ""

    def to_row(self) -> dict[str, Any]:
        return {
            "metric": self.name,
            "value": self.value if self.status == "MEASURED" else NOT_MEASURABLE,
            "unit": self.unit,
            "status": self.status,
            "definition": self.definition,
            "note": self.note,
            "required_data": self.required_data,
        }


class MetricSuite:
    """Computes every §6.5 metric, or says precisely why it cannot."""

    def __init__(self, result, labels: pd.DataFrame | None = None) -> None:
        self.result = result
        self.config = result.config
        self.labels = labels if labels is not None else result.held_out_labels
        self.review_cost = float(self.config.get("fully_loaded_review_cost_per_case_aed"))

    # ------------------------------------------------------------------ label

    def label_series(self) -> pd.Series:
        """``claim_sk -> 1/0``. Evaluation only."""
        if self.labels is None or self.labels.empty:
            return pd.Series(dtype=float)
        return self.labels.set_index("claim_sk")["fraud_label"].astype(float)

    def flagged_claims(self, level: str = "claim") -> set[str]:
        """Claims flagged, at a stated level of attribution.

        This distinction matters enough to be a method parameter rather than a
        convenience. A CLAIM-subject signal says "this claim is the finding". A
        PROVIDER-subject signal says "this provider is the finding", and it
        carries that provider's claim list as context — not as an assertion
        that each of those claims is individually suspect.

        Counting an entity lead as flagging every one of that entity's claims
        inflates apparent coverage enormously (on this dataset it takes claim
        coverage from ~10% to ~84%) and would make a claim-level precision
        figure meaningless. Both levels are therefore computed and reported
        separately, and the headline claim-level figure uses ``"claim"``.
        """
        if level == "claim":
            subjects = {"claim", "episode", "member"}
            return {
                c for s in self.result.signals
                if s.subject_type in subjects
                for c in s.claim_ids
            }
        if level == "entity":
            subjects = {"provider", "agent", "community", "cluster"}
            return {
                c for s in self.result.signals
                if s.subject_type in subjects
                for c in s.claim_ids
            }
        return {c for s in self.result.signals for c in s.claim_ids}

    def case_outcomes(self) -> pd.Series:
        """Per-case 'contains at least one labelled-fraud claim'. Evaluation only.

        This is the outcome series handed to the model promotion gate. It is
        *not* a reviewer outcome — it is a held-out label — and every consumer
        is told so.
        """
        labels = self.label_series()
        if labels.empty:
            return pd.Series(dtype=float)
        rows = {}
        for case in self.result.cases.cases.values():
            claims = [c for c in case.claim_ids if c in labels.index]
            rows[case.case_id] = float(labels.loc[claims].max()) if claims else 0.0
        return pd.Series(rows)

    def provider_outcomes(self) -> pd.Series:
        labels = self.label_series()
        if labels.empty:
            return pd.Series(dtype=float)
        claims = self.result.claims.set_index("claim_sk")
        joined = claims.join(labels.rename("fraud_label"))
        return joined.groupby("provider_sk")["fraud_label"].max()

    # ---------------------------------------------------------------- metrics

    def compute(self) -> list[Metric]:
        labels = self.label_series()
        flagged = self.flagged_claims()
        out: list[Metric] = []

        entity_flagged = self.flagged_claims("entity")
        any_flagged = self.flagged_claims("any")

        # ---- precision ----------------------------------------------------
        if labels.empty:
            out.append(Metric(
                "Precision (confirmed ÷ reviewed)", None, "ratio", NOT_MEASURABLE,
                "Confirmed issues divided by reviewed signals.",
                required_data="review_outcome.validated_category"))
        else:
            base_rate = float(labels.mean())
            valid = [c for c in flagged if c in labels.index]
            precision = float(labels.loc[valid].mean()) if valid else 0.0
            out.append(Metric(
                "Precision proxy — CLAIM-level controls", round(precision, 4), "ratio", "MEASURED",
                "Share of claims flagged by a CLAIM/EPISODE/MEMBER-subject control that carry a "
                "fraud label.",
                note=(f"{len(valid):,} claims flagged ({len(valid) / max(len(labels), 1):.1%} of "
                      f"the file) against a base rate of {base_rate:.1%} — a lift of "
                      f"{precision / max(base_rate, 1e-9):.2f}×. This is a PROXY for precision, "
                      f"not precision: real precision is confirmed ÷ REVIEWED and nothing has "
                      f"been reviewed. The labels are investigation-derived and selection-biased, "
                      f"so this is an UPPER BOUND.")))

            entity_valid = [c for c in entity_flagged if c in labels.index]
            entity_precision = float(labels.loc[entity_valid].mean()) if entity_valid else 0.0
            out.append(Metric(
                "Precision proxy — ENTITY leads (context claims)", round(entity_precision, 4),
                "ratio", "MEASURED",
                "Share of claims belonging to a flagged PROVIDER/AGENT/COMMUNITY that carry a "
                "fraud label.",
                note=(f"{len(entity_valid):,} claims sit under an entity lead. An entity lead "
                      f"does NOT assert that each of those claims is suspect — it asserts that "
                      f"the entity's pattern warrants review — so this figure is close to the "
                      f"base rate of {base_rate:.1%} by construction and is reported to make "
                      f"that visible, not as a performance claim.")))

            out.append(Metric(
                "Claim coverage — any control", round(len(any_flagged) / max(len(labels), 1), 4),
                "ratio", "MEASURED",
                "Share of all claims touched by at least one signal, at any attribution level.",
                note="Reported so that the difference between the two figures above is "
                     "inspectable rather than implicit."))

        # ---- flagged value vs confirmed value -----------------------------
        established = sum(
            c.exposure_aed for c in self.result.cases.cases.values() if c.exposure_established
        )
        unestablished = sum(
            c.exposure_aed for c in self.result.cases.cases.values() if not c.exposure_established
        )
        out.append(Metric(
            "Gross flagged value (established exposure)", round(established, 2), "AED", "MEASURED",
            "Sum of case exposure where an objective condition was shown to fail.",
            note=("GROSS FLAGGED VALUE IS NOT SAVINGS. Nothing here has been reviewed, "
                  "confirmed, prevented or recovered.")))
        out.append(Metric(
            "Gross flagged value (exposure NOT established)", round(unestablished, 2), "AED",
            "MEASURED",
            "Sum of case gross amounts where no objective condition has been shown to fail.",
            note=("Reported SEPARATELY and never added to the figure above: a model-only "
                  "lead shows its gross amount with 'exposure not yet established'.")))
        out.append(Metric(
            "Confirmed AED", None, "AED", NOT_MEASURABLE,
            "Value of issues a reviewer has actually validated.",
            required_data="review_outcome.confirmed_amount_aed"))
        out.append(Metric(
            "Prevented / recovered AED", None, "AED", NOT_MEASURABLE,
            "Value stopped pre-payment or recovered post-payment.",
            required_data="remittance.* (no payment records exist in the claim extract)"))

        # ---- net savings after review cost --------------------------------
        n_cases = len(self.result.cases.cases)
        review_cost_total = n_cases * self.review_cost
        out.append(Metric(
            "Net savings after review cost", None, "AED", NOT_MEASURABLE,
            "Recovered/prevented value minus the fully loaded cost of the review effort.",
            note=(f"The COST side is computable: {n_cases:,} cases × AED {self.review_cost:,.2f} "
                  f"= AED {review_cost_total:,.2f} of review effort. The SAVINGS side is not, "
                  f"because nothing has been confirmed. Reporting the cost alone would imply a "
                  f"negative return that is equally unevidenced."),
            required_data="review_outcome.confirmed_amount_aed and remittance.*"))

        # ---- abrasion, turnaround, overturn -------------------------------
        out.append(Metric(
            "Provider/member abrasion", None, "rate", NOT_MEASURABLE,
            "Complaints, appeals or relationship harm attributable to false positives.",
            required_data="complaint.* and review_outcome.appeal_result"))
        out.append(Metric(
            "Pend turnaround", None, "hours", NOT_MEASURABLE,
            "Time from a PREPAY_PEND disposition to resolution.",
            required_data="review_outcome.decided_at (populated as reviewers work the queue)"))
        out.append(Metric(
            "Overturn / appeal rate", None, "ratio", NOT_MEASURABLE,
            "Share of dispositions reversed on appeal.",
            required_data="review_outcome.appeal_result"))

        # ---- rule stability -----------------------------------------------
        out.append(Metric(
            "Rule stability", None, "revisions/rule", NOT_MEASURABLE,
            "Rate at which a rule's parameters or exclusions must be revised shortly after "
            "activation.",
            note="Every control in this build is in SHADOW; none has been activated, so none "
                 "has had a post-activation revision to count.",
            required_data="an activation history spanning at least one revision cycle"))

        # ---- alerts per reviewer ------------------------------------------
        capacity = int(self.config.get("alerts_per_reviewer_per_day")) * int(
            self.config.get("reviewer_count"))
        out.append(Metric(
            "Alerts per reviewer", round(n_cases / max(int(self.config.get("reviewer_count")), 1), 1),
            "cases/reviewer/run", "MEASURED",
            "Capacity and burnout indicator.",
            note=(f"{n_cases:,} cases across "
                  f"{count(int(self.config.get('reviewer_count')), 'reviewer')}. "
                  f"Daily capacity is {capacity} alerts, so this run's queue represents "
                  f"{n_cases / max(capacity, 1):.1f} days of review work.")))

        out.append(Metric(
            "Time to case disposition", None, "hours", NOT_MEASURABLE,
            "Time from case creation to a recorded disposition.",
            required_data="review_outcome.decided_at"))

        return out

    # ------------------------------------------------ precision@k and by-source

    def precision_at_capacity(self, capacities: Sequence[int] | None = None) -> pd.DataFrame:
        """Capacity-aware precision (§6.6).

        "capacity-aware precision evaluates precision specifically at the alert
        volume a review team can actually process, rather than at an arbitrary
        threshold that ignores operational capacity."
        """
        labels = self.label_series()
        if labels.empty:
            return pd.DataFrame()
        per_reviewer = int(self.config.get("alerts_per_reviewer_per_day"))
        reviewers = int(self.config.get("reviewer_count"))
        capacities = capacities or [
            per_reviewer, per_reviewer * reviewers, per_reviewer * reviewers * 5,
            per_reviewer * reviewers * 20,
        ]
        queue = self.result.cases.to_frame(self.result.tenant_id)
        outcomes = self.case_outcomes()
        rows = []
        for k in capacities:
            top = queue.head(k)
            if top.empty:
                continue
            y = outcomes.reindex(top["case_id"]).fillna(0.0)
            rows.append({
                "capacity_cases": int(k),
                "reviewer_days": round(k / max(per_reviewer * reviewers, 1), 2),
                "cases_reviewed": int(len(top)),
                "precision_proxy": round(float(y.mean()), 4),
                "labelled_cases": int(y.sum()),
                "exposure_reviewed_aed": round(float(top["exposure_aed"].sum()), 2),
            })
        return pd.DataFrame(rows)

    def by_ground_truth_source(self) -> pd.DataFrame:
        """Precision stratified by ``ground_truth_source`` (build brief §3.3).

        "Report metrics stratified by ``ground_truth_source`` and state in the
        report that ``expert_review`` and ``rule_engine`` labels are exactly the
        kind of biased ground truth §1.3 and §6.4 warn against."
        """
        if self.labels is None or self.labels.empty:
            return pd.DataFrame()
        flagged = self.flagged_claims("claim")
        df = self.labels.copy()
        df["flagged"] = df["claim_sk"].isin(flagged)
        rows = []
        for source, sub in df.groupby("ground_truth_source"):
            fraud = sub[sub["fraud_label"] == 1]
            rows.append({
                "ground_truth_source": source,
                "claims": int(len(sub)),
                "labelled_fraud": int(len(fraud)),
                "prevalence": round(len(fraud) / max(len(sub), 1), 4),
                "flagged": int(sub["flagged"].sum()),
                "flagged_and_labelled": int((sub["flagged"] & (sub["fraud_label"] == 1)).sum()),
                "precision_proxy": round(
                    float(sub.loc[sub["flagged"], "fraud_label"].mean()) if sub["flagged"].any() else 0.0, 4),
                "recall_proxy": round(
                    float(fraud["flagged"].mean()) if len(fraud) else 0.0, 4),
                "reliability_caveat": (
                    "rule_engine labels are produced by a rule process, so a rule-based detector "
                    "scoring well against them is close to tautological."
                    if source == "rule_engine" else
                    "expert_review labels carry investigation-selection bias: they exist because "
                    "someone chose to investigate."
                    if source == "expert_review" else
                    "pattern_detection labels are produced by a pattern process and share its "
                    "blind spots."
                ),
            })
        return pd.DataFrame(rows)

    def recall_by_fraud_type(self) -> pd.DataFrame:
        """Per-``fraud_type`` recall at both attribution levels."""
        if self.labels is None or self.labels.empty:
            return pd.DataFrame()
        claim_rules: dict[str, set[str]] = {}
        entity_rules: dict[str, set[str]] = {}
        for s in self.result.signals:
            target = claim_rules if s.subject_type in {"claim", "episode", "member"} else entity_rules
            for c in s.claim_ids:
                target.setdefault(c, set()).add(s.rule_id)
        df = self.labels.copy()
        df["claim_flagged"] = df["claim_sk"].isin(claim_rules)
        df["entity_flagged"] = df["claim_sk"].isin(entity_rules)
        rows = []
        for ftype, sub in df[df["fraud_label"] == 1].groupby("fraud_type"):
            rules: dict[str, int] = {}
            for c in sub.loc[sub["claim_flagged"], "claim_sk"]:
                for r in claim_rules.get(c, ()):
                    rules[r] = rules.get(r, 0) + 1
            top = sorted(rules.items(), key=lambda kv: -kv[1])[:4]
            rows.append({
                "fraud_type": ftype,
                "labelled_claims": int(len(sub)),
                "flagged_by_claim_control": int(sub["claim_flagged"].sum()),
                "recall_claim_level": round(float(sub["claim_flagged"].mean()), 4),
                "also_under_an_entity_lead": int(sub["entity_flagged"].sum()),
                "top_claim_level_controls": "; ".join(f"{r} ({n})" for r, n in top) or "— none —",
            })
        return pd.DataFrame(rows).sort_values("labelled_claims", ascending=False)

    def lift_over_composite(self) -> pd.DataFrame:
        """Each layer's review yield against the transparent composite (§4.8)."""
        outcomes = self.provider_outcomes()
        if outcomes.empty:
            return pd.DataFrame()
        capacity = int(self.config.get("alerts_per_reviewer_per_day")) * int(
            self.config.get("reviewer_count"))

        rankings: dict[str, pd.Series] = {}
        composite_results, _ = self.result.composite
        rankings["ANL-01-R01 transparent composite"] = pd.Series(
            {r.provider_sk: r.score for r in composite_results if not r.insufficient_evidence})
        if self.result.models is not None:
            for name, series in self.result.models.provider_scores().items():
                rankings[f"model: {name}"] = series

        rule_scores: dict[str, float] = {}
        for s in self.result.signals:
            if s.subject_type == "provider":
                rule_scores[s.subject_id] = max(rule_scores.get(s.subject_id, 0.0), s.score)
        if rule_scores:
            rankings["deterministic + statistical rules"] = pd.Series(rule_scores)

        baseline = None
        rows = []
        for label, series in rankings.items():
            common = series.index.intersection(outcomes.index)
            if len(common) == 0:
                continue
            k = min(capacity, len(common))
            top = series.loc[common].nlargest(k).index
            yield_rate = float(outcomes.loc[top].mean())
            if "transparent composite" in label:
                baseline = yield_rate
            rows.append({
                "layer": label,
                "entities_ranked": int(len(common)),
                "capacity_k": int(k),
                "review_yield_proxy": round(yield_rate, 4),
            })
        base_rate = float(outcomes.mean())
        for row in rows:
            row["lift_over_base_rate"] = round(row["review_yield_proxy"] / max(base_rate, 1e-9), 2)
            row["lift_over_composite"] = (
                round(row["review_yield_proxy"] / baseline, 2) if baseline else None
            )
        frame = pd.DataFrame(rows)
        if not frame.empty:
            frame.attrs["base_rate"] = base_rate
        return frame

    def calibration_curve(self, bins: int = 8) -> pd.DataFrame:
        """Predicted-vs-observed for the case priority score."""
        outcomes = self.case_outcomes()
        if outcomes.empty:
            return pd.DataFrame()
        queue = self.result.cases.to_frame(self.result.tenant_id)
        frame = queue[["case_id", "priority"]].copy()
        frame["outcome"] = frame["case_id"].map(outcomes).fillna(0.0)
        frame = frame.dropna(subset=["priority"])
        if frame.empty:
            return pd.DataFrame()
        frame["bin"] = pd.qcut(frame["priority"], q=min(bins, frame["priority"].nunique()),
                               duplicates="drop")
        grouped = frame.groupby("bin", observed=True).agg(
            cases=("case_id", "size"),
            mean_priority=("priority", "mean"),
            observed_rate=("outcome", "mean"),
        ).reset_index()
        grouped["bin"] = grouped["bin"].astype(str)
        grouped["predicted_rate"] = grouped["mean_priority"] / 100.0
        grouped["gap"] = grouped["predicted_rate"] - grouped["observed_rate"]
        return grouped.round(4)
