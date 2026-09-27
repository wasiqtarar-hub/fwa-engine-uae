"""The evaluation protocol as runnable code (manuscript §6.6, build brief §11.3).

    "Implement, as runnable code: synthetic-corruption injection (permanently
     marked synthetic — inject known upcoding and duplicate patterns into a
     clean subset and confirm the relevant control detects them), the
     random-audit sampler, temporal holdout, entity isolation, calibration
     check, capacity-aware precision, review-yield tracking, stability, and
     fairness/appeal monitoring across TPA, policy type and provider volume
     band."

Nine procedures. Each returns a result *and* an honest statement of what that
result can and cannot support.

The two that carry the most weight, and the reason why:

**The random-audit sampler** is, per §6.5, "the only metric in this list capable
of estimating the system's FALSE-NEGATIVE RATE rather than only its precision on
what it already flagged". It draws a stratified random sample *irrespective of
flag status* and reveals its labels. It is implemented in full — and its output
carries a warning that a simulated audit over labels produced by a detection
process cannot correct for that process's blind spots, which is precisely the
limitation §6.4 uses to block supervised modelling.

**Synthetic-corruption injection** proves a control detects the pattern it was
designed for, independently of whether the real data happens to contain that
pattern. Every injected record is marked ``SYNTHETIC_INJECTED`` permanently, is
confined to a copy, and never touches the real signal store.
"""

from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

import numpy as np
import pandas as pd
from .prose import count

__all__ = ["ProtocolResult", "EvaluationProtocol"]

SYNTHETIC_INJECTED = "SYNTHETIC_INJECTED"


@dataclass
class ProtocolResult:
    procedure: str
    status: str                 # PASS | FAIL | INFORMATIONAL | NOT_MEASURABLE_ON_THIS_DATASET
    summary: str
    detail: pd.DataFrame = field(default_factory=pd.DataFrame)
    values: dict[str, Any] = field(default_factory=dict)
    caveat: str = ""

    def to_row(self) -> dict[str, Any]:
        return {
            "procedure": self.procedure,
            "status": self.status,
            "summary": self.summary,
            "caveat": self.caveat,
            **{f"value_{k}": v for k, v in self.values.items()},
        }


class EvaluationProtocol:
    def __init__(self, result, metrics) -> None:
        self.result = result
        self.metrics = metrics
        self.config = result.config
        self.rng = np.random.default_rng(int(self.config.get("random_seed")))

    # ------------------------------------------------------------------- run

    def run_all(self) -> list[ProtocolResult]:
        return [
            self.synthetic_corruption_injection(),
            self.random_audit(),
            self.temporal_holdout(),
            self.entity_isolation(),
            self.calibration_check(),
            self.capacity_aware_precision(),
            self.review_yield(),
            self.stability(),
            self.fairness_monitoring(),
        ]

    # ------------------------------------------------- 1. corruption injection

    def synthetic_corruption_injection(self) -> ProtocolResult:
        """Inject known duplicate and upcoding patterns; confirm detection."""
        from ..engine.context import ControlContext
        from ..engine.evaluator import Evaluator
        from ..engine.signals import SignalStore

        claims = self.result.claims.copy()
        clean = claims.sample(min(400, len(claims)), random_state=int(self.config.get("random_seed")))

        injected_rows = []
        # (a) exact duplicates — PAY-01-R01's pattern, by construction
        for i, row in enumerate(clean.head(20).itertuples(index=False)):
            new = {c: getattr(row, c) for c in clean.columns}
            new["claim_sk"] = f"{SYNTHETIC_INJECTED}-DUP-{i:03d}"
            new["claim_date"] = pd.Timestamp(row.claim_date) + pd.Timedelta(days=3)
            injected_rows.append({**new, "_injected": "exact_duplicate"})
        # (b) upcoding proxy — a coding mismatch on an otherwise clean claim
        for i, row in enumerate(clean.tail(20).itertuples(index=False)):
            new = {c: getattr(row, c) for c in clean.columns}
            new["claim_sk"] = f"{SYNTHETIC_INJECTED}-UPC-{i:03d}"
            new["icd_code_matches_procedure"] = False
            new["gross_amount_aed"] = float(row.gross_amount_aed) * 3.0
            injected_rows.append({**new, "_injected": "upcoding"})

        injected = pd.DataFrame(injected_rows)
        corrupted = pd.concat([claims, injected.drop(columns=["_injected"])], ignore_index=True)

        ctx = ControlContext(
            config=self.config, tenant_id=self.result.tenant_id, claims=corrupted,
            features=self.result.features, dataset=self.result.dataset,
            peers=self.result.context.peers, provider_peers=self.result.context.provider_peers,
            provider_metrics=self.result.provider_metrics, episodes=self.result.episodes,
            episode_map=self.result.episode_map, lineage=self.result.lineage,
            run_date=self.result.context.run_date,
        )
        store = SignalStore()
        Evaluator(self.result.registry, store).run(
            ctx, rule_ids=["PAY-01-R01", "CLN-03-R01", "PAY-06-R03"]
        )
        detected = {c for s in store for c in s.claim_ids}

        rows = []
        for pattern, rule in (("exact_duplicate", "PAY-01-R01"), ("upcoding", "CLN-03-R01")):
            targets = injected.loc[injected["_injected"] == pattern, "claim_sk"].tolist()
            found = [t for t in targets if t in detected]
            rows.append({
                "injected_pattern": pattern,
                "expected_control": rule,
                "injected": len(targets),
                "detected": len(found),
                "detection_rate": round(len(found) / max(len(targets), 1), 3),
            })
        frame = pd.DataFrame(rows)
        all_detected = bool((frame["detection_rate"] >= 0.9).all())
        return ProtocolResult(
            procedure="Synthetic-corruption injection",
            status="PASS" if all_detected else "FAIL",
            summary=(
                f"Injected {len(injected)} permanently-marked {SYNTHETIC_INJECTED} records into a "
                f"copy of the claim frame and re-ran the target controls. "
                + "; ".join(
                    f"{r['injected_pattern']}: {r['detected']}/{r['injected']} detected by "
                    f"{r['expected_control']}" for r in rows)
            ),
            detail=frame,
            values={"injected": len(injected), "detected": len(detected & set(injected['claim_sk']))},
            caveat=(
                "This confirms a control detects the pattern it was DESIGNED for. It says nothing "
                "about real-world prevalence or precision. Every injected record is marked "
                f"{SYNTHETIC_INJECTED}, exists only in a copy, and never enters the real signal "
                "store."
            ),
        )

    # ------------------------------------------------------- 2. random audit

    def random_audit(self) -> ProtocolResult:
        """Stratified random audit — the only honest route to a false-negative rate."""
        labels = self.metrics.label_series()
        if labels.empty:
            return ProtocolResult(
                "Random-audit sampler", "NOT_MEASURABLE_ON_THIS_DATASET",
                "No outcome labels are available to reveal.",
            )
        n = int(self.config.get("random_audit_sample_size"))
        claims = self.result.claims.copy()
        claims["_stratum"] = claims["tpa"].astype(str) + "|" + claims["policy_type"].astype(str)

        # proportional stratified sample, irrespective of flag status
        sample_ids: list[str] = []
        for stratum, sub in claims.groupby("_stratum"):
            take = max(1, int(round(n * len(sub) / len(claims))))
            sample_ids.extend(
                sub["claim_sk"].sample(min(take, len(sub)), random_state=self.rng.integers(1e9)).tolist()
            )
        sample = claims[claims["claim_sk"].isin(sample_ids)]
        flagged = self.metrics.flagged_claims("claim")
        y = labels.reindex(sample["claim_sk"]).fillna(0.0)
        is_flagged = sample["claim_sk"].isin(flagged).to_numpy()
        y_arr = y.to_numpy()

        n_pos = int(y_arr.sum())
        missed = int(((y_arr == 1) & (~is_flagged)).sum())
        false_negative_rate = missed / max(n_pos, 1)
        miss_rate = missed / max(len(sample), 1)

        detail = pd.DataFrame([
            {"group": "audited claims", "n": len(sample)},
            {"group": "labelled fraud in sample", "n": n_pos},
            {"group": "labelled fraud the system FLAGGED", "n": n_pos - missed},
            {"group": "labelled fraud the system MISSED", "n": missed},
            {"group": "flagged but not labelled", "n": int(((y_arr == 0) & is_flagged).sum())},
        ])
        return ProtocolResult(
            procedure="Random-audit sampler",
            status="INFORMATIONAL",
            summary=(
                f"Stratified random sample of {len(sample):,} claims drawn by TPA × policy type, "
                f"IRRESPECTIVE of flag status. Estimated FALSE-NEGATIVE RATE "
                f"{false_negative_rate:.1%} ({missed} of {n_pos} labelled-fraud claims in the "
                f"sample were not flagged). Random-audit MISS RATE {miss_rate:.2%} of all audited "
                f"claims."
            ),
            detail=detail,
            values={
                "sample_size": len(sample),
                "false_negative_rate": round(false_negative_rate, 4),
                "random_audit_miss_rate": round(miss_rate, 4),
            },
            caveat=(
                "This is a SIMULATED audit over labels that were themselves produced by detection "
                "processes (pattern_detection, expert_review, rule_engine). It therefore cannot "
                "correct for those processes' blind spots — a claim no process ever flagged is "
                "labelled legitimate here whether or not it was. A genuine random audit needs "
                "HUMAN review of a representative sample regardless of flag status, and its "
                "absence blocks supervised modelling."
            ),
        )

    # --------------------------------------------------- 3/4. splits

    def temporal_holdout(self) -> ProtocolResult:
        models = self.result.models
        if models is None or models.anomaly is None or models.anomaly.split is None:
            return ProtocolResult("Temporal holdout", "NOT_MEASURABLE_ON_THIS_DATASET",
                                  "No model split was produced.")
        plan = models.anomaly.coverage()
        return ProtocolResult(
            "Temporal holdout", "PASS",
            f"Training data ends at {plan['split_date']}; scoring data begins after it. "
            f"{plan['train_claims']:,} training claims, {plan['score_claims']:,} scored claims.",
            detail=pd.DataFrame([plan]),
            values={"split_date": plan["split_date"]},
            caveat=plan.get("coverage_note", ""),
        )

    def entity_isolation(self) -> ProtocolResult:
        models = self.result.models
        if models is None or models.anomaly is None or models.anomaly.split is None:
            return ProtocolResult("Entity isolation", "NOT_MEASURABLE_ON_THIS_DATASET",
                                  "No model split was produced.")
        plan = models.anomaly.coverage()
        overlap = plan["provider_overlap"]
        return ProtocolResult(
            "Entity isolation", "PASS" if overlap == 0 else "FAIL",
            f"{count(overlap, 'provider')} appear in both the training and scoring sets "
            f"({plan['train_providers']} train / {plan['score_providers']} score).",
            values={"provider_overlap": overlap},
            caveat=plan.get("coverage_note", ""),
        )

    # ---------------------------------------------------- 5/6/7. metrics

    def calibration_check(self) -> ProtocolResult:
        curve = self.metrics.calibration_curve()
        if curve.empty:
            return ProtocolResult("Calibration check", "NOT_MEASURABLE_ON_THIS_DATASET",
                                  "No outcomes to calibrate against.")
        ece = float((curve["gap"].abs() * curve["cases"] / curve["cases"].sum()).sum())
        return ProtocolResult(
            "Calibration check", "INFORMATIONAL",
            f"Expected calibration error between the priority score (rescaled to [0,1]) and the "
            f"observed labelled-fraud rate is {ece:.3f} across {len(curve)} bins.",
            detail=curve, values={"ece": round(ece, 4)},
            caveat=(
                "The priority score is NOT a probability — it is a queue-ordering score from the "
                "priority formula. A poor calibration figure here is expected and is not evidence "
                "that the score is broken; it is evidence that it was never a probability."
            ),
        )

    def capacity_aware_precision(self) -> ProtocolResult:
        frame = self.metrics.precision_at_capacity()
        if frame.empty:
            return ProtocolResult("Capacity-aware precision", "NOT_MEASURABLE_ON_THIS_DATASET",
                                  "No outcomes available.")
        head = frame.iloc[min(1, len(frame) - 1)]
        return ProtocolResult(
            "Capacity-aware precision", "INFORMATIONAL",
            f"At one day of team capacity ({int(head['capacity_cases'])} cases), the precision "
            f"proxy is {head['precision_proxy']:.1%}.",
            detail=frame, values={"precision_at_one_day": float(head["precision_proxy"])},
            caveat="Precision proxy, not precision: nothing has been reviewed. Upper bound.",
        )

    def review_yield(self) -> ProtocolResult:
        return ProtocolResult(
            "Review-yield tracking", "NOT_MEASURABLE_ON_THIS_DATASET",
            "Review yield is the confirmed-to-flagged ratio and needs reviewer dispositions.",
            caveat=(
                "Review yield is 'the primary signal of whether a control or model is "
                "still earning its place in production'. Substituting a label-derived proxy here "
                "would be exactly the leakage this artefact is built to prevent, and would make "
                "the model promotion gate meaningless. It is reported as unmeasurable instead."
            ),
        )

    # ------------------------------------------------------- 8. stability

    def stability(self) -> ProtocolResult:
        """Do control outputs drift between periods absent a genuine change?"""
        signals = self.result.signals.to_dataframe(self.result.tenant_id)
        if signals.empty:
            return ProtocolResult("Stability", "NOT_MEASURABLE_ON_THIS_DATASET", "No signals.")
        signals = signals[signals["period_bucket"].str.len() == 7]
        if signals.empty:
            return ProtocolResult("Stability", "NOT_MEASURABLE_ON_THIS_DATASET",
                                  "No monthly-bucketed signals.")
        pivot = signals.pivot_table(index="period_bucket", columns="rule_id",
                                    values="signal_id", aggfunc="count").fillna(0)
        rows = []
        for rule in pivot.columns:
            series = pivot[rule]
            mean = float(series.mean())
            cv = float(series.std() / mean) if mean > 0 else float("nan")
            rows.append({
                "rule_id": rule, "periods": int((series > 0).sum()),
                "mean_signals_per_period": round(mean, 2),
                "coefficient_of_variation": round(cv, 3) if cv == cv else None,
            })
        frame = pd.DataFrame(rows).sort_values("coefficient_of_variation", ascending=False)
        unstable = frame[frame["coefficient_of_variation"] > 1.0]
        return ProtocolResult(
            "Stability", "INFORMATIONAL",
            f"{count(len(frame), 'control')} produced monthly signals. {len(unstable)} show a "
            f"coefficient of variation above 1.0 across periods.",
            detail=frame, values={"unstable_controls": len(unstable)},
            caveat=(
                "High variation here does not necessarily mean an unstable RULE: claim volume "
                "itself varies by month, and several controls are provider-level and fire once "
                "per provider per period by design."
            ),
        )

    # ---------------------------------------------------- 9. fairness

    def fairness_monitoring(self) -> ProtocolResult:
        """Flag rates across TPA, policy type and provider volume band (§6.6, §7.8).

        "fairness/appeal monitoring tracks whether disposition, overturn and
        abrasion rates vary in ways that would indicate the system is
        systematically harder on some legitimate provider or member segment
        than another."
        """
        claims = self.result.claims.copy()
        flagged = self.metrics.flagged_claims("claim")
        claims["flagged"] = claims["claim_sk"].isin(flagged)
        counts = claims.groupby("provider_sk").size()
        claims["provider_volume_band"] = claims["provider_sk"].map(
            lambda p: self.config.peer_groups.volume_band(int(counts.get(p, 0))))
        labels = self.metrics.label_series()
        if not labels.empty:
            claims["labelled"] = claims["claim_sk"].map(labels).fillna(0.0)
        else:
            claims["labelled"] = np.nan

        frames = []
        for dimension in ("tpa", "policy_type", "provider_volume_band"):
            grouped = claims.groupby(dimension).agg(
                claims=("claim_sk", "size"),
                flag_rate=("flagged", "mean"),
                labelled_rate=("labelled", "mean"),
            ).reset_index().rename(columns={dimension: "segment_value"})
            grouped["dimension"] = dimension
            # A flag rate that exceeds the labelled rate by a wide margin in one
            # segment and not another is the shape of an unfair control.
            grouped["flag_to_label_ratio"] = (
                grouped["flag_rate"] / grouped["labelled_rate"].replace(0, np.nan)
            )
            frames.append(grouped)
        frame = pd.concat(frames, ignore_index=True).round(4)

        spreads = {}
        for dimension, sub in frame.groupby("dimension"):
            spreads[dimension] = round(float(sub["flag_rate"].max() - sub["flag_rate"].min()), 4)
        worst = max(spreads, key=spreads.get) if spreads else ""
        return ProtocolResult(
            "Fairness monitoring", "INFORMATIONAL",
            f"Flag rates compared across TPA, policy type and provider volume band. Largest "
            f"between-segment spread is on {worst} ({spreads.get(worst, 0):.1%} points).",
            detail=frame, values=spreads,
            caveat=(
                "A spread is not by itself unfairness: segments genuinely differ in case mix, and "
                "the volume band in particular is CONSTRUCTED from claim count, so a relationship "
                "between it and flag rate is partly definitional. What would indicate unfairness "
                "is a segment whose FLAG-TO-LABEL ratio is markedly higher than others', and that "
                "column is reported here for exactly that comparison. The appeal half of "
                "'fairness/appeal monitoring' is NOT_MEASURABLE — no appeals exist."
            ),
        )
