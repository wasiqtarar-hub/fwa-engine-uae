"""The Table 6.1 release-acceptance gates, evaluated against this build (§6.4).

    "Table 6.1 reproduces the specification's five-gate acceptance framework,
     which governs the staged deployment sequence... No claim or gate in
     Table 6.1 is satisfied by anything in this manuscript; the table specifies
     what a FUTURE IMPLEMENTATION TEAM would need to demonstrate before
     advancing a control from shadow to active status."   — manuscript §6.4

This module is that future implementation team's first attempt. Each gate gets
an explicit **PASS / FAIL / NOT_ASSESSABLE** with its evidence, and the honest
expectation is that several are NOT_ASSESSABLE — a gate that needs a prospective
shadow period cannot be satisfied by a single retrospective run, and saying so
is the finding.

``NOT_ASSESSABLE`` is never a pass. A control whose gate cannot be evaluated
stays in shadow, which is why every control in this build is still in shadow at
the end of a run.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd
from .prose import count, verb

__all__ = ["GateResult", "ReleaseGates", "PASS", "FAIL", "NOT_ASSESSABLE"]

PASS = "PASS"
FAIL = "FAIL"
NOT_ASSESSABLE = "NOT_ASSESSABLE"


@dataclass
class GateResult:
    gate: str
    requirement: str
    status: str
    evidence: str
    measured: Any = None
    threshold: Any = None

    def to_row(self) -> dict[str, Any]:
        return {
            "gate": self.gate,
            "minimum_acceptance_evidence": self.requirement,
            "status": self.status,
            "measured": self.measured,
            "threshold": self.threshold,
            "evidence": self.evidence,
        }


class ReleaseGates:
    def __init__(self, result, metrics=None) -> None:
        self.result = result
        self.config = result.config
        self.metrics = metrics

    def evaluate(self) -> list[GateResult]:
        return [
            self._data_readiness(),
            self._hard_edit(),
            self._expert_edit(),
            self._statistical_rule(),
            self._model(),
            self._production(),
        ]

    # ------------------------------------------------------------------ gates

    def _data_readiness(self) -> GateResult:
        threshold = float(self.config.get("gate_key_linkage_min"))
        linkage = self.result.dataset.key_linkage_rate()
        report = self.result.dataset.population_report()
        not_populated = report[report["status"] == "NOT_POPULATED"]
        line_linkage = self.result.dataset.line_linkage_rate()
        if line_linkage is None:
            line_part = ("The LINE half of this gate is vacuous here — there are no claim lines "
                         "to link — so the measured figure covers claim-level linkage only.")
            measured = linkage
        else:
            line_part = (f"Line linkage (claim line → claim) is {line_linkage:.4%}; the measured "
                         f"figure is the lower of the two.")
            measured = min(linkage, line_linkage)
        status = PASS if measured >= threshold else FAIL
        return GateResult(
            "Data readiness",
            "≥99.5% key linkage for claims and lines; scenario-specific completeness reported "
            "explicitly, not hidden by imputation",
            status,
            f"Key linkage (claim → member and provider) is {linkage:.4%} against a required "
            f"{threshold:.1%}. {line_part} Scenario completeness IS reported explicitly: "
            f"{len(not_populated)} of {len(report)} canonical tables are NOT_POPULATED, each with "
            f"a stated reason, and no field is imputed to hide the gap.",
            measured=round(measured, 4), threshold=threshold,
        )

    def _hard_edit(self) -> GateResult:
        threshold = float(self.config.get("gate_hard_edit_reproducibility_min"))
        # What ran on THIS file (the evaluator's effective classification), not
        # the catalogue's static one: an unlocked control is executable here.
        run = self.result.evaluation.to_frame()
        effective = (dict(zip(run["rule_id"], run["data_support"]))
                     if {"rule_id", "data_support"} <= set(run.columns) else {})
        hard = [c for c in self.result.registry.all()
                if c.can_deny and effective.get(c.rule_id, c.data_support.value)
                != "NOT_EXECUTABLE_ON_THIS_DATASET"]
        signed_off = [c for c in hard if c.approved_by]
        return GateResult(
            "Hard edit",
            "Policy-owner sign-off; expected-versus-observed volume comparison; ≥99% decision "
            "reproducibility; documented exception coverage",
            NOT_ASSESSABLE,
            f"{count(len(hard), 'hard/expert control')} that may deny "
            f"{verb('is', len(hard))} executable here, and {len(signed_off)} "
            f"{verb('carries', len(signed_off))} a policy-owner sign-off — because no control "
            f"in this build "
            f"has been activated. Decision REPRODUCIBILITY is demonstrable (signal ids are a pure "
            f"function of rule, version, subject, fact and period, and the idempotency test "
            f"asserts a replay produces no duplicate), but reproducibility against an EXISTING "
            f"ADJUDICATION SYSTEM — which is what this gate means — cannot be measured without "
            f"that system's decisions. Exception coverage IS documented: every control declares "
            f"its exclusions or fails to register.",
            measured=None, threshold=threshold,
        )

    def _expert_edit(self) -> GateResult:
        return GateResult(
            "Expert edit",
            "Clinical/coding validation sample; documented false-positive rate and an appeal route",
            NOT_ASSESSABLE,
            "No clinical or coding validation sample has been reviewed, so no false-positive rate "
            "exists. An appeal route IS implemented — the Case Evidence page captures an appeal "
            "and its result against review_outcome — but it has never been exercised, and an "
            "untested route is not evidence.",
        )

    def _statistical_rule(self) -> GateResult:
        shadow_days = int(self.config.get("shadow_period_min_days"))
        baseline = float(self.config.get("gate_statistical_min_review_yield"))
        statistical = [c for c in self.result.registry.all()
                       if "S" in c.type_label and c.data_support.value != "NOT_EXECUTABLE_ON_THIS_DATASET"]
        peer_sizes = [
            s.evidence.get("peer_n") for s in self.result.signals
            if s.evidence.get("peer_n") is not None
        ]
        min_peer = int(self.config.get("min_peer_group_n"))
        met = all(p >= min_peer for p in peer_sizes) if peer_sizes else None
        return GateResult(
            "Statistical rule",
            "A prospective shadow period; minimum peer sizes met; stable volume; review yield "
            "above an agreed baseline",
            NOT_ASSESSABLE,
            f"{count(len(statistical), 'statistical control')} "
            f"{verb('executes', len(statistical))} here. MINIMUM PEER SIZES: "
            + (f"every peer group used met the cfg.min_peer_group_n floor of {min_peer}."
               if met else
               f"at least one peer comparison fell back below {min_peer} and is reported as "
               f"INSUFFICIENT_PEER_EVIDENCE rather than flagged." if met is False else
               "no peer-sized signals were produced.")
            + f" PROSPECTIVE SHADOW PERIOD: not satisfiable by a single retrospective run — the "
            f"gate requires {shadow_days} days of forward observation. REVIEW YIELD: "
            f"NOT_MEASURABLE (no reviewer dispositions), so the {baseline:.0%} baseline cannot be "
            f"tested.",
            threshold=f"review yield ≥ {baseline:.0%}, shadow ≥ {shadow_days} days",
        )

    def _model(self) -> GateResult:
        models = self.result.models
        if models is None or not models.gate_verdicts:
            return GateResult(
                "Model",
                "Temporal holdout evaluation; calibration check; demonstrated prospective lift "
                "over the simple statistical baseline; drift and explanation-quality tests",
                NOT_ASSESSABLE,
                "The model promotion gate was not run in this pass.",
            )
        promoted = [n for n, v in models.gate_verdicts.items() if v.promoted]
        lines = []
        for name, verdict in models.gate_verdicts.items():
            failed = [c.name for c in verdict.criteria if c.status == "FAIL"]
            unassessable = [c.name for c in verdict.criteria if c.status == NOT_ASSESSABLE]
            lines.append(
                f"{name}: {'PROMOTED' if verdict.promoted else 'REMAINS IN SHADOW'}"
                + (f"; failed {', '.join(failed)}" if failed else "")
                + (f"; not assessable {', '.join(unassessable)}" if unassessable else "")
            )
        return GateResult(
            "Model",
            "Temporal holdout evaluation; calibration check; demonstrated prospective lift over "
            "the simple statistical baseline; drift and explanation-quality tests",
            PASS if promoted else FAIL,
            " | ".join(lines)
            + ("" if promoted else
               " The transparent composite ANL-01-R01 continues to run and is the system's "
               "explainable fallback, so no detection capability is lost by leaving the models "
               "in shadow."),
            measured=f"{len(promoted)} promoted of {len(models.gate_verdicts)}",
        )

    def _production(self) -> GateResult:
        registry = self.result.registry
        audit = self.result.audit
        ok, message = audit.verify_chain()
        ceiling = int(self.config.get("alert_volume_ceiling_per_rule_per_run"))
        breaches = self.result.evaluation.ceiling_breaches
        owners = {c.owner for c in registry.all()}
        capabilities = [
            ("Rollback capability", "RuleRegistry.rollback() restores a prior registered version "
                                    "and re-enters it in SHADOW, never straight to active."),
            ("Kill switch", "RuleRegistry.kill_switch() routes a control's signals to "
                            "MONITOR_ONLY without a code deployment, with a mandatory reason."),
            ("Alert-volume ceiling", f"cfg.alert_volume_ceiling_per_rule_per_run = {ceiling}; "
                                     f"{len(breaches)} breach(es) this run "
                                     f"({', '.join(breaches) or 'none'}), each routed to "
                                     f"MONITOR_ONLY and audited."),
            ("Audit logs", f"Append-only and hash-chained. {message}"),
            ("Named responsible owner", f"{count(len(owners), 'distinct owner')} across the catalogue; "
                                        f"a control with no owner cannot be registered."),
        ]
        # Four of the five elements are software capabilities and are evidenced
        # above. The fifth — a named responsible owner ON CALL — is a rota, not
        # a capability, and no artefact can evidence it. A FAIL would be wrong
        # (the software elements are present and demonstrable) but so would a
        # PASS: it would report this build as meeting a production gate it
        # cannot meet, which is precisely the overstatement §6.5 and §11 exist
        # to prevent. NOT_ASSESSABLE is the honest third answer, and it never
        # counts as a pass.
        status = FAIL if not ok else NOT_ASSESSABLE
        return GateResult(
            "Production",
            "Rollback capability, kill switch, an alert-volume ceiling, audit logs and a named "
            "responsible owner on call",
            status,
            " ".join(f"{name}: {detail}" for name, detail in capabilities)
            + " ON CALL is the one element that cannot be evidenced by software: it is a rota, "
              "not a capability, and this artefact has no one on call. The four "
              "software elements are demonstrable and demonstrated; the gate as a whole is "
              "therefore reported NOT_ASSESSABLE rather than passed.",
        )

    # ---------------------------------------------------------------- reports

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame([g.to_row() for g in self.evaluate()])
