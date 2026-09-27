"""The evaluator — runs the catalogue and records what happened.

Responsibilities, each tied to a specific manuscript requirement:

* **Run only runnable controls.** A control that is retired, kill-switched,
  outside its effective dates or classified ``NOT_EXECUTABLE_ON_THIS_DATASET``
  is skipped, and the *reason* it was skipped is recorded as an
  :class:`~fwa.engine.evallib.EvaluationOutcome`. "Did not fire" and "could not
  be evaluated" are different operational facts and the Governance page shows
  them separately.
* **Enforce the alert-volume ceiling** (§3.10 NFR: "a runaway rule cannot flood
  the review queue"). A control that exceeds
  ``cfg.alert_volume_ceiling_per_rule_per_run`` has its signals routed to
  ``MONITOR_ONLY`` — not dropped, because silently discarding a rule's output
  would hide the coverage loss — and a governance event is written to the audit
  log.
* **Measure stage latency** against the Table 3.4 ceilings, so the Governance
  page can report observed against permitted rather than asserting compliance.
* **Stay idempotent.** Signal ids are deterministic (see
  :func:`fwa.engine.signals.Signal.create`), so replaying a run overwrites
  rather than appends. :meth:`Evaluator.run` twice on the same input produces a
  store of the same length — which is exactly what the §6.2 idempotency test
  asserts, for every control uniformly.
"""

from __future__ import annotations

import datetime as _dt
import time
import traceback
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from ..audit.log import AuditEventType
from ..enums import DataSupport, Disposition, RuleStatus, Stage
from .context import ControlContext
from .contract import AtomicControl
from .controls import CONTROL_IMPLEMENTATIONS
from .evallib import EvaluationOutcome
from .registry import RuleRegistry
from .signals import Signal, SignalStore
from .unlocks import UnlockDecision, decide, runtime_view

__all__ = ["Evaluator", "ControlRunRecord", "EvaluationReport"]


@dataclass
class ControlRunRecord:
    rule_id: str
    rule_version: str
    scenario_id: str
    type_label: str
    stage: str
    status: str
    data_support: str
    outcome: str
    signal_count: int
    expected_alert_volume: int | None
    ceiling_breached: bool
    elapsed_ms: float
    error: str = ""
    #: The static classification from the catalogue YAML. ``data_support``
    #: above is the EFFECTIVE classification on this run, which differs only
    #: when a dataset unlock (``rules/unlocks/``) was satisfied.
    catalogue_data_support: str = ""
    #: "catalogue", "unlocked" (was not executable, ran because the dataset
    #: carries the tables) or "upgraded" (a proxy replaced by the full check).
    support_basis: str = "catalogue"
    support_reason: str = ""
    #: What the dataset lacked for this control's unlock, if it has one.
    missing_for_unlock: str = ""

    def to_row(self) -> dict[str, Any]:
        return self.__dict__.copy()


@dataclass
class EvaluationReport:
    records: list[ControlRunRecord] = field(default_factory=list)
    stage_latency_ms: dict[str, float] = field(default_factory=dict)
    #: Rows the run covered, so the latency note states the real batch size
    #: rather than a number carried over from whichever file was loaded first.
    claim_count: int = 0
    ceiling_breaches: list[str] = field(default_factory=list)
    errors: list[tuple[str, str]] = field(default_factory=list)

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame([r.to_row() for r in self.records])

    def latency_report(self, config) -> pd.DataFrame:
        """Observed stage latency against the Table 3.4 ceilings."""
        ceilings = config.get("latency_ceiling_ms")
        scale = f"{self.claim_count:,} claims" if self.claim_count else "this file"
        rows = []
        for stage in Stage:
            observed = self.stage_latency_ms.get(stage.value)
            ceiling = ceilings.get(stage.value)
            rows.append(
                {
                    "stage": stage.value,
                    "ceiling_ms": ceiling,
                    "observed_ms": None if observed is None else round(observed, 1),
                    "within_ceiling": None if observed is None or ceiling is None else observed <= ceiling,
                    "note": f"Observed is the BATCH time for this stage across {scale}, not a "
                            "per-claim online latency. A real PREPAY_SYNC deployment would be "
                            "measured per claim; this artefact runs as a batch and says so.",
                }
            )
        return pd.DataFrame(rows)

    def summary(self) -> dict[str, Any]:
        by_outcome: dict[str, int] = {}
        for r in self.records:
            by_outcome[r.outcome] = by_outcome.get(r.outcome, 0) + 1
        return {
            "controls_considered": len(self.records),
            "controls_that_ran": sum(1 for r in self.records if r.outcome in
                                     (EvaluationOutcome.TRIGGERED.value, EvaluationOutcome.NOT_TRIGGERED.value)),
            "controls_that_fired": sum(1 for r in self.records if r.signal_count > 0),
            "total_signals": sum(r.signal_count for r in self.records),
            "by_outcome": by_outcome,
            "ceiling_breaches": list(self.ceiling_breaches),
            "errors": len(self.errors),
        }


class Evaluator:
    def __init__(
        self,
        registry: RuleRegistry,
        store: SignalStore | None = None,
        audit: Any = None,
    ) -> None:
        self.registry = registry
        self.store = store if store is not None else SignalStore()
        self.audit = audit
        #: rule_id -> the per-run executability decision, for the reports.
        self.decisions: dict[str, UnlockDecision] = {}

    # ------------------------------------------------------------------- run

    def run(
        self,
        ctx: ControlContext,
        *,
        stages: list[Stage] | None = None,
        rule_ids: list[str] | None = None,
    ) -> EvaluationReport:
        report = EvaluationReport(claim_count=int(len(ctx.claims)))
        ceiling = int(ctx.cfg("alert_volume_ceiling_per_rule_per_run"))
        as_of = ctx.run_date

        controls = self.registry.all()
        if rule_ids:
            controls = [c for c in controls if c.rule_id in set(rule_ids)]
        if stages:
            wanted = {s.value for s in stages}
            controls = [c for c in controls if c.stage.value in wanted]

        stage_clock: dict[str, float] = {}

        unlocks = getattr(self.registry, "unlocks", {}) or {}
        for control in controls:
            started = time.perf_counter()
            decision = decide(control, unlocks.get(control.rule_id), ctx.dataset)
            self.decisions[control.rule_id] = decision
            outcome, signals, error = self._run_one(ctx, runtime_view(control, decision), as_of,
                                                    decision)
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            stage_clock[control.stage.value] = stage_clock.get(control.stage.value, 0.0) + elapsed_ms

            breached = False
            if len(signals) > ceiling:
                breached = True
                report.ceiling_breaches.append(control.rule_id)
                for s in signals:
                    s.disposition = Disposition.MONITOR_ONLY
                    s.evidence["alert_volume_ceiling_breached"] = {
                        "signals": len(signals),
                        "ceiling": ceiling,
                        "action": "Routed to MONITOR_ONLY. The signals are NOT discarded — a "
                                  "silently dropped rule hides a coverage loss (§3.10).",
                    }
                if self.audit is not None:
                    self.audit.record(
                        AuditEventType.ALERT_CEILING_BREACHED,
                        actor="system", actor_role="ENGINE", tenant_id=ctx.tenant_id,
                        subject=control.rule_id,
                        reason=f"{len(signals)} signals exceeds the ceiling of {ceiling}; routed to MONITOR_ONLY.",
                        after={"signal_count": len(signals), "ceiling": ceiling},
                    )

            self.store.extend(signals)

            expected = None
            try:
                rec = ctx.config.parameters.record("alert_volume_ceiling_per_rule_per_run")
                expected = rec.expected_alert_volume
            except Exception:  # pragma: no cover
                expected = None

            report.records.append(
                ControlRunRecord(
                    rule_id=control.rule_id,
                    rule_version=control.version,
                    scenario_id=control.scenario_id,
                    type_label=control.type_label,
                    stage=control.stage.value,
                    status=control.status.value,
                    data_support=decision.effective_support.value,
                    outcome=outcome.value,
                    signal_count=len(signals),
                    expected_alert_volume=expected,
                    ceiling_breached=breached,
                    elapsed_ms=round(elapsed_ms, 2),
                    error=error,
                    catalogue_data_support=decision.catalogue_support.value,
                    support_basis=decision.basis,
                    support_reason=decision.reason,
                    missing_for_unlock="; ".join(decision.missing),
                )
            )
            if error:
                report.errors.append((control.rule_id, error))

        report.stage_latency_ms = stage_clock
        if self.audit is not None:
            self.audit.record(
                AuditEventType.PIPELINE_RUN,
                actor="system", actor_role="ENGINE", tenant_id=ctx.tenant_id,
                subject="evaluator",
                reason="Catalogue evaluation completed.",
                after=report.summary(),
            )
        return report

    # -------------------------------------------------------------- internals

    def _run_one(
        self, ctx: ControlContext, control: AtomicControl, as_of: _dt.date,
        decision: UnlockDecision | None = None,
    ) -> tuple[EvaluationOutcome, list[Signal], str]:
        if control.data_support is DataSupport.NOT_EXECUTABLE_ON_THIS_DATASET:
            return EvaluationOutcome.NOT_EXECUTABLE_ON_THIS_DATASET, [], ""
        if control.kill_switched:
            return EvaluationOutcome.KILL_SWITCHED, [], ""
        if control.status is RuleStatus.RETIRED:
            return EvaluationOutcome.NOT_EFFECTIVE, [], ""
        if not control.is_effective(as_of):
            return EvaluationOutcome.NOT_EFFECTIVE, [], ""

        if decision is not None and decision.basis != "catalogue":
            from .unlocked import UNLOCKED_IMPLEMENTATIONS

            impl = UNLOCKED_IMPLEMENTATIONS.get(control.implementation or "")
        else:
            impl = CONTROL_IMPLEMENTATIONS.get(control.implementation or "")
        if impl is None:
            return (
                EvaluationOutcome.NOT_EXECUTABLE_ON_THIS_DATASET,
                [],
                f"{control.rule_id} names implementation {control.implementation!r}, which does not exist.",
            )
        try:
            signals = impl(ctx, control) or []
        except Exception:  # pragma: no cover - surfaced, never swallowed
            return EvaluationOutcome.NULL_INPUT, [], traceback.format_exc(limit=4)
        if not signals:
            return EvaluationOutcome.NOT_TRIGGERED, [], ""
        return EvaluationOutcome.TRIGGERED, signals, ""
