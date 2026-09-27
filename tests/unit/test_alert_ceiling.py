"""The alert-volume ceiling (§3.10 NFR: "a runaway rule cannot flood the review queue").

The requirement has two halves, and the second is the one that is easy to get
wrong. A control that produces more signals than
``cfg.alert_volume_ceiling_per_rule_per_run`` must stop reaching the review
queue as an actionable disposition — *and its signals must survive*. Dropping
them would be the simpler implementation and the wrong one: a rule that silently
disappears takes its coverage with it, and nobody downstream can tell the
difference between "this control was throttled" and "this control found
nothing". So the ceiling routes to ``MONITOR_ONLY`` rather than discarding, says
so in the evidence, and writes a governance event that names the rule.

These tests drive the evaluator with a control whose signal count is chosen
relative to the ceiling, rather than relying on whichever real control happens
to be noisy on whichever dataset is loaded — the property under test is the
evaluator's, not the catalogue's.
"""

from __future__ import annotations

import datetime as _dt

import pandas as pd
import pytest

from fwa.audit.log import AuditEventType, AuditLog
from fwa.config import load_config
from fwa.engine.context import ControlContext
from fwa.engine.contract import AtomicControl
from fwa.engine.controls import CONTROL_IMPLEMENTATIONS
from fwa.engine.evaluator import Evaluator
from fwa.engine.registry import RuleRegistry
from fwa.engine.signals import Signal, SignalStore
from fwa.enums import Disposition
from fwa.features.frame import FeatureFrame

IMPL_NAME = "__test_alert_ceiling_noisy__"
TENANT = "T001"


def _control(**overrides) -> AtomicControl:
    payload = dict(
        rule_id="PAY-06-R09", scenario_id="PAY-06", name="Alert-ceiling fixture",
        version="1.0.0", type="S", stage="POSTPAY_DAILY", population="claim",
        inputs=["claim_header.gross_amount_aed"], expression="noisy",
        exclusions=["corrected_resubmission"],
        grouping_key=["tenant_id", "claim_sk"], score=50.0,
        disposition="POSTPAY_AUDIT", reason_code="FIXTURE",
        evidence_fields=["claim_sk"], owner="analytics",
        effective_from="2020-01-01", data_support="EXECUTABLE",
        data_support_reason="Fixture control used only by the alert-ceiling tests.",
        implementation=IMPL_NAME,
    )
    payload.update(overrides)
    return AtomicControl(**payload)


@pytest.fixture()
def harness(monkeypatch):
    """An evaluator wired to a control that emits exactly ``n`` signals.

    ``n`` is set per test through the returned ``run`` callable. The fake
    implementation is registered on the real ``CONTROL_IMPLEMENTATIONS`` map and
    removed again by monkeypatch, so no test leaks a rule into another's
    catalogue.
    """
    config = load_config()
    ceiling = int(config.get("alert_volume_ceiling_per_rule_per_run"))

    emitted: dict[str, int] = {"n": 0}

    def _impl(ctx, control):
        return [
            Signal.create(
                control=control, tenant_id=ctx.tenant_id,
                subject_type="claim", subject_id=f"C{i:06d}",
                evidence={"i": i}, fact_key=f"fixture:{i}",
                claim_ids=[f"C{i:06d}"], period_bucket="2026-01",
            )
            for i in range(emitted["n"])
        ]

    monkeypatch.setitem(CONTROL_IMPLEMENTATIONS, IMPL_NAME, _impl)

    def run(n: int):
        emitted["n"] = n
        registry = RuleRegistry([_control()])
        audit = AuditLog()
        store = SignalStore()
        ctx = ControlContext(
            config=config, tenant_id=TENANT,
            claims=pd.DataFrame({"claim_sk": ["C000000"], "tenant_id": [TENANT]}),
            features=FeatureFrame(pd.DataFrame({"claim_sk": ["C000000"]})),
            run_date=_dt.date(2026, 6, 1), audit=audit,
        )
        report = Evaluator(registry, store=store, audit=audit).run(ctx)
        return report, store, audit

    run.ceiling = ceiling
    return run


def test_a_control_under_the_ceiling_keeps_its_disposition(harness):
    """The ceiling must not touch a control that is behaving."""
    report, store, _ = harness(harness.ceiling - 1)
    record = report.records[0]
    assert record.ceiling_breached is False
    assert report.ceiling_breaches == []
    assert all(s.disposition is Disposition.POSTPAY_AUDIT for s in store.all())
    assert all("alert_volume_ceiling_breached" not in s.evidence for s in store.all())


def test_exactly_at_the_ceiling_is_not_a_breach(harness):
    """The ceiling is a maximum permitted volume, not a forbidden value.

    Asserted explicitly because an off-by-one here would throttle every control
    that sits exactly on a threshold a policy owner chose deliberately.
    """
    report, store, _ = harness(harness.ceiling)
    assert report.records[0].ceiling_breached is False
    assert all(s.disposition is Disposition.POSTPAY_AUDIT for s in store.all())


def test_a_breaching_control_is_routed_to_monitor_only(harness):
    report, store, _ = harness(harness.ceiling + 5)
    record = report.records[0]
    assert record.ceiling_breached is True
    assert record.rule_id in report.ceiling_breaches
    assert all(s.disposition is Disposition.MONITOR_ONLY for s in store.all())


def test_the_breaching_signals_are_kept_not_discarded(harness):
    """The half of the requirement that a naive implementation gets wrong.

    Throttling by deletion would leave the queue clean and the coverage gap
    invisible. The signals must still be in the store, and each must carry the
    reason its disposition was changed.
    """
    n = harness.ceiling + 5
    report, store, _ = harness(n)
    assert len(store.all()) == n, "signals were dropped; the coverage loss is now invisible"
    for signal in store.all():
        breach = signal.evidence["alert_volume_ceiling_breached"]
        assert breach["signals"] == n
        assert breach["ceiling"] == harness.ceiling
        assert "NOT discarded" in breach["action"]


def test_a_breach_writes_an_attributable_governance_event(harness):
    """§9.7: a control cannot be throttled without the throttling being auditable."""
    report, _, audit = harness(harness.ceiling + 5)
    events = audit.by_type(AuditEventType.ALERT_CEILING_BREACHED)
    assert len(events) == 1
    assert events[0].subject == "PAY-06-R09"
    assert str(harness.ceiling) in events[0].reason
    ok, detail = audit.verify_chain()
    assert ok is True, f"the ceiling event broke the hash chain: {detail}"
