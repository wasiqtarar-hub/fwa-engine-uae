"""Tenant scoping (build brief §13.5, manuscript §3.10 NFR).

    "Tenant scoping: every query is tenant-filtered; add a test that proves a
     user in tenant A cannot see tenant B's data."

Multi-tenancy is the kind of property that is easy to assert loosely ("the
query has a WHERE clause") and hard to assert usefully. These tests take the
other route: build data in two tenants, then check that every read path the
application actually uses — the signal store, the case correlator, the audit
log, the review queue and the session — returns nothing from the other tenant.
"""

from __future__ import annotations

import pytest

from fwa.audit.log import AuditEventType, AuditLog
from fwa.auth.service import AuthService
from fwa.cases.correlate import CaseCorrelator
from fwa.cases.priority import PriorityScorer
from fwa.engine.controls import CONTROL_IMPLEMENTATIONS
from fwa.engine.signals import SignalStore
from fwa.storage.db import Database

pytestmark = pytest.mark.governance

TENANT_A = "T001"
TENANT_B = "T002"


def _with(context, **overrides):
    import copy

    ctx = copy.copy(context)
    for key, value in overrides.items():
        setattr(ctx, key, value)
    return ctx


@pytest.fixture(scope="module")
def two_tenant_signals(context, registry):
    """The same control over the same claims, run once per tenant."""
    control = registry.get("ENT-03-R02")
    impl = CONTROL_IMPLEMENTATIONS[control.implementation]

    claims_a = context.claims.copy()
    claims_a["tenant_id"] = TENANT_A
    claims_b = context.claims.copy()
    claims_b["tenant_id"] = TENANT_B

    a = impl(_with(context, claims=claims_a, tenant_id=TENANT_A), control)
    b = impl(_with(context, claims=claims_b, tenant_id=TENANT_B), control)
    assert a and b, "Fixture produced no signals in one of the tenants."
    return a, b


def test_a_signal_carries_the_tenant_that_produced_it(two_tenant_signals):
    a, b = two_tenant_signals
    assert {s.tenant_id for s in a} == {TENANT_A}
    assert {s.tenant_id for s in b} == {TENANT_B}


def test_the_same_fact_in_two_tenants_is_two_distinct_signals(two_tenant_signals):
    """Tenant is part of the signal's identity, not a column bolted on after.

    If it were not, a replay in tenant B would overwrite tenant A's signal in
    the store — silent cross-tenant data loss rather than a visible leak, which
    is the harder failure to notice.
    """
    a, b = two_tenant_signals
    by_fact_a = {s.fact_key: s.signal_id for s in a}
    by_fact_b = {s.fact_key: s.signal_id for s in b}
    shared_facts = set(by_fact_a) & set(by_fact_b)
    assert shared_facts, "Fixture did not produce the same fact in both tenants."
    for fact in shared_facts:
        assert by_fact_a[fact] != by_fact_b[fact]


def test_signal_store_reads_are_tenant_filtered(two_tenant_signals):
    a, b = two_tenant_signals
    store = SignalStore()
    store.extend(a)
    store.extend(b)

    assert len(store) == len(a) + len(b)
    assert {s.tenant_id for s in store.all(TENANT_A)} == {TENANT_A}
    assert {s.tenant_id for s in store.all(TENANT_B)} == {TENANT_B}
    assert len(store.all(TENANT_A)) == len(a)


def test_every_signal_store_read_path_is_tenant_scoped(two_tenant_signals):
    """Not just ``all()`` — each accessor the app calls."""
    a, b = two_tenant_signals
    store = SignalStore()
    store.extend(a)
    store.extend(b)

    probe = b[0]
    assert store.by_rule(probe.rule_id, tenant_id=TENANT_A), "fixture sanity"
    assert all(s.tenant_id == TENANT_A for s in store.by_rule(probe.rule_id, tenant_id=TENANT_A))
    assert not store.by_subject(probe.subject_type, probe.subject_id, tenant_id=TENANT_A) or all(
        s.tenant_id == TENANT_A
        for s in store.by_subject(probe.subject_type, probe.subject_id, tenant_id=TENANT_A)
    )
    for claim_id in probe.claim_ids[:1]:
        assert all(
            s.tenant_id == TENANT_A for s in store.by_claim(claim_id, tenant_id=TENANT_A)
        )
    frame = store.to_dataframe(tenant_id=TENANT_A)
    if not frame.empty:
        assert set(frame["tenant_id"].unique()) == {TENANT_A}


def test_cases_and_the_queue_are_tenant_scoped(config, two_tenant_signals, registry):
    a, b = two_tenant_signals
    correlator = CaseCorrelator(config, PriorityScorer(config))
    correlator.correlate(list(a) + list(b), registry)

    frame_a = correlator.to_frame(tenant_id=TENANT_A)
    assert not frame_a.empty
    assert set(frame_a["tenant_id"].unique()) == {TENANT_A}

    queue_a = correlator.queue(tenant_id=TENANT_A)
    assert not queue_a.empty, "Tenant A's queue is empty; the fixture did not correlate."
    assert set(queue_a["tenant_id"].unique()) == {TENANT_A}

    ids_a = set(queue_a["case_id"])
    ids_b = set(correlator.queue(tenant_id=TENANT_B)["case_id"])
    assert not (ids_a & ids_b), "A case appears in both tenants' queues."


def test_case_fingerprints_differ_across_tenants():
    from fwa.cases.correlate import case_fingerprint

    args = ("provider", "P0001", "PAY", "2025-11")
    assert case_fingerprint(TENANT_A, *args) != case_fingerprint(TENANT_B, *args)


def test_audit_log_reads_are_tenant_filtered():
    log = AuditLog()
    log.record(AuditEventType.DISPOSITION_RECORDED, actor="a", tenant_id=TENANT_A, subject="S1",
               reason="tenant A event")
    log.record(AuditEventType.DISPOSITION_RECORDED, actor="b", tenant_id=TENANT_B, subject="S2",
               reason="tenant B event")

    assert {e.tenant_id for e in log.all(tenant_id=TENANT_A)} == {TENANT_A}
    assert [e.subject for e in log.all(tenant_id=TENANT_A)] == ["S1"]
    frame = log.to_dataframe(tenant_id=TENANT_B)
    assert set(frame["tenant_id"].unique()) == {TENANT_B}


def test_a_session_is_bound_to_one_tenant():
    """The seeded second-tenant reviewer must land in T002, not T001."""
    service = AuthService(Database("sqlite://"), audit=AuditLog())
    service.seed()

    a = service.login("reviewer", "reviewer")
    b = service.login("reviewer2", "reviewer2")

    assert a.tenant_id == TENANT_A
    assert b.tenant_id == TENANT_B
    assert a.role == b.role, "The two accounts should differ only by tenant."


def test_a_tenant_filtered_read_with_no_tenant_returns_everything_by_design(two_tenant_signals):
    """``tenant_id=None`` is the operator/admin view, and it is explicit.

    Recording this as a test rather than leaving it implicit matters: a default
    that silently means "all tenants" is the usual way a scoping bug ships, so
    the contract is that the caller must *ask* for the unscoped view.
    """
    a, b = two_tenant_signals
    store = SignalStore()
    store.extend(a)
    store.extend(b)
    assert len(store.all(tenant_id=None)) == len(a) + len(b)
