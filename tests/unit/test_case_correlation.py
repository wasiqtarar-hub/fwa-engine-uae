"""Case correlation, fingerprints, merge and split (manuscript §3.8, §4.12).

    "Signals correlate into cases along six dimensions... using a deterministic
     ``case_fingerprint = hash(tenant, case_type, primary_subject,
     scenario_family, period_bucket)``, with MERGE AND SPLIT operations always
     producing a RECORDED AUDIT EVENT rather than a silent case-identity change."

The worked example §4.12 gives is the acceptance test, and it has its own test
below: a PAY-08 resubmission signal, a DOC-02 cloned-note signal and a CLN-01
upcoding signal must converge into one coding case rather than three alerts
that hide the fact they describe one event.
"""

from __future__ import annotations

import datetime as _dt

import pytest

from fwa.audit.log import AuditEventType, AuditLog
from fwa.cases.correlate import (
    CORRELATION_DIMENSIONS, CaseCorrelator, case_family, case_fingerprint,
)
from fwa.cases.priority import PriorityScorer
from fwa.engine.signals import Signal
from fwa.enums import Disposition, SignalDomain


def _signal(rule_id, *, subject_type="provider", subject_id="P0001", fact=None,
            disposition=Disposition.POSTPAY_AUDIT, domain=SignalDomain.RULE,
            period="2026-03", claim_ids=("C1",), exposure=0.0, established=True,
            signal_id=None) -> Signal:
    scenario_id = "-".join(rule_id.split("-")[:2])
    return Signal(
        signal_id=signal_id or f"sig-{rule_id}-{subject_id}",
        rule_id=rule_id, rule_version="1.0.0", scenario_id=scenario_id,
        tenant_id="T001", subject_type=subject_type, subject_id=subject_id,
        claim_ids=list(claim_ids),
        event_time=_dt.datetime(2026, 3, 15), detection_time=_dt.datetime(2026, 3, 16),
        period_bucket=period, score=60.0, confidence=1.0, evidence_strength=0.5,
        disposition=disposition, reason_code="R", domain=domain,
        stage="PREPAY_SYNC", rule_status="shadow",
        exposure_aed=exposure, exposure_basis="fixture", exposure_established=established,
        evidence={}, fact_key=fact or f"fact-{rule_id}",
    )


@pytest.fixture()
def correlator(config):
    return CaseCorrelator(config, PriorityScorer(config), audit=AuditLog())


# ------------------------------------------------------------ the fingerprint


def test_the_fingerprint_is_a_pure_function_of_its_five_fields():
    args = ("T001", "provider_behaviour", "provider:P0001", "coding_integrity", "2026-03")
    assert case_fingerprint(*args) == case_fingerprint(*args)


@pytest.mark.parametrize("position", range(5))
def test_changing_any_one_field_changes_the_fingerprint(position):
    args = ["T001", "provider_behaviour", "provider:P0001", "coding_integrity", "2026-03"]
    other = list(args)
    other[position] = other[position] + "-X"
    assert case_fingerprint(*args) != case_fingerprint(*other)


def test_the_fingerprint_does_not_depend_on_the_signals():
    """A case identity derived from its contents would change every time a signal arrived.

    The reviewer's disposition recorded yesterday has to still attach to the
    same case today, so identity has to come from the five stable fields.
    """
    import inspect

    source = inspect.getsource(case_fingerprint)
    for forbidden in ("signal", "evidence", "score"):
        assert forbidden not in source.lower()


# ---------------------------------------------------------------- dimensions


def test_there_are_exactly_six_dimensions():
    assert len(CORRELATION_DIMENSIONS) == 6
    assert set(CORRELATION_DIMENSIONS) == {
        "claim_lineage", "episode", "provider_behaviour", "pharmacy_product",
        "network", "distribution",
    }
    for spec in CORRELATION_DIMENSIONS.values():
        assert spec["label"] and spec["subject"] and spec["description"]


def test_every_scenario_maps_to_a_case_family(registry):
    """No scenario falls through to an unnamed family."""
    for scenario in registry.scenarios():
        assert case_family(scenario)


def test_related_scenarios_share_a_family():
    assert case_family("CLN-01") == case_family("DOC-02") == case_family("PAY-08")
    assert case_family("PAY-01") == case_family("PAY-02")
    assert case_family("CLN-01") != case_family("PAY-01")


# ------------------------------------------------------ the §4.12 worked example


def test_the_manuscripts_worked_example_converges_into_one_case(correlator):
    """PAY-08 + DOC-02 + CLN-01 on one provider, one period → one coding case."""
    signals = [
        _signal("PAY-08-R01", fact="resubmission:C1", signal_id="a"),
        _signal("DOC-02-R01", fact="cloned_note:C1", domain=SignalDomain.DOCUMENT,
                signal_id="b"),
        _signal("CLN-01-R01", fact="coding:C1", domain=SignalDomain.STATISTICAL,
                signal_id="c"),
    ]
    cases = correlator.correlate(signals)

    coding = [c for c in cases.values() if c.scenario_family == "coding_integrity"]
    assert len(coding) == 1, (
        "Three signals describing one coding event produced "
        f"{len(coding)} cases instead of one."
    )
    assert len(coding[0].signal_ids) == 3
    assert len(coding[0].domains) == 3


def test_unrelated_families_do_not_converge(correlator):
    signals = [
        _signal("CLN-01-R01", signal_id="a"),
        _signal("PHR-01-R01", signal_id="b"),
    ]
    cases = correlator.correlate(signals)
    assert len({c.scenario_family for c in cases.values()}) == 2


def test_different_periods_do_not_converge(correlator):
    signals = [
        _signal("CLN-01-R01", period="2026-03", signal_id="a"),
        _signal("CLN-02-R01", period="2026-04", signal_id="b"),
    ]
    assert len(correlator.correlate(signals)) == 2


def test_different_subjects_do_not_converge(correlator):
    signals = [
        _signal("CLN-01-R01", subject_id="P0001", signal_id="a"),
        _signal("CLN-02-R01", subject_id="P0002", signal_id="b"),
    ]
    assert len(correlator.correlate(signals)) == 2


# -------------------------------------------------------------- disposition


def test_a_case_takes_the_strongest_disposition_among_its_signals(correlator):
    signals = [
        _signal("CLN-01-R01", disposition=Disposition.MONITOR_ONLY, signal_id="a"),
        _signal("CLN-02-R01", disposition=Disposition.REJECT, signal_id="b"),
        _signal("CLN-03-R01", disposition=Disposition.POSTPAY_AUDIT, signal_id="c"),
    ]
    case = next(iter(correlator.correlate(signals).values()))
    assert case.disposition is Disposition.REJECT


def test_a_model_signal_cannot_strengthen_a_cases_disposition(correlator):
    """MONITOR_ONLY sits at the bottom of the rank, so it never lifts anything."""
    without = correlator.correlate([
        _signal("CLN-01-R01", disposition=Disposition.POSTPAY_AUDIT, signal_id="a"),
    ])
    baseline = next(iter(without.values())).disposition

    fresh = CaseCorrelator(correlator.config, PriorityScorer(correlator.config))
    with_model = fresh.correlate([
        _signal("CLN-01-R01", disposition=Disposition.POSTPAY_AUDIT, signal_id="a"),
        _signal("CLN-02-R01", disposition=Disposition.MONITOR_ONLY,
                domain=SignalDomain.MODEL, signal_id="b"),
    ])
    assert next(iter(with_model.values())).disposition is baseline


# ----------------------------------------------------------------- exposure


def test_a_claim_touching_two_signals_is_counted_once(correlator):
    """Two signals on one claim must not produce twice the exposure."""
    signals = [
        _signal("CLN-01-R01", claim_ids=["C1"], exposure=500.0, signal_id="a"),
        _signal("CLN-02-R01", claim_ids=["C1"], exposure=500.0, signal_id="b"),
    ]
    case = next(iter(correlator.correlate(signals).values()))
    assert case.exposure_aed == pytest.approx(500.0)


def test_an_unestablished_exposure_does_not_become_established_by_correlation(correlator):
    signals = [
        _signal("ANL-01-R01", exposure=12_000.0, established=False,
                domain=SignalDomain.MODEL, disposition=Disposition.MONITOR_ONLY,
                signal_id="a"),
    ]
    case = next(iter(correlator.correlate(signals).values()))
    assert case.exposure_established is False
    assert "not established exposure" in case.exposure_basis


# --------------------------------------------------------------- merge/split


def test_a_merge_records_an_audit_event(correlator):
    correlator.correlate([
        _signal("CLN-01-R01", subject_id="P0001", signal_id="a"),
        _signal("PHR-01-R01", subject_id="P0001", signal_id="b"),
    ])
    ids = list(correlator.cases)
    merged = correlator.merge(ids, actor="rashid", reason="Same underlying billing event.")

    assert merged.merged_from == ids
    assert len(merged.signal_ids) == 2
    assert all(old not in correlator.cases for old in ids)

    events = correlator.audit.by_type(AuditEventType.CASE_MERGED)
    assert len(events) == 1
    assert "Same underlying billing event." in events[0].reason


def test_a_merge_requires_two_cases_and_a_reason(correlator):
    correlator.correlate([_signal("CLN-01-R01", signal_id="a")])
    ids = list(correlator.cases)
    with pytest.raises(ValueError):
        correlator.merge(ids, actor="rashid", reason="One case is not a merge.")
    with pytest.raises(ValueError):
        correlator.merge(ids * 2, actor="rashid", reason="   ")


def test_a_split_records_an_audit_event_and_replaces_the_original(correlator):
    correlator.correlate([
        _signal("CLN-01-R01", fact="f1", signal_id="a"),
        _signal("CLN-02-R01", fact="f2", signal_id="b"),
    ])
    case_id = next(iter(correlator.cases))
    children = correlator.split(case_id, [["a"], ["b"]],
                                actor="rashid", reason="Two unrelated events.")

    assert len(children) == 2
    assert case_id not in correlator.cases
    assert all(c.case_id in correlator.cases for c in children)

    events = correlator.audit.by_type(AuditEventType.CASE_SPLIT)
    assert len(events) == 1
    assert events[0].subject == case_id


def test_a_split_requires_a_reason(correlator):
    correlator.correlate([_signal("CLN-01-R01", signal_id="a")])
    case_id = next(iter(correlator.cases))
    with pytest.raises(ValueError):
        correlator.split(case_id, [["a"]], actor="rashid", reason="")


def test_merge_and_split_never_change_identity_silently(correlator):
    """The §4.12 requirement: both operations emit an event, every time."""
    correlator.correlate([
        _signal("CLN-01-R01", subject_id="P0001", signal_id="a"),
        _signal("PHR-01-R01", subject_id="P0001", signal_id="b"),
    ])
    before = len(correlator.audit)
    merged = correlator.merge(list(correlator.cases), actor="r", reason="One event.")
    correlator.split(merged.case_id, [["a"], ["b"]], actor="r", reason="Two after all.")
    assert len(correlator.audit) == before + 2
    assert correlator.audit.verify_chain()


# --------------------------------------------------------------- determinism


def test_the_same_signals_produce_the_same_case_ids(config):
    signals = [_signal("CLN-01-R01", signal_id="a"), _signal("CLN-02-R01", signal_id="b")]
    first = CaseCorrelator(config, PriorityScorer(config)).correlate(signals)
    second = CaseCorrelator(config, PriorityScorer(config)).correlate(signals)
    assert set(first) == set(second)


def test_a_real_run_produces_cases_that_carry_their_provenance(sample_cases):
    _, cases = sample_cases
    assert cases
    for case in list(cases.values())[:50]:
        assert case.case_id
        assert case.case_type in CORRELATION_DIMENSIONS
        assert case.signal_ids
        assert case.scenario_family
        assert case.period_bucket
        row = case.to_row()
        assert row["case_id"] == case.case_id
        assert "priority" in row
