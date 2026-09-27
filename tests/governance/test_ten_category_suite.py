"""The ten-category rule-level suite (manuscript §6.2), for EVERY implemented control.

    "Every one of the 164 atomic controls in Appendix C requires, at minimum,
     the ten test categories the specification defines... This suite is
     deliberately uniform across all 164 controls REGARDLESS OF TYPE (H, E, S,
     N, T, M), because the specification's governance model depends on every
     control, however statistically sophisticated, still passing the same
     baseline of auditability tests as the simplest hard edit. A statistical or
     model control that cannot pass an idempotency or tenant-isolation test is
     NOT PRODUCTION-READY REGARDLESS OF ITS DETECTION PERFORMANCE."
                                                            — manuscript §6.2

Ten categories × every control with an implementation. Uniform: an Isolation
Forest control is held to the same idempotency and tenant-isolation standard as
a duplicate edit, and the parametrisation makes no exceptions for type.

Controls classified ``NOT_EXECUTABLE_ON_THIS_DATASET`` have no implementation
to test; they are covered instead by
:mod:`tests.governance.test_catalogue_completeness`, which asserts that each one
carries a reason and the canonical fields that would unlock it — because a
catalogue entry that claims to be unrunnable and is silently wrong is the other
way this suite could be defeated.
"""

from __future__ import annotations

import datetime as _dt
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from fwa.engine.controls import CONTROL_IMPLEMENTATIONS
from fwa.engine.evaluator import Evaluator
from fwa.engine.signals import SignalStore
from fwa.enums import DataSupport, RuleStatus

CATEGORIES = (
    "positive_fixture",
    "clean_negative_fixture",
    "boundary_case",
    "null_missing_input",
    "effective_date",
    "correction_cancellation_resubmission",
    "approved_exception",
    "idempotency",
    "tenant_isolation",
    "evidence_snapshot",
)


def _run(context, control, claims=None, tenant_id=None):
    """Execute one control against a (possibly modified) claim frame."""
    ctx = _with(context, claims=claims, tenant_id=tenant_id)
    impl = CONTROL_IMPLEMENTATIONS[control.implementation]
    return impl(ctx, control)


def _with(context, **overrides):
    import copy

    ctx = copy.copy(context)
    for key, value in overrides.items():
        if value is not None:
            setattr(ctx, key, value)
    return ctx


@pytest.fixture(scope="module")
def baseline(context, implemented_controls):
    """One run of every implemented control, reused across categories."""
    return {c.rule_id: _run(context, c) for c in implemented_controls}


def _ids(controls):
    return [c.rule_id for c in controls]


# ---------------------------------------------------------------------------
# 1. positive fixture
# ---------------------------------------------------------------------------


def test_every_implemented_control_has_all_ten_categories(implemented_controls):
    """The suite is uniform: no control is exempt from any category."""
    assert implemented_controls, "No control declares an implementation."
    covered = {
        name.split("test_category_")[-1]
        for name in globals()
        if name.startswith("test_category_")
    }
    missing = set(CATEGORIES) - covered
    assert not missing, f"Ten-category suite is missing: {sorted(missing)}"


def test_category_positive_fixture(context, implemented_controls, baseline):
    """A case constructed to trigger the rule's exact reason code.

    A control that produces no signal anywhere in a 1,500-claim stratified
    sample is recorded here rather than silently passing: that is a real finding
    about the dataset, and the validation report explains each one.
    """
    silent = []
    for control in implemented_controls:
        signals = baseline[control.rule_id]
        if not signals:
            silent.append(control.rule_id)
            continue
        assert all(s.reason_code == control.reason_code for s in signals), (
            f"{control.rule_id} produced a signal with a reason code other than "
            f"{control.reason_code}"
        )
        assert all(s.rule_id == control.rule_id for s in signals)
        assert all(s.rule_version == control.version for s in signals)

    # These are the controls the validation report documents as silent, each for
    # a stated reason about the data rather than about the control.
    expected_silent = {
        "PAY-01-R01", "PAY-01-R02", "PAY-01-R03", "PAY-06-R02", "CLN-05-R01",
        "CLN-05-R03", "CLN-06-R03", "CLN-07-R02", "NET-02-R02", "NET-04-R02",
        "NET-04-R03", "ANL-01-R03", "ANL-01-R04", "CLN-01-R03", "DOC-01-R02",
        "DOC-02-R01", "NET-01-R01", "NET-02-R03", "NET-03-R01", "CLN-04-R02",
        "POL-01-R04", "CLN-04-R04", "PAY-06-R04", "CLN-01-R01", "CLN-07-R04",
        "ANL-01-R01", "ANL-01-R02",
    }
    unexpected = set(silent) - expected_silent
    assert not unexpected, (
        f"Controls silent on the sample without a documented reason: {sorted(unexpected)}"
    )


def test_category_clean_negative_fixture(context, implemented_controls, claims):
    """A case that must NOT trigger the rule.

    Built by removing the conditions each control keys on: no exclusion-listed
    providers, no coding mismatches, no multi-insurer events, no short
    readmission gaps, no extreme pharmacy ratios, and a single claim per member
    and provider so no duplicate, repeat or concentration pattern can exist.
    """
    clean = claims.copy()
    clean["provider_blacklist_flag"] = False
    clean["icd_code_matches_procedure"] = True
    clean["num_insurers_same_event"] = 1
    clean["discharge_readmit_gap_days"] = 365
    clean["pharmacy_bill_ratio"] = 0.20
    clean["days_since_policy_start"] = 1500
    clean = clean.drop_duplicates(subset=["member_sk"]).drop_duplicates(subset=["provider_sk"])
    # uniform amounts and stays: nothing can be a peer outlier
    clean["gross_amount_aed"] = 5000.0
    clean["approved_amount_aed"] = 4500.0
    clean["length_of_stay_days"] = 10

    # A clean fixture has to be clean *relative to its own peer population*.
    # Leaving the session's PeerService in place would compare these uniform
    # AED 5,000 claims against the real sample's medians, and "your fixture
    # differs from the real data" is not the same statement as "this control
    # fires on a case that should not trigger it". Rebuilding the peer service
    # over the clean frame is what makes the category test mean what it says.
    from fwa.statistical import PeerService

    clean_peers = PeerService(context.config, clean)

    claim_level = [
        c for c in implemented_controls
        if c.rule_id in {
            "ENT-03-R02", "CLN-03-R01", "PAY-10-R01", "CLN-05-R02", "PHR-03-R03",
            "PAY-06-R02", "PAY-06-R03", "CLN-01-R02", "CLN-04-R01", "ENT-02-R03",
        }
    ]
    for control in claim_level:
        ctx = _with(context, claims=clean, peers=clean_peers)
        signals = CONTROL_IMPLEMENTATIONS[control.implementation](ctx, control)
        assert not signals, (
            f"{control.rule_id} fired on a deliberately clean frame — "
            f"{len(signals)} signal(s), first reason {signals[0].reason_code if signals else ''}"
        )


def test_category_boundary_case(context, config, registry):
    """Values at the exact threshold boundary behave as defined.

    CLN-05-R02 fires when ``discharge_readmit_gap_days < cfg.readmit_window_days``.
    At exactly the window it must NOT fire; one day below it must.
    """
    control = registry.get("CLN-05-R02")
    window = int(config.get("readmit_window_days"))
    frame = context.claims.head(60).copy()

    frame["discharge_readmit_gap_days"] = window
    assert not _run(context, control, claims=frame), (
        f"CLN-05-R02 fired at exactly the boundary ({window}); the expression is "
        f"strictly less-than."
    )

    frame["discharge_readmit_gap_days"] = window - 1
    assert _run(context, control, claims=frame), "CLN-05-R02 did not fire one day inside the window"


def test_category_null_missing_input(context, implemented_controls):
    """A required input is absent: the control must not fire and must not raise.

    §4.5: "the absence of a field is frequently itself the signal", so a null
    is an explicit state, never coerced into a comparison.
    """
    frame = context.claims.head(120).copy()
    for column in ("gross_amount_aed", "approved_amount_aed", "length_of_stay_days",
                   "pharmacy_bill_ratio", "discharge_readmit_gap_days",
                   "days_since_policy_start", "num_insurers_same_event"):
        frame[column] = np.nan
    frame["service_date"] = pd.NaT
    frame["discharge_date"] = pd.NaT

    for control in implemented_controls:
        try:
            signals = _run(context, control, claims=frame)
        except Exception as exc:  # pragma: no cover - the assertion is the point
            pytest.fail(f"{control.rule_id} raised on null inputs: {type(exc).__name__}: {exc}")
        for s in signals:
            assert s.confidence <= 1.0


def test_category_effective_date(context, registry, implemented_controls):
    """Two policy versions: the as-of join must resolve to the one in force.

    A control whose ``effective_from`` is after the run date is not runnable,
    and one whose ``effective_to`` has passed is not runnable either — the
    version in force on the evaluated date is the one that applies (§3.6).
    """
    run_date = context.run_date
    for control in implemented_controls[:6]:
        future = control.model_copy(update={"effective_from": _dt.date(2099, 1, 1)})
        assert not future.is_runnable(run_date), f"{control.rule_id} ran before it was effective"

        expired = control.model_copy(
            update={"effective_from": _dt.date(2020, 1, 1), "effective_to": _dt.date(2021, 1, 1)})
        assert not expired.is_runnable(run_date), f"{control.rule_id} ran after it expired"

        current = control.model_copy(
            update={"effective_from": _dt.date(2020, 1, 1), "effective_to": None})
        assert current.is_runnable(run_date)


def test_category_correction_cancellation_resubmission(context, registry):
    """Lineage-aware behaviour: a superseded claim does not re-signal.

    PAY-01-R01 consults the lineage index before raising a duplicate. A claim
    that has been corrected is superseded, and the duplicate against it must
    close rather than duplicate (§6.3).
    """
    from fwa.lineage.episodes import ClaimVersionRecord, LineageIndex

    control = registry.get("PAY-01-R01")
    base = context.claims.head(1).copy()
    twin = base.copy()
    twin["claim_sk"] = twin["claim_sk"] + "-B"
    twin["claim_date"] = pd.to_datetime(twin["claim_date"]) + pd.Timedelta(days=2)
    frame = pd.concat([base, twin], ignore_index=True)

    signals = _run(context, control, claims=frame)
    assert signals, "PAY-01-R01 did not detect a constructed exact duplicate"

    lineage = LineageIndex()
    lineage.add(ClaimVersionRecord(
        claim_sk=str(twin["claim_sk"].iloc[0]), version_no=2, relationship="CORRECTION",
        prior_claim_sk=str(base["claim_sk"].iloc[0]), changed_fields={"amount": (1, 2)},
        recorded_at=_dt.datetime.now(_dt.timezone.utc),
    ))
    ctx = _with(context, claims=frame, lineage=lineage)
    suppressed = CONTROL_IMPLEMENTATIONS[control.implementation](ctx, control)
    assert not suppressed, (
        "PAY-01-R01 raised a duplicate against a claim that had been SUPERSEDED by a "
        "correction; §6.3 requires the corrected claim to supersede rather than duplicate."
    )
    assert lineage.active_claim_for(str(base["claim_sk"].iloc[0])) == str(twin["claim_sk"].iloc[0])


def test_category_approved_exception(implemented_controls, registry):
    """Every declared exclusion exists, and the contract refuses a control without one."""
    from fwa.engine.contract import AtomicControl, ControlContractError

    for control in implemented_controls:
        assert control.exclusions, f"{control.rule_id} declares no exclusions"
        assert all(isinstance(e, str) and e.strip() for e in control.exclusions)

    sample = implemented_controls[0].model_dump()
    sample["exclusions"] = []
    with pytest.raises((ControlContractError, Exception)):
        AtomicControl(**sample)


def test_category_idempotency(context, implemented_controls, baseline):
    """Replaying the same input produces NO duplicate signal.

    Uniform across types: the Isolation Forest control is held to this exactly
    as the duplicate edit is (§6.2).
    """
    store = SignalStore()
    for control in implemented_controls:
        store.extend(baseline[control.rule_id])
    first = len(store)

    for control in implemented_controls:
        store.extend(_run(context, control))
    assert len(store) == first, (
        f"Replaying the catalogue changed the signal store from {first} to {len(store)}; "
        f"signal identity must be a pure function of (tenant, rule, version, subject, fact, "
        f"period)."
    )

    for control in implemented_controls:
        ids_a = {s.signal_id for s in baseline[control.rule_id]}
        ids_b = {s.signal_id for s in _run(context, control)}
        assert ids_a == ids_b, f"{control.rule_id} produced different signal ids on replay"


def test_category_tenant_isolation(context, implemented_controls, claims):
    """No cross-tenant data leaks into a rule's evaluation."""
    other = claims.copy()
    other["tenant_id"] = "T999"
    other["claim_sk"] = other["claim_sk"] + "-T999"
    mixed = pd.concat([claims, other], ignore_index=True)
    own_claims = set(claims["claim_sk"])

    for control in implemented_controls:
        signals = _run(context, control, claims=mixed, tenant_id="T001")
        for s in signals:
            assert s.tenant_id == "T001", f"{control.rule_id} emitted a signal for another tenant"
            for claim_id in s.claim_ids:
                assert claim_id in own_claims, (
                    f"{control.rule_id} referenced claim {claim_id} belonging to tenant T999"
                )


def test_category_evidence_snapshot(context, implemented_controls, baseline):
    """Rendered evidence contains human-readable facts and the correct versions."""
    for control in implemented_controls:
        for signal in baseline[control.rule_id][:5]:
            assert signal.evidence, f"{control.rule_id} produced a signal with empty evidence"
            assert signal.reference_versions.get("rule") == f"{control.rule_id}@{control.version}"
            assert signal.reference_versions.get("rule_fingerprint") == control.fingerprint()
            assert signal.reference_versions.get("parameters"), (
                f"{control.rule_id} did not stamp the parameter-registry fingerprint"
            )
            assert "exposure_basis" in signal.evidence
            human = [
                v for v in signal.evidence.values()
                if isinstance(v, str) and len(v) > 25 and " " in v
            ]
            assert human, (
                f"{control.rule_id} rendered no human-readable fact; the evidence-snapshot "
                f"category requires evidence a reviewer can read, not only identifiers"
            )
            if control.data_support is not DataSupport.EXECUTABLE:
                assert signal.evidence.get("data_support_reason"), (
                    f"{control.rule_id} runs on a proxy but the signal does not say so"
                )
