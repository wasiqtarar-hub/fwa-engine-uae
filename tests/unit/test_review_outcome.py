"""Authentication, the review loop and the override path (build brief §13, §14.1).

    "Rationale is mandatory... Manual adjudication override is a distinct
     permission, NOT implied by any role... PHI masking by default with an
     explicit, audited unmask action."

The review outcome is where FR7 closes: a reviewer's disposition and its
rationale are the only feedback the system ever receives, so the tests here
care most about what the system refuses to accept — a disposition with no
rationale, an override by someone who was never granted the permission, an
unmask with no stated reason. Each refusal leaves an audit entry.
"""

from __future__ import annotations

import datetime as _dt

import pytest

from fwa.audit.log import AuditEventType, AuditLog
from fwa.auth.rbac import AccessDenied, Permission, Role
from fwa.auth.service import MAX_FAILED_ATTEMPTS, AuthenticationError, AuthService
from fwa.enums import Disposition
from fwa.storage.db import Database


@pytest.fixture()
def service():
    svc = AuthService(Database("sqlite://"), audit=AuditLog())
    svc.seed()
    return svc


@pytest.fixture()
def reviewer(service):
    return service.login("reviewer", "reviewer")


@pytest.fixture()
def admin(service):
    return service.login("admin", "admin")


class _Case:
    case_id = "CASE-0001"
    tenant_id = "T001"
    signal_ids = ["sig-a", "sig-b"]
    disposition = Disposition.PREPAY_PEND
    primary_subject_type = "provider"
    primary_subject_id = "H0141"


# -------------------------------------------------------------- authentication


def test_a_seeded_account_logs_in_and_must_change_its_password(service):
    state = service.login("reviewer", "reviewer")
    assert state.role is Role.CLAIMS_REVIEWER
    assert state.tenant_id == "T001"
    assert state.must_change_password is True, (
        "A seeded demo password that never has to change is a shipped credential."
    )


def test_a_wrong_password_is_refused_and_recorded(service):
    with pytest.raises(AuthenticationError):
        service.login("reviewer", "not-the-password")
    failures = service.audit.by_type(AuditEventType.LOGIN_FAILURE)
    assert len(failures) == 1
    assert failures[0].actor == "reviewer"


def test_an_unknown_account_gives_the_same_message_as_a_wrong_password(service):
    """Not leaking which half was wrong is the point."""
    with pytest.raises(AuthenticationError) as unknown:
        service.login("nobody", "x")
    with pytest.raises(AuthenticationError) as wrong:
        service.login("reviewer", "x")
    assert str(unknown.value) == str(wrong.value)


def test_an_account_locks_after_repeated_failures(service):
    for _ in range(MAX_FAILED_ATTEMPTS):
        with pytest.raises(AuthenticationError):
            service.login("reviewer", "wrong")
    with pytest.raises(AuthenticationError) as exc:
        service.login("reviewer", "reviewer")   # the correct password, now locked out
    assert "locked" in str(exc.value)


def test_a_successful_login_clears_the_failure_count(service):
    with pytest.raises(AuthenticationError):
        service.login("reviewer", "wrong")
    service.login("reviewer", "reviewer")
    with pytest.raises(AuthenticationError):
        service.login("reviewer", "wrong")
    service.login("reviewer", "reviewer")  # still not locked


def test_the_password_is_not_stored_in_clear(service):
    user = service.db.get_user("reviewer")
    assert "reviewer" not in user.password_hash
    assert user.password_hash.startswith("$argon2")


def test_changing_a_password_clears_the_forced_change(service, reviewer):
    service.change_own_password(reviewer, "reviewer", "a-longer-password")
    assert reviewer.must_change_password is False
    assert service.login("reviewer", "a-longer-password") is not None


def test_a_short_or_unchanged_password_is_refused(service, reviewer):
    with pytest.raises(ValueError):
        service.change_own_password(reviewer, "reviewer", "short")
    with pytest.raises(ValueError):
        service.change_own_password(reviewer, "reviewer", "reviewer")


def test_a_session_times_out(service, reviewer):
    reviewer.last_seen = _dt.datetime.now(_dt.timezone.utc) - _dt.timedelta(hours=4)
    assert reviewer.is_idle(timeout_minutes=30)
    reviewer.touch()
    assert not reviewer.is_idle(timeout_minutes=30)


# ------------------------------------------------------------- the review loop


def test_a_disposition_requires_a_rationale(service, reviewer):
    """FR7: the rationale is the only feedback the system ever gets."""
    with pytest.raises(ValueError) as exc:
        service.record_disposition(state=reviewer, case=_Case(),
                                   disposition="CLEARED", rationale="ok")
    assert "at least 10 characters" in str(exc.value)
    assert "FR7" in str(exc.value)


def test_a_whitespace_rationale_does_not_count(service, reviewer):
    with pytest.raises(ValueError):
        service.record_disposition(state=reviewer, case=_Case(),
                                   disposition="CLEARED", rationale="          ")


def test_a_recorded_disposition_captures_the_full_outcome(service, reviewer):
    outcome = service.record_disposition(
        state=reviewer, case=_Case(), disposition="CONFIRMED",
        rationale="Clinical record confirms the service was not delivered as billed.",
        validated_category="upcoding", confirmed_amount_aed=2400.0,
    )
    assert outcome.reviewer == "reviewer"
    assert outcome.reviewer_role == "CLAIMS_REVIEWER"
    assert outcome.validated_category == "upcoding"
    assert outcome.confirmed_amount_aed == pytest.approx(2400.0)
    assert outcome.original_disposition == Disposition.PREPAY_PEND.value
    assert outcome.tenant_id == "T001"
    assert outcome.signal_ids == "sig-a,sig-b"


def test_a_disposition_is_audited_with_its_before_and_after(service, reviewer):
    service.record_disposition(state=reviewer, case=_Case(), disposition="CLEARED",
                               rationale="Staged procedure, confirmed with the provider.")
    events = service.audit.by_type(AuditEventType.DISPOSITION_RECORDED)
    assert len(events) == 1
    assert events[0].before == {"disposition": "PREPAY_PEND"}
    assert events[0].after["disposition"] == "CLEARED"
    assert "Staged procedure" in events[0].reason
    assert service.audit.verify_chain()[0]


def test_a_role_without_the_permission_cannot_record_a_disposition(service):
    analyst = service.login("analyst", "analyst")
    with pytest.raises(AccessDenied):
        service.record_disposition(state=analyst, case=_Case(), disposition="CLEARED",
                                   rationale="An analyst should not be able to do this.")


# ------------------------------------------------------------------ override


def test_an_override_needs_a_permission_no_role_confers(service, reviewer, admin):
    """§13.2: "a distinct permission, NOT implied by any role" — including ADMIN."""
    for state in (reviewer, admin):
        with pytest.raises(AccessDenied):
            service.record_disposition(
                state=state, case=_Case(), disposition="REJECT", is_override=True,
                rationale="Overriding the engine's disposition on this case.",
            )


def test_an_individually_granted_override_works_and_is_audited_separately(
    service, reviewer, admin
):
    service.grant_permission(actor=admin, username="reviewer",
                             permission=Permission.MANUAL_OVERRIDE.value,
                             reason="Senior reviewer, approved by the governance forum.")
    granted = service.login("reviewer", "reviewer")

    outcome = service.record_disposition(
        state=granted, case=_Case(), disposition="REJECT", is_override=True,
        rationale="Contract terms make this non-payable; the engine lacks the contract.",
    )
    assert outcome.was_override is True

    overrides = service.audit.by_type(AuditEventType.MANUAL_OVERRIDE)
    assert len(overrides) == 1
    assert not service.audit.by_type(AuditEventType.DISPOSITION_RECORDED), (
        "An override recorded as an ordinary disposition is invisible to an auditor."
    )


# -------------------------------------------------------------------- unmask


def test_unmasking_requires_a_stated_reason(service, admin):
    with pytest.raises(ValueError) as exc:
        service.unmask(admin, "H0141", "why")
    assert "reason" in str(exc.value).lower()


def test_unmasking_is_refused_and_recorded_for_a_role_without_the_permission(service, reviewer):
    with pytest.raises(AccessDenied):
        service.unmask(reviewer, "H0141", "I would like to see who this provider is.")
    denials = service.audit.by_type(AuditEventType.ACCESS_DENIED)
    assert len(denials) == 1
    assert denials[0].subject == "H0141"


def test_a_permitted_unmask_is_audited_and_scoped_to_the_subject(service, admin):
    assert service.unmask(admin, "H0141", "Escalating to the regulator; identity required.")
    events = service.audit.by_type(AuditEventType.PHI_UNMASK)
    assert len(events) == 1
    assert events[0].subject == "H0141"
    assert "H0141" in admin.unmasked_subjects
    assert "H0142" not in admin.unmasked_subjects


# -------------------------------------------------------------------- appeals


def test_an_appeal_records_its_result_and_turnaround(service, reviewer, admin):
    outcome = service.record_disposition(
        state=reviewer, case=_Case(), disposition="CONFIRMED",
        rationale="Documentation does not support the billed level of service.",
    )
    appealed = service.capture_appeal(
        state=admin, review_id=outcome.review_id, result="OVERTURNED",
        reason="Provider supplied the missing operative note.",
    )
    assert appealed.appeal_raised is True
    assert appealed.appeal_result == "OVERTURNED"
    assert appealed.turnaround_hours is not None
    assert service.audit.by_type(AuditEventType.APPEAL_CAPTURED)


def test_an_appeal_on_an_unknown_review_raises(service, admin):
    with pytest.raises(ValueError):
        service.capture_appeal(state=admin, review_id="nope", result="UPHELD", reason="x")


# ------------------------------------------------- model promotion needs two people


def test_an_analyst_may_propose_a_promotion_but_not_approve_one(service):
    """Propose and approve are held by different roles, so they are different people."""
    analyst = service.login("analyst", "analyst")
    service.propose_model_promotion(
        state=analyst, model="IsolationForest",
        reason="Gate evidence attached; lift over the transparent composite is positive.",
    )
    assert service.audit.by_type(AuditEventType.MODEL_PROMOTION_PROPOSED)

    with pytest.raises(AccessDenied):
        service.approve_model_promotion(
            state=analyst, model="IsolationForest", proposer="analyst",
            reason="Approving my own proposal.",
        )


def test_even_an_admin_may_not_approve_a_promotion_they_proposed(service, admin):
    """ADMIN holds both permissions, so separation of duties is what stops it."""
    from fwa.auth.rbac import SeparationOfDutiesError

    service.propose_model_promotion(
        state=admin, model="LOF", reason="Proposing promotion with the gate evidence attached.",
    )
    with pytest.raises(SeparationOfDutiesError):
        service.approve_model_promotion(
            state=admin, model="LOF", proposer="admin", reason="Approving my own proposal.",
        )


def test_a_second_approver_may_approve(service, admin):
    policy = service.login("policy", "policy")
    service.propose_model_promotion(
        state=admin, model="LOF", reason="Proposing promotion with the gate evidence attached.",
    )
    service.approve_model_promotion(
        state=policy, model="LOF", proposer="admin",
        reason="Gate evidence reviewed in the governance forum.",
    )
    assert service.audit.by_type(AuditEventType.MODEL_PROMOTION_APPROVED)
