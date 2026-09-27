"""The atomic-control contract (§3.7) as an enforced object, not a document.

§3.7 calls every field in the control contract "a requirement, not
documentation". That sentence is only true if the model refuses to construct a
control that breaks it, so this module tests the refusals rather than the happy
path: a control with no declared exclusions, no owner, no evidence fields, a
malformed rule id, a non-semantic version, or a ``data_support`` classification
that contradicts whether the control is actually wired to an implementation.

The legality boundary itself (§3.3 — a statistical, network, document or model
control may never declare ``REJECT`` or ``REPRICE``) is exercised against the
shipped 164-control catalogue in ``tests/governance/test_disposition_legality.py``.
What is tested here is the narrower and equally load-bearing property that the
check runs *on construction*, for composite types too, and for the
``alternate_dispositions`` list as well as the primary disposition — because a
control that declares a legal primary disposition and an illegal alternate
would otherwise slip past a check that only looked at the primary.
"""

from __future__ import annotations

import datetime as _dt

import pytest
from pydantic import ValidationError

from fwa.engine.contract import (
    AtomicControl,
    ControlContractError,
    DispositionLegalityError,
    parse_types,
    validate_disposition_legality,
)
from fwa.enums import ControlType, DataSupport, Disposition, RuleStatus

VALID = dict(
    rule_id="PAY-04-R01", scenario_id="PAY-04", name="Missing mandatory authorization",
    version="1.0.0", type="H", stage="PREPAY_SYNC",
    population="claim_line where benefit_rule.authorization_required = true",
    inputs=["line.service_date", "line.activity_code"],
    expression="authorization_id is null and no policy_exception exists",
    parameters={"grace_minutes": 0},
    exclusions=["emergency_exception", "deemed_approval", "regulator_exception"],
    grouping_key=["payer_id", "member_sk", "episode_id", "scenario_id"],
    score=100.0, disposition="PREPAY_PEND", reason_code="AUTH_MISSING",
    evidence_fields=["activity_code", "service_date", "authorization_rule_version"],
    owner="claims_policy", effective_from="2026-01-01",
    data_support="NOT_EXECUTABLE_ON_THIS_DATASET",
    data_support_reason="No authorization table exists in this source file.",
)


def _control(**overrides) -> AtomicControl:
    payload = dict(VALID)
    payload.update(overrides)
    return AtomicControl(**payload)


def _refused(**overrides):
    """Build a control that must not construct, and return the message.

    Contract violations are raised as :class:`ControlContractError`, which is a
    ``ValueError``, so pydantic converts them into a ``ValidationError`` with
    the original message preserved. That wrapping is deliberate and is asserted
    directly in :func:`test_a_contract_violation_is_a_validation_error`; the
    §3.3 legality violation is the one that must NOT be wrapped, and it is
    tested separately for exactly that reason.
    """
    with pytest.raises(ValidationError) as excinfo:
        _control(**overrides)
    return str(excinfo.value)


# ---------------------------------------------------------------- the contract

def test_the_specification_example_constructs():
    """§3.7's own worked example must be a valid control, or the contract has drifted."""
    control = _control()
    assert control.rule_id == "PAY-04-R01"
    assert control.status is RuleStatus.SHADOW, "a control is born in shadow, never active"
    assert control.type == [ControlType.H]
    assert control.disposition is Disposition.PREPAY_PEND
    assert control.effective_from == _dt.date(2026, 1, 1)


def test_a_control_is_frozen_once_built():
    """A new rule version is a new row, never an in-place edit (§3.7, §5.4)."""
    control = _control()
    with pytest.raises(Exception):
        control.score = 10.0


def test_an_unknown_field_is_refused():
    """``extra='forbid'``: a typo'd contract field must not be silently accepted."""
    with pytest.raises(Exception):
        _control(dispostion="REJECT")


@pytest.mark.parametrize("field_name", ["exclusions", "grouping_key", "evidence_fields"])
def test_an_empty_required_list_is_refused(field_name):
    """The three list fields §3.7 identifies as making governance real.

    Empty exclusions means a reviewer discovers the legitimate exceptions; an
    empty grouping_key means case correlation needs a bespoke rule per scenario;
    empty evidence_fields means a signal that cannot render why it fired.
    """
    _refused(**{field_name: []})


def test_a_control_with_no_owner_is_refused():
    """§3.9 parameter governance is auditable only if someone owns the rule."""
    _refused(owner="   ")


@pytest.mark.parametrize("bad_id", ["PAY-4-R01", "PAY-04-01", "XXX-04-R01", "pay-04-r01"])
def test_a_malformed_rule_id_is_refused(bad_id):
    """Appendix C identifiers generate the traceability matrix; their shape is load-bearing."""
    _refused(rule_id=bad_id)


def test_a_rule_must_belong_to_the_scenario_it_names():
    _refused(rule_id="PAY-04-R01", scenario_id="CLN-01")


@pytest.mark.parametrize("bad_version", ["1.0", "v1.0.0", "latest"])
def test_a_non_semantic_version_is_refused(bad_version):
    """Rule versions are stamped onto every signal so a finding can be reproduced."""
    _refused(version=bad_version)


def test_score_must_lie_in_the_declared_range():
    _refused(score=140.0)


# -------------------------------------------------- data-support honesty rules

def test_an_unrunnable_control_may_not_name_an_implementation():
    """Otherwise the coverage matrix claims a control runs when it does not."""
    _refused(data_support="NOT_EXECUTABLE_ON_THIS_DATASET",
                 implementation="pay_04_r01_missing_authorization")


def test_a_runnable_control_must_name_an_implementation():
    """And the reverse: a classification of EXECUTABLE with nothing behind it."""
    _refused(data_support="EXECUTABLE", implementation=None)


def test_the_data_support_classification_must_state_a_reason():
    _refused(data_support_reason="")


def test_a_runnable_control_with_an_implementation_is_accepted():
    control = _control(data_support="PARTIAL", implementation="pay_10_r01_duplicate",
                       data_support_reason="Runs against a claim-header proxy for the line grain.")
    assert control.data_support is DataSupport.PARTIAL
    assert control.implementation == "pay_10_r01_duplicate"


# ------------------------------------------------------- the §3.3 boundary itself

def test_parse_types_reads_the_catalogues_composite_notation():
    assert parse_types("H") == [ControlType.H]
    assert parse_types("E/T") == [ControlType.E, ControlType.T]
    assert parse_types("H|S") == [ControlType.H, ControlType.S]


def test_an_unknown_type_is_refused():
    with pytest.raises(ControlContractError):
        parse_types("X")


@pytest.mark.parametrize("ctype", ["S", "N", "T", "M"])
@pytest.mark.parametrize("disposition", ["REJECT", "REPRICE"])
def test_a_scoring_control_may_never_deny(ctype, disposition):
    """The single most important refusal in the repository (§3.3)."""
    with pytest.raises(DispositionLegalityError):
        _control(type=ctype, disposition=disposition,
                 data_support="NOT_EXECUTABLE_ON_THIS_DATASET", implementation=None)


def test_a_composite_type_is_judged_by_its_weakest_component():
    """"A control that is part statistical is, for the purpose of §3.3, statistical."

    H alone may deny. H/S may not, and that is the case a per-primary-type check
    would wave through.
    """
    _control(type="H", disposition="REJECT")                      # legal
    with pytest.raises(DispositionLegalityError):
        _control(type="H/S", disposition="REJECT")


def test_an_illegal_alternate_disposition_is_caught_too():
    """A legal primary must not be a way to smuggle an illegal alternate through."""
    with pytest.raises(DispositionLegalityError):
        _control(type="S", disposition="POSTPAY_AUDIT", alternate_dispositions=["REJECT"])


def test_the_boundary_has_no_override_parameter():
    """§3.3 is unconditional: there is no threshold, flag or severity that relaxes it.

    Asserted against the validator's signature rather than its behaviour, so
    that adding a ``force`` or ``allow_deny`` argument later breaks a test
    instead of quietly widening the boundary.
    """
    import inspect

    params = list(inspect.signature(validate_disposition_legality).parameters)
    assert params == ["rule_id", "types", "disposition"]


def test_a_contract_violation_is_a_validation_error():
    """The wrapping that makes the legality exception's separate type necessary.

    An ordinary contract violation IS a ``ValueError``, so pydantic converts it
    into a ``ValidationError`` alongside missing fields and bad dates. That is
    fine for a malformed rule. It is not fine for an unsafe one, which is the
    point of the next test.
    """
    message = _refused(exclusions=[])
    assert "no declared exclusions" in message
    assert issubclass(ControlContractError, ValueError)


def test_a_legality_violation_is_not_a_value_error():
    """It must not be catchable by ``except ValidationError`` or ``except ControlContractError``.

    A caller skipping malformed controls must not also skip unsafe ones.
    """
    assert not issubclass(DispositionLegalityError, ValueError)
    assert not issubclass(DispositionLegalityError, ControlContractError)
