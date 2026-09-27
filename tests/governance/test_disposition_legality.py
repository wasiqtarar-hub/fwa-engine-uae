"""THE safety-boundary test (manuscript §3.3, build brief §1).

    "Enforce this with a validator that **raises at rule-registration time** if
     any control of type ``S``, ``N``, ``T`` or ``M`` declares ``REJECT`` or
     ``REPRICE``. Add a unit test that asserts the validator fires. It must be
     *impossible* to configure an anomaly score into a denial."

If any test in this file fails, the system's central claim is false.
"""

from __future__ import annotations

import pytest

from fwa.engine.contract import (
    AtomicControl, DispositionLegalityError, validate_disposition_legality,
)
from fwa.engine.registry import RuleRegistry
from fwa.enums import ControlType, DENYING_DISPOSITIONS, Disposition, TYPES_ALLOWED_TO_DENY

pytestmark = pytest.mark.governance

ILLEGAL_TYPES = [ControlType.S, ControlType.N, ControlType.T, ControlType.M]


@pytest.mark.parametrize("ctype", ILLEGAL_TYPES)
@pytest.mark.parametrize("disposition", sorted(DENYING_DISPOSITIONS, key=lambda d: d.value))
def test_validator_raises_for_every_illegal_combination(ctype, disposition):
    with pytest.raises(DispositionLegalityError) as exc:
        validate_disposition_legality("ANL-01-R99", [ctype], disposition)
    assert "SAFETY BOUNDARY VIOLATION" in str(exc.value)
    assert "A signal is not a fraud finding" in str(exc.value)


@pytest.mark.parametrize("ctype", [ControlType.H, ControlType.E])
@pytest.mark.parametrize("disposition", sorted(DENYING_DISPOSITIONS, key=lambda d: d.value))
def test_validator_permits_hard_and_expert_edits(ctype, disposition):
    validate_disposition_legality("PAY-01-R01", [ctype], disposition)


def test_composite_type_containing_a_score_may_not_deny():
    """A control that is PART statistical is statistical for §3.3 purposes."""
    with pytest.raises(DispositionLegalityError):
        validate_disposition_legality("PAY-01-R04", [ControlType.H, ControlType.S],
                                      Disposition.REJECT)


def _payload(**overrides):
    base = dict(
        rule_id="ANL-01-R99", scenario_id="ANL-01", name="Illegal test control",
        version="1.0.0", type="M", stage="MODEL_MONTHLY", population="claim",
        inputs=["claim_header.gross_amount"], expression="score > threshold",
        exclusions=["none_declared_by_policy_owner"],
        grouping_key=["tenant_id", "provider_sk"], score=50.0,
        disposition="REJECT", reason_code="ILLEGAL", evidence_fields=["score"],
        owner="analytics", effective_from="2026-01-01",
        data_support="NOT_EXECUTABLE_ON_THIS_DATASET",
        data_support_reason="Test fixture.",
    )
    base.update(overrides)
    return base


def test_contract_construction_raises_at_registration_time():
    """The check fires when the control is CONSTRUCTED, not when it executes."""
    with pytest.raises(DispositionLegalityError):
        AtomicControl(**_payload())


def test_registry_refuses_an_illegal_control(tmp_path):
    """The engine cannot be started with an illegal control in the catalogue."""
    import yaml

    rules = tmp_path / "ANL"
    rules.mkdir(parents=True)
    (rules / "ANL-01.yaml").write_text(yaml.safe_dump({
        "scenario": {"scenario_id": "ANL-01", "title": "t", "priority": "P1"},
        "controls": [_payload()],
    }), encoding="utf-8")
    with pytest.raises(DispositionLegalityError):
        RuleRegistry.from_directory(tmp_path, strict=True)


def test_registry_register_rechecks_even_for_a_hand_built_control():
    """Defence in depth: a control built by any other route is re-checked."""
    registry = RuleRegistry()
    legal = AtomicControl(**_payload(type="H", disposition="REJECT"))
    registry.register(legal)
    object.__setattr__(legal, "__dict__", dict(legal.__dict__))
    smuggled = legal.model_construct(**{**legal.model_dump(), "type": [ControlType.M],
                                        "disposition": Disposition.REJECT})
    with pytest.raises(DispositionLegalityError):
        registry.register(smuggled)


def test_no_shipped_control_violates_the_boundary(registry):
    """Every one of the 164 catalogue controls, checked."""
    for control in registry.all():
        if control.disposition in DENYING_DISPOSITIONS:
            offending = [t for t in control.type if t not in TYPES_ALLOWED_TO_DENY]
            assert not offending, (
                f"{control.rule_id} ({control.type_label}) declares "
                f"{control.disposition.value}"
            )


def test_there_is_no_override_flag():
    """No parameter, environment variable or argument relaxes the check.

    The scan runs over the *executable* source — the docstring is stripped
    first, so prose such as "There is no ``force=True``" is not mistaken for an
    escape hatch, and the word "enforce" is not mistaken for one either (hence
    the word-boundary match rather than a substring test).
    """
    import ast
    import inspect
    import re

    tree = ast.parse(inspect.getsource(validate_disposition_legality).lstrip())
    func = tree.body[0]
    if ast.get_docstring(func) is not None:
        func.body = func.body[1:]
    code = ast.unparse(func)

    for escape in ("force", "override", "skip", "bypass", "allow_illegal", "environ", "getenv"):
        assert re.search(rf"\b{escape}\b", code) is None, (
            f"validate_disposition_legality mentions {escape!r}; the boundary must have no "
            f"escape hatch."
        )
    # And it takes exactly the three arguments it needs — nothing that could be
    # threaded through by a caller to soften the outcome.
    params = list(inspect.signature(validate_disposition_legality).parameters)
    assert params == ["rule_id", "types", "disposition"], params


def test_model_controls_in_the_catalogue_may_never_deny(registry):
    """No Type-M control declares a denying disposition — the §3.3 rule itself."""
    for control in registry.by_type(ControlType.M):
        assert control.disposition not in DENYING_DISPOSITIONS, (
            f"{control.rule_id} ({control.type_label}) is a model control declaring "
            f"{control.disposition.value}"
        )


def test_pure_model_controls_in_the_catalogue_are_monitor_only(registry):
    """A control that is *only* Type M resolves to MONITOR_ONLY, as Appendix C chooses.

    Composite types are deliberately excluded. ``CLN-01-R02`` is typed ``S/M``
    and Appendix C gives it the output "Prepay sample", so ``PREPAY_PEND`` is
    both correct and legal: §3.3 forbids a score from *denying*, not from
    routing a claim to a human before payment. Asserting MONITOR_ONLY for every
    control with an M in its type would be asserting something stricter than
    the manuscript says, and would quietly misrepresent the catalogue.
    """
    pure_model = [c for c in registry.by_type(ControlType.M) if list(c.type) == [ControlType.M]]
    assert pure_model, "The catalogue should contain at least one pure Type-M control."
    for control in pure_model:
        assert control.disposition is Disposition.MONITOR_ONLY, (
            f"{control.rule_id} is a pure model control with disposition "
            f"{control.disposition.value}"
        )
