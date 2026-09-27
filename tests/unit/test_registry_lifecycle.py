"""Rule lifecycle, separation of duties, kill switch and rollback (§3.10 NFR, §9.7).

    "Kill switch per rule (deactivate without deployment), rule rollback to a
     previous version... The user who authors a rule may not approve its
     shadow→active transition."

The registry is the only object in the system that can change what executes, so
every transition it offers is tested here, including the ones that must *fail*.
Each test builds its own registry: drafting or killing a control in the
session-scoped catalogue would leak into every test that ran afterwards.
"""

from __future__ import annotations

import datetime as _dt

import pytest

from fwa.engine.contract import AtomicControl
from fwa.engine.registry import RuleNotFound, RuleRegistry, SeparationOfDutiesError
from fwa.enums import ControlType, Disposition, RuleStatus, Stage


def _payload(**overrides):
    base = dict(
        rule_id="PAY-01-R09", scenario_id="PAY-01", name="Lifecycle fixture",
        version="1.0.0", type="H", stage="PREPAY_SYNC", population="claim",
        inputs=["claim_header.gross_amount_aed"], expression="a == b",
        exclusions=["corrected_resubmission"],
        grouping_key=["tenant_id", "claim_sk"], score=70.0,
        disposition="REJECT", reason_code="FIXTURE", evidence_fields=["claim_sk"],
        owner="claims_policy", effective_from="2026-01-01",
        data_support="NOT_EXECUTABLE_ON_THIS_DATASET",
        data_support_reason="Fixture control used only by the lifecycle tests.",
    )
    base.update(overrides)
    return AtomicControl(**base)


@pytest.fixture()
def empty():
    return RuleRegistry()


# ------------------------------------------------------------------- drafting


def test_a_draft_enters_shadow_whoever_wrote_it(empty):
    drafted = empty.draft(_payload(status="active"), author="priya")
    assert drafted.status is RuleStatus.SHADOW
    assert drafted.authored_by == "priya"
    assert drafted.approved_by is None
    assert drafted.shadow_since == _dt.date.today()


def test_a_draft_is_retrievable_and_counted(empty):
    empty.draft(_payload(), author="priya")
    assert "PAY-01-R09" in empty
    assert len(empty) == 1
    assert empty.get("PAY-01-R09").rule_id == "PAY-01-R09"


def test_an_unknown_rule_raises(empty):
    with pytest.raises(RuleNotFound):
        empty.get("PAY-99-R99")


# --------------------------------------------------------- separation of duties


def test_the_author_may_not_approve_their_own_rule(empty):
    empty.draft(_payload(), author="priya")
    with pytest.raises(SeparationOfDutiesError) as exc:
        empty.activate("PAY-01-R09", actor="priya")
    assert "may not approve" in str(exc.value)


def test_a_different_approver_may_activate(empty):
    empty.draft(_payload(), author="priya")
    active = empty.activate("PAY-01-R09", actor="amina", note="Reviewed in the weekly forum.")
    assert active.status is RuleStatus.ACTIVE
    assert active.approved_by == "amina"
    assert active.approved_at is not None
    assert "weekly forum" in active.governance_note


def test_activating_an_already_active_rule_is_a_no_op(empty):
    empty.draft(_payload(), author="priya")
    once = empty.activate("PAY-01-R09", actor="amina")
    twice = empty.activate("PAY-01-R09", actor="amina")
    assert twice.approved_at == once.approved_at


def test_a_control_registered_without_an_author_can_be_activated(empty):
    """A catalogue control loaded from YAML has no author to conflict with."""
    empty.register(_payload())
    assert empty.activate("PAY-01-R09", actor="amina").status is RuleStatus.ACTIVE


# ----------------------------------------------------------------- kill switch


def test_a_kill_switch_deactivates_without_a_deployment(empty):
    empty.register(_payload())
    killed = empty.kill_switch("PAY-01-R09", actor="amina", reason="False positives after tariff change")
    assert killed.kill_switched is True
    assert "amina" in killed.kill_switch_reason
    assert "tariff" in killed.kill_switch_reason


def test_a_kill_switch_requires_a_stated_reason(empty):
    """§9.7: turning a control off is a governed act, not a toggle."""
    empty.register(_payload())
    with pytest.raises(ValueError):
        empty.kill_switch("PAY-01-R09", actor="amina", reason="   ")


def test_a_killed_control_does_not_execute(empty, context):
    """The evaluator reports KILL_SWITCHED — a distinct outcome from "found nothing"."""
    from fwa.engine.evaluator import Evaluator
    from fwa.engine.evallib import EvaluationOutcome

    control = _payload(
        rule_id="ENT-03-R02", scenario_id="ENT-03",
        data_support="EXECUTABLE", implementation="ent_03_r02_excluded_entity",
        data_support_reason="provider_blacklist_flag is present, as a documented proxy.",
    )
    empty.register(control)

    before = Evaluator(empty).run(context)
    assert before.records[0].outcome == EvaluationOutcome.TRIGGERED.value

    empty.kill_switch("ENT-03-R02", actor="amina", reason="Suspected data-quality issue.")
    after = Evaluator(empty).run(context)
    assert after.records[0].outcome == EvaluationOutcome.KILL_SWITCHED.value
    assert after.records[0].signal_count == 0


def test_a_killed_control_can_be_restored(empty):
    empty.register(_payload())
    empty.kill_switch("PAY-01-R09", actor="amina", reason="Investigating.")
    restored = empty.restore("PAY-01-R09", actor="amina", reason="Data issue resolved.")
    assert restored.kill_switched is False
    assert restored.kill_switch_reason == ""
    assert "Restored by amina" in restored.governance_note


# -------------------------------------------------------------------- retire


def test_retiring_keeps_the_prior_version_in_history(empty):
    empty.register(_payload())
    retired = empty.retire("PAY-01-R09", actor="amina", reason="Superseded by PAY-01-R10.")
    assert retired.status is RuleStatus.RETIRED
    assert "Superseded" in retired.governance_note
    assert empty.history("PAY-01-R09")


def test_a_retired_control_does_not_execute(empty, context):
    from fwa.engine.evaluator import Evaluator
    from fwa.engine.evallib import EvaluationOutcome

    empty.register(_payload(
        rule_id="ENT-03-R02", scenario_id="ENT-03",
        data_support="EXECUTABLE", implementation="ent_03_r02_excluded_entity",
        data_support_reason="provider_blacklist_flag is present, as a documented proxy.",
    ))
    empty.retire("ENT-03-R02", actor="amina", reason="No longer required.")
    report = Evaluator(empty).run(context)
    assert report.records[0].outcome == EvaluationOutcome.NOT_EFFECTIVE.value


# ------------------------------------------------------------------ versions


def test_a_new_version_pushes_the_old_one_into_history(empty):
    empty.register(_payload(version="1.0.0"))
    empty.register(_payload(version="1.1.0", score=80.0))
    assert empty.get("PAY-01-R09").version == "1.1.0"
    assert [c.version for c in empty.history("PAY-01-R09")] == ["1.0.0"]


def test_changing_a_rule_without_bumping_the_version_is_refused(empty):
    """Signals store ``rule_id@version``; silently changing v1.0.0 breaks reproducibility."""
    from fwa.engine.contract import ControlContractError

    empty.register(_payload(version="1.0.0", score=70.0))
    with pytest.raises(ControlContractError) as exc:
        empty.register(_payload(version="1.0.0", score=90.0))
    assert "Bump the version" in str(exc.value)


def test_re_registering_identical_content_is_allowed(empty):
    empty.register(_payload(version="1.0.0"))
    empty.register(_payload(version="1.0.0"))
    assert len(empty) == 1


# ------------------------------------------------------------------ rollback


def test_a_rollback_restores_the_previous_version(empty):
    empty.register(_payload(version="1.0.0", score=70.0))
    empty.register(_payload(version="1.1.0", score=95.0))
    rolled = empty.rollback("PAY-01-R09", actor="amina")
    assert rolled.version == "1.0.0"
    assert rolled.score == 70.0


def test_a_rollback_re_enters_shadow_never_straight_to_active(empty):
    """The safety property: an emergency rollback does not silently resume denying claims."""
    empty.register(_payload(version="1.0.0"))
    empty.register(_payload(version="1.1.0", score=95.0))
    empty.activate("PAY-01-R09", actor="amina")
    rolled = empty.rollback("PAY-01-R09", actor="amina")
    assert rolled.status is RuleStatus.SHADOW
    assert "Rolled back by amina" in rolled.governance_note


def test_a_rollback_can_target_a_named_version(empty):
    empty.register(_payload(version="1.0.0", score=10.0))
    empty.register(_payload(version="1.1.0", score=20.0))
    empty.register(_payload(version="1.2.0", score=30.0))
    assert empty.rollback("PAY-01-R09", actor="amina", to_version="1.0.0").score == 10.0


def test_a_rollback_to_an_unregistered_version_raises(empty):
    empty.register(_payload(version="1.0.0"))
    empty.register(_payload(version="1.1.0", score=20.0))
    with pytest.raises(RuleNotFound):
        empty.rollback("PAY-01-R09", actor="amina", to_version="9.9.9")


def test_a_rollback_with_no_history_raises(empty):
    empty.register(_payload())
    with pytest.raises(RuleNotFound):
        empty.rollback("PAY-01-R09", actor="amina")


# ----------------------------------------------------------------- selectors


def test_selectors_partition_the_catalogue(registry):
    assert len(registry.by_stage(Stage.PREPAY_SYNC)) > 0
    assert len(registry.by_type(ControlType.H)) > 0
    assert len(registry.by_status(RuleStatus.SHADOW)) == len(registry)
    assert len(registry.executable()) < len(registry)
    assert set(registry.families()) <= {"ENT", "PAY", "CLN", "PHR", "DOC", "NET", "ANL", "POL"}


def test_denying_controls_are_exactly_the_reject_and_reprice_ones(registry):
    denying = registry.denying_controls()
    assert denying
    for control in denying:
        assert control.disposition in (Disposition.REJECT, Disposition.REPRICE)


def test_runnable_respects_the_effective_date(empty):
    """``runnable`` is "executable here AND in force on this date" — both halves."""
    empty.register(_payload(
        effective_from="2027-01-01",
        data_support="EXECUTABLE", implementation="ent_03_r02_excluded_entity",
        data_support_reason="provider_blacklist_flag is present, as a documented proxy.",
    ))
    assert not empty.runnable(_dt.date(2026, 6, 1))
    assert [c.rule_id for c in empty.runnable(_dt.date(2027, 6, 1))] == ["PAY-01-R09"]


def test_a_control_this_dataset_cannot_run_is_never_runnable(empty):
    empty.register(_payload(effective_from="2020-01-01"))
    assert not empty.runnable(_dt.date(2026, 6, 1))
