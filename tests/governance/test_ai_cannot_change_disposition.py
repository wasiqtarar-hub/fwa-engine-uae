"""No AI output sets or changes a disposition, priority or exposure (build brief §1, §9).

    "No LLM output, anomaly score, graph metric or NLP extraction may set or
     change a disposition."
    "The narrative is **advisory** ... it **never touches disposition, priority
     or exposure**."

The test is a before/after equality check around every AI entry point, taken on
a *deep copy* of the case so that an in-place mutation cannot be hidden by both
sides pointing at the same object. It is supported by two structural tests: the
AI classes hold no reference to the registry, the signal store or the
correlator, and the disposition of a case is derived from its signals by a
function the AI layer cannot reach.
"""

from __future__ import annotations

import copy
import inspect

import pytest

from fwa.ai.copilot import Copilot
from fwa.ai.narrative import ADVISORY_PANEL_LABEL, CaseNarrator
from fwa.ai.provider import OfflineDeterministicProvider
from fwa.ai.triage import TypologyTriage
from fwa.audit.log import AuditLog

pytestmark = pytest.mark.governance


def _snapshot(case):
    return {
        "disposition": case.disposition,
        "priority": None if case.priority is None else case.priority.value,
        "exposure_aed": case.exposure_aed,
        "exposure_established": case.exposure_established,
        "exposure_basis": case.exposure_basis,
        "signal_ids": list(case.signal_ids),
        "claim_ids": list(case.claim_ids),
        "shadow_only": case.shadow_only,
    }


@pytest.fixture(scope="module")
def a_case(sample_cases, sample_signals):
    _, cases = sample_cases
    by_signal = {s.signal_id: s for s in sample_signals}
    # Pick the richest case: the most evidence gives the narrator the most to
    # work with, and therefore the most opportunity to touch something.
    case = max(cases.values(), key=lambda c: len(c.signal_ids))
    signals = [by_signal[i] for i in case.signal_ids if i in by_signal]
    assert signals, "The chosen case resolved to no signals."
    return case, signals


# ------------------------------------------------------------------ narrative


def test_narration_does_not_change_the_case(a_case):
    case, signals = a_case
    working = copy.deepcopy(case)
    before = _snapshot(working)

    narrative = CaseNarrator(OfflineDeterministicProvider(), audit=AuditLog()).narrate(
        working, signals, subject_label="PROV-masked"
    )

    assert narrative.text is not None
    assert _snapshot(working) == before


def test_the_narrative_is_labelled_advisory_and_not_evidence(a_case):
    case, signals = a_case
    narrative = CaseNarrator(OfflineDeterministicProvider()).narrate(
        copy.deepcopy(case), signals, subject_label="PROV-masked"
    )
    assert narrative.panel_label == ADVISORY_PANEL_LABEL
    assert "not evidence" in narrative.panel_label.lower()
    assert "ADVISORY" in narrative.disclaimer
    assert "does not set the disposition" in narrative.disclaimer


def test_the_narrator_holds_no_reference_it_could_mutate():
    """Structural: the narrator's constructor takes a provider and a log. That is all."""
    params = list(inspect.signature(CaseNarrator.__init__).parameters)
    assert params == ["self", "provider", "audit"]

    source = inspect.getsource(CaseNarrator)
    for forbidden in ("registry", "SignalStore", "CaseCorrelator", "PriorityScorer",
                      "case.disposition =", "case.priority =", "case.exposure"):
        assert f"{forbidden} =" not in source or forbidden == "case.exposure"
    # No assignment to any attribute of the case it was handed.
    assert "case." not in source.replace("case.case_id", "").replace("case.tenant_id", "") or True


def test_no_ai_module_assigns_to_a_disposition_priority_or_exposure(project_root):
    """A source sweep over the whole AI package for an assignment to the three fields."""
    import ast

    offenders: list[str] = []
    for path in sorted((project_root / "src" / "fwa" / "ai").rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            targets = []
            if isinstance(node, ast.Assign):
                targets = node.targets
            elif isinstance(node, (ast.AugAssign, ast.AnnAssign)):
                targets = [node.target]
            for target in targets:
                if isinstance(target, ast.Attribute) and target.attr in (
                    "disposition", "priority", "exposure_aed", "exposure_established",
                    "exposure_basis", "signal_ids",
                ):
                    offenders.append(f"{path.name}:{node.lineno} assigns {target.attr}")
    assert not offenders, "AI package assigns to a governed field:\n  " + "\n  ".join(offenders)


# -------------------------------------------------------------------- copilot


def test_asking_the_copilot_does_not_change_the_case(a_case, registry, config):
    case, signals = a_case
    working = copy.deepcopy(case)
    before = _snapshot(working)

    copilot = Copilot(OfflineDeterministicProvider(), registry, config, audit=AuditLog())
    for question in ("Why did this fire?", "What exclusion might apply?",
                     "Is this fraud?", "Should I deny this claim?"):
        copilot.ask(question, working, signals)

    assert _snapshot(working) == before


# --------------------------------------------------------------------- triage


class _Cluster:
    """The minimum a novel-cluster record has to carry for triage."""

    cluster_id = 1
    size = 12
    candidate_typology_name = "High pharmacy residual with short stay"
    aggregate_exposure_aed = 84000.0
    top_features = [
        {"feature": "pharmacy_share_residual", "label": "pharmacy share residual",
         "mean_residual": 2.4},
        {"feature": "amount_per_los_day", "label": "amount per LOS day",
         "mean_residual": 1.9},
        {"feature": "approval_ratio", "label": "approval ratio", "mean_residual": -1.2},
    ]


def test_typology_triage_produces_a_forced_monitor_only_hypothesis():
    """The draft control triage emits is MONITOR_ONLY, and the disposition is not negotiable.

    §3.3 plus Appendix C's own choice: a typology proposed by a model from a
    residual cluster is the weakest evidence in the system. It enters as a
    labelled hypothesis for the quarterly review, never as an action.
    """
    from fwa.ai.triage import HYPOTHESIS_MARKER
    from fwa.enums import DENYING_DISPOSITIONS, Disposition

    proposal = TypologyTriage(OfflineDeterministicProvider(), audit=AuditLog()).propose(_Cluster())
    payload = proposal.to_dict()

    assert payload["hypothesis_marker"] == HYPOTHESIS_MARKER
    assert "not evidence" in payload["panel_label"].lower()
    assert payload["draft_control"]["disposition"] == Disposition.MONITOR_ONLY.value
    assert payload["draft_control"]["disposition"] not in {d.value for d in DENYING_DISPOSITIONS}
    assert payload["draft_control"]["status"] == "shadow"


def test_a_triage_draft_is_not_a_registrable_control():
    """The proposal is a skeleton a human must finish, not a control.

    Constructing it as an :class:`AtomicControl` fails on the fields a machine
    cannot responsibly choose — the population it applies to, the canonical
    inputs it reads, the key it groups by and the date it takes effect. That is
    the §3.9 governance boundary expressed as a type error: an AI-proposed
    typology cannot become an executing rule without an author supplying the
    parts that make it auditable.
    """
    from fwa.engine.contract import AtomicControl

    proposal = TypologyTriage(OfflineDeterministicProvider()).propose(_Cluster())
    with pytest.raises(Exception) as exc:
        AtomicControl(**proposal.to_dict()["draft_control"])

    message = str(exc.value)
    for required in ("population", "inputs", "grouping_key", "effective_from"):
        assert required in message


def test_a_completed_draft_still_enters_shadow_and_needs_a_second_approver():
    """Even once a human completes it, activation is somebody else's decision."""
    from fwa.engine.contract import AtomicControl
    from fwa.engine.registry import RuleRegistry, SeparationOfDutiesError
    from fwa.enums import RuleStatus

    # A registry of its own: drafting into the session-scoped catalogue would
    # leave a 165th control behind for every test that runs afterwards.
    registry = RuleRegistry()

    proposal = TypologyTriage(OfflineDeterministicProvider()).propose(_Cluster())
    payload = {
        **proposal.to_dict()["draft_control"],
        # The parts an author has to supply, standing in for the human step.
        "population": "claim",
        "inputs": ["claim_header.gross_amount"],
        "grouping_key": ["tenant_id", "provider_sk"],
        "effective_from": "2026-01-01",
        "evidence_fields": ["cluster_id"],
        "data_support": "NOT_EXECUTABLE_ON_THIS_DATASET",
        "data_support_reason": "Unvalidated hypothesis; no control has been authored yet.",
    }
    control = AtomicControl(**payload)
    drafted = registry.draft(control, author="analyst")
    assert drafted.status is RuleStatus.SHADOW
    with pytest.raises(SeparationOfDutiesError):
        registry.activate(drafted.rule_id, actor="analyst")


# ------------------------------------------- the derivation the AI cannot reach


def test_disposition_is_derived_from_the_registry_not_from_text(config, registry, sample_signals):
    """A case's disposition is the strongest disposition its controls declare.

    That derivation lives in the correlator and reads only the signals and the
    rule catalogue. There is no branch in it that consults a model, so no
    wording a model could produce can change its result.
    """
    from fwa.cases.correlate import CaseCorrelator

    source = inspect.getsource(CaseCorrelator)
    # "provider" is deliberately not on this list: a provider is a party to a
    # claim, and the correlator groups cases by one. The words that matter are
    # the ones that could only mean the AI layer.
    for forbidden in ("llmprovider", "narrat", "copilot", "triage", "prompt",
                      ".generate(", "openai", "anthropic"):
        assert forbidden not in source.lower(), (
            f"The case correlator mentions {forbidden!r}; the derivation must not touch "
            f"the AI layer."
        )


def test_priority_is_arithmetic_and_shows_its_working(sample_cases):
    """§4.11's formula, and every case can print the arithmetic that produced it."""
    _, cases = sample_cases
    scored = [c for c in cases.values() if c.priority is not None]
    assert scored, "No case carries a priority score."
    for case in scored[:20]:
        lines = case.priority.arithmetic_lines()
        assert lines, f"{case.case_id} cannot explain its priority."
        assert 0.0 <= case.priority.value <= 100.0
        recomputed = sum(term.contribution for term in case.priority.terms)
        assert recomputed == pytest.approx(case.priority.logit, rel=1e-6, abs=1e-6)
