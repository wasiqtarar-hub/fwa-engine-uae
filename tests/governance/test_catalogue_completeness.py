"""The catalogue is complete and honest about itself (build brief §4).

    "Encode **all 39 scenarios and all 164 atomic controls** from Appendix C as
     YAML rule definitions... Each control must declare ``data_support``:
     ``EXECUTABLE``, ``PARTIAL`` or ``NOT_EXECUTABLE_ON_THIS_DATASET`` with a
     reason. **Do not silently drop the controls this dataset cannot support.**"

The interesting half of this file is not the count. It is the set of tests that
make a *dishonest* classification fail: a control that claims to be executable
with no implementation, one that claims to be unrunnable while quietly running,
one whose stated reason says nothing, and one that names no canonical field
that would unlock it. Those are the ways a catalogue can be complete on paper
and wrong in substance.
"""

from __future__ import annotations

import pytest
import yaml

from fwa.engine.controls import CONTROL_IMPLEMENTATIONS
from fwa.enums import DENYING_DISPOSITIONS, ControlType, DataSupport, TYPES_ALLOWED_TO_DENY

pytestmark = pytest.mark.governance

EXPECTED_SCENARIOS = 39
EXPECTED_CONTROLS = 164

FAMILIES = ("ENT", "PAY", "CLN", "PHR", "DOC", "NET", "ANL", "POL")


# ------------------------------------------------------------------ the count


def test_all_thirty_nine_scenarios_are_encoded(registry):
    assert len(registry.scenarios()) == EXPECTED_SCENARIOS


def test_all_one_hundred_and_sixty_four_controls_are_encoded(registry):
    assert len(registry) == EXPECTED_CONTROLS


def test_every_family_is_represented(registry):
    present = {c.rule_id.split("-")[0] for c in registry.all()}
    assert present == set(FAMILIES)


def test_nothing_failed_to_load(registry):
    assert not registry.load_errors, registry.load_errors


def test_rule_ids_are_unique_and_well_formed(registry):
    ids = [c.rule_id for c in registry.all()]
    assert len(ids) == len(set(ids))
    for rule_id in ids:
        family, scenario, control = rule_id.split("-")
        assert family in FAMILIES
        assert scenario.isdigit() and len(scenario) == 2
        assert control.startswith("R") and control[1:].isdigit()


def test_every_control_belongs_to_a_declared_scenario(registry):
    scenarios = set(registry.scenarios())
    for control in registry.all():
        assert control.scenario_id in scenarios


# ------------------------------------------------------------------- honesty


def test_every_control_declares_a_data_support_classification(registry):
    for control in registry.all():
        assert isinstance(control.data_support, DataSupport)


def test_every_classification_carries_a_substantive_reason(registry):
    thin = []
    for control in registry.all():
        reason = (control.data_support_reason or "").strip()
        if len(reason) < 25:
            thin.append(f"{control.rule_id} → {reason!r}")
    assert not thin, "Data-support reasons that say nothing:\n  " + "\n  ".join(thin)


def test_a_control_claiming_to_be_unrunnable_has_no_implementation(registry):
    """The classification and the code must agree, in this direction..."""
    for control in registry.all():
        if control.data_support is DataSupport.NOT_EXECUTABLE_ON_THIS_DATASET:
            assert not control.implementation, (
                f"{control.rule_id} is classified NOT_EXECUTABLE_ON_THIS_DATASET but names the "
                f"implementation {control.implementation!r}."
            )


def test_a_control_claiming_to_be_runnable_names_a_real_implementation(registry):
    """...and in the other. A named function that does not exist is a silent drop."""
    missing = []
    for control in registry.all():
        if control.data_support is DataSupport.NOT_EXECUTABLE_ON_THIS_DATASET:
            continue
        if not control.implementation:
            missing.append(f"{control.rule_id} names no implementation")
        elif control.implementation not in CONTROL_IMPLEMENTATIONS:
            missing.append(f"{control.rule_id} → {control.implementation} is not registered")
    assert not missing, "\n  ".join(missing)


def test_an_unrunnable_control_names_what_would_unlock_it(registry):
    """§4: say *what data* is missing, not merely that something is.

    This is the difference between a catalogue that documents a gap and one
    that excuses it. A reader should be able to take the named canonical fields
    to a payer and ask for them.
    """
    silent = []
    for control in registry.all():
        if control.data_support is not DataSupport.NOT_EXECUTABLE_ON_THIS_DATASET:
            continue
        if not control.required_canonical_fields:
            silent.append(control.rule_id)
    assert not silent, (
        "Unrunnable controls that do not name the canonical fields they need:\n  "
        + "\n  ".join(silent)
    )


def test_the_executable_share_is_reported_not_inflated(registry):
    """A sanity bound on the headline number, in both directions.

    Too high would mean controls are claiming data this file does not have;
    too low would mean the build gave up on controls it could run. The band is
    wide on purpose — the test is here to catch a change that moves the number
    without anyone noticing, not to pin it.
    """
    summary = registry.summary()
    runnable = (summary["by_data_support"].get("EXECUTABLE", 0)
                + summary["by_data_support"].get("PARTIAL", 0))
    assert 25 <= runnable <= 60, summary["by_data_support"]
    assert sum(summary["by_data_support"].values()) == EXPECTED_CONTROLS


def test_every_implemented_control_is_reachable_from_the_registry(registry):
    """No orphan implementations: code that no catalogue entry can call."""
    declared = {c.implementation for c in registry.all() if c.implementation}
    orphans = set(CONTROL_IMPLEMENTATIONS) - declared
    assert not orphans, f"Implementations no control declares: {sorted(orphans)}"


# ------------------------------------------------------------- the §3.7 shape


def test_every_control_satisfies_the_atomic_contract(registry):
    """§3.7's fields, on all 164 — including the ones that never run."""
    for control in registry.all():
        assert control.name
        assert control.version
        assert control.type
        assert control.stage
        assert control.population
        assert control.inputs
        assert control.expression
        assert control.exclusions, f"{control.rule_id} declares no exclusions"
        assert control.grouping_key
        assert control.reason_code
        assert control.evidence_fields
        assert control.owner
        assert control.effective_from


def test_every_control_declares_at_least_one_exclusion(registry):
    """§3.7: the exclusion list is where a control says what would clear it.

    An empty list is the assertion that nothing could legitimately explain the
    pattern, which is never true of a payment-integrity control.
    """
    for control in registry.all():
        assert len(control.exclusions) >= 1
        assert all(e.strip() for e in control.exclusions)


def test_every_denying_control_is_hard_or_expert(registry):
    """The §3.3 boundary, restated over the whole catalogue."""
    for control in registry.all():
        if control.disposition in DENYING_DISPOSITIONS:
            assert all(t in TYPES_ALLOWED_TO_DENY for t in control.type), (
                f"{control.rule_id} ({control.type_label}) may deny"
            )


def test_no_model_control_may_deny(registry):
    for control in registry.by_type(ControlType.M):
        assert control.disposition not in DENYING_DISPOSITIONS


def test_every_control_starts_in_shadow(registry):
    """§4.8/§6.3: nothing ships active. Activation is a governed, audited act."""
    summary = registry.summary()
    assert summary["by_status"]["active"] == 0
    assert summary["by_status"]["shadow"] == EXPECTED_CONTROLS


# ------------------------------------------------ the YAML is the source text


def test_the_yaml_files_round_trip(project_root, registry):
    """What is on disk is what the registry holds — no code-side patching."""
    on_disk: dict[str, dict] = {}
    for path in sorted((project_root / "rules").rglob("*.yaml")):
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        for entry in raw.get("controls", []) or []:
            on_disk[entry["rule_id"]] = entry

    assert len(on_disk) == EXPECTED_CONTROLS
    for control in registry.all():
        entry = on_disk[control.rule_id]
        assert entry["version"] == control.version
        assert entry["disposition"] == control.disposition.value
        assert entry["data_support"] == control.data_support.value


def test_every_control_has_a_stable_fingerprint(registry):
    """Signals store ``rule_id@version``; the fingerprint is what makes that mean something."""
    fingerprints = {c.rule_id: c.fingerprint() for c in registry.all()}
    assert len(set(fingerprints.values())) == EXPECTED_CONTROLS, "Two controls hash identically."
    for control in registry.all():
        assert control.fingerprint() == fingerprints[control.rule_id]


def test_coverage_rows_cover_every_control(registry):
    rows = registry.coverage_rows()
    assert len(rows) == EXPECTED_CONTROLS
    for row in rows:
        assert row["data_support_reason"]
        assert row["rule_fingerprint"]
