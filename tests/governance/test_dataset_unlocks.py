"""Dataset unlocks cannot weaken the catalogue, the contract or the boundary.

An unlock (``rules/unlocks/*.yaml``) lets a control that the claim-header
extract cannot support run on a file that does carry the tables it needs. That
is a new route to running code, so it gets the same scrutiny as the catalogue:

* every declaration is well formed, names a real control and a real
  implementation, and every unlocked implementation is declared (no orphans);
* an unlock never alters the registered control — type, disposition, score,
  status and fingerprint are the registry's, so the §3.3 legality check that
  ran at registration still governs every signal an unlocked control raises;
* on a file that does not populate the named tables the unlock is inert: the
  effective classification equals the static one, and the static count the
  catalogue test pins (25–60 runnable) is untouched;
* the answer key of a synthetic dataset is never read by detection code.
"""

from __future__ import annotations

import ast
import re

import pytest
import yaml

from fwa.canonical import CANONICAL_TABLES
from fwa.canonical.supplementary import SUPPLEMENTARY_TABLES
from fwa.engine.unlocked import UNLOCKED_IMPLEMENTATIONS
from fwa.engine.unlocks import DatasetUnlock, decide, load_unlocks, runtime_view
from fwa.enums import DENYING_DISPOSITIONS, DataSupport, TYPES_ALLOWED_TO_DENY

pytestmark = pytest.mark.governance

KNOWN_TABLES = set(CANONICAL_TABLES) | set(SUPPLEMENTARY_TABLES)


def test_every_unlock_names_a_catalogue_control(registry):
    for rule_id in registry.unlocks:
        assert rule_id in registry, rule_id


def test_every_unlock_names_a_registered_implementation(registry):
    missing = [f"{u.rule_id} -> {u.implementation}" for u in registry.unlocks.values()
               if u.implementation not in UNLOCKED_IMPLEMENTATIONS]
    assert not missing, "Unlocks naming no implementation:\n  " + "\n  ".join(missing)


def test_no_unlocked_implementation_is_an_orphan(registry):
    declared = {u.implementation for u in registry.unlocks.values()}
    orphans = sorted(set(UNLOCKED_IMPLEMENTATIONS) - declared)
    assert not orphans, f"Unlocked implementations no declaration reaches: {orphans}"


def test_unlocks_only_name_known_tables(registry):
    for u in registry.unlocks.values():
        unknown = [t for t in u.requires_tables if t not in KNOWN_TABLES]
        assert not unknown, f"{u.rule_id} requires unknown table(s) {unknown}"


def test_an_unlock_applies_only_where_the_catalogue_allows_it(registry):
    """NOT_EXECUTABLE controls may be unlocked; PARTIAL ones only by an explicit upgrade."""
    for u in registry.unlocks.values():
        control = registry.get(u.rule_id)
        if control.data_support is DataSupport.EXECUTABLE:
            pytest.fail(f"{u.rule_id} is already EXECUTABLE; an unlock has nothing to add.")
        if control.data_support is DataSupport.PARTIAL:
            assert u.upgrades_partial, f"{u.rule_id}: PARTIAL control unlocked without upgrades_partial"


def test_the_static_catalogue_is_unchanged_by_unlocks(registry):
    """The YAML classification — and the headline count the catalogue test pins — stands."""
    summary = registry.summary()["by_data_support"]
    runnable = summary.get("EXECUTABLE", 0) + summary.get("PARTIAL", 0)
    assert 25 <= runnable <= 60
    for u in registry.unlocks.values():
        control = registry.get(u.rule_id)
        if control.data_support is DataSupport.NOT_EXECUTABLE_ON_THIS_DATASET:
            assert not control.implementation


def test_a_runtime_view_keeps_every_safety_relevant_field(registry):
    class _Full:
        def is_populated(self, name):
            return True

        def get(self, name):
            import pandas as pd

            return pd.DataFrame({"x": [1]})

    for u in registry.unlocks.values():
        control = registry.get(u.rule_id)
        view = runtime_view(control, decide(control, u, _Full()))
        assert view.type == control.type
        assert view.disposition == control.disposition
        assert view.score == control.score
        assert view.status == control.status
        assert view.fingerprint() == control.fingerprint()
        if view.disposition in DENYING_DISPOSITIONS:
            assert all(t in TYPES_ALLOWED_TO_DENY for t in view.type)


def test_unlocks_are_inert_on_a_claim_header_file(registry, context):
    """On the claim-header sample nothing is unlocked: effective == static for all 164."""
    for control in registry.all():
        decision = decide(control, registry.unlocks.get(control.rule_id), context.dataset)
        assert decision.effective_support == control.data_support, control.rule_id
        assert decision.basis == "catalogue", control.rule_id


def test_an_unlock_needs_every_named_table_with_rows():
    import pandas as pd

    from fwa.canonical import CanonicalDataset

    unlock = DatasetUnlock(
        rule_id="PAY-04-R01", implementation="x", requires_tables=["claim_line", "authorization"],
        requires_columns=["claim_line.authorization_id"], data_support="EXECUTABLE",
        reason="Fixture: runs when claim lines and approvals are both present.",
    )
    ds = CanonicalDataset.empty()
    assert unlock.missing(ds) == ["claim_line", "authorization"]
    ds.set("claim_line", pd.DataFrame({"authorization_id": [None]}), "POPULATED", "fixture")
    ds.set("authorization", pd.DataFrame({"authorization_sk": ["A1"]}), "POPULATED", "fixture")
    assert unlock.missing(ds) == ["claim_line.authorization_id"]
    ds.set("claim_line", pd.DataFrame({"authorization_id": ["A1"]}), "POPULATED", "fixture")
    assert unlock.satisfied_by(ds)


@pytest.mark.parametrize("bad", [
    {"data_support": "NOT_EXECUTABLE_ON_THIS_DATASET"},
    {"requires_tables": []},
    {"reason": "short"},
    {"requires_columns": ["remittance.decision"]},
])
def test_a_malformed_unlock_is_refused(bad):
    base = dict(rule_id="PAY-04-R01", implementation="x", requires_tables=["claim_line"],
                data_support="EXECUTABLE", reason="A substantive reason of adequate length.")
    base.update(bad)
    with pytest.raises(Exception):
        DatasetUnlock(**base)


def test_every_unlock_file_loads(project_root):
    unlocks = load_unlocks(project_root / "rules")
    for path in (project_root / "rules" / "unlocks").glob("*.yaml"):
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        for entry in raw.get("unlocks", []) or []:
            assert entry["rule_id"] in unlocks


# --------------------------------------------------------------- answer key

_ANSWER_KEY = re.compile(r"strata|evaluation_labels")


def test_no_detection_module_reads_an_answer_key(project_root):
    """Synthetic answer keys (``*.strata.*``) and held-out labels stay out of detection.

    Only the adapters' ``held_out_labels`` (evaluation input) and ``fwa.evaluation``
    may name the held-out label file; nothing under ``src/`` may name a strata file.
    """
    offenders = []
    for path in sorted((project_root / "src" / "fwa").rglob("*.py")):
        rel = path.relative_to(project_root / "src" / "fwa").as_posix()
        if rel.startswith("evaluation/"):
            continue
        text = path.read_text(encoding="utf-8")
        tree = ast.parse(text)
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                if "strata" in node.value.lower() and ".strata" in node.value.lower():
                    offenders.append(f"{rel}: names a strata file ({node.value[:60]!r})")
                if "evaluation_labels" in node.value and not rel.startswith("canonical/"):
                    offenders.append(f"{rel}: names the held-out label file")
    assert not offenders, "\n".join(offenders)
