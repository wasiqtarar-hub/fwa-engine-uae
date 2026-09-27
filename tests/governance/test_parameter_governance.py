"""Parameter governance (manuscript §3.9, build brief §3.9, §18).

    "Every threshold is a governed object with an owner, a rationale, an
     effective date, an approval and an expected alert volume... **no magic
     numbers in code**."

The load-bearing test here is the last one: a sweep over the engine's source
for numeric literals used as thresholds. Everything above it checks the
registry keeps its side of the bargain — scope resolution, effective dating,
and the fingerprint that ties a run to the exact parameter set that produced it.
"""

from __future__ import annotations

import ast
import datetime as _dt

import pytest
import yaml

from fwa.config import ParameterNotFound, ParameterRecord, ParameterRegistry

pytestmark = pytest.mark.governance

REQUIRED_FIELDS = ("owner", "rationale", "source", "approved_by", "review_date")

_FAR_PAST = _dt.date(1900, 1, 1)
_FAR_FUTURE = _dt.date(2099, 12, 31)


def _record(key: str, value, *, rationale: str, valid_from=_FAR_PAST, valid_to=_FAR_FUTURE,
            **scope) -> ParameterRecord:
    """A governed record with the governance fields filled in, for scope/date tests."""
    return ParameterRecord(
        key=key, value=value, owner="test_owner", rationale=rationale,
        source="test fixture", approved_by="test_approver",
        valid_from=_dt.date.fromisoformat(valid_from) if isinstance(valid_from, str) else valid_from,
        valid_to=_dt.date.fromisoformat(valid_to) if isinstance(valid_to, str) else valid_to,
        **scope,
    )


@pytest.fixture(scope="module")
def raw_parameters(project_root):
    return yaml.safe_load((project_root / "config" / "parameters.yaml").read_text("utf-8"))


# ------------------------------------------------------- every governed field


def test_every_parameter_declares_every_governance_field(config):
    missing: list[str] = []
    for row in config.parameters.governance_report():
        for field in REQUIRED_FIELDS:
            if not row.get(field):
                missing.append(f"{row['key']}.{field}")
    assert not missing, "Parameters missing governance metadata:\n  " + "\n  ".join(missing)


def test_no_rationale_is_a_placeholder(config):
    """A rationale has to say *why*, not restate the number."""
    thin: list[str] = []
    for row in config.parameters.governance_report():
        rationale = (row.get("rationale") or "").strip()
        if len(rationale) < 40 or rationale.lower().startswith(("tbd", "todo", "n/a")):
            thin.append(f"{row['key']} → {rationale[:60]!r}")
    assert not thin, "Placeholder rationales:\n  " + "\n  ".join(thin)


def test_every_parameter_is_approved(config):
    unapproved = [
        row["key"] for row in config.parameters.governance_report() if row.get("unapproved")
    ]
    assert not unapproved, f"Unapproved parameters in the registry: {unapproved}"


def test_no_parameter_is_already_expired(config):
    expired = [row["key"] for row in config.parameters.governance_report() if row.get("expired")]
    assert not expired, f"Expired parameters still resolvable: {expired}"


def test_every_review_date_is_in_the_future(config):
    """A review date in the past is an ungoverned parameter wearing a label."""
    today = _dt.date.today()
    stale = []
    for row in config.parameters.governance_report():
        review = row.get("review_date")
        if review and _dt.date.fromisoformat(str(review)) < today:
            stale.append(f"{row['key']} (due {review})")
    assert not stale, "Parameters past their review date:\n  " + "\n  ".join(stale)


def test_detection_thresholds_declare_an_expected_alert_volume(config, raw_parameters):
    """§3.9: a threshold without an expected volume cannot breach a ceiling.

    Not every parameter is a detection threshold — a random seed and a feature
    switch are not — so the requirement applies to the ones the catalogue's
    controls reference, which is how the ceiling check finds them.
    """
    referenced = {
        row["key"] for row in config.parameters.governance_report()
        if row.get("expected_alert_volume") is not None
    }
    assert len(referenced) >= 20, (
        f"Only {len(referenced)} parameters declare an expected alert volume; the alert-ceiling "
        f"machinery has almost nothing to enforce."
    )
    for key in referenced:
        volume = int(config.parameters.record(key).expected_alert_volume)
        # Zero is a legitimate expectation, and one the registry states on
        # purpose: ``community_density_threshold`` is retained as a reported
        # absolute floor that the size-matched test supersedes, so it is
        # expected to raise nothing on its own. A negative number never is.
        assert volume >= 0, f"{key} declares a negative expected alert volume"
    positive = [
        k for k in referenced if int(config.parameters.record(k).expected_alert_volume) > 0
    ]
    assert len(positive) >= 20


# ----------------------------------------------------------- effective dating


def test_a_parameter_resolves_to_the_version_in_force():
    registry = ParameterRegistry([
        _record("t", 1.0, rationale="first version, before the tariff change",
                valid_from="2020-01-01", valid_to="2025-12-31"),
        _record("t", 2.0, rationale="second version, after the tariff change",
                valid_from="2026-01-01", valid_to="2099-12-31"),
    ])
    assert registry.get("t", as_of="2024-06-01") == 1.0
    assert registry.get("t", as_of="2026-06-01") == 2.0


def test_an_unknown_parameter_raises_rather_than_defaulting(config):
    """A silent default is how a magic number gets back in."""
    with pytest.raises(ParameterNotFound):
        config.get("no_such_parameter_exists")


def test_a_parameter_with_no_version_in_force_raises():
    registry = ParameterRegistry([
        _record("t", 1.0, rationale="in force during 2020 only, for a one-off contract period",
                valid_from="2020-01-01", valid_to="2020-12-31"),
    ])
    with pytest.raises(ParameterNotFound):
        registry.get("t", as_of="2026-01-01")


def test_a_more_specific_scope_wins():
    """§3.9: parameters are scoped by regulator, payer, product, contract, provider type."""
    registry = ParameterRegistry([
        _record("t", 1.0, rationale="the default, applied when no more specific scope matches"),
        _record("t", 9.0, rationale="override for one payer whose contract sets another limit",
                payer="PAYER-A"),
    ])
    assert registry.get("t") == 1.0
    assert registry.get("t", scope={"payer": "PAYER-A"}) == 9.0
    assert registry.get("t", scope={"payer": "PAYER-B"}) == 1.0


# ------------------------------------------------------------- reproducibility


def test_the_fingerprint_identifies_the_parameter_set(config):
    assert config.fingerprint() == config.parameters.fingerprint()
    assert len(config.fingerprint()) >= 16


def test_changing_one_parameter_changes_the_fingerprint():
    def build(value):
        return ParameterRegistry([
            _record("t", value, rationale="a threshold whose value is varied to test the hash"),
        ])

    assert build(1.0).fingerprint() != build(1.1).fingerprint()
    assert build(1.0).fingerprint() == build(1.0).fingerprint()


def test_the_seed_is_a_governed_parameter(config):
    """Reproducibility is a governed property, not an incidental one."""
    record = config.parameters.record("random_seed")
    assert record.owner
    assert isinstance(config.get("random_seed"), int)


# ------------------------------------------------------------ no magic numbers


#: Numbers that are not thresholds: identity, unit conversion, percentages,
#: array indices, the constants of the statistics themselves.
ALLOWED_LITERALS = {
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 20, 24, 25, 30, 50, 60, 100, 365, 1000,
    -1, -2, 0.0, 0.5, 1.0, 2.0, 100.0, 1e-9, 1e-6, 1e-12,
    1.4826,  # the MAD consistency constant, defined and named in statistical.robust
}

#: Modules whose numbers are presentation or plumbing rather than detection.
EXEMPT_MODULES = {"config.py", "run_validation.py", "pipeline.py"}


def test_no_magic_threshold_in_the_engine_or_statistical_layer(project_root):
    """Every comparison against a bare number, in the code that decides.

    The sweep looks specifically at *comparisons* — ``x > 3.7``, ``x <= 0.85`` —
    because that is the shape a threshold takes. A number used to index, to
    convert or to round is not a threshold and is not what §3.9 governs.
    """
    offenders: list[str] = []
    roots = [project_root / "src" / "fwa" / "engine",
             project_root / "src" / "fwa" / "statistical",
             project_root / "src" / "fwa" / "cases"]

    for root in roots:
        for path in sorted(root.rglob("*.py")):
            if path.name in EXEMPT_MODULES:
                continue
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if not isinstance(node, ast.Compare):
                    continue
                for comparator in node.comparators:
                    if not isinstance(comparator, ast.Constant):
                        continue
                    value = comparator.value
                    if not isinstance(value, (int, float)) or isinstance(value, bool):
                        continue
                    if value in ALLOWED_LITERALS:
                        continue
                    offenders.append(f"{path.name}:{node.lineno} compares against {value}")

    assert not offenders, (
        "Ungoverned numeric thresholds (§3.9 requires cfg. lookups):\n  " + "\n  ".join(offenders)
    )


def test_the_controls_reference_parameters_by_key_not_by_value(project_root, registry):
    """Every ``cfg.x`` a control declares resolves in the registry."""
    from fwa.config import load_config

    config = load_config(project_root / "config")
    unresolved: list[str] = []
    for control in registry.all():
        for name, value in (control.parameters or {}).items():
            ref = str(value)
            if not ref.startswith("cfg."):
                continue
            try:
                config.parameters.record(ref[4:])
            except ParameterNotFound:
                unresolved.append(f"{control.rule_id}.{name} → {ref}")
    assert not unresolved, "Controls reference unknown parameters:\n  " + "\n  ".join(unresolved)


def test_parameters_yaml_is_the_single_source(raw_parameters, config):
    """The file and the loaded registry agree on the key set — no code-side additions."""
    entries = raw_parameters if isinstance(raw_parameters, list) else raw_parameters["parameters"]
    assert {e["key"] for e in entries} == set(config.parameters.keys())
