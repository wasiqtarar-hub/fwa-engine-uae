"""Unlocked POL controls (src/fwa/engine/unlocked/pol.py) on a small hand-built fixture.

Reuses the clean mini-world of ``test_unlocked_ent`` and adds the enrolment
tables (employer rosters, policy events, applications, prior cover). Each
control must fire on its planted pattern, stay silent on the clean world and on
the declared look-alike exclusions, return nothing on an empty dataset, and be
idempotent.
"""

from __future__ import annotations

import copy
import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from fwa.config import load_config  # noqa: E402
from fwa.engine.registry import RuleRegistry  # noqa: E402
from fwa.engine.unlocked import pol  # noqa: E402
from fwa.engine.unlocks import decide, runtime_view  # noqa: E402

_spec = importlib.util.spec_from_file_location("_ent_world", Path(__file__).with_name("test_unlocked_ent.py"))
_ew = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_ew)

T = _ew.T
D = pd.Timestamp


@pytest.fixture(scope="module")
def cfg():
    return load_config(ROOT / "config")


@pytest.fixture(scope="module")
def reg():
    return RuleRegistry.from_directory(ROOT / "rules")


def _add_enrolment(w, member, employer, effective="2024-01-01", recorded="2023-12-28 10:00", agent="A1",
                   event_type="ADD_MEMBER", application="2023-12-15", declared="NONE", prior_cover=True):
    w.t.setdefault("employer_roster", []).append({
        "employer_id": employer, "member_sk": member, "sponsor_id": None, "relationship": "EMPLOYEE",
        "valid_from": D("2024-01-01"), "valid_to": D("2025-12-31"), "tenant_id": T})
    w.t.setdefault("policy_event", []).append({
        "policy_event_sk": f"PE-{member}", "coverage_id": f"CV-{member}", "event_type": event_type,
        "actor": "hr_portal", "event_time": D(recorded), "effective_date": D(effective), "agent_id": agent,
        "tenant_id": T})
    w.t.setdefault("policy_application", []).append({
        "application_sk": f"APP-{member}", "member_sk": member, "coverage_id": f"CV-{member}",
        "application_date": D(application), "declared_conditions": declared, "declared_prior_cover": prior_cover,
        "agent_id": agent, "tenant_id": T})


def _pol_base():
    w = _ew._base()
    for m in list(w.t["member"]):
        _add_enrolment(w, m["member_sk"], m["employer_id"])
    # a clean member whose declarations match the prior history on record
    w.member("M062", dob="1975-01-01", sex="F", employer="E1")
    _add_enrolment(w, "M062", "E1", application="2024-12-10", declared="E11", prior_cover=True)
    w.t["prior_coverage_history"] = [{"member_sk": "M062", "prior_payer": "OLDINS", "valid_from": D("2020-01-01"),
                                      "valid_to": D("2024-11-30"), "conditions_treated": "E11.9", "tenant_id": T}]
    return w


@pytest.fixture(scope="module")
def base_world():
    return _pol_base()


def _run(reg, cfg, rule_id, frames):
    ctx = _ew._ctx(cfg, frames)
    control = reg.get(rule_id)
    view = runtime_view(control, decide(control, reg.unlocks.get(rule_id), ctx.dataset))
    impl = pol.IMPLEMENTATIONS[reg.unlocks[rule_id].implementation]
    return impl.__wrapped__(ctx, view)


def _drop(w, table, member):
    w.t[table] = [r for r in w.t[table] if r.get("member_sk") != member and r.get("coverage_id") != f"CV-{member}"]


# ---------------------------------------------------------------------------
# plants
# ---------------------------------------------------------------------------


def plant_pol_01_r01(w):
    _drop(w, "employer_roster", "M010")
    return "member", "M010"


def plant_pol_01_r02(w):
    for e in w.t["policy_event"]:
        if e["coverage_id"] == "CV-M011":
            e["event_time"] = D("2025-06-01 16:40")
    return "member", "M011"


def plant_pol_01_r03(w):
    w.member("M070", dob="1970-05-05", sex="M", employer="E2")
    _add_enrolment(w, "M070", "E2", application="2024-12-15", declared="NONE", prior_cover=False)
    w.t["prior_coverage_history"].append({"member_sk": "M070", "prior_payer": "OLDINS", "valid_from": D("2020-01-01"),
                                          "valid_to": D("2024-11-30"), "conditions_treated": "E11.9;I10",
                                          "tenant_id": T})
    return "member", "M070"


def plant_pol_01_r05(w, event_type="ADD_MEMBER"):
    for p in ("P15", "P16"):
        w.t["provider"].append(_ew._provider(p, "CLINIC", "DUBAI", bank_account_token="BK-SHARED"))
        w.t["clinician_roster"].append(_ew._clin(f"D{p}", p))
    for k in range(10):
        m = f"M{80 + k:03d}"
        w.member(m, dob="1990-01-01", sex="M", employer="E7")
        _add_enrolment(w, m, "E7", effective="2025-03-01", recorded="2025-02-25 09:00", agent="A9",
                       event_type=event_type)
        p = "P15" if k % 2 else "P16"
        w.claim(m, p, D("2025-03-05") + pd.Timedelta(days=k * 2), [("99213", 230.0, 1.0)], clinician=f"D{p}",
                start_hour=9 + k % 5)
    return "employer", "E7"


PLANTS = {"POL-01-R01": plant_pol_01_r01, "POL-01-R02": plant_pol_01_r02, "POL-01-R03": plant_pol_01_r03,
          "POL-01-R05": plant_pol_01_r05}


# ---------------------------------------------------------------------------
# tests
# ---------------------------------------------------------------------------


def test_unlock_yaml_and_implementations_match(reg):
    raw = yaml.safe_load((ROOT / "rules" / "unlocks" / "POL.yaml").read_text("utf-8"))
    declared = {e["implementation"] for e in raw["unlocks"]}
    assert declared == set(pol.IMPLEMENTATIONS)
    rule_ids = {e["rule_id"] for e in raw["unlocks"]}
    assert rule_ids == set(PLANTS)
    for rid in rule_ids:
        assert reg.get(rid).data_support.value == "NOT_EXECUTABLE_ON_THIS_DATASET"
    assert "POL-01-R04" not in rule_ids


@pytest.mark.parametrize("rule_id", sorted(PLANTS))
def test_fires_on_planted_pattern(reg, cfg, base_world, rule_id):
    w = copy.deepcopy(base_world)
    subject_type, subject_id = PLANTS[rule_id](w)
    _ew._assert_fires(_run(reg, cfg, rule_id, w.frames()), subject_type, subject_id)


@pytest.mark.parametrize("rule_id", sorted(PLANTS))
def test_silent_on_clean_world(reg, cfg, base_world, rule_id):
    assert _run(reg, cfg, rule_id, base_world.frames()) == []


@pytest.mark.parametrize("rule_id", sorted(PLANTS))
def test_empty_dataset_returns_nothing(reg, cfg, rule_id):
    ctx = _ew._ctx(cfg, {})
    control = reg.get(rule_id)
    impl = pol.IMPLEMENTATIONS[reg.unlocks[rule_id].implementation]
    assert impl(ctx, control) == []
    assert impl.__wrapped__(ctx, control) == []


@pytest.mark.parametrize("rule_id", sorted(PLANTS))
def test_idempotent(reg, cfg, base_world, rule_id):
    w = copy.deepcopy(base_world)
    PLANTS[rule_id](w)
    frames = w.frames()
    a = sorted(s.signal_id for s in _run(reg, cfg, rule_id, frames))
    b = sorted(s.signal_id for s in _run(reg, cfg, rule_id, frames))
    assert a and a == b


def test_duplicate_active_identity_limb(reg, cfg, base_world):
    w = copy.deepcopy(base_world)
    w.member("M063", dob="1980-03-15", sex="M", token="TK-M020", employer="E2")
    _add_enrolment(w, "M063", "E2")
    signals = _run(reg, cfg, "POL-01-R01", w.frames())
    _ew._assert_fires(signals, "member", "M063")
    assert any(s.evidence.get("limb_fired") == "duplicate_active_identity" for s in signals)


def test_exclusions_hold(reg, cfg, base_world):
    # POL-01-R01: a newborn not yet on the roster
    w = copy.deepcopy(base_world)
    w.member("M064", dob="2025-05-01", sex="F", employer="E1")
    w.claim("M064", "P01", "2025-05-10", [("99213", 230.0, 1.0)], clinician="DP01", start_hour=18)
    assert _run(reg, cfg, "POL-01-R01", w.frames()) == []
    # POL-01-R02: an approved retroactive correction
    w = copy.deepcopy(base_world)
    for e in w.t["policy_event"]:
        if e["coverage_id"] == "CV-M011":
            e["event_time"] = D("2025-06-01 16:40")
            e["event_type"] = "RETRO_CORRECTION_APPROVED"
    assert _run(reg, cfg, "POL-01-R02", w.frames()) == []
    # POL-01-R03: the same conflict outside the contestability window
    w = copy.deepcopy(base_world)
    plant_pol_01_r03(w)
    w.t["policy_application"][-1]["application_date"] = D("2023-06-01")
    assert _run(reg, cfg, "POL-01-R03", w.frames()) == []
    # POL-01-R05: an enrolment campaign
    w = copy.deepcopy(base_world)
    plant_pol_01_r05(w, event_type="ADD_CAMPAIGN")
    assert _run(reg, cfg, "POL-01-R05", w.frames()) == []
