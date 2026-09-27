"""Unlocked PAY-01 to PAY-06 controls (src/fwa/engine/unlocked/pay_a.py).

Every control is exercised on a minimal hand-built multi-table fixture: it
fires on the planted pattern with the expected subject and a readable
plain-language sentence, stays silent on the clean variant of the same
fixture, returns [] when its tables are empty, and is idempotent.
"""

from __future__ import annotations

import datetime as _dt
from pathlib import Path

import pandas as pd
import pytest
import yaml

from fwa.canonical import CanonicalDataset
from fwa.engine.context import ControlContext
from fwa.engine.unlocked import pay_a

ROOT = Path(__file__).resolve().parents[2]
TENANT = "T001"


# ---------------------------------------------------------------------------
# fixture builder
# ---------------------------------------------------------------------------

CODES = [
    # code, type, family, sex, age_min, age_max, minutes, timed, desc
    ("S100", "procedure", "surgery", None, None, None, None, False, "knee arthroscopy"),
    ("S101", "procedure", "surgery", None, None, None, None, False, "diagnostic arthroscopy"),
    ("S200", "procedure", "endoscopy", None, None, None, None, False, "colonoscopy"),
    ("S201", "procedure", "endoscopy", None, None, None, None, False, "biopsy"),
    ("S202", "procedure", "endoscopy", None, None, None, None, False, "polyp removal"),
    ("PKG1", "package", "maternity", None, None, None, None, False, "normal delivery package"),
    ("C1", "procedure", "maternity", None, None, None, None, False, "delivery room"),
    ("C2", "procedure", "maternity", None, None, None, None, False, "newborn check"),
    ("T15", "procedure", "therapy", None, None, None, 15, True, "physiotherapy per 15 minutes"),
    ("OB1", "procedure", "obstetrics", "F", None, None, None, False, "obstetric ultrasound"),
    ("PED1", "procedure", "paediatrics", None, 0, 12, None, False, "child vaccination"),
    ("MRI", "imaging", "imaging", None, None, None, None, False, "MRI scan"),
    ("CT", "imaging", "imaging", None, None, None, None, False, "CT scan"),
    ("U1", "procedure", "injection", None, None, None, None, False, "joint injection"),
    ("OLD1", "procedure", "surgery", None, None, None, None, False, "retired procedure"),
    ("L1", "procedure", "laboratory", None, None, None, None, False, "blood count"),
    ("L2", "procedure", "laboratory", None, None, None, None, False, "ferritin"),
    ("CONS", "procedure", "consultation", None, None, None, None, False, "consultation"),
]


def _code_ref() -> list[dict]:
    return [dict(activity_code=c, activity_type=t, service_family=f, sex_restriction=s,
                 age_min=a0, age_max=a1, minutes=m, is_time_based=tb, description=d)
            for c, t, f, s, a0, a1, m, tb, d in CODES]


def _ctx(config, claims: list[dict], lines: list[dict], **tables) -> ControlContext:
    ds = CanonicalDataset.empty(source_system="fixture", tenant_id=TENANT)
    types = {c[0]: c[1] for c in CODES}
    hdr = pd.DataFrame([{
        "tenant_id": TENANT, "payer_id": "PAY1", "claim_type": "outpatient",
        "gross_amount": None, "cross_payer_match_token": None, "submission_date": None,
        "length_of_stay_days": 0, **c} for c in claims])
    hdr["service_date"] = pd.to_datetime(hdr["service_date"])
    hdr["claim_date"] = hdr["service_date"]
    ln = pd.DataFrame([{"tenant_id": TENANT, "units": 1, "indicator": None, "authorization_id": None,
                        "activity_type": types.get(line.get("activity_code")), **line} for line in lines])
    if not ln.empty:
        if "net_amount" not in ln.columns:
            ln["net_amount"] = ln["gross_amount"]
        if "service_date" not in ln.columns:
            ln["service_date"] = ln["claim_sk"].map(dict(zip(hdr["claim_sk"], hdr["service_date"])))
        # gross on the header = sum of its lines
        sums = ln.groupby("claim_sk")["gross_amount"].sum()
        hdr["gross_amount_aed"] = hdr["claim_sk"].map(sums).fillna(hdr.get("gross_amount_aed", 0.0))
    ds.set("claim_header", hdr, "POPULATED", "fixture")
    if not ln.empty:
        ds.set("claim_line", ln, "POPULATED", "fixture")
    tables.setdefault("activity_code_reference", _code_ref())
    for name, rows in tables.items():
        frame = pd.DataFrame(rows)
        if not frame.empty:
            ds.set(name, frame, "POPULATED", "fixture")
    return ControlContext(config=config, tenant_id=TENANT, claims=hdr, features=None, dataset=ds,
                          run_date=_dt.date(2026, 9, 20))


def _claim(sk, member="M1", provider="P1", date="2025-03-01", **kw):
    return {"claim_sk": sk, "member_sk": member, "provider_sk": provider, "service_date": date, **kw}


def _line(sk, claim, code, amount, **kw):
    return {"line_sk": sk, "claim_sk": claim, "activity_code": code, "gross_amount": amount, **kw}


EDIT = {"column_1_code": "S100", "column_2_code": "S101", "edit_type": "component",
        "modifier_allowed": False, "valid_from": "2020-01-01", "valid_to": None}
PACKAGE = [{"package_code": "PKG1", "component_code": "C1", "expected_zero_price": True},
           {"package_code": "PKG1", "component_code": "C2", "expected_zero_price": True}]
MEMBERS = [{"member_sk": "M1", "sex": "M", "date_of_birth": "1980-01-01", "sponsor_id": "SP1"},
           {"member_sk": "M2", "sex": "F", "date_of_birth": "1990-01-01", "sponsor_id": "SP2"}]
BENEFIT = [{"product": "BASIC", "service_family": "imaging", "covered": True,
            "authorization_required": True, "valid_from": "2020-01-01", "valid_to": None}]
COVER = [{"coverage_id": "CV1", "member_sk": "M1", "product": "BASIC", "payer_id": "PAY1",
          "valid_from": "2024-01-01", "valid_to": "2026-12-31"},
         {"coverage_id": "CV2", "member_sk": "M2", "product": "BASIC", "payer_id": "PAY1",
          "valid_from": "2024-01-01", "valid_to": "2026-12-31"}]


def _auth(status="approved", member="M1", provider="P1", code="MRI", units=None, value=None,
          vf="2025-01-01", vt="2025-12-31"):
    return dict(
        authorization=[{"authorization_sk": "A1", "request_id": "RQ1", "response_id": "RS1",
                        "status": status, "valid_from": vf, "valid_to": vt, "provider_sk": provider,
                        "member_sk": member, "tenant_id": TENANT}],
        authorization_line=[{"authorization_line_sk": "AL1", "authorization_sk": "A1",
                             "activity_code": code, "approved_units": units, "approved_value": value,
                             "denial_code": None, "tenant_id": TENANT}],
    )


# ---------------------------------------------------------------------------
# one fire / clean scenario per control
# ---------------------------------------------------------------------------


def s_pay01_r04(clean):
    claims = [_claim("C1", cross_payer_match_token="TK1"), _claim("CX", member="M2")]
    lines = [_line("L1", "C1", "MRI", 1000.0), _line("LX", "CX", "CONS", 100.0)]
    cob = [{"member_sk": "M2", "primary_payer": "PAY1", "secondary_payer": "PAY9"}]
    ours, theirs = 1000.0, 1000.0
    if clean:
        cob.append({"member_sk": "M1", "primary_payer": "PAY2", "secondary_payer": "PAY1",
                    "valid_from": "2024-01-01", "valid_to": "2026-12-31"})
        ours, theirs = 200.0, 800.0
    return claims, lines, dict(
        remittance=[{"remittance_sk": "R1", "claim_sk": "C1", "line_sk": "L1", "decision": "paid",
                     "payment_amount": ours, "tenant_id": TENANT}],
        other_payer_remittance=[{"claim_sk": None, "other_payer_id": "PAY2", "payment_amount": theirs,
                                 "cross_payer_match_token": "TK1", "tenant_id": TENANT}],
        coordination_of_benefits=cob,
    ), "claim", "C1"


def s_pay02_r01(clean):
    edit = dict(EDIT, modifier_allowed=clean)
    lines = [_line("L1", "C1", "S100", 4210.0), _line("L2", "C1", "S101", 1180.0, indicator="59")]
    return [_claim("C1")], lines, dict(bundling_edit_table=[edit]), "claim", "C1"


def s_pay02_r02(clean):
    later = "2025-03-20" if clean else "2025-03-01"
    claims = [_claim("C1", provider="P1"), _claim("C2", provider="P2", date=later)]
    lines = [_line("L1", "C1", "S100", 4210.0), _line("L2", "C2", "S101", 1180.0)]
    return claims, lines, dict(bundling_edit_table=[EDIT]), "claim", "C2"


def s_pay02_r03(clean):
    lines = [_line("L1", "C1", "PKG1", 5000.0), _line("L2", "C1", "C1", 0.0 if clean else 300.0),
             _line("L3", "C1", "C2", 0.0)]
    return [_claim("C1")], lines, dict(package_definition=PACKAGE), "claim", "C1"


def s_pay02_r04(clean):
    lines = [_line("L1", "C1", "PKG1", 5000.0), _line("L2", "C1", "C1", 300.0),
             _line("L3", "C1", "C2", 0.0)]
    rem = [{"remittance_sk": "R1", "claim_sk": "C1", "line_sk": "L1", "decision": "paid",
            "payment_amount": 5000.0},
           {"remittance_sk": "R2", "claim_sk": "C1", "line_sk": "L2",
            "decision": "denied" if clean else "paid", "payment_amount": 0.0 if clean else 300.0,
            "denial_code": "BUNDLED" if clean else None}]
    return [_claim("C1")], lines, dict(package_definition=PACKAGE, remittance=rem), "claim", "C1"


def s_pay02_r05(clean):
    claims, lines = [], []
    n = 0
    for p in ("P1", "P2", "P3", "P4", "P5"):
        count = 30 if p == "P1" else 10
        for i in range(count):
            n += 1
            ck = f"C{n}"
            claims.append(_claim(ck, member=f"M{n}", provider=p, date=f"2025-0{1 + i % 9}-1{i % 9}"))
            lines.append(_line(f"L{n}a", ck, "L1", 40.0))
            pair = (p == "P1" and not clean) or i == 0
            if pair:
                lines.append(_line(f"L{n}b", ck, "L2", 90.0))
    prov = [{"provider_sk": p, "provider_type": "clinic", "specialty": "general"}
            for p in ("P1", "P2", "P3", "P4", "P5")]
    return claims, lines, dict(provider=prov, bundling_edit_table=[EDIT]), "provider", "P1"


def s_pay03_r01(clean):
    date = "2024-06-01" if clean else "2025-03-01"
    lines = [_line("L1", "C1", "OLD1", 800.0), _line("L2", "C1", "S100", 100.0)]
    csv = [{"code": "OLD1", "code_system": "CPT", "valid_from": "2015-01-01", "valid_to": "2024-12-31"},
           {"code": "S100", "code_system": "CPT", "valid_from": "2015-01-01", "valid_to": None}]
    return [_claim("C1", date=date)], lines, dict(code_system_version=csv), "claim", "C1"


def s_pay03_r02(clean):
    member = "M2" if clean else "M1"
    lines = [_line("L1", "C1", "OB1", 600.0)]
    return [_claim("C1", member=member)], lines, dict(member=MEMBERS), "claim", "C1"


def s_pay03_r03(clean):
    lines = [_line("L1", "C1", "U1", 500.0, units=2 if clean else 5)]
    pol = [{"activity_code": "U1", "max_units_per_day": 2}]
    return [_claim("C1")], lines, dict(unit_maximum_policy=pol), "claim", "C1"


def s_pay03_r04(clean):
    end = "2025-03-01 10:00" if clean else "2025-03-01 09:20"
    lines = [_line("L1", "C1", "T15", 400.0, units=4, rendering_clinician_id="D1",
                   service_start_time="2025-03-01 09:00", service_end_time=end)]
    return [_claim("C1")], lines, {}, "claim", "C1"


def s_pay04_r01(clean):
    lines = [_line("L1", "C1", "MRI", 1500.0, authorization_id="A1" if clean else None)]
    tables = dict(benefit_rule_version=BENEFIT, coverage_period=COVER, **_auth())
    return [_claim("C1")], lines, tables, "claim", "C1"


def s_pay04_r02(clean):
    lines = [_line("L1", "C1", "MRI", 1500.0, authorization_id="A1")]
    return [_claim("C1")], lines, _auth(status="approved" if clean else "denied"), "claim", "C1"


def s_pay04_r03(clean):
    lines = [_line("L1", "C1", "MRI", 1500.0, authorization_id="RS1")]
    return [_claim("C1")], lines, _auth(code="MRI" if clean else "CT"), "claim", "C1"


def s_pay04_r04(clean):
    claims = [_claim(f"C{i}", date=f"2025-03-0{i}") for i in (1, 2, 3)]
    lines = [_line(f"L{i}", f"C{i}", "MRI", 500.0, authorization_id="A1") for i in (1, 2, 3)]
    return claims, lines, _auth(units=3 if clean else 2), "claim", "C3"


def s_pay04_r05(clean):
    claims = [_claim("C1", member="M1"), _claim("C2", member="M1" if clean else "M2")]
    lines = [_line("L1", "C1", "MRI", 500.0, authorization_id="A1"),
             _line("L2", "C2", "MRI", 500.0, authorization_id="A1")]
    return claims, lines, dict(member=MEMBERS, **_auth()), "claim", "C2"


def s_pay05_r01(clean):
    lines = [_line("L1", "C1", "S100", 4210.0), _line("L2", "C1", "S101", 1180.0, indicator="59")]
    return [_claim("C1")], lines, dict(bundling_edit_table=[dict(EDIT, modifier_allowed=clean)]), "claim", "C1"


def s_pay05_r02(clean):
    claims = [_claim("C1", submission_date="2025-03-02"), _claim("C2", submission_date="2025-11-30")]
    lines = [_line("L1", "C1", "S100", 4210.0, indicator="22"), _line("L2", "C2", "CONS", 100.0)]
    obs = [{"observation_sk": "O9", "line_sk": "L2", "attachment_ref": "x"}]
    if clean:
        obs.append({"observation_sk": "O1", "line_sk": "L1", "attachment_ref": "op-report.pdf"})
    docs = [{"document_sk": "D9", "claim_sk": "C2", "doc_type": "report"}]
    return claims, lines, dict(observation=obs, document=docs), "claim", "C1"


def s_pay05_r03(clean):
    claims, lines = [], []
    n = 0
    provs = ("P1", "P2", "P3", "P4", "P5", "P6")
    for p in provs:
        for i in range(80):
            n += 1
            ck = f"C{n}"
            claims.append(_claim(ck, member=f"M{n}", provider=p))
            flagged = i == 0 or (p == "P1" and not clean and i < 40)
            lines.append(_line(f"L{n}", ck, "CONS", 100.0, indicator="25" if flagged else None))
    prov = [{"provider_sk": p, "provider_type": "clinic", "specialty": "general"} for p in provs]
    return claims, lines, dict(provider=prov), "provider", "P1"


def s_pay05_r04(clean):
    claims, lines, rem = [], [], []
    edit = {"column_1_code": "S200", "column_2_code": "S201", "edit_type": "component",
            "modifier_allowed": True, "valid_from": "2025-05-01", "valid_to": None}
    dates_pre = ["2025-01-10", "2025-02-10", "2025-03-10", "2025-04-10"]
    dates_post = ["2025-06-05", "2025-06-20", "2025-07-05", "2025-07-20", "2025-08-05", "2025-08-20"]
    n = 0

    def add(date, indicator, paid, denied=False):
        nonlocal n
        n += 1
        ck = f"C{n}"
        claims.append(_claim(ck, member=f"M{n}", provider="P1", date=date))
        lines.append(_line(f"L{n}a", ck, "S200", 2000.0))
        lines.append(_line(f"L{n}b", ck, "S201", 400.0, indicator=indicator))
        rem.append({"remittance_sk": f"R{n}", "claim_sk": ck, "line_sk": f"L{n}b",
                    "decision": "denied" if denied else "paid", "payment_amount": paid,
                    "denial_code": "BUNDLED" if denied else None})
    for d in dates_pre:
        add(d, None, 400.0)
    add("2025-05-05", None, 0.0, denied=True)
    for d in dates_post:
        add(d, None if clean else "59", 0.0 if clean else 400.0, denied=clean)
    return claims, lines, dict(bundling_edit_table=[edit], remittance=rem), "provider", "P1"


def s_pay06_r01(clean):
    lines = [_line("L1", "C1", "MRI", 900.0 if clean else 1200.0)]
    tables = dict(
        tariff=[{"activity_code": "MRI", "network_tier": "A", "allowed_price": 1000.0,
                 "valid_from": "2024-01-01", "valid_to": None, "payer_id": None}],
        contract=[{"contract_sk": "K1", "provider_sk": "P1", "payer_id": "PAY1", "valid_from": "2024-01-01",
                   "valid_to": None, "tariff_basis": "tariff", "discount_pct": 10.0, "network_tier": "A"}],
    )
    return [_claim("C1")], lines, tables, "claim", "C1"


SCENARIOS = {
    "PAY-01-R04": ("pay_01_r04_cross_payer_duplicate", s_pay01_r04),
    "PAY-02-R01": ("pay_02_r01_same_claim_component", s_pay02_r01),
    "PAY-02-R02": ("pay_02_r02_cross_claim_component", s_pay02_r02),
    "PAY-02-R03": ("pay_02_r03_package_completeness", s_pay02_r03),
    "PAY-02-R04": ("pay_02_r04_inclusive_leakage", s_pay02_r04),
    "PAY-02-R05": ("pay_02_r05_novel_unbundling", s_pay02_r05),
    "PAY-03-R01": ("pay_03_r01_invalid_code", s_pay03_r01),
    "PAY-03-R02": ("pay_03_r02_demographic_impossibility", s_pay03_r02),
    "PAY-03-R03": ("pay_03_r03_unit_maximum", s_pay03_r03),
    "PAY-03-R04": ("pay_03_r04_time_unit_inconsistency", s_pay03_r04),
    "PAY-04-R01": ("pay_04_r01_missing_authorization", s_pay04_r01),
    "PAY-04-R02": ("pay_04_r02_invalid_timing_status", s_pay04_r02),
    "PAY-04-R03": ("pay_04_r03_scope_mismatch", s_pay04_r03),
    "PAY-04-R04": ("pay_04_r04_quantity_exhaustion", s_pay04_r04),
    "PAY-04-R05": ("pay_04_r05_authorization_reuse", s_pay04_r05),
    "PAY-05-R01": ("pay_05_r01_prohibited_indicator_pair", s_pay05_r01),
    "PAY-05-R02": ("pay_05_r02_indicator_support_absent", s_pay05_r02),
    "PAY-05-R03": ("pay_05_r03_indicator_rate_outlier", s_pay05_r03),
    "PAY-05-R04": ("pay_05_r04_post_edit_migration", s_pay05_r04),
    "PAY-06-R01": ("pay_06_r01_tariff_price_variance", s_pay06_r01),
}

FORBIDDEN_WORDS = ("fraud", "PAY-0", "_r0")


def _run(config, registry, rule_id, clean):
    impl, scenario = SCENARIOS[rule_id]
    claims, lines, tables, subject_type, subject_id = scenario(clean)
    ctx = _ctx(config, claims, lines, **tables)
    return pay_a.IMPLEMENTATIONS[impl](ctx, registry.get(rule_id)), subject_type, subject_id, ctx


@pytest.mark.parametrize("rule_id", sorted(SCENARIOS))
def test_fires_on_the_planted_pattern(config, registry, rule_id):
    signals, subject_type, subject_id, _ = _run(config, registry, rule_id, clean=False)
    assert signals, f"{rule_id} did not fire on its planted fixture"
    hit = [s for s in signals if s.subject_type == subject_type and s.subject_id == subject_id]
    assert hit, [(s.subject_type, s.subject_id) for s in signals]
    for s in signals:
        text = s.evidence.get("plain_language", "")
        assert len(text) > 30
        assert not any(w in text for w in FORBIDDEN_WORDS), text
        assert s.fact_key
        assert s.rule_id == rule_id


@pytest.mark.parametrize("rule_id", sorted(SCENARIOS))
def test_silent_on_the_clean_variant(config, registry, rule_id):
    signals, *_ = _run(config, registry, rule_id, clean=True)
    assert signals == [], [s.evidence.get("plain_language") for s in signals]


@pytest.mark.parametrize("rule_id", sorted(SCENARIOS))
def test_empty_tables_return_nothing(config, registry, rule_id):
    impl, _ = SCENARIOS[rule_id]
    ctx = _ctx(config, [_claim("C1")], [])
    assert pay_a.IMPLEMENTATIONS[impl](ctx, registry.get(rule_id)) == []
    empty = ControlContext(config=config, tenant_id=TENANT, claims=pd.DataFrame(), features=None,
                           dataset=CanonicalDataset.empty(), run_date=_dt.date(2026, 9, 20))
    assert pay_a.IMPLEMENTATIONS[impl](empty, registry.get(rule_id)) == []


@pytest.mark.parametrize("rule_id", sorted(SCENARIOS))
def test_idempotent(config, registry, rule_id):
    first, *_ = _run(config, registry, rule_id, clean=False)
    second, *_ = _run(config, registry, rule_id, clean=False)
    assert sorted(s.signal_id for s in first) == sorted(s.signal_id for s in second)
    assert len({s.signal_id for s in first}) == len(first)


def test_unlock_yaml_and_implementations_agree():
    raw = yaml.safe_load((ROOT / "rules" / "unlocks" / "PAY_A.yaml").read_text(encoding="utf-8"))
    declared = {u["implementation"] for u in raw["unlocks"]}
    assert declared == set(pay_a.IMPLEMENTATIONS)
    assert {u["rule_id"] for u in raw["unlocks"]} == set(SCENARIOS)
    for u in raw["unlocks"]:
        assert SCENARIOS[u["rule_id"]][0] == u["implementation"]


def test_hard_edits_establish_exposure_and_statistical_ones_do_not(config, registry):
    for rule_id in ("PAY-02-R01", "PAY-03-R02", "PAY-06-R01"):
        signals, *_ = _run(config, registry, rule_id, clean=False)
        assert all(s.exposure_aed > 0 for s in signals)
    for rule_id in ("PAY-02-R05", "PAY-05-R03", "PAY-05-R04"):
        signals, *_ = _run(config, registry, rule_id, clean=False)
        assert all(s.exposure_aed == 0 for s in signals)


def test_evaluator_runs_the_unlocked_control(config, registry):
    """Through the evaluator: the unlock is honoured and nothing errors."""
    from fwa.engine.evaluator import Evaluator

    _, _, _, ctx = _run(config, registry, "PAY-04-R02", clean=False)
    report = Evaluator(registry).run(ctx, rule_ids=["PAY-04-R02", "PAY-04-R03"])
    rec = {r.rule_id: r for r in report.records}
    assert rec["PAY-04-R02"].support_basis == "unlocked"
    assert rec["PAY-04-R02"].outcome == "TRIGGERED"
    assert rec["PAY-04-R03"].outcome == "NOT_TRIGGERED"
    assert not report.errors


def test_package_missing_core_activities(config, registry):
    """PAY-02-R03 second limb: a package billed without the activities every other claim lists."""
    claims, lines = [], []
    for i in range(12):
        ck = f"C{i}"
        claims.append(_claim(ck, member=f"M{i}"))
        lines.append(_line(f"L{i}p", ck, "PKG1", 5000.0))
        if i != 0:
            lines += [_line(f"L{i}a", ck, "C1", 0.0), _line(f"L{i}b", ck, "C2", 0.0)]
    ctx = _ctx(config, claims, lines, package_definition=PACKAGE)
    signals = pay_a.pay_02_r03_package_completeness(ctx, registry.get("PAY-02-R03"))
    assert [s.subject_id for s in signals] == ["C0"]
    assert "core activities" in signals[0].evidence["plain_language"]
