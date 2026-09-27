"""Unlocked ENT controls (src/fwa/engine/unlocked/ent.py) on a small hand-built fixture.

One clean mini-world (six clinics, a hospital, three pharmacies, sixty members,
eighteen months of claims) is built once. For each control a pattern is planted
into a copy of it. Each control must: fire on the planted copy with the expected
subject and a plain-language sentence; stay silent on the clean world; return
nothing (and not raise) on an empty dataset; and give the same signal ids twice.
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path

import pandas as pd
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from fwa.canonical import CanonicalDataset  # noqa: E402
from fwa.config import load_config  # noqa: E402
from fwa.engine.context import ControlContext  # noqa: E402
from fwa.engine.registry import RuleRegistry  # noqa: E402
from fwa.engine.unlocked import ent  # noqa: E402
from fwa.engine.unlocks import decide, runtime_view  # noqa: E402

T = "T001"
D = pd.Timestamp


@pytest.fixture(scope="module")
def cfg():
    return load_config(ROOT / "config")


@pytest.fixture(scope="module")
def reg():
    return RuleRegistry.from_directory(ROOT / "rules")


# ---------------------------------------------------------------------------
# the mini world
# ---------------------------------------------------------------------------

CODES = [
    # code, type, service_family, code_family, facilities, tele, minutes, time_based, sex, age_min, price
    ("99213", "CPT", "CONSULTATION", "EM_OFFICE_EST", "CLINIC;HOSPITAL", True, 20, False, None, 0, 230.0),
    ("99442", "CPT", "TELEHEALTH", "TELEHEALTH", "CLINIC;HOSPITAL", True, 20, True, None, 0, 140.0),
    ("97110", "CPT", "PHYSIOTHERAPY", "PHYSIO", "CLINIC;HOSPITAL", False, 15, True, None, 0, 120.0),
    ("80053", "CPT", "LAB", "LAB_PANEL", "CLINIC;HOSPITAL;DIAGNOSTIC_LAB", False, None, False, None, 0, 180.0),
    ("70551", "CPT", "ADVANCED_IMAGING", "IMAGING_MRI", "HOSPITAL;RADIOLOGY_CENTRE", False, 45, False, None, 0, 2200.0),
    ("47562", "CPT", "INPATIENT", "SURG_ABDOMINAL", "HOSPITAL", False, 90, False, None, 12, 9000.0),
    ("29881", "CPT", "DAY_SURGERY", "SURG_ORTHO", "HOSPITAL", False, 60, False, None, 16, 9000.0),
    ("19303", "CPT", "INPATIENT", "SURG_BREAST", "HOSPITAL", False, 120, False, "F", 18, 14000.0),
    ("15830", "CPT", "COSMETIC", "COSMETIC", "HOSPITAL", False, 90, False, None, 0, 18000.0),
    ("RM-WARD", "SERVICE", "INPATIENT", "ROOM_DAY", "HOSPITAL", False, None, False, None, 0, 900.0),
]
CLINICS = ["P02", "P03", "P04", "P05", "P06", "P07"]
PHARMACIES = ["P08", "P09", "P10"]
EMIRATE = {"P01": "DUBAI", "P02": "DUBAI", "P03": "RAS AL KHAIMAH", "P04": "DUBAI", "P05": "RAS AL KHAIMAH",
           "P06": "DUBAI", "P07": "RAS AL KHAIMAH", "P08": "DUBAI", "P09": "DUBAI", "P10": "DUBAI", "P11": "DUBAI"}
FTYPE = {"P01": "HOSPITAL", **{p: "CLINIC" for p in CLINICS}, **{p: "PHARMACY" for p in PHARMACIES},
         "P11": "DIAGNOSTIC_LAB"}
MONTHS = pd.date_range("2024-07-01", "2025-12-01", freq="MS")


def _provider(p, ftype, emirate, cred="2020-01-01", **over):
    row = {"provider_sk": p, "tenant_id": T, "source_provider_id": f"SRC-{p}", "regulator_id": f"REG-{p}",
           "provider_type": "FACILITY", "specialty": "GENERAL", "facility_type": ftype,
           "owner_entity_id": f"OWN-{p}", "bank_account_token": f"BK-{p}", "phone_token": f"PH-{p}",
           "address_token": f"AD-{p}", "emirate": emirate, "credentialing_date": cred,
           "ownership_changed_on": None, "licence_no": f"LIC-{p}"}
    row.update(over)
    return row


def _clin(cid, p, specialty="GENERAL_PRACTICE", role="PHYSICIAN", **over):
    row = {"clinician_id": cid, "provider_sk": p, "tenant_id": T, "specialty": specialty, "role": role,
           "licence_no": f"L-{cid}", "licence_valid_from": "2015-01-01", "licence_valid_to": "2099-12-31",
           "privileges": "EM_OFFICE_EST;LAB_PANEL", "leave_periods": "[]", "emirate": EMIRATE.get(p, "DUBAI")}
    row.update(over)
    return row


class W:
    """A mutable bag of tables plus helpers to add claims consistently."""

    def __init__(self):
        self.t: dict[str, list[dict]] = {k: [] for k in (
            "claim_header", "claim_line", "encounter", "remittance", "member", "coverage_period", "provider",
            "provider_status_period", "clinician_roster", "benefit_rule_version", "activity_code_reference",
            "prescription_dispense", "referral", "drug_policy")}
        self.n = 0

    def claim(self, member, provider, date, lines, *, setting="OUTPATIENT", clinician=None, discharge=None,
              bed="auto", duration=20.0, start_hour=9, ordering=None):
        self.n += 1
        c = f"C{self.n:05d}"
        date = D(date)
        discharge = D(discharge) if discharge is not None else date
        gross = 0.0
        for i, (code, amount, units) in enumerate(lines):
            st = date + pd.Timedelta(hours=start_hour, minutes=30 * i)
            self.t["claim_line"].append({
                "line_sk": f"{c}-{i}", "claim_sk": c, "tenant_id": T, "activity_type": "CPT", "activity_code": code,
                "units": units, "gross_amount": amount, "net_amount": amount, "patient_share": 0.0,
                "rendering_clinician_id": clinician, "ordering_clinician_id": ordering, "service_date": date,
                "service_start_time": st, "service_end_time": st + pd.Timedelta(minutes=20),
                "authorization_id": None})
            self.t["remittance"].append({"remittance_sk": f"R{c}-{i}", "claim_sk": c, "line_sk": f"{c}-{i}",
                                         "decision": "PAID", "payment_amount": amount, "tenant_id": T})
            gross += amount
        self.t["claim_header"].append({
            "claim_sk": c, "tenant_id": T, "member_sk": member, "provider_sk": provider, "payer_id": "PAY1",
            "encounter_sk": f"E{c}", "service_date": date, "discharge_date": discharge,
            "submission_date": discharge + pd.Timedelta(days=3), "gross_amount": gross, "net_amount": gross,
            "gross_amount_aed": gross, "approved_amount_aed": gross, "claim_type": setting,
            "diagnosis_primary": "J06.9", "length_of_stay_days": (discharge - date).days})
        inpatient = setting == "INPATIENT"
        self.t["encounter"].append({
            "encounter_sk": f"E{c}", "claim_sk": c, "tenant_id": T, "member_sk": member, "encounter_type": setting,
            "facility_id": provider, "admission_date": date if inpatient else None,
            "discharge_date": discharge if inpatient else None,
            "start_time": date + pd.Timedelta(hours=start_hour),
            "end_time": date + pd.Timedelta(hours=start_hour, minutes=duration),
            "bed_id": (f"BED-{self.n}" if bed == "auto" else bed) if inpatient else None,
            "duration_minutes": duration, "admission_type": "ELECTIVE" if inpatient else None,
            "discharge_type": "HOME" if inpatient else None})
        return c

    def rx(self, member, prescriber, pharmacy, date, product="RX1001", amount=60.0):
        k = len(self.t["prescription_dispense"]) + 1
        self.t["prescription_dispense"].append({
            "rx_sk": f"RX{k:05d}", "claim_sk": None, "tenant_id": T, "member_sk": member,
            "prescriber_id": prescriber, "pharmacy_id": pharmacy, "prescribed_product": product,
            "billed_product": product, "dispensed_product": product, "prescribed_qty": 10.0,
            "dispensed_qty": 10.0, "billed_qty": 10.0, "days_supply": 10.0, "billed_amount": amount,
            "fill_date": D(date), "prescribed_date": D(date)})

    def member(self, m, dob="1980-01-01", sex="M", token=None, employer="E1", **over):
        row = {"member_sk": m, "tenant_id": T, "protected_id_token": token or f"TK-{m}", "date_of_birth": dob,
               "sex": sex, "death_date": None, "death_source": None, "death_source_confidence": None,
               "sponsor_id": None, "employer_id": employer, "emirate": "DUBAI"}
        row.update(over)
        self.t["member"].append(row)
        self.t["coverage_period"].append({
            "coverage_id": f"CV-{m}", "member_sk": m, "tenant_id": T, "product": "GOLD", "payer_id": "PAY1",
            "valid_from": D("2024-01-01"), "valid_to": D("2025-12-31"), "network": "GN", "status": "ACTIVE",
            "agent_id": "A1"})

    def frames(self) -> dict[str, pd.DataFrame]:
        return {k: pd.DataFrame(v) for k, v in self.t.items() if v}


def _base() -> W:
    w = W()
    for code, typ, fam, cfam, fac, tele, mins, tb, sex, amin, price in CODES:
        w.t["activity_code_reference"].append({
            "activity_code": code, "description": f"Service {code}", "activity_type": typ, "service_family": fam,
            "code_family": cfam, "facility_types": fac, "is_telehealth_eligible": tele, "minutes": mins,
            "is_time_based": tb, "sex_restriction": sex, "age_min": amin, "age_max": 120,
            "specialty_required": None, "unit_price_reference": price})
    for fam in ("CONSULTATION", "TELEHEALTH", "PHYSIOTHERAPY", "LAB", "ADVANCED_IMAGING", "INPATIENT",
                "DAY_SURGERY", "COSMETIC"):
        w.t["benefit_rule_version"].append({
            "product": "GOLD", "service_family": fam, "covered": fam != "COSMETIC",
            "benefit_limit": 3000.0 if fam == "PHYSIOTHERAPY" else None, "patient_share_pct": 0.0,
            "authorization_required": False, "exceptions": "", "valid_from": D("2020-01-01"),
            "valid_to": D("2099-12-31"), "version_id": "BV1"})
    for p in ["P01", *CLINICS, *PHARMACIES, "P11"]:
        w.t["provider"].append(_provider(p, FTYPE[p], EMIRATE[p]))
        for st, val in (("LICENCE", "ACTIVE"), ("NETWORK", "GN")):
            w.t["provider_status_period"].append({"provider_sk": p, "status_type": st, "status_value": val,
                                                  "valid_from": D("2020-01-01"), "valid_to": D("2099-12-31"),
                                                  "source": "REGISTER"})
        if p == "P01" or p in CLINICS:
            w.t["clinician_roster"].append(_clin(f"D{p}", p, "GENERAL_SURGERY" if p == "P01" else "GENERAL_PRACTICE",
                                                 privileges="EM_OFFICE_EST;LAB_PANEL;SURG_ABDOMINAL"))
    for i in range(1, 61):
        w.member(f"M{i:03d}", dob=f"{1960 + i % 30}-03-15", sex="F" if i % 2 else "M",
                 employer=f"E{(i - 1) // 10 + 1}")
    w.t["drug_policy"] = [{"product": "RX1001", "is_high_cost": False, "unit_price": 6.0},
                          {"product": "J9999", "is_high_cost": True, "unit_price": 3000.0}]
    idx = 0
    for mi, month in enumerate(MONTHS):
        for pi, p in enumerate(CLINICS):
            for k in range(4):
                member = f"M{idx % 60 + 1:03d}"
                idx += 1
                date = month + pd.Timedelta(days=2 + 6 * k)
                w.claim(member, p, date, [("99213", 230.0, 1.0)], clinician=f"D{p}")
                if k == 0:
                    w.rx(member, f"D{p}", PHARMACIES[(mi + pi) % 3], date)
        member = f"M{mi % 60 + 1:03d}"
        adm = month + pd.Timedelta(days=10)
        w.claim(member, "P01", adm, [("47562", 9000.0, 1.0), ("RM-WARD", 1800.0, 2.0)], setting="INPATIENT",
                clinician="DP01", discharge=adm + pd.Timedelta(days=2))
        for k in range(2):
            w.claim(f"M{(mi + 30 + k) % 60 + 1:03d}", "P01", month + pd.Timedelta(days=4 + 12 * k),
                    [("99213", 230.0, 1.0)], clinician="DP01")
    return w


@pytest.fixture(scope="module")
def base_world() -> W:
    return _base()


def _ctx(cfg, frames: dict[str, pd.DataFrame]) -> ControlContext:
    ds = CanonicalDataset.empty(source_system="fixture", tenant_id=T)
    for name, frame in frames.items():
        ds.set(name, frame, "POPULATED", "fixture")
    claims = frames.get("claim_header", pd.DataFrame(columns=["claim_sk", "tenant_id"]))
    return ControlContext(config=cfg, tenant_id=T, claims=claims, features=None, dataset=ds,
                          run_date=D("2026-09-20").date())


def _run(reg, cfg, rule_id, frames):
    ctx = _ctx(cfg, frames)
    control = reg.get(rule_id)
    view = runtime_view(control, decide(control, reg.unlocks.get(rule_id), ctx.dataset))
    impl = ent.IMPLEMENTATIONS[reg.unlocks[rule_id].implementation]
    return impl.__wrapped__(ctx, view)


def _assert_fires(signals, subject_type, subject_id=None):
    assert signals, "expected at least one signal"
    subjects = {(s.subject_type, s.subject_id) for s in signals}
    assert any(st == subject_type for st, _ in subjects), subjects
    if subject_id is not None:
        assert (subject_type, subject_id) in subjects, subjects
    for s in signals:
        text = s.evidence.get("plain_language", "")
        assert text and "fraud" not in text.lower() and "ENT-" not in text


# ---------------------------------------------------------------------------
# plants: each returns (world, expected subject_type, expected subject_id or None)
# ---------------------------------------------------------------------------


def plant_ent_01_r01(w: W):
    c = w.claim("M001", "P02", "2026-02-15", [("99213", 230.0, 1.0)], clinician="DP02")
    return "claim", c


def plant_ent_01_r02(w: W):
    c = w.claim("M002", "P01", "2025-05-05", [("15830", 18000.0, 1.0)], clinician="DP01")
    return "claim", c


def plant_ent_01_r03(w: W):
    ids = [w.claim("M003", "P02", f"2025-0{m}-20", [("97110", 1000.0, 1.0)], clinician="DP02", start_hour=15)
           for m in range(3, 7)]
    return "claim", ids[-1]


def plant_ent_01_r04(w: W):
    w.t["provider"].append(_provider("P12", "CLINIC", "DUBAI"))
    w.t["provider_status_period"].append({"provider_sk": "P12", "status_type": "NETWORK", "status_value": "RN",
                                          "valid_from": D("2020-01-01"), "valid_to": D("2099-12-31")})
    w.t["clinician_roster"].append(_clin("DP12", "P12"))
    c = w.claim("M004", "P12", "2025-04-10", [("99213", 230.0, 1.0)], clinician="DP12")
    return "claim", c


def plant_ent_02_r01(w: W):
    w.member("M061", dob="1999-07-01", sex="M", token="TK-M005")
    w.claim("M061", "P02", "2025-04-11", [("99213", 230.0, 1.0)], clinician="DP02", start_hour=16)
    return "member", None


def plant_ent_02_r02(w: W):
    for m in w.t["member"]:
        if m["member_sk"] == "M006":
            m.update(death_date="2025-03-01", death_source="MOHAP death register", death_source_confidence=0.95)
    return "claim", None


def plant_ent_02_r04(w: W):
    for day in ("2025-02-03", "2025-02-17"):
        w.claim("M008", "P02", day, [("99213", 230.0, 1.0)], clinician="DP02", start_hour=17)
        w.claim("M008", "P03", day, [("99213", 230.0, 1.0)], clinician="DP03", start_hour=18)
    return "member", "M008"


def plant_ent_03_r01(w: W):
    for r in w.t["provider_status_period"]:
        if r["provider_sk"] == "P04" and r["status_type"] == "LICENCE":
            r["valid_to"] = D("2025-01-31")
    return "claim", None


def plant_ent_03_r03(w: W):
    for r in w.t["clinician_roster"]:
        if r["clinician_id"] == "DP02":
            r["privileges"] = "EM_OFFICE_EST;LAB_PANEL;NOT:IMAGING_MRI"
    c = w.claim("M009", "P02", "2025-06-06", [("70551", 2200.0, 1.0)], clinician="DP02", start_hour=16)
    return "claim", c


def plant_ent_03_r04(w: W):
    c = w.claim("M010", "P01", "2025-06-07", [("29881", 9000.0, 1.0)], clinician="DP01", start_hour=15)
    return "claim", c


def plant_ent_03_r05(w: W):
    w.t["provider"].append(_provider("P13", "CLINIC", "DUBAI", cred="2025-06-01"))
    w.t["clinician_roster"].append(_clin("DP13", "P13"))
    for k in range(60):
        w.claim(f"M{k % 60 + 1:03d}", "P13", D("2025-06-02") + pd.Timedelta(days=k % 85),
                [("99213", 230.0, 1.0)], clinician="DP13", start_hour=8 + k % 8)
    return "provider", "P13"


def plant_ent_04_r01(w: W):
    c = w.claim("M011", "P02", "2025-06-08", [("99213", 230.0, 1.0)], clinician=None, start_hour=16)
    return "claim", c


def plant_ent_04_r02(w: W):
    c = w.claim("M012", "P02", "2025-06-09", [("99213", 230.0, 1.0)], clinician="DP03", start_hour=16)
    return "claim", c


def plant_ent_04_r03(w: W):
    for k in range(3):
        w.n += 1
        c = f"C{w.n:05d}"
        st = D("2025-07-07 10:00") + pd.Timedelta(minutes=5 * k)
        w.t["claim_header"].append({"claim_sk": c, "tenant_id": T, "member_sk": f"M02{k}", "provider_sk": "P02",
                                    "service_date": D("2025-07-07"), "discharge_date": D("2025-07-07"),
                                    "gross_amount_aed": 120.0, "net_amount": 120.0, "claim_type": "OUTPATIENT"})
        w.t["claim_line"].append({"line_sk": f"{c}-0", "claim_sk": c, "tenant_id": T, "activity_code": "97110",
                                  "activity_type": "CPT", "units": 3.0, "gross_amount": 120.0, "net_amount": 120.0,
                                  "rendering_clinician_id": "DP02", "service_date": D("2025-07-07"),
                                  "service_start_time": st, "service_end_time": st + pd.Timedelta(minutes=45)})
    return "clinician", "DP02"


def plant_ent_04_r04(w: W):
    for r in w.t["clinician_roster"]:
        if r["clinician_id"] == "DP03":
            r["leave_periods"] = '[{"from": "2025-05-01", "to": "2025-05-20", "type": "annual leave"}]'
    return "clinician", "DP03"


def plant_ent_05_r01(w: W):
    w.t["clinician_roster"].append(_clin("DP08", "P08", role="PHARMACIST", specialty="PHARMACY"))
    c = w.claim("M013", "P08", "2025-06-10", [("99213", 230.0, 1.0)], clinician="DP08")
    return "claim", c


def plant_ent_05_r02(w: W):
    c = w.claim("M014", "P01", "2025-06-11", [("47562", 9000.0, 1.0)], setting="INPATIENT", clinician="DP01",
                discharge="2025-06-13", bed=None, start_hour=14)
    return "claim", c


def plant_ent_05_r03(w: W):
    ip = next(r for r in w.t["claim_header"] if r["claim_type"] == "INPATIENT" and r["service_date"] > D("2025-01-01"))
    c = w.claim(ip["member_sk"], "P01", ip["service_date"] + pd.Timedelta(days=1), [("99213", 230.0, 1.0)],
                clinician="DP01", start_hour=17)
    return "claim", c


def plant_ent_05_r04(w: W):
    k = 0
    for r in w.t["claim_header"]:
        if r["provider_sk"] == "P02" and r["service_date"] >= D("2025-04-01"):
            k += 1
            if k % 2:
                r["claim_type"] = "INPATIENT"
                for e in w.t["encounter"]:
                    if e["claim_sk"] == r["claim_sk"]:
                        e["encounter_type"] = "INPATIENT"
                        e["bed_id"] = "BED-X"
    return "provider", "P02"


def _tele_provider(w: W, p="P14"):
    w.t["provider"].append(_provider(p, "CLINIC", "DUBAI"))
    w.t["provider_status_period"].append({"provider_sk": p, "status_type": "LICENCE", "status_value": "ACTIVE",
                                          "valid_from": D("2020-01-01"), "valid_to": D("2099-12-31")})
    w.t["clinician_roster"].append(_clin(f"D{p}", p))


def plant_ent_06_r01(w: W):
    _tele_provider(w)
    c = w.claim("M015", "P14", "2025-06-12", [("99442", 140.0, 1.0), ("70551", 2200.0, 1.0)], setting="TELEHEALTH",
                clinician="DP14")
    return "claim", c


def plant_ent_06_r02(w: W):
    _tele_provider(w)
    c = w.claim("M016", "P14", "2025-06-13", [("99442", 140.0, 1.0)], setting="TELEHEALTH", clinician="DP14",
                duration=3.0)
    w.rx("M016", "DP14", "P09", "2025-06-13", product="J9999", amount=3000.0)
    return "claim", c


def plant_ent_06_r03(w: W):
    _tele_provider(w)
    for k in range(30):
        m = f"M{k % 60 + 1:03d}"
        date = D("2025-03-01") + pd.Timedelta(days=k * 3)
        w.claim(m, "P14", date, [("99442", 140.0, 1.0)], setting="TELEHEALTH", clinician="DP14", start_hour=12)
        w.rx(m, "DP14", "P10", date)
    return "provider", "P14"


def plant_ent_06_r04(w: W):
    _tele_provider(w)
    for k in range(30):
        m = f"M{k % 60 + 1:03d}"
        date = D("2025-03-01") + pd.Timedelta(days=k * 3)
        w.claim(m, "P14", date, [("99442", 140.0, 1.0)], setting="TELEHEALTH", clinician="DP14", start_hour=12)
        if k % 3:
            w.rx(m, "DP14", PHARMACIES[k % 3], date, product="J9999", amount=3000.0)
    return "provider", "P14"


PLANTS = {
    "ENT-01-R01": plant_ent_01_r01, "ENT-01-R02": plant_ent_01_r02, "ENT-01-R03": plant_ent_01_r03,
    "ENT-01-R04": plant_ent_01_r04, "ENT-02-R01": plant_ent_02_r01, "ENT-02-R02": plant_ent_02_r02,
    "ENT-02-R04": plant_ent_02_r04, "ENT-03-R01": plant_ent_03_r01, "ENT-03-R03": plant_ent_03_r03,
    "ENT-03-R04": plant_ent_03_r04, "ENT-03-R05": plant_ent_03_r05, "ENT-04-R01": plant_ent_04_r01,
    "ENT-04-R02": plant_ent_04_r02, "ENT-04-R03": plant_ent_04_r03, "ENT-04-R04": plant_ent_04_r04,
    "ENT-05-R01": plant_ent_05_r01, "ENT-05-R02": plant_ent_05_r02, "ENT-05-R03": plant_ent_05_r03,
    "ENT-05-R04": plant_ent_05_r04, "ENT-06-R01": plant_ent_06_r01, "ENT-06-R02": plant_ent_06_r02,
    "ENT-06-R03": plant_ent_06_r03, "ENT-06-R04": plant_ent_06_r04,
}


# ---------------------------------------------------------------------------
# tests
# ---------------------------------------------------------------------------


def test_unlock_yaml_and_implementations_match(reg):
    raw = yaml.safe_load((ROOT / "rules" / "unlocks" / "ENT.yaml").read_text("utf-8"))
    declared = {e["implementation"] for e in raw["unlocks"]}
    assert declared == set(ent.IMPLEMENTATIONS)
    rule_ids = {e["rule_id"] for e in raw["unlocks"]}
    assert rule_ids == set(PLANTS)
    for rid in rule_ids:
        assert reg.get(rid).data_support.value == "NOT_EXECUTABLE_ON_THIS_DATASET"
    assert not rule_ids & {"ENT-02-R03", "ENT-03-R02"}


@pytest.mark.parametrize("rule_id", sorted(PLANTS))
def test_fires_on_planted_pattern(reg, cfg, base_world, rule_id):
    w = copy.deepcopy(base_world)
    subject_type, subject_id = PLANTS[rule_id](w)
    signals = _run(reg, cfg, rule_id, w.frames())
    _assert_fires(signals, subject_type, subject_id)


@pytest.mark.parametrize("rule_id", sorted(PLANTS))
def test_silent_on_clean_world(reg, cfg, base_world, rule_id):
    assert _run(reg, cfg, rule_id, base_world.frames()) == []


@pytest.mark.parametrize("rule_id", sorted(PLANTS))
def test_empty_dataset_returns_nothing(reg, cfg, rule_id):
    ctx = _ctx(cfg, {})
    control = reg.get(rule_id)
    impl = ent.IMPLEMENTATIONS[reg.unlocks[rule_id].implementation]
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


def test_exclusions_hold(reg, cfg, base_world):
    """Look-alikes the declared exclusions cover stay silent."""
    # ENT-01-R01: an emergency visit after cover ended
    w = copy.deepcopy(base_world)
    w.claim("M001", "P01", "2026-02-15", [("99213", 230.0, 1.0)], setting="EMERGENCY", clinician="DP01")
    assert _run(reg, cfg, "ENT-01-R01", w.frames()) == []
    # ENT-02-R02: a death date from an unreliable source
    w = copy.deepcopy(base_world)
    for m in w.t["member"]:
        if m["member_sk"] == "M006":
            m.update(death_date="2025-03-01", death_source="unverified call", death_source_confidence=0.3)
    assert _run(reg, cfg, "ENT-02-R02", w.frames()) == []
    # ENT-03-R05: a burst right after an acquisition
    w = copy.deepcopy(base_world)
    plant_ent_03_r05(w)
    w.t["provider"][-1]["ownership_changed_on"] = "2025-05-20"
    assert _run(reg, cfg, "ENT-03-R05", w.frames()) == []
    # ENT-06-R03: concentration on a pharmacy under the same owner
    w = copy.deepcopy(base_world)
    plant_ent_06_r03(w)
    for p in w.t["provider"]:
        if p["provider_sk"] in ("P14", "P10"):
            p["owner_entity_id"] = "OWN-GROUP"
    assert _run(reg, cfg, "ENT-06-R03", w.frames()) == []


def test_line_edit_exposure_is_established_only_for_hard_controls(reg, cfg, base_world):
    w = copy.deepcopy(base_world)
    plant_ent_01_r03(w)
    s = _run(reg, cfg, "ENT-01-R03", w.frames())
    assert s and s[0].exposure_established and s[0].exposure_aed == pytest.approx(1000.0)
    w = copy.deepcopy(base_world)
    plant_ent_06_r04(w)
    s = _run(reg, cfg, "ENT-06-R04", w.frames())
    assert s and not s[0].exposure_established
