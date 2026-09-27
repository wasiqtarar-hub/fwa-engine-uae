"""Unit tests for the unlocked CLN implementations (src/fwa/engine/unlocked/cln.py).

Every control is exercised on a minimal hand-built multi-table fixture in two
variants — one carrying the planted pattern (it must fire, on the expected
subject, with a readable plain-language sentence) and the clean twin of the
same fixture (it must stay silent) — plus the empty-table and idempotency
checks the family brief requires.
"""

from __future__ import annotations

import datetime as _dt
from pathlib import Path

import pandas as pd
import pytest
import yaml

from fwa.canonical import CanonicalDataset
from fwa.config import load_config
from fwa.engine.context import ControlContext
from fwa.engine.unlocked import cln
from fwa.engine.unlocks import DatasetUnlock, decide, runtime_view

ROOT = Path(__file__).resolve().parents[2]
UNLOCK_FILE = ROOT / "rules" / "unlocks" / "CLN.yaml"
T = "T001"
D0 = _dt.date(2024, 7, 1)


# ----------------------------------------------------------------- fixtures


@pytest.fixture(scope="module")
def cfg():
    return load_config(ROOT / "config")


@pytest.fixture(scope="module")
def unlocks() -> dict[str, DatasetUnlock]:
    raw = yaml.safe_load(UNLOCK_FILE.read_text(encoding="utf-8"))
    return {u["rule_id"]: DatasetUnlock(**u) for u in raw["unlocks"]}


def _day(n: int) -> pd.Timestamp:
    return pd.Timestamp(D0 + _dt.timedelta(days=n))


REF = pd.DataFrame([
    # code, description, service_family, code_family, level, minutes, complexity, price
    dict(activity_code="99213", description="Office visit, level 3", service_family="CONSULTATION",
         code_family="EM_OFFICE_EST", code_level=3, minutes=20, complexity="LOW", unit_price_reference=230.0),
    dict(activity_code="99215", description="Office visit, level 5", service_family="CONSULTATION",
         code_family="EM_OFFICE_EST", code_level=5, minutes=40, complexity="HIGH", unit_price_reference=480.0),
    dict(activity_code="CR110B", description="Case rate pneumonia with CC", service_family="INPATIENT",
         code_family="CASE_RATE", code_level=2, minutes=None, complexity="MODERATE", unit_price_reference=11250.0),
    dict(activity_code="CR110C", description="Case rate pneumonia", service_family="INPATIENT",
         code_family="CASE_RATE", code_level=1, minutes=None, complexity="LOW", unit_price_reference=9000.0),
    dict(activity_code="93306", description="Echocardiography", service_family="PROCEDURE",
         code_family="ECHO", code_level=None, minutes=40, complexity="MODERATE", unit_price_reference=1100.0),
    dict(activity_code="73721", description="MRI knee", service_family="ADVANCED_IMAGING",
         code_family="IMAGING_MRI", code_level=None, minutes=45, complexity="MODERATE", unit_price_reference=2200.0),
    dict(activity_code="73562", description="X-ray knee", service_family="IMAGING",
         code_family="IMAGING_XR", code_level=None, minutes=10, complexity="LOW", unit_price_reference=180.0),
    dict(activity_code="71046", description="Chest X-ray", service_family="IMAGING",
         code_family="IMAGING_XR", code_level=None, minutes=10, complexity="LOW", unit_price_reference=160.0),
    dict(activity_code="83036", description="Haemoglobin A1c", service_family="LAB",
         code_family="LAB_CHEM", code_level=None, minutes=None, complexity="LOW", unit_price_reference=110.0),
    dict(activity_code="80061", description="Lipid panel", service_family="LAB",
         code_family="LAB_PANEL", code_level=None, minutes=None, complexity="LOW", unit_price_reference=130.0),
    dict(activity_code="82465", description="Cholesterol", service_family="LAB",
         code_family="LAB_CHEM", code_level=None, minutes=None, complexity="LOW", unit_price_reference=35.0),
    dict(activity_code="84478", description="Triglycerides", service_family="LAB",
         code_family="LAB_CHEM", code_level=None, minutes=None, complexity="LOW", unit_price_reference=35.0),
    dict(activity_code="83718", description="HDL cholesterol", service_family="LAB",
         code_family="LAB_CHEM", code_level=None, minutes=None, complexity="LOW", unit_price_reference=40.0),
    dict(activity_code="81162", description="BRCA1/2 gene analysis", service_family="LAB",
         code_family="LAB_MOLECULAR", code_level=None, minutes=None, complexity="HIGH", unit_price_reference=4000.0),
    dict(activity_code="20610", description="Joint injection", service_family="PROCEDURE",
         code_family="PROC_ORTHO", code_level=None, minutes=15, complexity="LOW", unit_price_reference=550.0),
    dict(activity_code="0999T", description="Unusual procedure", service_family="PROCEDURE",
         code_family="PROC_OTHER", code_level=None, minutes=15, complexity="LOW", unit_price_reference=900.0),
])

GROUPER = pd.DataFrame([
    dict(diagnosis_code="E87.1", is_cc=True, is_mcc=False, severity_weight=0.25),
    dict(diagnosis_code="I50.9", is_cc=True, is_mcc=False, severity_weight=0.30),
    dict(diagnosis_code="A41.9", is_cc=False, is_mcc=True, severity_weight=0.80),
    dict(diagnosis_code="I10", is_cc=False, is_mcc=False, severity_weight=0.0),
])


class Builder:
    """A tiny multi-table world: claims, lines, diagnoses and anything else."""

    def __init__(self) -> None:
        self.claims: list[dict] = []
        self.lines: list[dict] = []
        self.dx: list[dict] = []
        self.extra: dict[str, list[dict]] = {}

    def claim(self, cid, member, provider, day, *, dx="J06.9", ctype="OP", gross=None, los=0,
              secondary=(), poa="Y"):
        self.claims.append(dict(claim_sk=cid, tenant_id=T, member_sk=member, provider_sk=provider,
                                payer_id="PAY1", service_date=_day(day), discharge_date=_day(day + los),
                                claim_type=ctype, diagnosis_primary=dx, gross_amount_aed=gross,
                                approved_amount_aed=gross, length_of_stay_days=los))
        self.dx.append(dict(claim_sk=cid, code=dx, sequence=1, diagnosis_type="PRINCIPAL",
                            present_on_admission=poa, tenant_id=T))
        for i, code in enumerate(secondary, start=2):
            self.dx.append(dict(claim_sk=cid, code=code, sequence=i, diagnosis_type="SECONDARY",
                                present_on_admission="Y", tenant_id=T))
        return cid

    def line(self, cid, code, amount=100.0, *, units=1, day=None, **kw):
        n = len(self.lines) + 1
        header = next(c for c in self.claims if c["claim_sk"] == cid)
        row = dict(line_sk=f"L{n:05d}", claim_sk=cid, activity_code=code, units=units,
                   gross_amount=amount, net_amount=amount, patient_share=0.0,
                   service_date=header["service_date"] if day is None else _day(day), tenant_id=T)
        row.update(kw)
        self.lines.append(row)
        return row["line_sk"]

    def add(self, table, **row):
        self.extra.setdefault(table, []).append(row)

    def context(self, cfg) -> ControlContext:
        ds = CanonicalDataset.empty(source_system="fixture", tenant_id=T)
        header = pd.DataFrame(self.claims)
        if not header.empty:
            gross = header.groupby("claim_sk").size() * 0.0
            if self.lines:
                gross = pd.DataFrame(self.lines).groupby("claim_sk")["gross_amount"].sum()
            fill = header["claim_sk"].map(gross).fillna(100.0)
            header["gross_amount_aed"] = header["gross_amount_aed"].fillna(fill).astype(float)
            ds.set("claim_header", header, "POPULATED", "fixture")
        tables = {"claim_line": self.lines, "diagnosis": self.dx, **self.extra}
        for name, rows in tables.items():
            if rows:
                ds.set(name, pd.DataFrame(rows), "POPULATED", "fixture")
        ds.set("activity_code_reference", REF.copy(), "POPULATED", "fixture")
        ds.set("drg_grouper", GROUPER.copy(), "POPULATED", "fixture")
        return ControlContext(config=cfg, tenant_id=T, claims=header, features=None, dataset=ds,
                              run_date=_dt.date(2026, 1, 1))


def _run(rule_id, ctx, registry, unlocks):
    control = registry.get(rule_id)
    unlock = unlocks[rule_id]
    decision = decide(control, unlock, ctx.dataset)
    assert decision.basis in ("unlocked", "upgraded"), f"{rule_id} fixture misses {decision.missing}"
    fn = cln.IMPLEMENTATIONS[unlock.implementation]
    out = fn(ctx, runtime_view(control, decision))
    assert unlock.implementation not in cln.LAST_ERRORS, cln.LAST_ERRORS.get(unlock.implementation)
    return out


# ----------------------------------------------------------- the scenarios
#
# Each scenario builds (dirty=True) the planted pattern or (dirty=False) its
# clean twin and returns (builder, expected_subject_type, expected_subject_id).


def s_cln01r04(dirty):
    b = Builder()
    for m in range(12):
        for k in range(10):
            cid = b.claim(f"C{m:02d}{k}", f"M{k}", "P1", m * 30 + k)
            top = k < (8 if (dirty and m >= 6) else 2)
            b.line(cid, "99215" if top else "99213", 480.0 if top else 230.0)
    return b, "provider", "P1"


def s_cln01r01(dirty):
    b = Builder()
    for p in range(1, 7):
        b.add("provider", provider_sk=f"P{p}", specialty="FAMILY_MEDICINE", tenant_id=T)
        tops = 25 if (dirty and p == 1) else 2 + p % 3
        for k in range(30):
            cid = b.claim(f"C{p}{k:02d}", f"M{p}{k}", f"P{p}", k * 5)
            b.line(cid, "99215" if k < tops else "99213", 480.0 if k < tops else 230.0)
    return b, "provider", "P1"


def s_cln02r01(dirty):
    b = Builder()
    cid = b.claim("C1", "M1", "H1", 10, dx="J18.9", ctype="IP", los=4, secondary=["E87.1"])
    b.line(cid, "CR110B", 11250.0)
    if not dirty:  # the condition is on the patient's record elsewhere
        b.claim("C2", "M1", "H1", 200, dx="E87.1", ctype="OP")
    return b, "claim", "C1"


def s_cln02r02(dirty):
    b = Builder()
    for p in range(1, 6):
        for k in range(25):
            with_cc = k < (25 if (dirty and p == 1) else 3)
            b.claim(f"C{p}{k:02d}", f"M{p}{k}", f"H{p}", k * 7, dx="J18.9", ctype="IP", los=3,
                    secondary=["E87.1"] if with_cc else ["I10"])
    return b, "provider", "H1"


def s_cln02r03(dirty):
    b = Builder()
    for m in ("M2", "M3"):  # other patients show heart failure is long-standing
        b.claim(f"{m}a", m, "H1", 10, dx="I10", secondary=["I50.9"])
        b.claim(f"{m}b", m, "H1", 90, dx="I10", secondary=["I50.9"])
    for i, day in enumerate((10, 60, 110, 160, 200, 260)):
        sec = ["I50.9"] if (i == 4 or (not dirty and i == 5)) else []
        b.claim(f"M1c{i}", "M1", "H2", day, dx="J06.9", secondary=sec)
    return b, "claim", "M1c4"


def s_cln02r04(dirty):
    b = Builder()
    b.claim("C1", "M1", "H1", 10, dx="J18.9", ctype="IP", los=3, poa="N" if dirty else "Y")
    return b, "claim", "C1"


def s_cln02r05(dirty):
    b = Builder()
    for m in range(12):
        for k in range(5):
            sec = ["A41.9"] if (dirty and m >= 6) else (["E87.1"] if k == 0 else [])
            b.claim(f"C{m:02d}{k}", f"M{k}", "H1", m * 30 + k, dx="J18.9", ctype="IP", los=3, secondary=sec)
    return b, "provider", "H1"


def s_cln03r02(dirty):
    b = Builder()
    cid = b.claim("C1", "M1", "P1", 10, dx="J06.9" if dirty else "R07.9")
    b.line(cid, "93306", 1100.0)
    b.add("indication_policy", activity_code="93306", allowed_diagnosis_prefixes="I50;R07")
    return b, "claim", "C1"


def s_cln03r03(dirty):
    b = Builder()
    if not dirty:
        b.line(b.claim("C0", "M1", "P1", 5, dx="M17.11"), "73562", 180.0)
    b.line(b.claim("C1", "M1", "P1", 40, dx="M17.11"), "73721", 2200.0)
    b.add("care_pathway_policy", activity_code="73721", required_prior_code="73562", within_days=180)
    return b, "claim", "C1"


def s_cln03r04(dirty):
    b = Builder()
    for p in range(1, 6):
        b.add("provider", provider_sk=f"P{p}", specialty="ORTHOPAEDICS", tenant_id=T)
        for k in range(40):
            cid = b.claim(f"C{p}{k:02d}", f"M{p}{k}", f"P{p}", k * 3, dx="M17.11")
            code = "0999T" if (dirty and p == 1 and k < 12) else ("20610" if k % 2 else "73562")
            b.line(cid, code, 300.0)
    return b, "provider", "P1"


def s_cln04r03(dirty):
    b = Builder()
    first = b.line(b.claim("C1", "M1", "P1", 10, dx="E11.9"), "83036", 110.0)
    b.line(b.claim("C2", "M1", "P1", 20, dx="E11.9"), "83036", 110.0)
    b.add("repeat_interval_policy", activity_code="83036", min_interval_days=80, requires_result_before_repeat=True)
    other = b.line(b.claim("C3", "M2", "P1", 10, dx="E11.9"), "83036", 110.0)
    b.add("observation", observation_sk="O1", line_sk=other, claim_sk="C3", observation_code="83036",
          value="6.1", event_time=_day(11), tenant_id=T)
    if not dirty:
        b.add("observation", observation_sk="O2", line_sk=first, claim_sk="C1", observation_code="83036",
              value="9.4", event_time=_day(11), tenant_id=T)
    return b, "claim", "C2"


def s_cln04r05(dirty):
    b = Builder()
    n = 0
    for r in range(1, 7):
        for k in range(6):
            n += 1
            amount = 6000.0 + 50 * k if (dirty and r == 1) else 500.0 + 40 * k + 15 * r
            cid = b.claim(f"D{n:03d}", f"M{n}", "SPEC1", 40 + k, gross=amount)
            b.add("referral", referral_sk=f"R{n:03d}", referrer_provider_sk=f"GP{r}", recipient_provider_sk="SPEC1",
                  member_sk=f"M{n}", referral_date=_day(35 + k), resulting_claim_sk=cid, tenant_id=T)
    return b, "provider", "GP1"


def s_cln05r04(dirty):
    b = Builder()
    n = 0
    for h in range(1, 6):
        admits = 40 if (dirty and h == 1) else 5
        for k in range(40 + admits):
            n += 1
            ip = k >= 40
            cid = b.claim(f"C{n:04d}", f"M{n}", f"H{h}", k, dx="R07.9", ctype="IP" if ip else "OP", los=1 if ip else 0)
            b.add("encounter", encounter_sk=f"E{n:04d}", claim_sk=cid, encounter_type="INPATIENT" if ip else "EMERGENCY",
                  admission_type="EMERGENCY" if ip else None, observation_status=None, tenant_id=T)
    return b, "provider", "H1"


def s_cln06r01(dirty):
    b = Builder()
    line = b.line(b.claim("C1", "M1", "P1", 10, dx="R05.9"), "71046", 160.0, ordering_clinician_id="D1")
    later = b.line(b.claim("C2", "M2", "P1", 60, dx="R05.9"), "71046", 160.0)
    b.add("observation", observation_sk="O1", line_sk=later, claim_sk="C2", observation_code="71046",
          value="Normal chest", event_time=_day(60), tenant_id=T)
    if not dirty:
        b.add("observation", observation_sk="O2", line_sk=line, claim_sk="C1", observation_code="71046",
              value="Normal chest", event_time=_day(11), tenant_id=T)
    return b, "claim", "C1"


def s_cln06r04(dirty):
    b = Builder()
    b.claim("C1", "M1", "P1", 10, gross=800.0)
    b.claim("C2", "M2", "P1", 12, gross=300.0)
    b.add("member_confirmation", confirmation_sk="MC1", member_sk="M1", claim_sk="C1",
          service_confirmed=not dirty, response="I did not visit" if dirty else "Yes",
          response_date=_day(20), channel="APP_VERIFIED", tenant_id=T)
    b.add("attendance_record", record_sk="A1", member_sk="M2", provider_sk="P1", claim_sk="C2",
          attendance_date=_day(12), attended=True, source="CHECK_IN_KIOSK", tenant_id=T)
    return b, "claim", "C1"


def s_cln06r05(dirty):
    b = Builder()
    b.add("drug_policy", product="RX1010", description="Metformin", indication_prefixes="E11")
    for m in ("M1", "M2", "M3", "M4"):
        for i in range(3):
            b.claim(f"{m}c{i}", m, "P1", 10 + 40 * i, dx="E11.9")
        if m != "M1" or not dirty:
            b.add("prescription_dispense", rx_sk=f"RX{m}", claim_sk=f"{m}c0", member_sk=m,
                  billed_product="RX1010", dispensed_product="RX1010", tenant_id=T)
    b.add("observation", observation_sk="O1", line_sk="", claim_sk="M2c1", member_sk="M2",
          observation_code="83036", value="7.0", event_time=_day(51), tenant_id=T)
    return b, "member", "M1"


def s_cln07r01(dirty):
    b = Builder()
    for k in range(40 if dirty else 10):
        b.line(b.claim(f"C{k:03d}", f"M{k}", "P1", 10), "99215", 480.0, rendering_clinician_id="D1")
    return b, "clinician", "D1"


def s_cln07r03(dirty):
    b = Builder()
    b.add("equipment_inventory", provider_sk="P1", equipment_type="MRI", units=1, activity_codes="73721",
          minutes_per_use=45, hours_per_day=12)
    for k in range(25 if dirty else 10):
        b.line(b.claim(f"C{k:03d}", f"M{k}", "P1", 10, dx="M17.11"), "73721", 2200.0)
    return b, "provider", "P1"


def s_cln08r01(dirty):
    b = Builder()
    b.add("clinician_roster", clinician_id="D1", provider_sk="P1", role="PHYSICIAN",
          licence_valid_from=_day(-1000), licence_valid_to=_day(1000), tenant_id=T)
    b.line(b.claim("C1", "M1", "P1", 10, dx="E11.9"), "83036", 110.0,
           ordering_clinician_id=None if dirty else "D1")
    return b, "claim", "C1"


def s_cln08r02(dirty):
    b = Builder()
    cid = b.claim("C1", "M1", "P1", 10, dx="E78.5")
    if dirty:
        for code in ("82465", "84478", "83718"):
            b.line(cid, code, 60.0)
    else:
        b.line(cid, "80061", 130.0)
    for code in ("82465", "84478", "83718"):
        b.add("panel_definition", panel_code="80061", component_code=code)
    return b, "claim", "C1"


def s_cln08r03(dirty):
    b = Builder()
    line = b.line(b.claim("C1", "M1", "P1", 10, dx="E11.9"), "83036", 110.0)
    later = b.line(b.claim("C2", "M2", "P1", 60, dx="E11.9"), "83036", 110.0)
    for ln, cid in ((line, "C1"), (later, "C2")):
        b.add("remittance", remittance_sk=f"R{cid}", claim_sk=cid, line_sk=ln, decision="PAID",
              payment_amount=110.0, tenant_id=T)
    b.add("observation", observation_sk="O2", line_sk=later, claim_sk="C2", observation_code="83036",
          value="6.4", event_time=_day(60), tenant_id=T)
    if not dirty:
        b.add("observation", observation_sk="O1", line_sk=line, claim_sk="C1", observation_code="83036",
              value="7.9", event_time=_day(11), tenant_id=T)
    return b, "claim", "C1"


def s_cln08r04(dirty):
    b = Builder()
    n = 0
    for k in range(12):  # the reference laboratory billing directly
        n += 1
        b.line(b.claim(f"C{n:03d}", f"M{n}", "LAB1", k, dx="E11.9"), "83036", 100.0 + k)
    for p in range(1, 5):
        mark = 3.0 if (dirty and p == 1) else 1.1 + 0.02 * p
        for k in range(12):
            n += 1
            b.line(b.claim(f"C{n:03d}", f"M{n}", f"P{p}", k, dx="E11.9"), "83036", round(105.0 * mark, 2),
                   performing_entity_id="LAB1")
    return b, "provider", "P1"


def s_cln08r05(dirty):
    b = Builder()
    n = 0
    for p in range(1, 6):
        b.add("provider", provider_sk=f"P{p}", specialty="FAMILY_MEDICINE", tenant_id=T)
        genetic = 40 if (dirty and p == 1) else 2
        for k in range(80):
            n += 1
            code = "81162" if k < genetic else "83036"
            b.line(b.claim(f"C{n:04d}", f"M{n}", f"P{p}", k, dx="Z80.3"), code, 4000.0 if code == "81162" else 110.0)
    return b, "provider", "P1"


def s_cln04r01(dirty):
    b = Builder()
    b.add("repeat_interval_policy", activity_code="83036", min_interval_days=80, requires_result_before_repeat=True)
    b.line(b.claim("C1", "M1", "P1", 10, dx="E11.9"), "83036", 110.0)
    b.line(b.claim("C2", "M1", "P1", 40 if dirty else 120, dx="E11.9"), "83036", 110.0)
    return b, "member", "M1"


def s_cln04r02(dirty):
    b = Builder()
    n = 0
    for m in range(1, 1201):  # a population that sees a doctor once or twice a month
        for k in range(1 + m % 2):
            n += 1
            b.line(b.claim(f"C{n:05d}", f"M{m}", "P1", 10 + 7 * k), "99213", 230.0)
    for k in range(12 if dirty else 2):
        n += 1
        b.line(b.claim(f"C{n:05d}", "MX", "P1", 10 + 2 * k), "99213", 230.0)
    return b, "member", "MX"


def s_cln06r03(dirty):
    b = Builder()
    for k in range(4):
        cid = b.claim(f"C{k}", f"M{k}", "P1", 10 + 7 * k, dx="R07.9")
        for code, amount in (("99213", 230.0), ("93306", 1100.0), ("20610", 550.0)):
            minute = 0 if dirty else 7 * k
            b.line(cid, code, amount, service_start_time=_day(10 + 7 * k) + pd.Timedelta(hours=10, minutes=minute))
    return b, "provider", "P1"


SCENARIOS = {
    "CLN-04-R01": s_cln04r01, "CLN-04-R02": s_cln04r02, "CLN-06-R03": s_cln06r03,
    "CLN-01-R01": s_cln01r01, "CLN-01-R04": s_cln01r04, "CLN-02-R01": s_cln02r01, "CLN-02-R02": s_cln02r02,
    "CLN-02-R03": s_cln02r03, "CLN-02-R04": s_cln02r04, "CLN-02-R05": s_cln02r05,
    "CLN-03-R02": s_cln03r02, "CLN-03-R03": s_cln03r03, "CLN-03-R04": s_cln03r04,
    "CLN-04-R03": s_cln04r03, "CLN-04-R05": s_cln04r05, "CLN-05-R04": s_cln05r04,
    "CLN-06-R01": s_cln06r01, "CLN-06-R04": s_cln06r04, "CLN-06-R05": s_cln06r05,
    "CLN-07-R01": s_cln07r01, "CLN-07-R03": s_cln07r03, "CLN-08-R01": s_cln08r01,
    "CLN-08-R02": s_cln08r02, "CLN-08-R03": s_cln08r03, "CLN-08-R04": s_cln08r04,
    "CLN-08-R05": s_cln08r05,
}
IDS = sorted(SCENARIOS)


# ------------------------------------------------------------------- tests


def test_every_unlock_maps_to_an_implementation_and_back(unlocks):
    declared = {u.implementation for u in unlocks.values()}
    assert declared == set(cln.IMPLEMENTATIONS)
    assert set(unlocks) == set(SCENARIOS)
    for u in unlocks.values():
        assert u.rule_id.startswith("CLN-")


def test_every_unlocked_control_is_not_executable_in_the_catalogue(registry, unlocks):
    for rule_id, unlock in unlocks.items():
        static = registry.get(rule_id).data_support.value
        if unlock.upgrades_partial:
            assert static == "PARTIAL"   # the one optional upgrade: a proxy replaced by the real measure
        else:
            assert static == "NOT_EXECUTABLE_ON_THIS_DATASET"


@pytest.mark.parametrize("rule_id", IDS)
def test_fires_on_the_planted_pattern(rule_id, cfg, registry, unlocks):
    builder, subject_type, subject_id = SCENARIOS[rule_id](True)
    signals = _run(rule_id, builder.context(cfg), registry, unlocks)
    assert signals, f"{rule_id} did not fire on its planted fixture"
    hit = [s for s in signals if s.subject_type == subject_type and s.subject_id == subject_id]
    assert hit, f"{rule_id}: expected a {subject_type} signal on {subject_id}, got " \
                f"{[(s.subject_type, s.subject_id) for s in signals]}"
    for s in signals:
        text = s.evidence.get("plain_language", "")
        assert isinstance(text, str) and len(text) > 20
        assert "CLN-" not in text and "fraud" not in text.lower()
        assert s.rule_id == rule_id
        assert s.fact_key
        assert s.evidence.get("data_support") in (None, "PARTIAL")
        pd.io.json.ujson_dumps(s.evidence, default_handler=str)


@pytest.mark.parametrize("rule_id", IDS)
def test_silent_on_the_clean_twin(rule_id, cfg, registry, unlocks):
    builder, _, _ = SCENARIOS[rule_id](False)
    assert _run(rule_id, builder.context(cfg), registry, unlocks) == []


@pytest.mark.parametrize("rule_id", IDS)
def test_empty_tables_return_nothing(rule_id, cfg, registry, unlocks):
    ds = CanonicalDataset.empty(source_system="fixture", tenant_id=T)
    ctx = ControlContext(config=cfg, tenant_id=T, claims=pd.DataFrame(columns=["claim_sk", "tenant_id"]),
                         features=None, dataset=ds, run_date=_dt.date(2026, 1, 1))
    control = registry.get(rule_id)
    impl = cln.IMPLEMENTATIONS[unlocks[rule_id].implementation]
    assert impl(ctx, control) == []
    assert impl.__wrapped__(ctx, control) == []  # the body itself copes, not only the guard


@pytest.mark.parametrize("rule_id", IDS)
def test_idempotent(rule_id, cfg, registry, unlocks):
    builder, _, _ = SCENARIOS[rule_id](True)
    ctx = builder.context(cfg)
    first = sorted(s.signal_id for s in _run(rule_id, ctx, registry, unlocks))
    second = sorted(s.signal_id for s in _run(rule_id, builder.context(cfg), registry, unlocks))
    assert first and first == second


def test_other_tenant_rows_are_invisible(cfg, registry, unlocks):
    builder, _, _ = s_cln08r01(True)
    for row in builder.lines:
        row["tenant_id"] = "T002"
    assert _run("CLN-08-R01", builder.context(cfg), registry, unlocks) == []


def test_exclusions_hold(cfg, registry, unlocks):
    # a laboratory test under a screening diagnosis is excluded from the order check
    b = Builder()
    b.add("clinician_roster", clinician_id="D1", provider_sk="P1", role="PHYSICIAN", tenant_id=T)
    b.line(b.claim("C1", "M1", "P1", 10, dx="Z13.1"), "83036", 110.0)
    assert _run("CLN-08-R01", b.context(cfg), registry, unlocks) == []
    # a referral into the provider means the work-up may have been done elsewhere
    b, _, _ = s_cln03r03(True)
    b.add("referral", referral_sk="R1", referrer_provider_sk="P9", recipient_provider_sk="P1", member_sk="M1",
          referral_date=_day(30), tenant_id=T)
    assert _run("CLN-03-R03", b.context(cfg), registry, unlocks) == []
    # a member's denial given long after the service is not relied on
    b, _, _ = s_cln06r04(True)
    b.extra["member_confirmation"][0]["response_date"] = _day(400)
    assert _run("CLN-06-R04", b.context(cfg), registry, unlocks) == []
    # an oncology centre ordering genetic tests is excluded
    b, _, _ = s_cln08r05(True)
    b.extra["provider"][0]["specialty"] = "ONCOLOGY"
    assert _run("CLN-08-R05", b.context(cfg), registry, unlocks) == []
