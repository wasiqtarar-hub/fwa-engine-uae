"""Unlocked PHR controls (src/fwa/engine/unlocked/phr.py) on small hand-built fixtures.

Each control: fires on a minimal planted fixture with the expected subject and a
plain-language sentence; is silent on the clean variant of the same fixture;
returns nothing (no exception) when its tables are empty; and is idempotent.
"""

from __future__ import annotations

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
from fwa.engine.unlocked import phr  # noqa: E402
from fwa.engine.unlocks import decide, runtime_view  # noqa: E402

T = "T001"


@pytest.fixture(scope="module")
def cfg():
    return load_config(ROOT / "config")


@pytest.fixture(scope="module")
def reg():
    return RuleRegistry.from_directory(ROOT / "rules")


def _ds(tables: dict[str, pd.DataFrame]) -> CanonicalDataset:
    ds = CanonicalDataset.empty(source_system="fixture", tenant_id=T)
    for name, frame in tables.items():
        ds.set(name, frame, "POPULATED", "fixture")
    return ds


def _claims_from(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    if "claim_header" in tables:
        return tables["claim_header"]
    ids = set()
    for name in ("prescription_dispense", "claim_line"):
        if name in tables:
            ids |= set(tables[name]["claim_sk"].dropna())
    rows = []
    rx = tables.get("prescription_dispense")
    for c in sorted(ids):
        member, date = "M1", pd.Timestamp("2025-03-01")
        if rx is not None and c in set(rx["claim_sk"]):
            r = rx[rx["claim_sk"] == c].iloc[0]
            member, date = r.get("member_sk", "M1"), pd.Timestamp(r["fill_date"])
        cl = tables.get("claim_line")
        if cl is not None and c in set(cl["claim_sk"]):
            r = cl[cl["claim_sk"] == c].iloc[0]
            member = r.get("_member", member)
            date = pd.Timestamp(r["service_date"])
            prov = r.get("_provider", "HOSP1")
        else:
            prov = "HOSP1"
        rows.append({"claim_sk": c, "tenant_id": T, "member_sk": member, "provider_sk": prov,
                     "payer_id": "PAY1", "service_date": date, "discharge_date": date,
                     "length_of_stay_days": 0, "diagnosis_primary": "I10", "claim_type": "outpatient",
                     "gross_amount_aed": 100.0})
    return pd.DataFrame(rows)


def _ctx(cfg, tables: dict[str, pd.DataFrame]) -> ControlContext:
    claims = _claims_from(tables)
    tables = {k: v.drop(columns=[c for c in v.columns if c.startswith("_")]) if k != "claim_header" else v
              for k, v in tables.items()}
    return ControlContext(config=cfg, tenant_id=T, claims=claims, features=None,
                          dataset=_ds({k: v for k, v in tables.items() if k != "claim_header"}),
                          run_date=pd.Timestamp("2026-09-20").date())


def _run(reg, cfg, rule_id: str, tables: dict[str, pd.DataFrame]):
    ctx = _ctx(cfg, tables)
    control = reg.get(rule_id)
    decision = decide(control, reg.unlocks.get(rule_id), ctx.dataset)
    view = runtime_view(control, decision)
    impl = phr.IMPLEMENTATIONS[reg.unlocks[rule_id].implementation]
    return impl(ctx, view), decision


# ---------------------------------------------------------------------------
# fixture builders
# ---------------------------------------------------------------------------

DRUGS = pd.DataFrame([
    {"product": "ATOR-B", "description": "Atorvastatin 20mg brand", "equivalence_group": "ATOR20", "strength_mg": 20,
     "form": "tablet", "unit_price": 5.0, "max_duration_days": None, "max_mg_per_kg_day": 1.0,
     "is_high_cost": False, "is_controlled": False, "indication_prefixes": "E78", "therapeutic_class": "statin"},
    {"product": "ATOR-G", "description": "Atorvastatin 20mg generic", "equivalence_group": "ATOR20", "strength_mg": 20,
     "form": "tablet", "unit_price": 1.0, "max_duration_days": None, "max_mg_per_kg_day": 1.0,
     "is_high_cost": False, "is_controlled": False, "indication_prefixes": "E78", "therapeutic_class": "statin"},
    {"product": "ROSU", "description": "Rosuvastatin 40mg", "equivalence_group": "ROSU40", "strength_mg": 40,
     "form": "tablet", "unit_price": 9.0, "max_duration_days": None, "max_mg_per_kg_day": 1.0,
     "is_high_cost": False, "is_controlled": False, "indication_prefixes": "E78", "therapeutic_class": "statin"},
    {"product": "AMOX", "description": "Amoxicillin 500mg", "equivalence_group": "AMOX500", "strength_mg": 500,
     "form": "capsule", "unit_price": 0.5, "max_duration_days": 14, "max_mg_per_kg_day": 50.0,
     "is_high_cost": False, "is_controlled": False, "indication_prefixes": "J", "therapeutic_class": "antibiotic"},
    {"product": "BIO", "description": "Adalimumab 40mg", "equivalence_group": "ADA40", "strength_mg": 40,
     "form": "injection", "unit_price": 1500.0, "max_duration_days": None, "max_mg_per_kg_day": 1.0,
     "is_high_cost": True, "is_controlled": False, "indication_prefixes": "M05;L40", "therapeutic_class": "biologic",
     "vial_size_mg": None},
    {"product": "MTX", "description": "Methotrexate 10mg", "equivalence_group": "MTX10", "strength_mg": 10,
     "form": "tablet", "unit_price": 0.8, "max_duration_days": None, "max_mg_per_kg_day": 1.0,
     "is_high_cost": False, "is_controlled": False, "indication_prefixes": "M05;L40", "therapeutic_class": "dmard"},
    {"product": "ONC", "description": "Oncology vial drug", "equivalence_group": "ONC", "strength_mg": 100,
     "form": "vial", "unit_price": 2.0, "max_duration_days": None, "max_mg_per_kg_day": 10.0,
     "is_high_cost": True, "is_controlled": False, "indication_prefixes": "C", "therapeutic_class": "oncology",
     "vial_size_mg": 100.0},
])


def _rx(n: int = 1, **over) -> dict:
    base = {"rx_sk": f"RX{n}", "claim_sk": f"C{n}", "tenant_id": T, "member_sk": "M1",
            "prescriber_id": "DR1", "pharmacy_id": "PH1", "prescribed_product": "ATOR-B",
            "billed_product": "ATOR-B", "dispensed_product": "ATOR-B", "prescribed_qty": 30.0,
            "dispensed_qty": 30.0, "billed_qty": 30.0, "days_supply": 30.0, "billed_amount": 150.0,
            "fill_date": "2025-03-01", "strength_mg": 20.0, "form": "tablet", "dose_mg_per_day": 20.0,
            "authorization_id": None, "line_sk": f"L{n}"}
    base.update(over)
    return base


def _frame(rows) -> pd.DataFrame:
    return pd.DataFrame(rows)


def _assert_fires(signals, subject_type: str, subject_id: str | None = None):
    assert signals, "expected at least one signal"
    s = signals[0]
    assert s.subject_type == subject_type
    if subject_id is not None:
        assert s.subject_id == subject_id
    text = s.evidence.get("plain_language", "")
    assert text and "fraud" not in text.lower() and "PHR-" not in text


def _ids(signals):
    return sorted(s.signal_id for s in signals)


# ---------------------------------------------------------------------------
# the unlock declaration and the implementation map agree
# ---------------------------------------------------------------------------


def test_unlock_yaml_and_implementations_match(reg):
    raw = yaml.safe_load((ROOT / "rules" / "unlocks" / "PHR.yaml").read_text("utf-8"))
    declared = {e["implementation"] for e in raw["unlocks"]}
    assert declared == set(phr.IMPLEMENTATIONS)
    for e in raw["unlocks"]:
        assert reg.get(e["rule_id"]).data_support.value == "NOT_EXECUTABLE_ON_THIS_DATASET"
    assert "PHR-03-R03" not in {e["rule_id"] for e in raw["unlocks"]}


# ---------------------------------------------------------------------------
# per-control cases: (rule_id, planted tables, clean tables, expected subject_type)
# ---------------------------------------------------------------------------


def _auth_tables(units: float = 30.0, product: str = "ATOR-B"):
    return {
        "authorization": _frame([{"authorization_sk": "A1", "status": "approved", "member_sk": "M1", "tenant_id": T}]),
        "authorization_line": _frame([{"authorization_line_sk": "AL1", "authorization_sk": "A1",
                                       "activity_code": product, "approved_units": units, "tenant_id": T}]),
    }


def case_phr_01_r01():
    planted = {"prescription_dispense": _frame([_rx(1, billed_product="ROSU", dispensed_product="ROSU",
                                                    billed_amount=270.0)]), "drug_policy": DRUGS}
    # generic substituted for the prescribed brand is an approved equivalent → silent
    clean = {"prescription_dispense": _frame([_rx(1, billed_product="ATOR-G", dispensed_product="ATOR-G")]),
             "drug_policy": DRUGS}
    return planted, clean, "claim"


def case_phr_01_r02():
    planted = {"prescription_dispense": _frame([_rx(1, billed_qty=90.0, dispensed_qty=90.0, billed_amount=450.0)]),
               "drug_policy": DRUGS, **_auth_tables()}
    # 32 units against 30: inside the pack-conversion tolerance
    clean = {"prescription_dispense": _frame([_rx(1, billed_qty=32.0)]), "drug_policy": DRUGS, **_auth_tables()}
    return planted, clean, "claim"


def case_phr_01_r03():
    inv = _frame([{"pharmacy_id": "PH1", "product": "ATOR-B", "period": "2025-03", "opening_stock": 500,
                   "purchased_qty": 500, "closing_stock": 970, "tenant_id": T}])
    planted = {"prescription_dispense": _frame([_rx(1, dispensed_product="ATOR-G")]),
               "pharmacy_inventory": inv, "drug_policy": DRUGS}
    clean = {"prescription_dispense": _frame([_rx(1)]), "pharmacy_inventory": inv, "drug_policy": DRUGS}
    return planted, clean, "claim"


def case_phr_01_r04():
    def receipt(text):
        return _frame([{"receipt_sk": "R1", "member_sk": "M1", "claim_sk": "C1", "provider_sk": "PH1",
                        "amount_paid_aed": 30.0, "item_description": text, "receipt_date": "2025-03-01",
                        "tenant_id": T}])
    planted = {"prescription_dispense": _frame([_rx(1)]), "member_receipt": receipt("Perfume 100ml gift set"),
               "drug_policy": DRUGS}
    clean = {"prescription_dispense": _frame([_rx(1)]), "member_receipt": receipt("Atorvastatin 20mg x30"),
             "drug_policy": DRUGS}
    return planted, clean, "pharmacy"


def _series(dates, **over):
    return [_rx(i + 1, fill_date=d, **over) for i, d in enumerate(dates)]


def case_phr_02_r01():
    planted = {"prescription_dispense": _frame(_series(["2025-01-01", "2025-01-11", "2025-01-21"])),
               "drug_policy": DRUGS}
    clean = {"prescription_dispense": _frame(_series(["2025-01-01", "2025-01-29", "2025-02-27"])),
             "drug_policy": DRUGS}
    return planted, clean, "member"


def case_phr_02_r02():
    amox = dict(prescribed_product="AMOX", billed_product="AMOX", dispensed_product="AMOX",
                days_supply=10.0, strength_mg=500.0, dose_mg_per_day=1500.0, billed_qty=30.0)
    planted = {"prescription_dispense": _frame(_series(["2025-01-01", "2025-01-11", "2025-01-21"], **amox)),
               "drug_policy": DRUGS}
    clean = {"prescription_dispense": _frame(_series(["2025-01-01"], **amox)), "drug_policy": DRUGS}
    return planted, clean, "member"


def _roster(n: int):
    roster = _frame([{"clinician_id": f"DR{i}", "provider_sk": f"CLIN{i}", "specialty": "GP", "emirate": "Dubai",
                      "tenant_id": T} for i in range(1, n + 1)])
    prov = _frame([{"provider_sk": f"CLIN{i}", "owner_entity_id": f"OWN{i}", "emirate": "Dubai",
                    "provider_type": "clinic", "tenant_id": T} for i in range(1, n + 1)]
                  + [{"provider_sk": f"PH{i}", "owner_entity_id": f"POWN{i}", "emirate": "Dubai",
                      "provider_type": "pharmacy", "tenant_id": T} for i in range(1, 6)])
    return roster, prov


def case_phr_02_r03():
    roster, prov = _roster(4)
    rows = [_rx(i + 1, fill_date=f"2025-01-{1 + 25 * i:02d}" if i < 2 else "2025-02-20", prescriber_id=f"DR{i + 1}")
            for i in range(3)]
    rows[2]["fill_date"] = "2025-03-10"
    planted = {"prescription_dispense": _frame(rows), "drug_policy": DRUGS, "clinician_roster": roster,
               "provider": prov}
    # the same three prescribers, but all at one facility (care team) → silent
    roster2 = roster.assign(provider_sk="CLIN1")
    clean = {"prescription_dispense": _frame(rows), "drug_policy": DRUGS, "clinician_roster": roster2,
             "provider": prov}
    return planted, clean, "member"


def case_phr_02_r04():
    inv = _frame([{"pharmacy_id": "PH9", "product": "X", "period": "2025-01", "opening_stock": 1,
                   "purchased_qty": 0, "closing_stock": 1, "tenant_id": T}])
    rows = [_rx(i + 1, fill_date=d, pharmacy_id=f"PH{i + 1}")
            for i, d in enumerate(["2025-01-01", "2025-01-05", "2025-01-09"])]
    planted = {"prescription_dispense": _frame(rows), "drug_policy": DRUGS, "pharmacy_inventory": inv}
    # partial fills split between pharmacies are excluded
    partial = [dict(r, dispensed_qty=10.0, billed_qty=10.0) for r in rows]
    clean = {"prescription_dispense": _frame(partial), "drug_policy": DRUGS, "pharmacy_inventory": inv}
    return planted, clean, "member"


def case_phr_03_r01():
    members = _frame([{"member_sk": "M1", "weight_kg": 20.0, "height_cm": 110.0, "date_of_birth": "2018-01-01",
                       "tenant_id": T}])
    amox = dict(prescribed_product="AMOX", billed_product="AMOX", dispensed_product="AMOX", days_supply=7.0,
                strength_mg=500.0, billed_qty=42.0)
    planted = {"prescription_dispense": _frame([_rx(1, dose_mg_per_day=3000.0, **amox)]),
               "drug_policy": DRUGS, "member": members}
    clean = {"prescription_dispense": _frame([_rx(1, dose_mg_per_day=900.0, **amox)]),
             "drug_policy": DRUGS, "member": members}
    return planted, clean, "claim"


def case_phr_03_r02():
    step = _frame([{"product": "BIO", "required_prior_group": "MTX10", "lookback_days": 180}])
    dx = _frame([{"claim_sk": f"C{i}", "code": "M05.9", "sequence": 1, "tenant_id": T} for i in range(1, 4)])
    bio = dict(prescribed_product="BIO", billed_product="BIO", dispensed_product="BIO", billed_amount=3000.0)
    anchor = _rx(1, fill_date="2025-01-01", prescribed_product="ATOR-B")  # establishes history coverage
    planted_rows = [anchor, _rx(2, fill_date="2025-09-01", **bio)]
    clean_rows = [anchor, _rx(2, fill_date="2025-06-01", prescribed_product="MTX", billed_product="MTX",
                              dispensed_product="MTX"), _rx(3, fill_date="2025-09-01", **bio)]
    header = _frame([{"claim_sk": f"C{i}", "tenant_id": T, "member_sk": "M1", "provider_sk": "HOSP1",
                      "payer_id": "PAY1", "service_date": pd.Timestamp(d), "diagnosis_primary": "M05.9",
                      "length_of_stay_days": 0, "discharge_date": pd.Timestamp(d)}
                     for i, d in [(1, "2025-01-01"), (2, "2025-06-01"), (3, "2025-09-01")]])
    header.loc[0, "diagnosis_primary"] = "E78.0"
    planted = {"prescription_dispense": _frame(planted_rows), "drug_policy": DRUGS, "step_therapy_policy": step,
               "diagnosis": dx, "claim_header": header}
    clean = {"prescription_dispense": _frame(clean_rows), "drug_policy": DRUGS, "step_therapy_policy": step,
             "diagnosis": dx, "claim_header": header}
    return planted, clean, "claim"


def case_phr_03_r04():
    def world(bad_waste: float):
        lines, rx = [], []
        n = 0
        for p in range(1, 5):
            for k in range(6):
                n += 1
                waste = bad_waste if p == 1 else 50.0  # dose 250 mg → 3 vials → 50 mg expected waste
                lines.append({"line_sk": f"L{n}", "claim_sk": f"C{n}", "tenant_id": T, "product": "ONC",
                              "units": 250.0 + waste, "wastage_units": waste, "unit_price": 2.0,
                              "net_amount": 2.0 * (250.0 + waste), "service_date": "2025-04-01",
                              "activity_code": "J9999", "_provider": f"HOSP{p}", "_member": f"M{n}"})
                rx.append(_rx(n, member_sk=f"M{n}", billed_product="ONC", prescribed_product="ONC",
                              dispensed_product="ONC", dose_mg_per_day=250.0, fill_date="2025-04-01"))
        return {"claim_line": _frame(lines), "prescription_dispense": _frame(rx), "drug_policy": DRUGS}
    return world(350.0), world(50.0), "provider"


def _flows(top_share_rows: int, total: int = 25, n_presc: int = 6, pharmacy_top: str = "PH2"):
    rows, n = [], 0
    for d in range(1, n_presc + 1):
        for i in range(total):
            n += 1
            if d == 1:
                ph = pharmacy_top if i < top_share_rows else f"PH{3 + i % 3}"
            else:
                ph = f"PH{1 + i % 5}"
            rows.append(_rx(n, prescriber_id=f"DR{d}", pharmacy_id=ph, member_sk=f"M{n}",
                            fill_date=f"2025-{1 + i % 12:02d}-10"))
    return _frame(rows)


def case_phr_04_r01():
    roster, prov = _roster(6)
    planted = {"prescription_dispense": _flows(24), "clinician_roster": roster, "provider": prov}
    # the same concentration, but the pharmacy is the clinic's own (same owner) → on-site
    prov2 = prov.copy()
    prov2.loc[prov2["provider_sk"] == "PH2", "owner_entity_id"] = "OWN1"
    clean = {"prescription_dispense": _flows(24), "clinician_roster": roster, "provider": prov2}
    return planted, clean, "clinician"


def case_phr_04_r02():
    roster, prov = _roster(6)
    prov = prov.copy()
    prov["bank_account_token"] = prov["provider_sk"].map(lambda p: f"BANK-{p}")
    prov.loc[prov["provider_sk"] == "PH2", "bank_account_token"] = "BANK-CLIN1"
    planted = {"prescription_dispense": _flows(24), "clinician_roster": roster, "provider": prov}
    clean_prov = prov.copy()
    clean_prov.loc[clean_prov["provider_sk"] == "PH2", "owner_entity_id"] = "OWN1"  # declared corporate link
    clean = {"prescription_dispense": _flows(24), "clinician_roster": roster, "provider": clean_prov}
    return planted, clean, "provider"


def case_phr_04_r03():
    def world(steer: bool):
        rows, n = [], 0
        for i in range(10):
            n += 1
            rows.append(_rx(n, billed_product="BIO", prescribed_product="BIO", dispensed_product="BIO",
                            pharmacy_id="PH2" if steer else f"PH{1 + i % 4}", member_sk=f"M{n}"))
        for i in range(12):
            n += 1
            rows.append(_rx(n, pharmacy_id=f"PH{1 + i % 4}", member_sk=f"M{n}"))
        for p in range(1, 5):  # BIO is stocked widely, so availability does not explain it
            n += 1
            rows.append(_rx(n, prescriber_id="DR9", billed_product="BIO", prescribed_product="BIO",
                            dispensed_product="BIO", pharmacy_id=f"PH{p}", member_sk=f"M{n}"))
        return {"prescription_dispense": _frame(rows), "drug_policy": DRUGS}
    return world(True), world(False), "clinician"


def case_phr_04_r04():
    def world(ramp: bool):
        rows, n = [], 0
        for m in range(1, 13):
            for k in range(6):
                n += 1
                ph = "PH1"
                if ramp and m >= 8:
                    ph = "PH3" if k < 5 else "PH1"
                rows.append(_rx(n, pharmacy_id=ph, member_sk=f"M{n}", fill_date=f"2025-{m:02d}-{1 + k:02d}"))
        # PH3 has been dispensing for other prescribers all year (not a new site)
        for m in range(1, 13):
            n += 1
            rows.append(_rx(n, prescriber_id="DR7", pharmacy_id="PH3", member_sk=f"M{n}",
                            fill_date=f"2025-{m:02d}-15"))
        return {"prescription_dispense": _frame(rows)}
    return world(True), world(False), "clinician"


def _device_world(**serial_over):
    inv = _frame([{"serial_number": "SN1", "device_code": "E0100", "provider_sk": "HOSP1", "condition": "used",
                   "acquisition": "rental", "useful_life_days": 1825, "claim_sk": "C1", "line_sk": "L1",
                   "member_sk": "M1", "issued_date": "2025-02-01", "tenant_id": T, **serial_over}])
    lines = _frame([{"line_sk": "L1", "claim_sk": "C1", "tenant_id": T, "activity_code": "E0100",
                     "activity_description": "Wheelchair - new - purchase", "units": 1.0, "net_amount": 5000.0,
                     "gross_amount": 5000.0, "device_serial": "SN1", "service_date": "2025-02-01"}])
    return inv, lines


def case_phr_05_r01():
    inv, lines = _device_world()
    planted = {"claim_line": lines, "device_inventory": inv}
    inv2, _ = _device_world(condition="new", acquisition="purchase")
    clean = {"claim_line": lines, "device_inventory": inv2}
    return planted, clean, "claim"


def case_phr_05_r02():
    ref = _frame([{"activity_code": "IMP1", "description": "Knee implant", "is_implant": True,
                   "service_family": "implant"},
                  {"activity_code": "27447", "description": "Knee replacement", "is_implant": False,
                   "service_family": "surgery"},
                  {"activity_code": "99213", "description": "Office visit", "is_implant": False,
                   "service_family": "consultation"}])
    implant = {"line_sk": "L1", "claim_sk": "C1", "tenant_id": T, "activity_code": "IMP1", "units": 1.0,
               "net_amount": 12000.0, "gross_amount": 12000.0, "service_date": "2025-05-01"}
    visit = {"line_sk": "L2", "claim_sk": "C1", "tenant_id": T, "activity_code": "99213", "units": 1.0,
             "net_amount": 200.0, "gross_amount": 200.0, "service_date": "2025-05-01"}
    surgery = dict(visit, activity_code="27447", net_amount=20000.0)
    planted = {"claim_line": _frame([implant, visit]), "activity_code_reference": ref}
    clean = {"claim_line": _frame([implant, surgery]), "activity_code_reference": ref}
    return planted, clean, "claim"


def case_phr_05_r03():
    inv = _frame([{"serial_number": f"SN{i}", "device_code": "A4253", "provider_sk": "HOSP1", "condition": "new",
                   "acquisition": "purchase", "useful_life_days": 30, "tenant_id": T} for i in range(3)])

    def lines(bad: bool):
        rows = [{"line_sk": f"L{i}", "claim_sk": f"C{i}", "tenant_id": T, "activity_code": "A4253", "units": 2.0,
                 "net_amount": 100.0, "gross_amount": 100.0, "service_date": "2025-05-01"} for i in range(1, 13)]
        if bad:
            rows[0]["units"] = 20.0
            rows[0]["net_amount"] = 1000.0
        return _frame(rows)
    return ({"claim_line": lines(True), "device_inventory": inv},
            {"claim_line": lines(False), "device_inventory": inv}, "claim")


def case_phr_05_r04():
    inv = _frame([{"serial_number": "SN77", "device_code": "IMP1", "provider_sk": "HOSP1", "condition": "new",
                   "acquisition": "purchase", "useful_life_days": 3650, "tenant_id": T}])

    def lines(dup: bool):
        a = {"line_sk": "L1", "claim_sk": "C1", "tenant_id": T, "activity_code": "IMP1", "units": 1.0,
             "net_amount": 9000.0, "gross_amount": 9000.0, "service_date": "2025-05-01", "device_serial": "SN77",
             "_member": "M1"}
        b = dict(a, line_sk="L2", claim_sk="C2", service_date="2025-06-01", _member="M2",
                 device_serial="SN77" if dup else None)
        return _frame([a, b])
    return ({"claim_line": lines(True), "device_inventory": inv},
            {"claim_line": lines(False), "device_inventory": inv}, "provider")


CASES = {
    "PHR-01-R01": case_phr_01_r01, "PHR-01-R02": case_phr_01_r02, "PHR-01-R03": case_phr_01_r03,
    "PHR-01-R04": case_phr_01_r04, "PHR-02-R01": case_phr_02_r01, "PHR-02-R02": case_phr_02_r02,
    "PHR-02-R03": case_phr_02_r03, "PHR-02-R04": case_phr_02_r04, "PHR-03-R01": case_phr_03_r01,
    "PHR-03-R02": case_phr_03_r02, "PHR-03-R04": case_phr_03_r04, "PHR-04-R01": case_phr_04_r01,
    "PHR-04-R02": case_phr_04_r02, "PHR-04-R03": case_phr_04_r03, "PHR-04-R04": case_phr_04_r04,
    "PHR-05-R01": case_phr_05_r01, "PHR-05-R02": case_phr_05_r02, "PHR-05-R03": case_phr_05_r03,
    "PHR-05-R04": case_phr_05_r04,
}


def test_every_unlocked_control_has_a_case(reg):
    assert set(CASES) == set(u for u in reg.unlocks if u.startswith("PHR-"))


@pytest.mark.parametrize("rule_id", sorted(CASES))
def test_fires_on_planted_fixture(reg, cfg, rule_id):
    planted, _clean, subject = CASES[rule_id]()
    signals, decision = _run(reg, cfg, rule_id, planted)
    assert decision.basis == "unlocked", decision.missing
    _assert_fires(signals, subject)
    assert signals[0].rule_id == rule_id


@pytest.mark.parametrize("rule_id", sorted(CASES))
def test_silent_on_clean_fixture(reg, cfg, rule_id):
    _planted, clean, _subject = CASES[rule_id]()
    signals, _ = _run(reg, cfg, rule_id, clean)
    assert signals == [], [s.evidence.get("plain_language") for s in signals]


@pytest.mark.parametrize("rule_id", sorted(CASES))
def test_empty_tables_return_nothing(reg, cfg, rule_id):
    ctx = ControlContext(config=cfg, tenant_id=T, claims=pd.DataFrame(), features=None,
                         dataset=CanonicalDataset.empty(tenant_id=T), run_date=pd.Timestamp("2026-09-20").date())
    impl = phr.IMPLEMENTATIONS[reg.unlocks[rule_id].implementation]
    assert impl(ctx, reg.get(rule_id)) == []


@pytest.mark.parametrize("rule_id", sorted(CASES))
def test_idempotent(reg, cfg, rule_id):
    planted, _clean, _subject = CASES[rule_id]()
    a, _ = _run(reg, cfg, rule_id, planted)
    b, _ = _run(reg, cfg, rule_id, planted)
    assert _ids(a) == _ids(b) and a


def test_other_tenant_rows_are_invisible(reg, cfg):
    planted, _clean, _ = case_phr_01_r01()
    planted = {k: (v.assign(tenant_id="T999") if "tenant_id" in v.columns else v) for k, v in planted.items()}
    signals, _ = _run(reg, cfg, "PHR-01-R01", planted)
    assert signals == []
