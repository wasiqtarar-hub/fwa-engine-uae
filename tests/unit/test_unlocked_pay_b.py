"""Unit tests for the dataset-unlocked PAY-07 … PAY-12 implementations (``pay_b``).

Every control runs against a small hand-built multi-table world:

* the CLEAN world is internally consistent (shares match the benefit, every
  claim paid once, nothing cancelled, every override justified) and every
  control must be silent on it;
* each PLANT function adds one pattern to a copy of the clean world, and the
  control that looks for it must fire on the expected subject with a
  plain-language sentence;
* every control returns ``[]`` (and records no swallowed error) when the
  tables it needs are empty, and produces the same signal ids twice.
"""

from __future__ import annotations

import datetime as _dt
from pathlib import Path

import pandas as pd
import pytest
import yaml

from fwa.canonical import CanonicalDataset
from fwa.engine.context import ControlContext
from fwa.engine.unlocked import pay_b
from fwa.engine.unlocks import load_unlocks

ROOT = Path(__file__).resolve().parents[2]
T = "T001"
D0 = _dt.date(2025, 1, 1)
N_CLAIMS = 120
PROVIDERS = ["P1", "P2", "P3", "P4"]


# --------------------------------------------------------------------------- the clean world


def _d(offset: int) -> pd.Timestamp:
    return pd.Timestamp(D0 + _dt.timedelta(days=offset))


def clean_world() -> dict[str, pd.DataFrame]:
    heads, lines, remit, versions, events, receipts, conf, dx, enc = [], [], [], [], [], [], [], [], []
    for i in range(N_CLAIMS):
        c = f"C{i:03d}"
        prov = PROVIDERS[i % 4]
        mem = f"M{i % 8}"
        sd = _d(i)
        heads.append({"claim_sk": c, "tenant_id": T, "member_sk": mem, "provider_sk": prov, "payer_id": "PAYER1",
                      "service_date": sd, "gross_amount": 230.0, "net_amount": 184.0, "patient_share": 46.0,
                      "discount": 0.0, "gross_amount_aed": 230.0, "approved_amount_aed": 184.0,
                      "accident_indicator": False, "diagnosis_primary": "J06.9", "claim_type": "OUTPATIENT"})
        lines.append({"line_sk": f"L{c}", "claim_sk": c, "tenant_id": T, "activity_type": "CPT",
                      "activity_code": "99213", "units": 1.0, "gross_amount": 230.0, "net_amount": 184.0,
                      "patient_share": 46.0, "service_date": sd,
                      "activity_description": "Office or other outpatient visit, established patient, level 3"})
        remit.append({"remittance_sk": f"R{c}", "claim_sk": c, "line_sk": f"L{c}", "decision": "PAID",
                      "denial_code": None, "adjustment": 0.0, "payment_amount": 184.0,
                      "payment_reference": f"PAYREF{i:04d}", "settlement_date": sd + pd.Timedelta(days=20),
                      "tenant_id": T})
        versions.append({"claim_sk": c, "version_no": 1, "relationship": "ORIGINAL", "prior_claim_sk": None,
                         "changed_fields": "{}", "recorded_at": sd, "resubmission_type": None, "tenant_id": T})
        events.append({"event_sk": f"E{c}", "claim_sk": c, "actor": f"A{i % 3 + 1}", "actor_role": "ADJUDICATOR",
                       "event_type": "DECISION", "decision": "APPROVED", "override_reason": None,
                       "amount_before": 184.0, "amount_after": 184.0, "event_time": sd + pd.Timedelta(days=5),
                       "system_edit_result": "PASS", "tenant_id": T})
        receipts.append({"receipt_sk": f"RC{c}", "member_sk": mem, "claim_sk": c, "provider_sk": prov,
                         "amount_paid_aed": 46.0, "item_description": "Co-payment for office visit, established patient",
                         "receipt_date": sd, "tenant_id": T})
        conf.append({"confirmation_sk": f"MC{c}", "member_sk": mem, "claim_sk": c, "service_confirmed": True,
                     "response": "Yes, I attended", "response_date": sd + pd.Timedelta(days=10),
                     "channel": "SMS", "tenant_id": T})
        dx.append({"claim_sk": c, "code": "J06.9", "code_system": "ICD10CM", "sequence": 1, "tenant_id": T})
        enc.append({"encounter_sk": f"EN{c}", "claim_sk": c, "encounter_type": "OUTPATIENT", "tenant_id": T})
    w = {
        "claim_header": pd.DataFrame(heads), "claim_line": pd.DataFrame(lines), "remittance": pd.DataFrame(remit),
        "claim_version": pd.DataFrame(versions), "adjudication_event": pd.DataFrame(events),
        "member_receipt": pd.DataFrame(receipts), "member_confirmation": pd.DataFrame(conf),
        "diagnosis": pd.DataFrame(dx), "encounter": pd.DataFrame(enc),
        "activity_code_reference": pd.DataFrame([
            {"activity_code": "99213", "description": "Office visit, established patient, level 3",
             "service_family": "CONSULTATION", "code_family": "EM_OFFICE_EST"},
            {"activity_code": "99214", "description": "Office visit, established patient, level 4",
             "service_family": "CONSULTATION", "code_family": "EM_OFFICE_EST"},
            {"activity_code": "97110", "description": "Therapeutic exercise",
             "service_family": "PHYSIOTHERAPY", "code_family": "PT_THERAPY"},
            {"activity_code": "15823", "description": "Blepharoplasty, upper eyelid",
             "service_family": "DAY_SURGERY", "code_family": "SURG_EYE"},
        ]),
        "coverage_period": pd.DataFrame([
            {"coverage_id": f"CV{m}", "member_sk": f"M{m}", "product": "GOLD", "payer_id": "PAYER1",
             "valid_from": _d(-400), "valid_to": _d(400), "tenant_id": T} for m in range(8)]),
        "benefit_rule_version": pd.DataFrame([
            {"product": "GOLD", "service_family": f, "covered": True, "benefit_limit": 100000.0,
             "patient_share_pct": 20.0, "authorization_required": False, "exceptions": None,
             "valid_from": _d(-800), "valid_to": _d(800)}
            for f in ("CONSULTATION", "PHYSIOTHERAPY", "DAY_SURGERY")]),
        "disguise_risk_policy": pd.DataFrame([
            {"activity_code": "15823", "excluded_service": "Cosmetic upper-eyelid surgery (blepharoplasty)",
             "risk_diagnosis_prefixes": "Z41.1;L57;L90"}]),
        "document": pd.DataFrame([
            {"document_sk": "DOC1", "claim_sk": "C001", "doc_type": "clinical_note", "language": "en",
             "text": "Sore throat and cough for three days. Not cosmetic. Advised fluids.",
             "ocr_confidence": 0.95, "tenant_id": T}]),
        # C020 is a coordinated claim: the other insurer paid 100, we paid 84.
        "other_payer_remittance": pd.DataFrame([
            {"claim_sk": "C020", "other_payer_id": "OTHER1", "payment_amount": 100.0,
             "settlement_date": _d(25), "tenant_id": T}]),
        # C021 is an accident with its liable party recorded.
        "third_party_liability": pd.DataFrame([
            {"claim_sk": "C021", "member_sk": "M5", "accident_type": "ROAD", "liable_party": "MOTOR-INS-7",
             "reported_date": _d(21), "tenant_id": T}]),
        # C022: third party settled 100 first; we paid only the remaining 84.
        "third_party_settlement": pd.DataFrame([
            {"claim_sk": "C022", "settlement_amount": 100.0, "settlement_date": _d(23), "payer": "MOTOR-INS-7",
             "tenant_id": T}]),
        "recovery_ledger": pd.DataFrame([
            {"ledger_sk": "LG1", "provider_sk": "P1", "claim_sk": "C004", "credit_amount": 120.0,
             "credit_date": _d(30), "applied": True, "applied_date": _d(45), "tenant_id": T}]),
    }
    h = w["claim_header"]
    h.loc[h["claim_sk"] == "C021", "accident_indicator"] = True
    r = w["remittance"]
    r.loc[r["claim_sk"].isin(["C020", "C022"]), "payment_amount"] = 84.0
    return w


# --------------------------------------------------------------------------- helpers


@pytest.fixture(scope="module")
def controls(registry):
    return {rid: registry.get(rid) for rid in CASES}


def make_ctx(config, tables: dict[str, pd.DataFrame]) -> ControlContext:
    ds = CanonicalDataset.empty(source_system="fixture", tenant_id=T)
    for name, frame in tables.items():
        ds.set(name, frame, "POPULATED", "fixture")
    return ControlContext(config=config, tenant_id=T, claims=ds.get("claim_header"), features=None,
                          dataset=ds, run_date=_dt.date(2026, 9, 20))


def add_rows(w, table, rows):
    w[table] = pd.concat([w[table], pd.DataFrame(rows)], ignore_index=True)


def clone_claim(w, src: str, new: str, *, days: int = 3, code: str | None = None, dx: str | None = None,
                paid: float | None = 184.0, prior: str | None = None, rel: str = "RESUBMISSION",
                changed: str = "{}", rtype: str | None = None):
    h = w["claim_header"]
    row = h[h["claim_sk"] == src].iloc[0].to_dict()
    sd = pd.Timestamp(row["service_date"]) + pd.Timedelta(days=days)
    row.update(claim_sk=new, service_date=sd)
    if dx:
        row["diagnosis_primary"] = dx
    add_rows(w, "claim_header", [row])
    line = w["claim_line"][w["claim_line"]["claim_sk"] == src].iloc[0].to_dict()
    line.update(line_sk=f"L{new}", claim_sk=new, service_date=sd)
    if code:
        line["activity_code"] = code
    add_rows(w, "claim_line", [line])
    add_rows(w, "diagnosis", [{"claim_sk": new, "code": dx or "J06.9", "sequence": 1, "tenant_id": T}])
    add_rows(w, "claim_version", [{"claim_sk": new, "version_no": 2, "relationship": rel, "prior_claim_sk": prior,
                                   "changed_fields": changed, "recorded_at": sd, "resubmission_type": rtype,
                                   "tenant_id": T}])
    if paid is not None:
        add_rows(w, "remittance", [{"remittance_sk": f"R{new}", "claim_sk": new, "line_sk": f"L{new}",
                                    "decision": "PAID" if paid > 0 else "DENIED",
                                    "denial_code": None if paid > 0 else "MNEC-003", "adjustment": 0.0,
                                    "payment_amount": paid, "payment_reference": f"PAYREF-{new}",
                                    "settlement_date": sd + pd.Timedelta(days=20), "tenant_id": T}])


def deny(w, claim: str, code: str = "MNEC-003"):
    r = w["remittance"]
    m = r["claim_sk"] == claim
    r.loc[m, "decision"] = "DENIED"
    r.loc[m, "denial_code"] = code
    r.loc[m, "payment_amount"] = 0.0


def set_val(w, table, key, keyval, **values):
    f = w[table]
    for col, v in values.items():
        f.loc[f[key] == keyval, col] = v


# --------------------------------------------------------------------------- plants


def plant_07_r01(w):
    set_val(w, "claim_header", "claim_sk", "C000", patient_share=10.0, net_amount=220.0)
    return "C000"


def plant_07_r02(w):
    set_val(w, "claim_header", "claim_sk", "C001", net_amount=230.0)
    return "C001"


def plant_07_r03(w):
    h = w["claim_header"]
    h.loc[h["provider_sk"] == "P2", "patient_share"] = 0.0
    return "P2"


def plant_07_r04(w):
    set_val(w, "member_receipt", "claim_sk", "C002", amount_paid_aed=146.0)
    return "C002"


def plant_08_r01(w):
    add_rows(w, "claim_version", [{"claim_sk": "C003", "version_no": 2, "relationship": "RESUBMISSION",
                                   "prior_claim_sk": "CX-NOT-THERE", "changed_fields": "{}",
                                   "recorded_at": _d(5), "resubmission_type": "CORRECTED", "tenant_id": T}])
    return "C003"


def plant_08_r02(w):
    deny(w, "C004")
    clone_claim(w, "C004", "C004R", code="99214", dx="M54.5", prior="C004",
                changed='{"diagnosis_primary": ["J06.9", "M54.5"], "activity_code": ["99213", "99214"]}')
    return "C004R"


def plant_08_r03(w):
    deny(w, "C005")
    prev = "C005"
    for k, fields in enumerate(['{"diagnosis": 1}', '{"activity_code": 1}', '{"units": 1}', '{"provider": 1}']):
        new = f"C005R{k}"
        clone_claim(w, "C005", new, days=3 * (k + 1), prior=prev, changed=fields,
                    paid=184.0 if k == 3 else 0.0)
        prev = new
    return prev


def plant_08_r04(w):
    p1 = [c for c in w["claim_header"]["claim_sk"] if w["claim_header"].set_index("claim_sk").at[c, "provider_sk"] == "P1"]
    p3 = [c for c in w["claim_header"]["claim_sk"] if w["claim_header"].set_index("claim_sk").at[c, "provider_sk"] == "P3"]
    for c in p1[:9]:
        deny(w, c)
        clone_claim(w, c, c + "M", dx="M54.5", prior=c, changed='{"diagnosis_primary": ["J06.9", "M54.5"]}')
    for j, c in enumerate(p3[:9]):
        deny(w, c)
        clone_claim(w, c, c + "M", dx="M54.5", prior=c, changed='{"diagnosis_primary": ["J06.9", "M54.5"]}',
                    paid=184.0 if j < 2 else 0.0)
    return "P1"


def plant_09_r01(w):
    add_rows(w, "document", [{"document_sk": "DOC2", "claim_sk": "C006", "doc_type": "operative_note",
                              "language": "en", "text": "Botox injections to forehead lines for cosmetic "
                                                        "improvement at the patient's request.",
                              "ocr_confidence": 0.9, "tenant_id": T}])
    return "C006"


def plant_09_r02(w):
    h = w["claim_header"].set_index("claim_sk")
    p2 = [c for c in h.index if h.at[c, "provider_sk"] == "P2"][:8]
    p3 = [c for c in h.index if h.at[c, "provider_sk"] == "P3"][:8]
    for c in p2:
        deny(w, c, "NCOV-001")
        clone_claim(w, c, c + "S", days=5, code="97110", prior=None, rel="ORIGINAL")
    for c in p3:
        deny(w, c, "NCOV-001")
    return "P2"


def plant_09_r03(w):
    set_val(w, "claim_line", "claim_sk", "C007", activity_code="15823", gross_amount=5000.0, net_amount=4000.0)
    set_val(w, "diagnosis", "claim_sk", "C007", code="Z41.1")
    return "C007"


def plant_09_r04(w):
    set_val(w, "member_confirmation", "claim_sk", "C008", service_confirmed=False,
            response="I did not visit this clinic in January")
    return "C008"


def plant_10_r02(w):
    add_rows(w, "other_payer_remittance", [{"claim_sk": "C009", "other_payer_id": "OTHER1",
                                            "payment_amount": 230.0, "settlement_date": _d(15), "tenant_id": T}])
    return "C009"


def plant_10_r03(w):
    set_val(w, "claim_header", "claim_sk", "C010", accident_indicator=True)
    return "C010"


def plant_10_r04(w):
    add_rows(w, "third_party_settlement", [{"claim_sk": "C011", "settlement_amount": 150.0,
                                            "settlement_date": _d(12), "payer": "MOTOR-INS-2", "tenant_id": T}])
    return "C011"


def plant_11_r01(w):
    set_val(w, "adjudication_event", "claim_sk", "C012", event_type="OVERRIDE", override_reason=None,
            system_edit_result="FAIL", amount_before=0.0, amount_after=184.0)
    return "C012"


def plant_11_r02(w):
    h = w["claim_header"].set_index("claim_sk")
    p3 = [c for c in h.index if h.at[c, "provider_sk"] == "P3"][:10]
    p1 = [c for c in h.index if h.at[c, "provider_sk"] == "P1"][:10]
    rows = [{"event_sk": f"X{c}", "claim_sk": c, "actor": "A9", "actor_role": "SUPERVISOR", "event_type": "OVERRIDE",
             "decision": "APPROVED", "override_reason": "clinical justification", "amount_before": 0.0,
             "amount_after": 184.0, "event_time": _d(50), "system_edit_result": "FAIL", "tenant_id": T} for c in p3]
    rows += [{"event_sk": f"Y{c}", "claim_sk": c, "actor": "A9", "actor_role": "SUPERVISOR", "event_type": "DECISION",
              "decision": "APPROVED", "override_reason": None, "amount_before": 184.0, "amount_after": 184.0,
              "event_time": _d(50), "system_edit_result": "PASS", "tenant_id": T} for c in p1]
    add_rows(w, "adjudication_event", rows)
    return "A9"


def plant_11_r03(w):
    add_rows(w, "remittance", [{"remittance_sk": "RUP", "claim_sk": "C013", "line_sk": "LC013", "decision": "PAID",
                                "denial_code": None, "adjustment": 100.0, "payment_amount": 100.0,
                                "payment_reference": "PAYREF-UP", "settlement_date": _d(13 + 60), "tenant_id": T}])
    return "C013"


def plant_11_r04(w):
    h = w["claim_header"].set_index("claim_sk")
    p4 = [c for c in h.index if h.at[c, "provider_sk"] == "P4"][:14]
    p2 = [c for c in h.index if h.at[c, "provider_sk"] == "P2"][:14]
    rows = [{"event_sk": f"F{c}", "claim_sk": c, "actor": "A8", "actor_role": "SUPERVISOR", "event_type": "DECISION",
             "decision": "APPROVED", "override_reason": "reviewed", "amount_before": 184.0, "amount_after": 184.0,
             "event_time": _d(60), "system_edit_result": "WARN", "tenant_id": T} for c in p4]
    rows += [{"event_sk": f"G{c}", "claim_sk": c, "actor": "A1", "actor_role": "SUPERVISOR", "event_type": "DECISION",
              "decision": "PARTIAL", "override_reason": "reviewed", "amount_before": 184.0, "amount_after": 92.0,
              "event_time": _d(60), "system_edit_result": "WARN", "tenant_id": T} for c in p2]
    add_rows(w, "adjudication_event", rows)
    return "A8"


def plant_12_r01(w):
    add_rows(w, "claim_version", [{"claim_sk": "C014", "version_no": 2, "relationship": "CANCELLATION",
                                   "prior_claim_sk": None, "changed_fields": "{}", "recorded_at": _d(40),
                                   "resubmission_type": None, "tenant_id": T}])
    return "C014"


def plant_12_r02(w):
    add_rows(w, "remittance", [{"remittance_sk": "RDUP", "claim_sk": "C015", "line_sk": "LC015", "decision": "PAID",
                                "denial_code": None, "adjustment": 0.0, "payment_amount": 184.0,
                                "payment_reference": "PAYREF-DUP", "settlement_date": _d(50), "tenant_id": T}])
    return "C015"


def plant_12_r03(w):
    add_rows(w, "recovery_ledger", [{"ledger_sk": "LG2", "provider_sk": "P3", "claim_sk": "C016",
                                     "credit_amount": 500.0, "credit_date": _d(5), "applied": False,
                                     "applied_date": None, "tenant_id": T}])
    return "P3"


def plant_12_r04(w):
    h = w["claim_header"].set_index("claim_sk")
    p4 = [c for c in h.index if h.at[c, "provider_sk"] == "P4"][:4]
    add_rows(w, "remittance", [{"remittance_sk": f"NEG{c}", "claim_sk": c, "line_sk": f"L{c}", "decision": "PAID",
                                "denial_code": None, "adjustment": -500.0, "payment_amount": -500.0,
                                "payment_reference": f"OFF{c}", "settlement_date": _d(70), "tenant_id": T}
                               for c in p4])
    return "P4"


CASES = {
    "PAY-07-R01": ("pay_07_r01_expected_share_mismatch", plant_07_r01),
    "PAY-07-R02": ("pay_07_r02_share_shifted_to_payer", plant_07_r02),
    "PAY-07-R03": ("pay_07_r03_systematic_zero_share", plant_07_r03),
    "PAY-07-R04": ("pay_07_r04_balance_billing", plant_07_r04),
    "PAY-08-R01": ("pay_08_r01_broken_lineage", plant_08_r01),
    "PAY-08-R02": ("pay_08_r02_field_mutation", plant_08_r02),
    "PAY-08-R03": ("pay_08_r03_resubmission_loop", plant_08_r03),
    "PAY-08-R04": ("pay_08_r04_edit_learning", plant_08_r04),
    "PAY-09-R01": ("pay_09_r01_document_conflict", plant_09_r01),
    "PAY-09-R02": ("pay_09_r02_covered_code_substitution", plant_09_r02),
    "PAY-09-R03": ("pay_09_r03_cosmetic_disguise", plant_09_r03),
    "PAY-09-R04": ("pay_09_r04_member_confirmation_mismatch", plant_09_r04),
    "PAY-10-R02": ("pay_10_r02_multi_payer_overpayment", plant_10_r02),
    "PAY-10-R03": ("pay_10_r03_accident_without_liability", plant_10_r03),
    "PAY-10-R04": ("pay_10_r04_paid_after_settlement", plant_10_r04),
    "PAY-11-R01": ("pay_11_r01_unauthorised_override", plant_11_r01),
    "PAY-11-R02": ("pay_11_r02_adjudicator_provider_concentration", plant_11_r02),
    "PAY-11-R03": ("pay_11_r03_post_settlement_increase", plant_11_r03),
    "PAY-11-R04": ("pay_11_r04_invariant_full_pay", plant_11_r04),
    "PAY-12-R01": ("pay_12_r01_paid_cancelled_claim", plant_12_r01),
    "PAY-12-R02": ("pay_12_r02_duplicate_payment", plant_12_r02),
    "PAY-12-R03": ("pay_12_r03_unapplied_refund", plant_12_r03),
    "PAY-12-R04": ("pay_12_r04_offset_manipulation", plant_12_r04),
}


def _run(config, control, impl, tables):
    pay_b.LAST_ERRORS.pop(impl, None)
    sigs = pay_b.IMPLEMENTATIONS[impl](make_ctx(config, tables), control)
    assert impl not in pay_b.LAST_ERRORS, pay_b.LAST_ERRORS.get(impl)
    return sigs


# --------------------------------------------------------------------------- tests


@pytest.mark.parametrize("rule_id", sorted(CASES))
def test_fires_on_planted_pattern(config, controls, rule_id):
    impl, plant = CASES[rule_id]
    w = clean_world()
    subject = plant(w)
    sigs = _run(config, controls[rule_id], impl, w)
    assert sigs, f"{rule_id} did not fire on its planted pattern"
    assert subject in {s.subject_id for s in sigs}
    for s in sigs:
        text = s.evidence.get("plain_language", "")
        assert text and "fraud" not in text.lower() and rule_id not in text
        assert s.rule_id == rule_id


@pytest.mark.parametrize("rule_id", sorted(CASES))
def test_silent_on_clean_world(config, controls, rule_id):
    impl, _ = CASES[rule_id]
    assert _run(config, controls[rule_id], impl, clean_world()) == []


@pytest.mark.parametrize("rule_id", sorted(CASES))
def test_empty_tables_return_nothing(config, controls, rule_id):
    impl, _ = CASES[rule_id]
    assert _run(config, controls[rule_id], impl, {}) == []
    # header only, every other table empty
    assert _run(config, controls[rule_id], impl, {"claim_header": clean_world()["claim_header"]}) == []


@pytest.mark.parametrize("rule_id", sorted(CASES))
def test_idempotent(config, controls, rule_id):
    impl, plant = CASES[rule_id]
    w = clean_world()
    plant(w)
    a = [s.signal_id for s in _run(config, controls[rule_id], impl, w)]
    b = [s.signal_id for s in _run(config, controls[rule_id], impl, w)]
    assert a == b and len(set(a)) == len(a)


def test_exclusion_lookalikes_do_not_fire(config, controls):
    """Declared exclusions: legacy route, consented receipt, reconstructive diagnosis, emergency override,
    appeal-route attempts, coordinated secondary payment."""
    w = clean_world()
    add_rows(w, "claim_version", [{"claim_sk": "C003", "version_no": 2, "relationship": "RESUBMISSION",
                                   "prior_claim_sk": None, "changed_fields": "{}", "recorded_at": _d(5),
                                   "resubmission_type": "LEGACY", "tenant_id": T}])
    assert _run(config, controls["PAY-08-R01"], CASES["PAY-08-R01"][0], w) == []

    w = clean_world()
    add_rows(w, "member_receipt", [{"receipt_sk": "RCX", "member_sk": "M2", "claim_sk": "C002", "provider_sk": "P3",
                                    "amount_paid_aed": 300.0, "item_description": "Non-covered cosmetic cream, signed consent on file",
                                    "receipt_date": _d(2), "tenant_id": T}])
    assert _run(config, controls["PAY-07-R04"], CASES["PAY-07-R04"][0], w) == []

    w = clean_world()
    plant_09_r03(w)
    add_rows(w, "diagnosis", [{"claim_sk": "C007", "code": "C44.1", "sequence": 2, "tenant_id": T}])
    assert _run(config, controls["PAY-09-R03"], CASES["PAY-09-R03"][0], w) == []

    w = clean_world()
    set_val(w, "adjudication_event", "claim_sk", "C012", event_type="OVERRIDE", override_reason="Emergency escalation",
            system_edit_result="FAIL")
    assert _run(config, controls["PAY-11-R01"], CASES["PAY-11-R01"][0], w) == []

    w = clean_world()
    deny(w, "C005")
    prev = "C005"
    for k in range(4):
        new = f"C005A{k}"
        clone_claim(w, "C005", new, days=3 * (k + 1), prior=prev, changed="{}", rtype="APPEAL", paid=0.0)
        prev = new
    assert _run(config, controls["PAY-08-R03"], CASES["PAY-08-R03"][0], w) == []

    w = clean_world()
    set_val(w, "document", "document_sk", "DOC1", text="Eyelid surgery for visual field loss; no cosmetic intent.")
    assert _run(config, controls["PAY-09-R01"], CASES["PAY-09-R01"][0], w) == []


def test_share_below_benefit_carries_established_exposure(config, controls):
    w = clean_world()
    plant_07_r01(w)
    (sig,) = _run(config, controls["PAY-07-R01"], CASES["PAY-07-R01"][0], w)
    assert sig.exposure_established and round(sig.exposure_aed, 2) == 36.0
    assert "AED 46.00" in sig.evidence["plain_language"]


def test_statistical_controls_never_establish_exposure(config, controls):
    for rid in ("PAY-07-R03", "PAY-08-R04", "PAY-09-R02", "PAY-11-R02", "PAY-11-R04", "PAY-12-R04"):
        impl, plant = CASES[rid]
        w = clean_world()
        plant(w)
        for s in _run(config, controls[rid], impl, w):
            assert s.exposure_aed == 0.0 or not s.exposure_established or rid == "PAY-07-R03"


def test_tenant_rows_of_another_tenant_are_ignored(config, controls):
    w = clean_world()
    add_rows(w, "recovery_ledger", [{"ledger_sk": "LGX", "provider_sk": "P3", "claim_sk": "C016",
                                     "credit_amount": 500.0, "credit_date": _d(5), "applied": False,
                                     "applied_date": None, "tenant_id": "T999"}])
    assert _run(config, controls["PAY-12-R03"], CASES["PAY-12-R03"][0], w) == []


def test_unlock_yaml_and_implementations_match():
    raw = yaml.safe_load((ROOT / "rules" / "unlocks" / "PAY_B.yaml").read_text("utf-8"))
    declared = {u["rule_id"]: u["implementation"] for u in raw["unlocks"]}
    assert set(declared.values()) == set(pay_b.IMPLEMENTATIONS)
    assert set(declared) == set(CASES)
    assert {k: v[0] for k, v in CASES.items()} == declared
    unlocks = load_unlocks(ROOT / "rules")
    for rid in declared:
        assert unlocks[rid].implementation == declared[rid]


def test_every_scoped_control_is_statically_not_executable(registry):
    for rid in CASES:
        assert registry.get(rid).data_support.value == "NOT_EXECUTABLE_ON_THIS_DATASET"
