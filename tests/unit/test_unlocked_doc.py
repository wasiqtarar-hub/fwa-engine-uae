"""Unlocked DOC controls: each fires on a minimal planted fixture, is silent on its
clean twin, survives empty tables, and is idempotent."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import yaml

from fwa.canonical import CanonicalDataset
from fwa.engine.context import ControlContext
from fwa.engine.unlocked import doc as D

ROOT = Path(__file__).resolve().parents[2]


def _control(registry, rule_id):
    c = registry.get(rule_id)
    return c.model_copy(update={"data_support": c.data_support.__class__("PARTIAL"),
                                "data_support_reason": "unit-test fixture"})


def _ctx(config, tables: dict[str, pd.DataFrame], claims: pd.DataFrame) -> ControlContext:
    ds = CanonicalDataset.empty(source_system="fixture")
    ds.set("claim_header", claims, "POPULATED", "fixture")
    for name, frame in tables.items():
        ds.set(name, frame, "POPULATED", "fixture")
    return ControlContext(config=config, tenant_id="T001", claims=claims, features=None, dataset=ds,
                          run_date=pd.Timestamp("2026-09-20").date())


def _claims(n=3, **overrides):
    rows = []
    for i in range(n):
        rows.append({"claim_sk": f"C{i}", "member_sk": f"M{i}", "provider_sk": "P1", "tenant_id": "T001",
                     "service_date": pd.Timestamp("2025-01-10"), "submission_date": pd.Timestamp("2025-01-15"),
                     "claim_date": pd.Timestamp("2025-01-15"), "gross_amount_aed": 5000.0})
    df = pd.DataFrame(rows)
    for k, v in overrides.items():
        df[k] = v
    return df


REF = pd.DataFrame([
    {"activity_code": "SURG01", "description": "knee arthroscopy", "service_family": "surgery",
     "code_family": "surgery", "complexity": "high"},
    {"activity_code": "SURG02", "description": "shoulder arthroscopy", "service_family": "surgery",
     "code_family": "surgery", "complexity": "high"},
    {"activity_code": "IMG01", "description": "mri lumbar spine", "service_family": "imaging",
     "code_family": "imaging", "complexity": "medium"},
    {"activity_code": "CONS01", "description": "consultation", "service_family": "consultation",
     "code_family": "em", "complexity": "low"},
])


def _lines(codes_by_claim, auth=None):
    rows = []
    for claim, codes in codes_by_claim.items():
        for j, code in enumerate(codes):
            rows.append({"line_sk": f"{claim}-{j}", "claim_sk": claim, "activity_code": code,
                         "net_amount": 4000.0, "gross_amount": 4000.0, "tenant_id": "T001",
                         "authorization_id": (auth or {}).get(claim)})
    return pd.DataFrame(rows)


def _note(claim, member, body, *, doc_type="operative_note", sk=None, created="2025-01-12", prior=None,
          ocr=0.98, ref=None):
    text = (f"*** SYNTHETIC DOCUMENT ***\nOPERATIVE NOTE (SYNTHETIC)\nPatient reference: {member}\n"
            f"Claim reference: {claim}\n\n{body}\n--- END OF SYNTHETIC DOCUMENT ---\n")
    return {"document_sk": sk or f"D-{claim}", "claim_sk": claim, "member_sk": member, "provider_sk": "P1",
            "doc_type": doc_type, "language": "en", "text": text, "ocr_confidence": ocr,
            "created_at": pd.Timestamp(created), "version_no": 1 if prior is None else 2,
            "prior_document_sk": prior, "attachment_ref": ref, "is_synthetic": True, "tenant_id": "T001"}


def _assert_signal(sigs, subject, n=1):
    assert len(sigs) == n, [s.evidence.get("plain_language") for s in sigs]
    s = sigs[0]
    assert s.subject_id == subject
    assert s.evidence["plain_language"].strip()
    return s


# ------------------------------------------------------------------ DOC-01-R01

def _r01(present: bool):
    claims = _claims(2)
    lines = _lines({"C0": ["SURG01"], "C1": ["CONS01"]})
    policy = pd.DataFrame([{"service_family": "surgery", "activity_code": None, "doc_type": "operative note"}])
    docs = [_note("C1", "M1", "Seen in clinic.", doc_type="clinic_note", created="2025-04-01")]
    if present:
        docs.append(_note("C0", "M0", "Arthroscopy performed.", created="2025-01-12"))
    return lines, policy, pd.DataFrame(docs), claims


def test_doc_01_r01_fires_and_is_silent_when_document_present(config, registry):
    ctrl = _control(registry, "DOC-01-R01")
    lines, policy, docs, claims = _r01(False)
    tables = {"claim_line": lines, "document_requirement_policy": policy, "document": docs,
              "activity_code_reference": REF}
    s = _assert_signal(D.doc_01_r01_required_document_absent(_ctx(config, tables, claims), ctrl), "C0")
    assert s.evidence["required_document_types"] == ["operative_note"]
    lines, policy, docs, claims = _r01(True)
    tables.update(document=docs)
    assert D.doc_01_r01_required_document_absent(_ctx(config, tables, claims), ctrl) == []


def test_doc_01_r01_respects_the_deadline(config, registry):
    ctrl = _control(registry, "DOC-01-R01")
    lines, policy, docs, claims = _r01(False)
    docs["created_at"] = pd.Timestamp("2025-01-20")  # file "as of" only 5 days after submission
    tables = {"claim_line": lines, "document_requirement_policy": policy, "document": docs,
              "activity_code_reference": REF}
    assert D.doc_01_r01_required_document_absent(_ctx(config, tables, claims), ctrl) == []


# ------------------------------------------------------------------ DOC-01-R03

def _r03(supported: bool):
    claims = _claims(1)
    lines = _lines({"C0": ["IMG01"]})
    policy = pd.DataFrame([{"activity_code": "IMG01",
                            "required_terms": "radicular|neurological deficit;conservative treatment|physiotherapy"}])
    body = ("CLINICAL COURSE\nLow back pain for two weeks. There is no neurological deficit.\n"
            "Plan: MRI requested.")
    if supported:
        body = ("CLINICAL COURSE\nRadicular pain with a neurological deficit after six weeks of failed "
                "physiotherapy.\nPlan: MRI requested.")
    docs = pd.DataFrame([_note("C0", "M0", body, doc_type="clinical_note")])
    return {"claim_line": lines, "medical_necessity_policy": policy, "document": docs,
            "activity_code_reference": REF}, claims


def test_doc_01_r03_fires_with_span_and_is_silent_when_supported(config, registry):
    ctrl = _control(registry, "DOC-01-R03")
    tables, claims = _r03(False)
    s = _assert_signal(D.doc_01_r03_medical_necessity_absent(_ctx(config, tables, claims), ctrl), "C0")
    text = tables["document"].iloc[0]["text"]
    a, b = s.evidence["span_offsets"]
    assert text[a:b].strip() == s.evidence["source_span_text"]
    assert "no neurological deficit" in s.evidence["source_span_text"].lower()
    tables, claims = _r03(True)
    assert D.doc_01_r03_medical_necessity_absent(_ctx(config, tables, claims), ctrl) == []


# ------------------------------------------------------------------ DOC-01-R04

def _r04(drift: bool):
    claims = _claims(1)
    billed = "SURG02" if drift else "SURG01"
    if drift:
        billed = "IMG01"
    lines = _lines({"C0": [billed]}, auth={"C0": "A1"})
    auth = pd.DataFrame([{"authorization_sk": "A1", "request_id": "REQ1", "status": "approved",
                          "member_sk": "M0", "provider_sk": "P1", "tenant_id": "T001"}])
    auth_lines = pd.DataFrame([{"authorization_line_sk": "AL1", "authorization_sk": "A1",
                                "activity_code": "SURG01", "tenant_id": "T001"}])
    narrative = ("REQUEST NARRATIVE\nRequesting approval for knee arthroscopy (SURG01) for a torn "
                 "medial meniscus after failed physiotherapy.")
    docs = pd.DataFrame([_note("C0", "M0", narrative, doc_type="authorization_request", ref="REQ1")])
    return {"claim_line": lines, "authorization": auth, "authorization_line": auth_lines,
            "document": docs, "activity_code_reference": REF}, claims


def test_doc_01_r04_fires_on_different_service_and_is_silent_when_consistent(config, registry):
    ctrl = _control(registry, "DOC-01-R04")
    tables, claims = _r04(True)
    s = _assert_signal(D.doc_01_r04_authorization_narrative_drift(_ctx(config, tables, claims), ctrl), "C0")
    assert s.evidence["source_span_text"] in tables["document"].iloc[0]["text"]
    tables, claims = _r04(False)
    assert D.doc_01_r04_authorization_narrative_drift(_ctx(config, tables, claims), ctrl) == []


def test_doc_01_r04_sibling_code_is_not_material(config, registry):
    ctrl = _control(registry, "DOC-01-R04")
    tables, claims = _r04(False)
    tables["claim_line"]["activity_code"] = "SURG02"  # same family as the requested service
    assert D.doc_01_r04_authorization_narrative_drift(_ctx(config, tables, claims), ctrl) == []


# ------------------------------------------------------------------ DOC-02-R02

def _r02(wrong: bool):
    claims = _claims(1)
    members = pd.DataFrame([{"member_sk": "M0", "date_of_birth": pd.Timestamp("1990-03-01"), "sex": "M",
                             "tenant_id": "T001"}])
    body = ("Age: 67\nSex: Female\nCLINICAL COURSE\nKnee pain after a fall." if wrong
            else "Age: 34\nSex: Male\nCLINICAL COURSE\nKnee pain after a fall.")
    docs = pd.DataFrame([_note("C0", "M0", body)])
    return {"member": members, "document": docs}, claims


def test_doc_02_r02_fires_on_wrong_demographics_and_is_silent_on_correct(config, registry):
    ctrl = _control(registry, "DOC-02-R02")
    tables, claims = _r02(True)
    s = _assert_signal(D.doc_02_r02_contradictory_copied_facts(_ctx(config, tables, claims), ctrl), "C0")
    fields = {c["conflicting_field"] for c in s.evidence["conflicts"]}
    assert {"age", "sex"} <= fields
    text = tables["document"].iloc[0]["text"]
    a, b = s.evidence["span_offsets"]
    assert text[a:b].strip() == s.evidence["source_span_text"]
    tables, claims = _r02(False)
    assert D.doc_02_r02_contradictory_copied_facts(_ctx(config, tables, claims), ctrl) == []


def test_doc_02_r02_low_ocr_is_discarded(config, registry):
    ctrl = _control(registry, "DOC-02-R02")
    tables, claims = _r02(True)
    tables["document"]["ocr_confidence"] = 0.3
    assert D.doc_02_r02_contradictory_copied_facts(_ctx(config, tables, claims), ctrl) == []


# ------------------------------------------------------------------ DOC-02-R03

def _r03b(altered: bool, signed: bool = False):
    claims = _claims(1)
    remit = pd.DataFrame([{"remittance_sk": "R1", "claim_sk": "C0", "decision": "DENIED",
                           "denial_code": "MNEC-003", "settlement_date": pd.Timestamp("2025-02-10"),
                           "tenant_id": "T001"}])
    v1 = _note("C0", "M0", "Length of stay: 1 day(s)\nSeverity: mild\nCLINICAL COURSE\nUneventful.",
               sk="D1", created="2025-01-12")
    v2_body = ("Length of stay: 4 day(s)\nSeverity: severe\nCLINICAL COURSE\nUneventful." if altered
               else "Length of stay: 1 day(s)\nSeverity: mild\nCLINICAL COURSE\nUneventful. Typo corrected.")
    if signed:
        v2_body = "ADDENDUM\n" + v2_body + "\nSigned electronically by Dr A (SYNTHETIC)."
    v2 = _note("C0", "M0", v2_body, sk="D2", created="2025-02-20", prior="D1")
    return {"document": pd.DataFrame([v1, v2]), "remittance": remit}, claims


def test_doc_02_r03_fires_after_denial_and_respects_signed_addendum(config, registry):
    ctrl = _control(registry, "DOC-02-R03")
    tables, claims = _r03b(True)
    s = _assert_signal(D.doc_02_r03_post_denial_alteration(_ctx(config, tables, claims), ctrl), "C0")
    facts = {c["fact"] for c in s.evidence["changed_facts"]}
    assert "length of stay" in facts and "severity" in facts
    assert s.evidence["source_span_text"]
    tables, claims = _r03b(False)
    assert D.doc_02_r03_post_denial_alteration(_ctx(config, tables, claims), ctrl) == []
    tables, claims = _r03b(True, signed=True)
    assert D.doc_02_r03_post_denial_alteration(_ctx(config, tables, claims), ctrl) == []


# ------------------------------------------------------------------ DOC-02-R04

_RICH = ("CLINICAL COURSE\nArthroscopic examination revealed a complex tear of the posterior horn of the "
         "medial meniscus with grade two chondral softening over the medial femoral condyle. Partial "
         "meniscectomy was performed using a shaver and biter, preserving the peripheral rim. Tourniquet "
         "time was forty minutes, estimated blood loss minimal, portals closed with nylon sutures, and "
         "the patient was mobilised with crutches and a physiotherapy referral arranged.")
_THIN = "CLINICAL COURSE\nProcedure done. Patient stable."


def _r04b(generic: bool):
    rows, lines, notes = [], {}, []
    for i in range(12):
        provider = "P1" if i < 6 else f"P{i}"
        claim = f"C{i}"
        rows.append({"claim_sk": claim, "member_sk": f"M{i}", "provider_sk": provider, "tenant_id": "T001",
                     "service_date": pd.Timestamp("2025-01-10"), "gross_amount_aed": 9000.0})
        lines[claim] = ["SURG01"]
        body = _THIN if (generic and provider == "P1") else _RICH.replace("forty", f"forty{i}")
        notes.append(_note(claim, f"M{i}", body))
    claims = pd.DataFrame(rows)
    return {"claim_line": _lines(lines), "activity_code_reference": REF, "document": pd.DataFrame(notes)}, claims


def test_doc_02_r04_fires_on_generic_provider_and_is_silent_on_rich_notes(config, registry):
    ctrl = _control(registry, "DOC-02-R04")
    tables, claims = _r04b(True)
    s = _assert_signal(D.doc_02_r04_template_complexity_mismatch(_ctx(config, tables, claims), ctrl), "P1")
    assert s.subject_type == "provider"
    assert s.evidence["sample_generic_notes"][0]["source_span_text"]
    tables, claims = _r04b(False)
    assert D.doc_02_r04_template_complexity_mismatch(_ctx(config, tables, claims), ctrl) == []


# ------------------------------------------------------------------ generic properties

CASES = {
    "DOC-01-R01": (D.doc_01_r01_required_document_absent, lambda: (lambda t: ({"claim_line": t[0],
                   "document_requirement_policy": t[1], "document": t[2], "activity_code_reference": REF}, t[3]))(_r01(False))),
    "DOC-01-R03": (D.doc_01_r03_medical_necessity_absent, lambda: _r03(False)),
    "DOC-01-R04": (D.doc_01_r04_authorization_narrative_drift, lambda: _r04(True)),
    "DOC-02-R02": (D.doc_02_r02_contradictory_copied_facts, lambda: _r02(True)),
    "DOC-02-R03": (D.doc_02_r03_post_denial_alteration, lambda: _r03b(True)),
    "DOC-02-R04": (D.doc_02_r04_template_complexity_mismatch, lambda: _r04b(True)),
}


@pytest.mark.parametrize("rule_id", sorted(CASES))
def test_empty_tables_return_nothing(config, registry, rule_id):
    fn, _ = CASES[rule_id]
    ctx = _ctx(config, {}, pd.DataFrame(columns=["claim_sk", "member_sk", "provider_sk"]))
    assert fn(ctx, _control(registry, rule_id)) == []
    ctx2 = ControlContext(config=config, tenant_id="T001", claims=pd.DataFrame(), features=None, dataset=None)
    assert fn(ctx2, _control(registry, rule_id)) == []


@pytest.mark.parametrize("rule_id", sorted(CASES))
def test_idempotent(config, registry, rule_id):
    fn, build = CASES[rule_id]
    tables, claims = build()
    ctrl = _control(registry, rule_id)
    a = [s.signal_id for s in fn(_ctx(config, tables, claims), ctrl)]
    b = [s.signal_id for s in fn(_ctx(config, tables, claims), ctrl)]
    assert a and a == b


def test_unlock_yaml_matches_implementations():
    raw = yaml.safe_load((ROOT / "rules" / "unlocks" / "DOC.yaml").read_text(encoding="utf-8"))
    declared = {u["implementation"] for u in raw["unlocks"]}
    assert declared == set(D.IMPLEMENTATIONS)
    for u in raw["unlocks"]:
        assert u["implementation"] == D.IMPLEMENTATIONS[u["implementation"]].__name__
