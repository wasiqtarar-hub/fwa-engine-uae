"""Plant the DOC-family documentation patterns into the SYNTHETIC UAE demo dataset.

Every document touched or added here is SYNTHETIC (marked in its text and in
``is_synthetic``). Each pattern is planted at a handful of claims and recorded
in the answer key; where a control declares an exclusion, one or two
legitimate look-alikes are planted too (``positive=False``) so the exclusion is
visibly exercised.

Controls exercised: DOC-01-R01, DOC-01-R03, DOC-01-R04, DOC-02-R02, DOC-02-R03,
DOC-02-R04.
"""

from __future__ import annotations

import datetime as _dt
import re
from typing import Any

import pandas as pd

from .world import World

__all__ = ["inject"]

HEAD = "*** SYNTHETIC DOCUMENT — GENERATED FOR SOFTWARE TESTING. NOT A REAL CLINICAL RECORD. ***\n"
FOOT = "\n--- END OF SYNTHETIC DOCUMENT --- Every fact in this file is invented.\n"
TD = _dt.timedelta


# ----------------------------------------------------------------- helpers


def _docs(world: World) -> pd.DataFrame:
    return world.tables["document"]


def _has_doc(world: World, doc_type: str) -> pd.Series:
    return world.tables["claim_header"]["claim_sk"].isin(
        set(_docs(world).loc[_docs(world)["doc_type"] == doc_type, "claim_sk"]))


def _set_text(world: World, document_sk: str, text: str) -> None:
    d = _docs(world)
    d.loc[d["document_sk"] == document_sk, "text"] = text


def _new_doc(world: World, *, claim_sk: str, member_sk: str, provider_sk: str, doc_type: str, text: str,
             created: Any, language: str = "en", prior: str | None = None, version: int = 1,
             attachment: str | None = None) -> str:
    sk = world.new_id("DOCX", 6)
    world.append("document", [{
        "document_sk": sk, "claim_sk": claim_sk, "member_sk": member_sk, "provider_sk": provider_sk,
        "doc_type": doc_type, "language": language, "text": text,
        "ocr_confidence": round(float(world.rng.uniform(0.93, 0.995)), 3),
        "created_at": pd.Timestamp(created), "version_no": version, "prior_document_sk": prior,
        "attachment_ref": attachment or f"SYNTHETIC_{doc_type.lower()}_{claim_sk}_{sk}.txt",
        "is_synthetic": True, "tenant_id": "T001",
    }])
    return sk


def _as_of(world: World) -> pd.Timestamp:
    h = world.tables["claim_header"]
    return max(pd.to_datetime(h["submission_date"]).max(), pd.to_datetime(_docs(world)["created_at"]).max())


def _age(dob: Any, on: Any) -> int:
    dob, on = pd.Timestamp(dob), pd.Timestamp(on)
    return int(on.year - dob.year - ((on.month, on.day) < (dob.month, dob.day)))


def _service_date(world: World, claim_sk: str) -> pd.Timestamp:
    enc = world.tables["encounter"]
    row = enc[enc["claim_sk"] == claim_sk]
    if row.empty:
        return pd.Timestamp(world.tables["claim_header"].set_index("claim_sk").loc[claim_sk, "submission_date"])
    r = row.iloc[0]
    return pd.Timestamp(r["admission_date"]) if not pd.isna(r.get("admission_date")) else pd.Timestamp(r["start_time"]).normalize()


# ----------------------------------------------------------- DOC-01-R01


def _required_document_absent(world: World) -> None:
    as_of = _as_of(world)
    policy = world.tables["document_requirement_policy"]
    op_codes = set(policy.loc[policy["doc_type"] == "OPERATIVE_NOTE", "activity_code"].dropna())
    lines = world.tables["claim_line"]
    priced_op = set(lines.loc[lines["activity_code"].isin(op_codes) & (lines["gross_amount"] > 0), "claim_sk"])
    h = world.tables["claim_header"]
    # Only original claims that no later version replaces: a resubmission inherits the documents
    # already sent with the claim it replaces, so an absence there is not an absence.
    cv = world.tables.get("claim_version", pd.DataFrame(columns=["claim_sk", "prior_claim_sk"]))
    versioned = set(cv["prior_claim_sk"].dropna()) | set(cv.loc[cv["prior_claim_sk"].notna(), "claim_sk"])
    old = (pd.to_datetime(h["submission_date"]) < as_of - pd.Timedelta(days=90)) & ~h["claim_sk"].isin(versioned)

    picks = []
    surg = world.take_claims(4, lambda f: f["claim_sk"].isin(priced_op) & _has_doc(world, "OPERATIVE_NOTE") & old)
    picks += [(c, "OPERATIVE_NOTE") for c in surg["claim_sk"]]
    if not picks:  # no separately priced operation: an admission without its discharge summary instead
        adm = world.take_claims(3, lambda f: (f["claim_type"] == "INPATIENT") & _has_doc(world, "DISCHARGE_SUMMARY") & old)
        picks += [(c, "DISCHARGE_SUMMARY") for c in adm["claim_sk"]]
    rad = world.take_claims(8 - len(picks), lambda f: (f["claim_type"] == "RADIOLOGY")
                            & _has_doc(world, "IMAGING_REPORT") & old)
    picks += [(c, "IMAGING_REPORT") for c in rad["claim_sk"]]
    d = _docs(world)
    for claim, doc_type in picks:
        world.tables["document"] = d = d[~((d["claim_sk"] == claim) & (d["doc_type"] == doc_type))]
    if picks:
        world.record("DOC-01 required document never sent (operative note, discharge summary or imaging report)",
                     rule_ids=["DOC-01-R01"], claim_ids=[c for c, _ in picks])

    # Look-alike: the same absence on claims sent so recently that the deadline has not passed.
    bctx = world.context["base_ctx"]
    centres = [p for p, info in bctx.providers.items() if info["type"] == "RADIOLOGY_CENTRE"]
    service = _dt.date(2025, 12, 12)
    claimed = set(h["member_sk"])
    members = world.take_entities(
        "member", "member_sk", 2,
        lambda f: ~f["member_sk"].isin(claimed) & f["member_sk"].map(
            lambda m: bctx.members[m]["window"][0] <= service <= bctx.members[m]["window"][1]))
    fresh = []
    for msk in members["member_sk"]:
        lag = max(1, (as_of.date() - service).days - 8)
        emirate = bctx.members[msk]["emirate"]
        where = [p for p in centres if bctx.providers[p]["emirate"] == emirate] or centres
        try:
            claim = world.make_claim(member_sk=msk, provider_sk=where[0], service_date=service,
                                     claim_type="RADIOLOGY", submission_lag_days=lag)
        except RuntimeError:
            continue
        d = _docs(world)
        world.tables["document"] = d[~((d["claim_sk"] == claim) & (d["doc_type"] == "IMAGING_REPORT"))]
        fresh.append(claim)
    if fresh:
        world.record("DOC-01 report not yet due (inside the document deadline)", rule_ids=["DOC-01-R01"],
                     claim_ids=fresh, positive=False,
                     note="Latency exclusion: the claim was sent less than the deadline before the file's end.")


# ----------------------------------------------------------- DOC-01-R03


def _necessity_absent(world: World) -> None:
    necessity: dict[str, list[str]] = world.context.get("medical_necessity", {})
    lines = world.tables["claim_line"]
    nec_claims = set(lines.loc[lines["activity_code"].isin(necessity), "claim_sk"])
    taken = world.take_claims(
        8, lambda f: f["claim_sk"].isin(nec_claims) & (f["claim_type"] != "INPATIENT")
        & _has_doc(world, "CLINICAL_NOTE"))
    planted = []
    for i, claim in enumerate(taken["claim_sk"]):
        codes = [c for c in lines.loc[lines["claim_sk"] == claim, "activity_code"] if c in necessity]
        terms = list(dict.fromkeys(t for c in codes for t in necessity[c]))
        d = _docs(world)
        claim_docs = d[d["claim_sk"] == claim]
        note = claim_docs[claim_docs["doc_type"] == "CLINICAL_NOTE"].iloc[0]
        others = " ".join(claim_docs.loc[claim_docs["document_sk"] != note["document_sk"], "text"]).lower()
        droppable = [t for t in terms if t not in others]
        if not droppable:
            continue
        drop = droppable[-1]
        kept = [t for t in terms if t != drop]
        if i % 2 == 0:
            line = f"Clinical justification: {'; '.join(kept) or 'see history'}. No {drop} documented."
        else:
            line = f"Clinical justification: {'; '.join(kept) or 'see history'}."
        text = re.sub(r"Clinical justification:.*", line, note["text"])
        if drop in text.lower().replace(f"no {drop}", ""):
            continue
        _set_text(world, note["document_sk"], text)
        planted.append(claim)
    if planted:
        world.record("DOC-01 high-cost service without the required indication in the record",
                     rule_ids=["DOC-01-R03"], claim_ids=planted)


# ----------------------------------------------------------- DOC-01-R04


#: A plainer service that the request describes instead of the advanced one billed.
_REQUESTED_INSTEAD = {  # billed advanced scan -> the plainer service of ANOTHER family that was requested
    "72148": "72100", "73721": "73562", "73221": "73030", "70551": "95816", "70553": "95816",
    "74177": "76700", "74176": "76700", "71250": "71046", "71260": "71046", "78815": "76700",
    "70450": "95816",
}


def _auth_request_text(world: World, *, request_id: str, member: str, provider: str, dx: str,
                       code: str) -> str:
    codes = world.context["codes"]
    desc = codes.get(code, {}).get("description", code)
    icd = world.context["icd"].get(dx, (dx,))[0]
    return (HEAD + "PRIOR AUTHORISATION REQUEST (SYNTHETIC)\n"
            f"Request: {request_id}  Patient: {member}  Facility: {provider}\n"
            f"Diagnosis: {dx} {icd}\n\n"
            f"The patient presents with {icd.lower()}. We request approval for {code} {desc} to guide "
            f"further management; symptoms have persisted despite initial treatment.\n" + FOOT)


def _auth_narrative_drift(world: World) -> None:
    lines = world.tables["claim_line"]
    auth = world.tables["authorization"].set_index("authorization_sk")
    dx = world.tables["diagnosis"]
    principal = dx.sort_values("sequence").drop_duplicates("claim_sk").set_index("claim_sk")["code"]
    authed = lines[lines["authorization_id"].notna()]
    drift_ok = set(authed.loc[authed["activity_code"].isin(_REQUESTED_INSTEAD), "claim_sk"])
    other_ok = set(authed["claim_sk"])

    def plant(claims, drift: bool) -> list[str]:
        done = []
        for claim in claims:
            sub = authed[authed["claim_sk"] == claim]
            aid = sub["authorization_id"].iloc[0]
            if aid not in auth.index:
                continue
            a = auth.loc[aid]
            billed = sub["activity_code"].iloc[0]
            code = _REQUESTED_INSTEAD.get(billed, billed) if drift else billed
            if drift and code == billed:
                continue
            text = _auth_request_text(world, request_id=a["request_id"], member=a["member_sk"],
                                      provider=a["provider_sk"], dx=principal.get(claim, "R69"), code=code)
            _new_doc(world, claim_sk=claim, member_sk=a["member_sk"], provider_sk=a["provider_sk"],
                     doc_type="AUTHORIZATION_REQUEST", text=text,
                     created=pd.Timestamp(a["valid_from"]) - pd.Timedelta(hours=6),
                     attachment=a["request_id"])
            done.append(claim)
        return done

    pos = plant(world.take_claims(6, lambda f: f["claim_sk"].isin(drift_ok) & (f["claim_type"] != "INPATIENT"))
                ["claim_sk"], True)
    if pos:
        world.record("DOC-01 approval requested for a plain X-ray or ultrasound, advanced scan billed",
                     rule_ids=["DOC-01-R04"], claim_ids=pos)
    neg = plant(world.take_claims(12, lambda f: f["claim_sk"].isin(other_ok))["claim_sk"], False)
    if neg:
        world.record("DOC-01 approval request that describes exactly what was billed", rule_ids=["DOC-01-R04"],
                     claim_ids=neg, positive=False, note="Consistent request narratives: the control must stay silent.")


# ----------------------------------------------------------- DOC-02-R02


_SEX_EN = {"M": "Male", "F": "Female"}
_SEX_AR = {"M": "ذكر", "F": "أنثى"}


def _insert_demographics(text: str, age: int, sex: str) -> str:
    if "مرجع المريض" in text:
        return re.sub(r"(مرجع المريض:[^\n]*\n)", lambda m: m.group(1) + f"العمر: {age}\nالجنس: {_SEX_AR[sex]}\n",
                      text, count=1)
    return re.sub(r"(Patient reference:[^\n]*\n)", lambda m: m.group(1) + f"Age: {age}\nSex: {_SEX_EN[sex]}\n",
                  text, count=1)


def _copied_demographics(world: World) -> None:
    members = world.tables["member"].set_index("member_sk")
    taken = world.take_claims(30, lambda f: (f["claim_type"] == "INPATIENT") & _has_doc(world, "DISCHARGE_SUMMARY"))
    if taken.empty:
        return
    claims = list(taken["claim_sk"])
    positives, negatives = claims[:8], claims[8:]
    d = _docs(world)
    pool = members.index.to_numpy()
    for claim in claims:
        doc = d[(d["claim_sk"] == claim) & (d["doc_type"] == "DISCHARGE_SUMMARY")].iloc[0]
        m = members.loc[doc["member_sk"]]
        on = _service_date(world, claim)
        age, sex = _age(m["date_of_birth"], on), m["sex"]
        if claim in positives:
            for _ in range(50):  # the note of a different patient, copied over
                other = members.loc[pool[int(world.rng.integers(len(pool)))]]
                o_age, o_sex = _age(other["date_of_birth"], on), other["sex"]
                if o_sex != sex or abs(o_age - age) > 10:
                    age, sex = o_age, o_sex
                    break
        _set_text(world, doc["document_sk"], _insert_demographics(doc["text"], age, sex))
    world.record("DOC-02 discharge summary carries another patient's age or sex", rule_ids=["DOC-02-R02"],
                 claim_ids=positives)
    world.record("DOC-02 discharge summary with the patient's correct age and sex", rule_ids=["DOC-02-R02"],
                 claim_ids=negatives, positive=False, note="Correct demographics: the control must stay silent.")


# ----------------------------------------------------------- DOC-02-R03


def _post_denial_alteration(world: World) -> None:
    bctx = world.context["base_ctx"]
    hospitals = [p for p, info in bctx.providers.items() if info["type"] == "HOSPITAL"]
    members = world.take_entities("member", "member_sk", 16, lambda f: f["relationship"] == "PRINCIPAL")
    made: list[tuple[str, bool]] = []
    for i, msk in enumerate(members["member_sk"]):
        if len(made) >= 8:
            break
        m = bctx.members[msk]
        lo, hi = m["window"]
        if (hi - lo).days < 60:
            continue
        day = lo + TD(days=int(world.rng.integers(10, max(11, (hi - lo).days - 40))))
        hospital = [p for p in hospitals if bctx.providers[p]["emirate"] == m["emirate"]] or hospitals
        prov = hospital[int(world.rng.integers(len(hospital)))]
        try:
            claim = world.make_claim(member_sk=msk, provider_sk=prov, service_date=day, claim_type="INPATIENT",
                                     decision="DENIED", denial_code="MNEC-003")
        except RuntimeError:
            continue
        made.append((claim, len(made) >= 6))  # the last two become signed addenda
    positives, negatives = [], []
    d = _docs(world)
    remit = world.tables["remittance"]
    for claim, signed in made:
        docs = d[(d["claim_sk"] == claim) & (d["doc_type"] == "DISCHARGE_SUMMARY")]
        if docs.empty:
            continue
        doc = docs.iloc[0]
        denied_on = pd.to_datetime(remit.loc[remit["claim_sk"] == claim, "settlement_date"]).min()
        text = doc["text"]
        los = re.search(r"Length of stay:\s*(\d+)", text) or re.search(r"مدة الإقامة:\s*(\d+)", text)
        if los is None:
            continue
        old, new = int(los.group(1)), int(los.group(1)) + 3
        text = re.sub(r"(Length of stay:\s*)\d+", lambda m_: f"{m_.group(1)}{new}", text)
        text = re.sub(r"(recorded inpatient stay was\s+)\d+", lambda m_: f"{m_.group(1)}{new}", text)
        text = re.sub(r"(مدة الإقامة:\s*)\d+", lambda m_: f"{m_.group(1)}{new}", text)
        text = re.sub(r"(بلغت مدة الإقامة المسجلة\s+)\d+", lambda m_: f"{m_.group(1)}{new}", text)
        severity = "Severity: severe, requiring continuous monitoring\n"
        text = text.replace("PROCEDURES:", severity + "PROCEDURES:", 1) if "PROCEDURES:" in text else text + severity
        if signed:
            attending = re.search(r"Attending clinician:\s*(\S+)", text)
            who = attending.group(1).rstrip(".") if attending else "the attending clinician"
            text = ("ADDENDUM (SYNTHETIC) — late entry clarifying the length of stay.\n" + text
                    + f"\nSigned electronically by {who} on {(denied_on + pd.Timedelta(days=9)).date()}.\n")
        _new_doc(world, claim_sk=claim, member_sk=doc["member_sk"], provider_sk=doc["provider_sk"],
                 doc_type="DISCHARGE_SUMMARY", text=text, created=denied_on + pd.Timedelta(days=9, hours=11),
                 language=doc["language"], prior=doc["document_sk"], version=2)
        (negatives if signed else positives).append(claim)
    if positives:
        world.record("DOC-02 discharge summary rewritten after a denial (longer stay, severity added)",
                     rule_ids=["DOC-02-R03"], claim_ids=positives)
    if negatives:
        world.record("DOC-02 signed addendum after a denial", rule_ids=["DOC-02-R03"], claim_ids=negatives,
                     positive=False, note="Valid signed addendum: the declared exclusion.")


# ----------------------------------------------------------- DOC-02-R04


_THIN = {
    "en": "Patient admitted and treated. Discharged home.",
    "ar": "تم إدخال المريض وعلاجه وخروجه.",
}


def _thin_note(text: str, language: str) -> str:
    """Keep the template (headings, key: value lines, markers); replace the narrative with one line."""
    out, inserted = [], False
    for line in text.split("\n"):
        s = line.strip()
        template = (not s or s.startswith(("***", "---")) or "SYNTHETIC" in s or re.match(r"^[^\n:]{2,40}:", s)
                    or re.fullmatch(r"[A-Z /&()\-]{4,}", s) or s in ("المسار السريري", "الأدوية والمستلزمات", "الخروج")
                    or s.startswith(("تم إنشاء هذا الملف", "This file was generated")))
        if template:
            out.append(line)
        elif not inserted:
            out.append(_THIN.get(language, _THIN["en"]))
            inserted = True
    return "\n".join(out)


def _generic_notes(world: World) -> None:
    codes = world.context["codes"]
    high = {c for c, info in codes.items() if str(info.get("complexity", "")).upper() == "HIGH"}
    lines = world.tables["claim_line"]
    high_claims = set(lines.loc[lines["activity_code"].isin(high), "claim_sk"])
    d = _docs(world)
    ds = d[(d["doc_type"] == "DISCHARGE_SUMMARY") & d["claim_sk"].isin(high_claims)]
    h = world.tables["claim_header"]
    free = h[~h["claim_sk"].isin(world.used_claims) & h["claim_sk"].isin(set(ds["claim_sk"]))]
    counts = free["provider_sk"].value_counts()
    total = ds["claim_sk"].map(h.set_index("claim_sk")["provider_sk"]).value_counts()
    # A hospital with enough documented high-complexity admissions, most of them still unused.
    candidates = [p for p in counts.index if 8 <= counts[p] <= 30 and counts[p] >= 0.8 * total.get(p, 0)
                  and p not in world.used_entities]
    if not candidates:
        return
    provider = candidates[0]
    world.used_entities.add(provider)
    taken = world.take_claims(len(free), lambda f: (f["provider_sk"] == provider) & f["claim_sk"].isin(set(ds["claim_sk"])))
    for claim in taken["claim_sk"]:
        for doc in ds[ds["claim_sk"] == claim].itertuples(index=False):
            _set_text(world, doc.document_sk, _thin_note(doc.text, doc.language))
    world.record("DOC-02 one-line discharge summaries for high-complexity admissions", rule_ids=["DOC-02-R04"],
                 claim_ids=taken["claim_sk"], subject_type="provider", subject_ids=[provider])


# ------------------------------------------------------------------- entry


def inject(world: World) -> None:
    if "document" not in world.tables or world.tables["document"].empty:
        return
    _required_document_absent(world)
    _necessity_absent(world)
    _auth_narrative_drift(world)
    _copied_demographics(world)
    _post_denial_alteration(world)
    _generic_notes(world)
