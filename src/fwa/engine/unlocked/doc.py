"""Unlocked implementations for the DOC family.

Each function here runs only when a dataset unlock (`rules/unlocks/DOC.yaml`)
is satisfied by the loaded file. See :mod:`fwa.engine.unlocks`.

These controls read the ``document`` table the dataset itself supplies (on the
UAE demo file every document is SYNTHETIC and says so). They follow the §4.9
rule the existing document controls follow: a finding about what a document
SAYS carries the exact text it rests on and that text's character offsets. The
offsets are produced by :class:`fwa.nlp.pipeline.Extraction`, which refuses to
exist without a usable span, so a finding whose span cannot be located is
discarded rather than shown with a caveat. A finding about what a document
does NOT say (a required document or fact that is absent) shows the exact
passage that was searched, so the reviewer reads the same text the check read.
"""

from __future__ import annotations

import re
from typing import Any, Callable, Iterable

import numpy as np
import pandas as pd

from ...cases import exposure as _exp
from ...nlp.pipeline import DocumentPipeline, Extraction, MissingSpanError
from ...presentation import aed, count_phrase, pct, plain_date
from ..controls import _claim_frame, _f, _sig
from ..evallib import is_missing, period_bucket

__all__ = ["IMPLEMENTATIONS"]

_SYNTHETIC_MARKER = (
    "SYNTHETIC — the documents in this file were generated to exercise the engine; "
    "they are not real clinical records."
)
_NLP_VERSION = {"nlp_pipeline": "rule-based deterministic extraction v1.0.0"}


# ---------------------------------------------------------------------------
# shared helpers
# ---------------------------------------------------------------------------


def _table(ctx, name: str) -> pd.DataFrame:
    """A dataset table, tenant-scoped, or an empty frame."""
    ds = ctx.dataset
    if ds is None:
        return pd.DataFrame()
    try:
        frame = ds.get(name)
    except Exception:
        return pd.DataFrame()
    if frame is None or frame.empty:
        return pd.DataFrame(columns=[] if frame is None else frame.columns)
    if "tenant_id" in frame.columns and frame["tenant_id"].notna().any():
        frame = frame[frame["tenant_id"].isna() | (frame["tenant_id"].astype(str) == str(ctx.tenant_id))]
    return frame


def _s(value: Any) -> str:
    return "" if is_missing(value) else str(value).strip()


def _norm(value: Any) -> str:
    """Normalised category token: lower case, words joined by underscores."""
    return re.sub(r"[\s\-/]+", "_", _s(value).lower())


def _claims(ctx) -> pd.DataFrame:
    df = _claim_frame(ctx)
    if df is None or df.empty or "claim_sk" not in df.columns:
        return pd.DataFrame()
    df = df.copy()
    df["claim_sk"] = df["claim_sk"].astype(str)
    return df.drop_duplicates("claim_sk")


def _documents(ctx) -> pd.DataFrame:
    docs = _table(ctx, "document")
    if docs.empty or "text" not in docs.columns or "document_sk" not in docs.columns:
        return pd.DataFrame()
    docs = docs[docs["text"].notna()].copy()
    if docs.empty:
        return docs
    docs["text"] = docs["text"].astype(str)
    docs["document_sk"] = docs["document_sk"].astype(str)
    docs["claim_sk"] = docs["claim_sk"].map(_s) if "claim_sk" in docs.columns else ""
    docs["doc_type_norm"] = docs["doc_type"].map(_norm) if "doc_type" in docs.columns else ""
    ocr = pd.to_numeric(docs["ocr_confidence"], errors="coerce") if "ocr_confidence" in docs.columns \
        else pd.Series(np.nan, index=docs.index)
    docs["ocr"] = ocr.fillna(1.0).clip(0.0, 1.0)
    docs["created"] = pd.to_datetime(docs["created_at"], errors="coerce") if "created_at" in docs.columns \
        else pd.NaT
    docs["language_detected"] = docs["text"].map(DocumentPipeline.detect_language)
    return docs


def _lines(ctx) -> pd.DataFrame:
    lines = _table(ctx, "claim_line")
    if lines.empty or "claim_sk" not in lines.columns or "activity_code" not in lines.columns:
        return pd.DataFrame()
    lines = lines.copy()
    lines["claim_sk"] = lines["claim_sk"].astype(str)
    lines["activity_code"] = lines["activity_code"].map(_s)
    for col in ("net_amount", "gross_amount"):
        lines[col] = pd.to_numeric(lines[col], errors="coerce") if col in lines.columns else np.nan
    return lines


def _code_reference(ctx) -> pd.DataFrame:
    ref = _table(ctx, "activity_code_reference")
    if ref.empty or "activity_code" not in ref.columns:
        return pd.DataFrame(columns=["activity_code", "description", "service_family",
                                     "code_family", "complexity"])
    ref = ref.copy()
    ref["activity_code"] = ref["activity_code"].map(_s)
    return ref.drop_duplicates("activity_code")


def _is_auth_narrative(doc_type_norm: str) -> bool:
    return "auth" in doc_type_norm or "approval" in doc_type_norm


def _is_clinical(doc_type_norm: str) -> bool:
    """Clinical notes, as opposed to requests, invoices, consents and receipts."""
    if _is_auth_narrative(doc_type_norm):
        return False
    return not any(k in doc_type_norm for k in ("invoice", "receipt", "consent", "id_card", "letter"))


def _extraction(field: str, value: Any, text: str, start: int, end: int, ocr: float,
                language: str, raw_confidence: float) -> Extraction | None:
    """A span-anchored fact, or ``None`` when no usable span exists (discarded)."""
    try:
        return Extraction(
            field_name=field, value=value, span_text=text[start:end].strip(),
            span_start=int(start), span_end=int(end), raw_confidence=raw_confidence,
            ocr_confidence=float(ocr), language=language,
        )
    except MissingSpanError:
        return None


_SENTENCE_BREAK = re.compile(r"[.\n؟!]")


def _sentence_around(text: str, start: int, end: int) -> tuple[int, int]:
    """Offsets of the sentence that contains ``text[start:end]``."""
    left = start
    while left > 0 and not _SENTENCE_BREAK.match(text[left - 1]):
        left -= 1
    right = end
    while right < len(text) and not _SENTENCE_BREAK.match(text[right]):
        right += 1
    while left < right and text[left].isspace():
        left += 1
    return left, min(len(text), right + 1)


_NEGATION = re.compile(
    r"\b(no|not|without|denies|denied|negative for|ruled out|absent)\b|(?:^|\s)(لا|لم|بدون|ينفي|غير)\s",
    re.I,
)


def _negated(text: str, start: int, end: int) -> bool:
    """True when the match sits in a sentence with a negation cue before it."""
    left, _ = _sentence_around(text, start, end)
    return bool(_NEGATION.search(text[left:start]))


#: Lines of a synthetic note that are TEMPLATE, not patient-specific prose.
_TEMPLATE_LINE = re.compile(
    r"^\s*(\*\*\*.*\*\*\*|[A-Z][A-Z /&()\-]{3,}(\(SYNTHETIC\))?\s*$|---.*---|"
    r"[A-Za-z][A-Za-z /()\-]{1,40}:\s*.*|[؀-ۿ][؀-ۿ /()\-]{1,40}:\s*.*|"
    r"This file was generated.*|تم إنشاء هذا الملف.*|.*SYNTHETIC.*)$"
)


def _narrative_span(text: str) -> tuple[int, int] | None:
    """Offsets of the free-text narrative (everything that is not template lines)."""
    start = end = None
    pos = 0
    for line in text.splitlines(keepends=True):
        stripped = line.strip()
        if stripped and not _TEMPLATE_LINE.match(stripped):
            offset = pos + line.index(stripped[0])
            if start is None:
                start = offset
            end = pos + len(line.rstrip())
        pos += len(line)
    if start is None or end is None or end <= start:
        return None
    return start, end


def _narrative_tokens(text: str) -> set[str]:
    span = _narrative_span(text)
    if span is None:
        return set()
    body = text[span[0]: span[1]]
    lines = [ln for ln in body.splitlines() if ln.strip() and not _TEMPLATE_LINE.match(ln.strip())]
    return {t for t in re.findall(r"[^\W\d_]{3,}", " ".join(lines).lower())}


def _span_evidence(ex: Extraction) -> dict[str, Any]:
    d = ex.to_dict()
    return {
        "source_span_text": d["source_span_text"],
        "span_offsets": d["span_offsets"],
        "extraction_confidence": d["extraction_confidence"],
        "ocr_confidence": d["ocr_confidence"],
        "document_language": d["language"],
    }


def _passage(field: str, text: str, span: tuple[int, int], ocr: float, language: str,
             max_chars: int) -> Extraction | None:
    """The searched passage as a span (trimmed to ``max_chars`` from its start)."""
    start, end = span
    end = min(end, start + max_chars)
    return _extraction(field, "searched passage", text, start, end, ocr, language, 1.0)


def _claim_row(claims: pd.DataFrame, claim_sk: str) -> pd.Series | None:
    if claims.empty:
        return None
    hit = claims[claims["claim_sk"] == claim_sk]
    return None if hit.empty else hit.iloc[0]


def _service_date(row: pd.Series | None) -> Any:
    if row is None:
        return None
    for col in ("service_date", "admission_date", "claim_date"):
        if col in row.index and not is_missing(row[col]):
            return row[col]
    return None


def _describe(code: str, ref_index: dict[str, dict[str, Any]]) -> str:
    """The service in words (its reference description), falling back to the code."""
    desc = _s(ref_index.get(code, {}).get("description"))
    return desc or f"service {code}"


def _a(noun: str) -> str:
    return ("an " if noun[:1].lower() in "aeiou" else "a ") + noun


# ===========================================================================
# DOC-01-R01 — required document absent after the deadline
# ===========================================================================


def doc_01_r01_required_document_absent(ctx, control) -> list:
    """A document the requirement policy names for a billed service never arrived.

    Evaluated as of the file's own extraction point (the latest submission or
    document date it contains), so a claim whose deadline has not yet passed is
    not flagged (declared exclusion: latency). A document of the required type
    that arrived, even late, satisfies the requirement.
    """
    policy = _table(ctx, "document_requirement_policy")
    lines = _lines(ctx)
    docs = _documents(ctx)
    claims = _claims(ctx)
    if policy.empty or lines.empty or claims.empty or "doc_type" not in policy.columns:
        return []
    sla_days = int(ctx.cfg("doc01r01_document_sla_days"))
    ref = _code_reference(ctx)
    ref_index = ref.set_index("activity_code").to_dict("index") if not ref.empty else {}

    pol = policy.copy()
    pol["doc_type_norm"] = pol["doc_type"].map(_norm)
    pol["activity_code"] = pol["activity_code"].map(_s) if "activity_code" in pol.columns else ""
    pol["service_family"] = pol["service_family"].map(_norm) if "service_family" in pol.columns else ""
    pol = pol[pol["doc_type_norm"] != ""]
    if pol.empty:
        return []

    work = lines[["claim_sk", "activity_code", "net_amount", "gross_amount"]].copy()
    # A zero-priced line is a package component already paid inside another
    # code; it is not separately billed, so it carries no document requirement
    # of its own (its package's requirement applies instead).
    work = work[(work["gross_amount"].fillna(work["net_amount"]).fillna(0) > 0)]
    work["service_family"] = work["activity_code"].map(
        lambda c: _norm(ref_index.get(c, {}).get("service_family")))
    # On an admission billed as a case rate every line is billed under the case
    # rate's benefit family, so the family-level requirement is the admission's
    # (a discharge summary), not one per component line.
    work["code_family"] = work["activity_code"].map(lambda c: _norm(ref_index.get(c, {}).get("code_family")))
    case_family = work[work["code_family"] == "case_rate"].groupby("claim_sk")["service_family"].first()
    in_case = work["claim_sk"].isin(case_family.index)
    work.loc[in_case, "service_family"] = work.loc[in_case, "claim_sk"].map(case_family)
    by_code = pol[pol["activity_code"] != ""][["activity_code", "doc_type_norm"]]
    by_family = pol[(pol["activity_code"] == "") & (pol["service_family"] != "")][
        ["service_family", "doc_type_norm"]]
    need = pd.concat([
        work.merge(by_code, on="activity_code", how="inner"),
        work.merge(by_family, on="service_family", how="inner"),
    ], ignore_index=True)
    if need.empty:
        return []

    present = set()
    if not docs.empty:
        present = set(zip(docs["claim_sk"], docs["doc_type_norm"]))

    # Latency: the file's own "as of" point, not the wall clock.
    date_col = next((c for c in ("submission_date", "claim_date", "service_date") if c in claims.columns), None)
    if date_col is None:
        return []
    claims = claims.assign(_submitted=pd.to_datetime(claims[date_col], errors="coerce"))
    as_of_candidates = [claims["_submitted"].max()]
    if not docs.empty:
        as_of_candidates.append(docs["created"].max())
    as_of = max((d for d in as_of_candidates if not pd.isna(d)), default=pd.NaT)
    if pd.isna(as_of):
        return []
    submitted = claims.set_index("claim_sk")["_submitted"]

    need = need[need["claim_sk"].isin(submitted.index)]
    need["_missing"] = [
        (c, t) not in present for c, t in zip(need["claim_sk"], need["doc_type_norm"])
    ]
    need = need[need["_missing"]]
    if need.empty:
        return []

    docs_by_claim = {} if docs.empty else docs.groupby("claim_sk")["doc_type_norm"].apply(
        lambda s: sorted(set(s))).to_dict()
    indexed = claims.set_index("claim_sk")
    out = []
    for claim_sk, sub in need.groupby("claim_sk", sort=True):
        sub_date = submitted.get(claim_sk)
        if pd.isna(sub_date):
            continue
        elapsed = int((as_of - sub_date).days)
        if elapsed <= sla_days:
            continue  # declared exclusion: latency — the deadline has not passed yet
        row = indexed.loc[claim_sk]
        missing_types = sorted(set(sub["doc_type_norm"]))
        triggering = sub.drop_duplicates(["activity_code", "doc_type_norm"])
        services = sorted(set(triggering["activity_code"]))
        billed = float(sub.drop_duplicates("activity_code")["net_amount"].fillna(0).sum())
        on_file = docs_by_claim.get(claim_sk, [])
        missing_words = " and ".join(t.replace("_", " ") for t in missing_types)
        service_words = ", ".join(_describe(c, ref_index) for c in services[:3])
        out.append(_sig(
            ctx, control, subject_type="claim", subject_id=claim_sk,
            fact_key=f"docreq:{claim_sk}",
            claim_ids=[claim_sk],
            event_time=_service_date(row),
            period=period_bucket(_service_date(row)),
            evidence={
                "claim_sk": claim_sk,
                "provider_sk": _s(row.get("provider_sk")),
                "member_sk": _s(row.get("member_sk")),
                "required_document_types": missing_types,
                "document_types_on_file": on_file,
                "services_requiring_documents": [
                    {"activity_code": c, "description": _s(ref_index.get(c, {}).get("description")),
                     "required_document_type": t}
                    for c, t in zip(triggering["activity_code"], triggering["doc_type_norm"])
                ],
                "billed_amount_for_those_services_aed": round(billed, 2),
                "submitted_on": str(sub_date.date()),
                "evaluated_as_of": str(as_of.date()),
                "days_since_submission": elapsed,
                "document_deadline_days": sla_days,
                "source_span_note": (
                    "This finding is an ABSENCE: no document of the required type is attached to the "
                    "claim, so there is no passage to quote. The documents that are on file are listed."
                ),
                "synthetic_document_marker": _SYNTHETIC_MARKER,
                "plain_language": (
                    f"The claim bills {service_words} (AED {billed:,.0f}); policy requires "
                    f"{_a(missing_words)} for this service, and none was on file "
                    f"{elapsed} days after the claim was sent on {plain_date(sub_date)} "
                    f"(allowed: {sla_days} days). Documents on file: "
                    f"{', '.join(t.replace('_', ' ') for t in on_file) or 'none'}."
                ),
                "what_the_reviewer_must_verify": (
                    "Whether the provider can supply the required document, and whether a document "
                    "of another name already covers it."
                ),
            },
            exposure=_exp.no_exposure(
                "a missing document holds the claim for information; it does not show an amount was overpaid"
            ),
        ))
    return out


# ===========================================================================
# DOC-01-R03 — medical-necessity support absent
# ===========================================================================


def _split_terms(raw: Any) -> list[list[str]]:
    """``"indication|indicated for;severity|grade"`` → [[indication, indicated for], [severity, grade]]."""
    out = []
    for element in re.split(r"[;\n]", _s(raw)):
        alts = [a.strip().lower() for a in element.split("|") if a.strip()]
        if alts:
            out.append(alts)
    return out


def doc_01_r03_medical_necessity_absent(ctx, control) -> list:
    """Required indication/severity facts cannot be found in the claim's clinical record.

    Absence is not proof (declared exclusion): the claim is only considered
    when it HAS clinical documents read with adequate OCR confidence, a fact
    that appears only inside a negated sentence ("no documented failure of
    conservative treatment") does not count as present, and the finding goes to
    a qualified clinical reviewer — it never denies anything.
    """
    policy = _table(ctx, "medical_necessity_policy")
    lines = _lines(ctx)
    docs = _documents(ctx)
    claims = _claims(ctx)
    if policy.empty or lines.empty or docs.empty or claims.empty:
        return []
    if "required_terms" not in policy.columns:
        return []
    min_conf = float(ctx.cfg("nlp_min_extraction_confidence"))
    max_chars = int(ctx.cfg("doc01r03_span_display_max_chars"))
    ref = _code_reference(ctx)
    ref_index = ref.set_index("activity_code").to_dict("index") if not ref.empty else {}

    pol = policy.copy()
    pol["activity_code"] = pol["activity_code"].map(_s)
    pol["_terms"] = pol["required_terms"].map(_split_terms)
    pol = pol[(pol["activity_code"] != "") & pol["_terms"].map(bool)]
    if pol.empty:
        return []
    terms_by_code = dict(zip(pol["activity_code"], pol["_terms"]))
    hits = lines[lines["activity_code"].isin(terms_by_code)]
    if hits.empty:
        return []

    clinical = docs[docs["doc_type_norm"].map(_is_clinical) & (docs["ocr"] >= min_conf)]
    docs_by_claim = {k: g for k, g in clinical.groupby("claim_sk")}
    indexed = claims.set_index("claim_sk")
    out = []
    for claim_sk, sub in hits.groupby("claim_sk", sort=True):
        if claim_sk not in docs_by_claim or claim_sk not in indexed.index:
            continue  # no readable clinical record: absence would prove nothing
        claim_docs = docs_by_claim[claim_sk]
        row = indexed.loc[claim_sk]
        missing: list[dict[str, Any]] = []
        found: list[dict[str, Any]] = []
        negated: list[dict[str, Any]] = []
        for code in sorted(set(sub["activity_code"])):
            for alts in terms_by_code[code]:
                located = None
                neg_hit = None
                for d in claim_docs.itertuples(index=False):
                    lowered = d.text.lower()
                    for alt in alts:
                        for m in re.finditer(re.escape(alt), lowered):
                            ex = _extraction("necessity_element", alt, d.text, m.start(), m.end(),
                                             d.ocr, d.language_detected, 0.95)
                            if ex is None:
                                continue
                            if _negated(d.text, m.start(), m.end()):
                                if neg_hit is None:
                                    s0, s1 = _sentence_around(d.text, m.start(), m.end())
                                    sent = _extraction("negated_element", alt, d.text, s0, s1, d.ocr,
                                                       d.language_detected, 0.95)
                                    if sent is not None:
                                        neg_hit = (d.document_sk, sent)
                                continue
                            located = (d.document_sk, ex)
                            break
                        if located:
                            break
                    if located:
                        break
                element = " / ".join(alts)
                if located:
                    found.append({"activity_code": code, "element": element, "document_sk": located[0],
                                  **_span_evidence(located[1])})
                else:
                    missing.append({"activity_code": code, "element": element})
                    if neg_hit:
                        negated.append({"activity_code": code, "element": element,
                                        "document_sk": neg_hit[0], **_span_evidence(neg_hit[1])})
        if not missing:
            continue
        # The span a reviewer reads: the narrative that was searched (or the
        # negated sentence, which is the strongest thing the record says).
        best = claim_docs.sort_values("ocr", ascending=False).iloc[0]
        span = _narrative_span(best["text"])
        searched = None if span is None else _passage(
            "searched_passage", best["text"], span, best["ocr"], best["language_detected"], max_chars)
        if searched is None and not negated:
            continue  # no locatable passage: the finding is discarded, not shown with a caveat
        codes = sorted({m["activity_code"] for m in missing})
        billed = float(sub[sub["activity_code"].isin(codes)]["net_amount"].fillna(0).sum())
        span_ev = negated[0] if negated else {"document_sk": best["document_sk"], **_span_evidence(searched)}
        confidence = float(span_ev.get("extraction_confidence", best["ocr"]))
        missing_words = "; ".join(m["element"].replace(" / ", " or ") for m in missing[:4])
        out.append(_sig(
            ctx, control, subject_type="claim", subject_id=claim_sk,
            fact_key=f"necessity:{claim_sk}",
            claim_ids=[claim_sk],
            event_time=_service_date(row),
            period=period_bucket(_service_date(row)),
            confidence=round(min(confidence, 0.75), 3),
            data_quality_penalty=round(1.0 - float(best["ocr"]), 3),
            evidence={
                "claim_sk": claim_sk,
                "provider_sk": _s(row.get("provider_sk")),
                "member_sk": _s(row.get("member_sk")),
                "services": [_describe(c, ref_index) for c in codes],
                "billed_amount_for_those_services_aed": round(billed, 2),
                "missing_elements": missing,
                "elements_found": found,
                "elements_found_only_in_negated_sentences": negated,
                "documents_searched": sorted(claim_docs["document_sk"].tolist()),
                **span_ev,
                "absence_is_not_proof": (
                    "A fact that cannot be found is a reason for a qualified reviewer to read the "
                    "record, never a reason on its own to refuse the service."
                ),
                "synthetic_document_marker": _SYNTHETIC_MARKER,
                "plain_language": (
                    f"The claim bills {', '.join(_describe(c, ref_index) for c in codes[:2])} "
                    f"(AED {billed:,.0f}), but the clinical record on file "
                    f"({count_phrase(len(claim_docs), 'document')}) does not state the required {missing_words}"
                    + (" — the record mentions it only to say it is absent." if negated else ".")
                ),
                "what_the_reviewer_must_verify": (
                    "Whether the full clinical record shows why the service was needed; ask the "
                    "provider for further notes before any decision."
                ),
            },
            exposure=_exp.no_exposure(
                "missing support for medical necessity is a question for a clinical reviewer, "
                "not an established overpayment"
            ),
            extra_versions=_NLP_VERSION,
        ))
    return out


# ===========================================================================
# DOC-01-R04 — authorisation narrative drifts from the billed service
# ===========================================================================


_CODE_TOKEN = re.compile(r"(?<![\w.])[A-Z0-9][A-Z0-9.\-]{2,}[A-Z0-9](?![\w])")


def doc_01_r04_authorization_narrative_drift(ctx, control) -> list:
    """The approval request describes one service; the claim bills a different kind.

    The request narrative is read for the services it names (activity codes and
    their reference descriptions), each with its exact span. A billed line under
    that approval is flagged only when the narrative names at least one service,
    names neither the billed code nor its description, and every service it does
    name belongs to a DIFFERENT service family — a material difference, not a
    sibling code (declared exclusion: the finding is explained with the extracted
    conflicting facts).
    """
    lines = _lines(ctx)
    docs = _documents(ctx)
    auth = _table(ctx, "authorization")
    auth_lines = _table(ctx, "authorization_line")
    claims = _claims(ctx)
    if lines.empty or docs.empty or claims.empty or "authorization_id" not in lines.columns:
        return []
    ref = _code_reference(ctx)
    if ref.empty:
        return []
    ref_index = ref.set_index("activity_code").to_dict("index")
    known_codes = set(ref_index)

    narratives = docs[docs["doc_type_norm"].map(_is_auth_narrative)]
    if narratives.empty:
        return []
    lines = lines[lines["authorization_id"].map(_s) != ""].copy()
    if lines.empty:
        return []
    lines["authorization_id"] = lines["authorization_id"].map(_s)

    # Which narrative belongs to which approval: by reference, else by claim.
    auth_keys: dict[str, str] = {}
    if not auth.empty and "authorization_sk" in auth.columns:
        for r in auth.itertuples(index=False):
            sk = _s(getattr(r, "authorization_sk", ""))
            auth_keys[sk] = sk
            rid = _s(getattr(r, "request_id", ""))
            if rid:
                auth_keys[rid] = sk
    by_auth: dict[str, pd.Series] = {}
    for d in narratives.itertuples(index=False):
        ref_key = _s(getattr(d, "attachment_ref", ""))
        if ref_key and ref_key in auth_keys:
            by_auth.setdefault(auth_keys[ref_key], d)
    claim_auths = lines.groupby("claim_sk")["authorization_id"].apply(lambda s: sorted(set(s))).to_dict()
    for d in narratives.itertuples(index=False):
        ids = claim_auths.get(d.claim_sk, [])
        if len(ids) == 1:
            by_auth.setdefault(ids[0], d)
    if not by_auth:
        return []

    approved: dict[str, list[str]] = {}
    if not auth_lines.empty and {"authorization_sk", "activity_code"} <= set(auth_lines.columns):
        approved = auth_lines.groupby(auth_lines["authorization_sk"].map(_s))["activity_code"].apply(
            lambda s: sorted({_s(v) for v in s if _s(v)})).to_dict()

    def family(code: str) -> str:
        info = ref_index.get(code, {})
        return _norm(info.get("service_family")) or _norm(info.get("code_family")) or code

    indexed = claims.set_index("claim_sk")
    out = []
    for (claim_sk, auth_id), sub in lines.groupby(["claim_sk", "authorization_id"], sort=True):
        d = by_auth.get(auth_id)
        if d is None or claim_sk not in indexed.index:
            continue
        text = d.text
        lowered = text.lower()
        billed_codes = sorted(set(sub["activity_code"]) - {""})
        candidates = set(billed_codes) | set(approved.get(auth_id, []))
        mentions: dict[str, Extraction] = {}
        for m in _CODE_TOKEN.finditer(text):
            code = m.group(0)
            if code in known_codes and code not in mentions:
                ex = _extraction("requested_service", code, text, m.start(), m.end(), d.ocr,
                                 d.language_detected, 0.95)
                if ex is not None:
                    mentions[code] = ex
        for code in candidates:
            desc = _s(ref_index.get(code, {}).get("description")).lower()
            if code in mentions or len(desc) < 5:
                continue
            pos = lowered.find(desc)
            if pos >= 0:
                ex = _extraction("requested_service", code, text, pos, pos + len(desc), d.ocr,
                                 d.language_detected, 0.9)
                if ex is not None:
                    mentions[code] = ex
        if not mentions:
            continue  # nothing the narrative names can be anchored: no finding
        requested_families = {family(c) for c in mentions}
        drift = []
        for code in billed_codes:
            if code in mentions:
                continue
            if family(code) in requested_families:
                continue  # a sibling code in the same family is not a material difference
            drift.append(code)
        if not drift:
            continue
        min_conf = float(ctx.cfg("nlp_min_extraction_confidence"))
        spans = [ex for ex in mentions.values() if ex.confidence >= min_conf]
        if not spans:
            continue
        row = indexed.loc[claim_sk]
        billed = float(sub[sub["activity_code"].isin(drift)]["net_amount"].fillna(0).sum())
        first = sorted(spans, key=lambda e: e.span_start)[0]
        requested_words = ", ".join(_describe(c, ref_index) for c in sorted(mentions)[:2])
        billed_words = ", ".join(_describe(c, ref_index) for c in drift[:2])
        out.append(_sig(
            ctx, control, subject_type="claim", subject_id=claim_sk,
            fact_key=f"authdrift:{claim_sk}:{auth_id}",
            claim_ids=[claim_sk],
            event_time=_service_date(row),
            period=period_bucket(_service_date(row)),
            confidence=round(first.confidence * 0.8, 3),
            evidence={
                "claim_sk": claim_sk,
                "authorization_sk": auth_id,
                "provider_sk": _s(row.get("provider_sk")),
                "member_sk": _s(row.get("member_sk")),
                "narrative_document_sk": d.document_sk,
                "requested_services": [
                    {"activity_code": c, "service_family": family(c), **_span_evidence(ex)}
                    for c, ex in sorted(mentions.items())
                ],
                "billed_services_not_described": [
                    {"activity_code": c, "description": _s(ref_index.get(c, {}).get("description")),
                     "service_family": family(c)} for c in drift
                ],
                "approved_codes_on_authorisation": approved.get(auth_id, []),
                "billed_amount_for_those_services_aed": round(billed, 2),
                **_span_evidence(first),
                "comparison_basis": "services named in the request (codes and reference descriptions) "
                                    "against the billed lines; semantic paraphrase is not modelled",
                "synthetic_document_marker": _SYNTHETIC_MARKER,
                "plain_language": (
                    f"The approval request describes {requested_words}; the claim bills "
                    f"{billed_words} (AED {billed:,.0f}), a different kind of service that the "
                    f"request does not mention."
                ),
                "what_the_reviewer_must_verify": (
                    "Read the approval request beside the claim and confirm whether the billed "
                    "service is the one that was requested and approved."
                ),
            },
            exposure=_exp.no_exposure(
                "a narrative mismatch holds the claim for explanation; the approved and billed "
                "amounts have not been reconciled"
            ),
            extra_versions=_NLP_VERSION,
        ))
    return out


# ===========================================================================
# DOC-02-R02 — copied note with the wrong patient's details
# ===========================================================================


_AGE_PATTERNS = [
    re.compile(r"\bAge\s*:\s*(\d{1,3})\b", re.I),
    re.compile(r"\b(\d{1,3})[- ]year[- ]old\b", re.I),
    re.compile(r"العمر\s*:\s*(\d{1,3})"),
    re.compile(r"يبلغ من العمر\s+(\d{1,3})"),
]
_SEX_PATTERNS = [
    re.compile(r"\b(?:Sex|Gender)\s*:\s*(male|female|m|f)\b", re.I),
    re.compile(r"\b\d{1,3}[- ]year[- ]old\s+(male|female|man|woman)\b", re.I),
    re.compile(r"الجنس\s*:\s*(ذكر|أنثى|انثى)"),
]
_SEX_MAP = {"male": "M", "m": "M", "man": "M", "ذكر": "M",
            "female": "F", "f": "F", "woman": "F", "أنثى": "F", "انثى": "F"}
#: Findings that cannot apply to a patient of the other sex.
_SEX_SPECIFIC = {
    "M": [r"\bpregnan\w*", r"\bgravid\w*", r"\bovar(?:y|ian)\b", r"\buter(?:us|ine)\b", r"حامل", r"الحمل"],
    "F": [r"\bprostat\w*", r"\btestic\w*", r"\bscrot\w*", r"البروستاتا"],
}


def _member_sex(value: Any) -> str:
    v = _s(value).lower()
    return _SEX_MAP.get(v, "")


def doc_02_r02_contradictory_copied_facts(ctx, control) -> list:
    """A note states an age or sex that is not this patient's, or a finding impossible for them."""
    docs = _documents(ctx)
    members = _table(ctx, "member")
    claims = _claims(ctx)
    if docs.empty or members.empty or claims.empty or "member_sk" not in members.columns:
        return []
    min_conf = float(ctx.cfg("nlp_min_extraction_confidence"))
    tolerance = int(ctx.cfg("doc02r02_age_tolerance_years"))
    mem = members.copy()
    mem["member_sk"] = mem["member_sk"].astype(str)
    mem = mem.drop_duplicates("member_sk").set_index("member_sk")
    indexed = claims.set_index("claim_sk")

    out = []
    for d in docs[docs["doc_type_norm"].map(_is_clinical)].sort_values("document_sk").itertuples(index=False):
        claim_row = indexed.loc[d.claim_sk] if d.claim_sk in indexed.index else None
        member_sk = _s(getattr(d, "member_sk", "")) or (
            _s(claim_row.get("member_sk")) if claim_row is not None else "")
        if not member_sk or member_sk not in mem.index:
            continue
        m = mem.loc[member_sk]
        true_sex = _member_sex(m.get("sex"))
        dob = pd.to_datetime(m.get("date_of_birth"), errors="coerce")
        ref_date = pd.to_datetime(_service_date(claim_row), errors="coerce")
        if pd.isna(ref_date):
            ref_date = d.created
        true_age = None
        if not pd.isna(dob) and not pd.isna(ref_date):
            true_age = int(ref_date.year - dob.year - ((ref_date.month, ref_date.day) < (dob.month, dob.day)))

        conflicts: list[dict[str, Any]] = []
        text = d.text
        if true_age is not None:
            for pat in _AGE_PATTERNS:
                hit = pat.search(text)
                if not hit:
                    continue
                ex = _extraction("age", int(hit.group(1)), text, hit.start(), hit.end(), d.ocr,
                                 d.language_detected, 0.95)
                if ex is not None and ex.confidence >= min_conf and abs(int(ex.value) - true_age) > tolerance:
                    conflicts.append({"conflicting_field": "age", "document_value": int(ex.value),
                                      "patient_record_value": true_age, **_span_evidence(ex)})
                break
        if true_sex:
            stated = None
            for pat in _SEX_PATTERNS:
                hit = pat.search(text)
                if hit:
                    stated = hit
                    break
            if stated is not None:
                sex = _SEX_MAP.get(stated.group(1).lower(), "")
                ex = _extraction("sex", sex, text, stated.start(), stated.end(), d.ocr,
                                 d.language_detected, 0.95)
                if ex is not None and sex and sex != true_sex and ex.confidence >= min_conf:
                    conflicts.append({"conflicting_field": "sex", "document_value": sex,
                                      "patient_record_value": true_sex, **_span_evidence(ex)})
            for pattern in _SEX_SPECIFIC.get(true_sex, []):
                for hit in re.finditer(pattern, text, re.I):
                    if _negated(text, hit.start(), hit.end()):
                        continue
                    s0, s1 = _sentence_around(text, hit.start(), hit.end())
                    ex = _extraction("sex_specific_finding", hit.group(0), text, s0, s1, d.ocr,
                                     d.language_detected, 0.9)
                    if ex is not None and ex.confidence >= min_conf:
                        conflicts.append({"conflicting_field": "finding impossible for the patient's sex",
                                          "document_value": hit.group(0), "patient_record_value": true_sex,
                                          **_span_evidence(ex)})
                        break
                else:
                    continue
                break
        if not conflicts:
            continue
        first = conflicts[0]
        sex_word = {"M": "male", "F": "female"}
        parts = []
        for c in conflicts[:2]:
            if c["conflicting_field"] == "age":
                parts.append(f"age {c['document_value']} (the patient is {c['patient_record_value']})")
            elif c["conflicting_field"] == "sex":
                parts.append(f"sex {sex_word.get(c['document_value'], c['document_value'])} "
                             f"(the patient is {sex_word.get(c['patient_record_value'])})")
            else:
                parts.append(f"'{c['document_value']}' for a {sex_word.get(c['patient_record_value'])} patient")
        service_date = _service_date(claim_row)
        out.append(_sig(
            ctx, control, subject_type="claim", subject_id=d.claim_sk or d.document_sk,
            fact_key=f"docdemo:{d.document_sk}",
            claim_ids=[d.claim_sk] if d.claim_sk else [],
            event_time=service_date,
            period=period_bucket(service_date),
            confidence=float(first["extraction_confidence"]),
            data_quality_penalty=round(1.0 - float(d.ocr), 3),
            evidence={
                "claim_sk": d.claim_sk,
                "document_sk": d.document_sk,
                "document_type": d.doc_type_norm,
                "member_sk": member_sk,
                "provider_sk": _s(getattr(d, "provider_sk", "")) or (
                    _s(claim_row.get("provider_sk")) if claim_row is not None else ""),
                "conflicts": conflicts,
                "conflicting_field": first["conflicting_field"],
                "extracted_value": first["document_value"],
                "claim_value": first["patient_record_value"],
                **{k: first[k] for k in ("source_span_text", "span_offsets", "extraction_confidence",
                                         "ocr_confidence", "document_language")},
                "age_tolerance_years": tolerance,
                "laterality_limb": "not evaluated — the claim lines carry no body side to contradict",
                "synthetic_document_marker": _SYNTHETIC_MARKER,
                "plain_language": (
                    f"The note on this claim states {' and '.join(parts)}, which does not fit the "
                    f"patient's registration record — a sign the note may have been copied from "
                    f"another patient."
                ),
                "what_the_reviewer_must_verify": (
                    "Compare the highlighted passage with the patient's registration record and ask "
                    "the provider which patient the note was written for."
                ),
            },
            exposure=_exp.no_exposure(
                "a note with the wrong patient's details undermines the record; it does not by "
                "itself establish an amount"
            ),
            extra_versions=_NLP_VERSION,
        ))
    return out


# ===========================================================================
# DOC-02-R03 — document materially altered after a denial
# ===========================================================================


#: Reimbursement-relevant "Key: value" lines compared between versions.
_RELEVANT_KEYS = (
    "diagnosis", "primary diagnosis", "secondary diagnosis", "procedure", "procedure performed",
    "indication", "severity", "length of stay", "admission date", "discharge date", "units",
    "laterality", "level of care", "findings", "التشخيص الأساسي", "مدة الإقامة", "الإجراء",
    "تاريخ الدخول", "تاريخ الخروج", "الشدة", "دواعي الاستعمال",
)
_KV_LINE = re.compile(r"^[ \t]*([^\n:]{2,40}?)[ \t]*:[ \t]*(.+?)[ \t]*$", re.M)
_ADDENDUM = re.compile(r"\bADDENDUM\b|ملحق", re.I)
_SIGNED = re.compile(r"\bsigned(?:\s+electronically)?\s+by\b|\belectronically signed\b|توقيع", re.I)


def _kv_facts(text: str) -> dict[str, tuple[str, int, int]]:
    out: dict[str, tuple[str, int, int]] = {}
    for m in _KV_LINE.finditer(text):
        key = m.group(1).strip().lower()
        if key in _RELEVANT_KEYS and key not in out:
            out[key] = (m.group(2).strip(), m.start(), m.end())
    return out


def _denied(decision: Any) -> bool:
    v = _s(decision).lower()
    return v.startswith("den") or "reject" in v or v in ("d", "declined", "not_paid", "not paid")


def doc_02_r03_post_denial_alteration(ctx, control) -> list:
    """A later version of a claim's document, written after a denial, changes payment-relevant facts.

    Declared exclusion: a new version that is a SIGNED ADDENDUM (it says it is an
    addendum and carries a signature statement) has provenance and is not flagged.
    """
    docs = _documents(ctx)
    remit = _table(ctx, "remittance")
    claims = _claims(ctx)
    if docs.empty or remit.empty or "prior_document_sk" not in docs.columns or "decision" not in remit.columns:
        return []
    versions = docs[docs["prior_document_sk"].map(_s) != ""]
    if versions.empty:
        return []
    min_conf = float(ctx.cfg("nlp_min_extraction_confidence"))
    by_sk = docs.set_index("document_sk")

    denials = remit[remit["decision"].map(_denied)].copy()
    if denials.empty:
        return []
    denials["claim_sk"] = denials["claim_sk"].astype(str)
    denials["_when"] = pd.to_datetime(denials["settlement_date"], errors="coerce") \
        if "settlement_date" in denials.columns else pd.NaT
    denial_date = denials.groupby("claim_sk")["_when"].min().to_dict()
    denial_codes = denials.groupby("claim_sk")["denial_code"].apply(
        lambda s: sorted({_s(v) for v in s if _s(v)})).to_dict() if "denial_code" in denials.columns else {}

    # Resubmissions: the new document may sit on the corrected claim.
    prior_of: dict[str, str] = {}
    cv = _table(ctx, "claim_version")
    if not cv.empty and {"claim_sk", "prior_claim_sk"} <= set(cv.columns):
        for r in cv.itertuples(index=False):
            if _s(r.prior_claim_sk):
                prior_of[_s(r.claim_sk)] = _s(r.prior_claim_sk)

    pipeline = DocumentPipeline(ctx.config)
    indexed = claims.set_index("claim_sk") if not claims.empty else pd.DataFrame()
    out = []
    for new in versions.sort_values("document_sk").itertuples(index=False):
        prior_sk = _s(new.prior_document_sk)
        if prior_sk not in by_sk.index:
            continue
        old = by_sk.loc[prior_sk]
        if isinstance(old, pd.DataFrame):
            old = old.iloc[0]
        chain = [c for c in {new.claim_sk, _s(old["claim_sk"]), prior_of.get(new.claim_sk, "")} if c]
        dates = [denial_date[c] for c in chain if c in denial_date and not pd.isna(denial_date[c])]
        if not dates:
            continue
        denied_on = min(dates)
        if pd.isna(new.created) or new.created <= denied_on:
            continue  # written before the denial: not a post-denial alteration
        if _ADDENDUM.search(new.text) and _SIGNED.search(new.text):
            continue  # declared exclusion: valid signed addendum
        a_old = pipeline.analyse(claim_sk=prior_sk, text=old["text"], ocr_confidence=float(old["ocr"]))
        a_new = pipeline.analyse(claim_sk=new.document_sk, text=new.text, ocr_confidence=float(new.ocr))
        changes: list[dict[str, Any]] = []
        for field in ("length_of_stay", "admission_date", "discharge_date", "diagnosis_text"):
            eo, en = a_old.get(field), a_new.get(field)
            if en is None or en.confidence < min_conf:
                continue
            if eo is None:
                changes.append({"fact": field.replace("_", " "), "before": None, "after": en.value,
                                "after_span": _span_evidence(en), "before_span": None})
            elif str(eo.value).strip().lower() != str(en.value).strip().lower():
                changes.append({"fact": field.replace("_", " "), "before": eo.value, "after": en.value,
                                "after_span": _span_evidence(en), "before_span": _span_evidence(eo)})
        seen = {c["fact"] for c in changes}
        kv_old, kv_new = _kv_facts(old["text"]), _kv_facts(new.text)
        for key, (value, s0, s1) in kv_new.items():
            if key in seen or key in ("length of stay", "admission date", "discharge date", "primary diagnosis",
                                      "مدة الإقامة", "تاريخ الدخول", "تاريخ الخروج", "التشخيص الأساسي"):
                continue
            before = kv_old.get(key)
            if before is not None and before[0].strip().lower() == value.strip().lower():
                continue
            en = _extraction(key, value, new.text, s0, s1, new.ocr, new.language_detected, 0.95)
            if en is None or en.confidence < min_conf:
                continue
            eo = None if before is None else _extraction(key, before[0], old["text"], before[1], before[2],
                                                        float(old["ocr"]), a_old.language, 0.95)
            changes.append({"fact": key, "before": None if before is None else before[0], "after": value,
                            "after_span": _span_evidence(en),
                            "before_span": None if eo is None else _span_evidence(eo)})
        if not changes:
            continue
        claim_sk = new.claim_sk or _s(old["claim_sk"])
        row = indexed.loc[claim_sk] if not indexed.empty and claim_sk in indexed.index else None
        first = changes[0]
        change_words = "; ".join(
            f"{c['fact']} changed from '{c['before']}' to '{c['after']}'" if c["before"] is not None
            else f"{c['fact']} '{c['after']}' added" for c in changes[:3])
        codes = sorted({code for c in chain for code in denial_codes.get(c, [])})
        out.append(_sig(
            ctx, control, subject_type="claim", subject_id=claim_sk or new.document_sk,
            fact_key=f"postdenial:{new.document_sk}",
            claim_ids=sorted(set(chain)),
            event_time=_service_date(row) if row is not None else new.created,
            period=period_bucket(_service_date(row) if row is not None else new.created),
            confidence=round(float(first["after_span"]["extraction_confidence"]) * 0.9, 3),
            evidence={
                "claim_sk": claim_sk,
                "claims_in_chain": sorted(set(chain)),
                "provider_sk": _s(getattr(new, "provider_sk", "")) or (
                    _s(row.get("provider_sk")) if row is not None else ""),
                "member_sk": _s(getattr(new, "member_sk", "")) or (
                    _s(row.get("member_sk")) if row is not None else ""),
                "earlier_document_sk": prior_sk,
                "later_document_sk": new.document_sk,
                "earlier_version_written": str(pd.Timestamp(old["created"]).date()) if not pd.isna(old["created"]) else None,
                "later_version_written": str(new.created.date()),
                "denied_on": str(denied_on.date()),
                "denial_codes": codes,
                "changed_facts": changes,
                **first["after_span"],
                "addendum_provenance": "none — the later version is not a signed addendum",
                "synthetic_document_marker": _SYNTHETIC_MARKER,
                "plain_language": (
                    f"The claim was denied on {plain_date(denied_on)}; a new version of its "
                    f"document written on {plain_date(new.created)} changes payment-relevant facts "
                    f"({change_words}) and is not a signed addendum."
                ),
                "what_the_reviewer_must_verify": (
                    "Compare the two versions line by line and ask the provider for the signed "
                    "addendum or the reason the record was changed."
                ),
            },
            exposure=_exp.no_exposure(
                "an altered record holds the resubmission for review; the amount at stake is "
                "the resubmitted claim, not an established overpayment"
            ),
            extra_versions=_NLP_VERSION,
        ))
    return out


# ===========================================================================
# DOC-02-R04 — generic notes supporting high-complexity services
# ===========================================================================


def doc_02_r04_template_complexity_mismatch(ctx, control) -> list:
    """A provider backs high-complexity services with notes far thinner than peers write.

    "Generic" is measured on the patient-specific narrative only: template lines
    (headings, "Key: value" fields, markers) are removed first, so a shared EHR
    template does not count against anyone (declared exclusion). Each note is
    compared with high-complexity notes of the same type and language, within the
    provider's specialty when that group is large enough (declared exclusion:
    specialty), otherwise across all providers — the level used is recorded.
    """
    docs = _documents(ctx)
    lines = _lines(ctx)
    ref = _code_reference(ctx)
    claims = _claims(ctx)
    if docs.empty or lines.empty or ref.empty or claims.empty or "complexity" not in ref.columns:
        return []
    min_notes = int(ctx.cfg("doc02r04_min_notes_per_provider"))
    ratio_cut = float(ctx.cfg("doc02r04_generic_length_ratio"))
    share_cut = float(ctx.cfg("doc02r04_min_generic_share"))
    min_peer = int(ctx.cfg("min_peer_group_n"))
    max_chars = int(ctx.cfg("doc01r03_span_display_max_chars"))

    high_codes = set(ref.loc[ref["complexity"].map(_norm).isin({"high", "very_high", "complex"}),
                             "activity_code"])
    if not high_codes:
        return []
    high_lines = lines[lines["activity_code"].isin(high_codes)]
    if high_lines.empty:
        return []
    high_claims = set(high_lines["claim_sk"])
    notes = docs[docs["doc_type_norm"].map(_is_clinical) & docs["claim_sk"].isin(high_claims)].copy()
    if notes.empty:
        return []
    pinfo = claims.set_index("claim_sk")["provider_sk"].astype(str).to_dict()
    notes["provider_sk"] = notes["claim_sk"].map(pinfo)
    notes = notes[notes["provider_sk"].notna()]
    if notes.empty:
        return []
    specialty = {}
    prov = _table(ctx, "provider")
    if not prov.empty and "specialty" in prov.columns:
        specialty = dict(zip(prov["provider_sk"].astype(str), prov["specialty"].map(_norm)))
    notes["specialty"] = notes["provider_sk"].map(lambda p: specialty.get(p, ""))
    notes["specific_terms"] = notes["text"].map(lambda t: len(_narrative_tokens(t)))
    notes = notes[notes["specific_terms"] > 0].copy() if (notes["specific_terms"] > 0).any() else notes

    base_key = ["doc_type_norm", "language_detected"]
    overall = notes.groupby(base_key)["specific_terms"].agg(["median", "size"])
    by_spec = notes.groupby(base_key + ["specialty"])["specific_terms"].agg(["median", "size"])

    def peer(row) -> tuple[float, str, int]:
        k = (row.doc_type_norm, row.language_detected, row.specialty)
        if row.specialty and k in by_spec.index and by_spec.loc[k, "size"] >= min_peer:
            return float(by_spec.loc[k, "median"]), "specialty", int(by_spec.loc[k, "size"])
        k2 = (row.doc_type_norm, row.language_detected)
        return float(overall.loc[k2, "median"]), "all providers", int(overall.loc[k2, "size"])

    peers = [peer(r) for r in notes.itertuples(index=False)]
    notes["peer_median"] = [p[0] for p in peers]
    notes["peer_level"] = [p[1] for p in peers]
    notes["peer_n"] = [p[2] for p in peers]
    notes["generic"] = notes["specific_terms"] < ratio_cut * notes["peer_median"]
    overall_share = float(notes["generic"].mean())
    ref_index = ref.set_index("activity_code").to_dict("index")

    out = []
    for provider, sub in notes.groupby("provider_sk", sort=True):
        if len(sub) < min_notes:
            continue
        share = float(sub["generic"].mean())
        if share < share_cut:
            continue
        generic = sub[sub["generic"]].sort_values("document_sk")
        samples = []
        for r in generic.head(5).itertuples(index=False):
            span = _narrative_span(r.text)
            ex = None if span is None else _passage("generic_note", r.text, span, r.ocr,
                                                    r.language_detected, max_chars)
            if ex is None:
                continue
            codes = sorted(set(high_lines.loc[high_lines["claim_sk"] == r.claim_sk, "activity_code"]))
            samples.append({"document_sk": r.document_sk, "claim_sk": r.claim_sk,
                            "high_complexity_services": [_describe(c, ref_index) for c in codes],
                            "patient_specific_terms": int(r.specific_terms),
                            "peer_median_terms": round(float(r.peer_median), 1),
                            **_span_evidence(ex)})
        if not samples:
            continue  # nothing quotable: discarded, not shown with a caveat
        med = float(sub["specific_terms"].median())
        peer_med = float(sub["peer_median"].median())
        dates = claims[claims["claim_sk"].isin(set(sub["claim_sk"]))]
        last = dates["service_date"].max() if "service_date" in dates.columns else None
        out.append(_sig(
            ctx, control, subject_type="provider", subject_id=provider,
            fact_key=f"genericnotes:{provider}",
            claim_ids=sorted(set(generic["claim_sk"])),
            event_time=last,
            period=period_bucket(last),
            confidence=0.5,
            peer_level_used=str(sub["peer_level"].mode().iloc[0]),
            evidence={
                "provider_sk": provider,
                "high_complexity_notes": int(len(sub)),
                "generic_notes": int(sub["generic"].sum()),
                "generic_share": round(share, 3),
                "all_providers_generic_share": round(overall_share, 3),
                "provider_median_specific_terms": round(med, 1),
                "peer_median_specific_terms": round(peer_med, 1),
                "generic_length_ratio": ratio_cut,
                "peer_level_used": str(sub["peer_level"].mode().iloc[0]),
                "sample_generic_notes": samples,
                **{k: samples[0][k] for k in ("source_span_text", "span_offsets", "extraction_confidence",
                                              "ocr_confidence", "document_language")},
                "template_removed_before_measuring": True,
                "supporting_evidence_only": True,
                "synthetic_document_marker": _SYNTHETIC_MARKER,
                "plain_language": (
                    f"{pct(share)} of this provider's {len(sub)} notes for high-complexity services "
                    f"are generic (a median of {med:.0f} patient-specific words against {peer_med:.0f} "
                    f"for similar notes elsewhere); across all providers {pct(overall_share)} are."
                ),
                "what_the_reviewer_must_verify": (
                    "Read a sample of the notes and check whether a standard record template, rather "
                    "than missing detail, explains how short they are."
                ),
            },
            exposure=_exp.no_exposure("generic notes are supporting evidence only and carry no amount"),
            extra_versions=_NLP_VERSION,
        ))
    return out


IMPLEMENTATIONS: dict[str, Callable] = {
    fn.__name__: fn
    for fn in (
        doc_01_r01_required_document_absent,
        doc_01_r03_medical_necessity_absent,
        doc_01_r04_authorization_narrative_drift,
        doc_02_r02_contradictory_copied_facts,
        doc_02_r03_post_denial_alteration,
        doc_02_r04_template_complexity_mismatch,
    )
}
