"""Document controls (Type T) — DOC-01-R02 and DOC-02-R01.

Both run **only** against the clearly-labelled synthetic corpus, because
the claim extract contains no documents. Every signal says so on its face, in an
evidence field a reviewer cannot miss, so a demonstration of the pipeline can
never be mistaken for a finding about a real record.

Both also honour the §4.9 non-negotiable: the finding carries its exact source
span, and a finding whose span could not be located, or whose OCR-adjusted
confidence falls below ``cfg.nlp_min_extraction_confidence``, was already
discarded upstream by :class:`~fwa.nlp.pipeline.DocumentPipeline` and never
reaches this module.
"""

from __future__ import annotations

import itertools
import re
from typing import Any, Callable

import pandas as pd

from ..cases import exposure as _exp
from ..engine.evallib import period_bucket

__all__ = ["doc_01_r02_structured_fact_conflict", "doc_02_r01_cross_patient_near_duplicate"]

#: Boilerplate stripped before similarity comparison: "remove standard
#: boilerplate before comparison". Without this every document in a templated
#: corpus is a near-duplicate of every other, and the control would be
#: measuring the template rather than the clinician.
#:
#: Every entry is a TEMPLATE LINE, and each ``.*`` means "the rest of this
#: line". That is why the pattern below is compiled without ``re.DOTALL``:
#: with it, ``.`` also matches a newline, ``DISCHARGE SUMMARY.*`` runs greedily
#: from the first heading to the end of the file, and :func:`_strip_boilerplate`
#: returns an empty string for every document ever written. The token set is
#: then empty, no document reaches the comparison, and DOC-02-R01 cannot fire
#: on any dataset — silently, because a control that finds nothing looks
#: exactly like a control with nothing to find.
_BOILERPLATE = [
    r"\*\*\*.*?\*\*\*",
    r"DISCHARGE SUMMARY.*",
    r"Facility reference.*", r"Patient reference.*", r"Claim reference.*",
    r"Admission date.*", r"Discharge date.*", r"Length of stay.*", r"Primary diagnosis.*",
    r"CLINICAL COURSE", r"MEDICATION AND SUPPLIES", r"DISPOSITION",
    r"--- END OF.*", r"This file was generated for software testing.*",
    r"ملخص الخروج.*", r"المرجع المنشأة.*", r"مرجع المريض.*", r"مرجع المطالبة.*",
    r"تاريخ الدخول.*", r"تاريخ الخروج.*", r"مدة الإقامة.*", r"التشخيص الأساسي.*",
    r"المسار السريري", r"الأدوية والمستلزمات", r"الخروج",
    r"--- نهاية وثيقة.*", r"تم إنشاء هذا الملف.*",
]
_BOILERPLATE_RE = re.compile("|".join(_BOILERPLATE))


def _strip_boilerplate(text: str) -> str:
    """Remove the template, keep the narrative.

    What survives is the prose a clinician would have written: the course of
    the admission, the medication note, the discharge sentence. That is what
    DOC-02-R01 compares across members, and it is the only part of a templated
    document where a near-duplicate means anything.
    """
    return re.sub(r"\s+", " ", _BOILERPLATE_RE.sub(" ", text)).strip()


def doc_01_r02_structured_fact_conflict(ctx, control, _sig: Callable) -> list[Any]:
    """DOC-01-R02 — an extracted fact contradicts the claim."""
    pipeline = getattr(ctx.documents, "pipeline", None)
    if pipeline is None:
        return []
    claims = ctx.claims
    claims = claims[claims["tenant_id"] == ctx.tenant_id] if "tenant_id" in claims.columns else claims
    indexed = claims.set_index("claim_sk")

    out: list[Any] = []
    for claim_sk, analysis in pipeline.analyses.items():
        if claim_sk not in indexed.index:
            continue
        row = indexed.loc[claim_sk]
        for conflict in pipeline.compare(analysis, row):
            out.append(
                _sig(
                    ctx, control, subject_type="claim", subject_id=claim_sk,
                    fact_key=f"doc_conflict:{claim_sk}:{conflict['conflicting_field']}",
                    claim_ids=[claim_sk],
                    event_time=row["service_date"],
                    period=period_bucket(row["service_date"]),
                    confidence=float(conflict["extraction_confidence"]),
                    data_quality_penalty=round(1.0 - float(conflict["ocr_confidence"]), 3),
                    evidence={
                        **conflict,
                        "gross_amount_aed": round(float(row["gross_amount_aed"]), 2),
                        "provider_sk": row["provider_sk"],
                        "span_grounding_notice": (
                            "Every fact above is anchored to the exact text at the stated "
                            "offsets.: a finding with no source span is discarded by the "
                            "pipeline, not displayed with a caveat."
                        ),
                    },
                    exposure=_exp.no_exposure(
                        "a documentation conflict establishes that the record does not support "
                        "the claim as billed, not the amount by which it does not"
                    ),
                    extra_versions={"nlp_pipeline": "rule-based deterministic extraction v1.0.0"},
                )
            )
    return out


def doc_02_r01_cross_patient_near_duplicate(ctx, control, _sig: Callable) -> list[Any]:
    """DOC-02-R01 — near-identical patient-specific content across members."""
    corpus = ctx.documents
    pipeline = getattr(corpus, "pipeline", None)
    if pipeline is None:
        return []
    threshold = float(ctx.cfg("doc_clone_similarity_threshold"))
    claims = ctx.claims
    claims = claims[claims["tenant_id"] == ctx.tenant_id] if "tenant_id" in claims.columns else claims
    indexed = claims.set_index("claim_sk")

    stripped: dict[str, tuple[str, frozenset[str]]] = {}
    for claim_sk, analysis in pipeline.analyses.items():
        if claim_sk not in indexed.index:
            continue
        body = _strip_boilerplate(analysis.text)
        tokens = frozenset(t for t in re.findall(r"\w+", body.lower()) if len(t) > 2)
        if tokens:
            stripped[claim_sk] = (body, tokens)

    out: list[Any] = []
    seen_pairs: set[tuple[str, str]] = set()
    by_provider: dict[str, list[str]] = {}
    for claim_sk in stripped:
        by_provider.setdefault(str(indexed.loc[claim_sk, "provider_sk"]), []).append(claim_sk)

    for provider, claim_ids in by_provider.items():
        for a, b in itertools.combinations(sorted(claim_ids), 2):
            member_a = str(indexed.loc[a, "member_sk"])
            member_b = str(indexed.loc[b, "member_sk"])
            if member_a == member_b:
                continue  # declared exclusion: same_member_documents
            ta, tb = stripped[a][1], stripped[b][1]
            union = len(ta | tb)
            similarity = len(ta & tb) / union if union else 0.0
            if similarity <= threshold:
                continue
            pair = (a, b)
            if pair in seen_pairs:
                continue
            seen_pairs.add(pair)
            out.append(
                _sig(
                    ctx, control, subject_type="provider", subject_id=provider,
                    fact_key=f"doc_clone:{a}:{b}",
                    claim_ids=[a, b],
                    event_time=indexed.loc[a, "service_date"],
                    period=period_bucket(indexed.loc[a, "service_date"]),
                    confidence=0.4,
                    data_quality_penalty=0.5,
                    evidence={
                        "document_pair": [a, b],
                        "similarity": round(similarity, 4),
                        "threshold": threshold,
                        "member_sks": [member_a, member_b],
                        "provider_sk": provider,
                        "boilerplate_removed": True,
                        "excerpt_a": stripped[a][0][:220],
                        "excerpt_b": stripped[b][0][:220],
                        "synthetic_document_marker": (
                            "SYNTHETIC — both documents were generated by this artefact from a "
                            "shared template."
                        ),
                        "template_baseline_caveat": (
                            "This corpus is TEMPLATE-GENERATED, so a substantial baseline "
                            "similarity is expected even between unrelated documents. The "
                            "threshold sits above that baseline, but a similarity finding here "
                            "demonstrates the mechanism and says nothing about cloned "
                            "documentation in real clinical records."
                        ),
                    },
                    exposure=_exp.no_exposure(
                        "cloned documentation is an audit-sample trigger; it carries no directly "
                        "attributable over-payment"
                    ),
                )
            )
    return out
