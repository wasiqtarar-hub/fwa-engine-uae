"""The document / NLP pipeline (manuscript §4.9).

    "The document pipeline follows the specification's required sequence:
     document classification, OCR, language detection, clinical entity/negation
     extraction, structured comparison against the billed claim, and evidence
     rendering."                                            — manuscript §4.9

    "**Non-negotiable:** no free-form generative model output may be persisted
     as a factual clinical finding WITHOUT A SOURCE SPAN: every DOC-01/DOC-02
     and CLN-01-R03 style signal must display the exact supporting or
     conflicting text alongside its OCR-confidence-adjusted extraction
     confidence, so a reviewer is always evaluating a grounded excerpt, never an
     ungrounded model assertion."                           — manuscript §4.9

The span requirement is implemented as a **structural** property, not a check
applied afterwards: :class:`Extraction` cannot be constructed without a span,
and :meth:`DocumentPipeline.extract` discards any candidate finding whose span
cannot be located in the source text. The build brief is explicit that "a
finding with no span is DISCARDED by the pipeline, not displayed with a caveat",
and :attr:`DocumentPipeline.discarded` counts how many were dropped so the
discard rate is reportable rather than invisible.

Extraction here is **deterministic and rule-based**, not generative. That is a
deliberate choice for an offline artefact: a regex that finds "Length of stay :
14 day(s)" and returns the exact character offsets is trivially auditable, and
its failure modes are inspectable. The AI layer (:mod:`fwa.ai`) may *narrate*
these findings, but it never produces them.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Iterable

import pandas as pd

__all__ = ["Extraction", "DocumentAnalysis", "DocumentPipeline", "documentation_level_conflicts"]


class MissingSpanError(ValueError):
    """A finding was constructed without a source span. Always fatal."""


@dataclass(frozen=True)
class Extraction:
    """One extracted fact, with the exact text it came from."""

    field_name: str
    value: Any
    span_text: str
    span_start: int
    span_end: int
    raw_confidence: float
    ocr_confidence: float
    language: str

    def __post_init__(self) -> None:
        if not self.span_text or self.span_start < 0 or self.span_end <= self.span_start:
            raise MissingSpanError(
                f"Extraction of {self.field_name!r} has no usable source span. Persisting a "
                "clinical finding without one is forbidden, so the pipeline discards it."
            )

    @property
    def confidence(self) -> float:
        """Extraction confidence ADJUSTED FOR OCR QUALITY (§4.9).

        A perfect regex match on a badly scanned page is not a confident
        finding, and §6.3 requires that NLP results "visibly degrade their
        confidence when OCR quality is poor" rather than presenting a
        low-quality extraction with unwarranted certainty.
        """
        return round(self.raw_confidence * self.ocr_confidence, 4)

    def to_dict(self) -> dict[str, Any]:
        return {
            "field": self.field_name,
            "value": self.value,
            "source_span_text": self.span_text,
            "span_offsets": [self.span_start, self.span_end],
            "raw_extraction_confidence": self.raw_confidence,
            "ocr_confidence": self.ocr_confidence,
            "extraction_confidence": self.confidence,
            "language": self.language,
        }


@dataclass
class DocumentAnalysis:
    claim_sk: str
    document_type: str
    language: str
    ocr_confidence: float
    extractions: dict[str, Extraction] = field(default_factory=dict)
    negations: list[str] = field(default_factory=list)
    text: str = ""
    discarded: list[str] = field(default_factory=list)

    def get(self, field_name: str) -> Extraction | None:
        return self.extractions.get(field_name)


# ---------------------------------------------------------------------------
# patterns — English and Arabic, kept side by side so the language split is
# visible rather than buried in a language-detection branch
# ---------------------------------------------------------------------------

_PATTERNS = {
    "en": {
        "length_of_stay": re.compile(r"Length of stay\s*:\s*(\d+)\s*day", re.I),
        "length_of_stay_body": re.compile(r"recorded inpatient stay was\s+(\d+)\s*day", re.I),
        "admission_date": re.compile(r"Admission date\s*:\s*(\d{4}-\d{2}-\d{2})"),
        "admission_date_body": re.compile(r"admitted on\s+(\d{4}-\d{2}-\d{2})"),
        "discharge_date": re.compile(r"Discharge date\s*:\s*(\d{4}-\d{2}-\d{2})"),
        "diagnosis_text": re.compile(r"Primary diagnosis\s*:\s*\S+\s*—\s*(.+)"),
        "diagnosis_body": re.compile(r"presentation consistent with\s+(.+?)\.", re.S),
        "pharmacy_share": re.compile(r"approximately\s+(\d+)%\s+of the total billed", re.I),
    },
    "ar": {
        "length_of_stay": re.compile(r"مدة الإقامة\s*:\s*(\d+)\s*يوم"),
        "length_of_stay_body": re.compile(r"بلغت مدة الإقامة المسجلة\s+(\d+)\s*يوم"),
        "admission_date": re.compile(r"تاريخ الدخول\s*:\s*(\d{4}-\d{2}-\d{2})"),
        "admission_date_body": re.compile(r"بتاريخ\s+(\d{4}-\d{2}-\d{2})"),
        "discharge_date": re.compile(r"تاريخ الخروج\s*:\s*(\d{4}-\d{2}-\d{2})"),
        "diagnosis_text": re.compile(r"التشخيص الأساسي\s*:\s*\S+\s*—\s*(.+)"),
        "diagnosis_body": re.compile(r"تتوافق مع\s+(.+?)\."),
        "pharmacy_share": re.compile(r"حوالي\s+(\d+)%"),
    },
}

#: Negation cues. Extraction that falls inside a negated clause must not be
#: reported as an asserted finding — the classic clinical-NLP failure mode.
_NEGATION_CUES = {
    "en": ["no ", "not ", "without ", "denies ", "ruled out", "negative for", "absent"],
    "ar": ["لا ", "لم ", "بدون ", "ينفي", "غير "],
}


class DocumentPipeline:
    """Executes the §4.9 sequence over the synthetic corpus."""

    def __init__(self, config) -> None:
        self.config = config
        self.min_confidence = float(config.get("nlp_min_extraction_confidence"))
        self.analyses: dict[str, DocumentAnalysis] = {}
        self.discarded: int = 0
        self.discard_reasons: list[str] = []

    # ------------------------------------------------------------------- run

    def run(self, corpus) -> dict[str, DocumentAnalysis]:
        for claim_sk, doc in corpus.documents.items():
            analysis = self.analyse(
                claim_sk=claim_sk, text=doc.text, ocr_confidence=doc.ocr_confidence
            )
            self.analyses[claim_sk] = analysis
        return self.analyses

    def analyse(self, *, claim_sk: str, text: str, ocr_confidence: float) -> DocumentAnalysis:
        doc_type = self.classify(text)
        text = self.ocr_hook(text)
        language = self.detect_language(text)
        analysis = DocumentAnalysis(
            claim_sk=claim_sk, document_type=doc_type, language=language,
            ocr_confidence=ocr_confidence, text=text,
        )
        analysis.extractions = self.extract(text, language, ocr_confidence, analysis)
        analysis.negations = self.find_negations(text, language)
        return analysis

    # ---------------------------------------------------- §4.9 sequence steps

    @staticmethod
    def classify(text: str) -> str:
        """1. Document classification."""
        lowered = text.lower()
        if "discharge summary" in lowered or "ملخص الخروج" in text:
            return "discharge_summary"
        if "operative" in lowered:
            return "operative_note"
        if "invoice" in lowered:
            return "invoice"
        return "unclassified"

    @staticmethod
    def ocr_hook(text: str) -> str:
        """2. OCR hook.

        A seam, not an implementation. The corpus is generated as text, so there
        is nothing to OCR; a real deployment substitutes an OCR engine here and
        the rest of the pipeline is unchanged. Saying "OCR is implemented" of a
        pass-through would be untrue, so it is documented as the hook it is.
        """
        return text

    @staticmethod
    def detect_language(text: str) -> str:
        """3. Language detection — by Arabic script presence, deliberately simple."""
        arabic_chars = sum(1 for ch in text if "؀" <= ch <= "ۿ")
        return "ar" if arabic_chars > 20 else "en"

    def extract(
        self, text: str, language: str, ocr_confidence: float, analysis: DocumentAnalysis
    ) -> dict[str, Extraction]:
        """4. Clinical entity extraction — every result carries its span."""
        patterns = _PATTERNS.get(language, _PATTERNS["en"])
        out: dict[str, Extraction] = {}
        for name, pattern in patterns.items():
            match = pattern.search(text)
            if not match:
                continue
            raw_value = match.group(1).strip()
            # "_body" patterns are CORROBORATION for the same canonical field
            # as their header counterpart, not a separate field. Mapping them
            # onto one name keeps the extraction set aligned with the claim
            # fields it is compared against.
            field_name = {
                "length_of_stay_body": "length_of_stay",
                "admission_date_body": "admission_date",
                "diagnosis_body": "diagnosis_text",
            }.get(name, name)
            if field_name in out:
                continue  # header match wins; the body match is corroboration
            value: Any = raw_value
            confidence = 0.95
            if field_name in ("length_of_stay", "pharmacy_share"):
                try:
                    value = int(raw_value)
                except ValueError:
                    continue
            try:
                extraction = Extraction(
                    field_name=field_name,
                    value=value,
                    span_text=text[match.start(): match.end()].strip(),
                    span_start=match.start(),
                    span_end=match.end(),
                    raw_confidence=confidence,
                    ocr_confidence=ocr_confidence,
                    language=language,
                )
            except MissingSpanError as exc:
                self.discarded += 1
                self.discard_reasons.append(str(exc))
                analysis.discarded.append(field_name)
                continue
            out[field_name] = extraction
        return out

    @staticmethod
    def find_negations(text: str, language: str) -> list[str]:
        """4b. Negation detection.

        A sentence containing a negation cue is recorded so that a downstream
        comparison does not read a negated statement as an assertion.
        """
        cues = _NEGATION_CUES.get(language, _NEGATION_CUES["en"])
        sentences = re.split(r"[.\n]", text)
        return [s.strip() for s in sentences if any(c in s.lower() for c in cues) and s.strip()]

    def compare(
        self, analysis: DocumentAnalysis, claim_row: pd.Series
    ) -> list[dict[str, Any]]:
        """5. Structured comparison against the billed claim.

        Returns one dict per CONFLICT. A finding is emitted only when the
        extraction's OCR-adjusted confidence clears
        ``cfg.nlp_min_extraction_confidence`` — a low-quality extraction is not
        presented with unwarranted certainty (§6.3).
        """
        conflicts: list[dict[str, Any]] = []

        los = analysis.get("length_of_stay")
        if los is not None:
            claimed = int(claim_row["length_of_stay_days"])
            if int(los.value) != claimed:
                conflicts.append(self._conflict(
                    analysis, los, "length_of_stay_days", los.value, claimed,
                    f"The note states a stay of {los.value} day(s); the claim bills "
                    f"{claimed} day(s).",
                ))

        admit = analysis.get("admission_date")
        if admit is not None:
            claimed_admit = str(pd.Timestamp(claim_row["service_date"]).date())
            if str(admit.value) != claimed_admit:
                conflicts.append(self._conflict(
                    analysis, admit, "admission_date", admit.value, claimed_admit,
                    f"The note states an admission date of {admit.value}; the claim bills "
                    f"{claimed_admit}.",
                ))

        dx = analysis.get("diagnosis_text")
        if dx is not None:
            claimed_dx = str(claim_row.get("diagnosis_description", "") or "")
            if claimed_dx and str(dx.value).strip().lower() != claimed_dx.strip().lower():
                conflicts.append(self._conflict(
                    analysis, dx, "diagnosis_description", dx.value, claimed_dx,
                    f"The note describes '{dx.value}'; the claim codes "
                    f"'{claimed_dx}'.",
                ))

        return [c for c in conflicts if c is not None]

    def _conflict(
        self, analysis: DocumentAnalysis, extraction: Extraction,
        claim_field: str, extracted: Any, claimed: Any, sentence: str,
    ) -> dict[str, Any] | None:
        if extraction.confidence < self.min_confidence:
            self.discarded += 1
            self.discard_reasons.append(
                f"{analysis.claim_sk}/{extraction.field_name}: extraction confidence "
                f"{extraction.confidence:.2f} below cfg.nlp_min_extraction_confidence "
                f"{self.min_confidence:.2f} (OCR {extraction.ocr_confidence:.2f}) — discarded."
            )
            return None
        return {
            "claim_sk": analysis.claim_sk,
            "conflicting_field": claim_field,
            "extracted_value": extracted,
            "claim_value": claimed,
            "plain_language": sentence,
            "document_type": analysis.document_type,
            "document_language": analysis.language,
            "synthetic_document_marker": "SYNTHETIC — generated by this artefact; the claim extract "
                                         "contains no documents",
            "negated_sentences_in_document": len(analysis.negations),
            **extraction.to_dict(),
        }

    # ---------------------------------------------------------------- reports

    def quality_by_language_and_type(self) -> pd.DataFrame:
        """§4.9: report extraction quality SEPARATELY by language and document type.

        "Arabic, English and mixed-language clinical documentation in the UAE
        context cannot be assumed to perform identically under one aggregate
        accuracy figure."
        """
        rows = []
        for a in self.analyses.values():
            for name, ex in a.extractions.items():
                rows.append({
                    "language": a.language,
                    "document_type": a.document_type,
                    "field": name,
                    "extraction_confidence": ex.confidence,
                    "ocr_confidence": ex.ocr_confidence,
                    "below_threshold": ex.confidence < self.min_confidence,
                })
        if not rows:
            return pd.DataFrame()
        frame = pd.DataFrame(rows)
        return (
            frame.groupby(["language", "document_type", "field"], as_index=False)
            .agg(
                extractions=("extraction_confidence", "size"),
                mean_confidence=("extraction_confidence", "mean"),
                mean_ocr=("ocr_confidence", "mean"),
                below_threshold=("below_threshold", "sum"),
            )
            .round(3)
        )

    def summary(self) -> dict[str, Any]:
        langs: dict[str, int] = {}
        for a in self.analyses.values():
            langs[a.language] = langs.get(a.language, 0) + 1
        return {
            "documents_analysed": len(self.analyses),
            "by_language": langs,
            "findings_discarded": self.discarded,
            "discard_note": (
                "A finding with no source span, or with an OCR-adjusted confidence below "
                "cfg.nlp_min_extraction_confidence, is DISCARDED by the pipeline — not shown "
                "with a caveat."
            ),
        }


# ---------------------------------------------------------------------------
# CLN-01-R03 helper, called from fwa.engine.controls
# ---------------------------------------------------------------------------


def documentation_level_conflicts(ctx) -> list[dict[str, Any]]:
    """Conflicts between the documented stay and the billed intensity (CLN-01-R03)."""
    pipeline = getattr(ctx.documents, "pipeline", None)
    if pipeline is None:
        return []
    claims = ctx.claims.set_index("claim_sk")
    out: list[dict[str, Any]] = []
    for claim_sk, analysis in pipeline.analyses.items():
        if claim_sk not in claims.index:
            continue
        row = claims.loc[claim_sk]
        los = analysis.get("length_of_stay")
        if los is None:
            continue
        billed_los = int(row["length_of_stay_days"])
        if int(los.value) >= billed_los:
            continue  # only a SHORTER documented stay understates support for the bill
        conflict = pipeline._conflict(
            analysis, los, "length_of_stay_days", los.value, billed_los,
            f"The documentation supports a {los.value}-day stay; the claim bills "
            f"{billed_los} days at AED {float(row['gross_amount_aed']):,.0f}.",
        )
        if conflict is None:
            continue
        conflict["service_date"] = row["service_date"]
        conflict["billed_amount_aed"] = round(float(row["gross_amount_aed"]), 2)
        conflict["documented_days"] = int(los.value)
        conflict["billed_days"] = billed_los
        out.append(conflict)
    return out
