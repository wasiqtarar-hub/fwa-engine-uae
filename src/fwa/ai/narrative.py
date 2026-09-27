"""Grounded case narrative (build brief §9.1) — advisory, cited, validated.

    "Given a correlated case, produce a reviewer-facing summary that explains
     what fired, why it matters and what the reviewer should check. **Every
     factual sentence must carry an inline citation**... Post-generate, run a
     groundedness validator... The narrative is **advisory** — it is rendered in
     a visually distinct 'AI-generated summary — not evidence' panel, and it
     **never touches disposition, priority or exposure**."

The last clause is the one worth making structural. :class:`CaseNarrator` is a
*pure function of a case it is given*: it receives a read-only evidence bundle
and returns text. It has no reference to the registry, the signal store, the
case correlator or the priority scorer, so there is no object it could mutate
even if a future edit tried. The test
``tests/governance/test_ai_cannot_change_disposition.py`` asserts that a case's
disposition, priority and exposure are byte-identical before and after
narration.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

from ..audit.log import AuditEventType
from ..enums import REVIEWER_ASK, Disposition
from .groundedness import GroundednessValidator, ValidationResult
from .provider import LLMProvider, LLMResponse

__all__ = ["CaseNarrative", "CaseNarrator", "ADVISORY_PANEL_LABEL"]

ADVISORY_PANEL_LABEL = "AI-generated summary — not evidence"


@dataclass
class CaseNarrative:
    case_id: str
    text: str
    validation: ValidationResult
    response: LLMResponse
    panel_label: str = ADVISORY_PANEL_LABEL
    disclaimer: str = (
        "This summary was generated from the evidence on this case and then checked: any "
        "sentence without a resolving citation was removed before display. It is ADVISORY. "
        "It does not set the disposition, the priority or the exposure, and it is not itself "
        "evidence — the cited artefacts are."
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "panel_label": self.panel_label,
            "narrative": self.text,
            "disclaimer": self.disclaimer,
            **self.validation.to_dict(),
            **self.response.to_audit_payload(),
        }


class CaseNarrator:
    def __init__(self, provider: LLMProvider, audit=None) -> None:
        self.provider = provider
        self.audit = audit

    # ------------------------------------------------------------------ build

    def narrate(
        self,
        case,
        signals: Sequence[Any],
        *,
        model_features: Sequence[dict[str, Any]] = (),
        document_spans: Sequence[dict[str, Any]] = (),
        actor: str = "system",
        subject_label: str | None = None,
    ) -> CaseNarrative:
        # ``subject_label`` is the MASKED identifier the caller is permitted to
        # show. The narrator never reaches for the raw subject id: a generated
        # summary that names a provider a reviewer is not cleared to see would
        # route PHI straight around the §13.2 masking rule.
        evidence = self._bundle(case, signals, model_features, document_spans, subject_label)
        prompt = (
            "Summarise this correlated payment-integrity case for a claims reviewer. Say what "
            "fired, why it matters, and what the reviewer should check. Every factual sentence "
            "must carry an inline citation in square brackets to an artefact in the evidence."
        )
        response = self.provider.generate(prompt, evidence, task="narrative")

        validator = GroundednessValidator.from_case(case, signals, model_features, document_spans)
        validation = validator.validate(response.text)

        if self.audit is not None:
            self.audit.record(
                AuditEventType.AI_TEXT_GENERATED,
                actor=actor, actor_role="AI_LAYER", tenant_id=case.tenant_id,
                subject=case.case_id,
                reason="Grounded case narrative generated.",
                after={**response.to_audit_payload(),
                       "sentences_dropped": validation.drop_count},
            )
            if validation.drop_count:
                self.audit.record(
                    AuditEventType.AI_CLAIM_DROPPED_UNGROUNDED,
                    actor=actor, actor_role="AI_LAYER", tenant_id=case.tenant_id,
                    subject=case.case_id,
                    reason=f"{validation.drop_count} uncited or unresolvable sentence(s) dropped "
                           f"before display.",
                    after={"dropped": [s for s, _ in validation.dropped]},
                )
        return CaseNarrative(
            case_id=case.case_id, text=validation.text, validation=validation, response=response
        )

    # --------------------------------------------------------------- evidence

    @staticmethod
    def _bundle(case, signals, model_features, document_spans,
                subject_label: str | None = None) -> dict[str, Any]:
        ask = REVIEWER_ASK.get(case.disposition, "")
        exposure_line = (
            f"Exposure is AED {case.exposure_aed:,.2f}."
            if case.exposure_established else
            f"Gross amount is AED {case.exposure_aed:,.2f}, but exposure not yet established."
        )
        priority_line = ""
        if case.priority is not None:
            top = max(case.priority.terms, key=lambda t: t.contribution)
            priority_line = (
                f"Priority is {case.priority.value:.0f} of 100 ({case.priority.band_label}), "
                f"driven most by {top.label.lower()}."
            )
        return {
            "case_id": case.case_id,
            "subject_label": f"one {case.primary_subject_type} case on "
                             f"{subject_label or case.primary_subject_id}",
            "disposition": case.disposition.value,
            "reviewer_ask": ask,
            "signals": [
                {
                    "signal_id": s.signal_id,
                    "rule_id": s.rule_id,
                    "rule_version": s.rule_version,
                    "headline": CaseNarrator._headline(s),
                }
                for s in signals
            ],
            "model_features": [
                {**f, "model": f.get("model", "model")} for f in model_features
                if f.get("plain_language")
            ],
            "document_spans": list(document_spans),
            "exposure_line": exposure_line,
            "priority_line": priority_line,
            "suggested_checks": CaseNarrator._checks(case, signals),
            "evidence_ids": [s.signal_id for s in signals],
        }

    @staticmethod
    def _headline(signal) -> str:
        ev = signal.evidence or {}
        for key in ("plain_language", "what_the_reviewer_must_verify", "shrinkage_explanation"):
            if ev.get(key):
                return str(ev[key])
        return f"{signal.rule_id} fired with reason code {signal.reason_code}."

    @staticmethod
    def _checks(case, signals) -> list[str]:
        checks: list[str] = []
        ask = REVIEWER_ASK.get(case.disposition)
        if ask:
            checks.append(ask)
        families = {s.scenario_id.split("-")[0] for s in signals}
        if "PAY" in families:
            checks.append("Confirm the matched claim is genuinely the same service, not a "
                          "legitimate repeat or a staged procedure.")
        if "CLN" in families:
            checks.append("Obtain the clinical record and confirm what it supports.")
        if "NET" in families:
            checks.append("Check whether the relationship has a declared corporate or "
                          "contractual explanation before treating concentration as a finding.")
        proxies = {
            s.evidence.get("proxy_note") for s in signals if s.evidence.get("proxy_note")
        }
        if proxies:
            checks.append("Note that at least one signal here rests on a PROXY rather than the "
                          "field the control was designed for; weigh it accordingly.")
        checks.append("If you clear this case, record why — the rationale is what tunes the rule.")
        return checks
