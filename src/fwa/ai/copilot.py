"""The reviewer copilot — retrieval-bounded, with a hard-coded refusal (§9.3).

    "**Reviewer copilot**, strictly retrieval-bounded to the current case's
     evidence bundle, the rule text and the parameter registry. It answers 'why
     did this fire?', 'what exclusion might apply?', 'what would I need to
     confirm this?' It **refuses** to answer 'is this fraud?' — returning the
     §3.3 boundary statement instead. **Hard-code that refusal; do not leave it
     to the prompt.**"                                      — build brief §9.3

The refusal is implemented before any provider is called. :meth:`Copilot.ask`
classifies the question first; a conduct question returns the boundary statement
and never reaches the LLM. That ordering is the whole point: a refusal enforced
in the prompt is a refusal that a sufficiently determined rephrasing can talk
its way past, and the manuscript treats this boundary as the one property the
system must not be able to lose.

"Retrieval-bounded" is also structural. :meth:`Copilot.ask` builds the answer
from three sources and no others — the case's own signals, the rule text of the
controls that fired, and the parameter registry entries those rules reference.
There is no path from this class to the full claim table, to another tenant's
data, or to the held-out labels.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Sequence

from .. import SAFETY_BOUNDARY_STATEMENT
from ..audit.log import AuditEventType
from ..enums import DISPOSITION_MEANINGS, REVIEWER_ASK
from .provider import LLMProvider

__all__ = ["CopilotAnswer", "Copilot", "REFUSAL_PATTERNS"]

#: Questions about CONDUCT. Matching any of these returns the boundary
#: statement without calling a model. The list is deliberately broad: a false
#: refusal costs a reviewer one rephrasing, while a false answer costs the
#: system the property §3.3 exists to guarantee.
REFUSAL_PATTERNS = [
    r"\bis (this|it|that) fraud\b",
    r"\bis (this|it|that) fraudulent\b",
    r"\b(did|has|have) .{0,40}\b(commit|committed|commits)\b",
    r"\bare they (guilty|defrauding|cheating|stealing)\b",
    r"\bis (this|the) (provider|member|agent|clinic|hospital) (a )?(fraud|fraudster|criminal)\b",
    r"\bshould (i|we) (deny|reject|refuse)\b",
    r"\bwhat disposition should\b",
    r"\bdecide (this|the) (case|claim) for me\b",
    r"\bhow (guilty|suspicious) (is|are)\b",
    r"\bprove (that )?(they|he|she|it)\b",
    r"\bintent\b.{0,20}\b(prove|establish|confirm)\b",
    # ...and the same question with the words the other way round ("prove
    # intent", "establish intent"). Intent is precisely what §3.3 says this
    # system cannot establish, so the word order must not decide the answer.
    r"\b(prove|establish|confirm|show|demonstrate)\w*\b.{0,20}\bintent\w*\b",
    r"\b(deliberate|deliberately|intentional|intentionally|knowingly|on purpose)\b",
]

_REFUSAL_RE = [re.compile(p, re.I) for p in REFUSAL_PATTERNS]

REFUSAL_TEXT = (
    f"{SAFETY_BOUNDARY_STATEMENT}\n\n"
    "I can't answer that, and it isn't a limitation of the model — it is the boundary this "
    "system is built around. What I can do is show you what fired, what the rule actually "
    "tests, which exclusions were declared, and what evidence would confirm or clear it. "
    "The conclusion about conduct is yours to reach, on the evidence, and to record with "
    "your rationale."
)


@dataclass
class CopilotAnswer:
    question: str
    answer: str
    refused: bool
    sources: list[str] = field(default_factory=list)
    bounded_to: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "answer": self.answer,
            "refused": self.refused,
            "sources": self.sources,
            "retrieval_bounded_to": self.bounded_to,
            "panel_label": "AI-generated answer — not evidence",
        }


class Copilot:
    def __init__(self, provider: LLMProvider, registry, config, audit=None) -> None:
        self.provider = provider
        self.registry = registry
        self.config = config
        self.audit = audit

    # -------------------------------------------------------------------- ask

    def ask(
        self,
        question: str,
        case,
        signals: Sequence[Any],
        *,
        actor: str = "reviewer",
    ) -> CopilotAnswer:
        # ---- HARD-CODED REFUSAL, before any model call --------------------
        if self.is_conduct_question(question):
            if self.audit is not None:
                self.audit.record(
                    AuditEventType.AI_COPILOT_REFUSED,
                    actor=actor, actor_role="AI_LAYER", tenant_id=case.tenant_id,
                    subject=case.case_id,
                    reason="Conduct question refused at the safety boundary, before any model call.",
                    after={"question": question},
                )
            return CopilotAnswer(question=question, answer=REFUSAL_TEXT, refused=True)

        bundle, sources = self._retrieval_bundle(question, case, signals)
        response = self.provider.generate(
            f"Answer the reviewer's question using ONLY the supplied evidence: {question}",
            bundle, task="copilot",
        )
        return CopilotAnswer(
            question=question,
            answer=response.text,
            refused=False,
            sources=sources,
            bounded_to=[
                "this case's evidence bundle",
                "the rule text of the controls that fired",
                "the parameter registry entries those rules reference",
            ],
        )

    @staticmethod
    def is_conduct_question(question: str) -> bool:
        return any(rx.search(question or "") for rx in _REFUSAL_RE)

    # -------------------------------------------------------------- retrieval

    def _retrieval_bundle(self, question: str, case, signals) -> tuple[dict[str, Any], list[str]]:
        q = (question or "").lower()
        points: list[str] = []
        sources: list[str] = []

        rule_ids = sorted({s.rule_id for s in signals})
        for rule_id in rule_ids:
            try:
                control = self.registry.get(rule_id)
            except Exception:
                continue
            sources.append(f"{control.rule_id}@{control.version}")

            if "why" in q or "fire" in q or "trigger" in q:
                points.append(
                    f"{control.rule_id}@{control.version} ({control.name}) fired. It tests: "
                    f"{control.expression}"
                )
                for s in signals:
                    if s.rule_id != rule_id:
                        continue
                    headline = s.evidence.get("plain_language") or s.evidence.get(
                        "what_the_reviewer_must_verify")
                    if headline:
                        points.append(f"On this case: {headline}")

            if "exclusion" in q or "exception" in q or "clear" in q:
                points.append(
                    f"{control.rule_id} declares these exclusions: "
                    + ", ".join(control.exclusions)
                    + ". If one of them applies here, the signal should be cleared with that "
                      "exclusion named in your rationale."
                )

            if "confirm" in q or "need" in q or "evidence" in q or "check" in q:
                points.append(
                    f"{control.rule_id} renders these evidence fields: "
                    + ", ".join(control.evidence_fields)
                    + f". Its required canonical inputs are: " + ", ".join(control.inputs) + "."
                )
                if control.required_canonical_fields:
                    points.append(
                        "Fields this dataset does not carry, which would strengthen the finding: "
                        + ", ".join(control.required_canonical_fields)
                    )

            if "threshold" in q or "parameter" in q or "cfg" in q or "number" in q:
                for key, value in (control.parameters or {}).items():
                    ref = str(value)
                    if ref.startswith("cfg."):
                        param = ref[4:]
                        try:
                            rec = self.config.parameters.record(param)
                            points.append(
                                f"{key} = cfg.{param} = {rec.value} — owner {rec.owner}. "
                                f"Rationale: {rec.rationale.strip()[:220]}"
                            )
                            sources.append(f"param:{param}")
                        except Exception:
                            continue

            if "disposition" in q or "what happens" in q or "next" in q:
                points.append(
                    f"The case disposition is {case.disposition.value}: "
                    f"{DISPOSITION_MEANINGS.get(case.disposition, '')} "
                    f"What that asks of you: {REVIEWER_ASK.get(case.disposition, '')}"
                )

            if "proxy" in q or "reliable" in q or "trust" in q or "weak" in q:
                for s in signals:
                    note = s.evidence.get("proxy_note")
                    if note:
                        points.append(f"{s.rule_id}: {note}")

        if "priority" in q and case.priority is not None:
            for term in sorted(case.priority.terms, key=lambda t: -abs(t.contribution))[:3]:
                points.append(
                    f"Priority term '{term.label}' contributed {term.contribution:+.3f}. "
                    f"{term.explanation}"
                )
            sources.append(f"priority:{case.case_id}")

        if "exposure" in q or "worth" in q or "amount" in q:
            points.append(case.exposure_basis or "No exposure basis was recorded for this case.")
            if not case.exposure_established:
                points.append(
                    "Note: exposure on this case is NOT ESTABLISHED. The amount shown is gross, "
                    "not a recoverable or preventable figure."
                )
            sources.append(f"exposure:{case.case_id}")

        if not points:
            points.append(
                "I'm bounded to this case's evidence, the rule text of the controls that fired, "
                "and the parameter registry. None of those contains an answer to that. Try "
                "asking why a specific rule fired, what exclusions it declares, or what evidence "
                "would confirm it."
            )

        return {"answer_points": points, "evidence_ids": [s.signal_id for s in signals]}, sorted(set(sources))
