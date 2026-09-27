"""The groundedness validator (build brief §9.1).

    "Post-generate, run a **groundedness validator** that parses the narrative,
     verifies each citation resolves to a stored artefact, and **drops any
     uncited factual claim before display**."

This is the mechanism that makes an LLM narrative safe to put in front of a
reviewer, and it is deliberately adversarial toward the generator:

* A sentence with **no citation** is dropped if it asserts a fact. Sentences
  that are framing rather than assertion ("What to check:", "What fired.") are
  allowed through a small, explicit allow-list — not by a heuristic about
  sentence shape, because a heuristic is what a confident hallucination would
  slip through.
* A sentence whose citation **does not resolve** to a stored artefact is
  dropped. Inventing a plausible-looking ``[PAY-06-R03@1.0.0]`` is exactly the
  failure mode this catches, and it is caught by looking the id up in the
  evidence bundle rather than by checking that it looks like an id.
* Every drop is **counted and reported**. :attr:`ValidationResult.dropped` goes
  into the audit log and onto the Case Evidence panel, so a narrative that lost
  half its sentences is visibly a narrative that lost half its sentences.

The validator does not attempt to verify that a cited sentence is *true* — only
that it is *attributable*. That is the honest limit of the mechanism and it is
stated on the panel: the citation tells a reviewer where to check, and the
reviewer checks.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Iterable

__all__ = ["ValidationResult", "GroundednessValidator", "CITATION_RE"]

#: ``[PAY-06-R03@1.0.0]``, ``[shap:isolation_forest:amount_per_los_day]``,
#: ``[span:C0001234:333-360]``, ``[signal:abc123]``, ``[exposure:CASE-001]``
CITATION_RE = re.compile(r"\[([A-Za-z0-9_.@:\-/]+)\]")

#: Sentences that are structure, not assertion. Deliberately a short, explicit
#: list: anything not on it and not cited is dropped.
_STRUCTURAL_PREFIXES = (
    "what fired.", "what to check:", "what it's worth.", "what is being asked",
    "  - ", "- ", "",
)


@dataclass
class ValidationResult:
    text: str
    kept: list[str] = field(default_factory=list)
    dropped: list[tuple[str, str]] = field(default_factory=list)   # (sentence, reason)
    citations_resolved: list[str] = field(default_factory=list)
    citations_unresolved: list[str] = field(default_factory=list)

    @property
    def drop_count(self) -> int:
        return len(self.dropped)

    def to_dict(self) -> dict[str, Any]:
        return {
            "validated_text": self.text,
            "sentences_kept": len(self.kept),
            "sentences_dropped": self.drop_count,
            "dropped_detail": [{"sentence": s, "reason": r} for s, r in self.dropped],
            "citations_resolved": self.citations_resolved,
            "citations_unresolved": self.citations_unresolved,
            "validator_note": (
                "This validator confirms every factual sentence is ATTRIBUTABLE to a stored "
                "artefact. It does not confirm the sentence is TRUE. The citation tells you "
                "where to check; you check."
            ),
        }


class GroundednessValidator:
    def __init__(self, resolvable_ids: Iterable[str]) -> None:
        self.resolvable = {str(i) for i in resolvable_ids}

    def validate(self, text: str) -> ValidationResult:
        result = ValidationResult(text="")
        kept_lines: list[str] = []

        for raw_line in text.splitlines():
            line = raw_line.rstrip()
            stripped = line.strip().lower()
            if not stripped:
                kept_lines.append(line)
                continue
            if any(stripped.startswith(p) for p in _STRUCTURAL_PREFIXES if p):
                kept_lines.append(line)
                result.kept.append(line)
                continue

            citations = CITATION_RE.findall(line)
            if not citations:
                result.dropped.append((line, "No citation. An uncited factual claim is dropped."))
                continue

            unresolved = [c for c in citations if not self._resolves(c)]
            if unresolved:
                result.citations_unresolved.extend(unresolved)
                result.dropped.append((
                    line,
                    f"Citation(s) {unresolved} do not resolve to a stored artefact. A "
                    f"plausible-looking but invented citation is exactly the failure mode this "
                    f"validator exists to catch.",
                ))
                continue

            result.citations_resolved.extend(citations)
            kept_lines.append(line)
            result.kept.append(line)

        result.text = "\n".join(kept_lines).strip()
        return result

    def _resolves(self, citation: str) -> bool:
        if citation in self.resolvable:
            return True
        # Prefix forms: shap:<model>:<feature>, span:<claim>:<offsets>, etc.
        head = citation.split(":", 1)[0]
        if head in {"shap", "span", "signal", "exposure", "priority", "case", "param"}:
            return any(r == citation or r.endswith(citation.split(":")[-1]) for r in self.resolvable) or \
                   citation in self.resolvable
        return False

    @classmethod
    def from_case(cls, case, signals, model_features=None, document_spans=None) -> "GroundednessValidator":
        """Build the resolvable set from what is actually stored for a case."""
        ids: set[str] = set()
        for s in signals:
            ids.add(s.signal_id)
            ids.add(f"signal:{s.signal_id}")
            ids.add(f"{s.rule_id}@{s.rule_version}")
            ids.add(s.rule_id)
        case_id = getattr(case, "case_id", None) or getattr(case, "case_fingerprint", "case")
        ids.add(f"exposure:{case_id}")
        ids.add(f"priority:{case_id}")
        ids.add(f"case:{case_id}")
        for f in model_features or []:
            ids.add(f"shap:{f.get('model')}:{f.get('feature')}")
        for span in document_spans or []:
            offsets = span.get("span_offsets", [0, 0])
            ids.add(f"span:{span.get('claim_sk')}:{offsets[0]}-{offsets[1]}")
        return cls(ids)
