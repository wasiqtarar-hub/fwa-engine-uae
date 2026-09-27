"""The LLM provider interface, with an offline default (build brief §9).

    "Implement an ``LLMProvider`` interface with an **``OfflineDeterministicProvider``
     as the default**, so the whole tool runs with no API key, no network and
     reproducible output; add ``AnthropicProvider`` / ``OpenAIProvider`` as
     optional, configured via environment variable only."

The offline provider is not a stub. It produces the same *shape* of output a
hosted model would — cited sentences, YAML drafts, bounded answers — by
composing templates over the evidence it is given. That matters for three
reasons:

1. The artefact is **demonstrable in a viva** with no network and no key.
2. Every figure in ``reports/`` is **reproducible**, because the narrative text
   is a deterministic function of the evidence.
3. It makes the governance machinery testable. The groundedness validator, the
   copilot's hard refusal and the disposition-legality check on an
   LLM-authored rule all have to work regardless of which provider generated
   the text — and the only way to prove that is to exercise them against a
   provider whose output you control.

A hosted provider is selected by environment variable only
(``FWA_LLM_PROVIDER=anthropic`` plus ``ANTHROPIC_API_KEY``), never by a config
file, so that cloning this repository and running it cannot silently start
sending claim evidence to a third party.
"""

from __future__ import annotations

import abc
import hashlib
import json
import os
import textwrap
from dataclasses import dataclass, field
from typing import Any

__all__ = ["LLMResponse", "LLMProvider", "OfflineDeterministicProvider",
           "AnthropicProvider", "OpenAIProvider", "get_provider"]


@dataclass
class LLMResponse:
    text: str
    provider: str
    model: str
    prompt_hash: str
    evidence_ids: list[str] = field(default_factory=list)
    deterministic: bool = True

    def to_audit_payload(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "model": self.model,
            "prompt_hash": self.prompt_hash,
            "evidence_ids": self.evidence_ids,
            "deterministic": self.deterministic,
            "characters": len(self.text),
        }


def prompt_hash(prompt: str, evidence: Any) -> str:
    payload = json.dumps({"prompt": prompt, "evidence": evidence}, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


class LLMProvider(abc.ABC):
    name = "abstract"
    model = "abstract"
    deterministic = False

    @abc.abstractmethod
    def generate(self, prompt: str, evidence: dict[str, Any], *, task: str) -> LLMResponse:
        """Produce text for ``task`` given ``evidence``.

        ``task`` is one of ``narrative``, ``rule_draft``, ``copilot``,
        ``triage``. A provider may specialise on it; the offline provider does.
        """


class OfflineDeterministicProvider(LLMProvider):
    """The default. No API key, no network, byte-identical output for identical input."""

    name = "offline_deterministic"
    model = "template-composer-1.0.0"
    deterministic = True

    def generate(self, prompt: str, evidence: dict[str, Any], *, task: str) -> LLMResponse:
        handler = {
            "narrative": self._narrative,
            "rule_draft": self._rule_draft,
            "copilot": self._copilot,
            "triage": self._triage,
        }.get(task, self._fallback)
        text = handler(prompt, evidence)
        return LLMResponse(
            text=text,
            provider=self.name,
            model=self.model,
            prompt_hash=prompt_hash(prompt, evidence),
            evidence_ids=[str(k) for k in evidence.get("evidence_ids", [])],
            deterministic=True,
        )

    # -- task handlers -------------------------------------------------------

    def _narrative(self, prompt: str, evidence: dict[str, Any]) -> str:
        """Compose a cited summary. Every factual sentence carries a citation."""
        lines: list[str] = []
        subject = evidence.get("subject_label", "this case")
        signals = evidence.get("signals", [])

        lines.append(f"What fired. {len(signals)} signal(s) correlated into {subject}.")
        for s in signals:
            cite = f"[{s['rule_id']}@{s['rule_version']}]"
            lines.append(f"{s['headline']} {cite}")

        for feature in evidence.get("model_features", []):
            lines.append(
                f"{feature['plain_language']}. "
                f"[shap:{feature['model']}:{feature['feature']}]"
            )
        for span in evidence.get("document_spans", []):
            lines.append(
                f"The documentation states: \"{span['source_span_text']}\". "
                f"[span:{span['claim_sk']}:{span['span_offsets'][0]}-{span['span_offsets'][1]}]"
            )

        if evidence.get("exposure_line"):
            lines.append(f"{evidence['exposure_line']} [exposure:{evidence.get('case_id', 'case')}]")
        if evidence.get("priority_line"):
            lines.append(f"{evidence['priority_line']} [priority:{evidence.get('case_id', 'case')}]")

        lines.append("")
        lines.append("What to check:")
        for step in evidence.get("suggested_checks", []):
            lines.append(f"  - {step}")
        return "\n".join(lines)

    def _rule_draft(self, prompt: str, evidence: dict[str, Any]) -> str:
        return json.dumps(evidence.get("draft", {}), indent=2, default=str)

    def _copilot(self, prompt: str, evidence: dict[str, Any]) -> str:
        bullets = evidence.get("answer_points", [])
        if not bullets:
            return (
                "I can only answer from this case's evidence bundle, the rule text and the "
                "parameter registry, and none of them contains an answer to that question."
            )
        return "\n".join(f"- {b}" for b in bullets)

    def _triage(self, prompt: str, evidence: dict[str, Any]) -> str:
        cluster = evidence.get("cluster", {})
        features = ", ".join(
            f"{f['label']} {f['mean_residual']:+.2f}σ" for f in cluster.get("top_features", [])[:3]
        )
        return textwrap.dedent(f"""
            Candidate typology: {cluster.get('candidate_typology_name', 'unnamed')}
            Cluster size: {cluster.get('size', 0)} entities
            Distinguishing residuals: {features}

            Plausible mechanism (HYPOTHESIS, not a finding): entities in this cluster deviate
            together on the residuals above. One mechanism consistent with that pattern would
            produce exactly this combination; several innocuous mechanisms would too, and
            nothing in the data distinguishes between them.

            Recommended next step: quarterly typology review. If an analyst judges the
            pattern coherent and material, draft an atomic control — MONITOR_ONLY — and place
            it in shadow for a prospective period before any activation is considered.
        """).strip()

    def _fallback(self, prompt: str, evidence: dict[str, Any]) -> str:
        return "No offline handler is registered for this task."


class _HostedProvider(LLMProvider):
    """Shared plumbing for hosted providers. Configured by environment only."""

    env_key = ""
    deterministic = False

    def __init__(self, model: str | None = None) -> None:
        self.api_key = os.environ.get(self.env_key)
        if not self.api_key:
            raise RuntimeError(
                f"{self.name} requires {self.env_key} in the environment. It is deliberately "
                f"NOT configurable from config/parameters.yaml: cloning this repository and "
                f"running it must never start sending claim evidence to a third party."
            )
        self.model = model or self.default_model

    default_model = ""

    def _system_prompt(self, task: str) -> str:
        from .. import SAFETY_BOUNDARY_STATEMENT

        return (
            "You are assisting a payment-integrity reviewer.\n"
            f"{SAFETY_BOUNDARY_STATEMENT}\n"
            "Rules you must follow absolutely:\n"
            "1. Every factual sentence must carry an inline citation in square brackets to a "
            "signal_id, rule_id@version, SHAP feature contribution or document source span "
            "that appears in the evidence you were given.\n"
            "2. Do not state any fact that is not in the evidence.\n"
            "3. Do not assert or imply that anyone committed fraud, and do not recommend a "
            "disposition. You cannot set or change a disposition, priority or exposure.\n"
            "4. If asked whether something is fraud, decline and restate the boundary above.\n"
        )


class AnthropicProvider(_HostedProvider):
    name = "anthropic"
    env_key = "ANTHROPIC_API_KEY"
    default_model = "claude-sonnet-4-5"

    def generate(self, prompt: str, evidence: dict[str, Any], *, task: str) -> LLMResponse:
        import anthropic  # imported lazily so the offline path needs no dependency

        client = anthropic.Anthropic(api_key=self.api_key)
        message = client.messages.create(
            model=self.model,
            max_tokens=1200,
            system=self._system_prompt(task),
            messages=[{
                "role": "user",
                "content": prompt + "\n\nEVIDENCE:\n" + json.dumps(evidence, indent=2, default=str),
            }],
        )
        text = "".join(block.text for block in message.content if getattr(block, "type", "") == "text")
        return LLMResponse(
            text=text, provider=self.name, model=self.model,
            prompt_hash=prompt_hash(prompt, evidence),
            evidence_ids=[str(k) for k in evidence.get("evidence_ids", [])],
            deterministic=False,
        )


class OpenAIProvider(_HostedProvider):
    name = "openai"
    env_key = "OPENAI_API_KEY"
    default_model = "gpt-4o"

    def generate(self, prompt: str, evidence: dict[str, Any], *, task: str) -> LLMResponse:
        from openai import OpenAI  # lazy import

        client = OpenAI(api_key=self.api_key)
        completion = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self._system_prompt(task)},
                {"role": "user",
                 "content": prompt + "\n\nEVIDENCE:\n" + json.dumps(evidence, indent=2, default=str)},
            ],
            max_tokens=1200,
        )
        return LLMResponse(
            text=completion.choices[0].message.content or "",
            provider=self.name, model=self.model,
            prompt_hash=prompt_hash(prompt, evidence),
            evidence_ids=[str(k) for k in evidence.get("evidence_ids", [])],
            deterministic=False,
        )


_PROVIDERS: dict[str, type[LLMProvider]] = {
    "offline_deterministic": OfflineDeterministicProvider,
    "anthropic": AnthropicProvider,
    "openai": OpenAIProvider,
}


def get_provider(name: str | None = None) -> LLMProvider:
    """Resolve the provider.

    Environment wins over configuration, and the offline provider is the
    fallback for everything — including a hosted provider that cannot be
    constructed, so that a missing key degrades to a working offline artefact
    rather than to a crash.
    """
    requested = os.environ.get("FWA_LLM_PROVIDER") or name or "offline_deterministic"
    cls = _PROVIDERS.get(requested, OfflineDeterministicProvider)
    try:
        return cls()
    except Exception:
        return OfflineDeterministicProvider()
