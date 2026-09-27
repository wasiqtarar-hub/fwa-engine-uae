"""The AI layer (build brief §9) — governed, grounded, independently switchable.

Every feature here is OFF-ABLE, and the tool must remain fully functional with
all of them disabled — a property asserted by
``tests/governance/test_ai_disabled.py``. Nothing in this package may set or
change a disposition, a priority or an exposure; the narrative and the copilot
are read-only consumers of evidence, the rule author produces a draft that the
DETERMINISTIC validator then accepts or rejects, and the triage proposal is
forced to MONITOR_ONLY.
"""

from dataclasses import dataclass, field
from typing import Any

from .provider import (
    LLMProvider, LLMResponse, OfflineDeterministicProvider,
    AnthropicProvider, OpenAIProvider, get_provider,
)
from .groundedness import GroundednessValidator, ValidationResult, CITATION_RE
from .narrative import CaseNarrator, CaseNarrative, ADVISORY_PANEL_LABEL
from .copilot import Copilot, CopilotAnswer, REFUSAL_TEXT
from .rule_author import NaturalLanguageRuleAuthor, RuleDraft, TEN_CATEGORIES
from .triage import TypologyTriage, TypologyProposal, HYPOTHESIS_MARKER


@dataclass
class AILayer:
    """Assembles the four features behind independent switches."""

    config: Any
    provider: LLMProvider
    narrator: CaseNarrator | None = None
    copilot: Copilot | None = None
    rule_author: NaturalLanguageRuleAuthor | None = None
    triage: TypologyTriage | None = None
    enabled: bool = True
    features: dict[str, bool] = field(default_factory=dict)

    @classmethod
    def build(cls, config, registry=None, audit=None) -> "AILayer":
        enabled = bool(config.get("ai_enabled"))
        features = dict(config.get("ai_features_enabled"))
        provider = get_provider(config.get("ai_provider"))
        layer = cls(config=config, provider=provider, enabled=enabled, features=features)
        if not enabled:
            return layer
        if features.get("grounded_narrative"):
            layer.narrator = CaseNarrator(provider, audit=audit)
        if features.get("reviewer_copilot") and registry is not None:
            layer.copilot = Copilot(provider, registry, config, audit=audit)
        if features.get("rule_authoring") and registry is not None:
            layer.rule_author = NaturalLanguageRuleAuthor(provider, registry, config, audit=audit)
        if features.get("typology_triage"):
            layer.triage = TypologyTriage(provider, audit=audit)
        return layer

    def status(self) -> dict[str, Any]:
        return {
            "ai_enabled": self.enabled,
            "provider": self.provider.name,
            "model": self.provider.model,
            "deterministic": self.provider.deterministic,
            "features": {
                "grounded_narrative": self.narrator is not None,
                "reviewer_copilot": self.copilot is not None,
                "rule_authoring": self.rule_author is not None,
                "typology_triage": self.triage is not None,
            },
            "note": (
                "Every feature is independently switchable and the tool is fully functional "
                "with all of them off. No AI output sets or changes a disposition, a priority "
                "or an exposure."
            ),
        }


__all__ = [
    "LLMProvider", "LLMResponse", "OfflineDeterministicProvider", "AnthropicProvider",
    "OpenAIProvider", "get_provider", "GroundednessValidator", "ValidationResult", "CITATION_RE",
    "CaseNarrator", "CaseNarrative", "ADVISORY_PANEL_LABEL", "Copilot", "CopilotAnswer",
    "REFUSAL_TEXT", "NaturalLanguageRuleAuthor", "RuleDraft", "TEN_CATEGORIES",
    "TypologyTriage", "TypologyProposal", "HYPOTHESIS_MARKER", "AILayer",
]
