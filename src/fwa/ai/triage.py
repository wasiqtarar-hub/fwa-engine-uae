"""Novel-typology triage (build brief §9.4).

    "**Novel-typology triage.** Given an ANL-01-R04 cluster and its top residual
     features, propose a candidate typology name, a plausible mechanism and a
     **draft** atomic control for the quarterly review — always ``MONITOR_ONLY``,
     always flagged as an unvalidated hypothesis."

Two guards, both structural rather than prompt-level:

* The drafted control's disposition is forced to ``MONITOR_ONLY`` before the
  contract is constructed. Even if a provider returned something else, it would
  be overwritten here — and if that were somehow bypassed, the §3.3 validator
  would reject a Type-M control declaring anything that denies.
* The proposed mechanism is always rendered with its hypothesis marker
  attached, because a plausible-sounding mechanism is the most persuasive and
  least evidenced thing an LLM produces in this whole system.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..audit.log import AuditEventType
from ..enums import Disposition
from .provider import LLMProvider

__all__ = ["TypologyProposal", "TypologyTriage"]

HYPOTHESIS_MARKER = (
    "UNVALIDATED HYPOTHESIS — generated from a residual-space cluster. Nothing here has been "
    "confirmed against a case, an outcome or a reviewer. It is input to the quarterly typology "
    "review, not a finding."
)


@dataclass
class TypologyProposal:
    cluster_id: int
    name: str
    mechanism: str
    draft_control: dict[str, Any]
    size: int
    exposure_aed: float
    hypothesis_marker: str = HYPOTHESIS_MARKER

    def to_dict(self) -> dict[str, Any]:
        return {
            "cluster_id": self.cluster_id,
            "candidate_typology_name": self.name,
            "proposed_mechanism": self.mechanism,
            "cluster_size": self.size,
            "aggregate_exposure_aed": round(self.exposure_aed, 2),
            "draft_control": self.draft_control,
            "hypothesis_marker": self.hypothesis_marker,
            "panel_label": "AI-generated proposal — not evidence",
        }


class TypologyTriage:
    def __init__(self, provider: LLMProvider, audit=None) -> None:
        self.provider = provider
        self.audit = audit

    def propose(self, cluster, *, actor: str = "analyst") -> TypologyProposal:
        payload = {
            "cluster": {
                "cluster_id": cluster.cluster_id,
                "size": cluster.size,
                "candidate_typology_name": cluster.candidate_typology_name,
                "top_features": cluster.top_features,
                "aggregate_exposure_aed": cluster.aggregate_exposure_aed,
            }
        }
        response = self.provider.generate(
            "Propose a candidate typology name, a plausible mechanism and a DRAFT atomic "
            "control for this residual cluster. The disposition must be MONITOR_ONLY.",
            payload, task="triage",
        )
        draft = {
            "rule_id": f"ANL-01-R{90 + (cluster.cluster_id % 9):02d}",
            "scenario_id": "ANL-01",
            "name": cluster.candidate_typology_name[:70],
            "version": "0.1.0",
            "status": "shadow",
            "type": "M",
            "stage": "MODEL_MONTHLY",
            # Forced, not requested. §3.3 plus Appendix C's own choice.
            "disposition": Disposition.MONITOR_ONLY.value,
            "score": 30.0,
            "reason_code": "NOVEL_TYPOLOGY_CANDIDATE",
            "owner": "policy_owner",
            "exclusions": [
                "unvalidated_hypothesis_requires_quarterly_review",
                "no_production_action_until_converted_to_rule",
            ],
            "expression": " and ".join(
                f"{f['label']} residual {'above' if f['mean_residual'] > 0 else 'below'} "
                f"{abs(f['mean_residual']):.2f}σ"
                for f in cluster.top_features[:3]
            ),
            "evidence_fields": [f["feature"] for f in cluster.top_features],
        }
        if self.audit is not None:
            self.audit.record(
                AuditEventType.AI_TEXT_GENERATED,
                actor=actor, actor_role="ANALYST", subject=f"cluster-{cluster.cluster_id}",
                reason="Novel-typology triage proposal generated (MONITOR_ONLY, unvalidated).",
                after=response.to_audit_payload(),
            )
        return TypologyProposal(
            cluster_id=cluster.cluster_id,
            name=cluster.candidate_typology_name,
            mechanism=response.text,
            draft_control=draft,
            size=cluster.size,
            exposure_aed=cluster.aggregate_exposure_aed,
        )
