"""Entity resolution that never merges on its own (manuscript §4.10).

    "Entity resolution (deciding that two administrative records represent the
     same real-world actor) EXPOSES ITS MATCHED FIELDS AND A CONFIDENCE SCORE,
     and the design EXPLICITLY PROHIBITS AUTOMATIC MERGING of low-confidence
     matches."                                              — manuscript §4.10

This artefact goes one step further than the prohibition and merges *nothing*
automatically at any confidence: ``cfg.entity_resolution_auto_merge_min_confidence``
is set to 1.01, a value the score can never reach. Every candidate pair is
surfaced on the Network page for a human to confirm or reject, and a
confirmation writes an ``ENTITY_MERGE_CONFIRMED`` event to the audit log.

The honest limitation: the claim extract contains none of the administrative
identifiers (bank account, phone, address, device, owner) that real entity
resolution depends on, which is why NET-02-R01 is classified
``NOT_EXECUTABLE_ON_THIS_DATASET``. What *can* be done is behavioural
resolution — two provider records that serve overlapping member populations
through the same agents and TPAs with near-identical billing profiles — and that
is what the matcher below scores. It is a weaker signal and the confidence
reflects it; a behavioural match is a question, not an identification.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Iterable

import numpy as np
import pandas as pd

__all__ = ["MatchCandidate", "EntityResolver"]


@dataclass(frozen=True)
class MatchCandidate:
    entity_a: str
    entity_b: str
    entity_type: str
    confidence: float
    matched_fields: dict[str, Any]
    auto_merge_allowed: bool
    status: str = "AWAITING_HUMAN_CONFIRMATION"
    note: str = ""

    def to_row(self) -> dict[str, Any]:
        d = asdict(self)
        d["matched_fields"] = "; ".join(f"{k}={v}" for k, v in self.matched_fields.items())
        return d


class EntityResolver:
    """Behavioural candidate matching. Surfaces; never merges."""

    def __init__(self, config) -> None:
        self.config = config
        self.auto_merge_threshold = float(config.get("entity_resolution_auto_merge_min_confidence"))
        self.review_threshold = float(config.get("entity_resolution_review_min_confidence"))
        self.confirmed: list[tuple[str, str, str]] = []

    # ------------------------------------------------------------------ score

    def candidates(self, claims: pd.DataFrame, entity_column: str = "provider_sk") -> list[MatchCandidate]:
        profiles = self._profiles(claims, entity_column)
        entities = list(profiles.index)
        out: list[MatchCandidate] = []
        for i, a in enumerate(entities):
            for b in entities[i + 1:]:
                score, fields = self._score(profiles.loc[a], profiles.loc[b])
                if score < self.review_threshold:
                    continue
                out.append(
                    MatchCandidate(
                        entity_a=str(a), entity_b=str(b),
                        entity_type=entity_column.replace("_sk", ""),
                        confidence=round(score, 3),
                        matched_fields=fields,
                        auto_merge_allowed=score >= self.auto_merge_threshold,
                        note=(
                            "BEHAVIOURAL match only. the claim extract carries no bank, phone, address, "
                            "device or owner identifiers, so this is a similarity of billing and "
                            "network behaviour, not an identification. It is surfaced for human "
                            "confirmation and is never merged automatically."
                        ),
                    )
                )
        return sorted(out, key=lambda c: -c.confidence)

    def _profiles(self, claims: pd.DataFrame, entity_column: str) -> pd.DataFrame:
        grouped = claims.groupby(entity_column)
        prof = grouped.agg(
            claim_count=("claim_sk", "count"),
            mean_gross=("gross_amount_aed", "mean"),
            mean_los=("length_of_stay_days", "mean"),
            mean_pharmacy=("pharmacy_bill_ratio", "mean"),
        )
        prof["agents"] = grouped["agent_id"].apply(lambda s: frozenset(s.unique()))
        prof["tpas"] = grouped["tpa"].apply(lambda s: frozenset(s.unique()))
        prof["members"] = grouped["member_sk"].apply(lambda s: frozenset(s.unique()))
        prof["diagnoses"] = grouped["diagnosis_primary"].apply(lambda s: frozenset(s.unique()))
        return prof

    def _score(self, a: pd.Series, b: pd.Series) -> tuple[float, dict[str, Any]]:
        fields: dict[str, Any] = {}
        parts: list[float] = []

        member_j = _jaccard(a["members"], b["members"])
        if member_j > 0:
            fields["shared_members"] = len(a["members"] & b["members"])
        parts.append(member_j * 0.40)

        agent_j = _jaccard(a["agents"], b["agents"])
        fields["agent_overlap"] = round(agent_j, 3)
        parts.append(agent_j * 0.20)

        tpa_j = _jaccard(a["tpas"], b["tpas"])
        fields["tpa_overlap"] = round(tpa_j, 3)
        parts.append(tpa_j * 0.10)

        dx_j = _jaccard(a["diagnoses"], b["diagnoses"])
        fields["diagnosis_overlap"] = round(dx_j, 3)
        parts.append(dx_j * 0.10)

        billing = _closeness(a["mean_gross"], b["mean_gross"])
        los = _closeness(a["mean_los"], b["mean_los"])
        pharm = _closeness(a["mean_pharmacy"], b["mean_pharmacy"])
        fields["billing_profile_similarity"] = round((billing + los + pharm) / 3, 3)
        parts.append((billing + los + pharm) / 3 * 0.20)

        return float(sum(parts)), fields

    # ---------------------------------------------------------------- confirm

    def confirm(self, candidate: MatchCandidate, *, actor: str, reason: str, audit=None) -> MatchCandidate:
        """Record a human confirmation. The only way two entities are ever linked."""
        if not reason.strip():
            raise ValueError("Confirming an entity match requires a stated reason.")
        self.confirmed.append((candidate.entity_a, candidate.entity_b, actor))
        if audit is not None:
            from ..audit.log import AuditEventType

            audit.record(
                AuditEventType.ENTITY_MERGE_CONFIRMED,
                actor=actor, actor_role="ADMIN",
                subject=f"{candidate.entity_a}~{candidate.entity_b}",
                reason=reason,
                after={"confidence": candidate.confidence, "matched_fields": candidate.matched_fields},
            )
        return MatchCandidate(
            **{**asdict(candidate), "status": f"CONFIRMED_BY_{actor}"}
        )

    def to_frame(self, candidates: Iterable[MatchCandidate]) -> pd.DataFrame:
        rows = [c.to_row() for c in candidates]
        cols = ["entity_a", "entity_b", "entity_type", "confidence", "matched_fields",
                "auto_merge_allowed", "status", "note"]
        return pd.DataFrame(rows, columns=cols) if rows else pd.DataFrame(columns=cols)


def _jaccard(a: frozenset, b: frozenset) -> float:
    if not a or not b:
        return 0.0
    union = len(a | b)
    return len(a & b) / union if union else 0.0


def _closeness(a: float, b: float) -> float:
    if a is None or b is None or (isinstance(a, float) and a != a) or (isinstance(b, float) and b != b):
        return 0.0
    denom = max(abs(float(a)), abs(float(b)), 1e-9)
    return float(max(0.0, 1.0 - abs(float(a) - float(b)) / denom))
