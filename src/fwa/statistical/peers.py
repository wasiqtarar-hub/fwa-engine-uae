"""Peer-group service — six levels, back-off, and ``peer_level_used`` (§4.6).

    "A granular peer group that is too sparse to support a stable estimate backs
     off to its parent level, and the ``peer_level_used`` is ALWAYS RECORDED so
     that a later analyst can see exactly which comparison group produced a
     given result, a direct requirement of the reproducibility non-functional
     requirement."                                          — manuscript §4.6

Two things this service refuses to do, both deliberately:

* It will not use a group smaller than ``cfg.min_peer_group_n``. It backs off,
  and if every level in ``config/peer_groups.yaml``'s back-off order is too
  sparse it returns :data:`INSUFFICIENT_PEER_EVIDENCE` — a distinct outcome
  from "not anomalous". A control that receives it records
  ``EvaluationOutcome.INSUFFICIENT_PEER_EVIDENCE`` and raises nothing.
* It will not use the geography level. That level is ``NOT_POPULATED`` in this
  dataset and is skipped rather than approximated, so no result can silently
  depend on a location the source does not contain.

Every resolved group is stamped with a ``baseline_version`` — a hash of the
group definition, the row set and the configuration fingerprint. §6.3 requires
that "a provider peer-comparison result must be exactly reproducible from its
stored baseline version when re-run at a later date"; the version is what makes
that check possible, and the integration test does exactly that re-run.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

from ..config import AppConfig

__all__ = ["PeerGroup", "PeerService", "INSUFFICIENT_PEER_EVIDENCE"]

INSUFFICIENT_PEER_EVIDENCE = "INSUFFICIENT_PEER_EVIDENCE"


@dataclass(frozen=True)
class PeerGroup:
    """One resolved comparison group."""

    key: tuple[Any, ...]
    level_name: str                 # e.g. "L1_L2_L4_L6" or "GLOBAL"
    levels_used: tuple[str, ...]
    n_rows: int
    n_entities: int
    row_index: pd.Index
    baseline_version: str
    backoff_steps: int
    skipped_levels: tuple[str, ...] = ()

    @property
    def sufficient(self) -> bool:
        return self.level_name != INSUFFICIENT_PEER_EVIDENCE

    def describe(self) -> str:
        if not self.sufficient:
            return "No peer group large enough to support a comparison."
        if not self.levels_used:
            return f"All claims ({self.n_rows:,} claims, {self.n_entities:,} providers) — global fallback."
        return (
            f"{' + '.join(self.levels_used)} ({self.n_rows:,} claims, "
            f"{self.n_entities:,} providers)"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "peer_key": list(self.key),
            "peer_level_used": self.level_name,
            "peer_levels": list(self.levels_used),
            "peer_n": self.n_rows,
            "peer_entities": self.n_entities,
            "peer_baseline_version": self.baseline_version,
            "peer_backoff_steps": self.backoff_steps,
            "peer_levels_skipped": list(self.skipped_levels),
            "peer_group_description": self.describe(),
        }


class PeerService:
    """Resolves peer groups over a claim-level frame, with back-off."""

    #: peer-hierarchy canonical level -> column in the working frame
    COLUMN_FOR_LEVEL = {
        "activity_code_family": "diagnosis_primary",
        "specialty": "diagnosis_chapter",
        "encounter_facility_type": "policy_type",
        "payment_method_contract_family": "tpa",
        "provider_volume_band": "provider_volume_band",
    }

    def __init__(self, config: AppConfig, frame: pd.DataFrame, entity_column: str = "provider_sk") -> None:
        self.config = config
        self.entity_column = entity_column
        self.min_n = int(config.get("min_peer_group_n"))
        self._cfg_fingerprint = config.fingerprint()

        work = frame.copy()
        if "provider_volume_band" not in work.columns:
            counts = work[entity_column].value_counts()
            work["provider_volume_band"] = work[entity_column].map(
                lambda e: config.peer_groups.volume_band(int(counts.get(e, 0)))
            )
        if "diagnosis_chapter" not in work.columns and "diagnosis_primary" in work.columns:
            work["diagnosis_chapter"] = work["diagnosis_primary"].map(config.peer_groups.chapter_for)
        self.frame = work

        self.skipped_levels = tuple(
            l.canonical_name for l in config.peer_groups.levels if not l.populated
        )
        self._backoff = [
            (step["name"], tuple(step["levels"]))
            for step in config.peer_groups.backoff_order
        ]
        self._cache: dict[tuple, PeerGroup] = {}

    # ------------------------------------------------------------------ core

    def resolve(self, row: pd.Series | dict[str, Any]) -> PeerGroup:
        """Walk the back-off order until a group is big enough."""
        cache_key = tuple(
            (lvl, row.get(self.COLUMN_FOR_LEVEL[lvl]))
            for _, levels in self._backoff for lvl in levels
        )
        if cache_key in self._cache:
            return self._cache[cache_key]

        for steps, (name, levels) in enumerate(self._backoff):
            mask = pd.Series(True, index=self.frame.index)
            usable = True
            for lvl in levels:
                col = self.COLUMN_FOR_LEVEL.get(lvl)
                if col is None or col not in self.frame.columns:
                    usable = False
                    break
                value = row.get(col)
                if value is None or (isinstance(value, float) and value != value):
                    usable = False
                    break
                mask &= self.frame[col] == value
            if not usable:
                continue
            idx = self.frame.index[mask]
            if len(idx) >= self.min_n:
                group = self._make_group(name, levels, row, idx, steps)
                self._cache[cache_key] = group
                return group

        group = PeerGroup(
            key=(), level_name=INSUFFICIENT_PEER_EVIDENCE, levels_used=(), n_rows=0,
            n_entities=0, row_index=pd.Index([]), baseline_version="",
            backoff_steps=len(self._backoff), skipped_levels=self.skipped_levels,
        )
        self._cache[cache_key] = group
        return group

    def _make_group(
        self, name: str, levels: tuple[str, ...], row: Any, idx: pd.Index, steps: int
    ) -> PeerGroup:
        key = tuple(row.get(self.COLUMN_FOR_LEVEL[l]) for l in levels)
        entities = int(self.frame.loc[idx, self.entity_column].nunique())
        version = self._baseline_version(name, key, idx)
        return PeerGroup(
            key=key, level_name=name, levels_used=levels, n_rows=int(len(idx)),
            n_entities=entities, row_index=idx, baseline_version=version,
            backoff_steps=steps, skipped_levels=self.skipped_levels,
        )

    def _baseline_version(self, name: str, key: tuple, idx: pd.Index) -> str:
        """Hash of (definition, membership, config) — the §6.3 reproducibility key."""
        payload = json.dumps(
            {
                "level": name,
                "key": [str(k) for k in key],
                "n": len(idx),
                "member_hash": hashlib.sha256(
                    ",".join(sorted(str(i) for i in idx)).encode("utf-8")
                ).hexdigest()[:16],
                "config": self._cfg_fingerprint,
            },
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]

    # -------------------------------------------------------------- helpers

    def values(self, group: PeerGroup, column: str) -> pd.Series:
        if not group.sufficient:
            return pd.Series(dtype=float)
        return self.frame.loc[group.row_index, column]

    def resolve_frame(self, columns: Sequence[str] | None = None) -> pd.DataFrame:
        """Resolve a peer group for every row. Returns the peer metadata frame.

        Vectorised by grouping on the back-off keys rather than iterating rows,
        which matters: a per-row resolve over tens of thousands of claims is fine, but the
        composite and several controls need this repeatedly.
        """
        records: list[dict[str, Any]] = []
        for _, sub in self.frame.groupby(
            [self.COLUMN_FOR_LEVEL[l] for l in self._backoff[0][1] if self.COLUMN_FOR_LEVEL[l] in self.frame.columns],
            dropna=False, sort=False,
        ):
            group = self.resolve(sub.iloc[0])
            meta = group.to_dict()
            for idx in sub.index:
                records.append({"__row": idx, **meta})
        out = pd.DataFrame(records).set_index("__row")
        return out.reindex(self.frame.index)

    def entity_aggregates(
        self, group: PeerGroup, value_column: str, agg: str = "median"
    ) -> float | None:
        if not group.sufficient:
            return None
        series = self.frame.loc[group.row_index, value_column].astype(float).dropna()
        if series.empty:
            return None
        return float(getattr(series, agg)())

    def level_availability(self) -> pd.DataFrame:
        """What each of the six levels maps to here, and whether it is usable.

        Rendered on the Provider Analytics page so a reviewer can see that
        level 5 (geography) is absent rather than wondering why comparisons
        never mention location.
        """
        rows = []
        for lvl in self.config.peer_groups.levels:
            col = self.COLUMN_FOR_LEVEL.get(lvl.canonical_name)
            rows.append(
                {
                    "level": lvl.level,
                    "canonical_name": lvl.canonical_name,
                    "dataset_mapping": lvl.dataset_mapping or "—",
                    "availability": lvl.availability,
                    "distinct_values": (
                        int(self.frame[col].nunique()) if col and col in self.frame.columns else 0
                    ),
                    "proxy_note": lvl.proxy_note or "",
                    "required_canonical_fields": "; ".join(lvl.required_canonical_fields),
                }
            )
        return pd.DataFrame(rows)
