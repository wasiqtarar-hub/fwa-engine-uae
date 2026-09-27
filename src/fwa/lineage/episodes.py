"""Claim lineage and the episode builder (manuscript §4.1, §4.12, Epic 0).

Two distinct jobs live here, and the manuscript treats both as foundations that
nothing else is safe to build on top of:

**Lineage** — the original / correction / resubmission chain (``claim_version``
in Table 3.5). the claim extract has none: every ``claim_id`` is unique and appears
exactly once. That is recorded honestly rather than simulated, and the
consequence is stated where it matters: PAY-08 is ``NOT_EXECUTABLE``, and the
correction/cancellation/resubmission test category (§6.2 category 6) is
exercised against constructed fixtures. :class:`LineageIndex` still exists and
still works — it is what a real Shafafiya feed would populate — and the
integration test ``test_corrected_duplicate_supersedes`` drives it.

**Episodes** — the grouping of claims that describe one clinical event. This one
*can* be built from the claim extract, and it matters: §6.3 requires that "a package
component billed on a related but separate claim must correlate into the same
episode as its parent package", and CLN-05-R03 (discharge-readmit split) is
precisely the question of whether two stays are really one episode.

The episode rule implemented here is stated in full so a reviewer can reproduce
it: two claims for the **same member** join the same episode when the second
admission begins within ``cfg.readmit_window_days`` of the first discharge. The
episode records whether the join was **same-provider** (a readmission, which is
what CLN-05-R02/R03 are about) or **cross-provider** (which may be a transfer, a
facility/professional split, or two unrelated admissions — the source has no
transfer indicator, so the episode says so and carries reduced confidence rather
than asserting a transfer it cannot see).

On the claim extract this yields a small number of multi-claim episodes, which is
itself a finding worth stating: the dataset is overwhelmingly one-claim-per-
event, so episode-level controls have thin support here and the coverage matrix
classifies them ``PARTIAL`` accordingly.
"""

from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass, field
from typing import Any, Iterable

import pandas as pd

from ..engine.evallib import is_missing

__all__ = ["ClaimVersionRecord", "LineageIndex", "Episode", "EpisodeBuilder"]


# ---------------------------------------------------------------------------
# Lineage
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ClaimVersionRecord:
    claim_sk: str
    version_no: int
    relationship: str          # ORIGINAL | CORRECTION | RESUBMISSION | CANCELLATION
    prior_claim_sk: str | None
    changed_fields: dict[str, tuple[Any, Any]]
    recorded_at: _dt.datetime
    resubmission_type: str | None = None


class LineageIndex:
    """Claim-version history with supersession semantics.

    The property §6.3 asks for — "a corrected duplicate claim must close or
    supersede the prior open signal rather than create two concurrently open
    cases" — is implemented as :meth:`active_claim_for`, which resolves any
    claim in a chain to the one version that is currently live. The evaluator
    calls it before raising a duplicate signal, so a correction supersedes
    rather than duplicates.
    """

    def __init__(self) -> None:
        self._versions: dict[str, list[ClaimVersionRecord]] = {}
        self._successor: dict[str, str] = {}
        self._cancelled: set[str] = set()

    def add(self, record: ClaimVersionRecord) -> None:
        self._versions.setdefault(record.claim_sk, []).append(record)
        self._versions[record.claim_sk].sort(key=lambda r: r.version_no)
        if record.prior_claim_sk:
            self._successor[record.prior_claim_sk] = record.claim_sk
        if record.relationship == "CANCELLATION":
            self._cancelled.add(record.prior_claim_sk or record.claim_sk)

    @classmethod
    def from_frame(cls, frame: pd.DataFrame) -> "LineageIndex":
        idx = cls()
        if frame is None or frame.empty:
            return idx
        for _, row in frame.iterrows():
            changed = row.get("changed_fields")
            if isinstance(changed, str):
                import json

                try:
                    changed = json.loads(changed)
                except Exception:
                    changed = {}
            idx.add(
                ClaimVersionRecord(
                    claim_sk=str(row["claim_sk"]),
                    version_no=int(row.get("version_no", 1)),
                    relationship=str(row.get("relationship", "ORIGINAL")),
                    prior_claim_sk=None if is_missing(row.get("prior_claim_sk")) else str(row["prior_claim_sk"]),
                    changed_fields=changed or {},
                    recorded_at=pd.to_datetime(row.get("recorded_at")).to_pydatetime()
                    if not is_missing(row.get("recorded_at"))
                    else _dt.datetime.now(_dt.timezone.utc),
                    resubmission_type=row.get("resubmission_type"),
                )
            )
        return idx

    def active_claim_for(self, claim_sk: str) -> str:
        """Follow the supersession chain to the currently live claim."""
        seen = {claim_sk}
        current = claim_sk
        while current in self._successor:
            nxt = self._successor[current]
            if nxt in seen:  # cycle guard
                break
            seen.add(nxt)
            current = nxt
        return current

    def is_superseded(self, claim_sk: str) -> bool:
        return claim_sk in self._successor

    def is_cancelled(self, claim_sk: str) -> bool:
        return claim_sk in self._cancelled

    def chain(self, claim_sk: str) -> list[str]:
        out = [claim_sk]
        while out[-1] in self._successor:
            nxt = self._successor[out[-1]]
            if nxt in out:
                break
            out.append(nxt)
        return out

    def field_diff(self, claim_sk: str) -> dict[str, tuple[Any, Any]]:
        """Field-level diff between a resubmission and its original (§6.3).

        The Case Evidence page renders this directly: "a denied claim that is
        subsequently mutated and resubmitted must display the field-level
        difference between the original and resubmitted versions".
        """
        versions = self._versions.get(claim_sk, [])
        merged: dict[str, tuple[Any, Any]] = {}
        for v in versions:
            merged.update(v.changed_fields or {})
        return merged

    def __len__(self) -> int:
        return len(self._versions)


# ---------------------------------------------------------------------------
# Episodes
# ---------------------------------------------------------------------------


@dataclass
class Episode:
    episode_id: str
    member_sk: str
    provider_sk: str
    claim_sks: list[str] = field(default_factory=list)
    provider_sks: list[str] = field(default_factory=list)
    start_date: _dt.date | None = None
    end_date: _dt.date | None = None
    total_gross_aed: float = 0.0
    claim_count: int = 0
    linkage_reason: str = ""
    crosses_providers: bool = False
    linkage_confidence: float = 1.0

    def to_row(self) -> dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "member_sk": self.member_sk,
            "provider_sk": self.provider_sk,
            "provider_sks": ",".join(sorted(set(self.provider_sks))),
            "claim_sks": ",".join(self.claim_sks),
            "claim_count": self.claim_count,
            "start_date": self.start_date.isoformat() if self.start_date else None,
            "end_date": self.end_date.isoformat() if self.end_date else None,
            "total_gross_aed": round(self.total_gross_aed, 2),
            "crosses_providers": self.crosses_providers,
            "linkage_confidence": round(self.linkage_confidence, 3),
            "linkage_reason": self.linkage_reason,
        }


class EpisodeBuilder:
    """Groups claims into episodes.

    The rule is stated in the module docstring and reproduced in the linkage
    reason on every episode, so a reviewer looking at a two-claim episode can
    see *why* the system decided those two claims describe one event.
    """

    def __init__(self, readmit_window_days: int) -> None:
        self.readmit_window_days = int(readmit_window_days)

    def build(self, claim_header: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, str]]:
        """Return (episode frame, claim_sk → episode_id map)."""
        if claim_header is None or claim_header.empty:
            return pd.DataFrame(columns=list(Episode("", "", "").to_row())), {}

        df = claim_header[
            ["claim_sk", "member_sk", "provider_sk", "service_date", "discharge_date", "gross_amount_aed"]
        ].copy()
        df["service_date"] = pd.to_datetime(df["service_date"])
        df["discharge_date"] = pd.to_datetime(df["discharge_date"])
        df = df.sort_values(["member_sk", "service_date", "claim_sk"], kind="stable")

        episodes: list[Episode] = []
        mapping: dict[str, str] = {}
        current: Episode | None = None
        prev_discharge: pd.Timestamp | None = None
        prev_member: str | None = None
        prev_provider: str | None = None

        for row in df.itertuples(index=False):
            member = str(row.member_sk)
            provider = str(row.provider_sk)
            admit = row.service_date
            discharge = row.discharge_date
            gap = (
                (admit - prev_discharge).days
                if prev_discharge is not None and not pd.isna(admit) and not pd.isna(prev_discharge)
                else None
            )
            joins_previous = (
                current is not None
                and prev_member == member
                and gap is not None
                and 0 <= gap <= self.readmit_window_days
            )
            if not joins_previous:
                current = Episode(
                    episode_id=f"EP-{len(episodes) + 1:06d}",
                    member_sk=member,
                    provider_sk=provider,
                    start_date=None if pd.isna(admit) else admit.date(),
                    linkage_reason=(
                        f"New episode: this member had no admission ending within "
                        f"{self.readmit_window_days} days (cfg.readmit_window_days) before this one."
                    ),
                )
                episodes.append(current)
            else:
                same_provider = provider == prev_provider
                current.crosses_providers = current.crosses_providers or not same_provider
                if same_provider:
                    current.linkage_reason = (
                        f"Joined: readmission {gap} day(s) after the previous discharge for the same "
                        f"member at the SAME provider, inside the {self.readmit_window_days}-day "
                        f"window (cfg.readmit_window_days). This is the CLN-05-R02/R03 pattern."
                    )
                else:
                    current.linkage_confidence = min(current.linkage_confidence, 0.6)
                    current.linkage_reason = (
                        f"Joined with REDUCED CONFIDENCE: admission {gap} day(s) after the previous "
                        f"discharge for the same member but at a DIFFERENT provider. This may be a "
                        f"transfer, a facility/professional split, or two unrelated admissions — "
                        f"the claim extract has no transfer indicator, so the episode records the ambiguity "
                        f"rather than asserting a transfer it cannot see."
                    )
            current.claim_sks.append(str(row.claim_sk))
            current.provider_sks.append(provider)
            current.claim_count = len(current.claim_sks)
            current.total_gross_aed += float(row.gross_amount_aed or 0.0)
            if not pd.isna(discharge):
                current.end_date = discharge.date()
            mapping[str(row.claim_sk)] = current.episode_id
            prev_discharge = discharge
            prev_member = member
            prev_provider = provider

        return pd.DataFrame([e.to_row() for e in episodes]), mapping
