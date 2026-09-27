"""Effective-dated reference / policy service — the as-of join (manuscript §3.6).

    "Every join from a claim to coverage, contract, tariff, code, licence,
     network, edit or model reference data must be an AS-OF JOIN evaluated at
     SERVICE TIME, not submission time... Corrections must never overwrite a
     prior version; every effective-dated record must carry valid_from,
     valid_to, recorded_at, source and version_id."        — manuscript §3.6

This is one of the quieter requirements and one of the easiest to get wrong,
because a system that ignores it still *runs* — it just applies today's policy
to a two-year-old claim and produces confidently wrong answers. The service
below makes that failure impossible by construction: there is no lookup method
that does not take an ``as_of`` date, and :meth:`AsOfReferenceService.resolve`
raises rather than defaulting to today.

The effective-date test (§6.2 category 5) is written against this service: two
policy versions are registered, a claim is evaluated at a date between them, and
the test asserts the *earlier* version resolved.
"""

from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

import pandas as pd

from ..engine.evallib import as_of_date

__all__ = ["ReferenceRecord", "AsOfReferenceService", "ReferenceNotEffective"]


class ReferenceNotEffective(LookupError):
    """No version of a reference record was in force on the requested date."""


@dataclass(frozen=True)
class ReferenceRecord:
    """One version of one effective-dated reference row."""

    domain: str                      # 'benefit_rule' | 'tariff' | 'licence' | 'code' | 'peer_baseline' | ...
    key: tuple[Any, ...]             # the natural key within the domain
    payload: Mapping[str, Any]
    valid_from: _dt.date
    valid_to: _dt.date
    version_id: str
    source: str
    recorded_at: _dt.datetime = field(default_factory=lambda: _dt.datetime.now(_dt.timezone.utc))
    superseded_by: str | None = None

    def covers(self, day: _dt.date) -> bool:
        return self.valid_from <= day <= self.valid_to


class AsOfReferenceService:
    """Versioned reference store with as-of resolution and no overwrite."""

    def __init__(self) -> None:
        self._records: dict[tuple[str, tuple[Any, ...]], list[ReferenceRecord]] = {}

    # ---------------------------------------------------------------- write

    def register(
        self,
        domain: str,
        key: Sequence[Any] | Any,
        payload: Mapping[str, Any],
        *,
        valid_from: _dt.date | str,
        valid_to: _dt.date | str = _dt.date(2099, 12, 31),
        version_id: str,
        source: str,
    ) -> ReferenceRecord:
        """Add a version. Never replaces an existing one (§3.6)."""
        k = tuple(key) if isinstance(key, (list, tuple)) else (key,)
        rec = ReferenceRecord(
            domain=domain,
            key=k,
            payload=dict(payload),
            valid_from=_coerce(valid_from),
            valid_to=_coerce(valid_to),
            version_id=version_id,
            source=source,
        )
        bucket = self._records.setdefault((domain, k), [])
        for existing in bucket:
            if existing.version_id == version_id and existing.valid_from == rec.valid_from:
                raise ValueError(
                    f"Version {version_id} of {domain}{k} is already registered with the same "
                    "effective date. A correction must be a NEW version — corrections never "
                    "overwrite a prior version."
                )
        bucket.append(rec)
        bucket.sort(key=lambda r: (r.valid_from, r.recorded_at))
        return rec

    def register_frame(
        self,
        domain: str,
        frame: pd.DataFrame,
        *,
        key_columns: Sequence[str],
        valid_from_column: str = "valid_from",
        valid_to_column: str = "valid_to",
        version_column: str = "version_id",
        source_column: str = "source",
    ) -> int:
        n = 0
        for _, row in frame.iterrows():
            self.register(
                domain,
                [row[c] for c in key_columns],
                row.drop(labels=[c for c in (valid_from_column, valid_to_column) if c in row.index]).to_dict(),
                valid_from=row[valid_from_column],
                valid_to=row.get(valid_to_column, _dt.date(2099, 12, 31)),
                version_id=str(row.get(version_column, "1.0.0")),
                source=str(row.get(source_column, "unspecified")),
            )
            n += 1
        return n

    # ----------------------------------------------------------------- read

    def resolve(
        self,
        domain: str,
        key: Sequence[Any] | Any,
        service_time: _dt.date | _dt.datetime | str,
    ) -> ReferenceRecord:
        """Resolve the version in force **on the service date**.

        Raises :class:`ReferenceNotEffective` rather than returning the latest
        version, because "the latest version" is precisely the wrong answer for
        a claim submitted two years after service.
        """
        day = as_of_date(service_time)
        k = tuple(key) if isinstance(key, (list, tuple)) else (key,)
        bucket = self._records.get((domain, k), [])
        viable = [r for r in bucket if r.covers(day)]
        if not viable:
            raise ReferenceNotEffective(
                f"No {domain} version for key {k} was in force on {day.isoformat()}. "
                f"{len(bucket)} version(s) exist with ranges "
                f"{[(r.valid_from.isoformat(), r.valid_to.isoformat()) for r in bucket]}. "
                "The service will not fall back to the current version."
            )
        # If two versions overlap, the one recorded latest wins, and the overlap
        # is itself a governance defect worth surfacing.
        viable.sort(key=lambda r: (r.valid_from, r.recorded_at))
        return viable[-1]

    def try_resolve(
        self, domain: str, key: Sequence[Any] | Any, service_time: Any
    ) -> ReferenceRecord | None:
        try:
            return self.resolve(domain, key, service_time)
        except (ReferenceNotEffective, ValueError):
            return None

    def versions(self, domain: str, key: Sequence[Any] | Any) -> list[ReferenceRecord]:
        k = tuple(key) if isinstance(key, (list, tuple)) else (key,)
        return list(self._records.get((domain, k), []))

    def domains(self) -> list[str]:
        return sorted({d for d, _ in self._records})

    def version_stamp(self, domain: str, key: Sequence[Any] | Any, service_time: Any) -> str:
        """``domain@version`` string stamped onto a signal's reference_versions."""
        rec = self.try_resolve(domain, key, service_time)
        return f"{domain}@{rec.version_id}" if rec else f"{domain}@NOT_EFFECTIVE"

    def overlap_defects(self) -> list[dict[str, Any]]:
        """Reference rows whose validity windows overlap.

        Surfaced on the Governance page: an overlap means two versions of the
        same policy were in force simultaneously, which makes a decision
        non-reproducible.
        """
        defects = []
        for (domain, key), bucket in self._records.items():
            ordered = sorted(bucket, key=lambda r: r.valid_from)
            for a, b in zip(ordered, ordered[1:]):
                if b.valid_from <= a.valid_to:
                    defects.append(
                        {
                            "domain": domain,
                            "key": key,
                            "version_a": a.version_id,
                            "version_b": b.version_id,
                            "overlap_from": b.valid_from.isoformat(),
                            "overlap_to": min(a.valid_to, b.valid_to).isoformat(),
                        }
                    )
        return defects

    def __len__(self) -> int:
        return sum(len(v) for v in self._records.values())


def _coerce(value: Any) -> _dt.date:
    if isinstance(value, _dt.datetime):
        return value.date()
    if isinstance(value, _dt.date):
        return value
    if isinstance(value, str):
        return _dt.date.fromisoformat(value[:10])
    if hasattr(value, "date"):
        return value.date()
    raise TypeError(f"Cannot interpret {value!r} as an effective date")
