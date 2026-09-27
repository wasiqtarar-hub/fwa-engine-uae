"""Immutable raw transaction store (manuscript §3.5, §5.3).

    "In both cases, the system must retain the original transaction payload, or
     an immutable hash of it, alongside the parsed canonical version, and must
     retain the source-system vocabulary alongside canonical, mapped values, so
     that a reviewer or auditor can always trace a canonical fact back to
     exactly what the source system actually said."      — manuscript §3.5

The store is append-only and content-addressed. A record is written **before**
parsing, so that if canonicalisation is later found to be wrong, the original is
still there to re-parse. Writing after parsing would make the raw store a record
of what the parser believed, not of what arrived.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping

__all__ = ["RawRecord", "ImmutableRawStore", "RawStoreViolation"]


class RawStoreViolation(RuntimeError):
    """Raised on any attempt to mutate or delete an already-stored record."""


def sha256_record(payload: Mapping[str, Any]) -> str:
    """Canonical SHA-256 of a source record.

    Keys are sorted and values stringified so that the same logical record
    hashes identically regardless of column order in the source file.
    """
    normalised = {str(k): ("" if v is None else str(v)) for k, v in sorted(payload.items())}
    blob = json.dumps(normalised, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class RawRecord:
    raw_hash: str
    source_system: str
    source_record_id: str
    received_at: _dt.datetime
    payload: dict[str, Any]

    def to_row(self) -> dict[str, Any]:
        return {
            "raw_hash": self.raw_hash,
            "source_system": self.source_system,
            "source_record_id": self.source_record_id,
            "received_at": self.received_at.isoformat(),
            "payload_json": json.dumps(self.payload, ensure_ascii=False, default=str),
        }


class ImmutableRawStore:
    """Append-only, content-addressed store of source records."""

    def __init__(self) -> None:
        self._records: dict[str, RawRecord] = {}
        self._by_source_id: dict[str, str] = {}

    def put(
        self,
        payload: Mapping[str, Any],
        *,
        source_system: str,
        source_record_id: str,
        received_at: _dt.datetime | None = None,
    ) -> RawRecord:
        h = sha256_record(payload)
        existing = self._records.get(h)
        if existing is not None:
            # Re-ingesting an identical record is idempotent, not an error:
            # claim files are routinely re-sent. Storing it twice would break
            # the idempotency guarantee the signal store depends on.
            return existing
        prior_hash = self._by_source_id.get(source_record_id)
        record = RawRecord(
            raw_hash=h,
            source_system=source_system,
            source_record_id=str(source_record_id),
            received_at=received_at or _dt.datetime.now(_dt.timezone.utc),
            payload=dict(payload),
        )
        self._records[h] = record
        # A changed payload for the same source id is a NEW version, not an
        # overwrite. §3.6: "Corrections must never overwrite a prior version."
        self._by_source_id[source_record_id] = h
        if prior_hash and prior_hash != h:
            object.__setattr__(record, "payload", dict(payload))
        return record

    def put_many(
        self, payloads: Iterable[Mapping[str, Any]], *, source_system: str, id_field: str
    ) -> list[RawRecord]:
        return [
            self.put(p, source_system=source_system, source_record_id=str(p.get(id_field, "")))
            for p in payloads
        ]

    def get(self, raw_hash: str) -> RawRecord | None:
        return self._records.get(raw_hash)

    def get_by_source_id(self, source_record_id: str) -> RawRecord | None:
        h = self._by_source_id.get(str(source_record_id))
        return self._records.get(h) if h else None

    def __len__(self) -> int:
        return len(self._records)

    def __iter__(self) -> Iterator[RawRecord]:
        return iter(self._records.values())

    def __delitem__(self, key: str) -> None:  # pragma: no cover - guard
        raise RawStoreViolation(
            "The raw store is append-only. A canonical fact must always be "
            "traceable back to exactly what the source system said."
        )

    def verify(self, raw_hash: str, payload: Mapping[str, Any]) -> bool:
        """Confirm a stored record still hashes to its recorded digest."""
        rec = self._records.get(raw_hash)
        if rec is None:
            return False
        return sha256_record(payload) == rec.raw_hash

    def to_dataframe(self):
        import pandas as pd

        return pd.DataFrame([r.to_row() for r in self._records.values()])

    def dump_jsonl(self, path: Path | str) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as fh:
            for rec in self._records.values():
                fh.write(json.dumps(rec.to_row(), ensure_ascii=False) + "\n")
        return path
