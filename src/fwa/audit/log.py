"""Append-only audit log (manuscript §3.10 audit-logging NFR, §9.4, §9.7).

    "audit logging (every disposition, override and configuration change is
     permanently and attributably recorded)"              — manuscript §3.10

Three properties make this an audit log rather than a list of events:

1. **Append-only in the API, not merely by convention.** There is no ``delete``,
   no ``update``, and :meth:`AuditLog.__delitem__` raises. The build brief §13.2
   is explicit: "No role can delete from it."
2. **Hash-chained.** Each entry carries the hash of the previous entry, so a
   tampered or removed row is detectable by :meth:`AuditLog.verify_chain`. The
   Governance page runs that verification and shows the result, because an audit
   log nobody verifies is decoration.
3. **Attributable.** Every entry records the acting user, their role, their
   tenant and a reason. Entries with no actor are rejected — a configuration
   change that cannot be attributed is exactly what §3.10 exists to prevent.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
from typing import Any, Iterator

__all__ = ["AuditEvent", "AuditEventType", "AuditLog", "AuditLogViolation"]


class AuditLogViolation(RuntimeError):
    pass


class AuditEventType(str, Enum):
    # access
    LOGIN_SUCCESS = "LOGIN_SUCCESS"
    LOGIN_FAILURE = "LOGIN_FAILURE"
    LOGOUT = "LOGOUT"
    SESSION_TIMEOUT = "SESSION_TIMEOUT"
    PHI_UNMASK = "PHI_UNMASK"
    ACCESS_DENIED = "ACCESS_DENIED"
    # user administration
    USER_CREATED = "USER_CREATED"
    USER_DISABLED = "USER_DISABLED"
    USER_ROLE_CHANGED = "USER_ROLE_CHANGED"
    PASSWORD_RESET = "PASSWORD_RESET"
    PASSWORD_CHANGED = "PASSWORD_CHANGED"
    # review
    DISPOSITION_RECORDED = "DISPOSITION_RECORDED"
    CASE_ESCALATED = "CASE_ESCALATED"
    APPEAL_CAPTURED = "APPEAL_CAPTURED"
    MANUAL_OVERRIDE = "MANUAL_OVERRIDE"
    CASE_MERGED = "CASE_MERGED"
    CASE_SPLIT = "CASE_SPLIT"
    # configuration and governance
    RULE_DRAFTED = "RULE_DRAFTED"
    RULE_ACTIVATED = "RULE_ACTIVATED"
    RULE_RETIRED = "RULE_RETIRED"
    RULE_ROLLED_BACK = "RULE_ROLLED_BACK"
    PARAMETER_CHANGED = "PARAMETER_CHANGED"
    KILL_SWITCH_ENGAGED = "KILL_SWITCH_ENGAGED"
    KILL_SWITCH_RELEASED = "KILL_SWITCH_RELEASED"
    ALERT_CEILING_BREACHED = "ALERT_CEILING_BREACHED"
    MODEL_PROMOTION_PROPOSED = "MODEL_PROMOTION_PROPOSED"
    MODEL_PROMOTION_APPROVED = "MODEL_PROMOTION_APPROVED"
    MODEL_PROMOTION_BLOCKED = "MODEL_PROMOTION_BLOCKED"
    # AI layer
    AI_TEXT_GENERATED = "AI_TEXT_GENERATED"
    AI_CLAIM_DROPPED_UNGROUNDED = "AI_CLAIM_DROPPED_UNGROUNDED"
    AI_COPILOT_REFUSED = "AI_COPILOT_REFUSED"
    # pipeline
    PIPELINE_RUN = "PIPELINE_RUN"
    ENTITY_MERGE_PROPOSED = "ENTITY_MERGE_PROPOSED"
    ENTITY_MERGE_CONFIRMED = "ENTITY_MERGE_CONFIRMED"


@dataclass(frozen=True)
class AuditEvent:
    sequence: int
    event_time: str
    event_type: str
    actor: str
    actor_role: str
    tenant_id: str
    subject: str
    reason: str
    before: dict[str, Any] | None
    after: dict[str, Any] | None
    prev_hash: str
    entry_hash: str

    def to_row(self) -> dict[str, Any]:
        d = asdict(self)
        d["before"] = json.dumps(self.before, default=str) if self.before is not None else None
        d["after"] = json.dumps(self.after, default=str) if self.after is not None else None
        return d


GENESIS_HASH = "0" * 64


class AuditLog:
    """Hash-chained, append-only event log."""

    def __init__(self) -> None:
        self._events: list[AuditEvent] = []

    # ------------------------------------------------------------------ write

    def record(
        self,
        event_type: AuditEventType | str,
        *,
        actor: str,
        actor_role: str = "",
        tenant_id: str = "",
        subject: str = "",
        reason: str = "",
        before: dict[str, Any] | None = None,
        after: dict[str, Any] | None = None,
    ) -> AuditEvent:
        if not actor or not str(actor).strip():
            raise AuditLogViolation(
                "An audit entry with no actor is rejected. Every disposition, override and "
                "configuration change must be ATTRIBUTABLY recorded, and an unattributable "
                "entry records nothing that can be asked about later."
            )
        et = event_type.value if isinstance(event_type, AuditEventType) else str(event_type)
        mutating = et in {
            AuditEventType.MANUAL_OVERRIDE.value,
            AuditEventType.KILL_SWITCH_ENGAGED.value,
            AuditEventType.RULE_ACTIVATED.value,
            AuditEventType.RULE_RETIRED.value,
            AuditEventType.PARAMETER_CHANGED.value,
            AuditEventType.PHI_UNMASK.value,
        }
        if mutating and not reason.strip():
            raise AuditLogViolation(
                f"{et} requires a stated reason. Unmasking, overriding and kill-switching "
                "are reason-required actions."
            )
        prev_hash = self._events[-1].entry_hash if self._events else GENESIS_HASH
        seq = len(self._events) + 1
        ts = _dt.datetime.now(_dt.timezone.utc).isoformat()
        body = json.dumps(
            {
                "sequence": seq,
                "event_time": ts,
                "event_type": et,
                "actor": actor,
                "actor_role": actor_role,
                "tenant_id": tenant_id,
                "subject": subject,
                "reason": reason,
                "before": before,
                "after": after,
                "prev_hash": prev_hash,
            },
            sort_keys=True,
            default=str,
        )
        entry_hash = hashlib.sha256(body.encode("utf-8")).hexdigest()
        event = AuditEvent(
            sequence=seq,
            event_time=ts,
            event_type=et,
            actor=actor,
            actor_role=actor_role,
            tenant_id=tenant_id,
            subject=subject,
            reason=reason,
            before=before,
            after=after,
            prev_hash=prev_hash,
            entry_hash=entry_hash,
        )
        self._events.append(event)
        return event

    # ------------------------------------------------------------------- read

    def all(self, tenant_id: str | None = None) -> list[AuditEvent]:
        if tenant_id is None:
            return list(self._events)
        # AUDITOR and ADMIN read across their own tenant only; system events
        # (tenant_id == "") are visible to everyone with log access.
        return [e for e in self._events if e.tenant_id in (tenant_id, "")]

    def by_type(self, event_type: AuditEventType | str) -> list[AuditEvent]:
        et = event_type.value if isinstance(event_type, AuditEventType) else str(event_type)
        return [e for e in self._events if e.event_type == et]

    def by_actor(self, actor: str) -> list[AuditEvent]:
        return [e for e in self._events if e.actor == actor]

    def by_subject(self, subject: str) -> list[AuditEvent]:
        return [e for e in self._events if e.subject == subject]

    def verify_chain(self) -> tuple[bool, str]:
        """Re-derive every hash. Returns (ok, message)."""
        prev = GENESIS_HASH
        for e in self._events:
            if e.prev_hash != prev:
                return False, f"Chain broken at sequence {e.sequence}: prev_hash mismatch."
            body = json.dumps(
                {
                    "sequence": e.sequence,
                    "event_time": e.event_time,
                    "event_type": e.event_type,
                    "actor": e.actor,
                    "actor_role": e.actor_role,
                    "tenant_id": e.tenant_id,
                    "subject": e.subject,
                    "reason": e.reason,
                    "before": e.before,
                    "after": e.after,
                    "prev_hash": e.prev_hash,
                },
                sort_keys=True,
                default=str,
            )
            if hashlib.sha256(body.encode("utf-8")).hexdigest() != e.entry_hash:
                return False, f"Entry {e.sequence} has been altered since it was written."
            prev = e.entry_hash
        return True, f"Verified {len(self._events)} entries; hash chain intact."

    def to_dataframe(self, tenant_id: str | None = None):
        import pandas as pd

        rows = [e.to_row() for e in self.all(tenant_id)]
        cols = [
            "sequence", "event_time", "event_type", "actor", "actor_role", "tenant_id",
            "subject", "reason", "before", "after", "prev_hash", "entry_hash",
        ]
        return pd.DataFrame(rows, columns=cols) if rows else pd.DataFrame(columns=cols)

    def dump_jsonl(self, path: Path | str) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as fh:
            for e in self._events:
                fh.write(json.dumps(e.to_row(), ensure_ascii=False, default=str) + "\n")
        return path

    # ---- append-only guards -------------------------------------------------

    def __delitem__(self, index: int) -> None:  # pragma: no cover - guard
        raise AuditLogViolation("The audit log is append-only. No role may delete from it.")

    def __setitem__(self, index: int, value: Any) -> None:  # pragma: no cover - guard
        raise AuditLogViolation("The audit log is append-only. Entries are immutable once written.")

    def clear(self) -> None:  # pragma: no cover - guard
        raise AuditLogViolation("The audit log is append-only and cannot be cleared.")

    def __len__(self) -> int:
        return len(self._events)

    def __iter__(self) -> Iterator[AuditEvent]:
        return iter(self._events)
