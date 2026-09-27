"""Authentication and session management (build brief §13.3).

    "Keep it self-contained and offline: a ``users`` table (SQLAlchemy) with
     ``username``, ``password_hash`` (**bcrypt or Argon2 — never plaintext,
     never a bare SHA**), ``role``, ``tenant_id``, ``is_active``, ``created_by``,
     ``last_login``, ``failed_attempts``, ``must_change_password``. Session
     state held in Streamlit's ``st.session_state`` with an idle timeout
     (``cfg.session_timeout_minutes``) and a visible session banner showing
     user, role and tenant."

Argon2id is used, via ``argon2-cffi``. Not bcrypt, and emphatically not a bare
SHA — a fast hash over a password is the same mistake as storing the password.

**This is a dissertation artefact, not a production security system.** It
demonstrates the §9.4 access-control requirement: roles, separation of duties,
PHI masking, reason-required unmasking, an append-only access log. It has not
been hardened for real PHI, it has no rate limiting beyond a lockout counter, no
MFA, no password policy beyond a length check, no session-fixation defence
beyond a fresh id per login, and no threat model. The README says so, and so
does this docstring, because the one thing worse than an unhardened auth layer
is an unhardened auth layer that does not say so.
"""

from __future__ import annotations

import datetime as _dt
import secrets
from dataclasses import dataclass, field
from typing import Any, Iterable

from argon2 import PasswordHasher
from argon2.exceptions import VerifyMismatchError, VerificationError

from ..audit.log import AuditEventType
from ..storage.db import Database, ReviewOutcome, SessionRecord, User
from .rbac import (
    AccessDenied, Permission, Role, ROLE_DESCRIPTIONS, assert_separation_of_duties,
    has_permission, require, visible_pages,
)

__all__ = ["AuthService", "SessionState", "SEED_ACCOUNTS", "AuthenticationError"]

_HASHER = PasswordHasher()

MAX_FAILED_ATTEMPTS = 5

#: One demo account per role, seeded on first run. §13.3 asks for
#: "admin/admin style credentials, forced password change on first login,
#: printed once to the console and documented in the user guide".
SEED_ACCOUNTS: list[dict[str, str]] = [
    {"username": "admin", "password": "admin", "role": Role.ADMIN.value,
     "display_name": "Amina Al Nuaimi", "tenant_id": "T001"},
    {"username": "policy", "password": "policy", "role": Role.POLICY_OWNER.value,
     "display_name": "Priya Menon", "tenant_id": "T001"},
    {"username": "reviewer", "password": "reviewer", "role": Role.CLAIMS_REVIEWER.value,
     "display_name": "Rashid Haddad", "tenant_id": "T001"},
    {"username": "clinical", "password": "clinical", "role": Role.CLINICAL_REVIEWER.value,
     "display_name": "Dr Layla Idris", "tenant_id": "T001"},
    {"username": "siu", "password": "siu", "role": Role.SIU_INVESTIGATOR.value,
     "display_name": "Samir Iqbal", "tenant_id": "T001"},
    {"username": "analyst", "password": "analyst", "role": Role.ANALYST.value,
     "display_name": "Ana Lyszczak", "tenant_id": "T001"},
    {"username": "qa", "password": "qa", "role": Role.QA.value,
     "display_name": "Quentin Abara", "tenant_id": "T001"},
    {"username": "auditor", "password": "auditor", "role": Role.AUDITOR.value,
     "display_name": "Aisha Ud-Din", "tenant_id": "T001"},
    # A second tenant, so tenant isolation is demonstrable in the UI rather
    # than only in a test.
    {"username": "reviewer2", "password": "reviewer2", "role": Role.CLAIMS_REVIEWER.value,
     "display_name": "Second-tenant reviewer", "tenant_id": "T002"},
]


class AuthenticationError(PermissionError):
    pass


@dataclass
class SessionState:
    session_id: str
    username: str
    display_name: str
    role: Role
    tenant_id: str
    started_at: _dt.datetime
    last_seen: _dt.datetime
    must_change_password: bool = False
    extra_permissions: list[str] = field(default_factory=list)
    assigned_case_ids: list[str] = field(default_factory=list)
    unmasked_subjects: set[str] = field(default_factory=set)

    @property
    def banner(self) -> str:
        return f"{self.display_name} · {self.role.value} · tenant {self.tenant_id}"

    def can(self, permission: Permission | str) -> bool:
        return has_permission(self.role, permission, self.extra_permissions)

    def pages(self) -> list[str]:
        return visible_pages(self.role, self.extra_permissions)

    def is_idle(self, timeout_minutes: int, now: _dt.datetime | None = None) -> bool:
        now = now or _dt.datetime.now(_dt.timezone.utc)
        return (now - self.last_seen).total_seconds() > timeout_minutes * 60

    def touch(self) -> None:
        self.last_seen = _dt.datetime.now(_dt.timezone.utc)


class AuthService:
    def __init__(self, database: Database | None = None, audit=None, config=None) -> None:
        self.db = database or Database()
        self.audit = audit
        self.config = config
        self.seeded_credentials: list[tuple[str, str, str]] = []

    # ------------------------------------------------------------------- seed

    def seed(self, *, force: bool = False) -> list[tuple[str, str, str]]:
        """Create one demo account per role on first run.

        Returns the credentials so the caller can print them **once**. They are
        not written to a file, and every seeded account has
        ``must_change_password=True``.
        """
        created: list[tuple[str, str, str]] = []
        for spec in SEED_ACCOUNTS:
            if not force and self.db.get_user(spec["username"]) is not None:
                continue
            self.db.upsert_user(User(
                username=spec["username"],
                password_hash=_HASHER.hash(spec["password"]),
                role=spec["role"],
                tenant_id=spec["tenant_id"],
                display_name=spec["display_name"],
                is_active=True,
                created_by="system:seed",
                must_change_password=True,
            ))
            created.append((spec["username"], spec["password"], spec["role"]))
        self.seeded_credentials = created
        return created

    # ------------------------------------------------------------- login/out

    def login(self, username: str, password: str) -> SessionState:
        user = self.db.get_user(username)
        if user is None or not user.is_active:
            self._audit(AuditEventType.LOGIN_FAILURE, username or "unknown", "",
                        reason="Unknown or disabled account.")
            raise AuthenticationError("Incorrect username or password.")
        if user.failed_attempts >= MAX_FAILED_ATTEMPTS:
            self._audit(AuditEventType.LOGIN_FAILURE, username, user.role,
                        reason="Account locked after repeated failures.")
            raise AuthenticationError(
                f"This account is locked after {MAX_FAILED_ATTEMPTS} failed attempts. "
                "An administrator must reset it."
            )
        try:
            _HASHER.verify(user.password_hash, password)
        except (VerifyMismatchError, VerificationError):
            user.failed_attempts += 1
            self.db.upsert_user(user)
            self._audit(AuditEventType.LOGIN_FAILURE, username, user.role,
                        reason=f"Incorrect password (attempt {user.failed_attempts}).")
            raise AuthenticationError("Incorrect username or password.")

        if _HASHER.check_needs_rehash(user.password_hash):
            user.password_hash = _HASHER.hash(password)
        user.failed_attempts = 0
        user.last_login = _dt.datetime.now(_dt.timezone.utc)
        self.db.upsert_user(user)

        now = _dt.datetime.now(_dt.timezone.utc)
        state = SessionState(
            session_id=secrets.token_urlsafe(24),
            username=user.username,
            display_name=user.display_name or user.username,
            role=Role(user.role),
            tenant_id=user.tenant_id,
            started_at=now,
            last_seen=now,
            must_change_password=user.must_change_password,
            extra_permissions=[p for p in (user.extra_permissions or "").split(",") if p],
            assigned_case_ids=[c for c in (user.assigned_case_ids or "").split(",") if c],
        )
        self.db.add(SessionRecord(
            session_id=state.session_id, username=user.username, role=user.role,
            tenant_id=user.tenant_id, started_at=now, last_seen=now,
        ))
        self._audit(AuditEventType.LOGIN_SUCCESS, user.username, user.role,
                    tenant_id=user.tenant_id, reason="Session started.")
        return state

    def logout(self, state: SessionState, reason: str = "user_logout") -> None:
        self._audit(AuditEventType.LOGOUT if reason == "user_logout" else AuditEventType.SESSION_TIMEOUT,
                    state.username, state.role.value, tenant_id=state.tenant_id, reason=reason)

    def enforce_timeout(self, state: SessionState) -> bool:
        """Return True if the session has expired (§13.3 idle timeout)."""
        minutes = int(self.config.get("session_timeout_minutes")) if self.config else 30
        if state.is_idle(minutes):
            self.logout(state, reason=f"idle_timeout_{minutes}m")
            return True
        state.touch()
        return False

    # ---------------------------------------------------------- user admin

    def create_user(self, *, actor: SessionState, username: str, password: str,
                    role: str, tenant_id: str, display_name: str = "") -> User:
        require(actor.role, Permission.MANAGE_USERS, actor.extra_permissions)
        if self.db.get_user(username) is not None:
            raise ValueError(f"User {username!r} already exists.")
        if len(password) < 4:
            raise ValueError("Password is too short.")
        user = User(
            username=username, password_hash=_HASHER.hash(password), role=Role(role).value,
            tenant_id=tenant_id, display_name=display_name or username,
            created_by=actor.username, must_change_password=True,
        )
        self.db.upsert_user(user)
        self._audit(AuditEventType.USER_CREATED, actor.username, actor.role.value,
                    tenant_id=tenant_id, subject=username,
                    reason=f"Created with role {role}.", after=user.to_row())
        return user

    def set_active(self, *, actor: SessionState, username: str, active: bool, reason: str) -> User:
        require(actor.role, Permission.MANAGE_USERS, actor.extra_permissions)
        user = self.db.get_user(username)
        if user is None:
            raise ValueError(f"No such user {username!r}.")
        before = user.to_row()
        user.is_active = active
        self.db.upsert_user(user)
        self._audit(AuditEventType.USER_DISABLED, actor.username, actor.role.value,
                    subject=username, reason=reason, before=before, after=user.to_row())
        return user

    def change_role(self, *, actor: SessionState, username: str, role: str, reason: str) -> User:
        require(actor.role, Permission.MANAGE_USERS, actor.extra_permissions)
        user = self.db.get_user(username)
        if user is None:
            raise ValueError(f"No such user {username!r}.")
        before = user.to_row()
        user.role = Role(role).value
        self.db.upsert_user(user)
        self._audit(AuditEventType.USER_ROLE_CHANGED, actor.username, actor.role.value,
                    subject=username, reason=reason, before=before, after=user.to_row())
        return user

    def reset_password(self, *, actor: SessionState, username: str, new_password: str,
                       reason: str) -> None:
        require(actor.role, Permission.MANAGE_USERS, actor.extra_permissions)
        user = self.db.get_user(username)
        if user is None:
            raise ValueError(f"No such user {username!r}.")
        user.password_hash = _HASHER.hash(new_password)
        user.must_change_password = True
        user.failed_attempts = 0
        self.db.upsert_user(user)
        self._audit(AuditEventType.PASSWORD_RESET, actor.username, actor.role.value,
                    subject=username, reason=reason)

    def change_own_password(self, state: SessionState, current: str, new: str) -> None:
        user = self.db.get_user(state.username)
        if user is None:
            raise AuthenticationError("Session user no longer exists.")
        try:
            _HASHER.verify(user.password_hash, current)
        except (VerifyMismatchError, VerificationError):
            raise AuthenticationError("Current password is incorrect.")
        if len(new) < 8:
            raise ValueError("Choose a password of at least 8 characters.")
        if new == current:
            raise ValueError("The new password must differ from the current one.")
        user.password_hash = _HASHER.hash(new)
        user.must_change_password = False
        self.db.upsert_user(user)
        state.must_change_password = False
        self._audit(AuditEventType.PASSWORD_CHANGED, state.username, state.role.value,
                    tenant_id=state.tenant_id, reason="User changed their own password.")

    def grant_permission(self, *, actor: SessionState, username: str, permission: str,
                         reason: str) -> User:
        """Grant an individual permission — how ``MANUAL_OVERRIDE`` is conferred.

        §13.2: "Manual adjudication override is a distinct permission, NOT
        implied by any role."
        """
        require(actor.role, Permission.MANAGE_USERS, actor.extra_permissions)
        user = self.db.get_user(username)
        if user is None:
            raise ValueError(f"No such user {username!r}.")
        before = user.to_row()
        permissions = {p for p in (user.extra_permissions or "").split(",") if p}
        permissions.add(Permission(permission).value)
        user.extra_permissions = ",".join(sorted(permissions))
        self.db.upsert_user(user)
        self._audit(AuditEventType.USER_ROLE_CHANGED, actor.username, actor.role.value,
                    subject=username, reason=f"Granted {permission}: {reason}",
                    before=before, after=user.to_row())
        return user

    def assign_case(self, *, actor: SessionState, username: str, case_id: str, reason: str) -> User:
        """Assign an SIU case. Unmasking is permitted on assigned cases only."""
        require(actor.role, Permission.MANAGE_USERS, actor.extra_permissions)
        user = self.db.get_user(username)
        if user is None:
            raise ValueError(f"No such user {username!r}.")
        cases = {c for c in (user.assigned_case_ids or "").split(",") if c}
        cases.add(case_id)
        user.assigned_case_ids = ",".join(sorted(cases))
        self.db.upsert_user(user)
        self._audit(AuditEventType.USER_ROLE_CHANGED, actor.username, actor.role.value,
                    subject=username, reason=f"Assigned case {case_id}: {reason}")
        return user

    # --------------------------------------------------------- PHI unmasking

    def unmask(self, state: SessionState, subject_id: str, reason: str) -> bool:
        """Explicit, reason-required unmasking that writes an access event (§13.2)."""
        if not reason or len(reason.strip()) < 10:
            raise ValueError(
                "Unmasking requires a stated reason of at least 10 characters. It is an "
                "explicit, reason-required action precisely so that it leaves a trail."
            )
        if not state.can(Permission.VIEW_UNMASKED_IDENTITY):
            self._audit(AuditEventType.ACCESS_DENIED, state.username, state.role.value,
                        tenant_id=state.tenant_id, subject=subject_id,
                        reason="Unmasking refused: role does not hold VIEW_UNMASKED_IDENTITY.")
            raise AccessDenied(
                f"Role {state.role.value} may not view unmasked identity."
            )
        state.unmasked_subjects.add(subject_id)
        self._audit(AuditEventType.PHI_UNMASK, state.username, state.role.value,
                    tenant_id=state.tenant_id, subject=subject_id, reason=reason)
        return True

    # ----------------------------------------------------- review outcomes

    def record_disposition(
        self, *, state: SessionState, case, disposition: str, rationale: str,
        validated_category: str = "", confirmed_amount_aed: float = 0.0,
        is_override: bool = False, original_disposition: str = "",
    ) -> ReviewOutcome:
        """Record a reviewer's decision. Rationale is mandatory (§14.1)."""
        permission = (
            Permission.MANUAL_OVERRIDE if is_override else Permission.RECORD_DISPOSITION
        )
        require(state.role, permission, state.extra_permissions)
        if not rationale or len(rationale.strip()) < 10:
            raise ValueError(
                "A rationale of at least 10 characters is required before a disposition can be "
                "submitted. The rationale is the feedback that tunes the rule (FR7), and a "
                "disposition without one teaches the system nothing."
            )
        outcome = ReviewOutcome(
            review_id=secrets.token_hex(12),
            case_id=case.case_id,
            signal_ids=",".join(case.signal_ids),
            reviewer=state.username,
            reviewer_role=state.role.value,
            disposition=disposition,
            validated_category=validated_category,
            confirmed_amount_aed=float(confirmed_amount_aed),
            rationale=rationale.strip(),
            tenant_id=state.tenant_id,
            was_override=is_override,
            original_disposition=original_disposition or case.disposition.value,
        )
        self.db.add(outcome)
        self._audit(
            AuditEventType.MANUAL_OVERRIDE if is_override else AuditEventType.DISPOSITION_RECORDED,
            state.username, state.role.value, tenant_id=state.tenant_id, subject=case.case_id,
            reason=rationale.strip(),
            before={"disposition": outcome.original_disposition},
            after={"disposition": disposition, "validated_category": validated_category,
                   "confirmed_amount_aed": confirmed_amount_aed},
        )
        return outcome

    def capture_appeal(self, *, state: SessionState, review_id: str, result: str,
                       reason: str) -> ReviewOutcome:
        require(state.role, Permission.CAPTURE_APPEAL, state.extra_permissions)
        with self.db.session() as s:
            outcome = s.get(ReviewOutcome, review_id)
            if outcome is None:
                raise ValueError(f"No such review outcome {review_id!r}.")
            outcome.appeal_raised = True
            outcome.appeal_result = result
            outcome.appeal_decided_at = _dt.datetime.now(_dt.timezone.utc)
            if outcome.decided_at:
                # SQLite does not preserve tzinfo across a round trip, so a
                # stored timestamp comes back naive even though the column is
                # declared ``timezone=True``. Both sides are UTC by
                # construction, so a naive value is re-tagged rather than
                # shifted; subtracting them directly raises, and turnaround is
                # one of the few §6.5 metrics this build can compute at all.
                decided = outcome.decided_at
                if decided.tzinfo is None:
                    decided = decided.replace(tzinfo=_dt.timezone.utc)
                delta = outcome.appeal_decided_at - decided
                outcome.turnaround_hours = round(delta.total_seconds() / 3600, 3)
            s.commit()
            s.refresh(outcome)
        self._audit(AuditEventType.APPEAL_CAPTURED, state.username, state.role.value,
                    tenant_id=state.tenant_id, subject=review_id, reason=reason,
                    after={"appeal_result": result})
        return outcome

    # ----------------------------------------------------- model promotion

    def propose_model_promotion(self, *, state: SessionState, model: str, reason: str) -> None:
        require(state.role, Permission.PROPOSE_MODEL_PROMOTION, state.extra_permissions)
        self._audit(AuditEventType.MODEL_PROMOTION_PROPOSED, state.username, state.role.value,
                    tenant_id=state.tenant_id, subject=model, reason=reason)

    def approve_model_promotion(self, *, state: SessionState, model: str, proposer: str,
                                reason: str) -> None:
        require(state.role, Permission.APPROVE_MODEL_PROMOTION, state.extra_permissions)
        assert_separation_of_duties(state.username, proposer, "approve the promotion of")
        self._audit(AuditEventType.MODEL_PROMOTION_APPROVED, state.username, state.role.value,
                    tenant_id=state.tenant_id, subject=model, reason=reason)

    # ---------------------------------------------------------------- audit

    def access_log(self, tenant_id: str | None = None):
        import pandas as pd

        if self.audit is None:
            return pd.DataFrame()
        frame = self.audit.to_dataframe(tenant_id)
        access_events = {
            AuditEventType.LOGIN_SUCCESS.value, AuditEventType.LOGIN_FAILURE.value,
            AuditEventType.LOGOUT.value, AuditEventType.SESSION_TIMEOUT.value,
            AuditEventType.PHI_UNMASK.value, AuditEventType.ACCESS_DENIED.value,
            AuditEventType.USER_CREATED.value, AuditEventType.USER_DISABLED.value,
            AuditEventType.USER_ROLE_CHANGED.value, AuditEventType.PASSWORD_RESET.value,
            AuditEventType.PASSWORD_CHANGED.value,
        }
        return frame[frame["event_type"].isin(access_events)] if not frame.empty else frame

    def _audit(self, event_type, actor: str, actor_role: str, *, tenant_id: str = "",
               subject: str = "", reason: str = "", before=None, after=None) -> None:
        if self.audit is None:
            return
        self.audit.record(event_type, actor=actor, actor_role=actor_role, tenant_id=tenant_id,
                          subject=subject, reason=reason, before=before, after=after)
