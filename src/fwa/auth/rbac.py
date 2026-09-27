"""Roles, permissions and the access guard (build brief §13, manuscript §9.4).

The eight roles are §13.1's table, and the permission model enforces §13.2's
rules in code rather than in the UI:

* **Separation of duties.** The author of a rule may not approve it; the
  proposer of a model promotion may not approve it. Enforced in
  :func:`assert_separation_of_duties` and in
  :meth:`fwa.engine.registry.RuleRegistry.activate`, so a second entry point
  cannot bypass it.
* **PHI minimisation.** ``patient_id`` and ``hospital_id`` are **masked by
  default** for every role except ``ADMIN`` and ``SIU_INVESTIGATOR`` on assigned
  cases. Unmasking is an explicit, reason-required action that writes an access
  event to the audit log. Crucially, :func:`mask_identity` masks *identity*, not
  *evidence* — §13.2: "Never mask so thoroughly that the reviewer cannot do
  their job."
* **Override control.** ``MANUAL_OVERRIDE`` is a distinct permission held by no
  role by default. It has to be granted to a user individually.
* **Kill switch and rollback** are ``ADMIN``-only.
* **Tenant scoping.** Every session carries one tenant and every query is
  filtered by it.
* **The audit log is append-only.** No role has a delete permission over it,
  because the log has no delete operation to permit.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterable

__all__ = [
    "Role", "Permission", "ROLE_PERMISSIONS", "ROLE_DESCRIPTIONS", "PAGE_PERMISSIONS",
    "AccessDenied", "SeparationOfDutiesError", "has_permission", "require",
    "mask_identity", "assert_separation_of_duties", "visible_pages",
]


class AccessDenied(PermissionError):
    """The session's role does not hold the required permission."""


class SeparationOfDutiesError(PermissionError):
    """The actor may not approve their own work (§13.2)."""


class Role(str, Enum):
    ADMIN = "ADMIN"
    POLICY_OWNER = "POLICY_OWNER"
    CLAIMS_REVIEWER = "CLAIMS_REVIEWER"
    CLINICAL_REVIEWER = "CLINICAL_REVIEWER"
    SIU_INVESTIGATOR = "SIU_INVESTIGATOR"
    ANALYST = "ANALYST"
    QA = "QA"
    AUDITOR = "AUDITOR"


class Permission(str, Enum):
    # viewing
    VIEW_OVERVIEW = "VIEW_OVERVIEW"
    VIEW_QUEUE = "VIEW_QUEUE"
    VIEW_CASE_EVIDENCE = "VIEW_CASE_EVIDENCE"
    VIEW_PROVIDER_ANALYTICS = "VIEW_PROVIDER_ANALYTICS"
    VIEW_NETWORK = "VIEW_NETWORK"
    VIEW_MODELS = "VIEW_MODELS"
    VIEW_RULE_REGISTRY = "VIEW_RULE_REGISTRY"
    VIEW_GOVERNANCE = "VIEW_GOVERNANCE"
    VIEW_VALIDATION_REPORT = "VIEW_VALIDATION_REPORT"
    VIEW_AUDIT_LOG = "VIEW_AUDIT_LOG"
    VIEW_TEST_EVIDENCE = "VIEW_TEST_EVIDENCE"
    VIEW_HELP = "VIEW_HELP"
    VIEW_PARAMETERS = "VIEW_PARAMETERS"
    LOAD_DATASET = "LOAD_DATASET"
    VIEW_UNMASKED_IDENTITY = "VIEW_UNMASKED_IDENTITY"
    # acting
    RECORD_DISPOSITION = "RECORD_DISPOSITION"
    RECORD_CLINICAL_DISPOSITION = "RECORD_CLINICAL_DISPOSITION"
    CAPTURE_APPEAL = "CAPTURE_APPEAL"
    ESCALATE_TO_SIU = "ESCALATE_TO_SIU"
    INVESTIGATE_CASE = "INVESTIGATE_CASE"
    LINK_CASES = "LINK_CASES"
    MANUAL_OVERRIDE = "MANUAL_OVERRIDE"
    # governance
    AUTHOR_RULE = "AUTHOR_RULE"
    AUTHOR_EXPERT_RULE = "AUTHOR_EXPERT_RULE"
    APPROVE_RULE_ACTIVATION = "APPROVE_RULE_ACTIVATION"
    RETIRE_RULE = "RETIRE_RULE"
    SET_PARAMETER = "SET_PARAMETER"
    PROPOSE_MODEL_PROMOTION = "PROPOSE_MODEL_PROMOTION"
    APPROVE_MODEL_PROMOTION = "APPROVE_MODEL_PROMOTION"
    RUN_PIPELINE = "RUN_PIPELINE"
    RECALIBRATE_MODEL = "RECALIBRATE_MODEL"
    KILL_SWITCH = "KILL_SWITCH"
    ROLLBACK = "ROLLBACK"
    CONFIRM_ENTITY_MERGE = "CONFIRM_ENTITY_MERGE"
    RECORD_GATE_EVIDENCE = "RECORD_GATE_EVIDENCE"
    # administration
    MANAGE_USERS = "MANAGE_USERS"
    EXPORT_EVIDENCE = "EXPORT_EVIDENCE"


P = Permission
R = Role

#: §13.1, expressed as permissions.
ROLE_PERMISSIONS: dict[Role, set[Permission]] = {
    R.ADMIN: set(Permission) - {P.MANUAL_OVERRIDE},   # override is granted individually
    R.POLICY_OWNER: {
        P.VIEW_OVERVIEW, P.VIEW_QUEUE, P.VIEW_CASE_EVIDENCE, P.VIEW_PROVIDER_ANALYTICS,
        P.VIEW_MODELS, P.VIEW_RULE_REGISTRY, P.VIEW_GOVERNANCE, P.VIEW_VALIDATION_REPORT,
        P.VIEW_HELP, P.VIEW_PARAMETERS, P.AUTHOR_RULE, P.APPROVE_RULE_ACTIVATION, P.RETIRE_RULE, P.SET_PARAMETER,
        P.APPROVE_MODEL_PROMOTION, P.RECORD_GATE_EVIDENCE,
    },
    R.CLAIMS_REVIEWER: {
        P.VIEW_OVERVIEW, P.VIEW_QUEUE, P.VIEW_CASE_EVIDENCE, P.VIEW_HELP,
        P.RECORD_DISPOSITION, P.CAPTURE_APPEAL, P.ESCALATE_TO_SIU,
    },
    R.CLINICAL_REVIEWER: {
        P.VIEW_OVERVIEW, P.VIEW_QUEUE, P.VIEW_CASE_EVIDENCE, P.VIEW_RULE_REGISTRY, P.VIEW_HELP,
        P.RECORD_CLINICAL_DISPOSITION, P.RECORD_DISPOSITION, P.CAPTURE_APPEAL,
        P.ESCALATE_TO_SIU, P.AUTHOR_EXPERT_RULE,
    },
    R.SIU_INVESTIGATOR: {
        P.VIEW_OVERVIEW, P.VIEW_QUEUE, P.VIEW_CASE_EVIDENCE, P.VIEW_NETWORK, P.VIEW_HELP,
        P.INVESTIGATE_CASE, P.LINK_CASES, P.RECORD_DISPOSITION,
        # unmasking is permitted only on ASSIGNED cases — enforced in mask_identity
        P.VIEW_UNMASKED_IDENTITY,
    },
    R.ANALYST: {
        P.VIEW_OVERVIEW, P.VIEW_QUEUE, P.VIEW_PROVIDER_ANALYTICS, P.VIEW_NETWORK, P.VIEW_MODELS,
        P.VIEW_VALIDATION_REPORT, P.VIEW_HELP, P.VIEW_PARAMETERS, P.LOAD_DATASET,
        P.RUN_PIPELINE, P.RECALIBRATE_MODEL,
        P.PROPOSE_MODEL_PROMOTION,      # propose only — never approve
    },
    R.QA: {
        P.VIEW_OVERVIEW, P.VIEW_TEST_EVIDENCE, P.VIEW_RULE_REGISTRY, P.VIEW_VALIDATION_REPORT,
        P.VIEW_GOVERNANCE, P.VIEW_HELP, P.VIEW_PARAMETERS, P.RECORD_GATE_EVIDENCE,
    },
    R.AUDITOR: {
        P.VIEW_OVERVIEW, P.VIEW_QUEUE, P.VIEW_CASE_EVIDENCE, P.VIEW_PROVIDER_ANALYTICS,
        P.VIEW_NETWORK, P.VIEW_MODELS, P.VIEW_RULE_REGISTRY, P.VIEW_GOVERNANCE,
        P.VIEW_VALIDATION_REPORT, P.VIEW_AUDIT_LOG, P.VIEW_TEST_EVIDENCE, P.VIEW_HELP,
        P.VIEW_PARAMETERS,
        P.EXPORT_EVIDENCE,              # read-only everything; nothing mutating
    },
}

ROLE_DESCRIPTIONS: dict[Role, dict[str, str]] = {
    R.ADMIN: {
        "job_title": "Administrator",
        "sees": "Everything, including unmasked identity",
        "can_do": "Manage users, roles and tenants; activate rules; kill switch; rollback; "
                  "view the audit log",
    },
    R.POLICY_OWNER: {
        "job_title": "Policy owner",
        "sees": "Rule registry, shadow statistics, parameters",
        "can_do": "Author, edit and retire rules; set parameters; approve shadow→active; sign off",
    },
    R.CLAIMS_REVIEWER: {
        "job_title": "Claims reviewer",
        "sees": "Own queue, case evidence, masked identity by default",
        "can_do": "Record a disposition with rationale; capture appeals; escalate to SIU",
    },
    R.CLINICAL_REVIEWER: {
        "job_title": "Clinical/coding policy",
        "sees": "Clinical cases, documents, coding evidence",
        "can_do": "Clinical disposition; author Type-E expert rules; validate coding findings",
    },
    R.SIU_INVESTIGATOR: {
        "job_title": "SIU",
        "sees": "SIU_LEAD cases, the network graph, unmasked identity on ASSIGNED cases",
        "can_do": "Investigate; link cases; record an investigation outcome",
    },
    R.ANALYST: {
        "job_title": "Analytics / ML engineering / Technical operator",
        "sees": "Models, drift, peer baselines, validation reports",
        "can_do": "Run pipelines; recalibrate; PROPOSE a model promotion (cannot approve one)",
    },
    R.QA: {
        "job_title": "QA",
        "sees": "Test fixtures, gate evidence",
        "can_do": "Attach and confirm the ten-category suite; record gate evidence",
    },
    R.AUDITOR: {
        "job_title": "Regulator / internal audit",
        "sees": "Read-only everything, including the audit log",
        "can_do": "Nothing mutating. Export evidence",
    },
}

#: Streamlit page -> the permission that gates it. Pages not in this map are
#: unreachable. `require_role` reads this, and the navigation renders ONLY the
#: entries a role can use rather than showing disabled ones (§13.3).
PAGE_PERMISSIONS: dict[str, Permission] = {
    "Overview": P.VIEW_OVERVIEW,
    "Data": P.LOAD_DATASET,
    "Review Queue": P.VIEW_QUEUE,
    "Case Evidence": P.VIEW_CASE_EVIDENCE,
    "Provider Analytics": P.VIEW_PROVIDER_ANALYTICS,
    "Network": P.VIEW_NETWORK,
    "Models": P.VIEW_MODELS,
    "Rule Registry": P.VIEW_RULE_REGISTRY,
    "Governance": P.VIEW_GOVERNANCE,
    "Parameters & Models": P.VIEW_PARAMETERS,
    "Validation Report": P.VIEW_VALIDATION_REPORT,
    "User Management": P.MANAGE_USERS,
    "Help": P.VIEW_HELP,
}


def has_permission(role: Role | str, permission: Permission | str,
                   extra_permissions: Iterable[str] = ()) -> bool:
    role = Role(role) if not isinstance(role, Role) else role
    permission = Permission(permission) if not isinstance(permission, Permission) else permission
    if permission.value in {str(p) for p in extra_permissions}:
        return True
    return permission in ROLE_PERMISSIONS.get(role, set())


def require(role: Role | str, permission: Permission | str,
            extra_permissions: Iterable[str] = ()) -> None:
    if not has_permission(role, permission, extra_permissions):
        raise AccessDenied(
            f"Role {getattr(role, 'value', role)} does not hold "
            f"{getattr(permission, 'value', permission)}."
        )


def visible_pages(role: Role | str, extra_permissions: Iterable[str] = ()) -> list[str]:
    """Only the pages this role can use. §13.3: render, don't disable."""
    return [
        page for page, permission in PAGE_PERMISSIONS.items()
        if has_permission(role, permission, extra_permissions)
    ]


def assert_separation_of_duties(actor: str, originator: str | None, action: str) -> None:
    """§13.2 — the actor may not approve their own work."""
    if originator and actor == originator:
        raise SeparationOfDutiesError(
            f"{actor} originated this item and may not {action} it. Separation of duties: "
            f"the user who authors a rule may not approve its shadow→active transition, and "
            f"the user who proposes a model promotion may not approve it."
        )


# ---------------------------------------------------------------------------
# PHI masking
# ---------------------------------------------------------------------------


def mask_identity(
    value: Any,
    role: Role | str,
    *,
    unmask_granted: bool = False,
    assigned: bool = False,
    extra_permissions: Iterable[str] = (),
) -> str:
    """Mask a member or provider identifier unless the role may see it.

    ``ADMIN`` sees identity. ``SIU_INVESTIGATOR`` sees it **on assigned cases
    only**. Everyone else gets a stable pseudonym, so a reviewer can still tell
    two providers apart, correlate across pages and describe "the provider in
    case X" — they simply cannot name them. That is the §13.2 distinction: mask
    IDENTITY, not EVIDENCE.
    """
    if value is None:
        return "—"
    text = str(value)
    role = Role(role) if not isinstance(role, Role) else role
    if role is Role.ADMIN:
        return text
    if unmask_granted:
        return text
    if role is Role.SIU_INVESTIGATOR and assigned:
        return text
    if has_permission(role, Permission.VIEW_UNMASKED_IDENTITY, extra_permissions) and assigned:
        return text
    digest = hashlib.sha256(f"phi-mask:{text}".encode("utf-8")).hexdigest()[:8].upper()
    prefix = "PROV" if text.upper().startswith("H") else "MBR"
    return f"{prefix}-{digest}"


def mask_frame(frame, columns: Iterable[str], role: Role | str, **kwargs):
    """Apply :func:`mask_identity` across a DataFrame's identity columns."""
    out = frame.copy()
    for column in columns:
        if column in out.columns:
            out[column] = out[column].map(lambda v: mask_identity(v, role, **kwargs))
    return out
