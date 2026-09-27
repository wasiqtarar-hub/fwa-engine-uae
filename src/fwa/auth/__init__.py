"""Authentication and role-based access control (build brief §13, manuscript §9.4)."""

from .rbac import (
    Role, Permission, ROLE_PERMISSIONS, ROLE_DESCRIPTIONS, PAGE_PERMISSIONS,
    AccessDenied, SeparationOfDutiesError, has_permission, require, visible_pages,
    mask_identity, mask_frame, assert_separation_of_duties,
)
from .service import AuthService, SessionState, SEED_ACCOUNTS, AuthenticationError

__all__ = [
    "Role", "Permission", "ROLE_PERMISSIONS", "ROLE_DESCRIPTIONS", "PAGE_PERMISSIONS",
    "AccessDenied", "SeparationOfDutiesError", "has_permission", "require", "visible_pages",
    "mask_identity", "mask_frame", "assert_separation_of_duties",
    "AuthService", "SessionState", "SEED_ACCOUNTS", "AuthenticationError",
]
