"""The page registry, shared by the app shell and the UI tests.

Keyed by the INTERNAL page name that role-based access control is defined on
(:data:`fwa.auth.PAGE_PERMISSIONS`). Those names are never shown: the sidebar
uses the plain titles in ``config/plain_language/pages.yaml``. The URL path is
derived from the internal name so existing bookmarks keep working.
"""

from __future__ import annotations

#: internal page name -> page module
PAGE_MODULES: dict[str, str] = {
    "Overview": "pages.overview",
    "Data": "pages.data",
    "Review Queue": "pages.review_queue",
    "Case Evidence": "pages.case_evidence",
    "Provider Analytics": "pages.provider_analytics",
    "Network": "pages.network",
    "Models": "pages.models",
    "Rule Registry": "pages.rule_registry",
    "Governance": "pages.governance",
    "Parameters & Models": "pages.parameters",
    "Validation Report": "pages.validation_report",
    "User Management": "pages.user_management",
    "Help": "pages.help",
}


def url_path(name: str) -> str:
    return name.lower().replace(" ", "-")
