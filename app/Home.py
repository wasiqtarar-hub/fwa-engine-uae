"""Entry point: login, role-aware navigation, and the forced password change.

    "Gate every page behind a single ``require_role(...)`` decorator/guard so no
     page can be reached by URL without authorisation, and **render only the
     navigation entries the role can use** rather than showing disabled ones."
                                                            — build brief §13.3

Navigation is built from :func:`fwa.auth.rbac.visible_pages`, so the sidebar a
CLAIMS_REVIEWER sees contains four entries and the sidebar an ADMIN sees
contains all thirteen. A page that is not in a role's list is not registered with
Streamlit's router at all for that session, and :func:`app.common.require_page`
re-checks on entry so the guard fails closed either way.

Run with::

    streamlit run app/Home.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from common import (  # noqa: E402
    auth_service, banner, boot, boundary_note, config, empty_state, note, session,
    session_banner,
)
from dataset import active_dataset  # noqa: E402
from fwa import SAFETY_BOUNDARY_STATEMENT, __version__  # noqa: E402
from fwa.auth import AuthenticationError, PAGE_PERMISSIONS, ROLE_DESCRIPTIONS, Role  # noqa: E402

PAGE_MODULES = {
    "Overview": ("pages.overview", "📊"),
    "Data": ("pages.data", "📁"),
    "Review Queue": ("pages.review_queue", "📋"),
    "Case Evidence": ("pages.case_evidence", "🔍"),
    "Provider Analytics": ("pages.provider_analytics", "🏥"),
    "Network": ("pages.network", "🕸️"),
    "Models": ("pages.models", "🧪"),
    "Rule Registry": ("pages.rule_registry", "📐"),
    "Governance": ("pages.governance", "⚖️"),
    "Parameters & Models": ("pages.parameters", "🎚️"),
    "Validation Report": ("pages.validation_report", "📄"),
    "User Management": ("pages.user_management", "👤"),
    "Help": ("pages.help", "❓"),
}


# ---------------------------------------------------------------------------
# page 0 — login, the only unauthenticated view
# ---------------------------------------------------------------------------


def login_view() -> None:
    boot("Sign in", "🛡️")
    left, right = st.columns([1.05, 1], gap="large")

    with left:
        st.markdown("# UAE FWA Engine")
        st.markdown(
            "##### Governed fraud, waste and abuse detection for medical claims\n"
            f"Payment-integrity detection engine, v{__version__}."
        )
        banner()
        boundary_note()
        st.markdown(
            "**What this tool does.** It runs a catalogue of 164 governed controls over a claim "
            "file, correlates what they find into cases, prices the exposure conservatively, and "
            "puts a prioritised, evidence-backed queue in front of a human reviewer.\n\n"
            "**What it does not do.** It does not decide that anyone committed fraud. Only a "
            "hard, objective, effective-dated condition may deny a claim; no statistical score, "
            "model score, network metric or AI output can, at any threshold, under any "
            "configuration."
        )

    with right:
        st.markdown("### Sign in")
        with st.form("login", border=True):
            username = st.text_input("Username", autocomplete="username")
            password = st.text_input("Password", type="password", autocomplete="current-password")
            submitted = st.form_submit_button("Sign in", type="primary", width="stretch")
        if submitted:
            try:
                state = auth_service().login(username.strip(), password)
            except AuthenticationError as exc:
                st.error(str(exc))
            else:
                st.session_state["fwa_session"] = state
                st.rerun()

        with st.expander("Demo accounts for this artefact", expanded=False):
            st.caption(
                "Seeded on first run and printed once to the console. Every account is forced to "
                "change its password at first sign-in. This authentication layer demonstrates the "
                "access-control model and is **not hardened for real PHI**."
            )
            st.table([
                {"username": u, "password": u, "role": r.value,
                 "sees": ROLE_DESCRIPTIONS[r]["sees"]}
                for u, r in [
                    ("admin", Role.ADMIN), ("policy", Role.POLICY_OWNER),
                    ("reviewer", Role.CLAIMS_REVIEWER), ("clinical", Role.CLINICAL_REVIEWER),
                    ("siu", Role.SIU_INVESTIGATOR), ("analyst", Role.ANALYST),
                    ("qa", Role.QA), ("auditor", Role.AUDITOR),
                ]
            ])


# ---------------------------------------------------------------------------
# forced password change (§13.3)
# ---------------------------------------------------------------------------


def force_password_change(state) -> None:
    boot("Change your password", "🔐")
    session_banner(state)
    st.markdown("## Choose a new password")
    note(
        "This account was seeded with a demo password and is required to change it before "
        "going any further (forced password change on first login)."
    )
    with st.form("change_password", border=True):
        current = st.text_input("Current password", type="password")
        new = st.text_input("New password", type="password",
                            help="At least 8 characters, and different from the current one.")
        confirm = st.text_input("Confirm new password", type="password")
        submitted = st.form_submit_button("Change password", type="primary")
    if submitted:
        if new != confirm:
            st.error("The two new passwords do not match.")
            return
        try:
            auth_service().change_own_password(state, current, new)
        except Exception as exc:
            st.error(str(exc))
        else:
            st.success("Password changed. Loading your workspace…")
            st.rerun()


# ---------------------------------------------------------------------------
# authenticated shell
# ---------------------------------------------------------------------------


def build_navigation(state) -> None:
    import importlib

    allowed = state.pages()
    pages = []
    for name in PAGE_MODULES:
        if name not in allowed:
            continue                      # not registered at all for this role
        module_name, icon = PAGE_MODULES[name]
        module = importlib.import_module(module_name)
        pages.append(st.Page(module.render, title=name, icon=icon,
                             url_path=name.lower().replace(" ", "-"),
                             default=(name == allowed[0])))

    with st.sidebar:
        st.markdown("### 🛡️ UAE FWA Engine")
        st.caption(f"v{__version__} · {state.role.value}")
        st.markdown("---")

    # ``expanded`` matters at thirteen pages. Streamlit collapses a long
    # navigation behind "View N more", and a role-gated sidebar whose point is
    # that it shows exactly what the role can reach undoes itself the moment
    # three of those entries are hidden behind a disclosure.
    navigation = st.navigation(pages, position="sidebar", expanded=True)

    with st.sidebar:
        st.markdown("---")
        st.caption(f"**{state.display_name}**")
        st.caption(f"{ROLE_DESCRIPTIONS[state.role]['job_title']}")
        st.caption(f"Tenant `{state.tenant_id}`")
        if st.button("Sign out", width="stretch"):
            auth_service().logout(state)
            st.session_state.pop("fwa_session", None)
            st.rerun()
        spec = active_dataset()
        st.markdown(
            f"<div class='fwa-sidebar-dataset'>"
            f"<span>Dataset</span><strong>{spec.name}</strong>"
            f"<span>{spec.rows:,} claims · {spec.short_sha}</span></div>",
            unsafe_allow_html=True,
        )

    navigation.run()


def main() -> None:
    """Always route through ``st.navigation``.

    Calling it unconditionally is what disables Streamlit's automatic discovery
    of ``app/pages/``. Without that, every page file in the directory would
    appear in the sidebar — including to an unauthenticated visitor — which
    would make the §13.3 requirement ("render only the navigation entries the
    role can use") false the moment anyone looked at the sidebar.
    """
    state = session()
    if state is None:
        st.navigation([st.Page(login_view, title="Sign in", url_path="sign-in", default=True)],
                      position="hidden").run()
        return
    boot("Workspace", "🛡️")
    if state.must_change_password:
        st.navigation(
            [st.Page(lambda: force_password_change(state), title="Change your password",
                     url_path="change-password", default=True)],
            position="hidden",
        ).run()
        return
    build_navigation(state)


main()
