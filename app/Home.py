"""Entry point: login, role-aware navigation, and the forced password change.

    "Gate every page behind a single ``require_role(...)`` decorator/guard so no
     page can be reached by URL without authorisation, and **render only the
     navigation entries the role can use** rather than showing disabled ones."
                                                            — build brief §13.3

Navigation is built from :func:`fwa.auth.rbac.visible_pages`, so a page a role
cannot use is not registered with Streamlit's router at all for that session,
and :func:`app.common.require_page` re-checks on entry so the guard fails
closed either way.

On top of role-based access sits the **Simple / Advanced** switch. Simple view
(the default, remembered per user) lists the pages a reviewer needs day to
day; Advanced adds the analytical and governance pages. The switch only ever
*narrows* what a role may see — it can never add a page the role lacks.

Page titles are the plain titles in ``config/plain_language/pages.yaml``; the
internal names that access control is defined on are never shown.

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
    auth_service, banner, boot, boundary_note, config, mode_toggle, note, pages_for_mode,
    session, session_banner,
)
from dataset import active_dataset  # noqa: E402
from navigation import PAGE_MODULES, url_path  # noqa: E402
from fwa import __version__  # noqa: E402
from fwa.auth import AuthenticationError, Role  # noqa: E402
from fwa.presentation import label, page_info, vocabulary  # noqa: E402


# ---------------------------------------------------------------------------
# page 0 — login, the only unauthenticated view
# ---------------------------------------------------------------------------


def login_view() -> None:
    boot("Sign in", "🛡️")
    left, right = st.columns([1.05, 1], gap="large")

    with left:
        st.markdown("# UAE FWA Engine")
        st.markdown(
            "##### Finds claims worth a closer look for possible fraud, waste and abuse\n"
            f"Payment-integrity review tool, version {__version__}."
        )
        banner()
        boundary_note()
        st.markdown(
            "**What this tool does.** It runs 164 checks over a claim file, groups what they find "
            "into cases, estimates the money at risk carefully, and puts a queue of cases, most "
            "urgent first, in front of a human reviewer, with the reasons for each in plain words.\n\n"
            "**What it does not do.** It never decides that anyone committed fraud. Only a clear, "
            "dated rule (for example, no cover on the date of service) may deny a claim; no "
            "comparison, pattern-finding model, connection analysis or AI text can, at any setting."
        )

    with right:
        st.markdown("### Sign in")
        with st.form("login", border=True):
            username = st.text_input("Username", autocomplete="username",
                                     help="Your username. The demo accounts are listed below.")
            password = st.text_input("Password", type="password", autocomplete="current-password",
                                     help="Your password. Demo accounts must change it at first "
                                          "sign-in.")
            submitted = st.form_submit_button("Sign in", type="primary", width="stretch")
        if submitted:
            try:
                state = auth_service().login(username.strip(), password)
            except AuthenticationError as exc:
                st.error(str(exc))
            else:
                st.session_state["fwa_session"] = state
                st.rerun()

        with st.expander("Demo accounts for this tool", expanded=False):
            st.caption(
                "Created on first start and printed once to the console. Every account must "
                "change its password at first sign-in. This sign-in demonstrates the access model; "
                "it is **not hardened for real patient data**."
            )
            sees = vocabulary().role_sees
            st.table([
                {"Username": u, "Password": u, "Role": label(r.value, "role"),
                 "What they see": sees.get(r.value, "")}
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
        "This account was created with a demo password, so you need to choose your own before "
        "going any further."
    )
    with st.form("change_password", border=True):
        current = st.text_input("Current password", type="password",
                                help="The password you just signed in with.")
        new = st.text_input("New password", type="password",
                            help="At least 8 characters, and different from the current one.")
        confirm = st.text_input("Confirm new password", type="password",
                                help="Type the new password again to rule out a typing mistake.")
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

    allowed = state.pages()                       # role-based access first
    visible = pages_for_mode(allowed) or allowed  # then the Simple/Advanced filter
    pages = []
    for name in PAGE_MODULES:
        if name not in visible:
            continue                      # not registered at all for this session
        info = page_info(name)
        module = importlib.import_module(PAGE_MODULES[name])
        pages.append(st.Page(module.render, title=info["title"], icon=info.get("icon") or None,
                             url_path=url_path(name), default=(name == visible[0])))

    with st.sidebar:
        st.markdown("### 🛡️ UAE FWA Engine")
        st.caption(f"Version {__version__} · {label(state.role.value, 'role')}")
        mode_toggle()
        st.markdown("---")

    # ``expanded`` matters at thirteen pages: Streamlit otherwise collapses a
    # long navigation behind "View N more", which would hide pages the role
    # can reach.
    navigation = st.navigation(pages, position="sidebar", expanded=True)

    with st.sidebar:
        st.markdown("---")
        st.caption(f"**{state.display_name}**")
        st.caption(label(state.role.value, "role"))
        st.caption(f"Organisation {state.tenant_id}")
        if st.button("Sign out", width="stretch", help="End your session on this computer."):
            auth_service().logout(state)
            st.session_state.pop("fwa_session", None)
            st.rerun()
        spec = active_dataset()
        st.markdown(
            f"<div class='fwa-sidebar-dataset'>"
            f"<span>Data file</span><strong>{spec.name}</strong>"
            f"<span>{spec.rows:,} claims</span></div>",
            unsafe_allow_html=True,
        )

    navigation.run()


def main() -> None:
    """Always route through ``st.navigation``.

    Calling it unconditionally is what disables Streamlit's automatic discovery
    of ``app/pages/``. Without that, every page file in the directory would
    appear in the sidebar — including to an unauthenticated visitor.
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
