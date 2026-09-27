"""User Management — ADMIN only (build brief §13.3).

"Provide an admin **User Management** page: create/disable users, assign roles
and tenants, reset passwords, view the access log."

Plus the two §13.2 controls that have nowhere else to live: granting the
distinct MANUAL_OVERRIDE permission, and assigning an SIU case (which is what
makes unmasking permissible for that investigator, on that case, and nothing
else).
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from common import (
    auth_service, banner, dataframe, note, pipeline, require_page, session_banner, tiles,
)
from fwa.auth import Permission, ROLE_DESCRIPTIONS, Role


def render() -> None:
    state = require_page("User Management", Permission.MANAGE_USERS)
    session_banner(state)

    st.markdown("# User management")
    banner()

    service = auth_service()
    users = service.db.list_users()

    tab_users, tab_roles, tab_permissions, tab_access = st.tabs(
        ["Users", "Roles", "Permissions & assignment", "Access log"])

    with tab_users:
        tiles([
            ("Users", f"{len(users)}", ""),
            ("Active", f"{sum(1 for u in users if u.is_active)}", ""),
            ("Must change password", f"{sum(1 for u in users if u.must_change_password)}", ""),
            ("Tenants", f"{len({u.tenant_id for u in users})}",
             "every query is filtered by the session user's tenant"),
        ])
        dataframe(pd.DataFrame([u.to_row() for u in users]), height=330)

        c1, c2 = st.columns(2, gap="large")
        with c1:
            st.markdown("### Create a user")
            with st.form("create_user"):
                username = st.text_input("Username")
                display = st.text_input("Display name")
                role = st.selectbox("Role", [r.value for r in Role])
                tenant = st.text_input("Tenant", value=state.tenant_id)
                password = st.text_input("Initial password", type="password")
                if st.form_submit_button("Create", type="primary"):
                    try:
                        service.create_user(actor=state, username=username.strip(),
                                            password=password, role=role, tenant_id=tenant.strip(),
                                            display_name=display.strip())
                    except Exception as exc:
                        st.error(str(exc))
                    else:
                        st.success(f"{username} created with role {role}. They must change this "
                                   f"password at first sign-in.")
                        st.rerun()
        with c2:
            st.markdown("### Enable, disable or reset")
            target = st.selectbox("User", [u.username for u in users])
            reason = st.text_input("Reason (required)", key="admin_reason")
            a, b, c = st.columns(3)
            if a.button("Disable"):
                _guarded(lambda: service.set_active(actor=state, username=target, active=False,
                                                    reason=reason), reason)
            if b.button("Enable"):
                _guarded(lambda: service.set_active(actor=state, username=target, active=True,
                                                    reason=reason), reason)
            new_password = st.text_input("New password", type="password", key="reset_pw")
            if c.button("Reset password"):
                _guarded(lambda: service.reset_password(actor=state, username=target,
                                                        new_password=new_password, reason=reason),
                         reason)

    with tab_roles:
        st.markdown("### The eight roles")
        dataframe(pd.DataFrame([
            {"role": r.value, **ROLE_DESCRIPTIONS[r]} for r in Role
        ]), height=340)
        st.markdown("### Change a user's role")
        with st.form("change_role"):
            target = st.selectbox("User", [u.username for u in users], key="role_target")
            role = st.selectbox("New role", [r.value for r in Role], key="role_new")
            reason = st.text_input("Reason (required)", key="role_reason")
            if st.form_submit_button("Change role"):
                _guarded(lambda: service.change_role(actor=state, username=target, role=role,
                                                     reason=reason), reason)

    with tab_permissions:
        st.markdown("### Grant an individual permission")
        note(
            "Manual adjudication override (PAY-11) is a DISTINCT permission, not implied by any "
            "role. It is granted to a named user, for a stated reason, and every "
            "override that user then performs is logged with the before/after disposition."
        )
        with st.form("grant_permission"):
            target = st.selectbox("User", [u.username for u in users], key="perm_target")
            permission = st.selectbox(
                "Permission",
                [Permission.MANUAL_OVERRIDE.value, Permission.VIEW_UNMASKED_IDENTITY.value,
                 Permission.EXPORT_EVIDENCE.value, Permission.CONFIRM_ENTITY_MERGE.value],
            )
            reason = st.text_input("Reason (required)", key="perm_reason")
            if st.form_submit_button("Grant"):
                _guarded(lambda: service.grant_permission(actor=state, username=target,
                                                          permission=permission, reason=reason),
                         reason)

        st.markdown("### Assign an SIU case")
        note(
            "An SIU investigator may see unmasked identity ON ASSIGNED CASES ONLY. Assignment is "
            "what scopes that, and unmasking still requires an explicit reason that is written "
            "to the access log."
        )
        result = pipeline()
        queue = result.queue(state.tenant_id)
        siu_cases = queue.loc[queue["disposition"] == "SIU_LEAD", "case_id"].tolist()
        with st.form("assign_case"):
            investigator = st.selectbox(
                "Investigator",
                [u.username for u in users if u.role == Role.SIU_INVESTIGATOR.value] or ["siu"],
            )
            case_id = st.selectbox("Case", siu_cases or list(queue["case_id"])[:50])
            reason = st.text_input("Reason (required)", key="assign_reason")
            if st.form_submit_button("Assign"):
                _guarded(lambda: service.assign_case(actor=state, username=investigator,
                                                     case_id=case_id, reason=reason), reason)

    with tab_access:
        st.markdown("### Access log")
        frame = service.access_log()
        if frame is None or frame.empty:
            st.caption("No access events recorded in this session yet.")
        else:
            dataframe(frame[["event_time", "event_type", "actor", "actor_role", "subject",
                             "reason"]], height=440)
        note(
            "Every sign-in, failed sign-in, refusal, unmasking and user-administration action is "
            "here. The log is append-only and hash-chained; no role can delete from it."
        )


def _guarded(action, reason: str) -> None:
    if len(reason.strip()) < 10:
        st.error("A reason of at least 10 characters is required — every administrative action "
                 "is attributably recorded.")
        return
    try:
        action()
    except Exception as exc:
        st.error(str(exc))
    else:
        st.success("Done, and written to the audit log.")
        st.rerun()
