"""Users and access (internal name "User Management") — ADMIN only (build brief §13.3).

"Provide an admin **User Management** page: create/disable users, assign roles
and tenants, reset passwords, view the access log."

Plus the two §13.2 controls that have nowhere else to live: granting the
distinct MANUAL_OVERRIDE permission, and assigning an SIU case (which is what
makes unmasking permissible for that investigator, on that case, and nothing
else).

Plain role and permission names are the default view; the raw role and
permission values, event types and hashes stay reachable through "Show all
columns" and "Technical details". Every action still goes through the auth
service, which enforces the permission check and writes the audit entry.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from common import (
    auth_service, banner, dataframe, empty_state, friendly_failure, friendly_table,
    headline_cards, note, page_header, pipeline, require_page, session_banner,
)
from components.governance_pages import (
    actor_role_label, permission_label, permission_meaning, plain_text, plain_when, role_can_do,
)
from fwa.auth import Permission, ROLE_PERMISSIONS, Role
from fwa.presentation import label, vocabulary

MIN_REASON = 10

#: The individually grantable permissions, in the order they are offered.
GRANTABLE = [Permission.MANUAL_OVERRIDE.value, Permission.VIEW_UNMASKED_IDENTITY.value,
             Permission.EXPORT_EVIDENCE.value, Permission.CONFIRM_ENTITY_MERGE.value]


def _role(value) -> str:
    return label(value, "role")


def render() -> None:
    state = require_page("User Management", Permission.MANAGE_USERS)
    session_banner(state)
    page_header("User Management")
    banner()

    service = auth_service()
    users = service.db.list_users()

    tab_users, tab_roles, tab_permissions, tab_access = st.tabs(
        ["People", "Roles", "Individual permissions and case assignment", "Sign-in and access log"])

    with tab_users:
        with friendly_failure("the list of users"):
            _people(service, users, state)
    with tab_roles:
        with friendly_failure("the roles"):
            _roles(service, users, state)
    with tab_permissions:
        with friendly_failure("the permission and assignment forms"):
            _permissions(service, users, state)
    with tab_access:
        with friendly_failure("the access log"):
            _access_log(service, state)


# ---------------------------------------------------------------------------
# people
# ---------------------------------------------------------------------------


def _people(service, users, state) -> None:
    active = sum(1 for u in users if u.is_active)
    headline_cards([
        (f"{len(users)}", f"people have an account; {active} can sign in."),
        (f"{sum(1 for u in users if u.must_change_password)}",
         "must change their password at their next sign-in."),
        (f"{len({u.tenant_id for u in users})}",
         "organisations. Everyone only ever sees their own organisation's data."),
    ])
    frame = pd.DataFrame([u.to_row() for u in users])
    if frame.empty:
        empty_state("No users yet", "Create the first user below.")
    else:
        frame["role_words"] = [_role(r) for r in frame["role"]]
        frame["extra_words"] = [
            ", ".join(permission_label(p) for p in (x or [])) if isinstance(x, (list, tuple, set))
            else (", ".join(permission_label(p.strip()) for p in str(x).split(",") if p.strip()) or "None")
            for x in frame["extra_permissions"]]
        frame["extra_words"] = frame["extra_words"].replace("", "None")
        frame = frame.sort_values(["is_active", "display_name"], ascending=[False, True])
        friendly_table(
            frame, key="um_users",
            columns=["display_name", "username", "role_words", "tenant_id", "is_active",
                     "last_login", "must_change_password", "extra_words", "created_by",
                     "created_at", "failed_attempts", "role", "extra_permissions"],
            rename={"role_words": "Role", "extra_words": "Extra permissions"},
            dates=["last_login", "created_at"], height=330,
        )

    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.markdown("#### Create a user")
        with st.form("create_user"):
            username = st.text_input(
                "User name (used to sign in)",
                help="A short sign-in name, for example 'jsmith'. It cannot be changed later and "
                     "appears in the audit log beside everything this person does.")
            display = st.text_input(
                "Full name", help="The name shown on screen, for example 'Jane Smith'.")
            role = st.selectbox(
                "Role", [r.value for r in Role], format_func=_role,
                help="Decides which pages and actions this person gets. Pick the narrowest role "
                     "that fits the job; see the Roles tab for what each one can do.")
            tenant = st.text_input(
                "Organisation reference", value=state.tenant_id,
                help="The organisation this person belongs to. They will only ever see that "
                     "organisation's data. Leave it as it is unless you manage several.")
            password = st.text_input(
                "First password", type="password",
                help="A temporary password to give to the person. They must change it the first "
                     "time they sign in.")
            if st.form_submit_button("Create user", type="primary"):
                try:
                    service.create_user(actor=state, username=username.strip(),
                                        password=password, role=role, tenant_id=tenant.strip(),
                                        display_name=display.strip())
                except Exception as exc:  # noqa: BLE001 - the service explains its refusal
                    st.error(f"The user could not be created. {plain_text(exc)}")
                else:
                    st.success(f"{username} was created as {_role(role)}. They must change this "
                               "password at their first sign-in. Recorded in the audit log.")
                    st.rerun()
    with c2:
        st.markdown("#### Switch a user off or on, or reset a password")
        if not users:
            empty_state("No users yet", "Create a user first.")
            return
        names = {u.username: f"{u.display_name or u.username} ({u.username})" for u in users}
        target = st.selectbox(
            "User", [u.username for u in users], format_func=lambda n: names.get(n, n),
            key="um_target",
            help="The person whose access you are changing.")
        chosen = next((u for u in users if u.username == target), None)
        if chosen is not None:
            st.caption(f"{names[target]} has the role “{_role(chosen.role)}” and is "
                       f"currently {'able' if chosen.is_active else 'not able'} to sign in.")
        reason = st.text_input(
            "Reason (required, at least 10 characters)", key="admin_reason",
            help="Why you are making this change, for example 'Left the company on 30 Sep'. It is "
                 "stored in the audit log with your name.")
        a, b, c = st.columns(3)
        if a.button("Switch off", key="um_disable",
                    help="The person can no longer sign in. Their history stays in the audit log."):
            _guarded(lambda: service.set_active(actor=state, username=target, active=False,
                                                reason=reason), reason)
        if b.button("Switch on", key="um_enable",
                    help="Lets a switched-off person sign in again."):
            _guarded(lambda: service.set_active(actor=state, username=target, active=True,
                                                reason=reason), reason)
        new_password = st.text_input(
            "New password", type="password", key="reset_pw",
            help="A temporary password for the person. Only needed for 'Reset password'; they "
                 "must change it at their next sign-in.")
        if c.button("Reset password", key="um_reset",
                    help="Sets the new password above and makes the person change it when they "
                         "next sign in."):
            _guarded(lambda: service.reset_password(actor=state, username=target,
                                                    new_password=new_password, reason=reason),
                     reason)


# ---------------------------------------------------------------------------
# roles
# ---------------------------------------------------------------------------


def _roles(service, users, state) -> None:
    st.markdown("#### The eight roles")
    sees = vocabulary().role_sees
    dataframe(pd.DataFrame([
        {"Role": _role(r.value), "Sees": sees.get(r.value, ""), "Can do": role_can_do(r.value),
         "People with this role": sum(1 for u in users if u.role == r.value)}
        for r in Role
    ]), height=340)
    with st.expander("Every permission each role holds", expanded=False):
        pick = st.selectbox("Role", [r.value for r in Role], format_func=_role, key="um_role_perm",
                            help="Pick a role to list every permission it carries.")
        perms = sorted(ROLE_PERMISSIONS[Role(pick)], key=lambda p: permission_label(p.value))
        st.markdown("\n".join(f"- {permission_label(p.value)}" for p in perms) or "None")
        st.caption("Manually overriding a payment decision is never part of any role; it is granted "
                   "to named people only, on the next tab.")
        st.caption("Technical names: " + ", ".join(
            p.value for p in sorted(ROLE_PERMISSIONS[Role(pick)], key=lambda p: p.value)))

    st.markdown("#### Change a user's role")
    if not users:
        empty_state("No users yet", "Create a user first.")
        return
    names = {u.username: f"{u.display_name or u.username} ({u.username})" for u in users}
    with st.form("change_role"):
        target = st.selectbox("User", [u.username for u in users], key="role_target",
                              format_func=lambda n: names.get(n, n),
                              help="The person whose role you are changing.")
        role = st.selectbox("New role", [r.value for r in Role], key="role_new", format_func=_role,
                            help="The role they will have from their next page load. Their old "
                                 "role's access ends immediately.")
        reason = st.text_input("Reason (required, at least 10 characters)", key="role_reason",
                               help="Why the role is changing, for example 'Moved to the "
                                    "investigations team'. Stored in the audit log.")
        if st.form_submit_button("Change role"):
            _guarded(lambda: service.change_role(actor=state, username=target, role=role,
                                                 reason=reason), reason)


# ---------------------------------------------------------------------------
# permissions and assignment
# ---------------------------------------------------------------------------


def _permissions(service, users, state) -> None:
    st.markdown("#### Grant an individual permission")
    note("Manually overriding a payment decision is a separate permission that no role includes. "
         "It is granted to a named person for a stated reason, and every override they make is "
         "logged with the decision before and after.")
    if not users:
        empty_state("No users yet", "Create a user first.")
        return
    names = {u.username: f"{u.display_name or u.username} ({u.username})" for u in users}
    with st.form("grant_permission"):
        target = st.selectbox("User", [u.username for u in users], key="perm_target",
                              format_func=lambda n: names.get(n, n),
                              help="The person who will receive the permission.")
        permission = st.selectbox(
            "Permission", GRANTABLE, format_func=permission_label,
            help="What the person will be allowed to do in addition to their role. "
                 + " ".join(f"{permission_label(p)}: {permission_meaning(p)}" for p in GRANTABLE))
        reason = st.text_input("Reason (required, at least 10 characters)", key="perm_reason",
                               help="Why this person needs it, for example 'Covering the claims "
                                    "manager during leave'. Stored in the audit log.")
        if st.form_submit_button("Grant permission"):
            _guarded(lambda: service.grant_permission(actor=state, username=target,
                                                      permission=permission, reason=reason),
                     reason)

    st.markdown("#### Assign a case to an investigator")
    note("An investigator sees real patient and provider identities only on cases assigned to "
         "them. Assigning the case is what allows that, and each reveal still needs a reason that "
         "is written to the access log.")
    result = pipeline()
    queue = result.queue(state.tenant_id)
    siu_cases = queue.loc[queue["disposition"] == "SIU_LEAD", "case_id"].tolist() \
        if not queue.empty else []
    investigators = [u.username for u in users if u.role == Role.SIU_INVESTIGATOR.value] or ["siu"]
    cases = siu_cases or (list(queue["case_id"])[:50] if not queue.empty else [])
    if not cases:
        empty_state("No cases to assign", "This file produced no cases.")
        return
    st.caption(f"{len(siu_cases)} cases are suggested for the investigations team on this file"
               + ("." if siu_cases else "; the first 50 cases in the queue are offered instead."))
    with st.form("assign_case"):
        investigator = st.selectbox(
            "Investigator", investigators, format_func=lambda n: names.get(n, n),
            help="The investigator who will work the case and may see its real identities.")
        case_id = st.selectbox("Case", cases,
                               help="The case to assign. Cases suggested for referral to "
                                    "investigations are listed.")
        reason = st.text_input("Reason (required, at least 10 characters)", key="assign_reason",
                               help="Why this investigator should take this case. Stored in the "
                                    "audit log.")
        if st.form_submit_button("Assign case"):
            _guarded(lambda: service.assign_case(actor=state, username=investigator,
                                                 case_id=case_id, reason=reason), reason)


# ---------------------------------------------------------------------------
# access log
# ---------------------------------------------------------------------------


def _access_log(service, state) -> None:
    st.markdown("#### Sign-in and access log")
    note("Every sign-in, failed sign-in, refused page, identity reveal and change to a user is "
         "recorded here. Entries can only be added, never changed or deleted, and each links to "
         "the one before it so tampering shows.")
    frame = service.access_log()
    if frame is None or frame.empty:
        empty_state("No access events yet", "Nothing has been recorded in this session.")
        return
    frame = frame.sort_values("sequence", ascending=False) if "sequence" in frame else frame
    frame = frame.assign(
        when=[plain_when(t) for t in frame["event_time"]],
        event_words=[label(t, "audit") for t in frame["event_type"]],
        role_words=[actor_role_label(r) for r in frame["actor_role"]],
        reason_words=[plain_text(r) or "—" for r in frame["reason"]],
    )
    friendly_table(
        frame, key="um_access",
        columns=["when", "event_words", "actor", "role_words", "subject", "reason_words",
                 "event_time", "event_type", "actor_role", "reason"],
        rename={"when": "When", "event_words": "What happened", "role_words": "Their role",
                "reason_words": "Reason given"},
        max_default=6, height=440,
    )


def _guarded(action, reason: str) -> None:
    if len(reason.strip()) < MIN_REASON:
        st.error(f"Please give a reason of at least {MIN_REASON} characters. Every administrative "
                 "action is recorded with the name of the person who did it.")
        return
    try:
        action()
    except Exception as exc:  # noqa: BLE001 - the service explains its refusal
        st.error(f"That change was refused. {plain_text(exc)}")
    else:
        st.success("Done, and written to the audit log.")
        st.rerun()
