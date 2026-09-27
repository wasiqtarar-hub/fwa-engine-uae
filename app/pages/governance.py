"""Governance — parameters, audit log, kill switch, rollback, ceilings, latency.

Build brief §14, item 8 (Roles 3 & 4, §9.4, §9.5, §9.7–9.9): "parameter registry
with owner, rationale, source, approval, expected alert volume and expiry;
append-only audit log; per-rule and per-model **kill switch** (routing signals to
``MONITOR_ONLY``) and version **rollback**; alert-volume ceiling; latency
monitoring against the Table 3.4 ceilings; model-health report."

Everything a §3.10 non-functional requirement promises is either demonstrable on
this page or it is not implemented.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from common import (
    banner, bar, dataframe, empty_state, mono, note, pipeline, require_page, session_banner,
    status_chip, tiles,
)
from fwa.auth import Permission
from fwa.enums import RuleStatus


def render() -> None:
    state = require_page("Governance", Permission.VIEW_GOVERNANCE)
    session_banner(state)

    st.markdown("# Governance")
    banner()

    result = pipeline()
    config = result.config

    tabs = st.tabs([
        "Parameter registry", "Audit log", "Kill switch & rollback",
        "Alert ceilings", "Latency", "Model health",
    ])

    # ======================================================================
    with tabs[0]:
        st.markdown("## Parameter registry")
        st.markdown(
            "Every threshold in this system lives here, effective-dated, with an owner, a "
            "rationale, a source, an approver and a review date. Nothing in `src/` contains a "
            "bare numeric threshold — a governance test scans for them."
        )
        rows = pd.DataFrame(config.parameters.governance_report())
        expired = int(rows["expired"].sum())
        unapproved = int(rows["unapproved"].sum())
        tiles([
            ("Registered parameters", f"{len(rows):,}", ""),
            ("Past review date", f"{expired}", "surfaced here rather than rotting unnoticed"),
            ("Unapproved", f"{unapproved}", "a value with no approver is a governance defect"),
            ("Registry fingerprint", config.fingerprint(),
             "stamped onto every signal for reproducibility"),
        ])
        search = st.text_input("Filter", placeholder="e.g. readmit, peer, cusum, ai_")
        view = rows if not search else rows[
            rows["key"].str.contains(search, case=False)
            | rows["rationale"].str.contains(search, case=False)
        ]
        dataframe(view[["key", "value", "owner", "approved_by", "expected_alert_volume",
                        "review_date", "expired"]], height=380)
        chosen = st.selectbox("Inspect a parameter", list(rows["key"]))
        record = config.parameters.record(chosen)
        st.markdown(
            f"<div class='fwa-card'><h4>{record.key} = <code>{record.value}</code></h4>"
            f"<p><strong>Owner:</strong> {record.owner} · "
            f"<strong>Approved by:</strong> {record.approved_by} · "
            f"<strong>Effective:</strong> {record.valid_from} → {record.valid_to}</p>"
            f"<p><strong>Rationale.</strong> {record.rationale}</p>"
            f"<p class='fwa-sub'><strong>Source.</strong> {record.source}</p></div>",
            unsafe_allow_html=True,
        )
        note(
            "Every value in this registry is a CONFIGURATION CHOICE made for this build. "
            "None of them is a universal medical or actuarial fact, and the design "
            "is explicit on that point. Where the honest answer to 'where did this come from?' "
            "is 'chosen for this artefact', the source field says exactly that."
        )

    # ======================================================================
    with tabs[1]:
        st.markdown("## Audit log")
        ok, message = result.audit.verify_chain()
        st.markdown(
            (status_chip("PASS") if ok else status_chip("FAIL")) + f" {message}",
            unsafe_allow_html=True,
        )
        note(
            "Append-only and hash-chained: each entry carries the hash of the previous one, so a "
            "removed or altered row is detectable. No role has a delete permission over it, "
            "because the log has no delete operation to permit."
        )
        if not state.can(Permission.VIEW_AUDIT_LOG):
            st.info(
                f"The full audit log is visible to AUDITOR and ADMIN. The {state.role.value} "
                f"role sees the chain verification above but not the entries."
            )
        else:
            frame = result.audit.to_dataframe(state.tenant_id)
            if frame.empty:
                empty_state("No audit entries yet", "Nothing has been recorded in this session.")
            else:
                types = st.multiselect("Event type", sorted(frame["event_type"].unique()),
                                       default=sorted(frame["event_type"].unique()))
                view = frame[frame["event_type"].isin(types)]
                dataframe(view[["sequence", "event_time", "event_type", "actor", "actor_role",
                                "subject", "reason"]], height=400)
                counts = frame["event_type"].value_counts().reset_index()
                counts.columns = ["event_type", "events"]
                bar(counts, "event_type", "events", title="Audit events by type",
                    horizontal=True, height=320)

    # ======================================================================
    with tabs[2]:
        st.markdown("## Kill switch and rollback")
        st.markdown(
            "Both are **configuration changes, not code deployments** — that is the "
            "requirement. A kill-switched control does not vanish: its signals are routed to "
            "`MONITOR_ONLY`, so the loss of coverage is visible in the queue rather than silent."
        )
        registry = result.registry
        killed = [c for c in registry.all() if c.kill_switched]
        tiles([
            ("Controls kill-switched", f"{len(killed)}", ""),
            ("Controls active", f"{len(registry.by_status(RuleStatus.ACTIVE))}", ""),
            ("Controls in shadow", f"{len(registry.by_status(RuleStatus.SHADOW))}", ""),
            ("Controls retired", f"{len(registry.by_status(RuleStatus.RETIRED))}", ""),
        ])
        if killed:
            dataframe(pd.DataFrame([
                {"rule_id": c.rule_id, "reason": c.kill_switch_reason} for c in killed]))

        if not state.can(Permission.KILL_SWITCH):
            note(f"The kill switch is ADMIN-only. The {state.role.value} role may see its state "
                 f"but not operate it.")
        else:
            st.markdown("### Operate")
            executable = [c.rule_id for c in registry.executable()]
            rule_id = st.selectbox("Control", executable)
            reason = st.text_input("Reason (required)")
            c1, c2, c3 = st.columns(3)
            if c1.button("Engage kill switch", type="primary"):
                if len(reason.strip()) < 10:
                    st.error("A reason of at least 10 characters is required.")
                else:
                    registry.kill_switch(rule_id, actor=state.username, reason=reason)
                    result.audit.record(
                        "KILL_SWITCH_ENGAGED", actor=state.username, actor_role=state.role.value,
                        tenant_id=state.tenant_id, subject=rule_id, reason=reason)
                    st.success(f"{rule_id} kill-switched. Its signals now route to MONITOR_ONLY "
                               f"on the next run. Logged with your username, the time and the "
                               f"reason.")
            if c2.button("Release kill switch"):
                if len(reason.strip()) < 10:
                    st.error("A reason of at least 10 characters is required.")
                else:
                    registry.restore(rule_id, actor=state.username, reason=reason)
                    result.audit.record(
                        "KILL_SWITCH_RELEASED", actor=state.username,
                        actor_role=state.role.value, tenant_id=state.tenant_id,
                        subject=rule_id, reason=reason)
                    st.success(f"{rule_id} restored.")
            if c3.button("Roll back to the previous version"):
                try:
                    rolled = registry.rollback(rule_id, actor=state.username)
                except Exception as exc:
                    st.error(str(exc))
                else:
                    result.audit.record(
                        "RULE_ROLLED_BACK", actor=state.username, actor_role=state.role.value,
                        tenant_id=state.tenant_id, subject=rule_id,
                        reason=f"Rolled back to v{rolled.version}.")
                    st.success(f"{rule_id} rolled back to v{rolled.version}, re-entering SHADOW. "
                               f"A rollback never goes straight to active.")

    # ======================================================================
    with tabs[3]:
        st.markdown("## Alert-volume ceiling")
        ceiling = int(config.get("alert_volume_ceiling_per_rule_per_run"))
        breaches = result.evaluation.ceiling_breaches
        tiles([
            ("Ceiling per rule per run", f"{ceiling:,}", "cfg.alert_volume_ceiling_per_rule_per_run"),
            ("Breaches this run", f"{len(breaches)}", ", ".join(breaches) or "none"),
        ])
        note(
            "A control that exceeds the ceiling has its signals routed to MONITOR_ONLY and "
            "raises a governance event. The signals are NOT discarded: silently dropping a "
            "rule's output would hide a coverage loss."
        )
        run = result.evaluation.to_frame()
        fired = run[run["signal_count"] > 0].sort_values("signal_count", ascending=False)
        if not fired.empty:
            bar(fired.head(20), "rule_id", "signal_count", title="Signals per control",
                horizontal=True, height=440)

        st.markdown("### Expected against observed volume")
        rows = []
        for control in result.registry.executable():
            observed = int(run.loc[run["rule_id"] == control.rule_id, "signal_count"].sum()) \
                if not run.empty else 0
            expected = None
            for key, value in (control.parameters or {}).items():
                if isinstance(value, str) and value.startswith("cfg."):
                    try:
                        expected = config.parameters.record(value[4:]).expected_alert_volume
                    except Exception:
                        expected = None
                    if expected is not None:
                        break
            rows.append({"rule_id": control.rule_id, "observed": observed,
                         "governed_expectation": expected,
                         "divergence": None if expected is None else observed - expected})
        dataframe(pd.DataFrame(rows))
        note(
            "`expected_alert_volume` is a GOVERNANCE COMMITMENT made by the parameter's owner, "
            "not a measured result. Divergence between it and the observed volume is exactly "
            "what the hard-edit release gate's 'expected-versus-observed volume comparison' "
            "asks a team to look at."
        )

    # ======================================================================
    with tabs[4]:
        st.markdown("## Latency against the stage ceilings")
        dataframe(result.evaluation.latency_report(config))
        note(
            f"Observed figures are BATCH times for this stage across {len(result.claims):,} "
            "claims, not per-claim online latency. A real PREPAY_SYNC deployment is measured per "
            "claim against a 500 ms ceiling; this artefact runs as a batch and reports it as a "
            "batch rather than dividing by the row count and calling the result a latency."
        )
        timing = pd.DataFrame([{"stage": k, "seconds": round(v, 2)}
                               for k, v in result.timings.items()])
        bar(timing, "stage", "seconds", title="Pipeline stage timings", horizontal=True, height=340)

    # ======================================================================
    with tabs[5]:
        st.markdown("## Model-health report")
        layer = result.models
        if layer is None or not layer.drift:
            empty_state("No model health to report", "The model layer did not run.")
        else:
            from fwa.models.drift import DriftMonitor

            for name, results in layer.drift.items():
                st.markdown(f"### {name}")
                dataframe(pd.DataFrame([d.to_row() for d in results])[
                    ["monitor", "value", "warn_threshold", "breach_threshold", "status"]])
                st.markdown(f"**{DriftMonitor.recommendation(results)}**")
            note(
                "A breach RECOMMENDS recalibration or the kill switch; it does not apply either. "
                "Deactivating a model is a governed action with a named owner, not "
                "something a monitor does on its own."
            )
            if layer.supervised is not None:
                st.markdown("### Supervised gate")
                st.markdown(status_chip(layer.supervised.status), unsafe_allow_html=True)
                st.markdown(layer.supervised.summary)
