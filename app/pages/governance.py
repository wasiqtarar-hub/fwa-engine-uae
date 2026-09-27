"""Governance — ownership of settings, audit log, kill switch, rollback, ceilings, latency.

Build brief §14, item 8 (Roles 3 & 4, §9.4, §9.5, §9.7–9.9): "parameter registry
with owner, rationale, source, approval, expected alert volume and expiry;
append-only audit log; per-rule and per-model **kill switch** (routing signals to
``MONITOR_ONLY``) and version **rollback**; alert-volume ceiling; latency
monitoring against the Table 3.4 ceilings; model-health report."

Everything a §3.10 non-functional requirement promises is either demonstrable on
this page or it is not implemented. Plain wording is the default view; raw keys,
event types, hashes and monitor notes are in "Technical details" expanders and
"Show all columns". Every action keeps its permission check, its required reason
and its audit record.
"""

from __future__ import annotations

import html

import pandas as pd
import streamlit as st

from common import (
    bar, banner, dataframe, empty_state, friendly_failure, friendly_table, headline_cards,
    help_text, note, page_header, pipeline, require_page, session_banner, status_badge,
    synthetic_banner,
)
from components.governance_pages import (
    actor_role_label, check_title, format_value, owner_label, plain_seconds, plain_text,
    plain_when,
)
from fwa.auth import Permission
from fwa.enums import RuleStatus
from fwa.presentation import (
    count_phrase, label, model_label, parameter_text, plain_date, plain_number, standard_text,
    status,
)

MIN_REASON = 10


def render() -> None:
    state = require_page("Governance", Permission.VIEW_GOVERNANCE)
    session_banner(state)
    page_header("Governance")
    banner()
    synthetic_banner()

    result = pipeline()
    config = result.config

    tabs = st.tabs([
        "Who owns each setting", "Audit log", "Switch a check off or roll it back",
        "Flag limits and expected volumes", "Speed of each stage", "Model health",
    ])
    with tabs[0]:
        with friendly_failure("who owns each setting"):
            _ownership(config)
    with tabs[1]:
        with friendly_failure("the audit log"):
            _audit(result, state)
    with tabs[2]:
        with friendly_failure("the switch-off and roll-back controls"):
            _kill_switch(result, state)
    with tabs[3]:
        with friendly_failure("the flag limits"):
            _ceilings(result, config)
    with tabs[4]:
        with friendly_failure("the stage timings"):
            _latency(result, config)
    with tabs[5]:
        with friendly_failure("the model health report"):
            _model_health(result)


# ---------------------------------------------------------------------------
# 1. ownership
# ---------------------------------------------------------------------------


def _ownership(config) -> None:
    st.markdown("### Who owns each setting")
    st.markdown(
        "Every number the checks use is a governed setting: it has a **named owner** who answers "
        "for it, someone who **approved** it, the **dates** it applies between and a written "
        "**reason**. No threshold is hidden in the software; an automated test looks for any that "
        "are. To read what each setting does and what changing it would do, open "
        "**Settings and thresholds**."
    )
    rows = pd.DataFrame(config.parameters.governance_report())
    if rows.empty:
        empty_state("No governed settings found", "The settings registry is empty.")
        return
    rows["setting_words"] = [parameter_text(k)["label"] for k in rows["key"]]
    rows["value_words"] = [format_value(k, v) for k, v in zip(rows["key"], rows["value"])]
    rows["owner_words"] = [owner_label(o) for o in rows["owner"]]
    rows["approver_words"] = [owner_label(a) for a in rows["approved_by"]]
    expired = int(rows["expired"].sum())
    unapproved = int(rows["unapproved"].sum())
    headline_cards([
        (f"{len(rows):,}", "settings are governed, each with an owner and an approver."),
        (f"{rows['owner'].nunique()}", "teams own them."),
        (f"{expired}", "are past their review date. They still apply; this list makes sure they "
                       "are not forgotten."),
        (f"{unapproved}", "have no approver. A value nobody signed off is a governance gap."),
    ])

    by_owner = rows.groupby("owner_words").size().reset_index(name="Settings") \
        .sort_values("Settings", ascending=False)
    top = by_owner.iloc[0]
    bar(by_owner, "owner_words", "Settings", horizontal=True, height=340,
        title=f"{top['owner_words']} owns the most settings ({top['Settings']})",
        x_label="Owner", y_label="Number of settings",
        how_to_read="Each bar is a team; its length is how many settings that team answers for.")

    c1, c2 = st.columns([2, 1])
    with c1:
        search = st.text_input(
            "Search settings", key="gov_param_search", placeholder="e.g. readmission, peer, days",
            help="Type a word to narrow the list to settings whose name, owner or reason mentions "
                 "it. Leave it empty to see every setting.",
        )
    with c2:
        owners = sorted(rows["owner"].unique(), key=owner_label)
        chosen = st.multiselect(
            "Owned by", owners, default=[], format_func=owner_label, key="gov_owner",
            help="Show only the settings owned by these teams. Leave it empty for all teams.",
        )
    view = rows
    if search.strip():
        q = search.strip().lower()
        hay = (view["setting_words"] + " " + view["owner_words"] + " " + view["key"] + " "
               + view["rationale"].astype(str)).str.lower()
        view = view[hay.str.contains(q, regex=False)]
    if chosen:
        view = view[view["owner"].isin(chosen)]
    view = view.sort_values(["expired", "unapproved", "owner_words", "setting_words"],
                            ascending=[False, False, True, True])
    friendly_table(
        view, key="gov_params",
        columns=["setting_words", "value_words", "owner_words", "approver_words", "valid_from",
                 "valid_to", "review_date", "expired", "expected_alert_volume", "unapproved",
                 "key", "value", "rationale", "source"],
        dates=["valid_from", "valid_to", "review_date"], height=380,
        empty_title="No setting matches", empty_body="Clear the search or the owner filter.",
    )

    keys = list(view["key"]) or list(rows["key"])
    labels = rows.set_index("key")["setting_words"]
    pick = st.selectbox(
        "Look at one setting's ownership", keys, format_func=lambda k: labels.get(k, k),
        key="gov_param_pick",
        help="Pick a setting to see who owns it, who approved it and when it applies.",
    )
    record = config.parameters.record(pick)
    esc = html.escape
    st.markdown(
        f"<div class='fwa-card'><h4>{esc(parameter_text(pick)['label'])}: "
        f"{esc(format_value(pick, record.value))}</h4>"
        f"<p><strong>Owner:</strong> {esc(owner_label(record.owner))} · "
        f"<strong>Approved by:</strong> {esc(owner_label(record.approved_by))} · "
        f"<strong>In force:</strong> {esc(plain_date(record.valid_from))} to "
        f"{esc(plain_date(record.valid_to))}</p>"
        f"<p class='fwa-sub'>{esc(parameter_text(pick)['meaning'])}</p></div>",
        unsafe_allow_html=True,
    )
    with st.expander("Technical details: key, reason and source as recorded", expanded=False):
        st.markdown(f"`{record.key}` = `{record.value}`")
        st.markdown(f"**Reason given.** {record.rationale}")
        st.markdown(f"**Source.** {record.source}")
    note("Every value is a configuration choice made for this build, not a universal medical or "
         "actuarial fact. Where the honest answer to 'where did this come from?' is 'chosen for "
         "this tool', the recorded source says exactly that.")


# ---------------------------------------------------------------------------
# 2. audit log
# ---------------------------------------------------------------------------


def _audit(result, state) -> None:
    st.markdown("### Audit log")
    ok, message = result.audit.verify_chain()
    if ok:
        st.markdown(status_badge("PASS") + " **The log is intact.** Every entry still links to the "
                    "one before it, so nothing has been removed or changed.",
                    unsafe_allow_html=True)
    else:
        st.markdown(status_badge("FAIL") + " **The log has been altered.** An entry no longer "
                    "links to the one before it. Report this to an administrator.",
                    unsafe_allow_html=True)
    st.caption(plain_text(message))
    note("Entries can only be added, never edited or deleted: each carries a fingerprint of the one "
         "before it, so a removed or altered entry shows up. No role can delete from it, because "
         "the log has no delete operation at all.")
    if not state.can(Permission.VIEW_AUDIT_LOG):
        st.info(f"Only auditors and administrators can read the entries. As "
                f"{label(state.role.value, 'role')} you can see the check above but not the entries.")
        return
    frame = result.audit.to_dataframe(state.tenant_id)
    if frame.empty:
        empty_state("No audit entries yet", "Nothing has been recorded in this session.")
        return
    types = sorted(frame["event_type"].unique(), key=lambda t: label(t, "audit"))
    chosen = st.multiselect(
        "Kind of event", types, default=[], format_func=lambda t: label(t, "audit"),
        key="gov_audit_types",
        help="Show only these kinds of event, for example sign-ins or checks switched off. Leave "
             "it empty to show everything.",
    )
    view = frame[frame["event_type"].isin(chosen)] if chosen else frame
    view = view.sort_values("sequence", ascending=False)
    view = view.assign(
        when=[plain_when(t) for t in view["event_time"]],
        event_words=[label(t, "audit") for t in view["event_type"]],
        role_words=[actor_role_label(r) for r in view["actor_role"]],
        reason_words=[plain_text(r) or "—" for r in view["reason"]],
    )
    friendly_table(
        view, key="gov_audit",
        columns=["when", "event_words", "actor", "role_words", "subject", "reason_words",
                 "sequence", "event_time", "event_type", "actor_role", "tenant_id", "reason",
                 "before", "after", "prev_hash", "entry_hash"],
        rename={"when": "When", "event_words": "What happened", "role_words": "Their role",
                "reason_words": "Reason given"},
        max_default=6, height=380,
    )
    counts = frame.assign(event=[label(t, "audit") for t in frame["event_type"]]) \
        .groupby("event").size().reset_index(name="Entries").sort_values("Entries", ascending=False)
    bar(counts, "event", "Entries", horizontal=True, height=320,
        title=f"Most entries are “{counts.iloc[0]['event']}” ({counts.iloc[0]['Entries']})",
        x_label="Kind of event", y_label="Number of entries",
        how_to_read="Each bar is a kind of event; its length is how many times it was recorded.")


# ---------------------------------------------------------------------------
# 3. kill switch and rollback
# ---------------------------------------------------------------------------


def _kill_switch(result, state) -> None:
    st.markdown("### Switch a check off, back on, or roll it back")
    st.markdown(
        "**Switching a check off** (the kill switch) is for a check that is misbehaving, for "
        "example flooding the queue. Its flags are not deleted: they are marked watch-only, so "
        "the gap in coverage stays visible. **Rolling back** returns a check to its previous "
        "version, which always restarts in watch-only mode, never straight into use. Both are "
        "settings changes, not software changes, and each needs a written reason that is stored "
        "in the audit log with your name and the time."
    )
    registry = result.registry
    killed = [c for c in registry.all() if c.kill_switched]
    headline_cards([
        (f"{len(killed)}", "checks are switched off."),
        (f"{len(registry.by_status(RuleStatus.ACTIVE))}", "checks are approved and in use."),
        (f"{len(registry.by_status(RuleStatus.SHADOW))}", "checks are in watch-only mode."),
        (f"{len(registry.by_status(RuleStatus.RETIRED))}", "checks are retired."),
    ])
    if killed:
        dataframe(pd.DataFrame([
            {"Check": check_title(c.rule_id, c.name), "Switched off by, and why": c.kill_switch_reason}
            for c in killed]))

    if not state.can(Permission.KILL_SWITCH):
        note(f"Only an administrator can switch checks off or roll them back. As "
             f"{label(state.role.value, 'role')} you can see their state but not change it.")
        return

    run = result.evaluation.to_frame()
    ran = set(run.loc[~run["data_support"].astype(str).str.startswith("NOT_EXECUTABLE"), "rule_id"]) \
        if not run.empty else set()
    candidates = sorted({c.rule_id for c in registry.executable()} | ran)
    if not candidates:
        empty_state("No check ran on this file", "There is nothing to switch off.")
        return
    flags = dict(zip(run["rule_id"], run["signal_count"])) if not run.empty else {}
    titles = {r: check_title(r, registry.get(r).name) for r in candidates}
    rule_id = st.selectbox(
        "Check", candidates, format_func=lambda r: titles.get(r, r), key="gov_kill_pick",
        help="Pick the check to switch off, switch back on or roll back. Only checks that can run "
             "on this file are listed.",
    )
    control = registry.get(rule_id)
    n = int(flags.get(rule_id, 0))
    if control.kill_switched:
        st.markdown(f"This check is **switched off**. Switching it back on would let its "
                    f"{count_phrase(n, 'flag')} reach the queue normally again from the next run.")
    else:
        st.markdown(f"If you switch it off, its {count_phrase(n, 'flag')} on this run would be kept "
                    "but marked watch-only from the next run, so no case would be driven by it.")
    history = registry.history(rule_id)
    st.caption(f"Previous versions available to roll back to: {len(history)}."
               + ("" if history else " Roll-back is not possible until a second version exists."))
    reason = st.text_input(
        "Reason (required, at least 10 characters)", key="gov_kill_reason",
        help="Say why you are making this change, for example 'Flooding the queue after a coding "
             "update; switching off until fixed'. It is stored in the audit log with your name.",
    )
    c1, c2, c3 = st.columns(3)
    if c1.button("Switch this check off", type="primary", key="gov_kill",
                 help="Engage the kill switch. Its flags are kept but marked watch-only."):
        if len(reason.strip()) < MIN_REASON:
            st.error(f"Please give a reason of at least {MIN_REASON} characters.")
        else:
            registry.kill_switch(rule_id, actor=state.username, reason=reason)
            result.audit.record(
                "KILL_SWITCH_ENGAGED", actor=state.username, actor_role=state.role.value,
                tenant_id=state.tenant_id, subject=rule_id, reason=reason)
            st.success(f"“{titles[rule_id]}” is switched off. From the next run its flags "
                       "are marked watch-only. Recorded in the audit log with your name, the time "
                       "and your reason.")
    if c2.button("Switch it back on", key="gov_release",
                 help="Release the kill switch so the check's flags reach the queue normally."):
        if len(reason.strip()) < MIN_REASON:
            st.error(f"Please give a reason of at least {MIN_REASON} characters.")
        else:
            registry.restore(rule_id, actor=state.username, reason=reason)
            result.audit.record(
                "KILL_SWITCH_RELEASED", actor=state.username, actor_role=state.role.value,
                tenant_id=state.tenant_id, subject=rule_id, reason=reason)
            st.success(f"“{titles[rule_id]}” is switched back on. Recorded in the audit log.")
    if c3.button("Roll back to the previous version", key="gov_rollback",
                 help="Return the check to its previous version. It restarts in watch-only mode."):
        if len(reason.strip()) < MIN_REASON:
            st.error(f"Please give a reason of at least {MIN_REASON} characters.")
        else:
            try:
                rolled = registry.rollback(rule_id, actor=state.username)
            except Exception as exc:  # noqa: BLE001 - no earlier version is the usual cause
                st.error("This check has no earlier version to roll back to.")
                st.caption(str(exc))
            else:
                result.audit.record(
                    "RULE_ROLLED_BACK", actor=state.username, actor_role=state.role.value,
                    tenant_id=state.tenant_id, subject=rule_id,
                    reason=f"Rolled back to v{rolled.version}. {reason.strip()}")
                st.success(f"“{titles[rule_id]}” is back on version {rolled.version} and "
                           "restarts in watch-only mode. A roll-back never goes straight into use.")


# ---------------------------------------------------------------------------
# 4. ceilings and expected volume
# ---------------------------------------------------------------------------


def _ceilings(result, config) -> None:
    st.markdown("### Flag limits and expected volumes")
    ceiling = int(config.get("alert_volume_ceiling_per_rule_per_run"))
    breaches = list(result.evaluation.ceiling_breaches)
    st.markdown(
        f"**Each check may raise at most {ceiling:,} flags per run; beyond that its flags are "
        "kept but marked watch-only.** The limit stops one runaway check from flooding the queue. "
        "The flags are never thrown away, because silently dropping them would hide a gap in "
        f"coverage. (Setting: {parameter_text('alert_volume_ceiling_per_rule_per_run')['label']}, "
        f"owned by {owner_label(config.parameters.record('alert_volume_ceiling_per_rule_per_run').owner)}.)"
    )
    run = result.evaluation.to_frame()
    headline_cards([
        (f"{ceiling:,}", "flags per check per run is the limit."),
        (f"{len(breaches)}", "checks went over it on this run"
                             + (": " + ", ".join(check_title(b) for b in breaches) + "."
                                if breaches else ".")),
    ])
    fired = run[run["signal_count"] > 0].sort_values("signal_count", ascending=False) \
        if not run.empty else run
    if fired.empty:
        empty_state("No check raised a flag", "There is nothing to compare with the limit.")
    else:
        top = fired.head(20).assign(Check=[check_title(r) for r in fired.head(20)["rule_id"]])
        top = top.rename(columns={"signal_count": "Flags"})
        title = (f"No check went over the limit of {ceiling:,} flags" if not breaches else
                 f"{len(breaches)} checks went over the limit of {ceiling:,} flags")
        bar(top, "Check", "Flags", horizontal=True, height=460, title=title,
            x_label="Check", y_label="Flags on this run",
            how_to_read=f"The 20 busiest checks. Any bar longer than {ceiling:,} went over the "
                        "limit and its flags were marked watch-only.")

    st.markdown("#### Expected against actual flags")
    st.markdown(
        "Where a check's settings carry an **expected number of flags**, that number is a "
        "commitment by the setting's owner, **not a measured result**. A large gap between "
        "expected and actual is what the release checks ask a team to look into."
    )
    rows = []
    for control in result.registry.executable():
        observed = int(run.loc[run["rule_id"] == control.rule_id, "signal_count"].sum()) \
            if not run.empty else 0
        expected = None
        for _key, value in (control.parameters or {}).items():
            if isinstance(value, str) and value.startswith("cfg."):
                try:
                    expected = config.parameters.record(value[4:]).expected_alert_volume
                except Exception:  # noqa: BLE001
                    expected = None
                if expected is not None:
                    break
        rows.append({"rule_id": control.rule_id, "Check": check_title(control.rule_id, control.name),
                     "Actual flags": observed,
                     "Expected flags (owner's commitment)": "not set" if expected is None else f"{expected:,}",
                     "Difference": "not comparable" if expected is None else f"{observed - expected:+,}"})
    frame = pd.DataFrame(rows)
    if frame.empty:
        empty_state("No check to compare", "No check in the catalogue runs fully on this file.")
    else:
        dataframe(frame.drop(columns="rule_id").sort_values("Actual flags", ascending=False))
        with st.expander("Technical details: rule ids", expanded=False):
            dataframe(frame)


# ---------------------------------------------------------------------------
# 5. latency
# ---------------------------------------------------------------------------


def _latency(result, config) -> None:
    st.markdown("### Speed of each stage against its time limit")
    st.markdown(
        "Each stage of checking has a time limit from the design (for example, checks run "
        "before payment must be quick). The times below are for the **whole file in one batch**, "
        f"here {plain_number(len(result.claims))} claims, not the time per claim. A real "
        "before-payment deployment would be timed claim by claim; dividing the batch time by the "
        "number of claims and calling it a per-claim time would be misleading, so it is not done."
    )
    report = result.evaluation.latency_report(config)
    rows = []
    for _, r in report.iterrows():
        within = r["within_ceiling"]
        verdict = ("Not measured" if within is None or (isinstance(within, float) and pd.isna(within))
                   else ("Within the limit" if bool(within) else "Over the limit (batch time)"))
        rows.append({"Stage": label(r["stage"], "stage"),
                     "Time limit": plain_seconds(r["ceiling_ms"]),
                     "Time taken for the whole file": plain_seconds(r["observed_ms"]),
                     "Result": verdict})
    dataframe(pd.DataFrame(rows))
    with st.expander("Technical details: the latency report as recorded", expanded=False):
        dataframe(report.astype(str))

    timing = pd.DataFrame([{"Step": k.capitalize(), "Seconds": round(v, 2)}
                           for k, v in result.timings.items()]).sort_values("Seconds", ascending=False)
    if not timing.empty:
        slow = timing.iloc[0]
        bar(timing, "Step", "Seconds", horizontal=True, height=340,
            title=f"{slow['Step']} took the longest on this run ({slow['Seconds']:.0f} seconds)",
            x_label="Step of the run", y_label="Seconds",
            how_to_read="Each bar is one step of the whole run over this file; longer bars took "
                        "more time.")


def _drift_questions(results, config) -> list[dict[str, str]]:
    """Each monitor as the same plain question the Models page asks (shared wording)."""
    try:
        from components.analysis_pages import drift_rows

        rows = drift_rows(results, warn=float(config.get("psi_warn_threshold")),
                          breach=float(config.get("psi_breach_threshold")))
        return [{"Question": r["question"], "Result": r["label"], "In words": r["sentence"]}
                for r in rows]
    except Exception:  # noqa: BLE001 - fall back to the simple table below
        return []


def _recommendation(results) -> str:
    """The drift monitor's recommendation in words (same three outcomes, same meaning)."""
    statuses = {getattr(r, "status", "") for r in results}
    if "BREACH" in statuses:
        return ("Over the action level: re-tune the model, or switch it off if the problem "
                "continues. Neither happens automatically; switching a model off is a governed "
                "decision by a named owner.")
    if "WARN" in statuses:
        return "Close to the action level: keep watching and plan a re-tuning review."
    return "Within limits: no action needed."


# ---------------------------------------------------------------------------
# 6. model health
# ---------------------------------------------------------------------------


def _model_health(result) -> None:
    st.markdown("### Model health", help=help_text("psi_drift"))
    st.markdown(
        "For each pattern-finding model the engine checks whether the claims it now scores still "
        "look like the claims it learned from (**drift**), whether it flags about as many as "
        "expected, and whether its flags turn out to be right. A breach **recommends** re-tuning "
        "the model or switching it off; it never does either by itself. Switching a model off is "
        "a governed action with a named owner."
    )
    layer = result.models
    if layer is None or not layer.drift:
        empty_state("No model health to report",
                    "The pattern-finding models did not run on this file, usually because it has "
                    "too few months or providers to learn from.")
        return
    from fwa.models.drift import DriftMonitor

    for name, results in layer.drift.items():
        st.markdown(f"#### {model_label(name)}")
        rows = _drift_questions(results, result.config)
        if rows:
            dataframe(pd.DataFrame(rows))
            st.markdown(f"**{_recommendation(results)}**")
            with st.expander(f"Technical details: {model_label(name)} monitor notes", expanded=False):
                st.markdown(f"Recommendation, verbatim: {DriftMonitor.recommendation(results)}")
                dataframe(pd.DataFrame([d.to_row() for d in results]).astype(str))
            continue
        for d in results:
            r = d.to_row()
            value = r.get("value")
            rows.append({
                "What is checked": plain_text(r.get("monitor")),
                "Measured value": "can't be measured yet" if value is None else plain_number(value, 2),
                "Warning level": "—" if r.get("warn_threshold") is None else plain_number(r["warn_threshold"], 2),
                "Action level": "—" if r.get("breach_threshold") is None else plain_number(r["breach_threshold"], 2),
                "Result": status(r.get("status")).label,
            })
        dataframe(pd.DataFrame(rows))
        st.markdown(f"**{_recommendation(results)}**")
        with st.expander(f"Technical details: {model_label(name)} monitor notes", expanded=False):
            st.markdown(f"Recommendation, verbatim: {DriftMonitor.recommendation(results)}")
            dataframe(pd.DataFrame([d.to_row() for d in results]).astype(str))
    note("A breach recommends re-tuning or switching the model off; it does not do either. "
         + standard_text("shadow"))
    if layer.supervised is not None:
        st.markdown("#### Could a model trained on past decisions be used?",
                    help=help_text("supervised_model"))
        st.markdown(status_badge(layer.supervised.status), unsafe_allow_html=True)
        st.markdown(status(layer.supervised.status).meaning)
        with st.expander("Technical details: the supervised gate's summary", expanded=False):
            st.markdown(layer.supervised.summary)
