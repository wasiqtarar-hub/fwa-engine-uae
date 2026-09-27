"""Checks library (internal name "Rule Registry") — Role 2, the policy owner.

Build brief §14, item 7: "browse the catalogue by scenario; view a control's
YAML, version history, shadow-period volume and review yield; author a rule in
**natural language → validated YAML draft → shadow**; the shadow→active
approval flow with sign-off capture."

§9.6's claim is that a policy owner can add a rule **without a code
deployment**. This page is where that claim is either true or it is not: a rule
drafted here becomes configuration in the registry, in shadow, with its
ten-category test stubs generated alongside, and nobody touches Python.

Plain wording is the default view. Rule ids, versions, types, stages,
fingerprints, expressions, parameters and exclusions are one click away:
"Show all columns" on the table and a "Technical details" expander per check.
Whether a check could run is the EFFECTIVE support on this run (a dataset
unlock can make a catalogue "couldn't run" check run), with the catalogue's
own classification kept beside it.
"""

from __future__ import annotations

import html

import pandas as pd
import streamlit as st
import yaml

from common import (
    SEQUENCE, SEQUENCE_DARK, active_theme, banner, bar, dataframe, disposition_chip,
    empty_state, friendly_failure, friendly_table, headline_cards, mono, note,
    page_header, pipeline, require_page, session_banner, status_badge, synthetic_banner,
)
from components.governance_pages import (
    basis_label, check_rows, family_label, owner_label, plain_text,
)
from fwa.auth import Permission
from fwa.engine.contract import DispositionLegalityError  # noqa: F401 - raised by the validator path
from fwa.engine.registry import SeparationOfDutiesError
from fwa.enums import RuleStatus
from fwa.presentation import (
    control_text, count_phrase, label, plain_date, standard_text, status,
)

_SUPPORT_ORDER = ["EXECUTABLE", "PARTIAL", "NOT_EXECUTABLE_ON_THIS_DATASET"]


def _support_colours() -> dict[str, str]:
    seq = SEQUENCE_DARK if active_theme() == "dark" else SEQUENCE
    return {status("EXECUTABLE").label: seq[2], status("PARTIAL").label: seq[3],
            status("NOT_EXECUTABLE_ON_THIS_DATASET").label: seq[5]}


def render() -> None:
    state = require_page("Rule Registry", Permission.VIEW_RULE_REGISTRY)
    session_banner(state)
    page_header("Rule Registry")
    banner()
    synthetic_banner()

    result = pipeline()
    registry = result.registry
    rows = pd.DataFrame(check_rows(result))

    tab_browse, tab_control, tab_author, tab_approve = st.tabs([
        "All checks", "Look at one check", "Draft a new check", "Approve a check for use",
    ])

    with tab_browse:
        with friendly_failure("the list of checks"):
            _browse(result, rows)
    with tab_control:
        with friendly_failure("this check's details"):
            _one_check(result, rows, state)
    with tab_author:
        with friendly_failure("the check-drafting form",
                              "Try again, or ask an administrator to check that drafting is "
                              "switched on."):
            _author(result, registry, state)
    with tab_approve:
        with friendly_failure("the approval form"):
            _approve(result, registry, state, rows)


# ---------------------------------------------------------------------------
# browse
# ---------------------------------------------------------------------------


def _browse(result, rows: pd.DataFrame) -> None:
    if rows.empty:
        empty_state("No checks are loaded", "The checks library is empty. Ask an administrator "
                                            "to check the rules folder.")
        return
    total = len(rows)
    by = rows["data_support"].value_counts()
    full, part = int(by.get("EXECUTABLE", 0)), int(by.get("PARTIAL", 0))
    cant = int(by.get("NOT_EXECUTABLE_ON_THIS_DATASET", 0))
    fired = int((rows["signal_count"] > 0).sum())
    flags = int(rows["signal_count"].sum())
    may_deny = int(rows["can_deny"].sum())
    unlocked = int((rows["support_basis"] != "catalogue").sum())

    headline_cards([
        (f"{full + part} of {total}", f"checks could run on this file: {full} fully and {part} in "
                                      "a simplified form."),
        (f"{cant}", "checks could not run because the file lacks the information they need. "
                    "Each one says what is missing."),
        (f"{fired}", f"checks raised at least one flag, {count_phrase(flags, 'flag')} in all. A "
                     "flag is a reason to look, not proof of fraud."),
        (f"{may_deny}", "checks may ever recommend denying or re-pricing a claim. All are hard "
                        "or expert rules; no comparison or model can."),
    ])
    note(standard_text("shadow") + " The library refuses to load any comparison, connection, "
         "document or model check that tries to deny or re-price a claim.")
    if unlocked:
        st.caption(f"{unlocked} of these checks could run only because this file carries extra "
                   "tables; the catalogue on its own classes them differently. Tick 'Show all "
                   "columns' to see both.")

    f1, f2, f3 = st.columns([2, 2, 1])
    families = sorted(rows["family"].unique())
    with f1:
        chosen_families = st.multiselect(
            "Area", families, default=[], format_func=family_label, key="rr_family",
            help="Show only checks in these areas, for example payment accuracy or medicines. "
                 "Leave it empty to show every area.",
        )
    supports = [s for s in _SUPPORT_ORDER if s in set(rows["data_support"])]
    with f2:
        chosen_supports = st.multiselect(
            "Could it run on this file?", supports, default=[],
            format_func=lambda s: status(s).label, key="rr_support",
            help="Filter by whether each check ran fully, ran in a simplified form, or could not "
                 "run because the file lacks information. Leave it empty to show all.",
        )
    with f3:
        only_firing = st.checkbox(
            "Only checks that raised flags", value=False, key="rr_firing",
            help="Tick to hide checks that raised nothing on this file, so you can focus on what "
                 "they found.",
        )
    query = st.text_input(
        "Search checks", key="rr_search", placeholder="e.g. duplicate, refill, licence",
        help="Type a word to find checks whose name or description mentions it. Leave empty to "
             "see every check.",
    )

    view = rows
    if chosen_families:
        view = view[view["family"].isin(chosen_families)]
    if chosen_supports:
        view = view[view["data_support"].isin(chosen_supports)]
    if only_firing:
        view = view[view["signal_count"] > 0]
    if query.strip():
        q = query.strip().lower()
        hay = (view["check_title"] + " " + view["check_what"] + " " + view["rule_id"]).str.lower()
        view = view[hay.str.contains(q, regex=False)]
    view = view.assign(_order=view["data_support"].map({s: i for i, s in enumerate(_SUPPORT_ORDER)}))
    view = view.sort_values(["signal_count", "_order", "check_title"],
                            ascending=[False, True, True]).drop(columns="_order")

    st.caption(f"Showing {len(view):,} of {total:,} checks, those that raised the most flags first.")
    friendly_table(
        view, key="rr_table",
        columns=["check_title", "check_what", "family_words", "data_support", "missing_words",
                 "signal_count", "disposition", "run_outcome",
                 # technical columns, behind "Show all columns"
                 "rule_id", "scenario_id", "rule_version", "type", "stage", "status",
                 "catalogue_data_support", "support_basis", "basis_words", "support_reason",
                 "missing_for_unlock", "required_canonical_fields", "owner", "can_deny",
                 "exclusions", "implementation", "catalogue_trigger", "catalogue_disposition",
                 "governance_note", "rule_fingerprint", "check_why", "owner_words"],
        rename={"disposition": "What policy suggests if it fires",
                "signal_count": "Flags on this run",
                "data_support": "Could it run on this file?"},
        value_kinds={"catalogue_data_support": "status"},
        height=430,
        empty_title="No check matches",
        empty_body="Nothing matched the filters or search. Clear them to see every check.",
    )

    counts = rows.assign(Area=rows["family"].map(family_label),
                         Support=rows["data_support"].map(lambda s: status(s).label))
    counts = counts.groupby(["Area", "Support"]).size().reset_index(name="Checks")
    if cant > total / 2:
        title = f"Most checks ({cant} of {total}) could not run on this file"
    else:
        title = f"{full + part} of {total} checks could run on this file"
    bar(counts, "Area", "Checks", title=title, colour="Support", colour_map=_support_colours(),
        horizontal=True, height=360, x_label="Area", y_label="Number of checks",
        legend_title="Could it run?",
        how_to_read="Each bar is one area of checks. Green ran fully, amber ran in a simplified "
                    "form, grey could not run because information was missing from the file.")

    missing = rows[rows["data_support"] == "NOT_EXECUTABLE_ON_THIS_DATASET"]
    if not missing.empty:
        with st.expander(f"What the file is missing, grouped ({len(missing)} checks could not run)",
                         expanded=False):
            groups = missing["missing_words"].replace("", "not recorded").value_counts()
            for need, n in groups.items():
                st.markdown(f"- needs {need} ({count_phrase(n, 'check')})")


# ---------------------------------------------------------------------------
# one check
# ---------------------------------------------------------------------------


def _one_check(result, rows: pd.DataFrame, state) -> None:
    registry = result.registry
    if rows.empty:
        empty_state("No checks are loaded", "The checks library is empty.")
        return
    st.markdown("### Look at one check")
    families = sorted(rows["family"].unique())
    c1, c2 = st.columns([1, 2])
    with c1:
        family = st.selectbox(
            "Area", families, format_func=family_label, key="rr_one_family",
            help="Choose an area first to shorten the list of checks.",
        )
    subset = rows[rows["family"] == family].sort_values(["signal_count", "check_title"],
                                                        ascending=[False, True])
    titles = _unique_titles(subset)
    with c2:
        rule_id = st.selectbox(
            "Check", list(subset["rule_id"]), format_func=lambda r: titles.get(r, r),
            key="rr_one_check",
            help="Pick a check to read what it looks for, whether it could run on this file and "
                 "what it found. Checks that raised the most flags are listed first.",
        )
    row = subset.set_index("rule_id").loc[rule_id]
    control = registry.get(rule_id)
    text = control_text(rule_id, fallback_name=control.name)
    esc = html.escape

    st.markdown(f"#### {esc(text.title)}")
    st.markdown(
        status_badge(row["data_support"]) + " " + status_badge(control.status.value) + " "
        + (status_badge("KILL_SWITCHED") if getattr(control, "kill_switched", False) else ""),
        unsafe_allow_html=True,
    )
    st.markdown(f"**What it looks for.** {text.what or text.finding}")
    if text.why:
        st.markdown(f"**Why it matters.** {text.why}")

    disp = status(control.disposition.value)
    st.markdown(
        "**What policy suggests if it fires.** " + disposition_chip(control.disposition)
        + f" {esc(disp.meaning)}", unsafe_allow_html=True)
    if control.can_deny:
        st.caption("This is a hard or expert rule, so policy allows it to recommend denying or "
                   "re-pricing, but only after a reviewer confirms the rule that failed.")
    else:
        st.caption("This check can never deny or re-price a claim on its own. "
                   + standard_text("priority_not_decision"))

    # ---- could it run -----------------------------------------------------
    support = row["data_support"]
    st.markdown("**Could it run on this file?** " + status(support).label + ". "
                + status(support).meaning)
    if row["support_basis"] != "catalogue":
        st.markdown(f"*{basis_label(row['support_basis'])}.* The catalogue on its own classes it "
                    f"as “{status(row['catalogue_data_support']).label}”.")
    if support != "EXECUTABLE":
        need = row["missing_words"] or text.needs
        if need:
            st.markdown(f"**What the file is missing.** It needs {need}.")
        if text.needs and text.needs not in (need or ""):
            st.caption(f"In full, this check needs {text.needs}.")

    signals = result.signals.by_rule(rule_id, state.tenant_id)
    headline_cards([
        (f"{len(signals):,}", "flags raised on this run. Each is a reason to look, not proof."),
        (label(row["run_outcome"]) if row["run_outcome"] else "Not run",
         "is what happened when the engine tried this check."),
        ("Can't be measured yet", "how often its flags turn out to be right: no reviewer has "
                                  "recorded a decision on them. No number is invented instead."),
    ])
    if text.next_steps:
        st.markdown("**What a reviewer checks next when it fires**\n\n" + "\n".join(
            f"{i}. {s}" for i, s in enumerate(text.next_steps, start=1)))
    st.caption(f"Owned by {owner_label(control.owner)}. {standard_text('shadow')}")

    with st.expander("Technical details: identifiers, contract and version history", expanded=False):
        st.markdown(
            f"**{control.rule_id}** · version `{control.version}` · `{control.qualified_id}` · "
            f"type `{control.type_label}` ({control.primary_type.label}) · stage "
            f"`{control.stage.value}` ({label(control.stage.value, 'stage')}) · status "
            f"`{control.status.value}` · disposition `{control.disposition.value}` · owner "
            f"`{control.owner}` · fingerprint `{control.fingerprint()}`")
        st.markdown("**From the control catalogue, verbatim**")
        dataframe(pd.DataFrame([
            {"column": "Implementable trigger", "value": control.catalogue_trigger},
            {"column": "Configuration and exclusions", "value": control.catalogue_config_exclusions},
            {"column": "Output/disposition", "value": control.catalogue_disposition},
        ]))
        if control.governance_note:
            st.markdown(f"**Governance note.** {control.governance_note}")
        st.markdown(f"**Data support on this run:** `{support}` (catalogue: "
                    f"`{row['catalogue_data_support']}`, basis `{row['support_basis']}`)")
        st.markdown(f"> {row['support_reason'] or control.data_support_reason}")
        if row["missing_for_unlock"]:
            st.markdown(f"**Missing for the dataset unlock:** `{row['missing_for_unlock']}`")
        if control.required_canonical_fields:
            st.markdown("**Required canonical fields**")
            st.markdown("\n".join(f"- `{f}`" for f in control.required_canonical_fields))
        st.markdown("**The atomic-control contract (expression, parameters, exclusions)**")
        mono(yaml.safe_dump(control.to_yaml_dict(), sort_keys=False, allow_unicode=True, width=88))
        history = registry.history(rule_id)
        if history:
            st.markdown("**Version history**")
            dataframe(pd.DataFrame([
                {"version": h.version, "status": h.status.value, "fingerprint": h.fingerprint()}
                for h in history
            ]))


def _unique_titles(frame: pd.DataFrame) -> dict[str, str]:
    """Plain titles for a selectbox, disambiguated when two checks share one."""
    seen: dict[str, int] = {}
    out: dict[str, str] = {}
    for rid, title in zip(frame["rule_id"], frame["check_title"]):
        seen[title] = seen.get(title, 0) + 1
        out[rid] = title if seen[title] == 1 else f"{title} ({seen[title]})"
    return out


# ---------------------------------------------------------------------------
# author
# ---------------------------------------------------------------------------


def _scenario_words(registry, scenario_id: str) -> str:
    controls = registry.by_scenario(scenario_id)
    first = control_text(controls[0].rule_id, fallback_name=controls[0].name).title if controls else ""
    return f"{family_label(scenario_id.split('-')[0])}: {first} ({scenario_id})"


def _author(result, registry, state) -> None:
    st.markdown("### Draft a new check in plain English")
    if not state.can(Permission.AUTHOR_RULE) and not state.can(Permission.AUTHOR_EXPERT_RULE):
        note(f"As {label(state.role.value, 'role')} you can browse the checks library but not "
             "draft checks. Policy owners and clinical reviewers can.")
        return
    if result.ai.rule_author is None:
        empty_state("Drafting checks is switched off",
                    "The AI drafting feature is turned off in the settings. Every AI feature can "
                    "be switched off on its own and the tool keeps working without them.")
        return
    st.markdown(
        "Describe the policy in a sentence. The AI assistant **drafts** a check; an automatic "
        "checker that follows fixed rules **decides** whether the draft is allowed; and a "
        "**different person** must approve it before it is used. A new check always starts in "
        "watch-only (shadow) mode, and no software is changed."
    )
    example = st.selectbox(
        "Start from an example, or write your own below",
        [
            "Pend any claim where the pharmacy share exceeds twice the diagnosis-peer "
            "median and the stay is under two days.",
            "Audit providers whose approval ratio is unusually low compared to peers.",
            "Reject any claim where the billed amount is more than three times the peer "
            "median.",
        ],
        key="rr_example",
        help="Pick an example to see how drafting works. The third one asks a comparison to deny "
             "a claim, which is not allowed, so you can see the automatic checker refuse it.",
    )
    request = st.text_area(
        "What should the check do?", value=example, height=90, key="rr_request",
        help="Write the policy in plain English: what to look for, compared with what, and what "
             "should happen. Change the example text to write your own.",
    )
    scenarios = registry.scenarios()
    scenario_id = st.selectbox(
        "Which group of checks it belongs to", scenarios,
        index=scenarios.index("ANL-01") if "ANL-01" in scenarios else 0,
        format_func=lambda s: _scenario_words(registry, s), key="rr_scenario",
        help="The existing group the new check will join. It decides the area, owner and reporting "
             "line. Choose the group closest to the policy you described.",
    )
    if st.button("Draft the check", type="primary", key="rr_draft",
                 help="Ask the assistant for a draft. Nothing is saved until you press "
                      "'Add to the library' below."):
        draft = result.ai.rule_author.draft(request, author=state.username, scenario_id=scenario_id)
        st.session_state["fwa_draft"] = draft

    draft = st.session_state.get("fwa_draft")
    if draft is None:
        return
    if draft.valid:
        st.success("The automatic checker accepted this draft: it follows the check contract and "
                   "the rule that only hard or expert rules may deny a claim.")
    else:
        st.error("**The automatic checker refused this draft.** It would break one of the safety "
                 "rules, for example by letting a comparison deny a claim. The reasons are below.")
        for error in draft.errors:
            st.markdown(f"> {plain_text(error)}")
        note("This is the safeguard working, not failing. The assistant drafted what was asked "
             "for, and the checker refused it through the same code that checks a hand-written "
             "rule. There is no special treatment for AI drafts and no override.")
    st.caption("The draft was written by the AI assistant. It is configuration, not evidence, and "
               "it is not in use. The full draft is under Technical details.")
    with st.expander("Technical details: the drafted check, changes and test stubs", expanded=False):
        st.markdown("**The drafted contract**")
        mono(draft.yaml_text)
        if not draft.valid:
            st.markdown("**Validator messages, verbatim**")
            for error in draft.errors:
                st.markdown(f"> {error}")
        st.markdown("**Difference from the library**")
        mono(draft.diff)
        st.markdown("**Ten-category test stubs generated alongside** (every check needs all ten, "
                    "whatever its type, before activation can even be proposed)")
        st.markdown("\n".join(f"- `{c}`" for c in sorted(draft.test_fixture_stubs)))
        if "positive_fixture" in draft.test_fixture_stubs:
            st.markdown("**One stub**")
            mono(draft.test_fixture_stubs["positive_fixture"])

    if draft.valid and st.button(
            "Add to the library (watch-only)", key="rr_register",
            help="Adds the draft to the checks library in watch-only (shadow) mode. It will not "
                 "affect any claim until a different policy owner approves it."):
        control = result.ai.rule_author.register(draft, author=state.username)
        st.success(
            f"“{control_text(control.rule_id, fallback_name=control.name).title}” "
            f"({control.rule_id}) was added in watch-only mode. You drafted it, so you may not "
            "approve it for use: a different policy owner must."
        )


# ---------------------------------------------------------------------------
# approve
# ---------------------------------------------------------------------------


def _approve(result, registry, state, rows: pd.DataFrame) -> None:
    st.markdown("### Approve a check for real use")
    st.markdown(
        "Moving a check from watch-only (shadow) to in use needs a written sign-off from a "
        "policy owner who did **not** draft it. The approval is recorded in the audit log with "
        "your name, the time and your note."
    )
    if not state.can(Permission.APPROVE_RULE_ACTIVATION):
        note(f"As {label(state.role.value, 'role')} you cannot approve a check for use. Only a "
             "policy owner can, and never one who drafted the check.")
        return
    shadow = registry.by_status(RuleStatus.SHADOW)
    if not shadow:
        empty_state("No check is waiting", "Every check has already been approved or retired.")
        return
    st.caption(f"{count_phrase(len(shadow), 'check')} in watch-only mode. Newly drafted checks are "
               "listed first.")
    authored = [c for c in shadow if c.authored_by]
    options = authored + [c for c in shadow if not c.authored_by][: max(0, 50 - len(authored))]
    titles = {c.rule_id: control_text(c.rule_id, fallback_name=c.name).title
              + (" (newly drafted)" if c.authored_by else "") for c in options}
    rule_id = st.selectbox(
        "Check to approve", [c.rule_id for c in options], format_func=lambda r: titles.get(r, r),
        key="rr_approve_pick",
        help="Pick the watch-only check you want to put into use. Checks drafted on this page "
             "come first.",
    )
    control = registry.get(rule_id)
    st.markdown(
        f"Drafted by **{control.authored_by or 'the published catalogue'}**, in watch-only mode "
        f"since **{plain_date(control.shadow_since) if control.shadow_since else 'the library was loaded'}**."
    )
    signals = result.signals.by_rule(rule_id, state.tenant_id)
    min_days = result.config.get("shadow_period_min_days")
    headline_cards([
        (f"{len(signals):,}", "flags raised in watch-only mode on this run."),
        (f"{min_days} days", "of watching new claims is required first. One look back at an old "
                             "file does not count."),
        ("Can't be measured yet", "how often its flags are right: no reviewer has recorded a "
                                  "decision."),
    ])
    sign_off = st.text_area(
        "Sign-off note (required, at least 10 characters)", height=90, key="rr_signoff",
        help="Say why the check is ready: what you reviewed, the watch-only results you looked at "
             "and who agreed. The note is stored in the audit log with your name.",
    )
    st.caption("If you approve, this check's flags will be used for real decisions. If you drafted "
               "it yourself the approval will be refused.")
    if st.button("Approve for use", type="primary", key="rr_approve",
                 help="Records your approval and moves the check from watch-only to in use."):
        if len(sign_off.strip()) < 10:
            st.error("Please write a sign-off note of at least 10 characters.")
            return
        try:
            activated = registry.activate(rule_id, actor=state.username, note=sign_off)
        except SeparationOfDutiesError as exc:
            st.error("You drafted this check, so you may not approve it. A different policy owner "
                     "must approve it (separation of duties).")
            st.caption(str(exc))
            return
        result.audit.record(
            "RULE_ACTIVATED", actor=state.username, actor_role=state.role.value,
            tenant_id=state.tenant_id, subject=rule_id, reason=sign_off,
            after={"status": activated.status.value},
        )
        st.success(f"“{titles.get(rule_id, rule_id)}” is now in use, approved by "
                   f"{state.username} and recorded in the audit log.")
        st.warning(
            "What this run could not show: the release check for comparison rules needs a "
            "watch-only period on new claims and a record of how often its flags were right. "
            "Neither exists after one look back at an old file, so that release check is marked "
            f"“{status('NOT_ASSESSABLE').label}”."
        )
