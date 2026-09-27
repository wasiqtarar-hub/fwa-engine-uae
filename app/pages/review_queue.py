"""Review queue — the claims reviewer's working surface, in plain words.

The list of cases, most urgent first; how long the queue would take the team
(shown live as the team size and daily pace change); plain filters; a table of
at most eight friendly columns; and a detail panel that explains the selected
case in plain words and points to Case evidence.

Safety wording travels with the numbers: urgency orders the queue only and
never changes a disposition; a disposition is what policy suggests, never a
finding of fraud; established and not-yet-established amounts are shown as two
figures and never added; precision cannot be measured until reviewers record
decisions, and is said so in words. The label-based precision *proxy* survives
only inside a technical expander, labelled as an upper-bound proxy.
"""

from __future__ import annotations

import html

import pandas as pd
import streamlit as st

from common import (
    auth_service, banner, case_explanation, dataframe, disposition_chip, empty_state,
    friendly_failure, friendly_table, help_text, mask, page_header, pipeline, priority_chip,
    render_explanation, require_page, session_banner, synthetic_banner,
)
from components.reviewer_flow import (
    case_label_func, link_to, policy_label, split_amounts, subject_for, tour_step,
    tour_toggle, urgency_text, workload_sentence,
)
from fwa.auth import Permission
from fwa.enums import REVIEWER_ASK
from fwa.presentation import aed, control_text, label, plain_month, standard_text, status


def render() -> None:
    state = require_page("Review Queue", Permission.VIEW_QUEUE)
    session_banner(state)

    page_header("Review Queue")
    synthetic_banner()
    banner()

    result = pipeline()
    queue = result.queue(state.tenant_id)

    if queue.empty:
        empty_state(
            "Your queue is empty",
            "No cases were found for your organisation. Either no check raised a flag on this file, "
            "or every check that did has been switched off. The Governance page shows which; the "
            "Overview page shows which checks could run.",
        )
        return

    tour_toggle("Review Queue", key="rq_tour")
    tour_step("Review Queue", 0)

    # ---- workload -----------------------------------------------------------
    config = result.config
    per_reviewer = int(config.get("alerts_per_reviewer_per_day"))
    default_reviewers = int(config.get("reviewer_count"))

    st.markdown("## How long will this take?")
    c1, c2 = st.columns(2)
    with c1:
        reviewers = st.slider(
            "How many reviewers are on the team", 1, 20, default_reviewers, key="rq_reviewers",
            help="The number of people working this queue. Change it to see how a bigger or "
                 "smaller team changes how long the queue would take. It changes nothing in the "
                 "data or the order of the cases.",
        )
    with c2:
        per_day = st.slider(
            "How many cases one reviewer can handle per day", 5, 80, per_reviewer, step=5,
            key="rq_per_day",
            help="A realistic daily pace for one reviewer. Change it if your cases are quicker or "
                 "slower than usual. It is the basis for 'what the team can get through', which is "
                 "where precision has to be judged: at the volume a team can actually process. "
                 + help_text("capacity"),
        )
    capacity = reviewers * per_day
    st.markdown(workload_sentence(len(queue), reviewers, per_day))
    top = queue.head(capacity)
    est, unest = split_amounts(top)
    st.markdown(
        f"The first day's work — the {min(capacity, len(queue)):,} most urgent cases — involves "
        f"**{aed(est)} established** and, separately, **{aed(unest)} not yet established**. "
        f"The two are never added together. {html.escape(standard_text('not_savings'))}"
    )
    _precision_words(state)
    with st.expander("Technical details: label-based precision proxy (not precision)"):
        _precision_proxy(result, queue, capacity)

    # ---- filters ----------------------------------------------------------
    st.markdown("## Narrow the list")
    filtered = _filters(queue)
    st.caption(
        f"{len(filtered):,} of {len(queue):,} cases shown. Identities are hidden for the "
        f"{label(state.role.value, 'role').lower()} role unless you are allowed to see them; revealing "
        f"one needs a reason and is written to the access log."
    )
    if len(filtered) != len(queue) and not filtered.empty:
        st.markdown(workload_sentence(len(filtered), reviewers, per_day, what="your narrowed list"))

    if filtered.empty:
        empty_state("No cases match these filters",
                    "Tick more options above, or move the urgency slider back towards 0.")
        return

    # ---- the queue --------------------------------------------------------
    st.markdown("## Cases, most urgent first")
    with friendly_failure("the list of cases"):
        _queue_table(filtered.head(500), state)
        if len(filtered) > 500:
            st.caption(f"Showing the 500 most urgent of {len(filtered):,} cases. Narrow the list to "
                       f"see others.")

    # ---- detail panel -----------------------------------------------------
    st.markdown("## Read one case")
    tour_step("Review Queue", 1)
    with friendly_failure("the selected case", "Pick another case, or open Case evidence."):
        _detail_panel(result, filtered.head(500), state)

    # ---- what the queue is made of ---------------------------------------
    with st.expander("Technical details: which checks make up this list"):
        rows = []
        for case_row in filtered.head(2000).itertuples(index=False):
            for signal in result.signals_for_case(case_row.case_id):
                rows.append({"check": control_text(signal.rule_id).title, "rule_id": signal.rule_id,
                             "disposition": signal.disposition.value})
        if rows:
            frame = pd.DataFrame(rows).value_counts().reset_index(name="signals")
            dataframe(frame.sort_values("signals", ascending=False))
        else:
            st.caption("No flags behind the listed cases.")


# ---------------------------------------------------------------------------
# sections
# ---------------------------------------------------------------------------


def _precision_words(state) -> None:
    try:
        reviewed = len(auth_service().db.outcomes_frame(state.tenant_id))
    except Exception:
        reviewed = 0
    if reviewed == 0:
        st.markdown(
            "<div class='fwa-caveat'><strong>How often are the flags right?</strong> We can't measure "
            "this yet, because no reviewer has recorded a decision. It becomes measurable as "
            "decisions are recorded on the Case evidence page; until then it is left blank rather "
            "than guessed.</div>", unsafe_allow_html=True)
    else:
        st.markdown(
            f"<div class='fwa-caveat'><strong>How often are the flags right?</strong> {reviewed:,} "
            f"decision(s) have been recorded so far. The Validation report says whether that is "
            f"enough to measure it; this page never estimates it.</div>", unsafe_allow_html=True)


def _precision_proxy(result, queue, capacity: int) -> None:
    from fwa.evaluation import MetricSuite

    try:
        outcomes = MetricSuite(result).case_outcomes()
    except Exception:
        outcomes = pd.Series(dtype=float)
    if outcomes.empty:
        st.caption("This file carries no investigation labels, so not even a proxy can be computed.")
        return
    top = queue.head(capacity)
    y = outcomes.reindex(top["case_id"]).fillna(0.0)
    st.markdown(
        f"**Label-based proxy @ capacity: {y.mean():.1%}** of the top {min(capacity, len(queue)):,} "
        f"cases contain a claim carrying an investigation label in the source file."
    )
    st.caption(
        "This is an UPPER-BOUND PROXY, not precision. Real precision is confirmed ÷ reviewed, and "
        "nothing here has been reviewed. The labels are investigation-derived and selection-biased. "
        "On a synthetic file the labels were planted to test the checks, so the proxy measures the "
        "test, not detection performance."
    )


def _filters(queue: pd.DataFrame) -> pd.DataFrame:
    f1, f2 = st.columns(2)
    disp_opts = sorted(queue["disposition"].unique())
    fam_opts = sorted(queue["scenario_family"].unique())
    dim_opts = sorted(queue["case_type"].unique())
    with f1:
        dispositions = st.multiselect(
            "What policy suggests", disp_opts, default=disp_opts, key="rq_f_disp",
            format_func=lambda v: status(v).label,
            help="Show only cases where the rules suggest these next steps. Untick one to hide it, "
                 "for example to work only on cases to hold before paying. A suggestion is for you "
                 "to confirm; none is a finding of fraud.",
        )
        families = st.multiselect(
            "Kind of pattern", fam_opts, default=fam_opts, key="rq_f_fam",
            format_func=lambda v: label(v, "family"),
            help="Show only cases about these kinds of problem, for example if you specialise in "
                 "coding or in medicines.",
        )
    with f2:
        dimensions = st.multiselect(
            "How the flags were grouped", dim_opts, default=dim_opts, key="rq_f_dim",
            format_func=lambda v: label(v, "dimension"),
            help="Each case groups flags about one thing: one claim, one episode of care, one "
                 "provider, a connected group or an agent. Untick groupings you don't work on.",
        )
        min_priority = st.slider(
            "Show only cases at least this urgent (0 = show everything, 100 = only the most urgent)",
            0, 100, 0, key="rq_f_min",
            help="Hides cases below this urgency score. Raise it when the queue is too long to work "
                 "through and you only want the most urgent cases. "
                 + standard_text("priority_not_decision"),
        )
    return queue[
        queue["disposition"].isin(dispositions)
        & queue["case_type"].isin(dimensions)
        & queue["scenario_family"].isin(families)
        & (queue["priority"].fillna(0) >= min_priority)
    ]


def _month(period) -> str:
    p = str(period or "")
    if len(p) == 7 and p[4] == "-":
        return plain_month(p + "-01")
    return p or "—"


def _queue_table(frame: pd.DataFrame, state) -> None:
    assigned = set(state.assigned_case_ids)
    view = pd.DataFrame({
        "case_id": frame["case_id"].values,
        "subject_masked": [subject_for(r.subject_type, r.subject_id, state,
                                       assigned=r.case_id in assigned)
                           for r in frame.itertuples(index=False)],
        "policy_suggests": [policy_label(v) for v in frame["disposition"]],
        "urgency": [urgency_text(b, p) for b, p in zip(frame["priority_band"], frame["priority"])],
        "amount_at_risk": [aed(a) if e else f"{aed(a)} (not yet established)"
                           for a, e in zip(frame["exposure_aed"], frame["exposure_established"])],
        "pattern_kind": [label(v, "family") for v in frame["scenario_family"]],
        "claim_count": frame["claim_count"].astype(int).values,
        "case_month": [_month(v) for v in frame["period_bucket"]],
        "flag_count": frame["signal_count"].astype(int).values,
        "looked_at_as": [label(v, "dimension") for v in frame["case_type"]],
        "priority_score": frame["priority"].fillna(0).astype(float).round(1).values,
        "evidence_kinds": [", ".join(label(d, "domain") for d in str(v).split(",") if d)
                           for v in frame["domains"]],
        "watch_only": ["Yes" if v else "No" for v in frame["shadow_only"]],
    })
    friendly_table(
        view, key="rq_table",
        columns=["case_id", "subject_masked", "policy_suggests", "urgency", "amount_at_risk",
                 "pattern_kind", "claim_count", "case_month"],
        keep_numeric=["claim_count", "flag_count", "priority_score"],
        height=430,
    )


def _detail_panel(result, frame: pd.DataFrame, state) -> None:
    ids = list(frame["case_id"])
    current = st.session_state.get("fwa_selected_case")
    index = ids.index(current) if current in ids else 0
    case_id = st.selectbox(
        "Choose a case to read", ids, index=index, key="rq_pick",
        format_func=case_label_func(result, state),
        help="Pick any case from the list above. Its plain explanation appears below, and it "
             "becomes the current case on the Case evidence page.",
    )
    st.session_state["fwa_selected_case"] = case_id
    case = result.cases.cases[case_id]

    chips = disposition_chip(case.disposition)
    if case.priority is not None:
        chips += " " + priority_chip(case.priority.band, "", case.priority.value)
    st.markdown(chips, unsafe_allow_html=True)

    expl = case_explanation(result, case, state)
    render_explanation(expl, key=f"rq_expl_{case_id}", compact=True)

    phrase = status(case.disposition)
    st.markdown(
        f"<div class='fwa-card'><h4>What policy asks of you</h4>"
        f"<p><strong>{html.escape(phrase.label)}.</strong> {html.escape(phrase.meaning)}</p>"
        f"<p class='fwa-sub'>{html.escape(REVIEWER_ASK.get(case.disposition, ''))}</p></div>",
        unsafe_allow_html=True,
    )
    link_to(state, "Case Evidence",
            "Open this case in Case evidence to see all the evidence and record your decision",
            help="Takes you to the Case evidence page with this case already selected.")
