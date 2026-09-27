"""Review Queue — Role 1, the claims reviewer's working surface (§9.2, §14.1).

    "prioritised cases with scenario family, subject, exposure, disposition;
     filters by disposition/stage/scenario; capacity slider that recomputes
     precision@capacity live."                             — build brief §14

§9.2 makes this page the reviewer's entire working surface, so the design bar
applies hardest here: progressive disclosure, plain language leading, one
primary action, and identity masked by default.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from common import (
    DISPOSITION_COLOURS, banner, dataframe, disposition_chip, empty_state, guide, mask,
    note, pipeline, priority_chip, require_page, session_banner, tiles,
)
from fwa.auth import Permission
from fwa.enums import DISPOSITION_MEANINGS, Disposition, REVIEWER_ASK


def render() -> None:
    state = require_page("Review Queue", Permission.VIEW_QUEUE)
    session_banner(state)

    st.markdown("# Review queue")
    banner()

    result = pipeline()
    queue = result.queue(state.tenant_id)

    if queue.empty:
        empty_state(
            "Your queue is empty",
            "No cases are assigned to tenant "
            f"{state.tenant_id}. That can mean the pipeline found nothing, or that every "
            "control producing signals for this tenant has been kill-switched. The Governance "
            "page shows which, and the Overview page shows what ran.",
        )
        return

    # ---- guided demo ------------------------------------------------------
    if st.toggle("Walk me through a case", value=False,
                 help="Annotates this page and the Case Evidence page step by step. This is the "
                      "guided demo."):
        st.session_state["fwa_guided"] = True
        guide("Step 1 of 5 — the queue",
              "Cases are ordered by priority, which is computed from the priority formula and orders "
              "the queue ONLY. It never sets or overrides a disposition. The chip beside each "
              "case is the disposition: what policy says should happen next.")
    else:
        st.session_state["fwa_guided"] = False

    # ---- capacity ---------------------------------------------------------
    config = result.config
    per_reviewer = int(config.get("alerts_per_reviewer_per_day"))
    default_reviewers = int(config.get("reviewer_count"))

    st.markdown("## Capacity")
    c1, c2, c3 = st.columns([1, 1, 2])
    with c1:
        reviewers = st.slider("Reviewers on the team", 1, 20, default_reviewers)
    with c2:
        per_day = st.slider("Alerts per reviewer per day", 5, 80, per_reviewer, step=5)
    capacity = reviewers * per_day
    with c3:
        st.markdown("&nbsp;", unsafe_allow_html=True)
        tiles([("Daily capacity", f"{capacity:,} cases",
                f"{reviewers} reviewer(s) × {per_day}/day — the calibration basis")])

    # precision@capacity, recomputed live
    from fwa.evaluation import MetricSuite

    metrics = MetricSuite(result)
    outcomes = metrics.case_outcomes()
    if not outcomes.empty:
        top = queue.head(capacity)
        y = outcomes.reindex(top["case_id"]).fillna(0.0)
        backlog_days = len(queue) / max(capacity, 1)
        tiles([
            ("Precision proxy @ capacity", f"{y.mean():.1%}",
             f"of the top {min(capacity, len(queue)):,} cases carry a fraud label"),
            ("Exposure inside capacity", f"AED {top['exposure_aed'].sum():,.0f}",
             "established and unestablished combined — see Overview for the split"),
            ("Queue backlog", f"{backlog_days:.1f} days",
             f"{len(queue):,} cases at {capacity:,} per day"),
        ])
        note(
            "This is a PROXY for precision, not precision. Real precision is confirmed ÷ "
            "reviewed, and nothing here has been reviewed. The labels behind it are "
            "investigation-derived and selection-biased, so treat this as an upper bound. "
            "Precision has to be evaluated at the volume a team can actually process — "
            "that is what the sliders above are for."
        )

    # ---- filters ----------------------------------------------------------
    st.markdown("## Filters")
    f1, f2, f3, f4 = st.columns(4)
    with f1:
        dispositions = st.multiselect(
            "Disposition", sorted(queue["disposition"].unique()),
            default=sorted(queue["disposition"].unique()),
        )
    with f2:
        dimensions = st.multiselect(
            "Correlation dimension", sorted(queue["dimension_label"].unique()),
            default=sorted(queue["dimension_label"].unique()),
        )
    with f3:
        families = st.multiselect(
            "Scenario family", sorted(queue["scenario_family"].unique()),
            default=sorted(queue["scenario_family"].unique()),
        )
    with f4:
        min_priority = st.slider("Minimum priority", 0, 100, 0)

    filtered = queue[
        queue["disposition"].isin(dispositions)
        & queue["dimension_label"].isin(dimensions)
        & queue["scenario_family"].isin(families)
        & (queue["priority"].fillna(0) >= min_priority)
    ]

    st.caption(
        f"{len(filtered):,} of {len(queue):,} cases shown. "
        f"Identity is masked by default for the {state.role.value} role; unmasking is an "
        f"explicit, reason-required action recorded in the access log."
    )

    if filtered.empty:
        empty_state("No cases match these filters",
                    "Widen a filter above, or lower the minimum priority.")
        return

    # ---- the queue --------------------------------------------------------
    st.markdown("## Cases")
    view = filtered.head(200).copy()
    view["subject"] = [
        mask(row.subject_id, state, assigned=row.case_id in state.assigned_case_ids)
        for row in view.itertuples(index=False)
    ]
    view["exposure"] = [
        (f"AED {r.exposure_aed:,.0f}" if r.exposure_established
         else f"AED {r.exposure_aed:,.0f} — not established")
        for r in view.itertuples(index=False)
    ]
    display = view[[
        "case_id", "dimension_label", "scenario_family", "subject_type", "subject",
        "disposition", "priority", "priority_band_label", "exposure", "signal_count",
        "period_bucket",
    ]].rename(columns={
        "case_id": "Case", "dimension_label": "Dimension", "scenario_family": "Family",
        "subject_type": "Subject type", "subject": "Subject", "disposition": "Disposition",
        "priority": "Priority", "priority_band_label": "Band", "exposure": "Exposure",
        "signal_count": "Signals", "period_bucket": "Period",
    })
    dataframe(display, height=430)

    # ---- open a case ------------------------------------------------------
    st.markdown("## Open a case")
    labels = {
        f"{row.case_id} · {row.dimension_label} · {row.disposition} · "
        f"priority {row.priority:.0f}": row.case_id
        for row in filtered.head(200).itertuples(index=False)
    }
    chosen = st.selectbox("Case", list(labels), index=0)
    case_id = labels[chosen]
    case = result.cases.cases[case_id]

    st.markdown(
        disposition_chip(case.disposition)
        + " "
        + priority_chip(case.priority.band, case.priority.band_label, case.priority.value),
        unsafe_allow_html=True,
    )
    st.markdown(
        f"<div class='fwa-card'><h4>What policy asks of you</h4>"
        f"<p>{DISPOSITION_MEANINGS.get(case.disposition, '')}</p>"
        f"<p class='fwa-sub'>{REVIEWER_ASK.get(case.disposition, '')}</p></div>",
        unsafe_allow_html=True,
    )

    st.session_state["fwa_selected_case"] = case_id
    st.info(
        "This case is now the current case. Open **Case Evidence** in the sidebar to see its "
        "full bundle and record a disposition."
    )

    # ---- what the queue is made of ---------------------------------------
    with st.expander("What is in this queue, by control"):
        rows = []
        for case_row in filtered.itertuples(index=False):
            for signal in result.signals_for_case(case_row.case_id):
                rows.append({"rule_id": signal.rule_id, "disposition": signal.disposition.value})
        if rows:
            frame = pd.DataFrame(rows).value_counts().reset_index(name="signals")
            dataframe(frame.sort_values("signals", ascending=False))
        else:
            st.caption("No signals behind the filtered cases.")
