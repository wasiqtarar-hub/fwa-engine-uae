"""Overview — dataset profile, pipeline status, headline metrics with caveats.

Dataset profile, pipeline run status, and the headline metrics with their
honesty caveats.

The caveats are not footnotes here. Every headline number on this page carries
the sentence that says what it is not, because a number shown without that
sentence is the failure mode the whole dissertation argues against.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from common import (
    banner, boundary_note, dataframe, empty_state, note, pipeline, priority_chip,
    require_page, session_banner, status_chip, tiles, bar, disposition_colours,
)
from fwa.auth import Permission


def _source_currency(result) -> str:
    """The adapter's own declared currency, not a constant.

    The FX conversion is real and dated; writing the source currency into the
    page as a literal would make this row silently wrong the first time a
    dataset in another currency was loaded.
    """
    from fwa.canonical import get_adapter
    try:
        return get_adapter(result.adapter_name.lower()).currency
    except Exception:
        return "the source currency"


def render() -> None:
    state = require_page("Overview", Permission.VIEW_OVERVIEW)
    session_banner(state)

    st.markdown("# Overview")
    banner()
    boundary_note()

    result = pipeline()
    summary = result.summary()
    registry = result.registry.summary()
    queue = result.queue()

    # ---- headline tiles ---------------------------------------------------
    st.markdown("## This run")
    tiles([
        ("Claims ingested", f"{summary['claims']:,}",
         "SHA-256 of every raw row stored before parsing"),
        ("Controls in the catalogue", f"{registry['total_controls']}",
         f"{registry['total_scenarios']} scenarios, the whole catalogue"),
        ("Controls that ran", f"{summary['controls_executed']}",
         f"{registry['by_data_support'].get('NOT_EXECUTABLE_ON_THIS_DATASET', 0)} cannot run on "
         f"this dataset — each with a stated reason"),
        ("Signals", f"{summary['signals']:,}", "A signal is not a fraud finding"),
        ("Cases", f"{summary['cases']:,}", "After correlation along the six dimensions"),
        ("Controls active", f"{registry['by_status']['active']}",
         "Every control is in SHADOW; none has been activated"),
    ])

    st.markdown(
        "<div class='fwa-note'>Nothing on this page has been reviewed by a human. "
        "Precision in the strict sense is <em>confirmed ÷ reviewed</em>; with nothing reviewed it "
        "is <code>NOT_MEASURABLE_ON_THIS_DATASET</code> and is reported as such throughout.</div>",
        unsafe_allow_html=True,
    )

    # ---- exposure ---------------------------------------------------------
    st.markdown("## Exposure — and why it is not savings")
    established = queue.loc[queue["exposure_established"], "exposure_aed"].sum()
    unestablished = queue.loc[~queue["exposure_established"], "exposure_aed"].sum()
    left, right = st.columns(2)
    with left:
        tiles([("Established exposure", f"AED {established:,.0f}",
                "An objective condition was shown to fail")])
    with right:
        tiles([("Exposure NOT established", f"AED {unestablished:,.0f}",
                "Gross amount only — “exposure not yet established”")])
    note(
        "These two figures are never added together, anywhere in this system. Gross flagged "
        "value is not savings: nothing here has been reviewed, confirmed, prevented or "
        "recovered, and reporting it as if it were would overstate the system's contribution — "
        "which this system treats as an ethical failure, not a rounding one."
    )

    # ---- queue shape ------------------------------------------------------
    st.markdown("## The queue")
    if queue.empty:
        empty_state("No cases in the queue",
                    "The pipeline produced no signals. Check the Governance page for "
                    "kill-switched controls or a breached alert-volume ceiling.")
    else:
        c1, c2 = st.columns(2)
        with c1:
            counts = queue["disposition"].value_counts().reset_index()
            counts.columns = ["disposition", "cases"]
            bar(counts, "disposition", "cases", title="Cases by disposition",
                colour="disposition", colour_map=disposition_colours(), horizontal=True, height=300)
        with c2:
            dims = queue["dimension_label"].value_counts().reset_index()
            dims.columns = ["correlation dimension", "cases"]
            bar(dims, "correlation dimension", "cases",
                title="Cases by correlation dimension", horizontal=True, height=300)

        bands = (
            queue.groupby(["priority_band", "priority_band_label"], as_index=False)
            .size().rename(columns={"size": "cases"}).sort_values("priority_band")
        )
        st.markdown("**Priority bands.** Priority orders the queue only; it never sets or "
                    "overrides a disposition.")
        st.markdown(
            " ".join(
                priority_chip(r.priority_band, r.priority_band_label, None) + f" {r.cases}"
                for r in bands.itertuples(index=False)
            ),
            unsafe_allow_html=True,
        )

    # ---- dataset profile --------------------------------------------------
    st.markdown("## Dataset profile")
    claims = result.claims
    profile = pd.DataFrame([
        {"attribute": "Rows", "value": f"{len(claims):,}"},
        {"attribute": "Distinct members", "value": f"{claims['member_sk'].nunique():,}"},
        {"attribute": "Distinct providers", "value": f"{claims['provider_sk'].nunique():,}"},
        {"attribute": "Distinct TPAs", "value": f"{claims['tpa'].nunique():,}"},
        {"attribute": "Distinct agents", "value": f"{claims['agent_id'].nunique():,}"},
        {"attribute": "Primary diagnoses", "value": f"{claims['diagnosis_primary'].nunique():,}"},
        {"attribute": "Service dates",
         "value": f"{pd.to_datetime(claims['service_date']).min():%d %b %Y} – "
                  f"{pd.to_datetime(claims['service_date']).max():%d %b %Y}"},
        {"attribute": "Source file", "value": result.source_name or "—"},
        {"attribute": "Source currency",
         "value": f"{_source_currency(result)}, converted to AED at the service-date rate"},
        {"attribute": "Adapter", "value": result.adapter_name},
    ])
    left, right = st.columns([1, 1.15], gap="large")
    with left:
        dataframe(profile, height=340)
    with right:
        st.markdown("**Canonical model population**")
        pop = result.dataset.population_report()
        pop_view = pop[["table", "status", "rows"]].copy()
        dataframe(pop_view, height=340)

    with st.expander("Why so many canonical tables are empty — and why that is the honest answer"):
        st.markdown(
            "A table that is empty because the source has no such data is a **finding**. A table "
            "filled with plausible invented rows would be a lie that propagates into every "
            "downstream metric, and would make the coverage matrix — which is a deliverable in "
            "its own right — untrue.\n\n"
            "The four structural gaps, in order of consequence:"
        )
        st.markdown(
            "1. **No `authorization` table.** The whole PAY-04 scenario is unrunnable, including "
            "PAY-04-R01 — the canonical worked example of the atomic-control contract.\n"
            "2. **Claim-header level only.** No activity codes, units or line amounts, so PAY-02, "
            "PAY-03, PAY-05, CLN-08 and most of PHR have no input at all.\n"
            "3. **No lineage and no remittance.** PAY-08 and PAY-12 are dead; "
            "prevented/recovered AED is NOT_MEASURABLE.\n"
            "4. **No geography.** Peer-hierarchy level 5 is NOT_POPULATED and every "
            "travel-impossibility control is classified out rather than run against a fabricated "
            "location."
        )
        dataframe(pop[["table", "status", "reason"]])

    # ---- catalogue coverage ----------------------------------------------
    st.markdown("## Catalogue coverage")
    support = registry["by_data_support"]
    cov = pd.DataFrame([
        {"classification": k, "controls": v,
         "share": f"{v / registry['total_controls']:.1%}"}
        for k, v in support.items()
    ])
    left, right = st.columns([1, 1.4], gap="large")
    with left:
        dataframe(cov)
        st.markdown(
            f"Only **{registry['controls_that_may_deny']}** of "
            f"{registry['total_controls']} controls may deny or reprice a claim — and every one "
            f"is type H or H/E. The registry refuses to load a statistical, network, document or "
            f"model control that declares `REJECT` or `REPRICE`."
        )
    with right:
        run = result.evaluation.to_frame()
        fired = run[run["signal_count"] > 0].sort_values("signal_count", ascending=False)
        bar(fired.head(14), "rule_id", "signal_count",
            title="Controls that produced signals", horizontal=True, height=400)

    # ---- pipeline status --------------------------------------------------
    st.markdown("## Pipeline run status")
    timing = pd.DataFrame([
        {"stage": k, "seconds": round(v, 2)} for k, v in result.timings.items()
    ])
    left, right = st.columns([1.1, 1], gap="large")
    with left:
        dataframe(timing)
    with right:
        st.markdown(
            f"**Reproducibility.** Seed `{summary['seed']}` (`cfg.random_seed`), "
            f"parameter-registry fingerprint `{summary['parameters_fingerprint']}`.\n\n"
            f"Signal identity is a pure function of (tenant, rule, version, subject, fact, "
            f"period), so replaying a run overwrites rather than appends — which is what the "
            f"idempotency test asserts for every control uniformly.\n\n"
            f"**AI layer:** provider `{result.ai.provider.name}`, deterministic "
            f"`{result.ai.provider.deterministic}`. The tool is fully functional with the AI "
            f"layer switched off entirely, and a governance test asserts it."
        )
        errors = result.evaluation.errors
        if errors:
            st.error(f"{len(errors)} control(s) raised during evaluation — see the Governance page.")
        else:
            st.markdown(status_chip("PASS") + " No control raised during evaluation.",
                        unsafe_allow_html=True)
