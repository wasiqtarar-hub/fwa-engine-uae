"""Provider Analytics — peers, shrinkage, change points, and the §4.6 demonstration.

Build brief §14, item 4: "peer comparison with shrinkage and posterior
intervals, CUSUM/EWMA change points with pre/post means and change date, the
**small-sample vs large-sample shrinkage demonstration**."

That last item is the reason this page exists in the form it does. §4.6's
argument for empirical-Bayes shrinkage is a worked example — two providers with
the same 20% observed rate and utterly different evidential weight — and an
artefact that implements shrinkage without showing that example has implemented
the arithmetic and skipped the point.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from common import (
    banner, bar, dataframe, empty_state, histogram, line, mask, mono, note, pipeline,
    require_page, session_banner, tiles,
)
from fwa.auth import Permission
from fwa.statistical import robust_stats
from fwa.statistical.changepoint import detect_changepoints, monthly_provider_metrics
from fwa.statistical.shrinkage import shrinkage_demonstration


def render() -> None:
    state = require_page("Provider Analytics", Permission.VIEW_PROVIDER_ANALYTICS)
    session_banner(state)

    st.markdown("# Provider analytics")
    banner()

    result = pipeline()
    config = result.config
    composite_results, composite_frame = result.composite

    tab_composite, tab_shrinkage, tab_change, tab_peers = st.tabs([
        "Transparent composite", "Shrinkage", "Change detection", "Peer hierarchy",
    ])

    # ======================================================================
    with tab_composite:
        st.markdown("## ANL-01-R01 — the transparent composite")
        st.markdown(
            "This is the score **every model must beat** and the system's explainable "
            "fallback when every model is switched off. It is a weighted sum of robust "
            "(median/MAD) residuals across scenario-aligned features, and it is reconstructable "
            "by hand from the weights in `config/parameters.yaml`."
        )
        threshold = float(config.get("composite_flag_threshold"))
        scored = composite_frame[~composite_frame["insufficient_evidence"]]
        flagged = scored[scored["composite_score"] > threshold]
        tiles([
            ("Providers scored", f"{len(scored):,}", ""),
            ("Above threshold", f"{len(flagged):,}", f"cfg.composite_flag_threshold = {threshold}"),
            ("Insufficient evidence", f"{int(composite_frame['insufficient_evidence'].sum()):,}",
             "reported as such, never flagged on a small-group estimate"),
        ])
        histogram(scored["composite_score"], title="Composite score distribution",
                  vline=threshold, vline_label=f"threshold {threshold}")
        note(
            "The threshold's position in this distribution is visible rather than asserted. "
            "the promotion gate requires an operational threshold calibrated to reviewer capacity, not to a "
            "fixed significance level."
        )

        view = flagged.copy()
        view["provider"] = view["provider_sk"].map(lambda p: mask(p, state))
        dataframe(view[["provider", "composite_score", "peer_level_used", "peer_n",
                        "claim_count", "exposure_aed", "top_features"]], height=320)

        st.markdown("### Decomposition")
        options = {mask(r.provider_sk, state): r.provider_sk for r in composite_results
                   if not r.insufficient_evidence}
        if options:
            chosen_label = st.selectbox("Provider", list(options))
            chosen = options[chosen_label]
            record = next(r for r in composite_results if r.provider_sk == chosen)
            st.markdown(f"> {record.explain()}")
            contributions = pd.DataFrame(record.contributions)
            dataframe(contributions[[
                "label", "unit", "value", "peer_median", "residual", "weight", "contribution",
                "scale_source", "scenario_alignment",
            ]])
            lines = [
                f"{c['weight']:+.2f} × residual {0.0 if c['residual'] is None else c['residual']:+.3f}"
                f" = {c['contribution']:+.4f} {c['label']}"
                for c in record.contributions
            ]
            lines.append(f"{'':>34}total = {record.score:+.4f}")
            mono("\n".join(lines))
            positive = contributions[contributions["contribution"] > 0]
            if not positive.empty:
                bar(positive.sort_values("contribution"), "label", "contribution",
                    title="Per-feature contribution", horizontal=True, height=300)

    # ======================================================================
    with tab_shrinkage:
        st.markdown("## Empirical-Bayes shrinkage")
        st.markdown(
            "> *“a provider with five claims and one adverse event has an observed rate of 20%, "
            "which is statistically almost meaningless, while a provider with five thousand "
            "claims and the same 20% rate is a genuine outlier.”*"
        )
        st.markdown("### The demonstration")
        demo = shrinkage_demonstration(
            max_posterior_width=float(config.get("max_posterior_width")),
            interval_mass=float(config.get("posterior_interval_mass")),
        )
        for row in demo.itertuples(index=False):
            st.markdown(
                f"<div class='fwa-card'><h4>{row.entity}</h4>"
                f"<p>{row.explanation}</p>"
                f"<p class='fwa-sub'>observed {row.observed_rate:.1%} → shrunk "
                f"{row.shrunk_rate:.1%} · interval width {row.posterior_width:.3f} · "
                f"{'EXCLUDED FROM RANKING' if row.excluded_from_ranking else 'ranked'}</p>"
                f"</div>",
                unsafe_allow_html=True,
            )
        fig_frame = demo[["entity", "observed_rate", "shrunk_rate", "posterior_low",
                          "posterior_high"]].copy()
        dataframe(fig_frame.round(4))
        note(
            "Both providers show the SAME observed rate. The five-claim provider is pulled far "
            "toward the peer mean and carries a wide interval; the five-thousand-claim provider "
            "barely moves. That difference is the entire argument for the estimator — and it is "
            "why the peer design requires every shrunk rate to be reported WITH its posterior interval, and "
            "entities whose intervals are too wide to be EXCLUDED FROM RANKING rather than "
            "flagged on a falsely precise point estimate."
        )

        st.markdown("### Shrinkage on this dataset")
        signals = [s for s in result.signals if s.evidence.get("shrunk_rate") is not None]
        if not signals:
            empty_state("No shrunk rates in this run",
                        "No statistical control produced a shrunk rate. Check the Governance "
                        "page for kill-switched controls.")
        else:
            rows = [{
                "provider": mask(s.subject_id, state),
                "rule_id": s.rule_id,
                "observed_rate": s.evidence.get("observed_rate"),
                "shrunk_rate": s.evidence.get("shrunk_rate"),
                "posterior_interval": str(s.evidence.get("posterior_interval")),
                "peer_mean": s.evidence.get("peer_mean"),
                "prior_strength": s.evidence.get("prior_strength"),
                "claims": s.evidence.get("claim_count"),
            } for s in signals]
            dataframe(pd.DataFrame(rows), height=320)

    # ======================================================================
    with tab_change:
        st.markdown("## Change detection")
        st.markdown(
            "CUSUM (Page, 1954) and EWMA on **monthly normalised** provider metrics. A change "
            "point may not be declared before `cfg.min_history_periods` periods of history, and "
            "every declaration stores its pre-change mean, post-change mean, change date and "
            "confidence — so a reviewer sees *what* changed and *when*, not a binary alert."
        )
        series = monthly_provider_metrics(result.claims)
        changepoints = detect_changepoints(
            series, entity_column="provider_sk", period_column="period",
            value_column="mean_gross_aed", metric_name="mean claim value",
            min_history_periods=int(config.get("min_history_periods")),
            cusum_k=float(config.get("cusum_k")), cusum_h=float(config.get("cusum_h")),
            ewma_lambda=float(config.get("ewma_lambda")), ewma_L=float(config.get("ewma_L")),
        )
        tiles([
            ("Change points declared", f"{len(changepoints):,}", "on mean claim value"),
            ("Minimum history", f"{config.get('min_history_periods')} periods",
             "cfg.min_history_periods — a declaration inside this window is suppressed"),
            ("CUSUM decision interval", f"h = {config.get('cusum_h')}",
             "calibrated to the ~18-period series length available here"),
        ])
        if not changepoints:
            empty_state("No change points declared",
                        "With a median of 25 claims per provider across 38 months, a monthly "
                        "provider series is sparse. That is a data-density finding, not evidence "
                        "that no provider changed behaviour.")
        else:
            frame = pd.DataFrame([c.to_dict() for c in changepoints])
            frame["entity"] = frame["entity"].map(lambda p: mask(p, state))
            dataframe(frame[["entity", "metric", "method", "change_date", "pre_change_mean",
                             "post_change_mean", "change_magnitude", "direction", "confidence",
                             "periods_of_history"]], height=300)

            picks = {mask(c.entity, state): c for c in changepoints}
            label = st.selectbox("Inspect a provider's series", list(picks))
            cp = picks[label]
            sub = series[series["provider_sk"] == cp.entity].sort_values("period")
            line(sub, "period", "mean_gross_aed",
                 title=f"{label} — mean claim value by month")
            st.markdown(f"> {cp.explain()}")
            note(
                "the change-detection design requires known structural changes — a tariff revision, a new contract, an "
                "ownership change — to be explicitly segmented before a change point is "
                "attributed to provider behaviour. the claim extract contains none of those records, so "
                "no segment can be declared and every change point here carries that caveat."
            )

    # ======================================================================
    with tab_peers:
        st.markdown("## The six-level peer hierarchy")
        st.markdown(
            "A group too sparse to support a stable estimate **backs off** to its parent level, "
            "and `peer_level_used` is always recorded — reproducibility depends on it."
        )
        dataframe(result.context.peers.level_availability())
        note(
            "Level 5 (geography) is NOT_POPULATED: the claim extract contains no geographic field of "
            "any kind. It is skipped rather than approximated, and every control that depends on "
            "geography is classified NOT_EXECUTABLE rather than run against a fabricated "
            "location."
        )
        used = pd.Series([s.peer_level_used for s in result.signals if s.peer_level_used])
        if not used.empty:
            counts = used.value_counts().reset_index()
            counts.columns = ["peer_level_used", "signals"]
            bar(counts, "peer_level_used", "signals",
                title="Which peer level actually produced each signal", horizontal=True)
