"""Models — score distributions, SHAP, drift, and the two gate verdicts.

Build brief §14, item 6: "IF/LOF score distributions, SHAP summary, drift panel
(PSI, alert volume, review yield), the **promotion gate verdict vs the
transparent composite**, and the supervised gate's ``PROMOTION_BLOCKED`` verdict
with its unmet prerequisites."

The two verdicts are the point of the page. A models page that shows only
scores invites the reader to treat the model as the product; showing the gate
first makes clear that the model is a candidate, and that the transparent
composite is what it has to beat.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from common import (
    banner, bar, dataframe, empty_state, histogram, mask, note, pipeline, require_page,
    session_banner, status_chip, tiles,
)
from fwa.auth import Permission
from fwa.evaluation import MetricSuite


def render() -> None:
    state = require_page("Models", Permission.VIEW_MODELS)
    session_banner(state)

    st.markdown("# Models")
    banner()

    result = pipeline()
    layer = result.models
    if layer is None or layer.anomaly is None or not layer.anomaly.models:
        empty_state("No model was trained",
                    "The temporal + entity-isolated split left an empty training or scoring set.")
        return

    metrics = MetricSuite(result)
    if not layer.gate_verdicts:
        capacity = int(result.config.get("alerts_per_reviewer_per_day")) * int(
            result.config.get("reviewer_count"))
        layer.run_gate(outcomes=metrics.provider_outcomes(), capacity=capacity)

    tab_gate, tab_scores, tab_shap, tab_drift, tab_clusters, tab_supervised = st.tabs([
        "Promotion gate", "Scores", "Explanation", "Drift", "Novel clusters", "Supervised gate",
    ])

    # ======================================================================
    with tab_gate:
        st.markdown("## Promotion gate")
        st.markdown(
            "A model reaches `active` only on **demonstrated prospective lift over ANL-01-R01**, "
            "the transparent composite, plus calibration, drift and explanation-quality checks. "
            "`NOT_ASSESSABLE` is never a pass."
        )
        for name, verdict in layer.gate_verdicts.items():
            st.markdown(
                f"<div class='fwa-card'><h4>{name} — "
                f"{status_chip('PROMOTED' if verdict.promoted else 'FAIL')}</h4>"
                f"<p>{verdict.summary}</p></div>",
                unsafe_allow_html=True,
            )
            dataframe(verdict.to_frame())

        st.markdown("### Lift over the transparent composite")
        lift = metrics.lift_over_composite()
        if lift.empty:
            empty_state("No lift computed", "No outcome series was available.")
        else:
            dataframe(lift)
            bar(lift, "layer", "review_yield_proxy",
                title="Review-yield proxy by layer, at team capacity", horizontal=True, height=280)
            note(
                "The promotion gate defines lift as FUTURE-PERIOD REVIEW YIELD, not in-sample "
                "separation — and "
                "review yield needs confirmed reviewer outcomes, which do not exist yet. The "
                "figures above use a held-out LABEL PROXY and are the closest this dataset "
                "permits. They are an upper bound, not the measurement the gate actually "
                "requires, and the gate records that on every verdict."
            )

    # ======================================================================
    with tab_scores:
        st.markdown("## Score distributions")
        dataframe(layer.anomaly.summary())
        coverage = layer.coverage()
        tiles([
            ("Split date", coverage.get("split_date", "—"), "strictly temporal"),
            ("Training claims", f"{coverage.get('train_claims', 0):,}", "earlier period only"),
            ("Scored claims", f"{coverage.get('score_claims', 0):,}", "later period only"),
            ("Provider overlap", f"{coverage.get('provider_overlap', 0)}",
             "entity isolation: zero is the requirement"),
        ])
        note(coverage.get("coverage_note", ""))

        for name, scores in layer.anomaly.models.items():
            histogram(scores.scores, title=f"{name} — score distribution on the scoring period",
                      vline=scores.threshold,
                      vline_label=f"capacity threshold {scores.threshold:.3f}")
            st.caption(
                f"The threshold is set by **reviewer capacity** "
                f"({result.config.get('alerts_per_reviewer_per_day')} alerts × "
                f"{result.config.get('reviewer_count')} reviewers), not by a fixed significance "
                f"level or by the contamination parameter."
            )

        st.markdown("### Highest-scoring entities")
        provider_scores = layer.provider_scores()
        rows = []
        for name, series in provider_scores.items():
            for provider, score in series.nlargest(10).items():
                rows.append({"model": name, "provider": mask(provider, state),
                             "score": round(float(score), 4)})
        dataframe(pd.DataFrame(rows))

    # ======================================================================
    with tab_shap:
        st.markdown("## Explanation")
        st.markdown(
            "the promotion gate requires top contributing features **in original units with peer comparison**, "
            "for every flagged entity — not bare SHAP magnitudes. A model whose contributions "
            "cannot be rendered that way fails the gate's explanation-quality check."
        )
        for name, explainer in layer.explainers.items():
            st.markdown(f"**{name}** — {explainer.method}")
            if not explainer.available:
                note(explainer.failure_reason or "No explainer available.")
                continue
            matrix = layer.anomaly.score_matrix
            importance = explainer.global_importance(matrix, sample=200)
            if not importance.empty:
                bar(importance.head(12).sort_values("mean_abs_shap"), "label", "mean_abs_shap",
                    title=f"{name} — mean |SHAP| by feature", horizontal=True, height=340)

        st.markdown("### A worked explanation")
        model_name = st.selectbox("Model", list(layer.anomaly.models))
        scores = layer.anomaly.models[model_name]
        candidates = list(scores.flagged_index)[:25]
        if not candidates:
            empty_state("No flagged claims to explain", "No claim reached the capacity threshold.")
        else:
            claim = st.selectbox("Claim", candidates)
            contributions = layer.explain_claim(model_name, str(claim), top_n=6)
            for c in contributions:
                st.markdown(f"- {c.get('plain_language', c.get('label'))}")
            dataframe(pd.DataFrame([
                {k: c.get(k) for k in ("label", "value_display", "peer_display", "ratio_to_peer",
                                       "shap_value", "direction")}
                for c in contributions if c.get("feature") not in
                ("EXPLANATION_UNAVAILABLE", "EXPLANATION_FAILED")
            ]))

    # ======================================================================
    with tab_drift:
        st.markdown("## Drift monitoring")
        from fwa.models.drift import DriftMonitor

        for name, results in layer.drift.items():
            st.markdown(f"### {name}")
            frame = pd.DataFrame([d.to_row() for d in results])
            dataframe(frame[["monitor", "value", "warn_threshold", "breach_threshold", "status"]])
            st.markdown(f"**{DriftMonitor.recommendation(results)}**")
            for d in results:
                if d.note:
                    note(d.note)
                if not d.detail.empty and d.monitor.startswith("Population"):
                    bar(d.detail.head(12).sort_values("psi"), "feature", "psi",
                        title="Highest-PSI features", horizontal=True, height=320)

    # ======================================================================
    with tab_clusters:
        st.markdown("## ANL-01-R04 — novel-cluster discovery")
        st.markdown(
            "Candidate emerging typologies for the quarterly typology review. Disposition "
            "is `MONITOR_ONLY` and cannot be anything else: **no production action until "
            "converted to a rule**, and that rule would itself enter the registry in shadow."
        )
        if not layer.clusters:
            empty_state("No clusters found",
                        "HDBSCAN found no coherent cluster of at least "
                        f"{result.config.get('novel_cluster_min_size')} entities in the residual "
                        "space. Noise points are excluded — treating noise as a typology is the "
                        "easiest way to manufacture a finding.")
        else:
            dataframe(pd.DataFrame([c.to_row() for c in layer.clusters]))
            picks = {c.candidate_typology_name: c for c in layer.clusters}
            chosen = st.selectbox("Cluster", list(picks))
            cluster = picks[chosen]
            dataframe(pd.DataFrame(cluster.top_features))
            note(cluster.note)
            if result.ai.triage is not None and st.button("Draft a candidate control (MONITOR_ONLY)"):
                proposal = result.ai.triage.propose(cluster, actor=state.username)
                from common import ai_panel

                ai_panel(proposal.mechanism,
                         footer=proposal.hypothesis_marker,
                         label="AI-generated proposal — not evidence")
                st.json(proposal.draft_control)

    # ======================================================================
    with tab_supervised:
        st.markdown("## Supervised-modelling gate")
        if layer.supervised is None:
            empty_state("The supervised gate did not run", "")
        else:
            st.markdown(
                f"### {status_chip(layer.supervised.status)}", unsafe_allow_html=True)
            st.markdown(layer.supervised.summary)
            dataframe(layer.supervised.to_frame())
            note(
                "Nothing is trained here. There is no `fit`, no estimator and no classifier "
                "imported anywhere in `fwa/models/supervised_gate.py`. That is the point being "
                "made: whether to build a supervised model is a governance question with a "
                "governance answer, not a modelling experiment with an AUC. The promotion gate "
                "treats fitting "
                "a propensity model to investigation-derived labels as DISQUALIFYING, not merely "
                "undesirable."
            )

        if state.can(Permission.PROPOSE_MODEL_PROMOTION):
            st.markdown("### Propose a promotion")
            st.caption(
                "An analyst may PROPOSE; only a policy owner may approve, and never the same "
                "person (separation of duties)."
            )
            with st.form("propose_promotion"):
                model = st.selectbox("Model", list(layer.anomaly.models))
                reason = st.text_area("Reason")
                if st.form_submit_button("Propose promotion"):
                    if len(reason.strip()) < 10:
                        st.error("A reason of at least 10 characters is required.")
                    else:
                        from common import auth_service

                        auth_service().propose_model_promotion(
                            state=state, model=model, reason=reason)
                        st.success(
                            "Proposal recorded to the audit log. A policy owner who did not "
                            "propose it must approve it, and the gate verdict above must pass "
                            "first."
                        )
