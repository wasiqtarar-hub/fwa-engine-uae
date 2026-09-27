"""Pattern-finding models — what the two models found, why, and whether they can be trusted.

Build brief §14, item 6: score distributions, the reasons behind a score, drift,
the **promotion gate verdict against the simple scorecard** (the transparent
composite) and the supervised gate's verdict with its unmet prerequisites.

The page leads with the verdict, in words: a models page that shows only scores
invites the reader to treat the model as the product, so the first thing a
reader sees is that the models are candidates, that about half the claims are
deliberately not scored so the test stays fair, and that neither model is ready
to be trusted for real decisions unless all five gate questions are answered yes.

Presentation only. Internal names (``isolation_forest``, gate criterion names,
``PROMOTION_BLOCKED``…) are translated at display time and never renamed. The
optional exploratory scores (every claim, including providers the model learned
from) are shown under a banner and are never passed to the gate, the lift
comparison or any metric on this page.
"""

from __future__ import annotations

import html

import pandas as pd
import streamlit as st

from common import (
    banner, bar, dataframe, empty_state, friendly_failure,
    friendly_table, headline_cards, help_text, histogram, mask, page_header, pipeline,
    render_model_claim, require_page, session_banner, status_badge, synthetic_banner,
)
from components.analysis_pages import (
    checklist_html, drift_rows, gate_checklist, overall_verdict, words,
)
from fwa.auth import Permission
from fwa.presentation import (
    aed, feature_phrase, feature_unit, model_label, pct, plain_date, standard_text,
)
from fwa.presentation.explain_case import explain_model_claim

EXPLORATORY_NOTICE = (
    "Exploratory scores. The model has seen some of these hospitals during training, so these "
    "scores are not used for the promotion gate."
)

#: Mean-|SHAP| tables are expensive (the Local Outlier Factor needs a sampling
#: explainer), so each is computed once per fitted layer, not once per rerun.
_IMPORTANCE_CACHE: dict[tuple[int, str], tuple[object, pd.DataFrame]] = {}


def render() -> None:
    state = require_page("Models", Permission.VIEW_MODELS)
    session_banner(state)
    page_header("Models")
    banner()
    synthetic_banner()

    result = pipeline()
    config = result.config
    layer = result.models
    per_day = int(config.get("alerts_per_reviewer_per_day"))
    reviewers = int(config.get("reviewer_count"))
    capacity = max(per_day * reviewers, 1)

    models = {}
    if layer is not None and getattr(layer, "anomaly", None) is not None:
        models = dict(getattr(layer.anomaly, "models", {}) or {})
    messages = [str(m) for m in (getattr(layer, "messages", None) or []) if str(m).strip()]

    if models and not layer.gate_verdicts:
        with friendly_failure("the promotion checklist",
                              "The models ran, but the checklist could not be worked out. Try "
                              "another dataset, or ask an administrator to check the log."):
            from fwa.evaluation import MetricSuite

            layer.run_gate(outcomes=MetricSuite(result).provider_outcomes(), capacity=capacity)

    tab_names = ["Summary", "Flagged claims and their reasons", "Is it still reliable?",
                 "New patterns to study", "Why no model learns from past outcomes"]
    can_propose = state.can(Permission.PROPOSE_MODEL_PROMOTION)
    if can_propose:
        tab_names.append("Propose a model for use")
    tabs = st.tabs(tab_names)

    with tabs[0]:
        _summary(result, layer, models, messages, capacity, per_day, reviewers)
    with tabs[1]:
        with friendly_failure("the flagged claims"):
            _claims(result, layer, models, state, capacity)
    with tabs[2]:
        with friendly_failure("the reliability checks"):
            _drift(layer, models, config)
    with tabs[3]:
        with friendly_failure("the new patterns"):
            _clusters(result, layer, state, config)
    with tabs[4]:
        with friendly_failure("the supervised-model checklist"):
            _supervised(layer)
    if can_propose:
        with tabs[5]:
            _propose(layer, models, state)


# =============================================================================
# Summary
# =============================================================================


def _explain_models() -> None:
    plain, more, gloss = words("model_plain"), words("model_more"), words("model_glossary")
    cols = st.columns(2)
    for col, name in zip(cols, ("isolation_forest", "local_outlier_factor")):
        with col:
            st.markdown(f"**{model_label(name)}**", help=help_text(gloss.get(name, name)))
            st.markdown(plain.get(name, ""))
            if more.get(name):
                st.caption(more[name])
    st.markdown(
        "<div class='fwa-caveat'><strong>What a score does not mean.</strong> A high score means a "
        "claim is <em>unusual</em>, not that anything is wrong with it. "
        f"{html.escape(standard_text('not_proof'))} {html.escape(standard_text('shadow'))}</div>",
        unsafe_allow_html=True,
    )


def _summary(result, layer, models, messages, capacity, per_day, reviewers) -> None:
    st.markdown("### What the two pattern-finders do")
    _explain_models()

    for m in messages:
        st.info(m, icon="ℹ️")

    if not models:
        if not messages:
            st.info(
                "Neither pattern-finder could run on this file. They need enough claims spread over "
                "time and across enough providers to learn from one group and be tested fairly on "
                "another. Load a larger file on the Data page to see them work.", icon="ℹ️")
        return

    coverage = layer.coverage() or {}
    st.markdown("### What each model did")
    train = int(coverage.get("train_claims", 0) or 0)
    scored_total = int(coverage.get("score_claims", 0) or 0)
    total = int(coverage.get("total_claims", 0) or 0)
    excluded = int(coverage.get("excluded_claims", max(total - train - scored_total, 0)) or 0)
    split = plain_date(coverage.get("split_date"), missing="the split date")
    for name, scores in models.items():
        flagged = len(scores.flagged_index)
        st.markdown(f"**{model_label(name)}**")
        headline_cards([
            (f"{int(getattr(scores, 'training_rows', 0) or train):,}",
             "claims it learned from: earlier claims from one half of the providers"),
            (f"{len(scores.scores):,}",
             "claims it scored: later claims from the other half of the providers"),
            (f"{flagged:,}",
             "claims it flagged: the most unusual ones, as many as the review team can handle"),
        ])

    if total:
        st.markdown(
            f"**Why about half the claims have no score.** To keep the test fair, both models learned "
            f"from {train:,} claims dated before {split} at {coverage.get('train_providers', 0):,} "
            f"providers, then scored {scored_total:,} claims dated from {split} onwards at the other "
            f"{coverage.get('score_providers', 0):,} providers. The remaining {excluded:,} of "
            f"{total:,} claims ({pct(excluded / max(total, 1))}) are scored by neither model: they are "
            f"earlier claims from the providers being scored, or later claims from the providers it "
            f"learned from. A model that scored everything would partly be marking its own homework. "
            f"You can still see a score for every claim with the exploratory switch on the next tab.",
            help=help_text("temporal_split"),
        )
    st.markdown(
        f"**Why that many are flagged.** Each model flags its {capacity:,} most unusual claims "
        f"because that is how many the review team can handle ({per_day} a day × {reviewers} "
        f"reviewers). The cut-off follows team size, not a fixed rule.",
        help=help_text("capacity"),
    )

    st.markdown("### Is it ready to be trusted?")
    st.caption("Five questions every model must answer yes to before it may be used for real "
               "decisions. ✅ yes · ❌ no · ❓ can't be judged yet, which never counts as yes.",
               help=help_text("promotion_gate"))
    if not layer.gate_verdicts:
        empty_state("No checklist available",
                    "The promotion checklist was not worked out for this run.")
    for name, verdict in layer.gate_verdicts.items():
        icon, headline, sentence = overall_verdict(verdict)
        st.markdown(
            f"<div class='fwa-card'><h4>{html.escape(model_label(name))}: {icon} "
            f"{html.escape(headline)}</h4><p>{html.escape(sentence)}</p>"
            f"{checklist_html(gate_checklist(verdict, capacity=capacity, drift=layer.drift.get(name, [])))}"
            f"</div>",
            unsafe_allow_html=True,
        )
        with st.expander(f"Technical details: {model_label(name)} gate evidence", expanded=False):
            dataframe(verdict.to_frame())
            st.caption(verdict.summary)

    with st.expander("How the comparison with the simple scorecard was measured (technical)",
                     expanded=False):
        _lift(result)

    with st.expander("Technical details: model settings", expanded=False):
        gloss = words("hyperparameters")
        rows = []
        for name, scores in models.items():
            for key, value in (scores.params or {}).items():
                rows.append({"Model": model_label(name), "Setting": key, "Value": str(value),
                             "What it means": gloss.get(key, "")})
        dataframe(pd.DataFrame(rows))
        st.caption("Hyperparameters as passed to scikit-learn. The number of claims flagged is set by "
                   "reviewer capacity, not by the contamination setting.")
        dataframe(layer.anomaly.summary())
        st.json(coverage)
        for note in getattr(layer, "notes", []) or []:
            st.caption(note)


def _lift(result) -> None:
    from fwa.evaluation import MetricSuite

    lift = MetricSuite(result).lift_over_composite()
    if lift.empty:
        st.markdown("We can't measure this yet: no outcomes are available to compare the methods with.")
        return
    names = {"deterministic + statistical rules": "Rules and comparisons"}

    def plain(layer_name: str) -> str:
        if "transparent composite" in layer_name:
            return "Simple scorecard"
        if layer_name.startswith("model: "):
            return model_label(layer_name.split("model: ", 1)[1])
        return names.get(layer_name, layer_name)

    view = lift.copy()
    view["method"] = view["layer"].map(plain)
    best = view.sort_values("review_yield_proxy", ascending=False).iloc[0]
    bar(view.sort_values("review_yield_proxy"), "method", "review_yield_proxy",
        title=f"{best['method']} had the highest share of real issues in its top cases",
        horizontal=True, height=260, x_label="Method",
        y_label="Share of top cases that were real issues (0–1)",
        how_to_read="Each bar is one method's share of real issues among the cases the team can "
                    "review. Longer is better. The figures use held-out investigation labels, so they "
                    "are a best case, not the review yield the gate really needs.")
    dataframe(lift)
    st.caption(
        "The gate defines lift as future-period review yield, which needs confirmed reviewer "
        "outcomes that do not exist yet. The figures above use a held-out label proxy: an upper "
        "bound, not the measurement the gate requires. Exploratory scores are never used here."
    )


# =============================================================================
# Flagged claims
# =============================================================================


def _claims(result, layer, models, state, capacity) -> None:
    if not models:
        empty_state("No claims were scored",
                    "Neither pattern-finder could run on this file, so there are no scores to show.")
        return

    exploratory = st.toggle(
        "Exploratory: score every claim", value=False, key="models_exploratory",
        help="Off (recommended): only the claims in the fair test are shown — later claims from "
             "providers the model never learned from. On: the same models score every claim, "
             "including providers they learned from. Turn it on to look up a claim that has no "
             "score; never use these scores to judge the model. They are kept out of the "
             "promotion checklist and every figure on this page.",
    )
    name = st.selectbox(
        "Which pattern-finder", list(models), format_func=model_label, key="models_claims_model",
        help="Choose whose scores to show. The two models look for different kinds of unusual; "
             "comparing them shows whether a claim is unusual in more than one way.",
    )
    scores = models[name]
    governed = scores.scores

    series = governed
    if exploratory:
        fn = getattr(layer, "exploratory_scores", None)
        if callable(fn):
            try:
                series = fn(name)
            except Exception:
                series = None
        else:
            series = None
        notice = getattr(layer, "exploratory_notice", None) or EXPLORATORY_NOTICE
        if series is None or len(series) == 0:
            st.info("Exploratory scores are not available for this model on this file, so the fair-"
                    "test scores are shown instead.", icon="ℹ️")
            series, exploratory = governed, False
        else:
            head, _, tail = notice.partition(". ")
            st.markdown(f"<div class='fwa-banner'><strong>{html.escape(head)}.</strong> "
                        f"{html.escape(tail)}</div>", unsafe_allow_html=True)

    series = pd.Series(series).dropna()
    label = model_label(name)
    if exploratory:
        histogram(series.values,
                  title=f"Exploratory: {label} scores for all {len(series):,} claims",
                  x_label="How unusual (higher = more unusual)", y_label="Number of claims",
                  how_to_read="Most claims sit on the left. Claims far to the right are the most "
                              "unusual. These exploratory scores include providers the model learned "
                              "from, so no cut-off line is drawn and nothing here counts towards the "
                              "checklist.")
    else:
        histogram(series.values,
                  title=f"{label} scored {len(series):,} claims; the {len(scores.flagged_index):,} "
                        f"most unusual are flagged",
                  vline=scores.threshold, vline_label="flag line (team capacity)",
                  x_label="How unusual (higher = more unusual)", y_label="Number of claims",
                  how_to_read="Most claims sit on the left. Claims to the right of the dashed line "
                              f"are the {len(scores.flagged_index):,} most unusual — as many as the "
                              "team can review — and are flagged.")

    # ---- the claim list -----------------------------------------------------
    st.markdown("#### The most unusual claims")
    top = series.sort_values(ascending=False).head(25)
    claims = result.claims.copy()
    claims.index = claims["claim_sk"].astype(str)
    rank = series.rank(pct=True)
    flagged = set(map(str, scores.flagged_index))
    rows = []
    for claim_sk, score in top.items():
        c = claims.loc[str(claim_sk)] if str(claim_sk) in claims.index else None
        rows.append({
            "Claim": str(claim_sk),
            "Provider": mask(c["provider_sk"], state) if c is not None else "—",
            "Date of service": plain_date(c["service_date"]) if c is not None else "—",
            "Amount billed": aed(c["gross_amount_aed"]) if c is not None else "—",
            "How unusual": f"top {max(1, round((1 - float(rank[claim_sk])) * 100))}%",
            "Flagged": ("Not used: exploratory" if exploratory and str(claim_sk) not in flagged
                        else "Yes" if str(claim_sk) in flagged else "No"),
            "_score": round(float(score), 4),
        })
    frame = pd.DataFrame(rows)
    friendly_table(frame.rename(columns={"_score": "Raw score"}), key=f"models_top_{name}",
                   columns=["Claim", "Provider", "Date of service", "Amount billed", "How unusual",
                            "Flagged"],
                   empty_title="No scored claims", empty_body="This model scored no claims on this file.")

    with st.expander("Providers with the most unusual claims (fair-test scores only)", expanded=False):
        provider_scores = layer.provider_scores().get(name)
        if provider_scores is None or provider_scores.empty:
            st.markdown("No provider scores are available for this model.")
        else:
            top_p = provider_scores.nlargest(10)
            prank = provider_scores.rank(pct=True)
            friendly_table(pd.DataFrame([{
                "Provider": mask(p, state),
                "Most unusual claim": f"top {max(1, round((1 - float(prank[p])) * 100))}% of providers",
                "Highest raw score": round(float(s), 4),
            } for p, s in top_p.items()]), key=f"models_prov_{name}",
                columns=["Provider", "Most unusual claim"])
            st.caption("A provider's score is the score of its most unusual claim in the fair test. "
                       "This is the ranking the promotion checklist compares with the scorecard.")

    # ---- reasons behind one score ---------------------------------------------
    st.markdown("#### Reasons behind a score", help=help_text("shap"))
    candidates = [str(c) for c in (top.index if exploratory else list(scores.flagged_index)[:25])]
    if not candidates:
        empty_state("No flagged claims to explain", "No claim reached the flag line.")
    else:
        claim = st.selectbox(
            "Pick a claim to see why it was ranked unusual", candidates, key=f"models_claim_{name}",
            help="The explanation lists the few facts about this claim that pushed its score up the "
                 "most, each compared with what is typical. Pick another claim to compare.",
        )
        expl = explain_model_claim(layer, name, claim, exploratory=exploratory)
        if expl is None and exploratory:
            expl = _exploratory_explanation(layer, name, claim, series)
        if expl is None and exploratory:
            st.info("This claim was not part of the fair test, so the model's reasons for its "
                    "exploratory score are not worked out. Its rank above is for exploring only.",
                    icon="ℹ️")
        else:
            render_model_claim(expl)
            if expl is not None and expl.technical:
                with st.expander("Technical details: contributions to this score", expanded=False):
                    tech = pd.DataFrame(expl.technical)
                    tech.insert(1, "in words", [feature_phrase(f) for f in tech["feature"]])
                    dataframe(tech)
                    st.caption("SHAP values on the robust-scaled feature matrix; peer median is the "
                               "median over all claims.")

    # ---- global importance -----------------------------------------------------
    st.markdown("#### What the model pays most attention to overall")
    explainer = layer.explainers.get(name)
    if explainer is None or not getattr(explainer, "available", False):
        reason = getattr(explainer, "failure_reason", "") if explainer is not None else ""
        st.info("The reasons behind this model's scores could not be worked out on this file, so "
                "this chart is not available." + (" " if reason else ""), icon="ℹ️")
        if reason:
            with st.expander("Technical details", expanded=False):
                st.code(reason)
        return
    importance = _importance(layer, name, explainer)
    if importance.empty:
        st.info("The overall picture could not be worked out for this model on this file.", icon="ℹ️")
        return
    view = importance.head(10).copy()
    view["what"] = [_short(feature_phrase(f)) for f in view["feature"]]
    bar(view.sort_values("mean_abs_shap"), "what", "mean_abs_shap",
        title=f"{label} relies most on {_lower(view.iloc[0]['what'])}", horizontal=True, height=380,
        x_label="Claim detail", y_label="Average influence on the score",
        how_to_read="Longer bars are the claim details that move this model's scores the most, on "
                    "average across the claims it scored. Influence is not the same as wrongdoing.")
    with st.expander("Technical details: mean absolute SHAP by feature", expanded=False):
        dataframe(importance)


def _exploratory_explanation(layer, name, claim, series):
    """Reasons for a claim that only has an exploratory score (outside the fair test)."""
    fn = getattr(layer, "explain_exploratory_claim", None)
    if not callable(fn):
        return None
    try:
        rows = fn(name, str(claim), top_n=5) or []
    except Exception:
        return None
    from fwa.presentation.explain_case import ModelClaimExplanation

    reasons = []
    for r in rows:
        feat = str(r.get("feature") or "")
        if not feat or feat.startswith("EXPLANATION_") or (r.get("shap_value") or 0) <= 0:
            continue
        value, typical = r.get("value_display"), r.get("peer_display")
        if feature_unit(feat) == "flag" or feat.endswith("__missing"):
            try:
                on = float(r.get("value")) >= 0.5
            except (TypeError, ValueError):
                continue
            phrase = feature_phrase(feat)
            reasons.append(f"{phrase[:1].upper() + phrase[1:]}." if on else f"The absence of {phrase}.")
            if len(reasons) >= 3:
                break
            continue
        if value in (None, "not available"):
            continue
        reasons.append(f"{_short(feature_phrase(feat), 400)} was {value}"
                       + (f", compared with a typical {typical}." if typical not in (None, "not available") else "."))
        if len(reasons) >= 3:
            break
    rank = float(series.rank(pct=True).get(str(claim), float("nan")))
    top = max(1, round((1 - rank) * 100)) if rank == rank else None
    headline = (f"Exploratory only: an automated pattern-finder ({model_label(name)}) ranked this claim "
                + (f"among the top {top}% of all claims" if top else "as unusual")
                + (f". The main reason: {_lower(reasons[0].rstrip('.'))}." if reasons else "."))
    caveats = ["Exploratory score: the model has seen some of this provider's claims during training, "
               "so this score is not used for the promotion gate.",
               standard_text("model_only"), standard_text("not_proof")]
    return ModelClaimExplanation(
        claim_sk=str(claim), model=name, headline=headline, reasons=reasons,
        caveats=[c for c in caveats if c], percentile=rank if rank == rank else None,
        technical=[{"feature": r.get("feature"), "value": r.get("value"),
                    "peer_median": r.get("peer_median"), "shap_value": r.get("shap_value")} for r in rows],
        exploratory=True,
    )


def _short(text: str, n: int = 70) -> str:
    text = text[:1].upper() + text[1:]
    return text if len(text) <= n else text[: n - 1].rstrip() + "…"


def _lower(text: str) -> str:
    return text[:1].lower() + text[1:]


def _importance(layer, name, explainer) -> pd.DataFrame:
    key = (id(layer), name)
    cached = _IMPORTANCE_CACHE.get(key)
    if cached is not None and cached[0] is layer:
        return cached[1]
    matrix = layer.anomaly.score_matrix
    try:
        frame = explainer.global_importance(matrix, sample=200) if matrix is not None else pd.DataFrame()
    except Exception:
        frame = pd.DataFrame()
    _IMPORTANCE_CACHE[key] = (layer, frame)
    return frame


# =============================================================================
# Drift
# =============================================================================


def _drift(layer, models, config) -> None:
    warn = float(config.get("psi_warn_threshold"))
    breach = float(config.get("psi_breach_threshold"))
    st.markdown(
        "A model learns what \"usual\" looks like from past claims. If the claims it scores now look "
        "different, its sense of unusual may be out of date. The main measure is the **population "
        f"stability index (PSI)**: a single number for how much the claims have shifted. Below "
        f"{warn:g} means little change, {warn:g} to {breach:g} is worth watching, and above "
        f"{breach:g} is a big change.",
        help=help_text("psi_drift"),
    )
    if not models or not getattr(layer, "drift", None):
        empty_state("No reliability checks ran", "The models did not run on this file, so there is "
                                                 "nothing to check.")
        return
    from fwa.models.drift import DriftMonitor

    for name, results in layer.drift.items():
        st.markdown(f"### {model_label(name)}")
        rows = drift_rows(results, warn=warn, breach=breach)
        st.markdown("\n".join(
            f"- {r['icon']} **{r['question']}** {r['label']}. {r['sentence']}" for r in rows))
        worst = "BREACH" if any(r.status == "BREACH" for r in results) else (
            "WARN" if any(r.status == "WARN" for r in results) else "OK")
        advice = {
            "BREACH": "**What to do:** recalibrate the model, or switch it off if the change "
                      "persists. Neither happens automatically: a named owner must decide.",
            "WARN": "**What to do:** keep a close eye on it and schedule a recalibration review.",
            "OK": "**What to do:** nothing for now.",
        }[worst]
        st.markdown(f"{status_badge(worst)} {advice}", unsafe_allow_html=True)
        for d in results:
            detail = getattr(d, "detail", None)
            if d.monitor.startswith("Population") and detail is not None and not detail.empty \
                    and "feature" in detail.columns:
                view = detail.head(10).copy()
                view["what"] = [_short(feature_phrase(f)) for f in view["feature"]]
                bar(view.sort_values("psi"), "what", "psi",
                    title=f"The biggest shift is in {_lower(view.iloc[0]['what'])}",
                    horizontal=True, height=340, x_label="Claim detail",
                    y_label="How much it shifted (PSI)",
                    how_to_read=f"Each bar is one claim detail. Past {warn:g} is worth watching; "
                                f"past {breach:g} is a big change between the claims the model "
                                "learned from and the claims it scored.")
        with st.expander(f"Technical details: {model_label(name)} drift monitors", expanded=False):
            dataframe(pd.DataFrame([d.to_row() for d in results]))
            st.caption(DriftMonitor.recommendation(results))


# =============================================================================
# Novel clusters
# =============================================================================


def _clusters(result, layer, state, config) -> None:
    st.markdown(
        "The engine also looks for **groups of providers that differ from their peers in the same "
        "way**. Each group is a possible new pattern to study at the quarterly review. It is an "
        "untested idea, never a finding: it can only ever be watched, and nothing happens to any "
        "claim until a person turns it into a check, which itself starts in watch-only mode.",
        help=help_text("clustering"),
    )
    clusters = list(getattr(layer, "clusters", []) or []) if layer is not None else []
    if not clusters:
        empty_state(
            "No new patterns found",
            f"No group of at least {config.get('novel_cluster_min_size')} providers differed from "
            "their peers in the same way. Providers that fit no group are left out on purpose: "
            "treating stragglers as a pattern is the easiest way to invent a finding.")
        return
    rows = []
    for i, c in enumerate(clusters, start=1):
        diffs = []
        for f in c.top_features[:3]:
            diffs.append(f"{feature_phrase(f['feature'])}: {'higher' if f['mean_residual'] > 0 else 'lower'}")
        rows.append({
            "Pattern": f"Pattern {i}",
            "Providers": c.size,
            "What they share": "; ".join(diffs),
            "Total billed by them": aed(c.aggregate_exposure_aed),
            "How consistent": pct(c.coherence),
            "Examples": ", ".join(mask(m, state) for m in c.members[:5]) + ("…" if len(c.members) > 5 else ""),
        })
    friendly_table(pd.DataFrame(rows), key="models_clusters")
    st.caption("\"Total billed\" is everything these providers billed in the scored period; it is not "
               "money at risk and not money saved.")
    picks = {f"Pattern {i}": c for i, c in enumerate(clusters, start=1)}
    chosen = st.selectbox("Look at one pattern", list(picks), key="models_cluster_pick",
                          help="Shows how the providers in this pattern differ from similar providers.")
    cluster = picks[chosen]
    st.markdown("\n".join(
        f"- {_short(feature_phrase(f['feature']), 200)}: "
        f"{'higher' if f['mean_residual'] > 0 else 'lower'} than at similar providers "
        f"({abs(f['mean_residual']):.1f} typical spreads {'above' if f['mean_residual'] > 0 else 'below'})"
        for f in cluster.top_features))
    with st.expander("Technical details: cluster record", expanded=False):
        dataframe(pd.DataFrame([cluster.to_row()]))
        dataframe(pd.DataFrame(cluster.top_features))
        st.caption(cluster.note)
    if result.ai.triage is not None and st.button(
            "Draft a candidate check for this pattern (watch-only)", key="models_draft",
            help="Asks the offline assistant to draft a check that could test this pattern. The draft "
                 "is a suggestion, not evidence; it would start in watch-only mode and need a "
                 "separate person's approval."):
        proposal = result.ai.triage.propose(cluster, actor=state.username)
        from common import ai_panel

        ai_panel(proposal.mechanism, footer=proposal.hypothesis_marker,
                 label="AI-generated proposal — not evidence")
        with st.expander("Technical details: the drafted check", expanded=False):
            st.json(proposal.draft_control)


# =============================================================================
# Supervised gate
# =============================================================================


def _supervised(layer) -> None:
    st.markdown(
        "A **supervised model** would learn from past investigation results to predict new ones. "
        "None is trained here, on purpose: if the past results only cover claims someone chose to "
        "investigate, the model would learn to repeat those choices and miss everything nobody "
        "looked at.",
        help=help_text("supervised_model"),
    )
    sup = getattr(layer, "supervised", None) if layer is not None else None
    if sup is None:
        empty_state("This checklist did not run", "The model layer did not run on this file.")
        return
    blocked = sup.status != "PROMOTION_PERMITTED"
    st.markdown(f"#### Should a model be trained on past outcomes? **{'Not yet' if blocked else 'It may be proposed'}.**")
    questions = words("supervised_prereqs")
    lines = []
    for p in sup.prerequisites:
        q = questions.get(p.name, p.name)
        line = f"- {'✅' if p.met else '❌'} **{q}** {'Yes.' if p.met else 'No.'}"
        if not p.met:
            line += f" What would change this: {p.what_would_satisfy_it}"
        lines.append(line)
    st.markdown("\n".join(lines))
    unmet = len(sup.unmet)
    st.markdown(
        f"{unmet} of {len(sup.prerequisites)} conditions are not met, so "
        "no supervised model is trained. Building one on these labels is treated as disqualifying, "
        "not merely undesirable." if blocked else
        "All conditions are met. A supervised model may be proposed, "
        "and would still start in watch-only mode."
    )
    with st.expander("Technical details: prerequisite evidence", expanded=False):
        dataframe(sup.to_frame())
        st.caption(sup.summary)
        st.caption("Nothing is trained here: there is no fit, no estimator and no classifier "
                   "imported anywhere in the supervised-gate module.")


# =============================================================================
# Promotion proposal (propose only; separation of duties)
# =============================================================================


def _propose(layer, models, state) -> None:
    st.markdown(
        "An analyst may **propose** a pattern-finder for real use. Only a policy owner may approve "
        "it, and never the person who proposed it. A proposal changes nothing on its own: it is "
        "written to the audit log, and the five-question checklist must be all yes before approval."
    )
    names = list(models) or ["isolation_forest", "local_outlier_factor"]
    verdicts = getattr(layer, "gate_verdicts", {}) or {}
    with st.form("propose_promotion"):
        model = st.selectbox(
            "Which pattern-finder to propose", names, format_func=model_label,
            help="The model you think is ready for real use. Check its answers on the Summary tab "
                 "first: a model with any ❌ or ❓ cannot be approved.")
        reason = st.text_area(
            "Why you are proposing it (at least 10 characters)",
            help="Your reason is saved in the audit log with your name, so the approver and any "
                 "auditor can see why it was proposed.")
        if st.form_submit_button("Send the proposal"):
            if len(reason.strip()) < 10:
                st.error("Please give a reason of at least 10 characters.")
            else:
                from common import auth_service

                auth_service().propose_model_promotion(state=state, model=model, reason=reason)
                v = verdicts.get(model)
                still = (" Its checklist is not all yes yet, so it cannot be approved until it is."
                         if v is not None and not v.promoted else "")
                st.success("Proposal saved to the audit log. A policy owner who did not propose it "
                           "must approve it." + still)
