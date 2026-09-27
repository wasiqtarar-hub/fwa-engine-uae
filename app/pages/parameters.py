"""Parameters & Models — every number this run used, and where it came from.

There are two kinds of number in this system and confusing them is the most
expensive mistake a reader of it can make.

A **configured threshold** is a governance decision. Somebody owns it, somebody
approved it, it has a rationale in prose and a date from which it takes effect.
It does not change because the data changed. `min_peer_group_n = 30` is a
statement about how much evidence the organisation requires before it will let
a comparison be made, and it would be the same number on a dataset a hundred
times the size.

A **fitted quantity** is a measurement. The Beta prior's α and β, a peer
group's median and MAD, a model's decision threshold: every one of these is
estimated from the file that is currently loaded and is meaningless away from
it. Load a different dataset and all of them change.

The tabs below keep the two apart on purpose. A threshold shown without its
owner invites the reader to treat it as if it fell out of the data; a fitted
value shown without its sample size invites them to treat an estimate from
eleven observations as if it were a fact.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from common import (
    banner, dataframe, empty_state, note, pipeline, require_page, session_banner, tiles,
)
from dataset import active_dataset, model_parameters
from fwa.auth import Permission


def render() -> None:
    state = require_page("Parameters & Models", Permission.VIEW_PARAMETERS)
    session_banner(state)

    st.markdown("# Parameters & Models")
    banner()

    result = pipeline()
    spec = active_dataset()

    st.markdown(
        "Every number this run used, separated by where it came from. **Configured** values are "
        "governance decisions with an owner and an approval; **fitted** values are estimates "
        "measured from the dataset currently loaded and change the moment it does."
    )

    tab_fitted, tab_configured, tab_models, tab_peers = st.tabs([
        "Fitted on this run", "Configured thresholds", "Model hyperparameters", "Peer baselines",
    ])

    # =====================================================================
    with tab_fitted:
        _fitted(result, spec)

    with tab_configured:
        _configured(result)

    with tab_models:
        _models(result)

    with tab_peers:
        _peers(result)


# ---------------------------------------------------------------------------
# 1. fitted
# ---------------------------------------------------------------------------


def _fitted(result, spec) -> None:
    st.markdown("## Quantities estimated from this dataset")
    st.markdown(
        f"Everything on this tab was measured from **{spec.name}** "
        f"({spec.rows:,} claims, sha256 `{spec.short_sha}`). None of it is configuration, and "
        "none of it transfers to another dataset."
    )

    # ---- the composite's robust location and scale -----------------------
    results, _frame = result.composite
    contributions = [
        dict(c, provider_sk=r.provider_sk, peer_level=r.peer_level_used, peer_n=r.peer_n)
        for r in results if not r.insufficient_evidence for c in r.contributions
    ]
    if not contributions:
        empty_state("No composite feature had enough peer evidence",
                    "Every provider on this dataset fell below the minimum denominator or into "
                    "a peer group smaller than cfg.min_peer_group_n.")
    else:
        detail = pd.DataFrame(contributions)
        fitted = (
            detail.groupby(["feature", "label", "unit"], as_index=False)
            .agg(
                peer_median=("peer_median", "median"),
                peer_scale=("peer_scale", "median"),
                weight=("weight", "first"),
                providers_scored=("provider_sk", "nunique"),
                mean_residual=("residual", "mean"),
                max_residual=("residual", "max"),
            )
        )
        st.markdown("### Robust location and scale, per composite feature")
        st.markdown(
            "The composite scores a provider by how far it sits from its peer group on each "
            "feature, measured in robust standard deviations — median for location, MAD × 1.4826 "
            "for scale. Both are recomputed from whichever dataset is loaded; the **weight** "
            "beside them is the only configured number in this table."
        )
        dataframe(fitted, height=300)

        scale_sources = detail["scale_source"].value_counts().rename_axis("scale estimator") \
            .reset_index(name="feature-provider pairs")
        st.markdown("**Which scale estimator was used**")
        st.markdown(
            "MAD is preferred. Where a peer group's MAD is zero — every member identical on that "
            "feature — the fallback ladder steps to the IQR and then to the standard deviation, "
            "and the step taken is recorded rather than hidden, because a residual computed on a "
            "fallback scale is weaker evidence than one computed on a MAD."
        )
        dataframe(scale_sources)

    # ---- shrinkage priors -------------------------------------------------
    st.markdown("### Empirical-Bayes priors fitted to the peer population")
    st.markdown(
        "A provider with five claims and one adverse event has an observed rate of 20%, which "
        "is close to meaningless; a provider with five thousand claims and the same rate is an "
        "outlier. The Beta prior below is what pulls the first toward the peer mean and leaves "
        "the second almost untouched. Its strength — α + β — is the number of pseudo-"
        "observations the prior is worth, fitted by the method of moments on this dataset's own "
        "peer population."
    )
    priors = _prior_rows(result)
    if priors.empty:
        note("No Beta prior was fitted on this run: no rate-based control had a peer population "
             "large enough to estimate one from.")
    else:
        dataframe(priors, height=220)

    # ---- as-of and run facts ---------------------------------------------
    st.markdown("### Run facts")
    summary = result.summary()
    tiles([
        ("Parameter fingerprint", summary["parameters_fingerprint"],
         "sha256 of the whole registry — stamped on every signal"),
        ("Random seed", f"{result.seed}", "fixed, so this run reproduces"),
        ("Controls that ran", f"{summary['controls_executed']:,}",
         f"of {summary['controls']:,} in the catalogue"),
        ("Run started", result.run_started.strftime("%Y-%m-%d %H:%M UTC"), ""),
    ])
    timings = pd.DataFrame(
        [{"stage": k, "seconds": round(v, 2)} for k, v in result.timings.items()]
    ).sort_values("seconds", ascending=False)
    with st.expander("Where the time went on this run"):
        dataframe(timings, height=320)


def _prior_rows(result) -> pd.DataFrame:
    """Beta priors recorded on this run's signals, de-duplicated.

    The priors are not held in one place: each control that uses shrinkage fits
    one on its own peer population and stamps the result onto the signals it
    raises. Reading them back off the signals is therefore the only way to
    report what was actually used, rather than re-fitting now and reporting
    something that merely resembles it.

    What the signal carries is the peer mean and the prior strength, which is
    the pair a reader needs — α and β follow from them exactly, because the
    method of moments defines α = mean × strength and β = (1 − mean) ×
    strength. They are derived here rather than stored twice.
    """
    rows: dict[tuple, dict] = {}
    for signal in result.signals.all():
        ev = getattr(signal, "evidence", None) or {}
        strength, mean = ev.get("prior_strength"), ev.get("peer_mean")
        if strength is None or mean is None:
            continue
        key = (signal.rule_id, round(float(strength), 4), round(float(mean), 6))
        rows.setdefault(key, {
            "fitted for": signal.rule_id,
            "peer mean": round(float(mean), 4),
            "prior strength (α+β)": round(float(strength), 2),
            "α (= mean × strength)": round(float(mean) * float(strength), 4),
            "β (= (1−mean) × strength)": round((1 - float(mean)) * float(strength), 4),
            "entities in the peer population": ev.get("peer_population", "—"),
        })
    frame = pd.DataFrame(list(rows.values()))
    return frame.sort_values("fitted for") if not frame.empty else frame


# ---------------------------------------------------------------------------
# 2. configured
# ---------------------------------------------------------------------------


def _configured(result) -> None:
    st.markdown("## Governed configuration")
    st.markdown(
        "Every `cfg.*` value the controls reference, with the owner accountable for it, who "
        "approved it, the dates between which it is in force and the prose rationale. No "
        "threshold in this system is a literal in a source file: a number without an owner is a "
        "number nobody can be asked about."
    )
    report = pd.DataFrame(result.config.parameters.governance_report())
    if report.empty:
        empty_state("No parameters", "The registry is empty.")
        return

    expired = int(report["expired"].sum()) if "expired" in report else 0
    unapproved = int(report["unapproved"].sum()) if "unapproved" in report else 0
    tiles([
        ("Parameters under governance", f"{len(report):,}", ""),
        ("Past their review date", f"{expired:,}",
         "still in force — expiry is surfaced, not silently applied"),
        ("Unapproved", f"{unapproved:,}", "set but never signed off"),
        ("Registry fingerprint", result.config.fingerprint(), "changes if any value changes"),
    ])

    query = st.text_input("Filter by name, owner or rationale", key="param_filter")
    view = report
    if query.strip():
        q = query.strip().lower()
        mask = view.apply(lambda r: q in " ".join(str(v).lower() for v in r.values), axis=1)
        view = view[mask]
    dataframe(view, height=460)

    st.download_button(
        "Download the parameter registry as CSV",
        report.to_csv(index=False).encode("utf-8"),
        file_name="parameter_registry.csv",
        mime="text/csv",
    )


# ---------------------------------------------------------------------------
# 3. model hyperparameters
# ---------------------------------------------------------------------------


def _models(result) -> None:
    st.markdown("## Model hyperparameters, as actually fitted")
    layer = result.models
    if layer is None or layer.anomaly is None or not layer.anomaly.models:
        empty_state("No model was trained on this dataset",
                    " ".join(getattr(layer, "notes", []) or [])
                    or "The temporal and entity-isolated split left an empty training or "
                       "scoring set.")
        return

    st.markdown(
        "These are the values the fitted estimators are actually carrying, read off the trained "
        "objects rather than off the configuration that requested them — so a hyperparameter "
        "that was clipped, defaulted or overridden shows here as what it became."
    )
    params = model_parameters(result)
    dataframe(params, height=400)

    split = layer.anomaly.split
    if split is not None:
        st.markdown("### The split the models were fitted under")
        st.markdown(
            "Training and scoring are separated **in time and by entity**. Without the entity "
            "separation a model can score a provider it was trained on and post a lift figure "
            "that is partly memory; without the temporal separation it can use the future to "
            "explain the past. Both would flatter the model against the transparent composite, "
            "which is the comparison the promotion gate turns on."
        )
        plan = split.to_dict() if hasattr(split, "to_dict") else dict(split.__dict__)
        dataframe(pd.DataFrame([{"property": k, "value": str(v)} for k, v in plan.items()]),
                  height=280)

    st.markdown("### Explanation method actually used")
    st.markdown(
        "SHAP's exact tree path method is used where the estimator has a tree structure to "
        "exploit. Where it does not, the kernel approximation is used over a k-means-summarised "
        "background — which is an approximation, so it is named here rather than being presented "
        "as the exact method."
    )
    dataframe(pd.DataFrame([
        {"model": name,
         "explainer": ex.method,
         "available": "yes" if ex.available else "no",
         "why not": ex.failure_reason or "—"}
        for name, ex in layer.explainers.items()
    ]))


# ---------------------------------------------------------------------------
# 4. peer baselines
# ---------------------------------------------------------------------------


def _peers(result) -> None:
    st.markdown("## Peer baselines on this dataset")
    st.markdown(
        "A provider is never compared to the whole book. It is compared to a peer group built by "
        "walking a back-off order until the group is large enough to say anything — and where "
        "the walk runs out, the comparison is refused rather than made on a group of four."
    )

    groups = result.config.peer_groups
    levels = pd.DataFrame([
        {"level": l.level,
         "canonical name": l.canonical_name,
         "mapped to": l.dataset_mapping or "—",
         "availability": l.availability,
         "stand-in used": l.proxy_note or "—"}
        for l in groups.levels
    ])
    st.markdown("### Hierarchy levels")
    st.markdown(
        "A level that this dataset cannot populate is skipped and said to be skipped. Filling it "
        "with a plausible stand-in would make every peer comparison built on it quietly wrong."
    )
    dataframe(levels)

    st.markdown("### Back-off order")
    backoff = pd.DataFrame([
        {"step": i + 1, "group": step["name"], "levels": ", ".join(step["levels"])}
        for i, step in enumerate(groups.backoff_order)
    ])
    dataframe(backoff)

    results, _ = result.composite
    st.markdown("### Which step each provider's comparison actually landed on")
    used = pd.Series([r.peer_level_used for r in results]).value_counts() \
        .rename_axis("peer level used").reset_index(name="providers")
    dataframe(used)
    note(
        f"`min_peer_group_n` is {result.config.get('min_peer_group_n')} on this run. A provider "
        "whose walk ends without reaching it is returned as INSUFFICIENT_PEER_EVIDENCE and is "
        "not scored — which is a different statement from scoring zero, and is kept different "
        "everywhere it appears."
    )
