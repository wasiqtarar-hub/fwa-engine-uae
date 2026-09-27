"""Settings and thresholds (internal name "Parameters & Models").

There are two kinds of number in this system and confusing them is the most
expensive mistake a reader of it can make.

A **number chosen by policy** is a governance decision. Somebody owns it,
somebody approved it, it has a rationale in prose and a date from which it
takes effect. It does not change because the data changed.
``min_peer_group_n = 30`` is a statement about how much evidence the
organisation requires before it will let a comparison be made, and it would be
the same number on a dataset a hundred times the size.

A **number measured from your data** is an estimate. The Beta prior's α and β,
a peer group's median and spread, a model's decision threshold, the split
date: every one of these is estimated from the file that is currently loaded
and is meaningless away from it. Load a different dataset and all of them
change.

The tabs keep the two apart on purpose. Plain wording is the default view;
the raw keys, fingerprints, α/β and hyperparameter names are one click away
in "Technical details" expanders and the "Show all columns" option.
"""

from __future__ import annotations

import html

import pandas as pd
import streamlit as st

from common import (
    banner, dataframe, empty_state, friendly_failure, friendly_table,
    headline_cards, help_text, note, page_header, pipeline, require_page, session_banner,
    synthetic_banner,
)
from components.governance_pages import (
    availability_label, check_title, explainer_label, format_value, model_parameter_words,
    owner_label, peer_level_label, plain_text, plain_when, scale_source_label,
)
from dataset import active_dataset, model_parameters
from fwa.auth import Permission
from fwa.presentation import (
    aed, field_label, model_label, parameter_text, pct, plain_date, plain_number, status,
)

MEASURED = "Measured from your data"
CHOSEN = "Chosen by policy"

MEASURED_SENTENCE = (
    "**Measured from your data** means the engine estimated the number from the claim file "
    "loaded now (typical values, spreads, fitted starting assumptions, model thresholds, the "
    "training split), so it changes whenever the file changes."
)
CHOSEN_SENTENCE = (
    "**Chosen by policy** means a named owner set the number and someone approved it, with "
    "effective dates and a written reason; it stays the same whatever file is loaded until "
    "the owner changes it."
)


def render() -> None:
    state = require_page("Parameters & Models", Permission.VIEW_PARAMETERS)
    session_banner(state)
    page_header("Parameters & Models")
    banner()
    synthetic_banner()

    result = pipeline()
    spec = active_dataset()

    st.markdown(f"- {CHOSEN_SENTENCE}\n- {MEASURED_SENTENCE}")

    tab_policy, tab_measured, tab_models, tab_peers = st.tabs([
        "Chosen by policy", "Measured from your data", "Pattern-finder settings",
        "How similar providers are grouped",
    ])

    with tab_policy:
        with friendly_failure("the governed settings",
                              "Check that config/parameters.yaml loads, or ask an administrator."):
            _policy(result)
    with tab_measured:
        with friendly_failure("the numbers measured from this file"):
            _measured(result, spec)
    with tab_models:
        with friendly_failure("the pattern-finder settings"):
            _models(result)
    with tab_peers:
        with friendly_failure("the comparison groups"):
            _peers(result)


# ---------------------------------------------------------------------------
# 1. chosen by policy
# ---------------------------------------------------------------------------


def _policy_frame(result) -> pd.DataFrame:
    report = pd.DataFrame(result.config.parameters.governance_report())
    if report.empty:
        return report
    texts = {k: parameter_text(k) for k in report["key"]}
    report["setting_words"] = [texts[k]["label"] for k in report["key"]]
    report["meaning_words"] = [texts[k]["meaning"] or "No plain description recorded yet."
                               for k in report["key"]]
    report["raise_words"] = [texts[k]["raise"] or "Not described yet." for k in report["key"]]
    report["lower_words"] = [texts[k]["lower"] or "Not described yet." for k in report["key"]]
    report["value_words"] = [format_value(k, v) for k, v in zip(report["key"], report["value"])]
    report["owner_words"] = [owner_label(o) for o in report["owner"]]
    report["approver_words"] = [owner_label(a) for a in report["approved_by"]]
    return report


def _policy(result) -> None:
    st.markdown("### Settings chosen by policy")
    report = _policy_frame(result)
    if report.empty:
        empty_state("No governed settings found",
                    "The settings registry is empty. Ask an administrator to check the "
                    "configuration file.")
        return

    expired = int(report["expired"].sum()) if "expired" in report else 0
    unapproved = int(report["unapproved"].sum()) if "unapproved" in report else 0
    owners = report["owner"].nunique()
    headline_cards([
        (f"{len(report):,}", "settings are governed: each has a named owner, an approver, "
                             "effective dates and a written reason."),
        (f"{owners}", "teams own them. Nobody can change a value without it being recorded."),
        (f"{expired}", "are past their review date. They still apply; the date is shown so it "
                       "is not forgotten."),
        (f"{unapproved}", "have no approver. A value nobody signed off is a governance gap."),
    ])

    c1, c2 = st.columns([2, 1])
    with c1:
        query = st.text_input(
            "Search settings", key="param_filter", placeholder="e.g. refill, duplicate, days",
            help="Type a word to narrow the list to settings whose name, description, owner or "
                 "reason mentions it. Leave it empty to see every setting.",
        )
    with c2:
        owner_options = sorted(report["owner"].unique(), key=owner_label)
        chosen_owners = st.multiselect(
            "Owned by", owner_options, default=[], format_func=owner_label, key="param_owner",
            help="Show only the settings owned by these teams, for example to prepare a review "
                 "with one owner. Leave it empty to show every team.",
        )

    view = report
    if query.strip():
        q = query.strip().lower()
        hay = (view["setting_words"] + " " + view["meaning_words"] + " " + view["owner_words"]
               + " " + view["key"] + " " + view["rationale"].astype(str)).str.lower()
        view = view[hay.str.contains(q, regex=False)]
    if chosen_owners:
        view = view[view["owner"].isin(chosen_owners)]
    view = view.sort_values(["expired", "unapproved", "setting_words"],
                            ascending=[False, False, True])
    st.caption(f"Showing {len(view):,} of {len(report):,} settings. Past-review and unapproved "
               "settings are listed first.")
    friendly_table(
        view, key="param_table",
        columns=["setting_words", "value_words", "raise_words", "lower_words", "owner_words",
                 "approver_words", "review_date", "expired", "meaning_words", "key", "rationale",
                 "source", "valid_from", "valid_to", "expected_alert_volume", "unapproved"],
        dates=["review_date", "valid_from", "valid_to"], height=420,
        empty_title="No setting matches",
        empty_body="Nothing matched the search or the owner filter. Clear them to see every "
                   "setting.",
    )

    st.markdown("#### Look at one setting")
    keys = list(view["key"]) or list(report["key"])
    by_key = report.set_index("key")
    chosen = st.selectbox(
        "Setting", keys, format_func=lambda k: by_key.loc[k, "setting_words"], key="param_pick",
        help="Pick a setting to read what it controls, what raising or lowering it would do, "
             "who owns it and when it applies.",
    )
    _setting_card(result, chosen, by_key.loc[chosen])

    st.download_button(
        "Download every setting as a spreadsheet (CSV)",
        pd.DataFrame(result.config.parameters.governance_report()).to_csv(index=False).encode("utf-8"),
        file_name="parameter_registry.csv", mime="text/csv",
        help="The full governed registry exactly as stored, with the technical names, for "
             "audit or for a change request.",
    )


def _setting_card(result, key: str, row) -> None:
    record = result.config.parameters.record(key)
    text = parameter_text(key)
    esc = html.escape
    st.markdown(
        f"<div class='fwa-card'><h4>{esc(text['label'])}: {esc(format_value(key, record.value))}</h4>"
        f"<p>{esc(text['meaning'] or 'No plain description has been written for this setting yet.')}</p>"
        f"<p><strong>If you raise it:</strong> {esc(text['raise'] or 'not described yet.')}<br>"
        f"<strong>If you lower it:</strong> {esc(text['lower'] or 'not described yet.')}</p>"
        f"<p class='fwa-sub'><strong>Owner:</strong> {esc(owner_label(record.owner))} · "
        f"<strong>Approved by:</strong> {esc(owner_label(record.approved_by))} · "
        f"<strong>In force:</strong> {esc(plain_date(record.valid_from))} to "
        f"{esc(plain_date(record.valid_to))}"
        + (f" · <strong>Review due:</strong> {esc(plain_date(record.review_date))}"
           if record.review_date else "")
        + "</p></div>",
        unsafe_allow_html=True,
    )
    with st.expander("Technical details: the setting as recorded", expanded=False):
        st.markdown(f"**Key** `{record.key}` · **raw value** `{record.value}`")
        st.markdown(f"**Reason given.** {record.rationale}")
        st.markdown(f"**Source.** {record.source}")
        scope = {f: getattr(record, f) for f in ("regulator", "payer", "product", "contract",
                                                 "provider_type", "code_family")}
        st.markdown("**Scope.** " + ", ".join(f"{k} = `{v}`" for k, v in scope.items()))
        if record.expected_alert_volume is not None:
            st.markdown(f"**Expected alerts on a 5,000-claim file** (a commitment by the owner, not "
                        f"a measured result): {record.expected_alert_volume}")
    note("Every value here is a configuration choice made for this build, not a universal "
         "medical, actuarial or regulatory fact. Where the honest answer to 'where did this come "
         "from?' is 'chosen for this tool', the recorded source says exactly that.")


# ---------------------------------------------------------------------------
# 2. measured from your data
# ---------------------------------------------------------------------------


def _measured(result, spec) -> None:
    st.markdown("### Numbers measured from this file")
    st.markdown(
        f"Everything on this tab was measured from **{spec.name}** ({spec.rows:,} claims) and "
        "does not carry over to another file."
    )

    # ---- the scorecard's typical values and spreads ----------------------
    results, _frame = result.composite
    contributions = [
        dict(c, provider_sk=r.provider_sk, peer_level=r.peer_level_used, peer_n=r.peer_n)
        for r in results if not r.insufficient_evidence for c in r.contributions
    ]
    st.markdown("#### Typical values and spreads used by the simple scorecard",
                help=help_text("median_mad"))
    st.markdown(
        "The simple scorecard compares each provider with similar providers on a few measures. "
        "For each measure it works out the **typical value** (the middle provider) and the "
        "**typical spread** around it, both measured from this file. How far a provider sits from "
        "the typical value, counted in spreads, is its **distance**. Only the **weight** is chosen "
        "by policy."
    )
    if not contributions:
        empty_state("No measure had enough similar providers to compare",
                    "Every provider in this file fell below the minimum number of claims or into "
                    "a comparison group that was too small. A larger file, or one with more "
                    "providers, would let the scorecard run.")
    else:
        detail = pd.DataFrame(contributions)
        fitted = (
            detail.groupby(["feature", "label", "unit"], as_index=False)
            .agg(peer_median=("peer_median", "median"), peer_scale=("peer_scale", "median"),
                 weight=("weight", "first"), providers_scored=("provider_sk", "nunique"),
                 mean_residual=("residual", "mean"), max_residual=("residual", "max"))
        )

        def _fmt(v, unit):
            if unit == "AED":
                return aed(v)
            if str(unit).startswith("AED/"):
                return f"{aed(v)} per {str(unit).split('/', 1)[1]}"
            if unit in ("ratio", "share", "%"):
                return pct(v)
            return plain_number(v, 2)

        shown = pd.DataFrame({
            "Measure": fitted["label"],
            "Typical value (measured)": [_fmt(v, u) for v, u in zip(fitted["peer_median"], fitted["unit"])],
            "Typical spread (measured)": [_fmt(v, u) for v, u in zip(fitted["peer_scale"], fitted["unit"])],
            "Weight (chosen by policy)": [pct(w) for w in fitted["weight"]],
            "Providers compared": fitted["providers_scored"],
            "Average distance (spreads)": [plain_number(v, 2) for v in fitted["mean_residual"]],
            "Largest distance (spreads)": [plain_number(v, 2) for v in fitted["max_residual"]],
        })
        dataframe(shown, height=300)
        st.caption("Typical value and spread are the median across comparison groups. A distance "
                   "of 3 means three typical spreads away from similar providers: unusual, but a "
                   "reason to look, not proof of anything.")

        counts = detail["scale_source"].value_counts()
        st.markdown("**How the spread was measured.** " + "; ".join(
            f"{scale_source_label(k)}: {n:,} provider-measure pairs" for k, n in counts.items())
            + ". The median spread is preferred; when every provider in a group is identical it "
              "falls back to a wider measure, and that step is recorded because it makes the "
              "comparison weaker evidence.")
        with st.expander("Technical details: robust location and scale per feature", expanded=False):
            dataframe(fitted, height=300)
            st.caption("peer_median = median; peer_scale = MAD × 1.4826 (or the IQR/SD fallback "
                       "named in scale_source); residual = (value − median) / scale.")

    # ---- fitted starting assumptions (priors) ------------------------------
    st.markdown("#### Starting assumptions fitted for small providers", help=help_text("shrinkage"))
    st.markdown(
        "A provider with five claims and one problem has a rate of 20%, which means very little; "
        "one with five thousand claims and the same rate is genuinely unusual. To be fair to small "
        "providers, some checks start from the **typical rate among similar providers** and move "
        "towards the provider's own rate as its claims add up. The **strength** says how many "
        "claims' worth of weight that starting point carries. Both are measured from this file."
    )
    priors = _prior_rows(result)
    if priors.empty:
        note("No starting assumption was fitted on this run: no rate-based check had enough "
             "similar providers to estimate one from.")
    else:
        plain = pd.DataFrame({
            "Check": [check_title(r) for r in priors["rule_id"]],
            "Typical rate among similar providers": [pct(m) for m in priors["peer_mean"]],
            "Worth about this many claims": [plain_number(s, 0) for s in priors["prior_strength"]],
            "Providers it was measured on": [plain_number(p) for p in priors["peer_population"]],
        })
        dataframe(plain, height=220)
        with st.expander("Technical details: the Beta priors", expanded=False):
            dataframe(priors, height=220)
            st.caption("α = mean × strength and β = (1 − mean) × strength (method of moments), "
                       "read back off this run's signals rather than re-fitted.")

    # ---- run facts ---------------------------------------------------------
    st.markdown("#### This run")
    summary = result.summary()
    headline_cards([
        (f"{summary['controls_executed']:,} of {summary['controls']:,}",
         "checks ran on this file; the rest lacked the information they need."),
        (plain_when(result.run_started), "is when this run started."),
        (f"{result.seed}", "is the fixed randomness seed, so running this file again gives the "
                            "same numbers."),
    ])
    with st.expander("Technical details: fingerprint and where the time went", expanded=False):
        st.markdown(f"Parameter fingerprint `{summary['parameters_fingerprint']}` (a hash of the "
                    "whole settings registry, stamped on every flag).")
        timings = pd.DataFrame(
            [{"stage": k, "seconds": round(v, 2)} for k, v in result.timings.items()]
        ).sort_values("seconds", ascending=False)
        dataframe(timings, height=320)


def _prior_rows(result) -> pd.DataFrame:
    """Beta priors recorded on this run's signals, de-duplicated.

    The priors are not held in one place: each control that uses shrinkage fits
    one on its own peer population and stamps the result onto the signals it
    raises. Reading them back off the signals is therefore the only way to
    report what was actually used. α and β follow from the peer mean and the
    prior strength exactly (method of moments), so they are derived here.
    """
    rows: dict[tuple, dict] = {}
    for signal in result.signals.all():
        ev = getattr(signal, "evidence", None) or {}
        strength, mean = ev.get("prior_strength"), ev.get("peer_mean")
        if strength is None or mean is None:
            continue
        try:
            strength, mean = float(strength), float(mean)
        except (TypeError, ValueError):
            continue
        key = (signal.rule_id, round(strength, 4), round(mean, 6))
        rows.setdefault(key, {
            "rule_id": signal.rule_id,
            "peer_mean": round(mean, 4),
            "prior_strength": round(strength, 2),
            "alpha": round(mean * strength, 4),
            "beta": round((1 - mean) * strength, 4),
            "peer_population": ev.get("peer_population"),
        })
    frame = pd.DataFrame(list(rows.values()))
    return frame.sort_values("rule_id") if not frame.empty else frame


# ---------------------------------------------------------------------------
# 3. model settings
# ---------------------------------------------------------------------------


def _models(result) -> None:
    st.markdown("### Pattern-finder settings, as actually used")
    layer = result.models
    if layer is None or layer.anomaly is None or not layer.anomaly.models:
        empty_state("No pattern-finder was trained on this file",
                    plain_text(" ".join(getattr(layer, "notes", []) or []))
                    or "Separating earlier claims (for learning) from later claims by other "
                       "providers (for scoring) left nothing to learn from or nothing to score. "
                       "A file with more months and more providers would fix this.")
        return

    st.markdown(
        "Some of these settings were **chosen by policy** (for example how many trees the model "
        "builds); others were **measured from your data** when the model was trained (for "
        "example the score above which a claim is flagged). Each row says which. The values are "
        "read off the trained models, so a setting that was adjusted shows what it became."
    )
    params = model_parameters(result)
    if params.empty:
        empty_state("No settings recorded", "The models did not report their settings on this run.")
    else:
        rows = []
        for _, r in params.iterrows():
            name, kind = model_parameter_words(r["parameter"])
            value = str(r["value"])
            if value in ("True", "False"):
                value = "Yes" if value == "True" else "No"
            elif r["parameter"] == "contamination":
                try:
                    value = pct(float(value))
                except ValueError:
                    pass
            rows.append({"Model": model_label(r["model"]), "Setting": name, "Value": value,
                         "Where the number comes from": kind})
        dataframe(pd.DataFrame(rows), height=400)
        st.caption("The flagging score is set so that the number flagged matches what reviewers can "
                   "handle, not by the starting guess of the share of unusual claims.")
        with st.expander("Technical details: hyperparameters as fitted", expanded=False):
            dataframe(params, height=400)

    split = layer.anomaly.split
    if split is not None:
        st.markdown("#### How claims were divided between learning and scoring",
                    help=help_text("temporal_split"))
        plan = split.to_dict() if hasattr(split, "to_dict") else dict(split.__dict__)
        headline_cards([
            (plain_date(plan.get("split_date")), "is the dividing date: the models learned from "
                                                 "claims before it and scored claims after it."),
            (plain_number(plan.get("train_claims")),
             f"claims from {plain_number(plan.get('train_providers'))} providers were used for "
             "learning."),
            (plain_number(plan.get("score_claims")),
             f"claims from {plain_number(plan.get('score_providers'))} other providers were scored."),
            (plain_number(plan.get("excluded_claims")),
             "claims got no score, because scoring a provider the model learned from would "
             "flatter it."),
        ])
        st.markdown(
            "Learning and scoring are kept apart **in time and by provider**. Without that, a "
            "model could score a provider it had already learned from, or use later claims to "
            "explain earlier ones. Both would make it look better than the simple scorecard it "
            "has to beat. The dividing date and the counts are measured from this file."
        )
        with st.expander("Technical details: the split plan", expanded=False):
            dataframe(pd.DataFrame([{"property": k, "value": str(v)} for k, v in plan.items()]),
                      height=280)

    st.markdown("#### How the models' reasons were worked out", help=help_text("shap"))
    rows = [{"Model": model_label(name),
             "Method": explainer_label(ex.method),
             "Available": "Yes" if ex.available else "No",
             "Why not": plain_text(ex.failure_reason) or "—"}
            for name, ex in layer.explainers.items()]
    if rows:
        dataframe(pd.DataFrame(rows))
        st.caption("Where a model is made of decision trees an exact method is used. Otherwise an "
                   "approximation is used and it is named as one.")
        with st.expander("Technical details: explainer classes", expanded=False):
            dataframe(pd.DataFrame([{"model": n, "explainer": ex.method,
                                     "available": ex.available, "failure_reason": ex.failure_reason}
                                    for n, ex in layer.explainers.items()]).astype(str))


# ---------------------------------------------------------------------------
# 4. peer groups
# ---------------------------------------------------------------------------


def _peers(result) -> None:
    st.markdown("### How similar providers are grouped", help=help_text("peer_group"))
    st.markdown(
        "A provider is never compared with the whole book. It is compared with a group of "
        "similar providers. The engine starts with the most specific grouping and, if the group "
        "is too small to say anything, steps back to a broader one. When it runs out of steps, "
        "it refuses to compare rather than compare against a handful."
    )
    groups = result.config.peer_groups
    min_n = result.config.get("min_peer_group_n")
    st.markdown(
        f"**Chosen by policy:** a group needs at least **{min_n} providers** "
        f"(setting: {parameter_text('min_peer_group_n')['label']}). "
        "**Measured from your data:** which grouping each provider's comparison actually used."
    )

    levels = pd.DataFrame([
        {"Grouping": peer_level_label(l.canonical_name),
         "Taken from": field_label(l.dataset_mapping) if l.dataset_mapping else "—",
         "In this file": availability_label(l.availability),
         "Note": plain_text(l.proxy_note) or "—"}
        for l in groups.levels
    ])
    st.markdown("#### The groupings the engine can use")
    st.caption("A grouping this file cannot fill is skipped and said to be skipped; filling it "
               "with a guess would make every comparison built on it quietly wrong.")
    dataframe(levels)

    name_of = {l.canonical_name: peer_level_label(l.canonical_name) for l in groups.levels}
    backoff = pd.DataFrame([
        {"Step": i + 1,
         "Providers are grouped by": " + ".join(name_of.get(x, peer_level_label(x))
                                                for x in step["levels"]) or "All providers together"}
        for i, step in enumerate(groups.backoff_order)
    ])
    st.markdown("#### The order in which groupings are tried")
    dataframe(backoff)

    results, _ = result.composite
    level_words = _level_words(groups)
    used = pd.Series([level_words.get(str(r.peer_level_used), peer_level_label(r.peer_level_used))
                      for r in results]).value_counts()
    st.markdown("#### Which grouping each provider's comparison landed on (measured)")
    if used.empty:
        empty_state("No provider was compared", "The scorecard did not run on this file.")
    else:
        dataframe(used.rename_axis("Grouping used").reset_index(name="Providers"))
    note(
        f"A provider whose steps run out before reaching {min_n} similar providers is reported as "
        f"“{status('INSUFFICIENT_PEER_EVIDENCE').label}” and is not scored. That is a "
        "different statement from scoring zero, and it is kept different everywhere it appears."
    )
    with st.expander("Technical details: hierarchy and back-off as configured", expanded=False):
        dataframe(pd.DataFrame([
            {"level": l.level, "canonical name": l.canonical_name,
             "mapped to": l.dataset_mapping or "—", "availability": l.availability,
             "stand-in used": l.proxy_note or "—"} for l in groups.levels]))
        dataframe(pd.DataFrame([
            {"step": i + 1, "group": s["name"], "levels": ", ".join(s["levels"])}
            for i, s in enumerate(groups.backoff_order)]))


def _level_words(groups) -> dict[str, str]:
    """Back-off group names (``L1_L2``, ``GLOBAL``) in words."""
    names = {l.canonical_name: peer_level_label(l.canonical_name) for l in groups.levels}
    by_number = {f"L{l.level}": names[l.canonical_name] for l in groups.levels}
    out = {"GLOBAL": peer_level_label("GLOBAL")}
    for step in groups.backoff_order:
        out[step["name"]] = " + ".join(names.get(x, peer_level_label(x)) for x in step["levels"]) \
            or peer_level_label("GLOBAL")
    out.update({k: v for k, v in by_number.items() if k not in out})
    return out
