"""Compare hospitals — the simple scorecard, small-provider adjustment, changes over time, peers.

Build brief §14, item 4: peer comparison with shrinkage and uncertainty ranges,
change points with the level before and after and the date of the change, and
the **small-sample vs large-sample shrinkage demonstration** (two providers with
the same 20% rate and very different evidential weight).

Plain words are the default view. The statistics (median and MAD residuals,
empirical-Bayes shrinkage with Beta priors, posterior intervals, CUSUM and
EWMA) are all still here, one click away in "Technical details" expanders, with
a gloss wherever the term itself appears. Presentation only: nothing here
changes a score, a threshold or a disposition.
"""

from __future__ import annotations

import html

import pandas as pd
import streamlit as st

from common import (
    banner, bar, dataframe, empty_state, friendly_failure, friendly_table, headline_cards,
    help_text, histogram, line, mask, mono, page_header, pipeline, require_page,
    session_banner, synthetic_banner,
)
from components.analysis_pages import (
    change_title, fmt_unit, peer_group_words, safe_float, words,
)
from fwa.auth import Permission
from fwa.presentation import (
    aed, control_title, pct, plain_month, plain_number, ratio_words, standard_text,
)
from fwa.statistical.changepoint import detect_changepoints, monthly_provider_metrics
from fwa.statistical.shrinkage import shrinkage_demonstration


def render() -> None:
    state = require_page("Provider Analytics", Permission.VIEW_PROVIDER_ANALYTICS)
    session_banner(state)
    page_header("Provider Analytics")
    banner()
    synthetic_banner()

    result = pipeline()
    config = result.config

    tab_score, tab_small, tab_change, tab_peers = st.tabs([
        "Simple scorecard", "Providers with few claims", "Changes over time", "Who is compared with whom",
    ])
    with tab_score:
        with friendly_failure("the simple scorecard"):
            _scorecard(result, config, state)
    with tab_small:
        with friendly_failure("the small-provider adjustment"):
            _shrinkage(result, config, state)
    with tab_change:
        with friendly_failure("the changes over time"):
            _changes(result, config, state)
    with tab_peers:
        with friendly_failure("the comparison groups"):
            _peers(result)


# =============================================================================
# The simple scorecard (transparent composite)
# =============================================================================


def _reason(c: dict) -> str | None:
    """One contribution as a sentence: label, value vs typical, ratio in words."""
    value, typical = safe_float(c.get("value")), safe_float(c.get("peer_median"))
    if value is None or typical is None:
        return None
    unit = str(c.get("unit") or "")
    label = str(c.get("label") or "")
    ratio = ratio_words(value / typical) if typical else "well above"
    tail = "" if ratio == "about the same as" else f" — {ratio} the typical level"
    return (f"{label}: {fmt_unit(value, unit)} against a typical {fmt_unit(typical, unit)} for "
            f"similar providers{tail}.")


def _reasons(record, n: int = 2) -> list[str]:
    out = []
    for c in sorted(record.contributions, key=lambda c: -float(c.get("contribution") or 0)):
        if float(c.get("contribution") or 0) <= 0:
            continue
        sentence = _reason(c)
        if sentence:
            out.append(sentence)
        if len(out) >= n:
            break
    return out


def _scorecard(result, config, state) -> None:
    composite_results, composite_frame = result.composite
    threshold = float(config.get("composite_flag_threshold"))
    st.markdown(
        "The **simple scorecard** is a check anyone can recompute by hand. For each provider it looks "
        "at seven measures — average claim amount, amount per hospital day, share of the bill spent "
        "on medicines, share of the bill approved, claims per patient, readmissions and diagnoses "
        "that don't match the procedure — and asks how far the provider sits from the typical value "
        "for similar providers. Each gap is multiplied by a published weight and the results are "
        "added up. It is also the benchmark every pattern-finding model has to beat.",
        help=help_text("transparent_composite"),
    )
    records = [r for r in composite_results if not r.insufficient_evidence]
    not_compared = [r for r in composite_results if r.insufficient_evidence]
    flagged = sorted([r for r in records if r.score > threshold], key=lambda r: -r.score)
    headline_cards([
        (f"{len(records):,}", "providers compared with similar providers"),
        (f"{len(flagged):,}", f"score above the flag line of {threshold:g} and are worth a look"),
        (f"{len(not_compared):,}", "not compared: too few claims or too few similar providers, "
                                   "so they are never flagged on a guess"),
    ])
    st.caption(standard_text("not_proof"))

    if not records:
        empty_state("No provider could be compared",
                    "Every provider had too few claims or too few similar providers. Load a larger "
                    "file on the Data page.")
        return

    rows = []
    for r in flagged:
        rows.append({
            "Provider": mask(r.provider_sk, state),
            "Scorecard result": round(r.score, 2),
            "Main reasons": " ".join(_reasons(r)) or "No single measure stands out.",
            "Compared with": peer_group_words(r.peer_level_used),
            "Similar providers": r.peer_n,
            "Claims": r.claim_count,
            "Amount billed": aed(r.exposure_aed),
            "Peer group code": r.peer_level_used,
        })
    st.markdown("#### Providers above the flag line")
    friendly_table(pd.DataFrame(rows), key="pa_flagged",
                   columns=["Provider", "Scorecard result", "Main reasons", "Compared with",
                            "Similar providers", "Claims", "Amount billed"],
                   keep_numeric=["Scorecard result", "Similar providers", "Claims"],
                   empty_title="No provider is above the flag line",
                   empty_body="Every compared provider sits within the usual range of its peers.")

    histogram([r.score for r in records],
              title=f"{len(flagged):,} of {len(records):,} providers score above the flag line of {threshold:g}",
              vline=threshold, vline_label="flag line",
              x_label="Scorecard result (higher = further from similar providers)",
              y_label="Number of providers",
              how_to_read="Each bar counts providers with a similar result. Most sit near zero, "
                          "meaning they bill like their peers; those right of the dashed line are "
                          "listed above.")

    st.markdown("#### Look at one provider")
    ordered = sorted(records, key=lambda r: -r.score)
    options = {mask(r.provider_sk, state): r for r in ordered}
    who = st.selectbox(
        "Provider", list(options), key="pa_provider",
        help="Pick a provider to see how each of the seven measures compares with similar "
             "providers. The list starts with the highest scorecard result.",
    )
    record = options[who]
    st.markdown(
        f"**{html.escape(who)}** is compared with {record.peer_n:,} similar providers "
        f"({peer_group_words(record.peer_level_used)}). Its scorecard result is "
        f"**{record.score:.2f}** against a flag line of {threshold:g}.")
    sentences = _reasons(record, n=7)
    if sentences:
        st.markdown("\n".join(f"- {s}" for s in sentences))
    else:
        st.markdown("None of the seven measures sits above the typical level for similar providers.")
    contributions = pd.DataFrame(record.contributions)
    positive = contributions[contributions["contribution"] > 0] if not contributions.empty else contributions
    if not positive.empty:
        top = positive.sort_values("contribution", ascending=False).iloc[0]["label"]
        bar(positive.sort_values("contribution"), "label", "contribution",
            title=f"{who}'s result comes mainly from its {str(top).lower()}",
            horizontal=True, height=300, x_label="Measure", y_label="Points added to the result",
            how_to_read="Each bar is how much one measure adds to the scorecard result. Longer bars "
                        "are the measures where this provider is furthest from similar providers.")

    with st.expander("Technical details: how the result was calculated", expanded=False):
        st.markdown(
            "Each gap is a **robust residual**: (provider value − peer median) ÷ (1.4826 × MAD), "
            "where MAD is the median absolute deviation — the typical distance from the middle "
            "value. Medians are used instead of averages because claim amounts are lopsided. Only "
            "gaps in the adverse direction count (approval ratio counts in both directions). The "
            "weights are the governed parameter composite_feature_weights.",
            help=help_text("median_mad"),
        )
        if not contributions.empty:
            dataframe(contributions[[c for c in (
                "label", "unit", "value", "peer_median", "residual", "weight", "contribution",
                "scale_source", "scenario_alignment") if c in contributions.columns]])
            lines = [
                f"{c['weight']:+.2f} × residual {0.0 if c['residual'] is None else c['residual']:+.3f}"
                f" = {c['contribution']:+.4f} {c['label']}"
                for c in record.contributions
            ]
            lines.append(f"{'':>34}total = {record.score:+.4f}")
            mono("\n".join(lines))
        st.caption(record.explain())
        dataframe(composite_frame, height=300)


# =============================================================================
# Shrinkage
# =============================================================================


def _shrinkage(result, config, state) -> None:
    st.markdown(
        "Providers with few claims are **pulled towards the typical rate** so a small number of "
        "claims can't make them look extreme. One odd claim out of five is 20%, but it says very "
        "little; the same 20% over five thousand claims is a real pattern. The engine also works out "
        "a likely range for each provider's true rate, and leaves out of the ranking any provider "
        "whose range is too wide to trust.",
        help=help_text("shrinkage"),
    )
    demo = shrinkage_demonstration(
        max_posterior_width=float(config.get("max_posterior_width")),
        interval_mass=float(config.get("posterior_interval_mass")),
    )
    st.markdown("#### The same 20% rate, two very different providers")
    cols = st.columns(len(demo))
    for col, row in zip(cols, demo.itertuples(index=False)):
        with col:
            ranked = ("Too uncertain to rank: it is left out of the ranking rather than flagged on "
                      "a guess." if row.excluded_from_ranking else "Precise enough to rank.")
            st.markdown(
                f"<div class='fwa-card'><h4>A provider with {row.opportunities:,.0f} claims</h4>"
                f"<p>{row.events:,.0f} of its {row.opportunities:,.0f} claims had the problem: "
                f"{pct(row.observed_rate)}. Pulled towards the typical rate of {pct(row.peer_mean)}, "
                f"its estimate becomes <strong>{pct(row.shrunk_rate)}</strong>, likely between "
                f"{pct(row.posterior_low)} and {pct(row.posterior_high)}.</p>"
                f"<p class='fwa-sub'>{ranked}</p></div>",
                unsafe_allow_html=True,
            )
    st.caption("Both providers show the same rate. The small one is pulled far towards the typical "
               "rate and has a wide range; the large one barely moves. That difference is the whole "
               "reason for the adjustment.")

    st.markdown("#### On this file")
    signals = [s for s in result.signals if s.evidence.get("shrunk_rate") is not None]
    if not signals:
        empty_state("No adjusted rates in this run",
                    "No comparison check produced an adjusted rate on this file. Some checks may be "
                    "switched off (see Governance) or need information the file lacks.")
    else:
        rows = []
        for s in signals:
            interval = s.evidence.get("posterior_interval") or []
            low, high = (interval + [None, None])[:2] if isinstance(interval, list) else (None, None)
            rows.append({
                "Provider": mask(s.subject_id, state),
                "Check": control_title(s.rule_id),
                "Claims": plain_number(s.evidence.get("claim_count"), missing="—"),
                "Rate seen": pct(s.evidence.get("observed_rate")),
                "After adjustment": pct(s.evidence.get("shrunk_rate")),
                "Typical rate": pct(s.evidence.get("peer_mean")),
                "Likely range": (f"{pct(low)} to {pct(high)}" if safe_float(low) is not None
                                 and safe_float(high) is not None else "—"),
                "Check id": s.rule_id,
                "Prior strength": s.evidence.get("prior_strength"),
                "_sort": safe_float(s.evidence.get("shrunk_rate")) or 0.0,
            })
        frame = pd.DataFrame(rows).sort_values("_sort", ascending=False).drop(columns="_sort")
        friendly_table(frame, key="pa_shrunk",
                       columns=["Provider", "Check", "Claims", "Rate seen", "After adjustment",
                                "Typical rate", "Likely range"],
                       height=320)

    with st.expander("Technical details: empirical-Bayes shrinkage", expanded=False):
        st.markdown(
            "Each rate is shrunk with an **empirical-Bayes Beta prior** fitted to the peer group: "
            "adjusted rate = (events + α) ÷ (claims + α + β). The **posterior interval** is the "
            f"central {pct(float(config.get('posterior_interval_mass')))} of the Beta posterior; a "
            "provider whose interval is wider than max_posterior_width "
            f"({config.get('max_posterior_width')}) is excluded from ranking. Prior strength "
            "(α + β) is how many claims' worth of weight the peer average carries.",
            help=help_text("posterior_interval"),
        )
        dataframe(demo.drop(columns=["explanation"], errors="ignore").round(4))
        for row in demo.itertuples(index=False):
            st.caption(row.explanation)


# =============================================================================
# Change detection
# =============================================================================


def _changes(result, config, state) -> None:
    min_history = int(config.get("min_history_periods"))
    st.markdown(
        "For each provider the engine follows its **average claim value month by month** and looks "
        "for a lasting shift away from its own earlier level. A change is only declared after at "
        f"least {min_history} months of history, and each one records the level before, the level "
        "after and the month it happened, so you see what changed and when, not just an alarm.",
        help=help_text("change_point"),
    )
    series = monthly_provider_metrics(result.claims)
    changepoints = detect_changepoints(
        series, entity_column="provider_sk", period_column="period",
        value_column="mean_gross_aed", metric_name="mean claim value",
        min_history_periods=min_history,
        cusum_k=float(config.get("cusum_k")), cusum_h=float(config.get("cusum_h")),
        ewma_lambda=float(config.get("ewma_lambda")), ewma_L=float(config.get("ewma_L")),
    )
    rises = sum(1 for c in changepoints if c.direction == "increase")
    headline_cards([
        (f"{len(changepoints):,}", "providers whose average claim value shifted clearly"),
        (f"{rises:,}", "of those shifts were rises"),
        (f"{min_history}", "months of history needed before a change can be declared"),
    ])
    if not changepoints:
        empty_state(
            "No clear changes found",
            "No provider's monthly average claim value moved clearly away from its earlier level. "
            "When providers have few claims each month the monthly figures are noisy, so this means "
            "no clear shift was seen, not proof that no provider changed how it bills.")
        return

    ordered = sorted(changepoints, key=lambda c: -abs(float(c.change_magnitude)))
    rows = []
    for c in ordered:
        rows.append({
            "Provider": mask(c.entity, state),
            "Changed in": plain_month(c.change_date),
            "Direction": "Rose" if c.direction == "increase" else "Fell",
            "Before (average claim)": aed(c.pre_change_mean),
            "After (average claim)": aed(c.post_change_mean),
            "Size of change": (ratio_words(c.post_change_mean / c.pre_change_mean) + " the earlier level"
                               if safe_float(c.pre_change_mean) else "—"),
            "Months of history": c.periods_of_history,
            "How clear": pct(c.confidence),
            "Method": c.method,
        })
    friendly_table(pd.DataFrame(rows), key="pa_changes",
                   columns=["Provider", "Changed in", "Direction", "Before (average claim)",
                            "After (average claim)", "Size of change", "Months of history", "How clear"],
                   keep_numeric=["Months of history"], height=300)

    picks = {mask(c.entity, state): c for c in ordered}
    label = st.selectbox(
        "See one provider's monthly figures", list(picks), key="pa_change_pick",
        help="Pick a provider to see its average claim value month by month. The list starts with "
             "the biggest change.",
    )
    cp = picks[label]
    sub = series[series["provider_sk"] == cp.entity].sort_values("period").copy()
    sub["month"] = pd.to_datetime(sub["period"], errors="coerce")
    line(sub, "month", "mean_gross_aed", title=change_title(label, cp),
         x_label="Month", y_label="Average claim value (AED)",
         how_to_read=f"Each point is one month's average claim value. The shift was detected in "
                     f"{plain_month(cp.change_date)}: about {aed(cp.pre_change_mean)} before and "
                     f"{aed(cp.post_change_mean)} after.")
    st.info(
        "The file has no records of price-list changes, new contracts or ownership changes, so a "
        "shift here could be caused by one of those rather than by how the provider bills. Check "
        "for such a change before drawing any conclusion.", icon="ℹ️")

    with st.expander("Technical details: CUSUM and EWMA", expanded=False):
        st.markdown(
            "Two standard change detectors run on each provider's monthly series. **CUSUM** "
            "(cumulative sum) adds up small, persistent deviations from the provider's own average "
            "until they cross a decision line; **EWMA** (exponentially weighted moving average) "
            "smooths the series and flags when the smoothed value leaves its control limits. CUSUM "
            "is preferred when both fire.",
            help=help_text("change_point"),
        )
        dataframe(pd.DataFrame([{
            "parameter": k, "value": config.get(k)} for k in
            ("min_history_periods", "cusum_k", "cusum_h", "ewma_lambda", "ewma_L")]))
        frame = pd.DataFrame([c.to_dict() for c in ordered])
        frame["entity"] = frame["entity"].map(lambda p: mask(p, state))
        dataframe(frame)
        st.caption(cp.explain().replace(cp.entity, label))


# =============================================================================
# Peer groups
# =============================================================================


def _peers(result) -> None:
    st.markdown(
        "A provider is only compared with **similar providers**: the same kind of service, the same "
        "specialty, the same contract and a similar size. When the most specific group has too few "
        "providers for a fair comparison, the engine steps back to a broader group and records which "
        "one it used, so every comparison can be reproduced.",
        help=help_text("peer_back_off"),
    )
    levels = words("peer_levels")
    availability = words("availability")
    try:
        table = result.context.peers.level_availability()
    except Exception:
        table = pd.DataFrame()
    if table.empty:
        empty_state("No comparison groups", "The comparison groups were not built for this file.")
    else:
        rows = []
        for r in table.itertuples(index=False):
            entry = levels.get(r.canonical_name, {}) or {}
            rows.append({
                "Level": int(r.level),
                "Compared on": entry.get("what", str(r.canonical_name).replace("_", " ")),
                "How this file does it": entry.get("how", ""),
                "Available": availability.get(str(r.availability), str(r.availability).replace("_", " ").lower()),
                "Groups found": int(r.distinct_values),
            })
        friendly_table(pd.DataFrame(rows), key="pa_levels", keep_numeric=["Level", "Groups found"])
        geo = table[table["canonical_name"] == "geography"]
        if not geo.empty and str(geo.iloc[0]["availability"]) == "NOT_POPULATED":
            st.caption("Location is never used in these comparisons: it is left out rather than "
                       "guessed, and the checks that need it do not run.")

    used = pd.Series([s.peer_level_used for s in result.signals if s.peer_level_used])
    if not used.empty:
        counts = used.map(peer_group_words).value_counts().reset_index()
        counts.columns = ["group", "signals"]
        top = counts.iloc[0]["group"]
        bar(counts.sort_values("signals"), "group", "signals",
            title=f"Most flags compared providers on: {top}",
            horizontal=True, height=300, x_label="Comparison group", y_label="Number of flags",
            how_to_read="Each bar counts the flags whose comparison used that kind of group. More "
                        "specific groups make fairer comparisons; broader ones are used when the "
                        "specific group is too small.")

    with st.expander("Technical details: the six-level peer hierarchy", expanded=False):
        dataframe(table)
        if not used.empty:
            raw = used.value_counts().reset_index()
            raw.columns = ["peer_level_used", "signals"]
            dataframe(raw)
