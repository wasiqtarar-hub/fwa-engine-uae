"""Overview — the loaded file in one screen, in plain words.

Start here (three steps), headline cards with a big number and a sentence, the
ten cases to look at first with their one-line explanation, one sentence on
which checks could run on this file (and what the file lacks for the rest),
charts with finding titles, and the file profile. Seeds, fingerprints, timings
and rule ids sit under "Technical details".

The caveats are not footnotes here. Every headline number carries the sentence
that says what it is not: a flag is a reason to look, not proof of fraud; money
flagged is not money saved; established and not-yet-established amounts are
never added together; nothing has been reviewed, so precision is not
measurable and is said so in words rather than shown as a number.
"""

from __future__ import annotations

import html

import pandas as pd
import streamlit as st

from common import (
    auth_service, banner, bar, boundary_note, chart_note, dataframe, disposition_chip,
    disposition_colours, empty_state, friendly_failure, friendly_table, headline_cards,
    help_text, page_header, pipeline, priority_chip, require_page, session_banner,
    status_badge, synthetic_banner, case_explanation,
)
from components.reviewer_flow import (
    days_words, link_to, open_case_button, split_amounts, tour_step, tour_toggle, workload_days,
)
from fwa.auth import Permission
from fwa.presentation import (
    aed, control_text, label, plain_date, standard_text, status, table_label, vocabulary,
)

_NOT_RUN = ("NOT_EXECUTABLE_ON_THIS_DATASET", "NOT_EXECUTABLE")


def _source_currency(result) -> str:
    """The adapter's own declared currency, not a constant."""
    from fwa.canonical import get_adapter
    try:
        return get_adapter(result.adapter_name.lower()).currency
    except Exception:
        return "the source currency"


def render() -> None:
    state = require_page("Overview", Permission.VIEW_OVERVIEW)
    session_banner(state)

    page_header("Overview")
    synthetic_banner()
    banner()

    result = pipeline()
    queue = result.queue()

    tour_toggle("Overview", key="ov_tour")

    # ---- start here ---------------------------------------------------------
    _start_here(state)

    # ---- headline cards -----------------------------------------------------
    st.markdown("## The file at a glance")
    tour_step("Overview", 0)
    with friendly_failure("the headline figures"):
        _headline_cards(result, queue, state)
    boundary_note()

    # ---- top 10 ---------------------------------------------------------------
    st.markdown("## Top 10 cases to look at first")
    with friendly_failure("the top ten cases",
                          "Open the Review queue to see the full list of cases."):
        _top_ten(result, queue, state)

    # ---- coverage -------------------------------------------------------------
    st.markdown("## Which checks could run on this file")
    with friendly_failure("the check coverage summary"):
        _coverage(result)

    # ---- queue shape ----------------------------------------------------------
    st.markdown("## What the cases are about")
    with friendly_failure("the charts about the queue"):
        _queue_charts(result, queue)

    # ---- the file -------------------------------------------------------------
    st.markdown("## About this file")
    with friendly_failure("the description of the file"):
        _profile(result)

    # ---- run status -----------------------------------------------------------
    st.markdown("## Did the run finish cleanly?")
    with friendly_failure("the run status"):
        _run_status(result)


# ---------------------------------------------------------------------------
# sections
# ---------------------------------------------------------------------------


def _start_here(state) -> None:
    steps = list(vocabulary().start_here or [])
    if not steps:
        return
    st.markdown("## Start here")
    cols = st.columns(len(steps))
    for col, step in zip(cols, steps):
        with col, st.container(border=True):
            st.markdown(f"**{html.escape(str(step.get('title', '')))}**")
            st.markdown(f"<div class='fwa-sub'>{html.escape(str(step.get('body', '')))}</div>",
                        unsafe_allow_html=True)
            page = str(step.get("page") or "")
            if page:
                from fwa.presentation import page_info

                link_to(state, page, f"Go to {page_info(page)['title']}")


def _reviewed_count(state) -> int | None:
    try:
        frame = auth_service().db.outcomes_frame(state.tenant_id)
        return 0 if frame is None else int(len(frame))
    except Exception:
        return None


def _headline_cards(result, queue, state) -> None:
    config = result.config
    per_day = int(config.get("alerts_per_reviewer_per_day"))
    reviewers = int(config.get("reviewer_count"))
    n_cases = len(queue)
    established, unestablished = split_amounts(queue)
    records = result.evaluation.records
    total = len(records)
    ran = sum(1 for r in records if r.outcome in ("TRIGGERED", "NOT_TRIGGERED"))
    fired = sum(1 for r in records if r.signal_count > 0)
    n_signals = len(result.signals)
    reviewed = _reviewed_count(state)

    cards = [
        (f"{n_cases:,} cases",
         f"need review. At current staffing ({reviewers} reviewers × {per_day} a day), that is "
         f"{days_words(workload_days(n_cases, reviewers, per_day))} of work."),
        (aed(established),
         "at risk where a clear rule was shown to fail (<strong>established</strong>)."),
        (aed(unestablished),
         "flagged but <strong>not yet established</strong>: the gross amount of claims behind "
         "unusual patterns, not money owed. Never added to the figure beside it."),
        (f"{ran} of {total} checks",
         f"ran on this file; {fired} of them raised at least one flag ({n_signals:,} flags in all). "
         "A flag is a reason to look, not proof of fraud."),
    ]
    if reviewed == 0 or reviewed is None:
        cards.append(("Nothing reviewed yet",
                      "No reviewer has recorded a decision, so we can't yet measure how often the "
                      "flags turn out to be right. That number is left blank rather than guessed."))
    else:
        cards.append((f"{reviewed:,} decisions",
                      "recorded by reviewers so far. How often flags turn out to be right is "
                      "reported on the Validation report once enough decisions build up; it is "
                      "never estimated here."))
    headline_cards(cards)
    st.markdown(f"<div class='fwa-caveat'><strong>Please keep in mind.</strong> "
                f"{html.escape(standard_text('not_savings'))} {html.escape(standard_text('shadow'))}"
                f"</div>", unsafe_allow_html=True)


def _top_ten(result, queue, state) -> None:
    if queue.empty:
        empty_state("No cases to show",
                    "No check raised a flag on this file. The Governance page shows whether any "
                    "check was switched off; the Data page shows which checks could run.")
        return
    st.caption("Sorted by urgency. Urgency only decides the order of the queue; it never changes "
               "what should happen to a claim.")
    for rank, row in enumerate(queue.head(10).itertuples(index=False), start=1):
        case = result.cases.cases.get(row.case_id)
        if case is None:
            continue
        try:
            headline = case_explanation(result, case, state).headline
        except Exception:
            headline = f"{label(row.scenario_family, 'family')}: {row.signal_count} flags."
        chips = (disposition_chip(case.disposition) + " "
                 + (priority_chip(case.priority.band, "", case.priority.value)
                    if case.priority is not None else ""))
        left, right = st.columns([6, 1], vertical_alignment="center")
        with left:
            st.markdown(
                f"<div class='fwa-card'><p><strong>{rank}.</strong> {html.escape(headline)}</p>"
                f"<p>{chips} <span class='fwa-tag'>{html.escape(row.case_id)}</span></p></div>",
                unsafe_allow_html=True,
            )
        with right:
            open_case_button(state, row.case_id, key=f"ov_open_{rank}", text="Open")
    link_to(state, "Review Queue", "See the whole queue")


def _required_tables(control) -> list[str]:
    tables: list[str] = []
    for f in getattr(control, "required_canonical_fields", []) or []:
        t = str(f).strip().split(" (")[0].split(".")[0].strip()
        if t and t not in tables:
            tables.append(t)
    return tables


def _lower_first(text: str) -> str:
    if len(text) > 1 and text[1].islower():
        return text[0].lower() + text[1:]
    return text


def _coverage(result) -> None:
    records = result.evaluation.records
    total = len(records)
    full = [r for r in records if r.data_support == "EXECUTABLE"]
    partial = [r for r in records if r.data_support == "PARTIAL"]
    not_run = [r for r in records if r.data_support in _NOT_RUN]

    pop = result.dataset.population_report()
    empty_tables = set(pop.loc[pop["rows"] == 0, "table"].astype(str))
    known_tables = set(pop["table"].astype(str))

    groups: dict[str, int] = {}
    for r in not_run:
        try:
            tables = _required_tables(result.registry.get(r.rule_id))
        except Exception:
            tables = []
        missing = [t for t in tables if t in empty_tables or t not in known_tables] or tables
        if not missing:
            missing = ["(not described)"]
        for t in missing:
            groups[t] = groups.get(t, 0) + 1

    ranked = sorted(groups.items(), key=lambda kv: (-kv[1], kv[0]))

    def phrase(t: str) -> str:
        return ("information the check catalogue does not name" if t == "(not described)"
                else _lower_first(table_label(t)))

    parts = [f"{phrase(t)} ({n} check{'s' if n != 1 else ''})" for t, n in ranked[:4]]
    lacks = (", ".join(parts[:-1]) + " and " + parts[-1]) if len(parts) > 1 else "".join(parts)
    sentence = (f"Of {total} checks, **{len(full)} ran fully**, **{len(partial)} ran in a simplified "
                f"form**, and **{len(not_run)} could not run**")
    sentence += (f" because the file lacks {lacks}." if not_run and lacks else ".")
    st.markdown(sentence)
    if len(ranked) > 1:
        st.caption("A check can need more than one kind of missing information, so it can appear in "
                   "more than one group below. Nothing is invented to fill a gap: a check that "
                   "cannot run is reported as such, never run on guessed data.")
    unlocked = [r for r in records if getattr(r, "support_basis", "catalogue") != "catalogue"]
    if unlocked:
        st.caption(f"{len(unlocked)} of these checks could run (or run in full) only because this file "
                   f"carries extra kinds of information that the basic claim file does not.")

    try:
        denying = result.registry.summary()["controls_that_may_deny"]
    except Exception:
        denying = None
    if denying is not None:
        st.caption(f"Only {denying} of the {total} checks are allowed to suggest denying a claim or "
                   f"paying a corrected amount, and every one of them is a hard or expert rule. "
                   f"Comparisons, connections, documents and pattern-finders can never deny.")

    if ranked:
        frame = pd.DataFrame([{"missing_group": _upper_first(phrase(t)), "checks_count": n}
                              for t, n in ranked])
        left, right = st.columns([1, 1.3], gap="large")
        with left:
            friendly_table(frame, key="ov_cov_groups", columns=["missing_group", "checks_count"],
                           keep_numeric=["checks_count"], height=320)
        with right:
            top_t, top_n = ranked[0]
            bar(frame.head(10).iloc[::-1], "missing_group", "checks_count",
                title=f"Most checks that could not run need {phrase(top_t)}",
                horizontal=True, height=340, x_label="What the file lacks",
                y_label="Number of checks that could not run",
                how_to_read="Each bar is one kind of information the file lacks; its length is how "
                            "many checks could not run without it. Longer bars are the gaps worth "
                            "filling first.")

    with st.expander("Technical details: check-by-check coverage"):
        rows = [{"rule_id": r.rule_id, "data_support": r.data_support,
                 "catalogue_data_support": r.catalogue_data_support, "support_basis": r.support_basis,
                 "outcome": r.outcome, "signal_count": r.signal_count,
                 "missing_for_unlock": r.missing_for_unlock} for r in records]
        dataframe(pd.DataFrame(rows), height=360)
        st.caption("data_support is the effective classification on this file; catalogue_data_support "
                   "is the static one in the catalogue; support_basis says whether a dataset unlock "
                   "changed it.")


def _upper_first(text: str) -> str:
    return text[:1].upper() + text[1:] if text else text


def _queue_charts(result, queue) -> None:
    if queue.empty:
        empty_state("No cases in the queue",
                    "No check raised a flag on this file. Check the Governance page for checks that "
                    "were switched off or raised too many flags.")
        return
    colours = disposition_colours()
    c1, c2 = st.columns(2)
    with c1:
        counts = queue["disposition"].value_counts().reset_index()
        counts.columns = ["disposition", "cases"]
        counts["label"] = [status(v).label for v in counts["disposition"]]
        cmap = {status(k).label: v for k, v in colours.items()}
        top = counts.iloc[0]
        share = top["cases"] / max(len(queue), 1)
        bar(counts.iloc[::-1], "label", "cases",
            title=f"{share:.0%} of cases are '{top['label']}'",
            colour="label", colour_map=cmap, horizontal=True, height=320,
            x_label="What policy suggests", y_label="Number of cases", legend_title="",
            how_to_read="Each bar counts the cases for which the rules suggest that next step. "
                        "Colours match the badges on every page. A suggestion is for a reviewer to "
                        "confirm; none of them is a finding of fraud.")
    with c2:
        fam = queue["scenario_family"].value_counts().reset_index()
        fam.columns = ["family", "cases"]
        fam["label"] = [label(v, "family") for v in fam["family"]]
        top = fam.iloc[0]
        bar(fam.iloc[::-1], "label", "cases",
            title=f"Most cases are about '{top['label'].lower()}' ({int(top['cases']):,})",
            horizontal=True, height=320, x_label="Kind of pattern", y_label="Number of cases",
            how_to_read="Each bar counts the cases whose flags point to that broad kind of problem.")

    bands = (queue.groupby("priority_band", as_index=False).size()
             .rename(columns={"size": "cases"}).sort_values("priority_band"))
    st.markdown("**How urgent the cases are.** " + html.escape(standard_text("priority_not_decision")),
                unsafe_allow_html=True)
    st.markdown(
        " ".join(priority_chip(r.priority_band) + f" {int(r.cases):,}"
                 for r in bands.itertuples(index=False) if r.priority_band),
        unsafe_allow_html=True,
    )

    run = result.evaluation.to_frame()
    if not run.empty:
        fired = run[run["signal_count"] > 0].sort_values("signal_count", ascending=False).head(10)
        if not fired.empty:
            fired = fired.assign(check=[_short(control_text(r).title) for r in fired["rule_id"]])
            top = fired.iloc[0]
            bar(fired.iloc[::-1], "check", "signal_count",
                title=f"The check that raised the most flags: {_short(top['check'], 50)}",
                horizontal=True, height=380, x_label="Check", y_label="Number of flags",
                how_to_read="Each bar is one check; its length is how many flags it raised on this "
                            "file. Many flags from one check is a workload, not a measure of fraud.")


def _short(text: str, n: int = 60) -> str:
    text = str(text)
    return text if len(text) <= n else text[: n - 1].rstrip() + "…"


def _profile(result) -> None:
    claims = result.claims
    items: list[tuple[str, str]] = [("Claims", f"{len(claims):,}")]
    for col, name in (("member_sk", "Different patients"), ("provider_sk", "Different providers"),
                      ("tpa", "Different claims administrators"), ("agent_id", "Different sales agents"),
                      ("diagnosis_primary", "Different main diagnoses")):
        if col in claims.columns:
            items.append((name, f"{claims[col].nunique():,}"))
    if "service_date" in claims.columns:
        dates = pd.to_datetime(claims["service_date"], errors="coerce")
        if dates.notna().any():
            items.append(("Treatment dates", f"{plain_date(dates.min())} to {plain_date(dates.max())}"))
    items.append(("File", result.source_name or "not recorded"))
    items.append(("Currency", f"{_source_currency(result)}, converted to AED at the rate on each "
                              f"treatment date"))
    profile = pd.DataFrame([{"profile_item": k, "profile_value": v} for k, v in items])

    pop = result.dataset.population_report()
    pop_view = pd.DataFrame({
        "table_name": [table_label(t) for t in pop["table"]],
        "table_status": [status(getattr(s, "value", s)).label for s in pop["status"]],
        "table_rows": pop["rows"].astype(int),
    })
    present = int((pop["rows"] > 0).sum())
    left, right = st.columns([1, 1.15], gap="large")
    with left:
        friendly_table(profile, key="ov_profile", columns=["profile_item", "profile_value"], height=340)
    with right:
        st.markdown(f"**Kinds of information in this file: {present} of {len(pop)} present.** "
                    "A kind of information the file does not contain is reported as missing, never "
                    "filled in with invented rows — invented rows would make every number on these "
                    "pages untrue.")
        friendly_table(pop_view.sort_values("table_rows", ascending=False), key="ov_population",
                       columns=["table_name", "table_status", "table_rows"],
                       keep_numeric=["table_rows"], height=300)
    with st.expander("Technical details: canonical tables and why each is empty or filled"):
        dataframe(pop[["table", "status", "rows", "reason"]].astype(str))
        st.caption(f"Adapter: {result.adapter_name}.")


def _run_status(result) -> None:
    errors = result.evaluation.errors
    if errors:
        st.markdown(status_badge("FAIL") + f" {len(errors)} check(s) stopped with an error. The "
                    "Governance page lists which, and why.", unsafe_allow_html=True)
    else:
        st.markdown(status_badge("PASS") + " Every check that could run finished without an error.",
                    unsafe_allow_html=True)
    st.markdown(
        "Running the same file again gives the same results: the random seed and every setting are "
        "fixed and recorded (see the technical details below). An AI assistant can write a short "
        "summary for each case; it works offline, is clearly labelled as not evidence, and the "
        "engine works fully without it."
    )
    summary = result.summary()
    with st.expander("Technical details: seed, settings fingerprint and timings"):
        st.markdown(
            f"Seed `{summary['seed']}` (`cfg.random_seed`), parameter-registry fingerprint "
            f"`{summary['parameters_fingerprint']}`. AI provider `{result.ai.provider.name}`, "
            f"deterministic `{result.ai.provider.deterministic}`.\n\n"
            "Signal identity is a pure function of (tenant, rule, version, subject, fact, period), so "
            "replaying a run overwrites rather than appends — which is what the idempotency test "
            "asserts for every control uniformly."
        )
        dataframe(pd.DataFrame([{"stage": k, "seconds": round(v, 2)}
                                for k, v in result.timings.items()]))
        if errors:
            dataframe(pd.DataFrame(errors, columns=["rule_id", "error"]))
