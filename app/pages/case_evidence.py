"""Case evidence — one case, explained in plain words, with the evidence beneath.

Build brief §14, item 3, and the evidence rules that follow it:

    "never display a bare score without its decomposition; always show
     ``rule_id@version`` beside any finding; AI-generated text always in its
     distinct advisory panel; the dataset strip is persistent and not
     dismissible."

The plain-language rework keeps every one of those, and moves the technical
half one click away: the case opens on the plain explanation (headline, ranked
reasons, strength, what it does not mean, what to check next, downloadable
one-page summary); then the one primary action, the decision form; then the
evidence panels in words, each with a "Technical details" expander carrying
the rule id and version, the raw facts, the priority arithmetic and the peer
statistics. The AI summary and the copilot keep their distinct dashed panel
and their "not evidence" label. Reading a case and deciding it stay separate
permissions.
"""

from __future__ import annotations

import html
import json

import pandas as pd
import streamlit as st

from common import (
    ai_panel, auth_service, banner, case_explanation, dataframe, disposition_chip, empty_state,
    friendly_failure, friendly_table, mask, note, page_header, pipeline, priority_chip,
    render_explanation, require_page, session_banner, source_span, synthetic_banner,
)
from components.reviewer_flow import (
    case_label_func, plain_or, subject_for, tag, tour_step, tour_toggle,
)
from fwa.auth import AccessDenied, Permission
from fwa.enums import Disposition, REVIEWER_ASK
from fwa.presentation import (
    aed, control_text, label, pct, plain_month, plain_number, standard_text, status,
)
from fwa.presentation.explain_case import reason_sentence
from fwa.presentation.plain_language import RAW_TOKEN_RE, RULE_ID_RE


def render() -> None:
    state = require_page("Case Evidence", Permission.VIEW_CASE_EVIDENCE)
    session_banner(state)

    page_header("Case Evidence")
    synthetic_banner()
    banner()

    result = pipeline()
    queue = result.queue(state.tenant_id)

    if queue.empty:
        empty_state("No cases available",
                    "No cases were found for your organisation on this file. Open the Overview to "
                    "see which checks could run, or load another file on the Data page.")
        return

    tour_toggle("Case Evidence", key="ce_tour")

    case_ids = list(queue["case_id"])
    current = st.session_state.get("fwa_selected_case")
    index = case_ids.index(current) if current in case_ids else 0
    case_id = st.selectbox(
        "Case", case_ids, index=index, key="ce_pick",
        format_func=case_label_func(result, state),
        help="Choose which case to read. Cases are listed most urgent first. The case you picked on "
             "the Review queue is selected for you.",
    )
    st.session_state["fwa_selected_case"] = case_id

    case = result.cases.cases[case_id]
    signals = result.signals_for_case(case_id)
    assigned = case_id in state.assigned_case_ids
    subject = mask(case.primary_subject_id, state, assigned=assigned)

    # ======================================================================
    # 1. THE PLAIN EXPLANATION — what was found, why, how strong, what next
    # ======================================================================
    tour_step("Case Evidence", 0)
    chips = disposition_chip(case.disposition)
    if case.priority is not None:
        chips += " " + priority_chip(case.priority.band, "", case.priority.value)
    period = _month(case.period_bucket)
    st.markdown(f"{chips} {tag(case.case_id)}" + (f" {tag(period)}" if period else ""),
                unsafe_allow_html=True)
    with friendly_failure("the plain explanation of this case",
                          "The evidence panels further down still show what each check found."):
        expl = case_explanation(result, case, state)
        st.markdown(
            f"<div class='fwa-ask'><strong>Amount at risk.</strong> {_amount_line(case)}<br>"
            f"<strong>What is being asked of you.</strong> "
            f"{html.escape(REVIEWER_ASK.get(case.disposition, ''))} "
            f"<span class='fwa-sub'>({html.escape(status(case.disposition).meaning)})</span></div>",
            unsafe_allow_html=True,
        )
        render_explanation(expl, key=f"ce_expl_{case_id}")

    # ======================================================================
    # 2. THE PRIMARY ACTION — one per screen (§14.1)
    # ======================================================================
    tour_step("Case Evidence", 1)
    _disposition_form(state, result, case, signals)

    # ======================================================================
    # 3. THE EVIDENCE — in words, technical detail one click away
    # ======================================================================
    st.markdown("## The evidence")
    tour_step("Case Evidence", 2)

    with friendly_failure("the list of checks that fired"), st.expander(
            f"What each check found ({len({s.rule_id for s in signals})} checks, "
            f"{len(signals)} flags)", expanded=True):
        _signals_panel(result, signals, state, assigned)
    with friendly_failure("the technical details"), st.expander(
            "Technical details: rule ids, versions and the raw facts behind each flag"):
        _signals_technical(signals, state, assigned)

    with friendly_failure("the urgency breakdown"), st.expander("Why this case is this urgent"):
        _priority_panel(case)
    with friendly_failure("the urgency arithmetic"), st.expander("Technical details: the urgency arithmetic"):
        _priority_technical(case)

    with friendly_failure("the amount breakdown"), st.expander("How the amount at risk was worked out"):
        _exposure_panel(case, signals)

    model_features = _model_features(result, signals)
    if model_features:
        with friendly_failure("the pattern-finder's reasons"), st.expander("What the pattern-finder saw"):
            _shap_panel(model_features)

    peer_signals = [s for s in signals if s.peer_level_used]
    if peer_signals:
        with friendly_failure("the peer comparison"), st.expander("Comparison with similar providers"):
            _peer_panel(peer_signals, state, assigned)

    document_spans = _document_spans(signals)
    if document_spans:
        with friendly_failure("the document evidence"), st.expander("What the documents say"):
            _document_panel(document_spans)

    # ---- AI narrative, in its distinct advisory panel ---------------------
    if result.ai.narrator is not None:
        tour_step("Case Evidence", 3)
        st.markdown("### AI-written summary")
        with friendly_failure("the AI-written summary", "The evidence above is unaffected."):
            _ai_summary(state, result, case, signals, model_features, document_spans, subject)

    # ---- copilot ----------------------------------------------------------
    if result.ai.copilot is not None:
        with friendly_failure("the question box", "The evidence above is unaffected."):
            _copilot_panel(state, result, case, signals)

    # ---- audit trail -------------------------------------------------------
    with friendly_failure("the audit trail"), st.expander("Who did what on this case (audit trail)"):
        events = result.audit.to_dataframe(state.tenant_id)
        if not events.empty:
            events = events[events["subject"] == case.case_id]
        friendly_table(
            events if not events.empty else pd.DataFrame(), key="ce_audit",
            columns=["event_time", "event_type", "actor", "actor_role", "reason"],
            dates=["event_time"],
            empty_title="Nothing recorded yet",
            empty_body="No one has recorded a decision or other action on this case.",
        )
        st.caption("The audit trail can only be added to. No role can edit or delete it, and each "
                       "entry is linked to the one before it so tampering would show.")


# ---------------------------------------------------------------------------
# wording helpers
# ---------------------------------------------------------------------------


def _month(period) -> str:
    p = str(period or "")
    if len(p) == 7 and p[4] == "-":
        return plain_month(p + "-01")
    return p


def _amount_line(case, *, with_basis: bool = False) -> str:
    # The engine's derivation sentence is technical ("largest ESTABLISHED
    # exposure …"), so it is shown only inside the collapsed amount panel.
    basis = plain_or(case.exposure_basis) if with_basis else ""
    if case.exposure_established:
        return (f"{aed(case.exposure_aed)} — <strong>established</strong>: a clear rule was shown to "
                f"fail. {html.escape(basis)}")
    return (f"{aed(case.exposure_aed)} — <strong>{html.escape(standard_text('not_established'))}"
            f"</strong>. This is the gross amount of the claims involved, not money owed. "
            f"{html.escape(basis)}")


# ---------------------------------------------------------------------------
# evidence panels
# ---------------------------------------------------------------------------


_NOTE_KEYS = ("proxy_note", "not_a_fraud_label_notice", "not_fraud_evidence_alone_notice",
              "never_individual_denial_notice", "segmentation_caveat", "safety_boundary_notice",
              "template_baseline_caveat")


def _signals_panel(result, signals, state, assigned: bool) -> None:
    by_rule: dict[str, list] = {}
    for s in sorted(signals, key=lambda s: -float(s.evidence_strength)):
        by_rule.setdefault(s.rule_id, []).append(s)
    masker = lambda v: mask(v, state, assigned=assigned)  # noqa: E731
    for rule_id, group in by_rule.items():
        top = group[0]
        text = control_text(rule_id)
        try:
            sentence = reason_sentence(top, mask=masker, claims=result.claims)
        except Exception:
            sentence = text.finding
        simplified = str((top.evidence or {}).get("data_support", "")) == "PARTIAL"
        meta = [f"Kind of evidence: {label(top.domain, 'domain').lower()}",
                f"how sure the check is: {pct(top.confidence)}"]
        if len(group) > 1:
            meta.append(f"{len(group) - 1} more flag(s) like it on this case")
        if simplified:
            meta.append("ran in a simplified form on this file, so it is weaker")
        st.markdown(
            f"<div class='fwa-card'><h4>{html.escape(text.title)} {disposition_chip(top.disposition)}"
            f"</h4><p>{html.escape(sentence)}</p>"
            + (f"<p class='fwa-sub'>What this check looks for: {html.escape(text.what)}</p>"
               if text.what and is_plain_text(text.what) else "")
            + f"<p class='fwa-sub'>{html.escape(' · '.join(meta))}</p></div>",
            unsafe_allow_html=True,
        )
        for key in _NOTE_KEYS:
            value = plain_or((top.evidence or {}).get(key))
            if value:
                note(value)
    st.caption("Identities are hidden for your role. Hiding changes WHO you see, never the evidence: "
               "a hidden name is replaced by the same stand-in everywhere, so two findings about the "
               "same provider still line up.")


def is_plain_text(text: str) -> bool:
    return bool(plain_or(text))


def _signals_technical(signals, state, assigned: bool) -> None:
    for signal in sorted(signals, key=lambda s: -s.evidence_strength):
        st.markdown(
            f"**{signal.rule_id}@{signal.rule_version}** — `{signal.reason_code}` · "
            f"`{signal.qualified_rule}` · domain {signal.domain.value} · stage {signal.stage.value} · "
            f"status {signal.rule_status.value} · confidence {signal.confidence:.2f} · evidence "
            f"strength {signal.evidence_strength:.2f}"
            + (f" · peer level {signal.peer_level_used}" if signal.peer_level_used else "")
            + f"\n\nExposure AED {signal.exposure_aed:,.2f} "
              f"({'established' if signal.exposure_established else 'EXPOSURE_NOT_ESTABLISHED'}): "
              f"{signal.exposure_basis or '—'}"
        )
        facts = {
            k: _mask_value(k, v, state, assigned)
            for k, v in dict(signal.evidence).items()
            if not isinstance(v, (dict, list)) or k in ("claim_sks", "overlapping_claims")
        }
        if facts:
            dataframe(pd.DataFrame([{"field": k, "value": _stringify(v)} for k, v in facts.items()]))


_TERM_WORDS = {
    "evidence_strength": ("How strong the evidence is",
                          "Stronger evidence pushes a case up. Flags resting on the same fact count once."),
    "log_exposure": ("How much money is involved",
                     "More money pushes a case up, but gently, so one very large claim cannot take over "
                     "the queue."),
    "independent_domain_count": ("How many independent kinds of evidence agree",
                                 "Two findings from different kinds of evidence are stronger than two "
                                 "of the same kind."),
    "validated_prior_history": ("Earlier confirmed findings",
                                "Findings a reviewer has already confirmed against the same subject. "
                                "It stays at zero until reviewers record decisions."),
    "recency": ("How recent it is", "Newer activity is looked at sooner."),
    "data_quality_penalty": ("Gaps in the data (lowers urgency)",
                             "A finding resting on a stand-in measure or incomplete dates is worth less."),
    "known_exception_strength": ("A known exception applies (lowers urgency)",
                                 "A declared legitimate exception lowers urgency."),
}


def _priority_panel(case) -> None:
    score = case.priority
    if score is None:
        st.caption("This case has no urgency score.")
        return
    st.markdown(
        f"**{score.value:.0f} out of 100 — {label(score.band, 'band')}.** "
        f"{html.escape(standard_text('priority_not_decision'))}"
    )
    terms = sorted(score.terms, key=lambda t: -abs(float(t.contribution)))
    lines = []
    for t in terms:
        name, meaning = _TERM_WORDS.get(t.name, (t.label, ""))
        c = float(t.contribution)
        if abs(c) < 1e-9:
            effect = "no effect on this case"
        elif c > 0:
            effect = "pushes it up" + (" a lot" if c >= 1 else "")
        else:
            effect = "pulls it down"
        lines.append(f"- **{name}** — {effect}. {meaning}")
    st.markdown("\n".join(lines))


def _priority_technical(case) -> None:
    score = case.priority
    if score is None:
        st.caption("No priority score.")
        return
    dataframe(pd.DataFrame([t.to_dict() for t in score.terms])[
        ["label", "raw_value", "normalised", "weight", "contribution", "explanation"]].astype(str))
    st.markdown("**The arithmetic, reconstructable by hand**")
    st.code("\n".join(score.arithmetic_lines()))
    st.caption(score.to_dict()["formula"])


def _exposure_panel(case, signals) -> None:
    st.markdown(_amount_line(case, with_basis=True), unsafe_allow_html=True)
    rows = [{
        "check_name": control_text(s.rule_id).title,
        "flagged_amount": aed(s.exposure_aed),
        "amount_status": "Established" if s.exposure_established else "Not yet established",
    } for s in signals]
    friendly_table(pd.DataFrame(rows), key="ce_exposure",
                   columns=["check_name", "flagged_amount", "amount_status"])
    st.markdown(
        "The case amount counts each claim **once**, taking the largest established amount any check "
        "attached to it. Adding the checks' amounts together would count a claim twice when two checks "
        "found it. Established and not-yet-established amounts are never added together. "
        + html.escape(standard_text("not_savings"))
    )
    st.caption("How each check derived its amount is listed under 'Technical details: rule ids, "
               "versions and the raw facts behind each flag' above.")


def _model_features(result, signals) -> list[dict]:
    features: list[dict] = []
    for signal in signals:
        contributions = signal.evidence.get("shap_top_features_original_units")
        if not contributions:
            continue
        model = signal.evidence.get("model_name", "model")
        for c in contributions:
            features.append({**c, "model": model, "rule_id": signal.rule_id})
    return features


def _shap_panel(features: list[dict]) -> None:
    st.markdown(
        "An automated pattern-finder ranked this as unusual. These are the measures that mattered "
        "most, each shown in its real units beside what is typical for similar providers. "
        + html.escape(standard_text("model_only"))
    )
    shown = 0
    for f in features:
        if f.get("feature") in ("EXPLANATION_UNAVAILABLE", "EXPLANATION_FAILED"):
            st.caption("The pattern-finder's reasons for this score could not be worked out.")
            continue
        text = plain_or(f.get("plain_language"))
        if text:
            st.markdown(f"- {html.escape(text)}")
            shown += 1
    if not shown:
        st.caption("The reasons are only available in technical form (below).")
    st.markdown("**Technical details: contributions (SHAP values) in original units**")
    frame = pd.DataFrame([
        {k: f.get(k) for k in ("model", "label", "value_display", "peer_display",
                               "ratio_to_peer", "shap_value", "direction")}
        for f in features if f.get("feature") not in ("EXPLANATION_UNAVAILABLE", "EXPLANATION_FAILED")
    ])
    if not frame.empty:
        dataframe(frame.astype(str))


def _document_spans(signals) -> list[dict]:
    spans = []
    for signal in signals:
        ev = signal.evidence
        if ev.get("source_span_text") and ev.get("span_offsets"):
            spans.append({
                "claim_sk": ev.get("claim_sk") or (signal.claim_ids[0] if signal.claim_ids else ""),
                "source_span_text": ev["source_span_text"],
                "span_offsets": ev["span_offsets"],
                "extraction_confidence": ev.get("extraction_confidence"),
                "ocr_confidence": ev.get("ocr_confidence"),
                "document_language": ev.get("document_language"),
                "conflicting_field": ev.get("conflicting_field"),
                "extracted_value": ev.get("extracted_value"),
                "claim_value": ev.get("claim_value"),
                "rule_id": signal.rule_id,
            })
    return spans


def _document_panel(spans: list[dict]) -> None:
    from fwa.presentation import field_label

    st.markdown(
        "Every finding below points to the exact words in the document. A finding that cannot be "
        "tied to exact words is thrown away, not shown with a warning."
    )
    for span in spans:
        lang = {"en": "English", "ar": "Arabic"}.get(str(span.get("document_language") or ""), "")
        conf = span.get("extraction_confidence")
        caption = f"Found by: {control_text(span['rule_id']).title}"
        if lang:
            caption += f" · language: {lang}"
        if conf is not None:
            caption += f" · how sure the reading is: {pct(conf)}"
        source_span(span["source_span_text"], caption=caption)
        if span.get("conflicting_field"):
            st.markdown(
                f"The document and the claim disagree on **{field_label(span['conflicting_field']).lower()}**: "
                f"the document says “{span['extracted_value']}”, the claim says “{span['claim_value']}”."
            )
    note("These documents are SYNTHETIC: they were generated by this tool to demonstrate the "
         "document checks. The claim file itself contains no documents.")
    st.markdown("**Technical details: offsets and confidence**")
    dataframe(pd.DataFrame(spans).astype(str))


def _peer_panel(signals, state, assigned: bool) -> None:
    st.markdown(
        "These checks compare this subject with a group of similar providers. Where the group was "
        "small, the comparison was widened to a broader group and the figure pulled towards the "
        "group's typical value, so a few claims cannot make a provider look extreme by chance."
    )
    rows = []
    for s in signals:
        ev = s.evidence
        median = (ev.get("peer_median") or ev.get("peer_median_aed") or ev.get("peer_median_los")
                  or ev.get("peer_mean"))
        value = (ev.get("gross_amount_aed") or ev.get("amount_per_los_day")
                 or ev.get("length_of_stay_days") or ev.get("shrunk_rate"))
        peer_n = ev.get("peer_n")
        title = control_text(s.rule_id).title
        parts = [f"**{html.escape(title)}**:"]
        if value is not None and median is not None:
            parts.append(f"this case's figure is {plain_number(value, 2)}; similar providers are "
                         f"typically at {plain_number(median, 2)}.")
        else:
            parts.append("compared with similar providers.")
        if peer_n:
            parts.append(f"Compared with {plain_number(peer_n)} similar providers.")
        st.markdown(" ".join(parts))
        rows.append({
            "rule_id": s.rule_id, "peer_level_used": s.peer_level_used, "peer_n": peer_n,
            "peer_median": median, "value": value, "residual": ev.get("robust_residual"),
            "posterior_interval": _stringify(ev.get("posterior_interval")),
            "baseline_version": ev.get("peer_baseline_version"),
        })
    st.markdown("**Technical details: peer level, shrinkage and baseline version**")
    dataframe(pd.DataFrame(rows).astype(str))
    st.caption("peer_level_used is recorded on every comparison so a later analyst can see exactly which "
               "comparison group produced a result; the baseline version makes it reproducible.")


def _readable_ai(text: str) -> str:
    """Rule ids and reason codes in the AI text become words (display only).

    The text as generated, with its original citations, is kept verbatim in
    the technical expander beneath the panel.
    """
    text = RULE_ID_RE.sub(lambda m: f"“{control_text(m.group(0)).title}”", text or "")
    return RAW_TOKEN_RE.sub(lambda m: f"“{label(m.group(0)).lower()}”", text)


def _ai_summary(state, result, case, signals, model_features, document_spans, subject) -> None:
    narrative = result.ai.narrator.narrate(
        case, signals, model_features=model_features, document_spans=document_spans,
        actor=state.username, subject_label=subject,
    )
    ai_panel(
        _readable_ai(narrative.text) or "No sentence in the generated summary could be traced back "
                                        "to the evidence, so nothing is shown.",
        footer=(
            f"{narrative.validation.drop_count} sentence(s) removed because they could not be traced "
            f"to the evidence · advisory only: this panel does not set what should happen, the "
            f"urgency or the amount."
        ),
    )
    with st.expander("Technical details: the AI summary as generated, with its sources"):
        st.markdown(
            f"Provider `{narrative.response.provider}` · model `{narrative.response.model}` · prompt "
            f"hash `{narrative.response.prompt_hash}`."
        )
        st.code(narrative.text or "", language="text")
        if narrative.validation.dropped:
            dataframe(pd.DataFrame(
                [{"sentence": s, "reason": r} for s, r in narrative.validation.dropped]))


def _copilot_panel(state, result, case, signals) -> None:
    tour_step("Case Evidence", 4)
    st.markdown("### Ask about this case")
    st.caption(
        "The assistant only knows this case's evidence, the wording of the checks that fired and the "
        "settings. Its answers are not evidence. Ask it whether this is fraud and it will refuse: "
        "only a reviewer, looking at the evidence, can decide."
    )
    examples = [
        "Why did this fire?",
        "What exclusion might apply?",
        "What would I need to confirm this?",
        "What threshold was used?",
        "Is this fraud?",
    ]
    cols = st.columns(len(examples))
    for col, example in zip(cols, examples):
        if col.button(example, key=f"copilot_{example}", width="stretch",
                      help="Puts this question in the box below."):
            st.session_state["fwa_copilot_q"] = example
    question = st.text_input(
        "Your question", value=st.session_state.get("fwa_copilot_q", ""),
        help="Ask about why a check fired, what legitimate exception might apply or what evidence "
             "would confirm it. The assistant answers only from this case.",
    )
    if question:
        answer = result.ai.copilot.ask(question, case, signals, actor=state.username)
        ai_panel(
            _readable_ai(answer.answer),
            footer=(
                "Declined at the safety boundary — this refusal is built in and happens before any "
                "model is asked."
                if answer.refused else
                f"Sources: {', '.join(control_text(s).title if RULE_ID_RE.fullmatch(str(s)) else str(s) for s in answer.sources) or 'case evidence'} · advisory only."
            ),
            label="AI-generated answer — not evidence",
        )


# ---------------------------------------------------------------------------
# the decision form — the one primary action (§14.1)
# ---------------------------------------------------------------------------

_ACTIONS = {
    "confirm": "Agree with what policy suggests",
    "adjust": "Change what should happen (needs a special permission)",
    "clear": "No issue found — clear the case",
    "escalate": "Refer to the investigations team",
}


def _disposition_form(state, result, case, signals) -> None:
    can_record = state.can(Permission.RECORD_DISPOSITION) or state.can(
        Permission.RECORD_CLINICAL_DISPOSITION)
    if not can_record:
        note(
            f"The {label(state.role.value, 'role').lower()} role can read case evidence but cannot "
            f"record a decision. Reading a case and deciding it are kept separate on purpose."
        )
        return

    # A KEYED CONTAINER, not a hand-written <div>: the key puts a stable
    # ``.st-key-…`` class on the real container, which the stylesheet targets.
    with st.container(key="fwa_primary_action"):
        st.markdown("#### Record your decision")
        _disposition_fields(state, result, case, signals)

    # The appeal is a SECONDARY action and stays outside the highlighted box.
    _appeal_capture(state, case)


def _disposition_fields(state, result, case, signals) -> None:
    dispositions = [d.value for d in Disposition]
    categories = ["", "billing_error", "coding_error", "documentation_gap", "duplicate",
                  "utilisation_concern", "network_concern", "suspected_intentional", "cleared"]
    with st.form(f"disposition_{case.case_id}", border=False):
        c1, c2 = st.columns([1, 1])
        with c1:
            action = st.radio(
                "Your decision", list(_ACTIONS), format_func=lambda k: _ACTIONS[k],
                help=f"Agreeing records that what policy suggests ({status(case.disposition).label}) "
                     "is right. Changing it is a manual override: it needs a separate permission that "
                     "an administrator grants individually. Clearing records that you found no issue. "
                     "Referring sends the case to the investigations team — that is not a finding of "
                     "fraud.",
            )
            adjusted = st.selectbox(
                "If you are changing it: what should happen instead", dispositions,
                index=dispositions.index(case.disposition.value),
                format_func=lambda v: status(v).label,
                disabled=action != "adjust",
                help="Only used when you choose 'Change what should happen'. Only hard or expert "
                     "rules may lead to a denial or a corrected payment.",
            )
        with c2:
            category = st.selectbox(
                "What you found (category)", categories,
                format_func=lambda v: "Choose a category (optional)" if not v else label(v, "category"),
                help="The kind of problem you confirmed, if any. It feeds the reports that measure "
                     "how often each check is right.",
            )
            confirmed = st.number_input(
                "Amount you confirmed (AED)", min_value=0.0,
                value=float(case.exposure_aed) if case.exposure_established else 0.0, step=100.0,
                help="The amount you actually confirmed — not the amount flagged. The difference "
                     "between the two is why money flagged is not money saved.",
            )
        rationale = st.text_area(
            "Why? (required)",
            placeholder="What did you check, what did you find, and why does that settle it? "
                        "This is the feedback that improves the checks.",
            height=110,
            help="Required. Your reasons are stored with your decision in the audit trail.",
        )
        submitted = st.form_submit_button("Save decision", type="primary")

    if submitted:
        is_override = action == "adjust"
        disposition = (
            adjusted if is_override else
            Disposition.MONITOR_ONLY.value if action == "clear" else
            Disposition.SIU_LEAD.value if action == "escalate" else
            case.disposition.value
        )
        try:
            outcome = auth_service().record_disposition(
                state=state, case=case, disposition=disposition, rationale=rationale,
                validated_category=category or ("cleared" if action == "clear" else ""),
                confirmed_amount_aed=confirmed if action != "clear" else 0.0,
                is_override=is_override,
            )
        except AccessDenied as exc:
            st.error(
                "**You don't have permission to change what should happen on a case.** Changing it "
                "(a manual override) is a separate permission that no role has automatically; an "
                "administrator grants it to individuals."
            )
            with st.expander("Technical details"):
                st.code(str(exc))
        except ValueError as exc:
            st.error(f"**The decision was not saved.** {exc}")
        else:
            st.success(
                f"Your decision was saved (reference {outcome.review_id}). It was written to the "
                f"review record and to the audit trail with your name, the time, what should happen "
                f"before and after, and your reasons."
            )


def _appeal_capture(state, case) -> None:
    if state.can(Permission.CAPTURE_APPEAL):
        with st.expander("Record an appeal against an earlier decision"):
            outcomes = auth_service().db.outcomes_frame(state.tenant_id)
            outcomes = outcomes[outcomes["case_id"] == case.case_id] if not outcomes.empty else outcomes
            if outcomes.empty:
                st.caption("No decisions have been recorded on this case yet, so there is nothing to "
                           "appeal.")
            else:
                review_id = st.selectbox(
                    "Which decision was appealed", list(outcomes["review_id"]),
                    help="The earlier decision on this case that the appeal was against.")
                result_text = st.selectbox(
                    "Result of the appeal", ["upheld", "overturned", "partially_overturned"],
                    format_func=lambda v: label(v, "category"),
                    help="Whether the appeal kept, reversed or partly reversed the decision.")
                reason = st.text_input("Reason for the result",
                                       help="At least 10 characters; stored in the audit trail.")
                if st.button("Save appeal"):
                    if len(reason.strip()) < 10:
                        st.error("Please give a reason of at least 10 characters.")
                    else:
                        auth_service().capture_appeal(
                            state=state, review_id=review_id, result=result_text, reason=reason)
                        st.success("Appeal saved. How often decisions are overturned becomes "
                                   "measurable as appeals build up.")


#: Evidence keys that carry an identity and must be masked before rendering.
_IDENTITY_KEYS = {
    "provider_sk", "member_sk", "prior_provider_sk", "agent_id", "top_provider_sk",
    "providers", "member_sks", "provider_sks", "subject_id", "entity",
}


def _mask_value(key: str, value, state, assigned: bool):
    if key not in _IDENTITY_KEYS:
        return value
    if isinstance(value, (list, tuple)):
        return [mask(v, state, assigned=assigned) for v in value]
    return mask(value, state, assigned=assigned)


def _stringify(value) -> str:
    if isinstance(value, (dict, list)):
        return json.dumps(value, default=str)[:400]
    return str(value)
