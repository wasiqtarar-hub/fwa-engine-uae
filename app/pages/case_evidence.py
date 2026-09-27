"""Case Evidence — the §6.7 / Table 6.2 evidence bundle, made real.

Build brief §14, item 3, and the evidence rules that follow it:

    "**never display a bare score without its decomposition; always show
     ``rule_id@version`` beside any finding; AI-generated text always in its
     distinct advisory panel; the dataset strip is persistent and not
     dismissible.**"

And the §14.1 usability bar, which matters most on this page:

    "The case view opens on a clear verdict line (what fired, what it's worth,
     what's being asked of you) with the full evidence, SHAP decomposition, peer
     statistics and audit trail in expandable sections beneath. Depth on demand,
     never depth by default."

So: one sentence at the top in plain language, the disposition control
immediately under it, and everything else collapsed.
"""

from __future__ import annotations

import json

import pandas as pd
import streamlit as st

from common import (
    ai_panel, auth_service, banner, boundary_note, dataframe, disposition_chip, empty_state,
    guide, mask, mono, note, pipeline, priority_chip, require_page, session_banner,
    source_span, tiles,
)
from fwa.auth import AccessDenied, Permission
from fwa.enums import DISPOSITION_MEANINGS, Disposition, REVIEWER_ASK


def render() -> None:
    state = require_page("Case Evidence", Permission.VIEW_CASE_EVIDENCE)
    session_banner(state)

    result = pipeline()
    queue = result.queue(state.tenant_id)

    st.markdown("# Case evidence")
    banner()

    if queue.empty:
        empty_state("No cases available", "Nothing has been correlated for your tenant.")
        return

    case_ids = list(queue["case_id"])
    current = st.session_state.get("fwa_selected_case")
    index = case_ids.index(current) if current in case_ids else 0
    case_id = st.selectbox("Case", case_ids, index=index)
    st.session_state["fwa_selected_case"] = case_id

    case = result.cases.cases[case_id]
    signals = result.signals_for_case(case_id)
    assigned = case_id in state.assigned_case_ids
    subject = mask(case.primary_subject_id, state, assigned=assigned)

    if st.session_state.get("fwa_guided"):
        guide("Step 2 of 5 — the verdict line",
              "Everything you need to decide whether to look deeper is in the next three lines: "
              "what fired, what it is worth, and what is being asked of you. Everything below "
              "them is collapsed on purpose.")

    # ======================================================================
    # 1. VERDICT LINE — what fired, what it's worth, what's asked of you
    # ======================================================================
    headline = _headline(signals, subject, case)
    st.markdown(
        f"<div class='fwa-verdict'>"
        f"<div class='fwa-line'>{headline}</div>"
        f"<div>{disposition_chip(case.disposition)} "
        f"{priority_chip(case.priority.band, case.priority.band_label, case.priority.value)} "
        f"<span class='fwa-tag'>{case.case_id}</span> "
        f"<span class='fwa-tag'>{case.period_bucket}</span></div>"
        f"<div class='fwa-ask'><strong>What it's worth.</strong> "
        f"{_exposure_line(case)}<br>"
        f"<strong>What is being asked of you.</strong> "
        f"{REVIEWER_ASK.get(case.disposition, '')}</div>"
        f"</div>",
        unsafe_allow_html=True,
    )

    # ======================================================================
    # 2. THE PRIMARY ACTION — one per screen (§14.1)
    # ======================================================================
    _disposition_form(state, result, case, signals)

    # ======================================================================
    # 3. EVERYTHING ELSE — depth on demand
    # ======================================================================
    if st.session_state.get("fwa_guided"):
        guide("Step 3 of 5 — the evidence",
              "Each panel below is one kind of evidence. Every finding carries its "
              "rule_id@version, and every peer comparison says which peer level produced it — "
              "that is what makes a result reproducible later.")

    with st.expander(f"Contributing signals ({len(signals)})", expanded=True):
        _signals_panel(signals, state, assigned)
        st.caption(
            "Identity fields are masked for your role. Masking hides WHO, never the evidence "
            " — a pseudonym is stable, so two findings about the same provider still "
            "correlate on the page."
        )

    with st.expander("Priority score — term by term"):
        _priority_panel(case)

    with st.expander("Exposure — how the amount was derived"):
        _exposure_panel(case, signals)

    model_features = _model_features(result, signals)
    if model_features:
        with st.expander("Model explanation — SHAP in original units"):
            _shap_panel(model_features)

    document_spans = _document_spans(signals)
    if document_spans:
        with st.expander("Documentation — source spans"):
            _document_panel(document_spans)

    peer_signals = [s for s in signals if s.peer_level_used]
    if peer_signals:
        with st.expander("Peer comparison and shrinkage"):
            _peer_panel(peer_signals)

    # ---- AI narrative, in its distinct advisory panel ---------------------
    if result.ai.narrator is not None:
        if st.session_state.get("fwa_guided"):
            guide("Step 4 of 5 — the AI panel",
                  "The panel below looks different from every evidence panel on purpose. It is "
                  "advisory. Every factual sentence in it carries a citation, and any sentence "
                  "whose citation did not resolve was removed before you saw it.")
        st.markdown("### Summary")
        narrative = result.ai.narrator.narrate(
            case, signals, model_features=model_features, document_spans=document_spans,
            actor=state.username, subject_label=subject,
        )
        ai_panel(
            narrative.text or "No sentence in the generated summary survived the groundedness "
                              "check, so nothing is shown.",
            footer=(
                f"{narrative.validation.drop_count} sentence(s) dropped as uncited or "
                f"unresolvable · provider {narrative.response.provider} · "
                f"model {narrative.response.model} · prompt hash "
                f"{narrative.response.prompt_hash} · advisory only: this panel does not set the "
                f"disposition, the priority or the exposure."
            ),
        )
        if narrative.validation.dropped:
            with st.expander(f"What was dropped, and why ({narrative.validation.drop_count})"):
                dataframe(pd.DataFrame(
                    [{"sentence": s, "reason": r} for s, r in narrative.validation.dropped]))

    # ---- copilot ----------------------------------------------------------
    if result.ai.copilot is not None:
        if st.session_state.get("fwa_guided"):
            guide("Step 5 of 5 — the copilot",
                  "Ask it why a rule fired, what exclusions it declares, or what evidence would "
                  "confirm it. Ask it whether this is fraud and it will refuse — that refusal is "
                  "hard-coded and happens before any model is called.")
        _copilot_panel(state, result, case, signals)

    # ---- audit trail -------------------------------------------------------
    with st.expander("Audit trail for this case"):
        events = result.audit.to_dataframe(state.tenant_id)
        if not events.empty:
            events = events[events["subject"] == case.case_id]
        dataframe(events[["event_time", "event_type", "actor", "actor_role", "reason"]]
                  if not events.empty else pd.DataFrame())
        st.caption("Append-only and hash-chained. No role can delete from it.")


# ---------------------------------------------------------------------------
# panels
# ---------------------------------------------------------------------------


def _headline(signals, subject: str, case) -> str:
    """One plain-language sentence. §14.1: lead with the sentence."""
    if not signals:
        return f"Case on {subject} with no surviving signals."
    primary = max(signals, key=lambda s: s.evidence_strength)
    plain = (
        primary.evidence.get("plain_language")
        or primary.evidence.get("shrinkage_explanation")
        or primary.evidence.get("what_the_reviewer_must_verify")
    )
    if plain:
        lead = str(plain)
    else:
        lead = (
            f"{primary.rule_id} fired on {subject} with reason code "
            f"{primary.reason_code}."
        )
    if len(signals) > 1:
        others = len({s.rule_id for s in signals}) - 1
        if others > 0:
            lead += f" {others} further control(s) fired on the same subject."
    return lead


def _exposure_line(case) -> str:
    if case.exposure_established:
        return f"AED {case.exposure_aed:,.2f}. {case.exposure_basis}"
    return (
        f"AED {case.exposure_aed:,.2f} — <strong>exposure not yet established</strong>. "
        f"{case.exposure_basis}"
    )


def _signals_panel(signals, state, assigned: bool) -> None:
    for signal in sorted(signals, key=lambda s: -s.evidence_strength):
        st.markdown(
            f"<div class='fwa-card'>"
            f"<h4>{signal.rule_id} — {signal.reason_code} "
            f"<span class='fwa-tag'>{signal.qualified_rule}</span>"
            f"{' ' + disposition_chip(signal.disposition)}</h4>"
            f"<p class='fwa-sub'>{signal.domain.value} evidence · stage {signal.stage.value} · "
            f"status {signal.rule_status.value} · confidence {signal.confidence:.0%}"
            + (f" · peer level {signal.peer_level_used}" if signal.peer_level_used else "")
            + "</p></div>",
            unsafe_allow_html=True,
        )
        evidence = dict(signal.evidence)
        for key in ("plain_language", "shrinkage_explanation", "what_the_reviewer_must_verify"):
            if evidence.get(key):
                st.markdown(f"> {evidence.pop(key)}")
        for key in ("proxy_note", "not_a_fraud_label_notice", "not_fraud_evidence_alone_notice",
                    "never_individual_denial_notice", "segmentation_caveat",
                    "safety_boundary_notice", "template_baseline_caveat"):
            if evidence.get(key):
                note(str(evidence.pop(key)))
        facts = {
            k: _mask_value(k, v, state, assigned)
            for k, v in evidence.items()
            if not isinstance(v, (dict, list)) or k in ("claim_sks", "overlapping_claims")
        }
        if facts:
            st.markdown("**The underlying facts**")
            dataframe(pd.DataFrame(
                [{"field": k, "value": _stringify(v)} for k, v in facts.items()]))
        st.markdown("---")


def _priority_panel(case) -> None:
    score = case.priority
    st.markdown(
        f"**{score.value:.1f} / 100** — {score.band} {score.band_label}. "
        f"Priority orders the queue only; it never sets or overrides a disposition."
    )
    dataframe(pd.DataFrame([t.to_dict() for t in score.terms])[
        ["label", "raw_value", "normalised", "weight", "contribution", "explanation"]])
    st.markdown("**The arithmetic, reconstructable by hand**")
    mono("\n".join(score.arithmetic_lines()))
    st.caption(score.to_dict()["formula"])


def _exposure_panel(case, signals) -> None:
    st.markdown(f"**{_exposure_line(case)}**", unsafe_allow_html=True)
    rows = [{
        "rule_id": s.rule_id,
        "exposure_aed": round(s.exposure_aed, 2),
        "established": s.exposure_established,
        "basis": s.exposure_basis,
    } for s in signals]
    dataframe(pd.DataFrame(rows))
    note(
        "Case exposure sums over DISTINCT CLAIMS, each counted once, taking the largest "
        "established exposure attributed to each. Summing the signals directly would "
        "double-count a claim that two controls both found."
    )


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
        "Contributions are shown **in original units beside the peer median**, not as bare SHAP "
        "magnitudes — that is what the promotion gate requires and what a reviewer can act on."
    )
    for f in features:
        if f.get("feature") in ("EXPLANATION_UNAVAILABLE", "EXPLANATION_FAILED"):
            note(str(f.get("note", "No explanation available for this model.")))
            continue
        st.markdown(f"- {f.get('plain_language', f.get('label'))}")
    frame = pd.DataFrame([
        {k: f.get(k) for k in ("model", "label", "value_display", "peer_display",
                               "ratio_to_peer", "shap_value", "direction")}
        for f in features if f.get("feature") not in ("EXPLANATION_UNAVAILABLE", "EXPLANATION_FAILED")
    ])
    dataframe(frame)


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
    st.markdown(
        "Every finding below is anchored to the exact text at the stated offsets. A finding with "
        "no source span is **discarded by the pipeline**, not displayed with a caveat."
    )
    for span in spans:
        source_span(
            span["source_span_text"],
            caption=(
                f"{span['rule_id']} · offsets {span['span_offsets']} · language "
                f"{span.get('document_language')} · extraction confidence "
                f"{span.get('extraction_confidence')} (OCR {span.get('ocr_confidence')})"
            ),
        )
        if span.get("conflicting_field"):
            st.markdown(
                f"Conflict on **{span['conflicting_field']}** — document says "
                f"`{span['extracted_value']}`, claim says `{span['claim_value']}`."
            )
    note(
        "These documents are SYNTHETIC and were generated by this artefact. the claim extract contains "
        "no documents of any kind. The pipeline is being demonstrated, not measured."
    )


def _peer_panel(signals) -> None:
    rows = []
    for s in signals:
        ev = s.evidence
        rows.append({
            "rule_id": s.rule_id,
            "peer_level_used": s.peer_level_used,
            "peer_n": ev.get("peer_n"),
            "peer_median": ev.get("peer_median") or ev.get("peer_median_aed")
            or ev.get("peer_median_los") or ev.get("peer_mean"),
            "value": ev.get("gross_amount_aed") or ev.get("amount_per_los_day")
            or ev.get("length_of_stay_days") or ev.get("shrunk_rate"),
            "residual": ev.get("robust_residual"),
            "posterior_interval": _stringify(ev.get("posterior_interval")),
            "baseline_version": ev.get("peer_baseline_version"),
        })
    dataframe(pd.DataFrame(rows))
    note(
        "`peer_level_used` is recorded on every peer comparison because the peer design requires it: a later "
        "analyst must be able to see exactly which comparison group produced a given result. The "
        "baseline version is the hash that makes the comparison reproducible."
    )


def _copilot_panel(state, result, case, signals) -> None:
    st.markdown("### Ask about this case")
    st.caption(
        "Bounded to this case's evidence, the rule text of the controls that fired, and the "
        "parameter registry. Nothing else."
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
        if col.button(example, key=f"copilot_{example}", width="stretch"):
            st.session_state["fwa_copilot_q"] = example
    question = st.text_input("Question", value=st.session_state.get("fwa_copilot_q", ""))
    if question:
        answer = result.ai.copilot.ask(question, case, signals, actor=state.username)
        ai_panel(
            answer.answer,
            footer=(
                "Refused at the safety boundary — this refusal is hard-coded and happens before "
                "any model is called."
                if answer.refused else
                f"Sources: {', '.join(answer.sources) or 'case evidence'} · advisory only."
            ),
            label="AI-generated answer — not evidence",
        )


# ---------------------------------------------------------------------------
# the disposition control — the one primary action (§14.1)
# ---------------------------------------------------------------------------


def _disposition_form(state, result, case, signals) -> None:
    can_record = state.can(Permission.RECORD_DISPOSITION) or state.can(
        Permission.RECORD_CLINICAL_DISPOSITION)
    if not can_record:
        note(
            f"The {state.role.value} role has read access to case evidence but may not record a "
            f"disposition. This is the separation between reading a case and deciding it."
        )
        return

    # A KEYED CONTAINER, not a hand-written <div>. Streamlit sanitises each
    # ``st.markdown`` block independently and closes any tag left open inside
    # it, so an opening div written on its own renders as an empty bordered box
    # and never wraps the form it was meant to frame. The key puts a stable
    # ``.st-key-…`` class on the real container, which the stylesheet targets.
    with st.container(key="fwa_primary_action"):
        st.markdown("#### Record your decision")
        _disposition_fields(state, result, case, signals)

    # The appeal is a SECONDARY action and stays outside the highlighted box.
    # §14.1 asks for one primary action per screen, and a box containing two
    # things to do is no longer pointing at one of them.
    _appeal_capture(state, case)


def _disposition_fields(state, result, case, signals) -> None:
    with st.form(f"disposition_{case.case_id}", border=False):
        c1, c2 = st.columns([1, 1])
        with c1:
            action = st.radio(
                "Decision",
                ["Confirm the disposition", "Adjust the disposition", "Clear — no issue",
                 "Escalate to SIU"],
                help="Confirming records that policy's disposition is correct. Adjusting is a "
                     "manual override and requires a distinct permission (PAY-11).",
            )
            adjusted = st.selectbox(
                "Adjusted disposition", [d.value for d in Disposition],
                index=[d.value for d in Disposition].index(case.disposition.value),
                disabled=action != "Adjust the disposition",
            )
        with c2:
            category = st.selectbox(
                "Validated category",
                ["", "billing_error", "coding_error", "documentation_gap", "duplicate",
                 "utilisation_concern", "network_concern", "suspected_intentional", "cleared"],
            )
            confirmed = st.number_input(
                "Confirmed amount (AED)", min_value=0.0,
                value=float(case.exposure_aed) if case.exposure_established else 0.0, step=100.0,
                help="What you have actually validated — not the flagged exposure. The "
                     "difference between the two is the whole point.",
            )
        rationale = st.text_area(
            "Rationale (required)",
            placeholder="What did you check, what did you find, and why does that settle it? "
                        "This is the feedback that tunes the rule (FR7).",
            height=110,
        )
        submitted = st.form_submit_button("Submit decision", type="primary")

    if submitted:
        is_override = action == "Adjust the disposition"
        disposition = (
            adjusted if is_override else
            Disposition.MONITOR_ONLY.value if action == "Clear — no issue" else
            Disposition.SIU_LEAD.value if action == "Escalate to SIU" else
            case.disposition.value
        )
        try:
            outcome = auth_service().record_disposition(
                state=state, case=case, disposition=disposition, rationale=rationale,
                validated_category=category or ("cleared" if action.startswith("Clear") else ""),
                confirmed_amount_aed=confirmed if not action.startswith("Clear") else 0.0,
                is_override=is_override,
            )
        except AccessDenied as exc:
            st.error(
                f"{exc} Manual adjudication override is a **distinct permission**, not implied by "
                f"any role. An administrator grants it individually."
            )
        except ValueError as exc:
            st.error(str(exc))
        else:
            st.success(
                f"Recorded as `{outcome.review_id}`. Written to `review_outcome` and to the "
                f"append-only audit log with your username, the time, the before/after "
                f"disposition and your rationale."
            )


def _appeal_capture(state, case) -> None:
    if state.can(Permission.CAPTURE_APPEAL):
        with st.expander("Capture an appeal against an earlier decision"):
            outcomes = auth_service().db.outcomes_frame(state.tenant_id)
            outcomes = outcomes[outcomes["case_id"] == case.case_id] if not outcomes.empty else outcomes
            if outcomes.empty:
                st.caption("No decisions have been recorded against this case yet.")
            else:
                review_id = st.selectbox("Decision", list(outcomes["review_id"]))
                result_text = st.selectbox("Appeal result",
                                           ["upheld", "overturned", "partially_overturned"])
                reason = st.text_input("Reason")
                if st.button("Record appeal"):
                    if len(reason.strip()) < 10:
                        st.error("A reason of at least 10 characters is required.")
                    else:
                        auth_service().capture_appeal(
                            state=state, review_id=review_id, result=result_text, reason=reason)
                        st.success("Appeal recorded. Overturn rate becomes measurable once "
                                   "appeals accumulate.")


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
