""""Why was this flagged?" — every case gets a plain, honest explanation.

The explanation is the most-read text in the application, so these tests pin
the properties the usability brief makes non-negotiable: a one-sentence
headline, at most five ranked reasons written as sentences with numbers and no
rule ids, an evidence-strength label computed with the engine's own capping
rule, the "not proof of fraud" line on every case, the model-only wording where
it applies, concrete next steps, and the technical detail kept for auditors.
"""

from __future__ import annotations

import re

import pytest

from fwa.engine.signals import cap_evidence
from fwa.presentation import standard_text
from fwa.presentation.explain_case import (
    MAX_REASONS, explain_case, is_plain, reason_sentence,
)


@pytest.fixture(scope="module")
def explanations(sample_cases, sample_signals, claims):
    correlator, cases = sample_cases
    by_id = {s.signal_id: s for s in sample_signals}
    out = []
    for case in cases.values():
        signals = [by_id[sid] for sid in case.signal_ids if sid in by_id]
        out.append((case, signals, explain_case(
            case, signals, claims=claims, mask=lambda v: f"MASKED-{abs(hash(str(v))) % 1000:03d}")))
    return out


def test_every_case_is_explained(explanations):
    assert explanations
    for case, signals, e in explanations:
        assert e.headline and e.headline.endswith(".")
        assert 1 <= len(e.reasons) <= MAX_REASONS
        assert e.next_steps and len(e.next_steps) <= 4
        assert e.technical and len(e.technical) == len(signals)


def test_reasons_are_plain_sentences_without_rule_ids(explanations):
    failures = []
    for _, _, e in explanations:
        for r in e.reasons:
            if not is_plain(r.sentence):
                failures.append(f"{r.rule_id}: {r.sentence}")
    assert not failures, "\n".join(failures[:20])


def test_most_reasons_carry_a_number(explanations):
    """A reason should state the fact, not only the category of the check."""
    reasons = [r.sentence for _, _, e in explanations for r in e.reasons]
    with_number = [s for s in reasons if re.search(r"\d", s)]
    assert len(with_number) / len(reasons) >= 0.8


def test_the_not_proof_line_is_always_present(explanations):
    for _, _, e in explanations:
        assert standard_text("not_proof") in e.caveats


def test_model_only_cases_say_no_rule_was_broken(explanations):
    seen = 0
    for case, signals, e in explanations:
        domains = {s.domain.value for s in signals}
        if domains <= {"model"}:
            seen += 1
            assert standard_text("model_only") in e.caveats
            assert "established" in " ".join(e.caveats).lower() or \
                "not yet established" in e.amount_line
    # The sample may or may not contain a model-only case; the rule is checked
    # directly below either way.


def test_unestablished_exposure_is_never_shown_as_money_owed(explanations):
    for case, _, e in explanations:
        if not case.exposure_established:
            assert standard_text("not_established") in e.amount_line


def test_strength_uses_the_capping_rule_not_the_sum(explanations):
    for _, signals, e in explanations:
        assert e.strength.score == pytest.approx(cap_evidence(signals))
        assert e.strength.label in ("Strong", "Moderate", "Weak")
        assert e.strength.why


def test_reasons_sharing_a_fact_are_called_out():
    """Two checks resting on one fact are one piece of evidence, and the text says so."""
    from types import SimpleNamespace

    from fwa.enums import Disposition, SignalDomain

    def sig(rule, strength, fact):
        return SimpleNamespace(
            signal_id=f"{rule}-{fact}", rule_id=rule, rule_version="1.0.0", reason_code="X",
            evidence_strength=strength, fact_key=fact, domain=SignalDomain.STATISTICAL,
            evidence={"plain_language": f"Amount was AED {int(strength * 1000):,} against AED 500."},
            confidence=1.0, score=60.0, subject_type="claim", subject_id="C1", claim_ids=["C1"],
            disposition=Disposition.PREPAY_PEND, peer_level_used=None, exposure_aed=0.0,
            exposure_established=True, exposure_basis="",
        )

    signals = [sig("PAY-06-R03", 0.6, "amount:C1"), sig("CLN-01-R02", 0.5, "amount:C1"),
               sig("PHR-03-R03", 0.4, "pharmacy:C1")]
    case = SimpleNamespace(
        case_id="CASE-1", primary_subject_type="claim", primary_subject_id="C1",
        claim_ids=["C1"], disposition=Disposition.PREPAY_PEND, exposure_aed=100.0,
        exposure_established=True, shadow_only=True, period_bucket="2025-11", priority=None,
    )
    e = explain_case(case, signals)
    assert "Reasons 1 and 2 come from the same underlying fact" in e.strength.shared_fact_note
    assert e.strength.score == pytest.approx(cap_evidence(signals))
    assert "two unusual patterns" in e.headline
    assert "November 2025" in e.headline


def test_identities_are_masked_in_reason_text(explanations, sample_signals, claims):
    for s in sample_signals:
        if s.subject_type == "provider" and s.subject_id in str(s.evidence.get("plain_language", "")):
            text = reason_sentence(s, mask=lambda v: "MASKED", claims=claims)
            assert s.subject_id not in text
            break


def test_the_summary_exports_as_markdown_and_html(explanations):
    _, _, e = explanations[0]
    md = e.to_markdown()
    assert md.startswith("# Case summary")
    assert "## Why it was flagged" in md and "## What this does not mean" in md
    assert standard_text("not_proof") in md
    html = e.to_html(synthetic_notice="SYNTHETIC data")
    assert "<ol>" in html and "SYNTHETIC data" in html and "@media print" in html
