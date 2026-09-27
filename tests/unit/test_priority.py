"""The §4.11 priority formula, term by term.

    priority = 100 * sigmoid(
        1.10*evidence_strength + 0.80*log1p(exposure_aed/1000)
      + 0.55*independent_domain_count + 0.45*validated_prior_history + 0.25*recency
      - 0.60*data_quality_penalty - 0.50*known_exception_strength)

    "A hard-edit disposition is never overridden by a low priority score:
     disposition is a policy decision, and priority only orders the review
     queue, never substitutes for the disposition rule itself."

The weights are reproduced verbatim from the manuscript, so the first test
checks them against the registry rather than against a copy in the test — a
weight that drifts silently is a scoring change nobody approved. The rest check
each term's direction and the decomposition that lets a reviewer reconstruct
the number by hand.
"""

from __future__ import annotations

import datetime as _dt
import math

import pytest

from fwa.cases.priority import PriorityScorer
from fwa.engine.signals import Signal
from fwa.enums import Disposition, SignalDomain

#: The manuscript's own numbers. If the registry ever disagrees with these, the
#: formula has changed without the manuscript changing with it.
MANUSCRIPT_WEIGHTS = {
    "evidence_strength": 1.10,
    "log_exposure": 0.80,
    "independent_domain_count": 0.55,
    "validated_prior_history": 0.45,
    "recency": 0.25,
    "data_quality_penalty": -0.60,
    "known_exception_strength": -0.50,
}

AS_OF = _dt.date(2026, 9, 20)


def _signal(**kwargs) -> Signal:
    defaults = dict(
        signal_id="sig-1", rule_id="PAY-01-R01", rule_version="1.0.0", scenario_id="PAY-01",
        tenant_id="T001", subject_type="claim", subject_id="C1", claim_ids=["C1"],
        event_time=_dt.datetime(2026, 9, 19), detection_time=_dt.datetime(2026, 9, 20),
        period_bucket="2026-09", score=70.0, confidence=1.0, evidence_strength=0.5,
        disposition=Disposition.POSTPAY_AUDIT, reason_code="R", domain=SignalDomain.RULE,
        stage="PREPAY_SYNC", rule_status="shadow", exposure_aed=0.0, exposure_basis="",
        exposure_established=True, evidence={}, fact_key="fact-1",
    )
    defaults.update(kwargs)
    return Signal(**defaults)


@pytest.fixture(scope="module")
def scorer(config):
    return PriorityScorer(config)


# ------------------------------------------------------------- the weights


def test_the_weights_are_the_manuscripts_weights(config):
    weights = dict(config.get("priority_weights"))
    assert weights == pytest.approx(MANUSCRIPT_WEIGHTS)


def test_the_penalties_are_negative_and_the_rest_positive(config):
    weights = dict(config.get("priority_weights"))
    for name in ("data_quality_penalty", "known_exception_strength"):
        assert weights[name] < 0
    for name in ("evidence_strength", "log_exposure", "independent_domain_count",
                 "validated_prior_history", "recency"):
        assert weights[name] > 0


# --------------------------------------------------------- the decomposition


def test_the_score_names_all_seven_terms(scorer):
    score = scorer.score([_signal()], exposure_aed=1000.0, as_of=AS_OF)
    assert {t.name for t in score.terms} == set(MANUSCRIPT_WEIGHTS)


def test_the_terms_sum_to_the_logit_and_the_logit_gives_the_value(scorer):
    """Reconstructable by hand, which is what the evidence panel promises."""
    score = scorer.score([_signal()], exposure_aed=5000.0, as_of=AS_OF)
    assert sum(t.contribution for t in score.terms) == pytest.approx(score.logit)
    assert score.value == pytest.approx(100.0 / (1.0 + math.exp(-score.logit)))


def test_each_term_shows_weight_times_normalised_value(scorer):
    score = scorer.score([_signal()], exposure_aed=5000.0, as_of=AS_OF)
    for term in score.terms:
        assert term.contribution == pytest.approx(term.weight * term.normalised)
        assert term.explanation, f"{term.name} does not explain itself"


def test_the_arithmetic_is_printable(scorer):
    lines = scorer.score([_signal()], exposure_aed=5000.0, as_of=AS_OF).arithmetic_lines()
    assert len(lines) == len(MANUSCRIPT_WEIGHTS) + 2
    assert "sigmoid" in lines[-1]


def test_the_dictionary_form_carries_the_formula_and_the_boundary(scorer):
    payload = scorer.score([_signal()], exposure_aed=1000.0, as_of=AS_OF).to_dict()
    assert "sigmoid" in payload["formula"]
    assert "ORDERS THE QUEUE ONLY" in payload["boundary_note"]


# --------------------------------------------------------------- directions


def test_stronger_evidence_raises_priority(scorer):
    weak = scorer.score([_signal(evidence_strength=0.1)], exposure_aed=1000.0, as_of=AS_OF)
    strong = scorer.score([_signal(evidence_strength=0.9)], exposure_aed=1000.0, as_of=AS_OF)
    assert strong.value > weak.value


def test_larger_exposure_raises_priority_but_logarithmically(scorer):
    """A single very large claim must not dominate the queue."""
    small = scorer.score([_signal()], exposure_aed=1_000.0, as_of=AS_OF)
    large = scorer.score([_signal()], exposure_aed=10_000.0, as_of=AS_OF)
    huge = scorer.score([_signal()], exposure_aed=1_000_000.0, as_of=AS_OF)

    assert small.value < large.value < huge.value
    first_step = large.value - small.value
    second_step = huge.value - large.value
    assert second_step < first_step * 3, "Exposure is entering close to linearly."


def test_more_independent_domains_raise_priority(scorer):
    one_domain = [_signal(signal_id="a", fact_key="f1"),
                  _signal(signal_id="b", fact_key="f2")]
    two_domains = [_signal(signal_id="a", fact_key="f1"),
                   _signal(signal_id="b", fact_key="f2", domain=SignalDomain.NETWORK)]

    assert (scorer.score(two_domains, exposure_aed=1000.0, as_of=AS_OF).value
            > scorer.score(one_domain, exposure_aed=1000.0, as_of=AS_OF).value)


def test_a_data_quality_penalty_lowers_priority(scorer):
    clean = scorer.score([_signal()], exposure_aed=1000.0, as_of=AS_OF)
    proxy = scorer.score([_signal(data_quality_penalty=0.5)], exposure_aed=1000.0, as_of=AS_OF)
    assert proxy.value < clean.value


def test_a_known_exception_lowers_priority(scorer):
    plain = scorer.score([_signal()], exposure_aed=1000.0, as_of=AS_OF)
    excepted = scorer.score([_signal(known_exception_strength=0.8)],
                            exposure_aed=1000.0, as_of=AS_OF)
    assert excepted.value < plain.value


def test_an_older_signal_scores_lower(scorer):
    recent = scorer.score([_signal(event_time=_dt.datetime(2026, 9, 19))],
                          exposure_aed=1000.0, as_of=AS_OF)
    stale = scorer.score([_signal(event_time=_dt.datetime(2025, 1, 1))],
                         exposure_aed=1000.0, as_of=AS_OF)
    assert stale.value < recent.value


def test_validated_prior_history_raises_priority_and_starts_at_zero(scorer):
    """It is confirmed reviewer history, not a system flag — so it is zero by default."""
    none = scorer.score([_signal()], exposure_aed=1000.0, as_of=AS_OF)
    history = scorer.score([_signal()], exposure_aed=1000.0, as_of=AS_OF,
                           validated_prior_history=1.0)
    assert history.value > none.value

    term = next(t for t in none.terms if t.name == "validated_prior_history")
    assert term.normalised == 0.0
    assert "CONFIRMED history" in term.explanation


# ------------------------------------------------------------ capped evidence


def test_repeating_one_fact_does_not_raise_priority(scorer):
    """§3.8 flowing through into the queue order, which is where it matters."""
    one = [_signal(signal_id="a", fact_key="amount:C1", evidence_strength=0.6)]
    five = [_signal(signal_id=f"s{i}", fact_key="amount:C1", evidence_strength=0.6)
            for i in range(5)]

    assert (scorer.score(five, exposure_aed=1000.0, as_of=AS_OF).value
            == pytest.approx(scorer.score(one, exposure_aed=1000.0, as_of=AS_OF).value))


def test_the_evidence_term_reports_facts_not_signals(scorer):
    signals = [_signal(signal_id=f"s{i}", fact_key="amount:C1") for i in range(3)]
    term = next(t for t in scorer.score(signals, exposure_aed=0.0, as_of=AS_OF).terms
                if t.name == "evidence_strength")
    assert "3 signal(s)" in term.explanation
    assert "1 distinct underlying fact" in term.explanation


# ------------------------------------------------------------------- bounds


@pytest.mark.parametrize("exposure", [0.0, 1.0, 1e6, 1e9])
def test_the_score_stays_inside_zero_to_one_hundred(scorer, exposure):
    score = scorer.score([_signal(evidence_strength=1.0)], exposure_aed=exposure, as_of=AS_OF)
    assert 0.0 <= score.value <= 100.0


def test_a_negative_exposure_is_treated_as_zero(scorer):
    negative = scorer.score([_signal()], exposure_aed=-5000.0, as_of=AS_OF)
    zero = scorer.score([_signal()], exposure_aed=0.0, as_of=AS_OF)
    assert negative.value == pytest.approx(zero.value)


def test_a_signal_with_no_event_time_scores_without_crashing(scorer):
    score = scorer.score([_signal(event_time=None)], exposure_aed=1000.0, as_of=AS_OF)
    assert 0.0 <= score.value <= 100.0


def test_no_signals_still_produces_a_score(scorer):
    assert 0.0 <= scorer.score([], exposure_aed=0.0, as_of=AS_OF).value <= 100.0


# -------------------------------------------------------------------- bands


@pytest.mark.parametrize("value,expected", [
    (95.0, "P1"), (80.0, "P1"), (70.0, "P2"), (55.0, "P3"), (40.0, "P4"), (10.0, "P5"),
])
def test_bands_are_assigned_at_the_documented_cuts(value, expected):
    assert PriorityScorer._band(value)[0] == expected


def test_every_score_carries_a_labelled_band_not_a_naked_number(scorer):
    """Build brief §14.1: a bare 73.4 tells a reviewer nothing about what to do."""
    score = scorer.score([_signal()], exposure_aed=1000.0, as_of=AS_OF)
    assert score.band
    assert score.band_label
    assert not score.band_label.replace(".", "").isdigit()


# ---------------------------------------------------- the boundary above it


def test_priority_does_not_appear_in_any_disposition_decision():
    """§4.11: priority orders the queue; it never substitutes for the disposition rule.

    Asserted on the source of the two places a disposition is decided — the
    legality validator and the correlator's resolution — because the property
    is the *absence* of a dependency, which no single run can demonstrate.
    """
    import inspect

    from fwa.cases.correlate import CaseCorrelator
    from fwa.engine.contract import validate_disposition_legality

    for func in (validate_disposition_legality, CaseCorrelator._strongest_disposition):
        source = inspect.getsource(func)
        assert "priority" not in source.lower(), (
            f"{func.__qualname__} consults the priority score to decide a disposition."
        )
