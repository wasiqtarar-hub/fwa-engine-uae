"""Empirical-Bayes shrinkage and posterior intervals (manuscript §4.6).

    "Apply empirical-Bayes shrinkage toward the peer mean for small denominators
     and report a posterior interval, so a provider with five claims never
     outranks a provider with five thousand on a noisy rate."

That sentence is a testable claim about ordering, and
:func:`test_five_claims_never_outranks_five_thousand` is it. The rest establish
the pieces it rests on: the prior is fitted from the peer distribution rather
than assumed, the shrinkage is strongest where the denominator is smallest, and
an estimate whose posterior interval is too wide is excluded from ranking
altogether rather than ranked with a caveat nobody reads.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fwa.statistical.shrinkage import (
    fit_beta_prior, shrink_frame, shrink_rate, shrinkage_demonstration,
)


@pytest.fixture(scope="module")
def prior():
    """A prior fitted from 200 entities whose true rates genuinely differ.

    Drawing every entity's rate from the same fixed probability would leave
    only binomial noise between them, and method-of-moments would read that
    near-zero dispersion as near-certainty about the peer mean — a prior so
    strong that nothing is ever shrunk far and nothing is ever excluded. Real
    provider populations are over-dispersed, so the fixture draws each entity's
    own rate from Beta(2, 18) (mean 10%) and then samples its claims.
    """
    rng = np.random.default_rng(20260920)
    opportunities = rng.integers(50, 500, size=200)
    rates = rng.beta(2.0, 18.0, size=200)
    events = rng.binomial(opportunities, rates)
    return fit_beta_prior(events, opportunities)


# ------------------------------------------------------------------ the prior


def test_the_prior_is_fitted_from_the_peers_not_assumed(prior):
    assert prior.peer_mean == pytest.approx(0.1, abs=0.03)
    assert prior.alpha > 0 and prior.beta > 0
    assert prior.n_entities == 200
    assert prior.method


def test_the_prior_mean_equals_alpha_over_alpha_plus_beta(prior):
    assert prior.alpha / (prior.alpha + prior.beta) == pytest.approx(prior.peer_mean, abs=0.02)


def test_too_few_entities_falls_back_to_a_weak_prior_and_says_so():
    """Three providers are not a peer distribution; the method must admit it."""
    weak = fit_beta_prior([1, 2], [10, 20], min_entities=3)
    assert weak.method
    assert weak.note
    assert weak.prior_strength < 10, "A prior fitted from nothing must not be strong."


def test_a_degenerate_peer_distribution_does_not_crash():
    """Every entity at the same rate leaves no variance to fit."""
    everybody_identical = fit_beta_prior([10] * 50, [100] * 50)
    assert everybody_identical.alpha > 0
    assert everybody_identical.beta > 0


def test_no_events_anywhere_is_handled():
    none_at_all = fit_beta_prior([0] * 50, [100] * 50)
    assert none_at_all.peer_mean == pytest.approx(0.0, abs=1e-6)
    assert none_at_all.alpha > 0


# ------------------------------------------------------------- the shrinkage


def test_the_headline_claim_five_claims_never_outranks_five_thousand(prior):
    """§4.6's own example, in the form the manuscript states it.

    Both providers show the same 20% observed rate against a peer base rate
    near 10%. The five-claim provider's 20% is one event out of five, which is
    entirely consistent with the peer rate; the five-thousand-claim provider's
    is not. After shrinkage the second ranks above the first, which is the
    ordering §4.6 exists to produce.
    """
    small = shrink_rate("small", 1, 5, prior)          # observed 20%
    large = shrink_rate("large", 1000, 5000, prior)    # observed 20%

    assert small.observed_rate == pytest.approx(large.observed_rate)
    assert small.shrunk_rate < large.shrunk_rate, (
        "A five-claim provider with the same observed rate must rank below a "
        "five-thousand-claim one."
    )
    assert small.posterior_width > large.posterior_width


def test_shrinkage_pulls_toward_the_peer_mean(prior):
    """An extreme small-denominator rate is pulled in; the direction depends on which side."""
    high = shrink_rate("high", 4, 5, prior)   # observed 80%, peers ~10%
    low = shrink_rate("low", 0, 5, prior)     # observed 0%

    assert high.shrunk_rate < high.observed_rate
    assert low.shrunk_rate > low.observed_rate
    assert low.shrunk_rate < high.shrunk_rate


def test_a_large_denominator_is_barely_moved(prior):
    large = shrink_rate("large", 2000, 10_000, prior)
    assert large.shrunk_rate == pytest.approx(large.observed_rate, abs=0.01)


def test_the_shrinkage_is_monotone_in_the_denominator(prior):
    """More evidence, less pull — at every step, not just at the extremes."""
    rates = [shrink_rate(f"n{n}", int(0.8 * n), n, prior) for n in (5, 20, 100, 500, 5000)]
    distances = [abs(r.shrunk_rate - prior.peer_mean) for r in rates]
    assert distances == sorted(distances), distances


def test_zero_opportunities_is_not_a_zero_rate(prior):
    """No denominator means no estimate, not an estimate of nothing."""
    nothing = shrink_rate("empty", 0, 0, prior)
    assert nothing.observed_rate is None
    assert nothing.excluded_from_ranking
    assert "no opportunities" in nothing.exclusion_reason


# ------------------------------------------------------- posterior intervals


def test_the_posterior_interval_brackets_the_estimate(prior):
    r = shrink_rate("e", 20, 100, prior)
    assert r.posterior_low <= r.shrunk_rate <= r.posterior_high
    assert 0.0 <= r.posterior_low and r.posterior_high <= 1.0
    assert r.interval_mass == pytest.approx(0.9)


def test_the_interval_narrows_as_evidence_accumulates(prior):
    widths = [shrink_rate(f"n{n}", int(0.2 * n), n, prior).posterior_width
              for n in (5, 50, 500, 5000)]
    assert widths == sorted(widths, reverse=True), widths


def test_a_too_wide_interval_is_excluded_from_ranking_not_caveated(prior):
    """§4.6: the estimate is withheld, and the reason is recorded.

    Ranking an unreliable estimate with a footnote is how a five-claim provider
    ends up at the top of a reviewer's queue anyway.
    """
    r = shrink_rate("tiny", 1, 2, prior, max_posterior_width=0.10)
    assert r.excluded_from_ranking is True
    assert r.exclusion_reason
    assert "width" in r.exclusion_reason.lower() or "interval" in r.exclusion_reason.lower()


def test_a_reliable_estimate_is_not_excluded(prior):
    r = shrink_rate("solid", 200, 1000, prior, max_posterior_width=0.25)
    assert r.excluded_from_ranking is False


def test_the_interval_mass_is_configurable(prior):
    narrow = shrink_rate("e", 20, 100, prior, interval_mass=0.5)
    wide = shrink_rate("e", 20, 100, prior, interval_mass=0.99)
    assert narrow.posterior_width < wide.posterior_width


# ------------------------------------------------------------- explainability


def test_every_estimate_can_explain_itself_in_words(prior):
    text = shrink_rate("P0001", 1, 5, prior).explain()
    assert text
    assert "5" in text
    assert any(token in text.lower() for token in ("peer", "shrunk", "adjust"))


def test_the_dictionary_form_carries_the_provenance(prior):
    payload = shrink_rate("P0001", 1, 5, prior).to_dict()
    for key in ("observed_rate", "shrunk_rate", "posterior_low", "posterior_high", "peer_mean"):
        assert key in payload


# --------------------------------------------------------- the frame helper


def test_shrink_frame_returns_a_row_per_entity_and_the_prior():
    df = pd.DataFrame({
        "provider_sk": [f"P{i:03d}" for i in range(40)],
        "events": list(range(40)),
        "opportunities": [100] * 40,
    })
    out, fitted = shrink_frame(df, "provider_sk", "events", "opportunities")
    assert len(out) == 40
    assert fitted.n_entities == 40
    assert {"shrunk_rate", "posterior_low", "posterior_high"} <= set(out.columns)


# --------------------------------------------------------- the demonstration


def test_the_demonstration_table_reproduces_the_manuscript_example():
    """§4.6's worked example, rendered for the Statistical Baselines page."""
    table = shrinkage_demonstration(rate=0.2, small_n=5, large_n=5000)
    assert len(table) == 2

    small = table.iloc[0]
    large = table.iloc[1]
    assert small["opportunities"] == 5
    assert large["opportunities"] == 5000
    assert small["observed_rate"] == pytest.approx(large["observed_rate"])
    assert small["shrunk_rate"] < large["shrunk_rate"]
    assert large["shrunk_rate"] == pytest.approx(0.2, abs=0.01)
