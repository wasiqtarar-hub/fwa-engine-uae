"""Empirical-Bayes shrinkage (manuscript §4.6).

    "a provider with five claims and one adverse event has an observed rate of
     20%, which is statistically almost meaningless, while a provider with five
     thousand claims and the same 20% rate is a genuine outlier. The
     specification's recommended empirical-Bayes shrinkage estimator addresses
     this directly:

         shrunk_rate_i = (events_i + alpha_peer) / (opportunities_i + alpha_peer + beta_peer)

     ... and requires that every shrunk rate be reported WITH ITS POSTERIOR
     INTERVAL, with entities whose intervals are too wide EXCLUDED FROM RANKING
     rather than flagged on an artificially precise point estimate."
                                                            — manuscript §4.6

Three properties of this module are manuscript requirements, not choices:

1. The estimator is exactly the formula above. It is implemented literally so a
   reader can check it against the manuscript line by line.
2. Every result carries a posterior interval (Beta(events + α, opportunities −
   events + β) quantiles).
3. An entity whose interval is wider than ``cfg.max_posterior_width`` is
   ``excluded_from_ranking=True``. That flag is honoured by every control and by
   the review queue; it is the difference between "this provider is unusual" and
   "we have no idea about this provider".

α and β are fitted from the peer population by the method of moments, with a
documented fallback when the moment equations are degenerate (which happens for
peer groups where almost every entity has the same rate).
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Sequence

import numpy as np
import pandas as pd
from scipy import stats as _st

__all__ = ["BetaPrior", "fit_beta_prior", "ShrunkRate", "shrink_rate", "shrink_frame"]


@dataclass(frozen=True)
class BetaPrior:
    """The peer population's fitted Beta(α, β) prior."""

    alpha: float
    beta: float
    peer_mean: float
    peer_var: float
    n_entities: int
    method: str        # "method_of_moments" | "weak_prior_fallback" | "jeffreys_fallback"
    note: str = ""

    @property
    def prior_strength(self) -> float:
        """α + β — the number of pseudo-observations the prior is worth.

        This is the quantity that decides how hard a five-claim provider is
        pulled toward the peer mean, so it is surfaced in the UI beside the
        shrinkage demonstration.
        """
        return self.alpha + self.beta

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["prior_strength"] = self.prior_strength
        return d


def fit_beta_prior(
    events: Sequence[float],
    opportunities: Sequence[float],
    *,
    min_entities: int = 3,
) -> BetaPrior:
    """Fit Beta(α, β) to the peer population by the method of moments.

    Given per-entity rates :math:`p_i = events_i / opportunities_i` with mean
    :math:`m` and variance :math:`v`, the moment equations give

    .. math::
        \\alpha = m\\left(\\frac{m(1-m)}{v} - 1\\right), \\qquad
        \\beta  = (1-m)\\left(\\frac{m(1-m)}{v} - 1\\right)

    The bracketed term is the prior strength. It is only valid when
    :math:`0 < v < m(1-m)`; outside that range the peer population is either
    degenerate (every entity identical, so ``v = 0``) or over-dispersed beyond
    what a Beta can represent. Both cases fall back to a deliberately WEAK
    prior rather than to an arbitrary strong one, because a strong prior fitted
    to a degenerate population would shrink every entity to the same value and
    silently destroy the signal.
    """
    e = np.asarray(events, dtype=float)
    o = np.asarray(opportunities, dtype=float)
    mask = (o > 0) & np.isfinite(e) & np.isfinite(o)
    e, o = e[mask], o[mask]
    n = int(e.size)
    if n < min_entities:
        return BetaPrior(
            alpha=0.5, beta=0.5, peer_mean=float(e.sum() / o.sum()) if o.sum() > 0 else 0.0,
            peer_var=0.0, n_entities=n, method="jeffreys_fallback",
            note=f"Only {n} entities in the peer population (minimum {min_entities}); "
                 "Jeffreys prior Beta(0.5, 0.5) used so that shrinkage is minimal and the "
                 "uncertainty shows up in the posterior interval instead.",
        )
    rates = e / o
    m = float(np.average(rates, weights=o))          # opportunity-weighted peer mean
    v = float(np.var(rates, ddof=1)) if n > 1 else 0.0
    ceiling = m * (1 - m)
    if v <= 0 or ceiling <= 0 or v >= ceiling:
        strength = 20.0
        return BetaPrior(
            alpha=max(m * strength, 1e-6), beta=max((1 - m) * strength, 1e-6),
            peer_mean=m, peer_var=v, n_entities=n, method="weak_prior_fallback",
            note=("Method-of-moments is undefined here: the between-entity variance is "
                  f"{v:.6f} against a Beta ceiling of {ceiling:.6f}. A deliberately weak "
                  "prior (strength 20) centred on the peer mean is used instead, so that "
                  "low-volume entities are pulled toward the peer mean without the prior "
                  "dominating a genuinely high-volume entity."),
        )
    strength = ceiling / v - 1.0
    strength = float(np.clip(strength, 1e-3, 1e6))
    return BetaPrior(
        alpha=m * strength, beta=(1 - m) * strength, peer_mean=m, peer_var=v,
        n_entities=n, method="method_of_moments",
    )


@dataclass(frozen=True)
class ShrunkRate:
    """One entity's shrunk rate with its posterior interval."""

    entity: str
    events: float
    opportunities: float
    observed_rate: float | None
    shrunk_rate: float
    posterior_low: float
    posterior_high: float
    posterior_width: float
    peer_mean: float
    prior_alpha: float
    prior_beta: float
    prior_strength: float
    interval_mass: float
    excluded_from_ranking: bool
    exclusion_reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def explain(self) -> str:
        """Plain-language sentence for the reviewer (§14.1: lead with the sentence)."""
        if self.observed_rate is None:
            return f"{self.entity} has no opportunities, so no rate can be estimated."
        pull = self.shrunk_rate - self.observed_rate
        direction = "up toward" if pull > 0 else "down toward"
        return (
            f"{self.entity} shows {self.events:.0f} event(s) in {self.opportunities:.0f} "
            f"opportunities — an observed rate of {self.observed_rate:.1%}. Because that rests "
            f"on {self.opportunities:.0f} observation(s), the estimate is pulled {direction} the "
            f"peer average of {self.peer_mean:.1%}, giving {self.shrunk_rate:.1%} "
            f"({self.interval_mass:.0%} interval {self.posterior_low:.1%}–{self.posterior_high:.1%})."
            + (f" NOT RANKED: {self.exclusion_reason}" if self.excluded_from_ranking else "")
        )


def shrink_rate(
    entity: str,
    events: float,
    opportunities: float,
    prior: BetaPrior,
    *,
    interval_mass: float = 0.90,
    max_posterior_width: float = 0.25,
) -> ShrunkRate:
    """Apply the §4.6 estimator to one entity and return it with its interval."""
    events = float(events)
    opportunities = float(opportunities)
    a_post = events + prior.alpha
    b_post = max(opportunities - events, 0.0) + prior.beta
    shrunk = a_post / (a_post + b_post)          # == (e + α) / (o + α + β)
    tail = (1.0 - interval_mass) / 2.0
    lo = float(_st.beta.ppf(tail, a_post, b_post))
    hi = float(_st.beta.ppf(1.0 - tail, a_post, b_post))
    width = hi - lo
    if opportunities <= 0:
        # No denominator at all. The posterior is exactly the prior, so the
        # interval may well be narrow enough to pass the width test — which
        # would rank an entity on the peer mean and nothing else. §4.6's
        # exclusion is about *evidence*, and zero opportunities is the complete
        # absence of it, so it is excluded regardless of the interval.
        excluded = True
        reason = (
            "no opportunities in the scored period — the posterior here is the peer prior "
            "and carries no evidence about this entity"
        )
    else:
        excluded = width > max_posterior_width
        reason = (
            f"posterior interval width {width:.3f} exceeds cfg.max_posterior_width "
            f"{max_posterior_width:.3f} — too little evidence to rank this entity"
            if excluded else ""
        )
    return ShrunkRate(
        entity=entity,
        events=events,
        opportunities=opportunities,
        observed_rate=(events / opportunities) if opportunities > 0 else None,
        shrunk_rate=float(shrunk),
        posterior_low=lo,
        posterior_high=hi,
        posterior_width=float(width),
        peer_mean=prior.peer_mean,
        prior_alpha=prior.alpha,
        prior_beta=prior.beta,
        prior_strength=prior.prior_strength,
        interval_mass=interval_mass,
        excluded_from_ranking=excluded,
        exclusion_reason=reason,
    )


def shrink_frame(
    df: pd.DataFrame,
    entity_column: str,
    events_column: str,
    opportunities_column: str,
    *,
    interval_mass: float = 0.90,
    max_posterior_width: float = 0.25,
) -> tuple[pd.DataFrame, BetaPrior]:
    """Fit the prior on the frame's population and shrink every row."""
    prior = fit_beta_prior(df[events_column], df[opportunities_column])
    rows = [
        shrink_rate(
            str(r[entity_column]), r[events_column], r[opportunities_column], prior,
            interval_mass=interval_mass, max_posterior_width=max_posterior_width,
        ).to_dict()
        for _, r in df.iterrows()
    ]
    return pd.DataFrame(rows), prior


def shrinkage_demonstration(
    prior: BetaPrior | None = None,
    *,
    rate: float = 0.20,
    small_n: int = 5,
    large_n: int = 5000,
    interval_mass: float = 0.90,
    max_posterior_width: float = 0.25,
) -> pd.DataFrame:
    """The §4.6 worked example, rendered for the Provider Analytics page.

    "a provider with five claims and one adverse event has an observed rate of
     20%, which is statistically almost meaningless, while a provider with five
     thousand claims and the same 20% rate is a genuine outlier"

    The build brief (§5.2) asks for this to be demonstrated explicitly in the
    UI. Both providers have the *same observed rate*; the table shows how far
    each shrinks and how wide each interval is, which is the whole argument for
    the estimator in one picture.
    """
    if prior is None:
        # A plausible peer population centred below the demonstration rate, so
        # the pull is visible.
        rng = np.random.default_rng(7)
        opportunities = rng.integers(20, 400, size=200).astype(float)
        true_rates = rng.beta(4, 46, size=200)          # peer mean ≈ 8%
        events = rng.binomial(opportunities.astype(int), true_rates).astype(float)
        prior = fit_beta_prior(events, opportunities)
    rows = []
    for n in (small_n, large_n):
        sr = shrink_rate(
            f"{n}-claim provider", round(rate * n), n, prior,
            interval_mass=interval_mass, max_posterior_width=max_posterior_width,
        )
        d = sr.to_dict()
        d["explanation"] = sr.explain()
        rows.append(d)
    return pd.DataFrame(rows)
