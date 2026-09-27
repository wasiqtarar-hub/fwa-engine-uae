"""Evidence capping (manuscript §3.8, build brief §1).

    "**Evidence capping is mandatory**: ``N`` signals derived from the same
     underlying fact contribute ``max(evidence_strength)``, never the sum. Three
     rules that all fire because one amount is high is ONE piece of evidence."

Two properties have to hold, and they pull in opposite directions, which is why
both are tested here:

* same fact → **no** accumulation, however many controls fired on it;
* distinct facts → accumulation, but *saturating*, so a pile of weak
  independent facts cannot be added up into certainty.

The third test is the one that matters most in practice: PAY-06-R03 and
CLN-01-R02 both key on a claim's amount and deliberately share a ``fact_key``,
so the capping is exercised by the shipped catalogue rather than only by a
fixture built to exercise it.
"""

from __future__ import annotations

import pytest

from fwa.engine.controls import CONTROL_IMPLEMENTATIONS
from fwa.engine.signals import cap_evidence

pytestmark = pytest.mark.governance


class _FakeSignal:
    """The two fields :func:`cap_evidence` reads, and nothing else."""

    def __init__(self, fact_key: str, strength: float, signal_id: str = "") -> None:
        self.fact_key = fact_key
        self.evidence_strength = strength
        self.signal_id = signal_id or f"sig-{fact_key}-{strength}"


# -------------------------------------------------------- the same-fact rule


def test_same_fact_contributes_the_maximum_not_the_sum():
    signals = [_FakeSignal("amount:C1", s) for s in (0.4, 0.5, 0.6)]
    assert cap_evidence(signals) == pytest.approx(0.6)
    assert cap_evidence(signals) < 0.4 + 0.5 + 0.6


@pytest.mark.parametrize("copies", [1, 2, 5, 20, 100])
def test_repeating_one_fact_never_increases_evidence(copies):
    """The headline claim: N rules on one fact is ONE piece of evidence."""
    one = cap_evidence([_FakeSignal("amount:C1", 0.7)])
    many = cap_evidence([_FakeSignal("amount:C1", 0.7) for _ in range(copies)])
    assert many == pytest.approx(one)


def test_same_fact_with_differing_strengths_takes_the_strongest():
    strengths = [0.1, 0.9, 0.3, 0.85]
    assert cap_evidence(
        [_FakeSignal("amount:C1", s) for s in strengths]
    ) == pytest.approx(max(strengths))


# --------------------------------------------------- the distinct-fact rule


def test_distinct_facts_accumulate():
    one = cap_evidence([_FakeSignal("amount:C1", 0.5)])
    two = cap_evidence([_FakeSignal("amount:C1", 0.5), _FakeSignal("coding:C1", 0.5)])
    assert two > one


def test_distinct_facts_saturate_rather_than_sum():
    """Ten weak facts must not manufacture certainty."""
    weak = [_FakeSignal(f"fact:{i}", 0.2) for i in range(10)]
    value = cap_evidence(weak)
    assert value < 1.0
    assert value < sum(s.evidence_strength for s in weak)


def test_result_is_always_a_probability():
    for strengths in ([1.0, 1.0, 1.0], [0.0], [0.99] * 50, []):
        value = cap_evidence([_FakeSignal(f"f{i}", s) for i, s in enumerate(strengths)])
        assert 0.0 <= value <= 1.0


def test_no_signals_is_no_evidence():
    assert cap_evidence([]) == 0.0


def test_a_certain_fact_saturates_at_one():
    assert cap_evidence([_FakeSignal("f", 1.0), _FakeSignal("g", 0.5)]) == pytest.approx(1.0)


def test_out_of_range_strengths_are_clamped_not_trusted():
    """A control that mis-declares strength cannot push evidence past certainty."""
    assert cap_evidence([_FakeSignal("f", 5.0)]) == pytest.approx(1.0)
    assert cap_evidence([_FakeSignal("f", -3.0)]) == pytest.approx(0.0)


# --------------------------------------------- the shipped catalogue does it


def test_two_shipped_controls_share_a_fact_key_on_the_same_claim(context, registry):
    """PAY-06-R03 and CLN-01-R02 both key on the claim amount, by design.

    Both are amount-outlier controls reached by different routes (a robust peer
    residual and an expected-level residual). If they were allowed to contribute
    independently, a single expensive claim would look like two pieces of
    evidence. They therefore emit the same ``fact_key``, and this test asserts
    that the overlap actually occurs in a real run — a claim on which both fired.
    """
    peer = CONTROL_IMPLEMENTATIONS[registry.get("PAY-06-R03").implementation](
        context, registry.get("PAY-06-R03"))
    residual = CONTROL_IMPLEMENTATIONS[registry.get("CLN-01-R02").implementation](
        context, registry.get("CLN-01-R02"))
    shared = {s.fact_key for s in peer} & {s.fact_key for s in residual}
    assert shared, (
        "PAY-06-R03 and CLN-01-R02 never fired on the same claim in the sample, so the "
        "catalogue's evidence-capping demonstration is not being exercised."
    )

    fact = sorted(shared)[0]
    pair = [s for s in list(peer) + list(residual) if s.fact_key == fact]
    assert len(pair) >= 2
    assert cap_evidence(pair) == pytest.approx(max(s.evidence_strength for s in pair))


def test_case_evidence_strength_is_capped_not_summed(context, registry):
    """End to end: the number the priority formula consumes is the capped one."""
    control = registry.get("PAY-06-R03")
    signals = CONTROL_IMPLEMENTATIONS[control.implementation](context, control)
    assert signals, "PAY-06-R03 produced no signals to cap."

    one_claim = [s for s in signals if s.fact_key == signals[0].fact_key]
    duplicated = one_claim * 4
    assert cap_evidence(duplicated) == pytest.approx(cap_evidence(one_claim))
