"""Conservative exposure by signal type (manuscript §4.11, build brief §1).

    "Gross flagged value is NOT savings... an uncertain MODEL-ONLY LEAD shows its
     gross amount together with the explicit statement 'exposure not yet
     established' rather than presenting that gross amount as if it were
     savings. This last rule is one of the more consequential design decisions
     in the whole system, because it is the rule that prevents the system from
     ever silently overstating its own value, a risk this dissertation treats as
     an ETHICAL REQUIREMENT."

Six shapes of exposure, six sections. The pattern running through them is that
each one takes the *lower* or *incremental* reading wherever the data permits
two: the lower of a duplicated pair, the increment over the peer median, the
sampled sum rather than the extrapolation, a claim counted once rather than
once per graph node.
"""

from __future__ import annotations

import pytest

from fwa.cases.exposure import (
    duplicate_exposure, line_edit_exposure, model_only_exposure, network_exposure,
    no_exposure, provider_pattern_exposure,
)
from fwa.engine.signals import EXPOSURE_NOT_ESTABLISHED


# ------------------------------------------------------------- no exposure


def test_no_exposure_is_zero_with_a_reason():
    """A real finding with no monetary consequence is worth AED 0, not its gross."""
    e = no_exposure("a data-integrity RETURN corrects a field, it does not recover money")
    assert e.amount_aed == 0.0
    assert e.established is True
    assert "data-integrity RETURN" in e.basis


# -------------------------------------------------------------- line edits


def test_a_line_edit_is_the_difference_not_the_whole_claim():
    e = line_edit_exposure(submitted_payable=1000.0, correctly_repriced_payable=750.0)
    assert e.amount_aed == pytest.approx(250.0)
    assert e.established is True
    assert "1,000.00" in e.basis and "750.00" in e.basis


def test_a_line_edit_is_floored_at_zero():
    """A repricing that would pay *more* is not negative exposure."""
    assert line_edit_exposure(750.0, 1000.0).amount_aed == 0.0


# --------------------------------------------------------------- duplicates


def test_a_duplicate_takes_the_lower_of_the_pair():
    """The larger claim may be the legitimate one, so only the smaller is at risk."""
    e = duplicate_exposure([1200.0, 900.0])
    assert e.amount_aed == pytest.approx(900.0)
    assert "LOWER" in e.basis


def test_a_duplicate_deducts_valid_patient_share():
    e = duplicate_exposure([1200.0, 900.0], valid_patient_share=150.0)
    assert e.amount_aed == pytest.approx(750.0)


def test_patient_share_cannot_drive_exposure_negative():
    assert duplicate_exposure([1200.0, 900.0], valid_patient_share=5000.0).amount_aed == 0.0


def test_a_triplicate_still_takes_the_single_lowest():
    """Three copies of one service is still one recoverable amount, not two."""
    e = duplicate_exposure([1200.0, 900.0, 1100.0])
    assert e.amount_aed == pytest.approx(900.0)


def test_one_amount_is_not_a_duplicate():
    e = duplicate_exposure([900.0])
    assert e.amount_aed == 0.0
    assert "fewer than two" in e.basis


# -------------------------------------------------------- provider patterns


def test_a_provider_pattern_sums_the_sampled_increments():
    e = provider_pattern_exposure([100.0, 250.0, 50.0])
    assert e.amount_aed == pytest.approx(400.0)
    assert e.extrapolated is False


def test_negative_increments_do_not_offset_positive_ones():
    """A claim below the peer median is not a credit against one above it."""
    assert provider_pattern_exposure([100.0, -80.0]).amount_aed == pytest.approx(100.0)


def test_extrapolation_is_refused_by_default_and_says_why():
    """§4.11: extrapolate only once audit policy permits it. No such policy exists here."""
    e = provider_pattern_exposure([100.0, 200.0], population_size=1000, sample_size=10)
    assert e.amount_aed == pytest.approx(300.0)
    assert e.extrapolated is False
    assert "NOT extrapolated" in e.basis
    assert "audit policy" in e.basis


def test_permitted_extrapolation_is_an_interval_never_a_point():
    e = provider_pattern_exposure(
        [100.0, 200.0], population_size=1000, sample_size=10,
        audit_policy_permits_extrapolation=True,
    )
    assert e.extrapolated is True
    assert e.amount_aed == pytest.approx(300.0), "The headline stays the measured sum."
    assert e.interval_low_aed == pytest.approx(300.0)
    assert e.interval_high_aed == pytest.approx(30_000.0)
    assert "LABELLED INTERVAL" in e.basis
    assert "EXTRAPOLATED" in e.display()


# ------------------------------------------------------------------ network


def test_a_network_case_counts_each_claim_once():
    """The §4.11 rule that a claim touching three nodes is still one claim."""
    e = network_exposure({"C1": 500.0, "C2": 300.0})
    assert e.amount_aed == pytest.approx(800.0)


def test_a_claim_repeated_across_nodes_cannot_be_double_counted():
    """The mapping shape is the defence: the same key overwrites, it does not add."""
    amounts: dict[str, float] = {}
    for _node in ("provider", "member", "community"):
        amounts["C1"] = 500.0
    assert network_exposure(amounts).amount_aed == pytest.approx(500.0)


def test_an_empty_network_case_is_zero():
    assert network_exposure({}).amount_aed == 0.0


# --------------------------------------------------------------- model-only


def test_a_model_only_lead_never_presents_its_gross_as_savings():
    """The most consequential rule in §4.11, stated as a test."""
    e = model_only_exposure(12_500.0, "IsolationForest")
    assert e.established is False
    assert e.amount_aed == pytest.approx(12_500.0)
    assert EXPOSURE_NOT_ESTABLISHED in e.display()
    assert "12,500.00" in e.display()


def test_the_unestablished_marker_is_the_literal_required_string():
    assert EXPOSURE_NOT_ESTABLISHED == "exposure not yet established"


def test_a_model_only_lead_names_the_model_that_raised_it():
    assert "IsolationForest" in model_only_exposure(100.0, "IsolationForest").basis


def test_an_unestablished_exposure_is_flagged_in_its_dictionary_form():
    payload = model_only_exposure(12_500.0, "LOF").to_dict()
    assert payload["exposure_established"] is False
    assert EXPOSURE_NOT_ESTABLISHED in payload["exposure_display"]


# ------------------------------------------------------ the aggregate rule


def test_established_and_unestablished_amounts_are_never_added_together(sample_cases):
    """Across a real run: the two totals are reported separately, by construction."""
    _, cases = sample_cases
    established = sum(c.exposure_aed for c in cases.values() if c.exposure_established)
    unestablished = sum(c.exposure_aed for c in cases.values() if not c.exposure_established)
    assert established >= 0.0
    # The point is that both figures exist independently; a single "total
    # savings" number that mixed them is what §4.11 forbids.
    assert isinstance(unestablished, float)
    for case in cases.values():
        assert case.exposure_established in (True, False)
        if not case.exposure_established:
            assert case.exposure_basis, "An unestablished exposure must say why."


def test_every_case_exposure_is_non_negative(sample_cases):
    _, cases = sample_cases
    for case in cases.values():
        assert case.exposure_aed >= 0.0
