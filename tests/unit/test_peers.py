"""The six-level peer hierarchy, back-off and ``peer_level_used`` (manuscript §4.6).

    "A granular peer group that is too sparse to support a stable estimate backs
     off to its parent level, and the ``peer_level_used`` is ALWAYS RECORDED."

Three properties matter and each is easy to get wrong in a way that looks fine:
a group that is too small must back off rather than produce a confident number
from four rows; a level the dataset does not populate must be *skipped and
named*, not silently approximated from something adjacent; and the resulting
group must be reproducible later from its stored ``baseline_version``.
"""

from __future__ import annotations

import pandas as pd
import pytest

from fwa.statistical.peers import INSUFFICIENT_PEER_EVIDENCE, PeerService


@pytest.fixture(scope="module")
def service(config, claims):
    return PeerService(config, claims)


# ------------------------------------------------------------- the hierarchy


def test_the_configuration_declares_six_levels(config):
    assert len(config.peer_groups.levels) == 6
    for level in config.peer_groups.levels:
        assert level.canonical_name
        assert level.availability
        # A populated level must say what it maps to in this dataset; an
        # unpopulated one must instead name the canonical fields that would
        # populate it, so the gap is actionable rather than merely noted.
        if level.populated:
            assert level.dataset_mapping
        else:
            assert level.required_canonical_fields


def test_the_unpopulated_level_is_named_not_quietly_dropped(config, service):
    """Geography is absent from claims.csv. Saying so is the requirement."""
    unpopulated = [l for l in config.peer_groups.levels if not l.populated]
    assert unpopulated, "Every level claims to be populated; that would be a false claim."
    assert service.skipped_levels
    for level in unpopulated:
        assert level.canonical_name in service.skipped_levels


def test_a_resolved_group_reports_the_levels_it_skipped(service, claims):
    group = service.resolve(claims.iloc[0])
    assert group.skipped_levels == service.skipped_levels


def test_level_availability_is_reportable_for_the_ui(service):
    availability = service.level_availability()
    assert len(availability) == 6
    assert {"level", "canonical_name", "dataset_mapping", "availability"} <= set(
        availability.columns
    )
    assert (availability["availability"] == "NOT_POPULATED").any()


# ----------------------------------------------------------------- back-off


def test_every_row_resolves_to_a_group_that_names_its_level(service, claims):
    for _, row in claims.head(200).iterrows():
        group = service.resolve(row)
        assert group.level_name, "A resolved group with no level_name is unauditable."


def test_a_sufficient_group_meets_the_minimum(service, claims, config):
    minimum = int(config.get("min_peer_group_n"))
    for _, row in claims.head(200).iterrows():
        group = service.resolve(row)
        if group.sufficient:
            assert group.n_rows >= minimum


def test_a_narrow_population_backs_off_rather_than_using_four_rows(config):
    """The failure this exists to prevent: a confident z-score from a tiny group."""
    frame = pd.DataFrame({
        "provider_sk": [f"P{i}" for i in range(40)],
        "diagnosis_primary": ["A00"] * 2 + ["B00"] * 38,
        "gross_amount_aed": [1000.0] * 40,
        "payer_id": ["PAY1"] * 40,
        "tpa": ["T"] * 40,
    })
    service = PeerService(config, frame)
    rare = service.resolve(frame.iloc[0])   # diagnosis A00 has two rows
    common = service.resolve(frame.iloc[5])

    assert rare.n_rows >= int(config.get("min_peer_group_n")) or not rare.sufficient
    if rare.sufficient:
        assert rare.backoff_steps >= common.backoff_steps


def test_a_population_too_small_for_any_level_is_insufficient_not_anomalous(config):
    """A distinct outcome. "No comparison possible" is not "nothing found"."""
    tiny = pd.DataFrame({
        "provider_sk": ["P1", "P2"],
        "diagnosis_primary": ["A00", "B00"],
        "gross_amount_aed": [1000.0, 2000.0],
        "payer_id": ["PAY1", "PAY1"],
        "tpa": ["T", "T"],
    })
    group = PeerService(config, tiny).resolve(tiny.iloc[0])
    assert not group.sufficient
    assert group.level_name == INSUFFICIENT_PEER_EVIDENCE
    assert group.n_rows == 0
    assert "No peer group large enough" in group.describe()


def test_values_from_an_insufficient_group_are_empty_not_the_whole_table(config):
    """The dangerous failure mode: falling back to the global population unannounced."""
    tiny = pd.DataFrame({
        "provider_sk": ["P1", "P2"],
        "diagnosis_primary": ["A00", "B00"],
        "gross_amount_aed": [1000.0, 2000.0],
        "payer_id": ["PAY1", "PAY1"],
        "tpa": ["T", "T"],
    })
    service = PeerService(config, tiny)
    group = service.resolve(tiny.iloc[0])
    assert len(service.values(group, "gross_amount_aed")) == 0


# ----------------------------------------------------------- reproducibility


def test_every_sufficient_group_carries_a_baseline_version(service, claims):
    for _, row in claims.head(100).iterrows():
        group = service.resolve(row)
        if group.sufficient:
            assert group.baseline_version
            assert len(group.baseline_version) >= 8


def test_the_same_data_reproduces_the_same_baseline_version(config, claims):
    """§6.3: a peer result must be reproducible from its stored baseline version."""
    a = PeerService(config, claims).resolve(claims.iloc[0])
    b = PeerService(config, claims).resolve(claims.iloc[0])
    assert a.baseline_version == b.baseline_version
    assert a.level_name == b.level_name
    assert a.n_rows == b.n_rows


def test_different_membership_produces_a_different_baseline_version(config, claims):
    """The version has to change when the comparison group changes, or it proves nothing."""
    full = PeerService(config, claims).resolve(claims.iloc[0])
    half = PeerService(config, claims.head(len(claims) // 2)).resolve(claims.iloc[0])
    if full.sufficient and half.sufficient:
        assert full.baseline_version != half.baseline_version


def test_resolution_is_cached_but_not_confused_between_rows(service, claims):
    """Caching must key on the row's peer attributes, not on call order."""
    first = service.resolve(claims.iloc[0])
    again = service.resolve(claims.iloc[0])
    assert first.baseline_version == again.baseline_version

    different = claims[claims["diagnosis_primary"] != claims.iloc[0]["diagnosis_primary"]]
    assert not different.empty, "The sample has only one diagnosis; the cache is untested."
    other = service.resolve(different.iloc[0])
    if first.sufficient and other.sufficient and first.levels_used:
        assert other.key != first.key, (
            "Two rows with different diagnoses resolved to the same peer key."
        )


# ------------------------------------------------------------------ helpers


def test_values_returns_the_groups_own_rows(service, claims):
    group = service.resolve(claims.iloc[0])
    if group.sufficient:
        values = service.values(group, "gross_amount_aed")
        assert len(values) == group.n_rows


def test_the_group_describes_itself_for_the_evidence_panel(service, claims):
    group = service.resolve(claims.iloc[0])
    description = group.describe()
    assert description
    payload = group.to_dict()
    assert payload["peer_level_used"] == group.level_name
    assert "peer_baseline_version" in payload


def test_an_aggregate_is_taken_over_the_groups_rows(service, claims):
    group = service.resolve(claims.iloc[0])
    if not group.sufficient:
        pytest.skip("No sufficient peer group in this sample.")
    mean = service.entity_aggregates(group, "gross_amount_aed", "mean")
    median = service.entity_aggregates(group, "gross_amount_aed", "median")
    assert mean is not None and median is not None
    values = service.values(group, "gross_amount_aed").astype(float)
    assert mean == pytest.approx(values.mean())
    assert median == pytest.approx(values.median())
