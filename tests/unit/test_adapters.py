"""Source adapters, the schema gap, and the honesty rules around both.

Three adapters exist: ``ShafafiyaAdapter`` and ``EClaimLinkAdapter``, both
marked ``NOT CERTIFIED``, and ``GenericIndiaTpaAdapter``, which is the one that
actually runs. AED is the internal currency, reached through an effective-dated
``config/fx.yaml``; every raw record is hashed before parsing.

The tests below are as much about labelling as about mapping, because the
labelling is the honest part. An adapter written against a published schema but
never run against a live regulator feed is *schema-complete*, not *certified*,
and that distinction has to survive into the output rather than living in a
README where nobody reading a signal will see it.
"""

from __future__ import annotations

import datetime as _dt
import json

import pandas as pd
import pytest

from fwa.canonical import ImmutableRawStore
from fwa.canonical.adapters import (
    NOT_CERTIFIED, AdapterError, EClaimLinkAdapter, GenericIndiaTpaAdapter, ShafafiyaAdapter,
    get_adapter,
)
from fwa.canonical.raw_store import sha256_record


# ----------------------------------------------------------- the three adapters


def test_all_three_adapters_are_registered():
    assert get_adapter("generic_india_tpa") is GenericIndiaTpaAdapter
    assert get_adapter("shafafiya") is ShafafiyaAdapter
    assert get_adapter("eclaimlink") is EClaimLinkAdapter


def test_an_unknown_adapter_names_the_ones_that_exist():
    with pytest.raises(KeyError) as exc:
        get_adapter("nonesuch")
    assert "eclaimlink" in str(exc.value)


def test_the_two_uae_adapters_are_marked_not_certified():
    """Schema-complete is not certified, and the distinction is load-bearing."""
    assert ShafafiyaAdapter.certification == NOT_CERTIFIED
    assert EClaimLinkAdapter.certification == NOT_CERTIFIED
    assert "no live regulator feed" in NOT_CERTIFIED


def test_the_working_adapter_does_not_claim_certification_either():
    """The adapter that actually runs is no more certified than the other two.

    It is easy to let the one adapter that is exercised on every run drift into
    sounding authoritative, precisely because it works. What "works" means here
    is that it maps a CSV's columns by name — not that any regulator has
    accepted its output, which is the claim a reader would otherwise be
    entitled to infer.
    """
    certification = GenericIndiaTpaAdapter.certification
    assert "NOT CERTIFIED" in certification.upper()
    assert "certified against a live feed" in certification


def test_the_two_regimes_are_separate_implementations_not_one_parameterised_mapper():
    """§3.5 forbids assuming field-level identity between DoH and DHA.

    The clearest way to re-introduce that assumption would be to make one
    adapter a configuration of the other. They therefore keep their own element
    maps, and the maps genuinely differ — eClaimLink carries an Emirates ID
    construct that Shafafiya does not.
    """
    assert ShafafiyaAdapter.ELEMENT_MAP is not EClaimLinkAdapter.ELEMENT_MAP
    assert ShafafiyaAdapter.ELEMENT_MAP != EClaimLinkAdapter.ELEMENT_MAP
    assert "Claim/EmiratesIDNumber" in EClaimLinkAdapter.ELEMENT_MAP
    assert "Claim/EmiratesIDNumber" not in ShafafiyaAdapter.ELEMENT_MAP


def test_the_shafafiya_map_covers_the_constructs_section_three_five_names():
    """Claim, Encounter, Activity, Observation, PatientShare, Authorization,
    Remittance denial codes and Resubmission type."""
    prefixes = {k.split("/")[0] for k in ShafafiyaAdapter.ELEMENT_MAP}
    assert {"Claim", "Encounter", "Activity", "Observation", "Authorization",
            "Remittance", "Resubmission", "Diagnosis"} <= prefixes
    assert "Claim/PatientShare" in ShafafiyaAdapter.ELEMENT_MAP
    assert "Remittance/DenialCode" in ShafafiyaAdapter.ELEMENT_MAP
    assert "Resubmission/Type" in ShafafiyaAdapter.ELEMENT_MAP


# ----------------------------------------------------- the UAE adapters run


@pytest.fixture()
def shafafiya_submission():
    return {
        "Claim": [{
            "Claim/ID": "SHF-0001", "Claim/MemberID": "M-100", "Claim/PayerID": "PAY-1",
            "Claim/ProviderID": "PRV-9", "Claim/Gross": 5200.0,
            "Claim/PatientShare": 200.0, "Claim/Net": 5000.0,
        }],
        "Encounter": [{
            "Encounter/FacilityID": "FAC-1", "Encounter/Type": "IP",
            "Encounter/Start": "2026-03-01T09:00:00", "Encounter/End": "2026-03-05T11:00:00",
        }],
        "Activity": [{
            "Activity/ID": "ACT-1", "Activity/Code": "99213", "Activity/Quantity": 1,
            "Activity/Net": 5000.0, "Activity/Start": "2026-03-01",
        }],
        "Observation": [{"Observation/Type": "Result", "Observation/Code": "HB",
                         "Observation/Value": "13.1", "Observation/ValueType": "g/dL"}],
    }


def test_the_shafafiya_adapter_maps_a_submission(config, shafafiya_submission):
    dataset = ShafafiyaAdapter(config, ImmutableRawStore()).load(shafafiya_submission)
    header = dataset["claim_header"]
    assert header.iloc[0]["source_claim_id"] == "SHF-0001"
    assert header.iloc[0]["patient_share"] == 200.0
    assert dataset["encounter"].iloc[0]["encounter_type"] == "IP"
    assert dataset["observation"].iloc[0]["observation_code"] == "HB"


def test_a_mapped_table_records_its_source_system_and_certification(config, shafafiya_submission):
    dataset = ShafafiyaAdapter(config, ImmutableRawStore()).load(shafafiya_submission)
    assert dataset.source_system == "SHAFAFIYA_DOH_ABU_DHABI"
    assert dataset.reasons["__certification__"] == NOT_CERTIFIED
    assert (dataset["claim_header"]["source_system"] == "SHAFAFIYA_DOH_ABU_DHABI").all()
    assert NOT_CERTIFIED in dataset.reasons["claim_header"]


def test_a_mapped_row_keeps_the_literal_source_values(config, shafafiya_submission):
    """§3.5: a canonical value must always be traceable to the source value."""
    dataset = ShafafiyaAdapter(config, ImmutableRawStore()).load(shafafiya_submission)
    vocabulary = json.loads(dataset["claim_header"].iloc[0]["source_vocabulary"])
    assert vocabulary["Claim/ID"] == "SHF-0001"


def test_a_construct_absent_from_the_submission_is_marked_not_populated(
    config, shafafiya_submission
):
    dataset = ShafafiyaAdapter(config, ImmutableRawStore()).load(shafafiya_submission)
    assert not dataset.is_populated("remittance")
    assert "remittance" in dataset.reasons["remittance"]


def test_the_eclaimlink_adapter_maps_its_own_constructs(config):
    submission = {
        "Claim": [{
            "Claim/ID": "DHA-0001", "Claim/IDPayer": "PAY-2", "Claim/MemberID": "M-200",
            "Claim/EmiratesIDNumber": "784-1990-1234567-1", "Claim/ProviderID": "PRV-3",
            "Claim/Gross": 3100.0, "Claim/PatientShare": 100.0, "Claim/Net": 3000.0,
        }],
    }
    dataset = EClaimLinkAdapter(config, ImmutableRawStore()).load(submission)
    assert dataset.source_system == "ECLAIMLINK_DHA_DUBAI"
    assert dataset["claim_header"].iloc[0]["source_claim_id"] == "DHA-0001"
    # The Emirates ID lands in the member table as a protected token, not on
    # the claim header where every control would see it.
    assert "protected_id_token" in dataset["member"].columns
    assert "protected_id_token" not in dataset["claim_header"].columns


def test_a_control_cannot_tell_which_regime_a_claim_came_from(config, shafafiya_submission):
    """One interface: the canonical column names are the same either way."""
    shafafiya = ShafafiyaAdapter(config, ImmutableRawStore()).load(shafafiya_submission)
    dha = EClaimLinkAdapter(config, ImmutableRawStore()).load({
        "Claim": [{"Claim/ID": "DHA-1", "Claim/MemberID": "M", "Claim/IDPayer": "P",
                   "Claim/ProviderID": "PRV", "Claim/Gross": 1.0, "Claim/Net": 1.0}],
    })
    shared = {"source_claim_id", "member_sk", "provider_sk", "gross_amount"}
    assert shared <= set(shafafiya["claim_header"].columns)
    assert shared <= set(dha["claim_header"].columns)


# -------------------------------------------------------- the raw-record hash


def test_the_raw_record_is_hashed_before_parsing(config, sample_claims):
    store = ImmutableRawStore()
    GenericIndiaTpaAdapter(config, store).load(sample_claims.head(50))
    assert len(store) == 50


def test_the_hash_is_a_function_of_the_record_not_its_key_order():
    a = sha256_record({"claim_id": "C1", "amount": 100})
    b = sha256_record({"amount": 100, "claim_id": "C1"})
    assert a == b
    assert len(a) == 64


def test_a_different_record_hashes_differently():
    assert sha256_record({"claim_id": "C1"}) != sha256_record({"claim_id": "C2"})


def test_the_raw_store_is_append_only_and_content_addressed(config, sample_claims):
    store = ImmutableRawStore()
    GenericIndiaTpaAdapter(config, store).load(sample_claims.head(20))
    before = len(store)
    GenericIndiaTpaAdapter(config, store).load(sample_claims.head(20))
    assert len(store) == before, "The same records were stored twice under different keys."


# ------------------------------------------------------------ AED conversion


def test_money_is_converted_at_the_service_date_not_with_a_literal(config, sample_claims):
    """Build brief §3.1: effective-dated FX, never a hard-coded rate."""
    dataset = GenericIndiaTpaAdapter(config, ImmutableRawStore()).load(sample_claims.head(200))
    header = dataset["claim_header"]

    assert "gross_amount_aed" in header.columns
    assert (header["gross_amount_aed"] > 0).all()

    # ``gross_amount`` is retained in the SOURCE currency beside the converted
    # figure, so a reviewer can reconcile an AED number against the original.
    row = header.iloc[0]
    expected = config.fx.convert(
        float(row["gross_amount"]), "INR", pd.Timestamp(row["service_date"]).date()
    )
    assert row["gross_amount_aed"] == pytest.approx(expected, rel=1e-6)
    assert row["gross_amount_aed"] != pytest.approx(float(row["gross_amount"]))


def test_the_rate_changes_between_years(config):
    """Two claims a year apart must not be converted at the same rate.

    If they were, the effective dating would be decorative — and comparing a
    2024 claim with a 2025 one on amount would silently compare two different
    currencies' worth of value.
    """
    early = config.fx.rate("INR", _dt.date(2024, 6, 1))
    late = config.fx.rate("INR", _dt.date(2025, 6, 1))
    assert early != late


def test_every_fx_rate_states_its_source_and_owner(config):
    """An exchange rate with no provenance is a magic number in disguise.

    The sources here also say what they are NOT — an indicative period average
    used by a dissertation artefact, not a treasury-grade feed — because an AED
    figure derived from a placeholder rate should not be read as a settlement
    amount.
    """
    rates = config.fx._rates
    assert rates
    for rate in rates:
        assert rate.source
        assert rate.owner
        assert rate.valid_from <= rate.valid_to
    assert any("NOT a treasury-grade rate feed" in r.source for r in rates)


def test_converting_without_a_date_is_not_possible(config):
    import inspect

    params = inspect.signature(config.fx.convert).parameters
    assert "as_of" in params
    assert params["as_of"].default is inspect.Parameter.empty


# ---------------------------------------------------- labels stay held out


def test_the_adapter_splits_the_labels_off_the_canonical_model(config, sample_claims):
    adapter = GenericIndiaTpaAdapter(config, ImmutableRawStore())
    dataset = adapter.load(sample_claims)
    labels = adapter.held_out_labels(sample_claims)

    for column in GenericIndiaTpaAdapter.LABEL_COLUMNS:
        assert column in labels.columns
        for frame in dataset.tables.values():
            assert column not in frame.columns


def test_the_held_out_labels_keep_the_join_key(config, sample_claims):
    """Held out, not thrown away: evaluation still has to join them back."""
    adapter = GenericIndiaTpaAdapter(config, ImmutableRawStore())
    labels = adapter.held_out_labels(sample_claims)
    assert "claim_sk" in labels.columns
    assert labels["claim_sk"].is_unique


# ------------------------------------------------------------- missingness


def test_absent_fields_are_recorded_as_missing_not_imputed(config, sample_claims):
    """§4.5: never silently impute. The absence is itself carried forward."""
    dataset = GenericIndiaTpaAdapter(config, ImmutableRawStore()).load(sample_claims.head(50))
    missingness = dataset["claim_header"].iloc[0]["missingness"]
    payload = json.loads(missingness)
    assert payload, "No missingness recorded for a dataset that is missing most of the model."
    for field, reason in payload.items():
        assert "NOT_IN_SOURCE" in reason or reason


# --------------------------------------------- loading a file with gaps in it


def test_a_source_missing_an_optional_column_still_maps(config, raw_claims):
    """A real extract will not have every column the shipped file has.

    The Data page exists so that an arbitrary claim file can be loaded. That
    makes "this file has no agent_id" an ordinary fact about a source rather
    than a programming error, and it has to be handled the way every other gap
    in this system is handled: the field is present and empty, the absence is
    recorded, and the controls that needed it are classified rather than run
    against a substitute.
    """
    without_agent = raw_claims.drop(columns=["agent_id"])
    adapter = GenericIndiaTpaAdapter(config, ImmutableRawStore())

    dataset = adapter.load(without_agent)

    assert len(dataset["claim_header"]) == len(without_agent)
    assert "agent_id" in dataset["claim_header"].columns
    assert dataset["claim_header"]["agent_id"].isna().all()
    recorded = dataset.reasons["__absent_source_columns__"]
    assert "agent_id" in recorded


def test_a_source_missing_an_identifier_is_refused_with_a_usable_message(config):
    """Some gaps are not gaps. Say which, and say what was found instead.

    Without a claim, member or provider identifier there is nothing to map a
    claim *to*, so mapping it into an empty shell would produce a run whose
    every number is meaningless. The refusal names the missing columns and the
    columns that are present, which is the difference between a message a user
    can act on and one they can only forward.
    """
    adapter = GenericIndiaTpaAdapter(config, ImmutableRawStore())
    with pytest.raises(AdapterError) as exc:
        adapter.load(pd.DataFrame({"something_else": [1, 2, 3]}))
    message = str(exc.value)
    assert "claim_id" in message and "patient_id" in message and "hospital_id" in message
    assert "something_else" in message           # what WAS there, not only what wasn't


def test_labels_are_optional_because_real_data_has_none(config, raw_claims):
    """A file with no ground-truth columns is the normal case, not a failure."""
    df = raw_claims.drop(columns=list(GenericIndiaTpaAdapter.LABEL_COLUMNS))
    adapter = GenericIndiaTpaAdapter(config, ImmutableRawStore())

    held_out = adapter.held_out_labels(df)

    assert held_out.empty
    assert set(GenericIndiaTpaAdapter.LABEL_COLUMNS) <= set(held_out.columns)


def test_a_feature_over_an_absent_entity_column_is_missing_not_zero(config, raw_claims):
    """An all-empty grouping column must not become a column of zeros.

    ``_prior_expanding`` groups by an entity and counts that entity's earlier
    claims. With no entity column at all, pandas groups every row to nothing
    and has no arrays left to concatenate — which surfaced as a bare
    ``ValueError`` from three frames inside pandas. The fix has to produce
    MISSING rather than 0, because zero prior claims is a statement about an
    agent that exists, and this file has no agents.
    """
    from fwa.features.store import _prior_expanding, _prior_window_count

    work = raw_claims.head(200).copy()
    work["agent_id"] = pd.NA
    work["service_date"] = pd.to_datetime(work["date_of_claim"], dayfirst=True, errors="coerce")
    work["gross_amount_aed"] = work["claim_amount_requested_inr"].astype(float)

    counts = _prior_expanding(work, "agent_id", "gross_amount_aed", "count")
    window = _prior_window_count(work, "agent_id", "service_date", 30)

    assert counts.isna().all(), "an absent entity must yield missing, never zero"
    assert window.isna().all()
    assert len(counts) == len(work) and len(window) == len(work)
