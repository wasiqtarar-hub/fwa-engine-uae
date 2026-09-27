"""UaeMultiTableAdapter: folder and zip load identically, derived columns are sane, labels stay out."""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT, ROOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from fwa.canonical import (  # noqa: E402
    AdapterError, GenericIndiaTpaAdapter, ImmutableRawStore, UaeMultiTableAdapter, get_adapter,
)
from fwa.canonical.model import CANONICAL_TABLES, PopulationStatus  # noqa: E402
from fwa.canonical.supplementary import SUPPLEMENTARY_TABLES  # noqa: E402
from fwa.features.frame import LABEL_COLUMNS  # noqa: E402
from tools.make_uae_demo_dataset import generate  # noqa: E402

DERIVED = ["service_date", "discharge_date", "claim_date", "length_of_stay_days", "policy_type",
           "diagnosis_primary", "diagnosis_chapter", "pharmacy_bill_ratio", "icd_code_matches_procedure",
           "provider_blacklist_flag", "previous_fraud_on_policy", "days_since_policy_start",
           "discharge_readmit_gap_days", "gross_amount_aed", "approved_amount_aed", "fx_rate_used",
           "is_cashless", "num_insurers_same_event", "agent_id", "tpa", "payer_id", "raw_hash",
           "source_vocabulary", "missingness"]


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    d = tmp_path_factory.mktemp("uae_adapter")
    generate(d / "uae", d / "uae.zip", n_claims=1500, seed=777, inject=False, verbose=False)
    return d


@pytest.fixture(scope="module")
def loaded(built, config):
    a = UaeMultiTableAdapter(config, ImmutableRawStore())
    return a, a.load(str(built / "uae"))


def test_registered_under_its_key():
    assert get_adapter("uae_multitable") is UaeMultiTableAdapter
    assert UaeMultiTableAdapter.currency == "AED"
    assert UaeMultiTableAdapter.source_system == "UAE_MULTITABLE"


def test_folder_and_zip_load_identically(built, loaded, config):
    _, ds_folder = loaded
    ds_zip = UaeMultiTableAdapter(config, ImmutableRawStore()).load(str(built / "uae.zip"))
    a = ds_folder["claim_header"].drop(columns=["source_vocabulary"])
    b = ds_zip["claim_header"].drop(columns=["source_vocabulary"])
    pd.testing.assert_frame_equal(a.reset_index(drop=True), b.reset_index(drop=True))
    assert ds_folder.status == ds_zip.status
    for name in ("claim_line", "remittance", "document"):
        assert len(ds_folder[name]) == len(ds_zip[name])


def test_derived_claim_header_columns_present_and_sane(loaded):
    adapter, ds = loaded
    h = ds["claim_header"]
    missing = [c for c in DERIVED if c not in h.columns]
    assert not missing, missing
    assert h["service_date"].notna().all()
    assert (h["discharge_date"] >= h["service_date"]).all()
    assert (h["claim_date"] >= h["discharge_date"]).all()
    assert (h["length_of_stay_days"] >= 0).all()
    ip = h["claim_type"] == "INPATIENT"
    assert (h.loc[~ip, "length_of_stay_days"] == 0).all()
    assert set(h["policy_type"]) <= {"group_corporate", "family", "individual"}
    assert h["pharmacy_bill_ratio"].between(0, 1).all()
    # pharmacy claims are (almost) all medicine; the rest carry glucose test strips, a supply
    assert h.loc[h["claim_type"] == "PHARMACY", "pharmacy_bill_ratio"].gt(0.99).mean() > 0.85
    assert (h["days_since_policy_start"] >= 0).all()
    assert (h["discharge_readmit_gap_days"].loc[~ip] == 999).all()
    assert np.allclose(h["gross_amount_aed"], h["gross_amount"])  # AED at rate 1.0
    assert (h["fx_rate_used"] == 1.0).all()
    assert (h["approved_amount_aed"] <= h["gross_amount_aed"] + 0.01).all()
    assert h["agent_id"].notna().all() and h["tpa"].notna().all()
    assert h["diagnosis_chapter"].ne("UNMAPPED").mean() > 0.99
    assert h["icd_code_matches_procedure"].mean() > 0.99      # the clean base is coded consistently
    assert not h["provider_blacklist_flag"].any()              # and has no excluded providers
    assert set(h["num_insurers_same_event"]) <= {1, 2}
    # raw rows were hashed into the store before parsing
    assert h["raw_hash"].notna().all()
    assert all(adapter.raw_store.get(x) is not None for x in h["raw_hash"].head(50))
    assert ds.reasons["__certification__"].startswith("NOT CERTIFIED")
    # provider volume band derived as for the generic adapter
    assert ds["provider"]["volume_band"].notna().all()


def test_every_table_has_a_status_and_reason(loaded):
    _, ds = loaded
    for name in list(CANONICAL_TABLES) + list(SUPPLEMENTARY_TABLES):
        assert ds.status.get(name) in (PopulationStatus.POPULATED, PopulationStatus.NOT_POPULATED), name
        assert ds.reasons.get(name), name
        if ds.status[name] == PopulationStatus.POPULATED:
            assert "SYNTHETIC" in ds.reasons[name]
    assert ds.status["review_outcome"] == PopulationStatus.NOT_POPULATED
    assert ds.is_populated("bundling_edit_table") and ds.is_populated("clinician_roster")


def test_labels_never_enter_canonical_tables(built, loaded):
    adapter, ds = loaded
    for name, frame in ds.tables.items():
        assert not (set(frame.columns) & LABEL_COLUMNS), name
    labels = adapter.held_out_labels(str(built / "uae"))
    assert set(LABEL_COLUMNS) <= set(labels.columns)
    assert len(labels) == len(ds["claim_header"])
    assert labels.equals(adapter.held_out_labels(str(built / "uae.zip")))


def test_label_columns_smuggled_into_a_table_are_dropped(built, config):
    frames = {p.stem: pd.read_csv(p, dtype=str, keep_default_na=False)
              for p in (built / "uae").glob("*.csv") if p.stem != "evaluation_labels"}
    frames["claim_header"]["fraud_label"] = "1"
    ds = UaeMultiTableAdapter(config, ImmutableRawStore()).load(frames)
    assert "fraud_label" not in ds["claim_header"].columns
    assert all("fraud_label" not in v for v in ds["claim_header"]["source_vocabulary"].head(20))


def test_missing_claim_header_raises(built, tmp_path, config):
    folder = tmp_path / "no_header"
    folder.mkdir()
    shutil.copy(built / "uae" / "member.csv", folder / "member.csv")
    with pytest.raises(AdapterError, match="claim_header"):
        UaeMultiTableAdapter(config, ImmutableRawStore()).load(str(folder))
    with pytest.raises(AdapterError):
        UaeMultiTableAdapter(config, ImmutableRawStore()).load(str(tmp_path / "does_not_exist"))


def test_missing_supplementary_tables_are_not_populated(built, tmp_path, config):
    folder = tmp_path / "partial"
    shutil.copytree(built / "uae", folder)
    for name in ("bundling_edit_table", "clinician_roster", "document"):
        (folder / f"{name}.csv").unlink()
    ds = UaeMultiTableAdapter(config, ImmutableRawStore()).load(str(folder))
    for name in ("bundling_edit_table", "clinician_roster", "document"):
        assert ds.status[name] == PopulationStatus.NOT_POPULATED
        assert not ds.is_populated(name)
        assert name in ds.reasons[name]
    assert ds.is_populated("claim_line")


def test_generic_csv_loading_unaffected(config):
    sample = pd.read_csv(ROOT / "data" / "claims.csv", nrows=300)
    ds = get_adapter("generic_india_tpa")(config, ImmutableRawStore()).load(sample)
    assert len(ds["claim_header"]) == 300
    assert ds.status["claim_line"] == PopulationStatus.NOT_POPULATED
    assert isinstance(GenericIndiaTpaAdapter(config).held_out_labels(sample), pd.DataFrame)
