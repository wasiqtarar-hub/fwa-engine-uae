"""Generated report text must describe the file that was loaded.

Several sentences in the reports were written against the one-row-per-claim
demo extract ("there are no claim lines to link", "no prescriptions or pharmacy
entities", "labels were produced by three detection processes"). On a
multi-table file they were false while still reading as authoritative. These
tests pin the data-aware versions: each sentence is either measured from the
data in front of it, or says whose statement it is.
"""

from __future__ import annotations

from types import SimpleNamespace

import pandas as pd

from fwa.canonical.model import CanonicalDataset, PopulationStatus
from fwa.evaluation.reports import label_source_sentence
from fwa.graph.build import GraphService, _unbuilt_edge_sources
from fwa.presentation.explain_case import _when


def _dataset(**tables: pd.DataFrame) -> CanonicalDataset:
    ds = CanonicalDataset.empty("TEST")
    for name, frame in tables.items():
        ds.set(name, frame, PopulationStatus.POPULATED, "test fixture")
    return ds


# ---------------------------------------------------------------- line linkage

def test_line_linkage_is_none_without_claim_lines():
    """No lines: the line half of the gate is vacuous, and 100% would be a lie."""
    ds = _dataset(claim_header=pd.DataFrame({"claim_sk": ["C1", "C2"]}))
    assert ds.line_linkage_rate() is None


def test_line_linkage_is_measured_when_lines_exist():
    ds = _dataset(
        claim_header=pd.DataFrame({"claim_sk": ["C1", "C2"]}),
        claim_line=pd.DataFrame({"claim_sk": ["C1", "C1", "C2", "C9"]}),
    )
    assert ds.line_linkage_rate() == 0.75


# ---------------------------------------------------------------- label sources

def _result_with_sources(*sources: str) -> SimpleNamespace:
    labels = pd.DataFrame({"claim_sk": [f"C{i}" for i in range(len(sources))],
                           "fraud_label": 1, "ground_truth_source": list(sources)})
    return SimpleNamespace(held_out_labels=labels)


def test_synthetic_answer_key_is_not_called_a_detection_process():
    text = label_source_sentence(_result_with_sources("synthetic_injection"))
    assert "answer key" in text
    assert "detection process" not in text
    assert "rule_engine" not in text


def test_detection_processes_are_named_from_the_labels():
    text = label_source_sentence(_result_with_sources("rule_engine", "expert_review"))
    assert "2 detection processes" in text
    assert "`expert_review`" in text and "`rule_engine`" in text
    assert "pattern_detection" not in text
    assert "tautological" in text


def test_no_labels_says_so():
    assert label_source_sentence(SimpleNamespace(held_out_labels=None)) == \
        "No labels accompany this dataset."


# ---------------------------------------------------------------- graph notes

def test_edges_the_file_has_but_the_graph_does_not_build_say_so():
    ds = _dataset(
        prescription_dispense=pd.DataFrame({"rx_sk": ["R1", "R2"]}),
        provider=pd.DataFrame({"provider_sk": ["P1", "P2"],
                               "bank_account_token": ["B1", None],
                               "owner_entity_id": [None, None]}),
    )
    sources = _unbuilt_edge_sources(ds)
    assert "prescriber_pharmacy" in sources and "shared_identifier" in sources
    assert "ownership" not in sources and "referral" not in sources

    inventory = GraphService.__new__(GraphService)
    inventory.full_graph = None
    notes = dict(zip(*[GraphService.edge_inventory(inventory, ds)[c]
                       for c in ("edge_type", "note")]))
    assert notes["prescriber—pharmacy"].startswith("NOT BUILT INTO THE GRAPH")
    assert notes["ownership"].startswith("ABSENT")


def test_without_a_dataset_the_notes_are_unchanged():
    inventory = GraphService.__new__(GraphService)
    inventory.full_graph = None
    notes = GraphService.edge_inventory(inventory)["note"].tolist()
    assert "ABSENT — no prescriptions or pharmacy entities." in notes


# ---------------------------------------------------------------- headline dates

def _claims(dates):
    return pd.DataFrame({"claim_sk": [f"C{i}" for i in range(len(dates))],
                         "service_date": pd.to_datetime(dates)})


def test_a_case_whose_claims_span_months_is_not_dated_to_one_month():
    """The bug: '322 claims in December 2025' for claims spread over 18 months."""
    claims = _claims(["2024-07-03", "2025-01-10", "2025-12-20"])
    when = _when(claims, list(claims["claim_sk"]), "2025-12", "December 2025")
    assert "December 2025" in when and when.startswith(" between ")
    assert "July 2024" in when


def test_a_case_within_one_month_names_that_month():
    claims = _claims(["2025-12-01", "2025-12-20"])
    assert _when(claims, list(claims["claim_sk"]), "2025-12", "December 2025") == \
        " in December 2025"
