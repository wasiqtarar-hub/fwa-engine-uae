"""Scenario-level integration tests (build brief §12, manuscript §6.3).

    "a corrected duplicate supersedes rather than duplicates the open signal; a
     package component on a separate claim correlates into its parent episode; a
     mutated resubmission displays the field-level diff against the original; a
     peer-comparison result is **exactly reproducible from its stored baseline
     version** when re-run later; a graph case sums each claim exactly once; NLP
     results link to source spans and **visibly degrade confidence** when OCR
     quality is poor."

Six named properties, six sections. These are the tests that cross layers, so
each one names the layer boundary it is really about: lineage→controls,
episodes→correlation, lineage→evidence, peers→reproducibility,
graph→exposure, NLP→confidence.
"""

from __future__ import annotations

import datetime as _dt

import pandas as pd
import pytest

from fwa.cases.exposure import network_exposure
from fwa.engine.controls import CONTROL_IMPLEMENTATIONS
from fwa.lineage.episodes import ClaimVersionRecord, EpisodeBuilder, LineageIndex


def _with(context, **overrides):
    import copy

    ctx = copy.copy(context)
    for key, value in overrides.items():
        setattr(ctx, key, value)
    return ctx


# ------------------------------------------- 1. a corrected duplicate supersedes


@pytest.fixture()
def duplicate_pair(context):
    """Two claims identical on the duplicate key — a genuine duplicate pair."""
    frame = context.claims.head(2).copy()
    row = frame.iloc[0]
    for column in ("payer_id", "member_sk", "provider_sk", "diagnosis_primary",
                   "gross_amount_aed", "service_date", "discharge_date"):
        frame[column] = row[column]
    frame["claim_date"] = pd.to_datetime(
        [row["service_date"], row["service_date"]]
    ) + pd.to_timedelta([0, 1], unit="D")
    return frame


def test_a_duplicate_pair_raises_one_signal(context, registry, duplicate_pair):
    control = registry.get("PAY-01-R01")
    signals = CONTROL_IMPLEMENTATIONS[control.implementation](
        _with(context, claims=duplicate_pair, lineage=LineageIndex()), control
    )
    assert len(signals) == 1, "The fixture is not a duplicate pair; the next test proves nothing."
    assert len(signals[0].claim_ids) == 2


def test_a_correction_supersedes_rather_than_duplicating_the_signal(
    context, registry, duplicate_pair
):
    """§6.3's first property, at the layer boundary where it is enforced.

    The second claim is recorded as a CORRECTION of the first. The duplicate
    control must then stay silent: two concurrently open cases describing one
    corrected claim is precisely the outcome the lineage index exists to
    prevent.
    """
    earlier, later = duplicate_pair.iloc[0]["claim_sk"], duplicate_pair.iloc[1]["claim_sk"]
    lineage = LineageIndex()
    lineage.add(ClaimVersionRecord(
        claim_sk=str(later), version_no=2, relationship="CORRECTION",
        prior_claim_sk=str(earlier), changed_fields={"gross_amount_aed": (1000.0, 900.0)},
        recorded_at=_dt.datetime(2026, 3, 2),
    ))

    control = registry.get("PAY-01-R01")
    signals = CONTROL_IMPLEMENTATIONS[control.implementation](
        _with(context, claims=duplicate_pair, lineage=lineage), control
    )
    assert signals == [], "A corrected claim produced a second open duplicate signal."


def test_a_cancellation_also_supersedes(context, registry, duplicate_pair):
    earlier, later = duplicate_pair.iloc[0]["claim_sk"], duplicate_pair.iloc[1]["claim_sk"]
    lineage = LineageIndex()
    lineage.add(ClaimVersionRecord(
        claim_sk=str(later), version_no=2, relationship="CANCELLATION",
        prior_claim_sk=str(earlier), changed_fields={}, recorded_at=_dt.datetime(2026, 3, 2),
    ))
    control = registry.get("PAY-01-R01")
    assert CONTROL_IMPLEMENTATIONS[control.implementation](
        _with(context, claims=duplicate_pair, lineage=lineage), control
    ) == []


def test_the_lineage_chain_resolves_to_the_live_version():
    lineage = LineageIndex()
    for version, (prior, current) in enumerate([("C1", "C2"), ("C2", "C3")], start=2):
        lineage.add(ClaimVersionRecord(
            claim_sk=current, version_no=version, relationship="RESUBMISSION",
            prior_claim_sk=prior, changed_fields={}, recorded_at=_dt.datetime(2026, 3, version),
        ))
    assert lineage.active_claim_for("C1") == "C3"
    assert lineage.chain("C1") == ["C1", "C2", "C3"]
    assert lineage.is_superseded("C1") and lineage.is_superseded("C2")
    assert not lineage.is_superseded("C3")


def test_a_cyclic_chain_terminates():
    """Defensive: corrupt lineage data must not hang the engine."""
    lineage = LineageIndex()
    lineage.add(ClaimVersionRecord("C2", 2, "CORRECTION", "C1", {}, _dt.datetime(2026, 1, 1)))
    lineage.add(ClaimVersionRecord("C1", 3, "CORRECTION", "C2", {}, _dt.datetime(2026, 1, 2)))
    assert lineage.active_claim_for("C1") in ("C1", "C2")


# -------------------------------- 2. a component correlates into its episode


def test_a_second_claim_inside_the_window_joins_the_same_episode(config):
    """A package component billed on its own claim belongs to the parent episode."""
    window = int(config.get("readmit_window_days"))
    frame = pd.DataFrame([
        {"claim_sk": "E1", "member_sk": "M1", "provider_sk": "P1",
         "service_date": pd.Timestamp("2026-03-01"), "discharge_date": pd.Timestamp("2026-03-05"),
         "gross_amount_aed": 5000.0},
        {"claim_sk": "E2", "member_sk": "M1", "provider_sk": "P1",
         "service_date": pd.Timestamp("2026-03-06"), "discharge_date": pd.Timestamp("2026-03-06"),
         "gross_amount_aed": 800.0},
    ])
    episodes, mapping = EpisodeBuilder(window).build(frame)
    assert mapping["E1"] == mapping["E2"], "A component one day later opened a new episode."
    assert len(episodes) == 1
    assert episodes.iloc[0]["claim_count"] == 2


def test_a_claim_outside_the_window_starts_a_new_episode(config):
    window = int(config.get("readmit_window_days"))
    frame = pd.DataFrame([
        {"claim_sk": "E1", "member_sk": "M1", "provider_sk": "P1",
         "service_date": pd.Timestamp("2026-03-01"), "discharge_date": pd.Timestamp("2026-03-05"),
         "gross_amount_aed": 5000.0},
        {"claim_sk": "E2", "member_sk": "M1", "provider_sk": "P1",
         "service_date": pd.Timestamp("2026-03-05") + pd.Timedelta(days=window + 5),
         "discharge_date": pd.Timestamp("2026-03-05") + pd.Timedelta(days=window + 5),
         "gross_amount_aed": 800.0},
    ])
    _, mapping = EpisodeBuilder(window).build(frame)
    assert mapping["E1"] != mapping["E2"]


def test_an_episode_states_why_its_claims_were_linked(config, context):
    """A reviewer looking at a two-claim episode must see the linkage reason."""
    episodes, _ = EpisodeBuilder(int(config.get("readmit_window_days"))).build(context.claims)
    multi = episodes[episodes["claim_count"] > 1]
    assert not multi.empty, "No multi-claim episodes in the sample."
    assert multi["linkage_reason"].str.len().min() > 20
    assert multi["linkage_confidence"].between(0.0, 1.0).all()


# -------------------------------- 3. a mutated resubmission shows its diff


def test_a_resubmission_displays_the_field_level_diff():
    """§6.3: "must display the field-level difference between the versions"."""
    lineage = LineageIndex()
    lineage.add(ClaimVersionRecord(
        claim_sk="C2", version_no=2, relationship="RESUBMISSION", prior_claim_sk="C1",
        changed_fields={
            "diagnosis_primary": ("J18.9", "J15.9"),
            "gross_amount_aed": (4200.0, 6800.0),
        },
        recorded_at=_dt.datetime(2026, 3, 2), resubmission_type="MUTATED",
    ))
    diff = lineage.field_diff("C2")
    assert set(diff) == {"diagnosis_primary", "gross_amount_aed"}
    assert diff["diagnosis_primary"] == ("J18.9", "J15.9")
    assert diff["gross_amount_aed"][1] > diff["gross_amount_aed"][0]


def test_diffs_accumulate_across_versions():
    lineage = LineageIndex()
    lineage.add(ClaimVersionRecord("C2", 2, "RESUBMISSION", "C1",
                                   {"diagnosis_primary": ("A", "B")}, _dt.datetime(2026, 3, 2)))
    lineage.add(ClaimVersionRecord("C2", 3, "RESUBMISSION", "C1",
                                   {"gross_amount_aed": (1.0, 2.0)}, _dt.datetime(2026, 3, 3)))
    assert set(lineage.field_diff("C2")) == {"diagnosis_primary", "gross_amount_aed"}


def test_a_claim_with_no_versions_has_an_empty_diff():
    assert LineageIndex().field_diff("C9") == {}


# ------------------------- 4. a peer result reproduces from its baseline version


def test_a_peer_comparison_reproduces_exactly_from_its_stored_baseline_version(
    config, claims, registry, context
):
    """§6.3's reproducibility requirement, run as the §6.3 text describes it.

    A control that compares against peers is executed, its signals' stored
    ``peer_baseline_version`` recorded, and the whole comparison re-run from a
    freshly built peer service. The versions and the resulting numbers must
    match exactly — not approximately.
    """
    from fwa.statistical import PeerService

    control = registry.get("PAY-06-R03")
    impl = CONTROL_IMPLEMENTATIONS[control.implementation]

    first = impl(_with(context, peers=PeerService(config, claims)), control)
    later = impl(_with(context, peers=PeerService(config, claims)), control)
    assert first, "PAY-06-R03 produced no peer-compared signals."

    by_id = {s.signal_id: s for s in later}
    for signal in first:
        other = by_id[signal.signal_id]
        assert signal.evidence["peer_baseline_version"] == other.evidence["peer_baseline_version"]
        assert signal.evidence["peer_median_aed"] == other.evidence["peer_median_aed"]
        assert signal.evidence["robust_residual"] == other.evidence["robust_residual"]
        assert signal.peer_level_used == other.peer_level_used


def test_every_peer_compared_signal_records_the_level_it_used(context, registry):
    control = registry.get("PAY-06-R03")
    signals = CONTROL_IMPLEMENTATIONS[control.implementation](context, control)
    for signal in signals:
        assert signal.peer_level_used
        assert signal.evidence["peer_level_used"] == signal.peer_level_used
        assert signal.evidence["peer_n"] > 0


# ------------------------------- 5. a graph case sums each claim exactly once


def test_a_graph_case_counts_a_claim_touching_three_nodes_once():
    """§4.11 and §6.3: the exact case the mapping shape is designed to prevent."""
    touched: dict[str, float] = {}
    for _node, claims in (
        ("provider P1", {"C1": 500.0, "C2": 300.0}),
        ("member M1", {"C1": 500.0}),
        ("community K3", {"C1": 500.0, "C2": 300.0}),
    ):
        touched.update(claims)

    assert network_exposure(touched).amount_aed == pytest.approx(800.0)


def test_a_network_signal_in_a_real_run_does_not_double_count(context, registry):
    control = registry.get("NET-02-R02")
    signals = CONTROL_IMPLEMENTATIONS[control.implementation](context, control)
    for signal in signals:
        assert len(signal.claim_ids) == len(set(signal.claim_ids)), (
            f"{signal.signal_id} lists a claim twice."
        )


def test_correlated_network_cases_do_not_double_count_a_shared_claim(sample_cases):
    for case in list(sample_cases[1].values())[:100]:
        assert len(case.claim_ids) == len(set(case.claim_ids))


# ------------------------- 6. NLP links to spans and degrades on poor OCR


def test_every_extraction_carries_a_source_span(pipeline):
    """§4.9: a clinical finding without a source span is not persisted at all."""
    analyses = pipeline.documents.pipeline.analyses
    assert analyses, "The document pipeline produced no analyses."
    for analysis in list(analyses.values())[:200]:
        for extraction in analysis.extractions.values():
            assert extraction.span_text
            assert extraction.span_end > extraction.span_start >= 0


def test_an_extraction_without_a_span_cannot_be_constructed():
    from fwa.nlp.pipeline import Extraction, MissingSpanError

    with pytest.raises(MissingSpanError):
        Extraction(field_name="diagnosis", value="J18.9", span_text="", span_start=0,
                   span_end=0, raw_confidence=0.9, ocr_confidence=1.0, language="en")


def test_poor_ocr_visibly_degrades_confidence():
    """§6.3: a perfect match on a bad scan is not a confident finding."""
    from fwa.nlp.pipeline import Extraction

    def extraction(ocr):
        return Extraction(field_name="diagnosis", value="J18.9", span_text="Dx: J18.9",
                          span_start=0, span_end=9, raw_confidence=0.95,
                          ocr_confidence=ocr, language="en")

    clean = extraction(1.0)
    poor = extraction(0.45)

    assert poor.confidence < clean.confidence
    assert poor.confidence == pytest.approx(0.95 * 0.45)
    assert poor.to_dict()["ocr_confidence"] == 0.45


def test_the_poor_ocr_subset_of_the_corpus_scores_lower_on_average(pipeline):
    """Not just arithmetic — the property holds across the generated corpus."""
    analyses = list(pipeline.documents.pipeline.analyses.values())
    scored = [(a.ocr_confidence, e.confidence)
              for a in analyses for e in a.extractions.values()]
    assert scored, "No extractions to compare."

    poor = [c for ocr, c in scored if ocr < 0.8]
    good = [c for ocr, c in scored if ocr >= 0.8]
    assert poor and good, "The corpus contains no poor-OCR documents to compare against."
    assert sum(poor) / len(poor) < sum(good) / len(good)


def test_every_document_is_marked_synthetic(pipeline):
    """No synthetic document may be mistakable for a real clinical record."""
    for document in list(pipeline.documents.corpus.documents.values())[:50]:
        assert "SYNTHETIC" in document.text.upper()
        assert "SYNTHETIC" in document.filename.upper()
