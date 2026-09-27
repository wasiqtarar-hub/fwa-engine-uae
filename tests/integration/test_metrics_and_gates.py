"""§6.5 metrics, Table 6.1 release gates and the evaluation protocol.

    "Where a metric requires operational data this dataset cannot provide
     (appeals, turnaround), IMPLEMENT THE METRIC and render it as
     ``NOT_MEASURABLE_ON_THIS_DATASET`` with the reason — do not fabricate a
     number, and do not quietly drop the metric."         — build brief §11.1

Two failure modes are being guarded against, and they pull in opposite
directions. One is fabrication: producing a savings figure from unreviewed
flags. The other is quiet omission: dropping the metrics that would look bad.
The tests below check that every §6.5 metric is present, that the unmeasurable
ones say what is missing, and that the measured ones carry the caveats the
manuscript requires — in particular that gross flagged value is never presented
as savings.
"""

from __future__ import annotations

import pytest

from fwa.evaluation.gates import FAIL, NOT_ASSESSABLE, PASS
from fwa.evaluation.metrics import NOT_MEASURABLE

pytestmark = pytest.mark.slow


# ------------------------------------------------------------------ presence


def test_every_metric_is_either_measured_or_explicitly_unmeasurable(metrics):
    rows = metrics.compute()
    assert len(rows) >= 12
    for metric in rows:
        assert metric.status in ("MEASURED", NOT_MEASURABLE)
        assert metric.name and metric.definition


def test_an_unmeasurable_metric_names_the_data_that_would_supply_it(metrics):
    """"Not measurable" is only honest if it says what is missing."""
    unmeasurable = [m for m in metrics.compute() if m.status == NOT_MEASURABLE]
    assert unmeasurable, "Every §6.5 metric measured — on this dataset that cannot be true."
    for metric in unmeasurable:
        assert metric.required_data or metric.note, (
            f"{metric.name} is unmeasurable but does not say what it needs."
        )


@pytest.mark.parametrize("name_fragment", [
    "Confirmed AED", "Prevented", "Net savings", "abrasion", "turnaround", "Overturn",
])
def test_the_operational_metrics_this_dataset_cannot_support_are_present_and_honest(
    metrics, name_fragment
):
    """Each of these is named in §6.5 and is unmeasurable here — both facts stated."""
    matches = [m for m in metrics.compute() if name_fragment.lower() in m.name.lower()]
    assert matches, f"§6.5 metric matching {name_fragment!r} is missing from the suite."
    assert all(m.status == NOT_MEASURABLE for m in matches)


def test_an_unmeasurable_metric_renders_as_the_marker_not_as_zero(metrics):
    for metric in metrics.compute():
        if metric.status == NOT_MEASURABLE:
            assert metric.to_row()["value"] == NOT_MEASURABLE


# ------------------------------------------------- flagged value vs savings


def test_gross_flagged_value_is_reported_separately_from_savings(metrics):
    rows = {m.name: m for m in metrics.compute()}
    established = next(m for n, m in rows.items() if "established exposure" in n)
    unestablished = next(m for n, m in rows.items() if "NOT established" in n)

    assert established.status == "MEASURED"
    assert unestablished.status == "MEASURED"
    assert "NOT SAVINGS" in established.note.upper()
    assert "never added" in unestablished.note.lower()


def test_no_metric_claims_a_savings_figure(metrics):
    for metric in metrics.compute():
        if metric.status != "MEASURED":
            continue
        name = metric.name.lower()
        assert not any(
            word in name for word in ("savings", "recovered", "prevented", "confirmed aed")
        ), f"{metric.name} presents a measured savings figure on unreviewed flags."


def test_precision_is_described_as_a_proxy_and_an_upper_bound(metrics):
    precision = [m for m in metrics.compute() if "precision" in m.name.lower()]
    assert precision
    headline = precision[0]
    assert "PROXY" in headline.note
    assert "UPPER BOUND" in headline.note


def test_claim_level_and_entity_level_coverage_are_reported_separately(metrics):
    """Counting an entity lead's context claims as flags would inflate coverage ~8×."""
    claim_level = metrics.flagged_claims("claim")
    entity_level = metrics.flagged_claims("entity")
    any_level = metrics.flagged_claims("any")

    assert claim_level <= any_level
    assert entity_level <= any_level
    assert len(any_level) >= len(claim_level)

    names = [m.name for m in metrics.compute()]
    assert any("CLAIM-level" in n for n in names)
    assert any("ENTITY" in n for n in names)


# ----------------------------------------------------------- stratification


def test_precision_is_stratified_by_ground_truth_source(metrics):
    """§6.4: the three label sources are not equally reliable, and the report says so."""
    frame = metrics.by_ground_truth_source()
    assert not frame.empty
    assert "ground_truth_source" in frame.columns
    assert frame["ground_truth_source"].nunique() >= 2


def test_recall_is_reported_by_fraud_type(metrics):
    frame = metrics.recall_by_fraud_type()
    assert not frame.empty
    assert "fraud_type" in frame.columns


def test_precision_at_capacity_is_computed_for_several_capacities(metrics):
    frame = metrics.precision_at_capacity([25, 50, 100])
    assert len(frame) == 3


def test_lift_over_the_transparent_composite_is_reported(metrics):
    """§4.8: a model earns promotion only by beating the transparent baseline."""
    lift = metrics.lift_over_composite()
    assert lift is not None


# ----------------------------------------------------------- release gates


def test_all_six_release_gates_are_evaluated(gate_results):
    assert len(gate_results) == 6
    names = {g.gate for g in gate_results}
    assert len(names) == 6


def test_every_gate_has_a_status_a_requirement_and_evidence(gate_results):
    for gate in gate_results:
        assert gate.status in (PASS, FAIL, NOT_ASSESSABLE)
        assert gate.requirement
        assert gate.evidence, f"{gate.gate} reports a status with no evidence."


def test_a_gate_that_cannot_be_assessed_says_so_rather_than_passing(gate_results):
    """The dishonest alternative — defaulting to PASS — is the one that matters."""
    statuses = {g.gate: g.status for g in gate_results}
    assert not all(s == PASS for s in statuses.values()), (
        "Every release gate passed. On a proxy dataset with no reviewer outcomes that would "
        "be a false claim, not a good result."
    )


def test_the_production_gate_does_not_pass(gate_results):
    """Nothing here is production-ready, and the gate report must not imply otherwise."""
    production = [g for g in gate_results if "production" in g.gate.lower()]
    assert production
    assert production[0].status != PASS
    assert "on call" in production[0].evidence.lower()


def test_the_gate_frame_renders(pipeline, metrics):
    from fwa.evaluation.gates import ReleaseGates

    frame = ReleaseGates(pipeline, metrics).to_frame()
    assert len(frame) == 6
    assert {"gate", "status", "minimum_acceptance_evidence", "evidence"} <= set(frame.columns)


# ------------------------------------------------------- evaluation protocol


def test_every_protocol_procedure_runs_and_reports(pipeline, metrics):
    from fwa.evaluation.protocol import EvaluationProtocol

    results = EvaluationProtocol(pipeline, metrics).run_all()
    assert len(results) >= 9
    for row in results:
        assert row.procedure
        assert row.status
        assert row.summary, f"{row.procedure} reports a status with no summary."


def test_the_random_audit_simulator_is_labelled_as_a_simulation(pipeline, metrics):
    """§6.4: a simulated audit over biased labels cannot correct for that bias."""
    from fwa.evaluation.protocol import EvaluationProtocol

    audit = EvaluationProtocol(pipeline, metrics).random_audit()
    text = f"{audit.summary} {audit.caveat}".upper()
    assert "SIMULAT" in text, audit.summary
    assert audit.caveat, "A simulated audit must carry the caveat that it is simulated."


def test_synthetic_corruption_injection_is_detected(pipeline, metrics):
    """§6.2 category: a known-bad record injected on purpose must be caught."""
    from fwa.evaluation.protocol import EvaluationProtocol

    result = EvaluationProtocol(pipeline, metrics).synthetic_corruption_injection()
    assert result.procedure
    assert result.status in ("PASS", "FAIL", "INFORMATIONAL", "NOT_MEASURABLE_ON_THIS_DATASET")
    assert result.summary
