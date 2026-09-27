"""The whole pipeline, end to end, on the real file (build brief §2, §16).

    "``pip install -e . && streamlit run app/Home.py`` ... The validation run
     must be reproducible: same input, same seed, same output."

These tests run the epics in §4.13 order over ``claims.csv`` and then assert
the properties that only exist once everything is joined up: that the stages
happened in order and produced what the next one needed, that the run is
reproducible from its seed, that the audit chain survives a full run, and that
the five required reports are written with content in them.
"""

from __future__ import annotations

import datetime as _dt

import pandas as pd
import pytest

from fwa.enums import DENYING_DISPOSITIONS, Disposition, TYPES_ALLOWED_TO_DENY

pytestmark = pytest.mark.slow


# ---------------------------------------------------------------- the stages


def test_the_run_produces_every_layer(pipeline):
    assert pipeline.dataset is not None
    assert not pipeline.claims.empty
    assert pipeline.features is not None
    assert len(pipeline.registry) == 164
    assert pipeline.signals
    assert pipeline.evaluation is not None
    assert pipeline.graph is not None
    assert pipeline.models is not None
    assert pipeline.documents is not None
    assert pipeline.ai is not None
    assert pipeline.cases.cases


def test_every_epic_is_timed(pipeline):
    """§4.13's sequencing, visible in the run rather than only in the plan."""
    assert pipeline.timings
    assert all(v >= 0 for v in pipeline.timings.values())


def test_the_adapter_and_tenant_are_recorded_on_the_result(pipeline):
    assert pipeline.adapter_name
    assert pipeline.tenant_id == "T001"
    assert pipeline.seed == int(pipeline.config.get("random_seed"))


def test_raw_records_are_hashed_before_parsing(pipeline):
    """§3: a SHA-256 of the raw record, taken before any interpretation."""
    assert pipeline.raw_store is not None
    assert len(pipeline.raw_store) == len(pipeline.claims)
    for digest in list(pipeline.dataset.raw_hashes.values())[:20]:
        assert len(digest) == 64
        int(digest, 16)  # a real hex digest


TABLE_3_5 = [
    "member", "coverage_period", "benefit_rule_version", "provider",
    "provider_status_period", "claim_header", "claim_line", "diagnosis", "encounter",
    "observation", "authorization", "authorization_line", "claim_version", "remittance",
    "prescription_dispense", "policy_event",
]


def test_the_canonical_population_report_is_honest(pipeline):
    """Table 3.5's sixteen canonical tables, each either populated or explicitly not.

    ``review_outcome`` makes seventeen. It is not one of Table 3.5's sixteen —
    it is the §3.8 reviewer-outcome capture the build brief requires, and it is
    listed separately here so that the count cannot drift without someone
    noticing which table was added.
    """
    report = pipeline.dataset.population_report()
    tables = set(report["table"])
    assert set(TABLE_3_5) <= tables
    assert tables - set(TABLE_3_5) == {"review_outcome"}
    assert len(report) == 17
    assert set(report["status"]) <= {"POPULATED", "PARTIAL", "NOT_POPULATED"}
    not_populated = report[report["status"] == "NOT_POPULATED"]
    assert not not_populated.empty, "Every canonical table populated from one CSV is not credible."
    assert (not_populated["reason"].str.len() > 20).all()


# ------------------------------------------------------------- the boundary


def test_no_signal_in_a_real_run_denies_from_a_score(pipeline):
    """The §3.3 boundary, over every signal the system actually produced."""
    for signal in pipeline.signals:
        if signal.disposition in DENYING_DISPOSITIONS:
            control = pipeline.registry.get(signal.rule_id)
            assert all(t in TYPES_ALLOWED_TO_DENY for t in control.type), (
                f"{signal.signal_id} from {signal.rule_id} ({control.type_label}) denies."
            )


def test_no_model_derived_signal_denies_a_claim(pipeline):
    """§3.3, at the signal level rather than the catalogue level.

    Not every model-domain signal is MONITOR_ONLY — CLN-01-R02 is typed ``S/M``
    and Appendix C routes it to a prepay sample, which is a request for a human
    look, not a denial. What must never happen is a model-derived signal
    denying or repricing, and that is what this asserts.
    """
    from fwa.enums import SignalDomain

    model_signals = [s for s in pipeline.signals if s.domain is SignalDomain.MODEL]
    assert model_signals, "No model-derived signals in the run; the assertion is vacuous."
    for signal in model_signals:
        assert signal.disposition not in DENYING_DISPOSITIONS, (
            f"{signal.rule_id} produced a model-domain signal with disposition "
            f"{signal.disposition.value}."
        )


def test_every_pure_anomaly_signal_is_monitor_only(pipeline):
    """A signal from a control whose type is only ``M`` can do nothing but monitor."""
    from fwa.enums import ControlType

    for signal in pipeline.signals:
        control = pipeline.registry.get(signal.rule_id)
        if list(control.type) == [ControlType.M]:
            assert signal.disposition is Disposition.MONITOR_ONLY


def test_every_signal_carries_the_rule_version_that_produced_it(pipeline):
    for signal in pipeline.signals:
        assert signal.rule_version == pipeline.registry.get(signal.rule_id).version


def test_every_signal_carries_evidence(pipeline):
    for signal in pipeline.signals:
        assert signal.evidence, f"{signal.signal_id} has no evidence to show a reviewer."


def test_every_shadow_signal_is_marked_as_shadow(pipeline):
    """Nothing in this catalogue is active, so nothing in the queue should look live."""
    assert all(s.is_shadow for s in pipeline.signals)


# ----------------------------------------------------------- reproducibility


def test_the_same_input_and_seed_produce_the_same_signals(project_root):
    """§6.3 and build brief §2: the validation run is reproducible.

    Both runs are configured identically — documents off, AI off — because the
    claim is that the *same* run repeats, not that two differently configured
    runs agree. (They do not, and should not: the synthetic document corpus
    feeds DOC-01-R02 and CLN-01-R03, so a run with documents legitimately
    produces signals a run without them cannot.)
    """
    from fwa.pipeline import run_pipeline

    kwargs = dict(generate_documents=None, enable_ai=False, verbose=False)
    first = run_pipeline(project_root / "data" / "claims.csv", **kwargs)
    second = run_pipeline(project_root / "data" / "claims.csv", **kwargs)

    assert {s.signal_id for s in first.signals} == {s.signal_id for s in second.signals}
    assert set(first.cases.cases) == set(second.cases.cases)

    by_id = {s.signal_id: s for s in second.signals}
    for signal in first.signals:
        other = by_id[signal.signal_id]
        assert signal.disposition == other.disposition
        assert signal.exposure_aed == pytest.approx(other.exposure_aed)
        assert signal.evidence_strength == pytest.approx(other.evidence_strength)


def test_signal_ids_are_deterministic_functions_of_their_content(pipeline):
    """Idempotency at the level that makes replay safe."""
    by_id = {}
    for signal in pipeline.signals:
        key = (signal.tenant_id, signal.rule_id, signal.rule_version,
               signal.subject_type, signal.subject_id, signal.fact_key, signal.period_bucket)
        if key in by_id:
            assert by_id[key] == signal.signal_id
        by_id[key] = signal.signal_id


def test_replaying_the_store_adds_no_duplicates(pipeline):
    from fwa.engine.signals import SignalStore

    store = SignalStore()
    store.extend(pipeline.signals)
    before = len(store)
    store.extend(pipeline.signals)
    assert len(store) == before


# ------------------------------------------------------------- alert volumes


def test_a_ceiling_breach_is_routed_to_monitor_only_not_discarded(pipeline):
    """§3.10: a runaway rule must not flood the queue — and must not vanish either."""
    breaches = pipeline.evaluation.ceiling_breaches
    for rule_id in breaches:
        signals = [s for s in pipeline.signals if s.rule_id == rule_id]
        assert signals, f"{rule_id} breached the ceiling and then disappeared."
        assert all(s.disposition is Disposition.MONITOR_ONLY for s in signals)
        assert all("alert_volume_ceiling_breached" in s.evidence for s in signals)


def test_no_control_errored_during_the_run(pipeline):
    assert not pipeline.evaluation.errors, pipeline.evaluation.errors


def test_stage_latency_is_measured_against_the_table_34_ceilings(pipeline):
    report = pipeline.evaluation.latency_report(pipeline.config)
    assert not report.empty
    assert {"stage", "observed_ms", "ceiling_ms"} <= set(report.columns)


# ------------------------------------------------------------------- audit


def test_the_audit_chain_verifies_after_a_full_run(pipeline):
    ok, message = pipeline.audit.verify_chain()
    assert ok, message
    assert len(pipeline.audit) > 0


def test_the_audit_log_cannot_be_edited(pipeline):
    from fwa.audit.log import AuditLogViolation

    with pytest.raises(AuditLogViolation):
        del pipeline.audit[0]
    with pytest.raises(AuditLogViolation):
        pipeline.audit.clear()


# -------------------------------------------------------------------- queue


def test_the_queue_is_ordered_by_priority(pipeline):
    queue = pipeline.queue("T001")
    assert not queue.empty
    priorities = queue["priority"].dropna().tolist()
    assert priorities == sorted(priorities, reverse=True)


def test_every_case_resolves_back_to_its_signals(pipeline):
    for case_id in list(pipeline.cases.cases)[:50]:
        signals = pipeline.signals_for_case(case_id)
        assert signals, f"{case_id} resolves to no signals."


def test_the_summary_renders_without_fabricating_a_savings_figure(pipeline):
    summary = pipeline.summary()
    assert summary
    text = str(summary).lower()
    assert "savings" not in text or "not savings" in text


# ------------------------------------------------------------------ reports


def test_the_five_required_reports_are_written(pipeline, tmp_path):
    from fwa.evaluation.reports import ReportBuilder

    written = ReportBuilder(pipeline, output_dir=tmp_path).build_all()
    names = {p.name for p in written}

    for required in ("validation_report.md", "control_coverage_matrix.csv",
                     "traceability_matrix.csv", "model_card.md", "limitations.md"):
        assert required in names, f"{required} was not written."

    for path in written:
        assert path.exists()
        assert path.stat().st_size > 0, f"{path.name} is empty."


def test_the_coverage_matrix_has_a_row_per_control(pipeline, tmp_path):
    from fwa.evaluation.reports import ReportBuilder

    ReportBuilder(pipeline, output_dir=tmp_path).build_all()
    matrix = pd.read_csv(tmp_path / "control_coverage_matrix.csv")
    assert len(matrix) == 164
    assert matrix["data_support_reason"].notna().all()


def test_the_validation_report_states_what_the_system_is_not(pipeline, tmp_path):
    from fwa.evaluation.reports import ReportBuilder

    ReportBuilder(pipeline, output_dir=tmp_path).build_all()
    text = (tmp_path / "validation_report.md").read_text(encoding="utf-8")
    assert "signal is not a fraud finding" in text.lower()
    assert "proxy" in text.lower()


def test_the_limitations_document_is_substantial(pipeline, tmp_path):
    """§11.5: the limitations are a deliverable, not a footnote."""
    from fwa.evaluation.reports import ReportBuilder

    ReportBuilder(pipeline, output_dir=tmp_path).build_all()
    text = (tmp_path / "limitations.md").read_text(encoding="utf-8")
    assert len(text) > 2000
    for topic in ("proxy", "label", "not certified"):
        assert topic in text.lower(), f"limitations.md does not discuss {topic}."


def test_the_model_card_records_that_the_models_are_not_promoted(pipeline, tmp_path):
    from fwa.evaluation.reports import ReportBuilder

    ReportBuilder(pipeline, output_dir=tmp_path).build_all()
    text = (tmp_path / "model_card.md").read_text(encoding="utf-8")
    assert "shadow" in text.lower()


def test_the_traceability_matrix_covers_the_requirements(pipeline, tmp_path):
    from fwa.evaluation.reports import ReportBuilder

    ReportBuilder(pipeline, output_dir=tmp_path).build_all()
    matrix = pd.read_csv(tmp_path / "traceability_matrix.csv")
    assert len(matrix) >= 15
    assert matrix.notna().all().any()


def test_the_reports_reproduce_every_finding_across_two_runs(project_root, tmp_path):
    """§6.3 and build brief §17, checked on the files rather than on the objects.

    The comparison drops the columns that are *measurements of the run itself* —
    per-control ``elapsed_ms`` and each case's ``opened_at`` — because those are
    wall-clock readings and reproducing them byte-for-byte would mean not
    measuring them. Everything a reader would cite is compared: which controls
    ran, what they found, which cases resulted, and with what disposition,
    exposure and priority.
    """
    from fwa.evaluation.reports import ReportBuilder
    from fwa.pipeline import run_pipeline

    def build(into):
        result = run_pipeline(
            project_root / "data" / "claims.csv",
            generate_documents=None, enable_ai=False, verbose=False,
        )
        ReportBuilder(result, output_dir=into).build_all()
        return into

    first = build(tmp_path / "run1")
    second = build(tmp_path / "run2")

    volatile = {"elapsed_ms", "opened_at"}
    compared = 0
    for name in ("control_run_report.csv", "control_coverage_matrix.csv", "case_queue.csv",
                 "metrics.csv", "precision_by_ground_truth_source.csv", "recall_by_fraud_type.csv"):
        a = pd.read_csv(first / name)
        b = pd.read_csv(second / name)
        columns = [c for c in a.columns if c not in volatile]
        pd.testing.assert_frame_equal(
            a[columns].sort_index(axis=0).reset_index(drop=True),
            b[columns].sort_index(axis=0).reset_index(drop=True),
            check_exact=False, rtol=1e-9,
            obj=f"{name} differs between two runs on the same input",
        )
        compared += 1
    assert compared == 6


def test_the_run_does_not_claim_more_reproducibility_than_it_has(project_root):
    """The footer says findings reproduce, not that the files are byte-identical."""
    import inspect

    from fwa import run_validation

    source = inspect.getsource(run_validation.main)
    assert "reproduces these files exactly" not in source
    assert "reproduces every finding exactly" in source
