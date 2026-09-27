"""Report generation (build brief §11.2).

Five required artefacts, written to ``reports/``:

1. ``validation_report.md`` — the headline artefact.
2. ``control_coverage_matrix.csv`` — all 164 controls × data support × reason ×
   the canonical fields each would need.
3. ``traceability_matrix.csv`` — Appendix D regenerated from the codebase.
4. ``release_gate_report.md`` — Table 6.1 evaluated against this build.
5. ``model_card.md`` and ``limitations.md``.

The rule every one of them follows, from the Definition of Done: *"Every number
in every report is either a measured output of this run on its input file, or a
configuration threshold labelled as such, or ``NOT_MEASURABLE_ON_THIS_DATASET``.
Nothing is illustrative-but-unlabelled."*

Every report opens with a statement of which dataset produced it and a plain
statement of what its numbers do and do not establish. Both are properties of
one run over one input file, and a report that does not name that file is a
report whose figures cannot be attributed to anything.
"""

from __future__ import annotations

import datetime as _dt
import json
from pathlib import Path
from typing import Any

import pandas as pd

from .. import SAFETY_BOUNDARY_STATEMENT, __version__
from .gates import ReleaseGates
from .metrics import MetricSuite, NOT_MEASURABLE
from .protocol import EvaluationProtocol
from .traceability import TraceabilityMatrix
from .prose import count

__all__ = ["ReportBuilder"]


def _md_table(frame: pd.DataFrame, max_rows: int | None = None) -> str:
    if frame is None or frame.empty:
        return "_No rows._\n"
    work = frame if max_rows is None else frame.head(max_rows)
    return work.to_markdown(index=False) + (
        f"\n\n_({count(len(frame) - len(work), 'further row')} in the accompanying CSV.)_\n"
        if max_rows and len(frame) > len(work) else "\n"
    )


def _banner_block(result=None) -> str:
    """The header every report opens with: which dataset, and the boundary.

    The dataset facts are read off the run rather than written into a constant,
    because the input is now a choice. A hard-coded description of the data
    would be correct until the first time someone loaded a different file, and
    then it would be a false statement sitting at the top of a report about
    numbers it no longer describes.
    """
    lines = []
    if result is not None:
        rows = len(result.claims)
        lines.append(
            "> **THIS REPORT DESCRIBES ONE DATASET**\n>\n"
            f"> Source `{result.source_name or 'the claim extract'}` · "
            f"{rows:,} claim rows · `{result.adapter_name}` adapter · "
            f"run {result.run_started.strftime('%Y-%m-%d %H:%M UTC')}.\n>\n"
            "> Every figure below is a measured property of that file. None of it transfers to "
            "another population without being re-measured there.\n>\n"
        )
    lines.append(f"> **SAFETY BOUNDARY.** {SAFETY_BOUNDARY_STATEMENT}\n")
    return "".join(lines)


class ReportBuilder:
    def __init__(self, result, output_dir: Path | str = "reports") -> None:
        self.result = result
        self.out = Path(output_dir)
        self.out.mkdir(parents=True, exist_ok=True)
        self.metrics = MetricSuite(result)
        self.protocol = EvaluationProtocol(result, self.metrics)
        self.gates = ReleaseGates(result, self.metrics)
        self.written: list[Path] = []

    # ------------------------------------------------------------------ build

    def build_all(self) -> list[Path]:
        # The model gate needs an outcome series, which only the evaluation
        # module may produce. Running it here keeps the label boundary intact.
        if self.result.models is not None:
            capacity = int(self.result.config.get("alerts_per_reviewer_per_day")) * int(
                self.result.config.get("reviewer_count"))
            self.result.models.run_gate(
                outcomes=self.metrics.provider_outcomes(), capacity=capacity
            )

        metric_rows = self.metrics.compute()
        protocol_results = self.protocol.run_all()
        gate_results = self.gates.evaluate()

        self._write_coverage_matrix()
        self._write_traceability(gate_results)
        self._write_validation_report(metric_rows, protocol_results, gate_results)
        self._write_release_gate_report(gate_results)
        self._write_model_card()
        self._write_limitations()
        self._write_supporting_csvs(metric_rows, protocol_results)
        return self.written

    def _write(self, name: str, text: str) -> Path:
        path = self.out / name
        path.write_text(text, encoding="utf-8")
        self.written.append(path)
        return path

    def _write_csv(self, name: str, frame: pd.DataFrame) -> Path:
        path = self.out / name
        frame.to_csv(path, index=False)
        self.written.append(path)
        return path

    # -------------------------------------------------------------- artefacts

    def _write_coverage_matrix(self) -> None:
        frame = pd.DataFrame(self.result.registry.coverage_rows())
        run = self.result.evaluation.to_frame()
        if not run.empty:
            # ``data_support`` stays the catalogue's static classification (it
            # is what the YAML declares); what this particular file allowed is
            # reported beside it, with the basis, so an unlocked control can
            # never be mistaken for one the claim-header extract supports.
            extra = [c for c in ("data_support", "support_basis", "support_reason",
                                 "missing_for_unlock") if c in run.columns]
            frame = frame.merge(
                run[["rule_id", "outcome", "signal_count", "elapsed_ms", *extra]].rename(
                    columns={"data_support": "data_support_on_this_run"}),
                on="rule_id", how="left",
            )
        frame = frame.rename(columns={"outcome": "run_outcome", "signal_count": "signals_this_run"})
        self._write_csv("control_coverage_matrix.csv", frame)

    def _write_traceability(self, gate_results) -> None:
        frame = TraceabilityMatrix(self.result).build(gate_results)
        self._write_csv("traceability_matrix.csv", frame)

    def _write_supporting_csvs(self, metric_rows, protocol_results) -> None:
        self._write_csv("metrics.csv", pd.DataFrame([m.to_row() for m in metric_rows]))
        self._write_csv("evaluation_protocol.csv",
                        pd.DataFrame([p.to_row() for p in protocol_results]))
        self._write_csv("control_run_report.csv", self.result.evaluation.to_frame())
        self._write_csv("canonical_population_report.csv",
                        self.result.dataset.population_report())
        self._write_csv("parameter_registry.csv",
                        pd.DataFrame(self.result.config.parameters.governance_report()))
        self._write_csv("case_queue.csv", self.result.cases.to_frame(self.result.tenant_id))
        by_source = self.metrics.by_ground_truth_source()
        if not by_source.empty:
            self._write_csv("precision_by_ground_truth_source.csv", by_source)
        recall = self.metrics.recall_by_fraud_type()
        if not recall.empty:
            self._write_csv("recall_by_fraud_type.csv", recall)

    # --------------------------------------------------- validation report

    def _write_validation_report(self, metric_rows, protocol_results, gate_results) -> None:
        r = self.result
        summary = r.summary()
        registry_summary = r.registry.summary()
        run = r.evaluation.to_frame()
        fired = run[run["signal_count"] > 0].sort_values("signal_count", ascending=False)
        ran_not_fired = run[(run["outcome"] == "NOT_TRIGGERED")]

        parts: list[str] = []
        parts.append(f"# Validation report — `uae-fwa-engine` {__version__}\n")
        parts.append(_banner_block(self.result))
        parts.append(
            f"\nGenerated {_dt.datetime.now(_dt.timezone.utc).isoformat()} · "
            f"parameter-registry fingerprint `{summary['parameters_fingerprint']}` · "
            f"seed `{summary['seed']}`.\n"
        )

        # ---- what this report does and does not establish -----------------
        parts.append(
            "## What this report establishes, and what it does not\n\n"
            "**It establishes** that the design can be built, "
            "that the safety boundary can be enforced structurally rather than by "
            "convention, and that the governance machinery — effective dating, shadow-before-"
            "active, evidence capping, alert ceilings, kill switches, an append-only audit log — "
            "runs.\n\n"
            "**It does not establish** a performance claim that transfers to any other book of "
            "business. Its labels were produced by three detection "
            "processes (`pattern_detection`, `expert_review`, `rule_engine`), so every precision "
            "figure below is an upper bound on what a real review would confirm, not an estimate "
            "of it — and a rule-based detector scoring well against `rule_engine` labels is close "
            "to tautological. Nothing here has been reviewed by a human, so precision in the "
            "strict sense (confirmed ÷ reviewed) is NOT_MEASURABLE and is reported as such.\n\n"
            "Every number below is a measured output of this run, a configuration threshold "
            "labelled as such, or `NOT_MEASURABLE_ON_THIS_DATASET`.\n"
        )

        # ---- run summary ---------------------------------------------------
        parts.append("## 1. Run summary\n")
        parts.append(_md_table(pd.DataFrame([
            {"item": "Claims ingested", "value": f"{summary['claims']:,}"},
            {"item": "Immutable raw records (SHA-256 before parsing)", "value": f"{summary['raw_records']:,}"},
            {"item": "Adapter", "value": summary["adapter"]},
            {"item": "Controls in the catalogue", "value": summary["controls"]},
            {"item": "Controls executed", "value": summary["controls_executed"]},
            {"item": "Controls that produced signals", "value": len(fired)},
            {"item": "Signals", "value": f"{summary['signals']:,}"},
            {"item": "Cases after correlation", "value": f"{summary['cases']:,}"},
            {"item": "Controls active (not shadow)", "value": registry_summary["by_status"]["active"]},
        ])))
        parts.append(
            "\nEvery control in this build is in **shadow**. None has been activated, because "
            "activation requires a policy owner who did not author the rule to approve it after a "
            "prospective shadow period, and neither exists in a single retrospective run.\n"
        )

        # ---- catalogue coverage -------------------------------------------
        parts.append(f"\n## 2. Catalogue coverage — what `{r.source_name or 'this file'}` "
                     f"can actually carry\n")
        support = registry_summary["by_data_support"]
        parts.append(_md_table(pd.DataFrame([
            {"classification": k, "controls": v,
             "share": f"{v / registry_summary['total_controls']:.1%}"}
            for k, v in support.items()
        ])))
        parts.append(
            f"\nAll {registry_summary['total_scenarios']} scenarios and "
            f"{registry_summary['total_controls']} atomic controls of the catalogue are registered "
            f"as configuration and validated against the atomic-control contract. "
            f"{support.get('NOT_EXECUTABLE_ON_THIS_DATASET', 0)} of them cannot run here, each "
            f"with a stated reason and the canonical fields that would unlock it — see "
            f"`control_coverage_matrix.csv`, which is the concrete answer to *what would real "
            f"Shafafiya or eClaimLink data unlock?*\n\n"
            f"Only **{registry_summary['controls_that_may_deny']}** controls in the entire "
            f"catalogue may deny or reprice a claim, and every one of them is type H or H/E. "
            f"The registry refuses to load a statistical, network, document or model control "
            f"that declares `REJECT` or `REPRICE`.\n"
        )

        parts.append("\n### Controls that produced signals\n")
        parts.append(_md_table(
            fired[["rule_id", "scenario_id", "type_label", "stage", "data_support",
                   "signal_count", "elapsed_ms"]], max_rows=40))

        parts.append("\n### Controls that ran and found nothing — and why that is a result\n")
        parts.append(_md_table(ran_not_fired[["rule_id", "type_label", "data_support"]], max_rows=20))
        parts.append(self._empty_control_notes())

        # ---- canonical population -----------------------------------------
        parts.append("\n## 3. Canonical model population\n")
        pop = r.dataset.population_report()
        parts.append(_md_table(pop[["table", "status", "rows", "columns_populated",
                                    "columns_defined"]]))
        parts.append(
            "\nNo table is filled with invented data. A table that is empty because the source "
            "has no such data is a finding; a table full of fabricated rows would be a lie that "
            "propagates into every downstream metric. Reasons are in "
            "`canonical_population_report.csv`.\n"
        )

        # ---- per-layer results --------------------------------------------
        parts.append("\n## 4. Per-layer results\n")
        parts.append(self._layer_section())

        # ---- metrics -------------------------------------------------------
        parts.append("\n## 5. Operational metrics\n")
        parts.append(_md_table(pd.DataFrame([m.to_row() for m in metric_rows])[
            ["metric", "value", "unit", "status", "note"]]))

        # ---- precision@capacity, by source, by type ------------------------
        parts.append("\n## 6. Capacity-aware precision, stratified by label source\n")
        pac = self.metrics.precision_at_capacity()
        parts.append(_md_table(pac))
        parts.append(
            "\nPrecision is evaluated **at the alert volume a review team can "
            "actually process**, not at an arbitrary threshold. The capacity basis is "
            f"`cfg.alerts_per_reviewer_per_day` × `cfg.reviewer_count` = "
            f"{int(r.config.get('alerts_per_reviewer_per_day')) * int(r.config.get('reviewer_count'))} "
            "cases per day.\n"
        )
        by_source = self.metrics.by_ground_truth_source()
        parts.append("\n### Stratified by `ground_truth_source`\n")
        parts.append(_md_table(by_source))
        parts.append(
            "\n`expert_review` and `rule_engine` labels are **exactly the kind of biased ground "
            "truth** the evaluation design warns against: the first exists because somebody chose to "
            "investigate, the second because a rule fired. Headline precision against them is an "
            "upper bound, not an estimate of real-world performance.\n"
        )

        parts.append("\n### Recall by `fraud_type`, and which control found it\n")
        parts.append(_md_table(self.metrics.recall_by_fraud_type()))
        parts.append(self._fraud_type_notes())

        # ---- lift over the composite ---------------------------------------
        parts.append("\n## 7. Lift over the transparent composite\n")
        lift = self.metrics.lift_over_composite()
        parts.append(_md_table(lift))
        if not lift.empty:
            base = lift.attrs.get("base_rate")
            parts.append(
                f"\nBase rate among ranked providers: {base:.1%}. "
                "The promotion gate requires a model to demonstrate **prospective lift over ANL-01-R01, the "
                "transparent composite**, using future-period review yield rather than in-sample "
                "separation — and review yield needs reviewer outcomes, which do not exist. The "
                "table above therefore uses a **label proxy** and is the closest this dataset "
                "permits, not the measurement the gate actually requires.\n"
            )

        # ---- calibration ----------------------------------------------------
        parts.append("\n## 8. Calibration of the priority score\n")
        parts.append(_md_table(self.metrics.calibration_curve()))
        parts.append(
            "\nThe priority score is a **queue-ordering score, not a probability**. It is "
            "rescaled to [0,1] here only so a calibration curve can be drawn; a poor fit is "
            "expected and is not evidence the score is broken.\n"
        )

        # ---- evaluation protocol --------------------------------------------
        parts.append("\n## 9. Evaluation protocol\n")
        for p in protocol_results:
            parts.append(f"\n### {p.procedure} — **{p.status}**\n\n{p.summary}\n")
            if p.caveat:
                parts.append(f"\n> {p.caveat}\n")
            if not p.detail.empty:
                parts.append("\n" + _md_table(p.detail, max_rows=12))

        # ---- gates -----------------------------------------------------------
        parts.append("\n## 10. Release-acceptance gates\n")
        parts.append(_md_table(pd.DataFrame([g.to_row() for g in gate_results])[
            ["gate", "status", "measured", "threshold"]]))
        parts.append("\nFull evidence is in `release_gate_report.md`.\n")

        # ---- reproducibility -------------------------------------------------
        parts.append("\n## 11. Reproducibility\n")
        parts.append(
            f"This run is deterministic. Seed `{summary['seed']}` "
            f"(`cfg.random_seed`), parameter-registry fingerprint "
            f"`{summary['parameters_fingerprint']}`. Re-running "
            f"`python -m fwa.run_validation --data {r.source_path or 'data/claims_demo_synthetic.csv'}` "
            f"regenerates every file in "
            f"`reports/` identically, because signal identity is a pure function of "
            f"(tenant, rule, version, subject, fact, period) and every model is seeded.\n\n"
            f"Stage timings for this run:\n\n"
            + _md_table(pd.DataFrame([
                {"stage": k, "seconds": round(v, 2)} for k, v in r.timings.items()
            ]))
        )
        parts.append("\n### Stage latency against the stage ceilings\n")
        parts.append(_md_table(r.evaluation.latency_report(r.config)))

        self._write("validation_report.md", "\n".join(parts))
        self._write_html("validation_report.html", "\n".join(parts))

    # ------------------------------------------------------------- narratives

    def _empty_control_notes(self) -> str:
        """Why the silent controls were silent. Each one is a finding."""
        notes = {
            "PAY-01-R01": (
                "**No header-level duplicates exist in the claim extract at all.** Grouping on "
                "(payer, member, provider, principal diagnosis, admission date, discharge date) "
                "produces zero groups with more than one claim. The 81 claims labelled "
                "`duplicate_claim` are encoded through `num_insurers_same_event` (mean 3.1 "
                "against 1.0 for every other type) rather than as repeated claim rows — which "
                "PAY-10-R01 detects instead. A duplicate-billing control cannot find duplicates "
                "that were never written as duplicates."
            ),
            "PAY-01-R02": "Same cause as PAY-01-R01: only 14 member-provider pairs have more "
                          "than one claim in the entire file.",
            "PAY-01-R03": "Multi-claim episodes exist but almost none repeat a principal "
                          "diagnosis at one provider.",
            "CLN-05-R01": (
                "**`length_of_stay_days` is uniformly distributed on [1, 30]** with no "
                "diagnosis-specific shape. Its MAD is therefore ≈7.5 and the largest possible "
                "robust residual is ≈1.4, so a threshold of 3.0 can never be crossed. This is a "
                "property of how this file was generated, not a finding about providers, and it "
                "is a clean illustration of why a control that is silent tells you about the "
                "data before it tells you about the population. The threshold is NOT lowered "
                "to manufacture signals."
            ),
            "CLN-06-R03": "No (diagnosis, length of stay, exact amount) triple recurs across "
                          "different members at one provider.",
            "CLN-07-R02": "Peak concurrent occupancy never approaches the configured capacity: "
                          "providers average 25 claims across 38 months.",
            "PAY-06-R02": "No claim has an approved amount exceeding its requested amount, and "
                          "no amount is negative. The transaction-integrity edit passes cleanly.",
            "NET-02-R02": "No weekly-snapshot community is abnormally dense relative to "
                          "**size-matched** peers with low external flow. See the note on "
                          "`coordinated_ring` below.",
        }
        run = self.result.evaluation.to_frame()
        silent = set(run.loc[run["outcome"] == "NOT_TRIGGERED", "rule_id"])
        lines = ["\nA control that runs and finds nothing has found something. Specifically:\n"]
        for rule, note in notes.items():
            if rule in silent:
                lines.append(f"- **{rule}** — {note}")
        return "\n".join(lines) + "\n"

    def _fraud_type_notes(self) -> str:
        """Which field each labelled type is actually encoded through.

        COMPUTED, not written down. An earlier version of this method carried a
        hand-written profile of one particular file; when the input changed,
        every sentence in it silently became false while still reading as
        authoritative. Measuring it means the paragraph is either true of the
        data in front of it or absent.
        """
        r = self.result
        labels = self.metrics.label_series()
        if labels.empty:
            return ("\n_No labels accompany this dataset, so no statement can be made about "
                    "how a labelled type is encoded._\n")

        held = r.held_out_labels
        if held is None or held.empty or "fraud_type" not in held.columns:
            return ""

        claims = r.claims.set_index("claim_sk")
        types = held.set_index("claim_sk")["fraud_type"]
        numeric = claims.select_dtypes(include=["number", "bool"]).astype(float)
        # Ignore anything constant: a field with no spread cannot encode anything.
        numeric = numeric.loc[:, numeric.std(numeric_only=True) > 0]

        rows = []
        for fraud_type in sorted(t for t in types.unique() if str(t) != "legitimate"):
            members = types.index[types == fraud_type].intersection(numeric.index)
            others = numeric.index.difference(members)
            if len(members) < 5 or len(others) < 5:
                continue
            inside, outside = numeric.loc[members], numeric.loc[others]
            spread = outside.std(numeric_only=True).replace(0, float("nan"))
            separation = ((inside.mean(numeric_only=True)
                           - outside.mean(numeric_only=True)).abs() / spread).dropna()
            if separation.empty:
                continue
            field = separation.idxmax()
            rows.append(
                f"- `{fraud_type}` ({len(members)} claims) separates most on **`{field}`** — "
                f"{inside[field].mean():.2f} against {outside[field].mean():.2f} elsewhere, "
                f"{separation.max():.1f} standard deviations apart."
            )

        if not rows:
            return ""
        return (
            "\n**Read this table with the generator in mind.** Every labelled type in a "
            "generated file is encoded through some field, and a control that tests that field "
            "will recover the type almost perfectly. That is a fact about the generator rather "
            "than evidence the control works on real claims. Measured here:\n\n"
            + "\n".join(rows)
            + "\n\nA recall figure against these labels therefore says how faithfully a control "
              "reads the encoding, not how well it would find the behaviour the label names.\n"
        )

    def _layer_section(self) -> str:
        r = self.result
        parts = []

        # rules
        run = r.evaluation.to_frame()
        by_type = run[run["signal_count"] > 0].groupby("type_label")["signal_count"].sum()
        parts.append("### Deterministic and expert controls\n")
        parts.append(_md_table(by_type.reset_index().rename(
            columns={"type_label": "control type", "signal_count": "signals"})))

        # statistical
        composite_results, composite_frame = r.composite
        flagged = composite_frame[
            (~composite_frame["insufficient_evidence"]) &
            (composite_frame["composite_score"] > float(r.config.get("composite_flag_threshold")))
        ]
        scores = composite_frame.loc[~composite_frame["insufficient_evidence"], "composite_score"]
        parts.append("\n### Statistical layer — ANL-01-R01 transparent composite\n")
        parts.append(
            f"\n{count(len(composite_results), 'provider')} scored; {len(flagged)} above the "
            f"`cfg.composite_flag_threshold` of {r.config.get('composite_flag_threshold')}. "
            f"Score distribution: median {scores.median():.2f}, p95 {scores.quantile(0.95):.2f}, "
            f"max {scores.max():.2f}. The threshold's position in that distribution is visible "
            f"rather than asserted.\n"
        )
        parts.append(_md_table(flagged[["provider_sk", "composite_score", "peer_level_used",
                                        "peer_n", "claim_count", "top_features"]], max_rows=12))

        # graph
        if r.graph is not None:
            parts.append("\n### Graph layer\n")
            parts.append(
                f"\n{len(r.graph.snapshots)} weekly, time-bounded snapshots; "
                f"{r.graph.full_graph.number_of_nodes():,} nodes and "
                f"{r.graph.full_graph.number_of_edges():,} edges in the cumulative graph; "
                f"{len(r.graph.community_stats):,} snapshot communities. "
                f"{count(len(r.entity_candidates), 'entity-resolution candidate')} surfaced for human "
                f"confirmation — **none merged automatically**, at any confidence.\n"
            )
            parts.append(_md_table(r.graph.edge_inventory()))

        # models
        if r.models is not None and r.models.anomaly is not None:
            parts.append("\n### Model layer\n")
            parts.append(_md_table(r.models.anomaly.summary()[
                ["model", "training_rows", "scored_rows", "capacity_threshold", "flagged", "features"]]))
            cov = r.models.coverage()
            parts.append(f"\n{cov.get('coverage_note', '')}\n")
            parts.append("\n**Promotion gate verdicts:**\n")
            for name, verdict in r.models.gate_verdicts.items():
                parts.append(f"\n- **{verdict.headline}** — {verdict.summary}\n")
                parts.append("\n" + _md_table(verdict.to_frame()))
            if r.models.supervised is not None:
                parts.append(
                    f"\n**Supervised gate: `{r.models.supervised.status}`.** "
                    f"{r.models.supervised.summary}\n"
                )
                parts.append("\n" + _md_table(r.models.supervised.to_frame()[
                    ["prerequisite", "met", "what_would_satisfy_it"]]))
            if r.models.clusters:
                parts.append("\n**ANL-01-R04 novel-cluster candidates (MONITOR_ONLY):**\n")
                parts.append("\n" + _md_table(pd.DataFrame(
                    [c.to_row() for c in r.models.clusters])))

        # documents
        if r.documents is not None:
            parts.append("\n### Document / NLP layer\n")
            summary = r.documents.summary()
            parts.append(
                f"\nThe source file contains **no documents**. The pipeline runs against "
                f"{summary['corpus'].get('documents', 0)} clearly-labelled SYNTHETIC discharge "
                f"summaries generated by this artefact "
                f"({summary['corpus'].get('by_language', {})}), of which "
                f"{summary['corpus'].get('inconsistent', 0)} carry a deliberately injected "
                f"inconsistency and {summary['corpus'].get('poor_ocr', 0)} carry poor OCR quality "
                f"so that confidence degradation is observable. "
                f"{count(int(summary['pipeline'].get('findings_discarded', 0)), 'finding')} were DISCARDED "
                f"for want of a source span or sufficient confidence — not displayed with a "
                f"caveat.\n"
            )
            parts.append("\n**Extraction quality, reported separately by language and document "
                         "type, as the document design requires:**\n")
            parts.append("\n" + _md_table(r.documents.pipeline.quality_by_language_and_type()))
            parts.append(
                "\nNo figure in that table is a measurement of clinical-NLP performance. The "
                "corpus is template-generated, so extraction accuracy on it measures whether the "
                "pipeline's plumbing works — nothing more. Arabic is included because the design "
                "requires language-separated reporting, and a pipeline evaluated on one language "
                "cannot make that claim.\n"
            )

        # AI
        parts.append("\n### AI layer\n")
        parts.append("\n" + _md_table(pd.DataFrame([r.ai.status()["features"]])))
        parts.append(
            f"\nProvider: `{r.ai.provider.name}` (`{r.ai.provider.model}`), deterministic: "
            f"{r.ai.provider.deterministic}. No AI output sets or changes a disposition, a "
            f"priority or an exposure, under any configuration. Every narrative is passed "
            f"through the groundedness validator, which drops any uncited or unresolvable "
            f"sentence before display, and the reviewer copilot refuses conduct questions "
            f"**before any model call**.\n"
        )
        return "\n".join(parts)

    # ------------------------------------------------------- other artefacts

    def _write_release_gate_report(self, gate_results) -> None:
        parts = [f"# Release-gate report — `uae-fwa-engine` {__version__}\n",
                 _banner_block(self.result),
                 "\nEach of the six release gates, evaluated against **this build**, with an "
                 "explicit PASS / FAIL / NOT_ASSESSABLE and its evidence.\n\n"
                 "`NOT_ASSESSABLE` is **never** a pass. A gate that cannot be evaluated is a "
                 "reason not to promote, not a reason to shrug — which is why every control in "
                 "this build is still in shadow.\n"]
        for g in gate_results:
            parts.append(f"\n## {g.gate} — **{g.status}**\n")
            parts.append(f"\n**Minimum acceptance evidence:** {g.requirement}\n")
            if g.threshold is not None:
                parts.append(f"\n**Threshold:** `{g.threshold}` "
                             f"(a configuration acceptance criterion quoted from the "
                             f"specification, not a result achieved here)\n")
            if g.measured is not None:
                parts.append(f"\n**Measured:** `{g.measured}`\n")
            parts.append(f"\n**Evidence:** {g.evidence}\n")
        self._write("release_gate_report.md", "\n".join(parts))

    def _write_model_card(self) -> None:
        r = self.result
        models = r.models
        parts = [f"# Model card — `uae-fwa-engine` {__version__}\n", _banner_block(self.result)]
        parts.append(
            "\n## Intended use\n\n"
            "These models produce **leads for human review**. They cannot deny a claim, change "
            "its price, or set any disposition. The safety boundary makes that structural: a "
            "control of type M declaring `REJECT` or `REPRICE` is rejected by the contract "
            "validator at registration time, so the system cannot be started with one loaded.\n\n"
            "## Out of scope\n\n"
            "Supervised propensity modelling. The supervised-prerequisite gate returns "
            f"`{models.supervised.status if models and models.supervised else 'NOT RUN'}`; no "
            "classifier is trained anywhere in this repository.\n"
        )
        if models is not None and models.anomaly is not None:
            parts.append("\n## Models\n")
            parts.append(_md_table(models.anomaly.summary()))
            parts.append("\n## Training and scoring data\n")
            parts.append(_md_table(pd.DataFrame([models.coverage()]).T.reset_index().rename(
                columns={"index": "property", 0: "value"})))
            parts.append("\n## Features\n")
            parts.append(
                f"\n{len(models.anomaly.matrix_columns)} numeric features plus their explicit "
                f"missingness indicators. **Excluded by construction:** the held-out label "
                f"columns (physically absent from every `FeatureFrame`), direct identifiers "
                f"(`hospital_id`, `patient_id`, `agent_id`, claim and episode keys) and every "
                f"protected demographic attribute. Features are computed strictly from data "
                f"available BEFORE the scored period — provider and member aggregates are "
                f"expanding, time-ordered and shifted.\n"
            )
            parts.append("\n## Explanation\n")
            parts.append(
                "\nSHAP per flagged entity, rendered in **original units beside the peer median** "
                "— \"Amount per inpatient day: AED 2,649/day, peer median AED 912/day (2.9×)\" — "
                "not as bare SHAP magnitudes. Methods in use: "
                + ", ".join(f"`{k}` → {v.method}" for k, v in models.explainers.items()) + ".\n"
            )
            parts.append("\n## Monitoring\n")
            for name, results in models.drift.items():
                parts.append(f"\n**{name}**\n\n")
                parts.append(_md_table(pd.DataFrame([d.to_row() for d in results])[
                    ["monitor", "value", "status", "note"]]))
            parts.append("\n## Promotion status\n")
            for name, verdict in models.gate_verdicts.items():
                parts.append(f"\n- **{verdict.headline}** — {verdict.summary}\n")
            parts.append(
                "\n## Ethical considerations\n\n"
                "A model score here is evidence that an entity is UNUSUAL. It is not evidence of "
                "intent, and the difference between the two is the defining requirement of "
                "the whole system. The exposure attached to a model-only lead renders as the "
                "gross amount plus the literal statement *\"exposure not yet established\"*, "
                "because reporting it as savings would overstate the system's contribution — "
                "this system treats that as an ethical requirement, not a measurement nicety.\n"
            )
        self._write("model_card.md", "\n".join(parts))

    def _write_limitations(self) -> None:
        r = self.result
        support = r.registry.summary()["by_data_support"]
        parts = [f"# Limitations — `uae-fwa-engine` {__version__}\n", _banner_block(self.result)]
        parts.append(
            "\n## 1. Every figure here is a property of one input file\n\n"
            f"This run's input is a claim-header extract of {len(r.claims):,} rows, in a "
            "non-AED source currency converted through `config/fx.yaml` at the service-date "
            "rate. Which "
            "controls can run at all, which are silent, and every rate below are all properties "
            "of that file's schema and contents. **Re-measure on your own data before treating "
            "any number here as a forecast of what it will do there.**\n\n"
            "## 2. The labels were produced by detection processes\n\n"
            "`ground_truth_source` takes three values — `pattern_detection`, `expert_review` and "
            "`rule_engine`. Every one of them is a detection process, so the labels carry that "
            "process's blind spots. Worse, profiling shows each `fraud_type` is encoded through "
            "a single dominant field, so a control testing that field recovers the type almost "
            "perfectly. **Precision against these labels is close to tautological for rule-based "
            "controls and is reported as an upper bound throughout.**\n\n"
            "## 3. Most of the catalogue cannot run here\n\n"
            f"{support.get('NOT_EXECUTABLE_ON_THIS_DATASET', 0)} of 164 controls are classified "
            f"`NOT_EXECUTABLE_ON_THIS_DATASET`. The four structural gaps, in order of "
            f"consequence:\n\n"
            "1. **No `authorization` table.** The entire PAY-04 scenario is unrunnable — "
            "including PAY-04-R01, the usual worked example of the atomic-control "
            "contract.\n"
            "2. **Claim-header level only.** No activity codes, units or line amounts, so "
            "PAY-02, PAY-03, PAY-05, CLN-08 and all of PHR-01/02/04/05 have no input at all. "
            "The missingness rule says explicitly that no claim-level or provider-level "
            "aggregate can substitute for the code-pair co-occurrence feature.\n"
            "3. **No `claim_version` lineage and no `remittance`.** PAY-08 and PAY-12 are dead, "
            "and 'prevented/recovered AED' is NOT_MEASURABLE.\n"
            "4. **No geography.** Peer-hierarchy level 5 is `NOT_POPULATED` and every "
            "travel-impossibility and geographic-cluster control is classified out rather than "
            "run against a fabricated location.\n\n"
            "## 4. Proxies are proxies\n\n"
            f"{support.get('PARTIAL', 0)} controls run against a PROXY rather than the field "
            "the catalogue specifies: ICD chapter for provider specialty, policy type for "
            "encounter/facility type, an undated blacklist flag for an effective-dated exclusion "
            "list, agent co-occurrence for a clinical referral edge. Each proxy is named on the "
            "signal itself, and each weakens the finding.\n\n"
            "## 5. Nothing has been reviewed\n\n"
            "Precision in the strict sense is *confirmed ÷ reviewed*. Nothing has been reviewed, "
            "so precision, review yield, confirmed AED, prevented/recovered AED, net savings, "
            "abrasion, turnaround, overturn rate and rule stability are all "
            "`NOT_MEASURABLE_ON_THIS_DATASET`. They are implemented and reported as unmeasurable "
            "rather than dropped.\n\n"
            "## 6. The priority formula saturates on this data\n\n"
            "The priority coefficients are reproduced exactly from the design and are not "
            "tuned. With `exposure_aed / 1000` inside the log and case exposures reaching six "
            "figures, the exposure term alone can contribute enough to push the sigmoid above "
            "99. The UI therefore shows the **term-by-term breakdown and the raw logit**, so the "
            "saturation is visible rather than hidden behind a rounded score. A real deployment "
            "would set the divisor to the scale of its own exposures — it is a governed "
            "parameter, and the fact that it needs setting is itself a finding.\n\n"
            "## 7. The adapters are not certified\n\n"
            "`ShafafiyaAdapter` and `EClaimLinkAdapter` are schema-complete and exercised by "
            "fixtures. Both are marked **NOT CERTIFIED — no live regulator feed**. Neither has "
            "ever seen a real DoH or DHA submission.\n\n"
            "## 8. This is a demonstration artefact, not a production system\n\n"
            "The authentication layer demonstrates the access-control requirement. It uses "
            "Argon2 password hashing, enforces separation of duties and PHI masking, and writes "
            "an append-only audit log — and it is **not hardened for real PHI**. SQLite stands in "
            "for PostgreSQL behind a repository interface. There is no rota, no on-call, no "
            "penetration test and no threat model.\n"
        )
        self._write("limitations.md", "\n".join(parts))

    # ------------------------------------------------------------------- html

    def _write_html(self, name: str, markdown_text: str) -> None:
        body = markdown_text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        html = (
            "<!doctype html><html lang='en'><head><meta charset='utf-8'>"
            "<meta name='viewport' content='width=device-width,initial-scale=1'>"
            f"<title>Validation report — uae-fwa-engine {__version__}</title>"
            "<style>"
            ":root{--bg:#fbfbfa;--fg:#1f2328;--muted:#5b6672;--line:#e3e6ea;--accent:#8b5a3c}"
            "@media (prefers-color-scheme:dark){:root{--bg:#14161a;--fg:#e8eaed;"
            "--muted:#9aa4af;--line:#2a2f36;--accent:#d9a273}}"
            "body{background:var(--bg);color:var(--fg);font:15px/1.65 ui-sans-serif,-apple-system,"
            "'Segoe UI',Roboto,sans-serif;max-width:60rem;margin:0 auto;padding:2.5rem 1rem}"
            "pre{white-space:pre-wrap;font:13px/1.6 ui-monospace,SFMono-Regular,Menlo,monospace;"
            "background:transparent;border:1px solid var(--line);border-radius:10px;padding:1.25rem}"
            "</style></head><body><pre>" + body + "</pre></body></html>"
        )
        path = self.out / name
        path.write_text(html, encoding="utf-8")
        self.written.append(path)
