"""The orchestration module (manuscript §4.13 epic sequencing, §5.2).

    "Prefect/Airflow/MLflow/Docker: provide a THIN orchestration module and a
     ``Dockerfile``, but the tool must run with ``pip install -e . && streamlit
     run app/Home.py`` and **no API key and no network access**." — build brief §2

This is that thin module. It runs the epics in the order §4.13 specifies, and
the order is a design decision rather than project management: nothing can be
built on an unpopulated canonical model, the statistical composite must exist
before the models because it is their benchmark, and the AI layer comes last
because it consumes everything else and produces nothing the rest depends on.

The result is a single immutable :class:`PipelineResult` that the Streamlit app
caches and the validation report reads. One object, one source of truth — the
UI and the report cannot disagree about what the run found, because they are
looking at the same run.
"""

from __future__ import annotations

import datetime as _dt
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .ai import AILayer
from .audit.log import AuditEventType, AuditLog
from .canonical import CanonicalDataset, GenericIndiaTpaAdapter, ImmutableRawStore, get_adapter
from .cases import CaseCorrelator, PriorityScorer
from .config import AppConfig, load_config
from .engine.context import ControlContext
from .engine.evaluator import Evaluator, EvaluationReport
from .engine.registry import RuleRegistry
from .engine.signals import SignalStore
from .features.store import FeatureStore
from .graph import EntityResolver, GraphService
from .lineage.episodes import EpisodeBuilder, LineageIndex
from .models import ModelLayer
from .nlp import DocumentLayer, DocumentPipeline, SyntheticCorpus
from .statistical import PeerService, TransparentComposite

__all__ = ["PipelineResult", "run_pipeline", "EPICS"]

#: The §4.13 epic sequence. Printed after each stage so a run's progress maps
#: onto the manuscript's own build order.
EPICS = (
    ("Epic 0", "Foundations — raw store, canonical model, reference service, lineage, registry"),
    ("Epic 1", "P0 hard/expert edits — feature store and the executable PAY/CLN/ENT/PHR/POL controls"),
    ("Epic 2", "P1 provider analytics — peers, shrinkage, change detection, transparent composite"),
    ("Epic 3", "Networks — typed graph, entity resolution, NET scenarios, POL-01"),
    ("Epic 4", "Mature models and AI — IF/LOF, SHAP, drift, gates, clusters, NLP, then the AI layer"),
)


@dataclass
class PipelineResult:
    config: AppConfig
    dataset: CanonicalDataset
    raw_store: ImmutableRawStore
    adapter_name: str
    claims: pd.DataFrame
    features: Any
    feature_store: FeatureStore
    episodes: pd.DataFrame
    episode_map: dict[str, str]
    lineage: LineageIndex
    registry: RuleRegistry
    signals: SignalStore
    evaluation: EvaluationReport
    composite: tuple
    composite_frame: pd.DataFrame
    provider_metrics: pd.DataFrame
    graph: GraphService | None
    entity_candidates: list
    models: ModelLayer | None
    documents: DocumentLayer | None
    ai: AILayer
    cases: CaseCorrelator
    audit: AuditLog
    context: ControlContext
    timings: dict[str, float] = field(default_factory=dict)
    run_started: _dt.datetime = field(default_factory=lambda: _dt.datetime.now(_dt.timezone.utc))
    tenant_id: str = "T001"
    held_out_labels: pd.DataFrame | None = None
    seed: int = 0
    #: The input file this run describes. Carried on the result because every
    #: report and every page is a statement about one file, and a figure whose
    #: source cannot be named is a figure that cannot be checked.
    source_name: str = ""
    source_path: str = ""

    # ---- convenience views used by the app and the reports -----------------

    def queue(self, tenant_id: str | None = None, limit: int | None = None) -> pd.DataFrame:
        return self.cases.queue(tenant_id or self.tenant_id, limit)

    def signals_for_case(self, case_id: str) -> list:
        case = self.cases.cases.get(case_id)
        if case is None:
            return []
        return [s for s in (self.signals.get(sid) for sid in case.signal_ids) if s is not None]

    def summary(self) -> dict[str, Any]:
        return {
            "run_started": self.run_started.isoformat(),
            "adapter": self.adapter_name,
            "claims": len(self.claims),
            "raw_records": len(self.raw_store),
            "controls": len(self.registry),
            "controls_executed": self.evaluation.summary()["controls_that_ran"],
            "signals": len(self.signals),
            "cases": len(self.cases.cases),
            "parameters_fingerprint": self.config.fingerprint(),
            "seed": self.seed,
            "timings_seconds": {k: round(v, 2) for k, v in self.timings.items()},
        }


def run_pipeline(
    data_path: str | Path = "data/claims_demo_synthetic.csv",
    *,
    config: AppConfig | None = None,
    tenant_id: str = "T001",
    adapter: str = "generic_india_tpa",
    generate_documents: int | None = 800,
    enable_ai: bool | None = None,
    enable_experimental: bool = False,
    document_dir: str | Path = "data/synthetic_documents",
    verbose: bool = True,
) -> PipelineResult:
    """Run the whole thing, in §4.13 epic order."""
    config = config or load_config()
    seed = int(config.get("random_seed"))
    np.random.seed(seed)
    audit = AuditLog()
    timings: dict[str, float] = {}

    def _step(label: str):
        started = time.perf_counter()

        class _Ctx:
            def __enter__(self_inner):
                if verbose:
                    print(f"  … {label}", flush=True)
                return self_inner

            def __exit__(self_inner, *exc):
                timings[label] = time.perf_counter() - started
                if verbose:
                    print(f"  ✓ {label} ({timings[label]:.1f}s)", flush=True)
                return False

        return _Ctx()

    if verbose:
        print(f"\n{EPICS[0][0]}: {EPICS[0][1]}")

    # ---------------------------------------------------- Epic 0: foundations
    with _step("adapter and canonical model"):
        raw_store = ImmutableRawStore()
        adapter_cls = get_adapter(adapter)
        source_adapter = adapter_cls(config, raw_store, tenant_id=tenant_id)
        dataset = source_adapter.load(str(data_path))
        claims = dataset["claim_header"]
        held_out = (
            source_adapter.held_out_labels(str(data_path))
            if isinstance(source_adapter, GenericIndiaTpaAdapter) else None
        )

    with _step("lineage and episodes"):
        lineage = LineageIndex.from_frame(dataset.get("claim_version"))
        episodes, episode_map = EpisodeBuilder(config.get("readmit_window_days")).build(claims)

    with _step("rule registry"):
        registry = RuleRegistry.from_directory()

    if verbose:
        print(f"\n{EPICS[1][0]}: {EPICS[1][1]}")
    with _step("feature store"):
        feature_store = FeatureStore(config)
        feature_store.build(dataset, episode_map)
        features = feature_store.combined_claim_features()
        features.assert_no_labels()

    if verbose:
        print(f"\n{EPICS[2][0]}: {EPICS[2][1]}")
    with _step("peers, shrinkage and the transparent composite"):
        composite_engine = TransparentComposite(config)
        provider_metrics = composite_engine.provider_metrics(claims)
        composite = composite_engine.compute(claims)
        peers = PeerService(config, claims)
        provider_peers = PeerService(config, provider_metrics)

    if verbose:
        print(f"\n{EPICS[3][0]}: {EPICS[3][1]}")
    with _step("graph and entity resolution"):
        graph = GraphService(config, seed=seed).build(claims)
        resolver = EntityResolver(config)
        entity_candidates = resolver.candidates(claims)

    if verbose:
        print(f"\n{EPICS[4][0]}: {EPICS[4][1]}")
    documents: DocumentLayer | None = None
    if generate_documents:
        with _step("synthetic document corpus and NLP pipeline"):
            corpus = SyntheticCorpus(config, output_dir=document_dir)
            corpus.generate(claims, n=generate_documents, write=True)
            doc_pipeline = DocumentPipeline(config)
            doc_pipeline.run(corpus)
            documents = DocumentLayer(corpus=corpus, pipeline=doc_pipeline)

    with _step("unsupervised models, SHAP, drift, clusters"):
        models = ModelLayer.build(
            config, claims, features, composite[0], enable_experimental=enable_experimental
        )

    with _step("AI layer"):
        if enable_ai is not None:
            ai = AILayer.build(config, registry=registry, audit=audit) if enable_ai else \
                AILayer(config=config, provider=AILayer.build(config).provider, enabled=False)
        else:
            ai = AILayer.build(config, registry=registry, audit=audit)

    # ------------------------------------------------ evaluate the catalogue
    context = ControlContext(
        config=config,
        tenant_id=tenant_id,
        claims=claims,
        features=features,
        dataset=dataset,
        peers=peers,
        provider_peers=provider_peers,
        provider_metrics=provider_metrics,
        episodes=episodes,
        episode_map=episode_map,
        lineage=lineage,
        graph=graph,
        documents=documents,
        models=models,
        composite=composite,
        audit=audit,
        run_date=_dt.date.today(),
        reference_versions={
            "peer_config": config.peer_groups.icd_chapter_map.get("version", "1.0.0"),
            "fx_table": "config/fx.yaml",
        },
    )

    with _step("control evaluation"):
        signals = SignalStore()
        evaluator = Evaluator(registry, signals, audit=audit)
        evaluation = evaluator.run(context)

    with _step("case correlation and priority"):
        correlator = CaseCorrelator(config, PriorityScorer(config), audit=audit)
        correlator.correlate(signals, registry=registry)

    audit.record(
        AuditEventType.PIPELINE_RUN,
        actor="system", actor_role="ENGINE", tenant_id=tenant_id, subject="pipeline",
        reason="Full pipeline run completed.",
        after={"claims": len(claims), "signals": len(signals), "cases": len(correlator.cases)},
    )

    result = PipelineResult(
        config=config, dataset=dataset, raw_store=raw_store, adapter_name=source_adapter.source_system,
        claims=claims, features=features, feature_store=feature_store,
        episodes=episodes, episode_map=episode_map, lineage=lineage,
        registry=registry, signals=signals, evaluation=evaluation,
        composite=composite, composite_frame=composite[1], provider_metrics=provider_metrics,
        graph=graph, entity_candidates=entity_candidates, models=models, documents=documents,
        ai=ai, cases=correlator, audit=audit, context=context, timings=timings,
        tenant_id=tenant_id, held_out_labels=held_out, seed=seed,
        source_name=Path(str(data_path)).name, source_path=str(data_path),
    )
    if verbose:
        s = result.summary()
        print(
            f"\nRun complete: {s['signals']:,} signal(s) from "
            f"{s['controls_executed']} executed control(s) of {s['controls']}, "
            f"correlated into {s['cases']:,} case(s). "
            f"Parameter fingerprint {s['parameters_fingerprint']}, seed {s['seed']}."
        )
    return result
