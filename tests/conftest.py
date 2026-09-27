"""Shared fixtures.

The suite runs against a **sample of the real file**, not a hand-made toy
frame, for a specific reason: a fixture built to make a control fire tells you
the control fires on the fixture. A sample of ``claims.csv`` that happens to
contain the triggering pattern tells you the control fires on the data, and the
ten-category suite then asserts the *governance* properties (idempotency,
tenant isolation, effective dating, evidence snapshot) on that real behaviour.

The sample is deterministic (fixed seed) and stratified so that every
implemented control has something to find. It is small enough that the whole
suite runs in well under a minute.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fwa.audit.log import AuditLog  # noqa: E402
from fwa.canonical import GenericIndiaTpaAdapter, ImmutableRawStore  # noqa: E402
from fwa.config import load_config  # noqa: E402
from fwa.engine.context import ControlContext  # noqa: E402
from fwa.engine.registry import RuleRegistry  # noqa: E402
from fwa.features.store import FeatureStore  # noqa: E402
from fwa.graph import GraphService  # noqa: E402
from fwa.lineage.episodes import EpisodeBuilder, LineageIndex  # noqa: E402
from fwa.statistical import PeerService, TransparentComposite  # noqa: E402

SAMPLE_SIZE = 1500


@pytest.fixture(scope="session")
def project_root() -> Path:
    return ROOT


@pytest.fixture(scope="session")
def config():
    return load_config(ROOT / "config")


@pytest.fixture(scope="session")
def registry() -> RuleRegistry:
    return RuleRegistry.from_directory(ROOT / "rules")


@pytest.fixture(scope="session")
def raw_claims() -> pd.DataFrame:
    return pd.read_csv(ROOT / "data" / "claims.csv")


def _short_repeat_stratum(raw: pd.DataFrame, interval_days: int) -> pd.DataFrame:
    """Every claim belonging to a same-patient, same-diagnosis pair repeated inside the interval.

    Four such pairs exist in the whole file. A uniform random sample of 1,500
    rows almost never contains both halves of one, so CLN-04-R01 would be silent
    on the fixture for a reason that has nothing to do with the control. Rather
    than adding it to the documented-silent allowlist — which would say
    something false about the dataset — the stratum pulls the pairs in so the
    positive-fixture category tests the control it names.
    """
    work = raw[["claim_id", "patient_id", "diagnosis_primary", "date_of_claim"]].copy()
    # dayfirst=True, matching the adapter (build brief §3) — parsing these the
    # other way round would select a different, wrong set of pairs.
    work["_d"] = pd.to_datetime(work["date_of_claim"], dayfirst=True, errors="coerce")
    work = work.sort_values("_d")
    keep: list[str] = []
    for _, group in work.groupby(["patient_id", "diagnosis_primary"], sort=False):
        if len(group) < 2:
            continue
        gaps = group["_d"].diff().dt.days
        hits = gaps[(gaps >= 0) & (gaps <= interval_days)]
        for idx in hits.index:
            position = group.index.get_loc(idx)
            keep.extend(group.iloc[max(0, position - 1): position + 1]["claim_id"].tolist())
    return raw[raw["claim_id"].isin(set(keep))]


@pytest.fixture(scope="session")
def sample_claims(raw_claims, config) -> pd.DataFrame:
    """A deterministic, stratified sample that exercises every implemented control."""
    seed = int(config.get("random_seed"))
    parts = [
        _short_repeat_stratum(raw_claims, int(config.get("min_repeat_interval_days"))),
        raw_claims[raw_claims["provider_blacklist_flag"]].head(220),
        raw_claims[~raw_claims["icd_code_matches_procedure"]].head(200),
        raw_claims[raw_claims["pharmacy_bill_ratio"] > 0.7].head(120),
        raw_claims[raw_claims["discharge_readmit_gap_days"] < 5].head(120),
        raw_claims[raw_claims["num_insurers_same_event"] > 1].head(120),
        raw_claims[raw_claims["days_since_policy_start"] < 90].head(60),
        raw_claims.sample(SAMPLE_SIZE, random_state=seed),
    ]
    frame = pd.concat(parts).drop_duplicates(subset=["claim_id"]).reset_index(drop=True)
    return frame


@pytest.fixture(scope="session")
def dataset(sample_claims, config):
    adapter = GenericIndiaTpaAdapter(config, ImmutableRawStore(), tenant_id="T001")
    return adapter.load(sample_claims)


@pytest.fixture(scope="session")
def labels(sample_claims, config) -> pd.DataFrame:
    adapter = GenericIndiaTpaAdapter(config, ImmutableRawStore())
    return adapter.held_out_labels(sample_claims)


@pytest.fixture(scope="session")
def claims(dataset) -> pd.DataFrame:
    return dataset["claim_header"]


@pytest.fixture(scope="session")
def context(config, dataset, claims) -> ControlContext:
    """A control context over the sample, with every layer the controls need."""
    episodes, episode_map = EpisodeBuilder(config.get("readmit_window_days")).build(claims)
    store = FeatureStore(config)
    store.build(dataset, episode_map)
    composite_engine = TransparentComposite(config)
    provider_metrics = composite_engine.provider_metrics(claims)
    composite = composite_engine.compute(claims)
    graph = GraphService(config, seed=int(config.get("random_seed"))).build(claims)
    return ControlContext(
        config=config,
        tenant_id="T001",
        claims=claims,
        features=store.combined_claim_features(),
        dataset=dataset,
        peers=PeerService(config, claims),
        provider_peers=PeerService(config, provider_metrics),
        provider_metrics=provider_metrics,
        episodes=episodes,
        episode_map=episode_map,
        lineage=LineageIndex(),
        graph=graph,
        composite=composite,
        audit=AuditLog(),
        run_date=pd.Timestamp("2026-09-20").date(),
    )


@pytest.fixture(scope="session")
def implemented_controls(registry):
    """Every control this build actually runs — the ten-category suite's subject."""
    return [c for c in registry.all() if c.implementation]


@pytest.fixture()
def audit() -> AuditLog:
    return AuditLog()


@pytest.fixture(scope="session")
def evaluated(registry, context):
    """One full catalogue run over the sample: the report and the signal store.

    Session-scoped because the run is the expensive part and every test that
    needs signals wants the *same* signals — several of the governance
    properties (idempotency, AI non-interference) are statements about a run
    being repeatable, which is only meaningful against one shared baseline.
    """
    from fwa.engine.evaluator import Evaluator
    from fwa.engine.signals import SignalStore

    store = SignalStore()
    evaluator = Evaluator(registry, store, audit=AuditLog())
    report = evaluator.run(context)
    return report, store


@pytest.fixture(scope="session")
def sample_signals(evaluated):
    _, store = evaluated
    return list(store)


@pytest.fixture(scope="session")
def sample_cases(config, registry, sample_signals):
    from fwa.cases.correlate import CaseCorrelator
    from fwa.cases.priority import PriorityScorer

    correlator = CaseCorrelator(config, PriorityScorer(config), audit=AuditLog())
    return correlator, correlator.correlate(sample_signals, registry)
