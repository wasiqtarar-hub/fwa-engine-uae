"""The model layer never crashes; it degrades and says why (usability brief, Phase 6).

Every case here is a file the tool can realistically be handed — one claim, a
handful of claims, one hospital, one date, a column that is always empty — and
every case asserts the same three things:

* no exception escapes ``ModelLayer.build`` / ``run_gate`` / the helpers;
* ``layer.messages`` carries a plain sentence naming the real numbers;
* the layer is in a sensible state (no models, or the one that could not be
  trained is skipped with its reason in ``anomaly.skipped``).

Frames are built the way the pipeline builds them: the generic adapter over
rows of the shipped ``claims.csv``, then episodes, the feature store and the
transparent composite.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
import pytest

from fwa.canonical import GenericIndiaTpaAdapter, ImmutableRawStore
from fwa.features.store import FeatureStore
from fwa.lineage.episodes import EpisodeBuilder
from fwa.models import ModelLayer, NovelClusterDiscovery
from fwa.models.anomaly import MIN_PROVIDERS, _bucket
from fwa.models.explain import ShapExplainer
from fwa.statistical import TransparentComposite


class _Override:
    """A config that answers some keys differently and delegates the rest."""

    def __init__(self, config, **overrides) -> None:
        self._config = config
        self._overrides = overrides

    def get(self, key, *args, **kwargs):
        if key in self._overrides:
            return self._overrides[key]
        return self._config.get(key, *args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._config, name)


def _inputs(config, raw: pd.DataFrame):
    dataset = GenericIndiaTpaAdapter(config, ImmutableRawStore(), tenant_id="T001").load(
        raw.reset_index(drop=True)
    )
    claims = dataset["claim_header"]
    _, episode_map = EpisodeBuilder(config.get("readmit_window_days")).build(claims)
    store = FeatureStore(config)
    store.build(dataset, episode_map)
    features = store.combined_claim_features()
    composite = TransparentComposite(config).compute(claims)
    return claims, features, composite


def _build(config, raw: pd.DataFrame, *, features_hook=None) -> ModelLayer:
    claims, features, composite = _inputs(config, raw)
    if features_hook is not None:
        features_hook(features)
    layer = ModelLayer.build(config, claims, features, composite[0])
    # the rest of the surface the pages touch must not raise either
    layer.run_gate()
    layer.provider_scores()
    layer.summary()
    layer.coverage()
    for name in list(layer.explainers):
        layer.explanation_sample(name, n=2)
    return layer


def _assert_plain(layer: ModelLayer) -> None:
    assert layer.messages, "a degraded layer must say why, in plain language"
    for m in layer.messages:
        assert isinstance(m, str) and m.endswith((".", ")")), m
        assert "Traceback" not in m and "Error" not in m, m


@pytest.fixture(scope="module")
def healthy(config, raw_claims):
    """A small but trainable file: 700 claims, both models fit."""
    return _build(config, raw_claims.sample(700, random_state=11))


# ------------------------------------------------------------------ tiny files


def test_one_claim(config, raw_claims):
    layer = _build(config, raw_claims.head(1))
    assert layer.anomaly.models == {}
    _assert_plain(layer)
    assert any(f"At least {MIN_PROVIDERS} are needed (this file has 1)" in m for m in layer.messages)
    assert "all" in layer.anomaly.skipped


@pytest.mark.parametrize("n", [5, 30])
def test_a_handful_of_claims_skips_lof_and_says_how_many_are_needed(config, raw_claims, n):
    layer = _build(config, raw_claims.sample(n, random_state=3))
    k = int(config.get("lof_n_neighbors"))
    assert "local_outlier_factor" not in layer.anomaly.models
    _assert_plain(layer)
    if layer.anomaly.models:  # Isolation Forest can still fit on a few rows
        assert "local_outlier_factor" in layer.anomaly.skipped
        assert any(f"at least {k + 1} training claims" in m for m in layer.messages)
    else:
        assert layer.anomaly.skipped


def test_pipeline_on_a_five_claim_file_does_not_crash(raw_claims, tmp_path):
    from fwa.pipeline import run_pipeline

    path = tmp_path / "five.csv"
    raw_claims.head(5).to_csv(path, index=False)
    result = run_pipeline(path, generate_documents=None, enable_ai=False, verbose=False,
                          document_dir=tmp_path / "docs")
    assert result.models is not None
    assert result.models.messages


# -------------------------------------------------------------- too few providers


def test_one_provider_cannot_be_held_out(config, raw_claims):
    top = raw_claims["hospital_id"].value_counts().index[0]
    layer = _build(config, raw_claims[raw_claims["hospital_id"] == top])
    assert layer.anomaly.models == {}
    _assert_plain(layer)
    assert any("Not enough hospitals in this file to train the model fairly. At least 2 are "
               "needed (this file has 1)" in m for m in layer.messages)


def test_two_providers_on_the_same_side_of_the_hold_out(config, raw_claims):
    counts = raw_claims["hospital_id"].value_counts()
    same = [h for h in counts.index if _bucket(str(h), "provider-split") == 0][:2]
    layer = _build(config, raw_claims[raw_claims["hospital_id"].isin(same)])
    assert layer.anomaly.models == {}
    _assert_plain(layer)
    assert any("Not enough hospitals" in m and "2 to learn from and 0 to check" in m
               for m in layer.messages)


# ------------------------------------------------ n_neighbors > training samples


def test_lof_neighbourhood_larger_than_the_training_set(config, raw_claims):
    cfg = _Override(config, lof_n_neighbors=100_000)
    claims, features, composite = _inputs(cfg, raw_claims.sample(500, random_state=5))
    layer = ModelLayer.build(cfg, claims, features, composite[0])
    layer.run_gate()
    assert "isolation_forest" in layer.anomaly.models
    assert "local_outlier_factor" not in layer.anomaly.models
    rows = layer.anomaly.split.to_dict()["train_claims"]
    assert any(f"at least 100001 training claims are needed (this file leaves {rows}" in m
               for m in layer.messages)


# ------------------------------------------------ all-missing / constant columns


def test_all_missing_columns_do_not_crash(config, raw_claims):
    def blank_half(features):
        df = features.df
        numeric = [c for c in features.model_matrix()[1] if not c.endswith("__missing")]
        for c in numeric[::2]:
            df[c] = np.nan

    layer = _build(config, raw_claims.sample(500, random_state=6), features_hook=blank_half)
    assert layer.anomaly.models  # still trainable; missing is a value, not an error
    for scores in layer.anomaly.models.values():
        assert np.isfinite(scores.scores).all()


def test_every_column_constant_skips_both_models(config, raw_claims):
    def flatten(features):
        df = features.df
        for c in features.model_matrix()[1]:
            df[c] = False if c.endswith("__missing") else 1.0

    layer = _build(config, raw_claims.sample(400, random_state=7), features_hook=flatten)
    assert layer.anomaly.models == {}
    _assert_plain(layer)
    assert any("exactly the same values" in m for m in layer.messages)


# ---------------------------------------------------------------- one date


def test_every_claim_on_one_date(config, raw_claims):
    one = raw_claims.sample(300, random_state=8).copy()
    for col in ("date_of_admission", "date_of_discharge", "date_of_claim"):
        one[col] = "15/03/2024"
    layer = _build(config, one)
    assert layer.anomaly.models == {}
    _assert_plain(layer)
    assert any("same date (2024-03-15)" in m and "(this file has 1)" in m for m in layer.messages)


# ------------------------------------------------------------------- SHAP


class _Raises:
    def shap_values(self, *a, **k):
        raise RuntimeError("synthetic SHAP failure")


class _Slow:
    def shap_values(self, *a, **k):
        time.sleep(2.0)
        return np.zeros((1, 3))


def test_shap_failure_is_a_failed_explanation_and_fails_the_gate_check(healthy):
    name = "local_outlier_factor"
    explainer = healthy.explainers[name]
    saved = explainer._explainer
    try:
        explainer._explainer = _Raises()
        claim = str(healthy.anomaly.models[name].flagged_index[0])
        rows = healthy.explain_claim(name, claim)
        assert rows[0]["feature"] == "EXPLANATION_FAILED"
        assert rows[0]["plain_language"]
        verdicts = healthy.run_gate()
        quality = {c.name: c.status for c in verdicts[name].criteria}["Explanation quality"]
        assert quality == "FAIL"
        assert not verdicts[name].promoted
        assert any("could not be calculated" in m for m in healthy.messages)
    finally:
        explainer._explainer = saved
        healthy.gate_verdicts.clear()


def test_shap_timeout_is_bounded(healthy):
    name = "isolation_forest"
    explainer = healthy.explainers[name]
    saved, saved_t = explainer._explainer, explainer.explain_timeout_seconds
    try:
        explainer._explainer = _Slow()
        explainer.explain_timeout_seconds = 0.2
        claim = str(healthy.anomaly.models[name].flagged_index[0])
        started = time.perf_counter()
        rows = healthy.explain_claim(name, claim)
        assert time.perf_counter() - started < 1.5
        assert rows[0]["feature"] == "EXPLANATION_FAILED"
        assert "longer than" in rows[0]["plain_language"]
        assert "budget" in explainer.last_failure
    finally:
        explainer._explainer, explainer.explain_timeout_seconds = saved, saved_t


def test_shap_construction_failure_is_reported(config, raw_claims, monkeypatch):
    import shap

    def boom(*a, **k):
        raise RuntimeError("synthetic constructor failure")

    monkeypatch.setattr(shap, "TreeExplainer", boom)
    monkeypatch.setattr(shap, "KernelExplainer", boom)
    layer = _build(config, raw_claims.sample(500, random_state=9))
    assert layer.anomaly.models
    assert all(not e.available for e in layer.explainers.values())
    assert any("Explanations are not available for the" in m for m in layer.messages)
    for verdict in layer.gate_verdicts.values():
        assert not verdict.promoted


def test_explainer_on_a_one_row_background_does_not_raise():
    from sklearn.neighbors import LocalOutlierFactor

    X = pd.DataFrame({"a": [0.0, 1.0, 2.0], "b": [1.0, 0.0, 1.0]})
    lof = LocalOutlierFactor(n_neighbors=2, novelty=True).fit(X)
    explainer = ShapExplainer(lof, ["a", "b"], X.head(1))
    assert not explainer.available and explainer.failure_reason
    assert explainer.explain(X.iloc[0], X.iloc[0])[0]["feature"] == "EXPLANATION_UNAVAILABLE"


# ---------------------------------------------------------------- clusters


def test_cluster_discovery_on_tiny_input(config):
    d = NovelClusterDiscovery(config)
    need = 2 * int(config.get("novel_cluster_min_size"))
    out = d.discover(pd.DataFrame({"x": [0.1, 0.2, 0.3]}, index=["A", "B", "C"]))
    assert out == []
    assert f"At least {need} are needed" in d.message and "(twice novel_cluster_min_size; this file has 3)" in d.message


def test_cluster_discovery_finding_nothing(config, monkeypatch):
    d = NovelClusterDiscovery(config)
    monkeypatch.setattr(d, "_fit", lambda X: np.full(len(X), -1))
    frame = pd.DataFrame(np.random.default_rng(0).normal(size=(40, 3)), columns=list("abc"))
    assert d.discover(frame) == []
    assert d.completed and "No groups of similar unusual hospitals were found" in d.message


def test_cluster_discovery_failure_degrades(config, raw_claims, monkeypatch):
    def boom(self, X):
        raise ValueError("synthetic HDBSCAN failure")

    monkeypatch.setattr(NovelClusterDiscovery, "_fit", boom)
    layer = _build(config, raw_claims.sample(700, random_state=11))
    assert layer.clusters == []
    assert any("groups of similar unusual hospitals" in m for m in layer.messages)
    assert any("synthetic HDBSCAN failure" in n for n in layer.notes)


# ----------------------------------------------------- exploratory degradation


def test_exploratory_scores_on_an_untrained_layer_are_empty_with_a_message(config, raw_claims):
    layer = _build(config, raw_claims.head(1))
    scores = layer.exploratory_scores("isolation_forest")
    assert scores.empty
    assert layer.explain_exploratory_claim("isolation_forest", "x") == []
    assert any("Exploratory scores are not available" in m for m in layer.messages)


def test_exploratory_scores_cover_every_claim(healthy):
    for name, model in healthy.anomaly.models.items():
        scores = healthy.exploratory_scores(name)
        assert len(scores) == healthy.anomaly.split.total_claims
        # on the governed scoring set they ARE the governed scores
        governed = model.scores
        assert np.allclose(scores.reindex(governed.index).values, governed.values)
    rows = healthy.explain_exploratory_claim(
        "isolation_forest", str(healthy.anomaly.train_matrix.index[0]), top_n=3)
    assert rows and all(r["exploratory"] for r in rows)
    assert rows[0]["exploratory_notice"] == healthy.exploratory_notice
