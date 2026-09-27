"""Exploratory scores never reach a gate or a metric.

``ModelLayer.exploratory_scores()`` scores EVERY claim with the governed,
already-fitted models, including training-period claims and held-out
providers' claims. That is useful to look at and useless as evidence: the model
has seen some of those hospitals, so any lift, precision or calibration figure
computed on them would be inflated by exactly the leakage entity isolation
exists to prevent. The promotion gate, the metric suite, the release gates,
``provider_scores()``, ``explanation_sample()`` and ANL-01-R03 must therefore
behave as if the exploratory scores did not exist.

This file proves it two ways:

* **Behaviourally** — a small pipeline run is measured (gate verdicts, metric
  outputs, release gates, provider scores, governed score indexes, the gate's
  explanation sample), every exploratory score and an exploratory explanation
  are then computed, and the same measurements are taken again. They must be
  identical.
* **Statically** — no module under ``src/fwa/evaluation`` and neither
  ``models/gate.py`` nor ``models/controls.py`` references an exploratory
  identifier, and the governed helpers on the layer do not either.
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytestmark = pytest.mark.governance

ROOT = Path(__file__).resolve().parents[2]

#: Identifiers that belong to the exploratory path. Any of them appearing in a
#: governed module is a breach, however it is spelled (attribute, name or a
#: getattr string).
EXPLORATORY_IDENTIFIERS = {
    "exploratory_scores", "explain_exploratory_claim", "exploratory_notice",
    "_exploratory_cache", "_exploratory_scaled", "_exploratory_matrix",
    "_exploratory_matrix_scaled", "EXPLORATORY_NOTICE",
}


def _governed_modules() -> list[Path]:
    files = sorted((ROOT / "src" / "fwa" / "evaluation").glob("*.py"))
    files += [ROOT / "src" / "fwa" / "models" / "gate.py",
              ROOT / "src" / "fwa" / "models" / "controls.py"]
    return files


def _references(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    hits = []
    for node in ast.walk(tree):
        name = None
        if isinstance(node, ast.Attribute):
            name = node.attr
        elif isinstance(node, ast.Name):
            name = node.id
        elif isinstance(node, ast.alias):
            name = node.asname or node.name
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            name = node.value if node.value in EXPLORATORY_IDENTIFIERS else None
        if name and ("exploratory" in name.lower() or name in EXPLORATORY_IDENTIFIERS):
            hits.append(f"{path.name}:{getattr(node, 'lineno', '?')} {name}")
    return hits


# ------------------------------------------------------------------- static


def test_no_governed_module_references_the_exploratory_path():
    files = _governed_modules()
    assert len(files) > 3
    offences = [hit for f in files for hit in _references(f)]
    assert not offences, (
        "Exploratory scores must never feed a gate or a metric; found: " + "; ".join(offences)
    )


def test_the_governed_layer_helpers_do_not_read_the_exploratory_cache():
    from fwa.models.anomaly import AnomalyModels
    from fwa.models.layer import ModelLayer

    governed = [
        ModelLayer.provider_scores, ModelLayer.explanation_sample, ModelLayer.explain_claim,
        ModelLayer.run_gate, ModelLayer._build_residuals, ModelLayer.summary,
        ModelLayer.coverage, AnomalyModels.recalibrate, AnomalyModels.summary,
        AnomalyModels.coverage,
    ]
    for fn in governed:
        source = inspect.getsource(fn)
        assert "exploratory" not in source.lower(), f"{fn.__qualname__} touches the exploratory path"


# --------------------------------------------------------------- behavioural


@pytest.fixture(scope="module")
def run(raw_claims, config, tmp_path_factory):
    from fwa.pipeline import run_pipeline

    work = tmp_path_factory.mktemp("exploratory")
    path = work / "claims_sample.csv"
    raw_claims.sample(1200, random_state=int(config.get("random_seed"))).to_csv(path, index=False)
    result = run_pipeline(path, generate_documents=None, enable_ai=False, verbose=False,
                          document_dir=work / "docs")
    assert result.models is not None and result.models.anomaly.models, "fixture must train models"
    return result


def _measure(result) -> dict:
    """Everything a gate or a metric reads, in a comparable form."""
    from fwa.evaluation.gates import ReleaseGates
    from fwa.evaluation.metrics import MetricSuite

    np.random.seed(0)  # KernelExplainer samples from the global generator
    layer = result.models
    metrics = MetricSuite(result)
    capacity = int(result.config.get("alerts_per_reviewer_per_day")) * int(
        result.config.get("reviewer_count"))
    verdicts = layer.run_gate(outcomes=metrics.provider_outcomes(), capacity=capacity)
    return {
        "verdicts": {k: (v.promoted, v.summary, v.to_frame()) for k, v in verdicts.items()},
        "metrics": pd.DataFrame([m.to_row() for m in metrics.compute()]),
        "lift": metrics.lift_over_composite(),
        "precision": metrics.precision_at_capacity(),
        "release_gates": pd.DataFrame([g.to_row() for g in ReleaseGates(result, metrics).evaluate()]),
        "provider_scores": {k: v.copy() for k, v in layer.provider_scores().items()},
        "score_index": {k: set(m.scores.index) for k, m in layer.anomaly.models.items()},
        "scores": {k: m.scores.copy() for k, m in layer.anomaly.models.items()},
        "flagged": {k: list(m.flagged_index) for k, m in layer.anomaly.models.items()},
        "score_frame": layer.anomaly.score_frame.copy(),
        "samples": {k: layer.explanation_sample(k, n=5) for k in layer.anomaly.models},
    }


def _assert_same(before: dict, after: dict) -> None:
    assert before["verdicts"].keys() == after["verdicts"].keys()
    for name, (promoted, summary, frame) in before["verdicts"].items():
        p2, s2, f2 = after["verdicts"][name]
        assert (promoted, summary) == (p2, s2)
        pd.testing.assert_frame_equal(frame, f2)
    for key in ("metrics", "lift", "precision", "release_gates", "score_frame"):
        pd.testing.assert_frame_equal(before[key], after[key], obj=key)
    for name, series in before["provider_scores"].items():
        pd.testing.assert_series_equal(series, after["provider_scores"][name])
    for name, series in before["scores"].items():
        pd.testing.assert_series_equal(series, after["scores"][name])
    assert before["score_index"] == after["score_index"]
    assert before["flagged"] == after["flagged"]
    assert repr(before["samples"]) == repr(after["samples"])


def test_exploratory_scoring_changes_no_gate_and_no_metric(run):
    layer = run.models
    before = _measure(run)

    governed_ids = set(layer.anomaly.score_frame["claim_sk"].astype(str))
    for name in layer.anomaly.models:
        scores = layer.exploratory_scores(name)
        # it really is "every claim", i.e. strictly more than the governed set
        assert len(scores) == len(run.claims)
        assert governed_ids < set(scores.index)
        training_claim = str(layer.anomaly.train_matrix.index[0])
        rows = layer.explain_exploratory_claim(name, training_claim, top_n=3)
        assert rows and all(r.get("exploratory") is True for r in rows)

    after = _measure(run)
    _assert_same(before, after)

    # and the governed scoring set is still exactly the governed scoring set
    for name, index in after["score_index"].items():
        assert index == governed_ids, name
    for name, series in after["provider_scores"].items():
        assert set(series.index) <= set(layer.anomaly.split.score_providers), name


def test_exploratory_scores_carry_the_banner(run):
    assert run.models.exploratory_notice == (
        "Exploratory scores. The model has seen some of these hospitals during training, "
        "so these scores are not used for the promotion gate."
    )
