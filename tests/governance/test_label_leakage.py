"""Labels are held-out evaluation only (build brief §3.3).

    "``fraud_label``, ``fraud_type``, ``fraud_confidence`` and
     ``ground_truth_source`` are **held-out evaluation data only**. They must
     never be an input to any rule, any statistical baseline, any peer group,
     any feature, any model, any graph metric or any AI prompt... Enforce this
     structurally: a ``FeatureFrame`` that physically excludes the label columns,
     and a leakage test that breaks the build if they reappear."

This file is that test. It attacks the boundary from five directions: the
frame's constructor, its accessors, the adapter that splits the columns off,
the feature store the engine actually consumes, and a source-level sweep of
every module that is not the evaluation package.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pandas as pd
import pytest

from fwa.features.frame import FeatureFrame, LabelLeakageError

pytestmark = pytest.mark.governance

LABELS = ["fraud_label", "fraud_type", "fraud_confidence", "ground_truth_source"]

#: The only place in the package that is permitted to read a label at all.
EVALUATION_PACKAGE = "fwa/evaluation"

#: Modules that legitimately *name* the label columns in order to keep them out:
#: the adapter splits them off the source frame, the frame refuses them, the
#: store asserts their absence. Naming a column to exclude it is the opposite of
#: leaking it, so these are allowed — and listed here so that adding a new one
#: is a deliberate act that shows up in review.
QUARANTINE_MODULES = {
    "fwa/canonical/adapters.py",
    "fwa/features/frame.py",
    "fwa/features/store.py",
    "fwa/models/supervised_gate.py",
    "fwa/pipeline.py",
}


# --------------------------------------------------------------------- frame


@pytest.mark.parametrize("label", LABELS)
def test_feature_frame_refuses_a_label_column(label):
    df = pd.DataFrame({"claim_sk": ["C1"], "gross_amount_aed": [100.0], label: [1]})
    with pytest.raises(LabelLeakageError) as exc:
        FeatureFrame(df)
    assert label in str(exc.value)


def test_feature_frame_refuses_all_four_at_once():
    df = pd.DataFrame({"claim_sk": ["C1"], **{c: [0] for c in LABELS}})
    with pytest.raises(LabelLeakageError):
        FeatureFrame(df)


@pytest.mark.parametrize("label", LABELS)
def test_feature_frame_refuses_a_label_added_after_construction(label):
    frame = FeatureFrame(pd.DataFrame({"claim_sk": ["C1"], "gross_amount_aed": [100.0]}))
    with pytest.raises(LabelLeakageError):
        frame.add(label, [1])


@pytest.mark.parametrize("label", LABELS)
def test_feature_frame_refuses_to_read_a_label_by_any_accessor(label):
    frame = FeatureFrame(pd.DataFrame({"claim_sk": ["C1"], "gross_amount_aed": [100.0]}))
    with pytest.raises(LabelLeakageError):
        frame[label]
    with pytest.raises(LabelLeakageError):
        getattr(frame, label)
    assert label not in frame


def test_model_matrix_contains_no_label_and_no_identifier():
    frame = FeatureFrame(pd.DataFrame({
        "claim_sk": ["C1", "C2"], "member_sk": ["M1", "M2"], "provider_sk": ["P1", "P1"],
        "gross_amount_aed": [100.0, 200.0], "approval_ratio": [0.9, 0.8],
    }))
    matrix, columns = frame.model_matrix()
    for banned in LABELS + ["claim_sk", "member_sk", "provider_sk"]:
        assert banned not in columns
    assert matrix.shape[1] == len(columns)


# ------------------------------------------------------------------- adapter


def test_adapter_strips_labels_from_every_canonical_table(dataset):
    """No canonical table the engine reads carries a label column."""
    for name, frame in dataset.tables.items():
        for label in LABELS:
            assert label not in frame.columns, f"{label} survived into canonical table {name}"


def test_adapter_returns_labels_separately_and_completely(labels, sample_claims):
    """Held out, not discarded: evaluation still gets every row."""
    for label in LABELS:
        assert label in labels.columns
    assert len(labels) == len(sample_claims)


def test_engine_feature_frame_has_no_labels(context):
    """The features the controls actually consume are label-free."""
    features = context.features
    columns = features.columns if hasattr(features, "columns") else list(features)
    for label in LABELS:
        assert label not in columns


# -------------------------------------------------------------- source sweep


def _python_files(root: Path):
    for path in sorted((root / "src" / "fwa").rglob("*.py")):
        yield path, path.relative_to(root / "src").as_posix()


def test_no_module_outside_evaluation_reads_a_label(project_root):
    """A source-level sweep, because a test on behaviour only covers what it calls.

    The sweep parses each module rather than grepping it, so a label named in a
    docstring or a comment — which is how these columns are *discussed* all over
    the codebase — is not mistaken for one being read.
    """
    offenders: list[str] = []
    for path, relative in _python_files(project_root):
        if relative.startswith(EVALUATION_PACKAGE) or relative in QUARANTINE_MODULES:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            text = None
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                text = node.value
            elif isinstance(node, ast.Attribute):
                text = node.attr
            if text in LABELS:
                offenders.append(f"{relative}:{getattr(node, 'lineno', '?')} → {text}")
    assert not offenders, (
        "Label columns referenced outside the evaluation package:\n  " + "\n  ".join(offenders)
    )


def test_no_ai_prompt_can_carry_a_label(project_root):
    """§3.3 names AI prompts explicitly; the AI package must not know the columns exist."""
    for path, relative in _python_files(project_root):
        if not relative.startswith("fwa/ai/"):
            continue
        source = path.read_text(encoding="utf-8")
        for label in LABELS:
            assert label not in source, f"{relative} mentions {label}"


# ------------------------------------------------------- supervised training


def test_no_supervised_classifier_is_trained_anywhere(project_root):
    """§3.3: "Do not build a supervised classifier on this label. Train nothing."

    The sweep looks for a ``.fit(X, y)`` — a fit call with a second positional
    argument is supervised training by definition, whatever the estimator is
    called. Unsupervised estimators (IsolationForest, LOF, HDBSCAN) fit on one
    argument and are unaffected.
    """
    offenders: list[str] = []
    for path, relative in _python_files(project_root):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
                continue
            if node.func.attr in ("fit", "fit_predict", "partial_fit") and len(node.args) >= 2:
                offenders.append(f"{relative}:{node.lineno} → {node.func.attr} with a target")
    assert not offenders, "Supervised training found:\n  " + "\n  ".join(offenders)


def test_supervised_gate_blocks_promotion_and_trains_nothing():
    from fwa.models.supervised_gate import PROMOTION_BLOCKED, SupervisedGate

    # Deliberately answered as favourably as the artefact honestly can — and it
    # still blocks, because the prerequisites it cannot satisfy are the ones
    # about outcome capture and unbiased ground truth, not about plumbing.
    verdict = SupervisedGate().evaluate(
        review_outcomes=None,
        random_audit_performed=True,
        temporal_split_available=True,
        entity_isolation_available=True,
        calibration_and_stability_available=True,
    )
    assert verdict.status == PROMOTION_BLOCKED
    assert verdict.unmet, "The gate must name the prerequisites that are not met."
    frame = verdict.to_frame()
    assert not frame.empty
    assert not frame["met"].all(), "A blocked gate cannot report every prerequisite satisfied."
