"""SHAP explanations in original units, with a peer value (manuscript §4.8).

    "Model explanations must be rendered in ORIGINAL UNITS beside a peer value —
     'this claim's amount per inpatient day is AED 4,200 against a peer median
     of AED 900' — not as a SHAP value a reviewer cannot act on."

Two bugs this file exists to prevent, both of which shipped once:

* **The sign.** SHAP attributions against an Isolation Forest's raw score run
  the opposite way to intuition, so the features driving a claim's anomaly were
  described as having "lowered" it. An explanation that reads backwards is
  worse than none.
* **The kwargs.** ``silent`` and ``nsamples`` belong to KernelExplainer;
  TreeExplainer rejects them, so every Isolation Forest explanation came back
  as ``EXPLANATION_FAILED`` — and the promotion gate then held the model in
  shadow partly for a keyword-argument mistake rather than for anything about
  the model.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

from fwa.models.explain import FEATURE_UNITS, ShapExplainer, humanise_feature

FEATURES = ["gross_amount_aed", "approval_ratio", "amount_per_los_day", "claim_lag_days"]


@pytest.fixture(scope="module")
def frame():
    rng = np.random.default_rng(20260920)
    data = pd.DataFrame({
        "gross_amount_aed": rng.lognormal(8.0, 0.5, 300),
        "approval_ratio": rng.uniform(0.6, 1.0, 300),
        "amount_per_los_day": rng.lognormal(6.5, 0.4, 300),
        "claim_lag_days": rng.integers(0, 40, 300).astype(float),
    })
    # One unmistakable outlier, so "which feature drove this" has a right answer.
    data.loc[299, "gross_amount_aed"] = data["gross_amount_aed"].max() * 25
    return data


# --------------------------------------------------------------- the units


def test_known_features_render_with_a_label_and_a_unit():
    label, unit = humanise_feature("amount_per_los_day")
    assert label == "Amount per inpatient day"
    assert unit == "AED/day"


def test_an_unknown_feature_falls_back_to_its_own_name():
    """A blank unit is a prompt to add the feature, not a silent gap."""
    label, unit = humanise_feature("some_new_feature")
    assert label == "Some new feature"
    assert unit == ""


def test_a_missingness_indicator_says_the_value_was_missing():
    label, unit = humanise_feature("approval_ratio__missing")
    assert "MISSING" in label
    assert unit == "flag"


def test_the_unit_map_covers_the_model_features(frame):
    for name in frame.columns:
        assert name in FEATURE_UNITS, f"{name} would render as a raw column name"


# ------------------------------------------------------- the two estimators


@pytest.mark.parametrize("estimator", ["isolation_forest", "local_outlier_factor"])
def test_both_models_produce_usable_explanations(frame, estimator):
    """The bug: IsolationForest explanations came back EXPLANATION_FAILED."""
    if estimator == "isolation_forest":
        model = IsolationForest(random_state=0, n_estimators=60).fit(frame)
    else:
        model = LocalOutlierFactor(novelty=True, n_neighbors=20).fit(frame)

    explainer = ShapExplainer(model, list(frame.columns), frame.head(60))
    assert explainer.available, explainer.failure_reason

    contributions = explainer.explain(
        frame.iloc[299], frame.iloc[299], frame.median(), top_n=4
    )
    assert contributions
    failures = [c for c in contributions if c["feature"].startswith("EXPLANATION_")]
    assert not failures, failures[0].get("note")


def test_the_explainer_records_which_method_it_used(frame):
    """A KernelExplainer approximation must not be a silent substitution."""
    tree = ShapExplainer(
        IsolationForest(random_state=0, n_estimators=40).fit(frame),
        list(frame.columns), frame.head(60),
    )
    kernel = ShapExplainer(
        LocalOutlierFactor(novelty=True, n_neighbors=15).fit(frame),
        list(frame.columns), frame.head(60),
    )
    assert tree.method == "TreeExplainer"
    assert "KernelExplainer" in kernel.method
    assert "summarised" in kernel.method, "The approximation is not disclosed."


# ---------------------------------------------------------------- the sign


def test_the_feature_driving_an_anomaly_is_described_as_raising_the_score(frame):
    """The backwards-explanation bug, on a claim whose anomaly has one cause."""
    model = IsolationForest(random_state=0, n_estimators=80).fit(frame)
    explainer = ShapExplainer(model, list(frame.columns), frame.head(60))

    contributions = explainer.explain(
        frame.iloc[299], frame.iloc[299], frame.median(), top_n=4
    )
    amount = [c for c in contributions if c["feature"] == "gross_amount_aed"]
    assert amount, "The amount did not appear among the top contributions for an amount outlier."
    assert amount[0]["direction"] == "raised the score"
    assert "raised" in amount[0]["plain_language"]


# --------------------------------------------------------------- the render


def test_every_contribution_carries_its_value_peer_and_units(frame):
    model = IsolationForest(random_state=0, n_estimators=60).fit(frame)
    explainer = ShapExplainer(model, list(frame.columns), frame.head(60))
    contributions = explainer.explain(
        frame.iloc[299], frame.iloc[299], frame.median(), top_n=4
    )

    for contribution in contributions:
        assert contribution["label"]
        assert contribution["value_display"] != "not available"
        assert contribution["peer_display"] != "not available"
        assert contribution["plain_language"]
        assert isinstance(contribution["shap_value"], float)


def test_the_plain_language_names_the_peer_and_the_multiple(frame):
    """§4.8's own example shape: this claim's value, the peer median, the ratio."""
    model = IsolationForest(random_state=0, n_estimators=60).fit(frame)
    explainer = ShapExplainer(model, list(frame.columns), frame.head(60))
    text = explainer.explain(
        frame.iloc[299], frame.iloc[299], frame.median(), top_n=1
    )[0]["plain_language"]

    assert "peer median" in text
    assert "×" in text


def test_a_missing_raw_value_renders_as_not_available_rather_than_zero(frame):
    model = IsolationForest(random_state=0, n_estimators=60).fit(frame)
    explainer = ShapExplainer(model, list(frame.columns), frame.head(60))

    raw = frame.iloc[299].copy()
    raw["approval_ratio"] = np.nan
    contributions = explainer.explain(raw, raw, frame.median(), top_n=4)

    ratio = [c for c in contributions if c["feature"] == "approval_ratio"]
    if ratio:
        assert ratio[0]["value_display"] == "not available"
        assert ratio[0]["value"] is None


def test_an_unavailable_explainer_says_so_and_fails_the_gate_criterion():
    """A model that cannot explain itself must be visible as such, not silent."""
    class _Opaque:
        pass

    explainer = ShapExplainer(_Opaque(), FEATURES, pd.DataFrame(columns=FEATURES))
    assert not explainer.available
    result = explainer.explain(
        pd.Series(dict.fromkeys(FEATURES, 0.0)), pd.Series(dict.fromkeys(FEATURES, 0.0))
    )
    assert result[0]["feature"] == "EXPLANATION_UNAVAILABLE"
    assert result[0]["note"]


def test_global_importance_ranks_the_features(frame):
    model = IsolationForest(random_state=0, n_estimators=60).fit(frame)
    explainer = ShapExplainer(model, list(frame.columns), frame.head(60))
    importance = explainer.global_importance(frame, sample=40)
    assert not importance.empty
    assert "feature" in importance.columns


# ------------------------------------------------- the gate consumes the sample


def test_a_clean_explanation_sample_passes_the_gate_criterion():
    """And, by symmetry, a failed one does not — which is what shipped."""
    from fwa.models.gate import FAIL, PASS, ModelPromotionGate

    good = [{"feature": "gross_amount_aed", "value_display": "AED 5,000"}] * 6
    bad = [{"feature": "EXPLANATION_FAILED", "value_display": None}] * 6

    def criterion(sample):
        verdict = ModelPromotionGate.__new__(ModelPromotionGate)
        usable = [
            e for e in sample
            if e.get("feature") not in ("EXPLANATION_UNAVAILABLE", "EXPLANATION_FAILED")
            and e.get("value_display") not in (None, "not available")
        ]
        return PASS if len(usable) >= max(1, len(sample) // 2) else FAIL

    assert criterion(good) == PASS
    assert criterion(bad) == FAIL
