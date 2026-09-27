"""The model layer, assembled (manuscript §4.8).

One object that owns the whole Type-M path: split, train, score, explain,
monitor drift, run the promotion gate, discover novel clusters and evaluate the
supervised prerequisites. The Streamlit Models page and the validation report
both read from here, so what the UI shows and what the report prints cannot
diverge.

Ordering matters and is not arbitrary. Models are trained **after** the
transparent composite exists, because the composite is the benchmark (§4.8),
and the gate is run **after** drift and explanations, because two of its five
criteria depend on them.

**Never crash.** A small or degenerate file (one claim, one hospital, one
date, constant columns) degrades rather than raising: the layer comes back with
fewer models, or none, and says why twice — :attr:`ModelLayer.notes` carries
the technical reason (also logged) and :attr:`ModelLayer.messages` carries a
plain sentence the pages show as-is.

**Exploratory scoring.** The governed split leaves roughly half the claims
unscored by construction. :meth:`ModelLayer.exploratory_scores` scores *every*
claim with the SAME fitted models, for looking only. Those scores live in a
private cache that the promotion gate, the metrics, the release gates,
:meth:`provider_scores`, :meth:`explanation_sample` and ANL-01-R03 never read;
``tests/governance/test_exploratory_scores_excluded.py`` enforces that.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from .anomaly import AnomalyModels, ModelScores
from .clusters import NovelCluster, NovelClusterDiscovery
from .drift import DriftMonitor, DriftResult
from .explain import ShapExplainer, humanise_feature
from .gate import GateCriterion, GateVerdict, ModelPromotionGate
from .supervised_gate import SupervisedGate, SupervisedGateVerdict

__all__ = ["ModelLayer", "EXPLORATORY_NOTICE", "MODEL_LABELS"]

_log = logging.getLogger(__name__)

#: The banner every exploratory score is shown under. Exact text, by brief.
EXPLORATORY_NOTICE = (
    "Exploratory scores. The model has seen some of these hospitals during training, "
    "so these scores are not used for the promotion gate."
)

#: Plain names for the pages' sentences.
MODEL_LABELS = {
    "isolation_forest": "Isolation Forest",
    "local_outlier_factor": "Local Outlier Factor",
}


def _label(model_name: str) -> str:
    return MODEL_LABELS.get(model_name, model_name.replace("_", " ").title())


@dataclass
class ModelLayer:
    config: Any
    anomaly: AnomalyModels | None = None
    explainers: dict[str, ShapExplainer] = field(default_factory=dict)
    drift: dict[str, list[DriftResult]] = field(default_factory=dict)
    gate_verdicts: dict[str, GateVerdict] = field(default_factory=dict)
    clusters: list[NovelCluster] = field(default_factory=list)
    supervised: SupervisedGateVerdict | None = None
    peer_medians: pd.Series | None = None
    raw_features: pd.DataFrame | None = None
    composite_ranking: pd.Series | None = None
    residual_frame: pd.DataFrame | None = None
    #: technical detail (also logged).
    notes: list[str] = field(default_factory=list)
    #: plain-language sentences for the UI, one per degradation.
    messages: list[str] = field(default_factory=list)
    # Opt-in exploratory scores, model name -> claim_sk-indexed Series. Private
    # and separate from ``anomaly.models`` on purpose: no governed calculation
    # reads it (see the module docstring).
    _exploratory_cache: dict[str, pd.Series] = field(
        default_factory=dict, repr=False, compare=False
    )
    _exploratory_scaled: pd.DataFrame | None = field(default=None, repr=False, compare=False)

    #: Banner text for any page that shows exploratory scores.
    exploratory_notice: str = field(default=EXPLORATORY_NOTICE, init=False, repr=False)

    # ------------------------------------------------------------ messaging

    def _say(self, plain: str | None = None, technical: str | None = None, *, log: bool = True) -> None:
        if technical:
            if log:
                _log.warning(technical)
            if technical not in self.notes:
                self.notes.append(technical)
        if plain and plain not in self.messages:
            self.messages.append(plain)

    # ------------------------------------------------------------------ build

    @classmethod
    def build(
        cls,
        config,
        claims: pd.DataFrame,
        features,
        composite_results: list | None = None,
        *,
        enable_experimental: bool = False,
    ) -> "ModelLayer":
        layer = cls(config=config)
        layer.anomaly = AnomalyModels(config)
        try:
            models = layer.anomaly.fit_score(claims, features)
        except Exception as exc:  # noqa: BLE001 - last-resort guard; fit_score degrades itself
            layer.anomaly.models = {}
            models = {}
            layer._say(
                "The models could not be trained on this file, so no model scores are shown. "
                "The technical reason is in the log.",
                f"AnomalyModels.fit_score raised {type(exc).__name__}: {exc}",
            )
        for plain in layer.anomaly.messages:
            layer._say(plain)
        for name, reason in layer.anomaly.skipped.items():
            layer._say(technical=f"{name} not trained: {reason}", log=False)  # logged by _skip
        if not models:
            layer.notes.append(
                "No model could be trained: the temporal + entity-isolated split left an empty "
                "training or scoring set."
            )
            return layer

        raw = features.df.copy()
        raw.index = raw["claim_sk"].astype(str).values
        layer.raw_features = raw
        numeric = raw.select_dtypes(include=[np.number])
        layer.peer_medians = numeric.median()

        for name, scores in models.items():
            explainer = ShapExplainer(
                scores.explainer, scores.feature_columns, layer.anomaly.train_matrix
            )
            layer.explainers[name] = explainer
            if not explainer.available:
                layer._say(
                    f"Explanations are not available for the {_label(name)} model on this file, "
                    f"so it fails the explanation-quality check and stays in shadow.",
                    f"{name}: SHAP unavailable: {explainer.failure_reason}",
                )

        if composite_results:
            layer.composite_ranking = pd.Series(
                {r.provider_sk: r.score for r in composite_results if not r.insufficient_evidence}
            )

        monitor = DriftMonitor(config)
        for name, scores in models.items():
            try:
                layer.drift[name] = monitor.run(
                    train_matrix=layer.anomaly.train_matrix,
                    score_matrix=layer.anomaly.score_matrix,
                    scores=scores.scores,
                    reference_scores=None,
                    alert_volume=len(scores.flagged_index),
                    expected_alert_volume=None,
                    review_outcomes=None,
                )
            except Exception as exc:  # noqa: BLE001
                layer.drift[name] = []
                layer._say(
                    f"Drift checks could not be run for the {_label(name)} model on this file, "
                    f"so the promotion gate marks drift as not assessable.",
                    f"{name}: DriftMonitor.run raised {type(exc).__name__}: {exc}",
                )

        try:
            layer.residual_frame = layer._build_residuals()
        except Exception as exc:  # noqa: BLE001
            layer.residual_frame = pd.DataFrame()
            layer._say(technical=f"_build_residuals raised {type(exc).__name__}: {exc}")
        discovery = NovelClusterDiscovery(config)
        try:
            layer.clusters = discovery.discover(
                layer.residual_frame,
                exposures=layer._provider_exposure(claims),
                feature_labels={c: humanise_feature(c)[0] for c in layer.residual_frame.columns},
            )
        except Exception as exc:  # noqa: BLE001
            layer.clusters = []
            discovery.skip_reason = f"discover raised {type(exc).__name__}: {exc}"
            discovery.message = (
                "The search for groups of similar unusual hospitals could not be completed on "
                "this file, so no candidate groups are shown."
            )
        if not layer.clusters and discovery.completed:
            # Ran normally and found only noise: a result, not a degradation,
            # so it is a message for the page and nothing is added to notes.
            layer._say(discovery.message or None)
        elif not layer.clusters and (discovery.message or discovery.skip_reason):
            layer._say(discovery.message or None,
                       f"Novel-cluster discovery: {discovery.skip_reason}" if discovery.skip_reason else None)

        try:
            layer.supervised = SupervisedGate().evaluate(
                review_outcomes=None,
                random_audit_performed=False,
                temporal_split_available=True,
                entity_isolation_available=layer.anomaly.coverage().get("provider_overlap", 1) == 0,
                calibration_and_stability_available=False,
            )
        except Exception as exc:  # noqa: BLE001
            layer.supervised = None
            layer._say(technical=f"SupervisedGate.evaluate raised {type(exc).__name__}: {exc}")
        if enable_experimental:
            layer.notes.append(
                "--enable-experimental was set. One-Class SVM, DBSCAN, GMM and autoencoder "
                "variants are PERMITTED EXPERIMENTS and are clearly labelled as such; "
                "none may be promoted without clearing the same gate."
            )
        return layer

    # ------------------------------------------------------------------- gate

    def run_gate(self, outcomes: pd.Series | None = None, capacity: int | None = None) -> dict[str, GateVerdict]:
        """Run the Table 6.1 model gate. ``outcomes`` comes from fwa.evaluation."""
        if self.anomaly is None:
            return {}
        gate = ModelPromotionGate(self.config)
        provider_scores = self.provider_scores()
        for name, scores in self.anomaly.models.items():
            try:
                sample = self.explanation_sample(name, n=5)
                if any(e.get("feature") == "EXPLANATION_FAILED" for e in sample):
                    explainer = self.explainers.get(name)
                    self._say(
                        f"Some explanations for the {_label(name)} model could not be calculated, "
                        f"which counts against its explanation-quality check.",
                        f"{name}: explanation failure in gate sample: "
                        f"{explainer.last_failure if explainer is not None else ''}",
                    )
                self.gate_verdicts[name] = gate.evaluate(
                    model_name=name,
                    model_scores=provider_scores.get(name, pd.Series(dtype=float)),
                    composite_ranking=self.composite_ranking if self.composite_ranking is not None else pd.Series(dtype=float),
                    split_plan=self.anomaly.coverage(),
                    drift_results=self.drift.get(name, []),
                    explanation_sample=sample,
                    outcomes=outcomes,
                    capacity=capacity,
                )
            except Exception as exc:  # noqa: BLE001 - an unevaluable gate never promotes
                reason = f"{type(exc).__name__}: {exc}"
                verdict = GateVerdict(
                    model=name,
                    promoted=False,
                    criteria=[GateCriterion(
                        "Gate evaluation", "NOT_ASSESSABLE",
                        f"The promotion gate could not be evaluated on this file ({reason}). "
                        f"NOT_ASSESSABLE is not a pass.")],
                )
                verdict.summary = (
                    f"{name} REMAINS IN SHADOW. The promotion gate could not be evaluated on this "
                    f"file, and a gate that cannot be evaluated is not passed."
                )
                self.gate_verdicts[name] = verdict
                self._say(
                    f"The promotion gate could not be checked for the {_label(name)} model on "
                    f"this file, so the model stays in shadow.",
                    f"{name}: ModelPromotionGate.evaluate raised {reason}",
                )
        return self.gate_verdicts

    # ---------------------------------------------------------------- helpers

    def provider_scores(self) -> dict[str, pd.Series]:
        """Aggregate claim-level scores to the provider entity (§4.8 flags entities)."""
        out: dict[str, pd.Series] = {}
        if self.anomaly is None or self.anomaly.score_frame is None:
            return out
        frame = self.anomaly.score_frame
        for name in self.anomaly.models:
            col = f"{name}_score"
            if col not in frame.columns:
                continue
            out[name] = frame.groupby("provider_sk")[col].max()
        return out

    def explanation_sample(self, model_name: str, n: int = 5) -> list[dict[str, Any]]:
        explainer = self.explainers.get(model_name)
        scores = self.anomaly.models.get(model_name) if self.anomaly else None
        if explainer is None or scores is None or self.anomaly.score_matrix is None:
            return []
        out: list[dict[str, Any]] = []
        for claim_sk in list(scores.flagged_index)[:n]:
            out.extend(self.explain_claim(model_name, str(claim_sk), top_n=3))
        return out

    def explain_claim(self, model_name: str, claim_sk: str, top_n: int = 5) -> list[dict[str, Any]]:
        explainer = self.explainers.get(model_name)
        if explainer is None or self.anomaly is None or self.anomaly.score_matrix is None:
            return []
        if claim_sk not in self.anomaly.score_matrix.index:
            return []
        scaled = self.anomaly.score_matrix.loc[claim_sk]
        raw = self.raw_features.loc[claim_sk] if self.raw_features is not None and claim_sk in self.raw_features.index else pd.Series(dtype=float)
        return explainer.explain(scaled, raw, self.peer_medians, top_n=top_n)

    # ------------------------------------------------------ exploratory (opt-in)

    def exploratory_scores(self, model_name: str) -> pd.Series:
        """EXPLORATORY: score every claim with the governed, already-fitted model.

        Same estimator, same scaler, fitted on the governed training set exactly
        as for the governed scores — nothing is refitted. Returns a
        ``claim_sk``-indexed Series (higher = more unusual) covering training
        period claims and held-out providers' claims too, so it is shown under
        :attr:`exploratory_notice` and is NEVER used by the promotion gate, the
        metrics, the release gates or any control. On the governed scoring set
        the values equal the governed scores. Computed lazily and cached; when
        impossible, returns an empty Series and adds a plain sentence to
        :attr:`messages`.
        """
        cached = self._exploratory_cache.get(model_name)
        if cached is not None:
            return cached.copy()
        empty = pd.Series(dtype=float, name=f"{model_name}_exploratory_score")
        empty.index.name = "claim_sk"
        anomaly = self.anomaly
        if anomaly is None or not anomaly.models:
            self._say("Exploratory scores are not available because no model could be trained "
                      "on this file.")
            return empty
        model = anomaly.models.get(model_name)
        if model is None:
            self._say(f"Exploratory scores are not available for the {_label(model_name)} model "
                      f"because it was not trained on this file.")
            return empty
        try:
            scaled = self._exploratory_matrix_scaled()
            estimator = model.explainer
            values = pd.Series(
                -estimator.score_samples(scaled[model.feature_columns]),
                index=scaled.index,
                name=f"{model_name}_exploratory_score",
            )
            values.index.name = "claim_sk"
        except Exception as exc:  # noqa: BLE001
            self._say(
                f"Exploratory scores could not be calculated for the {_label(model_name)} model "
                f"on this file. The technical reason is in the log.",
                f"{model_name}: exploratory scoring raised {type(exc).__name__}: {exc}",
            )
            return empty
        self._exploratory_cache[model_name] = values
        return values.copy()

    def _exploratory_matrix_scaled(self) -> pd.DataFrame:
        if self._exploratory_scaled is None:
            anomaly = self.anomaly
            full = getattr(anomaly, "_exploratory_matrix", None)
            if full is None or anomaly.scaler is None:
                raise RuntimeError("No full feature matrix or fitted scaler is available.")
            columns = list(anomaly.matrix_columns)
            self._exploratory_scaled = pd.DataFrame(
                anomaly.scaler.transform(full[columns]), index=full.index, columns=columns
            )
        return self._exploratory_scaled

    def explain_exploratory_claim(
        self, model_name: str, claim_sk: str, top_n: int = 5
    ) -> list[dict[str, Any]]:
        """EXPLORATORY: explain one claim's exploratory score with the same explainer.

        Every returned row carries ``exploratory=True`` and the notice. The
        global NumPy random state is saved and restored around the SHAP call so
        that asking for an exploratory explanation can never perturb a later
        governed one (KernelExplainer samples from the global generator).
        """
        scores = self.exploratory_scores(model_name)
        explainer = self.explainers.get(model_name)
        claim_sk = str(claim_sk)
        if scores.empty or explainer is None or self._exploratory_scaled is None:
            return []
        if claim_sk not in self._exploratory_scaled.index:
            self._say(f"Claim {claim_sk} is not in this file, so it has no exploratory score.")
            return []
        row = self._exploratory_scaled.loc[claim_sk]
        raw = (self.raw_features.loc[claim_sk]
               if self.raw_features is not None and claim_sk in self.raw_features.index
               else pd.Series(dtype=float))
        state = np.random.get_state()
        try:
            rows = explainer.explain(row, raw, self.peer_medians, top_n=top_n)
        finally:
            np.random.set_state(state)
        return [
            {**r, "exploratory": True, "exploratory_notice": EXPLORATORY_NOTICE,
             "exploratory_score": float(scores.get(claim_sk, np.nan))}
            for r in rows
        ]

    # ------------------------------------------------------------- internals

    def _build_residuals(self) -> pd.DataFrame:
        """Residual space for cluster discovery: provider metrics minus peer medians."""
        if self.anomaly is None or self.anomaly.score_matrix is None or self.raw_features is None:
            return pd.DataFrame()
        scored = self.anomaly.score_frame
        if scored is None or scored.empty:
            return pd.DataFrame()
        # A CURATED residual space, not the full model matrix. Clustering 95
        # providers in 104 dimensions is dominated by the curse of
        # dimensionality, and — more importantly — a typology has to be
        # DESCRIBABLE. These are the scenario-aligned quantities a reviewer can
        # read a candidate typology out of.
        preferred = [
            "gross_amount_aed", "amount_per_los_day", "approval_ratio", "length_of_stay_days",
            "pharmacy_bill_ratio", "coding_mismatch", "readmission_flag",
            "member_utilisation_velocity", "num_insurers_same_event", "claim_lag_days",
            "days_since_policy_start", "is_early_tenure",
        ]
        numeric_cols = [
            c for c in preferred
            if c in self.raw_features.columns
            and pd.api.types.is_numeric_dtype(self.raw_features[c])
        ] or [
            c for c in self.anomaly.matrix_columns
            if not c.endswith("__missing") and c in self.raw_features.columns
        ]
        joined = scored[["claim_sk", "provider_sk"]].copy()
        joined = joined.join(
            self.raw_features[numeric_cols], on="claim_sk"
        )
        prov = joined.groupby("provider_sk")[numeric_cols].mean()
        med = prov.median()
        mad = (prov - med).abs().median()
        # A feature whose MAD is zero has no robust scale in this population —
        # every provider sits at the same value. Including it as a column of
        # zeros would add dimensions without information and dilute the
        # distance metric the clustering depends on, so it is DROPPED and the
        # drop is visible in the residual frame's column list.
        usable = mad[mad > 0].index
        resid = ((prov[usable] - med[usable]) / (1.4826 * mad[usable]))
        return resid.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    @staticmethod
    def _provider_exposure(claims: pd.DataFrame) -> pd.Series:
        return claims.groupby("provider_sk")["gross_amount_aed"].sum()

    # ---------------------------------------------------------------- reports

    def coverage(self) -> dict[str, Any]:
        """The split plan and its coverage cost, for the model card and report."""
        return self.anomaly.coverage() if self.anomaly is not None else {}

    def summary(self) -> dict[str, Any]:
        return {
            "models": [] if self.anomaly is None else list(self.anomaly.models),
            "split": {} if self.anomaly is None else self.anomaly.coverage(),
            "gate_verdicts": {k: v.headline for k, v in self.gate_verdicts.items()},
            "novel_clusters": len(self.clusters),
            "supervised_gate": None if self.supervised is None else self.supervised.status,
            "notes": self.notes,
        }
