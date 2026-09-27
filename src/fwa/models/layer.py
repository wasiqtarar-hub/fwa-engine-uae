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
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from .anomaly import AnomalyModels, ModelScores
from .clusters import NovelCluster, NovelClusterDiscovery
from .drift import DriftMonitor, DriftResult
from .explain import ShapExplainer, humanise_feature
from .gate import GateVerdict, ModelPromotionGate
from .supervised_gate import SupervisedGate, SupervisedGateVerdict

__all__ = ["ModelLayer"]


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
    notes: list[str] = field(default_factory=list)

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
        models = layer.anomaly.fit_score(claims, features)
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
            layer.explainers[name] = ShapExplainer(
                scores.explainer, scores.feature_columns, layer.anomaly.train_matrix
            )

        if composite_results:
            layer.composite_ranking = pd.Series(
                {r.provider_sk: r.score for r in composite_results if not r.insufficient_evidence}
            )

        monitor = DriftMonitor(config)
        for name, scores in models.items():
            layer.drift[name] = monitor.run(
                train_matrix=layer.anomaly.train_matrix,
                score_matrix=layer.anomaly.score_matrix,
                scores=scores.scores,
                reference_scores=None,
                alert_volume=len(scores.flagged_index),
                expected_alert_volume=None,
                review_outcomes=None,
            )

        layer.residual_frame = layer._build_residuals()
        discovery = NovelClusterDiscovery(config)
        layer.clusters = discovery.discover(
            layer.residual_frame,
            exposures=layer._provider_exposure(claims),
            feature_labels={c: humanise_feature(c)[0] for c in layer.residual_frame.columns},
        )

        layer.supervised = SupervisedGate().evaluate(
            review_outcomes=None,
            random_audit_performed=False,
            temporal_split_available=True,
            entity_isolation_available=layer.anomaly.coverage().get("provider_overlap", 1) == 0,
            calibration_and_stability_available=False,
        )
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
            sample = self.explanation_sample(name, n=5)
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
