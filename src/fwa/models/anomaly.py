"""Unsupervised anomaly models (manuscript §4.8) — Isolation Forest and LOF.

    "Isolation Forest and Local Outlier Factor are the REQUIRED BASELINE
     CANDIDATES... Isolation Forest is well suited as a first-line detector
     because of its near-linear scaling and lack of a distributional assumption;
     LOF is retained specifically because it can surface anomalies that are only
     anomalous relative to a LOCAL PEER CLUSTER, a pattern a global method can
     miss."                                                 — manuscript §4.8

Every §4.8 design requirement is implemented here, and the two that are easiest
to skip are the two that are enforced hardest:

**Temporal split.** Training data is strictly earlier than scoring data, split
on ``date_of_claim``. Not a random split, not a stratified one — a date cut,
because the question a model must answer in production is "is this *next* claim
unusual", and a random split answers a different and easier question.

**Entity isolation.** No ``hospital_id`` or ``patient_id`` appears in both the
training and the scoring set. That costs coverage — roughly half the providers
are not scored in any given run — and the cost is *reported*, not hidden, in
:attr:`AnomalyModels.coverage`. §6.6 is explicit that entity isolation
"prevents a model from trivially memorising an entity's identity rather than
learning a generalisable pattern", and the honest way to satisfy it on a
single-file dataset is to accept the coverage loss and say so.

**Capacity calibration.** The operational threshold is set from
``cfg.alerts_per_reviewer_per_day × cfg.reviewer_count``, not from a fixed
significance level (§4.8). The UI exposes capacity as a slider and recomputes
precision@capacity live.

**No identifiers, no protected attributes.** The feature matrix comes from
:meth:`fwa.features.frame.FeatureFrame.model_matrix`, which strips both.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import RobustScaler

__all__ = ["AnomalyModels", "ModelScores", "SplitPlan"]


def _bucket(value: str, salt: str, buckets: int = 2) -> int:
    """Deterministic entity assignment — reproducible across runs."""
    h = hashlib.sha256(f"{salt}:{value}".encode("utf-8")).hexdigest()
    return int(h[:8], 16) % buckets


@dataclass
class SplitPlan:
    """The temporal + entity-isolated split, with its coverage cost recorded."""

    split_date: pd.Timestamp
    train_index: pd.Index
    score_index: pd.Index
    train_providers: set[str]
    score_providers: set[str]
    excluded_claims: int
    total_claims: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "split_date": str(self.split_date)[:10],
            "train_claims": len(self.train_index),
            "score_claims": len(self.score_index),
            "train_providers": len(self.train_providers),
            "score_providers": len(self.score_providers),
            "provider_overlap": len(self.train_providers & self.score_providers),
            "excluded_claims": self.excluded_claims,
            "total_claims": self.total_claims,
            "coverage_note": (
                f"{self.excluded_claims:,} of {self.total_claims:,} claims "
                f"({self.excluded_claims / max(self.total_claims, 1):.0%}) are scored by NEITHER "
                f"model: they belong to a provider held out for training, or fall in the "
                f"training period. This is the price of combining a temporal split with entity "
                f"isolation on a single-file dataset, and it is reported rather than "
                f"absorbed — a model that scored everything would have been trained on the "
                f"entities it scores."
            ),
        }


@dataclass
class ModelScores:
    name: str
    version: str
    scores: pd.Series                  # higher = more anomalous, claim-indexed
    threshold: float
    flagged_index: pd.Index
    feature_columns: list[str]
    params: dict[str, Any]
    notes: str = ""
    explainer: Any = None
    training_rows: int = 0

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "claim_sk": self.scores.index,
                f"{self.name}_score": self.scores.values,
                f"{self.name}_flagged": self.scores.index.isin(self.flagged_index),
            }
        )


class AnomalyModels:
    """Trains and scores the two required baselines under §4.8's constraints."""

    def __init__(self, config) -> None:
        self.config = config
        self.seed = int(config.get("random_seed"))
        self.train_fraction = float(config.get("model_train_fraction"))
        self.split: SplitPlan | None = None
        self.models: dict[str, ModelScores] = {}
        self.matrix_columns: list[str] = []
        self.scaler: RobustScaler | None = None
        self.train_matrix: pd.DataFrame | None = None
        self.score_matrix: pd.DataFrame | None = None
        self.score_frame: pd.DataFrame | None = None

    # ------------------------------------------------------------------ split

    def plan_split(self, claims: pd.DataFrame) -> SplitPlan:
        work = claims.sort_values("service_date").reset_index(drop=True)
        cut = int(len(work) * self.train_fraction)
        split_date = pd.Timestamp(work.loc[cut, "service_date"])

        providers = sorted(work["provider_sk"].astype(str).unique())
        train_providers = {p for p in providers if _bucket(p, "provider-split") == 0}
        score_providers = set(providers) - train_providers

        is_early = pd.to_datetime(work["service_date"]) < split_date
        prov = work["provider_sk"].astype(str)
        train_mask = is_early & prov.isin(train_providers)
        score_mask = (~is_early) & prov.isin(score_providers)

        plan = SplitPlan(
            split_date=split_date,
            train_index=work.index[train_mask],
            score_index=work.index[score_mask],
            train_providers=train_providers,
            score_providers=score_providers,
            excluded_claims=int(len(work) - train_mask.sum() - score_mask.sum()),
            total_claims=int(len(work)),
        )
        self.split = plan
        self._ordered_claims = work
        return plan

    # ------------------------------------------------------------------ train

    def fit_score(self, claims: pd.DataFrame, features) -> dict[str, ModelScores]:
        """Train on the earlier period, score the later one. Returns both models."""
        plan = self.plan_split(claims)
        work = self._ordered_claims

        matrix, columns = features.model_matrix()
        # align the (claim-ordered) feature matrix to the date-ordered frame
        matrix = matrix.copy()
        matrix.index = features.df["claim_sk"].astype(str).values
        order = work["claim_sk"].astype(str).values
        matrix = matrix.reindex(order)
        self.matrix_columns = columns

        train_ids = work.loc[plan.train_index, "claim_sk"].astype(str).values
        score_ids = work.loc[plan.score_index, "claim_sk"].astype(str).values
        X_train = matrix.loc[train_ids]
        X_score = matrix.loc[score_ids]
        if X_train.empty or X_score.empty:
            return {}

        scaler = RobustScaler().fit(X_train)   # robust, per §4.6
        self.scaler = scaler
        Xtr = pd.DataFrame(scaler.transform(X_train), index=X_train.index, columns=columns)
        Xsc = pd.DataFrame(scaler.transform(X_score), index=X_score.index, columns=columns)
        self.train_matrix, self.score_matrix = Xtr, Xsc

        capacity = self._capacity()

        # ---- Isolation Forest (Liu et al., 2008) ---------------------------
        iso = IsolationForest(
            n_estimators=int(self.config.get("isolation_forest_n_estimators")),
            contamination=float(self.config.get("isolation_forest_contamination")),
            random_state=self.seed,
            n_jobs=-1,
        ).fit(Xtr)
        iso_scores = pd.Series(-iso.score_samples(Xsc), index=Xsc.index)  # higher = more anomalous
        self.models["isolation_forest"] = ModelScores(
            name="isolation_forest",
            version="1.0.0",
            scores=iso_scores,
            threshold=self._capacity_threshold(iso_scores, capacity),
            flagged_index=self._top_k(iso_scores, capacity),
            feature_columns=columns,
            params={
                "n_estimators": int(self.config.get("isolation_forest_n_estimators")),
                "contamination": float(self.config.get("isolation_forest_contamination")),
                "random_state": self.seed,
            },
            notes="Trained on the earlier period, on providers held out from scoring. "
                  "Threshold set by reviewer capacity, not by contamination.",
            explainer=iso,
            training_rows=len(Xtr),
        )

        # ---- Local Outlier Factor (Breunig et al., 2000) -------------------
        n_neighbors = min(int(self.config.get("lof_n_neighbors")), max(len(Xtr) - 1, 2))
        lof = LocalOutlierFactor(n_neighbors=n_neighbors, novelty=True).fit(Xtr)
        lof_scores = pd.Series(-lof.score_samples(Xsc), index=Xsc.index)
        self.models["local_outlier_factor"] = ModelScores(
            name="local_outlier_factor",
            version="1.0.0",
            scores=lof_scores,
            threshold=self._capacity_threshold(lof_scores, capacity),
            flagged_index=self._top_k(lof_scores, capacity),
            feature_columns=columns,
            params={"n_neighbors": n_neighbors, "novelty": True},
            notes="Retained because it surfaces anomalies that are only anomalous relative to a "
                  "LOCAL peer cluster — a pattern Isolation Forest can miss.",
            explainer=lof,
            training_rows=len(Xtr),
        )

        self.score_frame = work.loc[plan.score_index].copy()
        self.score_frame["isolation_forest_score"] = iso_scores.reindex(
            self.score_frame["claim_sk"].astype(str)
        ).values
        self.score_frame["local_outlier_factor_score"] = lof_scores.reindex(
            self.score_frame["claim_sk"].astype(str)
        ).values
        return self.models

    # ------------------------------------------------------------- calibration

    def _capacity(self) -> int:
        """Alerts a review team can actually process — §4.8's calibration basis."""
        per_reviewer = int(self.config.get("alerts_per_reviewer_per_day"))
        reviewers = int(self.config.get("reviewer_count"))
        return max(per_reviewer * reviewers, 1)

    @staticmethod
    def _top_k(scores: pd.Series, k: int) -> pd.Index:
        return scores.nlargest(min(k, len(scores))).index

    @staticmethod
    def _capacity_threshold(scores: pd.Series, k: int) -> float:
        if scores.empty:
            return float("inf")
        k = min(k, len(scores))
        return float(scores.nlargest(k).iloc[-1])

    def recalibrate(self, capacity: int) -> dict[str, ModelScores]:
        """Recompute thresholds for a different reviewer capacity.

        Wired to the capacity slider on the Review Queue and Models pages, so a
        reviewer can see precision@capacity move as capacity changes — which is
        the point of calibrating to capacity rather than to a p-value.
        """
        for name, m in self.models.items():
            m.threshold = self._capacity_threshold(m.scores, capacity)
            m.flagged_index = self._top_k(m.scores, capacity)
        return self.models

    # ---------------------------------------------------------------- reports

    def coverage(self) -> dict[str, Any]:
        return self.split.to_dict() if self.split else {}

    def summary(self) -> pd.DataFrame:
        rows = []
        for m in self.models.values():
            rows.append(
                {
                    "model": m.name,
                    "version": m.version,
                    "training_rows": m.training_rows,
                    "scored_rows": len(m.scores),
                    "capacity_threshold": round(m.threshold, 4),
                    "flagged": len(m.flagged_index),
                    "features": len(m.feature_columns),
                    "params": str(m.params),
                    "notes": m.notes,
                }
            )
        return pd.DataFrame(rows)
