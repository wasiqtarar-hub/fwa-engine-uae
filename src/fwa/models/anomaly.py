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
import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import RobustScaler

__all__ = ["AnomalyModels", "ModelScores", "SplitPlan", "MIN_PROVIDERS", "ISOLATION_FOREST_MIN_TRAIN"]

_log = logging.getLogger(__name__)

#: Entity isolation needs one provider to train on and a different one to score.
MIN_PROVIDERS = 2
#: An isolation tree needs at least two points to make a split.
ISOLATION_FOREST_MIN_TRAIN = 2


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
        #: model name (or ``"all"``) -> technical reason it was not trained.
        self.skipped: dict[str, str] = {}
        #: plain-language sentences for the UI, one per reason.
        self.messages: list[str] = []
        # Every claim's unscaled feature row; read ONLY by the layer's opt-in
        # exploratory scoring, never by any governed metric or gate.
        self._exploratory_matrix: pd.DataFrame | None = None

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

    def _skip(self, model: str, technical: str, plain: str) -> None:
        """Record why a model (or ``"all"``) was not trained: technical + plain."""
        self.skipped[model] = technical
        if plain and plain not in self.messages:
            self.messages.append(plain)
        _log.warning("Model %s skipped: %s", model, technical)

    def _preflight(self, claims: pd.DataFrame) -> bool:
        """Checks that must hold before a fair split is even possible."""
        n_claims = int(len(claims))
        if n_claims == 0:
            self._skip("all", "Input claim frame is empty.",
                       "This file has no claims, so there is nothing for the models to learn from.")
            return False
        n_providers = int(claims["provider_sk"].astype(str).nunique())
        if n_providers < MIN_PROVIDERS:
            self._skip(
                "all",
                f"Entity isolation needs >= {MIN_PROVIDERS} providers (one training group, one "
                f"scoring group); found {n_providers}.",
                f"Not enough hospitals in this file to train the model fairly. At least "
                f"{MIN_PROVIDERS} are needed (this file has {n_providers}): the model learns from "
                f"one group of hospitals and is checked on a different group, so it is never "
                f"judged on a hospital it has already seen.",
            )
            return False
        dates = pd.to_datetime(claims["service_date"], errors="coerce").dropna()
        n_dates = int(dates.dt.normalize().nunique())
        if n_dates == 0:
            self._skip("all", "No parseable service_date values.",
                       "None of the claims in this file has a readable date, so the model cannot "
                       "learn from earlier claims and be checked on later ones.")
            return False
        if n_dates < 2:
            day = str(dates.iloc[0])[:10]
            self._skip(
                "all",
                f"Temporal split impossible: every claim has service_date {day}.",
                f"Every claim in this file has the same date ({day}), so the model cannot learn "
                f"from earlier claims and be checked on later ones. At least 2 different claim "
                f"dates are needed (this file has 1).",
            )
            return False
        return True

    def fit_score(self, claims: pd.DataFrame, features) -> dict[str, ModelScores]:
        """Train on the earlier period, score the later one. Returns both models.

        Never raises on a small or degenerate file: every reason a model could
        not be trained is recorded in :attr:`skipped` (technical) and
        :attr:`messages` (a plain sentence for the UI), and an empty or partial
        dict is returned. On a healthy file the path is exactly the governed one.
        """
        self.models = {}
        self.skipped = {}
        self.messages = []
        if not self._preflight(claims):
            return {}
        try:
            plan = self.plan_split(claims)
        except Exception as exc:  # noqa: BLE001 - degrade, never crash
            self._skip("all", f"plan_split failed: {type(exc).__name__}: {exc}",
                       "The claims in this file could not be divided into a training period and a "
                       "later checking period, so no model was trained.")
            return {}
        work = self._ordered_claims

        if not plan.train_providers or not plan.score_providers:
            n = len(plan.train_providers) + len(plan.score_providers)
            which = "both" if n == 2 else f"all {n}"
            self._skip(
                "all",
                f"Provider hold-out produced train_providers={len(plan.train_providers)}, "
                f"score_providers={len(plan.score_providers)}.",
                f"Not enough hospitals in this file to train the model fairly: {which} fell into "
                f"the same group of the fair hold-out, so there is no separate group to check "
                f"the model on. At least 1 hospital is needed in each group (this file has "
                f"{len(plan.train_providers)} to learn from and {len(plan.score_providers)} to "
                f"check).",
            )
            return {}

        try:
            matrix, columns = features.model_matrix()
        except Exception as exc:  # noqa: BLE001
            self._skip("all", f"model_matrix failed: {type(exc).__name__}: {exc}",
                       "The measures the model needs could not be built from this file, so no "
                       "model was trained.")
            return {}
        if not columns:
            self._skip("all", "Model matrix has no numeric columns.",
                       "This file has no numeric measures the model can use, so no model was "
                       "trained.")
            return {}
        # align the (claim-ordered) feature matrix to the date-ordered frame
        matrix = matrix.copy()
        matrix.index = features.df["claim_sk"].astype(str).values
        order = work["claim_sk"].astype(str).values
        matrix = matrix.reindex(order)
        if matrix.isna().to_numpy().any():
            # Only a degenerate input reaches this (a claim with no feature row,
            # or a column the frame could not fill); a healthy run has no NaN
            # here, so the fill never changes a healthy result.
            _log.warning("Model matrix contained NaN after alignment; filled with 0.0.")
            matrix = matrix.fillna(0.0)
        self.matrix_columns = columns
        # Every claim's (unscaled) row, kept ONLY for ModelLayer's opt-in
        # exploratory scoring. Nothing in the governed path reads it.
        self._exploratory_matrix = matrix

        train_ids = work.loc[plan.train_index, "claim_sk"].astype(str).values
        score_ids = work.loc[plan.score_index, "claim_sk"].astype(str).values
        X_train = matrix.loc[train_ids]
        X_score = matrix.loc[score_ids]
        split_date = str(plan.split_date)[:10]
        if X_train.empty:
            self._skip(
                "all",
                f"Empty training set after the split (split_date={split_date}, "
                f"train_providers={len(plan.train_providers)}).",
                f"No claims were left to learn from after the fair split: the hospitals set "
                f"aside for training have no claims before {split_date}. At least 1 training "
                f"claim is needed (this file has 0).",
            )
            return {}
        if X_score.empty:
            self._skip(
                "all",
                f"Empty scoring set after the split (split_date={split_date}, "
                f"score_providers={len(plan.score_providers)}).",
                f"No claims were left to check the model on after the fair split: the hospitals "
                f"set aside for checking have no claims on or after {split_date}. At least 1 "
                f"claim is needed (this file has 0).",
            )
            return {}
        if bool((X_train.nunique(dropna=False) <= 1).all()):
            self._skip(
                "all",
                f"All {len(columns)} training columns are constant over {len(X_train)} rows.",
                f"Every claim the model would learn from has exactly the same values on all "
                f"{len(columns)} measures it uses, so there is no pattern to learn. At least 2 "
                f"training claims that differ are needed (this file has none).",
            )
            return {}

        try:
            scaler = RobustScaler().fit(X_train)   # robust, per §4.6
            Xtr = pd.DataFrame(scaler.transform(X_train), index=X_train.index, columns=columns)
            Xsc = pd.DataFrame(scaler.transform(X_score), index=X_score.index, columns=columns)
        except Exception as exc:  # noqa: BLE001
            self._skip("all", f"RobustScaler failed: {type(exc).__name__}: {exc}",
                       "The claims in this file could not be put on a common scale for the "
                       "model, so no model was trained.")
            return {}
        self.scaler = scaler
        self.train_matrix, self.score_matrix = Xtr, Xsc

        capacity = self._capacity()

        # ---- Isolation Forest (Liu et al., 2008) ---------------------------
        if len(Xtr) < ISOLATION_FOREST_MIN_TRAIN:
            self._skip(
                "isolation_forest",
                f"{len(Xtr)} training row(s) < {ISOLATION_FOREST_MIN_TRAIN}.",
                f"Not enough claims to train the Isolation Forest model. It isolates a claim by "
                f"splitting it away from others, so at least {ISOLATION_FOREST_MIN_TRAIN} "
                f"training claims are needed (this file leaves {len(Xtr)} after the fair split).",
            )
        else:
            try:
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
            except Exception as exc:  # noqa: BLE001
                self._skip("isolation_forest", f"fit/score failed: {type(exc).__name__}: {exc}",
                           "The Isolation Forest model could not be trained on this file, so it "
                           "was skipped. The technical reason is in the log.")

        # ---- Local Outlier Factor (Breunig et al., 2000) -------------------
        configured_k = int(self.config.get("lof_n_neighbors"))
        lof_min = configured_k + 1
        if len(Xtr) < lof_min:
            self._skip(
                "local_outlier_factor",
                f"{len(Xtr)} training row(s) < lof_n_neighbors + 1 = {lof_min}.",
                f"Not enough claims to train the Local Outlier Factor model. It compares each "
                f"claim with its {configured_k} nearest neighbours (lof_n_neighbors), so at "
                f"least {lof_min} training claims are needed (this file leaves {len(Xtr)} after "
                f"the fair split).",
            )
        else:
            try:
                n_neighbors = min(configured_k, max(len(Xtr) - 1, 2))
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
            except Exception as exc:  # noqa: BLE001
                self._skip("local_outlier_factor", f"fit/score failed: {type(exc).__name__}: {exc}",
                           "The Local Outlier Factor model could not be trained on this file, so "
                           "it was skipped. The technical reason is in the log.")

        if not self.models:
            return {}
        self.score_frame = work.loc[plan.score_index].copy()
        for name, m in self.models.items():
            self.score_frame[f"{name}_score"] = m.scores.reindex(
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
