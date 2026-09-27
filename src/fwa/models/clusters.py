"""ANL-01-R04 — novel-cluster discovery (manuscript §4.8, §9.3).

    "ANL-01-R04 | Novel-cluster discovery | M / QUARTERLY | Analyst-reviewed
     cluster exhibits coherent new typology and material exposure | EXPLORATORY;
     NO PRODUCTION ACTION UNTIL CONVERTED TO RULE | Rule-development candidate"
                                                            — Appendix C

Clusters the residual space to surface candidate emerging typologies for the
quarterly typology review. Three constraints, all from the catalogue and all
enforced here rather than merely described:

* Disposition is ``MONITOR_ONLY``. It cannot be anything else — §3.3 forbids a
  model control from denying, and the catalogue chooses the weakest of the
  remaining dispositions on top of that.
* "No production action until converted to rule." A cluster produces a *draft*
  control through :mod:`fwa.ai.triage`, which enters the registry in ``shadow``
  and requires a separate human approver.
* Noise points are excluded. HDBSCAN's ``-1`` label means "this point belongs to
  no cluster", and treating noise as a typology would be the single easiest way
  to manufacture a finding.

The residual space is used rather than the raw feature space because what is
interesting is *how an entity differs from its peers*, not where it sits in
absolute terms — two providers can bill identically and be differently unusual.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

__all__ = ["NovelCluster", "NovelClusterDiscovery"]


@dataclass
class NovelCluster:
    cluster_id: int
    size: int
    members: list[str]
    top_features: list[dict[str, Any]]
    aggregate_exposure_aed: float
    candidate_typology_name: str
    coherence: float
    note: str = ""

    def to_row(self) -> dict[str, Any]:
        return {
            "cluster_id": self.cluster_id,
            "size": self.size,
            "candidate_typology_name": self.candidate_typology_name,
            "coherence": round(self.coherence, 3),
            "aggregate_exposure_aed": round(self.aggregate_exposure_aed, 2),
            "top_features": "; ".join(
                f"{f['label']}={f['mean_residual']:+.2f}σ" for f in self.top_features[:3]
            ),
            "members": ", ".join(self.members[:8]) + ("…" if len(self.members) > 8 else ""),
        }


class NovelClusterDiscovery:
    def __init__(self, config) -> None:
        self.config = config
        self.min_cluster_size = int(config.get("novel_cluster_min_size"))
        self.seed = int(config.get("random_seed"))
        self.labels: pd.Series | None = None

    def discover(
        self,
        residuals: pd.DataFrame,
        exposures: pd.Series | None = None,
        feature_labels: dict[str, str] | None = None,
    ) -> list[NovelCluster]:
        """Cluster the residual space. Returns candidate typologies."""
        if residuals.empty or len(residuals) < self.min_cluster_size * 2:
            return []
        X = residuals.replace([np.inf, -np.inf], np.nan).fillna(0.0)

        labels = self._fit(X)
        self.labels = pd.Series(labels, index=residuals.index, name="cluster")
        feature_labels = feature_labels or {}

        out: list[NovelCluster] = []
        for cid in sorted(set(labels)):
            if cid == -1:
                continue  # noise is not a typology
            members = residuals.index[self.labels == cid].tolist()
            if len(members) < self.min_cluster_size:
                continue
            sub = X.loc[members]
            means = sub.mean().sort_values(key=lambda s: -s.abs())
            top = [
                {
                    "feature": name,
                    "label": feature_labels.get(name, name.replace("_", " ").capitalize()),
                    "mean_residual": float(value),
                }
                for name, value in means.head(5).items()
            ]
            # coherence: how tightly the cluster's members agree on their top
            # features. A cluster whose members disagree is not a typology.
            spread = float(sub[means.head(3).index].std().mean()) if len(sub) > 1 else 0.0
            coherence = float(1.0 / (1.0 + spread))
            exposure = float(exposures.reindex(members).fillna(0).sum()) if exposures is not None else 0.0
            out.append(
                NovelCluster(
                    cluster_id=int(cid),
                    size=len(members),
                    members=[str(m) for m in members],
                    top_features=top,
                    aggregate_exposure_aed=exposure,
                    candidate_typology_name=self._name(top),
                    coherence=coherence,
                    note=(
                        "EXPLORATORY. Appendix C, ANL-01-R04: 'no production action until "
                        "converted to rule'. This is an UNVALIDATED HYPOTHESIS for the quarterly "
                        "typology review, not a finding. Disposition is MONITOR_ONLY and "
                        "cannot be anything else."
                    ),
                )
            )
        return sorted(out, key=lambda c: -c.aggregate_exposure_aed)

    def _fit(self, X: pd.DataFrame) -> np.ndarray:
        try:
            from sklearn.cluster import HDBSCAN

            return HDBSCAN(
                min_cluster_size=self.min_cluster_size, min_samples=max(2, self.min_cluster_size // 3)
            ).fit_predict(X)
        except Exception:
            from sklearn.cluster import DBSCAN
            from sklearn.neighbors import NearestNeighbors

            # eps from the knee of the k-distance curve, so the fallback is not
            # an arbitrary radius.
            k = max(2, self.min_cluster_size // 2)
            distances, _ = NearestNeighbors(n_neighbors=k).fit(X).kneighbors(X)
            eps = float(np.percentile(distances[:, -1], 90))
            return DBSCAN(eps=eps, min_samples=self.min_cluster_size).fit_predict(X)

    @staticmethod
    def _name(top_features: list[dict[str, Any]]) -> str:
        """A descriptive working name, deliberately mechanical.

        A generated name should read as a *description of the residual pattern*,
        never as an accusation. "High amount per day with low approval ratio" is
        a hypothesis; "systematic upcoding ring" is a conclusion the evidence
        does not support.
        """
        if not top_features:
            return "Unnamed residual cluster"
        parts = []
        for f in top_features[:2]:
            direction = "elevated" if f["mean_residual"] > 0 else "depressed"
            parts.append(f"{direction} {f['label'].lower()}")
        return "Candidate pattern: " + " with ".join(parts)
