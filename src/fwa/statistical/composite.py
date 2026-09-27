"""ANL-01-R01 — the transparent composite (manuscript §4.8).

    "Every unsupervised model candidate is required to be compared against
     ANL-01-R01, a TRANSPARENT, AUDITABLE ROBUST-COMPOSITE SCORE built from
     weighted residuals across scenario-aligned features, using future-period
     review yield, not in-sample separation, as the comparison criterion. This
     ordering is deliberate: it means the system always has an EXPLAINABLE
     STATISTICAL FALLBACK that a reviewer can understand without any model at
     all, and it means an opaque model is only added to the production signal
     set once it has been shown, prospectively, to catch something the
     transparent composite does not."                      — manuscript §4.8

This module is therefore load-bearing twice over: it is a detection control in
its own right, and it is the *benchmark* the model promotion gate measures
against. If the models never beat it, the honest finding is that the models
never beat it — and the system still works, because this is what it falls back
to.

"Auditable" is meant literally. Every provider's score decomposes into
per-feature contributions, and each contribution is reported alongside the
provider's value and the peer median **in original units** (AED, days, ratio),
not as a bare residual. A reviewer with the weights from
``config/parameters.yaml`` can reconstruct the score with a calculator, and the
Provider Analytics page shows exactly that arithmetic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from ..config import AppConfig
from .peers import PeerService, INSUFFICIENT_PEER_EVIDENCE
from .robust import robust_stats

__all__ = ["CompositeFeature", "CompositeResult", "TransparentComposite", "COMPOSITE_FEATURES"]


@dataclass(frozen=True)
class CompositeFeature:
    """One scenario-aligned feature of the composite."""

    key: str
    label: str
    unit: str
    scenario_alignment: str
    direction: str = "high"     # "high" = only high values are adverse; "both" = either tail


#: The seven features, each aligned to a scenario family so that a reviewer can
#: see *which kind of risk* a contribution represents rather than only that the
#: number was large.
COMPOSITE_FEATURES: tuple[CompositeFeature, ...] = (
    CompositeFeature("amount_residual", "Mean claim amount", "AED",
                     "PAY-06 tariff/billed-amount manipulation"),
    CompositeFeature("amount_per_los_day_residual", "Amount per inpatient day", "AED/day",
                     "CLN-01 coding intensity"),
    CompositeFeature("pharmacy_share_residual", "Pharmacy share of the bill", "ratio",
                     "PHR-03 pharmacy spike"),
    CompositeFeature("approval_ratio_residual", "Approved-to-billed ratio", "ratio",
                     "PAY-06-R04 approval pattern", direction="both"),
    CompositeFeature("utilisation_velocity_residual", "Claims per distinct member", "claims/member",
                     "CLN-04 unnecessary or repeated services"),
    CompositeFeature("readmission_rate_residual", "Readmission rate", "ratio",
                     "CLN-05 readmission irregularity"),
    CompositeFeature("coding_mismatch_rate_residual", "Coding-mismatch rate", "ratio",
                     "CLN-01 upcoding proxy"),
)

#: composite feature key -> the provider-level metric column it is built from
_METRIC_FOR_FEATURE = {
    "amount_residual": "mean_gross_aed",
    "amount_per_los_day_residual": "mean_amount_per_los_day",
    "pharmacy_share_residual": "mean_pharmacy_ratio",
    "approval_ratio_residual": "mean_approval_ratio",
    "utilisation_velocity_residual": "claims_per_member",
    "readmission_rate_residual": "readmission_rate",
    "coding_mismatch_rate_residual": "coding_mismatch_rate",
}


@dataclass
class CompositeResult:
    provider_sk: str
    score: float
    peer_level_used: str
    peer_n: int
    peer_baseline_version: str
    claim_count: int
    contributions: list[dict[str, Any]] = field(default_factory=list)
    exposure_aed: float = 0.0
    insufficient_evidence: bool = False

    def top_contributions(self, n: int = 3) -> list[dict[str, Any]]:
        return sorted(self.contributions, key=lambda c: -abs(c["contribution"]))[:n]

    def explain(self) -> str:
        """Plain-language lead sentence (§14.1: lead with the sentence, keep the number)."""
        if self.insufficient_evidence:
            return (
                f"Provider {self.provider_sk} could not be compared: no peer group met the "
                f"minimum size, so no composite score was computed."
            )
        top = self.top_contributions(2)
        if not top:
            return f"Provider {self.provider_sk} scores {self.score:.2f} with no dominant feature."
        parts = []
        for c in top:
            if c["peer_median"] in (None, 0) or c["value"] is None:
                continue
            ratio = c["value"] / c["peer_median"] if c["peer_median"] else float("nan")
            parts.append(
                f"{c['label'].lower()} of {_fmt(c['value'], c['unit'])} against a peer median of "
                f"{_fmt(c['peer_median'], c['unit'])} ({ratio:.1f}×)"
            )
        return (
            f"This provider bills {' and '.join(parts)} "
            f"(peer group: {self.peer_level_used}, {self.peer_n} providers). "
            f"Composite score {self.score:.2f}."
        )

    def to_row(self) -> dict[str, Any]:
        return {
            "provider_sk": self.provider_sk,
            "composite_score": round(self.score, 4),
            "peer_level_used": self.peer_level_used,
            "peer_n": self.peer_n,
            "peer_baseline_version": self.peer_baseline_version,
            "claim_count": self.claim_count,
            "exposure_aed": round(self.exposure_aed, 2),
            "insufficient_evidence": self.insufficient_evidence,
            "top_features": "; ".join(
                f"{c['label']}={_fmt(c['value'], c['unit'])} vs peer {_fmt(c['peer_median'], c['unit'])}"
                for c in self.top_contributions(3)
            ),
        }


def _fmt(value: Any, unit: str) -> str:
    if value is None or (isinstance(float(value) if value is not None else 0.0, float) and value != value):
        return "n/a"
    v = float(value)
    if unit == "AED":
        return f"AED {v:,.0f}"
    if unit == "AED/day":
        return f"AED {v:,.0f}/day"
    if unit == "ratio":
        return f"{v:.1%}"
    return f"{v:,.2f} {unit}"


class TransparentComposite:
    """Builds provider-level metrics, residualises them against peers, weights them."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.weights: dict[str, float] = dict(config.get("composite_feature_weights"))
        self.threshold = float(config.get("composite_flag_threshold"))
        self.min_claims = int(config.get("min_entity_opportunities"))

    # ------------------------------------------------------------------ build

    def provider_metrics(self, claims: pd.DataFrame) -> pd.DataFrame:
        """Aggregate the claim frame to one row per provider."""
        work = claims.copy()
        work["approval_ratio"] = np.where(
            work["gross_amount_aed"] > 0,
            work["approved_amount_aed"] / work["gross_amount_aed"].replace(0, np.nan),
            np.nan,
        )
        work["amount_per_los_day"] = np.where(
            work["length_of_stay_days"] > 0,
            work["gross_amount_aed"] / work["length_of_stay_days"].replace(0, np.nan),
            np.nan,
        )
        work["readmission_flag"] = (
            work["discharge_readmit_gap_days"] < self.config.get("readmit_window_days")
        ).astype(float)
        work["coding_mismatch"] = (~work["icd_code_matches_procedure"].astype(bool)).astype(float)

        agg = work.groupby("provider_sk").agg(
            claim_count=("claim_sk", "count"),
            distinct_members=("member_sk", "nunique"),
            mean_gross_aed=("gross_amount_aed", "mean"),
            total_gross_aed=("gross_amount_aed", "sum"),
            mean_amount_per_los_day=("amount_per_los_day", "mean"),
            mean_pharmacy_ratio=("pharmacy_bill_ratio", "mean"),
            mean_approval_ratio=("approval_ratio", "mean"),
            readmission_rate=("readmission_flag", "mean"),
            coding_mismatch_rate=("coding_mismatch", "mean"),
        ).reset_index()
        agg["claims_per_member"] = agg["claim_count"] / agg["distinct_members"].replace(0, np.nan)

        # peer-grouping dimensions: a provider's dominant service line and mix
        modes = work.groupby("provider_sk").agg(
            diagnosis_primary=("diagnosis_primary", lambda s: s.mode().iat[0] if not s.mode().empty else None),
            diagnosis_chapter=("diagnosis_chapter", lambda s: s.mode().iat[0] if not s.mode().empty else None),
            policy_type=("policy_type", lambda s: s.mode().iat[0] if not s.mode().empty else None),
            tpa=("tpa", lambda s: s.mode().iat[0] if not s.mode().empty else None),
        ).reset_index()
        out = agg.merge(modes, on="provider_sk", how="left")
        out["provider_volume_band"] = out["claim_count"].map(self.config.peer_groups.volume_band)
        return out

    # ------------------------------------------------------------------ score

    def compute(self, claims: pd.DataFrame) -> tuple[list[CompositeResult], pd.DataFrame]:
        metrics = self.provider_metrics(claims)
        peers = PeerService(self.config, metrics, entity_column="provider_sk")

        results: list[CompositeResult] = []
        for _, row in metrics.iterrows():
            if row["claim_count"] < self.min_claims:
                results.append(
                    CompositeResult(
                        provider_sk=str(row["provider_sk"]), score=0.0,
                        peer_level_used="BELOW_MINIMUM_DENOMINATOR", peer_n=0,
                        peer_baseline_version="", claim_count=int(row["claim_count"]),
                        insufficient_evidence=True,
                    )
                )
                continue
            group = peers.resolve(row)
            if not group.sufficient:
                results.append(
                    CompositeResult(
                        provider_sk=str(row["provider_sk"]), score=0.0,
                        peer_level_used=INSUFFICIENT_PEER_EVIDENCE, peer_n=0,
                        peer_baseline_version="", claim_count=int(row["claim_count"]),
                        insufficient_evidence=True,
                    )
                )
                continue

            contributions: list[dict[str, Any]] = []
            score = 0.0
            for feat in COMPOSITE_FEATURES:
                metric_col = _METRIC_FOR_FEATURE[feat.key]
                weight = float(self.weights.get(feat.key, 0.0))
                peer_values = peers.values(group, metric_col)
                stats = robust_stats(peer_values)
                value = row.get(metric_col)
                resid = stats.z(value) if pd.notna(value) else None
                if resid is None:
                    contributions.append(
                        {
                            "feature": feat.key, "label": feat.label, "unit": feat.unit,
                            "scenario_alignment": feat.scenario_alignment,
                            "value": None if pd.isna(value) else float(value),
                            "peer_median": None if stats.n == 0 else stats.median,
                            "peer_scale": None if stats.n == 0 else stats.scale,
                            "residual": None, "weight": weight, "contribution": 0.0,
                            "scale_source": stats.fallback_used or "mad",
                            "note": "No residual: the peer scale is degenerate or the value is missing. "
                                    "Contributes zero rather than being imputed.",
                        }
                    )
                    continue
                effective = abs(resid) if feat.direction == "both" else max(resid, 0.0)
                contribution = weight * effective
                score += contribution
                contributions.append(
                    {
                        "feature": feat.key, "label": feat.label, "unit": feat.unit,
                        "scenario_alignment": feat.scenario_alignment,
                        "value": float(value), "peer_median": stats.median, "peer_scale": stats.scale,
                        "residual": float(resid), "weight": weight,
                        "contribution": float(contribution),
                        "scale_source": stats.fallback_used or "mad",
                        "note": "",
                    }
                )
            results.append(
                CompositeResult(
                    provider_sk=str(row["provider_sk"]),
                    score=float(score),
                    peer_level_used=group.level_name,
                    peer_n=group.n_entities,
                    peer_baseline_version=group.baseline_version,
                    claim_count=int(row["claim_count"]),
                    contributions=contributions,
                    exposure_aed=float(row["total_gross_aed"]),
                )
            )
        frame = pd.DataFrame([r.to_row() for r in results]).sort_values(
            "composite_score", ascending=False
        )
        return results, frame

    def flagged(self, results: list[CompositeResult]) -> list[CompositeResult]:
        return [r for r in results if not r.insufficient_evidence and r.score > self.threshold]
