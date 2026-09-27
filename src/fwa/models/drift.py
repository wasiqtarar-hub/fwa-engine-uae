"""Drift monitoring (manuscript §4.8, §9.8).

    "is continuously monitored for POPULATION STABILITY, SCORE-DISTRIBUTION
     DRIFT, ALERT VOLUME and REVIEW YIELD"                  — manuscript §4.8

Four monitors, one verdict. A breach recommends recalibration or the kill
switch (§9.7); it does not apply either automatically, because deactivating a
control is a governed action with a named owner (§3.10), not something a
monitor does on its own.

The Population Stability Index is the conventional measure and is banded
conventionally (0.10 warn / 0.25 breach, both governed parameters). It is worth
being explicit about what PSI can and cannot say: it detects that the input
distribution moved, not that the model got worse. That is why review yield —
the confirmed-to-flagged ratio, which needs real reviewer outcomes — is the
monitor §6.6 calls "the primary signal of whether a control or model is still
earning its place in production", and why this artefact reports it as
``NOT_MEASURABLE_ON_THIS_DATASET`` until dispositions accumulate rather than
substituting a label-derived proxy.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

__all__ = ["DriftMonitor", "DriftResult", "population_stability_index"]


def population_stability_index(
    expected: pd.Series, actual: pd.Series, bins: int = 10
) -> tuple[float, pd.DataFrame]:
    """PSI between a reference and a current distribution, plus the bin detail."""
    exp = pd.Series(expected).astype(float).dropna()
    act = pd.Series(actual).astype(float).dropna()
    if exp.empty or act.empty:
        return float("nan"), pd.DataFrame()
    quantiles = np.unique(np.quantile(exp, np.linspace(0, 1, bins + 1)))
    if len(quantiles) < 3:
        return 0.0, pd.DataFrame()
    quantiles[0], quantiles[-1] = -np.inf, np.inf
    e_counts = pd.cut(exp, quantiles, duplicates="drop").value_counts(sort=False)
    a_counts = pd.cut(act, quantiles, duplicates="drop").value_counts(sort=False)
    e_share = (e_counts / max(e_counts.sum(), 1)).replace(0, 1e-6)
    a_share = (a_counts / max(a_counts.sum(), 1)).replace(0, 1e-6)
    contrib = (a_share - e_share) * np.log(a_share / e_share)
    detail = pd.DataFrame(
        {
            "bin": [str(i) for i in e_share.index],
            "expected_share": e_share.values,
            "actual_share": a_share.values,
            "contribution": contrib.values,
        }
    )
    return float(contrib.sum()), detail


@dataclass
class DriftResult:
    monitor: str
    value: float | None
    warn_threshold: float | None
    breach_threshold: float | None
    status: str                  # OK | WARN | BREACH | NOT_MEASURABLE
    detail: pd.DataFrame = field(default_factory=pd.DataFrame)
    note: str = ""

    def to_row(self) -> dict[str, Any]:
        return {
            "monitor": self.monitor,
            "value": None if self.value is None else round(float(self.value), 4),
            "warn_threshold": self.warn_threshold,
            "breach_threshold": self.breach_threshold,
            "status": self.status,
            "note": self.note,
        }


class DriftMonitor:
    def __init__(self, config) -> None:
        self.config = config
        self.warn = float(config.get("psi_warn_threshold"))
        self.breach = float(config.get("psi_breach_threshold"))

    def run(
        self,
        *,
        train_matrix: pd.DataFrame,
        score_matrix: pd.DataFrame,
        scores: pd.Series,
        reference_scores: pd.Series | None = None,
        alert_volume: int = 0,
        expected_alert_volume: int | None = None,
        review_outcomes: pd.DataFrame | None = None,
    ) -> list[DriftResult]:
        results: list[DriftResult] = []

        # 1. population stability across the feature matrix
        psis = []
        rows = []
        for col in train_matrix.columns:
            psi, _ = population_stability_index(train_matrix[col], score_matrix[col])
            if not np.isnan(psi):
                psis.append(psi)
                rows.append({"feature": col, "psi": round(psi, 4)})
        worst = max(psis) if psis else float("nan")
        results.append(
            DriftResult(
                monitor="Population stability (max feature PSI)",
                value=worst,
                warn_threshold=self.warn,
                breach_threshold=self.breach,
                status=self._band(worst),
                detail=pd.DataFrame(rows).sort_values("psi", ascending=False) if rows else pd.DataFrame(),
                note=(
                    "PSI detects that the INPUT distribution moved between the training and "
                    "scoring periods. It does not say the model got worse. Expect a structural "
                    "breach on this dataset: the prior-history features (a member's or "
                    "provider's earlier claim count, spend and rates) are EXPANDING windows, so "
                    "they are near-empty at the start of the file and well populated later. "
                    "That is real drift in the input, and the honest consequence is that the "
                    "model gate fails on the drift criterion and the models stay in shadow. "
                    + ("Top drifting features: "
                       + ", ".join(f"{r['feature']} (PSI {r['psi']})" for r in
                                   sorted(rows, key=lambda x: -x['psi'])[:5])
                       if rows else "")
                ),
            )
        )

        # 2. score-distribution drift
        if reference_scores is not None and len(reference_scores):
            psi, detail = population_stability_index(reference_scores, scores)
            results.append(
                DriftResult(
                    monitor="Score-distribution drift (PSI)",
                    value=psi, warn_threshold=self.warn, breach_threshold=self.breach,
                    status=self._band(psi), detail=detail,
                    note="Reference distribution is the model's scores on the training period.",
                )
            )
        else:
            results.append(
                DriftResult(
                    monitor="Score-distribution drift (PSI)",
                    value=None, warn_threshold=self.warn, breach_threshold=self.breach,
                    status="NOT_MEASURABLE",
                    note="Requires a stored reference score distribution from a prior run. A "
                         "single-run artefact has none; it is created on the first run and "
                         "compared on the second.",
                )
            )

        # 3. alert volume
        if expected_alert_volume:
            ratio = alert_volume / max(expected_alert_volume, 1)
            status = "OK" if 0.5 <= ratio <= 2.0 else ("WARN" if 0.25 <= ratio <= 4.0 else "BREACH")
            results.append(
                DriftResult(
                    monitor="Alert volume vs expectation",
                    value=ratio, warn_threshold=2.0, breach_threshold=4.0, status=status,
                    note=f"{alert_volume} alerts against a governed expectation of "
                         f"{expected_alert_volume} (config/parameters.yaml).",
                )
            )
        else:
            results.append(
                DriftResult(
                    monitor="Alert volume vs expectation", value=float(alert_volume),
                    warn_threshold=None, breach_threshold=None, status="OK",
                    note="No expected_alert_volume is registered for this model.",
                )
            )

        # 4. review yield
        if review_outcomes is not None and not review_outcomes.empty:
            confirmed = int((review_outcomes["validated_category"] != "cleared").sum())
            yield_rate = confirmed / max(len(review_outcomes), 1)
            baseline = float(self.config.get("gate_statistical_min_review_yield"))
            results.append(
                DriftResult(
                    monitor="Review yield (confirmed ÷ reviewed)",
                    value=yield_rate, warn_threshold=baseline, breach_threshold=baseline / 2,
                    status="OK" if yield_rate >= baseline else ("WARN" if yield_rate >= baseline / 2 else "BREACH"),
                    note=f"{confirmed} confirmed of {len(review_outcomes)} reviewed.",
                )
            )
        else:
            results.append(
                DriftResult(
                    monitor="Review yield (confirmed ÷ reviewed)",
                    value=None, warn_threshold=None, breach_threshold=None,
                    status="NOT_MEASURABLE",
                    note="NOT_MEASURABLE_ON_THIS_DATASET until reviewers record dispositions. "
                         "Review yield is the primary signal of whether a model is still "
                         "earning its place; substituting a label-derived proxy here would be "
                         "exactly the leakage the artefact forbids.",
                )
            )
        return results

    def _band(self, value: float) -> str:
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return "NOT_MEASURABLE"
        if value >= self.breach:
            return "BREACH"
        if value >= self.warn:
            return "WARN"
        return "OK"

    @staticmethod
    def recommendation(results: list[DriftResult]) -> str:
        """What §9.8 asks an operator to do on a breach."""
        if any(r.status == "BREACH" for r in results):
            return (
                "BREACH — recommend RECALIBRATION, or the KILL SWITCH if the breach persists. "
                "Neither is applied automatically: deactivating a model is a governed action "
                "with a named owner, not something a monitor does on its own."
            )
        if any(r.status == "WARN" for r in results):
            return "WARN — monitor closely; schedule a recalibration review."
        return "OK — no drift action required."
