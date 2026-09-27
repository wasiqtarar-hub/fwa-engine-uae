"""Model controls (Type M) — ANL-01-R03 and ANL-01-R04.

Both are ``MONITOR_ONLY`` and structurally cannot be anything else: §3.3
forbids a model control from declaring ``REJECT`` or ``REPRICE``, the contract
validator refuses to load one that tries, and the catalogue independently
chooses the weakest remaining disposition for both.

ANL-01-R03's trigger is deliberately *incremental*: "Benchmarked model flags
entity AND ADDS LIFT BEYOND R01/R02". An entity the transparent composite has
already surfaced is not raised again by the model — the model's job is to catch
what the composite misses, and a model signal on an entity the composite
already found adds queue volume without adding evidence.
"""

from __future__ import annotations

from typing import Any, Callable

import pandas as pd

from ..cases import exposure as _exp
from ..engine.evallib import period_bucket

__all__ = ["anl_01_r03_unsupervised_incremental", "anl_01_r04_novel_cluster"]


def anl_01_r03_unsupervised_incremental(ctx, control, _sig: Callable) -> list[Any]:
    layer = ctx.models
    if layer is None or layer.anomaly is None or not layer.anomaly.models:
        return []

    df = ctx.claims
    df = df[df["tenant_id"] == ctx.tenant_id] if "tenant_id" in df.columns else df
    claim_index = df.set_index("claim_sk")

    # Entities the transparent composite (R01) and the changepoint control
    # (R02) have already surfaced. The model only speaks about the remainder.
    already: set[str] = set()
    if ctx.composite is not None:
        results, _ = ctx.composite
        threshold = float(ctx.cfg("composite_flag_threshold"))
        already |= {r.provider_sk for r in results
                    if not r.insufficient_evidence and r.score > threshold}

    capacity = int(ctx.cfg("alerts_per_reviewer_per_day")) * int(ctx.cfg("reviewer_count"))
    provider_scores = layer.provider_scores()

    out: list[Any] = []
    for model_name, scores in layer.anomaly.models.items():
        prov = provider_scores.get(model_name)
        if prov is None or prov.empty:
            continue
        ranked = prov.sort_values(ascending=False)
        threshold_value = float(ranked.iloc[min(capacity, len(ranked)) - 1])
        percentiles = ranked.rank(pct=True)

        gate = layer.gate_verdicts.get(model_name)
        gate_note = gate.summary if gate else (
            "The promotion gate has not been run for this model in this pass; the control "
            "therefore runs in shadow and its signals are MONITOR_ONLY."
        )

        for provider, score in ranked.head(capacity).items():
            provider = str(provider)
            if provider in already:
                continue  # declared exclusion: already_flagged_by_transparent_composite
            sub = df[df["provider_sk"] == provider]
            if sub.empty:
                continue
            top_claim = sub.loc[sub["gross_amount_aed"].idxmax(), "claim_sk"]
            contributions = layer.explain_claim(model_name, str(top_claim), top_n=5)
            out.append(
                _sig(
                    ctx, control, subject_type="provider", subject_id=provider,
                    fact_key=f"model:{model_name}:{provider}",
                    claim_ids=sub["claim_sk"].tolist(),
                    event_time=sub["service_date"].max(),
                    period=period_bucket(sub["service_date"].max()),
                    confidence=0.5,
                    evidence={
                        "provider_sk": provider,
                        "model_name": model_name,
                        "model_version": scores.version,
                        "score": round(float(score), 4),
                        "score_percentile": round(float(percentiles.get(provider, float("nan"))), 4),
                        "capacity_threshold": round(threshold_value, 4),
                        "capacity_basis": (
                            f"{ctx.cfg('alerts_per_reviewer_per_day')} alerts/reviewer/day × "
                            f"{ctx.cfg('reviewer_count')} reviewers = {capacity} alerts.: "
                            f"calibrate to reviewer capacity and expected value, NOT to a fixed "
                            f"significance level."
                        ),
                        "shap_top_features_original_units": contributions,
                        "explained_claim_sk": str(top_claim),
                        "incremental_over_composite": True,
                        "training_period": layer.anomaly.coverage().get("split_date"),
                        "scoring_period": f"after {layer.anomaly.coverage().get('split_date')}",
                        "entity_isolation": layer.anomaly.coverage().get("provider_overlap") == 0,
                        "feature_count": len(scores.feature_columns),
                        "promotion_gate_verdict": gate_note,
                        "safety_boundary_notice": (
                            "This is a MODEL SIGNAL. It cannot deny, reprice or change any "
                            "disposition, at any score, under any configuration. It is a "
                            "lead for a human to assess."
                        ),
                    },
                    exposure=_exp.model_only_exposure(
                        float(sub["gross_amount_aed"].sum()), model_name
                    ),
                    extra_versions={
                        "model": f"{model_name}@{scores.version}",
                        "feature_set": ",".join(scores.feature_columns[:6]) + "…",
                    },
                )
            )
    return out


def anl_01_r04_novel_cluster(ctx, control, _sig: Callable) -> list[Any]:
    layer = ctx.models
    if layer is None or not layer.clusters:
        return []
    df = ctx.claims
    df = df[df["tenant_id"] == ctx.tenant_id] if "tenant_id" in df.columns else df

    out: list[Any] = []
    for cluster in layer.clusters:
        sub = df[df["provider_sk"].isin(cluster.members)]
        out.append(
            _sig(
                ctx, control, subject_type="cluster", subject_id=f"CL-{cluster.cluster_id:03d}",
                fact_key=f"novel_cluster:{cluster.cluster_id}",
                claim_ids=sub["claim_sk"].tolist(),
                event_time=sub["service_date"].max() if not sub.empty else None,
                period=period_bucket(sub["service_date"].max()) if not sub.empty else "",
                confidence=cluster.coherence,
                evidence={
                    "cluster_id": cluster.cluster_id,
                    "cluster_size": cluster.size,
                    "members": cluster.members,
                    "top_residual_features": cluster.top_features,
                    "coherence": round(cluster.coherence, 3),
                    "aggregate_exposure_aed": round(cluster.aggregate_exposure_aed, 2),
                    "candidate_typology_name": cluster.candidate_typology_name,
                    "unvalidated_hypothesis_notice": cluster.note,
                    "next_step": (
                        "Quarterly typology review. If an analyst judges the pattern "
                        "coherent, it is drafted as an atomic control which enters the registry "
                        "in SHADOW and requires a separate approver."
                    ),
                },
                exposure=_exp.model_only_exposure(
                    cluster.aggregate_exposure_aed, "novel-cluster discovery"
                ),
            )
        )
    return out
