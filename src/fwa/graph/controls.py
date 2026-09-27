"""Network controls (Type N / S-N), manuscript §4.10, Appendix C NET-01..04.

These are separated from :mod:`fwa.engine.controls` only because they need the
graph service; the contract and the signal shape are identical. Each takes the
shared ``_sig`` signal builder so that every signal in the system is constructed
the same way regardless of which layer produced it.

The property §4.11 and §6.3 both insist on — **a network case sums each distinct
claim exactly once** — is implemented in
:func:`fwa.cases.exposure.network_exposure`, which takes a mapping keyed by
``claim_sk`` so double counting is structurally impossible rather than merely
avoided.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
import pandas as pd

from ..cases import exposure as _exp
from ..engine.evallib import period_bucket
from ..statistical.robust import robust_stats

__all__ = [
    "net_01_r01_referral_concentration",
    "net_02_r02_dense_member_circulation",
    "net_02_r03_synchronized_billing",
    "net_03_r01_repeated_high_value_dyad",
    "net_04_r02_originator_provider_concentration",
    "net_04_r03_early_tenure_cluster",
]

_INFERRED_EDGE_NOTE = (
    "The agent→provider edge is INFERRED from co-occurrence on the same claim. It is not an "
    "observed clinical referral — the claim extract has no referral records — so this is a "
    "distribution-channel concentration and carries less evidentiary weight than the "
    "catalogue control it implements."
)


def _claims(ctx) -> pd.DataFrame:
    df = ctx.claims
    return df[df["tenant_id"] == ctx.tenant_id] if "tenant_id" in df.columns else df


def net_01_r01_referral_concentration(ctx, control, _sig: Callable) -> list[Any]:
    """NET-01-R01 — top-recipient share and HHI for an originator."""
    df = _claims(ctx)
    absolute_floor = float(ctx.cfg("referral_concentration_threshold"))
    residual_cut = float(ctx.cfg("referral_concentration_peer_residual_threshold"))
    min_claims = int(ctx.cfg("min_entity_opportunities"))

    # Appendix C compares the top-recipient share to PEERS. The absolute floor is
    # evaluated and reported, but the executing test is the peer-relative one:
    # on the claim extract no agent comes near the absolute floor, because 20 agents
    # spread across 200 providers.
    profile = {}
    for agent, sub in df.groupby("agent_id", sort=False):
        if len(sub) < min_claims:
            continue
        counts = sub["provider_sk"].value_counts()
        shares = counts / counts.sum()
        profile[str(agent)] = (float(shares.iloc[0]), float((shares ** 2).sum()), counts)
    if not profile:
        return []
    peer_stats = robust_stats([v[0] for v in profile.values()])

    out = []
    for agent, sub in df.groupby("agent_id", sort=False):
        if str(agent) not in profile:
            continue
        top_share, hhi, counts = profile[str(agent)]
        resid = peer_stats.z(top_share)
        meets_absolute = top_share > absolute_floor
        meets_peer = resid is not None and resid > residual_cut
        if not (meets_absolute or meets_peer):
            continue
        top_provider = str(counts.index[0])
        # Exposure is the CONCENTRATED FLOW — the claims this agent sent to the
        # single provider the finding is about — not the agent's whole book.
        # §4.11 is deliberately conservative about what may be called exposure.
        concentrated = sub[sub["provider_sk"].astype(str) == top_provider]
        claim_amounts = dict(
            zip(concentrated["claim_sk"].astype(str), concentrated["gross_amount_aed"].astype(float))
        )
        out.append(
            _sig(
                ctx, control, subject_type="agent", subject_id=agent,
                fact_key=f"concentration:{agent}",
                claim_ids=list(claim_amounts),
                event_time=sub["service_date"].max(),
                period=period_bucket(sub["service_date"].max()),
                confidence=0.6,
                evidence={
                    "agent_id": agent,
                    "top_provider_sk": top_provider,
                    "top_provider_share": round(top_share, 3),
                    "hhi": round(hhi, 4),
                    "peer_median_top_share": round(peer_stats.median, 4),
                    "peer_robust_residual": None if resid is None else round(resid, 2),
                    "peer_residual_threshold": residual_cut,
                    "absolute_floor": absolute_floor,
                    "absolute_floor_met": meets_absolute,
                    "limb_fired": "peer_relative" if meets_peer else "absolute",
                    "claim_count": int(len(sub)),
                    "distinct_providers": int(counts.size),
                    "edge_type": "agent_provider (inferred)",
                    "peer_comparison": "by agent portfolio size only — specialty and geography "
                                       "are unavailable",
                    "plain_language": (
                        f"Agent {agent} sends {top_share:.0%} of {len(sub)} claims to a single "
                        f"provider ({top_provider}), across {counts.size} provider(s) in total."
                    ),
                    "proxy_note": _INFERRED_EDGE_NOTE,
                },
                exposure=_exp.network_exposure(claim_amounts),
            )
        )
    return out


def net_02_r02_dense_member_circulation(ctx, control, _sig: Callable) -> list[Any]:
    """NET-02-R02 — abnormally dense community with low external flow."""
    graph = ctx.graph
    if graph is None or graph.community_stats.empty:
        return []
    min_size = int(ctx.cfg("community_min_size"))
    absolute_floor = float(ctx.cfg("community_density_threshold"))
    peer_multiple = float(ctx.cfg("community_density_peer_multiple"))
    external_max = float(ctx.cfg("community_external_flow_max"))
    min_members = int(ctx.cfg("community_min_distinct_members"))

    stats = graph.community_stats.copy()
    eligible = stats[stats["size"] >= min_size]
    if eligible.empty:
        return []
    # SIZE-MATCHED peer density: Appendix C's exclusion column requires
    # "specialty/geography/SIZE matched communities", and in a bipartite claims
    # graph density falls roughly as 1/size, so an unmatched comparison would
    # flag every small community and no large one.
    eligible = eligible.assign(
        size_matched_density=eligible.groupby("size")["internal_density"].transform("median")
    )

    out = []
    for row in eligible.itertuples(index=False):
        dense = row.internal_density > peer_multiple * row.size_matched_density
        closed = row.external_ratio < external_max
        circulating = row.member_count >= min_members and row.provider_count >= min_members
        if not (dense and closed and circulating):
            continue
        claim_amounts = _amounts_for(ctx, row.claim_ids)
        out.append(
            _sig(
                ctx, control, subject_type="community", subject_id=str(row.community_id),
                fact_key=f"community:{row.community_id}",
                claim_ids=list(claim_amounts),
                event_time=None,
                period="",
                confidence=0.55,
                evidence={
                    "community_id": str(row.community_id),
                    "size": int(row.size),
                    "internal_density": float(row.internal_density),
                    "size_matched_peer_density": round(float(row.size_matched_density), 4),
                    "density_peer_multiple": peer_multiple,
                    "absolute_floor": absolute_floor,
                    "absolute_floor_met": bool(row.internal_density > absolute_floor),
                    "external_flow_max": external_max,
                    "external_edges": int(row.external_edges),
                    "external_ratio": float(row.external_ratio),
                    "member_count": int(row.member_count),
                    "provider_count": int(row.provider_count),
                    "agent_count": int(row.agent_count),
                    "claim_count": int(row.claim_count),
                    "snapshot_window": getattr(row, "window", ""),
                    "window_start": getattr(row, "window_start", ""),
                    "window_end": getattr(row, "window_end", ""),
                    "time_bounded_note": "Communities are detected on WEEKLY, time-bounded "
                                         "snapshots. A community that is dense across "
                                         "three years is usually a busy hospital; one that is "
                                         "dense inside a single week is a different observation.",
                    "each_claim_counted_once": True,
                    "peer_matching": "SIZE-matched only. Specialty and geography matching are "
                                     "unavailable in the claim extract.",
                    "plain_language": (
                        f"Community {row.community_id} contains {row.member_count} member(s), "
                        f"{row.provider_count} provider(s) and {row.agent_count} agent(s) with an "
                        f"internal edge density of {row.internal_density:.0%} against a "
                        f"size-matched peer median of {row.size_matched_density:.0%}, with "
                        f"{row.external_ratio:.0%} of its edges leaving the community."
                    ),
                },
                exposure=_exp.network_exposure(claim_amounts),
            )
        )
    return out


def net_02_r03_synchronized_billing(ctx, control, _sig: Callable) -> list[Any]:
    """NET-02-R03 — shared date and near-identical amounts inside a community."""
    graph = ctx.graph
    if graph is None or graph.community_stats.empty:
        return []
    tol = float(ctx.cfg("duplicate_amount_tolerance"))
    min_size = int(ctx.cfg("community_min_size"))
    df = _claims(ctx).set_index("claim_sk")
    out = []
    for row in graph.community_stats.itertuples(index=False):
        if row.size < min_size or row.claim_count < 2:
            continue
        ids = [c for c in row.claim_ids if c in df.index]
        if len(ids) < 2:
            continue
        sub = df.loc[ids]
        for day, grp in sub.groupby(pd.to_datetime(sub["service_date"]).dt.date, sort=False):
            if len(grp) < 2 or grp["member_sk"].nunique() < 2:
                continue
            amounts = grp["gross_amount_aed"].astype(float)
            spread = float(amounts.max() - amounts.min())
            if amounts.max() == 0 or spread / float(amounts.max()) > tol:
                continue
            claim_amounts = dict(zip(grp.index.astype(str), amounts))
            out.append(
                _sig(
                    ctx, control, subject_type="community", subject_id=str(row.community_id),
                    fact_key=f"sync:{row.community_id}:{day}",
                    claim_ids=list(claim_amounts),
                    event_time=day,
                    period=period_bucket(day),
                    confidence=0.5,
                    evidence={
                        "community_id": str(row.community_id),
                        "synchronised_claims": list(claim_amounts),
                        "shared_date": str(day),
                        "amount_spread_aed": round(spread, 2),
                        "amount_tolerance": tol,
                        "distinct_members": int(grp["member_sk"].nunique()),
                        "distinct_providers": int(grp["provider_sk"].nunique()),
                        "limbs_available": ["time", "amount"],
                        "limbs_unavailable": ["code signature (no activity codes)",
                                              "resubmission signature (no claim lineage)"],
                        "proxy_note": "Only two of the catalogue's four synchronisation limbs can be "
                                      "evaluated. The catalogue already treats this control as "
                                      "'supporting evidence' with a MONITOR_ONLY disposition.",
                    },
                    exposure=_exp.network_exposure(claim_amounts),
                )
            )
    return out


def net_03_r01_repeated_high_value_dyad(ctx, control, _sig: Callable) -> list[Any]:
    """NET-03-R01 — a member-provider pair with abnormal frequency and value."""
    df = _claims(ctx)
    dyads = df.groupby(["member_sk", "provider_sk"]).agg(
        claim_count=("claim_sk", "count"),
        total_aed=("gross_amount_aed", "sum"),
    ).reset_index()
    repeat = dyads[dyads["claim_count"] >= 2]
    if repeat.empty:
        return []
    stats = robust_stats(dyads["total_aed"])
    if stats.scale <= 0:
        return []
    threshold = float(ctx.cfg("robust_residual_flag_threshold"))
    out = []
    for row in repeat.itertuples(index=False):
        resid = stats.z(float(row.total_aed))
        if resid is None or resid <= threshold:
            continue
        sub = df[(df["member_sk"] == row.member_sk) & (df["provider_sk"] == row.provider_sk)]
        claim_amounts = dict(zip(sub["claim_sk"].astype(str), sub["gross_amount_aed"].astype(float)))
        out.append(
            _sig(
                ctx, control, subject_type="member", subject_id=row.member_sk,
                fact_key=f"dyad:{row.member_sk}:{row.provider_sk}",
                claim_ids=list(claim_amounts),
                event_time=sub["service_date"].max(),
                period=period_bucket(sub["service_date"].max()),
                confidence=0.75,
                evidence={
                    "member_sk": row.member_sk,
                    "provider_sk": row.provider_sk,
                    "dyad_claim_count": int(row.claim_count),
                    "dyad_total_aed": round(float(row.total_aed), 2),
                    "dyad_value_median_aed": round(stats.median, 2),
                    "robust_residual": round(resid, 2),
                    "threshold": threshold,
                    "diagnoses": sorted(sub["diagnosis_primary"].unique().tolist()),
                    "limbs_evaluated": ["frequency", "value"],
                    "limbs_unavailable": ["benefit exhaustion (no benefit limits in the claim extract)"],
                    "plain_language": (
                        f"This member and provider have {row.claim_count} claims together totalling "
                        f"AED {float(row.total_aed):,.0f}, against a median pair total of "
                        f"AED {stats.median:,.0f}."
                    ),
                },
                exposure=_exp.network_exposure(claim_amounts),
            )
        )
    return out


def net_04_r02_originator_provider_concentration(ctx, control, _sig: Callable) -> list[Any]:
    """NET-04-R02 — an originator's members concentrated into a small provider set."""
    df = _claims(ctx)
    min_claims = int(ctx.cfg("min_entity_opportunities"))
    agg = df.groupby("agent_id").agg(
        claim_count=("claim_sk", "count"),
        distinct_providers=("provider_sk", "nunique"),
        distinct_members=("member_sk", "nunique"),
    ).reset_index()
    eligible = agg[agg["claim_count"] >= min_claims]
    if eligible.empty:
        return []
    # Expected distinct providers for a portfolio of this size, from the
    # observed relationship across all agents. Deliberately empirical: there is
    # no catchment or specialty model to predict it from.
    ratio = (eligible["distinct_providers"] / eligible["claim_count"]).median()
    stats = robust_stats(eligible["distinct_providers"] / eligible["claim_count"])
    if stats.scale <= 0:
        return []
    out = []
    cut = -float(ctx.cfg("referral_concentration_peer_residual_threshold"))
    for row in eligible.itertuples(index=False):
        observed_ratio = row.distinct_providers / row.claim_count
        resid = stats.z(observed_ratio)
        if resid is None or resid >= cut:
            continue  # only *low* provider diversity is adverse
        sub = df[df["agent_id"] == row.agent_id]
        counts = sub["provider_sk"].value_counts()
        # The suspect flow is the share going to the providers this agent is
        # concentrated into, not every claim they originated.
        top_providers = set(counts.head(max(1, int(row.distinct_providers * 0.2))).index)
        concentrated = sub[sub["provider_sk"].isin(top_providers)]
        claim_amounts = dict(
            zip(concentrated["claim_sk"].astype(str), concentrated["gross_amount_aed"].astype(float))
        )
        out.append(
            _sig(
                ctx, control, subject_type="agent", subject_id=row.agent_id,
                fact_key=f"originator_concentration:{row.agent_id}",
                claim_ids=list(claim_amounts),
                event_time=sub["service_date"].max(),
                period=period_bucket(sub["service_date"].max()),
                confidence=0.6,
                evidence={
                    "agent_id": row.agent_id,
                    "distinct_providers": int(row.distinct_providers),
                    "expected_distinct_providers": round(ratio * row.claim_count, 1),
                    "claim_count": int(row.claim_count),
                    "distinct_members": int(row.distinct_members),
                    "concentration_ratio": round(observed_ratio, 4),
                    "robust_residual": round(resid, 2),
                    "top_provider_share": round(float(counts.iloc[0] / counts.sum()), 3),
                    "employer_limb": "UNAVAILABLE — no employer identifiers in the claim extract, so only "
                                     "the agent limb of this control runs and the employer-clinic "
                                     "exclusion cannot be applied",
                    "proxy_note": _INFERRED_EDGE_NOTE,
                },
                exposure=_exp.network_exposure(claim_amounts),
            )
        )
    return out


def net_04_r03_early_tenure_cluster(ctx, control, _sig: Callable) -> list[Any]:
    """NET-04-R03 — early-tenure high-value claims clustering by originator."""
    df = _claims(ctx)
    early_days = int(ctx.cfg("early_tenure_days"))
    min_claims = int(ctx.cfg("min_entity_opportunities"))
    max_width = float(ctx.cfg("max_posterior_width"))
    interval_mass = float(ctx.cfg("posterior_interval_mass"))

    peer_median = df.groupby("diagnosis_primary")["gross_amount_aed"].transform("median")
    work = df.assign(
        early_high=(
            (df["days_since_policy_start"] < early_days)
            & (df["gross_amount_aed"] > peer_median)
        ).astype(int)
    )
    agg = work.groupby("agent_id").agg(
        events=("early_high", "sum"), opportunities=("claim_sk", "count")
    ).reset_index()
    eligible = agg[agg["opportunities"] >= min_claims]
    if eligible.empty:
        return []

    from ..statistical.shrinkage import fit_beta_prior, shrink_rate

    prior = fit_beta_prior(eligible["events"], eligible["opportunities"])
    overall = float(eligible["events"].sum() / max(eligible["opportunities"].sum(), 1))

    out = []
    for row in eligible.itertuples(index=False):
        sr = shrink_rate(str(row.agent_id), row.events, row.opportunities, prior,
                         interval_mass=interval_mass, max_posterior_width=max_width)
        if sr.excluded_from_ranking or sr.shrunk_rate <= max(overall * 1.5, overall + 0.05):
            continue
        sub = work[(work["agent_id"] == row.agent_id) & (work["early_high"] == 1)]
        claim_amounts = dict(zip(sub["claim_sk"].astype(str), sub["gross_amount_aed"].astype(float)))
        out.append(
            _sig(
                ctx, control, subject_type="agent", subject_id=row.agent_id,
                fact_key=f"early_tenure_cluster:{row.agent_id}",
                claim_ids=list(claim_amounts),
                event_time=sub["service_date"].max() if not sub.empty else None,
                period=period_bucket(sub["service_date"].max()) if not sub.empty else "",
                confidence=0.6,
                evidence={
                    "agent_id": row.agent_id,
                    "early_tenure_high_value_count": int(row.events),
                    "agent_claim_count": int(row.opportunities),
                    "agent_rate": round(sr.observed_rate or 0.0, 4),
                    "overall_rate": round(overall, 4),
                    "shrunk_rate": round(sr.shrunk_rate, 4),
                    "posterior_interval": [round(sr.posterior_low, 4), round(sr.posterior_high, 4)],
                    "early_tenure_days": early_days,
                    "shrinkage_explanation": sr.explain(),
                    "never_individual_denial_notice": (
                        "Appendix C, NET-04-R03: 'Review, NEVER INDIVIDUAL DENIAL.' This is a "
                        "distribution-channel review signal. No individual claim may be denied on it."
                    ),
                    "proxy_note": "'Beyond morbidity expectation' is approximated by the "
                                  "diagnosis-peer amount median; no morbidity model exists here.",
                },
                exposure=_exp.network_exposure(claim_amounts),
            )
        )
    return out


def _amounts_for(ctx, claim_ids) -> dict[str, float]:
    df = _claims(ctx).set_index("claim_sk")
    ids = [c for c in claim_ids if c in df.index]
    return dict(zip(ids, df.loc[ids, "gross_amount_aed"].astype(float)))
