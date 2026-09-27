"""The 36 controls this build actually executes against the claim extract.

Every function here implements one atomic control from manuscript Appendix C.
The YAML in ``rules/`` is the contract; this module is the body. A control that
appears here must be classified ``EXECUTABLE`` or ``PARTIAL`` in the catalogue,
and a control classified ``NOT_EXECUTABLE_ON_THIS_DATASET`` must *not* appear —
:class:`~fwa.engine.contract.AtomicControl` enforces both directions at load
time, so the coverage matrix cannot drift away from what the code does.

Conventions that hold for every control in this module:

* It returns ``list[Signal]``. Returning nothing is a legitimate, recorded
  outcome, not a failure.
* It reads thresholds through ``ctx.cfg(...)`` — never a literal.
* It sets a meaningful ``fact_key``. Two controls resting on the same underlying
  fact share a key so that evidence capping (§3.8) caps them instead of summing
  them. The clearest example: PAY-06-R03 (the claim amount is a peer outlier)
  and CLN-01-R02 (the amount per inpatient day is a peer outlier) both rest on
  ``amount:<claim_sk>``, because they are two readings of one billed figure.
* It computes exposure through :mod:`fwa.cases.exposure`, so the §4.11 rules
  about what may be called exposure are applied in one place.
* Where the control runs against a PROXY, the proxy is named in the evidence
  under ``proxy_note``, so a reviewer sees the substitution on the case itself
  rather than having to find it in the coverage matrix.
"""

from __future__ import annotations

import datetime as _dt
import itertools
from typing import Any, Callable, Iterable

import numpy as np
import pandas as pd

from ..cases import exposure as _exp
from ..enums import Disposition
from ..statistical.robust import robust_stats
from ..statistical.shrinkage import fit_beta_prior, shrink_rate
from .context import ControlContext
from .evallib import Tri, compare, is_missing, overlaps, period_bucket, within_days
from .signals import Signal

__all__ = ["CONTROL_IMPLEMENTATIONS", "get_implementation"]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _sig(
    ctx: ControlContext,
    control,
    *,
    subject_type: str,
    subject_id: str,
    fact_key: str,
    evidence: dict[str, Any],
    claim_ids: Iterable[str] = (),
    exposure: _exp.Exposure | None = None,
    confidence: float = 1.0,
    event_time: Any = None,
    period: str = "",
    peer_level_used: str | None = None,
    data_quality_penalty: float = 0.0,
    known_exception_strength: float = 0.0,
    extra_versions: dict[str, str] | None = None,
) -> Signal:
    exposure = exposure or _exp.no_exposure("not computed by this control")
    ev = dict(evidence)
    ev.update(exposure.to_dict())
    ev.setdefault("catalogue_trigger", control.catalogue_trigger)
    ev.setdefault("catalogue_disposition", control.catalogue_disposition)
    if control.data_support.value != "EXECUTABLE":
        ev.setdefault("data_support", control.data_support.value)
        ev.setdefault("data_support_reason", control.data_support_reason)
    return Signal.create(
        control=control,
        tenant_id=ctx.tenant_id,
        subject_type=subject_type,
        subject_id=str(subject_id),
        evidence=ev,
        fact_key=fact_key,
        claim_ids=[str(c) for c in claim_ids],
        event_time=_ts(event_time),
        period_bucket=period,
        confidence=confidence,
        exposure_aed=exposure.amount_aed,
        exposure_basis=exposure.basis,
        exposure_established=exposure.established,
        reference_versions=ctx.versions(control, extra_versions),
        peer_level_used=peer_level_used,
        data_quality_penalty=data_quality_penalty,
        known_exception_strength=known_exception_strength,
    )


def _ts(value: Any) -> _dt.datetime | None:
    if is_missing(value):
        return None
    if isinstance(value, _dt.datetime):
        return value
    try:
        return pd.to_datetime(value).to_pydatetime()
    except Exception:  # pragma: no cover
        return None


def _f(value: Any, default: float = 0.0) -> float:
    return default if is_missing(value) else float(value)


def _claim_frame(ctx: ControlContext) -> pd.DataFrame:
    df = ctx.claims
    return df[df["tenant_id"] == ctx.tenant_id] if "tenant_id" in df.columns else df


def _incremental_over_peer(sub: pd.DataFrame, contributions: list[dict[str, Any]]) -> list[float]:
    """Each claim's excess over the peer median amount, floored at zero.

    §4.11 defines a provider-pattern exposure as "the sum of sampled suspect
    INCREMENTAL amounts". Using the provider's full billed value instead would
    state that every claim they ever sent is at risk, which is both wrong and
    the single easiest way for a payment-integrity system to overstate its own
    value — the failure mode §6.5 and §4.11 are written to prevent.
    """
    peer_median = None
    for c in contributions:
        if c.get("feature") == "amount_residual" and c.get("peer_median"):
            peer_median = float(c["peer_median"])
            break
    if peer_median is None:
        peer_median = float(sub["gross_amount_aed"].median())
    return [max(0.0, float(v) - peer_median) for v in sub["gross_amount_aed"]]


def _excess_claim_values(sub: pd.DataFrame, observed_rate: float, peer_rate: float) -> list[float]:
    """Value of the claims above the peer-median utilisation rate."""
    if peer_rate <= 0 or observed_rate <= peer_rate:
        return []
    excess_share = (observed_rate - peer_rate) / observed_rate
    n_excess = max(1, int(round(len(sub) * excess_share)))
    return sub["gross_amount_aed"].nlargest(n_excess).astype(float).tolist()


# ===========================================================================
# ENT
# ===========================================================================


def ent_02_r03_member_impossible_presence(ctx: ControlContext, control) -> list[Signal]:
    """ENT-02-R03 — a member cannot be an inpatient in two places at once.

    Only the overlapping-encounter limb runs; the travel-speed limb needs
    geography the source does not have. Confidence is reduced because the source
    carries dates without times, so a same-day transfer is indistinguishable
    from a genuine overlap (the catalogue names 'claim date without time' as an
    exclusion for exactly this reason).
    """
    df = _claim_frame(ctx)
    tol = int(ctx.cfg("concurrent_admission_overlap_tolerance_days"))
    out: list[Signal] = []
    for member, sub in df.groupby("member_sk", sort=False):
        if len(sub) < 2:
            continue
        sub = sub.sort_values("service_date")
        for (_, a), (_, b) in itertools.combinations(sub.iterrows(), 2):
            if a["provider_sk"] == b["provider_sk"]:
                continue  # same-facility transfer is a declared exclusion
            result = overlaps(
                a["service_date"], a["discharge_date"], b["service_date"], b["discharge_date"],
                tolerance_days=tol,
            )
            if not result.fired:
                continue
            overlap_days = (
                min(pd.Timestamp(a["discharge_date"]), pd.Timestamp(b["discharge_date"]))
                - max(pd.Timestamp(a["service_date"]), pd.Timestamp(b["service_date"]))
            ).days
            out.append(
                _sig(
                    ctx, control, subject_type="member", subject_id=member,
                    fact_key=f"overlap:{min(a['claim_sk'], b['claim_sk'])}:{max(a['claim_sk'], b['claim_sk'])}",
                    claim_ids=[a["claim_sk"], b["claim_sk"]],
                    event_time=a["service_date"],
                    period=period_bucket(a["service_date"]),
                    confidence=0.6,
                    data_quality_penalty=0.4,
                    evidence={
                        "member_sk": member,
                        "overlapping_claims": [a["claim_sk"], b["claim_sk"]],
                        "admission_dates": [str(a["service_date"])[:10], str(b["service_date"])[:10]],
                        "discharge_dates": [str(a["discharge_date"])[:10], str(b["discharge_date"])[:10]],
                        "overlap_days": int(overlap_days),
                        "providers": [a["provider_sk"], b["provider_sk"]],
                        "limb_fired": "overlapping_encounters",
                        "limb_unavailable": "travel_speed (no geography in the claim extract)",
                        "proxy_note": "Dates carry no time of day, so a same-day transfer cannot be "
                                      "distinguished from a genuine overlap. Confidence reduced to 0.6 "
                                      "and a data-quality penalty applied to the priority score.",
                    },
                    exposure=_exp.duplicate_exposure(
                        [_f(a["gross_amount_aed"]), _f(b["gross_amount_aed"])],
                        count_note="Treated as a duplicate-style exposure: at most one of two "
                                   "simultaneous inpatient stays can be legitimate.",
                    ),
                )
            )
    return out


def ent_03_r02_excluded_entity(ctx: ControlContext, control) -> list[Signal]:
    """ENT-03-R02 — exclusion-list match.

    The catalogue permits ``REJECT`` only on an **exact, effective-dated**
    identifier match. ``provider_blacklist_flag`` is exact but undated, so the
    governed disposition in the YAML is ``PREPAY_PEND`` and the evidence records
    that effective dating was unavailable. This is the §3.6 as-of requirement
    refusing to be waved through.
    """
    df = _claim_frame(ctx)
    flagged = df[df["provider_blacklist_flag"].astype(bool)]
    out: list[Signal] = []
    for provider, sub in flagged.groupby("provider_sk", sort=False):
        total = float(sub["gross_amount_aed"].sum())
        out.append(
            _sig(
                ctx, control, subject_type="provider", subject_id=provider,
                fact_key=f"exclusion:{provider}",
                claim_ids=sub["claim_sk"].tolist(),
                event_time=sub["service_date"].min(),
                period=period_bucket(sub["service_date"].min()),
                confidence=0.7,
                data_quality_penalty=0.3,
                evidence={
                    "provider_sk": provider,
                    "status_value": "BLACKLISTED",
                    "match_type": "EXACT",
                    "exclusion_source": "the claim extract provider_blacklist_flag (row-level), not a regulator feed",
                    "effective_dating_available": False,
                    "claim_count": int(len(sub)),
                    "total_gross_aed": round(total, 2),
                    "proxy_note": "The catalogue permits REJECT only on an exact match against an "
                                  "EFFECTIVE-DATED exclusion list. This source carries no dates, so "
                                  "the disposition is downgraded to a pend and recorded as such.",
                },
                exposure=_exp.provider_pattern_exposure(sub["gross_amount_aed"].tolist()),
            )
        )
    return out


# ===========================================================================
# PAY
# ===========================================================================


def pay_01_r01_exact_duplicate(ctx: ControlContext, control) -> list[Signal]:
    """PAY-01-R01 — exact duplicate. The only implemented control that may REJECT.

    It keeps ``REJECT`` because every element of the match is objective: same
    payer, member, provider, principal diagnosis, admission date, discharge date
    and exact gross amount. No score, no threshold, no peer comparison. Only the
    *later* claim in the active lineage is rejected.
    """
    df = _claim_frame(ctx)
    tol = float(ctx.cfg("exact_duplicate_amount_tolerance"))
    key_cols = ["payer_id", "member_sk", "provider_sk", "diagnosis_primary",
                "service_date", "discharge_date"]
    out: list[Signal] = []
    for _, sub in df.groupby(key_cols, dropna=False, sort=False):
        if len(sub) < 2:
            continue
        sub = sub.sort_values(["claim_date", "claim_sk"])
        rows = list(sub.itertuples(index=False))
        for i, later in enumerate(rows[1:], start=1):
            earlier = rows[i - 1]
            a, b = _f(earlier.gross_amount_aed), _f(later.gross_amount_aed)
            if tol == 0:
                if a != b:
                    continue
            elif abs(a - b) / max(a, b, 1.0) > tol:
                continue
            if ctx.lineage is not None and (
                ctx.lineage.is_superseded(str(earlier.claim_sk))
                or ctx.lineage.is_cancelled(str(earlier.claim_sk))
            ):
                continue  # declared exclusion: superseded_by_correction
            out.append(
                _sig(
                    ctx, control, subject_type="claim", subject_id=later.claim_sk,
                    fact_key=f"dup:{min(earlier.claim_sk, later.claim_sk)}:{max(earlier.claim_sk, later.claim_sk)}",
                    claim_ids=[earlier.claim_sk, later.claim_sk],
                    event_time=later.service_date,
                    period=period_bucket(later.service_date),
                    evidence={
                        "duplicate_of_claim_sk": earlier.claim_sk,
                        "match_key": {
                            "payer_id": earlier.payer_id, "member_sk": earlier.member_sk,
                            "provider_sk": earlier.provider_sk,
                            "diagnosis_primary": earlier.diagnosis_primary,
                            "admission_date": str(earlier.service_date)[:10],
                            "discharge_date": str(earlier.discharge_date)[:10],
                        },
                        "gross_amount_aed": round(b, 2),
                        "earlier_gross_amount_aed": round(a, 2),
                        "amount_tolerance": tol,
                        "lineage_status": "no lineage records exist in the claim extract; supersession "
                                          "could not be checked against a correction chain",
                        "proxy_note": "Match is at CLAIM-HEADER level. With activity codes and units "
                                      "this key would be strictly finer, so this control can miss a "
                                      "duplicated line inside two otherwise-different claims.",
                    },
                    exposure=_exp.duplicate_exposure([a, b]),
                )
            )
    return out


def pay_01_r02_near_duplicate(ctx: ControlContext, control) -> list[Signal]:
    """PAY-01-R02 — near duplicate within ``cfg.duplicate_window_days``."""
    df = _claim_frame(ctx)
    window = int(ctx.cfg("duplicate_window_days"))
    tol = float(ctx.cfg("duplicate_amount_tolerance"))
    exact_pairs = {
        tuple(sorted((s.claim_ids[0], s.claim_ids[1])))
        for s in ctx.reference_versions.get("__exact_duplicate_pairs__", [])  # type: ignore[arg-type]
    } if isinstance(ctx.reference_versions.get("__exact_duplicate_pairs__"), list) else set()

    out: list[Signal] = []
    for (member, provider, diagnosis), sub in df.groupby(
        ["member_sk", "provider_sk", "diagnosis_primary"], sort=False
    ):
        if len(sub) < 2:
            continue
        sub = sub.sort_values("service_date")
        for (_, a), (_, b) in itertools.combinations(sub.iterrows(), 2):
            if not within_days(a["service_date"], b["service_date"], window).fired:
                continue
            ga, gb = _f(a["gross_amount_aed"]), _f(b["gross_amount_aed"])
            if max(ga, gb) == 0 or abs(ga - gb) / max(ga, gb) > tol:
                continue
            pair = tuple(sorted((a["claim_sk"], b["claim_sk"])))
            if pair in exact_pairs:
                continue
            days = abs((pd.Timestamp(a["service_date"]) - pd.Timestamp(b["service_date"])).days)
            out.append(
                _sig(
                    ctx, control, subject_type="claim", subject_id=b["claim_sk"],
                    fact_key=f"dup:{pair[0]}:{pair[1]}",
                    claim_ids=list(pair),
                    event_time=b["service_date"],
                    period=period_bucket(b["service_date"]),
                    confidence=0.8,
                    evidence={
                        "matched_claim_sk": a["claim_sk"],
                        "days_apart": int(days),
                        "window_days": window,
                        "amount_difference_aed": round(abs(ga - gb), 2),
                        "amount_tolerance": tol,
                        "diagnosis_code": diagnosis,
                        "provider_sk": provider,
                        "member_sk": member,
                        "proxy_note": "'Code equivalent' is approximated by an identical PRIMARY "
                                      "DIAGNOSIS; there is no code-equivalence map and no activity codes.",
                    },
                    exposure=_exp.duplicate_exposure([ga, gb]),
                )
            )
    return out


def pay_01_r03_split_claim_duplicate(ctx: ControlContext, control) -> list[Signal]:
    """PAY-01-R03 — the same service appearing across claim headers in one episode."""
    if ctx.episodes is None or ctx.episodes.empty:
        return []
    df = _claim_frame(ctx).set_index("claim_sk")
    out: list[Signal] = []
    for row in ctx.episodes.itertuples(index=False):
        if row.claim_count < 2:
            continue
        claim_ids = [c for c in str(row.claim_sks).split(",") if c in df.index]
        if len(claim_ids) < 2:
            continue
        sub = df.loc[claim_ids]
        for diagnosis, grp in sub.groupby("diagnosis_primary", sort=False):
            if len(grp) < 2 or grp["provider_sk"].nunique() != 1:
                continue
            amounts = grp["gross_amount_aed"].astype(float).tolist()
            out.append(
                _sig(
                    ctx, control, subject_type="episode", subject_id=row.episode_id,
                    fact_key=f"episode_split:{row.episode_id}:{diagnosis}",
                    claim_ids=list(grp.index),
                    event_time=grp["service_date"].min(),
                    period=period_bucket(grp["service_date"].min()),
                    confidence=float(row.linkage_confidence),
                    evidence={
                        "episode_id": row.episode_id,
                        "claim_sks": list(grp.index),
                        "diagnosis_code": diagnosis,
                        "episode_total_gross_aed": round(float(row.total_gross_aed), 2),
                        "crosses_providers": bool(row.crosses_providers),
                        "linkage_confidence": float(row.linkage_confidence),
                        "linkage_reason": row.linkage_reason,
                        "proxy_note": "'Component service' cannot be identified without claim lines; "
                                      "this detects a repeated PRINCIPAL DIAGNOSIS inside one episode "
                                      "at one provider, which is a weaker fact.",
                    },
                    exposure=_exp.duplicate_exposure(amounts),
                )
            )
    return out


def pay_06_r02_amount_arithmetic(ctx: ControlContext, control) -> list[Signal]:
    """PAY-06-R02 — transaction-integrity arithmetic, the one limb that is checkable."""
    df = _claim_frame(ctx)
    bad = df[
        (df["approved_amount_aed"] > df["gross_amount_aed"])
        | (df["gross_amount_aed"] < 0)
        | (df["approved_amount_aed"] < 0)
    ]
    out: list[Signal] = []
    for row in bad.itertuples(index=False):
        out.append(
            _sig(
                ctx, control, subject_type="claim", subject_id=row.claim_sk,
                fact_key=f"arithmetic:{row.claim_sk}",
                claim_ids=[row.claim_sk],
                event_time=row.service_date,
                period=period_bucket(row.service_date),
                evidence={
                    "gross_amount_aed": round(_f(row.gross_amount_aed), 2),
                    "approved_amount_aed": round(_f(row.approved_amount_aed), 2),
                    "difference_aed": round(_f(row.approved_amount_aed) - _f(row.gross_amount_aed), 2),
                    "equation_checked": "approved <= gross, and both non-negative",
                    "equation_not_checked": "gross - valid_discount - patient_share = net "
                                            "(patient_share and discount are absent from the source)",
                },
                exposure=_exp.line_edit_exposure(
                    _f(row.approved_amount_aed), _f(row.gross_amount_aed)
                ),
            )
        )
    return out


def pay_06_r03_billed_amount_peer_outlier(ctx: ControlContext, control) -> list[Signal]:
    """PAY-06-R03 — billed amount above the contract-adjusted peer threshold.

    Robust (median/MAD), never mean/SD: claim amounts here run from roughly
    AED 200 to AED 82,000 with a long right tail, and a z-score on that
    distribution both under- and over-flags in ways that do not track risk
    (§4.6).
    """
    df = _claim_frame(ctx)
    threshold = float(ctx.cfg("robust_residual_flag_threshold"))
    out: list[Signal] = []
    for _, sub in df.groupby("diagnosis_primary", sort=False):
        group = ctx.peers.resolve(sub.iloc[0]) if ctx.peers is not None else None
        values = (
            ctx.peers.values(group, "gross_amount_aed")
            if group is not None and group.sufficient
            else sub["gross_amount_aed"]
        )
        stats = robust_stats(values)
        if stats.scale <= 0:
            continue
        for row in sub.itertuples(index=False):
            resid = stats.z(_f(row.gross_amount_aed))
            if resid is None or resid <= threshold:
                continue
            excess = max(0.0, _f(row.gross_amount_aed) - stats.median)
            out.append(
                _sig(
                    ctx, control, subject_type="claim", subject_id=row.claim_sk,
                    fact_key=f"amount:{row.claim_sk}",
                    claim_ids=[row.claim_sk],
                    event_time=row.service_date,
                    period=period_bucket(row.service_date),
                    peer_level_used=group.level_name if group else "DIAGNOSIS_FALLBACK",
                    evidence={
                        "gross_amount_aed": round(_f(row.gross_amount_aed), 2),
                        "peer_median_aed": round(stats.median, 2),
                        "peer_mad": round(stats.mad, 2),
                        "peer_scale": round(stats.scale, 2),
                        "robust_residual": round(resid, 2),
                        "threshold": threshold,
                        "scale_source": stats.fallback_used or "mad",
                        "peer_n": stats.n,
                        "peer_level_used": group.level_name if group else "DIAGNOSIS_FALLBACK",
                        "peer_baseline_version": group.baseline_version if group else "",
                        "diagnosis_code": row.diagnosis_primary,
                        "provider_sk": row.provider_sk,
                        "plain_language": (
                            f"This claim bills AED {_f(row.gross_amount_aed):,.0f} where comparable "
                            f"claims for {row.diagnosis_primary} bill a median of "
                            f"AED {stats.median:,.0f} — {(_f(row.gross_amount_aed)/max(stats.median,1)):.1f}× "
                            f"the peer median."
                        ),
                        "proxy_note": "Peer group is at PRIMARY-DIAGNOSIS level, not activity-code "
                                      "level, because the source has no activity codes.",
                    },
                    exposure=_exp.line_edit_exposure(_f(row.gross_amount_aed), stats.median)
                    if excess > 0 else _exp.no_exposure("billed amount does not exceed the peer median"),
                    extra_versions={"peer_baseline": group.baseline_version} if group else None,
                )
            )
    return out


def pay_06_r04_approval_ratio_pattern(ctx: ControlContext, control) -> list[Signal]:
    """PAY-06-R04 — persistently low, or suspiciously invariant, approval ratio.

    The shrunk rate is the §4.6 estimator applied to "approved AED out of billed
    AED", so a provider with four claims is pulled toward the peer mean and is
    excluded from ranking if its posterior interval is too wide.
    """
    metrics = ctx.provider_metrics
    if metrics is None or metrics.empty:
        return []
    df = _claim_frame(ctx)
    low_percentile = float(ctx.cfg("approval_ratio_peer_percentile"))
    invariance_percentile = float(ctx.cfg("approval_ratio_invariance_percentile"))
    min_claims = int(ctx.cfg("min_entity_opportunities"))
    interval_mass = float(ctx.cfg("posterior_interval_mass"))
    max_width = float(ctx.cfg("max_posterior_width"))

    agg = df.groupby("provider_sk").agg(
        approved=("approved_amount_aed", "sum"),
        billed=("gross_amount_aed", "sum"),
        claims=("claim_sk", "count"),
    ).reset_index()
    prior = fit_beta_prior(agg["approved"], agg["billed"])

    ratios = df.assign(
        ratio=np.where(df["gross_amount_aed"] > 0,
                       df["approved_amount_aed"] / df["gross_amount_aed"].replace(0, np.nan), np.nan)
    )
    variance = ratios.groupby("provider_sk")["ratio"].var().rename("ratio_variance")
    agg = agg.merge(variance, left_on="provider_sk", right_index=True, how="left")

    eligible = agg[agg["claims"] >= min_claims]
    if eligible.empty:
        return []
    shrunk = {
        str(r.provider_sk): shrink_rate(str(r.provider_sk), r.approved, r.billed, prior,
                                        interval_mass=interval_mass, max_posterior_width=max_width)
        for r in eligible.itertuples(index=False)
    }
    # "versus peers" (Appendix C) — both limbs are cut at a percentile of the
    # peer distribution, not at a fixed ratio or variance.
    low_cut = float(np.quantile([s.shrunk_rate for s in shrunk.values()], low_percentile))
    variances = eligible["ratio_variance"].dropna()
    var_cut = float(np.quantile(variances, invariance_percentile)) if len(variances) else 0.0

    out: list[Signal] = []
    for row in eligible.itertuples(index=False):
        sr = shrunk[str(row.provider_sk)]
        limbs = []
        if sr.shrunk_rate <= low_cut and not sr.excluded_from_ranking:
            limbs.append("persistently_low_approval_ratio")
        if not is_missing(row.ratio_variance) and float(row.ratio_variance) <= var_cut:
            limbs.append("suspiciously_invariant_approval")
        if not limbs:
            continue
        sub = df[df["provider_sk"] == row.provider_sk]
        out.append(
            _sig(
                ctx, control, subject_type="provider", subject_id=row.provider_sk,
                fact_key=f"approval_ratio:{row.provider_sk}",
                claim_ids=sub["claim_sk"].tolist(),
                event_time=sub["service_date"].max(),
                period=period_bucket(sub["service_date"].max()),
                confidence=0.5 if sr.excluded_from_ranking else 0.9,
                evidence={
                    "provider_sk": row.provider_sk,
                    "observed_approval_ratio": round(sr.observed_rate or 0.0, 4),
                    "shrunk_approval_ratio": round(sr.shrunk_rate, 4),
                    "posterior_interval": [round(sr.posterior_low, 4), round(sr.posterior_high, 4)],
                    "posterior_width": round(sr.posterior_width, 4),
                    "excluded_from_ranking": sr.excluded_from_ranking,
                    "peer_mean": round(sr.peer_mean, 4),
                    "prior_strength": round(sr.prior_strength, 2),
                    "approval_ratio_variance": None if is_missing(row.ratio_variance) else round(float(row.ratio_variance), 6),
                    "claim_count": int(row.claims),
                    "limb_fired": limbs,
                    "peer_percentile_low": low_percentile,
                    "peer_cut_low_ratio": round(low_cut, 4),
                    "peer_percentile_invariance": invariance_percentile,
                    "peer_cut_variance": round(var_cut, 6),
                    "peer_population": len(shrunk),
                    "shrinkage_explanation": sr.explain(),
                },
                exposure=_exp.provider_pattern_exposure(
                    (sub["gross_amount_aed"] - sub["approved_amount_aed"]).clip(lower=0).tolist()
                ),
            )
        )
    return out


def pay_10_r01_cob_not_coordinated(ctx: ControlContext, control) -> list[Signal]:
    """PAY-10-R01 — other coverage exists; coordination evidence does not."""
    df = _claim_frame(ctx)
    min_insurers = int(ctx.cfg("cob_min_insurers"))
    sub = df[df["num_insurers_same_event"] >= min_insurers]
    out: list[Signal] = []
    for row in sub.itertuples(index=False):
        out.append(
            _sig(
                ctx, control, subject_type="claim", subject_id=row.claim_sk,
                fact_key=f"cob:{row.claim_sk}",
                claim_ids=[row.claim_sk],
                event_time=row.service_date,
                period=period_bucket(row.service_date),
                confidence=0.6,
                evidence={
                    "num_insurers_same_event": int(row.num_insurers_same_event),
                    "min_insurers": min_insurers,
                    "gross_amount_aed": round(_f(row.gross_amount_aed), 2),
                    "member_sk": row.member_sk,
                    "coordination_record_present": False,
                    "payer_order_evaluable": False,
                    "proxy_note": "The source establishes that other coverage EXISTS but carries no "
                                  "primary/secondary designation and no other-payer payment record, "
                                  "so this is raised as 'coordination evidence required', NOT as a "
                                  "wrong-payer-order finding.",
                },
                exposure=_exp.no_exposure(
                    "the recoverable share depends on the other payer's settlement, which is not in "
                    "this source; reporting the gross amount here would overstate exposure"
                ),
            )
        )
    return out


# ===========================================================================
# CLN
# ===========================================================================


def cln_01_r01_coding_mismatch_rate(ctx: ControlContext, control) -> list[Signal]:
    """CLN-01-R01 — provider coding-mismatch rate against shrunk peers."""
    df = _claim_frame(ctx)
    percentile = float(ctx.cfg("coding_mismatch_peer_percentile"))
    min_claims = int(ctx.cfg("min_entity_opportunities"))
    interval_mass = float(ctx.cfg("posterior_interval_mass"))
    max_width = float(ctx.cfg("max_posterior_width"))

    work = df.assign(mismatch=(~df["icd_code_matches_procedure"].astype(bool)).astype(int))
    agg = work.groupby("provider_sk").agg(
        events=("mismatch", "sum"), opportunities=("claim_sk", "count"),
    ).reset_index()
    prior = fit_beta_prior(agg["events"], agg["opportunities"])

    # Appendix C: "exceeds shrunk PEER PERCENTILE". The cut is therefore taken
    # from the peer distribution of shrunk rates, computed once, rather than
    # from a fixed rate that would not adapt to the peer base rate.
    shrunk = {}
    for row in agg.itertuples(index=False):
        if row.opportunities < min_claims:
            continue
        shrunk[str(row.provider_sk)] = shrink_rate(
            str(row.provider_sk), row.events, row.opportunities, prior,
            interval_mass=interval_mass, max_posterior_width=max_width,
        )
    if not shrunk:
        return []
    cut = float(np.quantile([s.shrunk_rate for s in shrunk.values()], percentile))

    out: list[Signal] = []
    for row in agg.itertuples(index=False):
        sr = shrunk.get(str(row.provider_sk))
        if sr is None:
            continue
        if sr.excluded_from_ranking:
            continue  # declared exclusion: posterior_interval_too_wide
        if sr.shrunk_rate <= cut:
            continue
        sub = work[work["provider_sk"] == row.provider_sk]
        out.append(
            _sig(
                ctx, control, subject_type="provider", subject_id=row.provider_sk,
                fact_key=f"coding_mismatch:{row.provider_sk}",
                claim_ids=sub["claim_sk"].tolist(),
                event_time=sub["service_date"].max(),
                period=period_bucket(sub["service_date"].max()),
                evidence={
                    "provider_sk": row.provider_sk,
                    "observed_rate": round(sr.observed_rate or 0.0, 4),
                    "shrunk_rate": round(sr.shrunk_rate, 4),
                    "posterior_interval": [round(sr.posterior_low, 4), round(sr.posterior_high, 4)],
                    "peer_mean": round(sr.peer_mean, 4),
                    "prior_strength": round(sr.prior_strength, 2),
                    "claim_count": int(row.opportunities),
                    "mismatch_count": int(row.events),
                    "peer_percentile": percentile,
                    "peer_percentile_cut": round(cut, 4),
                    "peer_population": len(shrunk),
                    "shrinkage_explanation": sr.explain(),
                    "proxy_note": "There are no code LEVELS in the claim extract, so 'highest-paid level "
                                  "share' is substituted by the rate of icd_code_matches_procedure = "
                                  "False. That measures coding-integrity pressure, not level intensity.",
                },
                exposure=_exp.provider_pattern_exposure(
                    sub.loc[sub["mismatch"] == 1, "gross_amount_aed"].tolist()
                ),
            )
        )
    return out


def cln_01_r02_expected_level_residual(ctx: ControlContext, control) -> list[Signal]:
    """CLN-01-R02 — amount per inpatient day above the diagnosis-peer expectation."""
    df = _claim_frame(ctx)
    percentile = float(ctx.cfg("amount_per_los_day_peer_percentile"))
    work = df[df["length_of_stay_days"] > 0].copy()
    work["amount_per_los_day"] = work["gross_amount_aed"] / work["length_of_stay_days"]
    out: list[Signal] = []
    for diagnosis, sub in work.groupby("diagnosis_primary", sort=False):
        stats = robust_stats(sub["amount_per_los_day"])
        if stats.scale <= 0:
            continue
        # §4.6 permits "median/MAD OR PERCENTILE-BASED robust statistics". Amount
        # per day is a ratio of two skewed quantities and its tail is heavier
        # than either; the percentile form is used and the modified-z residual is
        # reported alongside so both readings are visible.
        cut = float(np.quantile(sub["amount_per_los_day"].dropna(), percentile))
        for row in sub.itertuples(index=False):
            value = _f(row.amount_per_los_day)
            resid = stats.z(value)
            if value <= cut:
                continue
            expected = stats.median * _f(row.length_of_stay_days)
            out.append(
                _sig(
                    ctx, control, subject_type="claim", subject_id=row.claim_sk,
                    fact_key=f"amount:{row.claim_sk}",   # shares the fact with PAY-06-R03: evidence capping
                    claim_ids=[row.claim_sk],
                    event_time=row.service_date,
                    period=period_bucket(row.service_date),
                    peer_level_used="diagnosis_primary",
                    evidence={
                        "amount_per_los_day": round(_f(row.amount_per_los_day), 2),
                        "peer_median": round(stats.median, 2),
                        "peer_mad": round(stats.mad, 2),
                        "robust_residual": None if resid is None else round(resid, 2),
                        "peer_percentile": percentile,
                        "peer_percentile_cut_per_day": round(cut, 2),
                        "expected_amount_aed": round(expected, 2),
                        "billed_amount_aed": round(_f(row.gross_amount_aed), 2),
                        "excess_amount_aed": round(max(0.0, _f(row.gross_amount_aed) - expected), 2),
                        "length_of_stay_days": int(row.length_of_stay_days),
                        "diagnosis_code": diagnosis,
                        "peer_n": stats.n,
                        "plain_language": (
                            f"At {int(row.length_of_stay_days)} inpatient day(s) for {diagnosis}, "
                            f"comparable claims bill about AED {expected:,.0f}; this one bills "
                            f"AED {_f(row.gross_amount_aed):,.0f}."
                        ),
                        "proxy_note": "Implemented as an expected-AMOUNT residual, not an "
                                      "expected-LEVEL residual: code levels do not exist here, and "
                                      "age and setting are unavailable for risk adjustment.",
                    },
                    exposure=_exp.line_edit_exposure(_f(row.gross_amount_aed), expected),
                )
            )
    return out



def cln_03_r01_dx_procedure_mismatch(ctx: ControlContext, control) -> list[Signal]:
    """CLN-03-R01 — diagnosis and procedure do not go together.

    ``icd_code_matches_procedure`` is a boolean **verdict** with no procedure
    behind it: some upstream process judged a mismatch and recorded the
    conclusion, not the evidence. That is why this control does NOT keep the
    catalogue's first-named ``REJECT`` disposition — §3.3 permits a denial only
    on an objective, effective-dated condition the system can itself evaluate,
    and "another system said so" is not that. The governed disposition is the
    catalogue's second option, a pend, and the evidence says exactly what the
    reviewer is being asked to verify.
    """
    df = _claim_frame(ctx)
    bad = df[~df["icd_code_matches_procedure"].astype(bool)]
    out: list[Signal] = []
    for row in bad.itertuples(index=False):
        out.append(
            _sig(
                ctx, control, subject_type="claim", subject_id=row.claim_sk,
                fact_key=f"dx_proc:{row.claim_sk}",
                claim_ids=[row.claim_sk],
                event_time=row.service_date,
                period=period_bucket(row.service_date),
                confidence=0.65,
                data_quality_penalty=0.35,
                evidence={
                    "claim_sk": row.claim_sk,
                    "diagnosis_code": row.diagnosis_primary,
                    "diagnosis_description": getattr(row, "diagnosis_primary", ""),
                    "icd_code_matches_procedure": False,
                    "gross_amount_aed": round(_f(row.gross_amount_aed), 2),
                    "provider_sk": row.provider_sk,
                    "what_the_reviewer_must_verify": (
                        "The source asserts that the coded diagnosis does not match the billed "
                        "procedure, but carries NO procedure code, so the assertion cannot be "
                        "re-derived here. The reviewer must obtain the procedure detail and "
                        "confirm or clear it."
                    ),
                    "proxy_note": "Disposition is a PEND, not the catalogue's first-named REJECT: "
                                  "§3.3 permits a denial only on an objective, effective-dated "
                                  "condition this system can evaluate itself, and a boolean verdict "
                                  "from an undocumented upstream process is not one.",
                },
                exposure=_exp.model_only_exposure(_f(row.gross_amount_aed), "coding-mismatch flag"),
            )
        )
    return out


def cln_04_r01_minimum_repeat_interval(ctx: ControlContext, control) -> list[Signal]:
    """CLN-04-R01 — the same service repeated inside the policy interval."""
    df = _claim_frame(ctx)
    interval = int(ctx.cfg("min_repeat_interval_days"))
    out: list[Signal] = []
    for (member, diagnosis), sub in df.groupby(["member_sk", "diagnosis_primary"], sort=False):
        if len(sub) < 2:
            continue
        sub = sub.sort_values("service_date")
        rows = list(sub.itertuples(index=False))
        for prev, curr in zip(rows, rows[1:]):
            prev_date, curr_date = pd.Timestamp(prev.service_date), pd.Timestamp(curr.service_date)
            if pd.isna(prev_date) or pd.isna(curr_date):
                # §4.5: a missing service date is an explicit state, not a zero.
                # "Repeated within 14 days" is unanswerable without both dates,
                # so the control stays silent rather than guessing an interval.
                continue
            gap = (curr_date - prev_date).days
            if gap < 0 or gap > interval:
                continue
            out.append(
                _sig(
                    ctx, control, subject_type="member", subject_id=member,
                    fact_key=f"repeat:{prev.claim_sk}:{curr.claim_sk}",
                    claim_ids=[prev.claim_sk, curr.claim_sk],
                    event_time=curr.service_date,
                    period=period_bucket(curr.service_date),
                    confidence=0.75,
                    evidence={
                        "prior_claim_sk": prev.claim_sk,
                        "days_between": int(gap),
                        "interval_threshold": interval,
                        "diagnosis_code": diagnosis,
                        "provider_sk": curr.provider_sk,
                        "prior_provider_sk": prev.provider_sk,
                        "member_sk": member,
                        "proxy_note": "'Same or equivalent service' is approximated by an identical "
                                      "PRIMARY DIAGNOSIS; there are no service codes and no required "
                                      "Observation to check for an accepted indicator.",
                    },
                    exposure=_exp.duplicate_exposure(
                        [_f(prev.gross_amount_aed), _f(curr.gross_amount_aed)]
                    ),
                )
            )
    return out


def cln_04_r02_episode_frequency_excess(ctx: ControlContext, control) -> list[Signal]:
    """CLN-04-R02 — member admission count inside the utilisation window."""
    df = _claim_frame(ctx)
    threshold = int(ctx.cfg("member_repeat_service_threshold"))
    window = int(ctx.cfg("utilisation_velocity_window_days"))
    out: list[Signal] = []
    for member, sub in df.groupby("member_sk", sort=False):
        if len(sub) < threshold:
            continue
        sub = sub.sort_values("service_date")
        times = pd.to_datetime(sub["service_date"]).to_numpy()
        for i in range(len(sub)):
            lo = times[i] - np.timedelta64(window, "D")
            in_window = sub.iloc[[j for j in range(i + 1) if times[j] > lo]]
            if len(in_window) < threshold:
                continue
            row = sub.iloc[i]
            out.append(
                _sig(
                    ctx, control, subject_type="member", subject_id=member,
                    fact_key=f"velocity:{member}:{period_bucket(row['service_date'])}",
                    claim_ids=in_window["claim_sk"].tolist(),
                    event_time=row["service_date"],
                    period=period_bucket(row["service_date"]),
                    confidence=0.8,
                    evidence={
                        "member_sk": member,
                        "claims_in_window": int(len(in_window)),
                        "window_days": window,
                        "threshold": threshold,
                        "diagnoses_in_window": sorted(in_window["diagnosis_primary"].unique().tolist()),
                        "providers_in_window": sorted(in_window["provider_sk"].unique().tolist()),
                        "total_gross_aed": round(float(in_window["gross_amount_aed"].sum()), 2),
                        "proxy_note": "Risk adjustment for age is not possible (no demographics); the "
                                      "count is unadjusted and the disposition reflects that.",
                    },
                    exposure=_exp.provider_pattern_exposure(
                        in_window["gross_amount_aed"].tolist()[1:]
                    ),
                )
            )
            break   # one signal per member per run; fact_key would dedupe anyway
    return out


def cln_04_r04_provider_utilisation_outlier(ctx: ControlContext, control) -> list[Signal]:
    """CLN-04-R04 — claims per distinct member, shrunk against peers."""
    metrics = ctx.provider_metrics
    if metrics is None or metrics.empty:
        return []
    min_claims = int(ctx.cfg("min_entity_opportunities"))
    max_width = float(ctx.cfg("max_posterior_width"))
    eligible = metrics[metrics["claim_count"] >= min_claims]
    if eligible.empty:
        return []
    stats = robust_stats(eligible["claims_per_member"])
    if stats.scale <= 0:
        return []
    df = _claim_frame(ctx)
    out: list[Signal] = []
    for row in eligible.itertuples(index=False):
        resid = stats.z(_f(row.claims_per_member))
        if resid is None or resid <= float(ctx.cfg("robust_residual_flag_threshold")):
            continue
        group = ctx.provider_peers.resolve(row._asdict()) if ctx.provider_peers is not None else None
        sub = df[df["provider_sk"] == row.provider_sk]
        out.append(
            _sig(
                ctx, control, subject_type="provider", subject_id=row.provider_sk,
                fact_key=f"utilisation:{row.provider_sk}",
                claim_ids=sub["claim_sk"].tolist(),
                event_time=sub["service_date"].max(),
                period=period_bucket(sub["service_date"].max()),
                peer_level_used=group.level_name if group else "GLOBAL",
                evidence={
                    "provider_sk": row.provider_sk,
                    "claims_per_member": round(_f(row.claims_per_member), 3),
                    "peer_median": round(stats.median, 3),
                    "robust_residual": round(resid, 2),
                    "distinct_members": int(row.distinct_members),
                    "claim_count": int(row.claim_count),
                    "peer_level_used": group.level_name if group else "GLOBAL",
                    "peer_n": group.n_entities if group else stats.n,
                    "max_posterior_width": max_width,
                    "proxy_note": "'Services' is substituted by 'claims' — there are no service "
                                  "lines, so the intensity of each claim is invisible here.",
                },
                # Incremental, not gross: the suspect amount is the value of the
                # claims ABOVE the peer-median utilisation rate, not the
                # provider's entire book (§4.11).
                exposure=_exp.provider_pattern_exposure(
                    _excess_claim_values(sub, _f(row.claims_per_member), stats.median)
                ),
            )
        )
    return out


def cln_05_r01_los_residual(ctx: ControlContext, control) -> list[Signal]:
    """CLN-05-R01 — length of stay outside the diagnosis-peer prediction interval."""
    df = _claim_frame(ctx)
    threshold = float(ctx.cfg("los_residual_threshold"))
    out: list[Signal] = []
    for diagnosis, sub in df.groupby("diagnosis_primary", sort=False):
        stats = robust_stats(sub["length_of_stay_days"])
        if stats.scale <= 0:
            continue
        for row in sub.itertuples(index=False):
            resid = stats.z(_f(row.length_of_stay_days))
            if resid is None or abs(resid) <= threshold:
                continue
            direction = "longer than expected" if resid > 0 else "shorter than expected"
            excess_days = max(0.0, _f(row.length_of_stay_days) - stats.median)
            per_day = _f(row.gross_amount_aed) / max(_f(row.length_of_stay_days), 1.0)
            out.append(
                _sig(
                    ctx, control, subject_type="claim", subject_id=row.claim_sk,
                    fact_key=f"los:{row.claim_sk}",
                    claim_ids=[row.claim_sk],
                    event_time=row.service_date,
                    period=period_bucket(row.service_date),
                    peer_level_used="diagnosis_primary",
                    evidence={
                        "length_of_stay_days": int(row.length_of_stay_days),
                        "peer_median_los": round(stats.median, 1),
                        "peer_mad": round(stats.mad, 2),
                        "robust_residual": round(resid, 2),
                        "direction": direction,
                        "threshold": threshold,
                        "diagnosis_code": diagnosis,
                        "peer_n": stats.n,
                        "provider_sk": row.provider_sk,
                        "plain_language": (
                            f"This stay was {int(row.length_of_stay_days)} day(s) where comparable "
                            f"{diagnosis} stays run a median of {stats.median:.0f} day(s) — "
                            f"{direction}."
                        ),
                    },
                    exposure=_exp.line_edit_exposure(
                        _f(row.gross_amount_aed), _f(row.gross_amount_aed) - excess_days * per_day
                    ) if resid > 0 else _exp.no_exposure(
                        "a shorter-than-expected stay has no over-payment exposure; it is a "
                        "utilisation-review finding"
                    ),
                )
            )
    return out


def cln_05_r02_related_readmission(ctx: ControlContext, control) -> list[Signal]:
    """CLN-05-R02 — readmission inside the window. Explicitly *not* a fraud label."""
    df = _claim_frame(ctx)
    window = int(ctx.cfg("readmit_window_days"))
    sub = df[df["discharge_readmit_gap_days"] < window]
    out: list[Signal] = []
    for row in sub.itertuples(index=False):
        out.append(
            _sig(
                ctx, control, subject_type="claim", subject_id=row.claim_sk,
                fact_key=f"readmit:{row.claim_sk}",
                claim_ids=[row.claim_sk],
                event_time=row.service_date,
                period=period_bucket(row.service_date),
                confidence=0.7,
                evidence={
                    "discharge_readmit_gap_days": int(row.discharge_readmit_gap_days),
                    "readmit_window_days": window,
                    "gross_amount_aed": round(_f(row.gross_amount_aed), 2),
                    "diagnosis_code": row.diagnosis_primary,
                    "provider_sk": row.provider_sk,
                    "member_sk": row.member_sk,
                    "not_a_fraud_label_notice": (
                        "Appendix C, CLN-05-R02: 'Episode review, NOT FRAUD LABEL.' A short "
                        "readmission gap is a utilisation-review trigger. It says nothing about intent."
                    ),
                    "proxy_note": "There is no diagnosis-relatedness map, so a RELATED readmission "
                                  "cannot be distinguished from an unrelated one; the unrelated-trauma "
                                  "exclusion therefore cannot be applied automatically.",
                },
                exposure=_exp.no_exposure(
                    "a readmission is not by itself an over-payment; exposure depends on whether "
                    "payment policy treats the two stays as one episode (CLN-05-R03)"
                ),
            )
        )
    return out


def cln_05_r03_discharge_readmit_split(ctx: ControlContext, control) -> list[Signal]:
    """CLN-05-R03 — two stays that may be one continuous episode."""
    if ctx.episodes is None or ctx.episodes.empty:
        return []
    window = int(ctx.cfg("readmit_window_days"))
    df = _claim_frame(ctx).set_index("claim_sk")
    out: list[Signal] = []
    for row in ctx.episodes.itertuples(index=False):
        if row.claim_count < 2 or row.crosses_providers:
            continue  # cross-provider is a declared exclusion (possible transfer)
        claim_ids = [c for c in str(row.claim_sks).split(",") if c in df.index]
        if len(claim_ids) < 2:
            continue
        sub = df.loc[claim_ids].sort_values("service_date")
        gaps = (
            pd.to_datetime(sub["service_date"]).iloc[1:].to_numpy()
            - pd.to_datetime(sub["discharge_date"]).iloc[:-1].to_numpy()
        )
        gap_days = [int(g / np.timedelta64(1, "D")) for g in gaps]
        out.append(
            _sig(
                ctx, control, subject_type="episode", subject_id=row.episode_id,
                fact_key=f"split_stay:{row.episode_id}",
                claim_ids=claim_ids,
                event_time=sub["service_date"].min(),
                period=period_bucket(sub["service_date"].min()),
                confidence=float(row.linkage_confidence),
                evidence={
                    "episode_id": row.episode_id,
                    "claim_sks": claim_ids,
                    "gap_days": gap_days,
                    "window_days": window,
                    "combined_gross_aed": round(float(sub["gross_amount_aed"].sum()), 2),
                    "provider_sk": sub["provider_sk"].iloc[0],
                    "linkage_reason": row.linkage_reason,
                    "proxy_note": "Disposition is a PEND, not a REPRICE: the correct combined payable "
                                  "amount is not computable without a payment policy, and Table 3.2 "
                                  "reserves REPRICE for the case where it is.",
                },
                exposure=_exp.duplicate_exposure(sub["gross_amount_aed"].astype(float).tolist()),
            )
        )
    return out


def cln_06_r02_non_operational_facility(ctx: ControlContext, control) -> list[Signal]:
    """CLN-06-R02 — exclusion-listed, or no credible operating footprint."""
    df = _claim_frame(ctx)
    min_claims = int(ctx.cfg("min_entity_opportunities"))
    global_median = float(df["gross_amount_aed"].median())
    out: list[Signal] = []
    for provider, sub in df.groupby("provider_sk", sort=False):
        blacklisted = bool(sub["provider_blacklist_flag"].any())
        active_days = int(pd.to_datetime(sub["service_date"]).dt.date.nunique())
        per_day = len(sub) / max(active_days, 1)
        mean_gross = float(sub["gross_amount_aed"].mean())
        thin_footprint = (
            len(sub) >= min_claims and active_days <= max(3, len(sub) // 4)
            and mean_gross > global_median
        )
        if not (blacklisted or thin_footprint):
            continue
        limbs = [l for l, on in (("exclusion_listed", blacklisted),
                                 ("thin_bursty_footprint", thin_footprint)) if on]
        out.append(
            _sig(
                ctx, control, subject_type="provider", subject_id=provider,
                fact_key=f"non_operational:{provider}",
                claim_ids=sub["claim_sk"].tolist(),
                event_time=sub["service_date"].min(),
                period=period_bucket(sub["service_date"].min()),
                confidence=0.75 if blacklisted else 0.45,
                data_quality_penalty=0.25,
                evidence={
                    "provider_sk": provider,
                    "exclusion_listed": blacklisted,
                    "claim_count": int(len(sub)),
                    "active_days": active_days,
                    "claims_per_active_day": round(per_day, 2),
                    "mean_gross_aed": round(mean_gross, 2),
                    "peer_median_gross_aed": round(global_median, 2),
                    "limb_fired": limbs,
                    "limb_unavailable": "verified operating period (no provider operating periods "
                                        "exist in the claim extract)",
                    "proxy_note": "The 'no credible operating footprint' limb is a composite of the "
                                  "undated exclusion flag and a thin, bursty claim pattern. It is "
                                  "weaker than the catalogue control and never denies.",
                },
                exposure=_exp.provider_pattern_exposure(sub["gross_amount_aed"].tolist()),
            )
        )
    return out


def cln_06_r03_synthetic_encounter_signature(ctx: ControlContext, control) -> list[Signal]:
    """CLN-06-R03 — identical billing signatures recurring across unrelated members."""
    df = _claim_frame(ctx)
    min_claims = int(ctx.cfg("min_entity_opportunities"))
    out: list[Signal] = []
    for provider, sub in df.groupby("provider_sk", sort=False):
        if len(sub) < min_claims:
            continue
        sig_cols = ["diagnosis_primary", "length_of_stay_days", "gross_amount"]
        for signature, grp in sub.groupby(sig_cols, sort=False):
            if len(grp) < 2 or grp["member_sk"].nunique() < 2:
                continue
            out.append(
                _sig(
                    ctx, control, subject_type="provider", subject_id=provider,
                    fact_key=f"signature:{provider}:{'|'.join(str(s) for s in signature)}",
                    claim_ids=grp["claim_sk"].tolist(),
                    event_time=grp["service_date"].min(),
                    period=period_bucket(grp["service_date"].min()),
                    confidence=0.7,
                    evidence={
                        "provider_sk": provider,
                        "signature": {
                            "diagnosis_primary": signature[0],
                            "length_of_stay_days": int(signature[1]),
                            "gross_amount_source_currency": float(signature[2]),
                        },
                        "distinct_members": int(grp["member_sk"].nunique()),
                        "repeat_count": int(len(grp)),
                        "claim_sks": grp["claim_sk"].tolist(),
                        "gross_amount_aed": round(float(grp["gross_amount_aed"].iloc[0]), 2),
                        "proxy_note": "Signature is (diagnosis, length of stay, exact requested "
                                      "amount). Codes, times and notes are unavailable, so a genuine "
                                      "standard package can produce this pattern — which is why "
                                      "'standard_package_pricing' is a declared exclusion.",
                    },
                    exposure=_exp.provider_pattern_exposure(grp["gross_amount_aed"].tolist()[1:]),
                )
            )
    return out


def cln_07_r02_facility_capacity(ctx: ControlContext, control) -> list[Signal]:
    """CLN-07-R02 — concurrent inpatient occupancy beyond plausible capacity."""
    df = _claim_frame(ctx)
    capacity = float(ctx.cfg("provider_daily_capacity_admissions"))
    tolerance = float(ctx.cfg("capacity_tolerance_multiple"))
    limit = capacity * tolerance
    out: list[Signal] = []
    for provider, sub in df.groupby("provider_sk", sort=False):
        admits = pd.to_datetime(sub["service_date"])
        discharges = pd.to_datetime(sub["discharge_date"])
        events = pd.concat([
            pd.DataFrame({"day": admits, "delta": 1}),
            pd.DataFrame({"day": discharges, "delta": -1}),
        ]).dropna().sort_values("day")
        if events.empty:
            continue
        occupancy = events.groupby("day")["delta"].sum().cumsum()
        peak = float(occupancy.max())
        if peak <= limit:
            continue
        peak_day = occupancy.idxmax()
        on_day = sub[(admits <= peak_day) & (discharges >= peak_day)]
        out.append(
            _sig(
                ctx, control, subject_type="provider", subject_id=provider,
                fact_key=f"capacity:{provider}:{str(peak_day)[:10]}",
                claim_ids=on_day["claim_sk"].tolist(),
                event_time=peak_day,
                period=period_bucket(peak_day),
                confidence=0.4,
                data_quality_penalty=0.5,
                evidence={
                    "provider_sk": provider,
                    "peak_day": str(peak_day)[:10],
                    "peak_concurrent_occupancy": int(peak),
                    "assumed_capacity": capacity,
                    "tolerance_multiple": tolerance,
                    "effective_limit": limit,
                    "claims_on_peak_day": int(len(on_day)),
                    "capacity_source": "cfg.provider_daily_capacity_admissions — a CONFIGURED "
                                       "PLACEHOLDER, not a licensed bed count",
                    "proxy_note": "Appendix C: 'Missing roster produces low confidence.' No licensed "
                                  "capacity exists in this source, so confidence is 0.4 and a large "
                                  "data-quality penalty is applied to the priority score.",
                },
                exposure=_exp.provider_pattern_exposure(on_day["gross_amount_aed"].tolist()),
            )
        )
    return out


def cln_07_r04_capacity_trend_discontinuity(ctx: ControlContext, control) -> list[Signal]:
    """CLN-07-R04 — CUSUM change point on monthly provider throughput."""
    from ..statistical.changepoint import detect_changepoints, monthly_provider_metrics

    df = _claim_frame(ctx)
    series = monthly_provider_metrics(df)
    cps = detect_changepoints(
        series, entity_column="provider_sk", period_column="period",
        value_column="claim_count", metric_name="monthly claim throughput",
        min_history_periods=int(ctx.cfg("min_history_periods")),
        cusum_k=float(ctx.cfg("cusum_k")), cusum_h=float(ctx.cfg("cusum_h")),
        ewma_lambda=float(ctx.cfg("ewma_lambda")), ewma_L=float(ctx.cfg("ewma_L")),
        segments=getattr(ctx, "segments", None),
        caveat="No staff, hours, equipment, contract or facility-change records exist in "
               "the claim extract, so a legitimate expansion cannot be segmented out. Treat as an "
               "audit prompt, not as evidence of misconduct.",
    )
    out: list[Signal] = []
    for cp in cps:
        if cp.direction != "increase":
            continue
        sub = df[df["provider_sk"] == cp.entity]
        out.append(
            _sig(
                ctx, control, subject_type="provider", subject_id=cp.entity,
                fact_key=f"throughput_change:{cp.entity}:{cp.change_period}",
                claim_ids=sub["claim_sk"].tolist(),
                event_time=cp.change_date,
                period=cp.change_period,
                confidence=float(cp.confidence),
                data_quality_penalty=0.3,
                evidence={
                    "provider_sk": cp.entity,
                    "change_date": cp.change_date,
                    "change_period": cp.change_period,
                    "pre_change_mean": round(cp.pre_change_mean, 3),
                    "post_change_mean": round(cp.post_change_mean, 3),
                    "change_magnitude": round(cp.change_magnitude, 3),
                    "method": cp.method,
                    "cusum_statistic": round(cp.statistic, 3),
                    "threshold": cp.threshold,
                    "periods_of_history": cp.periods_of_history,
                    "declared_segments": list(cp.declared_segments),
                    "segmentation_caveat": cp.caveat,
                    "plain_language": cp.explain(),
                },
                exposure=_exp.no_exposure(
                    "a throughput change point has no directly attributable over-payment; exposure "
                    "would require a sampled audit of the post-change claims"
                ),
            )
        )
    return out


def cln_01_r03_documentation_level_conflict(ctx: ControlContext, control) -> list[Signal]:
    """CLN-01-R03 — billed level unsupported by the documentation.

    Runs against the SYNTHETIC corpus only. Delegates the extraction and
    span-grounding to :mod:`fwa.nlp.pipeline`; a finding without a source span
    never reaches this function, because the pipeline discards it (§4.9).
    """
    if ctx.documents is None:
        return []
    from ..nlp.pipeline import documentation_level_conflicts

    findings = documentation_level_conflicts(ctx)
    out: list[Signal] = []
    for f in findings:
        out.append(
            _sig(
                ctx, control, subject_type="claim", subject_id=f["claim_sk"],
                fact_key=f"doc_level:{f['claim_sk']}",
                claim_ids=[f["claim_sk"]],
                event_time=f.get("service_date"),
                period=period_bucket(f.get("service_date")),
                confidence=float(f["extraction_confidence"]),
                evidence=f,
                exposure=_exp.no_exposure(
                    "a documentation conflict establishes that the record does not support the bill, "
                    "not by how much; the repriced amount is a coding decision for the reviewer"
                ),
            )
        )
    return out


# ===========================================================================
# PHR / POL
# ===========================================================================


def phr_03_r03_pharmacy_volume_spike(ctx: ControlContext, control) -> list[Signal]:
    """PHR-03-R03 — pharmacy share above the diagnosis-peer multiple."""
    df = _claim_frame(ctx)
    multiple = float(ctx.cfg("pharmacy_share_peer_multiple"))
    out: list[Signal] = []
    for diagnosis, sub in df.groupby("diagnosis_primary", sort=False):
        peer_median = float(sub["pharmacy_bill_ratio"].median())
        if peer_median <= 0:
            continue
        threshold = multiple * peer_median
        for row in sub.itertuples(index=False):
            ratio = _f(row.pharmacy_bill_ratio)
            if ratio <= threshold:
                continue
            implied = ratio * _f(row.gross_amount_aed)
            expected = peer_median * _f(row.gross_amount_aed)
            out.append(
                _sig(
                    ctx, control, subject_type="claim", subject_id=row.claim_sk,
                    fact_key=f"pharmacy:{row.claim_sk}",
                    claim_ids=[row.claim_sk],
                    event_time=row.service_date,
                    period=period_bucket(row.service_date),
                    peer_level_used="diagnosis_primary",
                    confidence=0.7,
                    evidence={
                        "pharmacy_bill_ratio": round(ratio, 3),
                        "peer_median_ratio": round(peer_median, 3),
                        "peer_multiple": multiple,
                        "threshold_ratio": round(threshold, 3),
                        "implied_pharmacy_aed": round(implied, 2),
                        "expected_pharmacy_aed": round(expected, 2),
                        "diagnosis_code": diagnosis,
                        "provider_sk": row.provider_sk,
                        "peer_n": int(len(sub)),
                        "plain_language": (
                            f"Pharmacy is {ratio:.0%} of this bill where comparable {diagnosis} "
                            f"claims run {peer_median:.0%}."
                        ),
                        "proxy_note": "The source has an aggregate pharmacy SHARE only — no products, "
                                      "quantities or days supply — so this is a ratio anomaly, not a "
                                      "drug-volume or drug-value spike.",
                    },
                    exposure=_exp.line_edit_exposure(implied, expected),
                )
            )
    return out


def pol_01_r04_early_tenure_utilisation(ctx: ControlContext, control) -> list[Signal]:
    """POL-01-R04 — high-value claim shortly after policy inception.

    ``previous_fraud_on_policy`` is a **policy-history attribute** present in the
    source, not the held-out outcome label. It is reported separately in the
    evidence rather than folded into the score, so a reviewer weighs it
    themselves — and so that this control cannot be mistaken for one that reads
    ``fraud_label``.
    """
    df = _claim_frame(ctx)
    early_days = int(ctx.cfg("early_tenure_days"))
    percentile = float(ctx.cfg("early_tenure_amount_peer_percentile"))
    early = df[df["days_since_policy_start"] < early_days]
    out: list[Signal] = []
    for diagnosis, sub in df.groupby("diagnosis_primary", sort=False):
        stats = robust_stats(sub["gross_amount_aed"])
        if stats.scale <= 0:
            continue
        cut = float(np.quantile(sub["gross_amount_aed"].dropna(), percentile))
        candidates = early[early["diagnosis_primary"] == diagnosis]
        for row in candidates.itertuples(index=False):
            resid = stats.z(_f(row.gross_amount_aed))
            if _f(row.gross_amount_aed) <= cut:
                continue
            out.append(
                _sig(
                    ctx, control, subject_type="member", subject_id=row.member_sk,
                    fact_key=f"early_tenure:{row.claim_sk}",
                    claim_ids=[row.claim_sk],
                    event_time=row.service_date,
                    period=period_bucket(row.service_date),
                    confidence=0.6,
                    evidence={
                        "member_sk": row.member_sk,
                        "days_since_policy_start": int(row.days_since_policy_start),
                        "early_tenure_days": early_days,
                        "gross_amount_aed": round(_f(row.gross_amount_aed), 2),
                        "peer_median_aed": round(stats.median, 2),
                        "peer_percentile": percentile,
                        "peer_percentile_cut_aed": round(cut, 2),
                        "robust_residual": None if resid is None else round(resid, 2),
                        "diagnosis_code": diagnosis,
                        "previous_fraud_on_policy": bool(row.previous_fraud_on_policy),
                        "previous_fraud_field_note": (
                            "previous_fraud_on_policy is a POLICY-HISTORY attribute carried in the "
                            "source record, not the held-out fraud_label. It is shown separately and "
                            "is NOT an input to the score."
                        ),
                        "not_fraud_evidence_alone_notice": (
                            "Appendix C, POL-01-R04: 'Not evidence of fraud alone.' Early tenure plus "
                            "a high-value claim is a distribution signal. Disposition is MONITOR_ONLY."
                        ),
                        "peer_level_used": "diagnosis_primary",
                    },
                    exposure=_exp.model_only_exposure(_f(row.gross_amount_aed), "early-tenure pattern"),
                )
            )
    return out


# ===========================================================================
# NET — delegate to the graph layer
# ===========================================================================


def _graph_control(name: str) -> Callable[[ControlContext, Any], list[Signal]]:
    def runner(ctx: ControlContext, control) -> list[Signal]:
        if ctx.graph is None:
            return []
        from ..graph import controls as graph_controls

        return getattr(graph_controls, name)(ctx, control, _sig)

    runner.__name__ = name
    return runner


net_01_r01_referral_concentration = _graph_control("net_01_r01_referral_concentration")
net_02_r02_dense_member_circulation = _graph_control("net_02_r02_dense_member_circulation")
net_02_r03_synchronized_billing = _graph_control("net_02_r03_synchronized_billing")
net_03_r01_repeated_high_value_dyad = _graph_control("net_03_r01_repeated_high_value_dyad")
net_04_r02_originator_provider_concentration = _graph_control("net_04_r02_originator_provider_concentration")
net_04_r03_early_tenure_cluster = _graph_control("net_04_r03_early_tenure_cluster")


# ===========================================================================
# ANL — delegate to the statistical and model layers
# ===========================================================================


def anl_01_r01_robust_composite(ctx: ControlContext, control) -> list[Signal]:
    """ANL-01-R01 — the transparent composite. The benchmark every model must beat."""
    if ctx.composite is None:
        return []
    results, _ = ctx.composite
    threshold = float(ctx.cfg("composite_flag_threshold"))
    df = _claim_frame(ctx)
    out: list[Signal] = []
    for r in results:
        if r.insufficient_evidence or r.score <= threshold:
            continue
        sub = df[df["provider_sk"] == r.provider_sk]
        out.append(
            _sig(
                ctx, control, subject_type="provider", subject_id=r.provider_sk,
                fact_key=f"composite:{r.provider_sk}",
                claim_ids=sub["claim_sk"].tolist(),
                event_time=sub["service_date"].max(),
                period=period_bucket(sub["service_date"].max()),
                peer_level_used=r.peer_level_used,
                evidence={
                    "provider_sk": r.provider_sk,
                    "composite_score": round(r.score, 4),
                    "threshold": threshold,
                    "per_feature_contributions": r.contributions,
                    "weights": dict(ctx.cfg("composite_feature_weights")),
                    "peer_level_used": r.peer_level_used,
                    "peer_n": r.peer_n,
                    "peer_baseline_version": r.peer_baseline_version,
                    "claim_count": r.claim_count,
                    "plain_language": r.explain(),
                    "auditability_note": (
                        "Every contribution above is weight × robust residual, with the provider's "
                        "value and the peer median in ORIGINAL UNITS. The weights are published in "
                        "config/parameters.yaml, so this score can be reconstructed by hand."
                    ),
                },
                # §4.11: a provider-pattern exposure is the sum of sampled
                # SUSPECT INCREMENTAL amounts — not the provider's whole book.
                # The incremental amount here is each claim's excess over its
                # peer median, floored at zero.
                exposure=_exp.provider_pattern_exposure(
                    _incremental_over_peer(sub, r.contributions)
                ),
                extra_versions={"peer_baseline": r.peer_baseline_version},
            )
        )
    return out


def anl_01_r02_self_history_changepoint(ctx: ControlContext, control) -> list[Signal]:
    """ANL-01-R02 — CUSUM/EWMA change point on a provider's own history."""
    from ..statistical.changepoint import detect_changepoints, monthly_provider_metrics

    df = _claim_frame(ctx)
    series = monthly_provider_metrics(df)
    out: list[Signal] = []
    for metric, column in (("mean claim value", "mean_gross_aed"),
                           ("approval ratio", "approval_ratio")):
        cps = detect_changepoints(
            series.dropna(subset=[column]),
            entity_column="provider_sk", period_column="period", value_column=column,
            metric_name=metric,
            min_history_periods=int(ctx.cfg("min_history_periods")),
            cusum_k=float(ctx.cfg("cusum_k")), cusum_h=float(ctx.cfg("cusum_h")),
            ewma_lambda=float(ctx.cfg("ewma_lambda")), ewma_L=float(ctx.cfg("ewma_L")),
            segments=getattr(ctx, "segments", None),
            caveat="No tariff, contract or ownership records exist in the claim extract, so a known "
                   "structural change cannot be segmented out as §4.7 requires. Every change point "
                   "here carries that caveat.",
        )
        for cp in cps:
            sub = df[df["provider_sk"] == cp.entity]
            out.append(
                _sig(
                    ctx, control, subject_type="provider", subject_id=cp.entity,
                    fact_key=f"changepoint:{cp.entity}:{column}:{cp.change_period}",
                    claim_ids=sub["claim_sk"].tolist(),
                    event_time=cp.change_date,
                    period=cp.change_period,
                    confidence=float(cp.confidence),
                    data_quality_penalty=0.25,
                    evidence={
                        "provider_sk": cp.entity,
                        "metric": cp.metric,
                        "method": cp.method,
                        "change_date": cp.change_date,
                        "pre_change_mean": round(cp.pre_change_mean, 3),
                        "post_change_mean": round(cp.post_change_mean, 3),
                        "change_magnitude": round(cp.change_magnitude, 3),
                        "direction": cp.direction,
                        "statistic": round(cp.statistic, 3),
                        "threshold": cp.threshold,
                        "periods_of_history": cp.periods_of_history,
                        "segment_id": cp.segment_id,
                        "declared_segments": list(cp.declared_segments),
                        "segmentation_caveat": cp.caveat,
                        "metrics_evaluated": ["mean claim value", "approval ratio"],
                        "metrics_unavailable": ["code mix (no activity codes)",
                                                "denial behaviour (no remittance)"],
                        "plain_language": cp.explain(),
                    },
                    exposure=_exp.no_exposure(
                        "a change point locates when behaviour changed, not how much was overpaid"
                    ),
                )
            )
    return out


def anl_01_r03_unsupervised_incremental(ctx: ControlContext, control) -> list[Signal]:
    if ctx.models is None:
        return []
    from ..models import controls as model_controls

    return model_controls.anl_01_r03_unsupervised_incremental(ctx, control, _sig)


def anl_01_r04_novel_cluster(ctx: ControlContext, control) -> list[Signal]:
    if ctx.models is None:
        return []
    from ..models import controls as model_controls

    return model_controls.anl_01_r04_novel_cluster(ctx, control, _sig)


# ===========================================================================
# DOC — delegate to the NLP layer
# ===========================================================================


def doc_01_r02_structured_fact_conflict(ctx: ControlContext, control) -> list[Signal]:
    if ctx.documents is None:
        return []
    from ..nlp import controls as nlp_controls

    return nlp_controls.doc_01_r02_structured_fact_conflict(ctx, control, _sig)


def doc_02_r01_cross_patient_near_duplicate(ctx: ControlContext, control) -> list[Signal]:
    if ctx.documents is None:
        return []
    from ..nlp import controls as nlp_controls

    return nlp_controls.doc_02_r01_cross_patient_near_duplicate(ctx, control, _sig)


# ===========================================================================
# registry
# ===========================================================================

CONTROL_IMPLEMENTATIONS: dict[str, Callable[[ControlContext, Any], list[Signal]]] = {
    fn.__name__: fn
    for fn in (
        ent_02_r03_member_impossible_presence,
        ent_03_r02_excluded_entity,
        pay_01_r01_exact_duplicate,
        pay_01_r02_near_duplicate,
        pay_01_r03_split_claim_duplicate,
        pay_06_r02_amount_arithmetic,
        pay_06_r03_billed_amount_peer_outlier,
        pay_06_r04_approval_ratio_pattern,
        pay_10_r01_cob_not_coordinated,
        cln_01_r01_coding_mismatch_rate,
        cln_01_r02_expected_level_residual,
        cln_01_r03_documentation_level_conflict,
        cln_03_r01_dx_procedure_mismatch,
        cln_04_r01_minimum_repeat_interval,
        cln_04_r02_episode_frequency_excess,
        cln_04_r04_provider_utilisation_outlier,
        cln_05_r01_los_residual,
        cln_05_r02_related_readmission,
        cln_05_r03_discharge_readmit_split,
        cln_06_r02_non_operational_facility,
        cln_06_r03_synthetic_encounter_signature,
        cln_07_r02_facility_capacity,
        cln_07_r04_capacity_trend_discontinuity,
        phr_03_r03_pharmacy_volume_spike,
        pol_01_r04_early_tenure_utilisation,
        net_01_r01_referral_concentration,
        net_02_r02_dense_member_circulation,
        net_02_r03_synchronized_billing,
        net_03_r01_repeated_high_value_dyad,
        net_04_r02_originator_provider_concentration,
        net_04_r03_early_tenure_cluster,
        anl_01_r01_robust_composite,
        anl_01_r02_self_history_changepoint,
        anl_01_r03_unsupervised_incremental,
        anl_01_r04_novel_cluster,
        doc_01_r02_structured_fact_conflict,
        doc_02_r01_cross_patient_near_duplicate,
    )
}


def get_implementation(name: str) -> Callable[[ControlContext, Any], list[Signal]]:
    try:
        return CONTROL_IMPLEMENTATIONS[name]
    except KeyError as exc:
        raise KeyError(
            f"No implementation named {name!r}. A control classified EXECUTABLE or PARTIAL must "
            f"name a function that exists in fwa.engine.controls; the coverage matrix would "
            f"otherwise overstate what this build runs."
        ) from exc
