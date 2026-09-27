"""Unlocked implementations for the NET family.

Each function here runs only when a dataset unlock (`rules/unlocks/NET.yaml`)
is satisfied by the loaded file. See :mod:`fwa.engine.unlocks`.

The graph layer (:mod:`fwa.graph`) builds its snapshots from claims alone, so
it is bipartite: members and agents connect to providers, never provider to
provider. The controls here use the relationship tables a richer file carries —
DIRECTED referrals, provider administrative identifiers, employers, contracts,
complaints — which is what makes reciprocity, closed chains and shared-identity
links observable at all. Every network finding is a reason to look: none of
them establishes an overpayment, and exposure is stated conservatively.
"""

from __future__ import annotations

import re
from typing import Any, Callable

import numpy as np
import pandas as pd
from scipy import stats as _st

from ...cases import exposure as _exp
from ...presentation import aed, count_phrase, pct, plain_date, times_phrase
from ..controls import _claim_frame, _f, _sig
from ..evallib import is_missing, period_bucket

__all__ = ["IMPLEMENTATIONS"]


# ---------------------------------------------------------------------------
# shared helpers
# ---------------------------------------------------------------------------


def _table(ctx, name: str) -> pd.DataFrame:
    ds = ctx.dataset
    if ds is None:
        return pd.DataFrame()
    try:
        frame = ds.get(name)
    except Exception:
        return pd.DataFrame()
    if frame is None or frame.empty:
        return pd.DataFrame(columns=[] if frame is None else frame.columns)
    if "tenant_id" in frame.columns and frame["tenant_id"].notna().any():
        frame = frame[frame["tenant_id"].isna() | (frame["tenant_id"].astype(str) == str(ctx.tenant_id))]
    return frame


def _s(value: Any) -> str:
    return "" if is_missing(value) else str(value).strip()


def _norm(value: Any) -> str:
    return re.sub(r"[\s\-/]+", "_", _s(value).lower())


def _claims(ctx) -> pd.DataFrame:
    df = _claim_frame(ctx)
    if df is None or df.empty or "claim_sk" not in df.columns:
        return pd.DataFrame()
    df = df.copy()
    df["claim_sk"] = df["claim_sk"].astype(str)
    for col in ("member_sk", "provider_sk"):
        if col in df.columns:
            df[col] = df[col].map(_s)
    if "service_date" in df.columns:
        df["service_date"] = pd.to_datetime(df["service_date"], errors="coerce")
    if "gross_amount_aed" not in df.columns:
        df["gross_amount_aed"] = pd.to_numeric(df.get("gross_amount"), errors="coerce")
    df["gross_amount_aed"] = pd.to_numeric(df["gross_amount_aed"], errors="coerce").fillna(0.0)
    return df.drop_duplicates("claim_sk")


def _referrals(ctx) -> pd.DataFrame:
    """Directed provider→provider referrals with both ends known and distinct."""
    ref = _table(ctx, "referral")
    if ref.empty:
        return pd.DataFrame()
    ref = ref.copy()
    a = ref["referrer_provider_sk"].map(_s) if "referrer_provider_sk" in ref.columns else pd.Series("", index=ref.index)
    b = ref["recipient_provider_sk"].map(_s) if "recipient_provider_sk" in ref.columns else pd.Series("", index=ref.index)
    if "referrer_id" in ref.columns:
        a = a.where(a != "", ref["referrer_id"].map(_s))
    if "recipient_id" in ref.columns:
        b = b.where(b != "", ref["recipient_id"].map(_s))
    ref["src"], ref["dst"] = a, b
    ref = ref[(ref["src"] != "") & (ref["dst"] != "") & (ref["src"] != ref["dst"])]
    if ref.empty:
        return ref
    ref["referral_sk"] = ref["referral_sk"].map(_s) if "referral_sk" in ref.columns else ref.index.astype(str)
    ref["member_sk"] = ref["member_sk"].map(_s) if "member_sk" in ref.columns else ""
    ref["resulting_claim_sk"] = ref["resulting_claim_sk"].map(_s) if "resulting_claim_sk" in ref.columns else ""
    ref["referral_date"] = pd.to_datetime(ref["referral_date"], errors="coerce") \
        if "referral_date" in ref.columns else pd.NaT
    return ref


def _providers(ctx) -> pd.DataFrame:
    prov = _table(ctx, "provider")
    if prov.empty or "provider_sk" not in prov.columns:
        return pd.DataFrame()
    prov = prov.copy()
    prov["provider_sk"] = prov["provider_sk"].map(_s)
    return prov.drop_duplicates("provider_sk").set_index("provider_sk")


def _amounts(claims: pd.DataFrame, ids) -> dict[str, float]:
    ids = {str(i) for i in ids if _s(i)}
    if claims.empty or not ids:
        return {}
    sub = claims[claims["claim_sk"].isin(ids)]
    return dict(zip(sub["claim_sk"], sub["gross_amount_aed"].astype(float)))


def _at_stake(claim_amounts: dict[str, float], what: str) -> _exp.Exposure:
    """The value of the distinct claims a network finding touches — NOT established.

    Each claim is counted once (keyed by claim_sk, as :func:`network_exposure`
    requires), but the total is a gross amount resting on a pattern, so it is
    shown beside "exposure not yet established" and never enters a savings total.
    """
    total = sum(float(v) for v in claim_amounts.values())
    return _exp.Exposure(
        total,
        f"Network pattern ({what}): {len(claim_amounts)} DISTINCT claim(s) summed once, gross "
        f"AED {total:,.2f}. {_exp.EXPOSURE_NOT_ESTABLISHED} — a pattern shows where to look, not "
        f"how much of this was overpaid.",
        established=False,
    )


_ADMIN_TOKENS = ("bank_account_token", "phone_token", "address_token")
_TOKEN_WORDS = {"bank_account_token": "bank account", "phone_token": "phone number",
                "address_token": "address", "owner_entity_id": "owner"}


def _shared_tokens(prov: pd.DataFrame, a: str, b: str, columns=_ADMIN_TOKENS) -> list[str]:
    if a not in prov.index or b not in prov.index:
        return []
    out = []
    for col in columns:
        if col in prov.columns:
            va, vb = _s(prov.at[a, col]), _s(prov.at[b, col])
            if va and va == vb:
                out.append(col)
    return out


def _same_owner(prov: pd.DataFrame, a: str, b: str) -> bool:
    return bool(_shared_tokens(prov, a, b, ("owner_entity_id",)))


# ===========================================================================
# NET-01-R02 — reciprocal referral loop
# ===========================================================================


def net_01_r02_reciprocal_referral_loop(ctx, control) -> list:
    """A→B and B→A referral flows both far stronger than each side's opportunity implies.

    Opportunity: how much of A's outgoing referrals would reach B if A referred in
    proportion to B's share of ALL referrals received. Each direction's LIFT is its
    observed share over that expectation; both must exceed the governed lift and
    both must carry a minimum count. Declared exclusion: two providers under the
    same declared owner are one organisation's team, and are not flagged.
    """
    ref = _referrals(ctx)
    if ref.empty:
        return []
    min_n = int(ctx.cfg("net01r02_min_reciprocal_referrals"))
    min_lift = float(ctx.cfg("net01r02_min_lift"))
    min_share = float(ctx.cfg("net01r02_min_direction_share"))
    prov = _providers(ctx)
    claims = _claims(ctx)

    pair = ref.groupby(["src", "dst"]).size()
    out_deg = ref.groupby("src").size()
    in_share = ref.groupby("dst").size() / float(len(ref))
    out = []
    seen: set[tuple[str, str]] = set()
    for (a, b), n_ab in pair.items():
        if (b, a) not in pair.index:
            continue
        key = tuple(sorted((a, b)))
        if key in seen:
            continue
        seen.add(key)
        n_ba = int(pair[(b, a)])
        n_ab = int(n_ab)
        if min(n_ab, n_ba) < min_n:
            continue
        share_ab, share_ba = n_ab / float(out_deg[a]), n_ba / float(out_deg[b])
        lift_ab = share_ab / max(float(in_share.get(b, 0.0)), 1e-9)
        lift_ba = share_ba / max(float(in_share.get(a, 0.0)), 1e-9)
        if min(lift_ab, lift_ba) < min_lift or min(share_ab, share_ba) < min_share:
            continue
        if not prov.empty and _same_owner(prov, a, b):
            continue  # declared exclusion: one organisation's multidisciplinary team
        a, b = (a, b) if n_ab >= n_ba else (b, a)
        n_ab, n_ba = max(n_ab, n_ba), min(n_ab, n_ba)
        share_ab, share_ba = n_ab / float(out_deg[a]), n_ba / float(out_deg[b])
        loop = ref[((ref["src"] == a) & (ref["dst"] == b)) | ((ref["src"] == b) & (ref["dst"] == a))]
        last = loop["referral_date"].max()
        claim_amounts = _amounts(claims, loop["resulting_claim_sk"])
        out.append(_sig(
            ctx, control, subject_type="provider", subject_id=a,
            fact_key=f"refloop:{key[0]}:{key[1]}",
            claim_ids=sorted(claim_amounts),
            event_time=last,
            period=period_bucket(last),
            confidence=0.6,
            evidence={
                "provider_sk": a,
                "counterpart_provider_sk": b,
                "community_id": f"{key[0]}|{key[1]}",
                "referrals_a_to_b": n_ab,
                "referrals_b_to_a": n_ba,
                "share_of_a_outgoing_to_b": round(share_ab, 3),
                "share_of_b_outgoing_to_a": round(share_ba, 3),
                "expected_share_a_to_b": round(float(in_share.get(b, 0.0)), 4),
                "expected_share_b_to_a": round(float(in_share.get(a, 0.0)), 4),
                "lift_a_to_b": round(share_ab / max(float(in_share.get(b, 0.0)), 1e-9), 1),
                "lift_b_to_a": round(share_ba / max(float(in_share.get(a, 0.0)), 1e-9), 1),
                "distinct_members": int(loop["member_sk"].replace("", np.nan).nunique()),
                "first_referral": str(loop["referral_date"].min().date()) if not pd.isna(loop["referral_date"].min()) else None,
                "last_referral": str(last.date()) if not pd.isna(last) else None,
                "min_lift": min_lift,
                "edge_type": "referral (directed, observed)",
                "plain_language": (
                    f"These two providers refer to each other: {n_ab} referrals one way "
                    f"({pct(share_ab)} of the first provider's referrals) and {n_ba} back "
                    f"({pct(share_ba)} of the second's), each "
                    f"{times_phrase(min(share_ab / max(float(in_share.get(b, 0.0)), 1e-9), share_ba / max(float(in_share.get(a, 0.0)), 1e-9)), 1.0, 'what their share of all referrals would predict')}."
                ),
                "what_the_reviewer_must_verify": (
                    "Whether the two providers work as a recognised care team; review a sample of "
                    "the referred patients and the reasons for referral."
                ),
            },
            exposure=_exp.no_exposure("a referral loop is graph evidence; it carries no amount of its own"),
        ))
    return out


# ===========================================================================
# NET-01-R03 — referrals converting to high-cost services
# ===========================================================================


def net_01_r03_high_cost_conversion(ctx, control) -> list:
    """Referred members of one recipient turn into high-cost services far more than expected.

    A referral CONVERTS when the claim it resulted in (the referral's own
    ``resulting_claim_sk``, else the member's first claim at the recipient within
    the conversion window) carries a line in the top tail of line values. The
    expected number of conversions is indirectly standardised on the recipient's
    specialty and the referral's stated reason (declared exclusions: case mix and
    referral indication), and the observed count must exceed it by the governed
    ratio with a Poisson tail probability below the governed level.
    """
    ref = _referrals(ctx)
    lines = _table(ctx, "claim_line")
    claims = _claims(ctx)
    if ref.empty or lines.empty or claims.empty or "net_amount" not in lines.columns:
        return []
    q = float(ctx.cfg("net01r03_high_cost_line_percentile"))
    window = int(ctx.cfg("net01r03_conversion_window_days"))
    min_ref = int(ctx.cfg("net01r03_min_referrals"))
    min_ratio = float(ctx.cfg("net01r03_min_observed_to_expected"))
    alpha = float(ctx.cfg("net01r03_max_tail_probability"))

    lines = lines.copy()
    lines["claim_sk"] = lines["claim_sk"].astype(str)
    lines["net_amount"] = pd.to_numeric(lines["net_amount"], errors="coerce")
    positive = lines["net_amount"][lines["net_amount"] > 0]
    if positive.empty:
        return []
    cut = float(positive.quantile(q))
    high = lines[lines["net_amount"] >= cut].groupby("claim_sk")["net_amount"].sum()

    ref = ref.copy()
    missing = ref["resulting_claim_sk"] == ""
    if missing.any() and "service_date" in claims.columns:
        cand = claims[["claim_sk", "member_sk", "provider_sk", "service_date"]].rename(
            columns={"provider_sk": "dst", "claim_sk": "_c"})
        m = ref[missing][["referral_sk", "member_sk", "dst", "referral_date"]].merge(
            cand, on=["member_sk", "dst"], how="inner")
        m = m[(m["service_date"] >= m["referral_date"])
              & ((m["service_date"] - m["referral_date"]).dt.days <= window)]
        first = m.sort_values("service_date").drop_duplicates("referral_sk").set_index("referral_sk")["_c"]
        ref.loc[missing, "resulting_claim_sk"] = ref.loc[missing, "referral_sk"].map(first).fillna("")
    ref["converted"] = ref["resulting_claim_sk"].isin(high.index)
    ref["high_value"] = ref["resulting_claim_sk"].map(high).fillna(0.0)
    prov = _providers(ctx)
    ref["specialty"] = ref["dst"].map(
        lambda p: _norm(prov.at[p, "specialty"]) if not prov.empty and p in prov.index and "specialty" in prov.columns else "")
    ref["reason"] = ref["reason_code"].map(_norm) if "reason_code" in ref.columns else ""

    out = []
    for dst, sub in ref.groupby("dst", sort=True):
        n = len(sub)
        if n < min_ref:
            continue
        # Leave the provider out of its own expectation, so one heavy recipient
        # cannot raise the bar it is measured against.
        others = ref[ref["dst"] != dst]
        rates = others.groupby(["specialty", "reason"])["converted"].mean()
        overall = float(others["converted"].mean()) if len(others) else float(ref["converted"].mean())
        expected = float(sum(rates.get((s, r), overall) for s, r in zip(sub["specialty"], sub["reason"])))
        observed = int(sub["converted"].sum())
        if observed == 0:
            continue
        ratio = observed / max(expected, 1e-9)
        tail = float(_st.poisson.sf(observed - 1, max(expected, 1e-9)))
        if ratio < min_ratio or tail > alpha:
            continue
        conv = sub[sub["converted"]]
        last = sub["referral_date"].max()
        referrers = sub["src"].value_counts()
        out.append(_sig(
            ctx, control, subject_type="provider", subject_id=dst,
            fact_key=f"refconvert:{dst}",
            claim_ids=sorted(set(conv["resulting_claim_sk"])),
            event_time=last,
            period=period_bucket(last),
            confidence=0.6,
            peer_level_used="recipient specialty × referral reason",
            evidence={
                "provider_sk": dst,
                "community_id": f"referrals-to:{dst}",
                "referrals_received": n,
                "converted_to_high_cost": observed,
                "conversion_rate": round(observed / n, 3),
                "expected_conversions": round(expected, 2),
                "observed_to_expected": round(ratio, 2),
                "poisson_tail_probability": float(f"{tail:.3g}"),
                "high_cost_line_threshold_aed": round(cut, 2),
                "high_cost_value_aed": round(float(conv["high_value"].sum()), 2),
                "top_referrers": [{"provider_sk": k, "referrals": int(v)} for k, v in referrers.head(3).items()],
                "standardised_on": ["recipient specialty", "referral reason"],
                "plain_language": (
                    f"{observed} of {n} patients referred to this provider "
                    f"({pct(observed / n)}) went on to a high-cost service (a line of "
                    f"{aed(cut)} or more); for similar referrals elsewhere about "
                    f"{expected:.1f} would be expected."
                ),
                "what_the_reviewer_must_verify": (
                    "Check the reasons for referral and the clinical need for a sample of the "
                    "high-cost services."
                ),
            },
            exposure=_at_stake(_amounts(claims, conv["resulting_claim_sk"]), "referred patients converted to high-cost services"),
        ))
    return out


# ===========================================================================
# NET-01-R04 — closed downstream chain with hidden links
# ===========================================================================


_DOWNSTREAM = ("lab", "imaging", "radiolog", "pharm", "diagnostic", "scan")


def _is_downstream(prov: pd.DataFrame, p: str) -> bool:
    if p not in prov.index:
        return False
    kind = " ".join(_norm(prov.at[p, c]) for c in ("provider_type", "facility_type", "specialty")
                    if c in prov.columns)
    return any(k in kind for k in _DOWNSTREAM)


def net_01_r04_closed_downstream_chain(ctx, control) -> list:
    """A referrer's lab/imaging/pharmacy referrals stay inside providers it is covertly linked to.

    "Linked" means an exactly shared bank account, phone or address token between
    the referrer and the downstream provider. A downstream provider under the
    SAME DECLARED OWNER is a disclosed, integrated group and is excluded, as is one
    the payer's own contract places in a narrow network (declared exclusions:
    ownership / narrow network).
    """
    ref = _referrals(ctx)
    prov = _providers(ctx)
    claims = _claims(ctx)
    if ref.empty or prov.empty:
        return []
    min_n = int(ctx.cfg("net01r04_min_downstream_referrals"))
    min_share = float(ctx.cfg("net01r04_min_retained_share"))
    narrow = set()
    contract = _table(ctx, "contract")
    if not contract.empty and "network_tier" in contract.columns:
        narrow = set(contract.loc[contract["network_tier"].map(_norm).str.contains("narrow"), "provider_sk"].map(_s))

    down = ref[ref["dst"].map(lambda p: _is_downstream(prov, p))].copy()
    if down.empty:
        return []
    out = []
    for src, sub in down.groupby("src", sort=True):
        if len(sub) < min_n:
            continue
        links = {d: _shared_tokens(prov, src, d) for d in sub["dst"].unique()}
        declared = {d for d in links if _same_owner(prov, src, d)}
        chain = {d for d, t in links.items() if t and d not in declared and d not in narrow}
        if not chain:
            continue
        retained = sub[sub["dst"].isin(chain)]
        share = len(retained) / float(len(sub))
        if share < min_share:
            continue
        claim_amounts = _amounts(claims, retained["resulting_claim_sk"])
        last = retained["referral_date"].max()
        members = retained["member_sk"].replace("", np.nan).nunique()
        link_words = sorted({_TOKEN_WORDS[t] for d in chain for t in links[d]})
        out.append(_sig(
            ctx, control, subject_type="provider", subject_id=src,
            fact_key=f"closedchain:{src}",
            claim_ids=sorted(claim_amounts),
            event_time=last,
            period=period_bucket(last),
            confidence=0.65,
            evidence={
                "provider_sk": src,
                "community_id": f"chain:{src}",
                "downstream_referrals": int(len(sub)),
                "retained_in_linked_chain": int(len(retained)),
                "retained_share": round(share, 3),
                "linked_downstream_providers": [
                    {"provider_sk": d, "provider_type": _s(prov.at[d, "provider_type"]) if "provider_type" in prov.columns else "",
                     "shared_identifiers": [_TOKEN_WORDS[t] for t in links[d]],
                     "referrals": int((retained["dst"] == d).sum())} for d in sorted(chain)
                ],
                "declared_same_owner_excluded": sorted(declared),
                "narrow_network_excluded": sorted(set(links) & narrow),
                "distinct_members_retained": int(members),
                "plain_language": (
                    f"{len(retained)} of this provider's {len(sub)} laboratory, imaging and pharmacy "
                    f"referrals ({pct(share)}, covering {count_phrase(members, 'patient')}) go to "
                    f"{count_phrase(len(chain), 'provider')} that share its {' or '.join(link_words)} "
                    f"but declare a different owner."
                ),
                "what_the_reviewer_must_verify": (
                    "Confirm who owns and runs the linked providers, and whether an insurer narrow "
                    "network explains the referral pattern."
                ),
            },
            exposure=_at_stake(claim_amounts, "referrals kept inside a linked chain"),
        ))
    return out


# ===========================================================================
# NET-02-R01 — separate providers sharing administrative identifiers
# ===========================================================================


def net_02_r01_shared_admin_identity(ctx, control) -> list:
    """Legally distinct providers share an exact bank, phone or address token.

    Exact matches only (a fuzzy match can never be a hard flag). Declared
    exclusions: providers under the same declared owner (a known group), and a
    token shared by more providers than a shared-services arrangement plausibly
    explains is recorded as a shared service rather than flagged.
    """
    prov = _providers(ctx)
    if prov.empty:
        return []
    max_group = int(ctx.cfg("net02r01_shared_service_max_providers"))
    frame = prov.reset_index()
    legal = next((c for c in ("regulator_id", "licence_no", "source_provider_id") if c in frame.columns), None)
    links: dict[tuple[str, ...], list[tuple[str, str]]] = {}
    for col in _ADMIN_TOKENS:
        if col not in frame.columns:
            continue
        vals = frame[["provider_sk", col] + ([legal] if legal else [])].copy()
        vals[col] = vals[col].map(_s)
        vals = vals[vals[col] != ""]
        for token, grp in vals.groupby(col, sort=True):
            members = sorted(set(grp["provider_sk"]))
            if len(members) < 2 or len(members) > max_group:
                continue
            if legal and grp[legal].map(_s).nunique() < 2:
                continue  # one legal entity under several records: an identity question, not a link
            owners = {_s(prov.at[p, "owner_entity_id"]) for p in members} if "owner_entity_id" in prov.columns else {""}
            if len(owners) == 1 and "" not in owners:
                continue  # declared exclusion: known group under one declared owner
            links.setdefault(tuple(members), []).append((col, token))
    claims = _claims(ctx)
    out = []
    for members, shared in sorted(links.items()):
        words = sorted({_TOKEN_WORDS[c] for c, _ in shared})
        n_claims = int(claims["provider_sk"].isin(members).sum()) if not claims.empty and "provider_sk" in claims.columns else 0
        owners = [_s(prov.at[p, "owner_entity_id"]) if "owner_entity_id" in prov.columns else "" for p in members]
        out.append(_sig(
            ctx, control, subject_type="provider", subject_id=members[0],
            fact_key=f"sharedadmin:{'|'.join(members)}",
            confidence=0.9 if any(c != "address_token" for c, _ in shared) else 0.6,
            evidence={
                "provider_sk": members[0],
                "linked_provider_sks": list(members),
                "community_id": "|".join(members),
                "shared_identifiers": [{"identifier": _TOKEN_WORDS[c], "token": t} for c, t in shared],
                "declared_owners": owners,
                "match_type": "exact token match",
                "claims_billed_by_linked_providers": n_claims,
                "plain_language": (
                    f"{count_phrase(len(members), 'provider')} registered as separate organisations "
                    f"with different declared owners share the same {' and '.join(words)}."
                ),
                "what_the_reviewer_must_verify": (
                    "Confirm the match is exact and not a shared-services arrangement or a known "
                    "group, then record the link for the network review."
                ),
            },
            exposure=_exp.no_exposure("a shared identifier is a graph edge; it carries no amount"),
        ))
    return out


# ===========================================================================
# NET-02-R04 — peripheral provider suddenly central, no contract to explain it
# ===========================================================================


def net_02_r04_structural_change(ctx, control) -> list:
    """A provider moves from the network's edge to its centre within a few months.

    Centrality is each provider's monthly percentile rank of distinct members
    served. A RISE is a provider averaging at or below the peripheral percentile
    over the window before a month and at or above the central percentile over
    the window from that month on, with material value in the later window.
    Declared exclusions: a contract starting in the run-up (new contract), a
    same-owner provider that stops billing as this one rises (merger), and a
    provider credentialed during the window (a new entrant, not a change).
    A recorded ownership change before the rise is reported as the structural
    change the catalogue names, not required.
    """
    claims = _claims(ctx)
    prov = _providers(ctx)
    if claims.empty or prov.empty or "service_date" not in claims.columns:
        return []
    w = int(ctx.cfg("net02r04_window_months"))
    low = float(ctx.cfg("net02r04_peripheral_percentile"))
    high = float(ctx.cfg("net02r04_central_percentile"))
    min_value = float(ctx.cfg("net02r04_min_post_value_aed"))

    df = claims.dropna(subset=["service_date"])
    df = df[df["provider_sk"] != ""]
    if df.empty:
        return []
    df = df.assign(month=df["service_date"].dt.to_period("M"))
    deg = df.groupby(["provider_sk", "month"])["member_sk"].nunique().unstack(fill_value=0)
    val = df.groupby(["provider_sk", "month"])["gross_amount_aed"].sum().unstack(fill_value=0.0)
    months = sorted(deg.columns)
    if len(months) < 2 * w:
        return []
    deg = deg.reindex(columns=months, fill_value=0)
    val = val.reindex(index=deg.index, columns=months, fill_value=0.0)
    rank = deg.rank(axis=0, pct=True)

    contracts = _table(ctx, "contract")
    contract_starts: dict[str, list[pd.Timestamp]] = {}
    if not contracts.empty and {"provider_sk", "valid_from"} <= set(contracts.columns):
        c = contracts.assign(_p=contracts["provider_sk"].map(_s),
                             _d=pd.to_datetime(contracts["valid_from"], errors="coerce"))
        contract_starts = c.dropna(subset=["_d"]).groupby("_p")["_d"].apply(list).to_dict()
    cred = pd.to_datetime(prov["credentialing_date"], errors="coerce") if "credentialing_date" in prov.columns \
        else pd.Series(pd.NaT, index=prov.index)
    owned = pd.to_datetime(prov["ownership_changed_on"], errors="coerce") if "ownership_changed_on" in prov.columns \
        else pd.Series(pd.NaT, index=prov.index)
    owner = prov["owner_entity_id"].map(_s) if "owner_entity_id" in prov.columns else pd.Series("", index=prov.index)

    rv = rank.to_numpy()
    out = []
    for i, p in enumerate(rank.index):
        for t in range(w, len(months) - w + 1):
            pre = float(rv[i, t - w:t].mean())
            post = float(rv[i, t:t + w].mean())
            if pre > low or post < high:
                continue
            start = months[t].to_timestamp()
            window_start = months[t - w].to_timestamp()
            post_value = float(val.iloc[i, t:t + w].sum())
            if post_value < min_value:
                continue
            c_start = pd.to_datetime(cred.get(p), errors="coerce")
            if not pd.isna(c_start) and c_start >= window_start:
                break  # declared exclusion: a new entrant, not a change in an existing node
            if any(window_start <= d <= start + pd.offsets.MonthEnd(1) for d in contract_starts.get(p, [])):
                break  # declared exclusion: a new contract explains the growth
            siblings = [q for q in owner.index if q != p and owner.get(p) and owner.get(q) == owner.get(p)]
            merged = [q for q in siblings if q in deg.index
                      and deg.loc[q, months[t - w:t]].sum() > 0 and deg.loc[q, months[t:t + w]].sum() == 0]
            if merged:
                break  # declared exclusion: a merger moved another provider's patients here
            pre_value = float(val.iloc[i, t - w:t].sum())
            pre_members = int(deg.iloc[i, t - w:t].sum())
            post_members = int(deg.iloc[i, t:t + w].sum())
            oc = owned.get(p)
            ownership_note = None
            if not pd.isna(oc) and window_start - pd.DateOffset(months=w) <= oc <= start + pd.offsets.MonthEnd(1):
                ownership_note = str(pd.Timestamp(oc).date())
            post_claims = df[(df["provider_sk"] == p) & (df["month"] >= months[t]) & (df["month"] <= months[t + w - 1])]
            out.append(_sig(
                ctx, control, subject_type="provider", subject_id=p,
                fact_key=f"structural:{p}:{months[t]}",
                claim_ids=sorted(post_claims["claim_sk"]),
                event_time=start,
                period=str(months[t]),
                confidence=0.6 if ownership_note else 0.45,
                evidence={
                    "provider_sk": p,
                    "community_id": f"provider:{p}",
                    "change_month": str(months[t]),
                    "window_months": w,
                    "centrality_percentile_before": round(pre, 3),
                    "centrality_percentile_after": round(post, 3),
                    "members_served_before": pre_members,
                    "members_served_after": post_members,
                    "value_before_aed": round(pre_value, 2),
                    "value_after_aed": round(post_value, 2),
                    "ownership_changed_on": ownership_note,
                    "contract_starts_checked": [str(d.date()) for d in contract_starts.get(p, [])],
                    "exclusions_checked": ["new contract", "merger", "new entrant"],
                    "centrality_measure": "monthly percentile rank of distinct members served",
                    "plain_language": (
                        f"In the {w} months before {plain_date(start)} this provider saw about "
                        f"{pre_members / w:.0f} patients a month ({aed(pre_value)} in total, busier than "
                        f"{pct(pre)} of providers); in the {w} months from then it saw about "
                        f"{post_members / w:.0f} a month ({aed(post_value)}, busier than {pct(post)})"
                        + (f", after its ownership changed on {plain_date(ownership_note)}" if ownership_note else "")
                        + "; no new contract or merger explains the rise."
                    ),
                    "what_the_reviewer_must_verify": (
                        "Check for a merger, ownership change or new contract, and review the new "
                        "connections and the money flowing through them."
                    ),
                },
                exposure=_exp.no_exposure(
                    "a structural change locates when a network position changed, not an overpayment"
                ),
            ))
            break
    return out


# ===========================================================================
# NET-03-R02 — provider serving a small closed employer group
# ===========================================================================


def net_03_r02_closed_member_group(ctx, control) -> list:
    """A provider's patients come overwhelmingly from one employer that barely uses anyone else.

    Declared exclusions: an on-site or occupational clinic (employer on-site
    care), and clinic catchment — the employer's share of the provider's patients
    is compared with that employer's share of all members in the provider's own
    emirate, so a local clinic serving a local workforce is not singled out.
    """
    claims = _claims(ctx)
    members = _table(ctx, "member")
    prov = _providers(ctx)
    if claims.empty or members.empty or "employer_id" not in members.columns:
        return []
    min_members = int(ctx.cfg("net03r02_min_members"))
    min_share = float(ctx.cfg("net03r02_min_employer_share"))
    min_lift = float(ctx.cfg("net03r02_min_lift"))
    max_providers = int(ctx.cfg("net03r02_max_group_providers"))
    min_closure = float(ctx.cfg("net03r02_min_closure_share"))

    mem = members.copy()
    mem["member_sk"] = mem["member_sk"].map(_s)
    mem["employer"] = mem["employer_id"].map(_s)
    if "sponsor_id" in mem.columns:
        mem["employer"] = mem["employer"].where(mem["employer"] != "", mem["sponsor_id"].map(_s))
    mem["emirate"] = mem["emirate"].map(_s) if "emirate" in mem.columns else ""
    mem = mem.drop_duplicates("member_sk")
    emp = dict(zip(mem["member_sk"], mem["employer"]))
    df = claims.assign(employer=claims["member_sk"].map(emp).fillna(""))
    df = df[(df["employer"] != "") & (df["provider_sk"] != "")]
    if df.empty:
        return []

    base = mem[mem["employer"] != ""]
    emirate_emp_share = base.groupby("emirate")["employer"].value_counts(normalize=True)
    all_emp_share = base["employer"].value_counts(normalize=True)
    pm = df.groupby("provider_sk")["member_sk"].nunique()

    out = []
    seen_groups: set[tuple[str, tuple[str, ...]]] = set()
    for p, sub in df.groupby("provider_sk", sort=True):
        n_mem = int(pm[p])
        if n_mem < min_members:
            continue
        if not prov.empty and p in prov.index:
            kind = " ".join(_norm(prov.at[p, c]) for c in ("provider_type", "facility_type", "specialty")
                            if c in prov.columns)
            if any(k in kind for k in ("onsite", "on_site", "occupational", "employer")):
                continue  # declared exclusion: employer on-site care
        per_emp = sub.groupby("employer")["member_sk"].nunique().sort_values(ascending=False)
        top, top_n = str(per_emp.index[0]), int(per_emp.iloc[0])
        share = top_n / float(n_mem)
        if share < min_share:
            continue
        p_emirate = _s(prov.at[p, "emirate"]) if not prov.empty and p in prov.index and "emirate" in prov.columns else ""
        local = float(emirate_emp_share.get((p_emirate, top), np.nan)) if p_emirate else np.nan
        expected = local if not np.isnan(local) and local > 0 else float(all_emp_share.get(top, 0.0))
        lift = share / max(expected, 1e-9)
        if lift < min_lift:
            continue
        group_members = set(sub.loc[sub["employer"] == top, "member_sk"])
        their = df[df["member_sk"].isin(group_members)]
        by_prov = their["provider_sk"].value_counts()
        core = by_prov.head(max_providers)
        closure = float(core.sum()) / float(by_prov.sum())
        if closure < min_closure or p not in core.index:
            continue
        group_key = (top, tuple(sorted(core.index)))
        if group_key in seen_groups:
            continue
        seen_groups.add(group_key)
        subject = str(core.index[0])
        grp_claims = their[their["provider_sk"].isin(core.index)]
        last = grp_claims["service_date"].max() if "service_date" in grp_claims.columns else None
        out.append(_sig(
            ctx, control, subject_type="provider", subject_id=subject,
            fact_key=f"closedgroup:{top}:{'|'.join(sorted(core.index))}",
            claim_ids=sorted(grp_claims["claim_sk"]),
            event_time=last,
            period=period_bucket(last),
            confidence=0.55,
            evidence={
                "provider_sk": subject,
                "group_providers": sorted(core.index),
                "community_id": f"{top}:{'|'.join(sorted(core.index))}",
                "employer_id": top,
                "provider_distinct_members": n_mem,
                "members_from_employer": top_n,
                "employer_share_of_provider_members": round(share, 3),
                "employer_share_of_local_membership": round(expected, 4),
                "lift": round(lift, 1),
                "group_claims_at_these_providers_share": round(closure, 3),
                "catchment_basis": f"members in {p_emirate}" if p_emirate and not np.isnan(local) else "all members",
                "plain_language": (
                    f"{pct(share)} of the {n_mem} patients this provider serves ({top_n}) work for one "
                    f"employer, which accounts for only {pct(expected)} of members locally; "
                    f"{pct(closure)} of that group's claims go to the same "
                    f"{count_phrase(len(core), 'provider')}."
                ),
                "what_the_reviewer_must_verify": (
                    "Check whether the provider is an on-site or contracted employer clinic, and how "
                    "the patients in the group came to it."
                ),
            },
            exposure=_at_stake(_amounts(claims, grp_claims["claim_sk"]), "claims of a closed employer group"),
        ))
    return out


# ===========================================================================
# NET-03-R03 — inducement signature
# ===========================================================================


def net_03_r03_inducement_signature(ctx, control) -> list:
    """Zero patient share and repeated high-value services co-occur in one provider's patients.

    Limb 1 counts claims with no patient share where the member's product
    normally charges one (a product whose benefit rules all waive the share is an
    approved programme — declared exclusion). Limb 2 counts patients with repeated
    claims carrying a top-tail line value. Both must exceed the pooled rate of all
    providers by the governed multiple, and enough patients must show both. The
    non-medical substitution limb is reported when dispensing records exist.
    """
    claims = _claims(ctx)
    lines = _table(ctx, "claim_line")
    if claims.empty or lines.empty or "net_amount" not in lines.columns:
        return []
    min_claims = int(ctx.cfg("net03r03_min_claims"))
    multiple = float(ctx.cfg("net03r03_rate_multiple"))
    q = float(ctx.cfg("net03r03_high_value_line_percentile"))
    min_both = int(ctx.cfg("net03r03_min_members_with_both"))

    lines = lines.copy()
    lines["claim_sk"] = lines["claim_sk"].astype(str)
    lines["net_amount"] = pd.to_numeric(lines["net_amount"], errors="coerce")
    df = claims.copy()
    if "patient_share" in df.columns and pd.to_numeric(df["patient_share"], errors="coerce").notna().any():
        df["_share"] = pd.to_numeric(df["patient_share"], errors="coerce")
    elif "patient_share" in lines.columns:
        df["_share"] = df["claim_sk"].map(pd.to_numeric(lines["patient_share"], errors="coerce").groupby(lines["claim_sk"]).sum())
    else:
        return []
    df = df[df["_share"].notna() & (df["gross_amount_aed"] > 0)]
    if df.empty:
        return []

    # Approved programmes: a claim is only EXPECTED to carry a patient share when
    # the member's product charges one for the benefit family of at least one of
    # its billed lines (a case-rate admission bills every line under the case
    # rate's family). A product that waives the share for that family is an
    # approved programme, and its zero-share claims are not counted.
    eligible_products = None
    benefit = _table(ctx, "benefit_rule_version")
    cover = _table(ctx, "coverage_period")
    ref = _table(ctx, "activity_code_reference")
    if not benefit.empty and not cover.empty and {"product", "patient_share_pct", "service_family"} <= set(benefit.columns):
        pct_share = pd.to_numeric(benefit["patient_share_pct"], errors="coerce").fillna(0)
        charging = {(a, b) for a, b, v in zip(benefit["product"].map(_s), benefit["service_family"].map(_norm), pct_share) if v > 0}
        eligible_products = {a for a, _ in charging}
        prod = cover.assign(_m=cover["member_sk"].map(_s), _p=cover["product"].map(_s)).drop_duplicates("_m", keep="last")
        product_of = dict(zip(prod["_m"], prod["_p"]))
        fam = {}
        cfam = {}
        if not ref.empty and {"activity_code", "service_family"} <= set(ref.columns):
            fam = dict(zip(ref["activity_code"].map(_s), ref["service_family"].map(_norm)))
            if "code_family" in ref.columns:
                cfam = dict(zip(ref["activity_code"].map(_s), ref["code_family"].map(_norm)))
        billed = lines[lines["net_amount"].fillna(0) > 0].assign(_code=lambda f: f["activity_code"].map(_s))
        billed = billed.assign(_fam=billed["_code"].map(fam).fillna(""),
                               _case=billed["_code"].map(cfam).fillna("") == "case_rate")
        case_fam = billed[billed["_case"]].groupby("claim_sk")["_fam"].first()
        in_case = billed["claim_sk"].isin(case_fam.index)
        billed.loc[in_case, "_fam"] = billed.loc[in_case, "claim_sk"].map(case_fam)
        billed = billed.assign(_prod=billed["claim_sk"].map(dict(zip(df["claim_sk"], df["member_sk"]))).map(product_of))
        billed["_charges"] = [(p, f) in charging for p, f in zip(billed["_prod"], billed["_fam"])]
        expects = billed.groupby("claim_sk")["_charges"].any()
        df = df[df["claim_sk"].map(expects).fillna(False).astype(bool)]
        if df.empty:
            return []
    df["_zero"] = df["_share"] <= 0

    positive = lines["net_amount"][lines["net_amount"] > 0]
    if positive.empty:
        return []
    cut = float(positive.quantile(q))
    high_claims = set(lines.loc[lines["net_amount"] >= cut, "claim_sk"])
    df["_high"] = df["claim_sk"].isin(high_claims)

    per_member = df.groupby(["provider_sk", "member_sk"]).agg(
        high=("_high", "sum"), zero=("_zero", "sum"), n=("claim_sk", "size")).reset_index()
    per_member["repeat_high"] = per_member["high"] >= 2
    pooled_zero = float(df["_zero"].mean())
    pooled_repeat = float(per_member["repeat_high"].mean())

    rx = _table(ctx, "prescription_dispense")
    subst = {}
    if not rx.empty and {"billed_product", "dispensed_product", "claim_sk"} <= set(rx.columns):
        diff = rx[(rx["billed_product"].map(_s) != "") & (rx["dispensed_product"].map(_s) != "")
                  & (rx["billed_product"].map(_s) != rx["dispensed_product"].map(_s))]
        cp = dict(zip(df["claim_sk"], df["provider_sk"]))
        subst = diff["claim_sk"].astype(str).map(cp).value_counts().to_dict()

    out = []
    for p, sub in df.groupby("provider_sk", sort=True):
        if len(sub) < min_claims or not p:
            continue
        zero_rate = float(sub["_zero"].mean())
        pm = per_member[per_member["provider_sk"] == p]
        repeat_rate = float(pm["repeat_high"].mean())
        if zero_rate < multiple * pooled_zero or repeat_rate < multiple * pooled_repeat:
            continue
        both = pm[(pm["zero"] > 0) & pm["repeat_high"]]
        if len(both) < min_both:
            continue
        hit_claims = sub[sub["member_sk"].isin(set(both["member_sk"])) & (sub["_zero"] | sub["_high"])]
        last = sub["service_date"].max() if "service_date" in sub.columns else None
        out.append(_sig(
            ctx, control, subject_type="provider", subject_id=p,
            fact_key=f"inducement:{p}",
            claim_ids=sorted(hit_claims["claim_sk"]),
            event_time=last,
            period=period_bucket(last),
            confidence=0.55,
            evidence={
                "provider_sk": p,
                "community_id": f"patients-of:{p}",
                "claims_considered": int(len(sub)),
                "zero_patient_share_rate": round(zero_rate, 3),
                "all_providers_zero_patient_share_rate": round(pooled_zero, 3),
                "patients_with_repeated_high_value_share": round(repeat_rate, 3),
                "all_providers_repeated_high_value_share": round(pooled_repeat, 3),
                "patients_with_both": int(len(both)),
                "high_value_line_threshold_aed": round(cut, 2),
                "substitution_limb": ({"claims_with_billed_product_not_dispensed": int(subst.get(p, 0))}
                                      if subst or not rx.empty else "not evaluated — no dispensing records"),
                "approved_programme_exclusion": ("products whose benefit rules charge no patient share are excluded"
                                                 if eligible_products is not None else
                                                 "not applied — no benefit rules or cover records"),
                "plain_language": (
                    f"{pct(zero_rate)} of this provider's {len(sub)} claims charged the patient nothing "
                    f"(all providers: {pct(pooled_zero)}), and {pct(repeat_rate)} of its patients had "
                    f"repeated high-value services (all providers: {pct(pooled_repeat)}); "
                    f"{count_phrase(len(both), 'patient')} show both."
                ),
                "what_the_reviewer_must_verify": (
                    "Check whether an approved programme waives patient payments, and contact a "
                    "sample of the patients to confirm what they received."
                ),
            },
            exposure=_at_stake(_amounts(claims, hit_claims["claim_sk"]), "zero-share and high-value claims of the patients showing both"),
        ))
    return out


# ===========================================================================
# NET-03-R04 — complaints and confirmations corroborating a mismatch
# ===========================================================================


#: Words that make a complaint an allegation of a service, item or charge mismatch.
#: A billing QUERY that was explained and resolved is not one.
_MISMATCH_TYPES = ("not_received", "not_provided", "never_received", "never_had", "did_not_receive",
                   "overcharg", "charged_for", "mismatch", "phantom", "not_rendered", "wrong_item",
                   "service_not", "item_not")
_RESOLVED = ("resolved", "explained", "withdrawn")
_UNRELIABLE_CHANNELS = ("anonymous", "unverified", "unknown", "unauthenticated")


def net_03_r04_complaint_corroboration(ctx, control) -> list:
    """Several patients of one provider independently say they did not get what was billed.

    Counts only evidence linked to a claim the patient actually has at that
    provider, from an authenticated channel (declared exclusion: evidence
    reliability): a member confirmation that the service was NOT received, or a
    complaint about a service, item or charge that did not match. A provider is
    flagged when enough DISTINCT patients corroborate each other.
    """
    claims = _claims(ctx)
    conf = _table(ctx, "member_confirmation")
    comp = _table(ctx, "complaint")
    if claims.empty or (conf.empty and comp.empty):
        return []
    min_members = int(ctx.cfg("net03r04_min_distinct_members"))
    owner = dict(zip(claims["claim_sk"], zip(claims["member_sk"], claims["provider_sk"])))
    items: list[dict[str, Any]] = []
    if not conf.empty and {"claim_sk", "service_confirmed"} <= set(conf.columns):
        for r in conf.itertuples(index=False):
            c = _s(r.claim_sk)
            confirmed = r.service_confirmed
            if isinstance(confirmed, str):
                confirmed = confirmed.strip().lower() in ("true", "yes", "1", "y")
            if c not in owner or is_missing(confirmed) or bool(confirmed):
                continue
            channel = _norm(getattr(r, "channel", ""))
            if any(k in channel for k in _UNRELIABLE_CHANNELS):
                continue
            m, p = owner[c]
            if _s(getattr(r, "member_sk", "")) and _s(r.member_sk) != m:
                continue  # the respondent is not the claim's patient
            items.append({"kind": "confirmation", "claim_sk": c, "member_sk": m, "provider_sk": p,
                          "date": _s(getattr(r, "response_date", "")), "channel": channel,
                          "text": _s(getattr(r, "response", ""))})
    if not comp.empty and "member_sk" in comp.columns:
        for r in comp.itertuples(index=False):
            c = _s(getattr(r, "claim_sk", ""))
            kind = _norm(getattr(r, "complaint_type", "")) + " " + _norm(getattr(r, "text", ""))
            if not any(k in kind for k in _MISMATCH_TYPES) or any(k in kind for k in _RESOLVED):
                continue
            m = _s(r.member_sk)
            p = _s(getattr(r, "provider_sk", ""))
            if c:
                if c not in owner or owner[c][0] != m:
                    continue
                p = p or owner[c][1]
            elif not ((claims["member_sk"] == m) & (claims["provider_sk"] == p)).any():
                continue  # no claim links this patient to this provider
            items.append({"kind": "complaint", "claim_sk": c, "member_sk": m, "provider_sk": p,
                          "date": _s(getattr(r, "complaint_date", "")),
                          "type": _s(getattr(r, "complaint_type", "")), "text": _s(getattr(r, "text", ""))})
    if not items:
        return []
    ev = pd.DataFrame(items)
    out = []
    for p, sub in ev.groupby("provider_sk", sort=True):
        n_mem = sub["member_sk"].nunique()
        if not p or n_mem < min_members:
            continue
        cids = sorted({c for c in sub["claim_sk"] if c})
        amounts = _amounts(claims, cids)
        dates = pd.to_datetime(sub["date"], errors="coerce")
        last = dates.max()
        out.append(_sig(
            ctx, control, subject_type="provider", subject_id=p,
            fact_key=f"corroboration:{p}",
            claim_ids=cids,
            event_time=last,
            period=period_bucket(last),
            confidence=0.7,
            evidence={
                "provider_sk": p,
                "community_id": f"patients-of:{p}",
                "distinct_patients": int(n_mem),
                "confirmations_not_received": int((sub["kind"] == "confirmation").sum()),
                "complaints": int((sub["kind"] == "complaint").sum()),
                "items": sub.sort_values(["member_sk", "claim_sk"]).head(10).to_dict("records"),
                "billed_on_those_claims_aed": round(sum(amounts.values()), 2),
                "reliability_rule": "linked to the patient's own claim at this provider, authenticated channel",
                "plain_language": (
                    f"{count_phrase(n_mem, 'patient')} of this provider independently reported not "
                    f"receiving what was billed ({count_phrase(int((sub['kind'] == 'confirmation').sum()), 'verification reply')} "
                    f"and {count_phrase(int((sub['kind'] == 'complaint').sum()), 'complaint')}) on "
                    f"claims worth {aed(sum(amounts.values()))}."
                ),
                "what_the_reviewer_must_verify": (
                    "Check how reliable each patient's account is and compare it with the claim "
                    "before escalating."
                ),
            },
            exposure=_at_stake(amounts, "claims the patients say they did not receive as billed"),
        ))
    return out


# ===========================================================================
# NET-04-R04 — shared identifiers across agents, employers, patients and providers
# ===========================================================================


#: (table, id column, actor type, identifier columns) read when present.
_ACTOR_SOURCES = (
    ("provider", "provider_sk", "provider",
     ("owner_entity_id", "bank_account_token", "phone_token", "address_token")),
    ("member", "member_sk", "member",
     ("protected_id_token", "bank_account_token", "phone_token", "address_token")),
    ("employer_roster", "employer_id", "employer", ("employer_id",)),
    ("member", "employer_id", "employer", ("employer_id",)),
    ("coverage_period", "agent_id", "agent", ("agent_id",)),
    ("policy_application", "agent_id", "agent", ("agent_id",)),
)
_ROLE_WORDS = {
    ("provider", "owner_entity_id"): "the owner recorded for a provider",
    ("provider", "bank_account_token"): "a provider's bank account",
    ("provider", "phone_token"): "a provider's phone number",
    ("provider", "address_token"): "a provider's address",
    ("member", "protected_id_token"): "a patient's personal identifier",
    ("member", "bank_account_token"): "a patient's bank account",
    ("member", "phone_token"): "a patient's phone number",
    ("member", "address_token"): "a patient's address",
    ("employer", "employer_id"): "an employer's identifier",
    ("agent", "agent_id"): "an agent's identifier",
}
_ID_WORDS = {"owner_entity_id": "owner identifier", "bank_account_token": "bank account",
             "phone_token": "phone number", "address_token": "address", "protected_id_token": "personal identifier",
             "sponsor_id": "sponsor identifier", "employer_id": "employer identifier", "agent_id": "agent identifier"}


def net_04_r04_shared_identifiers_actors(ctx, control) -> list:
    """One identifier value turns up on actors of DIFFERENT kinds that should be independent.

    For example, the owner recorded for a clinic is the agent who sold the
    policies of the patients it treats, or the sponsor of those patients. Exact
    matches only. Declared exclusion (known corporate links): a provider recorded
    as an on-site or occupational clinic sharing an identifier with an employer.
    """
    rows = []
    for table, key, actor, cols in _ACTOR_SOURCES:
        frame = _table(ctx, table)
        if frame.empty or key not in frame.columns:
            continue
        for col in cols:
            if col not in frame.columns:
                continue
            sub = frame[[key, col]].copy()
            sub.columns = ["actor_id", "value"]
            sub["actor_id"] = sub["actor_id"].map(_s)
            sub["value"] = sub["value"].map(_s)
            sub = sub[(sub["actor_id"] != "") & (sub["value"] != "")].drop_duplicates()
            sub["actor_type"] = actor
            sub["identifier"] = col
            rows.append(sub)
    if not rows:
        return []
    ids = pd.concat(rows, ignore_index=True).drop_duplicates(["actor_type", "actor_id", "value"])
    kinds = ids.groupby("value")["actor_type"].nunique()
    shared = ids[ids["value"].isin(kinds[kinds >= 2].index)]
    if shared.empty:
        return []
    prov = _providers(ctx)
    claims = _claims(ctx)
    max_actors = int(ctx.cfg("net02r01_shared_service_max_providers"))
    out = []
    for value, grp in shared.groupby("value", sort=True):
        # An identifier on a provider's side that equals the provider's own key
        # is the same actor, not a link.
        actors = grp.drop_duplicates(["actor_type", "actor_id"])
        if len(actors) > max_actors * 2:
            continue  # shared by too many parties to be a covert link (a registry code, a mall address)
        provs = sorted(set(actors.loc[actors["actor_type"] == "provider", "actor_id"]))
        if provs and not prov.empty and set(actors["actor_type"]) <= {"provider", "employer"}:
            kinds_p = " ".join(_norm(prov.at[p, c]) for p in provs if p in prov.index
                               for c in ("provider_type", "facility_type", "specialty") if c in prov.columns)
            if any(k in kinds_p for k in ("onsite", "on_site", "occupational")):
                continue  # declared exclusion: a known corporate link
        agents = sorted(set(actors.loc[actors["actor_type"] == "agent", "actor_id"]))
        members = sorted(set(actors.loc[actors["actor_type"] == "member", "actor_id"]))
        employers = sorted(set(actors.loc[actors["actor_type"] == "employer", "actor_id"]))
        # The business the link touches: claims by these providers for patients
        # sold by these agents / sponsored under this identifier.
        related = claims
        if provs and not claims.empty:
            related = claims[claims["provider_sk"].isin(provs)]
            if agents and "agent_id" in related.columns:
                related = related[related["agent_id"].map(_s).isin(agents)]
            elif members:
                related = related[related["member_sk"].isin(members)]
            elif employers:
                mem = _table(ctx, "member")
                if not mem.empty and "employer_id" in mem.columns:
                    emp_members = set(mem.loc[mem["employer_id"].map(_s).isin(employers), "member_sk"].map(_s))
                    related = related[related["member_sk"].isin(emp_members)]
        else:
            related = claims.iloc[0:0] if not claims.empty else claims
        amounts = _amounts(claims, related["claim_sk"]) if not related.empty else {}
        roles = sorted({_ROLE_WORDS.get((a, i), f"a {a}'s {_ID_WORDS.get(i, i)}")
                        for a, i in zip(grp["actor_type"], grp["identifier"])})
        subject_type, subject_id = ("provider", provs[0]) if provs else (
            ("agent", agents[0]) if agents else (actors.iloc[0]["actor_type"], actors.iloc[0]["actor_id"]))
        last = related["service_date"].max() if not related.empty and "service_date" in related.columns else None
        parts = []
        if provs:
            parts.append(count_phrase(len(provs), "provider"))
        if agents:
            parts.append(count_phrase(len(agents), "agent"))
        if employers:
            parts.append(count_phrase(len(employers), "employer"))
        if members:
            parts.append(count_phrase(len(members), "patient"))
        out.append(_sig(
            ctx, control, subject_type=subject_type, subject_id=subject_id,
            fact_key=f"sharedactor:{value}",
            claim_ids=sorted(amounts),
            event_time=last,
            period=period_bucket(last) if last is not None else "",
            confidence=0.7,
            evidence={
                "provider_sk": provs[0] if provs else "",
                "linked_provider_sks": provs,
                "agent_id": agents[0] if agents else "",
                "linked_agent_ids": agents,
                "employer_ids": employers,
                "member_sks": members[:20],
                "community_id": f"identifier:{value}",
                "shared_value": value,
                "roles_sharing_it": roles,
                "match_type": "exact value match across actor types",
                "claims_touched": len(amounts),
                "value_of_claims_touched_aed": round(sum(amounts.values()), 2),
                "plain_language": (
                    f"The same identifier is both {' and '.join(roles[:4])}, linking "
                    f"{' and '.join(parts)} that should be independent; the providers billed "
                    f"{count_phrase(len(amounts), 'claim')} ({aed(sum(amounts.values()))}) for the "
                    f"linked parties' patients."
                ),
                "what_the_reviewer_must_verify": (
                    "Rule out a lawful or known corporate link, then map every party sharing the "
                    "identifier for the network review."
                ),
            },
            exposure=_at_stake(amounts, "claims touched by a cross-party identifier link"),
        ))
    return out


IMPLEMENTATIONS: dict[str, Callable] = {
    fn.__name__: fn
    for fn in (
        net_01_r02_reciprocal_referral_loop,
        net_01_r03_high_cost_conversion,
        net_01_r04_closed_downstream_chain,
        net_02_r01_shared_admin_identity,
        net_02_r04_structural_change,
        net_03_r02_closed_member_group,
        net_03_r03_inducement_signature,
        net_03_r04_complaint_corroboration,
        net_04_r04_shared_identifiers_actors,
    )
}
