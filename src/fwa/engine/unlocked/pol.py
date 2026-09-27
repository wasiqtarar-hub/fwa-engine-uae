"""Unlocked implementations for the POL family.

Each function here runs only when a dataset unlock (`rules/unlocks/POL.yaml`)
is satisfied by the loaded file. See :mod:`fwa.engine.unlocks`. The data
helpers are shared with :mod:`.ent`.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
import pandas as pd

from ...cases import exposure as _exp
from ...presentation import aed, count_phrase, pct, plain_date
from ..controls import _sig
from ..evallib import is_missing
from .ent import (_claims, _col, _coverage, _d, _dstr, _has_word, _ids, _members, _num, _period, _providers, _s,
                  _safe, _table, _falsy, _up)

IMPLEMENTATIONS: dict[str, Callable] = {}

_ADD_WORDS = ("ADD", "ENROL", "INCLU", "REINSTAT", "CHANGE", "NEW")
_APPROVED_RETRO_WORDS = ("APPROV", "AUTHORISED", "AUTHORIZED")
_CAMPAIGN_WORDS = ("CAMPAIGN", "OPEN_ENROL", "OPEN ENROL")
_CONTINUATION_WORDS = ("CONTINU", "COBRA", "RUN-OFF", "RUNOFF")
_MANDATED_WORDS = ("MANDAT",)
_NONE_DECLARED = {"", "NONE", "NIL", "NO", "N", "NA", "N/A", "FALSE", "0", "[]"}


def _register(name: str):
    def deco(fn: Callable) -> Callable:
        wrapped = _safe(fn)
        IMPLEMENTATIONS[name] = wrapped
        return wrapped

    return deco


def _codes(value: Any) -> list[str]:
    if is_missing(value):
        return []
    text = str(value).strip().upper()
    if text in _NONE_DECLARED:
        return []
    for ch in "[]\"'":
        text = text.replace(ch, "")
    parts = [p.strip() for p in text.replace(",", ";").replace("|", ";").split(";")]
    return [p for p in parts if p and p not in _NONE_DECLARED]


def _events(ctx) -> pd.DataFrame:
    ev = _table(ctx, "policy_event")
    if ev.empty or "coverage_id" not in ev.columns:
        return pd.DataFrame(columns=["policy_event_sk", "coverage_id", "event_type", "actor", "event_time",
                                     "effective_date", "agent_id", "member_sk"])
    out = pd.DataFrame({
        "policy_event_sk": _s(_col(ev, "policy_event_sk")), "coverage_id": _s(ev["coverage_id"]),
        "event_type": _up(_col(ev, "event_type", "")), "actor": _col(ev, "actor"),
        "event_time": pd.to_datetime(_col(ev, "event_time"), errors="coerce"),
        "effective_date": _d(_col(ev, "effective_date")), "agent_id": _s(_col(ev, "agent_id")),
    })
    try:
        if getattr(out["event_time"].dt, "tz", None) is not None:
            out["event_time"] = out["event_time"].dt.tz_localize(None)
    except Exception:  # noqa: BLE001
        pass
    cov = _coverage(ctx)
    return out.merge(cov[["coverage_id", "member_sk", "vf", "status", "product", "agent_id"]].rename(
        columns={"agent_id": "cov_agent_id"}), on="coverage_id", how="left")


# ===========================================================================
# POL-01-R01 — eligibility / roster conflict
# ===========================================================================


@_register("pol_01_r01_roster_conflict")
def pol_01_r01_roster_conflict(ctx, control) -> list:
    """POL-01-R01 — employer-sponsored member not on the employer's roster when served, or a duplicate
    active identity (one Emirates-ID token with two overlapping active covers)."""
    mem = _members(ctx)
    roster = _table(ctx, "employer_roster")
    claims = _claims(ctx)
    cov = _coverage(ctx)
    if mem.empty or claims.empty or "employer_id" not in mem.columns:
        return []
    newborn_days = int(ctx.cfg("ent_newborn_days"))
    out = []
    # ---- limb 1: not on the roster -----------------------------------------
    if not roster.empty and "member_sk" in roster.columns:
        r = pd.DataFrame({"employer_id": _s(_col(roster, "employer_id")), "member_sk": _s(roster["member_sk"]),
                          "sponsor_id": _s(_col(roster, "sponsor_id")), "rvf": _d(_col(roster, "valid_from")),
                          "rvt": _d(_col(roster, "valid_to"))})
        m = pd.DataFrame({"member_sk": mem["member_sk"], "employer_id": _s(mem["employer_id"]),
                          "sponsor_id": _s(_col(mem, "sponsor_id")), "dob": _d(_col(mem, "date_of_birth"))})
        m = m.dropna(subset=["employer_id"])
        m = m[m["employer_id"].isin(set(r["employer_id"].dropna()))]  # only employers who supplied a roster
        c = claims[claims["member_sk"].isin(set(m["member_sk"]))][["claim_sk", "member_sk", "service_date", "gross"]]
        c = c.merge(m, on="member_sk", how="inner")
        # direct listing, or listed through the sponsor (dependant) on the same roster
        direct = c.merge(r[["employer_id", "member_sk", "rvf", "rvt"]], on=["employer_id", "member_sk"], how="left")
        dep = c.dropna(subset=["sponsor_id"])
        dep = dep[dep["sponsor_id"] != dep["member_sk"]]  # a principal is not their own sponsor route
        via = dep.merge(
            r[["employer_id", "member_sk", "rvf", "rvt"]].rename(columns={"member_sk": "sponsor_id"}),
            on=["employer_id", "sponsor_id"], how="left")
        via2 = dep.merge(
            r[["employer_id", "sponsor_id", "rvf", "rvt"]].dropna(subset=["sponsor_id"]),
            on=["employer_id", "sponsor_id"], how="left")
        allj = pd.concat([direct, via, via2], ignore_index=True)
        ok = allj["rvf"].notna() & (allj["service_date"] >= allj["rvf"]) & \
            ((allj["service_date"] <= allj["rvt"]) | allj["rvt"].isna())
        okc = allj.assign(ok=ok).groupby("claim_sk")["ok"].any()
        bad = c[~c["claim_sk"].map(okc).fillna(False).astype(bool)]
        # exclusions: continuation cover, newborns, mandated cover
        if not bad.empty:
            live = cov[~cov["void"]]
            j = bad.merge(live[["member_sk", "vf", "vt", "status", "product"]], on="member_sk", how="left")
            inside = (j["service_date"] >= j["vf"]) & ((j["service_date"] <= j["vt"]) | j["vt"].isna())
            j = j[inside]
            excl = _has_word(j["status"].fillna(""), _CONTINUATION_WORDS + _MANDATED_WORDS) | \
                _has_word(j["product"].fillna(""), _MANDATED_WORDS)
            bad = bad[~bad["claim_sk"].isin(set(j.loc[excl, "claim_sk"]))]
            newborn = (bad["service_date"] - bad["dob"]).dt.days.between(0, newborn_days)
            bad = bad[~newborn.fillna(False).astype(bool)]
        listed_ever = set(r["member_sk"].dropna()) | set(r["sponsor_id"].dropna())
        for member, g in bad.groupby("member_sk", sort=False):
            g = g.sort_values("service_date")
            emp = g["employer_id"].iloc[0]
            spons = g["sponsor_id"].iloc[0]
            ever = member in listed_ever or (not is_missing(spons) and spons in listed_ever)
            rr = r[(r["employer_id"] == emp) & ((r["member_sk"] == member) | (r["member_sk"] == spons)
                                                | (r["sponsor_id"] == spons))]
            window = (f"; the roster lists them only from {plain_date(rr['rvf'].min())} to {plain_date(rr['rvt'].max())}"
                      if ever and not rr.empty else "")
            first = g["service_date"].min()
            out.append(_sig(
                ctx, control, subject_type="member", subject_id=member,
                fact_key=f"roster:{member}", claim_ids=g["claim_sk"].tolist(),
                event_time=first, period=_period(first),
                evidence={
                    "member_sk": member, "employer_id": emp, "sponsor_id": None if is_missing(spons) else spons,
                    "limb_fired": "not_on_employer_roster" if not ever else "outside_roster_dates",
                    "claims_outside_roster": int(len(g)), "gross_outside_roster_aed": round(float(g["gross"].sum()), 2),
                    "first_service_date": _dstr(first), "last_service_date": _dstr(g["service_date"].max()),
                    "roster_rows": [[_dstr(a), _dstr(b)] for a, b in zip(rr["rvf"], rr["rvt"])][:10],
                    "reason_code": control.reason_code, "subject_id": member,
                    "plain_language": (f"This member is enrolled through employer {emp}, but "
                                       + ("the employer's roster does not list them or their sponsor"
                                          if not ever else "was not on that employer's roster on the service dates")
                                       + f"{window}; {count_phrase(len(g), 'claim')} ({aed(g['gross'].sum())}) were "
                                         f"billed from {plain_date(first)}."),
                    "what_the_reviewer_must_verify": "Ask the employer to confirm the relationship and check "
                                                     "continuation, newborn or mandated cover.",
                },
                exposure=_exp.no_exposure("entitlement must be confirmed with the employer before any amount is "
                                          "established"),
            ))
    # ---- limb 2: duplicate active identity -----------------------------------
    if "protected_id_token" in mem.columns and not cov.empty:
        tok = pd.DataFrame({"member_sk": mem["member_sk"], "token": _s(mem["protected_id_token"]),
                            "dob": _d(_col(mem, "date_of_birth")), "sex": _up(_col(mem, "sex", ""))}).dropna(subset=["token"])
        sizes = tok.groupby("token")["member_sk"].transform("nunique")
        placeholder_max = int(ctx.cfg("ent02_placeholder_token_max_members"))
        tok = tok[(sizes >= 2) & (sizes <= placeholder_max)]
        live = cov[~cov["void"]]
        for token, g in tok.groupby("token", sort=False):
            # same person (DOB and sex agree) — the DOB/sex conflict case belongs to ENT-02-R01
            if g["dob"].nunique() > 1 or g["sex"].nunique() > 1:
                continue
            cv = live[live["member_sk"].isin(set(g["member_sk"]))].sort_values("vf")
            pairs = cv.merge(cv, how="cross", suffixes=("_a", "_b"))
            pairs = pairs[(pairs["member_sk_a"] < pairs["member_sk_b"])]
            end_a = pairs["vt_a"].fillna(pd.Timestamp.max)
            end_b = pairs["vt_b"].fillna(pd.Timestamp.max)
            ovl = pairs[(pairs["vf_a"] <= end_b) & (pairs["vf_b"] <= end_a)]
            if ovl.empty:
                continue
            p0 = ovl.iloc[0]
            members = sorted({p0["member_sk_a"], p0["member_sk_b"]})
            later = p0["member_sk_b"] if p0["vf_b"] >= p0["vf_a"] else p0["member_sk_a"]
            start = max(p0["vf_a"], p0["vf_b"])
            end = min(end_a.loc[ovl.index[0]], end_b.loc[ovl.index[0]])
            cl = claims[claims["member_sk"].isin(members) & (claims["service_date"] >= start) & (claims["service_date"] <= end)]
            out.append(_sig(
                ctx, control, subject_type="member", subject_id=later,
                fact_key=f"dup_identity:{token}", claim_ids=cl["claim_sk"].tolist()[:100],
                event_time=start, period=_period(start),
                evidence={
                    "member_sk": later, "linked_member_sks": members, "protected_id_token": token,
                    "limb_fired": "duplicate_active_identity",
                    "overlap_from": _dstr(start), "overlap_to": _dstr(end if end != pd.Timestamp.max else None),
                    "coverage_ids": [p0["coverage_id_a"], p0["coverage_id_b"]], "claims_during_overlap": int(len(cl)),
                    "reason_code": control.reason_code, "subject_id": later,
                    "plain_language": (f"The same Emirates-ID reference, date of birth and sex are on two member records "
                                       f"with active cover at the same time from {plain_date(start)}; "
                                       f"{count_phrase(len(cl), 'claim')} were billed during the overlap."),
                    "what_the_reviewer_must_verify": "Confirm whether this is one person enrolled twice and which "
                                                     "enrolment should stand.",
                },
                exposure=_exp.no_exposure("which enrolment is valid is not yet established"),
            ))
    return out


# ===========================================================================
# POL-01-R02 — retroactive add after a costly service
# ===========================================================================


@_register("pol_01_r02_retroactive_event_after_service")
def pol_01_r02_retroactive_event_after_service(ctx, control) -> list:
    """POL-01-R02 — a member add/change recorded after a high-cost service it backdates cover over."""
    ev = _events(ctx)
    claims = _claims(ctx)
    if ev.empty or claims.empty:
        return []
    sla = int(ctx.cfg("pol01_retro_sla_days"))
    q = float(ctx.cfg("pol01_high_cost_quantile"))
    adds = ev[_has_word(ev["event_type"], _ADD_WORDS) & ev["event_time"].notna() & ev["effective_date"].notna()
              & ev["member_sk"].notna()].copy()
    adds = adds[~_has_word(adds["event_type"], _APPROVED_RETRO_WORDS)]
    adds["lag_days"] = (adds["event_time"].dt.normalize() - adds["effective_date"]).dt.days
    adds = adds[adds["lag_days"] > sla]
    if adds.empty:
        return []
    cut = float(np.quantile(claims["gross"].dropna(), q)) if claims["gross"].notna().any() else np.inf
    j = adds.merge(claims[["claim_sk", "member_sk", "service_date", "gross", "provider_sk"]], on="member_sk")
    j = j[(j["service_date"] >= j["effective_date"]) & (j["service_date"] < j["event_time"].dt.normalize())
          & (j["gross"] >= cut)]
    out = []
    for ev_id, g in j.groupby("policy_event_sk", sort=False, dropna=False):
        e0 = g.iloc[0]
        g = g.sort_values("service_date")
        total = float(g["gross"].sum())
        actor = None if is_missing(e0["actor"]) else str(e0["actor"])
        out.append(_sig(
            ctx, control, subject_type="member", subject_id=e0["member_sk"],
            fact_key=f"retro:{e0['member_sk']}:{_dstr(e0['effective_date'])}", claim_ids=g["claim_sk"].tolist(),
            event_time=e0["event_time"], period=_period(e0["event_time"]),
            evidence={
                "member_sk": e0["member_sk"], "policy_event_sk": None if is_missing(ev_id) else ev_id,
                "coverage_id": e0["coverage_id"], "event_type": e0["event_type"], "actor": actor,
                "agent_id": None if is_missing(e0["agent_id"]) else e0["agent_id"],
                "event_recorded": str(e0["event_time"]), "effective_date": _dstr(e0["effective_date"]),
                "backdated_by_days": int(e0["lag_days"]), "sla_days": sla,
                "high_cost_cut_aed": round(cut, 2), "high_cost_claims": int(len(g)), "high_cost_total_aed": round(total, 2),
                "reason_code": control.reason_code, "subject_id": e0["member_sk"],
                "plain_language": (f"This member was added on {plain_date(e0['event_time'])} with cover backdated to "
                                   f"{plain_date(e0['effective_date'])} ({int(e0['lag_days'])} days, the allowed window is "
                                   f"{sla}); in between, {count_phrase(len(g), 'high-cost claim')} totalling {aed(total)} "
                                   f"had already been incurred, the first on {plain_date(g['service_date'].iloc[0])}."),
                "what_the_reviewer_must_verify": "Confirm who made the change, whether they had authority, and whether "
                                                 "an approved backdated correction explains it.",
            },
            exposure=_exp.no_exposure("the claims are payable if the backdated enrolment is confirmed as legitimate"),
        ))
    return out


# ===========================================================================
# POL-01-R03 — application declaration vs prior history
# ===========================================================================


def _covered_by(code: str, declared: list[str]) -> bool:
    code = code.upper()
    return any(code.startswith(d) or d.startswith(code) for d in declared)


@_register("pol_01_r03_application_history_conflict")
def pol_01_r03_application_history_conflict(ctx, control) -> list:
    """POL-01-R03 — a declaration on the application contradicted by prior cover or conditions on record."""
    app = _table(ctx, "policy_application")
    prior = _table(ctx, "prior_coverage_history")
    if app.empty or prior.empty or "member_sk" not in app.columns or "member_sk" not in prior.columns:
        return []
    contest = int(ctx.cfg("pol01_contestability_days"))
    lookback = int(ctx.cfg("pol01_prior_history_lookback_days"))
    claims = _claims(ctx)
    as_of = claims["service_date"].max() if not claims.empty else pd.Timestamp(ctx.run_date)
    a = pd.DataFrame({"application_sk": _s(_col(app, "application_sk")), "member_sk": _s(app["member_sk"]),
                      "coverage_id": _s(_col(app, "coverage_id")), "application_date": _d(_col(app, "application_date")),
                      "declared": _col(app, "declared_conditions"), "declared_prior_cover": _col(app, "declared_prior_cover"),
                      "agent_id": _s(_col(app, "agent_id"))}).dropna(subset=["member_sk", "application_date"])
    # legal usability: only inside the contestability window
    a = a[(as_of - a["application_date"]).dt.days <= contest]
    p = pd.DataFrame({"member_sk": _s(prior["member_sk"]), "prior_payer": _col(prior, "prior_payer"),
                      "pvf": _d(_col(prior, "valid_from")), "pvt": _d(_col(prior, "valid_to")),
                      "conditions": _col(prior, "conditions_treated")})
    j = a.merge(p, on="member_sk", how="inner")
    j = j[(j["pvf"] < j["application_date"]) &
          ((j["pvt"].fillna(j["application_date"])) >= j["application_date"] - pd.Timedelta(days=lookback))]
    if j.empty:
        return []
    dx = pd.Series(dtype=object)
    raw = ctx.claims if ctx.claims is not None else pd.DataFrame()
    if "diagnosis_primary" in raw.columns:
        dx = raw.set_index(raw["claim_sk"].astype(str))["diagnosis_primary"]
    # screen row by row on plain Python lists (fast), then build evidence only for conflicts
    j = j.assign(no_prior=_falsy(j["declared_prior_cover"]).values)
    decl_l = [_codes(v) for v in j["declared"].tolist()]
    cond_l = [_codes(v) for v in j["conditions"].tolist()]
    j = j.assign(conflict=[bool(np_) or any(not _covered_by(c, d) for c in cs)
                           for np_, d, cs in zip(j["no_prior"].tolist(), decl_l, cond_l)])
    j = j[j.groupby("application_sk", dropna=False)["conflict"].transform("any")]
    out = []
    for app_sk, g in j.groupby("application_sk", sort=False, dropna=False):
        g0 = g.iloc[0]
        declared = _codes(g0["declared"])
        history = sorted({c for v in g["conditions"] for c in _codes(v)})
        undisclosed = [c for c in history if not _covered_by(c, declared)]
        said_no_prior = bool(g0["no_prior"])
        limbs = []
        if undisclosed:
            limbs.append("undisclosed_conditions")
        if said_no_prior:
            limbs.append("prior_cover_not_declared")
        if not limbs:
            continue
        member = g0["member_sk"]
        mc = claims[(claims["member_sk"] == member) & (claims["service_date"] >= g0["application_date"])]
        related = [c for c in mc["claim_sk"] if any(str(dx.get(c, "")).upper().startswith(u[:3]) for u in undisclosed)]
        parts = []
        if said_no_prior:
            payers = sorted({str(x) for x in g["prior_payer"] if not is_missing(x)})
            parts.append(f"no earlier cover was declared, but cover with {', '.join(payers) or 'another insurer'} is on "
                         f"record from {plain_date(g['pvf'].min())}")
        if undisclosed:
            parts.append(f"{'no conditions were' if not declared else 'only ' + ', '.join(declared) + ' was'} declared, "
                         f"but earlier cover records treatment for {', '.join(undisclosed)}")
        tail = f"; {count_phrase(len(related), 'later claim')} relate to those conditions" if related else ""
        out.append(_sig(
            ctx, control, subject_type="member", subject_id=member,
            fact_key=f"application:{member}:{_dstr(g0['application_date'])}", claim_ids=related[:100],
            event_time=g0["application_date"], period=_period(g0["application_date"]), confidence=0.7,
            evidence={
                "member_sk": member, "application_sk": None if is_missing(app_sk) else app_sk,
                "application_date": _dstr(g0["application_date"]), "declared_conditions": declared,
                "declared_prior_cover": not said_no_prior,
                "prior_cover": [{"payer": None if is_missing(x.prior_payer) else str(x.prior_payer),
                                 "from": _dstr(x.pvf), "to": _dstr(x.pvt)} for x in g.itertuples(index=False)],
                "conditions_on_record": history, "undisclosed_conditions": undisclosed, "limb_fired": limbs,
                "agent_id": None if is_missing(g0["agent_id"]) else g0["agent_id"],
                "contestability_days": contest, "related_claims": related[:20],
                "legal_use_note": "Used only inside the contestability window; any decision on cover terms is an "
                                  "underwriter's, subject to privacy and non-discrimination law.",
                "reason_code": control.reason_code, "subject_id": member,
                "plain_language": (f"On the application dated {plain_date(g0['application_date'])} " + "; and ".join(parts)
                                   + tail + "."),
                "what_the_reviewer_must_verify": "Check with compliance that the history may be used, then refer to an "
                                                 "underwriter for human review.",
            },
            exposure=_exp.no_exposure("an application conflict affects cover terms, not a claim amount"),
        ))
    return out


# ===========================================================================
# POL-01-R05 — employer / broker enrolment cluster
# ===========================================================================


def _linked_provider_groups(prov: pd.DataFrame) -> dict[str, str]:
    """provider_sk -> group id; providers sharing an owner, bank account, phone or address share a group."""
    parent: dict[str, str] = {p: p for p in prov["provider_sk"]}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for col in ("owner_entity_id", "bank_account_token", "phone_token", "address_token"):
        if col not in prov.columns:
            continue
        for _, g in prov.dropna(subset=[col]).groupby(col):
            ids = list(g["provider_sk"])
            for other in ids[1:]:
                ra, rb = find(ids[0]), find(other)
                if ra != rb:
                    parent[rb] = ra
    return {p: find(p) for p in parent}


@_register("pol_01_r05_enrolment_cluster")
def pol_01_r05_enrolment_cluster(ctx, control) -> list:
    """POL-01-R05 — an employer's or broker's burst of new members who claim early at linked providers."""
    ev = _events(ctx)
    claims = _claims(ctx)
    mem = _members(ctx)
    prov = _providers(ctx)
    if ev.empty or claims.empty or prov.empty:
        return []
    window = int(ctx.cfg("pol01_cluster_window_days"))
    min_members = int(ctx.cfg("pol01_cluster_min_members"))
    early_days = int(ctx.cfg("early_tenure_days"))
    linked_min = float(ctx.cfg("pol01_cluster_linked_share"))
    q = float(ctx.cfg("pol01_cluster_peer_percentile"))
    adds = ev[_has_word(ev["event_type"], ("ADD", "ENROL", "INCLU", "NEW")) & ev["member_sk"].notna()].copy()
    adds = adds[~_has_word(adds["event_type"], _CAMPAIGN_WORDS) & ~_has_word(adds["actor"].fillna(""), _CAMPAIGN_WORDS)]
    if adds.empty:
        return []
    adds["joined"] = adds["effective_date"].fillna(adds["event_time"].dt.normalize())
    adds = adds.sort_values("joined").drop_duplicates("member_sk")
    emp = mem.set_index("member_sk")["employer_id"] if "employer_id" in mem.columns else pd.Series(dtype=object)
    adds["employer"] = adds["member_sk"].map(emp)
    adds["broker"] = adds["agent_id"].fillna(adds["cov_agent_id"])
    origin0 = adds["joined"].min()
    adds["bucket"] = ((adds["joined"] - origin0).dt.days // window).astype("Int64")
    groups = _linked_provider_groups(prov)
    gsize = pd.Series(groups).value_counts()
    c = claims[claims["member_sk"].isin(set(adds["member_sk"]))][["claim_sk", "member_sk", "service_date", "provider_sk", "gross"]]
    c = c.merge(adds[["member_sk", "joined"]], on="member_sk")
    early = c[((c["service_date"] - c["joined"]).dt.days >= 0) & ((c["service_date"] - c["joined"]).dt.days <= early_days)].copy()
    early["group"] = early["provider_sk"].map(groups)
    early["linked"] = early["group"].map(gsize).fillna(1) >= 2
    has_early = set(early["member_sk"])
    frames = []
    for kind, col in (("employer", "employer"), ("broker", "broker")):
        a = adds.dropna(subset=[col])[["member_sk", col, "bucket"]].rename(columns={col: "origin"})
        if a.empty:
            continue
        a = a.assign(kind=kind, early=a["member_sk"].isin(has_early))
        coh = a.groupby(["kind", "origin", "bucket"]).agg(n=("member_sk", "nunique"), k=("early", "sum")).reset_index()
        coh = coh[coh["n"] >= min_members]
        if coh.empty:
            continue
        a = a.merge(coh[["kind", "origin", "bucket"]], on=["kind", "origin", "bucket"])
        ec = early.merge(a[["member_sk", "kind", "origin", "bucket"]], on="member_sk")
        tot = ec.groupby(["kind", "origin", "bucket"]).size().rename("early_claims")
        lk = ec[ec["linked"]].groupby(["kind", "origin", "bucket", "group"]).size().rename("c").reset_index()
        lk = lk.sort_values("c", ascending=False).drop_duplicates(["kind", "origin", "bucket"])
        coh = coh.merge(tot.reset_index(), on=["kind", "origin", "bucket"], how="left").merge(
            lk, on=["kind", "origin", "bucket"], how="left")
        coh["rate"] = coh["k"] / coh["n"]
        coh["top_share"] = (coh["c"] / coh["early_claims"]).fillna(0.0)
        coh["top_group"] = coh["group"]
        mem = a.groupby(["kind", "origin", "bucket"])["member_sk"].apply(set).rename("members").reset_index()
        frames.append(coh.merge(mem, on=["kind", "origin", "bucket"]))
    if not frames:
        return []
    cohorts = pd.concat(frames, ignore_index=True)
    big = cohorts[cohorts["n"] >= min_members]
    if big.empty:
        return []
    out = []
    for kind in ("employer", "broker"):
        peers = big[big["kind"] == kind]
        min_peer = int(ctx.cfg("min_entity_opportunities"))
        if len(peers) < min_peer:
            continue
        cut = float(np.quantile(peers["rate"], q))
        med = float(peers["rate"].median())
        for r in peers.itertuples(index=False):
            if r.rate <= cut or r.top_share < linked_min or is_missing(r.top_group):
                continue
            e = early[early["member_sk"].isin(r.members)]
            ring = sorted(p for p, gid in groups.items() if gid == r.top_group)
            shared = []
            info = prov.set_index("provider_sk").loc[ring]
            for col in ("owner_entity_id", "bank_account_token", "phone_token", "address_token"):
                if col in info.columns and info[col].notna().any() and info[col].dropna().duplicated().any():
                    shared.append(col.replace("_token", "").replace("_", " "))
            start = origin0 + pd.Timedelta(days=int(r.bucket) * window)
            ring_claims = e[e["group"] == r.top_group]
            key = "agent_id" if kind == "broker" else "employer_id"
            out.append(_sig(
                ctx, control, subject_type="agent" if kind == "broker" else "employer", subject_id=r.origin,
                fact_key=f"enrol_cluster:{kind}:{r.origin}:{int(r.bucket)}", claim_ids=ring_claims["claim_sk"].tolist()[:200],
                event_time=start, period=_period(start), confidence=0.6,
                peer_level_used=f"{kind} enrolment cohorts",
                evidence={
                    key: r.origin, "cohort_start": _dstr(start), "cohort_window_days": window,
                    "members_added": int(r.n), "members_with_early_claims": int(e["member_sk"].nunique()),
                    "early_claim_rate": round(float(r.rate), 3), "peer_median_rate": round(med, 3),
                    "peer_percentile": q, "peer_cut_rate": round(cut, 3), "early_days": early_days,
                    "linked_providers": ring, "shared_identifiers": shared,
                    "share_of_early_claims_at_linked_providers": round(float(r.top_share), 3),
                    "reason_code": control.reason_code, "subject_id": r.origin,
                    "plain_language": (f"{count_phrase(r.n, 'member')} joined through this {kind} within {window} days "
                                       f"from {plain_date(start)}; {pct(r.rate)} claimed within {early_days} days "
                                       f"(typical: {pct(med)}), and {pct(r.top_share)} of those early claims went to "
                                       f"{count_phrase(len(ring), 'provider')} sharing "
                                       f"{', '.join(shared) or 'identifiers'}."),
                    "what_the_reviewer_must_verify": "List the members added and when, check the shared provider "
                                                     "identifiers, and check for an enrolment campaign.",
                },
                exposure=_exp.model_only_exposure(float(ring_claims["gross"].sum()), "enrolment cluster"),
            ))
    return out
