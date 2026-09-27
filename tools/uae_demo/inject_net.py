"""Plant the NET-family network patterns into the SYNTHETIC UAE demo dataset.

Each pattern is a small set of providers, members and claims built with the
base claim factory (so every claim is complete across the claim tables) plus,
where the pattern lives in the reference data, a change to the provider
register. Every planted item is recorded in the answer key; where a control
declares an exclusion a legitimate look-alike is planted too
(``positive=False``) so the exclusion is visibly exercised.

Controls exercised: NET-01-R02, NET-01-R03, NET-01-R04, NET-02-R01, NET-02-R04,
NET-03-R02, NET-03-R03, NET-03-R04, NET-04-R04. NET-04-R01 is not planted: it
needs reviewer-confirmed outcomes, which must never be synthesised.
"""

from __future__ import annotations

import datetime as _dt
import math
from typing import Any, Callable

import numpy as np
import pandas as pd

from .world import World

__all__ = ["inject"]

TD = _dt.timedelta
D = _dt.date


# ----------------------------------------------------------------- helpers


def _bctx(world: World):
    return world.context["base_ctx"]


def _claimed_members(world: World) -> set[str]:
    return set(world.tables["claim_header"]["member_sk"])


def _covers(world: World, member: str, lo: D, hi: D) -> bool:
    m = _bctx(world).members.get(member)
    if not m:
        return False
    a, b = m["window"]
    return a <= lo and b >= hi


def _members(world: World, n: int, *, emirate: str | None = None, lo: D, hi: D,
             extra: Callable[[pd.DataFrame], pd.Series] | None = None) -> list[str]:
    """``n`` unused members with no claims yet, covered from ``lo`` to ``hi``."""
    claimed = _claimed_members(world)

    def where(f: pd.DataFrame) -> pd.Series:
        mask = ~f["member_sk"].isin(claimed) & f["member_sk"].map(lambda m: _covers(world, m, lo, hi))
        if emirate is not None:
            mask &= f["emirate"] == emirate
        if extra is not None:
            mask &= extra(f)
        return mask

    return list(world.take_entities("member", "member_sk", n, where)["member_sk"])


def _provider(world: World, ptype: str, *, emirate: str | None = None,
              where: Callable[[pd.DataFrame], pd.Series] | None = None, pick: str = "random") -> str | None:
    frame = world.tables["provider"]
    mask = (frame["provider_type"] == ptype) & ~frame["provider_sk"].isin(world.used_entities)
    if ptype == "CLINIC":  # every clinic plant bills office visits: someone there must be privileged for them
        mask &= frame["provider_sk"].map(lambda p: _can(world, p, _OFFICE_VISIT)).astype(bool)
    if emirate is not None:
        mask &= frame["emirate"] == emirate
    if where is not None:
        mask &= where(frame).fillna(False).astype(bool)
    pool = frame[mask]
    if pool.empty:
        return None
    if pick == "quietest":
        volume = world.tables["claim_header"]["provider_sk"].value_counts()
        out_ref = world.tables["referral"]["referrer_provider_sk"].value_counts() if "referral" in world.tables else {}
        score = pool["provider_sk"].map(lambda p: volume.get(p, 0) + 5 * out_ref.get(p, 0))
        chosen = str(pool.loc[score.sort_values(kind="stable").index[0], "provider_sk"])
    else:
        chosen = str(pool.iloc[int(world.rng.integers(len(pool)))]["provider_sk"])
    world.used_entities.add(chosen)
    return chosen


def _clinician(world: World, provider: str) -> str | None:
    roster = _bctx(world).clinicians.get(provider) or []
    gp = [c for c in roster if c["specialty"] in ("GENERAL_PRACTICE", "FAMILY_MEDICINE", "INTERNAL_MEDICINE")]
    pool = gp or roster
    return pool[0]["id"] if pool else None


def _weekday(d: D) -> D:
    while d.weekday() >= 5:  # Friday/Saturday weekend is not modelled; keep to Mon-Fri
        d += TD(days=1)
    return d


def _day(world: World, lo: D, hi: D) -> D:
    return _weekday(lo + TD(days=int(world.rng.integers(0, max(1, (hi - lo).days)))))


def _families(world: World, codes) -> set[str]:
    b = _bctx(world)
    return {b.family_of(c) for c in codes}


def _can(world: World, provider: str, codes) -> bool:
    """Some clinician on the provider's roster is privileged for every one of ``codes``."""
    fams = _families(world, codes)
    return any(fams <= c["privileges"] for c in _bctx(world).clinicians.get(provider, []))


def _privileged(world: World, provider: str, codes, day: D) -> str | None:
    b = _bctx(world)
    fams = _families(world, codes)
    pool = [c["id"] for c in b.clinicians.get(provider, []) if fams <= c["privileges"] and b.available(c, day)]
    return pool[int(world.rng.integers(len(pool)))] if pool else None


_OFFICE_VISIT = ["99203", "99213"]


def _claim(world: World, **kw) -> str | None:
    """make_claim, with a rostered clinician privileged for every billed code.

    Explicit lines get a clinician privileged for all of them; a default outpatient claim
    (an office visit) gets one privileged for office visits. A plant no clinician at the
    provider could have rendered is skipped rather than written with no rendering clinician.
    """
    lines = kw.get("lines")
    if not lines and kw.get("claim_type") == "OUTPATIENT" and not kw.get("clinician_id"):
        cid = _privileged(world, kw["provider_sk"], _OFFICE_VISIT, kw["service_date"])
        if cid is None:
            return None
        kw["clinician_id"] = cid
    if lines and not kw.get("clinician_id"):
        codes = [(l if isinstance(l, str) else l["code"]) for l in lines]
        rendered = [c for c in codes if not _bctx(world).family_of(c).startswith(("LAB_", "IMAGING_"))]
        if rendered:
            cid = _privileged(world, kw["provider_sk"], rendered, kw["service_date"])
            if cid is None:
                return None  # never plant a service no clinician there could have rendered
            kw["clinician_id"] = cid
    try:
        return world.make_claim(**kw)
    except (RuntimeError, KeyError, ValueError):
        return None


def _set_provider(world: World, provider: str, **values: Any) -> None:
    prov = world.tables["provider"]
    world.set_values("provider", prov["provider_sk"] == provider, **values)


def _emirate(world: World, provider: str) -> str:
    return _bctx(world).providers[provider]["emirate"]


# ------------------------------------------------------------- NET-01-R02


def _reciprocal_loop(world: World) -> None:
    a = _provider(world, "CLINIC", emirate="Dubai", pick="quietest")
    if a is None:
        return
    b = _provider(world, "CLINIC", emirate=_emirate(world, a), pick="quietest")
    if b is None:
        return
    ca, cb = _clinician(world, a), _clinician(world, b)
    lo, hi = D(2025, 1, 5), D(2025, 11, 20)
    members = _members(world, 22, emirate=_emirate(world, a), lo=lo, hi=hi)
    claims = []
    for i, m in enumerate(members):
        to, frm = (b, ca) if i < 12 else (a, cb)
        c = _claim(world, member_sk=m, provider_sk=to, service_date=_day(world, lo, hi), claim_type="OUTPATIENT",
                   referral_from=frm)
        if c:
            claims.append(c)
    world.record("NET-01 two clinics referring patients back and forth", rule_ids=["NET-01-R02"],
                 claim_ids=claims, subject_type="provider", subject_ids=[a, b])


# ------------------------------------------------------------- NET-01-R03


def _high_cost_conversion(world: World) -> None:
    bctx = _bctx(world)
    ct = [p for p, info in bctx.providers.items()
          if info["type"] == "RADIOLOGY_CENTRE" and "CT" in info.get("equipment", {})]
    r = _provider(world, "RADIOLOGY_CENTRE", emirate="Dubai", where=lambda f: f["provider_sk"].isin(ct))         or _provider(world, "RADIOLOGY_CENTRE", where=lambda f: f["provider_sk"].isin(ct))
    if r is None:
        return
    emirate = _emirate(world, r)
    clinics = [p for p, info in bctx.providers.items() if info["type"] == "CLINIC" and info["emirate"] == emirate]
    lo, hi = D(2024, 9, 1), D(2025, 11, 15)
    members = _members(world, 24, emirate=emirate, lo=lo, hi=hi)
    claims = []
    for i, m in enumerate(members):
        referrer = _clinician(world, clinics[i % len(clinics)]) if clinics else None
        c = _claim(world, member_sk=m, provider_sk=r, service_date=_day(world, lo, hi), claim_type="RADIOLOGY",
                   lines=[{"code": "74177"}], diagnoses=["R10.9"], referral_from=referrer,
                   ordering_clinician_id=referrer)
        if c:
            claims.append(c)
    world.record("NET-01 radiology centre turning every referral into a contrast CT", rule_ids=["NET-01-R03"],
                 claim_ids=claims, subject_type="provider", subject_ids=[r])


# ------------------------------------------------------------- NET-01-R04 / NET-02-R01


def _closed_chain(world: World) -> None:
    prov = world.tables["provider"].set_index("provider_sk")
    a = _provider(world, "CLINIC", pick="quietest")
    if a is None:
        return
    emirate = _emirate(world, a)
    lab = _provider(world, "DIAGNOSTIC_LAB", emirate=emirate)
    rad = _provider(world, "RADIOLOGY_CENTRE", emirate=emirate)
    if lab is None or rad is None:
        return
    _set_provider(world, lab, bank_account_token=prov.at[a, "bank_account_token"])
    _set_provider(world, rad, phone_token=prov.at[a, "phone_token"])
    clin = _clinician(world, a)
    lo, hi = D(2024, 10, 1), D(2025, 11, 15)
    members = _members(world, 14, emirate=emirate, lo=lo, hi=hi)
    claims = []
    for i, m in enumerate(members):
        dest, ctype = (lab, "LAB") if i < 10 else (rad, "RADIOLOGY")
        c = _claim(world, member_sk=m, provider_sk=dest, service_date=_day(world, lo, hi), claim_type=ctype,
                   referral_from=clin)
        if c:
            claims.append(c)
    world.record("NET-01 clinic keeps its lab and imaging referrals inside providers sharing its bank account "
                 "and phone", rule_ids=["NET-01-R04", "NET-02-R01"], claim_ids=claims, subject_type="provider",
                 subject_ids=[a, lab, rad])

    # Look-alike: a declared group (same owner) sharing premises with its own laboratory.
    c3 = _provider(world, "CLINIC", pick="quietest")
    if c3 is None:
        return
    lab3 = _provider(world, "DIAGNOSTIC_LAB", emirate=_emirate(world, c3))
    if lab3 is None:
        return
    _set_provider(world, lab3, owner_entity_id=prov.at[c3, "owner_entity_id"],
                  address_token=prov.at[c3, "address_token"])
    clin3 = _clinician(world, c3)
    members = _members(world, 10, emirate=_emirate(world, c3), lo=lo, hi=hi)
    claims = [c for c in (_claim(world, member_sk=m, provider_sk=lab3, service_date=_day(world, lo, hi),
                                 claim_type="LAB", referral_from=clin3) for m in members) if c]
    world.record("NET-01 clinic referring to its own declared group laboratory", rule_ids=["NET-01-R04", "NET-02-R01"],
                 claim_ids=claims, subject_type="provider", subject_ids=[c3, lab3], positive=False,
                 note="Declared common owner: a disclosed, integrated group (known-group / ownership exclusion).")


def _shared_identity(world: World) -> None:
    prov = world.tables["provider"].set_index("provider_sk")
    p = _provider(world, "CLINIC")
    q = _provider(world, "CLINIC")
    if p and q:
        _set_provider(world, q, bank_account_token=prov.at[p, "bank_account_token"])
        world.record("NET-02 two separately owned clinics paid into one bank account", rule_ids=["NET-02-R01"],
                     subject_type="provider", subject_ids=[p, q])
    h = _provider(world, "HOSPITAL")
    ph = _provider(world, "PHARMACY")
    if h and ph:
        _set_provider(world, ph, phone_token=prov.at[h, "phone_token"], address_token=prov.at[h, "address_token"])
        world.record("NET-02 pharmacy sharing a hospital's phone number and address under a different owner",
                     rule_ids=["NET-02-R01"], subject_type="provider", subject_ids=[h, ph])
    # Look-alike: six providers in one medical tower share its address (a shared service).
    tower = [x for x in (_provider(world, "CLINIC") for _ in range(6)) if x]
    if len(tower) == 6:
        for x in tower[1:]:
            _set_provider(world, x, address_token=prov.at[tower[0], "address_token"])
        world.record("NET-02 six clinics in one medical tower share its address", rule_ids=["NET-02-R01"],
                     subject_type="provider", subject_ids=tower, positive=False,
                     note="Shared-services exclusion: more providers share the token than a covert link explains.")


# ------------------------------------------------------------- NET-02-R04


def _monthly_p95(world: World) -> int:
    h = world.tables["claim_header"]
    enc = world.tables["encounter"].drop_duplicates("claim_sk").set_index("claim_sk")
    month = pd.to_datetime(h["claim_sk"].map(enc["start_time"]), errors="coerce").dt.to_period("M")
    deg = h.assign(_m=month).groupby(["provider_sk", "_m"])["member_sk"].nunique()
    return int(math.ceil(float(np.quantile(deg.to_numpy(), 0.95)))) if len(deg) else 15


def _rise(world: World, provider: str, per_month: int, months: list[D]) -> list[str]:
    emirate = _emirate(world, provider)
    claims = []
    for start in months:
        end = (pd.Timestamp(start) + pd.offsets.MonthEnd(1)).date()
        for m in _members(world, per_month, emirate=emirate, lo=start, hi=end):
            c = _claim(world, member_sk=m, provider_sk=provider, service_date=_day(world, start, end - TD(days=2)),
                       claim_type="OUTPATIENT", lines=[{"code": "99204"}, {"code": "93000"}], diagnoses=["R07.9"])
            if c:
                claims.append(c)
    return claims


def _structural_change(world: World) -> None:
    per_month = _monthly_p95(world) + 8
    months = [D(2025, 6, 1), D(2025, 7, 1), D(2025, 8, 1), D(2025, 9, 1)]
    rise_codes = ["99204", "93000"]
    s = _provider(world, "CLINIC", emirate="Dubai", pick="quietest",
                  where=lambda f: (pd.to_datetime(f["credentialing_date"]) < pd.Timestamp("2023-01-01"))
                  & f["provider_sk"].map(lambda p: _can(world, p, rise_codes)))
    if s is None:
        return
    _set_provider(world, s, ownership_changed_on=D(2025, 4, 20))
    claims = _rise(world, s, per_month, months)
    world.record("NET-02 quiet clinic becomes one of the busiest right after an ownership change",
                 rule_ids=["NET-02-R04"], claim_ids=claims, subject_type="provider", subject_ids=[s])
    s2 = _provider(world, "CLINIC", emirate="Abu Dhabi", pick="quietest",
                   where=lambda f: (pd.to_datetime(f["credentialing_date"]) < pd.Timestamp("2023-01-01"))
                   & f["provider_sk"].map(lambda p: _can(world, p, rise_codes)))
    if s2 is None:
        return
    world.append("contract", [{
        "contract_sk": f"CTR-{s2}-NEW", "provider_sk": s2, "payer_id": world.tables["contract"]["payer_id"].iloc[0],
        "valid_from": D(2025, 6, 1), "valid_to": D(2026, 12, 31), "tariff_basis": "SYN-TARIFF-2024/2025",
        "discount_pct": 0.05, "network_tier": "TIER_2", "tenant_id": "T001"}])
    claims2 = _rise(world, s2, per_month, months)
    world.record("NET-02 quiet clinic grows fast after joining a new payer network", rule_ids=["NET-02-R04"],
                 claim_ids=claims2, subject_type="provider", subject_ids=[s2], positive=False,
                 note="New-contract exclusion: a contract starting as the growth begins explains it.")


# ------------------------------------------------------------- NET-03-R02 / NET-04-R04 look-alike


def _new_employer_group(world: World, n: int, lo: D, hi: D, emirate: str = "Dubai") -> tuple[str, list[str]] | None:
    """A new small employer: ``n`` individually insured, never-claimed members put on one employer's roster.

    Uses the preferred emirate when it still has enough such members, else the emirate that has the most.
    """
    m = world.tables["member"]
    claimed = _claimed_members(world)
    ok = (m["employer_id"].isna() & ~m["member_sk"].isin(claimed) & ~m["member_sk"].isin(world.used_entities)
          & m["member_sk"].map(lambda x: _covers(world, x, lo, hi)))
    counts = m[ok]["emirate"].value_counts()
    if counts.empty or counts.iloc[0] < n:
        return None
    if counts.get(emirate, 0) < n:
        emirate = str(counts.index[0])
    members = _members(world, n, emirate=emirate, lo=lo, hi=hi, extra=lambda f: f["employer_id"].isna())
    if len(members) < n:
        return None
    emp = world.new_id("EMPX", 4)
    mem = world.tables["member"]
    world.set_values("member", mem["member_sk"].isin(members), employer_id=emp)
    world.append("employer_roster", [{"employer_id": emp, "member_sk": x, "sponsor_id": x, "relationship": "PRINCIPAL",
                                      "valid_from": lo - TD(days=60), "valid_to": D(2099, 12, 31), "tenant_id": "T001"}
                                     for x in members])
    for x in members:
        _bctx(world).members[x]["employer"] = emp
    return emp, members


_GROUP_WINDOWS = [(D(2025, 3, 1), D(2025, 9, 30)), (D(2025, 1, 1), D(2025, 6, 30)), (D(2025, 6, 1), D(2025, 11, 30)),
                  (D(2024, 9, 1), D(2025, 2, 28)), (D(2025, 4, 1), D(2025, 7, 31))]


def _group_with_window(world: World, n: int, emirate: str):
    for lo, hi in _GROUP_WINDOWS:
        for size in (n, n - 2):
            found = _new_employer_group(world, size, lo, hi, emirate=emirate)
            if found is not None:
                return found, lo, hi
    return None, None, None


def _closed_group(world: World) -> None:
    found, lo, hi = _group_with_window(world, 14, "Dubai")
    if found is None:
        return
    emp, members = found
    emirate = _bctx(world).members[members[0]]["emirate"]
    k1 = _provider(world, "CLINIC", emirate=emirate, pick="quietest")
    k2 = _provider(world, "CLINIC", emirate=emirate, pick="quietest")
    if k1 is None or k2 is None:
        return
    claims = []
    for m in members:
        for j in range(3):
            c = _claim(world, member_sk=m, provider_sk=k1 if j < 2 else k2, service_date=_day(world, lo, hi),
                       claim_type="OUTPATIENT")
            if c:
                claims.append(c)
    world.record("NET-03 one small employer's staff all treated by the same two clinics", rule_ids=["NET-03-R02"],
                 claim_ids=claims, subject_type="provider", subject_ids=[k1, k2], note=f"employer {emp}")

    # Look-alike: an employer's own on-site clinic, owned by the employer (NET-03-R02 and NET-04-R04).
    found, lo, hi = _group_with_window(world, 10, "Abu Dhabi")
    if found is None:
        return
    emp2, members2 = found
    k3 = _provider(world, "CLINIC", emirate=_bctx(world).members[members2[0]]["emirate"], pick="quietest")
    if k3 is None:
        return
    _set_provider(world, k3, facility_type="EMPLOYER_ONSITE_CLINIC", owner_entity_id=emp2)
    claims = [c for m in members2 for c in (
        _claim(world, member_sk=m, provider_sk=k3, service_date=_day(world, lo, hi), claim_type="OUTPATIENT"),
        _claim(world, member_sk=m, provider_sk=k3, service_date=_day(world, lo, hi), claim_type="OUTPATIENT")) if c]
    world.record("NET-03 employer's own on-site clinic serving its staff", rule_ids=["NET-03-R02", "NET-04-R04"],
                 claim_ids=claims, subject_type="provider", subject_ids=[k3], positive=False,
                 note="Employer on-site care / known corporate link: the declared exclusions.")


# ------------------------------------------------------------- NET-03-R03


def _inducement(world: World) -> None:
    bctx = _bctx(world)
    able = lambda f: f["provider_sk"].map(lambda p: _can(world, p, ["99214", "20610"]))  # noqa: E731
    z = _provider(world, "CLINIC", emirate="Abu Dhabi", where=able) or _provider(world, "CLINIC", emirate="Dubai",
                                                                                  where=able)
    if z is None:
        return
    lo, hi = D(2025, 1, 10), D(2025, 10, 31)
    members = _members(world, 8, emirate=_emirate(world, z), lo=lo, hi=hi,
                       extra=lambda f: f["member_sk"].map(lambda m: bctx.members[m]["product"] in
                                                          ("GROUP_ESSENTIAL", "INDIVIDUAL_PLUS")))
    claims = []
    for m in members:
        for k in range(4):
            day = _weekday(lo + TD(days=35 * k + int(world.rng.integers(0, 10))))
            c = _claim(world, member_sk=m, provider_sk=z, service_date=day, claim_type="OUTPATIENT",
                       lines=[{"code": "99214"}, {"code": "20610"}], diagnoses=["M17.11"])
            if c:
                claims.append(c)
    if not claims:
        return
    # The patient pays nothing: the share is waived and billed to the insurer instead.
    lines = world.tables["claim_line"]
    mask = lines["claim_sk"].isin(set(claims))
    lines.loc[mask, "net_amount"] = lines.loc[mask, "gross_amount"]
    lines.loc[mask, "patient_share"] = 0.0
    world.recompute_header_amounts(claims)
    remit = world.tables["remittance"]
    net = dict(zip(lines.loc[mask, "line_sk"], lines.loc[mask, "net_amount"]))
    rm = remit["line_sk"].isin(set(net)) & (remit["decision"] == "PAID")
    remit.loc[rm, "payment_amount"] = remit.loc[rm, "line_sk"].map(net).round(2)
    world.record("NET-03 clinic waiving patient share on repeated knee injections for a small patient group",
                 rule_ids=["NET-03-R03"], claim_ids=claims, subject_type="provider", subject_ids=[z])


# ------------------------------------------------------------- NET-03-R04


def _complaints(world: World) -> None:
    h = world.tables["claim_header"]
    y = _provider(world, "CLINIC", where=lambda f: f["provider_sk"].map(h["provider_sk"].value_counts()).fillna(0) >= 30)
    if y is None:
        return
    taken = world.take_claims(3, lambda f: (f["provider_sk"] == y) & (f["claim_type"] == "OUTPATIENT"))
    taken = taken.drop_duplicates("member_sk")
    if len(taken) < 2:
        return
    rows_c, rows_k = [], []
    for i, r in enumerate(taken.itertuples(index=False)):
        when = pd.Timestamp(r.submission_date) + pd.Timedelta(days=int(world.rng.integers(5, 20)))
        if i < 2:
            rows_c.append({"confirmation_sk": world.new_id("MCFX", 5), "member_sk": r.member_sk, "claim_sk": r.claim_sk,
                           "service_confirmed": False,
                           "response": "SYNTHETIC reply: I did not receive the injection and dressing billed on this visit.",
                           "response_date": when.date(), "channel": "APP", "tenant_id": "T001"})
        else:
            rows_k.append({"complaint_sk": world.new_id("CMPX", 5), "member_sk": r.member_sk, "provider_sk": y,
                           "claim_sk": r.claim_sk, "complaint_type": "SERVICE_NOT_RECEIVED", "complaint_date": when.date(),
                           "text": "SYNTHETIC complaint: my statement shows a procedure that was never performed.",
                           "tenant_id": "T001"})
    world.append("member_confirmation", rows_c)
    world.append("complaint", rows_k)
    world.record("NET-03 several patients of one clinic say billed services were never given",
                 rule_ids=["NET-03-R04"], claim_ids=taken["claim_sk"], subject_type="provider", subject_ids=[y])

    # Look-alike: the same statements through an unauthenticated web form.
    y2 = _provider(world, "CLINIC", where=lambda f: f["provider_sk"].map(h["provider_sk"].value_counts()).fillna(0) >= 30)
    if y2 is None:
        return
    taken2 = world.take_claims(2, lambda f: (f["provider_sk"] == y2) & (f["claim_type"] == "OUTPATIENT"))
    taken2 = taken2.drop_duplicates("member_sk")
    world.append("member_confirmation", [{
        "confirmation_sk": world.new_id("MCFX", 5), "member_sk": r.member_sk, "claim_sk": r.claim_sk,
        "service_confirmed": False, "response": "SYNTHETIC reply: not sure this visit happened.",
        "response_date": (pd.Timestamp(r.submission_date) + pd.Timedelta(days=9)).date(),
        "channel": "UNVERIFIED_WEB_FORM", "tenant_id": "T001"} for r in taken2.itertuples(index=False)])
    world.record("NET-03 unverified web-form replies disputing visits", rule_ids=["NET-03-R04"],
                 claim_ids=taken2["claim_sk"], subject_type="provider", subject_ids=[y2], positive=False,
                 note="Evidence-reliability exclusion: the channel is not authenticated.")


# ------------------------------------------------------------- NET-04-R04


def _agent_owned_clinic(world: World) -> None:
    cover = world.tables["coverage_period"]
    agents = cover["agent_id"].value_counts()
    agent = next((str(a) for a in agents.index if str(a) not in world.used_entities), None)
    if agent is None:
        return
    world.used_entities.add(agent)
    sold = set(cover.loc[cover["agent_id"] == agent, "member_sk"])
    emirates = pd.Series([_bctx(world).members[m]["emirate"] for m in sold]).value_counts()
    emirate = str(emirates.index[0]) if len(emirates) else None
    w = _provider(world, "CLINIC", emirate=emirate)
    if w is None:
        return
    _set_provider(world, w, owner_entity_id=agent)
    lo, hi = D(2025, 2, 1), D(2025, 10, 31)
    members = _members(world, 10, emirate=emirate, lo=lo, hi=hi, extra=lambda f: f["member_sk"].isin(sold))
    claims = [c for c in (_claim(world, member_sk=m, provider_sk=w, service_date=_day(world, lo, hi),
                                 claim_type="OUTPATIENT") for m in members) if c]
    world.record("NET-04 clinic owned by the agent who sold its patients' policies", rule_ids=["NET-04-R04"],
                 claim_ids=claims, subject_type="provider", subject_ids=[w], note=f"agent {agent}")


# ------------------------------------------------------------------- entry


def inject(world: World) -> None:
    if "base_ctx" not in world.context:
        return
    _reciprocal_loop(world)
    _high_cost_conversion(world)
    _closed_chain(world)
    _shared_identity(world)
    _structural_change(world)
    _closed_group(world)
    _inducement(world)
    _complaints(world)
    _agent_owned_clinic(world)
