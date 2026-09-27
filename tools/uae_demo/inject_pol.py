"""Plant the POL (enrolment and policy) patterns in the SYNTHETIC UAE demo dataset.

Every control implemented in ``src/fwa/engine/unlocked/pol.py`` gets at least one
realistic pattern, recorded in the answer key; declared exclusions get a
look-alike (``positive=False``).

Patterns (all SYNTHETIC):

* POL-01-R01  employer-sponsored members who are not on their employer's roster
              (look-alike: a member on continuation cover after leaving), and one person
              enrolled twice under two member records with overlapping cover;
* POL-01-R02  members added to a policy weeks after an expensive treatment, with cover
              backdated over it (look-alike: an approved retroactive correction);
* POL-01-R03  applications declaring no prior cover or no existing conditions although
              earlier cover and treatment are on record (look-alike: the same conflict
              outside the contestability window);
* POL-01-R05  a new employer group whose members all claim within weeks of joining, at
              two clinics that share a bank account (look-alike: the same burst recorded
              as an enrolment campaign).

(POL-01-R04, early-tenure utilisation, already runs on the claim extract and is not
planted here.)
"""

from __future__ import annotations

import copy
import datetime as _dt
from typing import Any

import numpy as np
import pandas as pd

from .inject_ent import _Kit, _set_rows
from .world import World

D = _dt.date
TD = _dt.timedelta
DT = _dt.datetime
TENANT = "T001"


def _plant(name: str):
    def deco(fn):
        def run(k: _Kit) -> None:
            try:
                fn(k)
            except Exception as exc:  # noqa: BLE001
                k.w.notes.append(f"inject_pol.{name} failed: {exc!r}")
                print(f"    ! inject_pol.{name} failed: {exc!r}")
        run.__name__ = name
        return run
    return deco


@_plant("roster_conflict")
def _pol_01_r01(k: _Kit) -> None:
    w = k.w
    roster = w.tables["employer_roster"]
    active = k.claims[k.claims["sd"] >= pd.Timestamp("2024-09-01")]
    per = active.groupby("member_sk").size()
    on_roster = set(roster["member_sk"])
    mem = k.members(7, lambda f: f["employer_id"].notna() & f["member_sk"].isin(on_roster)
                    & f["member_sk"].isin(set(per[per >= 2].index)) & (f["relationship"] == "PRINCIPAL"))
    for i, m in enumerate(mem):
        roster = w.tables["employer_roster"]
        w.tables["employer_roster"] = roster[roster["member_sk"] != m].reset_index(drop=True)
        ids = active[active["member_sk"] == m]["claim_sk"].tolist()
        if i == len(mem) - 1:
            cov = w.tables["coverage_period"]
            _set_rows(w, "coverage_period", cov["member_sk"] == m, status="CONTINUATION")
            w.record("POL-01 member off the roster but on continuation cover (exclusion look-alike)",
                     rule_ids=["POL-01-R01"], claim_ids=ids, subject_type="member", subject_ids=[m], positive=False,
                     note="Continuation cover; must not fire.")
        else:
            w.used_claims.update(ids)
            w.record("POL-01 employer-sponsored member not on the employer's roster", rule_ids=["POL-01-R01"],
                     claim_ids=ids, subject_type="member", subject_ids=[m])
    # one person enrolled twice: same Emirates-ID reference, date of birth and sex, overlapping cover
    cov = w.tables["coverage_period"].copy()
    cov["vf"] = pd.to_datetime(cov["valid_from"])
    cov["vt"] = pd.to_datetime(cov["valid_to"])
    live = set(cov[(cov["vf"] <= pd.Timestamp("2025-01-01")) & (cov["vt"] >= pd.Timestamp("2025-03-01"))]["member_sk"])
    pair = k.members(4, lambda f: f["member_sk"].isin(live & set(active["member_sk"])))
    mt = w.tables["member"]
    for a, b in zip(pair[::2], pair[1::2]):
        src = mt[mt["member_sk"] == a].iloc[0]
        _set_rows(w, "member", mt["member_sk"] == b, protected_id_token=src["protected_id_token"],
                  date_of_birth=src["date_of_birth"], sex=src["sex"])
        ids = active[active["member_sk"] == b]["claim_sk"].tolist()
        w.record("POL-01 one person enrolled twice with overlapping cover", rule_ids=["POL-01-R01"], claim_ids=ids[:20],
                 subject_type="member", subject_ids=[a, b])


@_plant("retroactive_add")
def _pol_01_r02(k: _Kit) -> None:
    w = k.w
    ev = w.tables["policy_event"]
    cov = w.tables["coverage_period"].set_index("coverage_id")
    adds = ev[ev["event_type"] == "ADD"].copy()
    adds["member_sk"] = adds["coverage_id"].map(cov["member_sk"])
    adds["eff"] = pd.to_datetime(adds["effective_date"])
    adds = adds[adds["eff"] >= pd.Timestamp("2024-07-01")]
    q = float(k.claims["gross_amount"].quantile(0.95))
    big = k.claims[(k.claims["gross_amount"] >= q)].merge(adds[["member_sk", "eff", "policy_event_sk"]], on="member_sk")
    big = big[(big["sd"] >= big["eff"]) & ((big["sd"] - big["eff"]).dt.days <= 150)]
    cand = set(big["member_sk"])
    mem = k.members(6, lambda f: f["member_sk"].isin(cand))
    for i, m in enumerate(mem):
        b = big[big["member_sk"] == m].sort_values("sd").iloc[0]
        rec = (b["sd"] + pd.Timedelta(days=int(k.rng.integers(20, 45)))).to_pydatetime().replace(
            hour=int(k.rng.integers(17, 22)), minute=int(k.rng.integers(0, 60)))
        ev = w.tables["policy_event"]
        mask = ev["policy_event_sk"] == b["policy_event_sk"]
        look = i == len(mem) - 1
        _set_rows(w, "policy_event", mask, event_time=rec, actor="OPS-99" if not look else "UW-APPROVALS",
                  event_type="ADD_RETRO_APPROVED" if look else "ADD")
        ids = big[(big["member_sk"] == m) & (big["sd"] < pd.Timestamp(rec))]["claim_sk"].tolist()
        if look:
            w.record("POL-01 approved retroactive correction over a costly service (exclusion look-alike)",
                     rule_ids=["POL-01-R02"], claim_ids=ids, subject_type="member", subject_ids=[m], positive=False,
                     note="Approved retroactive correction; must not fire.")
        else:
            w.used_claims.update(ids)
            w.record("POL-01 member added after a costly service with cover backdated over it",
                     rule_ids=["POL-01-R02"], claim_ids=ids, subject_type="member", subject_ids=[m],
                     note=f"Add recorded {rec:%Y-%m-%d %H:%M}.")


@_plant("application_conflict")
def _pol_01_r03(k: _Kit) -> None:
    w = k.w
    app = w.tables["policy_application"].copy()
    prior = w.tables["prior_coverage_history"]
    app["ad"] = pd.to_datetime(app["application_date"])
    with_hist = set(prior[prior["conditions_treated"].astype(str).str.upper() != "NONE"]["member_sk"])
    recent = app[(app["ad"] >= pd.Timestamp("2024-03-01")) & app["member_sk"].isin(with_hist)]
    old = app[(app["ad"] < pd.Timestamp("2023-06-01")) & app["member_sk"].isin(with_hist)]
    mem = k.members(5, lambda f: f["member_sk"].isin(set(recent["member_sk"])))
    mem_old = k.members(1, lambda f: f["member_sk"].isin(set(old["member_sk"])))
    for i, m in enumerate(mem + mem_old):
        tab = w.tables["policy_application"]
        mask = tab["member_sk"] == m
        look = m in mem_old
        if i % 2 or look:
            _set_rows(w, "policy_application", mask, declared_conditions="NONE")
            what = "declared no existing conditions"
        else:
            _set_rows(w, "policy_application", mask, declared_conditions="NONE", declared_prior_cover=False)
            what = "declared no prior cover and no conditions"
        ids = k.claims[k.claims["member_sk"] == m]["claim_sk"].tolist()[:10]
        if look:
            w.record("POL-01 application conflict outside the contestability window (exclusion look-alike)",
                     rule_ids=["POL-01-R03"], claim_ids=ids, subject_type="member", subject_ids=[m], positive=False,
                     note="Application older than the contestability period; must not fire.")
        else:
            w.record(f"POL-01 application {what} although earlier treatment is on record", rule_ids=["POL-01-R03"],
                     claim_ids=ids, subject_type="member", subject_ids=[m])


def _new_member(k: _Kit, template: str, *, employer: str, agent: str, joined: D, end: D, emirate: str) -> str:
    """A new member (row, cover, roster entry, application) cloned from an existing adult."""
    w, ctx = k.w, k.ctx
    sk = w.new_id("MBRX", 6)
    mt = w.tables["member"]
    row = mt[mt["member_sk"] == template].iloc[0].to_dict()
    dob = D(int(k.rng.integers(1975, 2000)), int(k.rng.integers(1, 13)), int(k.rng.integers(1, 28)))
    sex = "M" if k.rng.random() < 0.7 else "F"
    row.update(member_sk=sk, source_member_id=f"CARD-{int(k.rng.integers(10 ** 9, 10 ** 10))}",
               protected_id_token=f"EIDTOK-{int(k.rng.integers(10 ** 15, 10 ** 16))}{int(k.rng.integers(10 ** 7, 10 ** 8))}",
               date_of_birth=dob, sex=sex, sponsor_id=sk, employer_id=employer, relationship="PRINCIPAL",
               emirate=emirate, death_date=None, death_source=None, death_source_confidence=None)
    w.append("member", [row])
    w.used_entities.add(sk)
    tm = ctx.members[template]
    cid = w.new_id("COVX", 6)
    cov = dict(coverage_id=cid, valid_from=joined, valid_to=end, product=tm["product"], payer=tm["payer"],
               network=tm["coverage"][0]["network"] if tm["coverage"] else None, agent=agent)
    w.append("coverage_period", [{
        "coverage_id": cid, "member_sk": sk, "product": tm["product"], "payer_id": tm["payer"], "valid_from": joined,
        "valid_to": end, "network": cov["network"], "status": "ACTIVE", "policy_inception_date": joined,
        "days_since_policy_start": None, "tenant_id": TENANT, "source_system": "UAE_MULTITABLE", "agent_id": agent,
        "product_tier": tm.get("tier") or None}])
    w.append("employer_roster", [{"employer_id": employer, "member_sk": sk, "sponsor_id": sk,
                                  "relationship": "PRINCIPAL", "valid_from": joined, "valid_to": end,
                                  "tenant_id": TENANT}])
    w.append("policy_application", [{"application_sk": w.new_id("APPX", 6), "member_sk": sk, "coverage_id": cid,
                                     "application_date": joined - TD(days=10), "declared_conditions": "NONE",
                                     "declared_prior_cover": False, "agent_id": agent, "tenant_id": TENANT}])
    m = copy.deepcopy(tm)
    m.update(sk=sk, dob=dob, sex=sex, emirate=emirate, employer=employer, sponsor=sk, relationship="PRINCIPAL",
             window=(joined, end), inception=joined, chronic=[], drugs={}, coverage=[cov], agent=agent)
    m.pop("cob", None)
    ctx.members[sk] = m
    return sk, cid


@_plant("enrolment_cluster")
def _pol_01_r05(k: _Kit) -> None:
    w, ctx = k.w, k.ctx
    for variant, emirate in (("cluster", "Sharjah"), ("campaign", "Ajman")):
        gp = [p for p, info in ctx.providers.items() if info["type"] == "CLINIC" and info["kind"]
              and info["kind"][0] in ("GP", "POLY") and info["emirate"] == emirate and p not in w.used_entities]
        if len(gp) < 2:
            gp = [p for p, info in ctx.providers.items() if info["type"] == "CLINIC" and info["kind"]
                  and info["kind"][0] in ("GP", "POLY") and p not in w.used_entities]
        pair = gp[:2]
        emirate = ctx.providers[pair[0]]["emirate"]
        w.used_entities.update(pair)
        prov = w.tables["provider"]
        if variant == "cluster":
            tok = prov.loc[prov["provider_sk"] == pair[0], "bank_account_token"].iloc[0]
            _set_rows(w, "provider", prov["provider_sk"] == pair[1], bank_account_token=tok)
        employer = w.new_id("EMPX", 4)
        agent = w.new_id("AGTX", 3)
        joined = D(2025, 4, 1) if variant == "cluster" else D(2025, 2, 1)
        adults = [m for m, info in ctx.members.items() if info.get("employer") and info["coverage"]
                  and info.get("relationship") == "PRINCIPAL"]
        template = adults[int(k.rng.integers(len(adults)))]
        made, members = [], []
        for i in range(12):
            msk, cid = _new_member(k, template, employer=employer, agent=agent, joined=joined,
                                   end=D(2026, 3, 31), emirate=emirate)
            members.append(msk)
            w.append("policy_event", [{
                "policy_event_sk": w.new_id("PEVX", 6), "coverage_id": cid,
                "event_type": "ADD" if variant == "cluster" else "ADD_CAMPAIGN",
                "actor": f"HR-{employer}" if variant == "cluster" else "OPEN_ENROLMENT_CAMPAIGN",
                "event_time": DT.combine(joined - TD(days=6), _dt.time(11, 30)), "effective_date": joined,
                "agent_id": agent, "tenant_id": TENANT}])
            for j in range(1 + (i % 2)):
                sd = joined + TD(days=int(k.rng.integers(3, 45)))
                c = k.claim(member_sk=msk, provider_sk=pair[(i + j) % 2], service_date=sd, claim_type="OUTPATIENT",
                            start_minute=int(k.rng.integers(9 * 60, 18 * 60)))
                if c:
                    made.append(c)
        if variant == "cluster":
            w.record("POL-01 new employer group claiming within weeks at two clinics sharing a bank account",
                     rule_ids=["POL-01-R05"], claim_ids=made, subject_type="employer", subject_ids=[employer, agent] + pair)
        else:
            w.record("POL-01 early claims after an enrolment campaign (exclusion look-alike)", rule_ids=["POL-01-R05"],
                     claim_ids=made, subject_type="employer", subject_ids=[employer], positive=False,
                     note="Enrolment campaign; must not fire.")


PLANTS = [_pol_01_r01, _pol_01_r02, _pol_01_r03, _pol_01_r05]


def inject(world: World) -> None:
    """Plant every POL pattern (see the module docstring)."""
    k = _Kit(world)
    for plant in PLANTS:
        plant(k)
        k.refresh()
