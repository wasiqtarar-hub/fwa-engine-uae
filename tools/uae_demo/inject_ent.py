"""Plant the ENT (entitlement) patterns in the SYNTHETIC UAE demo dataset.

Every control implemented in ``src/fwa/engine/unlocked/ent.py`` gets at least one
realistic pattern here, at a modest volume, recorded in the answer key with the
rule ids it is meant to exercise. Where a declared exclusion has a legitimate
look-alike, one is planted too (``positive=False``) so the exclusion is shown.

Patterns (all SYNTHETIC):

* ENT-01-R01  visits weeks after the member's last cover period ended
              (look-alike: an emergency visit after cover ended);
* ENT-01-R02  cosmetic rhinoplasty billed under a plan that excludes cosmetic care
              (look-alike: the same service paid in full by the patient);
* ENT-01-R03  a physiotherapy course billed past the plan's annual physiotherapy limit;
* ENT-01-R04  two clinics that left the payer network keep billing its members;
* ENT-02-R01  one Emirates-ID reference on two member records with different dates of birth;
* ENT-02-R02  services dated after a member's recorded death
              (look-alike: a death reported by an unreliable source);
* ENT-02-R04  one member's card used in two distant emirates on the same days
              (look-alike: same-day visits in neighbouring emirates);
* ENT-03-R01  clinics billing after their facility licence expired, clinicians after theirs
              (look-alike: a service inside the licence grace period);
* ENT-03-R03  GPs whose privileges exclude joint injections billing joint injections;
* ENT-03-R04  GPs billing specialist consultation codes;
* ENT-03-R05  a newly credentialed clinic that bills far above its peers straight away
              (look-alike: a burst right after an acquisition);
* ENT-04-R01  service lines with no treating clinician, the facility's code, or an unknown ID;
* ENT-04-R02  a clinician from another facility billed as the treating clinician;
* ENT-04-R03  physiotherapists billed for three overlapping timed sessions;
* ENT-04-R04  clinicians billed while on recorded annual leave, and one billed in two
              emirates twenty minutes apart;
* ENT-05-R01  consultations billed by pharmacies, laboratory panels by imaging centres;
* ENT-05-R02  inpatient stays with no bed assigned (look-alike: a transferred patient);
* ENT-05-R03  an outpatient visit billed at the same hospital during the patient's stay;
* ENT-05-R04  a hospital whose ordinary visits are recorded as day cases from one month on;
* ENT-06-R01  telehealth visits billing an ECG, which cannot be done remotely;
* ENT-06-R02  three-minute teleconsults that end in a prescription (look-alike: a short
              call with no order);
* ENT-06-R03  a telehealth clinic sending nearly every prescription to one pharmacy;
* ENT-06-R04  a telehealth clinic whose calls turn into MRI referrals far more often than peers'.
"""

from __future__ import annotations

import copy
import datetime as _dt
import json
from typing import Any, Callable

import numpy as np
import pandas as pd

from .world import World

D = _dt.date
TD = _dt.timedelta
DT = _dt.datetime
TENANT = "T001"

#: neighbouring emirates (a same-day visit to both is plausible); mirrors the governed
#: parameter ent02_adjacent_emirates, used here only to choose realistic plants.
_ADJACENT = {frozenset(p) for p in (
    ("Abu Dhabi", "Dubai"), ("Dubai", "Sharjah"), ("Dubai", "Ajman"), ("Sharjah", "Ajman"),
    ("Sharjah", "Umm Al Quwain"), ("Ajman", "Umm Al Quwain"), ("Umm Al Quwain", "Ras Al Khaimah"),
    ("Sharjah", "Ras Al Khaimah"), ("Sharjah", "Fujairah"), ("Ras Al Khaimah", "Fujairah"))}


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _d(v: Any) -> D | None:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    ts = pd.Timestamp(v)
    return None if pd.isna(ts) else ts.date()


class _Kit:
    def __init__(self, world: World) -> None:
        self.w = world
        self.ctx = world.context["base_ctx"]
        self.rng = world.rng
        self.refresh()

    def refresh(self) -> None:
        w = self.w
        lines = w.tables["claim_line"]
        first = pd.to_datetime(lines["service_date"], errors="coerce").groupby(lines["claim_sk"]).min()
        h = w.tables["claim_header"][["claim_sk", "member_sk", "provider_sk", "claim_type", "gross_amount"]].copy()
        h["sd"] = h["claim_sk"].map(first)
        prov = w.tables["provider"].set_index("provider_sk")
        h["ptype"] = h["provider_sk"].map(prov["provider_type"])
        h["emirate"] = h["provider_sk"].map(prov["emirate"])
        self.claims = h
        self.prov = prov

    # ------------------------------------------------------------- selection

    def providers(self, n: int, where: Callable[[pd.DataFrame], pd.Series]) -> list[str]:
        return list(self.w.take_entities("provider", "provider_sk", n, where)["provider_sk"])

    def members(self, n: int, where: Callable[[pd.DataFrame], pd.Series]) -> list[str]:
        return list(self.w.take_entities("member", "member_sk", n, where)["member_sk"])

    def clinicians(self, n: int, where: Callable[[pd.DataFrame], pd.Series]) -> list[str]:
        ros = self.w.tables["clinician_roster"]
        taken = self.w.take_entities("clinician_roster", "clinician_id", n, where)
        return list(dict.fromkeys(taken["clinician_id"]))

    def free_claims(self) -> pd.Series:
        return ~self.claims["claim_sk"].isin(self.w.used_claims)

    def clinic_of(self, clin: str) -> str:
        return self.ctx.clin_by_id[clin]["provider"]

    def claim(self, **kw) -> str | None:
        try:
            return self.w.make_claim(**kw)
        except Exception as exc:  # noqa: BLE001 - one failed plant must not sink the build
            self.w.notes.append(f"inject_ent: make_claim failed ({exc!r})")
            return None

    def date_in(self, lo: D, hi: D) -> D:
        span = max((hi - lo).days, 0)
        return lo + TD(days=int(self.rng.integers(0, span + 1)))

    def member_on_cover(self, member: str, on: D) -> bool:
        return self.ctx.coverage_at(member, on) is not None


def _set_rows(world: World, table: str, mask: pd.Series, **values: Any) -> None:
    """Set columns on the rows of ``world.tables[table]`` selected by ``mask`` (aligned on the index)."""
    frame = world.tables[table]
    mask = mask.reindex(frame.index, fill_value=False).astype(bool)
    idx = frame.index[mask]
    for col, val in values.items():
        if col not in frame.columns:
            frame[col] = None
        if isinstance(val, (np.ndarray, list, pd.Series)):
            vals = list(np.asarray(val).tolist())
        else:
            vals = [val] * len(idx)
        numeric = all(isinstance(v, (int, float, np.number)) and not isinstance(v, bool) for v in vals)
        if not (numeric and pd.api.types.is_numeric_dtype(frame[col].dtype)) and frame[col].dtype != object:
            frame[col] = frame[col].astype(object)
        for i, v in zip(idx, vals):
            frame.at[i, col] = v


def _plant(name: str):
    def deco(fn):
        def run(k: _Kit) -> None:
            try:
                fn(k)
            except Exception as exc:  # noqa: BLE001
                k.w.notes.append(f"inject_ent.{name} failed: {exc!r}")
                print(f"    ! inject_ent.{name} failed: {exc!r}")
        run.__name__ = name
        return run
    return deco


# ---------------------------------------------------------------------------
# ENT-01
# ---------------------------------------------------------------------------


@_plant("coverage_inactive")
def _ent_01_r01(k: _Kit) -> None:
    cov = k.w.tables["coverage_period"].copy()
    cov["vt"] = pd.to_datetime(cov["valid_to"])
    last = cov.groupby("member_sk")["vt"].max()
    ended = set(last[(last >= pd.Timestamp("2024-10-01")) & (last <= pd.Timestamp("2025-10-15"))].index)
    seen = k.claims.dropna(subset=["sd"]).groupby("member_sk")
    home = seen["provider_sk"].agg(lambda s: s.value_counts().index[0])
    clinics = set(k.prov.index[k.prov["provider_type"] == "CLINIC"])
    cand = {m for m in ended if m in home.index and home[m] in clinics}
    mem = k.members(9, lambda f: f["member_sk"].isin(cand))
    planted = []
    for i, m in enumerate(mem):
        end = last[m].date()
        sd = end + TD(days=int(k.rng.integers(20, 60)))
        if i == len(mem) - 1:
            # look-alike: an emergency visit after cover ended is covered by the emergency rule
            hosp = [p for p in k.prov.index if k.prov.loc[p, "provider_type"] == "HOSPITAL"
                    and k.prov.loc[p, "emirate"] == k.ctx.members[m]["emirate"]] or \
                   [p for p in k.prov.index if k.prov.loc[p, "provider_type"] == "HOSPITAL"]
            c = k.claim(member_sk=m, provider_sk=hosp[0], service_date=sd, claim_type="OUTPATIENT",
                        lines=["99283"], encounter_type="EMERGENCY")
            if c:
                k.w.record("ENT-01 emergency visit after cover ended (exclusion look-alike)",
                           rule_ids=["ENT-01-R01"], claim_ids=[c], subject_type="claim", subject_ids=[m],
                           positive=False, note="Emergency rule applies; must not fire.")
            continue
        c = k.claim(member_sk=m, provider_sk=home[m], service_date=sd, claim_type="OUTPATIENT")
        if c:
            planted.append(c)
    k.w.record("ENT-01 service after cover ended", rule_ids=["ENT-01-R01"], claim_ids=planted,
               subject_type="claim", note="Visit 20-60 days after the member's last cover period ended.")


@_plant("benefit_not_covered")
def _ent_01_r02(k: _Kit) -> None:
    hosp = list(k.prov.index[k.prov["provider_type"] == "HOSPITAL"])
    adults = k.w.tables["member"]
    mem = k.members(7, lambda f: (pd.to_datetime(f["date_of_birth"]) < pd.Timestamp("2000-01-01"))
                    & f["member_sk"].isin(set(k.claims["member_sk"])))
    planted = []
    for i, m in enumerate(mem):
        emirate = k.ctx.members[m]["emirate"]
        ph = [p for p in hosp if k.prov.loc[p, "emirate"] == emirate] or hosp
        sd = k.date_in(D(2024, 9, 1), D(2025, 11, 30))
        if not k.member_on_cover(m, sd):
            continue
        c = k.claim(member_sk=m, provider_sk=ph[int(k.rng.integers(len(ph)))], service_date=sd,
                    claim_type="OUTPATIENT", lines=["30400"], diagnoses=["J34.2"] if "J34.2" in k.ctx.icd else None)
        if not c:
            continue
        lines = k.w.tables["claim_line"]
        mask = lines["claim_sk"] == c
        if i == 0:
            # look-alike: the patient paid the whole service themselves (self-pay route)
            _set_rows(k.w, "claim_line", mask, patient_share=lines.loc[mask, "gross_amount"].values, net_amount=0.0)
            k.w.recompute_header_amounts([c])
            k.w.record("ENT-01 excluded cosmetic service paid by the patient (exclusion look-alike)",
                       rule_ids=["ENT-01-R02"], claim_ids=[c], subject_type="claim", positive=False,
                       note="Self-pay route; must not fire.")
            continue
        gross = lines.loc[mask, "gross_amount"].astype(float).values
        _set_rows(k.w, "claim_line", mask, patient_share=(gross * 0.2).round(2), net_amount=(gross * 0.8).round(2))
        k.w.recompute_header_amounts([c])
        planted.append(c)
    k.w.record("ENT-01 cosmetic service billed under a plan that excludes it", rule_ids=["ENT-01-R02"],
               claim_ids=planted, subject_type="claim", note="Rhinoplasty (COSMETIC family, covered=false).")


@_plant("benefit_limit_exceeded")
def _ent_01_r03(k: _Kit) -> None:
    ctx = k.ctx
    physio = [p for p, info in ctx.providers.items() if info["type"] == "CLINIC"
              and any("PHYSIO" in c["privileges"] for c in ctx.clinicians.get(p, []))]
    brv = k.w.tables["benefit_rule_version"]
    lim = brv[brv["service_family"] == "PHYSIOTHERAPY"].set_index(["product", "valid_from"])["benefit_limit"]
    cov = k.w.tables["coverage_period"].copy()
    cov["vf"] = pd.to_datetime(cov["valid_from"])
    cov["vt"] = pd.to_datetime(cov["valid_to"])
    lo = cov["vf"].clip(lower=pd.Timestamp("2024-07-01"))
    hi = cov["vt"].clip(upper=pd.Timestamp("2025-12-20"))
    ok_cov = cov[(hi - lo).dt.days >= 300]
    emirates = {ctx.providers[p]["emirate"] for p in physio}
    mem = k.members(2, lambda f: f["member_sk"].isin(set(ok_cov["member_sk"])) & f["emirate"].isin(emirates)
                    & (pd.to_datetime(f["date_of_birth"]) < pd.Timestamp("1995-01-01")))
    planted_all = []
    for m in mem:
        prov = [p for p in physio if ctx.providers[p]["emirate"] == ctx.members[m]["emirate"]][0]
        clin = next(c["id"] for c in ctx.clinicians[prov] if "PHYSIO" in c["privileges"])
        mcov = ok_cov[ok_cov["member_sk"] == m].iloc[0]
        year_start = max(mcov["vf"].date(), D(2024, 7, 1))
        product = mcov["product"]
        limit = float(brv[(brv["product"] == product) & (brv["service_family"] == "PHYSIOTHERAPY")]
                      ["benefit_limit"].max())
        used = 0.0
        day = year_start + TD(days=10)
        made = []
        for session in range(60):
            if used > limit * 1.08 or day > min(mcov["vt"].date(), D(2025, 12, 20)):
                break
            c = k.claim(member_sk=m, provider_sk=prov, service_date=day, claim_type="OUTPATIENT", clinician_id=clin,
                        lines=[{"code": "97110", "units": 4}, {"code": "97140", "units": 4},
                               {"code": "97530", "units": 4}], diagnoses=["M54.50"] if "M54.50" in ctx.icd else None,
                        start_minute=int(k.rng.integers(8 * 60, 16 * 60)))
            if c:
                made.append(c)
                ln = k.w.tables["claim_line"]
                used += float(ln.loc[ln["claim_sk"] == c, "net_amount"].sum())
            day += TD(days=5)
        planted_all += made
        k.w.record("ENT-01 physiotherapy course billed past the annual benefit limit", rule_ids=["ENT-01-R03"],
                   claim_ids=made[-3:], subject_type="member", subject_ids=[m],
                   note=f"Weekly sessions; the plan limit is AED {limit:,.0f}; the last sessions exceed it.")


@_plant("network_violation")
def _ent_01_r04(k: _Kit) -> None:
    after = pd.Timestamp("2025-10-01")
    vol = k.claims[k.claims["sd"] >= after].groupby("provider_sk").size()
    cand = set(vol[(vol >= 4) & (vol <= 12)].index)
    provs = k.providers(2, lambda f: (f["provider_type"] == "CLINIC") & f["provider_sk"].isin(cand))
    for p in provs:
        psp = k.w.tables["provider_status_period"]
        mask = (psp["provider_sk"] == p) & (psp["status_type"] == "NETWORK")
        _set_rows(k.w, "provider_status_period", mask, valid_to=D(2025, 9, 30))
        k.w.append("provider_status_period", [{
            "provider_sk": p, "status_type": "NETWORK", "status_value": "OUT_OF_NETWORK", "valid_from": D(2025, 10, 1),
            "valid_to": D(2026, 12, 31), "source": "Synthetic payer network file", "match_type": "EXACT",
            "version_id": "SYN-NET-2025.4"}])
        ids = k.claims[(k.claims["provider_sk"] == p) & (k.claims["sd"] >= after)]["claim_sk"].tolist()
        k.w.used_claims.update(ids)
        k.w.record("ENT-01 clinic billing after leaving the payer network", rule_ids=["ENT-01-R04"], claim_ids=ids,
                   subject_type="provider", subject_ids=[p], note="Network contract ended 30 Sep 2025.")


# ---------------------------------------------------------------------------
# ENT-02
# ---------------------------------------------------------------------------


@_plant("identity_conflict")
def _ent_02_r01(k: _Kit) -> None:
    active = set(k.claims["member_sk"])
    mem = k.members(8, lambda f: f["member_sk"].isin(active) & (pd.to_datetime(f["date_of_birth"]) < pd.Timestamp("2005-01-01")))
    m = k.w.tables["member"]
    for a, b in zip(mem[::2], mem[1::2]):
        tok = m.loc[m["member_sk"] == a, "protected_id_token"].iloc[0]
        _set_rows(k.w, "member", m["member_sk"] == b, protected_id_token=tok)
        ids = k.claims[k.claims["member_sk"] == b]["claim_sk"].tolist()
        k.w.record("ENT-02 one Emirates-ID reference on two records with different dates of birth",
                   rule_ids=["ENT-02-R01"], claim_ids=ids[:10], subject_type="member", subject_ids=[a, b])


@_plant("post_mortem")
def _ent_02_r02(k: _Kit) -> None:
    c = k.claims.dropna(subset=["sd"])
    c = c[c["sd"] >= pd.Timestamp("2024-10-01")]
    cnt = c.groupby("member_sk")["sd"].agg(["count", "min", "max"])
    cand = set(cnt[(cnt["count"] >= 4) & ((cnt["max"] - cnt["min"]).dt.days >= 60)].index)
    mem = k.members(6, lambda f: f["member_sk"].isin(cand))
    m = k.w.tables["member"]
    for i, msk in enumerate(mem):
        dates = sorted(c[c["member_sk"] == msk]["sd"])
        death = (dates[-2] - pd.Timedelta(days=10)).date()
        if death <= dates[0].date():
            death = dates[0].date() + TD(days=1)
        weak = i == len(mem) - 1
        _set_rows(k.w, "member", m["member_sk"] == msk, death_date=death,
                  death_source="Unverified report by a relative" if weak else "MOHAP civil death register",
                  death_source_confidence=0.4 if weak else 0.97)
        after = c[(c["member_sk"] == msk) & (c["sd"] > pd.Timestamp(death) + pd.Timedelta(days=2))]["claim_sk"].tolist()
        k.w.used_claims.update(after)
        if weak:
            k.w.record("ENT-02 death reported by an unreliable source (exclusion look-alike)", rule_ids=["ENT-02-R02"],
                       claim_ids=after, subject_type="member", subject_ids=[msk], positive=False,
                       note="Death-source confidence 0.4; must not fire.")
        else:
            k.w.record("ENT-02 services after the member's recorded death", rule_ids=["ENT-02-R02"], claim_ids=after,
                       subject_type="member", subject_ids=[msk], note=f"Death recorded {death}.")


@_plant("card_sharing")
def _ent_02_r04(k: _Kit) -> None:
    ctx = k.ctx
    clinics_by_em: dict[str, list[str]] = {}
    for p, info in ctx.providers.items():
        if info["type"] == "CLINIC" and info["kind"] and info["kind"][0] in ("GP", "POLY"):
            clinics_by_em.setdefault(info["emirate"], []).append(p)
    base = k.claims[(k.claims["claim_type"] == "OUTPATIENT") & (k.claims["ptype"] == "CLINIC") & k.free_claims()]
    per = base.groupby("member_sk").size()
    cand = set(per[per >= 3].index)
    mem = k.members(6, lambda f: f["member_sk"].isin(cand))
    for i, m in enumerate(mem):
        look = i == len(mem) - 1
        home_em = ctx.members[m]["emirate"]
        if look:
            far = [e for e in clinics_by_em if e != home_em and frozenset((e, home_em)) in _ADJACENT]
        else:
            far = [e for e in clinics_by_em if e != home_em and frozenset((e, home_em)) not in _ADJACENT]
        if not far:
            continue
        own = base[base["member_sk"] == m].sort_values("sd")
        made = []
        for _, row in own.head(2).iterrows():
            em = far[int(k.rng.integers(len(far)))]
            p = clinics_by_em[em][int(k.rng.integers(len(clinics_by_em[em])))]
            if k.prov.loc[row["provider_sk"], "emirate"] == em:
                continue
            c = k.claim(member_sk=m, provider_sk=p, service_date=row["sd"].date(), claim_type="OUTPATIENT",
                        start_minute=int(k.rng.integers(15 * 60, 19 * 60)))
            if c:
                made += [row["claim_sk"], c]
                k.w.used_claims.add(row["claim_sk"])
        if look:
            k.w.record("ENT-02 same-day visits in neighbouring emirates (look-alike)", rule_ids=["ENT-02-R04"],
                       claim_ids=made, subject_type="member", subject_ids=[m], positive=False,
                       note="Neighbouring emirates; must not count as distant use.")
        else:
            k.w.record("ENT-02 one member's cover used in two distant emirates on the same days",
                       rule_ids=["ENT-02-R04"], claim_ids=made, subject_type="member", subject_ids=[m])


# ---------------------------------------------------------------------------
# ENT-03
# ---------------------------------------------------------------------------


@_plant("inactive_licence")
def _ent_03_r01(k: _Kit) -> None:
    ctx = k.ctx
    cut = pd.Timestamp("2025-10-01")
    vol = k.claims[k.claims["sd"] >= cut + pd.Timedelta(days=15)].groupby("provider_sk").size()
    cand = set(vol[(vol >= 3) & (vol <= 10)].index)
    provs = k.providers(2, lambda f: (f["provider_type"] == "CLINIC") & f["provider_sk"].isin(cand))
    for p in provs:
        psp = k.w.tables["provider_status_period"]
        mask = (psp["provider_sk"] == p) & (psp["status_type"] == "LICENCE")
        _set_rows(k.w, "provider_status_period", mask, valid_to=D(2025, 9, 30))
        k.w.append("provider_status_period", [{
            "provider_sk": p, "status_type": "LICENCE", "status_value": "EXPIRED", "valid_from": D(2025, 10, 1),
            "valid_to": D(2026, 12, 31), "source": "Synthetic facility licence register", "match_type": "EXACT",
            "version_id": "SYN-REG-2025.4"}])
        ids = k.claims[(k.claims["provider_sk"] == p) & (k.claims["sd"] > cut + pd.Timedelta(days=14))]["claim_sk"].tolist()
        k.w.used_claims.update(ids)
        k.w.record("ENT-03 clinic billing after its facility licence expired", rule_ids=["ENT-03-R01"], claim_ids=ids,
                   subject_type="provider", subject_ids=[p], note="Licence expired 30 Sep 2025.")
    # clinicians
    lines = k.w.tables["claim_line"]
    ld = pd.to_datetime(lines["service_date"], errors="coerce")
    late = lines[ld >= pd.Timestamp("2025-06-01")].groupby("rendering_clinician_id")["claim_sk"].nunique()
    cand = set(late[(late >= 5) & (late <= 25)].index)
    clins = k.clinicians(4, lambda f: f["clinician_id"].isin(cand) & ~f["provider_sk"].isin(set(provs)))
    ros = k.w.tables["clinician_roster"]
    for i, cid in enumerate(clins):
        mine = lines[lines["rendering_clinician_id"] == cid]
        dates = sorted(pd.to_datetime(mine["service_date"]).dt.date.unique())
        if len(dates) < 4:
            continue
        grace_case = i == len(clins) - 1
        if grace_case:
            expiry = dates[-1] - TD(days=5)   # inside the 14-day grace: look-alike
        else:
            expiry = dates[-4] - TD(days=20)
        _set_rows(k.w, "clinician_roster", ros["clinician_id"] == cid, licence_valid_to=expiry)
        ctx.clin_by_id[cid]["lic_to"] = expiry
        after = mine[pd.to_datetime(mine["service_date"]).dt.date > expiry + TD(days=14)]["claim_sk"].unique().tolist()
        after_any = mine[pd.to_datetime(mine["service_date"]).dt.date > expiry]["claim_sk"].unique().tolist()
        if grace_case:
            k.w.record("ENT-03 service inside the licence grace period (exclusion look-alike)", rule_ids=["ENT-03-R01"],
                       claim_ids=after_any, subject_type="clinician", subject_ids=[cid], positive=False,
                       note="Within the 14-day approved grace; must not fire.")
        else:
            k.w.used_claims.update(after)
            k.w.record("ENT-03 clinician billing after their licence expired", rule_ids=["ENT-03-R01"],
                       claim_ids=after, subject_type="clinician", subject_ids=[cid], note=f"Licence ended {expiry}.")


def _gp_clinicians(k: _Kit, n: int) -> list[str]:
    ctx = k.ctx
    ok = {c for c, info in ctx.clin_by_id.items() if info["specialty"] == "GENERAL_PRACTICE"
          and ctx.providers[info["provider"]]["type"] == "CLINIC"}
    return k.clinicians(n, lambda f: f["clinician_id"].isin(ok))


def _patients_of(k: _Kit, provider: str, n: int) -> list[str]:
    seen = k.claims[k.claims["provider_sk"] == provider]["member_sk"].dropna().unique().tolist()
    k.rng.shuffle(seen)
    return seen[:n]


@_plant("hard_privilege")
def _ent_03_r03(k: _Kit) -> None:
    ros = k.w.tables["clinician_roster"]
    for cid in _gp_clinicians(k, 3):
        prov = k.clinic_of(cid)
        mask = ros["clinician_id"] == cid
        priv = str(ros.loc[mask, "privileges"].iloc[0])
        _set_rows(k.w, "clinician_roster", mask, privileges=priv + ";NOT:PROC_ORTHO")
        made = []
        for m in _patients_of(k, prov, 2):
            sd = k.date_in(D(2025, 2, 1), D(2025, 11, 30))
            if not k.member_on_cover(m, sd):
                continue
            c = k.claim(member_sk=m, provider_sk=prov, service_date=sd, claim_type="OUTPATIENT", clinician_id=cid,
                        lines=[{"code": "20610", "clinician": cid}], diagnoses=["M17.11"] if "M17.11" in k.ctx.icd else None)
            if c:
                made.append(c)
        k.w.record("ENT-03 GP whose privileges exclude joint injections billing one", rule_ids=["ENT-03-R03"],
                   claim_ids=made, subject_type="clinician", subject_ids=[cid],
                   note="Privilege list carries NOT:PROC_ORTHO.")


@_plant("soft_specialty")
def _ent_03_r04(k: _Kit) -> None:
    for cid in _gp_clinicians(k, 3):
        prov = k.clinic_of(cid)
        made = []
        for m in _patients_of(k, prov, 2):
            sd = k.date_in(D(2024, 9, 1), D(2025, 11, 30))
            if not k.member_on_cover(m, sd):
                continue
            c = k.claim(member_sk=m, provider_sk=prov, service_date=sd, claim_type="OUTPATIENT", clinician_id=cid,
                        lines=[{"code": "99244", "clinician": cid}])
            if c:
                made.append(c)
        k.w.record("ENT-03 GP billing specialist consultation codes", rule_ids=["ENT-03-R04"], claim_ids=made,
                   subject_type="clinician", subject_ids=[cid])


def _new_provider(k: _Kit, template: str, *, cred: D, facility: str | None = None, n_clin: int = 2,
                  specialty: str = "GENERAL_PRACTICE", ownership_changed_on: D | None = None) -> tuple[str, list[str]]:
    """A new provider (with roster, licence, network and contracts) cloned from an existing one."""
    w, ctx = k.w, k.ctx
    sk = w.new_id("PRV", 4)
    prov = w.tables["provider"]
    row = prov[prov["provider_sk"] == template].iloc[0].to_dict()
    tok = lambda p: f"{p}{int(k.rng.integers(10 ** 11, 10 ** 12))}"  # noqa: E731
    row.update(provider_sk=sk, source_provider_id=f"SRC-{sk}", credentialing_date=cred,
               owner_entity_id=tok("OWN-"), bank_account_token=tok("IBANTOK-"), phone_token=tok("TELTOK-"),
               address_token=tok("ADDRTOK-"), ownership_changed_on=ownership_changed_on,
               licence_no=f"SYN-LIC-{cred.year}-{int(k.rng.integers(10000, 99999))}")
    if facility:
        row["facility_type"] = facility
    w.append("provider", [row])
    w.used_entities.add(sk)
    w.append("provider_status_period", [
        {"provider_sk": sk, "status_type": "LICENCE", "status_value": "ACTIVE", "valid_from": cred,
         "valid_to": D(2027, 12, 31), "source": "Synthetic facility licence register", "match_type": "EXACT",
         "version_id": "SYN-REG-2025.2"},
        {"provider_sk": sk, "status_type": "NETWORK", "status_value": "IN_NETWORK", "valid_from": cred,
         "valid_to": D(2026, 12, 31), "source": "Synthetic payer network file", "match_type": "EXACT",
         "version_id": "SYN-NET-2025.2"}])
    info = copy.deepcopy(ctx.providers[template])
    info.update(sk=sk, specialties=set(), equipment={})
    ctx.providers[sk] = info
    contracts = []
    for (p, payer), c in list(ctx.contracts.items()):
        if p == template:
            ctx.contracts[(sk, payer)] = dict(c)
            contracts.append({"contract_sk": f"CTR-{sk}-{payer[-1]}", "provider_sk": sk, "payer_id": payer,
                              "valid_from": cred, "valid_to": D(2026, 12, 31), "tariff_basis": "SYN-TARIFF-2024/2025",
                              "discount_pct": c["discount"], "network_tier": c["tier"], "tenant_id": TENANT})
    w.append("contract", contracts)
    src = next((c for c in ctx.clinicians[template] if c["specialty"] == specialty), ctx.clinicians[template][0])
    ros = w.tables["clinician_roster"]
    src_row = ros[ros["clinician_id"] == src["id"]].iloc[0].to_dict()
    clins = []
    for _ in range(n_clin):
        cid = w.new_id("CL", 5)
        clin = copy.deepcopy(src)
        clin.update(id=cid, provider=sk, leave=[], lic_from=cred - TD(days=400))
        ctx.clinicians[sk].append(clin)
        ctx.clin_by_id[cid] = clin
        info["specialties"].add(clin["specialty"])
        r = dict(src_row)
        r.update(clinician_id=cid, provider_sk=sk, full_name_token=tok("NAMETOK-"),
                 licence_valid_from=cred - TD(days=400), leave_periods="[]")
        w.append("clinician_roster", [r])
        w.used_entities.add(cid)
        clins.append(cid)
    return sk, clins


@_plant("new_provider_burst")
def _ent_03_r05(k: _Kit) -> None:
    ctx = k.ctx
    gp = [p for p, info in ctx.providers.items() if info["type"] == "CLINIC" and info["kind"]
          and info["kind"][0] == "GP" and p not in k.w.used_entities]
    for variant in ("burst", "acquired"):
        template = gp[int(k.rng.integers(len(gp)))]
        cred = D(2025, 6, 15) if variant == "burst" else D(2025, 3, 1)
        sk, clins = _new_provider(k, template, cred=cred, facility="MEDICAL_CENTRE",
                                  ownership_changed_on=(cred + TD(days=10)) if variant == "acquired" else None)
        pats = _patients_of(k, template, 60)
        made = []
        for i in range(90):
            if len(made) >= 48:
                break
            m = pats[i % len(pats)]
            sd = cred + TD(days=int(k.rng.integers(1, 88)))
            if not k.member_on_cover(m, sd):
                continue
            c = k.claim(member_sk=m, provider_sk=sk, service_date=sd, claim_type="OUTPATIENT",
                        clinician_id=clins[i % len(clins)], start_minute=int(k.rng.integers(8 * 60, 20 * 60)))
            if c:
                made.append(c)
        if variant == "burst":
            k.w.record("ENT-03 newly credentialed clinic billing far above peers at once", rule_ids=["ENT-03-R05"],
                       claim_ids=made, subject_type="provider", subject_ids=[sk])
        else:
            k.w.record("ENT-03 burst right after an acquisition (exclusion look-alike)", rule_ids=["ENT-03-R05"],
                       claim_ids=made, subject_type="provider", subject_ids=[sk], positive=False,
                       note="Ownership changed at the start; must not fire.")


# ---------------------------------------------------------------------------
# ENT-04
# ---------------------------------------------------------------------------


def _em_line_mask(k: _Kit, claim_ids: list[str]) -> pd.Series:
    lines = k.w.tables["claim_line"]
    codes = lines["activity_code"].astype(str)
    em = codes.str.fullmatch(r"992\d\d")
    return lines["claim_sk"].isin(set(claim_ids)) & em


@_plant("missing_clinician")
def _ent_04_r01(k: _Kit) -> None:
    pool = k.claims[(k.claims["claim_type"] == "OUTPATIENT") & (k.claims["ptype"] == "CLINIC")]
    lines = k.w.tables["claim_line"]
    has_em = set(lines.loc[lines["activity_code"].astype(str).str.fullmatch(r"992\d\d"), "claim_sk"])
    taken = k.w.take_claims(6, lambda h: h["claim_sk"].isin(set(pool["claim_sk"]) & has_em))
    for i, c in enumerate(taken["claim_sk"]):
        mask = _em_line_mask(k, [c])
        if i < 3:
            val, what = None, "no treating clinician"
        elif i < 5:
            val, what = taken.iloc[i]["provider_sk"], "the facility's own code as treating clinician"
        else:
            val, what = "CL99999", "a clinician ID on no roster"
        _set_rows(k.w, "claim_line", mask, rendering_clinician_id=val)
        k.w.record("ENT-04 consultation line with " + what, rule_ids=["ENT-04-R01"], claim_ids=[c],
                   subject_type="claim")


@_plant("role_conflict")
def _ent_04_r02(k: _Kit) -> None:
    ctx = k.ctx
    pool = k.claims[(k.claims["claim_type"] == "OUTPATIENT") & (k.claims["ptype"] == "CLINIC")]
    lines = k.w.tables["claim_line"]
    has_em = set(lines.loc[lines["activity_code"].astype(str).str.fullmatch(r"992\d\d"), "claim_sk"])
    taken = k.w.take_claims(5, lambda h: h["claim_sk"].isin(set(pool["claim_sk"]) & has_em))
    gps = [c for c, info in ctx.clin_by_id.items() if info["specialty"] == "GENERAL_PRACTICE"]
    for _, row in taken.iterrows():
        home = {c["id"] for c in ctx.clinicians.get(row["provider_sk"], [])}
        other = [c for c in gps if c not in home and ctx.providers[ctx.clin_by_id[c]["provider"]]["emirate"]
                 == k.prov.loc[row["provider_sk"], "emirate"]] or [c for c in gps if c not in home]
        cid = other[int(k.rng.integers(len(other)))]
        mask = _em_line_mask(k, [row["claim_sk"]])
        _set_rows(k.w, "claim_line", mask, rendering_clinician_id=cid, ordering_clinician_id=None)
        k.w.record("ENT-04 clinician from another facility billed as the treating clinician",
                   rule_ids=["ENT-04-R02"], claim_ids=[row["claim_sk"]], subject_type="claim",
                   subject_ids=[cid])


@_plant("concurrent_sessions")
def _ent_04_r03(k: _Kit) -> None:
    ctx = k.ctx
    phys = {c for c, info in ctx.clin_by_id.items() if info["specialty"] == "PHYSIOTHERAPY"
            and ctx.providers[info["provider"]]["type"] == "CLINIC"}
    for cid in k.clinicians(2, lambda f: f["clinician_id"].isin(phys)):
        prov = k.clinic_of(cid)
        pats = _patients_of(k, prov, 40)
        if len(pats) < 3:
            em = k.ctx.providers[prov]["emirate"]
            pats += [m for m, info in k.ctx.members.items() if info["emirate"] == em][:40]
        day = k.date_in(D(2025, 3, 1), D(2025, 11, 20))
        made = []
        j = -1
        for m in pats:
            if len(made) >= 3:
                break
            if not k.member_on_cover(m, day):
                continue
            j += 1
            start = DT.combine(day, _dt.time(10, 0)) + TD(minutes=10 * j)
            c = k.claim(member_sk=m, provider_sk=prov, service_date=day, claim_type="OUTPATIENT", clinician_id=cid,
                        lines=[{"code": "97110", "units": 3, "clinician": cid, "start": start, "minutes": 45}],
                        start_minute=600 + 10 * j)
            if c:
                made.append(c)
        k.w.record("ENT-04 physiotherapist billed for three overlapping timed sessions", rule_ids=["ENT-04-R03"],
                   claim_ids=made, subject_type="clinician", subject_ids=[cid])


@_plant("leave_travel")
def _ent_04_r04(k: _Kit) -> None:
    ctx = k.ctx
    ros = k.w.tables["clinician_roster"]
    gp_ok = {c for c, info in ctx.clin_by_id.items() if info["specialty"] in ("GENERAL_PRACTICE", "FAMILY_MEDICINE")
             and ctx.providers[info["provider"]]["type"] == "CLINIC"}
    for cid in k.clinicians(3, lambda f: f["clinician_id"].isin(gp_ok)):
        periods = json.loads(ros.loc[ros["clinician_id"] == cid, "leave_periods"].iloc[0] or "[]")
        periods = [p for p in periods if p.get("type") == "ANNUAL" and D.fromisoformat(p["from"]) >= D(2024, 8, 1)]
        if not periods:
            continue
        lv = periods[0]
        a, b = D.fromisoformat(lv["from"]), D.fromisoformat(lv["to"])
        prov = k.clinic_of(cid)
        made = []
        for m in _patients_of(k, prov, 3):
            sd = k.date_in(a, b)
            if not k.member_on_cover(m, sd):
                continue
            c = k.claim(member_sk=m, provider_sk=prov, service_date=sd, claim_type="OUTPATIENT", clinician_id=cid,
                        start_minute=int(k.rng.integers(9 * 60, 16 * 60)))
            if c:
                made.append(c)
        k.w.record("ENT-04 clinician billed while on recorded annual leave", rule_ids=["ENT-04-R04"], claim_ids=made,
                   subject_type="clinician", subject_ids=[cid], note=f"Leave {a} to {b}.")
    # travel: one clinician billed in two emirates twenty minutes apart
    for cid in k.clinicians(1, lambda f: f["clinician_id"].isin(gp_ok - k.w.used_entities)):
        prov = k.clinic_of(cid)
        em = ctx.providers[prov]["emirate"]
        far = [p for p, info in ctx.providers.items() if info["type"] == "CLINIC" and info["emirate"] != em
               and frozenset((em, info["emirate"])) not in _ADJACENT]
        if not far:
            continue
        other = far[int(k.rng.integers(len(far)))]
        pats = _patients_of(k, prov, 2) + _patients_of(k, other, 2)
        day = k.date_in(D(2025, 2, 1), D(2025, 11, 20))
        s1 = DT.combine(day, _dt.time(10, 0))
        made = []
        c1 = k.claim(member_sk=pats[0], provider_sk=prov, service_date=day, claim_type="OUTPATIENT", clinician_id=cid,
                     lines=[{"code": "99213", "clinician": cid, "start": s1, "minutes": 20}], start_minute=600)
        c2 = k.claim(member_sk=pats[-1], provider_sk=other, service_date=day, claim_type="OUTPATIENT", clinician_id=cid,
                     lines=[{"code": "99213", "clinician": cid, "start": s1 + TD(minutes=40), "minutes": 20}],
                     start_minute=640)
        made = [c for c in (c1, c2) if c]
        k.w.record("ENT-04 clinician billed in two distant emirates twenty minutes apart", rule_ids=["ENT-04-R04"],
                   claim_ids=made, subject_type="clinician", subject_ids=[cid])


# ---------------------------------------------------------------------------
# ENT-05
# ---------------------------------------------------------------------------


@_plant("facility_type")
def _ent_05_r01(k: _Kit) -> None:
    ctx = k.ctx
    for ptype, code, n in (("PHARMACY", "99213", 3), ("RADIOLOGY_CENTRE", "80053", 2)):
        provs = k.providers(n, lambda f, t=ptype: f["provider_type"] == t)
        for p in provs:
            clin = ctx.clinicians[p][0]["id"] if ctx.clinicians.get(p) else None
            em = ctx.providers[p]["emirate"]
            pool = [m for m, info in ctx.members.items() if info["emirate"] == em and m not in k.w.used_entities]
            m = pool[int(k.rng.integers(len(pool)))]
            sd = k.date_in(D(2024, 9, 1), D(2025, 11, 30))
            if not k.member_on_cover(m, sd):
                sd = D(2025, 3, 3)
            c = k.claim(member_sk=m, provider_sk=p, service_date=sd, claim_type="OUTPATIENT",
                        lines=[{"code": code, "clinician": clin}])
            if c:
                k.w.record(f"ENT-05 service billed by a {ptype.lower().replace('_', ' ')} not licensed for it",
                           rule_ids=["ENT-05-R01"], claim_ids=[c], subject_type="claim", subject_ids=[p])


@_plant("admission_evidence")
def _ent_05_r02(k: _Kit) -> None:
    taken = k.w.take_claims(7, lambda h: (h["claim_type"] == "INPATIENT"), claim_type="INPATIENT")
    taken = taken[taken["claim_sk"].isin(set(k.claims[k.claims["sd"] <= pd.Timestamp("2025-12-01")]["claim_sk"]))]
    enc = k.w.tables["encounter"]
    for i, c in enumerate(taken["claim_sk"]):
        mask = enc["claim_sk"] == c
        if i == 0:
            _set_rows(k.w, "encounter", mask, bed_id=None, discharge_type="TRANSFER")
            k.w.record("ENT-05 stay without a bed record, patient transferred (exclusion look-alike)",
                       rule_ids=["ENT-05-R02"], claim_ids=[c], subject_type="claim", positive=False,
                       note="Transferred case; must not fire.")
        else:
            _set_rows(k.w, "encounter", mask, bed_id=None)
            k.w.record("ENT-05 inpatient stay billed with no bed assigned", rule_ids=["ENT-05-R02"], claim_ids=[c],
                       subject_type="claim")


@_plant("setting_conflict")
def _ent_05_r03(k: _Kit) -> None:
    enc = k.w.tables["encounter"]
    e = enc[enc["encounter_type"] == "INPATIENT"].copy()
    e["adm"] = pd.to_datetime(e["admission_date"])
    e["dis"] = pd.to_datetime(e["discharge_date"])
    e = e[(e["dis"] - e["adm"]).dt.days >= 3]
    e = e[~(e["admission_type"].astype(str).str.upper().str.contains("TRANSFER")
            | e["discharge_type"].astype(str).str.upper().str.contains("TRANSFER"))]
    taken = k.w.take_claims(5, lambda h: h["claim_sk"].isin(set(e["claim_sk"])))
    for _, row in taken.iterrows():
        er = e[e["claim_sk"] == row["claim_sk"]].iloc[0]
        sd = (er["adm"] + pd.Timedelta(days=1)).date()
        c = k.claim(member_sk=row["member_sk"], provider_sk=row["provider_sk"], service_date=sd, claim_type="OUTPATIENT",
                    start_minute=int(k.rng.integers(10 * 60, 15 * 60)))
        if c:
            k.w.record("ENT-05 outpatient visit billed at the same hospital during the stay", rule_ids=["ENT-05-R03"],
                       claim_ids=[row["claim_sk"], c], subject_type="claim")


@_plant("setting_shift")
def _ent_05_r04(k: _Kit) -> None:
    enc = k.w.tables["encounter"]
    h = k.claims.merge(enc[["claim_sk", "encounter_type"]].drop_duplicates("claim_sk"), on="claim_sk", how="left")
    hosp = h[h["ptype"] == "HOSPITAL"]
    share = hosp.groupby("provider_sk").agg(n=("claim_sk", "size"),
                                            ip=("encounter_type", lambda s: s.isin(["INPATIENT", "DAY_CASE"]).mean()))
    cand = set(share[(share["n"] >= 120) & (share["ip"] <= 0.35)].index)
    provs = k.providers(1, lambda f: f["provider_sk"].isin(cand))
    for p in provs:
        split = pd.Timestamp("2025-08-01")
        sub = h[(h["provider_sk"] == p) & (h["sd"] >= split) & (h["encounter_type"] == "OUTPATIENT")]
        conv = sub.sample(frac=0.85, random_state=int(k.rng.integers(1 << 31)))["claim_sk"].tolist()
        mask = enc["claim_sk"].isin(set(conv))
        sds = enc.loc[mask, "start_time"]
        _set_rows(k.w, "encounter", mask, encounter_type="DAY_CASE", observation_status="DAY_CASE_BAY",
                  admission_type="ELECTIVE", discharge_type="HOME")
        enc = k.w.tables["encounter"]
        for idx in enc.index[mask]:
            day = _d(enc.at[idx, "start_time"])
            enc.at[idx, "admission_date"] = day
            enc.at[idx, "discharge_date"] = day
            enc.at[idx, "bed_id"] = f"{p}-DC{int(k.rng.integers(1, 12)):02d}"
        k.w.used_claims.update(conv)
        k.w.record("ENT-05 hospital recording ordinary visits as day cases from August 2025",
                   rule_ids=["ENT-05-R04"], claim_ids=conv, subject_type="provider", subject_ids=[p])


# ---------------------------------------------------------------------------
# ENT-06
# ---------------------------------------------------------------------------


def _gp_clinic_members(k: _Kit, n: int) -> list[tuple[str, str, str]]:
    """(clinician, clinic, member) triples for GP telehealth plants."""
    out = []
    for cid in _gp_clinicians(k, n):
        prov = k.clinic_of(cid)
        pats = _patients_of(k, prov, 1)
        if pats:
            out.append((cid, prov, pats[0]))
    return out


@_plant("telehealth_eligibility")
def _ent_06_r01(k: _Kit) -> None:
    for cid, prov, m in _gp_clinic_members(k, 6):
        sd = k.date_in(D(2024, 9, 1), D(2025, 11, 30))
        if not k.member_on_cover(m, sd):
            continue
        c = k.claim(member_sk=m, provider_sk=prov, service_date=sd, claim_type="OUTPATIENT", clinician_id=cid,
                    encounter_type="TELEHEALTH", lines=[{"code": "99442", "clinician": cid},
                                                         {"code": "93000", "clinician": cid}])
        if c:
            k.w.record("ENT-06 telehealth visit billing an ECG", rule_ids=["ENT-06-R01"], claim_ids=[c],
                       subject_type="claim", subject_ids=[cid])


def _pharmacy_near(k: _Kit, emirate: str) -> list[str]:
    return [p for p, info in k.ctx.providers.items() if info["type"] == "PHARMACY" and info["emirate"] == emirate]


def _rx_claim(k: _Kit, member: str, pharmacy: str, prescriber: str, day: D, product: str = "RX1004") -> str | None:
    return k.claim(member_sk=member, provider_sk=pharmacy, service_date=day, claim_type="PHARMACY",
                   ordering_clinician_id=prescriber,
                   lines=[{"code": product, "rx": {"prescriber": prescriber, "prescribed_date": day}}])


@_plant("no_meaningful_interaction")
def _ent_06_r02(k: _Kit) -> None:
    triples = _gp_clinic_members(k, 7)
    for i, (cid, prov, m) in enumerate(triples):
        sd = k.date_in(D(2024, 9, 1), D(2025, 11, 30))
        if not k.member_on_cover(m, sd):
            continue
        c = k.claim(member_sk=m, provider_sk=prov, service_date=sd, claim_type="OUTPATIENT", clinician_id=cid,
                    encounter_type="TELEHEALTH", lines=[{"code": "99441", "clinician": cid}], documents=False)
        if not c:
            continue
        enc = k.w.tables["encounter"]
        mask = enc["claim_sk"] == c
        start = pd.Timestamp(enc.loc[mask, "start_time"].iloc[0])
        mins = float(k.rng.integers(2, 5))
        _set_rows(k.w, "encounter", mask, duration_minutes=mins, end_time=(start + pd.Timedelta(minutes=mins)).to_pydatetime())
        if i == len(triples) - 1:
            k.w.record("ENT-06 short telehealth call with no order (look-alike)", rule_ids=["ENT-06-R02"],
                       claim_ids=[c], subject_type="claim", positive=False, note="No order followed; must not fire.")
            continue
        ph = _pharmacy_near(k, k.ctx.providers[prov]["emirate"])
        rx = _rx_claim(k, m, ph[int(k.rng.integers(len(ph)))], cid, sd) if ph else None
        k.w.record("ENT-06 three-minute teleconsult ending in a prescription", rule_ids=["ENT-06-R02"],
                   claim_ids=[x for x in (c, rx) if x], subject_type="claim", subject_ids=[cid])


def _tele_clinic(k: _Kit, emirate_hint: str | None = None) -> tuple[str, list[str], list[str]]:
    ctx = k.ctx
    gp_all = [p for p, info in ctx.providers.items() if info["type"] == "CLINIC" and info["kind"]
              and info["kind"][0] in ("GP", "POLY")]
    gp = [p for p in gp_all if emirate_hint is None or ctx.providers[p]["emirate"] == emirate_hint] or gp_all
    template = gp[int(k.rng.integers(len(gp)))]
    sk, clins = _new_provider(k, template, cred=D(2023, 6, 1), facility="TELEHEALTH_PROVIDER", n_clin=2)
    em = ctx.providers[sk]["emirate"]
    pool = [m for m, info in ctx.members.items() if info["emirate"] == em and m not in k.w.used_entities]
    k.rng.shuffle(pool)
    return sk, clins, pool


@_plant("referral_concentration")
def _ent_06_r03(k: _Kit) -> None:
    sk, clins, pool = _tele_clinic(k, "Dubai")
    ph = _pharmacy_near(k, k.ctx.providers[sk]["emirate"])
    target = ph[int(k.rng.integers(len(ph)))]
    made = []
    i = 0
    for m in pool:
        if len(made) >= 36:
            break
        sd = k.date_in(D(2025, 1, 10), D(2025, 11, 30))
        if not k.member_on_cover(m, sd):
            continue
        clin = clins[i % 2]
        i += 1
        c = k.claim(member_sk=m, provider_sk=sk, service_date=sd, claim_type="OUTPATIENT", clinician_id=clin,
                    encounter_type="TELEHEALTH", lines=[{"code": "99442", "clinician": clin}])
        if not c:
            continue
        pharm = target if i % 30 else ph[int(k.rng.integers(len(ph)))]
        rx = _rx_claim(k, m, pharm, clin, sd)
        made += [x for x in (c, rx) if x]
    k.w.record("ENT-06 telehealth clinic sending nearly every prescription to one pharmacy", rule_ids=["ENT-06-R03"],
               claim_ids=made, subject_type="provider", subject_ids=[sk, target])


@_plant("conversion_spike")
def _ent_06_r04(k: _Kit) -> None:
    ctx = k.ctx
    sk, clins, pool = _tele_clinic(k, "Abu Dhabi")
    rads = [p for p, info in ctx.providers.items() if info["type"] == "RADIOLOGY_CENTRE" and info["emirate"] == "Abu Dhabi"
            and "MRI" in info.get("equipment", {})] or \
           [p for p, info in ctx.providers.items() if info["type"] == "RADIOLOGY_CENTRE" and "MRI" in info.get("equipment", {})]
    made = []
    i = 0
    for m in pool:
        if i >= 34:
            break
        sd = k.date_in(D(2025, 1, 10), D(2025, 11, 25))
        if not k.member_on_cover(m, sd):
            continue
        clin = clins[i % 2]
        i += 1
        c = k.claim(member_sk=m, provider_sk=sk, service_date=sd, claim_type="OUTPATIENT", clinician_id=clin,
                    encounter_type="TELEHEALTH", lines=[{"code": "99442", "clinician": clin}])
        if not c:
            continue
        made.append(c)
        if i % 3 and rads:
            rad = rads[int(k.rng.integers(len(rads)))]
            r = k.claim(member_sk=m, provider_sk=rad, service_date=sd + TD(days=int(k.rng.integers(1, 3))),
                        claim_type="RADIOLOGY", lines=["72148"], ordering_clinician_id=clin, referral_from=clin)
            if r:
                made.append(r)
    k.w.record("ENT-06 telehealth clinic whose calls turn into MRI referrals far more often than peers'",
               rule_ids=["ENT-06-R04"], claim_ids=made, subject_type="provider", subject_ids=[sk])


# ---------------------------------------------------------------------------


PLANTS = [_ent_01_r01, _ent_01_r02, _ent_01_r03, _ent_01_r04, _ent_02_r01, _ent_02_r02, _ent_02_r04,
          _ent_03_r01, _ent_03_r03, _ent_03_r04, _ent_03_r05, _ent_04_r01, _ent_04_r02, _ent_04_r03,
          _ent_04_r04, _ent_05_r01, _ent_05_r02, _ent_05_r03, _ent_05_r04, _ent_06_r01, _ent_06_r02,
          _ent_06_r03, _ent_06_r04]


def inject(world: World) -> None:
    """Plant every ENT pattern (see the module docstring)."""
    k = _Kit(world)
    for plant in PLANTS:
        plant(k)
        k.refresh()
