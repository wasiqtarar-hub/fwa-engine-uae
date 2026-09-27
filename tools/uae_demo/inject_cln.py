"""Plant the CLN (clinical and coding) patterns in the SYNTHETIC UAE demo dataset.

Every control implemented in ``src/fwa/engine/unlocked/cln.py`` gets at least one
realistic pattern here, at a modest volume, and every planted item is written to
the answer key with the rule ids it is meant to exercise. Where a declared
exclusion has a legitimate look-alike, one or two look-alikes are planted too
(``positive=False``) so the exclusion is demonstrated.

Patterns (all SYNTHETIC):

* CLN-01-R04  a clinic whose visit levels jump to the top level from April 2025
              (+ a look-alike whose jump coincides with a change of ownership);
              the same clinic also stands out on CLN-01-R01 (top-level share);
* CLN-02-R01  inpatient stays lifted one case-rate band by a complication code the
              discharge summary does not record (+ documented look-alikes);
* CLN-02-R02  a hospital that records (and documents) complication codes on most
              of its stays;
* CLN-02-R03  a long-standing condition (heart failure / COPD …) appearing on a
              single claim of a patient with a long history that never shows it;
* CLN-02-R04  main diagnoses marked "not present on admission", codes listed twice
              (+ obstetric look-alikes, which are exempt);
* CLN-02-R05  a hospital whose recorded severity jumps from May 2025 with no change
              in length of stay or case mix;
* CLN-03-R02  ECGs billed for sore throats with no cardiac reason on record
              (+ chest-pain look-alikes);
* CLN-03-R03  knee MRIs with no knee X-ray first (+ referred-in look-alikes);
* CLN-03-R04  a general practice billing EEGs, which no other practice bills;
* CLN-04-R03  HbA1c repeated before the first result was recorded
              (+ look-alikes whose first result arrived a day late);
* CLN-04-R05  a clinic whose minor visits cascade into CT, echo and specialist
              consultations at one hospital;
* CLN-05-R04  a hospital admitting far more of its presentations than others;
* CLN-06-R01  imaging paid with no report or result on file;
* CLN-06-R04  members denying a service, and check-in logs showing no attendance
              (+ unverified and very late denials as look-alikes);
* CLN-06-R05  members billed repeatedly for diabetes with no medicine or result;
* CLN-07-R01  clinicians billing 17+ hours of visits in a day;
* CLN-07-R03  a radiology centre billing more MRI scans in a day than its scanner
              can perform;
* CLN-08-R01  laboratory tests with no (or an unknown) ordering clinician
              (+ screening look-alikes);
* CLN-08-R02  panel components billed one by one, or with the panel itself
              (+ a reflex-test look-alike);
* CLN-08-R03  paid tests with no result; identical result sets for unrelated patients;
* CLN-08-R04  a clinic re-billing a reference laboratory's tests at three times its
              price (+ clinics passing tests through at a normal handling margin);
* CLN-08-R05  a clinic ordering genetic tests on a large share of its patients.
"""

from __future__ import annotations

import datetime as _dt
import math
import re
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .world import World

D = _dt.date
TD = _dt.timedelta
TENANT = "T001"

GENETIC_CODES = [
    # code, description, price (AED)
    ("81162", "BRCA1/BRCA2 gene analysis, full sequence and deletion/duplication", 4200.0),
    ("81225", "CYP2C19 gene analysis, common variants", 900.0),
    ("81479", "Unlisted molecular pathology procedure (gene panel)", 2600.0),
]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _d(v: Any) -> D | None:
    if v is None or (isinstance(v, float) and v != v):
        return None
    if isinstance(v, _dt.datetime):
        return v.date()
    if isinstance(v, D):
        return v
    try:
        return pd.Timestamp(v).date()
    except Exception:
        return None


class _Kit:
    def __init__(self, world: World) -> None:
        self.w = world
        self.ctx = world.context["base_ctx"]
        self.rng = world.rng
        self.codes = world.context["codes"]
        self.grouper = world.context.get("drg_grouper", {})
        self.planted = 0
        self.refresh()

    # ------------------------------------------------------------- views

    def refresh(self) -> None:
        """Claim-level facts (first service date, provider, type) from the current tables."""
        lines = self.w.tables["claim_line"]
        first = lines.groupby("claim_sk")["service_date"].min()
        h = self.w.tables["claim_header"][["claim_sk", "member_sk", "provider_sk", "claim_type"]].copy()
        h["sd"] = h["claim_sk"].map(first).map(_d)
        self.info = h.set_index("claim_sk")

    def free(self, frame: pd.DataFrame) -> pd.DataFrame:
        return frame[~frame.index.isin(self.w.used_claims)]

    def take_ids(self, ids: Iterable[str]) -> list[str]:
        ids = [c for c in ids if c not in self.w.used_claims]
        self.w.used_claims.update(ids)
        return ids

    # ------------------------------------------------------------ entities

    def covered(self, member: str, a: D, b: D) -> bool:
        return bool(self.ctx.coverage_at(member, a)) and bool(self.ctx.coverage_at(member, b))

    def members(self, n: int, a: D = D(2024, 8, 1), b: D = D(2025, 11, 30), *, age=(22, 70),
                where=None) -> list[str]:
        ctx = self.ctx

        def ok(f: pd.DataFrame) -> pd.Series:
            def good(sk: str) -> bool:
                m = ctx.members.get(sk)
                if not m or m.get("cob"):
                    return False
                dob = m["dob"]
                years = (a - dob).days / 365.25
                if not (age[0] <= years <= age[1]):
                    return False
                if not self.covered(sk, a, b):
                    return False
                return where(m) if where else True
            return f["member_sk"].map(good)

        got = self.w.take_entities("member", "member_sk", n, where=ok)
        return got["member_sk"].tolist()

    def providers(self, n: int, ptype: str, specialty: str | None = None, *, among=None) -> list[str]:
        def ok(f: pd.DataFrame) -> pd.Series:
            m = f["provider_type"] == ptype
            if specialty:
                m &= f["specialty"] == specialty
            if among is not None:
                m &= f["provider_sk"].isin(set(among))
            return m & f["provider_sk"].map(lambda p: bool(self.ctx.clinicians.get(p)))
        return self.w.take_entities("provider", "provider_sk", n, where=ok)["provider_sk"].tolist()

    def any_provider(self, ptype: str, specialty: str | None = None) -> str | None:
        """A provider of a type WITHOUT reserving it (for ordinary new claims)."""
        p = self.w.tables["provider"]
        m = p["provider_type"] == ptype
        if specialty:
            m &= p["specialty"] == specialty
        pool = [x for x in p.loc[m, "provider_sk"] if self.ctx.clinicians.get(x)
                and str(x) not in self.w.used_entities]
        return None if not pool else pool[int(self.rng.integers(len(pool)))]

    def clinician(self, provider: str, family: str, on: D) -> str | None:
        return self.ctx.pick_clinician(provider, family, on)

    def weekday(self, a: D, b: D) -> D:
        while True:
            d = a + TD(days=int(self.rng.integers(0, max((b - a).days, 1))))
            if d.weekday() < 5:
                return d

    # --------------------------------------------------------------- claims

    def orderer(self, provider: str | None, claim_type: str, on: D) -> str | None:
        """A plausible ordering clinician: the practice's own doctor, else a GP elsewhere."""
        if claim_type == "OUTPATIENT" and provider:
            c = self.clinician(provider, "EM_OFFICE_EST", on)
            if c:
                return c
        for _ in range(10):
            gp = self.any_provider("CLINIC", "GENERAL_PRACTICE")
            c = self.clinician(gp, "EM_OFFICE_EST", on) if gp else None
            if c:
                return c
        return None

    def claim(self, **kw: Any) -> str | None:
        kw.setdefault("strict", False)
        if "ordering_clinician_id" not in kw and kw.get("claim_type") in ("LAB", "RADIOLOGY", "OUTPATIENT"):
            kw["ordering_clinician_id"] = self.orderer(kw.get("provider_sk"), kw["claim_type"], kw["service_date"])
        try:
            c = self.w.make_claim(**kw)
        except Exception:  # a claim the factory cannot build is simply not planted
            return None
        return c

    def lines(self, claim_sk: str) -> pd.DataFrame:
        return self.w.lines_of([claim_sk])

    # ---------------------------------------------------- consistent edits

    def price(self, code: str) -> float:
        c = self.codes.get(code)
        return float(c["unit_price_reference"]) if c else 0.0

    def recode_line(self, line_idx, new_code: str) -> None:
        """Change a line's code, scaling its amounts (and its payment) by the tariff ratio."""
        L = self.w.tables["claim_line"]
        old = L.at[line_idx, "activity_code"]
        ratio = self.price(new_code) / max(self.price(old), 1e-9)
        self.scale_line(line_idx, ratio)
        L.at[line_idx, "activity_code"] = new_code
        L.at[line_idx, "activity_description"] = self.ctx.description(new_code)
        auth = L.at[line_idx, "authorization_id"]
        if isinstance(auth, str) and auth:
            al = self.w.tables.get("authorization_line")
            if al is not None and not al.empty:
                m = (al["authorization_sk"] == auth) & (al["activity_code"] == old)
                if m.any():
                    al.loc[m, "activity_code"] = new_code
                    al.loc[m, "approved_value"] = (al.loc[m, "approved_value"].astype(float) * ratio).round(2)

    def scale_line(self, line_idx, ratio: float) -> None:
        L = self.w.tables["claim_line"]
        for col in ("gross_amount", "net_amount", "patient_share", "unit_price"):
            v = L.at[line_idx, col]
            if v is not None and not (isinstance(v, float) and math.isnan(v)):
                L.at[line_idx, col] = round(float(v) * ratio, 2)
        line_sk = L.at[line_idx, "line_sk"]
        rem = self.w.tables.get("remittance")
        if rem is not None and not rem.empty:
            m = rem["line_sk"] == line_sk
            if m.any():
                rem.loc[m, "payment_amount"] = (rem.loc[m, "payment_amount"].astype(float) * ratio).round(2)

    def set_band(self, claim_sk: str, band: str) -> bool:
        L = self.w.tables["claim_line"]
        m = (L["claim_sk"] == claim_sk) & L["activity_code"].astype(str).str.match(r"^CR\d+[ABC]$")
        if not m.any():
            return False
        idx = L.index[m][0]
        code = str(L.at[idx, "activity_code"])
        if code[-1] == band:
            return True
        self.recode_line(idx, code[:-1] + band)
        self.w.recompute_header_amounts([claim_sk])
        return True

    def band(self, claim_sk: str) -> str | None:
        L = self.lines(claim_sk)
        cr = L[L["activity_code"].astype(str).str.match(r"^CR\d+[ABC]$")]
        return None if cr.empty else str(cr["activity_code"].iloc[0])[-1]

    def add_dx(self, claim_sk: str, code: str, *, poa: str = "Y") -> None:
        dx = self.w.tables["diagnosis"]
        rows = dx[dx["claim_sk"] == claim_sk]
        template = rows.iloc[0].to_dict() if not rows.empty else {"claim_sk": claim_sk, "tenant_id": TENANT}
        desc, chapter = self.w.context["icd"].get(code, (code, ""))
        template.update(code=code, diagnosis_type="SECONDARY", sequence=int(rows["sequence"].max()) + 1
                        if not rows.empty else 2, present_on_admission=poa, description=desc, chapter=chapter)
        self.w.append("diagnosis", [template])

    def doc_secondary(self, claim_sk: str, add: str | None = None, drop: str | None = None) -> None:
        """Edit the 'SECONDARY DIAGNOSES:' line of the claim's discharge summaries."""
        docs = self.w.tables.get("document")
        if docs is None or docs.empty:
            return
        m = (docs["claim_sk"] == claim_sk) & (docs["doc_type"] == "DISCHARGE_SUMMARY")
        for idx in docs.index[m]:
            text = str(docs.at[idx, "text"])
            lines = text.split("\n")
            for i, ln in enumerate(lines):
                if ln.startswith("SECONDARY DIAGNOSES:"):
                    items = [x.strip() for x in ln.split(":", 1)[1].split(";") if x.strip()]
                    items = [x for x in items if not x.lower().startswith("none")]
                    if drop:
                        items = [x for x in items if not x.startswith(drop)]
                    if add:
                        desc = self.w.context["icd"].get(add, (add, ""))[0]
                        items.append(f"{add} {desc}")
                    lines[i] = "SECONDARY DIAGNOSES: " + ("; ".join(items) if items else "None recorded")
            docs.at[idx, "text"] = "\n".join(lines)

    def drop_rows(self, table: str, mask) -> None:
        frame = self.w.tables.get(table)
        if frame is None or frame.empty:
            return
        self.w.tables[table] = frame[~mask].reset_index(drop=True)

    def record(self, stratum: str, rule_ids, claims, *, subject_type="claim", subjects=(), positive=True,
               note="") -> None:
        claims = [c for c in claims if c]
        if not claims and not subjects:
            return
        self.w.record(f"CLN {stratum}", rule_ids=rule_ids, claim_ids=claims, subject_type=subject_type,
                      subject_ids=list(subjects) or claims, positive=positive, note=note)


def _cc_for(kit: _Kit, principal: str, *, mcc: bool = False) -> str:
    """An acute complication code that fits the principal diagnosis chapter."""
    chapter = kit.w.context["icd"].get(principal, ("", ""))[1]
    from .reference import ACUTE_SECONDARY
    options = [c for c in ACUTE_SECONDARY.get(chapter, {}) if kit.grouper.get(c, (False, False))[1 if mcc else 0]]
    if not options:
        options = ["N17.9", "J96.00"] if mcc else ["E87.1", "E87.6", "D62"]
    return options[int(kit.rng.integers(len(options)))]


# ---------------------------------------------------------------------------
# CLN-01 / CLN-02 — coding
# ---------------------------------------------------------------------------


def _em_lines(kit: _Kit) -> pd.DataFrame:
    L = kit.w.tables["claim_line"]
    em = L[L["activity_code"].astype(str).isin(["99202", "99203", "99204", "99212", "99213", "99214"])].copy()
    em["provider_sk"] = em["claim_sk"].map(kit.info["provider_sk"])
    em["sd"] = em["service_date"].map(_d)
    return em


def _level_migration(kit: _Kit) -> None:
    em = _em_lines(kit)
    change = D(2025, 4, 1)
    ptype = kit.w.tables["provider"].set_index("provider_sk")["provider_type"]
    em = em[em["provider_sk"].map(ptype) == "CLINIC"]
    busiest = em.groupby("provider_sk").size().sort_values(ascending=False).index[:8].tolist()
    chosen = kit.providers(2, "CLINIC", among=busiest)
    top = {"99202": "99205", "99203": "99205", "99204": "99205",
           "99212": "99215", "99213": "99215", "99214": "99215"}
    for i, prov in enumerate(chosen):
        mine = em[(em["provider_sk"] == prov) & (em["sd"] >= change)]
        mine = mine[~mine["claim_sk"].isin(kit.w.used_claims)]
        mine = mine[kit.rng.random(len(mine)) < 0.8]
        claims = kit.take_ids(sorted(set(mine["claim_sk"])))
        mine = mine[mine["claim_sk"].isin(set(claims))]
        for idx, code in zip(mine.index, mine["activity_code"].astype(str)):
            kit.recode_line(idx, top[code])
        kit.w.recompute_header_amounts(claims)
        if i == 0:
            kit.record("level migration to top-level visits from April 2025", ["CLN-01-R04", "CLN-01-R01"],
                       claims, subject_type="provider", subjects=[prov])
        else:
            prv = kit.w.tables["provider"]
            kit.w.set_values("provider", prv["provider_sk"] == prov, ownership_changed_on=change - TD(days=12))
            kit.record("level jump coinciding with a declared change of ownership (look-alike)",
                       ["CLN-01-R04"], claims, subject_type="provider", subjects=[prov], positive=False,
                       note="excluded: known structural change")


_MEDICAL_CR = [("CR110", "J18.9"), ("CR115", "A09"), ("CR114", "N10"), ("CR117", "L03.115"), ("CR113", "E11.65")]


def _ip_claim(kit: _Kit, member: str, hospital: str, day: D, cr: str, dx: list[str], **kw) -> str | None:
    return kit.claim(member_sk=member, provider_sk=hospital, service_date=day, claim_type="INPATIENT",
                     case_rate=cr, diagnoses=dx, **kw)


def _undocumented_cc(kit: _Kit) -> None:
    members = kit.members(16)
    for i, m in enumerate(members):
        hosp = kit.any_provider("HOSPITAL")
        cr, principal = _MEDICAL_CR[i % len(_MEDICAL_CR)]
        cc = _cc_for(kit, principal)
        day = kit.weekday(D(2024, 9, 1), D(2025, 10, 15))
        c = _ip_claim(kit, m, hosp, day, cr, [principal, cc])
        if not c:
            continue
        kit.set_band(c, "B")
        if i < 13:
            kit.doc_secondary(c, drop=cc)
            kit.record("complication code lifts the case rate but is not in the discharge summary",
                       ["CLN-02-R01"], [c])
        else:
            kit.record("complication code that the discharge summary records (look-alike)", ["CLN-02-R01"],
                       [c], positive=False, note="supported by the documentation")


def _hospital_ip(kit: _Kit, hospital: str, since: D | None = None) -> list[str]:
    inf = kit.free(kit.info)
    sel = inf[(inf["provider_sk"] == hospital) & (inf["claim_type"] == "INPATIENT")]
    if since:
        sel = sel[sel["sd"] >= since]
    return sorted(sel.index)


def _upgrade_documented(kit: _Kit, claims: list[str], band: str, share: float) -> list[str]:
    done = []
    dx = kit.w.tables["diagnosis"]
    for c in claims:
        if kit.rng.random() > share:
            continue
        current = kit.band(c)
        if current is None or current == "A" or (band == "B" and current == "B"):
            continue
        principal = dx.loc[(dx["claim_sk"] == c) & (dx["diagnosis_type"] == "PRINCIPAL"), "code"]
        if principal.empty:
            continue
        code = _cc_for(kit, str(principal.iloc[0]), mcc=band == "A")
        kit.add_dx(c, code)
        kit.doc_secondary(c, add=code)
        kit.set_band(c, band)
        done.append(c)
    return done


def _hospitals_by_ip(kit: _Kit) -> list[str]:
    inf = kit.info
    ip = inf[inf["claim_type"] == "INPATIENT"].groupby("provider_sk").size().sort_values(ascending=False)
    return ip.index.tolist()


def _comorbidity_hospital(kit: _Kit) -> None:
    ranked = _hospitals_by_ip(kit)
    got = kit.providers(1, "HOSPITAL", among=ranked[3:8])
    if not got:
        return
    h = got[0]
    claims = kit.take_ids(_hospital_ip(kit, h))
    done = _upgrade_documented(kit, claims, "B", 0.75)
    kit.record("hospital records (and documents) complication codes on most stays", ["CLN-02-R02"], done,
               subject_type="provider", subjects=[h])


def _severity_shift(kit: _Kit) -> None:
    ranked = _hospitals_by_ip(kit)
    got = kit.providers(1, "HOSPITAL", among=ranked[:3])
    if not got:
        return
    h = got[0]
    claims = kit.take_ids(_hospital_ip(kit, h, since=D(2025, 5, 1)))
    done = _upgrade_documented(kit, claims, "A", 0.9)
    kit.record("hospital severity index jumps from May 2025 with unchanged stays", ["CLN-02-R05", "CLN-02-R02"],
               done, subject_type="provider", subjects=[h])


def _history_conflict(kit: _Kit) -> None:
    dx = kit.w.tables["diagnosis"]
    dx = dx.assign(member_sk=dx["claim_sk"].map(kit.info["member_sk"]), cat=dx["code"].astype(str).str[:3])
    per = dx.groupby(["cat", "member_sk"])["claim_sk"].nunique()
    recurrence = (per >= 2).groupby(level=0).mean()
    chronic_cc = [c for c, (cc, mcc, _) in kit.grouper.items() if (cc or mcc) and recurrence.get(c[:3], 0) >= 0.6]
    if not chronic_cc:
        chronic_cc = ["I50.9", "J44.1"]
    inf = kit.info.dropna(subset=["sd"])
    counts = inf.groupby("member_sk").size()
    rich = counts[counts >= 6].index
    drugs_for = {c: {p for p, d in kit.ctx.drugs.items() if c[:3] in (d["indication_prefixes"] or "")}
                 for c in chronic_cc}
    planted = 0
    for m in kit.rng.permutation(np.array(rich, dtype=object)):
        if planted >= 9:
            break
        if str(m) in kit.w.used_entities:
            continue
        member = kit.ctx.members.get(m)
        if not member:
            continue
        cats = set(dx.loc[dx["member_sk"] == m, "cat"])
        code = next((c for c in chronic_cc if c[:3] not in cats and not (set(member.get("drugs") or []) & drugs_for[c])),
                    None)
        if code is None:
            continue
        mine = inf[inf["member_sk"] == m].sort_values("sd")
        for c, row in mine.iterrows():
            if c in kit.w.used_claims or row["claim_type"] != "OUTPATIENT":
                continue
            prior = ((mine["sd"] < row["sd"]) & (mine["sd"] >= row["sd"] - TD(days=365))).sum()
            later = ((mine["sd"] > row["sd"]) & (mine["sd"] <= row["sd"] + TD(days=365))).sum()
            if prior >= 3 and later >= 1:
                kit.w.used_entities.add(str(m))
                kit.take_ids([c])
                kit.add_dx(c, code)
                kit.record("long-standing condition on one claim only, absent from a long history",
                           ["CLN-02-R03"], [c], subjects=[c])
                planted += 1
                break


def _poa_sequencing(kit: _Kit) -> None:
    dx = kit.w.tables["diagnosis"]
    enc = kit.w.tables["encounter"].set_index("claim_sk")
    inf = kit.free(kit.info)
    ip = inf[inf["claim_type"] == "INPATIENT"].index
    principal = dx[(dx["diagnosis_type"] == "PRINCIPAL") & dx["claim_sk"].isin(set(ip))]
    obstetric = principal["code"].astype(str).str[:1].isin(["O", "P"])
    adm = principal["claim_sk"].map(enc["admission_type"]).astype(str).str.upper()
    normal = principal[~obstetric & ~adm.str.contains("TRANSFER")]
    pick = normal.sample(n=min(8, len(normal)), random_state=int(kit.rng.integers(1 << 31)))
    claims = kit.take_ids(pick["claim_sk"])
    kit.w.set_values("diagnosis", kit.w.tables["diagnosis"].index.isin(pick.index[pick["claim_sk"].isin(claims)]),
                     present_on_admission="N")
    kit.record("main diagnosis marked not present on admission", ["CLN-02-R04"], claims)
    obs = principal[obstetric]
    obs = obs[~obs["claim_sk"].isin(kit.w.used_claims)].head(2)
    oc = kit.take_ids(obs["claim_sk"])
    kit.w.set_values("diagnosis", kit.w.tables["diagnosis"].index.isin(obs.index[obs["claim_sk"].isin(oc)]),
                     present_on_admission="N")
    kit.record("obstetric main diagnosis not present on admission (exempt look-alike)", ["CLN-02-R04"], oc,
               positive=False, note="excluded: obstetric coding rules")
    sec = kit.w.tables["diagnosis"]
    sec = sec[(sec["diagnosis_type"] == "SECONDARY") & ~sec["claim_sk"].isin(kit.w.used_claims)]
    sec = sec[sec["claim_sk"].isin(set(inf.index))].drop_duplicates("claim_sk").head(3)
    dup_claims = kit.take_ids(sec["claim_sk"])
    rows = sec[sec["claim_sk"].isin(dup_claims)].copy()
    rows["sequence"] = rows["sequence"].astype(int) + 10
    kit.w.append("diagnosis", rows)
    kit.record("the same diagnosis code listed twice on a claim", ["CLN-02-R04"], dup_claims)


# ---------------------------------------------------------------------------
# CLN-03 / CLN-04 — clinical consistency, repeats, cascades
# ---------------------------------------------------------------------------


_NO_CARDIAC = ("I10", "I20", "I21", "I25", "I48", "I50", "E11")


def _no_cardiac(m: dict) -> bool:
    return not any(str(c).startswith(_NO_CARDIAC) for c in (m.get("chronic") or []))


def _unsupported_indication(kit: _Kit) -> None:
    members = kit.members(14, where=_no_cardiac)
    for i, m in enumerate(members):
        gp = kit.any_provider("CLINIC", "GENERAL_PRACTICE")
        day = kit.weekday(D(2024, 8, 1), D(2025, 11, 15))
        dx = "J06.9" if i < 12 else "R07.9"
        c = kit.claim(member_sk=m, provider_sk=gp, service_date=day, claim_type="OUTPATIENT",
                      lines=["99213", "93000"], diagnoses=[dx])
        if i < 12:
            kit.record("ECG billed for an upper respiratory infection with no cardiac reason", ["CLN-03-R02"], [c])
        else:
            kit.record("ECG billed for chest pain (look-alike)", ["CLN-03-R02"], [c], positive=False)


def _had_code(kit: _Kit, member: str, code: str) -> bool:
    L = kit.w.tables["claim_line"]
    claims = set(kit.info.index[kit.info["member_sk"] == member])
    return bool(((L["activity_code"] == code) & L["claim_sk"].isin(claims)).any())


def _sequence_conflict(kit: _Kit) -> None:
    members = [m for m in kit.members(12, age=(30, 70)) if not _had_code(kit, m, "73562")]
    for i, m in enumerate(members):
        rc = kit.any_provider("RADIOLOGY_CENTRE")
        day = kit.weekday(D(2024, 9, 1), D(2025, 11, 15))
        if i < 9:
            c = kit.claim(member_sk=m, provider_sk=rc, service_date=day, claim_type="RADIOLOGY",
                          lines=["73721"], diagnoses=["M17.11"])
            kit.record("knee MRI with no knee X-ray first", ["CLN-03-R03"], [c])
        else:
            gp = kit.any_provider("CLINIC", "GENERAL_PRACTICE")
            ref = kit.clinician(gp, "EM_OFFICE_EST", day) if gp else None
            c = kit.claim(member_sk=m, provider_sk=rc, service_date=day, claim_type="RADIOLOGY",
                          lines=["73721"], diagnoses=["M17.11"], referral_from=ref)
            kit.record("knee MRI for a patient referred in from elsewhere (look-alike)", ["CLN-03-R03"], [c],
                       positive=False, note="excluded: work-up may have been done externally")


def _rare_code(kit: _Kit) -> None:
    got = kit.providers(1, "CLINIC", "GENERAL_PRACTICE")
    if not got:
        return
    gp = got[0]
    inf = kit.info
    n_claims = int((inf["provider_sk"] == gp).sum())
    n = max(30, int(n_claims * 0.25))
    members = kit.members(n)
    claims = []
    for m in members:
        day = kit.weekday(D(2024, 8, 1), D(2025, 11, 30))
        c = kit.claim(member_sk=m, provider_sk=gp, service_date=day, claim_type="OUTPATIENT",
                      lines=["99213", "95816"], diagnoses=["G40.909"])
        claims.append(c)
    kit.record("a general practice billing electroencephalograms", ["CLN-03-R04"], claims,
               subject_type="provider", subjects=[gp])


def _move_results(kit: _Kit, claim_sk: str, when: _dt.datetime | None = None, value: str | None = None) -> None:
    obs = kit.w.tables["observation"]
    m = obs["claim_sk"] == claim_sk
    if when is not None:
        obs.loc[m, "event_time"] = when
    if value is not None:
        obs.loc[m, "value"] = value


def _no_result_before_repeat(kit: _Kit) -> None:
    members = kit.members(10, where=lambda m: any(str(c).startswith("E11") for c in (m.get("chronic") or [])))
    if len(members) < 10:
        members += kit.members(10 - len(members))
    for i, m in enumerate(members):
        lab = kit.any_provider("DIAGNOSTIC_LAB")
        day = kit.weekday(D(2024, 9, 1), D(2025, 10, 1))
        first = kit.claim(member_sk=m, provider_sk=lab, service_date=day, claim_type="LAB",
                          lines=["36415", "83036"], diagnoses=["E11.9"])
        second = kit.claim(member_sk=m, provider_sk=lab, service_date=day + TD(days=12), claim_type="LAB",
                           lines=["36415", "83036"], diagnoses=["E11.9"])
        if not first or not second:
            continue
        repeat = _dt.datetime.combine(day + TD(days=12), _dt.time(9, 0))
        if i < 8:
            _move_results(kit, first, repeat + TD(days=25))
            kit.record("HbA1c repeated twelve days later, before the first result was recorded",
                       ["CLN-04-R03", "CLN-04-R01"], [first, second], subjects=[second])
        else:
            _move_results(kit, first, repeat + TD(days=1))
            kit.record("HbA1c repeated while the first result was a day from being reported (look-alike)",
                       ["CLN-04-R03"], [first, second], subjects=[second], positive=False,
                       note="excluded from CLN-04-R03 (result latency); still an early repeat for CLN-04-R01")


def _cascade(kit: _Kit) -> None:
    got = kit.providers(1, "CLINIC", "GENERAL_PRACTICE")
    hosp = kit.any_provider("HOSPITAL")
    if not got or not hosp:
        return
    gp = got[0]
    members = kit.members(16, age=(30, 70))
    claims = []
    for m in members:
        day = kit.weekday(D(2024, 8, 1), D(2025, 11, 1))
        referrer = kit.clinician(gp, "EM_OFFICE_EST", day) or (kit.ctx.clinicians[gp][0]["id"])
        first = kit.claim(member_sk=m, provider_sk=gp, service_date=day, claim_type="OUTPATIENT",
                          lines=["99212"], diagnoses=["R07.9"], clinician_id=referrer)
        follow = kit.claim(member_sk=m, provider_sk=hosp, service_date=day + TD(days=4), claim_type="OUTPATIENT",
                           lines=["99245", "93306", "71260", "93000", "36415", "80053", "84484", "71046",
                                  "85025", "86140"], diagnoses=["R07.9"], referral_from=referrer)
        claims += [first, follow]
        ref = kit.w.tables["referral"]
        kit.w.set_values("referral", ref["resulting_claim_sk"] == follow, specialty="CARDIOLOGY")
    kit.record("minor clinic visits reliably followed by a large specialist, CT and echo bundle",
               ["CLN-04-R05"], claims, subject_type="provider", subjects=[gp])


def _admission_rate(kit: _Kit) -> None:
    enc = kit.w.tables["encounter"]
    ptype = kit.w.tables["provider"].set_index("provider_sk")["provider_type"]
    e = enc[enc["facility_id"].map(ptype) == "HOSPITAL"]
    adm = (e["encounter_type"] == "INPATIENT") & (e["admission_type"].astype(str).str.upper() == "EMERGENCY")
    pres = e["encounter_type"].isin(["EMERGENCY", "OUTPATIENT"]) | adm
    per = pd.DataFrame({"pres": pres, "adm": adm, "h": e["facility_id"]}).groupby("h").sum()
    r0 = per["adm"].sum() / max(per["pres"].sum(), 1)
    mid = per.sort_values("pres").index.tolist()
    mid = mid[len(mid) // 4: len(mid) * 3 // 4]
    got = kit.providers(1, "HOSPITAL", among=mid)
    if not got:
        return
    h = got[0]
    P, A = float(per.at[h, "pres"]), float(per.at[h, "adm"])
    target = min(0.6, 2.6 * r0)
    x = int(math.ceil(max(25.0, (target * P - A) / max(1 - target, 0.2))))
    x = min(x, 160)
    x = max(x, 70)
    members = kit.members(x, age=(25, 75))
    claims = []
    # presentations other hospitals treat as outpatients (gastritis with a mild
    # gastroenteritis), admitted overnight under the gastroenteritis case rate
    for i, m in enumerate(members):
        day = kit.weekday(D(2024, 7, 15), D(2025, 11, 30))
        c = _ip_claim(kit, m, h, day, "CR115", ["K29.70", "A09"], los=1, admission_type="EMERGENCY")
        claims.append(c)
    kit.record("hospital admitting overnight presentations others treat as outpatients", ["CLN-05-R04"], claims,
               subject_type="provider", subjects=[h])


# ---------------------------------------------------------------------------
# CLN-06 / CLN-07 — did it happen, could it happen
# ---------------------------------------------------------------------------


def _claims_with(kit: _Kit, claim_type: str, family_prefix: str, n: int, before: D = D(2025, 11, 1)) -> list[str]:
    L = kit.w.tables["claim_line"]
    fam = L["activity_code"].map(lambda c: kit.codes.get(str(c), {}).get("code_family", ""))
    has = set(L.loc[fam.astype(str).str.startswith(family_prefix), "claim_sk"])
    inf = kit.free(kit.info)
    inf = inf[(inf["claim_type"] == claim_type) & inf.index.isin(has) & (inf["sd"] <= before)]
    pick = inf.sample(n=min(n, len(inf)), random_state=int(kit.rng.integers(1 << 31))).index
    return kit.take_ids(pick)


def _no_report(kit: _Kit) -> None:
    claims = _claims_with(kit, "RADIOLOGY", "IMAGING_", 10)
    obs = kit.w.tables["observation"]
    kit.drop_rows("observation", obs["claim_sk"].isin(set(claims)))
    docs = kit.w.tables["document"]
    kit.drop_rows("document", docs["claim_sk"].isin(set(claims)) & (docs["doc_type"] == "IMAGING_REPORT"))
    L = kit.w.tables["claim_line"]
    kit.w.set_values("claim_line", L["claim_sk"].isin(set(claims[:5])), ordering_clinician_id=None)
    kit.record("imaging paid with no report or result on file", ["CLN-06-R01"], claims)


def _member_denial(kit: _Kit) -> None:
    claims = _claims_with(kit, "OUTPATIENT", "EM_", 14)
    conf = kit.w.tables["member_confirmation"]
    att = kit.w.tables["attendance_record"]
    for i, c in enumerate(claims):
        sd = kit.info.at[c, "sd"]
        member, prov = kit.info.at[c, "member_sk"], kit.info.at[c, "provider_sk"]
        if i < 10:
            channel, delay = "APP", 12
        elif i < 12:
            channel, delay = "SMS_UNVERIFIED", 9
        else:
            channel, delay = "APP", 300
        row = dict(confirmation_sk=kit.w.new_id("MCF", 6) + "C", member_sk=member, claim_sk=c,
                   service_confirmed=False, response="NOT_RECEIVED", response_date=sd + TD(days=delay),
                   channel=channel, tenant_id=TENANT)
        m = conf["claim_sk"] == c
        if m.any():
            kit.w.set_values("member_confirmation", m, service_confirmed=False, response="NOT_RECEIVED",
                             response_date=row["response_date"], channel=channel)
        else:
            kit.w.append("member_confirmation", [row])
            conf = kit.w.tables["member_confirmation"]
        if i < 5:
            ma = att["claim_sk"] == c
            if ma.any():
                kit.w.set_values("attendance_record", ma, attended=False)
            else:
                kit.w.append("attendance_record", [dict(record_sk=kit.w.new_id("ATT", 7) + "C", member_sk=member,
                                                        provider_sk=prov, claim_sk=c, attendance_date=sd,
                                                        attended=False, source="CLINIC_CHECKIN_LOG",
                                                        tenant_id=TENANT)])
                att = kit.w.tables["attendance_record"]
    kit.record("member says the visit did not happen (verified app response)", ["CLN-06-R04"], claims[:10])
    kit.record("denial through an unverified channel (look-alike)", ["CLN-06-R04"], claims[10:12],
               positive=False, note="excluded: authentication quality")
    kit.record("denial given ten months after the visit (look-alike)", ["CLN-06-R04"], claims[12:],
               positive=False, note="excluded: recall")


def _no_footprint(kit: _Kit) -> None:
    obs = kit.w.tables["observation"]
    rx = kit.w.tables["prescription_dispense"]
    busy = set(obs["member_sk"].dropna()) | set(rx["member_sk"].dropna())
    members = kit.members(6, age=(35, 70),
                          where=lambda m: not (m.get("chronic") or m.get("drugs")) and m["sk"] not in busy)
    for m in members:
        gp = kit.any_provider("CLINIC", "GENERAL_PRACTICE")
        claims = []
        start = kit.weekday(D(2024, 8, 1), D(2025, 3, 1))
        for k in range(4):
            day = start + TD(days=60 * k + int(kit.rng.integers(0, 10)))
            claims.append(kit.claim(member_sk=m, provider_sk=gp, service_date=day, claim_type="OUTPATIENT",
                                    lines=["99213"], diagnoses=["E11.9"]))
        kit.record("diabetes billed repeatedly with no medicine or result ever recorded", ["CLN-06-R05"],
                   claims, subject_type="member", subjects=[m])


def _long_day(kit: _Kit) -> None:
    for k in range(3):
        gp = kit.any_provider("CLINIC", "GENERAL_PRACTICE")
        if not gp:
            continue
        day = kit.weekday(D(2024, 9, 1), D(2025, 10, 30))
        clin = kit.clinician(gp, "EM_OFFICE_EST", day)
        if not clin:
            continue
        members = kit.members(26)
        claims = [kit.claim(member_sk=m, provider_sk=gp, service_date=day, claim_type="OUTPATIENT",
                            lines=["99215"], diagnoses=["J06.9"], clinician_id=clin) for m in members]
        kit.w.used_entities.add(clin)
        kit.record("clinician billing more than seventeen hours of visits in one day", ["CLN-07-R01"], claims,
                   subject_type="clinician", subjects=[clin])


def _scanner_overbooked(kit: _Kit) -> None:
    inv = kit.w.tables["equipment_inventory"]
    ptype = kit.w.tables["provider"].set_index("provider_sk")["provider_type"]
    mri = inv[(inv["equipment_type"] == "MRI") & (inv["provider_sk"].map(ptype) == "RADIOLOGY_CENTRE")]
    got = kit.providers(1, "RADIOLOGY_CENTRE", among=mri["provider_sk"].tolist())
    if not got:
        return
    rc = got[0]
    row = mri[mri["provider_sk"] == rc].iloc[0]
    cap = int(row["units"]) * int(row["hours_per_day"]) * 60 // int(row["minutes_per_use"])
    n = int(math.ceil(cap * 1.25)) + 4
    day = kit.weekday(D(2025, 2, 1), D(2025, 9, 30))
    members = kit.members(n, age=(25, 65))
    claims = [kit.claim(member_sk=m, provider_sk=rc, service_date=day, claim_type="RADIOLOGY",
                        lines=["70551"], diagnoses=["G43.909"]) for m in members]
    kit.record(f"MRI scans billed beyond one day's scanner capacity ({cap})", ["CLN-07-R03"], claims,
               subject_type="provider", subjects=[rc])


# ---------------------------------------------------------------------------
# CLN-08 — laboratory
# ---------------------------------------------------------------------------


def _no_order(kit: _Kit) -> None:
    claims = _claims_with(kit, "LAB", "LAB_", 12)
    dx = kit.w.tables["diagnosis"]
    screening = set(dx.loc[dx["code"].astype(str).str.match(r"Z(0\d|1[0-3])"), "claim_sk"])
    claims = [c for c in claims if c not in screening][:8]
    L = kit.w.tables["claim_line"]
    labs = L["activity_code"].map(lambda c: str(kit.codes.get(str(c), {}).get("service_family", "")) == "LAB")
    for i, c in enumerate(claims):
        m = (L["claim_sk"] == c) & labs
        kit.w.set_values("claim_line", m, ordering_clinician_id=None if i < 4 else f"CL9{9000 + i}",
                         rendering_clinician_id=None)
    kit.record("laboratory tests with no ordering clinician, or one on no roster", ["CLN-08-R01"], claims)
    members = kit.members(2)
    look = []
    for m in members:
        lab = kit.any_provider("DIAGNOSTIC_LAB")
        c = kit.claim(member_sk=m, provider_sk=lab, service_date=kit.weekday(D(2024, 9, 1), D(2025, 10, 1)),
                      claim_type="LAB", lines=["36415", "80061"], diagnoses=["Z00.00"])
        if c:
            L = kit.w.tables["claim_line"]
            kit.w.set_values("claim_line", L["claim_sk"] == c, ordering_clinician_id=None)
            look.append(c)
    kit.record("screening-programme test without an individual order (look-alike)", ["CLN-08-R01"], look,
               positive=False, note="excluded: screening programme")


def _panels(kit: _Kit) -> None:
    comps = kit.w.context["panels"]["80048"]
    members = kit.members(12)
    for i, m in enumerate(members):
        lab = kit.any_provider("DIAGNOSTIC_LAB")
        day = kit.weekday(D(2024, 8, 1), D(2025, 11, 1))
        if i < 8:
            c = kit.claim(member_sk=m, provider_sk=lab, service_date=day, claim_type="LAB",
                          lines=["36415"] + list(comps), diagnoses=["E11.9"])
            kit.record("all basic metabolic panel components billed one by one", ["CLN-08-R02"], [c])
        elif i < 11:
            c = kit.claim(member_sk=m, provider_sk=lab, service_date=day, claim_type="LAB",
                          lines=["36415", "80061", "82465"], diagnoses=["E78.5"])
            kit.record("lipid panel billed together with its own cholesterol component", ["CLN-08-R02"], [c])
        else:
            c = kit.claim(member_sk=m, provider_sk=lab, service_date=day, claim_type="LAB",
                          lines=["36415", "80061", {"code": "82465", "indicator": "REFLEX"}], diagnoses=["E78.5"])
            kit.record("component repeated as a reflex test (look-alike)", ["CLN-08-R02"], [c], positive=False,
                       note="excluded: reflex testing")


def _results(kit: _Kit) -> None:
    claims = _claims_with(kit, "LAB", "LAB_", 8)
    obs = kit.w.tables["observation"]
    kit.drop_rows("observation", obs["claim_sk"].isin(set(claims)))
    docs = kit.w.tables["document"]
    kit.drop_rows("document", docs["claim_sk"].isin(set(claims)) & (docs["doc_type"] == "LAB_REPORT"))
    kit.record("laboratory tests paid with no result recorded", ["CLN-08-R03"], claims)
    for g in range(2):
        lab = kit.providers(1, "DIAGNOSTIC_LAB")
        if not lab:
            continue
        members = kit.members(3)
        sponsors = {kit.ctx.members[m].get("sponsor") for m in members}
        if len(sponsors) < 2:
            continue
        day = kit.weekday(D(2024, 9, 1), D(2025, 10, 1))
        claims = [kit.claim(member_sk=m, provider_sk=lab[0], service_date=day + TD(days=3 * k), claim_type="LAB",
                            lines=["36415", "80053"], diagnoses=["E78.5"]) for k, m in enumerate(members)]
        claims = [c for c in claims if c]
        if len(claims) < 2:
            continue
        obs = kit.w.tables["observation"]
        src = obs[obs["claim_sk"] == claims[0]].set_index("observation_code")["value"]
        for c in claims[1:]:
            m = obs["claim_sk"] == c
            obs.loc[m, "value"] = obs.loc[m, "observation_code"].map(src).fillna(obs.loc[m, "value"])
        kit.record("identical multi-value results reported for unrelated patients", ["CLN-08-R03"], claims,
                   subject_type="provider", subjects=lab)


def _passthrough(kit: _Kit) -> None:
    L = kit.w.tables["claim_line"]
    labs = kit.providers(1, "DIAGNOSTIC_LAB")
    clinics = kit.providers(4, "CLINIC", "GENERAL_PRACTICE")
    if not labs or len(clinics) < 4:
        return
    lab = labs[0]
    own = L[L["claim_sk"].map(kit.info["provider_sk"]) == lab]
    own_price = (own["gross_amount"].astype(float) / own["units"].astype(float).clip(lower=1)).groupby(
        own["activity_code"]).median()
    for i, clinic in enumerate(clinics):
        markup = 3.0 if i == 0 else (1.08, 1.12, 1.18)[i - 1]
        members = kit.members(13)
        made = []
        for m in members:
            day = kit.weekday(D(2024, 8, 1), D(2025, 11, 15))
            c = kit.claim(member_sk=m, provider_sk=clinic, service_date=day, claim_type="OUTPATIENT",
                          lines=["99213", "83036", "84443"], diagnoses=["E11.9"])
            if not c:
                continue
            made.append(c)
            L = kit.w.tables["claim_line"]
            for idx in L.index[(L["claim_sk"] == c) & L["activity_code"].isin(["83036", "84443"])]:
                code = L.at[idx, "activity_code"]
                base = float(own_price.get(code, kit.price(code)))
                current = float(L.at[idx, "gross_amount"]) / max(float(L.at[idx, "units"] or 1), 1.0)
                kit.scale_line(idx, base * markup / max(current, 1e-9))
                L.at[idx, "performing_entity_id"] = lab
            kit.w.recompute_header_amounts([c])
        if i == 0:
            kit.record("clinic re-billing a reference laboratory's tests at three times its price",
                       ["CLN-08-R04"], made, subject_type="provider", subjects=[clinic])
        else:
            kit.record("clinic passing reference-laboratory tests through at a normal margin (look-alike)",
                       ["CLN-08-R04"], made, subject_type="provider", subjects=[clinic], positive=False)


def _add_genetic_codes(kit: _Kit) -> None:
    """Molecular tests the reference build does not list: added consistently everywhere."""
    from . import base as _base
    ref = kit.w.tables["activity_code_reference"]
    rows, tariff, csv, umax = [], [], [], []
    for code, desc, price in GENETIC_CODES:
        if code in kit.codes:
            continue
        row = {"activity_code": code, "description": desc, "activity_type": "CPT", "service_family": "LAB",
               "code_family": "LAB_MOLECULAR", "code_level": None, "sex_restriction": None, "age_min": 0,
               "age_max": 120, "minutes": None, "is_time_based": False, "specialty_required": None,
               "facility_types": "CLINIC;HOSPITAL;DIAGNOSTIC_LAB", "complexity": "HIGH",
               "is_telehealth_eligible": False, "is_scarce_equipment": False, "is_implant": False,
               "unit_price_reference": price}
        kit.codes[code] = row
        rows.append(row)
        for tier in _base.TIERS:
            for payer in _base.PAYERS:
                for a, b in ((D(2024, 1, 1), D(2024, 12, 31)), (D(2025, 1, 1), D(2025, 12, 31))):
                    tariff.append({"activity_code": code, "network_tier": tier,
                                   "allowed_price": kit.ctx.allowed_price(code, tier, payer, a),
                                   "valid_from": a, "valid_to": b, "payer_id": payer})
        csv.append({"code": code, "code_system": "CPT", "valid_from": D(2018, 1, 1), "valid_to": D(2099, 12, 31)})
        umax.append({"activity_code": code, "max_units_per_day": 1.0, "rationale": "One per day by definition"})
    if rows:
        kit.w.append("activity_code_reference", rows)
        kit.w.append("tariff", tariff)
        kit.w.append("code_system_version", csv)
        kit.w.append("unit_maximum_policy", umax)
    del ref


def _genetic(kit: _Kit) -> None:
    _add_genetic_codes(kit)
    got = kit.providers(1, "CLINIC", "GENERAL_PRACTICE")
    if not got:
        return
    clinic = got[0]
    members = kit.members(34)
    claims = []
    for i, m in enumerate(members):
        day = kit.weekday(D(2024, 8, 1), D(2025, 11, 15))
        code = GENETIC_CODES[i % len(GENETIC_CODES)][0]
        claims.append(kit.claim(member_sk=m, provider_sk=clinic, service_date=day, claim_type="OUTPATIENT",
                                lines=["99213", "36415", code], diagnoses=["Z00.00"]))
    kit.record("clinic ordering genetic tests for a large share of its patients", ["CLN-08-R05"], claims,
               subject_type="provider", subjects=[clinic])
    # background: a handful of ordinary providers order one such test each
    back = kit.members(10)
    bc = []
    for i, m in enumerate(back):
        prov = kit.any_provider("CLINIC", "GENERAL_PRACTICE")
        c = kit.claim(member_sk=m, provider_sk=prov, service_date=kit.weekday(D(2024, 8, 1), D(2025, 11, 15)),
                      claim_type="OUTPATIENT", lines=["99213", "36415", GENETIC_CODES[i % 3][0]],
                      diagnoses=["Z00.00"])
        bc.append(c)
    kit.record("occasional genetic test at an ordinary practice (background)", ["CLN-08-R05"], bc,
               positive=False, note="normal rate")


def _frequency(kit: _Kit) -> None:
    """Patients seen at a clinic on ten or more separate days within a month."""
    members = kit.members(3, age=(25, 60))
    for m in members:
        gp = kit.any_provider("CLINIC", "GENERAL_PRACTICE")
        start = kit.weekday(D(2024, 9, 1), D(2025, 10, 1))
        claims = [kit.claim(member_sk=m, provider_sk=gp, service_date=start + TD(days=3 * k),
                            claim_type="OUTPATIENT", lines=["99213"], diagnoses=["J06.9"]) for k in range(10)]
        kit.record("ten consultations for a cold within one month", ["CLN-04-R02"], claims,
                   subject_type="member", subjects=[m])


def _identical_encounters(kit: _Kit) -> None:
    """One clinic billing the same three-service encounter, at the same clock time, for different people."""
    got = kit.providers(1, "CLINIC", "GENERAL_PRACTICE")
    if not got:
        return
    clinic = got[0]
    template = [{"code": "99214", "gross": 360.0}, {"code": "93000", "gross": 150.0},
                {"code": "71046", "gross": 160.0}]
    claims = []
    for m in kit.members(6, age=(25, 65)):
        day = kit.weekday(D(2024, 9, 1), D(2025, 10, 30))
        claims.append(kit.claim(member_sk=m, provider_sk=clinic, service_date=day, claim_type="OUTPATIENT",
                                lines=[dict(x) for x in template], diagnoses=["R07.9"], start_minute=600))
    L = kit.w.tables["claim_line"]
    for c in [c for c in claims if c]:
        m = L["claim_sk"] == c
        day = pd.Timestamp(kit.info.at[c, "sd"] if c in kit.info.index else L.loc[m, "service_date"].iloc[0])
        kit.w.set_values("claim_line", m, service_start_time=day.normalize() + pd.Timedelta(hours=10))
    kit.record("identical three-service encounter at 10:00 for different patients", ["CLN-06-R03"], claims,
               subject_type="provider", subjects=[clinic])


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------


def inject(world: World) -> None:
    kit = _Kit(world)
    steps = [
        _level_migration, _comorbidity_hospital, _severity_shift, _poa_sequencing, _history_conflict,
        _no_report, _member_denial, _no_order, _results,
    ]
    for step in steps:           # edits of existing claims first, on a consistent view
        step(kit)
        kit.refresh()
    for step in (_undocumented_cc, _unsupported_indication, _sequence_conflict, _rare_code,
                 _no_result_before_repeat, _cascade, _admission_rate, _no_footprint, _long_day,
                 _scanner_overbooked, _panels, _passthrough, _genetic, _frequency, _identical_encounters):
        step(kit)
        kit.refresh()
