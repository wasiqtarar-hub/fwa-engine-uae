"""Plant the PAY-01 to PAY-06 line-level patterns in the SYNTHETIC UAE demo dataset.

Every control implemented in ``src/fwa/engine/unlocked/pay_a.py`` gets at least
one realistic pattern here, at a modest volume, and every planted item is
written to the answer key with the rule ids it is meant to exercise. Where a
declared exclusion has a legitimate look-alike, one or two look-alikes are
planted too (``positive=False``) so the exclusion is demonstrated.

Patterns (all SYNTHETIC):

* PAY-01-R04  a service paid in full by a second insurer as well (matched by the
              privacy-protected cross-payer token), with and without a
              coordination-of-benefits record;
* PAY-02-R01  a lab panel billed with one of its own components on the same claim;
* PAY-02-R02  an ECG billed complete at one provider and its interpretation
              billed again by another provider the same day;
* PAY-02-R03  case-rate claims charging an included zero-price component, or
              billed without the activities every other such claim lists;
* PAY-02-R04  an included component charged AND paid on top of the case rate;
* PAY-02-R05  one clinic billing an unusual pair of separately charged tests;
* PAY-03-R01  a deleted code billed after its end date, an unknown code;
* PAY-03-R02  sex- and age-restricted services billed for patients outside them;
* PAY-03-R03  units above the daily maximum, on one claim or split over two;
* PAY-03-R04  timed units longer than the recorded window; one therapist in two
              places at once;
* PAY-04-R01  advanced imaging with no approval;
* PAY-04-R02  approval refused, or service after the approval expired;
* PAY-04-R03  approval for another code or another provider;
* PAY-04-R04  a specialty-drug course that runs past the approved units;
* PAY-04-R05  one approval reused for other patients;
* PAY-05-R01  modifier 59 used to split a pair whose edit does not allow it;
* PAY-05-R02  modifier 22 with no operative report;
* PAY-05-R03  two clinics appending modifier 25 to most of their visits;
* PAY-05-R04  after a new bundling edit, a clinic's plain components are refused
              and it switches to modifier 59, which gets them paid;
* PAY-06-R01  lines priced above the contracted tariff.
"""

from __future__ import annotations

import datetime as _dt
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .world import World

D = _dt.date
TD = _dt.timedelta
TENANT = "T001"
FAMILY = "PAY"


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


class _Kit:
    def __init__(self, world: World) -> None:
        self.w = world
        self.ctx = world.context["base_ctx"]
        self.rng = world.rng
        mem = world.tables["member"].copy()
        mem["dob"] = pd.to_datetime(mem["date_of_birth"], errors="coerce")
        self.members = mem.set_index("member_sk")
        self._member_pool: list[str] = []
        # Rows appended by earlier injectors can leave money columns as object dtype, which
        # World.recompute_header_amounts cannot round. Values are unchanged; only the dtype.
        lines = world.tables["claim_line"]
        for col in ("units", "gross_amount", "net_amount", "patient_share", "unit_price"):
            if col in lines.columns and lines[col].dtype == object:
                lines[col] = pd.to_numeric(lines[col], errors="coerce")

    # ------------------------------------------------------------- entities

    def members_where(self, n: int, *, sex: str | None = None, age: tuple[int, int] = (20, 60),
                      on: D = D(2024, 7, 15), covered_until: D = D(2025, 11, 30)) -> list[str]:
        ctx = self.ctx

        def ok(frame: pd.DataFrame) -> pd.Series:
            a = (pd.Timestamp(on) - frame["dob"]).dt.days / 365.25
            m = a.between(*age)
            if sex:
                m &= frame["sex"].astype(str).str.upper().str[0] == sex
            cov = frame["member_sk"].map(lambda s: bool(ctx.coverage_at(s, on)) and bool(ctx.coverage_at(s, covered_until)))
            no_cob = frame["member_sk"].map(lambda s: not ctx.members.get(s, {}).get("cob"))
            return m & cov & no_cob

        mem = self.w.tables["member"].copy()
        mem["dob"] = pd.to_datetime(mem["date_of_birth"], errors="coerce")
        got = self.w.take_entities("member", "member_sk", n, where=lambda f: ok(mem.loc[f.index]))
        return got["member_sk"].tolist()

    def providers_where(self, n: int, ptype: str, specialty: str | None = None) -> list[str]:
        """Up to ``n`` unused providers of the type (and specialty); on a small build, falls back
        to the type with any specialty, and finally to any provider of the type."""
        def ok(spec):
            def f(frame: pd.DataFrame) -> pd.Series:
                m = frame["provider_type"] == ptype
                if spec:
                    m &= frame["specialty"] == spec
                return m & frame["provider_sk"].map(lambda p: bool(self.ctx.clinicians.get(p)))
            return f
        got = self.w.take_entities("provider", "provider_sk", n, where=ok(specialty))["provider_sk"].tolist()
        if len(got) < n and specialty:
            got += self.w.take_entities("provider", "provider_sk", n - len(got), where=ok(None))["provider_sk"].tolist()
        if not got:
            prov = self.w.tables["provider"]
            pool = prov.loc[ok(None)(prov), "provider_sk"].tolist()
            got = pool[:n]
        return got

    # --------------------------------------------------------------- claims

    def claim(self, **kw: Any) -> str | None:
        kw.setdefault("strict", False)
        try:
            return self.w.make_claim(**kw)
        except Exception:  # a claim the factory cannot build is simply not planted
            return None

    def lines(self, claim_sk: str) -> pd.DataFrame:
        return self.w.lines_of([claim_sk])

    def line_of(self, claim_sk: str, code: str) -> pd.Series | None:
        ln = self.lines(claim_sk)
        ln = ln[ln["activity_code"] == code]
        return None if ln.empty else ln.iloc[0]

    def header(self, claim_sk: str) -> pd.Series:
        h = self.w.tables["claim_header"]
        return h[h["claim_sk"] == claim_sk].iloc[0]

    # ------------------------------------------------------------ amounts

    def reprice_line(self, line_sk: str, unit_price: float, *, paid: bool | None = None) -> None:
        """Change one line's unit price, keeping gross/net/patient share, remittance and header consistent."""
        lines = self.w.tables["claim_line"]
        m = lines["line_sk"] == line_sk
        row = lines[m].iloc[0]
        units = float(row["units"] or 1)
        old_gross = float(row["gross_amount"] or 0.0)
        share_pct = (float(row["patient_share"] or 0.0) / old_gross) if old_gross > 0 else self._share_pct(row)
        gross = round(unit_price * units, 2)
        pshare = round(gross * share_pct, 2)
        net = round(gross - pshare, 2)
        self.w.set_values("claim_line", m, unit_price=round(unit_price, 2), gross_amount=gross,
                          patient_share=pshare, net_amount=net)
        rem = self.w.tables["remittance"]
        rm = rem["line_sk"] == line_sk
        if rm.any():
            dec = str(rem.loc[rm, "decision"].iloc[0]).upper()
            if paid is True or (paid is None and dec != "DENIED"):
                self.w.set_values("remittance", rm, decision="PAID", payment_amount=net, adjustment=0.0,
                                  denial_code=None)
            else:
                self.w.set_values("remittance", rm, decision="DENIED", payment_amount=0.0, adjustment=net,
                                  denial_code="CLAI-012")
        self.w.recompute_header_amounts([row["claim_sk"]])

    def _share_pct(self, row: pd.Series) -> float:
        others = self.lines(row["claim_sk"])
        others = others[others["gross_amount"].astype(float) > 0]
        if others.empty:
            return 0.0
        return float(others["patient_share"].astype(float).sum() / others["gross_amount"].astype(float).sum())

    def contract_price(self, claim_sk: str, code: str, on: D) -> float:
        h = self.header(claim_sk)
        c = self.ctx.contracts.get((h["provider_sk"], h["payer_id"]), {"discount": 0.0, "tier": "TIER_2"})
        return round(self.ctx.allowed_price(code, c["tier"], h["payer_id"], on) * (1 - c["discount"]), 2)

    def set_remittance(self, line_sk: str, decision: str, denial: str | None = None) -> None:
        rem = self.w.tables["remittance"]
        rm = rem["line_sk"] == line_sk
        net = float(self.w.tables["claim_line"].loc[self.w.tables["claim_line"]["line_sk"] == line_sk,
                                                    "net_amount"].iloc[0])
        if decision == "DENIED":
            self.w.set_values("remittance", rm, decision="DENIED", payment_amount=0.0, adjustment=net,
                              denial_code=denial or "CLAI-012")
        else:
            self.w.set_values("remittance", rm, decision="PAID", payment_amount=net, adjustment=0.0,
                              denial_code=None)

    # --------------------------------------------------------- approvals

    def add_auth(self, *, member: str, provider: str, codes: dict[str, tuple[float, float]], status: str,
                 valid_from: D, valid_to: D, conditions: str = "Valid for the named facility and member only",
                 denial: str | None = None) -> str:
        pa = self.ctx.unique(self.ctx.auth_ids, "PA", 8)
        total = sum(v for _, v in codes.values())
        self.w.append("authorization", [{
            "authorization_sk": pa, "request_id": f"REQ{pa[2:]}", "response_id": f"RSP{pa[2:]}",
            "status": status, "valid_from": pd.Timestamp(valid_from) + pd.Timedelta(hours=9),
            "valid_to": pd.Timestamp(valid_to) + pd.Timedelta(hours=23, minutes=59), "provider_sk": provider,
            "facility_id": provider, "approved_amount": round(total, 2), "member_sk": member, "tenant_id": TENANT}])
        self.w.append("authorization_line", [{
            "authorization_line_sk": f"{pa}-{j:02d}", "authorization_sk": pa, "activity_code": code,
            "approved_units": float(u), "approved_value": round(float(v), 2), "conditions": conditions,
            "denial_code": denial, "tenant_id": TENANT} for j, (code, (u, v)) in enumerate(sorted(codes.items()), 1)])
        return pa

    def link(self, line_sk: str, auth_id: str | None) -> None:
        lines = self.w.tables["claim_line"]
        self.w.set_values("claim_line", lines["line_sk"] == line_sk, authorization_id=auth_id)

    def auth_of(self, claim_sk: str) -> str | None:
        ln = self.lines(claim_sk)
        ids = ln["authorization_id"].dropna()
        return None if ids.empty else str(ids.iloc[0])

    def set_auth(self, auth_id: str, **values: Any) -> None:
        a = self.w.tables["authorization"]
        self.w.set_values("authorization", a["authorization_sk"] == auth_id, **values)

    def set_auth_lines(self, auth_id: str, **values: Any) -> None:
        a = self.w.tables["authorization_line"]
        self.w.set_values("authorization_line", a["authorization_sk"] == auth_id, **values)

    def dates(self, n: int, start: D, end: D) -> list[D]:
        span = (end - start).days
        picks = sorted(int(x) for x in self.rng.choice(span, size=n, replace=n > span))
        out = []
        for p in picks:
            d = start + TD(days=p)
            while d.weekday() >= 5:  # a weekday, like most planned outpatient care
                d += TD(days=1)
            out.append(d)
        return out

    def record(self, stratum: str, rule_ids: list[str], claims: Iterable[str | None], *,
               subject_type: str = "claim", subject_ids: Iterable[str] = (), positive: bool = True,
               note: str = "") -> None:
        claims = [c for c in claims if c]
        if not claims:
            return
        self.w.record(f"{FAMILY} {stratum}", rule_ids=rule_ids, claim_ids=claims, subject_type=subject_type,
                      subject_ids=list(subject_ids) or claims, positive=positive, note=note)


# ---------------------------------------------------------------------------
# the injector
# ---------------------------------------------------------------------------


def inject(world: World) -> None:
    k = _Kit(world)
    for plant in (_cross_payer, _bundling, _packages, _novel_pair, _codes_and_demographics, _units_and_time,
                  _authorisations, _modifiers, _post_edit_migration, _tariff):
        # A small build (the test suite uses ~1,500 claims) can run a pattern's
        # member or date pool dry part-way through. That is a size limit, not a
        # bug: skip the rest of that pattern, say so, and carry on. Anything it
        # planted before running dry was already recorded or is ordinary data.
        try:
            plant(k)
        except (StopIteration, IndexError) as exc:
            note = f"inject_pay_a: skipped the rest of '{plant.__name__}' - pool exhausted on a small build ({exc!r})"
            print(note)
            world.notes.append(note)


# PAY-01-R04 ------------------------------------------------------------------


def _cross_payer(k: _Kit) -> None:
    w = k.w
    header = w.tables["claim_header"]
    payers = sorted(header["payer_id"].dropna().unique())
    cob_members = {m for m, info in k.ctx.members.items() if info.get("cob")}
    rem = w.tables["remittance"]
    fully_paid = set(rem.groupby("claim_sk")["decision"].apply(lambda s: (s.astype(str).str.upper() == "PAID").all())
                     .loc[lambda s: s].index)

    def paid_outpatient(extra):
        return lambda h: (h["claim_type"] == "OUTPATIENT") & h["claim_sk"].isin(fully_paid) & \
            (h["gross_amount"].astype(float) > 300) & extra(h)

    # (a) paid in full by another insurer, no coordination record at all
    picks = w.take_claims(5, paid_outpatient(lambda h: ~h["member_sk"].isin(cob_members)))
    rows = []
    for h in picks.itertuples(index=False):
        other = [p for p in payers if p != h.payer_id][int(k.rng.integers(len(payers) - 1))]
        rows.append({"claim_sk": None, "other_payer_id": other, "payment_amount": round(float(h.net_amount), 2),
                     "settlement_date": pd.Timestamp(h.settlement_date) + pd.Timedelta(days=int(k.rng.integers(5, 40))),
                     "cross_payer_match_token": h.cross_payer_match_token, "tenant_id": TENANT})
    w.append("other_payer_remittance", rows)
    k.record("cross-payer duplicate: second insurer also paid, no coordination record", ["PAY-01-R04"],
             picks["claim_sk"])

    # (b) coordination on record, but the secondary insurer paid the whole bill, not the patient share
    picks = w.take_claims(2, paid_outpatient(lambda h: h["member_sk"].isin(cob_members)))
    opr = w.tables["other_payer_remittance"]
    for h in picks.itertuples(index=False):
        m = opr["cross_payer_match_token"] == h.cross_payer_match_token
        if m.any():
            w.set_values("other_payer_remittance", m, payment_amount=round(float(h.gross_amount), 2))
    k.record("cross-payer duplicate: secondary insurer paid the full bill", ["PAY-01-R04"], picks["claim_sk"])

    # look-alikes: coordination on record, secondary paid only the patient share (base behaviour)
    look = w.take_claims(2, paid_outpatient(lambda h: h["member_sk"].isin(cob_members)))
    k.record("cross-payer look-alike: coordinated payment within the bill", ["PAY-01-R04"], look["claim_sk"],
             positive=False, note="Exclusion: coordination-of-benefits data, combined payment within the bill.")


# PAY-02-R01 / R02 / PAY-05-R01 ------------------------------------------------------------


def _bundling(k: _Kit) -> None:
    labs = k.providers_where(2, "DIAGNOSTIC_LAB")
    clinics = k.providers_where(2, "CLINIC", "MULTISPECIALTY")
    gps = k.providers_where(2, "CLINIC", "GENERAL_PRACTICE")
    physio = k.providers_where(1, "CLINIC", "PHYSIOTHERAPY")
    members = k.members_where(24)
    dates = k.dates(24, D(2024, 9, 1), D(2025, 9, 30))
    mi = iter(zip(members, dates))

    # PAY-02-R01: a panel billed with one of its own components on the same claim
    same = []
    for i, pair in enumerate([("80053", "82947"), ("80053", "84450"), ("85025", "85027"), ("85025", "85048"),
                              ("80061", "83718"), ("80048", "84295")]):
        m, d = next(mi)
        same.append(k.claim(member_sk=m, provider_sk=labs[i % len(labs)], service_date=d, claim_type="LAB",
                            lines=[{"code": "36415"}, {"code": pair[0]}, {"code": pair[1]}]))
    k.record("unbundling: panel component billed beside its panel", ["PAY-02-R01"], same)

    # PAY-05-R01 (+ PAY-02-R01): modifier 59 used to split a pair whose edit forbids a modifier
    bypass = []
    for i, pair in enumerate([("80053", "80048"), ("93000", "93005"), ("85025", "85018"), ("80061", "82465")]):
        m, d = next(mi)
        prov = labs[i % len(labs)] if pair[0] != "93000" else gps[0]
        ctype = "LAB" if pair[0] != "93000" else "OUTPATIENT"
        lines = ([{"code": "36415"}] if ctype == "LAB" else [{"code": "99213"}]) + \
            [{"code": pair[0]}, {"code": pair[1], "indicator": "59"}]
        bypass.append(k.claim(member_sk=m, provider_sk=prov, service_date=d, claim_type=ctype, lines=lines))
    k.record("modifier 59 used to bypass an edit that allows no modifier", ["PAY-05-R01", "PAY-02-R01"], bypass)

    # look-alike: the edit allows a modifier, and modifier 59 is used (manual therapy + therapeutic activity)
    ok = []
    for _ in range(2):
        m, d = next(mi)
        ok.append(k.claim(member_sk=m, provider_sk=physio[0] if physio else gps[0], service_date=d,
                          claim_type="OUTPATIENT", diagnoses=["M54.50"],
                          lines=[{"code": "97140", "units": 2}, {"code": "97530", "units": 2, "indicator": "59"}]))
    k.record("bundling look-alike: permitted modifier separates the pair", ["PAY-02-R01", "PAY-05-R01"], ok,
             positive=False, note="Exclusion: the edit allows a modifier and one is present.")

    # PAY-02-R02: ECG complete at one provider, interpretation billed again by another the same day
    cross, look = [], []
    for i in range(6):
        m, d = next(mi)
        a = k.claim(member_sk=m, provider_sk=gps[i % len(gps)], service_date=d, claim_type="OUTPATIENT",
                    diagnoses=["R07.9"], lines=[{"code": "99213"}, {"code": "93000"}])
        ind = "26" if i >= 4 else None
        b = k.claim(member_sk=m, provider_sk=clinics[i % len(clinics)], service_date=d, claim_type="OUTPATIENT",
                    diagnoses=["R07.9"], lines=[{"code": "93010", "indicator": ind}])
        (look if ind else cross).append((a, b))
    k.record("cross-provider unbundling: ECG interpretation billed again", ["PAY-02-R02"],
             [b for _, b in cross], note="Parent ECG claims: " + ", ".join(str(a) for a, _ in cross))
    k.record("cross-provider look-alike: professional component split", ["PAY-02-R02"], [b for _, b in look],
             positive=False, note="Exclusion: professional/facility split (indicator 26).")


# PAY-02-R03 / R04 ------------------------------------------------------------------------


def _packages(k: _Kit) -> None:
    w = k.w
    lines = w.tables["claim_line"]
    ref = w.tables["activity_code_reference"].set_index("activity_code")
    pkg_claims = set(lines.loc[lines["activity_type"] == "DRG", "claim_sk"])
    zero = lines[(lines["gross_amount"].astype(float) == 0) & lines["claim_sk"].isin(pkg_claims)]
    cpt_zero = zero[zero["activity_type"] == "CPT"]
    eligible = set(cpt_zero["claim_sk"])

    # charged included component: 4 refused at remittance (prepay return), 4 paid on top (leakage)
    picks = w.take_claims(8, lambda h: h["claim_sk"].isin(eligible) & (h["claim_type"] == "INPATIENT"))
    returned, leaked = [], []
    for i, h in enumerate(picks.itertuples(index=False)):
        cand = cpt_zero[cpt_zero["claim_sk"] == h.claim_sk]
        cand = cand[cand["activity_code"].map(lambda c: str(ref.loc[c, "code_family"]) if c in ref.index else "")
                    .str.startswith(("LAB_", "IMAGING_", "EM_"))]
        if cand.empty:
            continue
        row = cand.iloc[0]
        price = k.contract_price(h.claim_sk, row["activity_code"], pd.Timestamp(row["service_date"]).date())
        paid = i >= 4
        k.reprice_line(row["line_sk"], price, paid=paid)
        (leaked if paid else returned).append(h.claim_sk)
    k.record("package: included component charged, refused at remittance", ["PAY-02-R03"], returned)
    k.record("package leakage: included component charged and paid", ["PAY-02-R03", "PAY-02-R04", "PAY-04-R04"],
             leaked, note="The charge also exceeds the zero value the approval covered.")

    # package billed without its itemised core activities
    picks = w.take_claims(4, lambda h: h["claim_sk"].isin(eligible) & (h["claim_type"] == "INPATIENT"))
    stripped = []
    for h in picks.itertuples(index=False):
        drop = zero[zero["claim_sk"] == h.claim_sk]["line_sk"]
        for table in ("claim_line", "remittance", "observation"):
            t = w.tables[table]
            if "line_sk" in t.columns:
                w.tables[table] = t[~t["line_sk"].isin(set(drop))].reset_index(drop=True)
        stripped.append(h.claim_sk)
    k.record("package incomplete: core activities not itemised", ["PAY-02-R03"], stripped[:3])
    # look-alike: the same omission with a documented medical reason
    if len(stripped) > 3:
        h = k.header(stripped[3])
        w.append("document", [{
            "document_sk": w.new_id("DOCP", 6), "claim_sk": h["claim_sk"], "member_sk": h["member_sk"],
            "provider_sk": h["provider_sk"], "doc_type": "MEDICAL_OMISSION_NOTE", "language": "en",
            "text": ("CLINICAL NOTE (SYNTHETIC) — the planned procedure was abandoned after induction for a "
                     "documented clinical reason; the package was billed without the procedure activities."),
            "ocr_confidence": 1.0, "created_at": pd.Timestamp(h["submission_date"]), "version_no": 1,
            "prior_document_sk": None, "attachment_ref": f"SYNTHETIC_omission_note_{h['claim_sk']}_en.txt",
            "is_synthetic": True, "tenant_id": TENANT}])
        k.record("package look-alike: documented medical omission", ["PAY-02-R03"], [stripped[3]], positive=False,
                 note="Exclusion: documented medical omission.")


# PAY-02-R05 --------------------------------------------------------------------------------


def _novel_pair(k: _Kit) -> None:
    gp = k.providers_where(1, "CLINIC", "GENERAL_PRACTICE")
    if not gp:
        return
    members = k.members_where(22)
    dates = k.dates(len(members), D(2024, 8, 1), D(2025, 11, 15))
    claims = [k.claim(member_sk=m, provider_sk=gp[0], service_date=d, claim_type="LAB", diagnoses=["R50.9"],
                      lines=[{"code": "85652"}, {"code": "86140"}]) for m, d in zip(members, dates)]
    k.record("novel unbundling: inflammation markers always billed as a separate pair", ["PAY-02-R05"], claims,
             subject_type="provider", subject_ids=gp)


# PAY-03-R01 / R02 --------------------------------------------------------------------------


def _codes_and_demographics(k: _Kit) -> None:
    w = k.w
    gps = k.providers_where(2, "CLINIC", "GENERAL_PRACTICE")
    labs = k.providers_where(1, "DIAGNOSTIC_LAB")
    radiology = k.providers_where(1, "RADIOLOGY_CENTRE")
    obgyn = k.providers_where(1, "CLINIC", "OBSTETRICS_GYNAECOLOGY") or gps

    # a code deleted in 2021 billed in 2025
    members = k.members_where(4)
    dates = k.dates(4, D(2025, 1, 5), D(2025, 9, 30))
    old = [k.claim(member_sk=m, provider_sk=gps[i % len(gps)], service_date=d, claim_type="OUTPATIENT",
                   diagnoses=["J06.9"], lines=[{"code": "99201"}]) for i, (m, d) in enumerate(zip(members, dates))]
    k.record("invalid code: deleted code billed after its end date", ["PAY-03-R01"], old)

    # a code that exists in no code set (a mistyped visit code)
    picks = w.take_claims(2, lambda h: (h["claim_type"] == "OUTPATIENT"))
    unknown = []
    lines = w.tables["claim_line"]
    for h in picks.itertuples(index=False):
        ln = lines[(lines["claim_sk"] == h.claim_sk) & lines["activity_code"].astype(str).str.startswith("992")]
        if ln.empty:
            continue
        w.set_values("claim_line", lines["line_sk"] == ln.iloc[0]["line_sk"], activity_code="99299",
                     activity_description="Office visit (code not in the code set)")
        unknown.append(h.claim_sk)
    k.record("invalid code: code absent from the code set", ["PAY-03-R01"], unknown)

    # PAY-03-R02: sex-restricted and age-restricted services
    demo = []
    for code, sex, age, ctype, prov in (("84702", "M", (25, 60), "LAB", labs), ("84153", "F", (45, 70), "LAB", labs),
                                        ("76801", "M", (25, 50), "RADIOLOGY", radiology),
                                        ("77067", "F", (22, 30), "RADIOLOGY", radiology),
                                        ("84153", "M", (20, 30), "LAB", labs)):
        m = k.members_where(1, sex=sex, age=age)
        if not m or not prov:
            continue
        d = k.dates(1, D(2024, 10, 1), D(2025, 9, 30))[0]
        lines = ([{"code": "36415"}] if ctype == "LAB" else []) + [{"code": code}]
        demo.append(k.claim(member_sk=m[0], provider_sk=prov[0], service_date=d, claim_type=ctype, lines=lines))
    k.record("demographic impossibility: service outside its sex or age restriction", ["PAY-03-R02"], demo)

    # look-alike: the same kind of service with an approved clinical exception on file
    m = k.members_where(1, sex="M", age=(25, 50))
    if m and obgyn:
        d = D(2025, 4, 14)
        c = k.claim(member_sk=m[0], provider_sk=obgyn[0], service_date=d, claim_type="OUTPATIENT",
                    lines=[{"code": "99213"}, {"code": "76856"}])
        if c:
            ln = k.line_of(c, "76856")
            pa = k.add_auth(member=m[0], provider=obgyn[0], codes={"76856": (1, float(ln["gross_amount"]) * 1.1)},
                            status="APPROVED", valid_from=d - TD(days=5), valid_to=d + TD(days=30),
                            conditions="Clinical exception approved by medical policy (gender-affirming care)")
            k.link(ln["line_sk"], pa)
            k.record("demographic look-alike: approved clinical exception", ["PAY-03-R02"], [c], positive=False,
                     note="Exclusion: clinical exception approved by policy.")


# PAY-03-R03 / R04 --------------------------------------------------------------------------


def _units_and_time(k: _Kit) -> None:
    w = k.w
    physio = k.providers_where(2, "CLINIC", "PHYSIOTHERAPY")
    gps = k.providers_where(1, "CLINIC", "GENERAL_PRACTICE")
    psych = k.providers_where(1, "CLINIC", "PSYCHIATRY")
    members = k.members_where(16)
    dates = k.dates(16, D(2024, 9, 1), D(2025, 9, 30))
    mi = iter(zip(members, dates))

    units = []
    for code, n, prov, dx in (("97110", 7, physio, "M54.50"), ("97140", 4, physio, "M54.50"),
                              ("36415", 4, gps, "R50.9"), ("97530", 6, physio, "M25.561")):
        m, d = next(mi)
        units.append(k.claim(member_sk=m, provider_sk=prov[0], service_date=d, claim_type="OUTPATIENT",
                             diagnoses=[dx], lines=[{"code": code, "units": n}]))
    k.record("unit maximum: units above the daily maximum", ["PAY-03-R03"], units)
    # split across two claims from one provider on the same day
    m, d = next(mi)
    split = [k.claim(member_sk=m, provider_sk=physio[-1], service_date=d, claim_type="OUTPATIENT",
                     diagnoses=["M54.50"], lines=[{"code": "97110", "units": 3}], start_minute=start)
             for start in (9 * 60, 15 * 60)]
    k.record("unit maximum: daily units split over two claims", ["PAY-03-R03"], split)
    # look-alike: repeat-procedure indicator
    m, d = next(mi)
    rep = k.claim(member_sk=m, provider_sk=physio[0], service_date=d, claim_type="OUTPATIENT", diagnoses=["M54.50"],
                  lines=[{"code": "97110", "units": 4}, {"code": "97110", "units": 2, "indicator": "76"}])
    k.record("unit look-alike: repeat procedure indicator", ["PAY-03-R03"], [rep], positive=False,
             note="Exclusion: repeat/laterality indicator.")

    # PAY-03-R04: billed time longer than the recorded window
    timed = []
    lines = w.tables["claim_line"]
    for code, n, keep in (("97110", 4, 25), ("97140", 2, 12), ("97530", 3, 20), ("90837", 1, 35)):
        m, d = next(mi)
        prov = psych if code == "90837" and psych else physio
        c = k.claim(member_sk=m, provider_sk=prov[0], service_date=d, claim_type="OUTPATIENT",
                    diagnoses=["F41.1" if code == "90837" else "M54.50"], lines=[{"code": code, "units": n}])
        if c:
            ln = k.line_of(c, code)
            end = pd.Timestamp(ln["service_start_time"]) + pd.Timedelta(minutes=keep)
            w.set_values("claim_line", w.tables["claim_line"]["line_sk"] == ln["line_sk"], service_end_time=end)
            timed.append(c)
    k.record("time units: billed minutes exceed the recorded window", ["PAY-03-R04"], timed)
    # look-alike: within the rounding tolerance
    m, d = next(mi)
    c = k.claim(member_sk=m, provider_sk=physio[0], service_date=d, claim_type="OUTPATIENT", diagnoses=["M54.50"],
                lines=[{"code": "97110", "units": 4}])
    if c:
        ln = k.line_of(c, "97110")
        w.set_values("claim_line", w.tables["claim_line"]["line_sk"] == ln["line_sk"],
                     service_end_time=pd.Timestamp(ln["service_start_time"]) + pd.Timedelta(minutes=55))
        k.record("time look-alike: within the rounding tolerance", ["PAY-03-R04"], [c], positive=False,
                 note="Exclusion: rounding policy.")
    # one psychotherapist in two sessions at once
    if psych:
        clin = [c for c in k.ctx.clinicians.get(psych[0], []) if "PSYCHOTHERAPY" in c.get("privileges", set())]
        c0 = (clin or k.ctx.clinicians[psych[0]])[0]
        clin_id = c0["id"]
        m1, d = next(mi)
        m2, _ = next(mi)
        while any(a <= d <= b for a, b in c0.get("leave", [])) or d.weekday() >= 5:
            d += TD(days=1)
        pair = [k.claim(member_sk=mm, provider_sk=psych[0], service_date=d, claim_type="OUTPATIENT",
                        diagnoses=["F41.1"], clinician_id=clin_id, start_minute=10 * 60 + off,
                        lines=[{"code": "90837", "clinician": clin_id}]) for mm, off in ((m1, 0), (m2, 20))]
        k.record("time units: one clinician in two timed sessions at once", ["PAY-03-R04"], pair[1:],
                 note=f"Overlaps claim {pair[0]}.")


# PAY-04 --------------------------------------------------------------------------------


def _authorisations(k: _Kit) -> None:
    w = k.w
    rad = k.providers_where(3, "RADIOLOGY_CENTRE")
    hosp = k.providers_where(1, "HOSPITAL")
    members = k.members_where(30)
    dates = k.dates(30, D(2024, 9, 1), D(2025, 9, 30))
    mi = iter(zip(members, dates))

    def mri(m, d, prov, code="73721", dx="M23.211", **kw):
        return k.claim(member_sk=m, provider_sk=prov, service_date=d, claim_type="RADIOLOGY", diagnoses=[dx],
                       lines=[{"code": code}], **kw)

    # R01: advanced imaging with no approval at all
    missing = []
    for i in range(5):
        m, d = next(mi)
        missing.append(mri(m, d, rad[i % len(rad)], authorization=False))
    k.record("prior approval missing for advanced imaging", ["PAY-04-R01"], missing)
    # look-alike: emergency CT, no approval needed in advance
    m, d = next(mi)
    em = k.claim(member_sk=m, provider_sk=hosp[0] if hosp else rad[0], service_date=d, claim_type="OUTPATIENT",
                 encounter_type="EMERGENCY", diagnoses=["R51.9"], lines=[{"code": "99284"}, {"code": "70450"}],
                 authorization=False)
    k.record("approval look-alike: emergency service", ["PAY-04-R01"], [em], positive=False,
             note="Exclusion: emergency.")

    # R02: approval refused, or service after the approval expired
    bad = []
    for i in range(5):
        m, d = next(mi)
        c = mri(m, d, rad[i % len(rad)])
        pa = k.auth_of(c) if c else None
        if not pa:
            continue
        if i < 2:
            k.set_auth(pa, status="DENIED")
            k.set_auth_lines(pa, denial_code="MNEC-004")
        else:
            k.set_auth(pa, valid_from=pd.Timestamp(d - TD(days=70)), valid_to=pd.Timestamp(d - TD(days=8)))
        bad.append(c)
    k.record("approval refused, or service after the approval expired", ["PAY-04-R02"], bad)
    # look-alike: the first approval expired but an extension covers the date
    m, d = next(mi)
    c = mri(m, d, rad[0])
    pa = k.auth_of(c) if c else None
    if pa:
        k.set_auth(pa, valid_from=pd.Timestamp(d - TD(days=70)), valid_to=pd.Timestamp(d - TD(days=8)))
        ln = k.line_of(c, "73721")
        k.add_auth(member=m, provider=rad[0], codes={"73721": (1, float(ln["gross_amount"]) * 1.1)},
                   status="APPROVED", valid_from=d - TD(days=9), valid_to=d + TD(days=30),
                   conditions="Extension of an earlier approval")
        k.record("approval look-alike: approved extension", ["PAY-04-R02"], [c], positive=False,
                 note="Exclusion: approved extension.")

    # R03: approval for another code, or issued to another provider
    scope = []
    for i in range(5):
        m, d = next(mi)
        c = mri(m, d, rad[i % len(rad)])
        pa = k.auth_of(c) if c else None
        if not pa:
            continue
        if i < 3:
            k.set_auth_lines(pa, activity_code="73221")  # approved: shoulder MRI; billed: knee MRI
        else:
            other = rad[(i + 1) % len(rad)]
            k.set_auth(pa, provider_sk=other, facility_id=other)
        scope.append(c)
    k.record("approval scope: another code or another provider", ["PAY-04-R03"], scope)
    # look-alike: approved code equivalent to the billed one
    m, d = next(mi)
    c = mri(m, d, rad[0], code="70553", dx="G43.909")
    pa = k.auth_of(c) if c else None
    if pa:
        k.set_auth_lines(pa, activity_code="70551")
        k.record("approval look-alike: equivalent code", ["PAY-04-R03"], [c], positive=False,
                 note="Exclusion: equivalent-code map.")

    # R04: a specialty-drug course running past the approved units
    rheum = k.providers_where(2, "CLINIC", "MULTISPECIALTY")
    course_members = k.members_where(3, age=(30, 60))
    over, ok = [], []
    for j, m in enumerate(course_members):
        start = D(2024, 10, 7) + TD(days=21 * j)
        claims = [k.claim(member_sk=m, provider_sk=rheum[j % len(rheum)], service_date=start + TD(days=14 * i),
                          claim_type="OUTPATIENT", diagnoses=["M05.79"], authorization=False,
                          lines=[{"code": "96372"}, {"code": "J0135", "units": 2}]) for i in range(5)]
        claims = [c for c in claims if c]
        if len(claims) < 4:
            continue
        drug = [k.line_of(c, "J0135") for c in claims]
        value = sum(float(x["gross_amount"]) for x in drug)
        pa = k.add_auth(member=m, provider=rheum[j % len(rheum)], codes={"J0135": (6, value * 0.62)},
                        status="APPROVED", valid_from=start - TD(days=3), valid_to=start + TD(days=90))
        for x in drug:
            k.link(x["line_sk"], pa)
        if j < 2:
            over += claims[3:]
        else:  # look-alike: the doses beyond the approval were refused, so nothing past it was consumed
            for x in drug[3:]:
                k.set_remittance(x["line_sk"], "DENIED", "MNEC-003")
            ok.append(claims[-1])
    k.record("approval exhausted: drug course beyond the approved units", ["PAY-04-R04"], over)
    k.record("approval look-alike: doses beyond the approval refused", ["PAY-04-R04"], ok, positive=False,
             note="Exclusion: refused lines consume nothing.")

    # R05: one approval reused for other patients
    reuse = []
    for i in range(2):
        m, d = next(mi)
        c = mri(m, d, rad[i % len(rad)])
        pa = k.auth_of(c) if c else None
        if not pa:
            continue
        for extra in range(2):
            m2, _ = next(mi)
            c2 = mri(m2, d + TD(days=3 + 4 * extra), rad[i % len(rad)], authorization=False)
            if c2:
                k.link(k.line_of(c2, "73721")["line_sk"], pa)
                reuse.append(c2)
    k.record("approval reused for other patients", ["PAY-04-R05", "PAY-04-R04"], reuse,
             note="Each reuse also consumes units the approval never granted.")


# PAY-05-R02 / R03 --------------------------------------------------------------------------


def _modifiers(k: _Kit) -> None:
    w = k.w
    # R02: modifier 22 (increased procedural service) with no operative report
    derm = k.providers_where(1, "CLINIC", "DERMATOLOGY") + k.providers_where(1, "CLINIC", "GENERAL_PRACTICE")
    members = k.members_where(6)
    dates = k.dates(6, D(2024, 10, 1), D(2025, 8, 31))
    no_doc = []
    for i, (m, d) in enumerate(zip(members, dates)):
        c = k.claim(member_sk=m, provider_sk=derm[i % len(derm)], service_date=d, claim_type="OUTPATIENT",
                    diagnoses=["L02.91"], documents=False,
                    lines=[{"code": "99213"}, {"code": "10060", "indicator": "22"}])
        if not c:
            continue
        if i < 5:
            no_doc.append(c)
        else:  # look-alike: the operative report is attached to the line
            ln = k.line_of(c, "10060")
            w.append("observation", [{
                "observation_sk": w.new_id("OBSP", 7), "line_sk": ln["line_sk"], "observation_type": "REPORT",
                "value": "Operative report (SYNTHETIC): multiloculated abscess, extended drainage time.",
                "unit": None, "attachment_ref": f"SYNTHETIC_operative_report_{c}_en.txt",
                "event_time": pd.Timestamp(ln["service_end_time"]) + pd.Timedelta(hours=2), "tenant_id": TENANT,
                "observation_code": "10060", "claim_sk": c, "member_sk": m}])
            k.record("modifier look-alike: supporting report attached", ["PAY-05-R02"], [c], positive=False,
                     note="The required report is present.")
    k.record("modifier 22 without the supporting report", ["PAY-05-R02"], no_doc)

    # R03: two clinics appending modifier 25 to most visits
    header = w.tables["claim_header"]
    busy = header[header["claim_type"] == "OUTPATIENT"].groupby("provider_sk").size()
    busy = set(busy[busy >= 150].index)
    clinics = w.take_entities("provider", "provider_sk", 2, where=lambda f: (f["provider_type"] == "CLINIC") &
                              f["provider_sk"].isin(busy))["provider_sk"].tolist()
    lines = w.tables["claim_line"]
    for p in clinics:
        picks = w.take_claims(400, lambda h, p=p: (h["provider_sk"] == p) & (h["claim_type"] == "OUTPATIENT"))
        em = lines[lines["claim_sk"].isin(set(picks["claim_sk"])) &
                   lines["activity_code"].astype(str).str.match(r"^992\d\d$") & lines["indicator"].isna()]
        em = em.drop_duplicates("claim_sk")
        chosen = em.sample(frac=0.7, random_state=int(k.rng.integers(1 << 31)))
        w.set_values("claim_line", lines["line_sk"].isin(set(chosen["line_sk"])), indicator="25")
        k.record("modifier 25 appended to most visits", ["PAY-05-R03"], chosen["claim_sk"],
                 subject_type="provider", subject_ids=[p])


# PAY-05-R04 --------------------------------------------------------------------------------


def _post_edit_migration(k: _Kit) -> None:
    """A new edit bundles the injection into minor incision-and-drainage from 1 March 2025.

    The edit is appended to the bundling table (a policy change inside the period). One
    clinic billed the injection plainly before it; afterwards its plain injections are
    refused, and it then adds modifier 59 to every one, which gets them paid.
    """
    w = k.w
    launch = D(2025, 3, 1)
    w.append("bundling_edit_table", [{"column_1_code": "10060", "column_2_code": "96372",
                                      "edit_type": "COMPREHENSIVE_COMPONENT", "modifier_allowed": True,
                                      "valid_from": launch, "valid_to": D(2099, 12, 31)}])
    clinic = k.providers_where(1, "CLINIC", "GENERAL_PRACTICE")
    if not clinic:
        return
    p = clinic[0]
    members = k.members_where(16)
    pre_d = k.dates(5, D(2024, 11, 5), D(2025, 2, 25))
    denied_d = [D(2025, 3, 6), D(2025, 3, 19)]
    post_d = k.dates(9, D(2025, 4, 7), D(2025, 7, 25))
    mi = iter(members)

    def visit(d, indicator=None):
        return k.claim(member_sk=next(mi), provider_sk=p, service_date=d, claim_type="OUTPATIENT",
                       diagnoses=["L02.91"],
                       lines=[{"code": "10060"}, {"code": "96372", "indicator": indicator}])

    pre = [visit(d) for d in pre_d]
    denied = []
    for d in denied_d:
        c = visit(d)
        if c:
            k.set_remittance(k.line_of(c, "96372")["line_sk"], "DENIED", "CLAI-012")
            denied.append(c)
    post = [visit(d, "59") for d in post_d]
    k.record("post-edit migration: modifier 59 after a new edit", ["PAY-05-R04"], [c for c in post if c],
             subject_type="provider", subject_ids=[p],
             note="Pre-edit claims: " + ", ".join(str(c) for c in pre if c))
    k.record("post-edit: plain component billed with its parent after the edit", ["PAY-02-R01"], denied,
             note="Refused at remittance under the new edit; part of the PAY-05-R04 pattern.")


# PAY-06-R01 --------------------------------------------------------------------------------


def _tariff(k: _Kit) -> None:
    w = k.w
    lines = w.tables["claim_line"]
    rem = w.tables["remittance"]
    paid_lines = set(rem.loc[rem["decision"].astype(str).str.upper() == "PAID", "line_sk"])
    priced = lines[(lines["gross_amount"].astype(float) > 50) & lines["line_sk"].isin(paid_lines) &
                   (lines["activity_type"].isin(["CPT"]))]
    eligible = set(priced["claim_sk"])
    picks = w.take_claims(8, lambda h: h["claim_sk"].isin(eligible) & h["claim_type"].isin(["OUTPATIENT", "LAB"]))
    over, near = [], []
    for i, h in enumerate(picks.itertuples(index=False)):
        row = priced[priced["claim_sk"] == h.claim_sk].iloc[0]
        base = float(row["unit_price"])
        factor = 1.004 if i >= 6 else float(k.rng.uniform(1.18, 1.45))
        k.reprice_line(row["line_sk"], round(base * factor, 2))
        (near if i >= 6 else over).append(h.claim_sk)
    k.record("price above the contracted tariff", ["PAY-06-R01"], over)
    k.record("price look-alike: within the rounding tolerance", ["PAY-06-R01"], near, positive=False,
             note="Exclusion: rounding tolerance.")
