"""Plant the PHR (pharmacy, drugs and devices) patterns in the SYNTHETIC UAE demo dataset.

Every control implemented in ``src/fwa/engine/unlocked/phr.py`` gets at least one
realistic pattern here, at a modest volume, and every planted item is written to
the answer key with the rule ids it is meant to exercise. Where a declared
exclusion has a legitimate look-alike, one is planted too (``positive=False``) so
the exclusion is demonstrated.

Patterns (all SYNTHETIC):

* PHR-01-R01  a different, non-equivalent (dearer) medicine billed than the one prescribed;
              look-alike: a generic billed for a prescribed brand (approved equivalent);
* PHR-01-R02  three times the prescribed quantity billed; look-alike: a whole-pack rounding;
* PHR-01-R03  brand billed while the pharmacy's own record shows the generic handed over,
              fewer tablets dispensed than billed, and a month in which one pharmacy bills far
              more of a product than its stock records could supply;
* PHR-01-R04  receipts showing perfume, protein powder, skincare … against a billed medicine;
* PHR-02-R01  a chronic medicine refilled every ten days on a thirty-day supply (stockpiling);
              look-alike: an early refill with a dose increase;
* PHR-02-R02  a proton-pump inhibitor or tramadol continued for months on one prescription;
              look-alike: the same course with a specialist approval on file;
* PHR-02-R03  pregabalin from three or four unrelated prescribers within ninety days;
              look-alike: three prescribers of one clinic (a care team);
* PHR-02-R04  alprazolam collected from three pharmacies within ten days;
              look-alike: a prescription split into partial fills across pharmacies;
* PHR-03-R01  a daily dose two to three times the mg/kg limit for the patient's weight;
              look-alike: a short first (loading) course;
* PHR-03-R02  sitagliptin with no metformin first; isotretinoin with no acne diagnosis;
* PHR-03-R04  one hospital billing two extra vials of wastage on every infliximab infusion;
* PHR-04-R01  a GP whose prescriptions nearly all go to one pharmacy;
* PHR-04-R02  a clinic sending most prescriptions to a pharmacy that shares its bank account
              and phone line; look-alike: a declared same-owner pharmacy;
* PHR-04-R03  a specialist routing every adalimumab prescription to one pharmacy while other
              prescriptions spread normally;
* PHR-04-R04  a GP's prescriptions switching abruptly to an established pharmacy they never used;
* PHR-05-R01  used or rented equipment billed with the new-purchase modifier (NU);
              look-alike: a rent-to-own device;
* PHR-05-R02  implants billed with no procedure, or on the opposite side to the procedure;
              look-alike: a staged implant after an earlier procedure;
* PHR-05-R03  test strips billed ten times the usual quantity; crutches re-issued within weeks;
              look-alike: a child re-issued a device (growth);
* PHR-05-R04  one equipment serial billed for two patients.
"""

from __future__ import annotations

import datetime as _dt
from typing import Any

import numpy as np
import pandas as pd

from .world import World

D = _dt.date
TD = _dt.timedelta
TENANT = "T001"
CAPITALS = ("Dubai", "Abu Dhabi", "Sharjah")


class _Kit:
    def __init__(self, world: World) -> None:
        self.w = world
        self.ctx = world.context["base_ctx"]
        self.rng = world.rng
        self.drugs = world.context["drugs"]
        self.icd = world.context["icd"]
        mem = world.tables["member"].copy()
        mem["dob"] = pd.to_datetime(mem["date_of_birth"], errors="coerce")
        self.mem = mem
        prov = world.tables["provider"]
        self.prov = prov.set_index("provider_sk")
        roster = world.tables["clinician_roster"]
        self.roster = roster.drop_duplicates("clinician_id").set_index("clinician_id")
        rx = world.tables.get("prescription_dispense", pd.DataFrame())
        self.rx_count = rx.groupby("prescriber_id").size().to_dict() if not rx.empty else {}
        self.own_pharmacies: list[str] = []
        self.member_groups: dict[str, set[str]] = {}
        if not rx.empty:
            grp = rx["billed_product"].map(lambda p: self.drugs.get(p, {}).get("equivalence_group"))
            for m, g in zip(rx["member_sk"], grp):
                self.member_groups.setdefault(m, set()).add(g)

    # ------------------------------------------------------------ entities

    def members(self, n: int, *, age: tuple[float, float] = (25, 60), start: D = D(2024, 8, 1),
                end: D = D(2025, 12, 15), avoid_groups: tuple[str, ...] = (),
                emirate: str | None = None) -> list[str]:
        ctx = self.ctx
        mem = self.mem

        def ok(f: pd.DataFrame) -> pd.Series:
            sub = mem.loc[f.index]
            a = (pd.Timestamp(start) - sub["dob"]).dt.days / 365.25
            m = a.between(*age) & sub["weight_kg"].notna()
            if emirate:
                m &= sub["emirate"] == emirate
            cov = sub["member_sk"].map(lambda s: bool(ctx.coverage_at(s, start)) and bool(ctx.coverage_at(s, end))
                                       and not ctx.members.get(s, {}).get("cob"))
            clean = sub["member_sk"].map(lambda s: not (self.member_groups.get(s, set()) & set(avoid_groups)))
            return m & cov & clean

        return self.w.take_entities("member", "member_sk", n, where=ok)["member_sk"].tolist()

    def pharmacies(self, n: int, emirate: str | None = None, exclude: tuple[str, ...] = ()) -> list[str]:
        got = self._pharmacies(n, emirate, exclude)
        if len(got) < n and emirate:
            got += self._pharmacies(n - len(got), None, exclude + tuple(got))
        # small builds: reuse pharmacies this injector already took (never another injector's)
        for ph in self.own_pharmacies:
            if len(got) >= n:
                break
            if ph not in got and ph not in exclude:
                got.append(ph)
        self.own_pharmacies += [ph for ph in got if ph not in self.own_pharmacies]
        return got

    def _pharmacies(self, n: int, emirate: str | None = None, exclude: tuple[str, ...] = ()) -> list[str]:
        def ok(f: pd.DataFrame) -> pd.Series:
            m = (f["provider_type"] == "PHARMACY") & ~f["provider_sk"].isin(exclude)
            m &= f["provider_sk"].map(lambda p: bool(self.ctx.clinicians.get(p)))
            if emirate:
                m &= f["emirate"] == emirate
            return m
        return self.w.take_entities("provider", "provider_sk", n, where=ok)["provider_sk"].tolist()

    def providers(self, n: int, ptype: str, emirate: str | None = None, max_rx: int | None = None) -> list[str]:
        got: list[str] = []
        for em, cap in ((emirate, max_rx), (None, max_rx), (None, None if max_rx is None else max_rx * 5)):
            if len(got) >= n:
                break
            got += self._providers(n - len(got), ptype, em, cap)
        return got

    def _providers(self, n: int, ptype: str, emirate: str | None = None, max_rx: int | None = None) -> list[str]:
        prescribing = {"GENERAL_PRACTICE", "FAMILY_MEDICINE", "INTERNAL_MEDICINE", "PAEDIATRICS"}
        clin_rx: dict[str, int] = {}
        for cid, cnt in self.rx_count.items():
            p = self.roster["provider_sk"].get(cid)
            if p is not None:
                clin_rx[p] = clin_rx.get(p, 0) + cnt

        def ok(f: pd.DataFrame) -> pd.Series:
            m = (f["provider_type"] == ptype) & f["provider_sk"].map(lambda p: bool(self.ctx.clinicians.get(p)))
            if ptype == "CLINIC":
                m &= f["provider_sk"].map(lambda p: sum(
                    c.get("specialty") in prescribing for c in self.ctx.clinicians.get(p, [])) >= 2)
            if emirate:
                m &= f["emirate"] == emirate
            if max_rx is not None:
                m &= f["provider_sk"].map(lambda p: clin_rx.get(p, 0) <= max_rx)
            return m
        return self.w.take_entities("provider", "provider_sk", n, where=ok)["provider_sk"].tolist()

    def prescribers(self, n: int, *, specialties: tuple[str, ...] = ("GENERAL_PRACTICE", "FAMILY_MEDICINE"),
                    emirate: str | None = None, max_rx: int = 5, provider: str | None = None,
                    distinct_owners: bool = False) -> list[str]:
        got: list[str] = []
        for em, cap in ((emirate, max_rx), (None, max_rx), (None, max_rx * 5)):
            if len(got) >= n:
                break
            got += self._prescribers(n - len(got), specialties=specialties, emirate=em, max_rx=cap,
                                     provider=provider, distinct_owners=distinct_owners)
        return got

    def _prescribers(self, n: int, *, specialties: tuple[str, ...], emirate: str | None, max_rx: int,
                     provider: str | None, distinct_owners: bool) -> list[str]:
        prov = self.prov

        def ok(f: pd.DataFrame) -> pd.Series:
            m = f["specialty"].isin(specialties)
            m &= f["provider_sk"].map(lambda p: prov["provider_type"].get(p) == "CLINIC")
            m &= f["clinician_id"].map(lambda c: self.rx_count.get(c, 0) <= max_rx)
            m &= f["clinician_id"].map(lambda c: c in self.ctx.clin_by_id)
            if emirate:
                m &= f["emirate"] == emirate
            if provider:
                m &= f["provider_sk"] == provider
            return m
        if not distinct_owners:
            return self.w.take_entities("clinician_roster", "clinician_id", n, where=ok)["clinician_id"].tolist()
        out: list[str] = []
        owners: set[str] = set()
        for _ in range(n * 6):
            if len(out) >= n:
                break
            got = self.w.take_entities("clinician_roster", "clinician_id", 1, where=ok)
            if got.empty:
                break
            cid = got["clinician_id"].iloc[0]
            owner = prov["owner_entity_id"].get(got["provider_sk"].iloc[0])
            if owner in owners:
                continue
            owners.add(owner)
            out.append(cid)
        return out

    def free_day(self, clinician: str | None, d: D) -> D:
        """The first day on or after ``d`` the prescriber is licensed and not on leave."""
        c = self.ctx.clin_by_id.get(clinician or "")
        if not c:
            return d
        for _ in range(60):
            on_leave = any(a <= d <= b for a, b in c.get("leave", []))
            licensed = c.get("lic_from", D(1900, 1, 1)) <= d <= c.get("lic_to", D(2099, 1, 1))
            if not on_leave and licensed:
                return d
            d += TD(days=1)
        return d

    def written_on(self, clinician: str | None, d: D) -> D:
        """The last day on or before ``d`` the prescriber was licensed and not on leave."""
        c = self.ctx.clin_by_id.get(clinician or "")
        if not c:
            return d
        for _ in range(90):
            on_leave = any(a <= d <= b for a, b in c.get("leave", []))
            licensed = c.get("lic_from", D(1900, 1, 1)) <= d <= c.get("lic_to", D(2099, 1, 1))
            if not on_leave and licensed:
                return d
            d -= TD(days=1)
        return d

    def dx_for(self, product: str) -> str:
        prefixes = [p for p in (self.drugs[product].get("indication_prefixes") or "").split(";") if p]
        for p in prefixes:
            codes = sorted(c for c in self.icd if c.replace(".", "").startswith(p))
            if codes:
                return codes[0]
        return "R52"

    # --------------------------------------------------------------- claims

    def claim(self, **kw: Any) -> str | None:
        kw.setdefault("strict", False)
        try:
            return self.w.make_claim(**kw)
        except Exception:
            return None

    def fill(self, member: str, pharmacy: str, d: D, product: str, *, qty: float, days: float,
             prescriber: str, prescribed: str | None = None, dispensed: str | None = None,
             presc_qty: float | None = None, disp_qty: float | None = None, dose: float | None = None,
             prescribed_date: D | None = None, dx: str | None = None, wastage: float | None = None) -> str | None:
        drug = self.drugs[product]
        if dose is None and drug.get("regimen"):
            dose = float(drug["regimen"][0] * drug["strength_mg"])
        rx = {"prescriber": prescriber, "prescribed_date": self.written_on(prescriber, prescribed_date or d),
              "days_supply": float(days),
              "prescribed_qty": float(presc_qty if presc_qty is not None else qty),
              "dispensed_qty": float(disp_qty if disp_qty is not None else qty), "dose_mg_per_day": dose}
        if prescribed:
            rx["prescribed_product"] = prescribed
        if dispensed:
            rx["dispensed_product"] = dispensed
        line: dict[str, Any] = {"code": product, "units": float(qty), "rx": rx}
        if wastage is not None:
            line["wastage"] = wastage
        return self.claim(member_sk=member, provider_sk=pharmacy, service_date=d, claim_type="PHARMACY",
                          lines=[line], diagnoses=[dx or self.dx_for(product)], ordering_clinician_id=prescriber)

    def rx_of(self, claim_sk: str) -> pd.DataFrame:
        rx = self.w.tables["prescription_dispense"]
        return rx[rx["claim_sk"] == claim_sk]

    def add_auth(self, *, member: str, provider: str, code: str, units: float, value: float,
                 start: D, end: D, conditions: str) -> str:
        pa = self.ctx.unique(self.ctx.auth_ids, "PA", 8)
        self.w.append("authorization", [{
            "authorization_sk": pa, "request_id": f"REQ{pa[2:]}", "response_id": f"RSP{pa[2:]}",
            "status": "APPROVED", "valid_from": pd.Timestamp(start), "valid_to": pd.Timestamp(end) + pd.Timedelta(hours=23),
            "provider_sk": provider, "facility_id": provider, "approved_amount": round(value, 2),
            "member_sk": member, "tenant_id": TENANT}])
        self.w.append("authorization_line", [{
            "authorization_line_sk": f"{pa}-01", "authorization_sk": pa, "activity_code": code,
            "approved_units": float(units), "approved_value": round(value, 2), "conditions": conditions,
            "denial_code": None, "tenant_id": TENANT}])
        return pa


def _dated(kit: _Kit, prescriber: str, d: D) -> D:
    """Fill dates are kept as planned; ``fill`` dates the prescription on a working day before it."""
    return d


def _skip(kit: _Kit, pattern: str, why: str) -> None:
    """A pattern whose pool is empty on this (small) build is skipped, with a printed note."""
    msg = f"inject_phr: skipped '{pattern}' — {why}"
    print(msg)
    kit.w.notes.append(msg)


def _need(kit: _Kit, pattern: str, **pools: list) -> bool:
    """True when every named pool has at least one entry; otherwise note the skip."""
    empty = [name for name, pool in pools.items() if not pool]
    if empty:
        _skip(kit, pattern, "no eligible " + ", ".join(empty) + " left on this build")
        return False
    return True


def _cycle(pool: list, i: int):
    return pool[i % len(pool)]


# ---------------------------------------------------------------------------
# PHR-01
# ---------------------------------------------------------------------------


def _phr01(kit: _Kit) -> None:
    w = kit.w
    pharms = kit.pharmacies(2, "Dubai")
    doctors = kit.prescribers(3, emirate="Dubai", max_rx=40)
    if not _need(kit, "PHR-01 (all patterns)", pharmacies=pharms, prescribers=doctors):
        return
    ph_a, ph_b = pharms[0], _cycle(pharms, 1)

    # R01 — a different, dearer, non-equivalent product billed than prescribed
    pairs = [("RX1029", "RX1031"), ("RX1029", "RX1031"), ("RX1029", "RX1031"),
             ("RX1052", "RX1053"), ("RX1052", "RX1053"), ("RX1020", "RX1024")]
    members = kit.members(len(pairs) + 2, avoid_groups=("OMEPRAZOLE", "ESOMEPRAZOLE", "SERTRALINE",
                                                        "ESCITALOPRAM", "AMLODIPINE", "BISOPROLOL",
                                                        "ATORVASTATIN"))
    if _need(kit, "PHR-01-R01 non-equivalent product", members=members):
        planted, spare = members[:max(1, len(members) - 2)], members[max(1, len(members) - 2):]
        got = []
        for i, (m, (presc, billed)) in enumerate(zip(planted, pairs)):
            doc = _cycle(doctors, i)
            c = kit.fill(m, ph_a if i % 2 else ph_b, D(2024, 10, 3) + TD(days=37 * i), billed, qty=30, days=30,
                         prescriber=doc, prescribed=presc, dispensed=billed, dx=kit.dx_for(presc))
            if c:
                got.append(c)
        w.record("PHR-01 billed product differs from prescribed (non-equivalent)", rule_ids=["PHR-01-R01"],
                 claim_ids=got, subject_type="claim", subject_ids=got)
        look = []
        for j, m in enumerate(spare):  # generic billed for a prescribed brand: an approved equivalent
            c = kit.fill(m, ph_a, D(2025, 2, 10) + TD(days=45 * j), "RX1017", qty=30, days=30,
                         prescriber=_cycle(doctors, j), prescribed="RX1018", dispensed="RX1017")
            if c:
                look.append(c)
        if look:
            w.record("PHR-01 generic substituted for prescribed brand (approved equivalent)",
                     rule_ids=["PHR-01-R01"], claim_ids=look, subject_type="claim", subject_ids=look,
                     positive=False, note="Declared exclusion: generic/trade equivalence — must not fire.")

    # R02 — three times the prescribed quantity billed
    m2 = kit.members(6, avoid_groups=("SERTRALINE", "LOSARTAN"))
    if _need(kit, "PHR-01-R02 quantity over prescription", members=m2):
        planted, spare = (m2[:5], m2[5:]) if len(m2) > 1 else (m2, [])
        got = []
        for i, m in enumerate(planted):
            c = kit.fill(m, ph_b, D(2024, 11, 12) + TD(days=41 * i), "RX1052" if i % 2 else "RX1022", qty=90,
                         days=30, prescriber=_cycle(doctors, i), presc_qty=30, disp_qty=90)
            if c:
                got.append(c)
        w.record("PHR-01 billed quantity three times the prescription", rule_ids=["PHR-01-R02"],
                 claim_ids=got, subject_type="claim", subject_ids=got)
        for m in spare:
            c = kit.fill(m, ph_b, D(2025, 5, 6), "RX1022", qty=32, days=30, prescriber=doctors[0],
                         presc_qty=30, disp_qty=32)
            w.record("PHR-01 quantity rounded up to whole packs", rule_ids=["PHR-01-R02"],
                     claim_ids=[c] if c else [], positive=False,
                     note="Declared exclusion: package conversion within tolerance — must not fire.")

    # R03 — the pharmacy's own record disagrees with the bill
    m3 = kit.members(5, avoid_groups=("ATORVASTATIN", "LOSARTAN"))
    if _need(kit, "PHR-01-R03 dispensing record differs", members=m3):
        got = []
        for i, m in enumerate(m3):
            doc = _cycle(doctors, i)
            if i < 3:  # brand billed, generic handed over
                c = kit.fill(m, ph_a, D(2024, 12, 2) + TD(days=53 * i), "RX1018", qty=30, days=30, prescriber=doc,
                             prescribed="RX1018", dispensed="RX1017")
            else:  # sixty tablets billed, thirty handed over
                c = kit.fill(m, ph_a, D(2025, 3, 4) + TD(days=33 * i), "RX1022", qty=60, days=30, prescriber=doc,
                             presc_qty=60, disp_qty=30)
            if c:
                got.append(c)
        w.record("PHR-01 dispensing record differs from billed product or quantity", rule_ids=["PHR-01-R03"],
                 claim_ids=got, subject_type="claim", subject_ids=got)

    # R03 stock limb — one pharmacy bills a month of iron tablets its stock could not supply
    ph_s = kit.pharmacies(1, "Sharjah")
    m4 = kit.members(6, avoid_groups=("FERROUS_SULFATE",))
    doc_s = kit.prescribers(1, emirate="Sharjah", max_rx=40) or doctors[:1]
    if _need(kit, "PHR-01-R03 stock overrun", pharmacies=ph_s, members=m4):
        got = []
        for i, m in enumerate(m4):
            c = kit.fill(m, ph_s[0], D(2025, 5, 3) + TD(days=4 * i), "RX1045", qty=30, days=30, prescriber=doc_s[0])
            if c:
                got.append(c)
        inv = w.tables["pharmacy_inventory"]
        mask = (inv["pharmacy_id"] == ph_s[0]) & (inv["product"] == "RX1045") & (inv["period"] == "2025-05")
        if mask.any() and got:
            w.set_values("pharmacy_inventory", mask, opening_stock=0.0, purchased_qty=15.0 * len(got),
                         closing_stock=0.0)
            w.record("PHR-01 pharmacy billed more of a product in a month than its stock records allow",
                     rule_ids=["PHR-01-R03"], claim_ids=got, subject_type="pharmacy", subject_ids=ph_s)

    # R04 — receipts show a non-medical item against a billed medicine
    items = ["Perfume, 100 ml eau de parfum", "Protein powder, 2 kg tub", "Skincare gift set",
             "Infant formula, 800 g x 2", "Sunglasses, polarised"]
    m5 = kit.members(len(items), avoid_groups=("ESOMEPRAZOLE", "MONTELUKAST"))
    if _need(kit, "PHR-01-R04 non-medical receipt", members=m5):
        got = []
        for i, (m, item) in enumerate(zip(m5, items)):
            d = D(2024, 9, 20) + TD(days=71 * i)
            c = kit.fill(m, ph_b, d, "RX1030" if i % 2 == 0 else "RX1040", qty=30, days=30,
                         prescriber=_cycle(doctors, i))
            if not c:
                continue
            got.append(c)
            gross = float(kit.rx_of(c)["billed_amount"].sum())
            w.append("member_receipt", [{
                "receipt_sk": w.new_id("RCTP", 6), "member_sk": m, "claim_sk": c, "provider_sk": ph_b,
                "amount_paid_aed": round(max(gross, 45.0), 2), "item_description": item, "receipt_date": d,
                "tenant_id": TENANT}])
        w.record("PHR-01 receipt shows a non-medical item billed as a covered medicine", rule_ids=["PHR-01-R04"],
                 claim_ids=got, subject_type="pharmacy", subject_ids=[ph_b])


# ---------------------------------------------------------------------------
# PHR-02
# ---------------------------------------------------------------------------


def _phr02(kit: _Kit) -> None:
    w = kit.w
    pharms = kit.pharmacies(4, "Abu Dhabi")
    docs = kit.prescribers(2, emirate="Abu Dhabi", max_rx=40)
    if not _need(kit, "PHR-02 (all patterns)", pharmacies=pharms, prescribers=docs):
        return
    ph1, ph2 = pharms[0], _cycle(pharms, 1)

    # R01 — refilled every ten days on a thirty-day supply
    mem = kit.members(6, avoid_groups=("LOSARTAN", "SERTRALINE"))
    if _need(kit, "PHR-02-R01 early refills", members=mem):
        planted, spare = (mem[:5], mem[5:]) if len(mem) > 1 else (mem, [])
        for k, m in enumerate(planted):
            start = D(2024, 10, 1) + TD(days=45 * k)
            got = [c for c in (kit.fill(m, ph1, start + TD(days=10 * j), "RX1022", qty=30, days=30,
                                        prescriber=_cycle(docs, k), prescribed_date=start) for j in range(4)) if c]
            w.record("PHR-02 early refills (stockpiling)", rule_ids=["PHR-02-R01"], claim_ids=got,
                     subject_type="member", subject_ids=[m])
        for m in spare:  # look-alike: early refill because the dose was increased
            d0 = D(2025, 4, 2)
            a = kit.fill(m, ph1, d0, "RX1052", qty=30, days=30, prescriber=docs[0], dose=50.0)
            b = kit.fill(m, ph1, d0 + TD(days=10), "RX1052", qty=60, days=30, prescriber=docs[0], dose=100.0)
            w.record("PHR-02 early refill with a dose increase", rule_ids=["PHR-02-R01"],
                     claim_ids=[c for c in (a, b) if c], subject_type="member", subject_ids=[m], positive=False,
                     note="Declared exclusion: dose change — must not fire.")

    # R02 — continued for months on one prescription
    mem = kit.members(6, avoid_groups=("OMEPRAZOLE", "TRAMADOL"))
    if _need(kit, "PHR-02-R02 therapy duration", members=mem):
        plans = [("RX1029", 28, 7), ("RX1029", 28, 7), ("RX1029", 28, 6), ("RX1047", 7, 5), ("RX1047", 7, 5)]
        planted, spare = (mem[:5], mem[5:]) if len(mem) > 1 else (mem, [])
        for k, (m, (prod, days, n)) in enumerate(zip(planted, plans)):
            start = D(2024, 11, 5) + TD(days=31 * k)
            qty = days * (1 if prod == "RX1029" else 3)
            got = [c for c in (kit.fill(m, ph2, start + TD(days=days * j), prod, qty=qty, days=days,
                                        prescriber=_cycle(docs, k), prescribed_date=start) for j in range(n)) if c]
            w.record("PHR-02 therapy continued beyond the policy duration on one prescription",
                     rule_ids=["PHR-02-R02"], claim_ids=got, subject_type="member", subject_ids=[m])
        for m in spare:  # look-alike: the same long PPI course with a specialist approval on file
            start = D(2025, 1, 7)
            got = [c for c in (kit.fill(m, ph2, start + TD(days=28 * j), "RX1029", qty=28, days=28,
                                        prescriber=docs[0], prescribed_date=start) for j in range(6)) if c]
            if not got:
                continue
            pa = kit.add_auth(member=m, provider=ph2, code="RX1029", units=28 * 6, value=28 * 6 * 0.6, start=start,
                              end=start + TD(days=200),
                              conditions="Specialist approval: long-term PPI for Barrett's oesophagus")
            lines = w.tables["claim_line"]
            w.set_values("claim_line", lines["claim_sk"].isin(got), authorization_id=pa)
            w.record("PHR-02 long course with specialist approval", rule_ids=["PHR-02-R02"], claim_ids=got,
                     subject_type="member", subject_ids=[m], positive=False,
                     note="Declared exclusion: specialist approval — must not fire.")

    # R03 — pregabalin from several unrelated prescribers within ninety days
    unrelated = kit.prescribers(4, specialties=("GENERAL_PRACTICE", "FAMILY_MEDICINE", "ORTHOPAEDICS",
                                                "NEUROLOGY", "INTERNAL_MEDICINE"),
                                max_rx=60, distinct_owners=True)
    mem = kit.members(5, avoid_groups=("PREGABALIN",))
    if len(unrelated) < 3:
        _skip(kit, "PHR-02-R03 multiple prescribers", "fewer than three unrelated prescribers on this build")
    elif _need(kit, "PHR-02-R03 multiple prescribers", members=mem):
        planted, spare = (mem[:4], mem[4:]) if len(mem) > 1 else (mem, [])
        for k, m in enumerate(planted):
            start = D(2024, 12, 1) + TD(days=50 * k)
            n_presc = min(len(unrelated), 3 if k % 2 == 0 else 4)
            got = [c for c in (kit.fill(m, _cycle(pharms, 2), start + TD(days=29 * j), "RX1048", qty=60, days=30,
                                        prescriber=unrelated[j], dx="M54.5") for j in range(n_presc)) if c]
            w.record("PHR-02 same controlled medicine from several unrelated prescribers", rule_ids=["PHR-02-R03"],
                     claim_ids=got, subject_type="member", subject_ids=[m])
        clinics = kit.providers(1, "CLINIC", emirate="Abu Dhabi", max_rx=200) if spare else []
        team = [c["id"] for c in kit.ctx.clinicians.get(clinics[0], [])][:3] if clinics else []
        if spare and len(team) >= 3:  # look-alike: three prescribers of one clinic (a care team)
            start = D(2025, 6, 2)
            got = [c for c in (kit.fill(spare[0], _cycle(pharms, 2), start + TD(days=29 * j), "RX1048", qty=60,
                                        days=30, prescriber=team[j], dx="M54.5") for j in range(3)) if c]
            w.record("PHR-02 one care team prescribing the same medicine", rule_ids=["PHR-02-R03"], claim_ids=got,
                     subject_type="member", subject_ids=[spare[0]], positive=False,
                     note="Declared exclusion: care team / provider group identity — must not fire.")

    # R04 — alprazolam from three pharmacies within ten days
    mem = kit.members(5, avoid_groups=("ALPRAZOLAM",))
    trio = [pharms[0], _cycle(pharms, 2), _cycle(pharms, 3)]
    if len(set(trio)) < 3:
        _skip(kit, "PHR-02-R04 multiple pharmacies", "fewer than three pharmacies available on this build")
    elif _need(kit, "PHR-02-R04 multiple pharmacies", members=mem):
        planted, spare = (mem[:4], mem[4:]) if len(mem) > 1 else (mem, [])
        for k, m in enumerate(planted):
            doc = _cycle(docs, k)
            start = D(2025, 1, 15) + TD(days=47 * k)
            got = [c for c in (kit.fill(m, trio[j], start + TD(days=4 * j), "RX1049", qty=28, days=14,
                                        prescriber=doc, prescribed_date=start, dx="F41.1") for j in range(3)) if c]
            w.record("PHR-02 same medicine collected from several pharmacies",
                     rule_ids=["PHR-02-R04", "PHR-02-R01"], claim_ids=got, subject_type="member", subject_ids=[m])
        for m in spare:  # look-alike: one prescription split into partial fills across pharmacies
            start = D(2025, 8, 4)
            got = [c for c in (kit.fill(m, trio[j], start + TD(days=4 * j), "RX1049", qty=8, days=4,
                                        prescriber=docs[0], presc_qty=28, disp_qty=8, prescribed_date=start,
                                        dx="F41.1") for j in range(3)) if c]
            w.record("PHR-02 partial fills across pharmacies", rule_ids=["PHR-02-R04"], claim_ids=got,
                     subject_type="member", subject_ids=[m], positive=False,
                     note="Declared exclusion: partial fills — must not fire.")


# ---------------------------------------------------------------------------
# PHR-03
# ---------------------------------------------------------------------------


def _weight(kit: _Kit, member: str, default: float) -> float:
    v = kit.mem.set_index("member_sk")["weight_kg"].get(member)
    return default if v is None or pd.isna(v) else float(v)


def _phr03(kit: _Kit) -> None:
    w = kit.w
    phs = kit.pharmacies(1, "Sharjah")
    docs = kit.prescribers(2, specialties=("PAEDIATRICS", "GENERAL_PRACTICE", "FAMILY_MEDICINE"), max_rx=60)
    if _need(kit, "PHR-03-R01/R02 (pharmacy patterns)", pharmacies=phs, prescribers=docs):
        ph = phs[0]
        # R01 — daily dose far above the mg/kg limit for the patient's weight
        kids = kit.members(4, age=(4, 9), avoid_groups=("IBUPROFEN",))
        adults = kit.members(2, age=(25, 55), avoid_groups=("PARACETAMOL",))
        if _need(kit, "PHR-03-R01 dose over limit", members=kids + adults):
            planted_kids, spare = (kids[:3], kids[3:]) if len(kids) > 1 else (kids, [])
            got = []
            for k, m in enumerate(planted_kids):
                dose = round(_weight(kit, m, 20.0) * 40 * 2.5, 0)
                ml = round(dose / 20 * 5 / 100 + 0.5) * 100
                c = kit.fill(m, ph, D(2024, 11, 18) + TD(days=60 * k), "RX1036", qty=ml, days=5,
                             prescriber=_cycle(docs, k), dose=dose, dx="R50.9")
                if c:
                    got.append(c)
            for k, m in enumerate(adults):
                tabs = int(np.ceil(_weight(kit, m, 70.0) * 75 * 2.5 / 500))
                c = kit.fill(m, ph, D(2025, 3, 10) + TD(days=70 * k), "RX1033", qty=tabs * 5, days=5,
                             prescriber=_cycle(docs, k), dose=float(tabs * 500), dx="M54.5")
                if c:
                    got.append(c)
            w.record("PHR-03 daily dose above the mg/kg limit", rule_ids=["PHR-03-R01"], claim_ids=got,
                     subject_type="claim", subject_ids=got)
            for m in spare:  # look-alike: a short first (loading) course
                c = kit.fill(m, ph, D(2025, 6, 9), "RX1036", qty=100, days=2, prescriber=docs[0],
                             dose=round(_weight(kit, m, 20.0) * 40 * 1.6, 0), dx="R50.9")
                w.record("PHR-03 short loading course", rule_ids=["PHR-03-R01"], claim_ids=[c] if c else [],
                         positive=False, note="Declared exclusion: loading dose — must not fire.")

        # R02 — step therapy skipped / drug not indicated
        mem = kit.members(6, avoid_groups=("METFORMIN", "SITAGLIPTIN", "ISOTRETINOIN"))
        if _need(kit, "PHR-03-R02 step therapy / indication", members=mem):
            got = []
            half = max(1, len(mem) // 2)
            for k, m in enumerate(mem[:half]):  # sitagliptin with no metformin first
                c = kit.fill(m, ph, D(2025, 7, 20) + TD(days=35 * k), "RX1014", qty=30, days=30,
                             prescriber=_cycle(docs, k), dx="E11.9")
                if c:
                    got.append(c)
            for k, m in enumerate(mem[half:]):  # isotretinoin billed against a cold
                c = kit.fill(m, ph, D(2025, 1, 14) + TD(days=61 * k), "RX1062", qty=30, days=30,
                             prescriber=_cycle(docs, k), dx="J06.9")
                if c:
                    got.append(c)
            w.record("PHR-03 second-line drug without first-line history, or drug not indicated",
                     rule_ids=["PHR-03-R02"], claim_ids=got, subject_type="claim", subject_ids=got)

    # R04 — one provider bills two extra vials of wastage on every infliximab infusion
    lines = w.tables["claim_line"]
    header = w.tables["claim_header"]
    inf = lines[(lines["activity_code"] == "J1745") & lines["wastage_units"].notna()]
    inf = inf[~inf["claim_sk"].isin(w.used_claims)].merge(header[["claim_sk", "provider_sk"]], on="claim_sk")
    counts = inf.groupby("provider_sk").size().sort_values(ascending=False)
    target = next((p for p, n in counts.items() if n >= 5), None)
    if target is None or counts.size < 3:
        _skip(kit, "PHR-03-R04 wastage", "no provider with five infliximab infusions and two peers on this build")
        return
    ids = set(inf.loc[inf["provider_sk"] == target, "claim_sk"])
    ids = set(w.take_claims(len(ids), where=lambda h: h["claim_sk"].isin(ids))["claim_sk"])
    if not ids:
        _skip(kit, "PHR-03-R04 wastage", "the provider's infusion claims were already used")
        return
    lm = (lines["claim_sk"].isin(ids)) & (lines["activity_code"] == "J1745")
    extra = 20.0  # two 100 mg vials of 10 mg billing units
    for idx in lines.index[lm]:
        row = lines.loc[idx]
        units = float(row["units"]) + extra
        gross = round(float(row["unit_price"]) * units, 2)
        old_gross = float(row["gross_amount"] or 0)
        share = float(row["patient_share"] or 0) / old_gross if old_gross else 0.0
        pshare = round(gross * share, 2)
        lines.loc[idx, ["units", "wastage_units", "gross_amount", "patient_share", "net_amount"]] = [
            units, float(row["wastage_units"] or 0) + extra, gross, pshare, round(gross - pshare, 2)]
        rem = w.tables["remittance"]
        rm = (rem["line_sk"] == row["line_sk"]) & (rem["decision"] != "DENIED")
        if rm.any():
            w.set_values("remittance", rm, payment_amount=round(gross - pshare, 2))
        rx = w.tables["prescription_dispense"]
        xm = rx["line_sk"] == row["line_sk"]
        if xm.any():
            w.set_values("prescription_dispense", xm, billed_qty=units, dispensed_qty=units,
                         prescribed_qty=units, billed_amount=gross)
            period = pd.Timestamp(row["service_date"]).strftime("%Y-%m")
            inv = w.tables["pharmacy_inventory"]
            im = (inv["pharmacy_id"] == target) & (inv["product"] == "J1745") & (inv["period"] == period)
            if im.any():
                w.set_values("pharmacy_inventory", im, purchased_qty=inv.loc[im, "purchased_qty"].astype(float) + extra)
    auth_ids = set(lines.loc[lm, "authorization_id"].dropna())
    al = w.tables.get("authorization_line")
    if al is not None and auth_ids:
        am = al["authorization_sk"].isin(auth_ids) & (al["activity_code"] == "J1745")
        for aid, n_lines in lines[lm].groupby("authorization_id").size().items():
            one = am & (al["authorization_sk"] == aid)
            if one.any():
                w.set_values("authorization_line", one,
                             approved_units=al.loc[one, "approved_units"].astype(float) + extra * n_lines)
    w.recompute_header_amounts(ids)
    w.record("PHR-03 wastage billed far above vial arithmetic", rule_ids=["PHR-03-R04"], claim_ids=sorted(ids),
             subject_type="provider", subject_ids=[target])


# ---------------------------------------------------------------------------
# PHR-04
# ---------------------------------------------------------------------------


def _monthly_fills(kit: _Kit, members: list[str], prescriber: str, pharmacy_for, months: list[D],
                   product: str = "RX1020") -> list[str]:
    got = []
    for mi, month in enumerate(months):
        for k, m in enumerate(members):
            ph = pharmacy_for(mi, k)
            if ph is None:
                continue
            c = kit.fill(m, ph, month + TD(days=k % 20), product, qty=30, days=30, prescriber=prescriber,
                         dx=kit.dx_for(product))
            if c:
                got.append(c)
    return got


def _team_of(kit: _Kit, clinic: str) -> list[str]:
    return [c["id"] for c in kit.ctx.clinicians.get(clinic, [])
            if c.get("specialty") not in ("PHYSIOTHERAPY", "CLINICAL_PSYCHOLOGY")][:3]


def _phr04(kit: _Kit) -> None:
    w = kit.w
    prov = w.tables["provider"]

    # R01 — a GP sending nearly every prescription to one pharmacy
    phs = kit.pharmacies(2, "Dubai")
    gp = kit.prescribers(1, emirate="Dubai", max_rx=3)
    mem = kit.members(10, avoid_groups=("AMLODIPINE",))
    if _need(kit, "PHR-04-R01 concentration", pharmacies=phs, prescribers=gp, members=mem):
        last = (2, len(mem) - 1)
        got = _monthly_fills(kit, mem, gp[0], lambda mi, k: phs[0] if (mi, k) != last else _cycle(phs, 1),
                             [D(2025, 2, 3), D(2025, 3, 3), D(2025, 4, 3)])
        w.record("PHR-04 prescriber concentrated on one pharmacy", rule_ids=["PHR-04-R01"], claim_ids=got,
                 subject_type="clinician", subject_ids=gp)

    # R02 — a clinic's flow to a pharmacy sharing its bank account and phone line
    clinic = kit.providers(1, "CLINIC", emirate="Abu Dhabi", max_rx=6)
    ph_link = kit.pharmacies(1, "Abu Dhabi")
    team = _team_of(kit, clinic[0]) if clinic else []
    mem = kit.members(9, avoid_groups=("AMLODIPINE",))
    if _need(kit, "PHR-04-R02 value loop", clinics=clinic, pharmacies=ph_link, prescribers=team, members=mem):
        cm, pm = prov["provider_sk"] == clinic[0], prov["provider_sk"] == ph_link[0]
        w.set_values("provider", pm, bank_account_token=prov.loc[cm, "bank_account_token"].iloc[0],
                     phone_token=prov.loc[cm, "phone_token"].iloc[0])
        got = []
        for t, doc in enumerate(team):
            got += _monthly_fills(kit, mem[t::len(team)], doc, lambda mi, k: ph_link[0],
                                  [D(2025, 5, 5), D(2025, 6, 5), D(2025, 7, 5)])
        w.record("PHR-04 clinic-to-pharmacy flow with shared bank account and phone", rule_ids=["PHR-04-R02"],
                 claim_ids=got, subject_type="provider", subject_ids=[clinic[0], ph_link[0]])
    # look-alike: the same flow to a pharmacy the clinic declares it owns
    clinic2 = kit.providers(1, "CLINIC", emirate="Sharjah", max_rx=6)
    ph_own = kit.pharmacies(1, "Sharjah")
    team2 = _team_of(kit, clinic2[0]) if clinic2 else []
    mem = kit.members(9, avoid_groups=("AMLODIPINE",))
    if _need(kit, "PHR-04-R02 look-alike (declared owner)", clinics=clinic2, pharmacies=ph_own,
             prescribers=team2, members=mem):
        c2 = prov["provider_sk"] == clinic2[0]
        w.set_values("provider", prov["provider_sk"] == ph_own[0],
                     owner_entity_id=prov.loc[c2, "owner_entity_id"].iloc[0],
                     address_token=prov.loc[c2, "address_token"].iloc[0])
        got = []
        for t, doc in enumerate(team2):
            got += _monthly_fills(kit, mem[t::len(team2)], doc, lambda mi, k: ph_own[0],
                                  [D(2025, 5, 12), D(2025, 6, 12), D(2025, 7, 12)])
        w.record("PHR-04 clinic's flow to its own declared pharmacy", rule_ids=["PHR-04-R02", "PHR-04-R01"],
                 claim_ids=got, subject_type="provider", subject_ids=[clinic2[0], ph_own[0]], positive=False,
                 note="Declared exclusions: known corporate relationship / on-site pharmacy — must not fire.")

    # R03 — adalimumab routed to one pharmacy, other prescriptions spread normally
    phs = kit.pharmacies(4, "Dubai")
    rheum = kit.prescribers(1, specialties=("RHEUMATOLOGY", "INTERNAL_MEDICINE", "DERMATOLOGY",
                                            "FAMILY_MEDICINE", "GENERAL_PRACTICE"), emirate="Dubai", max_rx=5)
    mem = kit.members(10, avoid_groups=("METHOTREXATE", "ADALIMUMAB"))
    if len(set(phs)) < 2:
        _skip(kit, "PHR-04-R03 high-cost steering", "fewer than two pharmacies available on this build")
    elif _need(kit, "PHR-04-R03 high-cost steering", prescribers=rheum, members=mem):
        got = []
        for k, m in enumerate(mem):
            d0 = D(2024, 12, 2) + TD(days=11 * k)
            c1 = kit.fill(m, _cycle(phs, k), d0, "RX1060", qty=24, days=28, prescriber=rheum[0], dx="M05.79",
                          dose=round(15 / 7, 3))
            c2 = kit.fill(m, phs[0], d0 + TD(days=70), "J0135", qty=2, days=28, prescriber=rheum[0], dx="M05.79",
                          wastage=0.0)
            got += [c for c in (c1, c2) if c]
        w.record("PHR-04 high-cost prescriptions steered to one pharmacy", rule_ids=["PHR-04-R03"], claim_ids=got,
                 subject_type="clinician", subject_ids=rheum)
        # background: other prescribers' adalimumab at other pharmacies (the product is widely available)
        bg_docs = kit.prescribers(2, specialties=("INTERNAL_MEDICINE", "FAMILY_MEDICINE", "GENERAL_PRACTICE"),
                                  emirate="Dubai", max_rx=40)
        bg_mem = kit.members(4, avoid_groups=("METHOTREXATE", "ADALIMUMAB"))
        others = phs[1:]
        if bg_docs and bg_mem:
            got = []
            for k, m in enumerate(bg_mem):
                doc, ph = _cycle(bg_docs, k), _cycle(others, k)
                d0 = D(2025, 1, 20) + TD(days=23 * k)
                c1 = kit.fill(m, ph, d0, "RX1060", qty=24, days=28, prescriber=doc, dx="M05.79", dose=round(15 / 7, 3))
                c2 = kit.fill(m, ph, d0 + TD(days=75), "J0135", qty=2, days=28, prescriber=doc, dx="M05.79",
                              wastage=0.0)
                got += [c for c in (c1, c2) if c]
            w.record("PHR-04 adalimumab dispensed by several pharmacies (availability background)",
                     rule_ids=["PHR-04-R03"], claim_ids=got, positive=False,
                     note="Legitimate background so the product-availability exclusion does not apply.")

    # R04 — a GP's prescriptions switch abruptly to an established pharmacy
    ph_old = kit.pharmacies(1, "Abu Dhabi")
    rx = w.tables["prescription_dispense"]
    first_fill = pd.to_datetime(rx.groupby("pharmacy_id")["fill_date"].min())
    early = set(first_fill[first_fill < pd.Timestamp(2024, 9, 1)].index)
    ph_new = kit.pharmacies(1, "Abu Dhabi", exclude=tuple(set(prov["provider_sk"]) - early))
    gp2 = kit.prescribers(1, emirate="Abu Dhabi", max_rx=3)
    mem = kit.members(9, avoid_groups=("AMLODIPINE",))
    if len(mem) < 3:
        _skip(kit, "PHR-04-R04 new relationship", "fewer than three eligible members left on this build")
    elif _need(kit, "PHR-04-R04 new relationship", pharmacies=ph_old, established_pharmacies=ph_new,
               prescribers=gp2):
        before = [D(2024, 10, 1) + TD(days=31 * i) for i in range(6)]
        after = [D(2025, 4, 7), D(2025, 5, 7), D(2025, 6, 7)]
        got = _monthly_fills(kit, mem[:-1], gp2[0], lambda mi, k: ph_old[0], before)
        new = _monthly_fills(kit, mem[:-1], gp2[0], lambda mi, k: ph_new[0], after)
        tail = _monthly_fills(kit, mem[-1:], gp2[0], lambda mi, k: ph_old[0], after)
        w.record("PHR-04 new prescriber-pharmacy link rapidly dominant", rule_ids=["PHR-04-R04"],
                 claim_ids=new, subject_type="clinician", subject_ids=[gp2[0], ph_new[0]])
        w.record("PHR-04 established prescribing before the switch", rule_ids=["PHR-04-R04"],
                 claim_ids=got + tail, subject_type="clinician", subject_ids=gp2, positive=False,
                 note="History that makes the later link new; not itself a finding.")


# ---------------------------------------------------------------------------
# PHR-05
# ---------------------------------------------------------------------------


def _device_line(w: World, claim_sk: str | None, code: str) -> pd.Series | None:
    if not claim_sk:
        return None
    lines = w.tables["claim_line"]
    ln = lines[(lines["claim_sk"] == claim_sk) & (lines["activity_code"] == code)]
    return None if ln.empty else ln.iloc[0]


def _set_serial(w: World, serial: Any, **values: Any) -> None:
    inv = w.tables["device_inventory"]
    w.set_values("device_inventory", inv["serial_number"] == serial, **values)


def _phr05(kit: _Kit) -> None:
    w = kit.w
    clinics = kit.providers(1, "CLINIC", emirate="Dubai")
    hosps = kit.providers(1, "HOSPITAL", emirate="Dubai")
    clinic = clinics[0] if clinics else None
    doc = (kit.ctx.clinicians.get(clinic) or [{"id": None}])[0]["id"] if clinic else None

    def device_claim(member: str, when: D, code: str, **line: Any) -> tuple[str | None, pd.Series | None]:
        c = kit.claim(member_sk=member, provider_sk=clinic, service_date=when, claim_type="OUTPATIENT",
                      clinician_id=doc, lines=[{"code": "99213"}, {"code": code, "device": True, **line}],
                      diagnoses=["S82.201A" if code == "E0114" else "G47.33"])
        return c, _device_line(w, c, code)

    # R01 — used or rented equipment billed with the new-purchase modifier
    mem = kit.members(4) if clinic else []
    if _need(kit, "PHR-05-R01 condition mismatch", clinics=clinics, members=mem):
        cases = [("USED", "PURCHASE"), ("NEW", "RENTAL"), ("REFURBISHED_USED", "PURCHASE")]
        planted, spare = (mem[:3], mem[3:]) if len(mem) > 1 else (mem, [])
        got = []
        for k, (m, (cond, acq)) in enumerate(zip(planted, cases)):
            c, ln = device_claim(m, D(2025, 2, 11) + TD(days=40 * k), "E0601", indicator="NU")
            if ln is not None:
                _set_serial(w, ln["device_serial"], condition=cond, acquisition=acq)
                got.append(c)
        w.record("PHR-05 used or rented equipment billed as a new purchase", rule_ids=["PHR-05-R01"],
                 claim_ids=got, subject_type="claim", subject_ids=got)
        for m in spare:  # look-alike: rent-to-own
            c, ln = device_claim(m, D(2025, 7, 8), "E0601", indicator="NU")
            if ln is not None:
                _set_serial(w, ln["device_serial"], condition="NEW", acquisition="RENT_TO_OWN")
            w.record("PHR-05 rent-to-own device billed as purchase", rule_ids=["PHR-05-R01"],
                     claim_ids=[c] if c else [], positive=False,
                     note="Declared exclusion: rent-to-own policy — must not fire.")

    # R02 — implant with no procedure, or on the opposite side
    got = []
    if hosps:
        hdoc = next((c["id"] for c in kit.ctx.clinicians.get(hosps[0], []) if c.get("specialty") not in
                     ("PATHOLOGY", "RADIOLOGY", "ANAESTHESIA")), None)
        for k, m in enumerate(kit.members(2)):
            c = kit.claim(member_sk=m, provider_sk=hosps[0], service_date=D(2025, 3, 18) + TD(days=50 * k),
                          claim_type="OUTPATIENT", clinician_id=hdoc,
                          lines=[{"code": "99213"}, {"code": "C1781", "device": True}], diagnoses=["K40.90"])
            if c:
                got.append(c)
    lines = w.tables["claim_line"]
    rt = set(lines.loc[lines["activity_code"].isin(["C1776", "V2632"]) & (lines["indicator"] == "RT"), "claim_sk"])
    for c in w.take_claims(2, where=lambda h: h["claim_sk"].isin(rt))["claim_sk"]:
        m = (lines["claim_sk"] == c) & lines["activity_code"].isin(["C1776", "V2632"])
        w.set_values("claim_line", m, indicator="LT")
        got.append(c)
    if got:
        w.record("PHR-05 implant with no procedure, or on the opposite side", rule_ids=["PHR-05-R02"],
                 claim_ids=got, subject_type="claim", subject_ids=got)
    else:
        _skip(kit, "PHR-05-R02 implant not linked", "no hospital or implant claims available on this build")
    # look-alike: a staged implant a few weeks after the patient's earlier procedure
    lines = w.tables["claim_line"]
    prior = set(lines.loc[lines["activity_code"] == "C1781", "claim_sk"])
    p0 = w.take_claims(1, where=lambda h: h["claim_sk"].isin(prior))
    if not p0.empty:
        hrow = p0.iloc[0]
        d = pd.Timestamp(lines.loc[lines["claim_sk"] == hrow["claim_sk"], "service_date"].min()).date() + TD(days=40)
        if kit.ctx.coverage_at(hrow["member_sk"], d):
            c = kit.claim(member_sk=hrow["member_sk"], provider_sk=hrow["provider_sk"], service_date=d,
                          claim_type="OUTPATIENT", lines=[{"code": "99213"}, {"code": "C1781", "device": True}],
                          diagnoses=["K40.90"])
            if c:
                w.record("PHR-05 staged implant after an earlier procedure", rule_ids=["PHR-05-R02"],
                         claim_ids=[c], positive=False,
                         note="Declared exclusion: replacement/spare policy — must not fire.")

    # R03 — twenty strip packs on one claim; crutches re-issued within weeks
    phs = kit.pharmacies(1, "Dubai")
    got = []
    if phs:
        for k, m in enumerate(kit.members(3)):
            c = kit.claim(member_sk=m, provider_sk=phs[0], service_date=D(2025, 1, 21) + TD(days=66 * k),
                          claim_type="PHARMACY", lines=[{"code": "A4253", "units": 20.0}], diagnoses=["E11.9"])
            if c:
                got.append(c)
    if clinic:
        for k, m in enumerate(kit.members(2)):
            d0 = D(2025, 4, 1) + TD(days=45 * k)
            for dd in (d0, d0 + TD(days=40)):
                c, _ = device_claim(m, dd, "E0114")
                if c:
                    got.append(c)
        kid = kit.members(1, age=(6, 12))
        look = [c for c, _ in (device_claim(kid[0], dd, "E0114") for dd in (D(2025, 5, 5), D(2025, 6, 20)))
                if c] if kid else []
        if look:
            w.record("PHR-05 child re-issued a device (growth)", rule_ids=["PHR-05-R03"], claim_ids=look,
                     positive=False, note="Declared exclusion: member growth — must not fire.")
    if got:
        w.record("PHR-05 supplies beyond usual consumption or useful life", rule_ids=["PHR-05-R03"],
                 claim_ids=got, subject_type="claim", subject_ids=got)
    else:
        _skip(kit, "PHR-05-R03 supply excess", "no pharmacy, clinic or members available on this build")

    # R04 — one equipment serial billed for two patients
    mem = kit.members(6) if clinic else []
    if _need(kit, "PHR-05-R04 serial duplication", clinics=clinics, members=mem[1:]):
        got = []
        for k in range(len(mem) // 2):
            a, la = device_claim(mem[2 * k], D(2025, 1, 13) + TD(days=70 * k), "E0601")
            b, lb = device_claim(mem[2 * k + 1], D(2025, 2, 24) + TD(days=70 * k), "E0601")
            if la is None or lb is None:
                continue
            lines = w.tables["claim_line"]
            w.set_values("claim_line", lines["line_sk"] == lb["line_sk"], device_serial=la["device_serial"])
            inv = w.tables["device_inventory"]
            w.tables["device_inventory"] = inv[inv["serial_number"] != lb["device_serial"]].reset_index(drop=True)
            got += [a, b]
        w.record("PHR-05 one device serial billed for two patients", rule_ids=["PHR-05-R04"], claim_ids=got,
                 subject_type="provider", subject_ids=[clinic])


def inject(world: World) -> None:
    kit = _Kit(world)
    for step in (_phr01, _phr02, _phr03, _phr04, _phr05):
        step(kit)
