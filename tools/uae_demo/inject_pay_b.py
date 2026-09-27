"""Plant the PAY-07 to PAY-12 patterns in the SYNTHETIC UAE demo dataset.

Every control implemented in ``src/fwa/engine/unlocked/pay_b.py`` gets at least
one realistic pattern here, at a modest volume, and every planted item is
written to the answer key with the rule ids it is meant to exercise. Where a
declared exclusion has a legitimate look-alike, one or two look-alikes are
planted too (``positive=False``) so the exclusion is demonstrated.

Patterns (all SYNTHETIC):

* PAY-07-R01  patient share below the benefit's co-payment;
* PAY-07-R02  the share left in the amount claimed from the insurer, or waived and
              the whole bill claimed;
* PAY-07-R03  one clinic that records a zero or rounded-down share on most claims;
* PAY-07-R04  receipts / complaints showing members paid more than their share;
* PAY-08-R01  resubmissions with no, a non-existent or an unrelated original;
* PAY-08-R02  refused (medical necessity) claims resubmitted with diagnosis and
              code level swapped;
* PAY-08-R03  services resubmitted four times with a different change each time;
* PAY-08-R04  one clinic whose changed resubmissions after one refusal reason
              are always paid;
* PAY-09-R01  clinical notes describing cosmetic work on claims billing covered codes;
* PAY-09-R02  one clinic billing a paid office visit right after refused procedures;
* PAY-09-R03  eyelid / nose / skin procedures billed with cosmetic-risk diagnoses;
* PAY-09-R04  members who say the service did not happen, or whose receipt names
              a different item;
* PAY-10-R02  a second insurer paid the whole bill and we paid too;
* PAY-10-R03  road-traffic / work injuries with no liable-party record;
* PAY-10-R04  a third party settled first and we paid in full afterwards;
* PAY-11-R01  overrides with no reason or by an unauthorised role;
* PAY-11-R02  one adjudicator overriding one hospital's claims;
* PAY-11-R03  upward payment adjustments after settlement with no reason;
* PAY-11-R04  one adjudicator paying one provider's pended claims in full, always;
* PAY-12-R01  cancelled claims whose payment was never reversed;
* PAY-12-R02  a line paid twice; a payment reference reused;
* PAY-12-R03  provider refunds never applied (or applied months late);
* PAY-12-R04  one provider's credits offset against unrelated small claims, and
              negative lines on its claims.
"""

from __future__ import annotations

import datetime as _dt
import json
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .world import World

D = _dt.date
TD = _dt.timedelta
TENANT = "T001"

_EM_UP = {"99202": "99203", "99203": "99204", "99204": "99205", "99212": "99213", "99213": "99214",
          "99214": "99215"}
_SWAP_DX = ["J20.9", "M54.50", "J34.2", "L30.9"]
_EXTRA_ICD = {
    "L57.0": "Actinic keratosis", "L90.8": "Other atrophic disorders of skin",
    "L81.4": "Other melanin hyperpigmentation", "M95.0": "Acquired deformity of nose",
    "C44.1": "Other and unspecified malignant neoplasm of skin of eyelid",
    "V43.52XA": "Car driver injured in collision with other motor vehicle in traffic accident, initial encounter",
    "V47.5XXA": "Car driver injured in collision with fixed or stationary object in traffic accident, initial encounter",
    "Y99.0": "Civilian activity done for income or pay",
    "W23.0XXA": "Caught, crushed, jammed, or pinched between moving objects, initial encounter",
    "V00.131A": "Fall from skateboard, initial encounter",
}


def _d(value: Any) -> D:
    return pd.Timestamp(value).date()


class _Kit:
    def __init__(self, world: World) -> None:
        self.w = world
        self.rng = world.rng
        self.ctx = world.context["base_ctx"]
        rem = world.tables["remittance"]
        dec = rem["decision"].astype(str).str.upper()
        self.fully_paid = set(dec.groupby(rem["claim_sk"]).apply(lambda s: (s == "PAID").all()).loc[lambda s: s].index)
        self.cob_members = {m for m, info in self.ctx.members.items() if info.get("cob")}
        opr = world.tables.get("other_payer_remittance", pd.DataFrame())
        self.cob_claims = set(opr["claim_sk"].dropna()) if not opr.empty else set()
        cv = world.tables["claim_version"]
        self.lineage = set(cv.loc[cv["relationship"] != "ORIGINAL", "claim_sk"]) | set(cv["prior_claim_sk"].dropna())
        lines = world.tables["claim_line"]
        self.first_code = lines.sort_values("line_sk").groupby("claim_sk")["activity_code"].first()
        self.n_lines = lines.groupby("claim_sk").size()
        rct = world.tables.get("member_receipt", pd.DataFrame())
        self.receipted = set(rct["claim_sk"].dropna()) if not rct.empty else set()

    # ------------------------------------------------------------- selection

    def take(self, n: int, *, claim_type: str | None = "OUTPATIENT", share: bool = False,
             codes: Iterable[str] | None = None, single_line: bool = False, provider: str | None = None,
             extra=None, no_receipt: bool = False) -> pd.DataFrame:
        codes = set(codes) if codes else None

        def where(h: pd.DataFrame) -> pd.Series:
            m = h["claim_sk"].isin(self.fully_paid) & ~h["claim_sk"].isin(self.lineage)
            m &= ~h["member_sk"].isin(self.cob_members) & ~h["claim_sk"].isin(self.cob_claims)
            m &= ~h["accident_indicator"].fillna(False).astype(bool)
            if share:
                m &= h["patient_share"].astype(float) > 10
            if codes:
                m &= h["claim_sk"].map(self.first_code).isin(codes)
            if single_line:
                m &= h["claim_sk"].map(self.n_lines) == 1
            if provider:
                m &= h["provider_sk"] == provider
            else:  # keep provider-level patterns clean: never touch a reserved provider's claims
                m &= ~h["provider_sk"].astype(str).isin(self.w.used_entities)
            if no_receipt:
                m &= ~h["claim_sk"].isin(self.receipted)
            if extra is not None:
                m &= extra(h)
            return m

        return self.w.take_claims(n, where, claim_type=claim_type)

    def busy_provider(self, ptype: str, min_claims: int, claim_type: str = "OUTPATIENT",
                      max_claims: int | None = None) -> str | None:
        h = self.w.tables["claim_header"]
        ok = h[h["claim_sk"].isin(self.fully_paid) & (h["claim_type"] == claim_type)
               & ~h["claim_sk"].isin(self.w.used_claims)]
        counts = ok.groupby("provider_sk").size()
        prov = self.w.tables["provider"].set_index("provider_sk")
        cands = [p for p, c in counts.sort_values(ascending=False).items()
                 if c >= min_claims and (max_claims is None or c <= max_claims)
                 and str(prov.at[p, "provider_type"]) == ptype and p not in self.w.used_entities]
        if not cands and max_claims is not None:
            return self.busy_provider(ptype, min_claims, claim_type)
        if not cands:
            return None
        pick = cands[int(self.rng.integers(min(3, len(cands))))]
        self.w.used_entities.add(pick)
        return pick

    # ------------------------------------------------------------ accessors

    def header(self, claim: str) -> dict:
        h = self.w.tables["claim_header"]
        return h[h["claim_sk"] == claim].iloc[0].to_dict()

    def sdate(self, claim: str) -> D:
        ln = self.w.lines_of([claim])
        return _d(ln["service_date"].min())

    def settle(self, claim: str) -> D:
        rem = self.w.tables["remittance"]
        return _d(rem.loc[rem["claim_sk"] == claim, "settlement_date"].max())

    def claim(self, **kw: Any) -> str | None:
        kw.setdefault("strict", False)
        try:
            return self.w.make_claim(**kw)
        except Exception:
            return None

    def set_version(self, claim: str, **values: Any) -> None:
        cv = self.w.tables["claim_version"]
        if "changed_fields" in values and not isinstance(values["changed_fields"], str):
            values["changed_fields"] = json.dumps(values["changed_fields"], sort_keys=True)
        self.w.set_values("claim_version", cv["claim_sk"] == claim, **values)

    def set_share(self, claim: str, factor: float | None = None, *, value_of=None, keep_net: bool = False,
                  net_is_gross: bool = False) -> None:
        """Change every line's patient share and keep lines, header and remittance consistent."""
        lines = self.w.tables["claim_line"]
        m = lines["claim_sk"] == claim
        gross = lines.loc[m, "gross_amount"].astype(float)
        old = lines.loc[m, "patient_share"].astype(float)
        new = (old * factor).round(2) if value_of is None else value_of(old).round(2)
        lines.loc[m, "patient_share"] = new.values
        if net_is_gross:
            lines.loc[m, "net_amount"] = gross.values
        elif not keep_net:
            lines.loc[m, "net_amount"] = (gross - new).round(2).values
        h = self.w.tables["claim_header"]
        hm = h["claim_sk"] == claim
        h.loc[hm, "patient_share"] = round(float(lines.loc[m, "patient_share"].sum()), 2)
        h.loc[hm, "net_amount"] = round(float(lines.loc[m, "net_amount"].sum()), 2)
        rem = self.w.tables["remittance"]
        net_by_line = lines.loc[m].set_index("line_sk")["net_amount"]
        rm = (rem["claim_sk"] == claim) & (rem["decision"].astype(str).str.upper() == "PAID")
        rem.loc[rm, "payment_amount"] = rem.loc[rm, "line_sk"].map(net_by_line).round(2).values
        ev = self.w.tables["adjudication_event"]
        em = (ev["claim_sk"] == claim) & (ev["event_type"] != "AUTO_EDIT")
        total = round(float(rem.loc[rem["claim_sk"] == claim, "payment_amount"].sum()), 2)
        ev.loc[em, "amount_after"] = total
        ev.loc[ev["claim_sk"] == claim, "amount_before"] = round(float(lines.loc[m, "net_amount"].sum()), 2)

    def deny(self, claim: str, code: str, adjudicator: str | None = None) -> None:
        rem = self.w.tables["remittance"]
        rm = rem["claim_sk"] == claim
        rem.loc[rm, "adjustment"] = (rem.loc[rm, "adjustment"].astype(float)
                                     + rem.loc[rm, "payment_amount"].astype(float)).round(2)
        self.w.set_values("remittance", rm, decision="DENIED", denial_code=code, payment_amount=0.0,
                          payment_reference=None)
        ev = self.w.tables["adjudication_event"]
        net = float(self.header(claim)["net_amount"])
        keep = ev[(ev["claim_sk"] != claim) | (ev["event_type"] == "AUTO_EDIT")].copy()
        self.w.tables["adjudication_event"] = keep
        self.w.set_values("adjudication_event", keep["claim_sk"] == claim, decision="PEND",
                          system_edit_result="PEND")
        t0 = pd.Timestamp(self.settle(claim)) - pd.Timedelta(days=2) + pd.Timedelta(hours=11)
        self.event(claim, adjudicator or self.adjudicator(), "CLAIMS_ADJUDICATOR", "MANUAL_REVIEW", "DENY",
                   None, net, 0.0, t0, "DENIAL:" + code)

    def adjudicator(self) -> str:
        adj = self.ctx.adjudicators
        return adj[int(self.rng.integers(len(adj)))]

    def event(self, claim, actor, role, etype, decision, reason, before, after, when, edit) -> None:
        self.w.append("adjudication_event", [{
            "event_sk": self.w.new_id("AEV", 8), "claim_sk": claim, "actor": actor, "actor_role": role,
            "event_type": etype, "decision": decision, "override_reason": reason,
            "amount_before": round(float(before), 2), "amount_after": round(float(after), 2),
            "event_time": pd.Timestamp(when).to_pydatetime(), "system_edit_result": edit, "tenant_id": TENANT}])

    def remit_row(self, claim: str, line: str | None, decision: str, pay: float, adj: float, when: D,
                  ref: str | None = None) -> None:
        self.w.append("remittance", [{
            "remittance_sk": self.w.new_id("RAPB", 6), "claim_sk": claim, "line_sk": line, "decision": decision,
            "denial_code": None, "adjustment": round(adj, 2), "payment_amount": round(pay, 2),
            "payment_reference": ref or self.w.new_id("PAY-PB-", 8), "settlement_date": when, "tenant_id": TENANT}])

    def record(self, stratum: str, rule_ids: list[str], claims: Iterable[str | None], *, subject_type="claim",
               subjects: Iterable[str] = (), positive: bool = True, note: str = "") -> None:
        claims = [c for c in claims if c]
        self.w.record(f"PAY {stratum}", rule_ids=rule_ids, claim_ids=claims, subject_type=subject_type,
                      subject_ids=list(subjects) or claims, positive=positive, note=note)


# ---------------------------------------------------------------------------
# the injector
# ---------------------------------------------------------------------------


def inject(world: World) -> None:
    k = _Kit(world)
    _reference_codes(k)
    for plant in (_pay07, _pay08, _pay09, _pay10, _pay11, _pay12):
        plant(k)


def _reference_codes(k: _Kit) -> None:
    """Diagnosis codes the plants use that the base code list does not carry."""
    csv = k.w.tables.get("code_system_version")
    have = set(csv["code"]) if csv is not None and not csv.empty else set()
    rows = [{"code": c, "code_system": "ICD-10-CM", "valid_from": D(2018, 1, 1), "valid_to": D(2099, 12, 31)}
            for c in _EXTRA_ICD if c not in have]
    k.w.append("code_system_version", rows)
    icd = k.w.context.get("icd")
    if isinstance(icd, dict):
        for c, desc in _EXTRA_ICD.items():
            icd.setdefault(c, (desc, "XX_external_causes" if c[0] in "VWY" else "other"))


def _add_dx(k: _Kit, claim: str, code: str, *, principal: bool = False) -> None:
    dx = k.w.tables["diagnosis"]
    sub = dx[dx["claim_sk"] == claim]
    if principal and not sub.empty:
        m = (dx["claim_sk"] == claim) & (dx["sequence"] == sub["sequence"].min())
        k.w.set_values("diagnosis", m, code=code, description=_EXTRA_ICD.get(code, code))
        return
    base = sub.iloc[0].to_dict() if not sub.empty else {"claim_sk": claim, "code_system": "ICD-10-CM",
                                                        "tenant_id": TENANT}
    base.update(code=code, sequence=int(sub["sequence"].max() + 1) if not sub.empty else 1,
                diagnosis_type="SECONDARY", description=_EXTRA_ICD.get(code, code), present_on_admission=None,
                chapter="XX_external_causes" if code[0] in "VWY" else base.get("chapter"))
    k.w.append("diagnosis", [base])


# PAY-07 -----------------------------------------------------------------------


def _pay07(k: _Kit) -> None:
    w = k.w
    # R01: share recorded well below the benefit co-payment (net raised to match, payer pays more)
    picks = k.take(12, share=True, no_receipt=True)
    for c in picks["claim_sk"]:
        k.set_share(c, float(k.rng.choice([0.25, 0.4, 0.5])))
    k.record("patient share below the benefit co-payment", ["PAY-07-R01"], picks["claim_sk"])
    # R01 look-alike: a fils-level rounding difference
    picks = k.take(2, share=True, no_receipt=True)
    for c in picks["claim_sk"]:
        k.set_share(c, value_of=lambda s: (s - 0.4).clip(lower=0))
    k.record("patient share off by rounding only (look-alike)", ["PAY-07-R01"], picks["claim_sk"], positive=False)

    # R02 (a): share recorded but left in the amount claimed from the insurer
    picks = k.take(7, share=True, no_receipt=True)
    for c in picks["claim_sk"]:
        k.set_share(c, 1.0, net_is_gross=True)
    k.record("patient share left inside the amount claimed from the insurer", ["PAY-07-R02"], picks["claim_sk"])
    # R02 (b): share waived to zero and the whole bill claimed (also a share mismatch)
    picks = k.take(6, share=True, no_receipt=True)
    for c in picks["claim_sk"]:
        k.set_share(c, 0.0, net_is_gross=True)
    k.record("patient share waived and shifted onto the insurer", ["PAY-07-R02", "PAY-07-R01"], picks["claim_sk"])

    # R03: one clinic that does not collect the co-payment (zero, or rounded down to tens)
    prov = k.busy_provider("CLINIC", 40, max_claims=140) or k.busy_provider("CLINIC", 25, max_claims=400)
    if prov:
        picks = k.take(70, share=True, no_receipt=True, provider=prov, extra=lambda h: h["patient_share"].astype(float) > 12)
        ids = list(picks["claim_sk"])
        n = int(len(ids) * 0.7)
        chosen = ids[:n]
        for j, c in enumerate(chosen):
            if j % 3 == 2:
                k.set_share(c, value_of=lambda s: (np.floor(s / 10.0) * 10.0).where(s % 10 > 2.5, (s - 10).clip(lower=0)),
                            keep_net=True)
            else:
                k.set_share(c, 0.0, keep_net=True)
        k.record("clinic systematically records zero or rounded-down patient share", ["PAY-07-R03", "PAY-07-R01"],
                 chosen, subject_type="provider", subjects=[prov])

    # R04: members' receipts show they paid more than their share (balance billing)
    picks = k.take(9, share=True, no_receipt=True)
    lines = w.tables["claim_line"]
    receipts, complaints = [], []
    for j, h in enumerate(picks.itertuples(index=False)):
        desc = str(lines.loc[lines["claim_sk"] == h.claim_sk, "activity_description"].iloc[0])
        extra = float(k.rng.choice([150, 200, 250, 300, 400]))
        receipts.append({"receipt_sk": w.new_id("RCT", 7), "member_sk": h.member_sk, "claim_sk": h.claim_sk,
                         "provider_sk": h.provider_sk, "amount_paid_aed": round(float(h.patient_share) + extra, 2),
                         "item_description": f"Payment for {desc}", "receipt_date": k.sdate(h.claim_sk),
                         "tenant_id": TENANT})
        if j % 2 == 0:
            complaints.append({"complaint_sk": w.new_id("CMP", 6), "member_sk": h.member_sk,
                               "provider_sk": h.provider_sk, "claim_sk": h.claim_sk, "complaint_type": "BILLING_OVERCHARGE",
                               "complaint_date": k.sdate(h.claim_sk) + TD(days=int(k.rng.integers(3, 20))),
                               "text": f"SYNTHETIC complaint: the clinic asked me to pay AED {extra:.0f} more than my "
                                       "co-payment before they would release my results.", "tenant_id": TENANT})
    k.record("member receipt exceeds the approved patient share", ["PAY-07-R04"], picks["claim_sk"])
    picks = k.take(2, share=True)
    for h in picks.itertuples(index=False):
        complaints.append({"complaint_sk": w.new_id("CMP", 6), "member_sk": h.member_sk,
                           "provider_sk": h.provider_sk, "claim_sk": h.claim_sk, "complaint_type": "BILLING_OVERCHARGE",
                           "complaint_date": k.sdate(h.claim_sk) + TD(days=5),
                           "text": "SYNTHETIC complaint: I was charged more than my co-payment in cash at the desk.",
                           "tenant_id": TENANT})
    k.record("member complaint of being charged more than the share", ["PAY-07-R04"], picks["claim_sk"])
    # look-alike: a non-covered elective item bought with signed consent
    picks = k.take(2, share=True, no_receipt=True)
    for h in picks.itertuples(index=False):
        receipts.append({"receipt_sk": w.new_id("RCT", 7), "member_sk": h.member_sk, "claim_sk": h.claim_sk,
                         "provider_sk": h.provider_sk, "amount_paid_aed": round(float(h.patient_share) + 180.0, 2),
                         "item_description": "Non-covered elective cosmetic skin cream, signed consent on file",
                         "receipt_date": k.sdate(h.claim_sk), "tenant_id": TENANT})
    k.record("receipt for a consented non-covered item (look-alike)", ["PAY-07-R04", "PAY-09-R04"],
             picks["claim_sk"], positive=False)
    w.append("member_receipt", receipts)
    w.append("complaint", complaints)


# PAY-08 -----------------------------------------------------------------------


def _resubmit(k: _Kit, prior: str, *, days: int, code: str | None, dx: str | None, decision: str,
              denial: str | None = None, changed: dict | None = None, rtype: str = "CORRECTION",
              version: int = 2) -> str | None:
    h = k.header(prior)
    lines = k.w.lines_of([prior])
    orig_code = str(lines.sort_values("line_sk")["activity_code"].iloc[0])
    dxs = k.w.tables["diagnosis"]
    orig_dx = str(dxs[dxs["claim_sk"] == prior].sort_values("sequence")["code"].iloc[0])
    new = k.claim(member_sk=h["member_sk"], provider_sk=h["provider_sk"], service_date=k.sdate(prior),
                  claim_type=h["claim_type"], lines=[code or orig_code], diagnoses=[dx or orig_dx],
                  decision=decision, denial_code=denial,
                  submission_lag_days=(k.settle(prior) - k.sdate(prior)).days + days)
    if new is None:
        return None
    if changed is None:
        changed = {}
        if dx and dx != orig_dx:
            changed["diagnosis_primary"] = [orig_dx, dx]
        if code and code != orig_code:
            changed["activity_code"] = [orig_code, code]
    k.set_version(new, relationship="RESUBMISSION", version_no=version, prior_claim_sk=prior,
                  changed_fields=changed, resubmission_type=rtype)
    return new


def _denied_original(k: _Kit, h, code: str) -> str | None:
    lines = k.w.lines_of([h.claim_sk])
    first = str(lines.sort_values("line_sk")["activity_code"].iloc[0])
    dxs = k.w.tables["diagnosis"]
    dx = str(dxs[dxs["claim_sk"] == h.claim_sk].sort_values("sequence")["code"].iloc[0])
    return k.claim(member_sk=h.member_sk, provider_sk=h.provider_sk,
                   service_date=k.sdate(h.claim_sk) + TD(days=int(k.rng.integers(10, 40))),
                   claim_type="OUTPATIENT", lines=[first], diagnoses=[dx], decision="DENIED", denial_code=code)


def _pay08(k: _Kit) -> None:
    w = k.w
    # R01: broken lineage
    picks = k.take(7)
    ids = list(picks["claim_sk"])
    others = list(k.take(3)["claim_sk"])
    for j, c in enumerate(ids):
        prior = [None, None, f"CLM9{int(k.rng.integers(10**7, 10**8))}", f"CLM9{int(k.rng.integers(10**7, 10**8))}",
                 others[0] if others else None, others[1 % max(len(others), 1)] if others else None, None][j]
        k.set_version(c, relationship="RESUBMISSION", version_no=2, prior_claim_sk=prior, resubmission_type="CORRECTION",
                      changed_fields={"attachment_ref": [None, "provided"]})
    k.record("resubmission with a missing, unknown or unrelated original", ["PAY-08-R01"], ids)
    k.record("claims wrongly named as originals by other members' resubmissions", ["PAY-08-R01"], others,
             positive=False, note="referenced, not themselves at fault")
    picks = k.take(2)
    for c in picks["claim_sk"]:
        k.set_version(c, relationship="RESUBMISSION", version_no=2, prior_claim_sk=None, resubmission_type="LEGACY",
                      changed_fields={})
    k.record("legacy-route resubmission without a prior reference (look-alike)", ["PAY-08-R01"], picks["claim_sk"],
             positive=False)

    em = list(_EM_UP)
    # R02: refused for medical necessity, resubmitted with diagnosis and code level swapped
    picks = k.take(10, codes=em, single_line=True)
    r02 = []
    for j, h in enumerate(picks.itertuples(index=False)):
        orig = _denied_original(k, h, "MNEC-003")
        if not orig:
            continue
        code = _EM_UP.get(str(k.first_code.get(h.claim_sk)))
        new = _resubmit(k, orig, days=int(k.rng.integers(5, 20)), code=code, dx=_SWAP_DX[j % len(_SWAP_DX)],
                        decision="PAID" if j % 2 == 0 else "DENIED", denial=None if j % 2 == 0 else "MNEC-003")
        r02 += [orig, new]
    k.record("refused claim resubmitted with diagnosis and code swapped", ["PAY-08-R02"], r02)

    # R03: four resubmissions, a different change each time, paid on the last
    picks = k.take(4, codes=em, single_line=True)
    for h in picks.itertuples(index=False):
        orig = _denied_original(k, h, "MNEC-004")
        if not orig:
            continue
        chain, prev = [orig], orig
        code0 = str(k.first_code.get(h.claim_sk))
        steps = [dict(dx="J20.9", code=None), dict(dx="J20.9", code=_EM_UP.get(code0)),
                 dict(dx="M54.50", code=_EM_UP.get(code0)), dict(dx="L30.9", code=code0)]
        for s, step in enumerate(steps):
            last = s == len(steps) - 1
            new = _resubmit(k, prev, days=10 + 15 * s, code=step["code"], dx=step["dx"],
                            decision="PAID" if last else "DENIED", denial=None if last else "MNEC-004",
                            version=s + 2)
            if not new:
                break
            chain.append(new)
            prev = new
        k.record("service resubmitted four times with a different change each time", ["PAY-08-R03", "PAY-08-R02"],
                 chain)
    # look-alike: formal appeals with nothing changed
    picks = k.take(1, codes=em, single_line=True)
    for h in picks.itertuples(index=False):
        orig = _denied_original(k, h, "MNEC-004")
        chain, prev = [orig], orig
        for s in range(4):
            new = _resubmit(k, prev, days=10 + 15 * s, code=None, dx=None, decision="DENIED" if s < 3 else "PAID",
                            denial="MNEC-004" if s < 3 else None, changed={}, rtype="APPEAL", version=s + 2)
            if not new:
                break
            chain.append(new)
            prev = new
        k.record("four formal appeals of one refusal, nothing changed (look-alike)", ["PAY-08-R03"], chain,
                 positive=False)

    # R04: one clinic whose changed resubmissions after MNEC-004 always get paid
    prov = k.busy_provider("CLINIC", 25)
    if prov:
        picks = k.take(11, codes=em, single_line=True, provider=prov)
        ids = []
        for j, h in enumerate(picks.itertuples(index=False)):
            orig = _denied_original(k, h, "MNEC-004")
            if not orig:
                continue
            new = _resubmit(k, orig, days=int(k.rng.integers(5, 15)), code=_EM_UP.get(str(k.first_code.get(h.claim_sk))),
                            dx=_SWAP_DX[j % len(_SWAP_DX)], decision="PAID")
            ids += [orig, new]
        k.record("clinic's changed resubmissions after one refusal reason always paid", ["PAY-08-R04", "PAY-08-R02"],
                 ids, subject_type="provider", subjects=[prov])


# PAY-09 -----------------------------------------------------------------------

_COSMETIC_NOTES_EN = [
    "Botox injections to the glabellar and forehead lines for cosmetic improvement at the patient's request.",
    "Patient attended for aesthetic dermal filler to the nasolabial folds; tolerated well.",
    "Laser skin resurfacing performed for cosmetic reasons (fine lines). Patient satisfied with result.",
    "Consultation and session for a slimming programme; weight loss programme plan and injections given.",
    "Hair transplant follow-up session; grafts reviewed and scalp care advised.",
    "Teeth whitening session completed; shade improved by four levels.",
]
_COSMETIC_NOTES_AR = [
    "تم إجراء حقن البوتوكس لأغراض تجميلية بناءً على طلب المريضة.",
    "جلسة تقشير كيميائي تجميلي للوجه، ولا توجد مضاعفات.",
]


def _pay09(k: _Kit) -> None:
    w = k.w
    # R01: note describes cosmetic work, claim bills a covered visit / procedure
    picks = k.take(8, codes=list(_EM_UP) + ["17110", "11200", "99211", "99215"])
    docs = []
    for j, h in enumerate(picks.itertuples(index=False)):
        ar = j >= 6
        body = _COSMETIC_NOTES_AR[j - 6] if ar else _COSMETIC_NOTES_EN[j % len(_COSMETIC_NOTES_EN)]
        text = ("*** SYNTHETIC DOCUMENT — GENERATED FOR SOFTWARE TESTING. NOT A REAL CLINICAL RECORD. ***\n"
                f"CLINICAL NOTE (SYNTHETIC)\n{body}\n--- END OF SYNTHETIC DOCUMENT ---\n")
        docs.append({"document_sk": w.new_id("DOCPB", 6), "claim_sk": h.claim_sk, "member_sk": h.member_sk,
                     "provider_sk": h.provider_sk, "doc_type": "CLINICAL_NOTE", "language": "ar" if ar else "en",
                     "text": text, "ocr_confidence": round(float(k.rng.uniform(0.85, 0.99)), 3),
                     "created_at": pd.Timestamp(k.sdate(h.claim_sk)) + pd.Timedelta(hours=14), "version_no": 1,
                     "prior_document_sk": None,
                     "attachment_ref": f"SYNTHETIC_clinical_note_{h.claim_sk}_{'ar' if ar else 'en'}.txt",
                     "is_synthetic": True, "tenant_id": TENANT})
    k.record("clinical note describes a cosmetic service billed under a covered code", ["PAY-09-R01"],
             picks["claim_sk"])
    # look-alikes: negated mention; unreadable scan
    picks = k.take(2)
    for j, h in enumerate(picks.itertuples(index=False)):
        body = ("Eyelid review: visual field restricted by skin fold; functional indication, not cosmetic."
                if j == 0 else "Botox injection for chronic migraine prophylaxis; cosmetic effect incidental.")
        ocr = 0.97 if j == 0 else 0.42
        docs.append({"document_sk": w.new_id("DOCPB", 6), "claim_sk": h.claim_sk, "member_sk": h.member_sk,
                     "provider_sk": h.provider_sk, "doc_type": "CLINICAL_NOTE", "language": "en",
                     "text": "*** SYNTHETIC DOCUMENT ***\n" + body, "ocr_confidence": ocr,
                     "created_at": pd.Timestamp(k.sdate(h.claim_sk)), "version_no": 1, "prior_document_sk": None,
                     "attachment_ref": f"SYNTHETIC_note_{h.claim_sk}.txt", "is_synthetic": True, "tenant_id": TENANT})
    k.record("negated or low-confidence cosmetic mention (look-alike)", ["PAY-09-R01"], picks["claim_sk"],
             positive=False)
    w.append("document", docs)

    # R02: one clinic bills a paid office visit shortly after each refused procedure
    prov = k.busy_provider("CLINIC", 25, max_claims=250)
    if prov:
        picks = k.take(16, provider=prov, single_line=True)
        ids = []
        for h in picks.itertuples(index=False):
            k.deny(h.claim_sk, "MNEC-003")
            code = str(k.first_code.get(h.claim_sk))
            proxy = "99214" if not code.startswith("992") else ("99215" if code != "99215" else "99205")
            dxs = w.tables["diagnosis"]
            dx = str(dxs[dxs["claim_sk"] == h.claim_sk].sort_values("sequence")["code"].iloc[0])
            new = k.claim(member_sk=h.member_sk, provider_sk=prov,
                          service_date=k.sdate(h.claim_sk) + TD(days=int(k.rng.integers(2, 20))),
                          claim_type="OUTPATIENT", lines=[proxy if not code.startswith("99") else "97110"],
                          diagnoses=[dx], decision="PAID")
            ids += [h.claim_sk, new]
        k.record("clinic bills a payable substitute code after refused services", ["PAY-09-R02"], ids,
                 subject_type="provider", subjects=[prov])

    # R03: cosmetic-risk code and diagnosis combinations, outpatient / day case
    combos = [("15823", "L90.8", "DAY_CASE"), ("30520", "M95.0", "DAY_CASE"), ("11200", "L57.0", "OUTPATIENT"),
              ("17110", "L81.4", "OUTPATIENT"), ("15823", "L57.0", "DAY_CASE"), ("11200", "L81.4", "OUTPATIENT")]
    hosts = k.take(len(combos) + 1, share=False)
    ids = []
    for (code, dx, enc), h in zip(combos, hosts.itertuples(index=False)):
        prov_type = "HOSPITAL" if enc == "DAY_CASE" else None
        prov = h.provider_sk
        if prov_type:
            ptab = w.tables["provider"]
            hosp = ptab.loc[ptab["provider_type"] == "HOSPITAL", "provider_sk"].tolist()
            prov = hosp[int(k.rng.integers(len(hosp)))]
        new = k.claim(member_sk=h.member_sk, provider_sk=prov, service_date=k.sdate(h.claim_sk) + TD(days=21),
                      claim_type="OUTPATIENT", lines=[code], diagnoses=[dx], encounter_type=enc,
                      authorization=False, decision="PAID")
        ids.append(new)
    k.record("covered procedure billed with a cosmetic-risk diagnosis", ["PAY-09-R03"], ids)
    # look-alike: same code with a malignancy on the eyelid (reconstructive)
    h = list(hosts.itertuples(index=False))[-1]
    ptab = w.tables["provider"]
    hosp = ptab.loc[ptab["provider_type"] == "HOSPITAL", "provider_sk"].tolist()
    look = k.claim(member_sk=h.member_sk, provider_sk=hosp[0], service_date=k.sdate(h.claim_sk) + TD(days=25),
                   claim_type="OUTPATIENT", lines=["15823"], diagnoses=["L90.8", "C44.1"], encounter_type="DAY_CASE",
                   authorization=False, decision="PAID")
    k.record("eyelid surgery after skin cancer, reconstructive (look-alike)", ["PAY-09-R03"], [look], positive=False)

    # R04: member says the service did not happen / receipt names a different item
    picks = k.take(8)
    conf = []
    for j, h in enumerate(picks.itertuples(index=False)):
        conf.append({"confirmation_sk": w.new_id("MCF", 6), "member_sk": h.member_sk, "claim_sk": h.claim_sk,
                     "service_confirmed": False,
                     "response": ["NOT_ATTENDED: I did not visit this clinic on that date.",
                                  "NOT_RECEIVED: I only collected a sick-leave certificate; no examination.",
                                  "NOT_ATTENDED: I was travelling outside the UAE that week."][j % 3],
                     "response_date": k.sdate(h.claim_sk) + TD(days=int(k.rng.integers(20, 40))),
                     "channel": ["CALL", "APP", "SMS", "PROXY"][j % 4], "tenant_id": TENANT})
    w.append("member_confirmation", conf)
    k.record("member says the billed service did not take place", ["PAY-09-R04"], picks["claim_sk"])
    picks = k.take(5, share=True, no_receipt=True)
    rec = []
    for j, h in enumerate(picks.itertuples(index=False)):
        rec.append({"receipt_sk": w.new_id("RCT", 7), "member_sk": h.member_sk, "claim_sk": h.claim_sk,
                    "provider_sk": h.provider_sk, "amount_paid_aed": round(float(h.patient_share), 2),
                    "item_description": ["Teeth whitening session", "Vitamin IV drip (glow package)",
                                         "Hydrafacial skin treatment", "Slimming body wrap session",
                                         "Laser hair removal, underarms"][j % 5],
                    "receipt_date": k.sdate(h.claim_sk), "tenant_id": TENANT})
    w.append("member_receipt", rec)
    k.record("member receipt names a different item from the one billed", ["PAY-09-R04"], picks["claim_sk"])


# PAY-10 -----------------------------------------------------------------------


def _pay10(k: _Kit) -> None:
    w = k.w
    payers = sorted(w.tables["claim_header"]["payer_id"].dropna().unique())
    # R02: another insurer paid the whole bill and we paid our share as well
    picks = k.take(6, extra=lambda h: h["gross_amount"].astype(float) > 200)
    rows = []
    for h in picks.itertuples(index=False):
        other = [p for p in payers if p != h.payer_id][0]
        rows.append({"claim_sk": h.claim_sk, "other_payer_id": other,
                     "payment_amount": round(float(h.gross_amount) * float(k.rng.choice([0.8, 1.0])), 2),
                     "settlement_date": k.settle(h.claim_sk) + TD(days=int(k.rng.integers(5, 30))),
                     "cross_payer_match_token": h.cross_payer_match_token, "tenant_id": TENANT})
    k.record("second insurer paid the bill and we paid too", ["PAY-10-R02"], picks["claim_sk"])
    # look-alike: secondary insurer paid only the member's share (coordinated)
    picks = k.take(2, share=True)
    for h in picks.itertuples(index=False):
        other = [p for p in payers if p != h.payer_id][0]
        rows.append({"claim_sk": h.claim_sk, "other_payer_id": other, "payment_amount": round(float(h.patient_share), 2),
                     "settlement_date": k.settle(h.claim_sk) + TD(days=12),
                     "cross_payer_match_token": h.cross_payer_match_token, "tenant_id": TENANT})
    k.record("secondary insurer paid only the member's share (look-alike)", ["PAY-10-R02"], picks["claim_sk"],
             positive=False)
    w.append("other_payer_remittance", rows)

    # R03: road-traffic / work injuries with no liable-party record
    picks = k.take(9)
    ids = list(picks["claim_sk"])
    for j, c in enumerate(ids):
        if j < 3:
            _add_dx(k, c, "V43.52XA")
        elif j < 5:
            _add_dx(k, c, "V47.5XXA")
        elif j < 7:
            _add_dx(k, c, "W23.0XXA")
            _add_dx(k, c, "Y99.0")
        else:
            w.set_values("claim_header", w.tables["claim_header"]["claim_sk"] == c, accident_indicator=True)
    k.record("road-traffic or work injury with no liable-party record", ["PAY-10-R03"], ids)
    picks = k.take(2)
    for c in picks["claim_sk"]:
        _add_dx(k, c, "V00.131A")
    k.record("skateboard fall on the false-positive list (look-alike)", ["PAY-10-R03"], picks["claim_sk"],
             positive=False)

    # R04: a liable third party settled first; we then paid in full without offset
    picks = k.take(6, extra=lambda h: h["gross_amount"].astype(float) > 150)
    tpl, tps, ledger = [], [], []
    ids = list(picks["claim_sk"])
    for j, h in enumerate(picks.itertuples(index=False)):
        w.set_values("claim_header", w.tables["claim_header"]["claim_sk"] == h.claim_sk, accident_indicator=True)
        party = f"Synthetic Motor Insurer {1 + j % 3}"
        tpl.append({"claim_sk": h.claim_sk, "member_sk": h.member_sk, "accident_type": "ROAD_TRAFFIC",
                    "liable_party": party, "reported_date": k.sdate(h.claim_sk) + TD(days=1), "tenant_id": TENANT})
        sd = k.settle(h.claim_sk) - TD(days=int(k.rng.integers(5, 12)))
        amount = round(float(h.gross_amount) * float(k.rng.uniform(0.7, 1.0)), 2)
        tps.append({"claim_sk": h.claim_sk, "settlement_amount": amount, "settlement_date": sd, "payer": party,
                    "tenant_id": TENANT})
        if j == len(ids) - 1:
            ledger.append({"ledger_sk": w.new_id("LED", 6), "provider_sk": h.provider_sk, "claim_sk": h.claim_sk,
                           "credit_amount": amount, "credit_date": k.settle(h.claim_sk) + TD(days=20), "applied": True,
                           "applied_date": k.settle(h.claim_sk) + TD(days=31), "tenant_id": TENANT})
    k.record("paid in full after a documented third-party settlement", ["PAY-10-R04"], ids[:-1])
    k.record("paid after settlement but recovery already opened (look-alike)", ["PAY-10-R04"], ids[-1:],
             positive=False)
    w.append("third_party_liability", tpl)
    w.append("third_party_settlement", tps)
    w.append("recovery_ledger", ledger)


# PAY-11 -----------------------------------------------------------------------


def _override(k: _Kit, claim: str, actor: str, role: str, reason: str | None) -> None:
    """Replace a claim's review trail with a failed edit that a person overrode."""
    ev = k.w.tables["adjudication_event"]
    h = k.header(claim)
    net = float(h["net_amount"])
    k.w.tables["adjudication_event"] = ev[ev["claim_sk"] != claim].copy()
    t0 = pd.Timestamp(h["submission_date"]) + pd.Timedelta(hours=10)
    k.event(claim, "SYSTEM", "AUTO_ADJUDICATION_ENGINE", "AUTO_EDIT", "PEND", None, net, net, t0, "FAIL")
    k.event(claim, actor, role, "MANUAL_REVIEW", "APPROVE", reason, net, net, t0 + pd.Timedelta(days=3, hours=4),
            "OVERRIDDEN")


def _pay11(k: _Kit) -> None:
    w = k.w
    adj = list(k.ctx.adjudicators)
    # R01: overrides with no reason, or by a role that cannot override
    big = lambda h: h["net_amount"].astype(float) > 80  # noqa: E731
    picks = k.take(9, claim_type=None, extra=big)
    for j, c in enumerate(picks["claim_sk"]):
        if j % 2 == 0:
            _override(k, c, adj[j % len(adj)], "CLAIMS_ADJUDICATOR", None)
        else:
            _override(k, c, f"CLK{j:02d}", "CLAIMS_CLERK", "Provider called; approved.")
    k.record("payment check overridden without a reason or by an unauthorised role", ["PAY-11-R01"],
             picks["claim_sk"])
    picks = k.take(2, claim_type=None)
    for c in picks["claim_sk"]:
        _override(k, c, "CLK90", "CLAIMS_CLERK", "Emergency escalation: on-call medical director approved by phone.")
    k.record("override by a clerk under emergency escalation (look-alike)", ["PAY-11-R01"], picks["claim_sk"],
             positive=False)

    # R02: one adjudicator overriding one hospital's claims
    prov = k.busy_provider("HOSPITAL", 25, claim_type="INPATIENT") or k.busy_provider("HOSPITAL", 25)
    if prov:
        actor = adj[3]
        picks = k.take(10, claim_type=None, provider=prov)
        for c in picks["claim_sk"]:
            _override(k, c, actor, "CLAIMS_ADJUDICATOR",
                      str(k.rng.choice(["Supporting documents reviewed; service medically necessary.",
                                        "Authorisation confirmed with the provider; edit was a false positive."])))
        k.record("one adjudicator overrides one hospital's claims far more than colleagues",
                 ["PAY-11-R02"], picks["claim_sk"], subject_type="adjudicator", subjects=[actor],
                 note=f"provider {prov}")

    # R03: upward payment adjustments after settlement with no appeal or true-up
    picks = k.take(7, claim_type=None, extra=lambda h: h["net_amount"].astype(float) > 100)
    ids = list(picks["claim_sk"])
    for j, h in enumerate(picks.itertuples(index=False)):
        rem = w.tables["remittance"]
        line = rem.loc[rem["claim_sk"] == h.claim_sk, "line_sk"].iloc[0]
        inc = round(float(h.net_amount) * float(k.rng.choice([0.25, 0.4, 0.6])), 2)
        k.remit_row(h.claim_sk, line, "ADJUSTED", inc, -inc, k.settle(h.claim_sk) + TD(days=int(k.rng.integers(25, 70))))
        if j == len(ids) - 1:
            t = pd.Timestamp(k.settle(h.claim_sk)) + pd.Timedelta(days=20)
            k.event(h.claim_sk, adj[5], "SENIOR_ADJUDICATOR", "APPEAL_DECISION", "APPROVE",
                    "Appeal upheld: additional time documented", h.net_amount, h.net_amount + inc, t, "REVIEWED")
    k.record("payment increased after settlement with no reason on record", ["PAY-11-R03"], ids[:-1])
    k.record("payment increased after an upheld appeal (look-alike)", ["PAY-11-R03"], ids[-1:], positive=False)

    # R04: one adjudicator pays one provider's pended claims in full, every time
    prov = k.busy_provider("CLINIC", 30, max_claims=400)
    if prov:
        # the adjudicator who has decided the fewest of this provider's claims so far
        ev = w.tables["adjudication_event"]
        hp = w.tables["claim_header"]
        mine = ev[(ev["event_type"] == "MANUAL_REVIEW") & ev["claim_sk"].isin(hp.loc[hp["provider_sk"] == prov, "claim_sk"])]
        seen = mine["actor"].value_counts()
        actor = min((a for a in adj if a != adj[3]), key=lambda a: (int(seen.get(a, 0)), a))
        picks = k.take(15, claim_type=None, provider=prov, extra=lambda h: h["net_amount"].astype(float) > 50)
        for h in picks.itertuples(index=False):
            ev = w.tables["adjudication_event"]
            w.tables["adjudication_event"] = ev[ev["claim_sk"] != h.claim_sk].copy()
            t0 = pd.Timestamp(h.submission_date) + pd.Timedelta(hours=9)
            k.event(h.claim_sk, "SYSTEM", "AUTO_ADJUDICATION_ENGINE", "AUTO_EDIT", "PEND", None, h.net_amount,
                    h.net_amount, t0, "PEND")
            k.event(h.claim_sk, actor, "CLAIMS_ADJUDICATOR", "MANUAL_REVIEW", "APPROVE", None, h.net_amount,
                    h.net_amount, t0 + pd.Timedelta(days=2), "REVIEWED")
        k.record("one adjudicator pays one provider's pended claims in full, uniformly", ["PAY-11-R04"],
                 picks["claim_sk"], subject_type="adjudicator", subjects=[actor], note=f"provider {prov}")


# PAY-12 -----------------------------------------------------------------------


def _pay12(k: _Kit) -> None:
    w = k.w
    end = w.end
    early = lambda h: (pd.to_datetime(h["settlement_date"]) < pd.Timestamp(end - TD(days=150))) & (  # noqa: E731
        h["net_amount"].astype(float) > 80)
    big = lambda h: h["net_amount"].astype(float) > 80  # noqa: E731
    # R01: cancelled after payment, payment never reversed
    picks = k.take(8, claim_type=None, extra=early)
    ids = list(picks["claim_sk"])
    for j, h in enumerate(picks.itertuples(index=False)):
        when = pd.Timestamp(k.settle(h.claim_sk)) + pd.Timedelta(days=int(k.rng.integers(10, 60)), hours=10)
        k.set_version(h.claim_sk, relationship="CANCELLATION", version_no=2, prior_claim_sk=None,
                      recorded_at=when.to_pydatetime(), resubmission_type="CANCELLATION",
                      changed_fields={"status": ["ACTIVE", "CANCELLED"]})
        if j >= 6:  # look-alikes: reversed in the next payment run
            rem = w.tables["remittance"]
            for r in rem[rem["claim_sk"] == h.claim_sk].itertuples(index=False):
                if float(r.payment_amount) > 0:
                    k.remit_row(h.claim_sk, r.line_sk, "REVERSED", -float(r.payment_amount), float(r.payment_amount),
                                (when + pd.Timedelta(days=14)).date())
    k.record("claim cancelled after payment; payment never reversed", ["PAY-12-R01"], ids[:6])
    k.record("claim cancelled and reversed in the next payment run (look-alike)", ["PAY-12-R01", "PAY-12-R04"],
             ids[6:], positive=False)

    # R02 (a): the same line paid twice
    picks = k.take(7, claim_type=None, extra=big)
    ids = list(picks["claim_sk"])
    for h in picks.itertuples(index=False):
        rem = w.tables["remittance"]
        r = rem[(rem["claim_sk"] == h.claim_sk) & (rem["payment_amount"].astype(float) > 0)].iloc[0]
        k.remit_row(h.claim_sk, r["line_sk"], "PAID", float(r["payment_amount"]), 0.0,
                    k.settle(h.claim_sk) + TD(days=int(k.rng.integers(14, 45))))
    k.record("same line paid twice", ["PAY-12-R02"], ids)
    # R02 (b): a payment reference reused on another provider's claim, another day
    picks = k.take(2, claim_type=None)
    donors = k.take(2, claim_type=None)
    for h, d in zip(picks.itertuples(index=False), donors.itertuples(index=False)):
        rem = w.tables["remittance"]
        ref = rem.loc[rem["claim_sk"] == d.claim_sk, "payment_reference"].dropna().iloc[0]
        w.set_values("remittance", rem["claim_sk"] == h.claim_sk, payment_reference=ref)
    k.record("payment reference reused across providers and dates", ["PAY-12-R02"],
             list(picks["claim_sk"]) + list(donors["claim_sk"]))
    # look-alike: a line settled in two instalments within its payable amount
    picks = k.take(1, claim_type=None, extra=lambda h: h["net_amount"].astype(float) > 100)
    for h in picks.itertuples(index=False):
        rem = w.tables["remittance"]
        m = (rem["claim_sk"] == h.claim_sk) & (rem["payment_amount"].astype(float) > 0)
        r = rem[m].iloc[0]
        half = round(float(r["payment_amount"]) / 2, 2)
        w.set_values("remittance", rem["remittance_sk"] == r["remittance_sk"], payment_amount=half)
        k.remit_row(h.claim_sk, r["line_sk"], "PAID", round(float(r["payment_amount"]) - half, 2), 0.0,
                    k.settle(h.claim_sk) + TD(days=30))
    k.record("line settled in two instalments (look-alike)", ["PAY-12-R02"], picks["claim_sk"], positive=False)

    # R03: provider refunds never applied, or applied months late
    picks = k.take(6, claim_type=None, extra=early)
    rows = []
    for j, h in enumerate(picks.itertuples(index=False)):
        cd = k.settle(h.claim_sk) + TD(days=int(k.rng.integers(20, 60)))
        late = j == 5
        rows.append({"ledger_sk": w.new_id("LED", 6), "provider_sk": h.provider_sk, "claim_sk": h.claim_sk,
                     "credit_amount": round(float(h.net_amount) * float(k.rng.uniform(0.1, 0.5)), 2),
                     "credit_date": cd, "applied": late, "applied_date": cd + TD(days=140) if late else None,
                     "tenant_id": TENANT})
    w.append("recovery_ledger", rows)
    k.record("provider refund received but never applied (one applied months late)", ["PAY-12-R03"],
             picks["claim_sk"], subject_type="provider", subjects=list(picks["provider_sk"]))

    # R04: one provider's large credits offset against small unrelated claims; negative lines
    prov = k.busy_provider("CLINIC", 20)
    if prov:
        picks = k.take(9, provider=prov, extra=lambda h: h["net_amount"].astype(float) < 400)
        ids = list(picks["claim_sk"])
        for h in list(picks.itertuples(index=False))[:5]:
            rem = w.tables["remittance"]
            line = rem.loc[rem["claim_sk"] == h.claim_sk, "line_sk"].iloc[0]
            credit = float(k.rng.choice([1200, 1850, 2400, 3100]))
            k.remit_row(h.claim_sk, line, "OFFSET", -credit, credit,
                        k.settle(h.claim_sk) + TD(days=int(k.rng.integers(20, 60))))
        for h in list(picks.itertuples(index=False))[5:]:
            lines = w.tables["claim_line"]
            base = lines[lines["claim_sk"] == h.claim_sk].iloc[0].to_dict()
            gross = -round(float(base["gross_amount"]) * 0.5, 2)
            share = -round(float(base["patient_share"] or 0) * 0.5, 2)
            base.update(line_sk=f"{h.claim_sk}-L90", gross_amount=gross, patient_share=share,
                        net_amount=round(gross - share, 2), units=-0.5, indicator="ADJ",
                        unit_price=float(base["unit_price"] or 0))
            w.append("claim_line", [base])
            w.recompute_header_amounts([h.claim_sk])
            k.remit_row(h.claim_sk, base["line_sk"], "PAID", base["net_amount"], 0.0, k.settle(h.claim_sk))
        k.record("provider offsets credits against unrelated claims and bills negative lines", ["PAY-12-R04"], ids,
                 subject_type="provider", subjects=[prov])
