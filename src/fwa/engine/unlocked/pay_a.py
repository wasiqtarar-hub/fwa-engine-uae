"""Unlocked implementations for the PAY family (PAY-01 to PAY-06).

Each function here runs only when a dataset unlock (`rules/unlocks/PAY_A.yaml`)
is satisfied by the loaded file. See :mod:`fwa.engine.unlocks`.

These are the line-level payment-integrity checks the claim-header extract
could not support: cross-insurer duplicates, bundling and package edits, code
validity, demographic and unit edits, prior-authorisation compliance, modifier
(indicator) abuse and tariff pricing. Conventions shared by every function:

* Every table is tenant-scoped when it carries a ``tenant_id`` column, and
  every claim line is restricted to the claims already in ``ctx.claims``.
* Reference tables are applied AS OF the line's service date wherever they
  carry effective dates (§3.6) — never against today's date.
* Nothing raises. A missing table, column or value returns ``[]``.
* Line amounts on the UAE multi-table file are already in AED.
* Hard edits compute exposure with :func:`line_edit_exposure`; the statistical
  controls (PAY-02-R05, PAY-05-R03, PAY-05-R04) carry no established exposure.
"""

from __future__ import annotations

import math
from typing import Any, Callable, Iterable

import numpy as np
import pandas as pd

from ...cases import exposure as _exp
from ...presentation import aed, count_phrase, pct, plain_date, times_phrase
from ...statistical.shrinkage import fit_beta_prior, shrink_rate
from ..controls import _claim_frame, _f, _sig
from ..evallib import is_missing, period_bucket
from ..signals import Signal

IMPLEMENTATIONS: dict[str, Callable] = {}

# Plain vocabularies (not thresholds). Matching is case-insensitive.
_DENIED_STATUSES = {"denied", "rejected", "refused", "cancelled", "canceled", "withdrawn",
                    "revoked", "void", "voided", "expired"}
_PENDING_STATUSES = {"pending", "requested", "submitted", "in_review", "deemed", "deemed_approved"}
_REVERSAL_DECISIONS = {"reversed", "reversal", "cancelled", "canceled", "void", "voided"}
_DENIAL_DECISIONS = {"denied", "rejected", "refused", "declined"}
_DRUG_TYPES = ("drug", "pharm", "medic", "rx")


def _register(name: str):
    def deco(fn):
        IMPLEMENTATIONS[name] = fn
        return fn
    return deco


# ---------------------------------------------------------------------------
# data access helpers
# ---------------------------------------------------------------------------


def _table(ctx, name: str) -> pd.DataFrame:
    ds = getattr(ctx, "dataset", None)
    if ds is None:
        return pd.DataFrame()
    try:
        df = ds.get(name)
    except Exception:
        return pd.DataFrame()
    if df is None or df.empty:
        return pd.DataFrame(columns=[] if df is None else df.columns)
    if "tenant_id" in df.columns and df["tenant_id"].notna().any():
        df = df[df["tenant_id"].astype(str) == str(ctx.tenant_id)]
    return df


def _col(df: pd.DataFrame, name: str, default: Any = np.nan) -> pd.Series:
    if name in df.columns:
        return df[name]
    return pd.Series(default, index=df.index, dtype=object)


def _num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def _date(s: pd.Series) -> pd.Series:
    out = pd.to_datetime(s, errors="coerce")
    try:
        if getattr(out.dt, "tz", None) is not None:
            out = out.dt.tz_localize(None)
    except Exception:  # pragma: no cover
        pass
    return out


def _norm(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.lower().where(s.notna(), "")


def _truthy(s: pd.Series) -> pd.Series:
    if s.dtype == bool:
        return s.fillna(False)
    return s.map(lambda v: (not is_missing(v)) and str(v).strip().lower() in
                 {"true", "1", "yes", "y", "t", "1.0"}).astype(bool)


def _has(s: pd.Series) -> pd.Series:
    """Non-missing, non-blank."""
    return s.notna() & (s.astype(str).str.strip() != "") & (s.astype(str).str.lower() != "nan") \
        & (s.astype(str).str.lower() != "none")


def _in_force(dates: pd.Series, valid_from: pd.Series, valid_to: pd.Series) -> pd.Series:
    vf = _date(valid_from)
    vt = _date(valid_to)
    d = _date(dates)
    ok_from = vf.isna() | (d >= vf)
    ok_to = vt.isna() | (d <= vt)
    return ok_from & ok_to


def _str_id(v: Any) -> str:
    return "" if is_missing(v) else str(v)


def _day(v: Any) -> str | None:
    if is_missing(v):
        return None
    try:
        ts = pd.Timestamp(v)
        return None if pd.isna(ts) else ts.strftime("%Y-%m-%d")
    except Exception:
        return str(v)[:10]


def _money(v: Any) -> float:
    x = _f(v)
    return 0.0 if math.isnan(x) else round(float(x), 2)


_HEADER_COLS = ["claim_sk", "member_sk", "provider_sk", "payer_id", "claim_type", "service_date",
                "gross_amount_aed", "length_of_stay_days", "submission_date", "claim_date"]


def _header(ctx) -> pd.DataFrame:
    df = _claim_frame(ctx)
    out = pd.DataFrame({c: _col(df, c) for c in _HEADER_COLS})
    out["claim_sk"] = out["claim_sk"].astype(str)
    out["service_date"] = _date(out["service_date"])
    return out


def _lines(ctx) -> pd.DataFrame:
    """Claim lines of this tenant's claims, joined to their header, typed and cached."""
    raw = _table(ctx, "claim_line")
    cache = getattr(ctx, "_pay_a_cache", None)
    key = (id(raw), len(raw), id(ctx.claims), len(ctx.claims))
    if isinstance(cache, dict) and cache.get("lines_key") == key:
        return cache["lines"]
    if raw.empty or "claim_sk" not in raw.columns:
        out = pd.DataFrame()
    else:
        cl = raw.copy()
        cl["claim_sk"] = cl["claim_sk"].astype(str)
        hdr = _header(ctx)
        out = cl.merge(hdr, on="claim_sk", how="inner", suffixes=("", "_hdr"))
        if not out.empty:
            line_date = _date(_col(out, "service_date"))
            out["line_date"] = line_date.fillna(_date(out["service_date_hdr"]))
            out["line_date"] = out["line_date"].dt.normalize()
            out["line_sk"] = _col(out, "line_sk").astype(str)
            out["activity_code"] = _col(out, "activity_code").map(_str_id)
            out["units_n"] = _num(_col(out, "units")).fillna(1.0)
            gross = _num(_col(out, "gross_amount"))
            net = _num(_col(out, "net_amount"))
            out["billed"] = gross.fillna(net).fillna(0.0)
            out["payable"] = net.fillna(gross).fillna(0.0)
            up = _num(_col(out, "unit_price"))
            out["unit_price_n"] = up.fillna(out["billed"] / out["units_n"].where(out["units_n"] > 0))
            out["indicator_s"] = _col(out, "indicator").map(_str_id).str.strip()
            out["activity_type_s"] = _norm(_col(out, "activity_type"))
            out["member_sk"] = out["member_sk"].map(_str_id)
            out["provider_sk"] = out["provider_sk"].map(_str_id)
    try:
        ctx._pay_a_cache = {"lines_key": key, "lines": out}
    except Exception:  # pragma: no cover
        pass
    return out


def _codes(ctx) -> pd.DataFrame:
    ref = _table(ctx, "activity_code_reference")
    if ref.empty or "activity_code" not in ref.columns:
        return pd.DataFrame(columns=["activity_code"])
    ref = ref.copy()
    ref["activity_code"] = ref["activity_code"].map(_str_id)
    return ref.drop_duplicates("activity_code")


def _describer(ctx) -> Callable[[str], str]:
    ref = _codes(ctx)
    names = {}
    if "description" in ref.columns:
        names = {c: str(d) for c, d in zip(ref["activity_code"], ref["description"]) if not is_missing(d)}

    def describe(code: Any) -> str:
        c = _str_id(code)
        d = names.get(c)
        return f"{d} ({c})" if d else f"code {c}"
    return describe


def _is_drug(df: pd.DataFrame) -> pd.Series:
    t = df["activity_type_s"] if "activity_type_s" in df.columns else pd.Series("", index=df.index)
    mask = pd.Series(False, index=df.index)
    for token in _DRUG_TYPES:
        mask |= t.str.contains(token, regex=False)
    return mask


def _base_evidence(control, subject_id: str, **facts: Any) -> dict[str, Any]:
    ev = {
        "rule_id": control.rule_id,
        "scenario_id": control.scenario_id,
        "subject_id": str(subject_id),
        "reason_code": control.reason_code,
    }
    ev.update(facts)
    return ev


def _authorisations(ctx) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, str]]:
    """Authorisations, their lines, and a map from any identifier a claim may carry to the key."""
    auth = _table(ctx, "authorization")
    al = _table(ctx, "authorization_line")
    if auth.empty or "authorization_sk" not in auth.columns:
        return pd.DataFrame(), pd.DataFrame(), {}
    auth = auth.copy()
    auth["authorization_sk"] = auth["authorization_sk"].map(_str_id)
    auth["status_s"] = _norm(_col(auth, "status"))
    auth["vf"] = _date(_col(auth, "valid_from")).dt.normalize()
    auth["vt"] = _date(_col(auth, "valid_to")).dt.normalize()
    auth["member_sk"] = _col(auth, "member_sk").map(_str_id)
    auth["provider_sk"] = _col(auth, "provider_sk").map(_str_id)
    ids: dict[str, str] = {}
    for column in ("response_id", "request_id", "authorization_sk"):
        if column in auth.columns:
            for v, k in zip(auth[column], auth["authorization_sk"]):
                if not is_missing(v) and str(v).strip():
                    ids[str(v).strip()] = k
    if not al.empty and "authorization_sk" in al.columns:
        al = al.copy()
        al["authorization_sk"] = al["authorization_sk"].map(_str_id)
        al["activity_code"] = _col(al, "activity_code").map(_str_id)
        al["line_denied"] = _has(_col(al, "denial_code"))
        al["approved_units_n"] = _num(_col(al, "approved_units"))
        al["approved_value_n"] = _num(_col(al, "approved_value"))
    else:
        al = pd.DataFrame(columns=["authorization_sk", "activity_code", "line_denied",
                                   "approved_units_n", "approved_value_n"])
    return auth, al, ids


def _approved(status: pd.Series) -> pd.Series:
    s = _norm(status)
    return ~s.isin(_DENIED_STATUSES) & ~s.isin(_PENDING_STATUSES) & (s != "")


def _linked_lines(ctx) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Lines that carry an authorisation id, resolved to the authorisation they point at."""
    lines = _lines(ctx)
    auth, al, ids = _authorisations(ctx)
    if lines.empty or auth.empty:
        return pd.DataFrame(), auth, al
    ln = lines[_has(_col(lines, "authorization_id"))].copy()
    if ln.empty:
        return ln, auth, al
    ln["auth_key"] = ln["authorization_id"].astype(str).str.strip().map(ids)
    return ln, auth, al


def _approved_code_pairs(auth: pd.DataFrame, al: pd.DataFrame) -> pd.DataFrame:
    """(authorization_sk, activity_code, member, provider, validity) of every approved, undenied line."""
    if auth.empty or al.empty:
        return pd.DataFrame(columns=["authorization_sk", "activity_code", "member_sk", "vf", "vt"])
    ok = auth[_approved(auth["status_s"])]
    pairs = al[~al["line_denied"]].merge(
        ok[["authorization_sk", "member_sk", "provider_sk", "vf", "vt", "status_s"]],
        on="authorization_sk", how="inner")
    return pairs


def _emergency_claims(ctx) -> set[str]:
    out: set[str] = set()
    hdr = _claim_frame(ctx)
    if "claim_type" in hdr.columns:
        m = _norm(hdr["claim_type"]).str.contains("emerg", regex=False)
        out |= set(hdr.loc[m, "claim_sk"].astype(str))
    enc = _table(ctx, "encounter")
    if not enc.empty and "claim_sk" in enc.columns:
        m = pd.Series(False, index=enc.index)
        for c in ("encounter_type", "admission_type"):
            if c in enc.columns:
                m |= _norm(enc[c]).str.contains("emerg", regex=False)
        out |= set(enc.loc[m, "claim_sk"].astype(str))
    return out


def _inactive_lines(ctx) -> set[str]:
    """Lines that no longer stand: on a claim cancelled or replaced by a later version,
    or refused outright or reversed at remittance."""
    out: set[str] = set()
    lines = _lines(ctx)
    if lines.empty:
        return out
    cv = _table(ctx, "claim_version")
    dropped: set[str] = set()
    if not cv.empty and "claim_sk" in cv.columns:
        canc = _norm(_col(cv, "relationship")).str.contains("cancel|void|revers", regex=True)
        dropped = set(cv.loc[canc, "claim_sk"].astype(str)) | set(_col(cv, "prior_claim_sk").dropna().astype(str))
    out |= set(lines.loc[lines["claim_sk"].isin(dropped), "line_sk"])
    paid = _paid_by_line(ctx)
    if not paid.empty:
        gone = paid["reversed"].astype(bool) | (paid["denied"].astype(bool) & (paid["paid"] <= 0))
        out |= set(paid.loc[gone, "line_sk"])
    return out


def _paid_by_line(ctx) -> pd.DataFrame:
    """Remittance summed per line: payment, whether denied, whether reversed."""
    rem = _table(ctx, "remittance")
    if rem.empty or "line_sk" not in rem.columns:
        return pd.DataFrame(columns=["line_sk", "paid", "denied", "reversed", "denial_code"])
    cache = getattr(ctx, "_pay_a_cache", None)
    key = (id(rem), len(rem))
    if isinstance(cache, dict) and cache.get("paid_key") == key:
        return cache["paid"]
    r = rem[_has(rem["line_sk"])].copy()
    r["line_sk"] = r["line_sk"].astype(str)
    r["pay"] = _num(_col(r, "payment_amount")).fillna(0.0)
    dec = _norm(_col(r, "decision"))
    r["is_denied"] = dec.isin(_DENIAL_DECISIONS) | _has(_col(r, "denial_code"))
    r["is_reversed"] = dec.isin(_REVERSAL_DECISIONS)
    r["dcode"] = _col(r, "denial_code").map(_str_id)
    g = r.groupby("line_sk").agg(paid=("pay", "sum"), denied=("is_denied", "any"),
                                 reversed=("is_reversed", "any")).reset_index()
    codes = r[r["dcode"] != ""].drop_duplicates("line_sk").set_index("line_sk")["dcode"]
    g["denial_code"] = g["line_sk"].map(codes).fillna("")
    if isinstance(cache, dict):
        cache["paid_key"], cache["paid"] = key, g
    return g


def _peer_group(ctx) -> dict[str, str]:
    prov = _table(ctx, "provider")
    if prov.empty or "provider_sk" not in prov.columns:
        return {}
    parts = []
    for c in ("provider_type", "specialty"):
        if c in prov.columns:
            parts.append(prov[c].map(_str_id))
    if not parts:
        return {}
    label = parts[0]
    for p in parts[1:]:
        label = label + " / " + p
    return dict(zip(prov["provider_sk"].map(_str_id), label))


# ===========================================================================
# PAY-01-R04 — cross-payer duplicate
# ===========================================================================


@_register("pay_01_r04_cross_payer_duplicate")
def pay_01_r04_cross_payer_duplicate(ctx, control) -> list[Signal]:
    """The same service paid by this payer and, through a privacy-preserving
    match reference, by another insurer, beyond what coordination allows.

    A payment by the other insurer is legitimate when a coordination-of-benefits
    record places that insurer first (or this payer first) on the service date
    AND the two payments together stay within the billed amount. It fires when
    both insurers paid and either no coordination record covers the member, or
    the combined payments exceed the billed amount by more than the tolerance.
    """
    hdr = _header(ctx)
    opr = _table(ctx, "other_payer_remittance")
    if hdr.empty or opr.empty:
        return []
    tol = float(ctx.cfg("pay01_cross_payer_amount_tolerance_aed"))
    full = _claim_frame(ctx)
    hdr["token"] = _col(full, "cross_payer_match_token").map(_str_id).values
    hdr["gross"] = _num(_col(full, "gross_amount_aed")).fillna(_num(_col(full, "gross_amount"))).values

    # What this payer paid on each claim.
    rem = _table(ctx, "remittance")
    if rem.empty or "claim_sk" not in rem.columns:
        return []
    r = rem.copy()
    r["claim_sk"] = r["claim_sk"].astype(str)
    r["pay"] = _num(_col(r, "payment_amount")).fillna(0.0)
    r["rev"] = _norm(_col(r, "decision")).isin(_REVERSAL_DECISIONS)
    ours = r.groupby("claim_sk").agg(our_paid=("pay", "sum"), reversed=("rev", "any")).reset_index()

    o = opr.copy()
    o["claim_sk"] = _col(o, "claim_sk").map(_str_id)
    o["token"] = _col(o, "cross_payer_match_token").map(_str_id)
    o["other_paid"] = _num(_col(o, "payment_amount")).fillna(0.0)
    o["other_payer_id"] = _col(o, "other_payer_id").map(_str_id)
    o["other_settled"] = _col(o, "settlement_date")
    o = o[o["other_paid"] > 0]
    if o.empty:
        return []
    # Match on the privacy-preserving token first; fall back to a shared claim key.
    by_token = hdr[hdr["token"] != ""].merge(o[o["token"] != ""].drop(columns=["claim_sk"]),
                                             on="token", how="inner")
    by_claim = hdr.merge(o[o["token"] == ""].drop(columns=["token"]), on="claim_sk", how="inner")
    m = pd.concat([by_token, by_claim], ignore_index=True)
    if m.empty:
        return []
    m = m[m["other_payer_id"] != m["payer_id"].map(_str_id)]
    m = m.merge(ours, on="claim_sk", how="left")
    m["our_paid"] = m["our_paid"].fillna(0.0)
    m = m[(m["our_paid"] > 0) & ~m["reversed"].fillna(False).astype(bool)]
    if m.empty:
        return []
    agg = m.groupby("claim_sk").agg(
        other_paid=("other_paid", "sum"), other_payers=("other_payer_id", lambda s: sorted(set(s))),
        our_paid=("our_paid", "first"), gross=("gross", "first"), member_sk=("member_sk", "first"),
        provider_sk=("provider_sk", "first"), payer_id=("payer_id", "first"),
        service_date=("service_date", "first"), token=("token", "first"),
        other_settled=("other_settled", "max"),
    ).reset_index()

    cob = _table(ctx, "coordination_of_benefits")
    cob_ok: dict[str, list[dict[str, Any]]] = {}
    if not cob.empty and "member_sk" in cob.columns:
        for row in cob.itertuples(index=False):
            cob_ok.setdefault(_str_id(row.member_sk), []).append({
                "primary": _str_id(getattr(row, "primary_payer", None)),
                "secondary": _str_id(getattr(row, "secondary_payer", None)),
                "vf": getattr(row, "valid_from", None), "vt": getattr(row, "valid_to", None),
            })

    out: list[Signal] = []
    for row in agg.itertuples(index=False):
        payers = {_str_id(row.payer_id), *row.other_payers}
        cover = None
        for rec in cob_ok.get(_str_id(row.member_sk), []):
            if {rec["primary"], rec["secondary"]} <= payers:
                if bool(_in_force(pd.Series([row.service_date]), pd.Series([rec["vf"]]),
                                  pd.Series([rec["vt"]])).iloc[0]):
                    cover = rec
                    break
        gross = _money(row.gross)
        combined = _money(row.our_paid + row.other_paid)
        over = combined - gross
        if cover is not None and over <= tol:
            continue  # declared exclusion: coordination of benefits on record, within the bill
        correct = max(0.0, gross - float(row.other_paid))
        exposure = _exp.line_edit_exposure(float(row.our_paid), min(float(row.our_paid), correct))
        why = ("no coordination-of-benefits record covers this patient" if cover is None else
               f"together the insurers paid {aed(combined)} against a bill of {aed(gross)}")
        out.append(_sig(
            ctx, control, subject_type="claim", subject_id=row.claim_sk,
            fact_key=f"crosspay:{row.claim_sk}", claim_ids=[row.claim_sk],
            event_time=row.service_date, period=period_bucket(row.service_date),
            evidence=_base_evidence(
                control, row.claim_sk,
                plain_language=(
                    f"This payer paid {aed(row.our_paid)} for this service on {plain_date(row.service_date)} "
                    f"and another insurer paid {aed(row.other_paid)} for the same service (matched "
                    f"through a privacy-protected reference); {why}."
                ),
                what_the_reviewer_must_verify=(
                    "Confirm with the other insurer that it paid this same service, and that the "
                    "data sharing behind the match is authorised."),
                member_sk=row.member_sk, provider_sk=row.provider_sk,
                our_payer_id=_str_id(row.payer_id), other_payer_ids=list(row.other_payers),
                our_paid_aed=_money(row.our_paid), other_paid_aed=_money(row.other_paid),
                billed_amount_aed=gross, combined_paid_aed=combined,
                cross_payer_match_reference=row.token or None,
                coordination_record=None if cover is None else {
                    "primary_payer": cover["primary"], "secondary_payer": cover["secondary"]},
                amount_tolerance_aed=tol,
                other_settlement_date=_day(row.other_settled),
            ),
            exposure=exposure,
            confidence=0.9,
        ))
    return out


# ===========================================================================
# PAY-02 — unbundling, packages and inclusive leakage
# ===========================================================================


def _edits(ctx) -> pd.DataFrame:
    e = _table(ctx, "bundling_edit_table")
    if e.empty or "column_1_code" not in e.columns or "column_2_code" not in e.columns:
        return pd.DataFrame()
    e = e.copy()
    e["col1"] = e["column_1_code"].map(_str_id)
    e["col2"] = e["column_2_code"].map(_str_id)
    e["mod_ok"] = _truthy(_col(e, "modifier_allowed"))
    e["edit_type_s"] = _col(e, "edit_type").map(_str_id)
    e["e_vf"] = _col(e, "valid_from")
    e["e_vt"] = _col(e, "valid_to")
    return e[(e["col1"] != "") & (e["col2"] != "") & (e["col1"] != e["col2"])]


def _same_claim_pairs(ctx) -> pd.DataFrame:
    """Every (parent line, component line) on one claim that matches an edit in force."""
    lines = _lines(ctx)
    e = _edits(ctx)
    if lines.empty or e.empty:
        return pd.DataFrame()
    comp = lines.merge(e, left_on="activity_code", right_on="col2", how="inner")
    if comp.empty:
        return pd.DataFrame()
    par = lines[["claim_sk", "activity_code", "line_sk", "billed", "indicator_s"]].rename(columns={
        "activity_code": "col1", "line_sk": "parent_line_sk", "billed": "parent_billed",
        "indicator_s": "parent_indicator"})
    m = comp.merge(par, on=["claim_sk", "col1"], how="inner")
    m = m[m["line_sk"] != m["parent_line_sk"]]
    if m.empty:
        return m
    m = m[_in_force(m["line_date"], m["e_vf"], m["e_vt"])]
    return m.sort_values(["line_sk", "parent_billed"], ascending=[True, False])


@_register("pay_02_r01_same_claim_component")
def pay_02_r01_same_claim_component(ctx, control) -> list[Signal]:
    """A component billed as its own payable line beside its parent on the same
    claim, where the bundling edit in force on the service date includes it.
    A permitted indicator on the component separates the pair only when the
    edit allows a modifier."""
    m = _same_claim_pairs(ctx)
    if m.empty:
        return []
    m = m[m["billed"] > 0]
    m = m[~(m["mod_ok"] & (m["indicator_s"] != ""))]  # declared exclusion: permitted_indicator
    m = m.drop_duplicates("line_sk")
    desc = _describer(ctx)
    out: list[Signal] = []
    for r in m.itertuples(index=False):
        out.append(_sig(
            ctx, control, subject_type="claim", subject_id=r.claim_sk,
            fact_key=f"bundle:{r.line_sk}", claim_ids=[r.claim_sk],
            event_time=r.line_date, period=period_bucket(r.line_date),
            evidence=_base_evidence(
                control, r.claim_sk,
                plain_language=(
                    f"Lines billed for {desc(r.col1)} ({aed(r.parent_billed)}) and its included "
                    f"component, {desc(r.col2)} ({aed(r.billed)}), on the same claim; the bundling "
                    f"table in force on {plain_date(r.line_date)} says the component is included."
                ),
                what_the_reviewer_must_verify=(
                    "Check the bundling rule version that applied and whether a permitted modifier "
                    "allows separate payment."),
                member_sk=r.member_sk, provider_sk=r.provider_sk,
                parent_line_sk=r.parent_line_sk, parent_code=r.col1,
                component_line_sk=r.line_sk, component_code=r.col2,
                component_amount_aed=_money(r.billed), parent_amount_aed=_money(r.parent_billed),
                component_indicator=r.indicator_s or None, edit_type=r.edit_type_s or None,
                edit_allows_modifier=bool(r.mod_ok),
                edit_valid_from=_day(r.e_vf), edit_valid_to=_day(r.e_vt),
                service_date=_day(r.line_date),
            ),
            exposure=_exp.line_edit_exposure(_money(r.billed), 0.0),
        ))
    return out


@_register("pay_02_r02_cross_claim_component")
def pay_02_r02_cross_claim_component(ctx, control) -> list[Signal]:
    """An included component billed on a DIFFERENT claim for the same patient —
    by the same or another provider — within the global window of its parent."""
    lines = _lines(ctx)
    e = _edits(ctx)
    if lines.empty or e.empty:
        return []
    window = int(ctx.cfg("pay02_cross_claim_window_days"))
    split = {str(x).strip().upper() for x in (ctx.cfg("pay02_professional_split_indicators") or [])}
    comp = lines[lines["billed"] > 0].merge(e, left_on="activity_code", right_on="col2", how="inner")
    if comp.empty:
        return []
    par = lines[["claim_sk", "member_sk", "provider_sk", "activity_code", "line_sk", "billed",
                 "line_date", "indicator_s"]].rename(columns={
        "claim_sk": "parent_claim_sk", "provider_sk": "parent_provider_sk", "activity_code": "col1",
        "line_sk": "parent_line_sk", "billed": "parent_billed", "line_date": "parent_date",
        "indicator_s": "parent_indicator"})
    m = comp.merge(par, on=["member_sk", "col1"], how="inner")
    m = m[(m["parent_claim_sk"] != m["claim_sk"]) & (m["member_sk"] != "")]
    if m.empty:
        return []
    gap = (m["line_date"] - m["parent_date"]).dt.days
    m = m.assign(days_apart=gap)
    m = m[m["days_apart"].abs() <= window]
    m = m[_in_force(m["line_date"], m["e_vf"], m["e_vt"])]
    # declared exclusion: contractual professional / facility split
    ind_c = m["indicator_s"].str.upper()
    ind_p = m["parent_indicator"].str.upper()
    m = m[~(ind_c.isin(split) | ind_p.isin(split))]
    # a modifier the edit permits separates the pair here as it does on one claim
    m = m[~(m["mod_ok"] & (m["indicator_s"] != ""))]
    if m.empty:
        return []
    m = m.assign(abs_gap=m["days_apart"].abs()).sort_values(["line_sk", "abs_gap"]).drop_duplicates("line_sk")
    desc = _describer(ctx)
    out: list[Signal] = []
    for r in m.itertuples(index=False):
        same_prov = r.parent_provider_sk == r.provider_sk
        who = "the same provider" if same_prov else "a different provider"
        when = "the same day" if int(r.days_apart) == 0 else f"{abs(int(r.days_apart))} day(s) apart"
        out.append(_sig(
            ctx, control, subject_type="claim", subject_id=r.claim_sk,
            fact_key=f"xbundle:{r.line_sk}", claim_ids=[r.claim_sk, r.parent_claim_sk],
            event_time=r.line_date, period=period_bucket(r.line_date),
            evidence=_base_evidence(
                control, r.claim_sk,
                plain_language=(
                    f"{desc(r.col2)} ({aed(r.billed)}) was billed on a separate claim by {who}, "
                    f"{when} from {desc(r.col1)} ({aed(r.parent_billed)}) for the same patient; the "
                    f"bundling table says the component is included in that procedure."
                ),
                what_the_reviewer_must_verify=(
                    "Review the related claims together and check the contract for an agreed split "
                    "between the doctor's and the facility's parts."),
                member_sk=r.member_sk, provider_sk=r.provider_sk,
                parent_provider_sk=r.parent_provider_sk, parent_claim_sk=r.parent_claim_sk,
                parent_line_sk=r.parent_line_sk, parent_code=r.col1,
                component_line_sk=r.line_sk, component_code=r.col2,
                component_amount_aed=_money(r.billed), parent_amount_aed=_money(r.parent_billed),
                days_apart=int(r.days_apart), window_days=window, same_provider=bool(same_prov),
                edit_type=r.edit_type_s or None,
            ),
            exposure=_exp.line_edit_exposure(_money(r.billed), 0.0),
            confidence=0.9 if same_prov else 0.8,
        ))
    return out


def _packages(ctx) -> pd.DataFrame:
    p = _table(ctx, "package_definition")
    if p.empty or "package_code" not in p.columns or "component_code" not in p.columns:
        return pd.DataFrame()
    p = p.copy()
    p["package_code"] = p["package_code"].map(_str_id)
    p["component_code"] = p["component_code"].map(_str_id)
    p["zero"] = _truthy(_col(p, "expected_zero_price"))
    return p[(p["package_code"] != "") & (p["component_code"] != "")]


def _package_components_on_claims(ctx) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """(package lines, package definition, component lines on the same claim as their package)."""
    lines = _lines(ctx)
    pk = _packages(ctx)
    if lines.empty or pk.empty:
        return pd.DataFrame(), pk, pd.DataFrame()
    pkg_lines = lines[lines["activity_code"].isin(set(pk["package_code"]))]
    if pkg_lines.empty:
        return pkg_lines, pk, pd.DataFrame()
    head = pkg_lines[["claim_sk", "activity_code", "line_sk", "billed"]].rename(columns={
        "activity_code": "package_code", "line_sk": "package_line_sk", "billed": "package_billed"})
    comp = head.merge(pk, on="package_code", how="inner").merge(
        lines, left_on=["claim_sk", "component_code"], right_on=["claim_sk", "activity_code"], how="inner")
    return pkg_lines, pk, comp


@_register("pay_02_r03_package_completeness")
def pay_02_r03_package_completeness(ctx, control) -> list[Signal]:
    """A package claim that charges for an activity the package includes at zero
    price, or omits the included activities the package definition requires."""
    pkg_lines, pk, comp = _package_components_on_claims(ctx)
    if pkg_lines.empty:
        return []
    zero_tol = float(ctx.cfg("pay02_zero_price_tolerance_aed"))
    min_missing = float(ctx.cfg("pay02_package_missing_share"))
    desc = _describer(ctx)
    out: list[Signal] = []

    # Limb 1: an included activity charged.
    if not comp.empty:
        charged = comp[comp["zero"] & (comp["billed"] > zero_tol)].drop_duplicates("line_sk")
        for r in charged.itertuples(index=False):
            out.append(_sig(
                ctx, control, subject_type="claim", subject_id=r.claim_sk,
                fact_key=f"pkgline:{r.line_sk}", claim_ids=[r.claim_sk],
                event_time=r.line_date, period=period_bucket(r.line_date),
                evidence=_base_evidence(
                    control, r.claim_sk,
                    plain_language=(
                        f"The package {desc(r.package_code)} ({aed(r.package_billed)}) already includes "
                        f"{desc(r.component_code)}, which should be listed at zero price, but it was "
                        f"charged at {aed(r.billed)} on the same claim."
                    ),
                    what_the_reviewer_must_verify="Compare the claim with the package definition.",
                    member_sk=r.member_sk, provider_sk=r.provider_sk,
                    package_code=r.package_code, package_line_sk=r.package_line_sk,
                    component_code=r.component_code, component_line_sk=r.line_sk,
                    component_amount_aed=_money(r.billed), finding="included activity charged",
                ),
                exposure=_exp.line_edit_exposure(_money(r.billed), 0.0),
            ))

    # Limb 2: required activities absent. The definition lists every activity a
    # package MAY include (alternative visit levels, optional tests …) without
    # saying which are indispensable, so the required core is taken as the
    # listed components that appear on nearly every claim for that package.
    core_share = float(ctx.cfg("pay02_package_core_share"))
    min_claims = int(ctx.cfg("pay02_package_min_claims"))
    n_claims = pkg_lines.groupby("activity_code")["claim_sk"].nunique()
    required: dict[str, list[str]] = {}
    if not comp.empty:
        seen = comp[comp["zero"]].groupby(["package_code", "component_code"])["claim_sk"].nunique()
        for (pkg, code), n in seen.items():
            total = int(n_claims.get(pkg, 0))
            if total >= min_claims and n / total >= core_share:
                required.setdefault(pkg, []).append(code)
    lines = _lines(ctx)
    codes_by_claim = lines.groupby("claim_sk")["activity_code"].apply(set)
    docs = _table(ctx, "document")
    documented: set[str] = set()
    if not docs.empty and {"doc_type", "claim_sk"} <= set(docs.columns):
        mask = _norm(docs["doc_type"]).str.contains("omission", regex=False)
        documented = set(docs.loc[mask, "claim_sk"].astype(str))
    for r in pkg_lines.drop_duplicates(["claim_sk", "activity_code"]).itertuples(index=False):
        need = required.get(r.activity_code)
        if not need:
            continue
        have = codes_by_claim.get(r.claim_sk, set())
        missing = [c for c in need if c not in have]
        if not missing or len(missing) / len(need) < min_missing:
            continue
        if r.claim_sk in documented:
            continue  # declared exclusion: documented medical omission
        names = ", ".join(desc(c) for c in missing[:4]) + (" and others" if len(missing) > 4 else "")
        out.append(_sig(
            ctx, control, subject_type="claim", subject_id=r.claim_sk,
            fact_key=f"pkgmissing:{r.claim_sk}:{r.activity_code}", claim_ids=[r.claim_sk],
            event_time=r.line_date, period=period_bucket(r.line_date),
            evidence=_base_evidence(
                control, r.claim_sk,
                plain_language=(
                    f"The package {desc(r.activity_code)} was billed at {aed(r.billed)} but "
                    f"{len(missing)} of its {len(need)} core activities, which appear on nearly every "
                    f"other claim for this package, are not on the claim ({names})."
                ),
                what_the_reviewer_must_verify=(
                    "Check whether a documented medical reason explains the missing activities."),
                member_sk=r.member_sk, provider_sk=r.provider_sk,
                package_code=r.activity_code, package_line_sk=r.line_sk,
                package_amount_aed=_money(r.billed), required_components=list(need),
                core_share_threshold=core_share,
                missing_components=missing, missing_share=round(len(missing) / len(need), 3),
                missing_share_threshold=min_missing, finding="required activities absent",
            ),
            exposure=_exp.no_exposure(
                "the claim is returned for correction; a corrected package claim may be payable in full"),
        ))
    return out


@_register("pay_02_r04_inclusive_leakage")
def pay_02_r04_inclusive_leakage(ctx, control) -> list[Signal]:
    """A fee-for-service line PAID separately although the case-rate or package
    billed on the same claim already includes it. Carve-outs (components the
    package definition marks as separately payable) are excluded."""
    pkg_lines, pk, comp = _package_components_on_claims(ctx)
    if comp.empty:
        return []
    paid = _paid_by_line(ctx)
    if paid.empty:
        return []
    c = comp[comp["zero"]].merge(paid, on="line_sk", how="inner")  # carve-outs are not zero-priced
    c = c[(c["paid"] > float(ctx.cfg("pay02_zero_price_tolerance_aed"))) & ~c["reversed"].astype(bool)]
    c = c.drop_duplicates("line_sk")
    desc = _describer(ctx)
    out: list[Signal] = []
    for r in c.itertuples(index=False):
        out.append(_sig(
            ctx, control, subject_type="claim", subject_id=r.claim_sk,
            fact_key=f"pkgline:{r.line_sk}", claim_ids=[r.claim_sk],
            event_time=r.line_date, period=period_bucket(r.line_date),
            evidence=_base_evidence(
                control, r.claim_sk,
                plain_language=(
                    f"{desc(r.component_code)} was paid {aed(r.paid)} as a separate item although the "
                    f"package {desc(r.package_code)} paid on the same claim already includes it."
                ),
                what_the_reviewer_must_verify=(
                    "Check the inclusions, exclusions and carve-outs in force at the time and confirm "
                    "the separate payment."),
                member_sk=r.member_sk, provider_sk=r.provider_sk,
                package_code=r.package_code, package_line_sk=r.package_line_sk,
                component_code=r.component_code, component_line_sk=r.line_sk,
                paid_amount_aed=_money(r.paid), billed_amount_aed=_money(r.billed),
            ),
            exposure=_exp.line_edit_exposure(_money(r.paid), 0.0),
        ))
    return out


@_register("pay_02_r05_novel_unbundling")
def pay_02_r05_novel_unbundling(ctx, control) -> list[Signal]:
    """A provider bills a pair of separately charged services together far more
    often than peers of the same type and specialty, and the pair is in no
    bundling edit. Rates are shrunk toward the peer rate (§4.6) before comparing."""
    lines = _lines(ctx)
    if lines.empty:
        return []
    min_claims = int(ctx.cfg("pay02_novel_min_pair_claims"))
    ratio_cut = float(ctx.cfg("pay02_novel_rate_ratio"))
    min_rate = float(ctx.cfg("pay02_novel_min_shrunk_rate"))
    min_peers = int(ctx.cfg("pay02_novel_min_peer_providers"))
    interval_mass = float(ctx.cfg("posterior_interval_mass"))
    max_width = float(ctx.cfg("max_posterior_width"))

    ln = lines[(lines["billed"] > 0) & (lines["activity_code"] != "") & ~_is_drug(lines)]
    ln = ln[["claim_sk", "provider_sk", "activity_code", "billed", "line_date"]]
    per = ln.groupby(["claim_sk", "activity_code"], as_index=False).agg(
        provider_sk=("provider_sk", "first"), billed=("billed", "sum"), line_date=("line_date", "min"))
    if per.empty:
        return []
    pairs = per.merge(per[["claim_sk", "activity_code", "billed"]], on="claim_sk", suffixes=("_a", "_b"))
    pairs = pairs[pairs["activity_code_a"] < pairs["activity_code_b"]]
    if pairs.empty:
        return []
    e = _edits(ctx)
    if not e.empty:
        known = set(zip(e["col1"], e["col2"])) | set(zip(e["col2"], e["col1"]))
        keys = list(zip(pairs["activity_code_a"], pairs["activity_code_b"]))
        pairs = pairs[[k not in known for k in keys]]
    peer_of = _peer_group(ctx)
    per["peer"] = per["provider_sk"].map(lambda p: peer_of.get(p, "all providers"))
    # Opportunities: the provider's claims carrying either code of the pair.
    code_claims = per.groupby(["provider_sk", "activity_code"])["claim_sk"].nunique()
    ev = pairs.groupby(["provider_sk", "activity_code_a", "activity_code_b"]).agg(
        events=("claim_sk", "nunique"), extra_billed=("billed_b", "sum"),
        claim_ids=("claim_sk", lambda s: sorted(set(s))), last_date=("line_date", "max")).reset_index()
    if ev.empty:
        return []

    def opp(p, a, b, events):
        return int(code_claims.get((p, a), 0) + code_claims.get((p, b), 0) - events)

    ev["opportunities"] = [opp(p, a, b, n) for p, a, b, n in
                           zip(ev["provider_sk"], ev["activity_code_a"], ev["activity_code_b"], ev["events"])]
    ev["peer"] = ev["provider_sk"].map(lambda p: peer_of.get(p, "all providers"))
    candidates = ev[ev["events"] >= min_claims]
    if candidates.empty:
        return []
    # Peer denominators: every provider in the peer group that bills either code.
    provs_by_peer = per.groupby("peer")["provider_sk"].apply(lambda s: sorted(set(s))).to_dict()
    ev_index = ev.set_index(["provider_sk", "activity_code_a", "activity_code_b"])
    desc = _describer(ctx)
    out: list[Signal] = []
    for r in candidates.itertuples(index=False):
        peers = [p for p in provs_by_peer.get(r.peer, []) if p != r.provider_sk]
        e_list, o_list = [], []
        for p in peers:
            o = int(code_claims.get((p, r.activity_code_a), 0) + code_claims.get((p, r.activity_code_b), 0))
            if o <= 0:
                continue
            key = (p, r.activity_code_a, r.activity_code_b)
            n = int(ev_index["events"].get(key, 0)) if key in ev_index.index else 0
            e_list.append(n)
            o_list.append(o - n)
        if len(o_list) < min_peers:
            continue  # declared exclusion: peer adjustment needs a peer population
        prior = fit_beta_prior(e_list, o_list)
        peer_rate = float(sum(e_list) / max(sum(o_list), 1))
        sr = shrink_rate(str(r.provider_sk), r.events, max(r.opportunities, r.events), prior,
                         interval_mass=interval_mass, max_posterior_width=max_width)
        if sr.excluded_from_ranking:
            continue
        if sr.shrunk_rate < min_rate or sr.shrunk_rate < ratio_cut * max(peer_rate, 1e-9):
            continue
        observed = r.events / max(r.opportunities, r.events, 1)
        out.append(_sig(
            ctx, control, subject_type="provider", subject_id=r.provider_sk,
            fact_key=f"novelpair:{r.provider_sk}:{r.activity_code_a}:{r.activity_code_b}",
            claim_ids=r.claim_ids[:200], event_time=r.last_date, period=period_bucket(r.last_date),
            confidence=0.7, peer_level_used=f"provider type / specialty: {r.peer}",
            evidence=_base_evidence(
                control, r.provider_sk,
                plain_language=(
                    f"This provider billed {desc(r.activity_code_a)} and {desc(r.activity_code_b)} as "
                    f"separate charges together on {pct(observed)} of the claims where either appears "
                    f"({count_phrase(r.events, 'claim')}); similar providers do so on {pct(peer_rate)}. "
                    f"No bundling rule covers this pair."
                ),
                what_the_reviewer_must_verify=(
                    "Review a sample of these claims and ask the medical policy team whether a new "
                    "bundling rule is needed; this is a policy-review candidate, not a refusal."),
                provider_sk=r.provider_sk, code_a=r.activity_code_a, code_b=r.activity_code_b,
                claims_with_pair=int(r.events), claims_with_either_code=int(r.opportunities),
                observed_rate=round(observed, 4), shrunk_rate=round(sr.shrunk_rate, 4),
                posterior_interval=[round(sr.posterior_low, 4), round(sr.posterior_high, 4)],
                peer_rate=round(peer_rate, 4), peer_providers=len(o_list), peer_group=r.peer,
                rate_ratio_threshold=ratio_cut, second_code_billed_aed=_money(r.extra_billed),
                shrinkage_explanation=sr.explain(),
            ),
            exposure=_exp.no_exposure(
                "a combination no current rule prohibits is a policy-review candidate; nothing is "
                "established as over-paid until a bundling rule is adopted"),
        ))
    return out


# ===========================================================================
# PAY-03 — code validity, demographics, units and time
# ===========================================================================


@_register("pay_03_r01_invalid_code")
def pay_03_r01_invalid_code(ctx, control) -> list[Signal]:
    """An activity code (or diagnosis code) absent from its code set, or outside
    the code's effective dates on the service date."""
    csv = _table(ctx, "code_system_version")
    lines = _lines(ctx)
    if csv.empty or lines.empty or "code" not in csv.columns:
        return []
    csv = csv.copy()
    csv["code"] = csv["code"].map(_str_id)
    csv["system"] = _col(csv, "code_system").map(_str_id)
    listed = set(csv["code"])
    desc = _describer(ctx)
    obs = _table(ctx, "observation")
    observed_lines = set(obs["line_sk"].astype(str)) if not obs.empty and "line_sk" in obs.columns else set()
    auth, _, ids = _authorisations(ctx)
    approved_ids = set()
    if not auth.empty:
        ok = set(auth.loc[_approved(auth["status_s"]), "authorization_sk"])
        approved_ids = {k for k, v in ids.items() if v in ok}

    ln = lines[lines["activity_code"] != ""].copy()
    # A code set covers an activity type when any code of that type is listed in it.
    covered_types = set(ln.loc[ln["activity_code"].isin(listed), "activity_type_s"])
    absent = ln[~ln["activity_code"].isin(listed) & ln["activity_type_s"].isin(covered_types)].copy()
    absent["problem"] = "absent"
    absent["vf"] = pd.NaT
    absent["vt"] = pd.NaT
    present = ln.merge(csv[["code", "valid_from", "valid_to"]].rename(columns={"code": "activity_code"}),
                       on="activity_code", how="inner")
    present["active"] = _in_force(present["line_date"], present["valid_from"], present["valid_to"])
    act = present.groupby("line_sk")["active"].any()
    inactive = present[~present["line_sk"].map(act).astype(bool)].drop_duplicates("line_sk").copy()
    inactive["problem"] = "inactive"
    inactive["vf"] = _date(inactive["valid_from"])
    inactive["vt"] = _date(inactive["valid_to"])
    bad = pd.concat([absent, inactive], ignore_index=True)
    if bad.empty:
        return []
    # declared exclusion: approved unlisted-code pathway with the required Observation
    auth_ok = _col(bad, "authorization_id").map(lambda v: _str_id(v).strip() in approved_ids)
    bad = bad[~(bad["line_sk"].isin(observed_lines) & auth_ok)]
    out: list[Signal] = []
    for r in bad.drop_duplicates("line_sk").itertuples(index=False):
        if r.problem == "absent":
            text = (f"Activity code {r.activity_code} billed on {plain_date(r.line_date)} "
                    f"({aed(r.billed)}) does not exist in the code set used for this type of service.")
        else:
            span = (f"valid from {plain_date(r.vf, missing='the start')} to "
                    f"{plain_date(r.vt, missing='now')}")
            text = (f"{desc(r.activity_code)} was billed for {plain_date(r.line_date)} "
                    f"({aed(r.billed)}), but the code set shows it {span}, so it was not valid on the "
                    f"service date.")
        out.append(_sig(
            ctx, control, subject_type="claim", subject_id=r.claim_sk,
            fact_key=f"code:{r.line_sk}", claim_ids=[r.claim_sk],
            event_time=r.line_date, period=period_bucket(r.line_date),
            evidence=_base_evidence(
                control, r.claim_sk, plain_language=text,
                what_the_reviewer_must_verify=(
                    "Check the code in the code set version that applied on the service date, and "
                    "whether an approved route for unlisted codes applies."),
                member_sk=r.member_sk, provider_sk=r.provider_sk, line_sk=r.line_sk,
                activity_code=r.activity_code, service_date=_day(r.line_date), problem=r.problem,
                code_valid_from=_day(r.vf), code_valid_to=_day(r.vt), line_amount_aed=_money(r.billed),
            ),
            exposure=_exp.no_exposure(
                "the claim is returned for correction; the corrected claim may be payable"),
        ))
    # Diagnosis codes, for any code system the table covers.
    dx = _table(ctx, "diagnosis")
    if not dx.empty and {"claim_sk", "code"} <= set(dx.columns):
        systems = set(csv["system"]) - {""}
        d = dx.copy()
        d["claim_sk"] = d["claim_sk"].astype(str)
        d["code"] = d["code"].map(_str_id)
        d["system"] = _col(d, "code_system").map(_str_id)
        d = d[d["system"].isin(systems) & (d["code"] != "")]
        if not d.empty:
            hdr = _header(ctx)[["claim_sk", "member_sk", "provider_sk", "service_date"]]
            d = d.merge(hdr, on="claim_sk", how="inner")
            listed_pairs = set(zip(csv["code"], csv["system"]))
            d["listed"] = [(c, s) in listed_pairs for c, s in zip(d["code"], d["system"])]
            dm = d.merge(csv[["code", "system", "valid_from", "valid_to"]], on=["code", "system"], how="left")
            dm["active"] = _in_force(dm["service_date"], dm["valid_from"], dm["valid_to"]) & dm["listed"]
            actd = dm.groupby(["claim_sk", "code"])["active"].any()
            badd = d[[not actd.get((c, k), False) for c, k in zip(d["claim_sk"], d["code"])]]
            for r in badd.drop_duplicates(["claim_sk", "code"]).itertuples(index=False):
                problem = "absent" if not r.listed else "inactive"
                text = (f"Diagnosis code {r.code} ({r.system}) on this claim "
                        + ("is not in that code set." if problem == "absent" else
                           f"was not valid on the service date, {plain_date(r.service_date)}."))
                out.append(_sig(
                    ctx, control, subject_type="claim", subject_id=r.claim_sk,
                    fact_key=f"dxcode:{r.claim_sk}:{r.code}", claim_ids=[r.claim_sk],
                    event_time=r.service_date, period=period_bucket(r.service_date),
                    evidence=_base_evidence(
                        control, r.claim_sk, plain_language=text,
                        what_the_reviewer_must_verify="Check the diagnosis code against the code set in force.",
                        member_sk=_str_id(r.member_sk), provider_sk=_str_id(r.provider_sk),
                        diagnosis_code=r.code, code_system=r.system, problem=problem,
                        service_date=_day(r.service_date),
                    ),
                    exposure=_exp.no_exposure("the claim is returned for correction"),
                ))
    return out


def _sex(v: Any) -> str:
    s = _str_id(v).strip().upper()
    if s in {"M", "MALE"}:
        return "M"
    if s in {"F", "FEMALE"}:
        return "F"
    return ""


@_register("pay_03_r02_demographic_impossibility")
def pay_03_r02_demographic_impossibility(ctx, control) -> list[Signal]:
    """A service with an explicit sex or age restriction billed for a patient
    who does not meet it. A pre-approval covering that very code is treated as
    the policy-approved clinical exception."""
    lines = _lines(ctx)
    ref = _codes(ctx)
    mem = _table(ctx, "member")
    if lines.empty or ref.empty or mem.empty or "member_sk" not in mem.columns:
        return []
    mem = mem.copy()
    mem["member_sk"] = mem["member_sk"].map(_str_id)
    mem["m_sex"] = _col(mem, "sex").map(_sex)
    mem["dob"] = _date(_col(mem, "date_of_birth"))
    r = ref[["activity_code"]].copy()
    r["r_sex"] = _col(ref, "sex_restriction").map(_sex)
    r["age_min"] = _num(_col(ref, "age_min"))
    r["age_max"] = _num(_col(ref, "age_max"))
    r = r[(r["r_sex"] != "") | r["age_min"].notna() | r["age_max"].notna()]
    if r.empty:
        return []
    m = lines.merge(r, on="activity_code", how="inner").merge(
        mem[["member_sk", "m_sex", "dob"]].drop_duplicates("member_sk"), on="member_sk", how="inner")
    if m.empty:
        return []
    m["age"] = ((m["line_date"] - m["dob"]).dt.days / 365.25).apply(
        lambda x: math.floor(x) if not pd.isna(x) else np.nan)
    sex_bad = (m["r_sex"] != "") & (m["m_sex"] != "") & (m["r_sex"] != m["m_sex"])
    young = m["age_min"].notna() & m["age"].notna() & (m["age"] < m["age_min"])
    old = m["age_max"].notna() & m["age"].notna() & (m["age"] > m["age_max"])
    m = m[sex_bad | young | old].copy()
    if m.empty:
        return []
    m["sex_bad"] = sex_bad[m.index]
    # declared exclusion: gender / clinical exception approved by policy (an approval for this code)
    auth, al, ids = _authorisations(ctx)
    pairs = _approved_code_pairs(auth, al)
    if not pairs.empty:
        ok_keys = {(k, c) for k, c in zip(pairs["authorization_sk"], pairs["activity_code"])}
        auth_key = _col(m, "authorization_id").map(lambda v: ids.get(_str_id(v).strip()))
        excepted = [(k, c) in ok_keys for k, c in zip(auth_key, m["activity_code"])]
        m = m[[not x for x in excepted]]
    desc = _describer(ctx)
    words = {"M": "male", "F": "female"}
    out: list[Signal] = []
    for x in m.drop_duplicates("line_sk").itertuples(index=False):
        if x.sex_bad:
            text = (f"{desc(x.activity_code)} is restricted to {words[x.r_sex]} patients, but it was "
                    f"billed ({aed(x.billed)}) for a {words.get(x.m_sex, 'patient')} patient.")
        else:
            lo = "" if pd.isna(x.age_min) else f"from age {int(x.age_min)}"
            hi = "" if pd.isna(x.age_max) else f"up to age {int(x.age_max)}"
            rng = " ".join(s for s in (lo, hi) if s)
            text = (f"{desc(x.activity_code)} is allowed only {rng}, but it was billed "
                    f"({aed(x.billed)}) for a patient aged {int(x.age)} on {plain_date(x.line_date)}.")
        out.append(_sig(
            ctx, control, subject_type="claim", subject_id=x.claim_sk,
            fact_key=f"demog:{x.line_sk}", claim_ids=[x.claim_sk],
            event_time=x.line_date, period=period_bucket(x.line_date),
            evidence=_base_evidence(
                control, x.claim_sk, plain_language=text,
                what_the_reviewer_must_verify=(
                    "Confirm the patient's date of birth and sex, and whether an approved clinical or "
                    "gender exception applies."),
                member_sk=x.member_sk, provider_sk=x.provider_sk, line_sk=x.line_sk,
                activity_code=x.activity_code, patient_sex=x.m_sex or None,
                patient_age=None if pd.isna(x.age) else int(x.age),
                sex_restriction=x.r_sex or None,
                age_min=None if pd.isna(x.age_min) else int(x.age_min),
                age_max=None if pd.isna(x.age_max) else int(x.age_max),
                line_amount_aed=_money(x.billed), service_date=_day(x.line_date),
            ),
            exposure=_exp.line_edit_exposure(_money(x.payable), 0.0),
        ))
    return out


@_register("pay_03_r03_unit_maximum")
def pay_03_r03_unit_maximum(ctx, control) -> list[Signal]:
    """Units of one service for one patient at one provider on one day (across all of
    that provider's claims) above the daily maximum. On an inpatient claim the maximum applies per day of
    stay. Lines carrying a laterality or repeat indicator, and drug lines (dose
    rules belong to the pharmacy controls), are excluded."""
    lines = _lines(ctx)
    pol = _table(ctx, "unit_maximum_policy")
    if lines.empty or pol.empty or "activity_code" not in pol.columns:
        return []
    exempt = {str(x).strip().upper() for x in (ctx.cfg("pay03_unit_exempt_indicators") or [])}
    pol = pol.copy()
    pol["activity_code"] = pol["activity_code"].map(_str_id)
    pol["max_units"] = _num(_col(pol, "max_units_per_day"))
    pol = pol[pol["max_units"] > 0].drop_duplicates("activity_code")
    ln = lines[~lines["line_sk"].isin(_inactive_lines(ctx))]  # declared exclusion: repeat / resubmission
    ln = ln.merge(pol[["activity_code", "max_units"]], on="activity_code", how="inner")
    if ln.empty:
        return []
    ln = ln[~_is_drug(ln)]
    ln = ln.assign(exempt=ln["indicator_s"].str.upper().isin(exempt))
    inpatient = _norm(_col(ln, "claim_type")).str.contains("inpat", regex=False)
    los = _num(_col(ln, "length_of_stay_days")).fillna(1.0).clip(lower=1.0)
    ln = ln.assign(
        grp=np.where(inpatient, "C:" + ln["claim_sk"],
                     "D:" + ln["member_sk"] + ":" + ln["provider_sk"] + ":" + ln["line_date"].dt.strftime("%Y-%m-%d")),
        allowed_factor=np.where(inpatient, los, 1.0))
    g = ln.groupby(["grp", "activity_code"]).agg(
        units=("units_n", "sum"), max_units=("max_units", "first"), factor=("allowed_factor", "max"),
        exempt=("exempt", "any"), billed=("billed", "sum"), payable=("payable", "sum"),
        member_sk=("member_sk", "first"), provider_sk=("provider_sk", "first"),
        line_date=("line_date", "min")).reset_index()
    g["allowed"] = g["max_units"] * g["factor"]
    g = g[(g["units"] > g["allowed"]) & ~g["exempt"]]
    if g.empty:
        return []
    hit = ln.merge(g[["grp", "activity_code"]], on=["grp", "activity_code"], how="inner")
    lists = hit.groupby(["grp", "activity_code"]).agg(
        claims=("claim_sk", lambda s: sorted(set(s))), lines=("line_sk", lambda s: sorted(set(s)))).reset_index()
    g = g.merge(lists, on=["grp", "activity_code"], how="left")
    desc = _describer(ctx)
    out: list[Signal] = []
    for r in g.itertuples(index=False):
        excess = float(r.units - r.allowed)
        per_unit = float(r.payable) / float(r.units) if r.units else 0.0
        subject = r.claims[-1]
        stay = r.grp.startswith("C:")
        scope = (f"across this {int(r.factor)}-day stay" if stay
                 else f"on {plain_date(r.line_date)}" + (
                     f" across {count_phrase(len(r.claims), 'claim')}" if len(r.claims) > 1 else ""))
        out.append(_sig(
            ctx, control, subject_type="claim", subject_id=subject,
            fact_key=f"units:{r.grp}:{r.activity_code}", claim_ids=r.claims,
            event_time=r.line_date, period=period_bucket(r.line_date),
            evidence=_base_evidence(
                control, subject,
                plain_language=(
                    f"{r.units:g} units of {desc(r.activity_code)} were billed for this patient {scope}; "
                    f"the maximum is {r.allowed:g}, so {excess:g} units are over the limit."
                ),
                what_the_reviewer_must_verify=(
                    "Check whether both sides of the body, repeat services or inpatient rules explain "
                    "the units."),
                member_sk=r.member_sk, provider_sk=r.provider_sk, activity_code=r.activity_code,
                units_billed=float(r.units), max_units_per_day=float(r.max_units),
                units_allowed=float(r.allowed), excess_units=excess, line_sks=r.lines,
                inpatient_stay=bool(stay), service_date=_day(r.line_date),
                billed_amount_aed=_money(r.billed),
            ),
            exposure=_exp.line_edit_exposure(_money(r.payable), _money(r.payable) - excess * per_unit),
        ))
    return out


@_register("pay_03_r04_time_unit_inconsistency")
def pay_03_r04_time_unit_inconsistency(ctx, control) -> list[Signal]:
    """Time-based units that need more minutes than the recorded service window,
    or two timed services by one clinician that overlap (anaesthesia excepted)."""
    lines = _lines(ctx)
    ref = _codes(ctx)
    if lines.empty or ref.empty or "service_start_time" not in lines.columns:
        return []
    tol = float(ctx.cfg("pay03_time_rounding_tolerance_minutes"))
    r = ref[["activity_code"]].copy()
    r["timed"] = _truthy(_col(ref, "is_time_based"))
    r["unit_minutes"] = _num(_col(ref, "minutes"))
    fam = _norm(_col(ref, "service_family")) + " " + _norm(_col(ref, "code_family")) + " " + \
        _norm(_col(ref, "activity_type")) + " " + \
        _norm(_col(ref, "description"))
    r["anaesthesia"] = fam.str.contains("anesth|anaesth", regex=True)
    r = r[r["timed"] & (r["unit_minutes"] > 0)]
    if r.empty:
        return []
    m = lines.merge(r, on="activity_code", how="inner")
    m["start"] = _date(_col(m, "service_start_time"))
    m["end"] = _date(_col(m, "service_end_time"))
    m = m[m["start"].notna() & m["end"].notna()]
    if m.empty:
        return []
    m["window_min"] = (m["end"] - m["start"]).dt.total_seconds() / 60.0
    m["billed_min"] = m["units_n"] * m["unit_minutes"]
    desc = _describer(ctx)
    out: list[Signal] = []

    over = m[m["billed_min"] > m["window_min"] + tol]
    for x in over.itertuples(index=False):
        supported = max(0.0, math.floor((max(x.window_min, 0.0) + tol) / x.unit_minutes))
        excess = max(0.0, float(x.units_n) - supported)
        per_unit = float(x.payable) / float(x.units_n) if x.units_n else 0.0
        out.append(_sig(
            ctx, control, subject_type="claim", subject_id=x.claim_sk,
            fact_key=f"time:{x.line_sk}", claim_ids=[x.claim_sk],
            event_time=x.line_date, period=period_bucket(x.line_date),
            evidence=_base_evidence(
                control, x.claim_sk,
                plain_language=(
                    f"{x.units_n:g} units of {desc(x.activity_code)} ({x.unit_minutes:g} minutes each, "
                    f"{x.billed_min:g} minutes in all) were billed, but the recorded start and end "
                    f"times allow only {max(x.window_min, 0):.0f} minutes."
                ),
                what_the_reviewer_must_verify=(
                    "Compare the units with the start and end times and the rounding policy."),
                member_sk=x.member_sk, provider_sk=x.provider_sk, line_sk=x.line_sk,
                activity_code=x.activity_code, units_billed=float(x.units_n),
                minutes_per_unit=float(x.unit_minutes), billed_minutes=float(x.billed_min),
                recorded_minutes=round(float(x.window_min), 1), rounding_tolerance_minutes=tol,
                units_supported=supported, excess_units=excess,
                start_time=str(x.start), end_time=str(x.end),
                clinician_id=_str_id(getattr(x, "rendering_clinician_id", None)) or None,
                finding="billed time exceeds recorded time",
            ),
            exposure=_exp.line_edit_exposure(_money(x.payable), _money(x.payable) - excess * per_unit),
        ))

    # Overlap limb: one clinician, two timed services at once.
    if "rendering_clinician_id" in m.columns:
        o = m[_has(m["rendering_clinician_id"]) & ~m["anaesthesia"]].copy()
        o["clin"] = o["rendering_clinician_id"].astype(str)
        o = o.sort_values(["clin", "start", "line_sk"])
        prev = o.groupby("clin").shift(1)
        o["p_line"] = prev["line_sk"]
        o["p_claim"] = prev["claim_sk"]
        o["p_end"] = prev["end"]
        o["p_start"] = prev["start"]
        o["p_code"] = prev["activity_code"]
        o["p_member"] = prev["member_sk"]
        o["overlap_min"] = (o[["end", "p_end"]].min(axis=1) - o["start"]).dt.total_seconds() / 60.0
        hit = o[o["p_line"].notna() & (o["p_claim"] != o["claim_sk"]) & (o["overlap_min"] > tol)]
        flagged = set(over["line_sk"])
        for x in hit.itertuples(index=False):
            if x.line_sk in flagged:
                continue
            share = min(1.0, float(x.overlap_min) / max(float(x.billed_min), 1e-9))
            out.append(_sig(
                ctx, control, subject_type="claim", subject_id=x.claim_sk,
                fact_key=f"time:{x.line_sk}", claim_ids=[x.claim_sk, x.p_claim],
                event_time=x.line_date, period=period_bucket(x.line_date), confidence=0.85,
                evidence=_base_evidence(
                    control, x.claim_sk,
                    plain_language=(
                        f"The same clinician is recorded delivering {desc(x.activity_code)} from "
                        f"{x.start:%H:%M} to {x.end:%H:%M} on {plain_date(x.start)} while also "
                        f"delivering {desc(x.p_code)} to another patient until {x.p_end:%H:%M} — "
                        f"{x.overlap_min:.0f} minutes overlap."
                    ),
                    what_the_reviewer_must_verify=(
                        "Check both records' times; concurrent services by one clinician are only "
                        "possible under the anaesthesia rules."),
                    member_sk=x.member_sk, provider_sk=x.provider_sk, clinician_id=x.clin,
                    line_sk=x.line_sk, other_line_sk=x.p_line, other_claim_sk=x.p_claim,
                    activity_code=x.activity_code, other_activity_code=x.p_code,
                    overlap_minutes=round(float(x.overlap_min), 1), rounding_tolerance_minutes=tol,
                    start_time=str(x.start), end_time=str(x.end),
                    other_start_time=str(x.p_start), other_end_time=str(x.p_end),
                    finding="overlapping timed services",
                ),
                exposure=_exp.line_edit_exposure(_money(x.payable), _money(x.payable) * (1.0 - share)),
            ))
    return out


# ===========================================================================
# PAY-04 — prior authorisation
# ===========================================================================


def _needs_authorisation(ctx, lines: pd.DataFrame) -> pd.DataFrame:
    """Lines whose product and service family require approval on the service date."""
    br = _table(ctx, "benefit_rule_version")
    ref = _codes(ctx)
    if br.empty or ref.empty or "service_family" not in ref.columns:
        return pd.DataFrame()
    br = br.copy()
    br = br[_truthy(_col(br, "authorization_required"))]
    if br.empty:
        return pd.DataFrame()
    br["product"] = _col(br, "product").map(_str_id)
    br["service_family"] = _col(br, "service_family").map(lambda v: _str_id(v).lower())
    fam = ref[["activity_code"]].assign(service_family=_col(ref, "service_family").map(lambda v: _str_id(v).lower()))
    base = lines.drop(columns=[c for c in ("product", "service_family") if c in lines.columns])
    ln = base.merge(fam, on="activity_code", how="inner")
    # Benefit family: on a case-rate (package) claim every line takes the case-rate code's family,
    # as the benefit rules' own exceptions state; elsewhere the line code's family.
    case = ln[ln["activity_type_s"] == "drg"].drop_duplicates("claim_sk").set_index("claim_sk")["service_family"]
    if not case.empty:
        ln["service_family"] = ln["claim_sk"].map(case).fillna(ln["service_family"])
    # Product: the member's cover in force on the service date.
    prod = pd.Series("", index=ln.index)
    cov = _table(ctx, "coverage_period")
    if not cov.empty and {"member_sk", "product"} <= set(cov.columns):
        c = cov[["member_sk", "product"]].assign(valid_from=_col(cov, "valid_from"), valid_to=_col(cov, "valid_to"))
        c["member_sk"] = c["member_sk"].map(_str_id)
        j = ln[["line_sk", "member_sk", "line_date"]].merge(c, on="member_sk", how="inner")
        j = j[_in_force(j["line_date"], j["valid_from"], j["valid_to"])].drop_duplicates("line_sk")
        prod = ln["line_sk"].map(dict(zip(j["line_sk"], j["product"].map(_str_id)))).fillna("")
    ln = ln.assign(product_s=prod.values)
    exact = ln.merge(br[["product", "service_family", "valid_from", "valid_to"]],
                     left_on=["product_s", "service_family"], right_on=["product", "service_family"], how="inner")
    wild = ln.merge(br[br["product"].isin(["*", "", "all", "ALL"])][["service_family", "valid_from", "valid_to"]],
                    on="service_family", how="inner")
    both = pd.concat([exact, wild], ignore_index=True)
    if both.empty:
        return both
    both = both[_in_force(both["line_date"], both["valid_from"], both["valid_to"])]
    return both.drop_duplicates("line_sk")


@_register("pay_04_r01_missing_authorization")
def pay_04_r01_missing_authorization(ctx, control) -> list[Signal]:
    """A service the product's benefit rules say needs approval, with no link to
    any known approval. Emergencies are excluded; so is a service for which a
    pending or deemed request exists (deemed approval / response-time route)."""
    lines = _lines(ctx)
    if lines.empty:
        return []
    need = _needs_authorisation(ctx, lines[lines["billed"] > 0])
    if need.empty:
        return []
    auth, al, ids = _authorisations(ctx)
    aid = _col(need, "authorization_id").map(lambda v: _str_id(v).strip())
    need = need.assign(aid=aid.values)
    unlinked = need[(need["aid"] == "") | ~need["aid"].isin(set(ids))]
    if unlinked.empty:
        return []
    emerg = _emergency_claims(ctx)
    unlinked = unlinked[~unlinked["claim_sk"].isin(emerg)]  # declared exclusion: emergency
    # declared exclusion: deemed approval / response-time exception
    if not auth.empty and not al.empty:
        pend = auth[auth["status_s"].isin(_PENDING_STATUSES)][["authorization_sk", "member_sk"]]
        pend = pend.merge(al[["authorization_sk", "activity_code"]], on="authorization_sk")
        pk = set(zip(pend["member_sk"], pend["activity_code"]))
        unlinked = unlinked[[(mm, c) not in pk for mm, c in zip(unlinked["member_sk"], unlinked["activity_code"])]]
    desc = _describer(ctx)
    out: list[Signal] = []
    for r in unlinked.itertuples(index=False):
        what = "no approval reference" if r.aid == "" else f"an approval reference ({r.aid}) that matches no approval on record"
        out.append(_sig(
            ctx, control, subject_type="claim", subject_id=r.claim_sk,
            fact_key=f"auth:{r.line_sk}", claim_ids=[r.claim_sk],
            event_time=r.line_date, period=period_bucket(r.line_date),
            evidence=_base_evidence(
                control, r.claim_sk,
                plain_language=(
                    f"{desc(r.activity_code)} ({aed(r.billed)}, {plain_date(r.line_date)}) needs "
                    f"pre-approval under the patient's product ({r.product_s or 'unknown'}, "
                    f"{r.service_family}), but the line carries {what}."
                ),
                what_the_reviewer_must_verify=(
                    "Search for an approval that should have been linked, and check whether the "
                    "service was an emergency or approval was deemed given."),
                member_sk=r.member_sk, provider_sk=r.provider_sk, line_sk=r.line_sk,
                activity_code=r.activity_code, service_family=r.service_family,
                product=r.product_s or None, authorization_id=r.aid or None,
                line_amount_aed=_money(r.billed), service_date=_day(r.line_date),
            ),
            exposure=_exp.line_edit_exposure(_money(r.payable), 0.0),
        ))
    return out


@_register("pay_04_r02_invalid_timing_status")
def pay_04_r02_invalid_timing_status(ctx, control) -> list[Signal]:
    """A line linked to an approval that was denied or cancelled, or whose service
    date is outside the approval's validity. Another approved authorisation for
    the same patient and code covering the date counts as an extension; an
    emergency service shortly before the approval started counts as the urgent
    retrospective route."""
    ln, auth, al = _linked_lines(ctx)
    if ln.empty or auth.empty:
        return []
    retro = int(ctx.cfg("pay04_retrospective_route_days"))
    m = ln[ln["auth_key"].notna()].merge(
        auth[["authorization_sk", "status_s", "vf", "vt", "member_sk"]].rename(
            columns={"authorization_sk": "auth_key", "member_sk": "auth_member_sk"}),
        on="auth_key", how="inner")
    if m.empty:
        return []
    denied = m["status_s"].isin(_DENIED_STATUSES - {"expired"})
    before = m["vf"].notna() & (m["line_date"] < m["vf"])
    after = m["vt"].notna() & (m["line_date"] > m["vt"])
    m = m.assign(denied=denied, before=before, after=after)
    m = m[denied | before | after]
    if m.empty:
        return []
    # declared exclusion: approved extension
    pairs = _approved_code_pairs(auth, al)
    if not pairs.empty:
        j = m[["line_sk", "member_sk", "activity_code", "line_date", "auth_key"]].merge(
            pairs[["authorization_sk", "member_sk", "activity_code", "vf", "vt"]],
            on=["member_sk", "activity_code"], how="inner")
        j = j[(j["authorization_sk"] != j["auth_key"]) & _in_force(j["line_date"], j["vf"], j["vt"])]
        m = m[~m["line_sk"].isin(set(j["line_sk"]))]
    # declared exclusion: urgent retrospective route
    emerg = _emergency_claims(ctx)
    gap = (m["vf"] - m["line_date"]).dt.days
    m = m[~(m["before"] & ~m["denied"] & m["claim_sk"].isin(emerg) & (gap <= retro))]
    desc = _describer(ctx)
    out: list[Signal] = []
    for r in m.itertuples(index=False):
        if r.denied:
            why = f"the approval it links to ({r.authorization_id}) is recorded as {r.status_s}"
            finding = "approval denied or cancelled"
        else:
            why = (f"the approval ({r.authorization_id}) was valid only from "
                   f"{plain_date(r.vf, missing='an unrecorded date')} to {plain_date(r.vt, missing='an unrecorded date')}")
            finding = "service before approval started" if r.before else "service after approval expired"
        out.append(_sig(
            ctx, control, subject_type="claim", subject_id=r.claim_sk,
            fact_key=f"authvalid:{r.line_sk}", claim_ids=[r.claim_sk],
            event_time=r.line_date, period=period_bucket(r.line_date),
            evidence=_base_evidence(
                control, r.claim_sk,
                plain_language=(
                    f"{desc(r.activity_code)} was provided on {plain_date(r.line_date)} "
                    f"({aed(r.billed)}), but {why}."
                ),
                what_the_reviewer_must_verify=(
                    "Check the approval dates and status, and whether an approved extension or an "
                    "urgent after-the-event approval exists."),
                member_sk=r.member_sk, provider_sk=r.provider_sk, line_sk=r.line_sk,
                activity_code=r.activity_code, authorization_id=str(r.authorization_id),
                authorization_status=r.status_s, authorization_valid_from=_day(r.vf),
                authorization_valid_to=_day(r.vt), service_date=_day(r.line_date), finding=finding,
                line_amount_aed=_money(r.billed),
            ),
            exposure=_exp.line_edit_exposure(_money(r.payable), 0.0),
        ))
    return out


@_register("pay_04_r03_scope_mismatch")
def pay_04_r03_scope_mismatch(ctx, control) -> list[Signal]:
    """A line linked to a valid approval whose code, provider or facility is not
    what was approved. An equivalent code (code-equivalence map) is accepted;
    an approval whose conditions record a transfer covers another provider."""
    ln, auth, al = _linked_lines(ctx)
    if ln.empty or auth.empty:
        return []
    a = auth[_approved(auth["status_s"])].copy()
    a["conditions_all"] = ""
    if not al.empty and "conditions" in al.columns:
        cond = al.groupby("authorization_sk")["conditions"].apply(
            lambda s: " ".join(str(v) for v in s if not is_missing(v)).lower())
        a["conditions_all"] = a["authorization_sk"].map(cond).fillna("")
    a["facility_s"] = _col(a, "facility_id").map(_str_id)
    m = ln[ln["auth_key"].notna()].merge(
        a[["authorization_sk", "provider_sk", "facility_s", "conditions_all", "member_sk"]].rename(columns={
            "authorization_sk": "auth_key", "provider_sk": "auth_provider_sk", "member_sk": "auth_member_sk"}),
        on="auth_key", how="inner")
    if m.empty:
        return []
    approved_codes = al[~al["line_denied"]].groupby("authorization_sk")["activity_code"].apply(set) \
        if not al.empty else pd.Series(dtype=object)
    eq = _table(ctx, "code_equivalence_map")
    group_of: dict[str, set[str]] = {}
    if not eq.empty and {"code", "equivalence_group"} <= set(eq.columns):
        for c, g in zip(eq["code"].map(_str_id), eq["equivalence_group"].map(_str_id)):
            group_of.setdefault(c, set()).add(g)
    enc = _table(ctx, "encounter")
    fac = {}
    if not enc.empty and {"claim_sk", "facility_id"} <= set(enc.columns):
        fac = dict(zip(enc["claim_sk"].astype(str), enc["facility_id"].map(_str_id)))
    desc = _describer(ctx)
    out: list[Signal] = []
    for r in m.itertuples(index=False):
        problems = []
        codes = approved_codes.get(r.auth_key, set()) if len(approved_codes) else set()
        if codes and r.activity_code not in codes:
            mine = group_of.get(r.activity_code, set())
            if not any(mine & group_of.get(c, set()) for c in codes):
                problems.append(("code", f"the approval covers {', '.join(desc(c) for c in sorted(codes)[:3])}, "
                                         f"not {desc(r.activity_code)}"))
        transfer = "transfer" in r.conditions_all
        if r.auth_provider_sk and r.provider_sk and r.auth_provider_sk != r.provider_sk and not transfer:
            problems.append(("provider", "the approval was issued to a different provider"))
        cf = fac.get(r.claim_sk, "")
        if r.facility_s and cf and r.facility_s != cf and not transfer:
            problems.append(("facility", f"the approval names facility {r.facility_s} but the service was at {cf}"))
        if not problems:
            continue
        out.append(_sig(
            ctx, control, subject_type="claim", subject_id=r.claim_sk,
            fact_key=f"authscope:{r.line_sk}", claim_ids=[r.claim_sk],
            event_time=r.line_date, period=period_bucket(r.line_date),
            evidence=_base_evidence(
                control, r.claim_sk,
                plain_language=(
                    f"{desc(r.activity_code)} ({aed(r.billed)}, {plain_date(r.line_date)}) is linked to "
                    f"approval {r.authorization_id}, but " + "; and ".join(p[1] for p in problems) + "."
                ),
                what_the_reviewer_must_verify=(
                    "Compare the claim with the approval line by line; an equivalent code or an "
                    "approved transfer can explain a difference."),
                member_sk=r.member_sk, provider_sk=r.provider_sk,
                authorised_provider_sk=r.auth_provider_sk or None, line_sk=r.line_sk,
                activity_code=r.activity_code, authorization_id=str(r.authorization_id),
                approved_codes=sorted(codes), mismatched_dimensions=[p[0] for p in problems],
                authorised_facility=r.facility_s or None, claim_facility=cf or None,
                line_amount_aed=_money(r.billed),
            ),
            exposure=_exp.line_edit_exposure(_money(r.payable), 0.0),
        ))
    return out


@_register("pay_04_r04_quantity_exhaustion")
def pay_04_r04_quantity_exhaustion(ctx, control) -> list[Signal]:
    """Cumulative units or value claimed against one approved line beyond what
    was approved. Reversed or cancelled lines do not count toward the total."""
    ln, auth, al = _linked_lines(ctx)
    if ln.empty or al.empty:
        return []
    tol = float(ctx.cfg("pay04_value_tolerance_aed"))
    ok = auth[_approved(auth["status_s"])]["authorization_sk"]
    a = al[~al["line_denied"] & al["authorization_sk"].isin(set(ok))].groupby(
        ["authorization_sk", "activity_code"]).agg(
        approved_units=("approved_units_n", "sum"), approved_value=("approved_value_n", "sum"),
        has_units=("approved_units_n", lambda s: s.notna().any()),
        has_value=("approved_value_n", lambda s: s.notna().any())).reset_index()
    m = ln[ln["auth_key"].notna()].merge(
        a.rename(columns={"authorization_sk": "auth_key"}), on=["auth_key", "activity_code"], how="inner")
    if m.empty:
        return []
    # declared exclusion: cancelled / reversed claims. A claim replaced by a resubmission is
    # superseded rather than consumed twice, and a line refused outright consumed nothing.
    m = m[~m["line_sk"].isin(_inactive_lines(ctx))]
    m = m.sort_values(["auth_key", "activity_code", "line_date", "claim_sk", "line_sk"])
    g = m.groupby(["auth_key", "activity_code"])
    m["cum_units"] = g["units_n"].cumsum()
    m["cum_value"] = g["billed"].cumsum()
    m["prev_units"] = m["cum_units"] - m["units_n"]
    m["prev_value"] = m["cum_value"] - m["billed"]
    over_u = m["has_units"] & (m["cum_units"] > m["approved_units"])
    over_v = m["has_value"] & (m["cum_value"] > m["approved_value"] + tol)
    m = m.assign(over_u=over_u, over_v=over_v)
    hits = m[m["over_u"] | m["over_v"]]
    desc = _describer(ctx)
    out: list[Signal] = []
    for r in hits.itertuples(index=False):
        per_unit = float(r.payable) / float(r.units_n) if r.units_n else 0.0
        ex_units = max(0.0, min(float(r.units_n), float(r.cum_units - r.approved_units))) if r.over_u else 0.0
        ex_value = max(0.0, min(float(r.billed), float(r.cum_value - r.approved_value))) if r.over_v else 0.0
        excess_amount = max(ex_units * per_unit, min(ex_value, float(r.payable)))
        if r.over_u:
            text = (f"{desc(r.activity_code)}: this claim brings the units billed against approval "
                    f"{r.authorization_id} to {r.cum_units:g}, but only {r.approved_units:g} were approved "
                    f"({r.prev_units:g} had already been billed).")
        else:
            text = (f"{desc(r.activity_code)}: this claim brings the amount billed against approval "
                    f"{r.authorization_id} to {aed(r.cum_value)}, but only {aed(r.approved_value)} was "
                    f"approved ({aed(r.prev_value)} had already been billed).")
        out.append(_sig(
            ctx, control, subject_type="claim", subject_id=r.claim_sk,
            fact_key=f"authqty:{r.line_sk}", claim_ids=[r.claim_sk],
            event_time=r.line_date, period=period_bucket(r.line_date),
            evidence=_base_evidence(
                control, r.claim_sk, plain_language=text,
                what_the_reviewer_must_verify=(
                    "Check the running total against the approval after removing cancelled or "
                    "reversed claims and partial fills."),
                member_sk=r.member_sk, provider_sk=r.provider_sk, line_sk=r.line_sk,
                authorization_id=str(r.authorization_id), activity_code=r.activity_code,
                approved_units=None if not r.has_units else float(r.approved_units),
                approved_value_aed=None if not r.has_value else _money(r.approved_value),
                cumulative_units=float(r.cum_units), cumulative_value_aed=_money(r.cum_value),
                units_before_this_claim=float(r.prev_units), excess_units=ex_units,
                excess_value_aed=_money(ex_value), service_date=_day(r.line_date),
            ),
            exposure=_exp.line_edit_exposure(_money(r.payable), _money(r.payable) - excess_amount),
        ))
    return out


@_register("pay_04_r05_authorization_reuse")
def pay_04_r05_authorization_reuse(ctx, control) -> list[Signal]:
    """One approval consumed by claims for a patient other than the one it was
    issued to. Family members of the approved patient are excluded only when
    policy permits shared family approvals."""
    ln, auth, _ = _linked_lines(ctx)
    if ln.empty or auth.empty:
        return []
    family_ok = bool(ctx.cfg("pay04_family_shared_authorization_permitted"))
    m = ln[ln["auth_key"].notna()].merge(
        auth[["authorization_sk", "member_sk", "provider_sk"]].rename(columns={
            "authorization_sk": "auth_key", "member_sk": "auth_member_sk", "provider_sk": "auth_provider_sk"}),
        on="auth_key", how="inner")
    m = m[(m["auth_member_sk"] != "") & (m["member_sk"] != "") & (m["member_sk"] != m["auth_member_sk"])]
    if m.empty:
        return []
    if family_ok:
        mem = _table(ctx, "member")
        if not mem.empty and "sponsor_id" in mem.columns:
            sp = dict(zip(mem["member_sk"].map(_str_id), mem["sponsor_id"].map(_str_id)))
            same = [sp.get(a, "") != "" and sp.get(a) == sp.get(b) for a, b in zip(m["member_sk"], m["auth_member_sk"])]
            m = m[[not s for s in same]]
    users = ln[ln["auth_key"].notna()].groupby("auth_key")["member_sk"].apply(lambda s: sorted(set(s)))
    per_claim = m.groupby(["auth_key", "claim_sk"]).agg(
        authorization_id=("authorization_id", "first"), member_sk=("member_sk", "first"),
        auth_member_sk=("auth_member_sk", "first"), provider_sk=("provider_sk", "first"),
        auth_provider_sk=("auth_provider_sk", "first"), line_date=("line_date", "min"),
        billed=("billed", "sum"), payable=("payable", "sum")).reset_index()
    out: list[Signal] = []
    for r in per_claim.itertuples(index=False):
        n_members = len(users.get(r.auth_key, []))
        out.append(_sig(
            ctx, control, subject_type="claim", subject_id=r.claim_sk,
            fact_key=f"authreuse:{r.auth_key}:{r.claim_sk}", claim_ids=[r.claim_sk],
            event_time=r.line_date, period=period_bucket(r.line_date),
            evidence=_base_evidence(
                control, r.claim_sk,
                plain_language=(
                    f"Approval {r.authorization_id} was issued for a different patient, yet this claim "
                    f"({aed(r.billed)}, {plain_date(r.line_date)}) uses it; in all "
                    f"{count_phrase(n_members, 'patient')} have claims linked to this one approval."
                ),
                what_the_reviewer_must_verify=(
                    "List every claim linked to the approval and check whether policy allows a shared "
                    "or family approval."),
                member_sk=r.member_sk, authorised_member_sk=r.auth_member_sk,
                provider_sk=r.provider_sk, authorised_provider_sk=r.auth_provider_sk or None,
                authorization_id=str(r.authorization_id), patients_using_approval=n_members,
                claim_amount_aed=_money(r.billed), family_sharing_permitted=family_ok,
            ),
            exposure=_exp.line_edit_exposure(_money(r.payable), 0.0),
        ))
    return out


# ===========================================================================
# PAY-05 — modifiers (indicators) and edit bypass
# ===========================================================================


def _indicator_rules(ctx) -> pd.DataFrame:
    """Prohibited (indicator, scope) pairs from the payment-policy table.

    A row whose ``policy_key`` names an indicator prohibition carries the
    indicator in ``description`` and the scope in ``service_family`` — a
    service family, an activity code, or ``*`` for every service.
    """
    pp = _table(ctx, "payment_policy")
    if pp.empty or "policy_key" not in pp.columns:
        return pd.DataFrame(columns=["indicator", "scope"])
    key = _norm(pp["policy_key"])
    rows = pp[key.str.contains("indicator|modifier", regex=True) &
              key.str.contains("prohibit|not_allowed|disallow|invalid", regex=True)]
    if rows.empty:
        return pd.DataFrame(columns=["indicator", "scope"])
    return pd.DataFrame({
        "indicator": _col(rows, "description").map(lambda v: _str_id(v).strip().upper()).values,
        "scope": _col(rows, "service_family").map(lambda v: _str_id(v).strip().lower()).values,
        "value": _num(_col(rows, "value")).values,
    })


@_register("pay_05_r01_prohibited_indicator_pair")
def pay_05_r01_prohibited_indicator_pair(ctx, control) -> list[Signal]:
    """A code billed with an indicator the effective policy prohibits for it:
    either a payment-policy prohibition for the code or its service family, or
    an indicator used to separate a bundled pair whose edit does not allow a
    modifier."""
    lines = _lines(ctx)
    if lines.empty:
        return []
    ln = lines[lines["indicator_s"] != ""]
    if ln.empty:
        return []
    desc = _describer(ctx)
    ref = _codes(ctx)
    fam = dict(zip(ref["activity_code"], _col(ref, "service_family").map(lambda v: _str_id(v).lower()))) \
        if not ref.empty else {}
    hits: dict[str, dict[str, Any]] = {}
    rules = _indicator_rules(ctx)
    if not rules.empty:
        ind = ln["indicator_s"].str.upper()
        famcol = ln["activity_code"].map(fam).fillna("")
        for rule in rules.itertuples(index=False):
            if rule.indicator == "":
                continue
            scope = rule.scope
            mask = (ind == rule.indicator) & (
                (scope in ("", "*", "all")) | (famcol == scope) | (ln["activity_code"].str.lower() == scope))
            for r in ln[mask].itertuples(index=False):
                hits.setdefault(r.line_sk, {"row": r, "why": (
                    f"local policy does not allow indicator {r.indicator_s} with "
                    f"{'any service' if scope in ('', '*', 'all') else ('this service family (' + scope + ')')}"),
                    "basis": "payment policy table"})
    pairs = _same_claim_pairs(ctx)
    if not pairs.empty:
        bad = pairs[(pairs["indicator_s"] != "") & ~pairs["mod_ok"]].drop_duplicates("line_sk")
        for r in bad.itertuples(index=False):
            hits.setdefault(r.line_sk, {"row": r, "why": (
                f"it is used to bill {desc(r.col2)} separately from {desc(r.col1)} on the same claim, "
                f"and the bundling rule for that pair does not allow a modifier"),
                "basis": "bundling edit table"})
    out: list[Signal] = []
    for line_sk, h in hits.items():
        r = h["row"]
        out.append(_sig(
            ctx, control, subject_type="claim", subject_id=r.claim_sk,
            fact_key=f"indicator:{line_sk}", claim_ids=[r.claim_sk],
            event_time=r.line_date, period=period_bucket(r.line_date),
            evidence=_base_evidence(
                control, r.claim_sk,
                plain_language=(
                    f"{desc(r.activity_code)} ({aed(r.billed)}) was billed with indicator "
                    f"{r.indicator_s}, but {h['why']}."
                ),
                what_the_reviewer_must_verify=(
                    "Check the combination in the local policy table in force at the time before "
                    "returning the claim."),
                member_sk=r.member_sk, provider_sk=r.provider_sk, line_sk=line_sk,
                activity_code=r.activity_code, indicator=r.indicator_s, policy_basis=h["basis"],
                line_amount_aed=_money(r.billed), service_date=_day(r.line_date),
            ),
            exposure=_exp.no_exposure(
                "the claim is returned for correction; the corrected line may still be payable"),
        ))
    return out


@_register("pay_05_r02_indicator_support_absent")
def pay_05_r02_indicator_support_absent(ctx, control) -> list[Signal]:
    """A line carrying an indicator that needs a supporting report, with no
    Observation attached to the line and no document on the claim. Claims
    submitted within the document-arrival allowance are not yet judged."""
    lines = _lines(ctx)
    if lines.empty:
        return []
    need = {str(x).strip().upper() for x in (ctx.cfg("pay05_support_required_indicators") or [])}
    latency = int(ctx.cfg("pay05_document_arrival_days"))
    ln = lines[lines["indicator_s"].str.upper().isin(need)]
    if ln.empty:
        return []
    obs = _table(ctx, "observation")
    with_obs = set()
    if not obs.empty and "line_sk" in obs.columns:
        o = obs
        if "attachment_ref" in obs.columns or "value" in obs.columns:
            o = obs[_has(_col(obs, "attachment_ref")) | _has(_col(obs, "value"))]
        with_obs = set(o["line_sk"].astype(str))
    docs = _table(ctx, "document")
    with_doc = set(docs["claim_sk"].astype(str)) if not docs.empty and "claim_sk" in docs.columns else set()
    ln = ln[~ln["line_sk"].isin(with_obs) & ~ln["claim_sk"].isin(with_doc)]
    if ln.empty:
        return []
    # declared exclusion: document arrival latency, measured from submission to the file's last claim
    hdr = _claim_frame(ctx)
    sub = _date(_col(hdr, "submission_date")).fillna(_date(_col(hdr, "claim_date"))).fillna(
        _date(_col(hdr, "service_date")))
    as_of = sub.max()
    subm = dict(zip(hdr["claim_sk"].astype(str), sub))
    ln = ln.assign(submitted=ln["claim_sk"].map(subm))
    if not pd.isna(as_of):
        ln = ln[(as_of - _date(ln["submitted"])).dt.days > latency]
    desc = _describer(ctx)
    out: list[Signal] = []
    for r in ln.drop_duplicates("line_sk").itertuples(index=False):
        out.append(_sig(
            ctx, control, subject_type="claim", subject_id=r.claim_sk,
            fact_key=f"indsupport:{r.line_sk}", claim_ids=[r.claim_sk],
            event_time=r.line_date, period=period_bucket(r.line_date),
            evidence=_base_evidence(
                control, r.claim_sk,
                plain_language=(
                    f"{desc(r.activity_code)} ({aed(r.billed)}) was billed with indicator {r.indicator_s}, "
                    f"which needs a supporting report or note, but none is attached to the line or the "
                    f"claim more than {latency} days after it was submitted."
                ),
                what_the_reviewer_must_verify="Ask the provider for the required report or note.",
                member_sk=r.member_sk, provider_sk=r.provider_sk, line_sk=r.line_sk,
                activity_code=r.activity_code, indicator=r.indicator_s,
                submitted_on=_day(r.submitted), document_arrival_allowance_days=latency,
                line_amount_aed=_money(r.billed),
            ),
            exposure=_exp.no_exposure("the line is pended until the supporting record arrives"),
        ))
    return out


@_register("pay_05_r03_indicator_rate_outlier")
def pay_05_r03_indicator_rate_outlier(ctx, control) -> list[Signal]:
    """A provider's use rate of one indicator, shrunk toward its peer group's
    rate (same provider type and specialty), above the peer percentile and a
    multiple of the peer mean. Providers with too few lines are not ranked."""
    lines = _lines(ctx)
    if lines.empty:
        return []
    min_opp = int(ctx.cfg("min_entity_opportunities"))
    min_lines = max(min_opp, int(ctx.cfg("pay05_rate_min_lines")))
    pctl = float(ctx.cfg("pay05_rate_peer_percentile"))
    ratio_cut = float(ctx.cfg("pay05_rate_ratio_to_peer"))
    min_excess = float(ctx.cfg("pay05_rate_min_excess"))
    interval_mass = float(ctx.cfg("posterior_interval_mass"))
    max_width = float(ctx.cfg("max_posterior_width"))
    ln = lines[~_is_drug(lines) & (lines["provider_sk"] != "")]
    if ln.empty:
        return []
    peer_of = _peer_group(ctx)
    ln = ln.assign(peer=ln["provider_sk"].map(lambda p: peer_of.get(p, "all providers")),
                   ind=ln["indicator_s"].str.upper())
    opp = ln.groupby("provider_sk").agg(opportunities=("line_sk", "nunique"), peer=("peer", "first"))
    used = ln[ln["ind"] != ""].groupby(["provider_sk", "ind"]).agg(
        events=("line_sk", "nunique"), claims=("claim_sk", lambda s: sorted(set(s))),
        last=("line_date", "max")).reset_index()
    if used.empty:
        return []
    out: list[Signal] = []
    for ind, sub in used.groupby("ind"):
        frame = opp.join(sub.set_index("provider_sk")[["events", "claims", "last"]], how="left")
        frame["events"] = frame["events"].fillna(0.0)
        for peer, grp in frame.groupby("peer"):
            grp = grp[grp["opportunities"] >= min_lines]
            if len(grp) < int(ctx.cfg("pay02_novel_min_peer_providers")):
                continue
            prior = fit_beta_prior(grp["events"], grp["opportunities"])
            srs = {p: shrink_rate(str(p), r.events, r.opportunities, prior, interval_mass=interval_mass,
                                  max_posterior_width=max_width) for p, r in grp.iterrows()}
            rates = [s.shrunk_rate for s in srs.values()]
            cut = float(np.quantile(rates, pctl))
            for p, sr in srs.items():
                if sr.excluded_from_ranking or sr.shrunk_rate <= cut:
                    continue
                if sr.shrunk_rate < ratio_cut * max(prior.peer_mean, 1e-9):
                    continue
                if sr.shrunk_rate - prior.peer_mean < min_excess:
                    continue  # far above peers means a material gap, not a ratio of two small rates
                row = grp.loc[p]
                claims = row["claims"] if isinstance(row["claims"], list) else []
                out.append(_sig(
                    ctx, control, subject_type="provider", subject_id=p,
                    fact_key=f"indrate:{p}:{ind}", claim_ids=claims[:200],
                    event_time=row["last"], period=period_bucket(row["last"]),
                    confidence=0.7, peer_level_used=f"provider type / specialty: {peer}",
                    evidence=_base_evidence(
                        control, p,
                        plain_language=(
                            f"{pct(sr.observed_rate)} of this provider's service lines carry modifier "
                            f"{ind} ({int(row['events'])} of {int(row['opportunities'])}); for similar "
                            f"providers it is {pct(prior.peer_mean)} — {times_phrase(sr.shrunk_rate, prior.peer_mean, 'the peer rate')} "
                            f"after allowing for small numbers."
                        ),
                        what_the_reviewer_must_verify=(
                            "Review a sample of the claims using the modifier; facility type and payment "
                            "model can explain some difference."),
                        provider_sk=p, indicator=ind, lines_with_indicator=int(row["events"]),
                        lines_total=int(row["opportunities"]),
                        observed_rate=round(sr.observed_rate or 0.0, 4), shrunk_rate=round(sr.shrunk_rate, 4),
                        posterior_interval=[round(sr.posterior_low, 4), round(sr.posterior_high, 4)],
                        peer_mean=round(prior.peer_mean, 4), peer_percentile=pctl,
                        peer_percentile_cut=round(cut, 4), peer_group=peer, peer_providers=len(grp),
                        shrinkage_explanation=sr.explain(),
                    ),
                    exposure=_exp.no_exposure(
                        "a use rate above peers is a reason to audit, not an established over-payment"),
                ))
    return out


@_register("pay_05_r04_post_edit_migration")
def pay_05_r04_post_edit_migration(ctx, control) -> list[Signal]:
    """After a bundling edit takes effect, a provider starts escaping it — by
    adding a modifier to the component, or by switching to an unedited code of
    the same service family — and the escaped lines are paid while plain
    components are denied. Compared before and after the edit's start date,
    ignoring the launch/education period."""
    lines = _lines(ctx)
    e = _edits(ctx)
    if lines.empty or e.empty:
        return []
    window = int(ctx.cfg("pay05_migration_window_days"))
    edu = int(ctx.cfg("pay05_education_period_days"))
    min_post = int(ctx.cfg("pay05_migration_min_post_claims"))
    min_rise = float(ctx.cfg("pay05_migration_min_rate_increase"))
    first = lines["line_date"].min()
    last = lines["line_date"].max()
    e = e.assign(launch=_date(e["e_vf"]))
    e = e[e["launch"].notna() & (e["launch"] > first) & (e["launch"] <= last)]
    if e.empty:
        return []
    ref = _codes(ctx)
    fam = dict(zip(ref["activity_code"], _col(ref, "service_family").map(lambda v: _str_id(v).lower()))) \
        if not ref.empty else {}
    paid = _paid_by_line(ctx)
    paid_map = dict(zip(paid["line_sk"], paid["paid"])) if not paid.empty else {}
    denied_map = dict(zip(paid["line_sk"], paid["denied"])) if not paid.empty else {}
    edited_with: dict[str, set[str]] = {}
    for a, b in zip(e["col1"], e["col2"]):
        edited_with.setdefault(a, set()).add(b)
    desc = _describer(ctx)
    out: list[Signal] = []
    by_claim = lines.groupby("claim_sk")
    for ed in e.drop_duplicates(["col1", "col2"]).itertuples(index=False):
        parents = lines[(lines["activity_code"] == ed.col1)]
        if parents.empty:
            continue
        pc = parents.drop_duplicates("claim_sk")[["claim_sk", "provider_sk", "line_date"]]
        pre = pc[(pc["line_date"] < ed.launch) & (pc["line_date"] >= ed.launch - pd.Timedelta(days=window))]
        post = pc[(pc["line_date"] >= ed.launch + pd.Timedelta(days=edu)) &
                  (pc["line_date"] < ed.launch + pd.Timedelta(days=edu + window))]
        if post.empty:
            continue
        family = fam.get(ed.col2, "")
        siblings = {c for c, f in fam.items() if f and f == family and c not in (ed.col1, ed.col2)
                    and c not in edited_with.get(ed.col1, set())}
        claim_lines = lines[lines["claim_sk"].isin(set(pc["claim_sk"]))]
        comp = claim_lines[claim_lines["activity_code"] == ed.col2]
        mod_comp = comp[comp["indicator_s"] != ""]
        sib = claim_lines[claim_lines["activity_code"].isin(siblings)]
        escaped_lines = pd.concat([mod_comp.assign(route="modifier"), sib.assign(route="code")])
        plain_comp = comp[comp["indicator_s"] == ""]
        for prov, post_p in post.groupby("provider_sk"):
            pre_p = pre[pre["provider_sk"] == prov]
            post_claims = set(post_p["claim_sk"])
            pre_claims = set(pre_p["claim_sk"])
            if len(post_claims) < min_post:
                continue
            esc_post = escaped_lines[escaped_lines["claim_sk"].isin(post_claims)]
            esc_post = esc_post[esc_post["line_sk"].map(lambda s: paid_map.get(s, 0.0)) > 0]
            esc_pre = escaped_lines[escaped_lines["claim_sk"].isin(pre_claims)]
            rate_post = esc_post["claim_sk"].nunique() / len(post_claims)
            rate_pre = esc_pre["claim_sk"].nunique() / len(pre_claims) if pre_claims else 0.0
            if rate_post - rate_pre < min_rise:
                continue
            denied_plain = plain_comp[plain_comp["claim_sk"].isin(
                set(pc.loc[(pc["provider_sk"] == prov) & (pc["line_date"] >= ed.launch), "claim_sk"]))]
            n_denied = int(sum(bool(denied_map.get(s, False)) for s in denied_plain["line_sk"]))
            if n_denied < 1:
                continue  # the migration must follow denials under the new edit
            routes = esc_post["route"].value_counts().to_dict()
            main = "adding a modifier to" if routes.get("modifier", 0) >= routes.get("code", 0) else \
                "switching to an unedited code in place of"
            paid_amt = float(sum(paid_map.get(s, 0.0) for s in esc_post["line_sk"]))
            claims = sorted(post_claims & set(esc_post["claim_sk"]))
            last_d = post_p["line_date"].max()
            out.append(_sig(
                ctx, control, subject_type="provider", subject_id=prov,
                fact_key=f"migration:{prov}:{ed.col1}:{ed.col2}", claim_ids=claims[:200],
                event_time=last_d, period=period_bucket(last_d), confidence=0.7,
                evidence=_base_evidence(
                    control, prov,
                    plain_language=(
                        f"After the rule bundling {desc(ed.col2)} into {desc(ed.col1)} started on "
                        f"{plain_date(ed.launch)}, this provider had {count_phrase(n_denied, 'refusal')} and "
                        f"then began {main} the component: {pct(rate_post)} of its claims with the "
                        f"procedure were paid this way afterwards, against {pct(rate_pre)} before."
                    ),
                    what_the_reviewer_must_verify=(
                        "Check the date the rule started and any provider education period, then "
                        "review the claims where the change turned a refusal into payment."),
                    provider_sk=prov, parent_code=ed.col1, component_code=ed.col2,
                    edit_start_date=_day(ed.launch), education_period_days=edu, window_days=window,
                    claims_with_procedure_before=len(pre_claims), claims_with_procedure_after=len(post_claims),
                    escape_rate_before=round(rate_pre, 4), escape_rate_after=round(rate_post, 4),
                    denied_component_lines_after_edit=n_denied, escape_routes=routes,
                    escaped_paid_aed=_money(paid_amt), min_rate_increase=min_rise,
                ),
                exposure=_exp.no_exposure(
                    "a change in billing behaviour is an audit lead; each escaped line must be reviewed "
                    "before any amount is established"),
            ))
    return out


# ===========================================================================
# PAY-06-R01 — tariff / contract price
# ===========================================================================


@_register("pay_06_r01_tariff_price_variance")
def pay_06_r01_tariff_price_variance(ctx, control) -> list[Signal]:
    """A line's unit price above the allowed price in force on the service date:
    the tariff price for the provider's contracted network tier (payer-specific
    where the tariff names a payer), less the contract discount, plus the
    rounding tolerance. Codes carved out of the contract are excluded."""
    lines = _lines(ctx)
    tariff = _table(ctx, "tariff")
    contract = _table(ctx, "contract")
    if lines.empty or tariff.empty or contract.empty or "activity_code" not in tariff.columns:
        return []
    tol = float(ctx.cfg("pay06_price_rounding_tolerance_pct"))
    tol_abs = float(ctx.cfg("pay06_price_rounding_tolerance_aed"))
    c = contract.copy()
    c["provider_sk"] = _col(c, "provider_sk").map(_str_id)
    c["c_payer"] = _col(c, "payer_id").map(_str_id)
    c["tier"] = _col(c, "network_tier").map(_str_id)
    c["disc"] = _num(_col(c, "discount_pct")).fillna(0.0)
    c["basis"] = _col(c, "tariff_basis").map(_str_id)
    c["c_vf"] = _col(c, "valid_from")
    c["c_vt"] = _col(c, "valid_to")
    ln = lines[(lines["activity_code"] != "") & (lines["units_n"] > 0)]
    j = ln.merge(c[["provider_sk", "c_payer", "tier", "disc", "basis", "c_vf", "c_vt"]], on="provider_sk", how="inner")
    j = j[_in_force(j["line_date"], j["c_vf"], j["c_vt"])]
    j = j[(j["c_payer"] == "") | (j["c_payer"] == j["payer_id"].map(_str_id))]
    # declared exclusion: negotiated carve-outs
    j = j[~_norm(j["basis"]).str.contains("carve|per_diem|per diem|case_rate|capitat", regex=True)]
    if j.empty:
        return []
    t = tariff.copy()
    t["activity_code"] = t["activity_code"].map(_str_id)
    t["tier"] = _col(t, "network_tier").map(_str_id)
    t["t_payer"] = _col(t, "payer_id").map(_str_id)
    t["allowed_price"] = _num(_col(t, "allowed_price"))
    t = t[t["allowed_price"] > 0]
    k = j.merge(t[["activity_code", "tier", "t_payer", "allowed_price", "valid_from", "valid_to"]],
                on=["activity_code", "tier"], how="inner", suffixes=("", "_t"))
    if k.empty:
        return []
    k = k[_in_force(k["line_date"], k["valid_from_t"] if "valid_from_t" in k.columns else k["valid_from"],
                    k["valid_to_t"] if "valid_to_t" in k.columns else k["valid_to"])]
    k = k[(k["t_payer"] == "") | (k["t_payer"] == k["payer_id"].map(_str_id))]
    # payer-specific tariff wins over the generic one
    k = k.assign(spec=(k["t_payer"] != "").astype(int)).sort_values(["line_sk", "spec"], ascending=[True, False])
    k = k.drop_duplicates("line_sk")
    disc = k["disc"].where(k["disc"] <= 1.0, k["disc"] / 100.0)  # percent or fraction
    k = k.assign(allowed=k["allowed_price"] * (1.0 - disc), disc_frac=disc)
    k["submitted"] = k["billed"] / k["units_n"]
    k = k[k["submitted"] > k["allowed"] * (1.0 + tol) + tol_abs]
    desc = _describer(ctx)
    out: list[Signal] = []
    for r in k.itertuples(index=False):
        correct = float(r.allowed) * float(r.units_n)
        share = float(r.payable) / float(r.billed) if r.billed else 1.0
        out.append(_sig(
            ctx, control, subject_type="claim", subject_id=r.claim_sk,
            fact_key=f"price:{r.line_sk}", claim_ids=[r.claim_sk],
            event_time=r.line_date, period=period_bucket(r.line_date),
            evidence=_base_evidence(
                control, r.claim_sk,
                plain_language=(
                    f"{desc(r.activity_code)} was billed at {aed(r.submitted, 2)} per unit on "
                    f"{plain_date(r.line_date)}; the contracted price for this provider's network tier "
                    f"({r.tier}) is {aed(r.allowed, 2)} — {times_phrase(r.submitted, r.allowed, 'the agreed price')}."
                ),
                what_the_reviewer_must_verify=(
                    "Check the contract and tariff in force on the service date, and any agreed "
                    "carve-out, VAT or currency difference."),
                member_sk=r.member_sk, provider_sk=r.provider_sk, line_sk=r.line_sk,
                activity_code=r.activity_code, units=float(r.units_n),
                submitted_unit_price_aed=_money(r.submitted), tariff_price_aed=_money(r.allowed_price),
                contract_discount=round(float(r.disc_frac), 4), allowed_unit_price_aed=_money(r.allowed),
                network_tier=r.tier, rounding_tolerance_pct=tol, rounding_tolerance_aed=tol_abs,
                line_amount_aed=_money(r.billed), service_date=_day(r.line_date),
            ),
            exposure=_exp.line_edit_exposure(_money(r.payable), min(_money(r.payable), correct * share)),
        ))
    return out
