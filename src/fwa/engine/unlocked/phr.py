"""Unlocked implementations for the PHR family (pharmacy, drugs and supplies).

Each function here runs only when a dataset unlock (`rules/unlocks/PHR.yaml`)
is satisfied by the loaded file. See :mod:`fwa.engine.unlocks`.

Conventions shared by every control in this module:

* Every table is scoped to ``ctx.tenant_id`` before it is read.
* "Equivalent" medicines are those the drug policy puts in one equivalence
  group (falling back to the code-equivalence map, then to the product code
  itself). Generic-for-brand substitution inside one group is never a finding.
* Every threshold is a governed parameter (``config/parameters.yaml``, keys
  prefixed ``phr``).
* A hard or expert violation on one dispensing line carries a line-edit
  exposure (billed minus correctly repriced). Statistical and network patterns
  carry either no exposure or an amount explicitly marked NOT established.
* No function lets an exception escape: a malformed or absent table yields no
  signals, and the reason is logged.
"""

from __future__ import annotations

import functools
import logging
import math
import re
from typing import Any, Callable, Iterable

import numpy as np
import pandas as pd

from ...cases import exposure as _exp
from ...presentation import aed, count_phrase, pct, plain_date, plain_number, times_phrase
from ..controls import _f, _sig
from ..evallib import is_missing, period_bucket
from ..signals import Signal

__all__ = ["IMPLEMENTATIONS"]

_log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# plumbing
# ---------------------------------------------------------------------------


def _safe(fn: Callable) -> Callable:
    """An unlocked control never lets an exception escape; it logs and returns nothing."""

    @functools.wraps(fn)
    def wrapper(ctx, control) -> list[Signal]:
        try:
            return fn(ctx, control) or []
        except Exception:  # pragma: no cover - defensive; tests exercise the happy paths
            _log.exception("%s failed on this dataset; no signals raised.", fn.__name__)
            return []

    return wrapper


def _table(ctx, name: str) -> pd.DataFrame:
    """A tenant-scoped copy of one table (empty frame when absent)."""
    dataset = getattr(ctx, "dataset", None)
    if dataset is None:
        return pd.DataFrame()
    try:
        frame = dataset.get(name)
    except Exception:
        return pd.DataFrame()
    if frame is None or frame.empty:
        return pd.DataFrame(columns=[] if frame is None else list(frame.columns))
    if "tenant_id" in frame.columns and frame["tenant_id"].notna().any():
        frame = frame[frame["tenant_id"].astype(str) == str(ctx.tenant_id)]
    return frame.copy()


def _col(frame: pd.DataFrame, name: str) -> pd.Series:
    if name in frame.columns:
        return frame[name]
    return pd.Series([None] * len(frame), index=frame.index, dtype=object)


def _str(series: pd.Series) -> pd.Series:
    """Strings with blanks and missing values as None, as plain Python objects.

    Object dtype on purpose: row-wise access to arrow-backed strings is an order
    of magnitude slower, and several controls walk small groups row by row.
    """
    mask = series.isna().tolist()
    vals = [None if m else (str(v).strip() or None) for v, m in zip(series.tolist(), mask)]
    return pd.Series(vals, index=series.index, dtype=object)


def _num(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _date(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce")


def _unestablished(amount: float, why: str) -> _exp.Exposure:
    """An amount carried for scale only — never counted as savings."""
    return _exp.Exposure(
        float(max(0.0, amount)),
        f"Pattern lead: {why} The amount is shown for scale; it is not established as "
        f"recoverable or preventable until a reviewer confirms the finding.",
        established=False,
    )


def _iso(value: Any) -> str | None:
    if is_missing(value):
        return None
    try:
        ts = pd.Timestamp(value)
        return None if pd.isna(ts) else ts.date().isoformat()
    except Exception:
        return str(value)


def _r(value: Any, places: int = 2) -> float | None:
    if is_missing(value):
        return None
    try:
        return round(float(value), places)
    except (TypeError, ValueError):
        return None


def _prefixes(text: Any) -> list[str]:
    if is_missing(text):
        return []
    return [p.strip().upper() for p in re.split(r"[;,|\s]+", str(text)) if p.strip()]


def _cfg_list(ctx, key: str) -> list[str]:
    value = ctx.cfg(key)
    if value is None:
        return []
    if isinstance(value, str):
        return [v.strip().lower() for v in re.split(r"[;,|]+", value) if v.strip()]
    return [str(v).strip().lower() for v in value if str(v).strip()]


# ---------------------------------------------------------------------------
# shared frames
# ---------------------------------------------------------------------------

_RX_STR = ("rx_sk", "claim_sk", "prescriber_id", "pharmacy_id", "prescribed_product",
           "billed_product", "dispensed_product", "member_sk", "line_sk", "authorization_id", "form")
_RX_NUM = ("prescribed_qty", "dispensed_qty", "days_supply", "billed_qty", "billed_amount",
           "strength_mg", "dose_mg_per_day")


def _rx(ctx) -> pd.DataFrame:
    """Dispensing records, typed, with the claim's member, provider, payer and dates attached."""
    rx = _table(ctx, "prescription_dispense")
    if rx.empty:
        return pd.DataFrame()
    for c in _RX_STR:
        rx[c] = _str(_col(rx, c))
    for c in _RX_NUM:
        rx[c] = _num(_col(rx, c))
    rx["fill_date"] = _date(_col(rx, "fill_date"))
    rx["prescribed_date"] = _date(_col(rx, "prescribed_date"))
    claims = ctx.claims if ctx.claims is not None else pd.DataFrame()
    if not claims.empty and "claim_sk" in claims.columns:
        if "tenant_id" in claims.columns:
            claims = claims[claims["tenant_id"].astype(str) == str(ctx.tenant_id)]
        keep = [c for c in ("claim_sk", "member_sk", "provider_sk", "payer_id", "service_date",
                            "diagnosis_primary") if c in claims.columns]
        head = claims[keep].drop_duplicates("claim_sk").copy()
        head["claim_sk"] = head["claim_sk"].astype(str)
        head = head.rename(columns={"member_sk": "_claim_member", "service_date": "_claim_date"})
        rx = rx.merge(head, on="claim_sk", how="left")
        if "_claim_member" in rx.columns:
            rx["member_sk"] = rx["member_sk"].where(rx["member_sk"].notna(), _str(rx["_claim_member"]))
        if "_claim_date" in rx.columns:
            rx["fill_date"] = rx["fill_date"].where(rx["fill_date"].notna(), _date(rx["_claim_date"]))
    for c in ("provider_sk", "payer_id", "diagnosis_primary"):
        if c not in rx.columns:
            rx[c] = None
    rx = rx[rx["rx_sk"].notna() & ~rx["claim_sk"].isin(_superseded_claims(ctx))]
    rx["billed_amount"] = rx["billed_amount"].fillna(0.0)
    return rx.reset_index(drop=True)


def _superseded_claims(ctx) -> set[str]:
    """Claims replaced by a later version (resubmission, correction) or cancelled.

    Only the latest version of a claim describes what was dispensed; counting the
    original as well would make every resubmitted fill look like a second fill.
    """
    versions = _table(ctx, "claim_version")
    out: set[str] = set()
    if not versions.empty and {"relationship", "prior_claim_sk"} <= set(versions.columns):
        rel = _str(versions["relationship"]).map(lambda r: (r or "").upper())
        later = versions[rel != "ORIGINAL"]
        out |= set(_str(later["prior_claim_sk"]).dropna())
        cancelled = versions[rel.str.contains("CANCEL|VOID|REVERS", regex=True)]
        out |= set(_str(cancelled["claim_sk"]).dropna())
    return out


def _drugs(ctx) -> pd.DataFrame:
    """The drug policy, one row per product, with a resolved equivalence group."""
    drugs = _table(ctx, "drug_policy")
    if drugs.empty or "product" not in drugs.columns:
        return pd.DataFrame(columns=["product", "group"]).set_index("product")
    drugs["product"] = _str(drugs["product"])
    drugs = drugs[drugs["product"].notna()].drop_duplicates("product").set_index("product")
    for c in ("strength_mg", "unit_price", "max_duration_days", "max_mg_per_kg_day", "vial_size_mg"):
        drugs[c] = _num(_col(drugs, c))
    for c in ("description", "equivalence_group", "therapeutic_class", "form", "indication_prefixes"):
        drugs[c] = _str(_col(drugs, c))
    for c in ("is_high_cost", "is_controlled"):
        drugs[c] = _col(drugs, c).map(_truthy)
    equiv = _table(ctx, "code_equivalence_map")
    mapped: dict[str, str] = {}
    if not equiv.empty and {"code", "equivalence_group"} <= set(equiv.columns):
        for code, grp in zip(_str(equiv["code"]), _str(equiv["equivalence_group"])):
            if code and grp:
                mapped.setdefault(code, grp)
    group = drugs["equivalence_group"].copy()
    fallback = pd.Series([mapped.get(p, p) for p in drugs.index], index=drugs.index)
    drugs["group"] = group.where(group.notna(), fallback)
    return drugs


def _truthy(v: Any) -> bool:
    if is_missing(v):
        return False
    if isinstance(v, str):
        return v.strip().lower() in ("true", "yes", "y", "1", "t")
    return bool(v)


def _group_of(products: pd.Series, drugs: pd.DataFrame) -> pd.Series:
    """Equivalence group of each product (the product itself when unknown)."""
    lookup = drugs["group"].to_dict() if "group" in drugs.columns else {}
    return products.map(lambda p: None if p is None else lookup.get(p, p))


def _describe(product: Any, drugs: pd.DataFrame) -> str:
    if is_missing(product):
        return "an unrecorded product"
    desc = drugs["description"].get(product) if "description" in drugs.columns else None
    return f"{desc} ({product})" if not is_missing(desc) else str(product)


def _price(product: Any, drugs: pd.DataFrame) -> float | None:
    if is_missing(product) or "unit_price" not in drugs.columns:
        return None
    v = drugs["unit_price"].get(product)
    return None if is_missing(v) else float(v)


def _authorizations(ctx) -> tuple[pd.DataFrame, pd.DataFrame]:
    auth = _table(ctx, "authorization")
    lines = _table(ctx, "authorization_line")
    if not auth.empty:
        auth["authorization_sk"] = _str(_col(auth, "authorization_sk"))
        auth["status"] = _str(_col(auth, "status"))
    if not lines.empty:
        lines["authorization_sk"] = _str(_col(lines, "authorization_sk"))
        lines["activity_code"] = _str(_col(lines, "activity_code"))
        lines["approved_units"] = _num(_col(lines, "approved_units"))
        lines["conditions"] = _str(_col(lines, "conditions"))
        lines["denial_code"] = _str(_col(lines, "denial_code"))
    return auth, lines


_APPROVED = ("approved", "approve", "accepted", "partially approved", "partial", "granted")


def _approved_auth_ids(auth: pd.DataFrame) -> set[str]:
    if auth.empty or "status" not in auth.columns:
        return set()
    ok = auth["status"].map(lambda s: s is not None and str(s).strip().lower() in _APPROVED)
    return set(auth.loc[ok, "authorization_sk"].dropna())


def _prescriber_units(ctx) -> tuple[dict[str, str], dict[str, dict[str, Any]], pd.DataFrame]:
    """clinician → facility provider_sk; clinician → roster facts; the provider table."""
    roster = _table(ctx, "clinician_roster")
    providers = _table(ctx, "provider")
    facility: dict[str, str] = {}
    facts: dict[str, dict[str, Any]] = {}
    if not roster.empty and "clinician_id" in roster.columns:
        roster["clinician_id"] = _str(roster["clinician_id"])
        roster["provider_sk"] = _str(_col(roster, "provider_sk"))
        for row in roster.drop_duplicates("clinician_id").itertuples(index=False):
            if row.clinician_id is None:
                continue
            facility[row.clinician_id] = row.provider_sk
            facts[row.clinician_id] = {
                "specialty": getattr(row, "specialty", None),
                "emirate": getattr(row, "emirate", None),
            }
    if not providers.empty and "provider_sk" in providers.columns:
        providers["provider_sk"] = _str(providers["provider_sk"])
        providers = providers.drop_duplicates("provider_sk").set_index("provider_sk")
    else:
        providers = pd.DataFrame().rename_axis("provider_sk")
    return facility, facts, providers


def _pfield(providers: pd.DataFrame, provider_sk: Any, column: str) -> Any:
    if is_missing(provider_sk) or column not in providers.columns or provider_sk not in providers.index:
        return None
    v = providers.at[provider_sk, column]
    return None if is_missing(v) else v


def _member_frame(ctx) -> pd.DataFrame:
    members = _table(ctx, "member")
    if members.empty or "member_sk" not in members.columns:
        return pd.DataFrame().rename_axis("member_sk")
    members["member_sk"] = _str(members["member_sk"])
    members = members.drop_duplicates("member_sk").set_index("member_sk")
    members["weight_kg"] = _num(_col(members, "weight_kg"))
    members["height_cm"] = _num(_col(members, "height_cm"))
    members["date_of_birth"] = _date(_col(members, "date_of_birth"))
    return members


def _inpatient_discharges(ctx) -> dict[str, list[pd.Timestamp]]:
    """Member → discharge dates of stays of at least one night (claim header)."""
    df = ctx.claims if ctx.claims is not None else pd.DataFrame()
    if df.empty or "discharge_date" not in df.columns or "member_sk" not in df.columns:
        return {}
    if "tenant_id" in df.columns:
        df = df[df["tenant_id"].astype(str) == str(ctx.tenant_id)]
    los = _num(_col(df, "length_of_stay_days")).fillna(0)
    sub = df[(los >= 1) & df["discharge_date"].notna()]
    out: dict[str, list[pd.Timestamp]] = {}
    for member, d in zip(sub["member_sk"].astype(str), _date(sub["discharge_date"])):
        if not pd.isna(d):
            out.setdefault(member, []).append(d)
    return out


# ===========================================================================
# PHR-01  prescribed / authorised / billed / dispensed mismatch
# ===========================================================================


@_safe
def phr_01_r01_product_identity_mismatch(ctx, control) -> list[Signal]:
    """PHR-01-R01 — billed product is neither the prescribed/authorised product nor an equivalent."""
    rx = _rx(ctx)
    if rx.empty:
        return []
    drugs = _drugs(ctx)
    rx = rx[rx["billed_product"].notna()].copy()
    rx["g_billed"] = _group_of(rx["billed_product"], drugs)
    rx["g_presc"] = _group_of(rx["prescribed_product"], drugs)
    presc_bad = rx["prescribed_product"].notna() & (rx["g_billed"] != rx["g_presc"])

    # Authorised products: authorisation lines whose activity code is a known product.
    auth, lines = _authorizations(ctx)
    auth_groups: dict[str, set[str]] = {}
    auth_products: dict[str, list[str]] = {}
    if not lines.empty:
        known = lines[lines["activity_code"].isin(set(drugs.index))]
        for aid, code in zip(known["authorization_sk"], known["activity_code"]):
            if aid is None:
                continue
            auth_groups.setdefault(aid, set()).add(_group_of(pd.Series([code]), drugs).iloc[0])
            auth_products.setdefault(aid, []).append(code)
    auth_bad = rx.apply(
        lambda r: r["authorization_id"] in auth_groups and r["g_billed"] not in auth_groups[r["authorization_id"]],
        axis=1,
    ) if auth_groups else pd.Series(False, index=rx.index)
    hits = rx[presc_bad | auth_bad]
    out: list[Signal] = []
    for row in hits.itertuples(index=False):
        against_rx = bool(row.prescribed_product is not None and row.g_billed != row.g_presc)
        authorised = auth_products.get(row.authorization_id, [])
        reference = row.prescribed_product if against_rx else (authorised[0] if authorised else None)
        billed = _f(row.billed_amount)
        ref_price = _price(reference, drugs)
        qty = _f(row.billed_qty, _f(row.dispensed_qty))
        if ref_price is not None and qty > 0:
            exposure = _exp.line_edit_exposure(billed, min(billed, ref_price * qty))
        else:
            exposure = _exp.no_exposure("the reference product has no unit price in the drug policy")
        what = "prescription" if against_rx else "approval"
        out.append(_sig(
            ctx, control, subject_type="claim", subject_id=row.claim_sk or row.rx_sk,
            fact_key=f"rxproduct:{row.rx_sk}",
            claim_ids=[row.claim_sk] if row.claim_sk else [],
            event_time=row.fill_date, period=period_bucket(row.fill_date),
            evidence={
                "plain_language": (
                    f"The bill is for {_describe(row.billed_product, drugs)} at {aed(billed)}, but the "
                    f"{what} was for {_describe(reference, drugs)}; the drug policy does not list them "
                    f"as equivalents."
                ),
                "what_the_reviewer_must_verify": (
                    "Compare the invoice with the prescription and the approval, and check whether a "
                    "substitution was approved."
                ),
                "rx_sk": row.rx_sk, "claim_sk": row.claim_sk, "member_sk": row.member_sk,
                "billed_product": row.billed_product, "prescribed_product": row.prescribed_product,
                "authorised_products": authorised or None,
                "authorization_id": row.authorization_id,
                "billed_equivalence_group": row.g_billed, "prescribed_equivalence_group": row.g_presc,
                "billed_qty": _r(row.billed_qty), "billed_amount_aed": _r(billed),
                "reference_unit_price_aed": _r(ref_price),
                "fill_date": _iso(row.fill_date), "pharmacy_id": row.pharmacy_id,
                "clinician_id": row.prescriber_id,
                "mismatch_against": what,
            },
            exposure=exposure,
        ))
    return out


@_safe
def phr_01_r02_quantity_strength_mismatch(ctx, control) -> list[Signal]:
    """PHR-01-R02 — billed quantity, strength or form exceeds the order or the approval."""
    rx = _rx(ctx)
    if rx.empty:
        return []
    drugs = _drugs(ctx)
    tol = float(ctx.cfg("phr01_quantity_tolerance"))
    dose_tol = float(ctx.cfg("phr_dose_change_tolerance"))
    rx = rx[rx["billed_product"].notna()].copy()
    rx["g_billed"] = _group_of(rx["billed_product"], drugs)
    rx["g_presc"] = _group_of(rx["prescribed_product"], drugs)
    same = rx["prescribed_product"].isna() | (rx["g_billed"] == rx["g_presc"])
    rx = rx[same]
    if rx.empty:
        return []

    # approved units per authorisation for the billed equivalence group
    auth, lines = _authorizations(ctx)
    approved: dict[tuple[str, str], float] = {}
    if not lines.empty:
        lines = lines[lines["approved_units"].notna() & lines["activity_code"].notna()]
        groups = _group_of(lines["activity_code"], drugs)
        for aid, grp, units in zip(lines["authorization_sk"], groups, lines["approved_units"]):
            key = (aid, grp)
            approved[key] = approved.get(key, 0.0) + float(units)

    # titration: a dose change against the member's previous fill of the same group
    rx = rx.sort_values(["member_sk", "g_billed", "fill_date", "rx_sk"])
    prev_dose = rx.groupby(["member_sk", "g_billed"], dropna=False)["dose_mg_per_day"].shift(1)
    rx["titration"] = (
        prev_dose.notna() & rx["dose_mg_per_day"].notna()
        & ((rx["dose_mg_per_day"] - prev_dose).abs() > dose_tol * prev_dose.abs())
    )

    out: list[Signal] = []
    for row in rx.itertuples(index=False):
        qty = row.billed_qty if not is_missing(row.billed_qty) else row.dispensed_qty
        billed = _f(row.billed_amount)
        limits: list[tuple[str, float]] = []
        if not is_missing(row.prescribed_qty) and row.prescribed_qty > 0:
            limits.append(("prescription", float(row.prescribed_qty)))
        key = (row.authorization_id, row.g_billed)
        if row.authorization_id is not None and key in approved:
            limits.append(("approval", approved[key]))
        reasons: list[str] = []
        correct = billed
        facts: dict[str, Any] = {}
        if limits and not is_missing(qty) and qty > 0:
            source, allowed = min(limits, key=lambda t: t[1])
            if qty > allowed * (1.0 + tol):
                reasons.append(
                    f"{plain_number(qty)} units billed against {plain_number(allowed)} on the {source}"
                )
                correct = min(correct, billed * allowed / qty)
                facts.update({"allowed_qty": _r(allowed), "allowed_qty_source": source})
        presc = row.prescribed_product or row.billed_product
        presc_strength = drugs["strength_mg"].get(presc) if "strength_mg" in drugs.columns else None
        presc_form = drugs["form"].get(presc) if "form" in drugs.columns else None
        if (not row.titration and not is_missing(row.strength_mg) and not is_missing(presc_strength)
                and presc_strength > 0 and row.strength_mg > presc_strength * (1.0 + tol)):
            reasons.append(
                f"strength {plain_number(row.strength_mg)} mg billed against "
                f"{plain_number(presc_strength)} mg prescribed"
            )
            correct = min(correct, billed * float(presc_strength) / float(row.strength_mg))
            facts["prescribed_strength_mg"] = _r(presc_strength)
        if (row.form is not None and not is_missing(presc_form)
                and str(row.form).strip().lower() != str(presc_form).strip().lower()):
            reasons.append(f"form '{row.form}' billed against '{presc_form}' prescribed")
            facts["prescribed_form"] = presc_form
        if not reasons:
            continue
        out.append(_sig(
            ctx, control, subject_type="claim", subject_id=row.claim_sk or row.rx_sk,
            fact_key=f"rxqty:{row.rx_sk}",
            claim_ids=[row.claim_sk] if row.claim_sk else [],
            event_time=row.fill_date, period=period_bucket(row.fill_date),
            evidence={
                "plain_language": (
                    f"{_describe(row.billed_product, drugs)} billed at {aed(billed)} with "
                    + "; ".join(reasons) + "."
                ),
                "what_the_reviewer_must_verify": (
                    "Check the prescription and approval lines for a partial fill, dose adjustment or "
                    "pack-size conversion before repricing."
                ),
                "rx_sk": row.rx_sk, "claim_sk": row.claim_sk, "member_sk": row.member_sk,
                "billed_product": row.billed_product, "prescribed_product": row.prescribed_product,
                "billed_qty": _r(qty), "prescribed_qty": _r(row.prescribed_qty),
                "billed_strength_mg": _r(row.strength_mg), "billed_form": row.form,
                "authorization_id": row.authorization_id,
                "billed_amount_aed": _r(billed), "quantity_tolerance": tol,
                "fill_date": _iso(row.fill_date), "pharmacy_id": row.pharmacy_id,
                "clinician_id": row.prescriber_id, **facts,
            },
            exposure=_exp.line_edit_exposure(billed, correct) if correct < billed
            else _exp.no_exposure("the form differs but no quantity or strength excess was priced"),
        ))
    return out


_REVERSAL_WORDS = ("cancel", "void", "revers", "withdraw")


def _reversed_claims(ctx) -> set[str]:
    out: set[str] = set()
    versions = _table(ctx, "claim_version")
    if not versions.empty and "relationship" in versions.columns:
        rel = _str(versions["relationship"]).map(lambda s: s is not None and any(
            w in str(s).lower() for w in _REVERSAL_WORDS))
        sub = versions[rel]
        out |= set(_str(_col(sub, "claim_sk")).dropna()) | set(_str(_col(sub, "prior_claim_sk")).dropna())
    remit = _table(ctx, "remittance")
    if not remit.empty and "decision" in remit.columns:
        dec = _str(remit["decision"]).map(lambda s: s is not None and any(
            w in str(s).lower() for w in _REVERSAL_WORDS))
        out |= set(_str(_col(remit[dec], "claim_sk")).dropna())
    return out


@_safe
def phr_01_r03_dispense_claim_mismatch(ctx, control) -> list[Signal]:
    """PHR-01-R03 — the pharmacy's dispensing or stock record disagrees with the bill."""
    rx = _rx(ctx)
    if rx.empty:
        return []
    drugs = _drugs(ctx)
    tol = float(ctx.cfg("phr01_quantity_tolerance"))
    stock_tol = float(ctx.cfg("phr01_stock_timing_tolerance"))
    reversed_ = _reversed_claims(ctx)
    rx = rx[rx["billed_product"].notna() & ~rx["claim_sk"].isin(reversed_)].copy()
    out: list[Signal] = []

    # limb 1: the dispensing record itself names another product or a smaller quantity
    prod_diff = rx["dispensed_product"].notna() & (rx["dispensed_product"] != rx["billed_product"])
    qty_diff = (rx["dispensed_qty"].notna() & rx["billed_qty"].notna()
                & (rx["billed_qty"] > rx["dispensed_qty"] * (1.0 + tol)))
    for row in rx[prod_diff | qty_diff].itertuples(index=False):
        billed = _f(row.billed_amount)
        qty = _f(row.billed_qty, _f(row.dispensed_qty))
        parts = []
        correct = billed
        if row.dispensed_product is not None and row.dispensed_product != row.billed_product:
            parts.append(f"the pharmacy's own record shows {_describe(row.dispensed_product, drugs)} was handed over")
            price = _price(row.dispensed_product, drugs)
            dqty = _f(row.dispensed_qty, qty)
            if price is not None:
                correct = min(correct, price * dqty)
        elif not is_missing(row.dispensed_qty) and qty > 0:
            parts.append(
                f"the pharmacy's own record shows {plain_number(row.dispensed_qty)} units dispensed "
                f"against {plain_number(qty)} billed"
            )
            correct = min(correct, billed * float(row.dispensed_qty) / qty)
        out.append(_sig(
            ctx, control, subject_type="claim", subject_id=row.claim_sk or row.rx_sk,
            fact_key=f"rxdispense:{row.rx_sk}",
            claim_ids=[row.claim_sk] if row.claim_sk else [],
            event_time=row.fill_date, period=period_bucket(row.fill_date),
            evidence={
                "plain_language": (
                    f"Billed for {_describe(row.billed_product, drugs)} ({plain_number(qty)} units, "
                    f"{aed(billed)}), but " + "; ".join(parts) + "."
                ),
                "what_the_reviewer_must_verify": (
                    "Compare the invoice with the pharmacy's dispensing record for the same date and "
                    "rule out a reversed or re-billed claim."
                ),
                "rx_sk": row.rx_sk, "claim_sk": row.claim_sk, "member_sk": row.member_sk,
                "billed_product": row.billed_product, "dispensed_product": row.dispensed_product,
                "billed_qty": _r(qty), "dispensed_qty": _r(row.dispensed_qty),
                "billed_amount_aed": _r(billed), "correct_payable_aed": _r(correct),
                "fill_date": _iso(row.fill_date), "pharmacy_id": row.pharmacy_id,
                "mismatch_limb": "dispensing record",
            },
            exposure=_exp.line_edit_exposure(billed, correct),
        ))

    # limb 2: more billed in a month than the pharmacy's stock could supply
    inv = _table(ctx, "pharmacy_inventory")
    if inv.empty or not {"pharmacy_id", "product", "period"} <= set(inv.columns):
        return out
    inv["pharmacy_id"] = _str(inv["pharmacy_id"])
    inv["product"] = _str(inv["product"])
    inv["period"] = _str(inv["period"]).map(lambda p: None if p is None else str(p)[:7])
    for c in ("opening_stock", "purchased_qty", "closing_stock"):
        inv[c] = _num(_col(inv, c))
    inv = inv.groupby(["pharmacy_id", "product", "period"], as_index=False)[
        ["opening_stock", "purchased_qty"]].sum(min_count=1)
    work = rx[rx["pharmacy_id"].notna() & rx["fill_date"].notna()].copy()
    work["period"] = work["fill_date"].dt.strftime("%Y-%m")
    work["qty"] = work["billed_qty"].where(work["billed_qty"].notna(), work["dispensed_qty"])
    agg = work.groupby(["pharmacy_id", "billed_product", "period"], as_index=False).agg(
        billed_qty=("qty", "sum"), billed_amount=("billed_amount", "sum"),
        claim_ids=("claim_sk", lambda s: sorted({c for c in s if c})), fills=("rx_sk", "count"),
        first_fill=("fill_date", "min"),
    ).rename(columns={"billed_product": "product"})
    merged = agg.merge(inv, on=["pharmacy_id", "product", "period"], how="inner")
    merged["available"] = merged["opening_stock"].fillna(0) + merged["purchased_qty"].fillna(0)
    over = merged[merged["billed_qty"] > merged["available"] * (1.0 + stock_tol)]
    for row in over.itertuples(index=False):
        billed = _f(row.billed_amount)
        available = float(row.available)
        correct = billed * available / float(row.billed_qty) if row.billed_qty > 0 else billed
        out.append(_sig(
            ctx, control, subject_type="pharmacy", subject_id=row.pharmacy_id,
            fact_key=f"rxstock:{row.pharmacy_id}:{row.product}:{row.period}",
            claim_ids=row.claim_ids, event_time=row.first_fill, period=row.period,
            evidence={
                "plain_language": (
                    f"In {row.period} this pharmacy billed {plain_number(row.billed_qty)} units of "
                    f"{_describe(row.product, drugs)} across {count_phrase(row.fills, 'fill')}, but its "
                    f"stock record shows only {plain_number(available)} units available (opening stock "
                    f"plus purchases)."
                ),
                "what_the_reviewer_must_verify": (
                    "Check the pharmacy's purchase invoices and whether late-posted deliveries or "
                    "reversed claims explain the gap."
                ),
                "pharmacy_id": row.pharmacy_id, "product": row.product, "period": row.period,
                "billed_qty": _r(row.billed_qty), "opening_stock": _r(row.opening_stock),
                "purchased_qty": _r(row.purchased_qty), "available_qty": _r(available),
                "timing_tolerance": stock_tol, "billed_amount_aed": _r(billed),
                "fills": int(row.fills), "mismatch_limb": "stock record",
            },
            exposure=_exp.line_edit_exposure(billed, correct),
        ))
    return out


@_safe
def phr_01_r04_non_medical_substitution(ctx, control) -> list[Signal]:
    """PHR-01-R04 — a receipt or stock record shows an uncovered or non-medical item handed over."""
    rx = _rx(ctx)
    receipts = _table(ctx, "member_receipt")
    if rx.empty or receipts.empty or "claim_sk" not in receipts.columns:
        return []
    drugs = _drugs(ctx)
    terms = _cfg_list(ctx, "phr01_non_medical_terms")
    window = int(ctx.cfg("phr01_receipt_window_days"))
    base_conf = float(ctx.cfg("phr01_receipt_confidence"))
    if not terms:
        return []
    receipts["claim_sk"] = _str(receipts["claim_sk"])
    receipts["item_description"] = _str(_col(receipts, "item_description"))
    receipts["receipt_date"] = _date(_col(receipts, "receipt_date"))
    receipts["amount_paid_aed"] = _num(_col(receipts, "amount_paid_aed"))
    receipts = receipts[receipts["claim_sk"].notna() & receipts["item_description"].notna()]
    pattern = re.compile("|".join(re.escape(t) for t in terms), re.IGNORECASE)
    receipts["term"] = receipts["item_description"].map(
        lambda s: (lambda m: m.group(0).lower() if m else None)(pattern.search(s)))
    receipts = receipts[receipts["term"].notna()]
    if receipts.empty:
        return []
    covered = rx[rx["billed_product"].isin(set(drugs.index))] if len(drugs) else rx
    merged = covered.merge(
        receipts[["claim_sk", "receipt_sk", "item_description", "receipt_date", "amount_paid_aed", "term"]]
        if "receipt_sk" in receipts.columns else
        receipts.assign(receipt_sk=None)[["claim_sk", "receipt_sk", "item_description", "receipt_date",
                                          "amount_paid_aed", "term"]],
        on="claim_sk", how="inner")
    if merged.empty:
        return []
    # confirmation reliability: a receipt dated far from the fill is not tied to it
    gap = (merged["receipt_date"] - merged["fill_date"]).dt.days.abs()
    merged = merged[gap.isna() | (gap <= window)]
    # the receipt must not simply name the billed medicine
    merged = merged[[
        not _mentions(item, prod, drugs)
        for item, prod in zip(merged["item_description"], merged["billed_product"])
    ]]
    stock_short = _stock_shortfall(ctx)
    out: list[Signal] = []
    for row in merged.drop_duplicates("rx_sk").itertuples(index=False):
        period = None if pd.isna(row.fill_date) else row.fill_date.strftime("%Y-%m")
        corroborated = (row.pharmacy_id, row.billed_product, period) in stock_short
        conf = min(1.0, base_conf + (0.5 * (1.0 - base_conf) if corroborated else 0.0))
        billed = _f(row.billed_amount)
        text = (
            f"The claim bills {_describe(row.billed_product, drugs)} for {aed(billed)}, but the "
            f"patient's receipt from the same visit is for '{row.item_description}'."
        )
        if corroborated:
            text += " The pharmacy's stock record also could not cover the billed medicine that month."
        out.append(_sig(
            ctx, control, subject_type="pharmacy", subject_id=row.pharmacy_id or row.claim_sk,
            fact_key=f"rxproduct_receipt:{row.rx_sk}",
            claim_ids=[row.claim_sk], event_time=row.fill_date, period=period_bucket(row.fill_date),
            confidence=conf,
            evidence={
                "plain_language": text,
                "what_the_reviewer_must_verify": (
                    "Check how reliable the receipt is and ask the patient what they actually received."
                ),
                "rx_sk": row.rx_sk, "claim_sk": row.claim_sk, "member_sk": row.member_sk,
                "pharmacy_id": row.pharmacy_id, "billed_product": row.billed_product,
                "billed_amount_aed": _r(billed), "receipt_sk": row.receipt_sk,
                "receipt_item": row.item_description, "receipt_amount_aed": _r(row.amount_paid_aed),
                "receipt_date": _iso(row.receipt_date), "fill_date": _iso(row.fill_date),
                "non_medical_term_matched": row.term, "stock_record_corroborates": bool(corroborated),
            },
            exposure=_unestablished(billed, "the billed medicine may not have been supplied."),
        ))
    return out


def _mentions(item: str, product: Any, drugs: pd.DataFrame) -> bool:
    text = str(item).lower()
    if product and str(product).lower() in text:
        return True
    desc = drugs["description"].get(product) if "description" in drugs.columns and product else None
    if desc:
        first = str(desc).lower().split()[0]
        return len(first) > 3 and first in text
    return False


def _stock_shortfall(ctx) -> set[tuple[str, str, str]]:
    """(pharmacy, product, month) whose billed quantity exceeds opening stock plus purchases."""
    inv = _table(ctx, "pharmacy_inventory")
    rx = _rx(ctx)
    if inv.empty or rx.empty or not {"pharmacy_id", "product", "period"} <= set(inv.columns):
        return set()
    inv["key"] = list(zip(_str(inv["pharmacy_id"]), _str(inv["product"]),
                          _str(inv["period"]).map(lambda p: None if p is None else str(p)[:7])))
    inv["available"] = _num(_col(inv, "opening_stock")).fillna(0) + _num(_col(inv, "purchased_qty")).fillna(0)
    available = inv.groupby("key")["available"].sum().to_dict()
    rx = rx[rx["fill_date"].notna()]
    rx["key"] = list(zip(rx["pharmacy_id"], rx["billed_product"], rx["fill_date"].dt.strftime("%Y-%m")))
    billed = rx.groupby("key")["billed_qty"].sum().to_dict()
    return {k for k, q in billed.items() if k in available and q > available[k]}


# ===========================================================================
# PHR-02  early refill, stockpiling and convergence
# ===========================================================================


def _rx_grouped(ctx) -> tuple[pd.DataFrame, pd.DataFrame]:
    rx = _rx(ctx)
    drugs = _drugs(ctx)
    if rx.empty:
        return rx, drugs
    rx = rx[rx["member_sk"].notna() & rx["fill_date"].notna() & rx["billed_product"].notna()].copy()
    rx["group"] = _group_of(rx["billed_product"], drugs)
    rx = rx.sort_values(["member_sk", "group", "fill_date", "rx_sk"]).reset_index(drop=True)
    return rx, drugs


def _override_auth_ids(ctx) -> set[str]:
    """Approvals whose conditions record a lost-medicine, travel or early-refill override."""
    auth, lines = _authorizations(ctx)
    if lines.empty:
        return set()
    flag = lines["conditions"].map(lambda s: s is not None and any(
        w in str(s).lower() for w in ("lost", "travel", "override", "vacation", "early refill")))
    return set(lines.loc[flag, "authorization_sk"].dropna())


def _inpatient_claims(ctx) -> set[str]:
    """Claims that are inpatient stays (a night or more, or typed INPATIENT)."""
    df = ctx.claims if ctx.claims is not None else pd.DataFrame()
    if df.empty or "claim_sk" not in df.columns:
        return set()
    los = _num(_col(df, "length_of_stay_days")).fillna(0)
    typ = _col(df, "claim_type").astype(object).map(lambda t: str(t).upper() if t is not None else "")
    return set(df.loc[(los >= 1) | typ.isin(["INPATIENT"]), "claim_sk"].astype(str))


def _with_previous(rx: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    """Attach the previous fill of the same member and equivalence group (rx sorted by date)."""
    g = rx.groupby(keys, sort=False)
    out = rx.copy()
    for col in ("fill_date", "days_supply", "rx_sk", "claim_sk", "dose_mg_per_day", "pharmacy_id"):
        out[f"prev_{col}"] = g[col].shift(1)
    out["prev_end"] = out["prev_fill_date"] + pd.to_timedelta(out["prev_days_supply"].fillna(0), unit="D")
    out["remaining_days"] = (out["prev_end"] - out["fill_date"]).dt.days
    return out


@_safe
def phr_02_r01_refill_overlap(ctx, control) -> list[Signal]:
    """PHR-02-R01 — remaining supply at the next fill exceeds the allowed overlap.

    The allowed overlap is the larger of a fixed number of days and a fraction of
    the previous fill's days' supply (a long injectable interval earns a
    proportionate window). Administrations during an inpatient stay are not
    refills and are left out (the catalogue's inpatient-days exclusion).
    """
    rx, drugs = _rx_grouped(ctx)
    if rx.empty:
        return []
    allowed = float(ctx.cfg("phr02_allowed_overlap_days"))
    frac = float(ctx.cfg("phr02_allowed_overlap_fraction"))
    dose_tol = float(ctx.cfg("phr_dose_change_tolerance"))
    grace = int(ctx.cfg("phr02_discharge_grace_days"))
    overrides = _override_auth_ids(ctx)
    discharges = _inpatient_discharges(ctx)
    rx = rx[~rx["claim_sk"].isin(_inpatient_claims(ctx))]
    walked = _with_previous(rx, ["member_sk", "group"])
    walked["allowed"] = np.maximum(allowed, frac * walked["prev_days_supply"].fillna(0))
    walked["dose_changed"] = (
        walked["prev_dose_mg_per_day"].notna() & walked["dose_mg_per_day"].notna()
        & ((walked["dose_mg_per_day"] - walked["prev_dose_mg_per_day"]).abs()
           > dose_tol * walked["prev_dose_mg_per_day"].abs())
    )
    cand = walked[(walked["remaining_days"] > walked["allowed"]) & ~walked["dose_changed"]
                  & ~walked["authorization_id"].isin(overrides)]
    out: list[Signal] = []
    for row in cand.itertuples(index=False):
        dis = discharges.get(str(row.member_sk), [])
        if any(pd.Timedelta(0) <= (row.fill_date - d) <= pd.Timedelta(days=grace) for d in dis):
            continue  # declared exclusion: inpatient days / discharge medication
        days = _f(row.days_supply)
        billed = _f(row.billed_amount)
        excess = float(row.remaining_days) - float(row.allowed)
        share = min(1.0, excess / days) if days > 0 else 0.0
        same_day = not pd.isna(row.prev_fill_date) and row.prev_fill_date == row.fill_date
        when = "on the same day as the previous fill" if same_day else f"(last filled {plain_date(row.prev_fill_date)})"
        out.append(_sig(
            ctx, control, subject_type="member", subject_id=row.member_sk,
            fact_key=f"refill:{row.rx_sk}",
            claim_ids=sorted(c for c in {row.claim_sk, row.prev_claim_sk} if c),
            event_time=row.fill_date, period=period_bucket(row.fill_date),
            evidence={
                "plain_language": (
                    f"{_describe(row.billed_product, drugs)} was refilled on {plain_date(row.fill_date)} {when}, "
                    f"with about {plain_number(row.remaining_days)} of the previous "
                    f"{plain_number(row.prev_days_supply)} days' supply still left; the policy allows "
                    f"{plain_number(row.allowed)} days of overlap."
                ),
                "what_the_reviewer_must_verify": (
                    "Check for a dose change, a lost-medicine or travel override, or a recent hospital stay."
                ),
                "member_sk": row.member_sk, "rx_sk": row.rx_sk, "claim_sk": row.claim_sk,
                "previous_rx_sk": row.prev_rx_sk, "previous_fill_date": _iso(row.prev_fill_date),
                "previous_days_supply": _r(row.prev_days_supply),
                "fill_date": _iso(row.fill_date), "equivalence_group": row.group,
                "billed_product": row.billed_product, "days_supply": _r(days),
                "remaining_supply_days": _r(row.remaining_days), "allowed_overlap_days": _r(row.allowed),
                "billed_amount_aed": _r(billed), "pharmacy_id": row.pharmacy_id,
                "previous_pharmacy_id": row.prev_pharmacy_id, "clinician_id": row.prescriber_id,
            },
            exposure=_exp.line_edit_exposure(billed, billed * (1.0 - share)),
        ))
    return out


@_safe
def phr_02_r02_therapy_duration_excess(ctx, control) -> list[Signal]:
    """PHR-02-R02 — continuous fills run longer than the drug policy's maximum duration without review.

    A course is a run of fills of one equivalent medicine where each fill arrives
    within the continuity gap of the supply running out. It is "without review"
    when no approval covers it and every fill in it rests on the same prescription
    (no new prescription was written during the course).
    """
    rx, drugs = _rx_grouped(ctx)
    if rx.empty or "max_duration_days" not in drugs.columns:
        return []
    gap_allow = float(ctx.cfg("phr02_continuity_gap_days"))
    tol = float(ctx.cfg("phr02_duration_tolerance"))
    chronic = {c.upper() for c in _cfg_list(ctx, "phr02_chronic_therapeutic_classes")}
    # declared exclusion: chronic maintenance therapy is expected to run past a course limit
    acute = drugs[~drugs["therapeutic_class"].fillna("").astype(str).str.upper().isin(chronic)]
    limits = acute.groupby("group")["max_duration_days"].min().dropna()
    limits = limits[limits > 0]
    rx = rx[rx["group"].isin(set(limits.index)) & ~rx["claim_sk"].isin(_inpatient_claims(ctx))].copy()
    if rx.empty:
        return []
    auth, _lines = _authorizations(ctx)
    approved = _approved_auth_ids(auth)
    keys = ["member_sk", "group"]
    rx["end"] = rx["fill_date"] + pd.to_timedelta(rx["days_supply"].fillna(0), unit="D")
    rx["cum_end"] = rx.groupby(keys, sort=False)["end"].cummax()
    prev_end = rx.groupby(keys, sort=False)["cum_end"].shift(1)
    new_course = prev_end.isna() | (rx["fill_date"] > prev_end + pd.Timedelta(days=gap_allow))
    rx["course"] = new_course.cumsum()
    rx["approved"] = rx["authorization_id"].isin(approved)
    courses = rx.groupby("course").agg(
        member_sk=("member_sk", "first"), group=("group", "first"), start=("fill_date", "min"),
        end=("end", "max"), fills=("rx_sk", "count"), approved=("approved", "any"),
        prescriptions=("prescribed_date", "nunique"), product=("billed_product", "first"))
    courses["limit"] = courses["group"].map(limits).astype(float)
    courses["span"] = (courses["end"] - courses["start"]).dt.days
    flagged = courses[(courses["span"] > courses["limit"] * (1.0 + tol)) & ~courses["approved"]
                      & (courses["prescriptions"] <= 1)]
    out: list[Signal] = []
    for course_id, c in flagged.iterrows():
        ep = rx[rx["course"] == course_id]
        limit = float(c["limit"])
        limit_date = c["start"] + pd.Timedelta(days=limit)
        beyond = ep[ep["fill_date"] >= limit_date]
        beyond_amount = float(beyond["billed_amount"].sum())
        last = ep["fill_date"].max()
        out.append(_sig(
            ctx, control, subject_type="member", subject_id=c["member_sk"],
            fact_key=f"duration:{c['member_sk']}:{c['group']}:{c['start'].date().isoformat()}",
            claim_ids=sorted({x for x in ep["claim_sk"] if x}),
            event_time=last, period=period_bucket(last), confidence=0.8,
            evidence={
                "plain_language": (
                    f"{_describe(c['product'], drugs)} has been supplied continuously for "
                    f"{plain_number(c['span'])} days ({count_phrase(c['fills'], 'fill')} on one prescription from "
                    f"{plain_date(c['start'])}), against a policy maximum of {plain_number(limit)} days, "
                    f"with no approval on file."
                ),
                "what_the_reviewer_must_verify": (
                    "Check whether the condition is chronic or a specialist approved continuation."
                ),
                "member_sk": c["member_sk"], "equivalence_group": c["group"],
                "therapy_start": _iso(c["start"]), "supply_end": _iso(c["end"]),
                "continuous_days": _r(c["span"], 0), "max_duration_days": _r(limit, 0),
                "duration_tolerance": tol, "fills": int(c["fills"]), "fills_after_limit": int(len(beyond)),
                "amount_after_limit_aed": _r(beyond_amount), "continuity_gap_days": gap_allow,
                "rx_sks": list(ep["rx_sk"]),
            },
            exposure=_unestablished(beyond_amount, "fills supplied after the policy's duration limit."),
        ))
    return out


@_safe
def phr_02_r03_multi_prescriber(ctx, control) -> list[Signal]:
    """PHR-02-R03 — the same or an equivalent medicine from many unrelated prescribers in a window."""
    rx, drugs = _rx_grouped(ctx)
    if rx.empty:
        return []
    rx = rx[rx["prescriber_id"].notna()]
    window = pd.Timedelta(days=int(ctx.cfg("phr02_prescriber_window_days")))
    threshold = int(ctx.cfg("phr02_min_distinct_prescribers"))
    threshold_ctl = int(ctx.cfg("phr02_min_distinct_prescribers_controlled"))
    facility, _facts, providers = _prescriber_units(ctx)

    def unit(clinician: str) -> str:
        # care team / provider-group identity: prescribers at one facility, or at
        # facilities with one declared owner, count once.
        fac = facility.get(clinician)
        owner = _pfield(providers, fac, "owner_entity_id")
        return f"owner:{owner}" if owner else (f"facility:{fac}" if fac else f"clinician:{clinician}")

    rx["unit"] = rx["prescriber_id"].map(unit)
    rx = rx[rx.groupby(["member_sk", "group"])["unit"].transform("nunique") >= min(threshold, threshold_ctl)]
    controlled = drugs.groupby("group")["is_controlled"].any().to_dict() if "is_controlled" in drugs.columns else {}
    out: list[Signal] = []
    for (member, group), g in rx.groupby(["member_sk", "group"], sort=False):
        need = threshold_ctl if controlled.get(group) else threshold
        if g["unit"].nunique() < need:
            continue
        dates = list(g["fill_date"])
        units = list(g["unit"])
        best = None
        lo = 0
        for hi in range(len(dates)):
            while dates[hi] - dates[lo] > window:
                lo += 1
            distinct = len(set(units[lo:hi + 1]))
            if distinct >= need and (best is None or distinct > best[0]):
                best = (distinct, lo, hi)
        if best is None:
            continue
        n, lo, hi = best
        w = g.iloc[lo:hi + 1]
        out.append(_sig(
            ctx, control, subject_type="member", subject_id=member,
            fact_key=f"multiprescriber:{member}:{group}",
            claim_ids=sorted({c for c in w["claim_sk"] if c}),
            event_time=w["fill_date"].max(), period=period_bucket(w["fill_date"].max()),
            confidence=0.8,
            evidence={
                "plain_language": (
                    f"This patient received {_describe(w.iloc[0]['billed_product'], drugs)} or an "
                    f"equivalent from {n} unrelated prescribers between {plain_date(w['fill_date'].min())} "
                    f"and {plain_date(w['fill_date'].max())}; the threshold is {need}."
                ),
                "what_the_reviewer_must_verify": (
                    "Check whether the prescribers belong to one care team and review the full "
                    "prescription history."
                ),
                "member_sk": member, "equivalence_group": group,
                "distinct_prescriber_groups": int(n), "threshold": int(need),
                "controlled_drug": bool(controlled.get(group, False)),
                "prescribers": sorted(set(w["prescriber_id"])),
                "window_days": int(window.days), "fills_in_window": int(len(w)),
                "window_start": _iso(w["fill_date"].min()), "window_end": _iso(w["fill_date"].max()),
                "billed_amount_in_window_aed": _r(w["billed_amount"].sum()),
            },
            exposure=_unestablished(float(w["billed_amount"].sum()), "fills from several prescribers in one window."),
        ))
    return out


@_safe
def phr_02_r04_multi_pharmacy(ctx, control) -> list[Signal]:
    """PHR-02-R04 — overlapping supplies of one equivalent medicine from several pharmacies."""
    rx, drugs = _rx_grouped(ctx)
    if rx.empty:
        return []
    allowed = float(ctx.cfg("phr02_allowed_overlap_days"))
    need = int(ctx.cfg("phr02_min_distinct_pharmacies"))
    rx = rx[rx["pharmacy_id"].notna()].copy()
    # declared exclusion: partial fills are split legitimately between pharmacies
    partial = rx["dispensed_qty"].notna() & rx["prescribed_qty"].notna() & (rx["dispensed_qty"] < rx["prescribed_qty"])
    rx = rx[~partial]
    rx = rx[~rx["claim_sk"].isin(_inpatient_claims(ctx))]
    rx = rx[rx.groupby(["member_sk", "group"])["pharmacy_id"].transform("nunique") >= need].copy()
    if rx.empty:
        return []
    rx["end"] = rx["fill_date"] + pd.to_timedelta(rx["days_supply"].fillna(0), unit="D")
    out_of_stock = _out_of_stock(ctx)
    out: list[Signal] = []
    for (member, group), g in rx.groupby(["member_sk", "group"], sort=False):
        if g["pharmacy_id"].nunique() < need:
            continue
        starts, ends = list(g["fill_date"]), list(g["end"])
        pharm, products = list(g["pharmacy_id"]), list(g["billed_product"])
        best = None
        for i in range(len(g)):
            members = [i]
            for j in range(len(g)):
                if j == i:
                    continue
                overlap = (min(ends[i], ends[j]) - max(starts[i], starts[j])).days
                if overlap > allowed:
                    members.append(j)
            # declared exclusion: a pharmacy that had run out of stock is not counted
            counted = {pharm[k] for k in members
                       if (pharm[k], products[k], starts[k].strftime("%Y-%m")) not in out_of_stock}
            if len(counted) >= need and (best is None or len(counted) > len(best[0])):
                best = (counted, sorted(members))
        if best is None:
            continue
        pharmacies, idx = best
        w = g.iloc[idx]
        amount = float(w["billed_amount"].sum())
        out.append(_sig(
            ctx, control, subject_type="member", subject_id=member,
            fact_key=f"multipharmacy:{member}:{group}",
            claim_ids=sorted({c for c in w["claim_sk"] if c}),
            event_time=w["fill_date"].max(), period=period_bucket(w["fill_date"].max()),
            confidence=0.7,
            evidence={
                "plain_language": (
                    f"This patient held overlapping supplies of {_describe(w.iloc[0]['billed_product'], drugs)} "
                    f"or an equivalent from {len(pharmacies)} different pharmacies between "
                    f"{plain_date(w['fill_date'].min())} and {plain_date(w['end'].max())} "
                    f"({count_phrase(len(w), 'fill')}, {aed(amount)})."
                ),
                "what_the_reviewer_must_verify": (
                    "Check for stock shortages or partial fills that forced a second pharmacy, and review "
                    "the dates and quantities dispensed at each."
                ),
                "member_sk": member, "equivalence_group": group,
                "pharmacies": sorted(pharmacies), "distinct_pharmacies": len(pharmacies),
                "threshold": need, "allowed_overlap_days": allowed,
                "rx_sks": list(w["rx_sk"]), "billed_amount_aed": _r(amount),
                "first_fill": _iso(w["fill_date"].min()), "last_supply_end": _iso(w["end"].max()),
            },
            exposure=_unestablished(amount, "overlapping supplies from several pharmacies."),
        ))
    return out


def _out_of_stock(ctx) -> set[tuple[str, str, str]]:
    inv = _table(ctx, "pharmacy_inventory")
    if inv.empty or not {"pharmacy_id", "product", "period", "closing_stock"} <= set(inv.columns):
        return set()
    closing = _num(inv["closing_stock"])
    sub = inv[closing.notna() & (closing <= 0)]
    return set(zip(_str(sub["pharmacy_id"]), _str(sub["product"]),
                   _str(sub["period"]).map(lambda p: None if p is None else str(p)[:7])))


# ===========================================================================
# PHR-03  clinical appropriateness and wastage
# ===========================================================================


@_safe
def phr_03_r01_dose_weight_conflict(ctx, control) -> list[Signal]:
    """PHR-03-R01 — daily dose per kilogram above the drug policy's limit for the member's weight."""
    rx, drugs = _rx_grouped(ctx)
    members = _member_frame(ctx)
    if rx.empty or members.empty or "max_mg_per_kg_day" not in drugs.columns:
        return []
    tol = float(ctx.cfg("phr03_dose_rounding_tolerance"))
    loading = float(ctx.cfg("phr03_loading_dose_days"))
    rx["weight_kg"] = rx["member_sk"].map(members["weight_kg"])
    rx["height_cm"] = rx["member_sk"].map(members["height_cm"])
    rx["limit"] = rx["billed_product"].map(drugs["max_mg_per_kg_day"])
    strength = rx["strength_mg"].where(rx["strength_mg"].notna(), rx["billed_product"].map(drugs["strength_mg"]))
    qty = rx["billed_qty"].where(rx["billed_qty"].notna(), rx["dispensed_qty"])
    derived = qty * strength / rx["days_supply"].where(rx["days_supply"] > 0)
    rx["dose"] = rx["dose_mg_per_day"].where(rx["dose_mg_per_day"].notna(), derived)
    rx["dose_source"] = np.where(rx["dose_mg_per_day"].notna(), "recorded daily dose",
                                 "billed quantity × strength ÷ days' supply")
    rx["mgkg"] = rx["dose"] / rx["weight_kg"].where(rx["weight_kg"] > 0)
    # declared exclusion: a short first fill is a loading dose
    first = rx.groupby(["member_sk", "group"]).cumcount() == 0
    is_loading = first & (rx["days_supply"] <= loading)
    cand = rx[rx["limit"].notna() & (rx["limit"] > 0) & rx["mgkg"].notna()
              & (rx["mgkg"] > rx["limit"] * (1.0 + tol)) & ~is_loading]
    out: list[Signal] = []
    for row in cand.itertuples(index=False):
        billed = _f(row.billed_amount)
        correct = billed * float(row.limit) / float(row.mgkg)
        out.append(_sig(
            ctx, control, subject_type="claim", subject_id=row.claim_sk or row.rx_sk,
            fact_key=f"rxdose:{row.rx_sk}",
            claim_ids=[row.claim_sk] if row.claim_sk else [],
            event_time=row.fill_date, period=period_bucket(row.fill_date),
            evidence={
                "plain_language": (
                    f"{_describe(row.billed_product, drugs)} billed at {plain_number(row.dose, 0)} mg a day "
                    f"for a patient weighing {plain_number(row.weight_kg, 0)} kg — "
                    f"{plain_number(row.mgkg, 1)} mg/kg/day against a limit of "
                    f"{plain_number(row.limit, 1)} mg/kg/day ({times_phrase(row.mgkg, row.limit, 'the limit')})."
                ),
                "what_the_reviewer_must_verify": (
                    "Recheck the patient's weight and the dose given, allowing for loading doses and rounding."
                ),
                "rx_sk": row.rx_sk, "claim_sk": row.claim_sk, "member_sk": row.member_sk,
                "billed_product": row.billed_product, "daily_dose_mg": _r(row.dose),
                "dose_source": row.dose_source, "weight_kg": _r(row.weight_kg),
                "height_cm": _r(row.height_cm), "mg_per_kg_day": _r(row.mgkg, 3),
                "max_mg_per_kg_day": _r(row.limit, 3), "rounding_tolerance": tol,
                "billed_amount_aed": _r(billed), "fill_date": _iso(row.fill_date),
                "clinician_id": row.prescriber_id,
            },
            exposure=_exp.line_edit_exposure(billed, correct),
        ))
    return out


def _member_diagnoses(ctx) -> dict[str, list[tuple[pd.Timestamp, str]]]:
    """Member → (date, diagnosis code) from the diagnosis table and claim headers."""
    claims = ctx.claims if ctx.claims is not None else pd.DataFrame()
    if claims.empty or "claim_sk" not in claims.columns:
        return {}
    if "tenant_id" in claims.columns:
        claims = claims[claims["tenant_id"].astype(str) == str(ctx.tenant_id)]
    head = claims[[c for c in ("claim_sk", "member_sk", "service_date", "diagnosis_primary")
                   if c in claims.columns]].copy()
    head["claim_sk"] = head["claim_sk"].astype(str)
    head["service_date"] = _date(_col(head, "service_date"))
    rows = [head.rename(columns={"diagnosis_primary": "code"})[["member_sk", "service_date", "code"]]]
    dx = _table(ctx, "diagnosis")
    if not dx.empty and {"claim_sk", "code"} <= set(dx.columns):
        dx["claim_sk"] = _str(dx["claim_sk"])
        dx = dx.merge(head[["claim_sk", "member_sk", "service_date"]], on="claim_sk", how="inner")
        rows.append(dx[["member_sk", "service_date", "code"]])
    allrows = pd.concat(rows, ignore_index=True)
    allrows = allrows[allrows["code"].notna() & allrows["member_sk"].notna()]
    out: dict[str, list[tuple[pd.Timestamp, str]]] = {}
    for m, d, c in zip(allrows["member_sk"].astype(str), allrows["service_date"],
                       allrows["code"].astype(str).str.upper().str.replace(".", "", regex=False)):
        out.setdefault(m, []).append((d, c))
    return out


@_safe
def phr_03_r02_drug_diagnosis_step_conflict(ctx, control) -> list[Signal]:
    """PHR-03-R02 — no supporting diagnosis for the drug, or the required first-line therapy is absent."""
    rx, drugs = _rx_grouped(ctx)
    if rx.empty:
        return []
    lookback_dx = pd.Timedelta(days=int(ctx.cfg("phr03_indication_lookback_days")))
    rare = [p.upper() for p in _cfg_list(ctx, "phr03_rare_disease_prefixes")]
    auth, _lines = _authorizations(ctx)
    approved = _approved_auth_ids(auth)
    dx = _member_diagnoses(ctx)
    steps = _table(ctx, "step_therapy_policy")
    step_map: dict[str, tuple[str, int]] = {}
    if not steps.empty and {"product", "required_prior_group"} <= set(steps.columns):
        for p, grp, lb in zip(_str(steps["product"]), _str(steps["required_prior_group"]),
                              _num(_col(steps, "lookback_days"))):
            if p and grp:
                step_map[p] = (grp, int(lb) if not pd.isna(lb) else 365)
    history_start = rx["fill_date"].min()
    fills_by_member = {m: g for m, g in rx.groupby("member_sk", sort=False)}
    indications = drugs["indication_prefixes"].map(_prefixes).to_dict() if "indication_prefixes" in drugs.columns else {}
    out: list[Signal] = []
    for row in rx.itertuples(index=False):
        if row.authorization_id in approved:
            continue  # declared exclusion: authorisation override
        codes = [(d, c) for d, c in dx.get(str(row.member_sk), [])
                 if pd.isna(d) or (row.fill_date - lookback_dx <= d <= row.fill_date + pd.Timedelta(days=1))]
        code_set = sorted({c for _, c in codes})
        if rare and any(c.startswith(p) for c in code_set for p in rare):
            continue  # declared exclusion: rare disease
        problems: list[str] = []
        facts: dict[str, Any] = {}
        prefixes = indications.get(row.billed_product) or []
        if prefixes and code_set and not any(c.startswith(p) for c in code_set for p in prefixes):
            problems.append(
                f"none of the patient's diagnoses ({', '.join(code_set[:4])}) is an approved "
                f"indication (expected codes starting {', '.join(prefixes[:4])})"
            )
            facts.update({"indication_prefixes": prefixes, "member_diagnoses": code_set[:10]})
        if row.billed_product in step_map:
            req, lb = step_map[row.billed_product]
            since = row.fill_date - pd.Timedelta(days=lb)
            if not pd.isna(history_start) and since >= history_start:
                hist = fills_by_member.get(row.member_sk)
                prior = hist[(hist["group"] == req) & (hist["fill_date"] < row.fill_date)
                             & (hist["fill_date"] >= since)] if hist is not None else []
                if len(prior) == 0:
                    problems.append(
                        f"no fill of the required first-line treatment ({req}) in the previous {lb} days"
                    )
                    facts.update({"required_prior_group": req, "step_lookback_days": lb})
        if not problems:
            continue
        billed = _f(row.billed_amount)
        out.append(_sig(
            ctx, control, subject_type="claim", subject_id=row.claim_sk or row.rx_sk,
            fact_key=f"rxindication:{row.rx_sk}",
            claim_ids=[row.claim_sk] if row.claim_sk else [],
            event_time=row.fill_date, period=period_bucket(row.fill_date),
            evidence={
                "plain_language": (
                    f"{_describe(row.billed_product, drugs)}, billed at {aed(billed)}, was dispensed on "
                    f"{plain_date(row.fill_date)} with " + "; and ".join(problems) + "."
                ),
                "what_the_reviewer_must_verify": (
                    "Check the patient's history for the indication and earlier treatment, and for an "
                    "override or rare-disease exception."
                ),
                "rx_sk": row.rx_sk, "claim_sk": row.claim_sk, "member_sk": row.member_sk,
                "billed_product": row.billed_product, "fill_date": _iso(row.fill_date),
                "billed_amount_aed": _r(billed), "clinician_id": row.prescriber_id,
                "indication_lookback_days": int(lookback_dx.days), **facts,
            },
            exposure=_unestablished(billed, "the drug may be payable once the indication or earlier "
                                            "treatment is documented."),
        ))
    return out


@_safe
def phr_03_r04_wastage_anomaly(ctx, control) -> list[Signal]:
    """PHR-03-R04 — billed wastage far above the vial arithmetic, compared with peers giving the same drug.

    For each vial-based drug line the expected waste is what single-use vials
    force: ``ceil(dose ÷ vial) × vial − dose`` (this is how the single-use vial
    policy is honoured — that waste is never a finding). The billed waste in
    excess of it is compared per provider with the other providers billing the
    same drug.
    """
    lines = _table(ctx, "claim_line")
    rx = _rx(ctx)
    drugs = _drugs(ctx)
    if lines.empty or "wastage_units" not in lines.columns or "vial_size_mg" not in drugs.columns:
        return []
    multiple = float(ctx.cfg("phr03_wastage_peer_multiple"))
    min_lines = int(ctx.cfg("phr03_wastage_min_lines"))
    min_peers = int(ctx.cfg("phr03_wastage_min_peer_providers"))
    lines["line_sk"] = _str(_col(lines, "line_sk"))
    lines["claim_sk"] = _str(_col(lines, "claim_sk"))
    lines["product"] = _str(_col(lines, "product"))
    lines["wastage_units"] = _num(lines["wastage_units"])
    lines["units"] = _num(_col(lines, "units"))
    lines["unit_price"] = _num(_col(lines, "unit_price"))
    lines["net_amount"] = _num(_col(lines, "net_amount"))
    if not rx.empty:
        dose = rx.dropna(subset=["line_sk"]).drop_duplicates("line_sk").set_index("line_sk")
        lines["dose_mg"] = lines["line_sk"].map(dose["dose_mg_per_day"])
        lines["product"] = lines["product"].where(lines["product"].notna(),
                                                  lines["line_sk"].map(dose["billed_product"]))
    else:
        lines["dose_mg"] = np.nan
    lines["vial_mg"] = lines["product"].map(drugs["vial_size_mg"])
    work = lines[lines["vial_mg"].notna() & (lines["vial_mg"] > 0) & lines["wastage_units"].notna()].copy()
    if work.empty:
        return []
    # units on a vial-drug line are milligrams administered (+ wastage); fall back to units
    work["dose_mg"] = work["dose_mg"].where(work["dose_mg"].notna(), work["units"] - work["wastage_units"])
    work = work[work["dose_mg"] > 0]
    work["expected_waste"] = np.ceil(work["dose_mg"] / work["vial_mg"]) * work["vial_mg"] - work["dose_mg"]
    work["excess_waste"] = (work["wastage_units"] - work["expected_waste"]).clip(lower=0)
    work["excess_ratio"] = work["excess_waste"] / work["vial_mg"]
    claims = ctx.claims if ctx.claims is not None else pd.DataFrame()
    if claims.empty or not {"claim_sk", "provider_sk"} <= set(claims.columns):
        return []
    if "tenant_id" in claims.columns:
        claims = claims[claims["tenant_id"].astype(str) == str(ctx.tenant_id)]
    claims = claims[[c for c in ("claim_sk", "provider_sk", "service_date") if c in claims.columns]].copy()
    claims = claims.rename(columns={"service_date": "_claim_date"})
    claims["claim_sk"] = claims["claim_sk"].astype(str)
    work = work.merge(claims.drop_duplicates("claim_sk"), on="claim_sk", how="inner")
    line_date = _date(_col(work, "service_date"))
    work["service_date"] = line_date.where(line_date.notna(), _date(_col(work, "_claim_date")))
    per = work.groupby(["product", "provider_sk"], as_index=False).agg(
        lines=("line_sk", "count"), mean_excess_ratio=("excess_ratio", "mean"),
        excess_mg=("excess_waste", "sum"), billed_waste_mg=("wastage_units", "sum"),
        expected_waste_mg=("expected_waste", "sum"),
        unit_price=("unit_price", "median"), last=("service_date", "max"),
        claim_ids=("claim_sk", lambda s: sorted(set(s))),
    )
    out: list[Signal] = []
    for product, grp in per.groupby("product", sort=False):
        if len(grp) < min_peers:
            continue
        for row in grp[grp["lines"] >= min_lines].itertuples(index=False):
            others = grp[grp["provider_sk"] != row.provider_sk]
            peer = float(others["mean_excess_ratio"].median()) if len(others) else float("nan")
            floor = max(peer, 0.0)
            if not math.isfinite(peer) or row.mean_excess_ratio <= 0:
                continue
            # a peer median of zero excess means any regular excess stands out; require a
            # full vial's worth on average in that case so rounding noise never fires
            bar = floor * multiple if floor > 0 else float(ctx.cfg("phr03_wastage_min_excess_vials"))
            if row.mean_excess_ratio < bar:
                continue
            price = _f(row.unit_price)
            amount = float(row.excess_mg) * price if price > 0 else 0.0
            out.append(_sig(
                ctx, control, subject_type="provider", subject_id=row.provider_sk,
                fact_key=f"wastage:{row.provider_sk}:{product}",
                claim_ids=row.claim_ids[:50], event_time=row.last, period=period_bucket(row.last),
                confidence=0.7, peer_level_used="product",
                evidence={
                    "plain_language": (
                        f"Across {count_phrase(row.lines, 'line')} of {_describe(product, drugs)}, this provider "
                        f"billed {plain_number(row.billed_waste_mg)} mg of wasted drug where vial sizes "
                        f"explain {plain_number(row.expected_waste_mg)} mg; the excess averages "
                        f"{plain_number(row.mean_excess_ratio, 2)} vials a line against "
                        f"{plain_number(peer, 2)} for other providers giving the same drug."
                    ),
                    "what_the_reviewer_must_verify": (
                        "Compare the billed wastage with the dose recorded in the administration notes "
                        "and the single-use vial policy."
                    ),
                    "provider_sk": row.provider_sk, "product": product, "lines": int(row.lines),
                    "billed_wastage_mg": _r(row.billed_waste_mg),
                    "vial_arithmetic_wastage_mg": _r(row.expected_waste_mg),
                    "excess_wastage_mg": _r(row.excess_mg),
                    "mean_excess_vials_per_line": _r(row.mean_excess_ratio, 3),
                    "peer_median_excess_vials_per_line": _r(peer, 3),
                    "peer_providers": int(len(others)), "peer_multiple": multiple,
                    "vial_size_mg": _r(drugs["vial_size_mg"].get(product)),
                },
                exposure=_unestablished(amount, "billed wastage above what single-use vials require."),
            ))
    return out


# ===========================================================================
# PHR-04  prescriber–pharmacy steering
# ===========================================================================


def _edges(ctx) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, str], dict[str, dict[str, Any]], pd.DataFrame]:
    rx = _rx(ctx)
    drugs = _drugs(ctx)
    facility, facts, providers = _prescriber_units(ctx)
    if rx.empty:
        return rx, drugs, facility, facts, providers
    rx = rx[rx["prescriber_id"].notna() & rx["pharmacy_id"].notna()].copy()
    rx["facility"] = rx["prescriber_id"].map(facility)
    return rx, drugs, facility, facts, providers


def _same_owner(providers: pd.DataFrame, a: Any, b: Any) -> bool:
    if is_missing(a) or is_missing(b):
        return False
    if a == b:
        return True
    oa, ob = _pfield(providers, a, "owner_entity_id"), _pfield(providers, b, "owner_entity_id")
    return bool(oa) and oa == ob


@_safe
def phr_04_r01_top_pharmacy_concentration(ctx, control) -> list[Signal]:
    """PHR-04-R01 — a prescriber's share to one pharmacy far above peers of the same specialty and emirate."""
    rx, drugs, facility, facts, providers = _edges(ctx)
    if rx.empty:
        return []
    min_rx = int(ctx.cfg("phr04_min_prescriptions"))
    min_share = float(ctx.cfg("phr04_top_share_min"))
    margin = float(ctx.cfg("phr04_peer_share_margin"))
    min_peers = int(ctx.cfg("phr04_min_peer_prescribers"))
    min_pharm = int(ctx.cfg("phr04_narrow_network_min_pharmacies"))
    counts = rx.groupby(["prescriber_id", "pharmacy_id"]).size().rename("n").reset_index()
    tot = counts.groupby("prescriber_id")["n"].sum().rename("total")
    top = counts.sort_values(["prescriber_id", "n", "pharmacy_id"], ascending=[True, False, True]) \
        .drop_duplicates("prescriber_id").set_index("prescriber_id")
    pres = pd.DataFrame({"total": tot, "top_pharmacy": top["pharmacy_id"], "top_n": top["n"]})
    pres = pres[pres["total"] >= min_rx].copy()
    if pres.empty:
        return []
    pres["share"] = pres["top_n"] / pres["total"]
    pres["specialty"] = [facts.get(p, {}).get("specialty") for p in pres.index]
    pres["emirate"] = [facts.get(p, {}).get("emirate") or _pfield(providers, facility.get(p), "emirate")
                       for p in pres.index]
    # pharmacies serving each emirate (narrow-network exclusion)
    rx["pharm_emirate"] = rx["pharmacy_id"].map(lambda p: _pfield(providers, p, "emirate"))
    pharm_by_emirate = rx.groupby("pharm_emirate")["pharmacy_id"].nunique().to_dict()
    out: list[Signal] = []
    for pid, row in pres.iterrows():
        peers, level = _peer_shares(pres, pid, min_peers)
        if peers is None:
            continue
        peer_median = float(peers.median())
        if row["share"] < min_share or row["share"] - peer_median < margin:
            continue
        if _same_owner(providers, facility.get(pid), row["top_pharmacy"]):
            continue  # declared exclusion: on-site / in-house pharmacy
        if row["emirate"] and pharm_by_emirate.get(row["emirate"], min_pharm) < min_pharm:
            continue  # declared exclusion: narrow network
        sub = rx[(rx["prescriber_id"] == pid) & (rx["pharmacy_id"] == row["top_pharmacy"])]
        last = sub["fill_date"].max()
        out.append(_sig(
            ctx, control, subject_type="clinician", subject_id=pid,
            fact_key=f"topphar:{pid}:{row['top_pharmacy']}",
            claim_ids=sorted({c for c in sub["claim_sk"] if c})[:50],
            event_time=last, period=period_bucket(last), confidence=0.7, peer_level_used=level,
            evidence={
                "plain_language": (
                    f"{pct(row['share'])} of this prescriber's {plain_number(row['total'])} prescriptions "
                    f"were filled at one pharmacy; for similar prescribers ({level}) the typical share is "
                    f"{pct(peer_median)}."
                ),
                "what_the_reviewer_must_verify": (
                    "Check whether the pharmacy is on site or the only specialist pharmacy nearby, and "
                    "look for business links between the two."
                ),
                "clinician_id": pid, "provider_sk": facility.get(pid),
                "pharmacy_id": row["top_pharmacy"], "prescriptions": int(row["total"]),
                "top_pharmacy_prescriptions": int(row["top_n"]),
                "top_pharmacy_share": _r(row["share"], 3), "peer_median_share": _r(peer_median, 3),
                "peer_prescribers": int(len(peers)), "peer_group": level,
                "specialty": row["specialty"], "emirate": row["emirate"],
                "share_threshold": min_share, "peer_margin": margin,
            },
            exposure=_exp.no_exposure("a referral-concentration pattern is a lead, not an amount"),
        ))
    return out


def _peer_shares(pres: pd.DataFrame, pid: str, min_peers: int) -> tuple[pd.Series | None, str]:
    row = pres.loc[pid]
    others = pres.drop(index=pid)
    options = [
        ("same specialty and emirate",
         (others["specialty"] == row["specialty"]) & (others["emirate"] == row["emirate"])),
        ("same specialty", others["specialty"] == row["specialty"]),
        ("all prescribers", pd.Series(True, index=others.index)),
    ]
    for level, mask in options:
        peers = others.loc[mask.fillna(False), "share"]
        if len(peers) >= min_peers:
            return peers, level
    return None, ""


@_safe
def phr_04_r02_reciprocal_value_loop(ctx, control) -> list[Signal]:
    """PHR-04-R02 — concentrated facility→pharmacy flow between parties sharing bank, phone or address."""
    rx, drugs, facility, facts, providers = _edges(ctx)
    if rx.empty or providers.empty:
        return []
    min_share = float(ctx.cfg("phr04_loop_min_share"))
    min_rx = int(ctx.cfg("phr04_min_prescriptions"))
    rx = rx[rx["facility"].notna()]
    if rx.empty:
        return []
    counts = rx.groupby(["facility", "pharmacy_id"]).agg(
        n=("rx_sk", "count"), amount=("billed_amount", "sum"),
        prescribers=("prescriber_id", lambda s: sorted(set(s))),
        last=("fill_date", "max")).reset_index()
    totals = counts.groupby("facility")["n"].transform("sum")
    counts["share"] = counts["n"] / totals
    counts["total"] = totals
    cand = counts[(counts["share"] >= min_share) & (counts["n"] >= min_rx)]
    referrals = _table(ctx, "referral")
    out: list[Signal] = []
    for row in cand.itertuples(index=False):
        fac, ph = row.facility, row.pharmacy_id
        if fac == ph:
            continue
        shared = [
            label for col, label in (("bank_account_token", "bank account"), ("phone_token", "phone number"),
                                     ("address_token", "address"))
            if _pfield(providers, fac, col) and _pfield(providers, fac, col) == _pfield(providers, ph, col)
        ]
        if not shared:
            continue
        of, op = _pfield(providers, fac, "owner_entity_id"), _pfield(providers, ph, "owner_entity_id")
        if of and op and of == op:
            continue  # declared exclusion: a known (declared) corporate relationship
        n_ref = 0
        if not referrals.empty and {"referrer_provider_sk", "recipient_provider_sk"} <= set(referrals.columns):
            a, b = _str(referrals["referrer_provider_sk"]), _str(referrals["recipient_provider_sk"])
            n_ref = int((((a == fac) & (b == ph)) | ((a == ph) & (b == fac))).sum())
        sub = rx[(rx["facility"] == fac) & (rx["pharmacy_id"] == ph)]
        amounts = sub.groupby("claim_sk")["billed_amount"].sum().to_dict()
        out.append(_sig(
            ctx, control, subject_type="provider", subject_id=ph,
            fact_key=f"loop:{fac}:{ph}",
            claim_ids=sorted(c for c in amounts if c)[:50],
            event_time=row.last, period=period_bucket(row.last), confidence=0.7,
            evidence={
                "plain_language": (
                    f"{pct(row.share)} of the prescriptions written at this clinic "
                    f"({plain_number(row.n)} of {plain_number(row.total)}, {aed(row.amount)}) went to one "
                    f"pharmacy that shares its {' and '.join(shared)} with the clinic, although the two "
                    f"declare different owners."
                ),
                "what_the_reviewer_must_verify": (
                    "Check company registration records for a common owner and rule out a known "
                    "corporate relationship."
                ),
                "provider_sk": fac, "pharmacy_id": ph, "prescriptions": int(row.n),
                "facility_prescriptions": int(row.total), "flow_share": _r(row.share, 3),
                "flow_amount_aed": _r(row.amount), "shared_identifiers": shared,
                "facility_owner": of, "pharmacy_owner": op, "referrals_between": n_ref,
                "prescribers": row.prescribers, "share_threshold": min_share,
            },
            exposure=_unestablished(float(sum(amounts.values())),
                                    "every distinct claim on the linked flow, each counted once."),
        ))
    return out


@_safe
def phr_04_r03_high_cost_steering(ctx, control) -> list[Signal]:
    """PHR-04-R03 — a prescriber's pharmacy concentration is much stronger for high-cost drugs."""
    rx, drugs, facility, facts, providers = _edges(ctx)
    if rx.empty or "is_high_cost" not in drugs.columns:
        return []
    gap = float(ctx.cfg("phr04_high_cost_share_gap"))
    min_n = int(ctx.cfg("phr04_high_cost_min_rx"))
    min_avail = int(ctx.cfg("phr04_availability_min_pharmacies"))
    rx["high_cost"] = rx["billed_product"].map(drugs["is_high_cost"]).fillna(False).astype(bool)
    inv = _table(ctx, "pharmacy_inventory")
    stocked: dict[str, int] = {}
    if not inv.empty and {"pharmacy_id", "product"} <= set(inv.columns):
        stocked = inv.groupby(_str(inv["product"]))["pharmacy_id"].nunique().to_dict()
    else:
        stocked = rx.groupby("billed_product")["pharmacy_id"].nunique().to_dict()
    out: list[Signal] = []
    for pid, g in rx.groupby("prescriber_id", sort=False):
        hc, other = g[g["high_cost"]], g[~g["high_cost"]]
        if len(hc) < min_n or len(other) < min_n:
            continue
        top = hc["pharmacy_id"].value_counts()
        pharmacy = top.index[0]
        hc_share = float(top.iloc[0]) / len(hc)
        other_share = float((other["pharmacy_id"] == pharmacy).mean())
        if hc_share - other_share < gap:
            continue
        hc_products = sorted(set(hc.loc[hc["pharmacy_id"] == pharmacy, "billed_product"]))
        if hc_products and all(stocked.get(p, min_avail) < min_avail for p in hc_products):
            continue  # declared exclusion: the high-cost products are only stocked by a few pharmacies
        steered = hc[hc["pharmacy_id"] == pharmacy]
        amount = float(steered["billed_amount"].sum())
        last = g["fill_date"].max()
        out.append(_sig(
            ctx, control, subject_type="clinician", subject_id=pid,
            fact_key=f"hcsteer:{pid}:{pharmacy}",
            claim_ids=sorted({c for c in steered["claim_sk"] if c})[:50],
            event_time=last, period=period_bucket(last), confidence=0.7,
            evidence={
                "plain_language": (
                    f"{pct(hc_share)} of this prescriber's {plain_number(len(hc))} high-cost prescriptions "
                    f"went to one pharmacy, against {pct(other_share)} of their "
                    f"{plain_number(len(other))} other prescriptions ({aed(amount)} of high-cost drugs)."
                ),
                "what_the_reviewer_must_verify": (
                    "Check whether the high-cost drugs are only stocked by a few pharmacies, and compare "
                    "with similar prescribers."
                ),
                "clinician_id": pid, "provider_sk": facility.get(pid), "pharmacy_id": pharmacy,
                "high_cost_prescriptions": int(len(hc)), "other_prescriptions": int(len(other)),
                "high_cost_share_to_pharmacy": _r(hc_share, 3),
                "other_share_to_pharmacy": _r(other_share, 3), "share_gap_threshold": gap,
                "high_cost_products": hc_products[:10], "high_cost_amount_aed": _r(amount),
            },
            exposure=_unestablished(amount, "high-cost prescriptions routed to one pharmacy."),
        ))
    return out


@_safe
def phr_04_r04_rapid_relationship_formation(ctx, control) -> list[Signal]:
    """PHR-04-R04 — a new prescriber→pharmacy link that quickly takes most of the prescriber's volume."""
    rx, drugs, facility, facts, providers = _edges(ctx)
    if rx.empty:
        return []
    ramp = int(ctx.cfg("phr04_ramp_months"))
    prior_months = int(ctx.cfg("phr04_new_edge_min_prior_months"))
    dominant = float(ctx.cfg("phr04_dominant_share"))
    min_rx = int(ctx.cfg("phr04_ramp_min_rx"))
    rx = rx[rx["fill_date"].notna()].copy()
    rx["month"] = rx["fill_date"].dt.to_period("M")
    pharmacy_first = rx.groupby("pharmacy_id")["month"].min().to_dict()
    contracts = _table(ctx, "contract")
    contract_start: dict[str, list[pd.Period]] = {}
    if not contracts.empty and "provider_sk" in contracts.columns:
        for p, d in zip(_str(contracts["provider_sk"]), _date(_col(contracts, "valid_from"))):
            if p and not pd.isna(d):
                contract_start.setdefault(p, []).append(d.to_period("M"))
    out: list[Signal] = []
    for pid, g in rx.groupby("prescriber_id", sort=False):
        months = sorted(g["month"].unique())
        if len(months) <= prior_months:
            continue
        first_edge = g.groupby("pharmacy_id")["month"].min()
        for pharmacy, start in first_edge.items():
            prior = [m for m in months if m < start]
            if len(prior) < prior_months:
                continue  # the link is only "new" against an established prescribing history
            end = start + (ramp - 1)
            win = g[(g["month"] >= start) & (g["month"] <= end)]
            n_edge = int((win["pharmacy_id"] == pharmacy).sum())
            share = n_edge / len(win) if len(win) else 0.0
            if n_edge < min_rx or share < dominant:
                continue
            ph_first = pharmacy_first.get(pharmacy)
            if ph_first is not None and ph_first >= start - prior_months:
                continue  # declared exclusion: the pharmacy itself is a new site
            if any(start - ramp <= c <= end for c in contract_start.get(pharmacy, [])):
                continue  # declared exclusion: a contract launch explains the new link
            before = g[g["month"] < start]
            prev_top = before["pharmacy_id"].value_counts()
            edge = win[win["pharmacy_id"] == pharmacy]
            last = edge["fill_date"].max()
            out.append(_sig(
                ctx, control, subject_type="clinician", subject_id=pid,
                fact_key=f"newedge:{pid}:{pharmacy}",
                claim_ids=sorted({c for c in edge["claim_sk"] if c})[:50],
                event_time=last, period=period_bucket(last), confidence=0.6,
                evidence={
                    "plain_language": (
                        f"This prescriber sent no prescriptions to this pharmacy in the "
                        f"{count_phrase(len(prior), 'month')} before {start.strftime('%B %Y')}; within "
                        f"{count_phrase(ramp, 'month')} it took {pct(share)} of their prescriptions "
                        f"({plain_number(n_edge)} of {plain_number(len(win))})."
                    ),
                    "what_the_reviewer_must_verify": (
                        "Check whether a new site or contract launched then, and look for business links."
                    ),
                    "clinician_id": pid, "provider_sk": facility.get(pid), "pharmacy_id": pharmacy,
                    "link_first_month": str(start), "ramp_months": ramp,
                    "prescriptions_in_ramp": int(len(win)), "prescriptions_to_new_pharmacy": n_edge,
                    "share_in_ramp": _r(share, 3), "dominant_share_threshold": dominant,
                    "prior_active_months": int(len(prior)),
                    "previous_top_pharmacy": None if prev_top.empty else prev_top.index[0],
                },
                exposure=_exp.no_exposure("a newly formed relationship is monitored, not priced"),
            ))
    return out


# ===========================================================================
# PHR-05  devices, implants and supplies
# ===========================================================================


def _device_lines(ctx) -> pd.DataFrame:
    lines = _table(ctx, "claim_line")
    if lines.empty:
        return pd.DataFrame()
    for c in ("line_sk", "claim_sk", "activity_code", "activity_type", "device_serial", "product",
              "activity_description", "authorization_id", "indicator"):
        lines[c] = _str(_col(lines, c))
    for c in ("units", "net_amount", "gross_amount", "unit_price"):
        lines[c] = _num(_col(lines, c))
    lines["service_date"] = _date(_col(lines, "service_date"))
    claims = ctx.claims if ctx.claims is not None else pd.DataFrame()
    if not claims.empty and "claim_sk" in claims.columns:
        if "tenant_id" in claims.columns:
            claims = claims[claims["tenant_id"].astype(str) == str(ctx.tenant_id)]
        head = claims[[c for c in ("claim_sk", "member_sk", "provider_sk", "service_date")
                       if c in claims.columns]].drop_duplicates("claim_sk").copy()
        head["claim_sk"] = head["claim_sk"].astype(str)
        head = head.rename(columns={"service_date": "_claim_date"})
        lines = lines.merge(head, on="claim_sk", how="left")
        if "_claim_date" in lines.columns:
            lines["service_date"] = lines["service_date"].where(lines["service_date"].notna(),
                                                                _date(lines["_claim_date"]))
    for c in ("member_sk", "provider_sk"):
        if c not in lines.columns:
            lines[c] = None
    return lines


def _activity_ref(ctx) -> pd.DataFrame:
    ref = _table(ctx, "activity_code_reference")
    if ref.empty or "activity_code" not in ref.columns:
        return pd.DataFrame().rename_axis("activity_code")
    ref["activity_code"] = _str(ref["activity_code"])
    ref = ref.drop_duplicates("activity_code").set_index("activity_code")
    ref["is_implant"] = _col(ref, "is_implant").map(_truthy)
    for c in ("description", "activity_type", "service_family", "code_family"):
        ref[c] = _str(_col(ref, c))
    return ref


def _inventory(ctx) -> pd.DataFrame:
    inv = _table(ctx, "device_inventory")
    if inv.empty or "serial_number" not in inv.columns:
        return pd.DataFrame()
    for c in ("serial_number", "device_code", "provider_sk", "condition", "acquisition",
              "claim_sk", "line_sk", "member_sk"):
        inv[c] = _str(_col(inv, c))
    inv["useful_life_days"] = _num(_col(inv, "useful_life_days"))
    inv["issued_date"] = _date(_col(inv, "issued_date"))
    return inv


_WORDS_NEW = ("new",)
_WORDS_USED = ("used", "refurb", "pre-owned", "second")
_WORDS_RENT = ("rent", "rental", "lease", "hire")
_WORDS_BUY = ("purchase", "buy", "sale", "sold", "outright")


def _billed_terms(text: Any) -> tuple[str | None, str | None]:
    """What a line's description/indicator says: (condition, payment method)."""
    if is_missing(text):
        return None, None
    t = str(text).lower()
    cond = "used" if any(w in t for w in _WORDS_USED) else ("new" if re.search(r"\bnew\b", t) else None)
    pay = "rental" if any(w in t for w in _WORDS_RENT) else (
        "purchase" if any(w in t for w in _WORDS_BUY) else None)
    return cond, pay


def _inventory_terms(condition: Any, acquisition: Any) -> tuple[str | None, str | None, bool]:
    c = None if is_missing(condition) else str(condition).lower()
    a = None if is_missing(acquisition) else str(acquisition).lower()
    cond = None if c is None else ("used" if any(w in c for w in _WORDS_USED) else
                                   ("new" if "new" in c else None))
    rent_to_own = a is not None and ("own" in a)
    pay = None if a is None else ("rental" if any(w in a for w in _WORDS_RENT) and not rent_to_own else
                                  ("purchase" if any(w in a for w in _WORDS_BUY) else None))
    return cond, pay, rent_to_own


@_safe
def phr_05_r01_new_used_rental_mismatch(ctx, control) -> list[Signal]:
    """PHR-05-R01 — a device billed as new or as a purchase when its serial record says used or rental."""
    lines = _device_lines(ctx)
    inv = _inventory(ctx)
    if lines.empty or inv.empty:
        return []
    lines = lines[lines["device_serial"].notna()]
    if lines.empty:
        return []
    inv1 = inv.drop_duplicates("serial_number").set_index("serial_number")
    rental_factor = float(ctx.cfg("phr05_rental_price_fraction"))
    used_factor = float(ctx.cfg("phr05_used_price_fraction"))
    out: list[Signal] = []
    for row in lines.itertuples(index=False):
        if row.device_serial not in inv1.index:
            continue
        rec = inv1.loc[row.device_serial]
        text = " ".join(str(v) for v in (row.activity_description, row.indicator) if v)
        b_cond, b_pay = _billed_terms(text)
        i_cond, i_pay, rent_to_own = _inventory_terms(rec["condition"], rec["acquisition"])
        issues: list[str] = []
        billed = _f(row.net_amount, _f(row.gross_amount))
        correct = billed
        if b_cond == "new" and i_cond == "used":
            if "refurb" in str(rec["condition"]).lower() and "refurb" in text.lower():
                pass  # declared exclusion: refurbishment billed as such
            else:
                issues.append(f"billed as new, but the serial record says '{rec['condition']}'")
                correct = min(correct, billed * used_factor)
        if b_pay == "purchase" and i_pay == "rental" and not rent_to_own:
            issues.append(f"billed as a purchase, but the serial record says '{rec['acquisition']}'")
            correct = min(correct, billed * rental_factor)
        if not issues:
            continue
        out.append(_sig(
            ctx, control, subject_type="claim", subject_id=row.claim_sk,
            fact_key=f"devicecond:{row.line_sk}",
            claim_ids=[row.claim_sk], event_time=row.service_date, period=period_bucket(row.service_date),
            evidence={
                "plain_language": (
                    f"Device {row.activity_code} (serial {row.device_serial}, {aed(billed)}) was "
                    + "; and ".join(issues) + "."
                ),
                "what_the_reviewer_must_verify": (
                    "Compare the invoice with the supplier contract and the serial record, allowing for "
                    "refurbishment and rent-to-own terms."
                ),
                "claim_sk": row.claim_sk, "line_sk": row.line_sk, "member_sk": row.member_sk,
                "provider_sk": row.provider_sk, "activity_code": row.activity_code,
                "device_serial": row.device_serial, "billed_description": row.activity_description,
                "inventory_condition": rec["condition"], "inventory_acquisition": rec["acquisition"],
                "billed_amount_aed": _r(billed), "correct_payable_aed": _r(correct),
                "rental_price_fraction": rental_factor, "used_price_fraction": used_factor,
            },
            exposure=_exp.line_edit_exposure(billed, correct),
        ))
    return out


_OPERATIVE_WORDS = ("operative", "operation", "surgical", "implant", "procedure")


@_safe
def phr_05_r02_implant_not_linked(ctx, control) -> list[Signal]:
    """PHR-05-R02 — an implant billed on a claim with no qualifying procedure line."""
    lines = _device_lines(ctx)
    ref = _activity_ref(ctx)
    if lines.empty or ref.empty or not ref["is_implant"].any():
        return []
    lookback = pd.Timedelta(days=int(ctx.cfg("phr05_replacement_lookback_days")))
    implants = set(ref.index[ref["is_implant"]])
    lines["is_implant"] = lines["activity_code"].isin(implants)
    ref_type = ref["activity_type"].str.lower() if "activity_type" in ref.columns else pd.Series(dtype=object)
    fam = (ref["service_family"].fillna("") + " " + ref["code_family"].fillna("")).str.lower()
    qualifying = set(ref.index[(~ref["is_implant"]) & (
        fam.str.contains("surg|procedur|operat|orthop|cardiac|interven", regex=True)
        | ref_type.fillna("").str.contains("procedur|surg", regex=True))])
    lines["is_procedure"] = lines["activity_code"].isin(qualifying)
    per_claim = lines.groupby("claim_sk")["is_procedure"].any()
    docs = _table(ctx, "document")
    operative: set[str] = set()
    if not docs.empty and "claim_sk" in docs.columns:
        kind = _str(_col(docs, "doc_type")).fillna("").str.lower()
        operative = set(_str(docs.loc[kind.map(lambda k: any(w in k for w in _OPERATIVE_WORDS)), "claim_sk"]).dropna())
    # a procedure for the same member on another claim that day links the implant too
    procs = lines[lines["is_procedure"] & lines["member_sk"].notna() & lines["service_date"].notna()]
    proc_days = set(zip(procs["member_sk"], procs["service_date"].dt.normalize()))
    proc_by_member = procs.groupby("member_sk")["service_date"].apply(list).to_dict()
    cand = lines[lines["is_implant"] & ~lines["claim_sk"].map(per_claim).fillna(False).astype(bool)]
    out: list[Signal] = []
    for row in cand.itertuples(index=False):
        if pd.isna(row.service_date):
            continue
        if (row.member_sk, row.service_date.normalize()) in proc_days:
            continue
        earlier = [d for d in proc_by_member.get(row.member_sk, []) if row.service_date - lookback <= d < row.service_date]
        if earlier:
            continue  # declared exclusion: replacement/staged implant after an earlier procedure
        billed = _f(row.net_amount, _f(row.gross_amount))
        has_note = row.claim_sk in operative
        desc = ref["description"].get(row.activity_code) if "description" in ref.columns else None
        out.append(_sig(
            ctx, control, subject_type="claim", subject_id=row.claim_sk,
            fact_key=f"implant:{row.line_sk}",
            claim_ids=[row.claim_sk], event_time=row.service_date, period=period_bucket(row.service_date),
            confidence=0.7 if has_note else 0.9,
            evidence={
                "plain_language": (
                    f"An implant ({desc or row.activity_code}, {aed(billed)}) is billed on "
                    f"{plain_date(row.service_date)} with no procedure on this claim or on any claim for the "
                    f"patient that day" + ("" if has_note else ", and no operation note is attached") + "."
                ),
                "what_the_reviewer_must_verify": (
                    "Ask for the operation note and implant record, and check the replacement and spare-device policy."
                ),
                "claim_sk": row.claim_sk, "line_sk": row.line_sk, "member_sk": row.member_sk,
                "provider_sk": row.provider_sk, "activity_code": row.activity_code,
                "device_serial": row.device_serial, "billed_amount_aed": _r(billed),
                "operative_record_attached": bool(has_note),
                "laterality_checked": False,
                "replacement_lookback_days": int(lookback.days),
            },
            exposure=_unestablished(billed, "the implant may be payable once the procedure is documented."),
        ))
    return out


@_safe
def phr_05_r03_supply_quantity_excess(ctx, control) -> list[Signal]:
    """PHR-05-R03 — supply units far above the norm for the code, or a device replaced too soon."""
    lines = _device_lines(ctx)
    inv = _inventory(ctx)
    if lines.empty:
        return []
    multiple = float(ctx.cfg("phr05_supply_units_peer_multiple"))
    min_peer = int(ctx.cfg("phr05_supply_min_peer_lines"))
    life_fraction = float(ctx.cfg("phr05_useful_life_fraction"))
    adult_age = float(ctx.cfg("phr05_growth_exclusion_age"))
    out: list[Signal] = []
    supply_codes: set[str] = set()
    if not inv.empty:
        supply_codes |= set(inv["device_code"].dropna())
    ref = _activity_ref(ctx)
    if not ref.empty:
        text = (ref["service_family"].fillna("") + " " + ref["code_family"].fillna("") + " "
                + ref["activity_type"].fillna("")).str.lower()
        supply_codes |= set(ref.index[text.str.contains("supply|supplies|device|consumable|dme", regex=True)
                                      | ref["is_implant"]])
    sup = lines[lines["activity_code"].isin(supply_codes) & lines["units"].notna() & (lines["units"] > 0)].copy()

    # limb 1: units far above the code's norm
    if not sup.empty:
        stats = sup.groupby("activity_code")["units"].agg(["median", "count"])
        sup["norm"] = sup["activity_code"].map(stats["median"])
        sup["peer_n"] = sup["activity_code"].map(stats["count"])
        over = sup[(sup["peer_n"] >= min_peer) & (sup["units"] >= sup["norm"] * multiple)
                   & (sup["units"] > sup["norm"])]
        for row in over.itertuples(index=False):
            billed = _f(row.net_amount, _f(row.gross_amount))
            excess_share = 1.0 - float(row.norm) / float(row.units)
            out.append(_sig(
                ctx, control, subject_type="claim", subject_id=row.claim_sk,
                fact_key=f"supplyqty:{row.line_sk}",
                claim_ids=[row.claim_sk], event_time=row.service_date, period=period_bucket(row.service_date),
                confidence=0.7, peer_level_used="activity_code",
                evidence={
                    "plain_language": (
                        f"{plain_number(row.units)} units of supply {row.activity_code} billed ({aed(billed)}); "
                        f"the usual quantity for this item is {plain_number(row.norm)} "
                        f"({times_phrase(row.units, row.norm, 'the usual quantity')})."
                    ),
                    "what_the_reviewer_must_verify": (
                        "Compare the billed units with the procedure's usual supply list and the clinical notes."
                    ),
                    "claim_sk": row.claim_sk, "line_sk": row.line_sk, "member_sk": row.member_sk,
                    "provider_sk": row.provider_sk, "activity_code": row.activity_code,
                    "units": _r(row.units), "norm_units": _r(row.norm), "peer_lines": int(row.peer_n),
                    "peer_multiple": multiple, "billed_amount_aed": _r(billed), "limb": "quantity",
                },
                exposure=_unestablished(billed * excess_share, "units above the item's usual quantity."),
            ))

    # limb 2: the same device re-issued to a member before its useful life
    if not inv.empty:
        members = _member_frame(ctx)
        issued = inv[inv["member_sk"].notna() & inv["issued_date"].notna() & inv["device_code"].notna()]
        issued = issued.sort_values(["member_sk", "device_code", "issued_date", "serial_number"])
        prev_date = issued.groupby(["member_sk", "device_code"])["issued_date"].shift(1)
        prev_serial = issued.groupby(["member_sk", "device_code"])["serial_number"].shift(1)
        prev_life = issued.groupby(["member_sk", "device_code"])["useful_life_days"].shift(1)
        issued = issued.assign(prev_date=prev_date, prev_serial=prev_serial, prev_life=prev_life)
        issued["gap"] = (issued["issued_date"] - issued["prev_date"]).dt.days
        early = issued[issued["prev_date"].notna() & issued["prev_life"].notna()
                       & (issued["gap"] < issued["prev_life"] * life_fraction)]
        amount_by_line = lines.set_index("line_sk")["net_amount"].to_dict() if "line_sk" in lines.columns else {}
        for row in early.itertuples(index=False):
            dob = members["date_of_birth"].get(row.member_sk) if "date_of_birth" in members.columns else None
            if dob is not None and not pd.isna(dob):
                age = (row.issued_date - dob).days / 365.25
                if age < adult_age:
                    continue  # declared exclusion: member growth
            billed = _f(amount_by_line.get(row.line_sk))
            out.append(_sig(
                ctx, control, subject_type="member", subject_id=row.member_sk,
                fact_key=f"supplylife:{row.serial_number}",
                claim_ids=[c for c in [row.claim_sk] if c], event_time=row.issued_date,
                period=period_bucket(row.issued_date), confidence=0.7,
                evidence={
                    "plain_language": (
                        f"A replacement {row.device_code} was issued on {plain_date(row.issued_date)}, "
                        f"{plain_number(row.gap)} days after the previous one; its expected useful life is "
                        f"{plain_number(row.prev_life)} days."
                    ),
                    "what_the_reviewer_must_verify": (
                        "Check for damage, loss or a change in clinical condition that justified early replacement."
                    ),
                    "member_sk": row.member_sk, "claim_sk": row.claim_sk, "line_sk": row.line_sk,
                    "device_code": row.device_code, "serial_number": row.serial_number,
                    "previous_serial": row.prev_serial, "previous_issued": _iso(row.prev_date),
                    "issued_date": _iso(row.issued_date), "days_between": _r(row.gap, 0),
                    "useful_life_days": _r(row.prev_life, 0), "useful_life_fraction": life_fraction,
                    "billed_amount_aed": _r(billed), "limb": "useful life",
                },
                exposure=_unestablished(billed, "a device replaced before its useful life."),
            ))
    return out


@_safe
def phr_05_r04_serial_duplication(ctx, control) -> list[Signal]:
    """PHR-05-R04 — one device serial billed for more than one member (or more often than it exists)."""
    lines = _device_lines(ctx)
    if lines.empty:
        return []
    inv = _inventory(ctx)
    lines = lines[lines["device_serial"].notna() & lines["member_sk"].notna()]
    if lines.empty:
        return []
    # declared exclusion: bulk / non-serialised consumables share lot numbers
    if not inv.empty:
        known = set(inv["serial_number"].dropna())
        lines = lines[lines["device_serial"].isin(known)]
    bulk = lines["device_serial"].str.lower().str.contains("lot|batch|bulk", regex=True)
    lines = lines[~bulk]
    out: list[Signal] = []
    for serial, g in lines.groupby("device_serial", sort=False):
        if g["member_sk"].nunique() < 2:
            continue
        g = g.sort_values(["service_date", "line_sk"])
        first = g.iloc[0]
        later = g[g["member_sk"] != first["member_sk"]]
        amounts = [float(_f(v, _f(w))) for v, w in zip(g["net_amount"], g["gross_amount"])]
        claims = sorted(set(g["claim_sk"]))
        out.append(_sig(
            ctx, control, subject_type="provider", subject_id=first["provider_sk"] or serial,
            fact_key=f"serial:{serial}",
            claim_ids=claims, event_time=later["service_date"].min(),
            period=period_bucket(later["service_date"].min()),
            evidence={
                "plain_language": (
                    f"Device serial {serial} ({first['activity_code']}) is billed for "
                    f"{g['member_sk'].nunique()} different patients on {count_phrase(len(claims), 'claim')} "
                    f"({plain_date(g['service_date'].min())} to {plain_date(g['service_date'].max())}); one "
                    f"serialised device can only be supplied once."
                ),
                "what_the_reviewer_must_verify": (
                    "Confirm the serial against the supplier invoice and implant register."
                ),
                "device_serial": serial, "activity_code": first["activity_code"],
                "provider_sk": first["provider_sk"], "member_sks": sorted(set(g["member_sk"])),
                "member_sk": first["member_sk"], "line_sks": list(g["line_sk"]),
                "billed_amounts_aed": [_r(a) for a in amounts],
            },
            exposure=_exp.duplicate_exposure(amounts, count_note="Serial billed for several patients."),
        ))
    return out


IMPLEMENTATIONS: dict[str, Callable] = {
    "phr_01_r01_product_identity_mismatch": phr_01_r01_product_identity_mismatch,
    "phr_01_r02_quantity_strength_mismatch": phr_01_r02_quantity_strength_mismatch,
    "phr_01_r03_dispense_claim_mismatch": phr_01_r03_dispense_claim_mismatch,
    "phr_01_r04_non_medical_substitution": phr_01_r04_non_medical_substitution,
    "phr_02_r01_refill_overlap": phr_02_r01_refill_overlap,
    "phr_02_r02_therapy_duration_excess": phr_02_r02_therapy_duration_excess,
    "phr_02_r03_multi_prescriber": phr_02_r03_multi_prescriber,
    "phr_02_r04_multi_pharmacy": phr_02_r04_multi_pharmacy,
    "phr_03_r01_dose_weight_conflict": phr_03_r01_dose_weight_conflict,
    "phr_03_r02_drug_diagnosis_step_conflict": phr_03_r02_drug_diagnosis_step_conflict,
    "phr_03_r04_wastage_anomaly": phr_03_r04_wastage_anomaly,
    "phr_04_r01_top_pharmacy_concentration": phr_04_r01_top_pharmacy_concentration,
    "phr_04_r02_reciprocal_value_loop": phr_04_r02_reciprocal_value_loop,
    "phr_04_r03_high_cost_steering": phr_04_r03_high_cost_steering,
    "phr_04_r04_rapid_relationship_formation": phr_04_r04_rapid_relationship_formation,
    "phr_05_r01_new_used_rental_mismatch": phr_05_r01_new_used_rental_mismatch,
    "phr_05_r02_implant_not_linked": phr_05_r02_implant_not_linked,
    "phr_05_r03_supply_quantity_excess": phr_05_r03_supply_quantity_excess,
    "phr_05_r04_serial_duplication": phr_05_r04_serial_duplication,
}
