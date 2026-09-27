"""Unlocked implementations for the PAY family (PAY-07 to PAY-12).

Each function here runs only when a dataset unlock (`rules/unlocks/PAY_B.yaml`)
is satisfied by the loaded file. See :mod:`fwa.engine.unlocks`.

Scope: patient share and balance billing (PAY-07), denial-resubmission
mutation (PAY-08), disguised non-covered services (PAY-09), coordination of
benefits and third-party liability (PAY-10), internal adjudication and
override (PAY-11), and reversal / refund / remittance leakage (PAY-12).

Conventions shared by every function below:

* thresholds come from ``ctx.cfg`` (keys prefixed ``pay07_`` … ``pay12_``);
* every table read is tenant-scoped and every column read is optional — a
  missing table, column or value returns ``[]`` rather than raising;
* a statistical pattern (type S / N) never claims established exposure; an
  objective payment fact (a second payment, a payment after a cancellation)
  does, and says how it was computed.
"""

from __future__ import annotations

import functools
import json
import re
from typing import Any, Callable

import numpy as np
import pandas as pd

from ...cases import exposure as _exp
from ...presentation import aed, count_phrase, pct, plain_date, times_phrase
from ...statistical.shrinkage import fit_beta_prior, shrink_rate
from ..controls import _claim_frame, _f, _sig
from ..evallib import is_missing, period_bucket

__all__ = ["IMPLEMENTATIONS", "LAST_ERRORS"]

IMPLEMENTATIONS: dict[str, Callable] = {}

#: The last exception each implementation swallowed (for tests and debugging).
#: A control must never raise into the evaluator; it records here instead.
LAST_ERRORS: dict[str, str] = {}

_DENIED = {"DENIED", "REJECTED", "DENY", "REJECT", "DECLINED", "D"}
_RESUB = {"RESUBMISSION", "CORRECTION", "CANCELLATION", "REPLACEMENT", "VOID", "REVERSAL"}
_CANCEL = {"CANCELLATION", "CANCELLED", "CANCEL", "VOID", "REVERSAL", "REVERSED"}
_AUTO_ACTORS = {"SYSTEM", "AUTO", "AUTO_ADJUDICATION", "AUTOMATED", "ENGINE", "RULES_ENGINE"}


def _register(name: str):
    def deco(fn):
        @functools.wraps(fn)
        def wrapper(ctx, control):
            try:
                out = fn(ctx, control) or []
                LAST_ERRORS.pop(name, None)
                return out
            except Exception as exc:  # never escape into the evaluator
                import traceback

                LAST_ERRORS[name] = f"{type(exc).__name__}: {exc}\n{traceback.format_exc(limit=6)}"
                return []

        IMPLEMENTATIONS[name] = wrapper
        return wrapper

    return deco


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _tbl(ctx, name: str) -> pd.DataFrame:
    """A tenant-scoped copy of one table (empty frame when absent)."""
    ds = ctx.dataset
    if ds is None:
        return pd.DataFrame()
    try:
        frame = ds.get(name)
    except KeyError:
        return pd.DataFrame()
    if frame is None or frame.empty:
        return pd.DataFrame(columns=list(frame.columns) if frame is not None else [])
    if "tenant_id" in frame.columns and frame["tenant_id"].notna().any():
        t = frame["tenant_id"]
        frame = frame[t.isna() | (t.astype(str) == str(ctx.tenant_id))]
    return frame.copy()


def _rows(frame: pd.DataFrame):
    """Rows as attribute objects. (``itertuples`` renames underscore-prefixed columns.)"""
    from types import SimpleNamespace

    for rec in frame.to_dict("records"):
        yield SimpleNamespace(**{str(k): v for k, v in rec.items()})


def _col(frame: pd.DataFrame, name: str, default: Any = np.nan) -> pd.Series:
    if name in frame.columns:
        return frame[name]
    return pd.Series(default, index=frame.index, dtype=object)


def _up(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.strip().str.upper().replace({"NAN": "", "NONE": ""})


def _num(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _dt(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce")


def _s(value: Any) -> str:
    return "" if is_missing(value) else str(value)


def _date_str(value: Any) -> str | None:
    try:
        ts = pd.Timestamp(value)
        return None if pd.isna(ts) else str(ts.date())
    except Exception:
        return None


def _claims(ctx) -> pd.DataFrame:
    df = _claim_frame(ctx)
    if df is None or df.empty or "claim_sk" not in df.columns:
        return pd.DataFrame()
    df = df.copy()
    df["claim_sk"] = df["claim_sk"].astype(str)
    if "service_date" not in df.columns:
        df["service_date"] = pd.NaT
    df["_sdate"] = _dt(df["service_date"])
    for c in ("member_sk", "provider_sk", "payer_id"):
        if c not in df.columns:
            df[c] = None
    return df


def _as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [v.strip() for v in re.split(r"[;,|]", value) if v.strip()]
    return [str(v).strip() for v in value if str(v).strip()]


def _contains_any(text: pd.Series, terms: list[str]) -> pd.Series:
    terms = [t.lower() for t in terms if t]
    if not terms:
        return pd.Series(False, index=text.index)
    pattern = "|".join(re.escape(t) for t in terms)
    return text.fillna("").astype(str).str.lower().str.contains(pattern, regex=True)


def _remittance(ctx) -> pd.DataFrame:
    r = _tbl(ctx, "remittance")
    if r.empty or "claim_sk" not in r.columns:
        return pd.DataFrame()
    r["claim_sk"] = r["claim_sk"].astype(str)
    r["_decision"] = _up(_col(r, "decision"))
    r["_pay"] = _num(_col(r, "payment_amount")).fillna(0.0)
    r["_adj"] = _num(_col(r, "adjustment")).fillna(0.0)
    r["_denial"] = _up(_col(r, "denial_code"))
    r["_settle"] = _dt(_col(r, "settlement_date"))
    r["_ref"] = _up(_col(r, "payment_reference"))
    r["_line"] = _col(r, "line_sk").astype(object).where(_col(r, "line_sk").notna(), None)
    r["_denied"] = r["_decision"].isin(_DENIED) | ((r["_denial"] != "") & (r["_pay"] <= 0))
    return r


def _claim_payment_status(remit: pd.DataFrame) -> pd.DataFrame:
    """Per claim: total paid, whether any line was denied, denial codes."""
    if remit.empty:
        return pd.DataFrame(columns=["claim_sk", "paid", "any_denied", "all_denied", "denial_codes"])
    g = remit.groupby("claim_sk")
    out = pd.DataFrame({
        "paid": g["_pay"].sum(),
        "any_denied": g["_denied"].any(),
        "all_denied": g["_denied"].all(),
        "denial_codes": g["_denial"].agg(lambda s: sorted({x for x in s if x})),
    }).reset_index()
    return out


def _data_end(ctx, *frames_cols: tuple[pd.DataFrame, str]) -> pd.Timestamp | None:
    """The latest date the file knows about — the as-of for SLA and timing checks."""
    candidates = []
    claims = _claims(ctx)
    if not claims.empty:
        candidates.append(claims["_sdate"].max())
    for frame, col in frames_cols:
        if frame is not None and not frame.empty and col in frame.columns:
            candidates.append(_dt(frame[col]).max())
    candidates = [c for c in candidates if not pd.isna(c)]
    return max(candidates) if candidates else None


def _line_benefits(ctx) -> pd.DataFrame:
    """Every tenant claim line with the member's product and the benefit rule in force.

    Columns added: ``product``, ``service_family``, ``covered``, ``share_pct``
    (a fraction), ``exceptions``, ``expected_share`` (line gross × share %).
    Empty when any of the reference tables it needs is absent.
    """
    claims = _claims(ctx)
    lines = _tbl(ctx, "claim_line")
    acr = _tbl(ctx, "activity_code_reference")
    cov = _tbl(ctx, "coverage_period")
    brv = _tbl(ctx, "benefit_rule_version")
    if claims.empty or lines.empty or acr.empty or cov.empty or brv.empty:
        return pd.DataFrame()
    for frame, cols in ((lines, ("claim_sk", "activity_code")), (acr, ("activity_code", "service_family")),
                        (cov, ("member_sk", "product")), (brv, ("product", "service_family"))):
        if any(c not in frame.columns for c in cols):
            return pd.DataFrame()
    lines = lines.copy()
    lines["claim_sk"] = lines["claim_sk"].astype(str)
    if "line_sk" not in lines.columns:
        lines["line_sk"] = lines["claim_sk"] + ":" + lines.groupby("claim_sk").cumcount().astype(str)
    lines["line_sk"] = lines["line_sk"].astype(str)
    lines["_gross"] = _num(_col(lines, "gross_amount")).fillna(0.0)
    lines["_lshare"] = _num(_col(lines, "patient_share"))
    lines["_ldate"] = _dt(_col(lines, "service_date"))
    lines = lines.merge(claims[["claim_sk", "member_sk", "provider_sk", "_sdate"]], on="claim_sk", how="inner")
    lines["_date"] = lines["_ldate"].fillna(lines["_sdate"])
    lines = lines.drop(columns=[c for c in ("service_family", "product", "covered", "exceptions") if c in lines.columns])
    lines = lines.merge(acr[["activity_code", "service_family"]].drop_duplicates("activity_code"),
                        on="activity_code", how="left")
    # benefit family: dispensed products are PHARMACY, unknown codes PROCEDURE, and every line of an
    # inpatient claim takes the family of the claim's case-rate (DRG) line, or of its first line
    atype = _up(_col(lines, "activity_type"))
    lines["service_family"] = lines["service_family"].where(
        lines["service_family"].notna(), np.where(atype == "DRUG", "PHARMACY", "PROCEDURE"))
    if "claim_type" in claims.columns:
        ip = set(claims.loc[_up(claims["claim_type"]) == "INPATIENT", "claim_sk"])
        if ip:
            ipl = lines[lines["claim_sk"].isin(ip)].sort_values("line_sk")
            drg = ipl[_up(_col(ipl, "activity_type")) == "DRG"].groupby("claim_sk")["service_family"].first()
            first = ipl.groupby("claim_sk")["service_family"].first()
            fam = first.copy()
            fam.update(drg)
            m = lines["claim_sk"].isin(ip)
            lines.loc[m, "service_family"] = lines.loc[m, "claim_sk"].map(fam)

    cv = cov[["member_sk", "product"]].copy()
    cv["_vf"] = _dt(_col(cov, "valid_from"))
    cv["_vt"] = _dt(_col(cov, "valid_to"))
    m = lines[["line_sk", "member_sk", "_date"]].merge(cv, on="member_sk", how="inner")
    ok = (m["_vf"].isna() | (m["_vf"] <= m["_date"])) & (m["_vt"].isna() | (m["_date"] <= m["_vt"]))
    m = m[ok].sort_values("_vf", ascending=False).drop_duplicates("line_sk")
    lines = lines.merge(m[["line_sk", "product"]], on="line_sk", how="left")

    b = brv[["product", "service_family"]].copy()
    b["covered"] = _col(brv, "covered")
    b["_pct"] = _num(_col(brv, "patient_share_pct"))
    b["exceptions"] = _col(brv, "exceptions").astype(object)
    b["_bvf"] = _dt(_col(brv, "valid_from"))
    b["_bvt"] = _dt(_col(brv, "valid_to"))
    mb = lines[["line_sk", "product", "service_family", "_date"]].merge(b, on=["product", "service_family"], how="inner")
    ok = (mb["_bvf"].isna() | (mb["_bvf"] <= mb["_date"])) & (mb["_bvt"].isna() | (mb["_date"] <= mb["_bvt"]))
    mb = mb[ok].sort_values("_bvf", ascending=False).drop_duplicates("line_sk")
    lines = lines.merge(mb[["line_sk", "covered", "_pct", "exceptions"]], on="line_sk", how="left")
    pct_raw = lines["_pct"]
    lines["share_pct"] = np.where(pct_raw > 1.0, pct_raw / 100.0, pct_raw)
    lines["expected_share"] = (lines["_gross"] * lines["share_pct"]).round(2)
    return lines


def _claim_expected_share(ctx) -> pd.DataFrame:
    """Per claim: submitted share, expected share, and whether every line had a benefit rule."""
    lb = _line_benefits(ctx)
    claims = _claims(ctx)
    if lb.empty or claims.empty:
        return pd.DataFrame()
    exempt_terms = _as_list(ctx.cfg("pay07_share_exemption_terms"))
    lb["_exempt_line"] = _contains_any(lb["exceptions"].astype(str), exempt_terms)
    g = lb.groupby("claim_sk")
    agg = pd.DataFrame({
        "expected_share": g["expected_share"].sum(min_count=1),
        "lines": g["line_sk"].count(),
        "mapped": g["share_pct"].count(),
        "line_gross": g["_gross"].sum(),
        "product": g["product"].first(),
        "exempt": g["_exempt_line"].any(),
        "share_pct": g["share_pct"].max(),
    }).reset_index()
    agg = agg[agg["mapped"] == agg["lines"]]
    cols = ["claim_sk", "member_sk", "provider_sk", "payer_id", "_sdate"]
    extra = [c for c in ("patient_share", "gross_amount", "net_amount", "discount") if c in claims.columns]
    out = agg.merge(claims[cols + extra], on="claim_sk", how="inner")
    for c in ("patient_share", "gross_amount", "net_amount", "discount"):
        out[c] = _num(_col(out, c))
    return out


# ===========================================================================
# PAY-07 — patient share, copay, deductible and balance billing
# ===========================================================================


@_register("pay_07_r01_expected_share_mismatch")
def pay_07_r01_expected_share_mismatch(ctx, control) -> list:
    """PAY-07-R01 — submitted patient share differs from the benefit calculation."""
    ce = _claim_expected_share(ctx)
    if ce.empty:
        return []
    abs_tol = float(ctx.cfg("pay07_share_abs_tolerance_aed"))
    rel_tol = float(ctx.cfg("pay07_share_rel_tolerance"))
    ce = ce[ce["patient_share"].notna() & ~ce["exempt"]]
    gap = ce["patient_share"] - ce["expected_share"]
    tol = np.maximum(abs_tol, rel_tol * ce["expected_share"].abs())
    hits = ce[gap.abs() > tol].assign(_gap=gap[gap.abs() > tol])
    out = []
    for r in _rows(hits):
        under = r._gap < 0
        net = _f(r.net_amount, _f(r.gross_amount))
        direction = "lower" if under else "higher"
        plain = (
            f"The patient's share on this claim is {aed(r.patient_share, 2)}, but the plan's benefit terms "
            f"({pct(r.share_pct)} co-payment on {aed(r.line_gross, 2)} billed) work out to "
            f"{aed(r.expected_share, 2)} — {aed(abs(r._gap), 2)} {direction} than it should be."
        )
        out.append(_sig(
            ctx, control, subject_type="claim", subject_id=r.claim_sk,
            fact_key=f"share:{r.claim_sk}", claim_ids=[r.claim_sk],
            event_time=r._sdate, period=period_bucket(r._sdate),
            confidence=0.95 if under else 0.6,
            evidence={
                "plain_language": plain,
                "what_the_reviewer_must_verify": "Recalculate the share from the benefit terms in force, "
                                                 "allowing for deductible progress and any exemption.",
                "member_sk": r.member_sk, "provider_sk": r.provider_sk, "product": r.product,
                "submitted_patient_share_aed": round(float(r.patient_share), 2),
                "expected_patient_share_aed": round(float(r.expected_share), 2),
                "benefit_share_pct": None if is_missing(r.share_pct) else round(float(r.share_pct), 4),
                "billed_line_total_aed": round(float(r.line_gross), 2),
                "difference_aed": round(float(r._gap), 2),
                "tolerance_aed": round(float(max(abs_tol, rel_tol * abs(r.expected_share))), 2),
                "direction": "share_understated" if under else "share_overstated",
                "exclusion_note": "Deductible accumulators are not in the file, so a share ABOVE the "
                                  "calculation is raised at reduced confidence (it may be a deductible).",
            },
            exposure=(_exp.line_edit_exposure(net, net - abs(float(r._gap))) if under else
                      _exp.no_exposure("the share is higher than the calculation, so the insurer is not "
                                       "overpaying; the member may be")),
        ))
    return out


def _discount_already_in_gross(ctx, claims: pd.DataFrame) -> bool:
    """Which convention the file uses for the header discount.

    Some sources record ``gross_amount`` BEFORE the provider discount (so the
    catalogue equation is gross − discount − share = net); others record the
    contracted, already-discounted price as gross and carry ``discount`` only for
    information (then gross − share = net). The convention is read from the
    file itself — whichever equation holds on more of the discounted claims.
    """
    if claims.empty or "discount" not in claims.columns:
        return False
    tol = float(ctx.cfg("pay07_net_equation_tolerance_aed"))
    g, n = _num(_col(claims, "gross_amount")), _num(_col(claims, "net_amount"))
    ps, d = _num(_col(claims, "patient_share")).fillna(0.0), _num(claims["discount"]).fillna(0.0)
    m = (d > tol) & g.notna() & n.notna()
    if not m.any():
        return False
    after = ((n - (g - ps)).abs() <= tol)[m].mean()
    before = ((n - (g - d - ps)).abs() <= tol)[m].mean()
    return bool(after > before)


@_register("pay_07_r02_share_shifted_to_payer")
def pay_07_r02_share_shifted_to_payer(ctx, control) -> list:
    """PAY-07-R02 — gross − valid discount − patient share ≠ net in the payer's disfavour,
    or a waived patient share included in the amount claimed from the insurer."""
    claims = _claims(ctx)
    if claims.empty or not {"gross_amount", "net_amount", "patient_share"} <= set(claims.columns):
        return []
    tol = float(ctx.cfg("pay07_net_equation_tolerance_aed"))
    in_gross = _discount_already_in_gross(ctx, claims)
    df = claims.copy()
    df["_g"] = _num(df["gross_amount"])
    df["_n"] = _num(df["net_amount"])
    df["_ps"] = _num(df["patient_share"]).fillna(0.0)
    df["_disc"] = _num(_col(df, "discount")).fillna(0.0)
    df = df[df["_g"].notna() & df["_n"].notna()]
    df["_expected_net"] = df["_g"] - df["_ps"] - (0.0 if in_gross else df["_disc"])
    df["_diff"] = df["_n"] - df["_expected_net"]
    df["_limb"] = np.where(df["_diff"] > tol, "net_exceeds_equation", "")
    # limb 2: the share was waived (recorded as zero) and the whole bill is claimed from the insurer
    ce = _claim_expected_share(ctx)
    if not ce.empty:
        exp_share = ce.set_index("claim_sk")["expected_share"]
        exempt = ce.set_index("claim_sk")["exempt"]
        df["_exp_share"] = df["claim_sk"].map(exp_share)
        waived = ((df["_ps"] <= tol) & (df["_exp_share"] > tol) & ~df["claim_sk"].map(exempt).fillna(False).astype(bool)
                  & (df["_n"] >= df["_g"] - (0.0 if in_gross else df["_disc"]) - tol))
        df.loc[waived & (df["_limb"] == ""), "_limb"] = "waived_share_in_net"
        df.loc[waived, "_diff"] = np.maximum(df.loc[waived, "_diff"], df.loc[waived, "_exp_share"])
    else:
        df["_exp_share"] = np.nan
    hits = df[df["_limb"] != ""]
    out = []
    for r in _rows(hits):
        disc_words = (f"(already net of a discount of {aed(r._disc, 2)})" if in_gross
                      else f"less discount {aed(r._disc, 2)}")
        if r._limb == "waived_share_in_net":
            plain = (
                f"The patient's share is recorded as {aed(r._ps, 2)} although the plan's terms give "
                f"{aed(r._exp_share, 2)}, and the insurer is asked for the whole bill of {aed(r._n, 2)} "
                f"{disc_words}: the waived share has been shifted onto the insurer."
            )
            payable = float(r._n) - float(r._exp_share)
        else:
            plain = (
                f"Billed {aed(r._g, 2)} {disc_words}, less patient share {aed(r._ps, 2)}, should leave "
                f"{aed(r._expected_net, 2)} for the insurer, but the claim asks for {aed(r._n, 2)} — "
                f"{aed(r._diff, 2)} more."
            )
            payable = float(r._expected_net)
        out.append(_sig(
            ctx, control, subject_type="claim", subject_id=r.claim_sk,
            fact_key=f"net_eq:{r.claim_sk}", claim_ids=[r.claim_sk],
            event_time=r._sdate, period=period_bucket(r._sdate),
            evidence={
                "plain_language": plain,
                "what_the_reviewer_must_verify": "Check whether an approved discount or hardship waiver "
                                                 "explains the difference; otherwise reprice.",
                "member_sk": r.member_sk, "provider_sk": r.provider_sk,
                "gross_amount_aed": round(float(r._g), 2), "discount_aed": round(float(r._disc), 2),
                "discount_already_in_gross": in_gross,
                "patient_share_aed": round(float(r._ps), 2), "net_amount_aed": round(float(r._n), 2),
                "expected_net_aed": round(float(r._expected_net), 2),
                "benefit_patient_share_aed": None if is_missing(r._exp_share) else round(float(r._exp_share), 2),
                "excess_aed": round(float(r._diff), 2), "limb": r._limb, "tolerance_aed": tol,
            },
            exposure=_exp.line_edit_exposure(float(r._n), payable),
        ))
    return out


@_register("pay_07_r03_systematic_zero_share")
def pay_07_r03_systematic_zero_share(ctx, control) -> list:
    """PAY-07-R03 — provider zero/rounded-down share rate against product-adjusted peers."""
    ce = _claim_expected_share(ctx)
    if ce.empty:
        return []
    abs_tol = float(ctx.cfg("pay07_share_abs_tolerance_aed"))
    unit = float(ctx.cfg("pay07_rounding_unit_aed"))
    min_claims = int(ctx.cfg("pay07_zero_share_min_claims"))
    ratio_cut = float(ctx.cfg("pay07_zero_share_rate_ratio"))
    min_excess = float(ctx.cfg("pay07_zero_share_min_excess"))
    programmes = _as_list(ctx.cfg("pay07_zero_share_programmes"))
    mass = float(ctx.cfg("posterior_interval_mass"))
    width = float(ctx.cfg("max_posterior_width"))

    opp = ce[(ce["expected_share"] > abs_tol) & ~ce["exempt"] & ce["patient_share"].notna()].copy()
    if programmes:
        opp = opp[~_contains_any(opp["product"].astype(str), programmes)]
    if opp.empty:
        return []
    ps = opp["patient_share"]
    zero = ps <= abs_tol
    rounded = (ps < opp["expected_share"] - abs_tol) & np.isclose(np.mod(ps, unit), 0.0) & (ps > abs_tol) \
        if unit > 0 else pd.Series(False, index=opp.index)
    opp["_event"] = (zero | rounded).astype(float)
    opp["_zero"] = zero.astype(float)
    # product-adjusted expectation: each claim contributes its product's event rate among the
    # OTHER providers (leave-one-out, so a large provider does not set its own benchmark)
    opp["product"] = opp["product"].fillna("")
    tot = opp.groupby("product")["_event"].agg(["sum", "count"])
    own = opp.groupby(["provider_sk", "product"])["_event"].agg(["sum", "count"])
    key = pd.MultiIndex.from_arrays([opp["provider_sk"], opp["product"]])
    t_sum = opp["product"].map(tot["sum"]).to_numpy()
    t_cnt = opp["product"].map(tot["count"]).to_numpy()
    o_sum = own["sum"].reindex(key).to_numpy()
    o_cnt = own["count"].reindex(key).to_numpy()
    rest = t_cnt - o_cnt
    overall = float(opp["_event"].mean())
    opp["_exp_rate"] = np.where(rest > 0, (t_sum - o_sum) / np.where(rest > 0, rest, 1), overall)
    opp["_gapv"] = (opp["expected_share"] - opp["patient_share"]).clip(lower=0.0)
    g = opp.groupby("provider_sk")
    agg = pd.DataFrame({
        "n": g["_event"].size(), "events": g["_event"].sum(), "zeros": g["_zero"].sum(),
        "expected_events": g["_exp_rate"].sum(),
        "share_gap": g["_gapv"].sum(),
    }).reset_index()
    prior = fit_beta_prior(agg["events"], agg["n"])
    out = []
    for r in _rows(agg[agg["n"] >= min_claims]):
        sr = shrink_rate(str(r.provider_sk), r.events, r.n, prior, interval_mass=mass, max_posterior_width=width)
        peer_rate = float(r.expected_events) / float(r.n) if r.n else 0.0
        if sr.excluded_from_ranking or sr.shrunk_rate - peer_rate < min_excess:
            continue
        if peer_rate > 0 and sr.shrunk_rate < ratio_cut * peer_rate:
            continue
        sub = opp[(opp["provider_sk"] == r.provider_sk) & (opp["_event"] > 0)]
        observed = float(r.events) / float(r.n)
        plain = (
            f"{pct(observed)} of this provider's claims ({int(r.events)} of {int(r.n)}) record a zero or "
            f"rounded-down patient share where the plan expects one; for the same products across all "
            f"providers it is {pct(peer_rate)}."
        )
        out.append(_sig(
            ctx, control, subject_type="provider", subject_id=r.provider_sk,
            fact_key=f"zero_share:{r.provider_sk}", claim_ids=sub["claim_sk"].tolist(),
            event_time=sub["_sdate"].max(), period=period_bucket(sub["_sdate"].max()),
            confidence=0.85,
            evidence={
                "plain_language": plain,
                "what_the_reviewer_must_verify": "Check whether the products involved carry a zero-share "
                                                 "benefit or regulator programme, then sample the claims.",
                "provider_sk": r.provider_sk, "claims_with_expected_share": int(r.n),
                "zero_or_rounded_claims": int(r.events), "zero_share_claims": int(r.zeros),
                "observed_rate": round(observed, 4), "shrunk_rate": round(sr.shrunk_rate, 4),
                "posterior_interval": [round(sr.posterior_low, 4), round(sr.posterior_high, 4)],
                "product_adjusted_peer_rate": round(peer_rate, 4),
                "uncollected_share_aed": round(float(r.share_gap), 2),
                "rounding_unit_aed": unit, "shrinkage_explanation": sr.explain(),
            },
            exposure=_exp.provider_pattern_exposure(
                (sub["expected_share"] - sub["patient_share"]).clip(lower=0).tolist()),
        ))
    return out


@_register("pay_07_r04_balance_billing")
def pay_07_r04_balance_billing(ctx, control) -> list:
    """PAY-07-R04 — member receipts or complaints exceed the approved patient liability."""
    claims = _claims(ctx)
    receipts = _tbl(ctx, "member_receipt")
    if claims.empty or receipts.empty or "claim_sk" not in receipts.columns:
        return []
    tol = float(ctx.cfg("pay07_balance_bill_tolerance_aed"))
    consent_terms = _as_list(ctx.cfg("pay07_consented_item_terms"))
    complaint_terms = _as_list(ctx.cfg("pay07_overcharge_complaint_terms"))
    receipts["claim_sk"] = receipts["claim_sk"].astype(str)
    receipts["_amt"] = _num(_col(receipts, "amount_paid_aed")).fillna(0.0)
    desc = _col(receipts, "item_description").astype(str)
    receipts["_consented"] = _contains_any(desc, consent_terms)
    counted = receipts[~receipts["_consented"]]
    paid = counted.groupby("claim_sk")["_amt"].sum()
    excluded = receipts[receipts["_consented"]].groupby("claim_sk")["_amt"].sum()
    n_receipts = counted.groupby("claim_sk").size()

    comp = _tbl(ctx, "complaint")
    comp_claims: dict[str, list[str]] = {}
    if not comp.empty and "claim_sk" in comp.columns:
        text = _col(comp, "complaint_type").astype(str) + " " + _col(comp, "text").astype(str)
        comp = comp[_contains_any(text, complaint_terms) & comp["claim_sk"].notna()]
        for cs, sub in comp.groupby(comp["claim_sk"].astype(str)):
            comp_claims[cs] = sub.get("complaint_sk", pd.Series(dtype=object)).astype(str).tolist()

    df = claims[claims["claim_sk"].isin(set(paid.index) | set(comp_claims))].copy()
    df["_liab"] = _num(_col(df, "patient_share")).fillna(0.0)
    df["_paid"] = df["claim_sk"].map(paid).fillna(0.0)
    df["_excess"] = df["_paid"] - df["_liab"]
    out = []
    for r in _rows(df):
        has_receipt_excess = r._excess > tol
        complaints = comp_claims.get(r.claim_sk, [])
        if not has_receipt_excess and not complaints:
            continue
        if has_receipt_excess:
            plain = (
                f"The member's receipts show {aed(r._paid, 2)} paid to the provider for this claim, but the "
                f"approved patient share is {aed(r._liab, 2)} — {aed(r._excess, 2)} more than they owed."
            )
            if complaints:
                plain += " The member also complained about the charge."
        else:
            plain = ("The member complained of being charged more than their share for this claim "
                     f"(approved share {aed(r._liab, 2)}); no receipt amount is on file to confirm it.")
        out.append(_sig(
            ctx, control, subject_type="claim", subject_id=r.claim_sk,
            fact_key=f"balance_bill:{r.claim_sk}", claim_ids=[r.claim_sk],
            event_time=r._sdate, period=period_bucket(r._sdate),
            confidence=0.9 if has_receipt_excess else 0.5,
            evidence={
                "plain_language": plain,
                "what_the_reviewer_must_verify": "Confirm with the member what they paid and check for "
                                                 "signed consent for any non-covered item.",
                "member_sk": r.member_sk, "provider_sk": r.provider_sk,
                "receipted_amount_aed": round(float(r._paid), 2),
                "receipt_count": int(n_receipts.get(r.claim_sk, 0)),
                "approved_patient_share_aed": round(float(r._liab), 2),
                "excess_aed": round(float(max(r._excess, 0.0)), 2),
                "consented_non_covered_items_excluded_aed": round(float(excluded.get(r.claim_sk, 0.0)), 2),
                "overcharge_complaints": complaints, "tolerance_aed": tol,
            },
            exposure=_exp.no_exposure("the excess was paid by the member, not by the insurer; the remedy "
                                      "is a refund to the member"),
        ))
    return out


# ===========================================================================
# PAY-08 — denial-resubmission mutation and gaming
# ===========================================================================


def _versions(ctx) -> pd.DataFrame:
    cv = _tbl(ctx, "claim_version")
    if cv.empty or "claim_sk" not in cv.columns:
        return pd.DataFrame()
    cv["claim_sk"] = cv["claim_sk"].astype(str)
    cv["_rel"] = _up(_col(cv, "relationship"))
    cv["_type"] = _up(_col(cv, "resubmission_type"))
    prior = _col(cv, "prior_claim_sk")
    cv["_prior"] = pd.Series(
        [None if is_missing(v) or str(v).strip().lower() in ("nan", "none", "nat") else str(v) for v in prior],
        index=cv.index, dtype=object)
    cv["_at"] = _dt(_col(cv, "recorded_at"))
    cv["_vno"] = _num(_col(cv, "version_no"))
    cv["_changed"] = _col(cv, "changed_fields").map(_parse_changed)
    return cv


def _parse_changed(value: Any) -> dict:
    if isinstance(value, dict):
        return value
    if isinstance(value, (list, tuple)):
        return {str(k): None for k in value}
    if is_missing(value):
        return {}
    try:
        parsed = json.loads(str(value))
    except Exception:
        return {str(k).strip(): None for k in re.split(r"[;,|]", str(value)) if str(k).strip()}
    if isinstance(parsed, dict):
        return parsed
    if isinstance(parsed, list):
        return {str(k): None for k in parsed}
    return {}


def _resubmissions(ctx) -> pd.DataFrame:
    """Non-original versions that name (or should name) a prior claim."""
    cv = _versions(ctx)
    if cv.empty:
        return cv
    return cv[cv["_rel"].isin(_RESUB) | (cv["_prior"].notna() & (cv["_rel"] != "ORIGINAL"))]


def _computed_diffs(ctx, pairs: pd.DataFrame) -> dict[tuple[str, str], dict[str, list]]:
    """Field differences computed from the data itself between prior and resubmitted claims."""
    claims = _claims(ctx)
    lines = _tbl(ctx, "claim_line")
    enc = _tbl(ctx, "encounter")
    diffs: dict[tuple[str, str], dict[str, list]] = {}
    if pairs.empty or claims.empty:
        return diffs
    ids = set(pairs["claim_sk"]) | set(pairs["_prior"].dropna())
    ch = claims[claims["claim_sk"].isin(ids)].set_index("claim_sk")
    feats: dict[str, pd.Series] = {}
    for c in ("diagnosis_primary", "provider_sk"):
        if c in ch.columns and ch[c].notna().any():
            feats[c] = ch[c].astype(str)
    if "diagnosis_primary" not in feats:
        dx = _tbl(ctx, "diagnosis")
        if not dx.empty and {"claim_sk", "code"} <= set(dx.columns):
            dx = dx[dx["claim_sk"].astype(str).isin(ids)].copy()
            dx["claim_sk"] = dx["claim_sk"].astype(str)
            if "sequence" in dx.columns:
                dx = dx.sort_values("sequence")
            feats["diagnosis_primary"] = dx.groupby("claim_sk")["code"].first().astype(str)
    codes: dict[str, dict[str, float]] = {}
    if not lines.empty and {"claim_sk", "activity_code"} <= set(lines.columns):
        ls = lines[lines["claim_sk"].astype(str).isin(ids)].copy()
        ls["claim_sk"] = ls["claim_sk"].astype(str)
        ls["_u"] = _num(_col(ls, "units")).fillna(1.0)
        g = ls.groupby(["claim_sk", ls["activity_code"].astype(str)])["_u"].sum()
        for (cs, code), u in g.items():
            codes.setdefault(cs, {})[code] = float(u)
    if not enc.empty and {"claim_sk", "encounter_type"} <= set(enc.columns):
        e = enc[enc["claim_sk"].astype(str).isin(ids)]
        feats["encounter_type"] = e.groupby(e["claim_sk"].astype(str))["encounter_type"].first().astype(str)
    for r in _rows(pairs):
        d: dict[str, list] = {}
        for name, series in feats.items():
            a, b = series.get(r._prior), series.get(r.claim_sk)
            if a is None or b is None or is_missing(a) or is_missing(b):
                continue
            if str(a) != str(b):
                d[name] = [str(a), str(b)]
        # a resubmission may legitimately carry only the refused lines, so a code counts as
        # changed only when the new version bills a code the refused version did not
        pa, pb = codes.get(r._prior, {}), codes.get(r.claim_sk, {})
        if pa and pb:
            added = sorted(set(pb) - set(pa))
            if added:
                d["activity_code"] = [",".join(sorted(pa)), ",".join(sorted(pb))]
            unit_changes = [c for c in set(pa) & set(pb) if abs(pa[c] - pb[c]) > 1e-9]
            if unit_changes:
                d["units"] = [";".join(f"{c}x{pa[c]:g}" for c in sorted(unit_changes)),
                              ";".join(f"{c}x{pb[c]:g}" for c in sorted(unit_changes))]
        diffs[(r._prior, r.claim_sk)] = d
    return diffs


def _mutation_fields(ctx, changed: dict, computed: dict) -> dict[str, Any]:
    tokens = [t.lower() for t in _as_list(ctx.cfg("pay08_reimbursement_field_tokens"))]
    merged: dict[str, Any] = {}
    for k, v in {**changed, **computed}.items():
        if any(t in str(k).lower() for t in tokens):
            merged[str(k)] = v if not isinstance(v, tuple) else list(v)
    return merged


def _canonical_field(name: str) -> str:
    n = name.lower()
    if "ordering" in n or "referr" in n:
        return "ordering_clinician"
    if "attach" in n or "document" in n:
        return "attachment"
    if "diag" in n:
        return "diagnosis"
    if "unit" in n or "quantity" in n:
        return "units"
    if "provider" in n or "clinician" in n or "facility" in n:
        return "provider"
    if "encounter" in n or "setting" in n:
        return "setting"
    if "indicator" in n or "modifier" in n:
        return "indicator"
    if "code" in n or "activity" in n:
        return "code"
    return n


def _accepted(ctx, denial_codes: list[str], fields: dict) -> bool:
    """Every mutated field is an accepted correction for at least one of the denial codes."""
    matrix = ctx.cfg("pay08_accepted_correction_matrix") or {}
    if not denial_codes or not fields:
        return False
    allowed: set[str] = set()
    for code in denial_codes:
        for key, flds in matrix.items():
            if str(code).upper().startswith(str(key).upper()):
                allowed |= {_canonical_field(f) for f in _as_list(flds)}
    return bool(allowed) and all(_canonical_field(f) in allowed for f in fields)


def _mutated_resubmissions(ctx) -> pd.DataFrame:
    """Resubmissions/corrections of a denied claim, with their reimbursement-field mutations."""
    rs = _resubmissions(ctx)
    remit = _remittance(ctx)
    if rs.empty or remit.empty:
        return pd.DataFrame()
    rs = rs[rs["_prior"].notna() & ~rs["_rel"].isin(_CANCEL)]
    status = _claim_payment_status(remit).set_index("claim_sk")
    rs = rs[rs["_prior"].isin(status.index[status["any_denied"]])]
    if rs.empty:
        return pd.DataFrame()
    computed = _computed_diffs(ctx, rs)
    rows = []
    for r in _rows(rs):
        comp = computed.get((r._prior, r.claim_sk), {})
        fields = _mutation_fields(ctx, r._changed, comp)
        codes = status.at[r._prior, "denial_codes"] if r._prior in status.index else []
        paid_now = float(status.at[r.claim_sk, "paid"]) if r.claim_sk in status.index else 0.0
        denied_now = bool(status.at[r.claim_sk, "all_denied"]) if r.claim_sk in status.index else False
        rows.append({
            "claim_sk": r.claim_sk, "prior_claim_sk": r._prior, "relationship": r._rel,
            "resubmission_type": r._type, "recorded_at": r._at, "fields": fields,
            "denial_codes": list(codes), "accepted": _accepted(ctx, list(codes), fields),
            "paid_now": paid_now, "succeeded": paid_now > 0 and not denied_now,
        })
    return pd.DataFrame(rows)


@_register("pay_08_r01_broken_lineage")
def pay_08_r01_broken_lineage(ctx, control) -> list:
    """PAY-08-R01 — a correction/resubmission lacks a valid prior reference or points at an unrelated claim."""
    rs = _resubmissions(ctx)
    claims = _claims(ctx)
    if rs.empty or claims.empty:
        return []
    legacy = _as_list(ctx.cfg("pay08_legacy_route_types"))
    rs = rs[rs["claim_sk"].isin(set(claims["claim_sk"]))]
    if legacy:
        rs = rs[~_contains_any(rs["_type"], legacy)]
    known = set(claims["claim_sk"]) | set(_versions(ctx)["claim_sk"])
    head = claims.set_index("claim_sk")
    out = []
    for r in _rows(rs):
        problems = []
        prior = None if is_missing(r._prior) else r._prior
        me = head.loc[r.claim_sk]
        if prior is None and r._rel in _CANCEL:
            continue  # a cancellation without a prior reference cancels this claim itself
        if prior is None:
            problems.append("no_prior_reference")
        elif prior == r.claim_sk:
            problems.append("references_itself")
        elif prior not in known:
            problems.append("prior_claim_not_found")
        elif prior in head.index:
            pr = head.loc[prior]
            for field, label in (("member_sk", "different_member"), ("provider_sk", "different_provider"),
                                 ("payer_id", "different_payer")):
                if not is_missing(me.get(field)) and not is_missing(pr.get(field)) and str(me.get(field)) != str(pr.get(field)):
                    problems.append(label)
        if not problems:
            continue
        words = {
            "no_prior_reference": "does not name the original claim it replaces",
            "references_itself": "names itself as its own original",
            "prior_claim_not_found": f"names an original claim ({prior}) that does not exist in the file",
            "different_member": "names an original claim for a different member",
            "different_provider": "names an original claim from a different provider",
            "different_payer": "names an original claim with a different payer",
        }
        rel = r._rel.lower() or "resubmission"
        plain = f"This {rel} " + "; it also ".join(words[p] for p in problems) + "."
        out.append(_sig(
            ctx, control, subject_type="claim", subject_id=r.claim_sk,
            fact_key=f"lineage:{r.claim_sk}", claim_ids=[c for c in (r.claim_sk, prior) if c and c in head.index],
            event_time=me.get("_sdate"), period=period_bucket(me.get("_sdate")),
            evidence={
                "plain_language": plain,
                "what_the_reviewer_must_verify": "Find the true original claim and return this one for a "
                                                 "correct reference.",
                "member_sk": me.get("member_sk"), "provider_sk": me.get("provider_sk"),
                "relationship": r._rel, "resubmission_type": r._type or None,
                "prior_claim_sk": prior, "lineage_problems": problems,
            },
            exposure=_exp.no_exposure("a broken lineage is a data-integrity return; the amount is not "
                                      "itself shown to be unpayable"),
        ))
    return out


@_register("pay_08_r02_field_mutation")
def pay_08_r02_field_mutation(ctx, control) -> list:
    """PAY-08-R02 — a denied claim resubmitted with reimbursement-enabling fields changed."""
    mr = _mutated_resubmissions(ctx)
    claims = _claims(ctx)
    if mr.empty or claims.empty:
        return []
    head = claims.set_index("claim_sk")
    mr = mr[mr["fields"].map(bool) & ~mr["accepted"] & mr["claim_sk"].isin(head.index)]
    out = []
    for r in _rows(mr):
        me = head.loc[r.claim_sk]
        diffs = []
        for k, v in r.fields.items():
            if isinstance(v, (list, tuple)) and len(v) == 2:
                diffs.append(f"{k.replace('_', ' ')} {v[0]} → {v[1]}")
            else:
                diffs.append(k.replace("_", " "))
        codes = ", ".join(r.denial_codes) or "not recorded"
        plain = (
            f"After the original claim was refused (reason {codes}), it was resubmitted with "
            f"{'; '.join(diffs)} changed, and the refusal reason does not call for that correction."
        )
        out.append(_sig(
            ctx, control, subject_type="claim", subject_id=r.claim_sk,
            fact_key=f"mutation:{r.prior_claim_sk}:{r.claim_sk}",
            claim_ids=[r.prior_claim_sk, r.claim_sk],
            event_time=me.get("_sdate"), period=period_bucket(me.get("_sdate")),
            confidence=0.85,
            evidence={
                "plain_language": plain,
                "what_the_reviewer_must_verify": "Compare both versions side by side and ask for records "
                                                 "supporting the changed fields.",
                "member_sk": me.get("member_sk"), "provider_sk": me.get("provider_sk"),
                "prior_claim_sk": r.prior_claim_sk, "denial_codes": r.denial_codes,
                "field_diff": {k: (list(v) if isinstance(v, (list, tuple)) else v) for k, v in r.fields.items()},
                "resubmission_paid_aed": round(float(r.paid_now), 2),
                "accepted_by_correction_matrix": False,
            },
            exposure=_exp.no_exposure("the pend is for a field-level review; whether any part of the "
                                      "payment is wrong depends on the records"),
        ))
    return out


@_register("pay_08_r03_resubmission_loop")
def pay_08_r03_resubmission_loop(ctx, control) -> list:
    """PAY-08-R03 — attempts per underlying service exceed the limit, or variants until paid."""
    cv = _versions(ctx)
    claims = _claims(ctx)
    if cv.empty or claims.empty:
        return []
    max_attempts = int(ctx.cfg("pay08_max_attempts"))
    variant_min = int(ctx.cfg("pay08_variants_until_paid_min"))
    appeal_terms = _as_list(ctx.cfg("pay08_appeal_route_types"))
    links = cv[cv["_prior"].notna() & ~cv["_rel"].isin(_CANCEL) & (cv["_prior"] != cv["claim_sk"])]
    if links.empty:
        return []
    parent = dict(zip(links["claim_sk"], links["_prior"]))

    def root_of(c: str) -> str:
        seen = {c}
        while c in parent and parent[c] not in seen:
            c = parent[c]
            seen.add(c)
        return c

    links = links.assign(_root=links["claim_sk"].map(root_of))
    links["_appeal"] = _contains_any(links["_type"], appeal_terms) if appeal_terms else False
    remit = _remittance(ctx)
    status = _claim_payment_status(remit).set_index("claim_sk") if not remit.empty else pd.DataFrame()
    head = claims.set_index("claim_sk")
    out = []
    for root, sub in links.groupby("_root"):
        counted = sub[~sub["_appeal"]]
        attempts = len(counted)
        if attempts == 0:
            continue
        sub = sub.sort_values(["_at", "_vno"])
        chain = [root] + sub["claim_sk"].tolist()
        final = chain[-1]
        signatures = {tuple(sorted(d.keys())) for d in counted["_changed"] if d}
        final_paid = bool(not status.empty and final in status.index and status.at[final, "paid"] > 0
                          and not status.at[final, "all_denied"])
        limb = None
        if attempts > max_attempts:
            limb = "attempts_exceed_limit"
        elif attempts >= variant_min and len(signatures) >= variant_min and final_paid:
            limb = "variants_until_paid"
        if limb is None:
            continue
        anchor = head.loc[final] if final in head.index else (head.loc[root] if root in head.index else None)
        sdate = None if anchor is None else anchor.get("_sdate")
        plain = (
            f"The same service was sent {attempts + 1} times (the original and {count_phrase(attempts, 'resubmission')}), "
            f"with {count_phrase(len(signatures), 'different set')} of changed details"
            + (", until it was paid." if final_paid else "; the limit is " + str(max_attempts) + " resubmissions.")
        )
        out.append(_sig(
            ctx, control, subject_type="claim", subject_id=final,
            fact_key=f"chain:{root}", claim_ids=[c for c in chain if c in head.index],
            event_time=sdate, period=period_bucket(sdate), confidence=0.85,
            evidence={
                "plain_language": plain,
                "what_the_reviewer_must_verify": "Review every version together and check whether any "
                                                 "attempt was a formal appeal.",
                "member_sk": None if anchor is None else anchor.get("member_sk"),
                "provider_sk": None if anchor is None else anchor.get("provider_sk"),
                "original_claim_sk": root, "chain": chain, "attempts_counted": attempts,
                "appeal_attempts_excluded": int(sub["_appeal"].sum()),
                "distinct_change_sets": [list(s) for s in sorted(signatures)],
                "final_version_paid": final_paid, "max_attempts": max_attempts, "limb": limb,
            },
            exposure=_exp.no_exposure("repeated attempts are a pattern; the final payment may still be "
                                      "correct"),
        ))
    return out


@_register("pay_08_r04_edit_learning")
def pay_08_r04_edit_learning(ctx, control) -> list:
    """PAY-08-R04 — provider's mutated-resubmission success well above peers, concentrated on few denial codes."""
    mr = _mutated_resubmissions(ctx)
    claims = _claims(ctx)
    if mr.empty or claims.empty:
        return []
    mr = mr[mr["fields"].map(bool) & ~mr["accepted"]]  # accepted corrections are not edit-gaming
    mr = mr.merge(claims[["claim_sk", "provider_sk", "_sdate"]], on="claim_sk", how="inner")
    if mr.empty:
        return []
    min_n = int(ctx.cfg("pay08_success_min_resubmissions"))
    uplift = float(ctx.cfg("pay08_success_uplift"))
    conc_min = float(ctx.cfg("pay08_denial_concentration_min"))
    mass = float(ctx.cfg("posterior_interval_mass"))
    width = float(ctx.cfg("max_posterior_width"))
    mr["_ok"] = mr["succeeded"].astype(float)
    mr["_code"] = mr["denial_codes"].map(lambda c: c[0] if c else "")
    g = mr.groupby("provider_sk")
    agg = pd.DataFrame({"n": g["_ok"].size(), "ok": g["_ok"].sum()}).reset_index()
    prior = fit_beta_prior(agg["ok"], agg["n"])
    out = []
    for r in _rows(agg[agg["n"] >= min_n]):
        sub = mr[mr["provider_sk"] == r.provider_sk]
        others = mr[mr["provider_sk"] != r.provider_sk]
        peer_rate = float(others["_ok"].mean()) if len(others) else prior.peer_mean
        sr = shrink_rate(str(r.provider_sk), r.ok, r.n, prior, interval_mass=mass, max_posterior_width=width)
        if sr.shrunk_rate - peer_rate < uplift:
            continue
        codes = sub.loc[sub["_ok"] > 0, "_code"]
        codes = codes[codes != ""]
        if codes.empty:
            continue
        top_code = codes.value_counts().index[0]
        conc = float(codes.value_counts().iloc[0]) / float(len(codes))
        if conc < conc_min:
            continue
        observed = float(r.ok) / float(r.n)
        plain = (
            f"{pct(observed)} of this provider's changed resubmissions after a refusal were paid "
            f"({int(r.ok)} of {int(r.n)}), against {pct(peer_rate)} for other providers; "
            f"{pct(conc)} of the successes followed refusal reason {top_code}."
        )
        out.append(_sig(
            ctx, control, subject_type="provider", subject_id=r.provider_sk,
            fact_key=f"edit_learning:{r.provider_sk}", claim_ids=sub["claim_sk"].tolist(),
            event_time=sub["_sdate"].max(), period=period_bucket(sub["_sdate"].max()),
            confidence=0.5 if sr.excluded_from_ranking else 0.8,
            evidence={
                "plain_language": plain,
                "what_the_reviewer_must_verify": "Check whether a billing company or a policy change "
                                                 "explains the pattern, then sample the resubmissions.",
                "provider_sk": r.provider_sk, "mutated_resubmissions": int(r.n), "paid": int(r.ok),
                "observed_success_rate": round(observed, 4), "shrunk_success_rate": round(sr.shrunk_rate, 4),
                "peer_success_rate": round(peer_rate, 4), "top_denial_code": top_code,
                "top_denial_code_share": round(conc, 4),
                "excluded_from_ranking": sr.excluded_from_ranking, "shrinkage_explanation": sr.explain(),
                "exclusion_note": "Biller/vendor and policy-change adjustments are not in the file and "
                                  "could not be applied.",
            },
            exposure=_exp.no_exposure("a success-rate pattern does not show that any single payment "
                                      "was wrong"),
        ))
    return out


# ===========================================================================
# PAY-09 — disguised non-covered service or product
# ===========================================================================


def _negated(text: str, start: int, negations: list[str], window: int) -> bool:
    """A negation word (whole word, so 'no' does not match 'note') shortly before the term."""
    before = text[max(0, start - window):start].lower()
    for n in negations:
        n = n.strip().lower()
        if not n:
            continue
        tail = "" if n.endswith("-") else r"(?!\w)"
        if re.search(r"(?<!\w)" + re.escape(n) + tail, before):
            return True
    return False


@_register("pay_09_r01_document_conflict")
def pay_09_r01_document_conflict(ctx, control) -> list:
    """PAY-09-R01 — the attached note describes a non-covered service while a covered code is billed."""
    claims = _claims(ctx)
    docs = _tbl(ctx, "document")
    lines = _tbl(ctx, "claim_line")
    if claims.empty or docs.empty or lines.empty or not {"claim_sk", "text"} <= set(docs.columns):
        return []
    terms = _as_list(ctx.cfg("pay09_noncovered_terms"))
    policy = _tbl(ctx, "disguise_risk_policy")
    if not policy.empty and "excluded_service" in policy.columns:
        terms = terms + [t for t in policy["excluded_service"].dropna().astype(str) if len(t) > 3]
    negations = _as_list(ctx.cfg("pay09_negation_terms"))
    window = int(ctx.cfg("pay09_negation_window_chars"))
    min_ocr = float(ctx.cfg("pay09_min_ocr_confidence"))
    docs = docs[docs["claim_sk"].notna() & docs["text"].notna()].copy()
    docs["claim_sk"] = docs["claim_sk"].astype(str)
    docs = docs[docs["claim_sk"].isin(set(claims["claim_sk"]))]
    ocr = _num(_col(docs, "ocr_confidence"))
    docs = docs[ocr.isna() | (ocr >= min_ocr)]
    docs = docs[_contains_any(docs["text"], terms)]
    if docs.empty:
        return []
    lb = _line_benefits(ctx)
    covered_claims: dict[str, list[str]] | None = None
    if not lb.empty:
        lb = lb[lb["claim_sk"].isin(set(docs["claim_sk"]))]
        cov = lb[lb["covered"].astype(str).str.strip().str.lower().isin(("true", "1", "yes", "y"))]
        covered_claims = cov.groupby("claim_sk")["activity_code"].agg(lambda s: sorted(set(s.astype(str)))).to_dict()
    else:
        lines = lines.copy()
        lines["claim_sk"] = lines["claim_sk"].astype(str)
    head = claims.set_index("claim_sk")
    lowered = [t.lower() for t in terms]
    out = []
    for cs, sub in docs.groupby("claim_sk"):
        codes = (covered_claims or {}).get(cs) if covered_claims is not None else sorted(
            set(lines.loc[lines["claim_sk"] == cs, "activity_code"].dropna().astype(str)))
        if not codes:
            continue
        found = None
        for d in _rows(sub):
            text = str(d.text)
            low = text.lower()
            for t in lowered:
                pos = low.find(t)
                while pos >= 0:
                    if not _negated(low, pos, negations, window):
                        found = (d, t, text[max(0, pos - 60): pos + len(t) + 60])
                        break
                    pos = low.find(t, pos + 1)
                if found:
                    break
            if found:
                break
        if not found:
            continue
        d, term, span = found
        me = head.loc[cs]
        doc_sk = getattr(d, "document_sk", None)
        plain = (
            f"The attached {_s(getattr(d, 'doc_type', '')) or 'document'} describes '{term}', a service the "
            f"plan does not cover, while the claim bills covered code{'s' if len(codes) > 1 else ''} "
            f"{', '.join(codes[:4])}."
        )
        out.append(_sig(
            ctx, control, subject_type="claim", subject_id=cs,
            fact_key=f"disguise_doc:{cs}", claim_ids=[cs],
            event_time=me.get("_sdate"), period=period_bucket(me.get("_sdate")),
            confidence=0.7,
            evidence={
                "plain_language": plain,
                "what_the_reviewer_must_verify": "Have a clinician read the note and confirm what was done.",
                "member_sk": me.get("member_sk"), "provider_sk": me.get("provider_sk"),
                "document_sk": None if is_missing(doc_sk) else str(doc_sk),
                "document_language": _s(getattr(d, "language", "")) or None,
                "matched_term": term, "source_span": span.strip(), "billed_covered_codes": codes,
                "ocr_confidence": None if is_missing(getattr(d, "ocr_confidence", None)) else float(d.ocr_confidence),
                "proxy_note": "Term matching with a negation window stands in for clinical NLP.",
            },
            exposure=_exp.no_exposure("what the note describes must be confirmed by a clinician before "
                                      "any amount is treated as unpayable"),
        ))
    return out


@_register("pay_09_r02_covered_code_substitution")
def pay_09_r02_covered_code_substitution(ctx, control) -> list:
    """PAY-09-R02 — provider bills a payable proxy code unusually often right after denied services."""
    claims = _claims(ctx)
    lines = _tbl(ctx, "claim_line")
    remit = _remittance(ctx)
    if claims.empty or lines.empty or remit.empty or "activity_code" not in lines.columns:
        return []
    window = int(ctx.cfg("pay09_substitution_window_days"))
    min_events = int(ctx.cfg("pay09_substitution_min_events"))
    uplift = float(ctx.cfg("pay09_substitution_uplift"))
    mass = float(ctx.cfg("posterior_interval_mass"))
    width = float(ctx.cfg("max_posterior_width"))
    lines = lines.copy()
    lines["claim_sk"] = lines["claim_sk"].astype(str)
    if "line_sk" not in lines.columns:
        return []
    lines["line_sk"] = lines["line_sk"].astype(str)
    lines = lines.merge(claims[["claim_sk", "member_sk", "provider_sk", "_sdate"]], on="claim_sk", how="inner")
    lines["_date"] = _dt(_col(lines, "service_date")).fillna(lines["_sdate"])
    acr = _tbl(ctx, "activity_code_reference")
    if not acr.empty and {"activity_code", "code_family"} <= set(acr.columns):
        fam = acr.drop_duplicates("activity_code").set_index("activity_code")["code_family"]
        lines["_fam"] = lines["activity_code"].map(fam)
    else:
        lines["_fam"] = None
    r_line = remit[remit["_line"].notna()].copy()
    r_line["_line"] = r_line["_line"].astype(str)
    denied_lines = set(r_line.loc[r_line["_denied"], "_line"])
    paid_lines = set(r_line.loc[(~r_line["_denied"]) & (r_line["_pay"] > 0), "_line"]) - denied_lines
    # a substitute is billed in the same setting: an office visit after a refused inpatient stay is
    # ordinary follow-up care, so inpatient claims are left out and claim types must match
    ctype = _up(_col(claims, "claim_type"))
    lines["_ctype"] = lines["claim_sk"].map(dict(zip(claims["claim_sk"], ctype)))
    lines = lines[lines["_ctype"] != "INPATIENT"]
    cols = ["line_sk", "claim_sk", "member_sk", "provider_sk", "_date", "activity_code", "_fam", "_ctype"]
    D = lines[lines["line_sk"].isin(denied_lines)][cols]
    P = lines[lines["line_sk"].isin(paid_lines)][cols]
    if D.empty or P.empty:
        return []
    # the resubmission of the denied claim itself is PAY-08 territory, not a substitution
    cv = _versions(ctx)
    lineage_pairs: set[tuple[str, str]] = set()
    if not cv.empty:
        lineage_pairs = set(zip(cv["_prior"].fillna(""), cv["claim_sk"]))
    m = D.merge(P, on=["member_sk", "provider_sk", "_ctype"], suffixes=("_d", "_p"))
    gap = (m["_date_p"] - m["_date_d"]).dt.days
    m = m[(m["claim_sk_d"] != m["claim_sk_p"]) & (gap >= 0) & (gap <= window)
          & (m["activity_code_d"] != m["activity_code_p"])]
    same_family = m["_fam_d"].notna() & (m["_fam_d"] == m["_fam_p"])
    m = m[~same_family]
    if lineage_pairs:
        m = m[[(a, b) not in lineage_pairs for a, b in zip(m["claim_sk_d"], m["claim_sk_p"])]]
    subst = m.drop_duplicates("line_sk_d")
    D["_sub"] = D["line_sk"].isin(set(subst["line_sk_d"])).astype(float)
    g = D.groupby("provider_sk")
    agg = pd.DataFrame({"n": g["_sub"].size(), "k": g["_sub"].sum()}).reset_index()
    prior = fit_beta_prior(agg["k"], agg["n"])
    policy = _tbl(ctx, "disguise_risk_policy")
    listed = set(policy["activity_code"].astype(str)) if not policy.empty and "activity_code" in policy.columns else set()
    out = []
    for r in _rows(agg[agg["k"] >= min_events]):
        others = D[D["provider_sk"] != r.provider_sk]
        peer_rate = float(others["_sub"].mean()) if len(others) else prior.peer_mean
        sr = shrink_rate(str(r.provider_sk), r.k, r.n, prior, interval_mass=mass, max_posterior_width=width)
        if sr.shrunk_rate - peer_rate < uplift:
            continue
        ps = subst[subst["provider_sk"] == r.provider_sk]
        top = ps["activity_code_p"].value_counts()
        proxy, proxy_n = str(top.index[0]), int(top.iloc[0])
        denied_top = ps["activity_code_d"].value_counts().index[0]
        observed = float(r.k) / float(r.n)
        plain = (
            f"After {pct(observed)} of this provider's refused services ({int(r.k)} of {int(r.n)}), the same "
            f"patient was billed a different, paid code within {window} days — most often {proxy} after a "
            f"refused {denied_top}; for other providers it is {pct(peer_rate)}."
        )
        claim_ids = sorted(set(ps["claim_sk_d"]) | set(ps["claim_sk_p"]))
        out.append(_sig(
            ctx, control, subject_type="provider", subject_id=r.provider_sk,
            fact_key=f"substitution:{r.provider_sk}", claim_ids=claim_ids,
            event_time=ps["_date_p"].max(), period=period_bucket(ps["_date_p"].max()),
            confidence=0.5 if sr.excluded_from_ranking else 0.8,
            evidence={
                "plain_language": plain,
                "what_the_reviewer_must_verify": "Review the refusal history and sample the substitute-code "
                                                 "claims for record review.",
                "provider_sk": r.provider_sk, "denied_services": int(r.n), "followed_by_substitute": int(r.k),
                "observed_rate": round(observed, 4), "shrunk_rate": round(sr.shrunk_rate, 4),
                "peer_rate": round(peer_rate, 4), "most_common_substitute_code": proxy,
                "substitute_code_uses": proxy_n, "most_common_refused_code": str(denied_top),
                "substitute_code_on_disguise_risk_list": proxy in listed if listed else None,
                "window_days": window, "shrinkage_explanation": sr.explain(),
            },
            exposure=_exp.no_exposure("a substitution pattern does not show that any single paid line "
                                      "was for a non-covered service"),
        ))
    return out


@_register("pay_09_r03_cosmetic_disguise")
def pay_09_r03_cosmetic_disguise(ctx, control) -> list:
    """PAY-09-R03 — diagnosis, setting and code match the approved disguise-risk policy."""
    claims = _claims(ctx)
    lines = _tbl(ctx, "claim_line")
    policy = _tbl(ctx, "disguise_risk_policy")
    dx = _tbl(ctx, "diagnosis")
    if claims.empty or lines.empty or policy.empty or dx.empty:
        return []
    if not {"activity_code", "risk_diagnosis_prefixes"} <= set(policy.columns) or "code" not in dx.columns:
        return []
    settings = [s.upper() for s in _as_list(ctx.cfg("pay09_disguise_risk_settings"))]
    recon = [p.upper() for p in _as_list(ctx.cfg("pay09_reconstructive_dx_prefixes"))]
    lines = lines.copy()
    lines["claim_sk"] = lines["claim_sk"].astype(str)
    lines = lines[lines["claim_sk"].isin(set(claims["claim_sk"]))]
    pol = policy.drop_duplicates("activity_code").copy()
    pol["activity_code"] = pol["activity_code"].astype(str)
    lines["activity_code"] = lines["activity_code"].astype(str)
    hit = lines.merge(pol[["activity_code", "risk_diagnosis_prefixes"] +
                          (["excluded_service"] if "excluded_service" in pol.columns else [])],
                      on="activity_code", how="inner")
    if hit.empty:
        return []
    dx = dx.copy()
    dx["claim_sk"] = dx["claim_sk"].astype(str)
    dx["_code"] = _up(dx["code"]).str.replace(".", "", regex=False)
    all_prefixes = tuple({p.upper().replace(".", "") for v in pol["risk_diagnosis_prefixes"].dropna()
                          for p in _as_list(v)})
    if not all_prefixes:
        return []
    risky = set(dx.loc[dx["_code"].str.startswith(all_prefixes), "claim_sk"])
    hit = hit[hit["claim_sk"].isin(risky)]
    dx = dx[dx["claim_sk"].isin(risky)]
    codes_by_claim = dx.groupby("claim_sk")["_code"].agg(list).to_dict()
    raw_codes = dict(zip(dx["_code"], dx["code"].astype(str)))
    enc = _tbl(ctx, "encounter")
    setting_by_claim = {}
    if not enc.empty and {"claim_sk", "encounter_type"} <= set(enc.columns):
        setting_by_claim = dict(zip(enc["claim_sk"].astype(str), _up(enc["encounter_type"])))
    auth_ok = set()
    auth = _tbl(ctx, "authorization")
    if "authorization_id" in lines.columns and not auth.empty and "authorization_sk" in auth.columns:
        approved = set(auth.loc[_up(_col(auth, "status")).isin({"APPROVED", "APPROVED_WITH_CONDITIONS", "A"}),
                                "authorization_sk"].astype(str))
        auth_ok = set(lines.loc[lines["authorization_id"].astype(str).isin(approved), "line_sk"].astype(str)) \
            if "line_sk" in lines.columns else set()
    head = claims.set_index("claim_sk")
    out = []
    for r in _rows(hit):
        codes = codes_by_claim.get(r.claim_sk, [])
        prefixes = [p.upper().replace(".", "") for p in _as_list(r.risk_diagnosis_prefixes)]
        matched = [c for c in codes if any(c.startswith(p) for p in prefixes)]
        if not matched:
            continue
        if recon and any(c.startswith(p.replace(".", "")) for c in codes for p in recon):
            continue  # declared exclusion: reconstructive
        line_sk = str(getattr(r, "line_sk", ""))
        if line_sk and line_sk in auth_ok:
            continue  # declared exclusion: medical-necessity exception approved in advance
        setting = setting_by_claim.get(r.claim_sk, "")
        if settings and setting and not any(s in setting for s in settings):
            continue
        me = head.loc[r.claim_sk]
        excluded = _s(getattr(r, "excluded_service", "")) or "an excluded service"
        amount = _f(getattr(r, "net_amount", None), _f(getattr(r, "gross_amount", None)))
        plain = (
            f"Code {r.activity_code} ({aed(amount, 2)}) is billed with diagnosis {raw_codes.get(matched[0], matched[0])}"
            + (f" in {'an' if setting[:1] in 'AEIOU' else 'a'} {setting.lower().replace('_', ' ')} setting"
               if setting else "")
            + f" — a combination the plan's disguise-risk list links to {excluded.lower()}."
        )
        out.append(_sig(
            ctx, control, subject_type="claim", subject_id=r.claim_sk,
            fact_key=f"disguise_line:{line_sk or r.claim_sk}", claim_ids=[r.claim_sk],
            event_time=me.get("_sdate"), period=period_bucket(me.get("_sdate")), confidence=0.8,
            evidence={
                "plain_language": plain,
                "what_the_reviewer_must_verify": "Send for clinical review and check whether a "
                                                 "reconstructive or medical-need exception applies.",
                "member_sk": me.get("member_sk"), "provider_sk": me.get("provider_sk"),
                "line_sk": line_sk or None, "activity_code": r.activity_code,
                "matched_diagnoses": [raw_codes.get(m, m) for m in matched],
                "policy_risk_diagnosis_prefixes": prefixes, "policy_excluded_service": excluded,
                "encounter_type": setting or None, "line_amount_aed": round(amount, 2),
            },
            exposure=_exp.no_exposure("a clinician must confirm the service was cosmetic before the line "
                                      "is treated as unpayable"),
        ))
    return out


def _tokens(text: Any) -> set[str]:
    return {t for t in re.findall(r"[a-z؀-ۿ]{3,}", str(text).lower())}


@_register("pay_09_r04_member_confirmation_mismatch")
def pay_09_r04_member_confirmation_mismatch(ctx, control) -> list:
    """PAY-09-R04 — the member's confirmation or receipt describes a different service than billed."""
    claims = _claims(ctx)
    conf = _tbl(ctx, "member_confirmation")
    receipts = _tbl(ctx, "member_receipt")
    if claims.empty or (conf.empty and receipts.empty):
        return []
    unreliable = [c.upper() for c in _as_list(ctx.cfg("pay09_unreliable_confirmation_channels"))]
    min_sim = float(ctx.cfg("pay09_receipt_similarity_min"))
    consent_terms = _as_list(ctx.cfg("pay07_consented_item_terms"))
    generic = [g.lower() for g in _as_list(ctx.cfg("pay09_generic_receipt_terms"))]
    head = claims.set_index("claim_sk")
    findings: dict[str, dict[str, Any]] = {}
    if not conf.empty and {"claim_sk", "service_confirmed"} <= set(conf.columns):
        c = conf[conf["claim_sk"].notna()].copy()
        c["claim_sk"] = c["claim_sk"].astype(str)
        flag = c["service_confirmed"].map(lambda v: str(v).strip().lower() in ("false", "0", "no", "n"))
        for r in _rows(c[flag & c["claim_sk"].isin(head.index)]):
            channel = _s(getattr(r, "channel", "")).upper()
            findings.setdefault(r.claim_sk, {"limbs": [], "details": {}, "confidence": 0.8})
            f = findings[r.claim_sk]
            f["limbs"].append("member_did_not_confirm_service")
            f["details"]["confirmation"] = {
                "confirmation_sk": _s(getattr(r, "confirmation_sk", "")) or None,
                "response": _s(getattr(r, "response", "")) or None,
                "response_date": _date_str(getattr(r, "response_date", None)), "channel": channel or None,
            }
            if channel and channel in unreliable:
                f["confidence"] = 0.5
    if not receipts.empty and {"claim_sk", "item_description"} <= set(receipts.columns):
        lines = _tbl(ctx, "claim_line")
        acr = _tbl(ctx, "activity_code_reference")
        if not lines.empty and "claim_sk" in lines.columns:
            lines = lines.copy()
            lines["claim_sk"] = lines["claim_sk"].astype(str)
            desc = _col(lines, "activity_description").astype(object)
            if not acr.empty and {"activity_code", "description"} <= set(acr.columns):
                ref = acr.drop_duplicates("activity_code").set_index("activity_code")["description"]
                desc = desc.where(desc.notna(), lines["activity_code"].map(ref)) if "activity_code" in lines.columns else desc
            lines["_desc"] = desc.fillna("").astype(str)
            billed = lines.groupby("claim_sk")["_desc"].agg(list).to_dict()
            rc = receipts[receipts["claim_sk"].notna()].copy()
            rc["claim_sk"] = rc["claim_sk"].astype(str)
            rc = rc[~_contains_any(_col(rc, "item_description").astype(str), consent_terms)]
            for r in _rows(rc[rc["claim_sk"].isin(set(billed))]):
                item = _s(r.item_description)
                it = _tokens(item) - set(generic)
                if not it:
                    continue
                best, best_desc = 0.0, ""
                for d in billed[r.claim_sk]:
                    dt = _tokens(d) - set(generic)
                    if not dt:
                        continue
                    sim = len(it & dt) / len(it | dt)
                    if sim > best:
                        best, best_desc = sim, d
                if best >= min_sim:
                    continue
                findings.setdefault(r.claim_sk, {"limbs": [], "details": {}, "confidence": 0.7})
                f = findings[r.claim_sk]
                f["limbs"].append("receipt_describes_other_item")
                f["details"]["receipt"] = {
                    "receipt_sk": _s(getattr(r, "receipt_sk", "")) or None, "item_description": item,
                    "billed_descriptions": billed[r.claim_sk][:5], "best_similarity": round(best, 3),
                }
    out = []
    for cs, f in findings.items():
        me = head.loc[cs]
        limbs = sorted(set(f["limbs"]))
        billed_codes = ""
        if "receipt" in f["details"]:
            billed_codes = "; ".join(f["details"]["receipt"]["billed_descriptions"][:2])
        parts = []
        if "member_did_not_confirm_service" in limbs:
            resp = f["details"]["confirmation"].get("response")
            parts.append("the member said the billed service did not take place"
                         + (f" (\"{resp[:80]}\")" if resp else ""))
        if "receipt_describes_other_item" in limbs:
            parts.append(f"the member's receipt is for \"{f['details']['receipt']['item_description'][:60]}\" "
                         f"while the claim bills \"{billed_codes[:80]}\"")
        plain = "When asked, " + "; and ".join(parts) + "."
        out.append(_sig(
            ctx, control, subject_type="claim", subject_id=cs,
            fact_key=f"member_confirm:{cs}", claim_ids=[cs],
            event_time=me.get("_sdate"), period=period_bucket(me.get("_sdate")),
            confidence=f["confidence"],
            evidence={
                "plain_language": plain,
                "what_the_reviewer_must_verify": "Contact the member to confirm what they received and "
                                                 "check how reliable the confirmation is.",
                "member_sk": me.get("member_sk"), "provider_sk": me.get("provider_sk"),
                "limbs": limbs, **f["details"],
            },
            exposure=_exp.no_exposure("the member's account must be confirmed before any amount is "
                                      "treated as unpayable"),
        ))
    return out


# ===========================================================================
# PAY-10 — coordination of benefits and third-party liability
# ===========================================================================


@_register("pay_10_r02_multi_payer_overpayment")
def pay_10_r02_multi_payer_overpayment(ctx, control) -> list:
    """PAY-10-R02 — payments from every payer plus patient share exceed the allowable charge."""
    claims = _claims(ctx)
    remit = _remittance(ctx)
    other = _tbl(ctx, "other_payer_remittance")
    if claims.empty or remit.empty or other.empty or "claim_sk" not in other.columns:
        return []
    tol = float(ctx.cfg("pay10_multi_payer_tolerance_aed"))
    other["claim_sk"] = other["claim_sk"].astype(str)
    other["_amt"] = _num(_col(other, "payment_amount")).fillna(0.0)
    op = other.groupby("claim_sk").agg(other_paid=("_amt", "sum"),
                                       other_payers=("other_payer_id", lambda s: sorted(set(s.dropna().astype(str))))
                                       if "other_payer_id" in other.columns else ("_amt", "size"))
    ours = remit.groupby("claim_sk")["_pay"].sum().rename("our_paid")
    df = claims.merge(op, left_on="claim_sk", right_index=True, how="inner")
    df = df.merge(ours, left_on="claim_sk", right_index=True, how="left")
    df["our_paid"] = df["our_paid"].fillna(0.0)
    df["_share"] = _num(_col(df, "patient_share")).fillna(0.0)
    in_gross = _discount_already_in_gross(ctx, claims)
    df["_allow"] = _num(_col(df, "gross_amount")).fillna(0.0) - (
        0.0 if in_gross else _num(_col(df, "discount")).fillna(0.0))
    # what the member still owes after other payers: a secondary insurer that pays the member's
    # share settles that liability, so it is not counted twice
    df["_member_owes"] = (df["_share"] - df["other_paid"]).clip(lower=0.0)
    df["_total"] = df["other_paid"] + df["our_paid"] + df["_member_owes"]
    df["_over"] = df["_total"] - df["_allow"]
    hits = df[(df["_over"] > tol) & (df["our_paid"] > 0)]
    out = []
    for r in _rows(hits):
        recoverable = min(float(r.our_paid), float(r._over))
        plain = (
            f"Another insurer paid {aed(r.other_paid, 2)} and we paid {aed(r.our_paid, 2)}; with what the patient "
            f"still owes ({aed(r._member_owes, 2)}) that is {aed(r._total, 2)} against an allowed charge of "
            f"{aed(r._allow, 2)} — {aed(r._over, 2)} too much."
        )
        payers = r.other_payers if isinstance(r.other_payers, list) else []
        out.append(_sig(
            ctx, control, subject_type="claim", subject_id=r.claim_sk,
            fact_key=f"multi_payer:{r.claim_sk}", claim_ids=[r.claim_sk],
            event_time=r._sdate, period=period_bucket(r._sdate),
            evidence={
                "plain_language": plain,
                "what_the_reviewer_must_verify": "Obtain the other insurer's remittance and check whether a "
                                                 "lawful top-up applies.",
                "member_sk": r.member_sk, "provider_sk": r.provider_sk, "other_payers": payers,
                "other_payer_paid_aed": round(float(r.other_paid), 2), "our_paid_aed": round(float(r.our_paid), 2),
                "patient_share_aed": round(float(r._share), 2), "member_still_owes_aed": round(float(r._member_owes), 2),
                "allowed_charge_aed": round(float(r._allow), 2),
                "overpayment_aed": round(float(r._over), 2), "tolerance_aed": tol,
            },
            exposure=_exp.line_edit_exposure(float(r.our_paid), float(r.our_paid) - recoverable),
        ))
    return out


@_register("pay_10_r03_accident_without_liability")
def pay_10_r03_accident_without_liability(ctx, control) -> list:
    """PAY-10-R03 — accident indicator or external-cause diagnosis but no liable-party record."""
    claims = _claims(ctx)
    tpl = _tbl(ctx, "third_party_liability")
    if claims.empty:
        return []
    prefixes = [p.upper().replace(".", "") for p in _as_list(ctx.cfg("pay10_accident_dx_prefixes"))]
    false_pos = [p.upper().replace(".", "") for p in _as_list(ctx.cfg("pay10_accident_false_positive_dx_prefixes"))]
    dx = _tbl(ctx, "diagnosis")
    if tpl.empty:
        return []  # without a liability register "no liable party recorded" cannot be established
    flagged: dict[str, dict[str, Any]] = {}
    if "accident_indicator" in claims.columns:
        ind = claims["accident_indicator"].map(lambda v: str(v).strip().lower() in ("true", "1", "yes", "y"))
        for cs in claims.loc[ind, "claim_sk"]:
            flagged.setdefault(cs, {"indicator": True, "dx": []})
    fp_claims: set[str] = set()
    if not dx.empty and {"claim_sk", "code"} <= set(dx.columns) and prefixes:
        d = dx.copy()
        d["claim_sk"] = d["claim_sk"].astype(str)
        d["_c"] = _up(d["code"]).str.replace(".", "", regex=False)
        acc = d[d["_c"].map(lambda c: any(c.startswith(p) for p in prefixes))]
        fp = d[d["_c"].map(lambda c: any(c.startswith(p) for p in false_pos))] if false_pos else d.iloc[0:0]
        fp_claims = set(fp["claim_sk"])
        for cs, sub in acc.groupby("claim_sk"):
            flagged.setdefault(cs, {"indicator": False, "dx": []})["dx"] = sorted(set(sub["code"].astype(str)))
    if not flagged:
        return []
    have = set()
    if not tpl.empty and "claim_sk" in tpl.columns:
        t = tpl.copy()
        party = _col(t, "liable_party")
        t = t[party.notna() & (party.astype(str).str.strip() != "")] if "liable_party" in t.columns else t
        have = set(t["claim_sk"].astype(str))
    head = claims.set_index("claim_sk")
    out = []
    for cs, info in flagged.items():
        if cs in have or cs not in head.index:
            continue
        if not info["indicator"] and cs in fp_claims:
            continue  # declared exclusion: clinical false-positive list
        if info["indicator"] is False and not info["dx"]:
            continue
        me = head.loc[cs]
        basis = []
        if info["indicator"]:
            basis.append("is marked as an accident")
        if info["dx"]:
            basis.append(f"carries external-cause diagnosis {', '.join(info['dx'][:3])}")
        plain = (f"This claim ({aed(me.get('gross_amount_aed', me.get('gross_amount')), 0)}) "
                 f"{' and '.join(basis)}, but no liable party (motor insurer, employer or other) is recorded.")
        out.append(_sig(
            ctx, control, subject_type="claim", subject_id=cs,
            fact_key=f"tpl:{cs}", claim_ids=[cs],
            event_time=me.get("_sdate"), period=period_bucket(me.get("_sdate")),
            confidence=0.9 if info["indicator"] else 0.7,
            evidence={
                "plain_language": plain,
                "what_the_reviewer_must_verify": "Ask how the injury happened and hold the claim for "
                                                 "liable-party details.",
                "member_sk": me.get("member_sk"), "provider_sk": me.get("provider_sk"),
                "accident_indicator": bool(info["indicator"]), "external_cause_diagnoses": info["dx"],
                "liable_party_recorded": False,
            },
            exposure=_exp.no_exposure("the amount recoverable from a liable party is not known until the "
                                      "party is identified"),
        ))
    return out


@_register("pay_10_r04_paid_after_settlement")
def pay_10_r04_paid_after_settlement(ctx, control) -> list:
    """PAY-10-R04 — we paid after a documented third-party settlement, without an offset."""
    claims = _claims(ctx)
    remit = _remittance(ctx)
    tps = _tbl(ctx, "third_party_settlement")
    if claims.empty or remit.empty or tps.empty or "claim_sk" not in tps.columns:
        return []
    tol = float(ctx.cfg("pay10_settlement_tolerance_aed"))
    tps["claim_sk"] = tps["claim_sk"].astype(str)
    tps["_amt"] = _num(_col(tps, "settlement_amount")).fillna(0.0)
    tps["_date"] = _dt(_col(tps, "settlement_date"))
    st = tps.groupby("claim_sk").agg(settled=("_amt", "sum"), settled_on=("_date", "min"))
    ledger = _tbl(ctx, "recovery_ledger")
    in_recovery = set(ledger["claim_sk"].dropna().astype(str)) if not ledger.empty and "claim_sk" in ledger.columns else set()
    rm = remit.merge(st, left_on="claim_sk", right_index=True, how="inner")
    if rm.empty:
        return []
    after = rm[rm["_settle"].isna() | rm["settled_on"].isna() | (rm["_settle"] >= rm["settled_on"])]
    paid_after = after.groupby("claim_sk")["_pay"].sum()
    df = claims.merge(st, left_on="claim_sk", right_index=True, how="inner")
    df["_paid_after"] = df["claim_sk"].map(paid_after).fillna(0.0)
    in_gross = _discount_already_in_gross(ctx, claims)
    df["_allow"] = _num(_col(df, "gross_amount")).fillna(0.0) - _num(_col(df, "patient_share")).fillna(0.0) - (
        0.0 if in_gross else _num(_col(df, "discount")).fillna(0.0))
    df["_limit"] = (df["_allow"] - df["settled"]).clip(lower=0.0)
    df["_over"] = df["_paid_after"] - df["_limit"]
    hits = df[(df["_paid_after"] > tol) & (df["_over"] > tol) & ~df["claim_sk"].isin(in_recovery)]
    out = []
    for r in _rows(hits):
        recoverable = min(float(r._over), float(r.settled), float(r._paid_after))
        plain = (
            f"A liable third party settled {aed(r.settled, 2)} on {plain_date(r.settled_on)}, yet we paid "
            f"{aed(r._paid_after, 2)} afterwards without deducting it; at most {aed(r._limit, 2)} was still due."
        )
        out.append(_sig(
            ctx, control, subject_type="claim", subject_id=r.claim_sk,
            fact_key=f"tps:{r.claim_sk}", claim_ids=[r.claim_sk],
            event_time=r._sdate, period=period_bucket(r._sdate),
            evidence={
                "plain_language": plain,
                "what_the_reviewer_must_verify": "Confirm the settlement amount and date, and the recovery "
                                                 "policy, before starting recovery.",
                "member_sk": r.member_sk, "provider_sk": r.provider_sk,
                "third_party_settlement_aed": round(float(r.settled), 2),
                "settlement_date": _date_str(r.settled_on),
                "paid_after_settlement_aed": round(float(r._paid_after), 2),
                "still_due_after_offset_aed": round(float(r._limit), 2),
                "recovery_already_open": False,
            },
            exposure=_exp.line_edit_exposure(float(r._paid_after), float(r._paid_after) - recoverable),
        ))
    return out


# ===========================================================================
# PAY-11 — internal adjudication, override and payment manipulation
# ===========================================================================


def _events(ctx) -> pd.DataFrame:
    ev = _tbl(ctx, "adjudication_event")
    if ev.empty or not {"claim_sk", "actor"} <= set(ev.columns):
        return pd.DataFrame()
    ev["claim_sk"] = ev["claim_sk"].astype(str)
    ev["_actor"] = ev["actor"].astype(str)
    ev["_role"] = _up(_col(ev, "actor_role"))
    ev["_type"] = _up(_col(ev, "event_type"))
    ev["_decision"] = _up(_col(ev, "decision"))
    ev["_reason"] = _col(ev, "override_reason").fillna("").astype(str).str.strip()
    ev["_edit"] = _up(_col(ev, "system_edit_result"))
    ev["_before"] = _num(_col(ev, "amount_before"))
    ev["_after"] = _num(_col(ev, "amount_after"))
    ev["_time"] = _dt(_col(ev, "event_time"))
    ev["_auto"] = _up(ev["_actor"]).isin(_AUTO_ACTORS) | ev["_role"].isin(_AUTO_ACTORS)
    return ev


def _is_override(ctx, ev: pd.DataFrame) -> pd.Series:
    fail_terms = [t.upper() for t in _as_list(ctx.cfg("pay11_edit_fail_results"))]
    edit_failed = ev["_edit"].map(lambda v: any(t in v for t in fail_terms) if v else False)
    approved = ~ev["_decision"].isin(_DENIED) & (ev["_decision"] != "")
    return ev["_type"].str.contains("OVERRIDE") | (edit_failed & approved & ~ev["_auto"])


@_register("pay_11_r01_unauthorised_override")
def pay_11_r01_unauthorised_override(ctx, control) -> list:
    """PAY-11-R01 — a denial or edit overridden without the required role, approval or reason."""
    claims = _claims(ctx)
    ev = _events(ctx)
    if claims.empty or ev.empty:
        return []
    roles = [r.upper() for r in _as_list(ctx.cfg("pay11_authorised_override_roles"))]
    exempt = _as_list(ctx.cfg("pay11_override_exemption_terms"))
    ov = ev[_is_override(ctx, ev) & ev["claim_sk"].isin(set(claims["claim_sk"]))].copy()
    if ov.empty:
        return []
    ov["_no_reason"] = ov["_reason"] == ""
    ov["_bad_role"] = ~ov["_role"].isin(roles) if roles else False
    ov["_exempt"] = _contains_any(ov["_reason"], exempt) if exempt else False
    hits = ov[(ov["_no_reason"] | ov["_bad_role"]) & ~ov["_exempt"]]
    head = claims.set_index("claim_sk")
    out = []
    for r in _rows(hits):
        me = head.loc[r.claim_sk]
        problems = []
        if r._no_reason:
            problems.append("no reason was recorded")
        if r._bad_role:
            problems.append(f"the role '{(r._role or 'not recorded').lower()}' is not authorised to override")
        before, after = _f(r._before), _f(r._after)
        plain = (
            f"A refusal or payment check on this claim was overridden by adjudicator {r._actor} on "
            f"{plain_date(r._time)} ({aed(before, 2)} → {aed(after, 2)}), but {' and '.join(problems)}."
        )
        ev_sk = _s(getattr(r, "event_sk", "")) or f"{r.claim_sk}:{r._actor}:{_date_str(r._time)}"
        out.append(_sig(
            ctx, control, subject_type="claim", subject_id=r.claim_sk,
            fact_key=f"override:{ev_sk}", claim_ids=[r.claim_sk],
            event_time=me.get("_sdate"), period=period_bucket(me.get("_sdate")),
            evidence={
                "plain_language": plain,
                "what_the_reviewer_must_verify": "Identify the adjudicator's authority and check for an "
                                                 "emergency or delegated-authority record.",
                "member_sk": me.get("member_sk"), "provider_sk": me.get("provider_sk"),
                "adjudicator_id": r._actor, "actor_role": r._role or None, "event_sk": ev_sk,
                "system_edit_result": r._edit or None, "override_reason": r._reason or None,
                "amount_before_aed": round(before, 2), "amount_after_aed": round(after, 2),
                "problems": problems,
            },
            exposure=_exp.line_edit_exposure(after, before) if after > before else
            _exp.no_exposure("the override did not raise the amount"),
        ))
    return out


@_register("pay_11_r02_adjudicator_provider_concentration")
def pay_11_r02_adjudicator_provider_concentration(ctx, control) -> list:
    """PAY-11-R02 — an adjudicator overrides one provider's claims far more than colleagues do."""
    claims = _claims(ctx)
    ev = _events(ctx)
    if claims.empty or ev.empty:
        return []
    min_ov = int(ctx.cfg("pay11_concentration_min_overrides"))
    ratio_cut = float(ctx.cfg("pay11_concentration_ratio"))
    ev = ev[~ev["_auto"]].merge(claims[["claim_sk", "provider_sk", "_sdate"]], on="claim_sk", how="inner")
    if ev.empty:
        return []
    ev["_ov"] = _is_override(ctx, ev).astype(float)
    pair = ev.groupby(["_actor", "provider_sk"]).agg(n=("_ov", "size"), k=("_ov", "sum")).reset_index()
    actor_tot = ev.groupby("_actor").agg(an=("_ov", "size"), ak=("_ov", "sum"))
    prov_tot = ev.groupby("provider_sk").agg(pn=("_ov", "size"), pk=("_ov", "sum"))
    pair = pair.join(actor_tot, on="_actor").join(prov_tot, on="provider_sk")
    overall = float(ev["_ov"].mean()) if len(ev) else 0.0
    out = []
    for r in _rows(pair[pair["k"] >= min_ov]):
        rate = float(r.k) / float(r.n)
        colleagues_n, colleagues_k = float(r.pn - r.n), float(r.pk - r.k)
        elsewhere_n, elsewhere_k = float(r.an - r.n), float(r.ak - r.k)
        peer_rate = colleagues_k / colleagues_n if colleagues_n > 0 else np.nan
        own_rate = elsewhere_k / elsewhere_n if elsewhere_n > 0 else np.nan
        if np.isnan(peer_rate):
            continue  # declared exclusion: an assigned provider book has no colleague comparison
        base = max(peer_rate, 0.0 if np.isnan(own_rate) else own_rate, overall)
        if base > 0 and rate < ratio_cut * base:
            continue
        sub = ev[(ev["_actor"] == r._actor) & (ev["provider_sk"] == r.provider_sk) & (ev["_ov"] > 0)]
        plain = (
            f"Adjudicator {r._actor} overrode {pct(rate)} of the decisions they made on this provider's claims "
            f"({int(r.k)} of {int(r.n)}); colleagues overrode {pct(peer_rate)} of theirs on the same provider "
            f"and this adjudicator {pct(own_rate)} elsewhere."
        )
        out.append(_sig(
            ctx, control, subject_type="adjudicator", subject_id=r._actor,
            fact_key=f"adj_conc:{r._actor}:{r.provider_sk}", claim_ids=sorted(set(sub["claim_sk"])),
            event_time=sub["_sdate"].max(), period=period_bucket(sub["_sdate"].max()), confidence=0.8,
            evidence={
                "plain_language": plain,
                "what_the_reviewer_must_verify": "Check whether the provider is assigned to this adjudicator "
                                                 "and review a sample of the overrides.",
                "adjudicator_id": r._actor, "provider_sk": r.provider_sk,
                "decisions_on_provider": int(r.n), "overrides_on_provider": int(r.k),
                "override_rate": round(rate, 4), "colleague_override_rate_same_provider": round(peer_rate, 4),
                "own_override_rate_other_providers": None if np.isnan(own_rate) else round(own_rate, 4),
                "ratio_threshold": ratio_cut,
            },
            exposure=_exp.no_exposure("a concentration pattern does not show that any single override was "
                                      "wrong"),
        ))
    return out


@_register("pay_11_r03_post_settlement_increase")
def pay_11_r03_post_settlement_increase(ctx, control) -> list:
    """PAY-11-R03 — paid amount increased after settlement without a valid reason or workflow."""
    claims = _claims(ctx)
    remit = _remittance(ctx)
    if claims.empty or remit.empty:
        return []
    tol = float(ctx.cfg("pay11_upward_adjustment_tolerance_aed"))
    valid_terms = _as_list(ctx.cfg("pay11_valid_adjustment_terms"))
    r = remit[remit["_settle"].notna()]
    first = r[r["_pay"] > 0].groupby("claim_sk")["_settle"].min().rename("_first")
    r = r.join(first, on="claim_sk")
    is_adj = r["_decision"].str.contains("ADJUST") | (r["_adj"] < -tol)
    r = r.assign(_up=np.where(r["_adj"] < -tol, -r["_adj"], r["_pay"]))
    later_up = r[(r["_settle"] > r["_first"]) & is_adj & (r["_pay"] > tol) & (r["_up"] > tol)]
    if later_up.empty:
        return []
    # valid workflow evidence: an adjudication event or a claim version naming appeal / true-up
    justified: set[str] = set()
    ev = _events(ctx)
    if not ev.empty and valid_terms:
        text = ev["_type"] + " " + ev["_reason"]
        justified |= set(ev.loc[_contains_any(text, valid_terms), "claim_sk"])
    cv = _versions(ctx)
    if not cv.empty and valid_terms:
        text = cv["_rel"] + " " + cv["_type"]
        hit = cv[_contains_any(text, valid_terms)]
        justified |= set(hit["claim_sk"]) | set(hit["_prior"].dropna())
    first_paid = r[r["_settle"] == r["_first"]].groupby("claim_sk")["_pay"].sum()
    head = claims.set_index("claim_sk")
    out = []
    for cs, sub in later_up.groupby("claim_sk"):
        if cs in justified or cs not in head.index:
            continue
        me = head.loc[cs]
        inc = float(sub["_up"].sum())
        base = float(first_paid.get(cs, 0.0))
        plain = (
            f"This claim was settled at {aed(base, 2)} on {plain_date(sub['_first'].iloc[0])}, then increased by "
            f"{aed(inc, 2)} on {plain_date(sub['_settle'].max())} with no appeal or contract true-up on record."
        )
        out.append(_sig(
            ctx, control, subject_type="claim", subject_id=cs,
            fact_key=f"upward_adj:{cs}", claim_ids=[cs],
            event_time=me.get("_sdate"), period=period_bucket(me.get("_sdate")),
            evidence={
                "plain_language": plain,
                "what_the_reviewer_must_verify": "Find the adjustment record and its reason; refer to "
                                                 "internal audit if none exists.",
                "member_sk": me.get("member_sk"), "provider_sk": me.get("provider_sk"),
                "first_settlement_date": _date_str(sub["_first"].iloc[0]),
                "first_settlement_paid_aed": round(base, 2),
                "upward_adjustments": [
                    {"remittance_sk": _s(x.get("remittance_sk")) or None, "date": _date_str(x["_settle"]),
                     "increase_aed": round(float(x["_up"]), 2),
                     "payment_reference": x["_ref"] or None}
                    for _, x in sub.iterrows()
                ],
                "increase_aed": round(inc, 2),
            },
            exposure=_exp.line_edit_exposure(base + inc, base),
        ))
    return out


@_register("pay_11_r04_invariant_full_pay")
def pay_11_r04_invariant_full_pay(ctx, control) -> list:
    """PAY-11-R04 — an adjudicator/provider pair approves in full, uniformly, despite comparable edits."""
    claims = _claims(ctx)
    ev = _events(ctx)
    if claims.empty or ev.empty:
        return []
    min_n = int(ctx.cfg("pay11_fullpay_min_events"))
    uplift = float(ctx.cfg("pay11_fullpay_rate_uplift"))
    max_std = float(ctx.cfg("pay11_fullpay_max_ratio_std"))
    full_tol = float(ctx.cfg("pay11_fullpay_ratio_tolerance"))
    edit_terms = [t.upper() for t in _as_list(ctx.cfg("pay11_edit_hit_results"))]
    ev = ev[~ev["_auto"]].merge(claims[["claim_sk", "provider_sk", "_sdate"]], on="claim_sk", how="inner")
    ev = ev[ev["_edit"].map(lambda v: any(t in v for t in edit_terms) if v else False)]
    ev = ev[ev["_before"] > 0]
    if ev.empty:
        return []
    ev["_ratio"] = (ev["_after"] / ev["_before"]).clip(lower=0.0)
    ev["_full"] = ((ev["_ratio"] >= 1.0 - full_tol) & ~ev["_decision"].isin(_DENIED)).astype(float)
    pair = ev.groupby(["_actor", "provider_sk"]).agg(n=("_full", "size"), k=("_full", "sum"),
                                                     sd=("_ratio", "std")).reset_index()
    tot_k, tot_n = float(ev["_full"].sum()), float(len(ev))
    out = []
    for r in _rows(pair[pair["n"] >= min_n]):
        rate = float(r.k) / float(r.n)
        peer_n = tot_n - r.n
        peer_rate = (tot_k - r.k) / peer_n if peer_n > 0 else np.nan
        if np.isnan(peer_rate) or rate - peer_rate < uplift:
            continue
        sd = 0.0 if is_missing(r.sd) else float(r.sd)
        if sd > max_std:
            continue
        sub = ev[(ev["_actor"] == r._actor) & (ev["provider_sk"] == r.provider_sk)]
        plain = (
            f"Adjudicator {r._actor} paid {pct(rate)} of this provider's claims in full ({int(r.k)} of {int(r.n)}) "
            f"even though a payment check had flagged each one; on flagged claims elsewhere the full-payment "
            f"rate is {pct(peer_rate)}."
        )
        out.append(_sig(
            ctx, control, subject_type="adjudicator", subject_id=r._actor,
            fact_key=f"full_pay:{r._actor}:{r.provider_sk}", claim_ids=sorted(set(sub["claim_sk"])),
            event_time=sub["_sdate"].max(), period=period_bucket(sub["_sdate"].max()), confidence=0.8,
            evidence={
                "plain_language": plain,
                "what_the_reviewer_must_verify": "Review a sample of the approved flagged claims; "
                                                 "automatically assessed claims are already excluded.",
                "adjudicator_id": r._actor, "provider_sk": r.provider_sk,
                "flagged_decisions": int(r.n), "paid_in_full": int(r.k), "full_pay_rate": round(rate, 4),
                "peer_full_pay_rate_on_flagged_claims": round(peer_rate, 4),
                "approval_ratio_std": round(sd, 4),
            },
            exposure=_exp.no_exposure("uniform approval is a control-weakness pattern; no single payment is "
                                      "shown to be wrong"),
        ))
    return out


# ===========================================================================
# PAY-12 — reversal, refund, credit-balance and remittance leakage
# ===========================================================================


@_register("pay_12_r01_paid_cancelled_claim")
def pay_12_r01_paid_cancelled_claim(ctx, control) -> list:
    """PAY-12-R01 — a net payment remains after the claim was cancelled or reversed."""
    claims = _claims(ctx)
    remit = _remittance(ctx)
    cv = _versions(ctx)
    if claims.empty or remit.empty or cv.empty:
        return []
    tol_days = int(ctx.cfg("pay12_cancellation_timing_tolerance_days"))
    tol_amt = float(ctx.cfg("pay12_duplicate_payment_tolerance_aed"))
    canc = cv[cv["_rel"].isin(_CANCEL) | cv["_type"].isin(_CANCEL)].copy()
    if canc.empty:
        return []
    canc["_target"] = canc["_prior"].fillna(canc["claim_sk"])
    canc = canc.sort_values("_at").drop_duplicates("_target")
    end = _data_end(ctx, (remit, "settlement_date"), (cv, "recorded_at"))
    net = remit.groupby("claim_sk")["_pay"].sum()
    head = claims.set_index("claim_sk")
    out = []
    for r in _rows(canc):
        target = r._target
        if target not in head.index:
            continue
        paid = float(net.get(target, 0.0))
        # a separate cancellation record may itself carry the reversal
        if r.claim_sk != target:
            paid += float(net.get(r.claim_sk, 0.0)) if r.claim_sk not in head.index else 0.0
        if paid <= tol_amt:
            continue
        if end is not None and not pd.isna(r._at) and (end - r._at).days < tol_days:
            continue  # declared exclusion: timing tolerance / netting in a later run
        me = head.loc[target]
        plain = (
            f"This claim was cancelled on {plain_date(r._at)}, yet {aed(paid, 2)} paid against it has not been "
            f"reversed as of {plain_date(end)}."
        )
        out.append(_sig(
            ctx, control, subject_type="claim", subject_id=target,
            fact_key=f"cancel_paid:{target}", claim_ids=[target],
            event_time=me.get("_sdate"), period=period_bucket(me.get("_sdate")),
            evidence={
                "plain_language": plain,
                "what_the_reviewer_must_verify": "Confirm the cancellation and payment status and whether "
                                                 "the payment is due to be netted in a later run.",
                "member_sk": me.get("member_sk"), "provider_sk": me.get("provider_sk"),
                "cancellation_recorded": _date_str(r._at), "cancellation_record_claim_sk": r.claim_sk,
                "net_paid_aed": round(paid, 2), "as_of": _date_str(end), "timing_tolerance_days": tol_days,
            },
            exposure=_exp.line_edit_exposure(paid, 0.0),
        ))
    return out


@_register("pay_12_r02_duplicate_payment")
def pay_12_r02_duplicate_payment(ctx, control) -> list:
    """PAY-12-R02 — same line paid more than once, or a payment reference reused inconsistently."""
    claims = _claims(ctx)
    remit = _remittance(ctx)
    lines = _tbl(ctx, "claim_line")
    if claims.empty or remit.empty:
        return []
    tol = float(ctx.cfg("pay12_duplicate_payment_tolerance_aed"))
    head = claims.set_index("claim_sk")
    adjust_rows = remit["_decision"].str.contains("ADJUST") | (remit["_adj"] < -tol)
    pay = remit[(remit["_pay"] > 0) & ~adjust_rows & ~remit["_denied"]].copy()
    pay["_key"] = pay["_line"].map(lambda v: None if is_missing(v) else str(v))
    pay["_key"] = pay["_key"].fillna("claim:" + pay["claim_sk"])
    payable: dict[str, float] = {}
    if not lines.empty and "line_sk" in lines.columns:
        amt = _num(_col(lines, "net_amount")).fillna(_num(_col(lines, "gross_amount")))
        payable = dict(zip(lines["line_sk"].astype(str), amt))
    claim_payable = _num(_col(claims, "net_amount")).fillna(_num(_col(claims, "gross_amount")))
    claim_payable = dict(zip(claims["claim_sk"], claim_payable))
    out = []
    counts = pay["_key"].value_counts()
    multi = pay[pay["_key"].isin(set(counts[counts >= 2].index))]
    for key, sub in multi.groupby("_key"):
        cs = sub["claim_sk"].iloc[0]
        if cs not in head.index:
            continue
        limit = payable.get(key) if not key.startswith("claim:") else claim_payable.get(cs)
        total = float(sub["_pay"].sum())
        if limit is not None and not is_missing(limit) and total <= float(limit) + tol:
            continue  # declared exclusion: split settlement within the payable amount
        me = head.loc[cs]
        amounts = sub["_pay"].round(2).tolist()
        plain = (
            f"The same {'line' if not key.startswith('claim:') else 'claim'} was paid {len(sub)} times "
            f"({', '.join(aed(a, 2) for a in amounts)}), {aed(total, 2)} in total against "
            f"{aed(limit, 2) if limit is not None and not is_missing(limit) else 'a single'} payable amount."
        )
        out.append(_sig(
            ctx, control, subject_type="claim", subject_id=cs,
            fact_key=f"dup_pay:{key}", claim_ids=[cs],
            event_time=me.get("_sdate"), period=period_bucket(me.get("_sdate")),
            evidence={
                "plain_language": plain,
                "what_the_reviewer_must_verify": "Compare the payment records and check for a split "
                                                 "settlement or batch payment.",
                "member_sk": me.get("member_sk"), "provider_sk": me.get("provider_sk"),
                "line_sk": None if key.startswith("claim:") else key, "limb": "paid_more_than_once",
                "payments": [{"remittance_sk": _s(x.get("remittance_sk")) or None,
                              "payment_aed": round(float(x["_pay"]), 2), "date": _date_str(x["_settle"]),
                              "payment_reference": x["_ref"] or None} for _, x in sub.iterrows()],
                "payable_aed": None if limit is None or is_missing(limit) else round(float(limit), 2),
            },
            exposure=_exp.duplicate_exposure(amounts),
        ))
    # payment reference reused inconsistently: the same reference on different providers or dates
    refs = remit[(remit["_ref"] != "") & (remit["_pay"] > 0)].merge(
        claims[["claim_sk", "provider_sk"]], on="claim_sk", how="inner")
    if not refs.empty:
        rg = refs.groupby("_ref").agg(providers=("provider_sk", "nunique"), dates=("_settle", "nunique"))
        bad = rg[(rg["providers"] > 1) | (rg["dates"] > 1)]
        for ref, info in bad.iterrows():
            sub = refs[refs["_ref"] == ref]
            cids = sorted(set(sub["claim_sk"]))
            cs = cids[-1]
            if cs not in head.index:
                continue
            me = head.loc[cs]
            plain = (
                f"Payment reference {ref} was used for {count_phrase(len(sub), 'payment')} across "
                f"{count_phrase(int(info['providers']), 'provider')} and {count_phrase(int(info['dates']), 'settlement date')}; "
                f"one reference should identify one remittance."
            )
            out.append(_sig(
                ctx, control, subject_type="claim", subject_id=cs,
                fact_key=f"dup_ref:{ref}", claim_ids=cids,
                event_time=me.get("_sdate"), period=period_bucket(me.get("_sdate")), confidence=0.7,
                evidence={
                    "plain_language": plain,
                    "what_the_reviewer_must_verify": "Check with finance whether the reference belongs to a "
                                                     "batch payment.",
                    "member_sk": me.get("member_sk"), "provider_sk": me.get("provider_sk"),
                    "payment_reference": ref, "limb": "payment_reference_reused",
                    "payments": [{"claim_sk": x["claim_sk"], "payment_aed": round(float(x["_pay"]), 2),
                                  "date": _date_str(x["_settle"])} for _, x in sub.iterrows()],
                },
                exposure=_exp.no_exposure("a reused reference is a reconciliation fault; which payment, if "
                                          "any, is surplus is not yet known"),
            ))
    return out


@_register("pay_12_r03_unapplied_refund")
def pay_12_r03_unapplied_refund(ctx, control) -> list:
    """PAY-12-R03 — a provider refund or credit not applied to an outstanding recovery within the SLA."""
    ledger = _tbl(ctx, "recovery_ledger")
    if ledger.empty or "credit_amount" not in ledger.columns:
        return []
    sla = int(ctx.cfg("pay12_refund_application_sla_days"))
    dispute_terms = _as_list(ctx.cfg("pay12_refund_hold_terms"))
    ledger["_amt"] = _num(ledger["credit_amount"]).fillna(0.0)
    ledger["_cd"] = _dt(_col(ledger, "credit_date"))
    ledger["_ad"] = _dt(_col(ledger, "applied_date"))
    ledger["_applied"] = _col(ledger, "applied").map(lambda v: str(v).strip().lower() in ("true", "1", "yes", "y"))
    end = _data_end(ctx, (ledger, "credit_date"), (ledger, "applied_date"))
    if end is None:
        return []
    age = (end - ledger["_cd"]).dt.days
    late_applied = ledger["_applied"] & ledger["_ad"].notna() & ((ledger["_ad"] - ledger["_cd"]).dt.days > sla)
    unapplied = ~ledger["_applied"] & (age > sla)
    hold = pd.Series(False, index=ledger.index)
    for c in ("status", "note", "hold_reason"):
        if c in ledger.columns and dispute_terms:
            hold |= _contains_any(ledger[c].astype(str), dispute_terms)
    hits = ledger[(ledger["_amt"] > 0) & (unapplied | late_applied) & ~hold]
    claims = _claims(ctx)
    head = claims.set_index("claim_sk") if not claims.empty else pd.DataFrame()
    out = []
    for r in _rows(hits):
        cs = _s(getattr(r, "claim_sk", ""))
        prov = _s(getattr(r, "provider_sk", ""))
        if not prov and cs and cs in head.index:
            prov = _s(head.loc[cs].get("provider_sk"))
        lsk = _s(getattr(r, "ledger_sk", "")) or f"{prov}:{cs}:{_date_str(r._cd)}"
        if r._applied:
            days = int((r._ad - r._cd).days)
            plain = (f"A provider credit of {aed(r._amt, 2)} received on {plain_date(r._cd)} was applied only "
                     f"after {days} days, beyond the {sla}-day limit.")
        else:
            days = int((end - r._cd).days)
            plain = (f"A provider credit of {aed(r._amt, 2)} received on {plain_date(r._cd)} has not been applied "
                     f"to any recovery after {days} days; the limit is {sla} days.")
        sdate = r._cd
        out.append(_sig(
            ctx, control, subject_type="provider" if prov else "claim", subject_id=prov or cs,
            fact_key=f"credit:{lsk}", claim_ids=[cs] if cs else [],
            event_time=sdate, period=period_bucket(sdate),
            evidence={
                "plain_language": plain,
                "what_the_reviewer_must_verify": "Find the recovery the credit settles and check whether it "
                                                 "is under dispute or in the unapplied-cash queue.",
                "provider_sk": prov or None, "claim_sk": cs or None, "ledger_sk": lsk,
                "credit_amount_aed": round(float(r._amt), 2), "credit_date": _date_str(r._cd),
                "applied": bool(r._applied), "applied_date": _date_str(r._ad), "days_outstanding": days,
                "sla_days": sla, "as_of": _date_str(end),
            },
            exposure=_exp.no_exposure("the money has been received; the risk is that the recovery it settles "
                                      "is still shown as open or is never closed"),
        ))
    return out


@_register("pay_12_r04_offset_manipulation")
def pay_12_r04_offset_manipulation(ctx, control) -> list:
    """PAY-12-R04 — provider offsets credits against unrelated claims, bills negative lines, or delays reversals."""
    claims = _claims(ctx)
    remit = _remittance(ctx)
    if claims.empty or remit.empty:
        return []
    min_events = int(ctx.cfg("pay12_offset_min_events"))
    ratio_cut = float(ctx.cfg("pay12_offset_rate_ratio"))
    delay_ratio = float(ctx.cfg("pay12_reversal_delay_ratio"))
    tol = float(ctx.cfg("pay12_duplicate_payment_tolerance_aed"))
    rm = remit.merge(claims[["claim_sk", "provider_sk", "_sdate"]], on="claim_sk", how="inner")
    if rm.empty:
        return []
    # limb 1: a negative remittance larger than anything ever paid on that claim/line — the credit
    # is being taken against a claim it does not belong to
    key = rm["_line"].map(lambda v: None if is_missing(v) else str(v)).fillna("claim:" + rm["claim_sk"])
    rm = rm.assign(_key=key)
    pos = rm[rm["_pay"] > 0].groupby("_key")["_pay"].sum()
    neg = rm[rm["_pay"] < 0].copy()
    lines_all = _tbl(ctx, "claim_line")
    if not lines_all.empty and {"line_sk", "net_amount"} <= set(lines_all.columns):
        # the settlement of a negative billed line is limb 2, not a misapplied credit
        neg_line_sks = set(lines_all.loc[_num(lines_all["net_amount"]) < 0, "line_sk"].astype(str))
        neg = neg[~neg["_key"].isin(neg_line_sks)]
    neg["_neg"] = neg["_pay"]
    neg["_paid_before"] = neg["_key"].map(pos).fillna(0.0)
    neg["_unrelated"] = (neg["_neg"].abs() > neg["_paid_before"] + tol).astype(float)
    per_claims = rm.groupby("provider_sk")["claim_sk"].nunique()
    unrel = neg[neg["_unrelated"] > 0].groupby("provider_sk")
    unrel_n = unrel.size()
    # limb 2: negative billed lines used to reduce claim totals
    lines = _tbl(ctx, "claim_line")
    neg_lines = pd.Series(dtype=float)
    neg_line_claims: dict[Any, list[str]] = {}
    if not lines.empty and {"claim_sk", "net_amount"} <= set(lines.columns):
        ls = lines.copy()
        ls["claim_sk"] = ls["claim_sk"].astype(str)
        ls = ls.merge(claims[["claim_sk", "provider_sk"]], on="claim_sk", how="inner")
        nl = ls[(_num(ls["net_amount"]) < 0) | (_num(_col(ls, "gross_amount")) < 0)]
        neg_lines = nl.groupby("provider_sk")["claim_sk"].nunique()
        neg_line_claims = nl.groupby("provider_sk")["claim_sk"].agg(lambda s: sorted(set(s))).to_dict()
    # limb 3: days from cancellation to reversal, versus peers
    delay = pd.Series(dtype=float)
    cv = _versions(ctx)
    if not cv.empty:
        canc = cv[cv["_rel"].isin(_CANCEL)].copy()
        canc["_target"] = canc["_prior"].fillna(canc["claim_sk"])
        firstneg = neg.groupby("claim_sk")["_settle"].min().rename("_rev")
        canc = canc.merge(firstneg, left_on="_target", right_index=True, how="inner")
        canc = canc[canc["_rev"].notna() & canc["_at"].notna()]
        if not canc.empty:
            canc = canc.merge(claims[["claim_sk", "provider_sk"]], left_on="_target", right_on="claim_sk",
                              how="inner", suffixes=("", "_c"))
            canc["_days"] = (canc["_rev"] - canc["_at"]).dt.days.clip(lower=0)
            delay = canc.groupby("provider_sk")["_days"].median()
    providers = set(unrel_n.index) | set(neg_lines.index) | set(delay.index)
    out = []
    total_claims = float(per_claims.sum()) if len(per_claims) else 0.0
    for prov in providers:
        limbs, facts = [], {}
        n_claims = float(per_claims.get(prov, 0))
        u = int(unrel_n.get(prov, 0))
        if u >= min_events and n_claims > 0:
            rate = u / n_claims
            peer = (float(unrel_n.sum()) - u) / max(total_claims - n_claims, 1.0)
            if peer == 0 or rate >= ratio_cut * peer:
                limbs.append("credits_offset_against_unrelated_claims")
                facts["unrelated_offsets"] = u
                facts["unrelated_offset_rate"] = round(rate, 4)
                facts["peer_unrelated_offset_rate"] = round(peer, 6)
        nl_n = int(neg_lines.get(prov, 0))
        if nl_n >= min_events and n_claims > 0:
            rate = nl_n / n_claims
            peer = (float(neg_lines.sum()) - nl_n) / max(total_claims - n_claims, 1.0)
            if peer == 0 or rate >= ratio_cut * peer:
                limbs.append("negative_billed_lines")
                facts["claims_with_negative_lines"] = nl_n
                facts["negative_line_rate"] = round(rate, 4)
                facts["peer_negative_line_rate"] = round(peer, 6)
        if prov in delay.index and len(delay) > 1:
            d = float(delay[prov])
            peer_d = float(delay.drop(prov).median())
            if peer_d > 0 and d >= delay_ratio * peer_d:
                limbs.append("slow_reversals")
                facts["median_days_to_reverse"] = d
                facts["peer_median_days_to_reverse"] = peer_d
        if not limbs:
            continue
        cids = set()
        if "credits_offset_against_unrelated_claims" in limbs:
            cids |= set(neg.loc[(neg["provider_sk"] == prov) & (neg["_unrelated"] > 0), "claim_sk"])
        if "negative_billed_lines" in limbs:
            cids |= set(neg_line_claims.get(prov, []))
        sub = claims[claims["claim_sk"].isin(cids)]
        when = sub["_sdate"].max() if not sub.empty else None
        parts = []
        if "credits_offset_against_unrelated_claims" in limbs:
            parts.append(f"{count_phrase(facts['unrelated_offsets'], 'credit')} were taken against claims they "
                         f"did not belong to ({pct(facts['unrelated_offset_rate'])} of its claims; peers "
                         f"{pct(facts['peer_unrelated_offset_rate'])})")
        if "negative_billed_lines" in limbs:
            parts.append(f"{count_phrase(facts['claims_with_negative_lines'], 'claim')} carry negative lines that "
                         f"reduce the claim total ({pct(facts['negative_line_rate'])} of its claims; peers "
                         f"{pct(facts['peer_negative_line_rate'])})")
        if "slow_reversals" in limbs:
            parts.append(f"reversals take a median of {facts['median_days_to_reverse']:.0f} days against "
                         f"{facts['peer_median_days_to_reverse']:.0f} for peers")
        plain = "For this provider, " + "; ".join(parts) + "."
        out.append(_sig(
            ctx, control, subject_type="provider", subject_id=prov,
            fact_key=f"offsets:{prov}", claim_ids=sorted(cids),
            event_time=when, period=period_bucket(when), confidence=0.75,
            evidence={
                "plain_language": plain,
                "what_the_reviewer_must_verify": "Check the contract's netting rules, then review the offsets "
                                                 "and reversal timings.",
                "provider_sk": prov, "limbs": limbs, "claims_with_remittance": int(n_claims), **facts,
            },
            exposure=_exp.no_exposure("an offset pattern shows misallocated credits, not an amount "
                                      "overpaid"),
        ))
    return out
