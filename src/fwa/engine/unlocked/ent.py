"""Unlocked implementations for the ENT family.

Each function here runs only when a dataset unlock (`rules/unlocks/ENT.yaml`)
is satisfied by the loaded file. See :mod:`fwa.engine.unlocks`.

Conventions shared by every function in this module (and by :mod:`.pol`, which
imports the helpers below):

* Every table is read through :func:`_table`, which filters to the run's
  tenant when the table carries a ``tenant_id`` column and returns an empty
  frame, never ``None``.
* Every threshold is a governed parameter read through ``ctx.cfg``.
* Vocabulary (what counts as an inpatient encounter, a telehealth visit, an
  emergency) is matched case-insensitively on words, because two sources rarely
  spell a code value the same way. The word lists are vocabulary, not
  thresholds.
* Every public function is wrapped by :func:`_safe`: a missing column or an
  unexpected value makes the control return nothing rather than raise. The
  unwrapped function stays reachable as ``fn.__wrapped__`` so the unit tests
  see real errors.
"""

from __future__ import annotations

import datetime as _dt
import functools
import json
from typing import Any, Callable, Iterable

import numpy as np
import pandas as pd

from ...cases import exposure as _exp
from ...presentation import aed, count_phrase, pct, plain_date
from ..controls import _f, _sig
from ..evallib import is_missing, period_bucket

IMPLEMENTATIONS: dict[str, Callable] = {}

# ---------------------------------------------------------------------------
# vocabulary
# ---------------------------------------------------------------------------

_INPATIENT_WORDS = ("INPATIENT", "ICU", "INTENSIVE", "DAYCASE", "DAY CASE", "DAY-CASE", "DAY_CASE")
_INPATIENT_EXACT = {"IP", "I", "DC", "ADMISSION", "ADMITTED"}
_OUTPATIENT_WORDS = ("OUTPATIENT", "CLINIC", "AMBULATORY", "OFFICE")
_OUTPATIENT_EXACT = {"OP", "O"}
_TELE_WORDS = ("TELE", "VIRTUAL", "REMOTE", "VIDEO")
_EMERGENCY_WORDS = ("EMERG",)
_EMERGENCY_EXACT = {"ER", "ED", "A&E"}
_HOME_MOBILE_WORDS = ("HOME", "MOBILE")
_TRANSFER_WORDS = ("TRANSFER",)
_VOID_STATUS_WORDS = ("CANCEL", "VOID", "REJECT", "DECLIN", "RESCIND")
_DENIED_WORDS = ("DENI", "DENY", "REJECT", "REVERS", "RECOUP")
_APPROVED_WORDS = ("APPROV", "GRANT", "ACCEPT")
_PENDING_WORDS = ("PEND",)
_NON_CLINICIAN_ACTIVITY_WORDS = ("DRUG", "PHARM", "MEDICATION", "SUPPL", "CONSUM", "ROOM", "BED",
                                 "ACCOMM", "DRG", "PACKAGE", "DEVICE", "DISPENS")
_NON_INDEPENDENT_ROLE_WORDS = ("NURSE", "TECHNICIAN", "ASSISTANT", "TRAINEE", "INTERN", "STUDENT",
                               "AIDE", "SCRIBE")
_ANAESTHESIA_WORDS = ("ANAES", "ANES")
_PROHIBITED_PREFIXES = ("NOT:", "!", "PROHIBITED:", "EXCLUDED:", "-")
_IN_NETWORK_VALUES = {"IN_NETWORK", "IN NETWORK", "IN", "PARTICIPATING", "CONTRACTED"}


def _safe(fn: Callable) -> Callable:
    """Never let an exception escape a control: return ``[]`` instead."""

    @functools.wraps(fn)
    def wrapper(ctx, control):
        try:
            return fn(ctx, control) or []
        except Exception:  # noqa: BLE001 - a control must never crash the run
            return []

    return wrapper


def _register(name: str):
    def deco(fn: Callable) -> Callable:
        wrapped = _safe(fn)
        IMPLEMENTATIONS[name] = wrapped
        return wrapped

    return deco


# ---------------------------------------------------------------------------
# data helpers (shared with pol.py)
# ---------------------------------------------------------------------------


def _table(ctx, name: str) -> pd.DataFrame:
    ds = getattr(ctx, "dataset", None)
    if ds is None:
        return pd.DataFrame()
    try:
        frame = ds.get(name)
    except Exception:  # noqa: BLE001
        return pd.DataFrame()
    if frame is None:
        return pd.DataFrame()
    if "tenant_id" in frame.columns and frame["tenant_id"].notna().any():
        frame = frame[(frame["tenant_id"].astype(str) == str(ctx.tenant_id)) | frame["tenant_id"].isna()]
    return frame


def _d(series: Any) -> pd.Series:
    """Dates, normalised to midnight, NaT where unparseable."""
    out = pd.to_datetime(series, errors="coerce")
    try:
        if getattr(out.dt, "tz", None) is not None:
            out = out.dt.tz_localize(None)
    except Exception:  # noqa: BLE001
        pass
    return out.dt.normalize()


def _t(series: Any) -> pd.Series:
    """Timestamps (time of day kept), NaT where unparseable."""
    out = pd.to_datetime(series, errors="coerce")
    try:
        if getattr(out.dt, "tz", None) is not None:
            out = out.dt.tz_localize(None)
    except Exception:  # noqa: BLE001
        pass
    return out


def _num(series: Any) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _s(series: pd.Series) -> pd.Series:
    """String key column: NaN stays NaN, everything else becomes a stripped str."""
    return series.where(series.isna(), series.astype(str).str.strip()).replace({"": np.nan, "nan": np.nan,
                                                                                 "None": np.nan})


def _up(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.strip().str.upper()


def _has_word(series: pd.Series, words: Iterable[str], exact: Iterable[str] = ()) -> pd.Series:
    u = _up(series)
    mask = pd.Series(False, index=series.index)
    for w in words:
        mask |= u.str.contains(w, regex=False)
    exact = set(exact)
    if exact:
        mask |= u.isin(exact)
    return mask


def _truthy(series: pd.Series) -> pd.Series:
    u = _up(series)
    return u.isin({"TRUE", "T", "Y", "YES", "1", "1.0"})


def _falsy(series: pd.Series) -> pd.Series:
    u = _up(series)
    return u.isin({"FALSE", "F", "N", "NO", "0", "0.0"})


def _col(frame: pd.DataFrame, name: str, default: Any = np.nan) -> pd.Series:
    if name in frame.columns:
        return frame[name]
    return pd.Series(default, index=frame.index, dtype=object)


def _claims(ctx) -> pd.DataFrame:
    """The claim header as a tidy frame with parsed dates and AED amounts."""
    df = ctx.claims if ctx.claims is not None else pd.DataFrame()
    if df.empty or "claim_sk" not in df.columns:
        return pd.DataFrame(columns=["claim_sk", "member_sk", "provider_sk", "service_date",
                                     "discharge_date", "gross", "net", "claim_type", "encounter_sk"])
    if "tenant_id" in df.columns:
        df = df[df["tenant_id"].astype(str) == str(ctx.tenant_id)]
    out = pd.DataFrame(index=df.index)
    out["claim_sk"] = df["claim_sk"].astype(str)
    out["member_sk"] = _s(_col(df, "member_sk"))
    out["provider_sk"] = _s(_col(df, "provider_sk"))
    sd = _col(df, "service_date")
    if sd.isna().all():
        sd = _col(df, "submission_date")
    out["service_date"] = _d(sd)
    out["discharge_date"] = _d(_col(df, "discharge_date"))
    gross = _num(_col(df, "gross_amount_aed"))
    if gross.isna().all():
        gross = _num(_col(df, "gross_amount"))
    out["gross"] = gross.fillna(0.0)
    net = _num(_col(df, "net_amount"))
    out["net"] = net.fillna(out["gross"])
    out["claim_type"] = _up(_col(df, "claim_type", ""))
    out["encounter_sk"] = _s(_col(df, "encounter_sk"))
    out["submission_date"] = _d(_col(df, "submission_date"))
    out["agent_id"] = _s(_col(df, "agent_id"))
    return out.reset_index(drop=True)


def _lines(ctx, claims: pd.DataFrame | None = None) -> pd.DataFrame:
    """Claim lines with dates (falling back to the header date) and amounts."""
    lines = _table(ctx, "claim_line")
    if lines.empty or "claim_sk" not in lines.columns:
        return pd.DataFrame()
    claims = _claims(ctx) if claims is None else claims
    out = pd.DataFrame(index=lines.index)
    out["line_sk"] = _s(_col(lines, "line_sk")).fillna(pd.Series(lines.index.astype(str), index=lines.index))
    out["claim_sk"] = lines["claim_sk"].astype(str)
    out["activity_code"] = _s(_col(lines, "activity_code"))
    out["activity_type"] = _up(_col(lines, "activity_type", ""))
    out["units"] = _num(_col(lines, "units")).fillna(1.0)
    out["gross_line"] = _num(_col(lines, "gross_amount")).fillna(0.0)
    out["net_line"] = _num(_col(lines, "net_amount")).fillna(out["gross_line"])
    out["patient_share"] = _num(_col(lines, "patient_share")).fillna(0.0)
    out["rendering"] = _s(_col(lines, "rendering_clinician_id"))
    out["ordering"] = _s(_col(lines, "ordering_clinician_id"))
    out["authorization_id"] = _s(_col(lines, "authorization_id"))
    out["line_date"] = _d(_col(lines, "service_date"))
    out["start"] = _t(_col(lines, "service_start_time"))
    out["end"] = _t(_col(lines, "service_end_time"))
    out = out.merge(claims[["claim_sk", "member_sk", "provider_sk", "service_date", "gross", "net",
                            "claim_type", "encounter_sk"]], on="claim_sk", how="inner")
    out["line_date"] = out["line_date"].fillna(out["service_date"])
    return out


def _encounters(ctx) -> pd.DataFrame:
    enc = _table(ctx, "encounter")
    if enc.empty:
        return pd.DataFrame(columns=["encounter_sk", "claim_sk", "encounter_type"])
    out = pd.DataFrame(index=enc.index)
    out["encounter_sk"] = _s(_col(enc, "encounter_sk"))
    out["claim_sk"] = _s(_col(enc, "claim_sk"))
    out["encounter_type"] = _up(_col(enc, "encounter_type", ""))
    out["admission_date"] = _d(_col(enc, "admission_date"))
    out["discharge_date_enc"] = _d(_col(enc, "discharge_date"))
    out["start_time"] = _t(_col(enc, "start_time"))
    out["end_time"] = _t(_col(enc, "end_time"))
    out["bed_id"] = _s(_col(enc, "bed_id"))
    out["duration_minutes"] = _num(_col(enc, "duration_minutes"))
    out["location"] = _s(_col(enc, "location"))
    out["admission_type"] = _up(_col(enc, "admission_type", ""))
    out["discharge_type"] = _up(_col(enc, "discharge_type", ""))
    out["observation_status"] = _up(_col(enc, "observation_status", ""))
    out["enc_member_sk"] = _s(_col(enc, "member_sk"))
    return out


def _claim_setting(ctx, claims: pd.DataFrame) -> pd.DataFrame:
    """claims + their encounter (joined by claim_sk, else encounter_sk)."""
    enc = _encounters(ctx)
    base = claims.copy()
    if enc.empty:
        base["encounter_type"] = base["claim_type"]
        for c in ("admission_date", "discharge_date_enc", "bed_id", "duration_minutes", "location",
                  "admission_type", "discharge_type", "observation_status", "start_time", "end_time"):
            base[c] = np.nan
        base["has_encounter"] = False
        return base
    by_claim = enc.dropna(subset=["claim_sk"]).drop_duplicates("claim_sk")
    merged = base.merge(by_claim.drop(columns=["encounter_sk"]), on="claim_sk", how="left")
    missing = merged["encounter_type"].isna()
    if missing.any() and enc["encounter_sk"].notna().any():
        by_enc = enc.dropna(subset=["encounter_sk"]).drop_duplicates("encounter_sk").set_index("encounter_sk")
        for c in by_enc.columns:
            if c in ("claim_sk",):
                continue
            fill = merged.loc[missing, "encounter_sk"].map(by_enc[c])
            merged.loc[missing, c] = fill.values
    merged["has_encounter"] = merged["encounter_type"].notna()
    merged["encounter_type"] = merged["encounter_type"].fillna("").astype(str)
    blank = merged["encounter_type"] == ""
    merged.loc[blank, "encounter_type"] = merged.loc[blank, "claim_type"]
    return merged


def _is_inpatient(series: pd.Series) -> pd.Series:
    return _has_word(series, _INPATIENT_WORDS, _INPATIENT_EXACT)


def _is_outpatient(series: pd.Series) -> pd.Series:
    return _has_word(series, _OUTPATIENT_WORDS, _OUTPATIENT_EXACT) & ~_is_inpatient(series)


def _is_tele(series: pd.Series) -> pd.Series:
    return _has_word(series, _TELE_WORDS)


def _is_emergency(series: pd.Series) -> pd.Series:
    return _has_word(series, _EMERGENCY_WORDS, _EMERGENCY_EXACT)


def _providers(ctx) -> pd.DataFrame:
    prov = _table(ctx, "provider")
    if prov.empty or "provider_sk" not in prov.columns:
        return pd.DataFrame(columns=["provider_sk"])
    out = prov.copy()
    out["provider_sk"] = out["provider_sk"].astype(str)
    return out.drop_duplicates("provider_sk")


def _members(ctx) -> pd.DataFrame:
    mem = _table(ctx, "member")
    if mem.empty or "member_sk" not in mem.columns:
        return pd.DataFrame(columns=["member_sk"])
    out = mem.copy()
    out["member_sk"] = out["member_sk"].astype(str)
    return out.drop_duplicates("member_sk")


def _roster(ctx) -> pd.DataFrame:
    ros = _table(ctx, "clinician_roster")
    if ros.empty or "clinician_id" not in ros.columns:
        return pd.DataFrame(columns=["clinician_id", "provider_sk"])
    out = ros.copy()
    out["clinician_id"] = out["clinician_id"].astype(str).str.strip()
    out["provider_sk"] = _s(_col(out, "provider_sk"))
    return out


def _activity_ref(ctx) -> pd.DataFrame:
    ref = _table(ctx, "activity_code_reference")
    if ref.empty or "activity_code" not in ref.columns:
        return pd.DataFrame(columns=["activity_code"])
    out = ref.copy()
    out["activity_code"] = out["activity_code"].astype(str).str.strip()
    return out.drop_duplicates("activity_code")


def _age_years(dob: pd.Series, at: pd.Series) -> pd.Series:
    return (at - dob).dt.days / 365.25


def _coverage(ctx) -> pd.DataFrame:
    cov = _table(ctx, "coverage_period")
    if cov.empty or "member_sk" not in cov.columns:
        return pd.DataFrame(columns=["coverage_id", "member_sk", "vf", "vt", "status", "product", "network"])
    out = pd.DataFrame(index=cov.index)
    out["coverage_id"] = _s(_col(cov, "coverage_id")).fillna(pd.Series(cov.index.astype(str), index=cov.index))
    out["member_sk"] = cov["member_sk"].astype(str)
    out["vf"] = _d(_col(cov, "valid_from"))
    out["vt"] = _d(_col(cov, "valid_to"))
    out["status"] = _up(_col(cov, "status", ""))
    out["product"] = _s(_col(cov, "product"))
    out["network"] = _s(_col(cov, "network"))
    out["payer_id"] = _s(_col(cov, "payer_id"))
    out["agent_id"] = _s(_col(cov, "agent_id"))
    out["policy_inception_date"] = _d(_col(cov, "policy_inception_date"))
    out["void"] = _has_word(out["status"], _VOID_STATUS_WORDS)
    return out


def _claim_coverage(claims: pd.DataFrame, cov: pd.DataFrame) -> pd.DataFrame:
    """claim_sk -> the (non-void) coverage row in force on the service date."""
    live = cov[~cov["void"]]
    m = claims[["claim_sk", "member_sk", "service_date"]].merge(
        live[["coverage_id", "member_sk", "vf", "vt", "product", "network"]], on="member_sk", how="inner")
    inside = (m["service_date"] >= m["vf"]) & ((m["service_date"] <= m["vt"]) | m["vt"].isna())
    m = m[inside].sort_values(["claim_sk", "vf"]).drop_duplicates("claim_sk", keep="last")
    return m[["claim_sk", "coverage_id", "product", "network", "vf", "vt"]]


def _authorizations(ctx) -> pd.DataFrame:
    auth = _table(ctx, "authorization")
    if auth.empty:
        return pd.DataFrame(columns=["authorization_id", "auth_status"])
    key = "authorization_sk" if "authorization_sk" in auth.columns else None
    if key is None:
        return pd.DataFrame(columns=["authorization_id", "auth_status"])
    out = pd.DataFrame({"authorization_id": auth[key].astype(str), "auth_status": _up(_col(auth, "status", ""))})
    # a line may also cite the request id
    if "request_id" in auth.columns:
        alt = pd.DataFrame({"authorization_id": auth["request_id"].astype(str),
                            "auth_status": _up(_col(auth, "status", ""))})
        out = pd.concat([out, alt], ignore_index=True)
    return out.drop_duplicates("authorization_id")


def _fast_ts(value: Any) -> pd.Timestamp:
    """An ISO date string parsed without pandas' general-purpose parser; anything else falls back."""
    if isinstance(value, str) and len(value) >= 10:
        try:
            return pd.Timestamp(_dt.date.fromisoformat(value[:10]))
        except ValueError:
            pass
    return pd.to_datetime(value, errors="coerce")


def _parse_periods(value: Any) -> list[tuple[pd.Timestamp, pd.Timestamp, str]]:
    """Leave periods in any of the shapes a roster tends to use."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    items: Any = value
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        try:
            items = json.loads(text)
        except Exception:  # noqa: BLE001
            items = [p for p in text.replace(",", ";").split(";") if p.strip()]
    if isinstance(items, dict):
        items = [items]
    out: list[tuple[pd.Timestamp, pd.Timestamp, str]] = []
    for it in items if isinstance(items, (list, tuple)) else []:
        start = end = None
        kind = "leave"
        if isinstance(it, dict):
            start = it.get("from") or it.get("start") or it.get("valid_from") or it.get("start_date")
            end = it.get("to") or it.get("end") or it.get("valid_to") or it.get("end_date")
            kind = str(it.get("type") or it.get("kind") or it.get("reason") or "leave")
        elif isinstance(it, (list, tuple)) and len(it) >= 2:
            start, end = it[0], it[1]
        elif isinstance(it, str):
            for sep in ("/", "..", " to "):
                if sep in it:
                    start, end = it.split(sep, 1)
                    break
        s = _fast_ts(start)
        e = _fast_ts(end)
        if pd.isna(s):
            continue
        if pd.isna(e):
            e = s
        out.append((s.normalize(), e.normalize(), kind))
    return out


def _article(word: str) -> str:
    return "an" if str(word)[:1].lower() in "aeiou" else "a"


def _period(value: Any) -> str:
    return period_bucket(value)


def _ids(values: Iterable[Any], limit: int | None = None) -> list[str]:
    seen: list[str] = []
    for v in values:
        if is_missing(v):
            continue
        s = str(v)
        if s not in seen:
            seen.append(s)
    return seen if limit is None else seen[:limit]


def _dstr(value: Any) -> str | None:
    if is_missing(value):
        return None
    try:
        ts = pd.Timestamp(value)
        return None if pd.isna(ts) else str(ts.date())
    except Exception:  # noqa: BLE001
        return str(value)


# ===========================================================================
# ENT-01 — coverage, benefit and network
# ===========================================================================


@_register("ent_01_r01_coverage_inactive")
def ent_01_r01_coverage_inactive(ctx, control) -> list:
    """ENT-01-R01 — service date outside every non-void cover period of the member."""
    claims = _claims(ctx)
    cov = _coverage(ctx)
    if claims.empty or cov.empty or cov["vf"].isna().all():
        return []
    grace = int(ctx.cfg("ent01_coverage_grace_days"))
    newborn_days = int(ctx.cfg("ent_newborn_days"))
    live = cov[~cov["void"] & cov["vf"].notna()]
    known = set(cov["member_sk"])
    c = claims[claims["member_sk"].isin(known) & claims["service_date"].notna()]
    if c.empty:
        return []
    m = c[["claim_sk", "member_sk", "service_date"]].merge(
        live[["member_sk", "vf", "vt", "coverage_id", "status"]], on="member_sk", how="left")
    vt_grace = m["vt"] + pd.Timedelta(days=grace)
    m["inside"] = (m["service_date"] >= m["vf"]) & ((m["service_date"] <= vt_grace) | m["vt"].isna())
    covered = m.groupby("claim_sk")["inside"].any()
    cand = c[~c["claim_sk"].map(covered).fillna(False).astype(bool)]
    if cand.empty:
        return []
    # exclusions -------------------------------------------------------------
    setting = _claim_setting(ctx, cand)
    emergency = _is_emergency(setting["encounter_type"]) | _is_emergency(setting["claim_type"])
    cand = cand[~cand["claim_sk"].isin(set(setting.loc[emergency, "claim_sk"]))]
    mem = _members(ctx)
    if not cand.empty and "date_of_birth" in mem.columns:
        dob = cand["member_sk"].map(mem.set_index("member_sk")["date_of_birth"].pipe(_d))
        newborn = (cand["service_date"] - dob).dt.days.between(0, newborn_days)
        cand = cand[~newborn.fillna(False).astype(bool)]
    ev = _table(ctx, "policy_event")
    retro_claims: set[str] = set()
    if not cand.empty and not ev.empty and "coverage_id" in ev.columns:
        pe = pd.DataFrame({"coverage_id": _s(ev["coverage_id"]), "event_time": _d(_col(ev, "event_time")),
                           "effective_date": _d(_col(ev, "effective_date"))})
        pe = pe.merge(cov[["coverage_id", "member_sk"]], on="coverage_id", how="inner")
        j = cand[["claim_sk", "member_sk", "service_date"]].merge(pe, on="member_sk", how="inner")
        pending_backdate = (j["effective_date"] <= j["service_date"]) & (j["event_time"] >= j["service_date"])
        retro_claims = set(j.loc[pending_backdate, "claim_sk"])
    cand = cand[~cand["claim_sk"].isin(retro_claims)]
    if cand.empty:
        return []
    # evidence: the member's cover intervals --------------------------------
    intervals: dict[str, list] = {}
    for r in live[live["member_sk"].isin(set(cand["member_sk"]))].sort_values("vf").itertuples(index=False):
        intervals.setdefault(r.member_sk, []).append((r.vf, r.vt, r.coverage_id, r.status))
    out = []
    for row in cand.itertuples(index=False):
        ivs = intervals.get(row.member_sk, [])
        sd = row.service_date
        before = [iv for iv in ivs if not pd.isna(iv[1]) and iv[1] < sd]
        after = [iv for iv in ivs if iv[0] > sd]
        if before:
            ref = before[-1]
            gap = int((sd - ref[1]).days)
            where = f"{gap} days after that cover ended"
        elif after:
            ref = after[0]
            gap = int((ref[0] - sd).days)
            where = f"{gap} days before that cover started"
        else:
            ref, gap, where = (None, None, None, None), None, "and no cover period is on file"
        cover_txt = (f"{plain_date(ref[0])} to {plain_date(ref[1])}" if ref[0] is not None else "none")
        out.append(_sig(
            ctx, control, subject_type="claim", subject_id=row.claim_sk,
            fact_key=f"coverage:{row.claim_sk}", claim_ids=[row.claim_sk],
            event_time=sd, period=_period(sd),
            evidence={
                "member_sk": row.member_sk, "provider_sk": row.provider_sk,
                "service_date": _dstr(sd),
                "coverage_id": ref[2], "coverage_valid_from": _dstr(ref[0]), "coverage_valid_to": _dstr(ref[1]),
                "coverage_status": ref[3], "days_outside_cover": gap, "grace_days_applied": grace,
                "all_cover_periods": [[_dstr(a), _dstr(b)] for a, b, _, _ in ivs],
                "net_amount_aed": round(_f(row.net), 2),
                "reason_code": control.reason_code, "subject_id": row.claim_sk,
                "plain_language": (f"Service on {plain_date(sd)}, but the patient's cover on file ran "
                                   f"{cover_txt} — {where}. {aed(row.net)} is claimed."),
                "what_the_reviewer_must_verify": "Confirm the policy dates and whether a grace period, "
                                                 "newborn cover, emergency rule or backdated renewal applies.",
            },
            exposure=_exp.line_edit_exposure(_f(row.net), 0.0),
        ))
    return out


def _line_benefits(ctx, claims: pd.DataFrame) -> pd.DataFrame:
    """Lines joined to the product in force and the benefit row in force."""
    lines = _lines(ctx, claims)
    ref = _activity_ref(ctx)
    brv = _table(ctx, "benefit_rule_version")
    cov = _coverage(ctx)
    if lines.empty or ref.empty or brv.empty or cov.empty or "service_family" not in ref.columns:
        return pd.DataFrame()
    cc = _claim_coverage(claims, cov)
    lines = lines.merge(cc[["claim_sk", "coverage_id", "product"]], on="claim_sk", how="inner")
    lines = lines.merge(ref[["activity_code", "service_family"]], on="activity_code", how="left")
    b = pd.DataFrame({
        "product": _s(_col(brv, "product")), "service_family": _s(_col(brv, "service_family")),
        "covered_raw": _col(brv, "covered"), "benefit_limit": _num(_col(brv, "benefit_limit")),
        "b_vf": _d(_col(brv, "valid_from")), "b_vt": _d(_col(brv, "valid_to")),
        "version_id": _s(_col(brv, "version_id")),
        "exceptions": _col(brv, "exceptions", "").fillna("").astype(str),
        "authorization_required": _col(brv, "authorization_required"),
    })
    m = lines.merge(b, on=["product", "service_family"], how="inner")
    ok = ((m["line_date"] >= m["b_vf"]) | m["b_vf"].isna()) & ((m["line_date"] <= m["b_vt"]) | m["b_vt"].isna())
    m = m[ok].sort_values(["line_sk", "b_vf"]).drop_duplicates("line_sk", keep="last")
    return m


@_register("ent_01_r02_benefit_not_covered")
def ent_01_r02_benefit_not_covered(ctx, control) -> list:
    """ENT-01-R02 — a billed service the product's benefit version says is not covered."""
    claims = _claims(ctx)
    m = _line_benefits(ctx, claims)
    if m.empty:
        return []
    bad = m[_falsy(m["covered_raw"])]
    if bad.empty:
        return []
    exc = bad["exceptions"].str.upper()
    bad = bad[~exc.str.contains("MANDATORY") & ~exc.str.contains("MEDICAL_TOURISM")]
    # approved exception: an approved authorisation on the line
    auth = _authorizations(ctx)
    if not auth.empty and not bad.empty:
        st = bad["authorization_id"].map(auth.set_index("authorization_id")["auth_status"])
        bad = bad[~_has_word(st.fillna(""), _APPROVED_WORDS)]
    # self-pay route: the member paid the line in full
    bad = bad[~(bad["patient_share"] >= bad["gross_line"] - 0.005) | (bad["gross_line"] <= 0)]
    bad = bad[bad["net_line"] > 0]
    out = []
    for claim_sk, g in bad.groupby("claim_sk", sort=False):
        first = g.iloc[0]
        total = float(g["net_line"].sum())
        fams = _ids(g["service_family"])
        out.append(_sig(
            ctx, control, subject_type="claim", subject_id=claim_sk,
            fact_key=f"not_covered:{claim_sk}", claim_ids=[claim_sk],
            event_time=first["line_date"], period=_period(first["line_date"]),
            evidence={
                "member_sk": first["member_sk"], "provider_sk": first["provider_sk"],
                "product": first["product"], "service_family": fams,
                "benefit_version_id": first["version_id"],
                "lines": [{"line_sk": r.line_sk, "activity_code": r.activity_code,
                           "service_family": r.service_family, "net_amount_aed": round(_f(r.net_line), 2)}
                          for r in g.itertuples(index=False)],
                "not_covered_amount_aed": round(total, 2),
                "reason_code": control.reason_code, "subject_id": claim_sk,
                "plain_language": (f"{count_phrase(len(g), 'line')} billed for {', '.join(fams)} "
                                   f"({aed(total)}); the patient's plan ({first['product']}) does not cover "
                                   f"{'that service' if len(fams) == 1 else 'those services'} under benefit "
                                   f"version {first['version_id']}."),
                "what_the_reviewer_must_verify": "Confirm the benefit version in force and whether a mandatory "
                                                 "benefit, approved exception or self-pay arrangement applies.",
            },
            exposure=_exp.line_edit_exposure(total, 0.0),
        ))
    return out


@_register("ent_01_r03_benefit_limit_exceeded")
def ent_01_r03_benefit_limit_exceeded(ctx, control) -> list:
    """ENT-01-R03 — paid-to-date plus this claim exceeds the benefit limit for the cover period."""
    claims = _claims(ctx)
    m = _line_benefits(ctx, claims)
    if m.empty:
        return []
    m = m[m["benefit_limit"].notna() & (m["benefit_limit"] > 0) & ~_falsy(m["covered_raw"])]
    if m.empty:
        return []
    # reversals: a denied/reversed line does not count, and is not itself payable
    rem = _table(ctx, "remittance")
    m = m.copy()
    m["paid"] = np.nan
    if not rem.empty and "line_sk" in rem.columns:
        r = pd.DataFrame({"line_sk": _s(rem["line_sk"]), "decision": _up(_col(rem, "decision", "")),
                          "payment_amount": _num(_col(rem, "payment_amount"))}).dropna(subset=["line_sk"])
        r = r.groupby("line_sk").agg(decision=("decision", "last"), payment_amount=("payment_amount", "sum"))
        dec = m["line_sk"].map(r["decision"]).fillna("")
        m = m[~_has_word(dec, _DENIED_WORDS)]
        m["paid"] = m["line_sk"].map(r["payment_amount"])
    # pending authorisations are not counted as used
    auth = _authorizations(ctx)
    if not auth.empty:
        st = m["authorization_id"].map(auth.set_index("authorization_id")["auth_status"]).fillna("")
        m = m[~_has_word(st, _PENDING_WORDS)]
    if m.empty:
        return []
    # family basis where the benefit row says so
    fam = m["exceptions"].str.upper().str.contains("FAMILY")
    basis_key = m["coverage_id"].astype(str)
    if fam.any():
        mem = _members(ctx)
        if "sponsor_id" in mem.columns:
            sponsor = m["member_sk"].map(mem.set_index("member_sk")["sponsor_id"])
            year = m["line_date"].dt.year.astype("Int64").astype(str)
            basis_key = basis_key.where(~fam | sponsor.isna(), "FAM:" + sponsor.astype(str) + ":" + year)
    m["basis_key"] = basis_key + "|" + m["service_family"].astype(str)
    m["used"] = m["paid"].where(m["paid"].notna(), m["net_line"]).clip(lower=0)
    m = m.sort_values(["basis_key", "line_date", "claim_sk", "line_sk"])
    m["used_before"] = m.groupby("basis_key")["used"].cumsum() - m["used"]
    m["total"] = m["used_before"] + m["net_line"]
    m["excess"] = np.minimum(m["net_line"], (m["total"] - m["benefit_limit"]).clip(lower=0))
    m["excess"] = m["excess"].round(2)
    over = m[m["excess"] > 0]
    out = []
    for claim_sk, g in over.groupby("claim_sk", sort=False):
        first = g.iloc[0]
        excess = float(g["excess"].sum())
        claimed = float(g["net_line"].sum())
        limit = float(first["benefit_limit"])
        before = float(first["used_before"])
        out.append(_sig(
            ctx, control, subject_type="claim", subject_id=claim_sk,
            fact_key=f"limit:{claim_sk}", claim_ids=[claim_sk],
            event_time=first["line_date"], period=_period(first["line_date"]),
            evidence={
                "member_sk": first["member_sk"], "provider_sk": first["provider_sk"],
                "product": first["product"], "service_family": first["service_family"],
                "benefit_version_id": first["version_id"], "benefit_limit_aed": round(limit, 2),
                "used_before_this_claim_aed": round(before, 2),
                "requested_on_this_claim_aed": round(claimed, 2),
                "excess_over_limit_aed": round(excess, 2),
                "limit_basis": "family" if str(first["basis_key"]).startswith("FAM:") else "member, per cover period",
                "reason_code": control.reason_code, "subject_id": claim_sk,
                "plain_language": (f"The {first['service_family']} limit is {aed(limit)}; {aed(before)} had already "
                                   f"been used this cover period and this claim asks for {aed(claimed)}, "
                                   f"so {aed(excess)} is over the limit."),
                "what_the_reviewer_must_verify": "Confirm the limit, its reset date and whether it is per member "
                                                 "or per family; remove reversed claims from the running total.",
            },
            exposure=_exp.line_edit_exposure(claimed, claimed - excess),
        ))
    return out


@_register("ent_01_r04_network_referral_violation")
def ent_01_r04_network_referral_violation(ctx, control) -> list:
    """ENT-01-R04 — provider outside the member's network on the date, or a required referral missing."""
    claims = _claims(ctx)
    cov = _coverage(ctx)
    psp = _table(ctx, "provider_status_period")
    if claims.empty or cov.empty or psp.empty or cov["network"].isna().all():
        return []
    cc = _claim_coverage(claims, cov)
    c = claims.merge(cc[["claim_sk", "network", "product"]], on="claim_sk", how="inner")
    c = c[c["network"].notna()]
    st = pd.DataFrame({"provider_sk": _s(_col(psp, "provider_sk")), "status_type": _up(_col(psp, "status_type", "")),
                       "status_value": _up(_col(psp, "status_value", "")), "vf": _d(_col(psp, "valid_from")),
                       "vt": _d(_col(psp, "valid_to"))})
    net = st[st["status_type"].str.contains("NETWORK")]
    if net.empty or c.empty:
        return []
    c["network_u"] = _up(c["network"])
    c = c[c["provider_sk"].isin(set(net["provider_sk"]))]
    j = c[["claim_sk", "provider_sk", "service_date", "network_u"]].merge(net, on="provider_sk", how="inner")
    in_force = (j["service_date"] >= j["vf"].fillna(pd.Timestamp.min)) & \
               ((j["service_date"] <= j["vt"]) | j["vt"].isna())
    tokens = j["status_value"].str.replace(",", ";").str.split(";")
    listed = [nw in {t.strip() for t in toks} if isinstance(toks, list) else False
              for nw, toks in zip(j["network_u"], tokens)]
    generic_in = j["status_value"].str.strip().isin(_IN_NETWORK_VALUES)
    j["ok"] = in_force & (pd.Series(listed, index=j.index) | generic_in) & \
        ~j["status_value"].str.contains("OUT|TERMIN|SUSPEND", regex=True)
    in_net = j.groupby("claim_sk")["ok"].any()
    c["in_network"] = c["claim_sk"].map(in_net).fillna(False).astype(bool)
    c["limb"] = np.where(~c["in_network"], "out_of_network", "")
    # referral limb: products whose benefit rows require a referral for the service family
    lines = _lines(ctx, claims)
    ref = _activity_ref(ctx)
    brv = _table(ctx, "benefit_rule_version")
    referral = _table(ctx, "referral")
    lookback = int(ctx.cfg("ent01_referral_lookback_days"))
    need_ref: set[str] = set()
    if not lines.empty and not brv.empty and "service_family" in ref.columns and not referral.empty:
        exc = _up(_col(brv, "exceptions", ""))
        req = pd.DataFrame({"product": _s(_col(brv, "product")), "service_family": _s(_col(brv, "service_family"))})[
            exc.str.contains("REFERRAL")]
        if not req.empty:
            l2 = lines.merge(ref[["activity_code", "service_family"]], on="activity_code", how="left")
            l2 = l2.merge(c[["claim_sk", "product"]], on="claim_sk", how="inner")
            l2 = l2.merge(req.drop_duplicates(), on=["product", "service_family"], how="inner")
            if not l2.empty:
                rf = pd.DataFrame({"member_sk": _s(_col(referral, "member_sk")),
                                   "recipient_provider_sk": _s(_col(referral, "recipient_provider_sk")),
                                   "referral_date": _d(_col(referral, "referral_date"))})
                k = l2[["claim_sk", "member_sk", "provider_sk", "line_date"]].drop_duplicates("claim_sk")
                jj = k.merge(rf, on="member_sk", how="left")
                okr = (jj["recipient_provider_sk"] == jj["provider_sk"]) & \
                      (jj["referral_date"] <= jj["line_date"]) & \
                      (jj["referral_date"] >= jj["line_date"] - pd.Timedelta(days=lookback))
                has = jj.assign(okr=okr).groupby("claim_sk")["okr"].any()
                need_ref = set(has[~has].index)
    c.loc[c["claim_sk"].isin(need_ref), "limb"] = np.where(
        c.loc[c["claim_sk"].isin(need_ref), "limb"] == "", "referral_missing", "out_of_network_and_referral_missing")
    c = c[c["limb"] != ""]
    if c.empty:
        return []
    # exclusions: emergency; access gap (no in-network provider of that type in the member's emirate)
    setting = _claim_setting(ctx, c)
    emerg = set(setting.loc[_is_emergency(setting["encounter_type"]) | _is_emergency(setting["claim_type"]),
                            "claim_sk"])
    c = c[~c["claim_sk"].isin(emerg)]
    prov = _providers(ctx)
    if not c.empty and {"emirate", "facility_type"} <= set(prov.columns):
        pinfo = prov.set_index("provider_sk")
        c["p_emirate"] = c["provider_sk"].map(pinfo["emirate"])
        c["p_type"] = c["provider_sk"].map(pinfo["facility_type"])
        nw_prov = net[net["status_value"].notna()].copy()
        nw_prov["emirate"] = nw_prov["provider_sk"].map(pinfo["emirate"])
        nw_prov["ftype"] = nw_prov["provider_sk"].map(pinfo["facility_type"])
        nw_prov = nw_prov[~nw_prov["status_value"].str.contains("OUT|TERMIN|SUSPEND", regex=True)]
        avail = set()
        for r in nw_prov.itertuples(index=False):
            toks = {t.strip() for t in str(r.status_value).replace(",", ";").split(";")}
            if toks & _IN_NETWORK_VALUES:
                toks.add("*")
            for tok in toks:
                avail.add((tok, r.emirate, r.ftype))
        gap = [(nw, e, t) not in avail and ("*", e, t) not in avail
               for nw, e, t in zip(c["network_u"], c["p_emirate"], c["p_type"])]
        c = c[~(pd.Series(gap, index=c.index) & (c["limb"] == "out_of_network"))]
    out = []
    for row in c.itertuples(index=False):
        why = {"out_of_network": "the treating provider is not in that network on the service date",
               "referral_missing": "the plan requires a referral for this service and none is on file",
               "out_of_network_and_referral_missing": "the provider is outside that network and no required "
                                                      "referral is on file"}[row.limb]
        out.append(_sig(
            ctx, control, subject_type="claim", subject_id=row.claim_sk,
            fact_key=f"network:{row.claim_sk}", claim_ids=[row.claim_sk],
            event_time=row.service_date, period=_period(row.service_date),
            evidence={
                "member_sk": row.member_sk, "provider_sk": row.provider_sk,
                "member_network": row.network, "limb_fired": row.limb,
                "service_date": _dstr(row.service_date), "net_amount_aed": round(_f(row.net), 2),
                "referral_lookback_days": lookback,
                "reason_code": control.reason_code, "subject_id": row.claim_sk,
                "plain_language": (f"The patient's plan uses the {row.network} network, but {why} "
                                   f"({plain_date(row.service_date)}, {aed(row.net)} claimed)."),
                "what_the_reviewer_must_verify": "Confirm the provider's network status on the date and any "
                                                 "referral, emergency or access-gap exception.",
            },
            exposure=_exp.line_edit_exposure(_f(row.net), 0.0),
        ))
    return out


# ===========================================================================
# ENT-02 — member identity and presence
# ===========================================================================


@_register("ent_02_r01_identity_conflict")
def ent_02_r01_identity_conflict(ctx, control) -> list:
    """ENT-02-R01 — one protected national-ID token on member records with different DOB or sex."""
    mem = _members(ctx)
    if mem.empty or "protected_id_token" not in mem.columns:
        return []
    placeholder_max = int(ctx.cfg("ent02_placeholder_token_max_members"))
    newborn_days = int(ctx.cfg("ent_newborn_days"))
    m = pd.DataFrame({"member_sk": mem["member_sk"], "token": _s(mem["protected_id_token"]),
                      "dob": _d(_col(mem, "date_of_birth")), "sex": _up(_col(mem, "sex", ""))})
    m = m.dropna(subset=["token"])
    # placeholder tokens: blank-like or shared by implausibly many records
    tok_u = m["token"].str.upper()
    placeholder = tok_u.str.fullmatch(r"(0+|9+|X+|NA|N/A|UNKNOWN|PLACEHOLDER|TEMP.*|000-.*)")
    m = m[~placeholder]
    sizes = m.groupby("token")["member_sk"].transform("nunique")
    m = m[(sizes >= 2) & (sizes <= placeholder_max)]
    if m.empty:
        return []
    claims = _claims(ctx)
    first_service = claims.groupby("member_sk")["service_date"].min()
    out = []
    for token, g in m.groupby("token", sort=False):
        dobs = g["dob"].dropna().dt.date.unique()
        sexes = [s for s in g["sex"].unique() if s]
        dob_conflict = len(dobs) > 1
        sex_conflict = len(set(sexes)) > 1
        if not (dob_conflict or sex_conflict):
            continue
        # newborn exclusion: a baby enrolled under a parent's token before its own ID is issued
        ages_at_first = (g["member_sk"].map(first_service) - g["dob"]).dt.days
        if (ages_at_first.dropna() <= newborn_days).any() and not sex_conflict:
            continue
        members = sorted(g["member_sk"].astype(str))
        cl = claims[claims["member_sk"].isin(members)]
        facts = []
        if dob_conflict:
            facts.append("dates of birth " + " and ".join(plain_date(d) for d in sorted(dobs)))
        if sex_conflict:
            facts.append("sex recorded as " + " and ".join(sorted(set(sexes))))
        first_date = cl["service_date"].min() if not cl.empty else None
        out.append(_sig(
            ctx, control, subject_type="member", subject_id=members[0],
            fact_key=f"identity:{token}", claim_ids=cl["claim_sk"].tolist()[:50],
            event_time=first_date, period=_period(first_date),
            evidence={
                "member_sk": members[0], "linked_member_sks": members,
                "protected_id_token": token,
                "dates_of_birth": [str(d) for d in sorted(dobs)], "sexes": sorted(set(sexes)),
                "dob_conflict": bool(dob_conflict), "sex_conflict": bool(sex_conflict),
                "claim_count": int(len(cl)),
                "reason_code": control.reason_code, "subject_id": members[0],
                "plain_language": (f"One Emirates-ID reference is on {count_phrase(len(members), 'member record')} "
                                   f"with {'; '.join(facts)}; {count_phrase(len(cl), 'claim')} were billed "
                                   f"under these records."),
                "what_the_reviewer_must_verify": "Compare the claim details with the Emirates ID record and the "
                                                 "enrolment file; do not rely on names alone.",
            },
            exposure=_exp.no_exposure("identity must be resolved before any amount can be said to be at stake"),
        ))
    return out


@_register("ent_02_r02_post_mortem_service")
def ent_02_r02_post_mortem_service(ctx, control) -> list:
    """ENT-02-R02 — service start after a confirmed date of death plus the reporting tolerance."""
    mem = _members(ctx)
    claims = _claims(ctx)
    if mem.empty or claims.empty or "death_date" not in mem.columns:
        return []
    tol = int(ctx.cfg("death_reporting_tolerance_days"))
    min_conf = float(ctx.cfg("ent02_death_source_min_confidence"))
    d = pd.DataFrame({"member_sk": mem["member_sk"], "death_date": _d(mem["death_date"]),
                      "death_source": _col(mem, "death_source"),
                      "conf": _num(_col(mem, "death_source_confidence"))}).dropna(subset=["death_date"])
    d = d[d["conf"].isna() | (d["conf"] >= min_conf)]
    if d.empty:
        return []
    c = claims.merge(d, on="member_sk", how="inner")
    c = c[c["service_date"] > c["death_date"] + pd.Timedelta(days=tol)]
    if c.empty:
        return []
    # administrative items billed after an earlier service are permitted
    lines = _lines(ctx, c[["claim_sk", "member_sk", "provider_sk", "service_date", "gross", "net",
                           "claim_type", "encounter_sk"]])
    if not lines.empty:
        admin_only = lines.groupby("claim_sk")["activity_type"].apply(
            lambda s: bool(len(s)) and s.str.contains("ADMIN").all())
        c = c[~c["claim_sk"].map(admin_only).fillna(False).astype(bool)]
    out = []
    for row in c.itertuples(index=False):
        days = int((row.service_date - row.death_date).days)
        out.append(_sig(
            ctx, control, subject_type="claim", subject_id=row.claim_sk,
            fact_key=f"post_mortem:{row.claim_sk}", claim_ids=[row.claim_sk],
            event_time=row.service_date, period=_period(row.service_date),
            confidence=1.0 if pd.isna(row.conf) else float(min(1.0, row.conf)),
            evidence={
                "member_sk": row.member_sk, "provider_sk": row.provider_sk,
                "death_date": _dstr(row.death_date), "service_date": _dstr(row.service_date),
                "days_after_death": days, "death_source": None if is_missing(row.death_source) else str(row.death_source),
                "death_source_confidence": None if pd.isna(row.conf) else round(float(row.conf), 2),
                "reporting_tolerance_days": tol, "net_amount_aed": round(_f(row.net), 2),
                "reason_code": control.reason_code, "subject_id": row.claim_sk,
                "plain_language": (f"Service on {plain_date(row.service_date)}, {days} days after the patient's "
                                   f"recorded date of death ({plain_date(row.death_date)}); {aed(row.net)} is claimed."),
                "what_the_reviewer_must_verify": "Confirm the date of death with its source and ask the provider "
                                                 "for the record of the billed visit.",
            },
            exposure=_exp.line_edit_exposure(_f(row.net), 0.0),
        ))
    return out


@_register("ent_02_r04_card_sharing_pattern")
def ent_02_r04_card_sharing_pattern(ctx, control) -> list:
    """ENT-02-R04 — same-day use in different emirates and/or services that do not fit the member's age or sex."""
    claims = _claims(ctx)
    prov = _providers(ctx)
    mem = _members(ctx)
    if claims.empty or prov.empty:
        return []
    min_evidence = int(ctx.cfg("ent02_card_sharing_min_evidence"))
    setting = _claim_setting(ctx, claims)
    c = setting[~_is_tele(setting["encounter_type"])].copy()
    if "emirate" in prov.columns:
        c["emirate"] = c["provider_sk"].map(prov.set_index("provider_sk")["emirate"])
    else:
        c["emirate"] = np.nan
    c["emirate"] = c["emirate"].where(c["emirate"].notna(), c["location"])
    # limb 1: same member, same day, providers in two or more emirates
    # (neighbouring emirates are a short drive apart, so only non-adjacent pairs count)
    adjacent = {frozenset(str(x).strip().upper() for x in pair)
                for pair in (ctx.cfg("ent02_adjacent_emirates") or []) if len(pair) == 2}
    cg = c.dropna(subset=["emirate", "service_date"])
    cg = cg.assign(emirate_u=_up(cg["emirate"]))
    n_em = cg.groupby(["member_sk", "service_date"])["emirate_u"].nunique()
    multi = n_em[n_em >= 2].reset_index()[["member_sk", "service_date"]]
    geo_days: dict[str, list] = {}
    if not multi.empty:
        sets = cg.merge(multi, on=["member_sk", "service_date"]).groupby(["member_sk", "service_date"])[
            "emirate_u"].apply(lambda v: sorted(set(v)))
        for (member, day), ems in sets.items():
            if any(frozenset((a, b)) not in adjacent for i, a in enumerate(ems) for b in ems[i + 1:]):
                geo_days.setdefault(member, []).append(day)
    # limb 2: services inconsistent with the member's age or sex
    demo: dict[str, list[dict[str, Any]]] = {}
    ref = _activity_ref(ctx)
    lines = _lines(ctx, claims)
    if not lines.empty and not ref.empty and not mem.empty and "date_of_birth" in mem.columns:
        cols = [x for x in ("sex_restriction", "age_min", "age_max", "description") if x in ref.columns]
        l2 = lines[["claim_sk", "member_sk", "line_date", "activity_code"]].merge(
            ref[["activity_code"] + cols], on="activity_code", how="inner")
        info = mem.set_index("member_sk")
        l2["sex"] = _up(l2["member_sk"].map(info.get("sex", pd.Series(dtype=object))).fillna(""))
        dob = _d(l2["member_sk"].map(info["date_of_birth"]))
        l2["age"] = _age_years(dob, l2["line_date"])
        sr = _up(_col(l2, "sex_restriction", "")).str[:1]
        bad_sex = (sr.isin(["M", "F"])) & (l2["sex"].str[:1].isin(["M", "F"])) & (sr != l2["sex"].str[:1])
        amin, amax = _num(_col(l2, "age_min")), _num(_col(l2, "age_max"))
        bad_age = (amin.notna() & (l2["age"] < amin)) | (amax.notna() & (l2["age"] > amax))
        l2 = l2[bad_sex | bad_age].assign(why=np.where(bad_sex[bad_sex | bad_age], "sex", "age"))
        for r in l2.itertuples(index=False):
            demo.setdefault(r.member_sk, []).append({
                "claim_sk": r.claim_sk, "activity_code": r.activity_code, "date": _dstr(r.line_date),
                "reason": r.why, "member_sex": r.sex, "member_age": None if pd.isna(r.age) else round(float(r.age), 1)})
    # chronic / rare-disease pathway exclusion: members whose claims are dominated by one long-term diagnosis
    out = []
    members = set(geo_days) | set(demo)
    for member in sorted(members):
        days = geo_days.get(member, [])
        dem = demo.get(member, [])
        n_items = len(days) + len({d["claim_sk"] for d in dem})
        limbs = int(bool(days)) + int(bool(dem))
        if n_items < min_evidence:
            continue
        mc = c[c["member_sk"] == member]
        claim_ids = set()
        facts = []
        for dday in days:
            sub = mc[mc["service_date"] == dday]
            claim_ids.update(sub["claim_sk"])
            facts.append({"date": _dstr(dday), "emirates": sorted(sub["emirate"].dropna().astype(str).unique()),
                          "claims": sorted(sub["claim_sk"])})
        claim_ids.update(d["claim_sk"] for d in dem)
        sub_all = claims[claims["claim_sk"].isin(claim_ids)]
        parts = []
        if days:
            ex = facts[0]
            parts.append(f"seen in {' and '.join(ex['emirates'])} on the same day ({plain_date(ex['date'])})"
                         + (f" and on {len(days) - 1} other day(s)" if len(days) > 1 else ""))
        if dem:
            ex = dem[0]
            who = (f"a {ex['member_sex'].lower() or 'member'} aged {ex['member_age']:.0f}"
                   if ex["member_age"] is not None else "this member")
            parts.append(f"billed {count_phrase(len(dem), 'service')} that do not fit {who} "
                         f"(e.g. code {ex['activity_code']}, restricted by {ex['reason']})")
        first_date = sub_all["service_date"].min()
        out.append(_sig(
            ctx, control, subject_type="member", subject_id=member,
            fact_key=f"card_sharing:{member}", claim_ids=sorted(claim_ids),
            event_time=first_date, period=_period(first_date),
            confidence=0.5 if limbs == 1 else 0.7,
            evidence={
                "member_sk": member, "same_day_multi_emirate_days": facts,
                "demographic_inconsistent_services": dem[:20],
                "independent_evidence_items": n_items, "minimum_independent_evidence": min_evidence,
                "reason_code": control.reason_code, "subject_id": member,
                "plain_language": "This member was " + "; and ".join(parts) + ".",
                "what_the_reviewer_must_verify": "Review the member's visit history by place; check whether a "
                                                 "chronic or rare condition explains it before any referral.",
            },
            exposure=_exp.model_only_exposure(float(sub_all["gross"].sum()), "card-sharing pattern"),
        ))
    return out


# ===========================================================================
# ENT-03 — licences, privileges and provider activity
# ===========================================================================


def _status_rows(ctx) -> pd.DataFrame:
    psp = _table(ctx, "provider_status_period")
    if psp.empty:
        return pd.DataFrame(columns=["provider_sk", "status_type", "status_value", "vf", "vt"])
    return pd.DataFrame({"provider_sk": _s(_col(psp, "provider_sk")), "status_type": _up(_col(psp, "status_type", "")),
                         "status_value": _up(_col(psp, "status_value", "")), "vf": _d(_col(psp, "valid_from")),
                         "vt": _d(_col(psp, "valid_to")), "source": _col(psp, "source")})


_BAD_LICENCE_WORDS = ("EXPIR", "SUSPEND", "REVOK", "INACTIVE", "LAPSE", "CANCEL", "WITHDRAW")


@_register("ent_03_r01_inactive_licence")
def ent_03_r01_inactive_licence(ctx, control) -> list:
    """ENT-03-R01 — facility or treating/ordering clinician licence not active on the service date."""
    claims = _claims(ctx)
    if claims.empty:
        return []
    grace = int(ctx.cfg("ent03_licence_grace_days"))
    g = pd.Timedelta(days=grace)
    findings: list[dict[str, Any]] = []
    # facility limb -----------------------------------------------------------
    st = _status_rows(ctx)
    lic = st[st["status_type"].str.contains("LICEN")]
    if not lic.empty:
        c = claims[claims["provider_sk"].isin(set(lic["provider_sk"]))]
        j = c[["claim_sk", "provider_sk", "service_date", "net", "member_sk"]].merge(lic, on="provider_sk")
        good_val = ~_has_word(j["status_value"], _BAD_LICENCE_WORDS)
        active = good_val & (j["service_date"] >= j["vf"].fillna(pd.Timestamp.min) - g) & \
                 ((j["service_date"] <= j["vt"] + g) | j["vt"].isna())
        ok = j.assign(a=active).groupby("claim_sk")["a"].any()
        bad = c[~c["claim_sk"].map(ok).fillna(False).astype(bool)]
        last = lic.sort_values("vt").groupby("provider_sk").last()
        for r in bad.itertuples(index=False):
            lr = last.loc[r.provider_sk] if r.provider_sk in last.index else None
            findings.append({"claim_sk": r.claim_sk, "role": "facility", "who": r.provider_sk,
                             "provider_sk": r.provider_sk, "member_sk": r.member_sk, "service_date": r.service_date,
                             "net": r.net, "valid_from": None if lr is None else lr["vf"],
                             "valid_to": None if lr is None else lr["vt"],
                             "status": None if lr is None else lr["status_value"]})
    # clinician limb ----------------------------------------------------------
    ros = _roster(ctx)
    lines = _lines(ctx, claims)
    if not ros.empty and not lines.empty and "licence_valid_to" in ros.columns:
        r = ros[["clinician_id", "provider_sk"]].copy()
        r["lvf"] = _d(_col(ros, "licence_valid_from"))
        r["lvt"] = _d(_col(ros, "licence_valid_to"))
        r = r[r["lvf"].notna() | r["lvt"].notna()]
        per_clin = r.groupby("clinician_id").agg(lvf=("lvf", "min"), lvt=("lvt", "max"))
        for role, col in (("treating clinician", "rendering"), ("ordering clinician", "ordering")):
            l2 = lines.dropna(subset=[col])
            l2 = l2[l2[col].isin(per_clin.index)]
            if l2.empty:
                continue
            lvf = l2[col].map(per_clin["lvf"])
            lvt = l2[col].map(per_clin["lvt"])
            outside = (l2["line_date"] < lvf - g) | (l2["line_date"] > lvt + g)
            for cl, gg in l2[outside.fillna(False)].groupby("claim_sk", sort=False):
                f0 = gg.iloc[0]
                findings.append({"claim_sk": cl, "role": role, "who": f0[col], "provider_sk": f0["provider_sk"],
                                 "member_sk": f0["member_sk"], "service_date": f0["line_date"],
                                 "net": float(gg["net_line"].sum()),
                                 "valid_from": per_clin.loc[f0[col], "lvf"], "valid_to": per_clin.loc[f0[col], "lvt"],
                                 "status": None})
    if not findings:
        return []
    fr = pd.DataFrame(findings)
    out = []
    for claim_sk, gg in fr.groupby("claim_sk", sort=False):
        f0 = gg.iloc[0]
        net_amount = float(claims.loc[claims["claim_sk"] == claim_sk, "net"].sum())
        sd = f0["service_date"]
        detail = []
        for r in gg.itertuples(index=False):
            if r.valid_to is not None and not pd.isna(r.valid_to) and sd > r.valid_to:
                detail.append(f"the {r.role}'s licence expired on {plain_date(r.valid_to)}")
            elif r.valid_from is not None and not pd.isna(r.valid_from) and sd < r.valid_from:
                detail.append(f"the {r.role}'s licence only started on {plain_date(r.valid_from)}")
            else:
                detail.append(f"the {r.role}'s licence was not active ({r.status or 'no active period on file'})")
        subject_is_clin = f0["role"] != "facility"
        out.append(_sig(
            ctx, control, subject_type="claim", subject_id=claim_sk,
            fact_key=f"licence:{claim_sk}", claim_ids=[claim_sk],
            event_time=sd, period=_period(sd),
            evidence={
                "provider_sk": f0["provider_sk"], "member_sk": f0["member_sk"],
                "clinician_id": f0["who"] if subject_is_clin else None,
                "service_date": _dstr(sd),
                "licences": [{"role": r.role, "id": r.who, "valid_from": _dstr(r.valid_from),
                              "valid_to": _dstr(r.valid_to), "status": r.status} for r in gg.itertuples(index=False)],
                "grace_days_applied": grace, "net_amount_aed": round(net_amount, 2),
                "reason_code": control.reason_code, "subject_id": claim_sk,
                "plain_language": (f"Service on {plain_date(sd)}, but " + "; ".join(dict.fromkeys(detail))
                                   + f". {aed(net_amount)} is claimed."),
                "what_the_reviewer_must_verify": "Confirm the licence dates on the DoH/DHA register and whether an "
                                                 "approved grace period applies.",
            },
            exposure=_exp.line_edit_exposure(net_amount, 0.0),
        ))
    return out


def _prohibited_tokens(value: Any) -> set[str]:
    if is_missing(value):
        return set()
    out = set()
    for tok in str(value).replace(",", ";").replace("|", ";").split(";"):
        t = tok.strip().upper()
        for p in _PROHIBITED_PREFIXES:
            if t.startswith(p) and len(t) > len(p):
                out.add(t[len(p):].strip())
                break
    return out


def _allowed_tokens(value: Any) -> set[str]:
    if is_missing(value):
        return set()
    out = set()
    for tok in str(value).replace(",", ";").replace("|", ";").split(";"):
        t = tok.strip().upper()
        if t and not any(t.startswith(p) for p in _PROHIBITED_PREFIXES):
            out.add(t)
    return out


@_register("ent_03_r03_hard_privilege_violation")
def ent_03_r03_hard_privilege_violation(ctx, control) -> list:
    """ENT-03-R03 — a service the clinician or facility is explicitly prohibited from performing."""
    claims = _claims(ctx)
    lines = _lines(ctx, claims)
    if lines.empty:
        return []
    ref = _activity_ref(ctx)
    keys = ["activity_code"] + [c for c in ("code_family", "service_family") if c in ref.columns]
    if len(keys) > 1:
        lines = lines.merge(ref[keys], on="activity_code", how="left")
    for k in ("code_family", "service_family"):
        if k not in lines.columns:
            lines[k] = np.nan
    lines["tok_code"] = _up(lines["activity_code"])
    lines["tok_cf"] = _up(lines["code_family"])
    lines["tok_sf"] = _up(lines["service_family"])
    hits = []
    # clinician privileges
    ros = _roster(ctx)
    if not ros.empty and "privileges" in ros.columns:
        pro = ros[["clinician_id", "provider_sk"]].copy()
        pro["prohibited"] = ros["privileges"].map(_prohibited_tokens)
        pro = pro[pro["prohibited"].map(len) > 0]
        if not pro.empty:
            pc = pro.groupby("clinician_id")["prohibited"].apply(lambda s: set().union(*s))
            l2 = lines[lines["rendering"].isin(pc.index)]
            if not l2.empty:
                banned = l2["rendering"].map(pc)
                hit = [bool({a, b, c} & p) for a, b, c, p in zip(l2["tok_code"], l2["tok_cf"], l2["tok_sf"], banned)]
                for r in l2[pd.Series(hit, index=l2.index)].itertuples(index=False):
                    hits.append((r, "treating clinician", r.rendering))
    # facility privileges
    st = _status_rows(ctx)
    fp = st[st["status_type"].str.contains("PRIVILEGE")]
    if not fp.empty:
        fp = fp.assign(prohibited=fp["status_value"].map(_prohibited_tokens))
        fp = fp[fp["prohibited"].map(len) > 0]
        if not fp.empty:
            l2 = lines.merge(fp[["provider_sk", "prohibited", "vf", "vt"]], on="provider_sk", how="inner")
            inforce = (l2["line_date"] >= l2["vf"].fillna(pd.Timestamp.min)) & ((l2["line_date"] <= l2["vt"]) | l2["vt"].isna())
            hit = [bool({a, b, c} & p) for a, b, c, p in zip(l2["tok_code"], l2["tok_cf"], l2["tok_sf"], l2["prohibited"])]
            for r in l2[inforce & pd.Series(hit, index=l2.index)].itertuples(index=False):
                hits.append((r, "facility", r.provider_sk))
    if not hits:
        return []
    # exception: an approved authorisation on the line
    auth = _authorizations(ctx)
    approved = set()
    if not auth.empty:
        approved = set(auth.loc[_has_word(auth["auth_status"], _APPROVED_WORDS), "authorization_id"])
    by_claim: dict[str, list] = {}
    for r, role, who in hits:
        if r.authorization_id in approved:
            continue
        by_claim.setdefault(r.claim_sk, []).append((r, role, who))
    out = []
    for claim_sk, items in by_claim.items():
        uniq = {}
        for r, role, who in items:
            uniq[(r.line_sk, role)] = (r, role, who)
        items = list(uniq.values())
        r0, role0, who0 = items[0]
        total = float(sum(_f(r.net_line) for r, _, _ in {x[0].line_sk: x for x in items}.values()))
        out.append(_sig(
            ctx, control, subject_type="claim", subject_id=claim_sk,
            fact_key=f"privilege:{claim_sk}", claim_ids=[claim_sk],
            event_time=r0.line_date, period=_period(r0.line_date),
            evidence={
                "provider_sk": r0.provider_sk, "member_sk": r0.member_sk,
                "clinician_id": r0.rendering if role0 != "facility" else None,
                "lines": [{"line_sk": r.line_sk, "activity_code": r.activity_code, "prohibited_for": role,
                           "net_amount_aed": round(_f(r.net_line), 2)} for r, role, _ in items],
                "net_amount_aed": round(total, 2),
                "reason_code": control.reason_code, "subject_id": claim_sk,
                "plain_language": (f"Code {r0.activity_code} ({aed(r0.net_line)}) was billed under a {role0} whose "
                                   f"privileges on {plain_date(r0.line_date)} explicitly exclude that service."),
                "what_the_reviewer_must_verify": "Confirm the privileges in force on the service date and whether an "
                                                 "approved exception covers this service.",
            },
            exposure=_exp.line_edit_exposure(total, 0.0),
        ))
    return out


@_register("ent_03_r04_soft_specialty_mismatch")
def ent_03_r04_soft_specialty_mismatch(ctx, control) -> list:
    """ENT-03-R04 — service outside the curated specialty map (cfg.specialty_map) for the treating
    clinician, with no cross-specialty privilege for it. Not prohibited: a coding-review finding."""
    claims = _claims(ctx)
    lines = _lines(ctx, claims)
    ref = _activity_ref(ctx)
    ros = _roster(ctx)
    if lines.empty or ref.empty or ros.empty or "code_family" not in ref.columns or "specialty" not in ros.columns:
        return []
    smap = {str(k).upper(): {str(v).upper() for v in (vals or [])}
            for k, vals in (ctx.cfg("ent03_specialty_map") or {}).items()}
    if not smap:
        return []
    fam = _up(ref.set_index("activity_code")["code_family"])
    l2 = lines.dropna(subset=["rendering"]).copy()
    l2["code_family"] = _up(l2["activity_code"].map(fam).fillna(""))
    l2 = l2[l2["code_family"].isin(set(smap))]
    if l2.empty:
        return []
    rs = ros.drop_duplicates(["clinician_id", "provider_sk"])
    spec = rs.groupby("clinician_id")["specialty"].apply(
        lambda s: {str(x).strip().upper() for x in s if not is_missing(x)})
    priv = (rs.groupby("clinician_id")["privileges"].apply(lambda s: set().union(*[_allowed_tokens(x) for x in s]))
            if "privileges" in rs.columns else pd.Series(dtype=object))
    l2 = l2[l2["rendering"].isin(spec.index)]
    if l2.empty:
        return []
    mism = []
    for clin, code, cf in zip(l2["rendering"], l2["activity_code"], l2["code_family"]):
        if spec.get(clin, set()) & smap[cf]:
            mism.append(False)
            continue
        p = priv.get(clin, set()) if len(priv) else set()
        mism.append(not ({str(code).upper(), cf} & p))
    bad = l2[pd.Series(mism, index=l2.index)]
    desc = ref.set_index("activity_code")["description"] if "description" in ref.columns else pd.Series(dtype=object)
    out = []
    for claim_sk, g in bad.groupby("claim_sk", sort=False):
        r0 = g.iloc[0]
        clin_spec = ", ".join(sorted(spec.get(r0["rendering"], set()))).replace("_", " ").lower()
        expected = sorted(smap[r0["code_family"]])
        req_txt = ("a specialist (" + ", ".join(e.replace("_", " ").lower() for e in expected[:3]) + " and others)"
                   if len(expected) > 3 else " or ".join(e.replace("_", " ").lower() for e in expected))
        what = desc.get(r0["activity_code"]) if len(desc) else None
        what = f"{what} (code {r0['activity_code']})" if what and not is_missing(what) else f"Code {r0['activity_code']}"
        out.append(_sig(
            ctx, control, subject_type="claim", subject_id=claim_sk,
            fact_key=f"specialty:{claim_sk}", claim_ids=[claim_sk],
            event_time=r0["line_date"], period=_period(r0["line_date"]), confidence=0.7,
            evidence={
                "provider_sk": r0["provider_sk"], "member_sk": r0["member_sk"], "clinician_id": r0["rendering"],
                "clinician_specialty": clin_spec,
                "lines": [{"line_sk": r.line_sk, "activity_code": r.activity_code, "code_family": r.code_family,
                           "expected_specialties": sorted(smap[r.code_family])} for r in g.itertuples(index=False)],
                "reason_code": control.reason_code, "subject_id": claim_sk,
                "plain_language": (f"{what} is normally performed by {req_txt}, but the treating clinician is recorded "
                                   f"as {clin_spec or 'having no specialty'} and holds no cross-specialty privilege "
                                   f"for it."),
                "what_the_reviewer_must_verify": "Check the clinician's specialty and privileges and who actually "
                                                 "performed the service; route to coding review, not refusal.",
            },
            exposure=_exp.no_exposure("a specialty mismatch is a coding-review finding; no amount is established"),
        ))
    return out


@_register("ent_03_r05_dormant_new_provider_burst")
def ent_03_r05_dormant_new_provider_burst(ctx, control) -> list:
    """ENT-03-R05 — new or reactivated provider whose early volume or value tops its peer percentile."""
    claims = _claims(ctx)
    prov = _providers(ctx)
    if claims.empty or prov.empty:
        return []
    window = int(ctx.cfg("ent03_new_provider_window_days"))
    dormant_gap = int(ctx.cfg("ent03_dormant_gap_days"))
    q = float(ctx.cfg("ent03_burst_peer_percentile"))
    min_claims = int(ctx.cfg("ent03_burst_min_claims"))
    c = claims.dropna(subset=["service_date", "provider_sk"])
    if c.empty:
        return []
    data_start = c["service_date"].min()
    info = prov.set_index("provider_sk")
    cred = _d(info["credentialing_date"]) if "credentialing_date" in info.columns else pd.Series(dtype="datetime64[ns]")
    st = _status_rows(ctx)
    lic_start = st[st["status_type"].str.contains("LICEN")].groupby("provider_sk")["vf"].max()
    # start of each provider's (re)activation
    starts: dict[str, tuple[pd.Timestamp, str]] = {}
    for p, dt in cred.dropna().items():
        if dt > data_start:
            starts[p] = (dt, "new")
    for p, dt in lic_start.dropna().items():
        if dt > data_start and p not in starts:
            starts[p] = (dt, "reactivated")
    # dormant: a gap of dormant_gap days with no claims, then billing resumes
    cs = c.sort_values(["provider_sk", "service_date"])
    gaps = cs.groupby("provider_sk")["service_date"].diff().dt.days
    resumed = cs[gaps >= dormant_gap].groupby("provider_sk")["service_date"].min()
    for p, dt in resumed.items():
        starts.setdefault(p, (dt, "dormant then resumed"))
    if not starts:
        return []
    # peer yardstick: claims and value per 30 days for every provider, by facility type
    span = c.groupby("provider_sk")["service_date"].agg(["min", "max", "count"])
    months = ((span["max"] - span["min"]).dt.days.clip(lower=window) / 30.0)
    rate = (span["count"] / months).rename("rate")
    value = (c.groupby("provider_sk")["gross"].sum() / months).rename("value")
    peer = pd.concat([rate, value], axis=1)
    ftype = info["facility_type"] if "facility_type" in info.columns else pd.Series(dtype=object)
    peer["ftype"] = peer.index.map(ftype).fillna("ALL")
    min_peer = int(ctx.cfg("min_entity_opportunities"))
    own_change = _d(info["ownership_changed_on"]) if "ownership_changed_on" in info.columns else pd.Series(dtype="datetime64[ns]")
    out = []
    for p, (start, kind) in sorted(starts.items()):
        sub = c[(c["provider_sk"] == p) & (c["service_date"] >= start) &
                (c["service_date"] < start + pd.Timedelta(days=window))]
        if len(sub) < min_claims:
            continue
        # exclusions: an acquisition/merger around the start
        oc = own_change.get(p) if len(own_change) else None
        if oc is not None and not pd.isna(oc) and abs((start - oc).days) <= window:
            continue
        w_rate = len(sub) / (window / 30.0)
        w_value = float(sub["gross"].sum()) / (window / 30.0)
        ft = peer.loc[p, "ftype"] if p in peer.index else "ALL"
        others = peer[(peer["ftype"] == ft) & (~peer.index.isin(list(starts)))]
        if len(others) < min_peer:
            others = peer[~peer.index.isin(list(starts))]
        if len(others) < min_peer:
            continue
        cut_rate = float(np.quantile(others["rate"], q))
        cut_value = float(np.quantile(others["value"], q))
        over_rate, over_value = w_rate > cut_rate, w_value > cut_value
        if not (over_rate or over_value):
            continue
        med_rate = float(others["rate"].median())
        what = (f"{w_rate:.0f} claims a month" if over_rate else f"{aed(w_value)} a month")
        typical = (f"{med_rate:.0f}" if over_rate else aed(float(others["value"].median())))
        start_txt = {"new": "was credentialed", "reactivated": "had its licence reactivated",
                     "dormant then resumed": "resumed billing after a long dormant spell"}[kind]
        out.append(_sig(
            ctx, control, subject_type="provider", subject_id=p,
            fact_key=f"burst:{p}", claim_ids=sub["claim_sk"].tolist(),
            event_time=start, period=_period(start), confidence=0.6,
            peer_level_used=f"facility_type={ft}",
            evidence={
                "provider_sk": p, "activation_kind": kind, "activation_date": _dstr(start),
                "window_days": window, "claims_in_window": int(len(sub)),
                "claims_per_month_in_window": round(w_rate, 1), "value_per_month_in_window_aed": round(w_value, 2),
                "peer_percentile": q, "peer_cut_claims_per_month": round(cut_rate, 1),
                "peer_cut_value_per_month_aed": round(cut_value, 2),
                "peer_median_claims_per_month": round(med_rate, 1), "peer_n": int(len(others)),
                "reason_code": control.reason_code, "subject_id": p,
                "plain_language": (f"This provider {start_txt} on {plain_date(start)} and within {window} days was "
                                   f"billing {what}; similar providers typically bill {typical} a month "
                                   f"(the {pct(q)} mark is {cut_rate:.0f} claims / {aed(cut_value)})."),
                "what_the_reviewer_must_verify": "Confirm the credentialing and contract dates, any acquisition or "
                                                 "merger, and consider a site visit.",
            },
            exposure=_exp.model_only_exposure(float(sub["gross"].sum()), "new-provider burst"),
        ))
    return out


# ===========================================================================
# ENT-04 — clinicians
# ===========================================================================


def _clinician_required(lines: pd.DataFrame, ref: pd.DataFrame) -> pd.Series:
    at = lines["activity_type"]
    if not ref.empty and "activity_type" in ref.columns:
        at = at.where(at != "", lines["activity_code"].map(_up(ref.set_index("activity_code")["activity_type"])).fillna(""))
    fam = pd.Series("", index=lines.index)
    if not ref.empty and "service_family" in ref.columns:
        fam = lines["activity_code"].map(_up(ref.set_index("activity_code")["service_family"])).fillna("")
    return ~(_has_word(at, _NON_CLINICIAN_ACTIVITY_WORDS) | _has_word(fam, _NON_CLINICIAN_ACTIVITY_WORDS))


@_register("ent_04_r01_missing_rendering_clinician")
def ent_04_r01_missing_rendering_clinician(ctx, control) -> list:
    """ENT-04-R01 — a service that needs a treating clinician has none, a facility code, or an unknown ID."""
    claims = _claims(ctx)
    lines = _lines(ctx, claims)
    ros = _roster(ctx)
    if lines.empty or ros.empty:
        return []
    ref = _activity_ref(ctx)
    lines = lines[_clinician_required(lines, ref)]
    # role requirement by activity: a diagnostic test (laboratory or imaging) is performed by the
    # department; the role it requires is a named ordering clinician, so one on the line satisfies it
    if not ref.empty and "service_family" in ref.columns:
        fam = _up(lines["activity_code"].map(ref.set_index("activity_code")["service_family"]).fillna(""))
        diagnostic = fam.isin({"LAB", "IMAGING", "ADVANCED_IMAGING"})
        lines = lines[~(diagnostic & lines["rendering"].isna() & lines["ordering"].notna())]
    prov = _providers(ctx)
    facility_ids = set(prov["provider_sk"].astype(str))
    for col in ("source_provider_id", "regulator_id", "licence_no"):
        if col in prov.columns:
            facility_ids |= set(prov[col].dropna().astype(str))
    known = set(ros["clinician_id"])
    rid = lines["rendering"]
    blank = rid.isna()
    facility_only = rid.isin(facility_ids) & ~rid.isin(known)
    unknown = rid.notna() & ~rid.isin(known) & ~facility_only
    lines = lines.assign(problem=np.select([blank, facility_only, unknown], ["missing", "facility_code", "not_on_roster"], ""))
    bad = lines[lines["problem"] != ""]
    out = []
    words = {"missing": "no treating clinician", "facility_code": "the facility's own code instead of a clinician",
             "not_on_roster": "a clinician ID that is not on any clinician roster"}
    for claim_sk, g in bad.groupby("claim_sk", sort=False):
        r0 = g.iloc[0]
        kinds = list(dict.fromkeys(g["problem"]))
        out.append(_sig(
            ctx, control, subject_type="claim", subject_id=claim_sk,
            fact_key=f"clinician_missing:{claim_sk}", claim_ids=[claim_sk],
            event_time=r0["line_date"], period=_period(r0["line_date"]),
            evidence={
                "provider_sk": r0["provider_sk"], "member_sk": r0["member_sk"],
                "lines": [{"line_sk": r.line_sk, "activity_code": r.activity_code, "rendering_clinician_id": r.rendering,
                           "problem": r.problem} for r in g.itertuples(index=False)],
                "reason_code": control.reason_code, "subject_id": claim_sk,
                "plain_language": (f"{count_phrase(len(g), 'service line')} (e.g. code {r0['activity_code']}) "
                                   f"{'carries' if len(g) == 1 else 'carry'} {words[kinds[0]]}, although the service "
                                   f"needs a named treating clinician."),
                "what_the_reviewer_must_verify": "Return the claim for the treating clinician to be added, then check "
                                                 "that clinician's licence.",
            },
            exposure=_exp.no_exposure("a claim returned for correction; nothing is established as not payable yet"),
        ))
    return out


@_register("ent_04_r02_billing_rendering_role_conflict")
def ent_04_r02_billing_rendering_role_conflict(ctx, control) -> list:
    """ENT-04-R02 — treating clinician not affiliated with the billing facility, or a non-independent role
    billed as the treating clinician for a physician-level service without a supervising physician."""
    claims = _claims(ctx)
    lines = _lines(ctx, claims)
    ros = _roster(ctx)
    if lines.empty or ros.empty:
        return []
    ref = _activity_ref(ctx)
    lines = lines.dropna(subset=["rendering"])
    lines = lines[lines["rendering"].isin(set(ros["clinician_id"]))]
    if lines.empty:
        return []
    aff = set(zip(ros["clinician_id"], ros["provider_sk"].astype(str)))
    home = ros.groupby("clinician_id")["provider_sk"].apply(lambda s: sorted(s.dropna().astype(str).unique()))
    not_affiliated = pd.Series([(c, p) not in aff for c, p in zip(lines["rendering"], lines["provider_sk"])],
                               index=lines.index)
    # team care / incident-to: a supervising (ordering) clinician affiliated with the billing facility
    supervised = pd.Series([(o, p) in aff for o, p in zip(lines["ordering"], lines["provider_sk"])], index=lines.index)
    limb_a = not_affiliated & ~supervised
    # role rule: a non-independent role cannot be the treating clinician on a physician-level service alone
    role = ros.groupby("clinician_id")["role"].first() if "role" in ros.columns else pd.Series(dtype=object)
    limb_b = pd.Series(False, index=lines.index)
    if len(role) and "specialty_required" in ref.columns:
        needs_phys = lines["activity_code"].map(ref.set_index("activity_code")["specialty_required"]).notna()
        r_role = _up(lines["rendering"].map(role).fillna(""))
        o_role = _up(lines["ordering"].map(role).fillna(""))
        non_ind = _has_word(r_role, _NON_INDEPENDENT_ROLE_WORDS)
        sup_ok = lines["ordering"].notna() & (lines["ordering"] != lines["rendering"]) & \
            ~_has_word(o_role, _NON_INDEPENDENT_ROLE_WORDS) & (o_role != "")
        limb_b = needs_phys & non_ind & ~sup_ok
    bad = lines[limb_a | limb_b].assign(limb=np.where(limb_a[limb_a | limb_b], "not_affiliated_with_billing_facility",
                                                      "non_independent_role_unsupervised"))
    out = []
    for claim_sk, g in bad.groupby("claim_sk", sort=False):
        r0 = g.iloc[0]
        if r0["limb"] == "not_affiliated_with_billing_facility":
            homes = home.get(r0["rendering"], [])
            txt = (f"billed by facility {r0['provider_sk']} but the treating clinician {r0['rendering']} is only on "
                   f"the roster of {', '.join(homes) if homes else 'another facility'}, and no in-house clinician "
                   f"ordered or supervised the service")
        else:
            txt = (f"the treating clinician {r0['rendering']} holds a {str(role.get(r0['rendering'], '')).lower()} role, "
                   f"which may not bill code {r0['activity_code']} without a supervising physician, and none is recorded")
        out.append(_sig(
            ctx, control, subject_type="claim", subject_id=claim_sk,
            fact_key=f"role:{claim_sk}", claim_ids=[claim_sk],
            event_time=r0["line_date"], period=_period(r0["line_date"]),
            evidence={
                "provider_sk": r0["provider_sk"], "member_sk": r0["member_sk"], "clinician_id": r0["rendering"],
                "limb_fired": sorted(set(g["limb"])),
                "lines": [{"line_sk": r.line_sk, "activity_code": r.activity_code, "rendering_clinician_id": r.rendering,
                           "ordering_clinician_id": r.ordering, "limb": r.limb} for r in g.itertuples(index=False)],
                "reason_code": control.reason_code, "subject_id": claim_sk,
                "plain_language": f"{count_phrase(len(g), 'line')} {'was' if len(g) == 1 else 'were'} {txt}.",
                "what_the_reviewer_must_verify": "Check the team-care and supervision policy and ask the provider who "
                                                 "delivered the service.",
            },
            exposure=_exp.no_exposure("a role conflict pends the claim; the payable amount depends on who delivered it"),
        ))
    return out


def _timed_lines(ctx, claims: pd.DataFrame) -> pd.DataFrame:
    lines = _lines(ctx, claims)
    if lines.empty:
        return lines
    lines = lines.dropna(subset=["rendering", "start", "end"])
    lines = lines[lines["end"] > lines["start"]]
    # date-only records (midnight to midnight) cannot show concurrency
    midnight = (lines["start"] == lines["start"].dt.normalize()) & (lines["end"] == lines["end"].dt.normalize())
    return lines[~midnight]


@_register("ent_04_r03_concurrent_clinician_services")
def ent_04_r03_concurrent_clinician_services(ctx, control) -> list:
    """ENT-04-R03 — a clinician with more overlapping timed services than cfg.max_concurrency."""
    claims = _claims(ctx)
    lines = _timed_lines(ctx, claims)
    if lines.empty:
        return []
    limit = int(ctx.cfg("ent04_max_concurrency"))
    ref = _activity_ref(ctx)
    if not ref.empty:
        info = ref.set_index("activity_code")
        timed = pd.Series(True, index=lines.index)
        if "is_time_based" in info.columns:
            timed = _truthy(lines["activity_code"].map(info["is_time_based"]).fillna(""))
        cf = lines["activity_code"].map(info["code_family"]).fillna("") if "code_family" in info.columns else \
            pd.Series("", index=lines.index)
        desc = lines["activity_code"].map(info["description"]).fillna("") if "description" in info.columns else \
            pd.Series("", index=lines.index)
        anaes = _has_word(cf, _ANAESTHESIA_WORDS) | _has_word(desc, _ANAESTHESIA_WORDS)
        lines = lines[timed & ~anaes]
    # team procedures: the same claim, code and start time carried by more than one clinician
    team = lines.groupby(["claim_sk", "activity_code", "start"])["rendering"].transform("nunique") > 1
    lines = lines[~team]
    if lines.empty:
        return []
    ev = pd.concat([
        pd.DataFrame({"clin": lines["rendering"].values, "time": lines["start"].values, "delta": 1,
                      "line_sk": lines["line_sk"].values}),
        pd.DataFrame({"clin": lines["rendering"].values, "time": lines["end"].values, "delta": -1,
                      "line_sk": lines["line_sk"].values}),
    ], ignore_index=True).sort_values(["clin", "time", "delta"])
    ev["conc"] = ev.groupby("clin")["delta"].cumsum()
    ev["day"] = pd.to_datetime(ev["time"]).dt.normalize()
    peaks = ev[ev["delta"] == 1].groupby(["clin", "day"])["conc"].max()
    peaks = peaks[peaks > limit]
    out = []
    lines = lines.assign(day=lines["start"].dt.normalize())
    for (clin, day), peak in peaks.items():
        sub = lines[(lines["rendering"] == clin) & (lines["day"] == day)].sort_values("start")
        # keep only the lines that overlap at least one other
        s, e = sub["start"].values, sub["end"].values
        ov = [(np.sum((s < e[i]) & (e > s[i])) - 1) > 0 for i in range(len(sub))]
        sub = sub[ov]
        if sub.empty:
            continue
        first = sub.iloc[0]
        window_txt = f"{first['start']:%H:%M}–{sub['end'].max():%H:%M}"
        out.append(_sig(
            ctx, control, subject_type="clinician", subject_id=clin,
            fact_key=f"concurrency:{clin}:{day.date()}", claim_ids=_ids(sub["claim_sk"]),
            event_time=day, period=_period(day), confidence=0.7,
            evidence={
                "clinician_id": clin, "provider_sk": _ids(sub["provider_sk"]), "date": _dstr(day),
                "max_concurrent_services": int(peak), "max_concurrency_allowed": limit,
                "lines": [{"line_sk": r.line_sk, "claim_sk": r.claim_sk, "activity_code": r.activity_code,
                           "start": f"{r.start:%H:%M}", "end": f"{r.end:%H:%M}"} for r in sub.itertuples(index=False)][:30],
                "reason_code": control.reason_code, "subject_id": clin,
                "plain_language": (f"On {plain_date(day)} this clinician was billed for {int(peak)} timed services "
                                   f"running at the same time ({count_phrase(len(sub), 'overlapping service')} between "
                                   f"{window_txt}); the limit is {limit}."),
                "what_the_reviewer_must_verify": "Build the clinician's timeline for the day and check theatre or "
                                                 "appointment records, team procedures and anaesthesia rules.",
            },
            exposure=_exp.model_only_exposure(float(sub["net_line"].sum()), "overlapping timed services"),
        ))
    return out


@_register("ent_04_r04_geographic_leave_impossibility")
def ent_04_r04_geographic_leave_impossibility(ctx, control) -> list:
    """ENT-04-R04 — clinician billed during recorded leave, or in two emirates too close together in time."""
    claims = _claims(ctx)
    lines = _lines(ctx, claims)
    ros = _roster(ctx)
    if lines.empty or ros.empty:
        return []
    lines = lines.dropna(subset=["rendering"])
    findings: list[dict[str, Any]] = []
    # leave limb
    if "leave_periods" in ros.columns:
        leave = []
        for clin, val in zip(ros["clinician_id"], ros["leave_periods"]):
            for s, e, kind in _parse_periods(val):
                leave.append((clin, s, e, kind))
        if leave:
            lv = pd.DataFrame(leave, columns=["rendering", "ls", "le", "kind"]).drop_duplicates()
            j = lines.merge(lv, on="rendering", how="inner")
            # a stay or episode that began before the leave may carry the clinician on later lines
            # (standing orders, attribution to the admitting team): only claims that START inside
            # the leave are counted
            j = j[(j["line_date"] >= j["ls"]) & (j["line_date"] <= j["le"]) & (j["service_date"] >= j["ls"])]
            for (clin, ls, le, kind), g in j.groupby(["rendering", "ls", "le", "kind"], sort=False):
                findings.append({"clin": clin, "limb": "leave", "claims": _ids(g["claim_sk"]),
                                 "date": g["line_date"].min(), "providers": _ids(g["provider_sk"]),
                                 "net": float(g.drop_duplicates("claim_sk")["net"].sum()),
                                 "detail": {"leave_from": _dstr(ls), "leave_to": _dstr(le), "leave_type": kind,
                                            "services_during_leave": int(len(g))},
                                 "key": f"leave:{clin}:{ls.date()}",
                                 "text": (f"This clinician was billed for {count_phrase(g['claim_sk'].nunique(), 'claim')} "
                                          f"between {plain_date(g['line_date'].min())} and "
                                          f"{plain_date(g['line_date'].max())}, while on recorded {kind} from "
                                          f"{plain_date(ls)} to {plain_date(le)}.")})
    # travel limb (needs service times; date-only records are excluded)
    prov = _providers(ctx)
    min_travel = int(ctx.cfg("ent04_min_travel_minutes"))
    if "emirate" in prov.columns:
        t = lines.dropna(subset=["start", "end"]).copy()
        t = t[~((t["start"] == t["start"].dt.normalize()) & (t["end"] == t["end"].dt.normalize()))]
        t["emirate"] = t["provider_sk"].map(prov.set_index("provider_sk")["emirate"])
        t = t.dropna(subset=["emirate"]).sort_values(["rendering", "start"])
        if not t.empty:
            t["prev_emirate"] = t.groupby("rendering")["emirate"].shift()
            t["prev_end"] = t.groupby("rendering")["end"].shift()
            t["prev_claim"] = t.groupby("rendering")["claim_sk"].shift()
            t["prev_provider"] = t.groupby("rendering")["provider_sk"].shift()
            gap = (t["start"] - t["prev_end"]).dt.total_seconds() / 60.0
            same_day = t["start"].dt.normalize() == t["prev_end"].dt.normalize()
            hop = t[same_day & t["prev_emirate"].notna() & (t["prev_emirate"] != t["emirate"]) & (gap < min_travel)]
            for (clin, day), g in hop.groupby(["rendering", hop["start"].dt.normalize()], sort=False):
                r0 = g.iloc[0]
                gm = (r0["start"] - r0["prev_end"]).total_seconds() / 60.0
                findings.append({"clin": clin, "limb": "travel", "claims": _ids(list(g["prev_claim"]) + list(g["claim_sk"])),
                                 "date": day, "providers": _ids(list(g["prev_provider"]) + list(g["provider_sk"])),
                                 "net": 0.0,
                                 "detail": {"from_emirate": r0["prev_emirate"], "to_emirate": r0["emirate"],
                                            "minutes_between": round(gm, 0), "minimum_travel_minutes": min_travel},
                                 "key": f"travel:{clin}:{day.date()}",
                                 "text": (f"On {plain_date(day)} this clinician finished a service in {r0['prev_emirate']} "
                                          f"at {r0['prev_end']:%H:%M} and started another in {r0['emirate']} at "
                                          f"{r0['start']:%H:%M} — {gm:.0f} minutes apart, less than the {min_travel} "
                                          f"minutes the trip needs.")})
    out = []
    net_by_claim = claims.set_index("claim_sk")["net"]
    for f in findings:
        amount = float(net_by_claim.reindex(f["claims"]).fillna(0).sum())
        out.append(_sig(
            ctx, control, subject_type="clinician", subject_id=f["clin"],
            fact_key=f["key"], claim_ids=f["claims"], event_time=f["date"], period=_period(f["date"]),
            confidence=0.7 if f["limb"] == "leave" else 0.6,
            evidence={
                "clinician_id": f["clin"], "provider_sk": f["providers"], "limb_fired": f["limb"], **f["detail"],
                "reason_code": control.reason_code, "subject_id": f["clin"],
                "plain_language": f["text"],
                "what_the_reviewer_must_verify": "Confirm the leave or travel dates with the facility and how reliable "
                                                 "the recorded times and places are.",
            },
            exposure=_exp.model_only_exposure(amount, "clinician presence pattern"),
        ))
    return out


# ===========================================================================
# ENT-05 — facility type and setting
# ===========================================================================


@_register("ent_05_r01_facility_type_incompatible")
def ent_05_r01_facility_type_incompatible(ctx, control) -> list:
    """ENT-05-R01 — a service whose permitted facility types do not include the provider's licensed type."""
    claims = _claims(ctx)
    lines = _lines(ctx, claims)
    ref = _activity_ref(ctx)
    prov = _providers(ctx)
    if lines.empty or ref.empty or "facility_types" not in ref.columns or "facility_type" not in prov.columns:
        return []
    allowed = ref.set_index("activity_code")["facility_types"].map(_allowed_tokens)
    allowed = allowed[allowed.map(len) > 0]
    l2 = lines[lines["activity_code"].isin(allowed.index)].copy()
    pinfo = prov.set_index("provider_sk")
    l2["ftype"] = _up(l2["provider_sk"].map(pinfo["facility_type"]).fillna(""))
    l2["ptype"] = _up(l2["provider_sk"].map(pinfo["provider_type"]).fillna("")) if "provider_type" in pinfo.columns \
        else ""
    l2 = l2[(l2["ftype"] != "") | (l2["ptype"] != "")]
    # the licence category may be recorded as the provider type (HOSPITAL, CLINIC, PHARMACY …) or the facility type
    ok = [ft in al or pt in al for ft, pt, al in zip(l2["ftype"], l2["ptype"], l2["activity_code"].map(allowed))]
    bad = l2[~pd.Series(ok, index=l2.index)]
    if bad.empty:
        return []
    setting = _claim_setting(ctx, claims[claims["claim_sk"].isin(set(bad["claim_sk"]))])
    exempt = setting.loc[_is_tele(setting["encounter_type"]) | _has_word(setting["encounter_type"], _HOME_MOBILE_WORDS),
                         "claim_sk"]
    bad = bad[~bad["claim_sk"].isin(set(exempt))]
    desc = ref.set_index("activity_code")["description"] if "description" in ref.columns else pd.Series(dtype=object)
    out = []
    for claim_sk, g in bad.groupby("claim_sk", sort=False):
        r0 = g.iloc[0]
        total = float(g["net_line"].sum())
        what = desc.get(r0["activity_code"]) if len(desc) else None
        what = f"{what} (code {r0['activity_code']})" if what and not is_missing(what) else f"code {r0['activity_code']}"
        out.append(_sig(
            ctx, control, subject_type="claim", subject_id=claim_sk,
            fact_key=f"facility_type:{claim_sk}", claim_ids=[claim_sk],
            event_time=r0["line_date"], period=_period(r0["line_date"]),
            evidence={
                "provider_sk": r0["provider_sk"], "member_sk": r0["member_sk"],
                "provider_facility_type": r0["ftype"],
                "lines": [{"line_sk": r.line_sk, "activity_code": r.activity_code,
                           "permitted_facility_types": sorted(allowed[r.activity_code]),
                           "net_amount_aed": round(_f(r.net_line), 2)} for r in g.itertuples(index=False)],
                "net_amount_aed": round(total, 2),
                "reason_code": control.reason_code, "subject_id": claim_sk,
                "provider_type": r0["ptype"] or None,
                "plain_language": (f"{what[0].upper() + what[1:]} was billed by a provider licensed as a "
                                   f"{(r0['ptype'] or r0['ftype']).replace('_', ' ').lower()}; that service may only be "
                                   f"billed by a "
                                   f"{' or '.join(sorted(t.replace('_', ' ').lower() for t in allowed[r0['activity_code']]))} "
                                   f"({aed(total)})."),
                "what_the_reviewer_must_verify": "Confirm the facility type on the provider's licence and whether a "
                                                 "home, mobile or telehealth exception applies.",
            },
            exposure=_exp.line_edit_exposure(total, 0.0),
        ))
    return out


@_register("ent_05_r02_admission_evidence_conflict")
def ent_05_r02_admission_evidence_conflict(ctx, control) -> list:
    """ENT-05-R02 — inpatient/ICU/day-case billing with no admission, bed or discharge record."""
    claims = _claims(ctx)
    if claims.empty:
        return []
    latency = int(ctx.cfg("ent05_admission_data_latency_days"))
    setting = _claim_setting(ctx, claims)
    ip = setting[_is_inpatient(setting["claim_type"]) | _is_inpatient(setting["encounter_type"])].copy()
    if ip.empty:
        return []
    horizon = claims["service_date"].max() - pd.Timedelta(days=latency)
    ip = ip[ip["service_date"] <= horizon]
    transfer = _has_word(ip["admission_type"].fillna(""), _TRANSFER_WORDS) | \
        _has_word(ip["discharge_type"].fillna(""), _TRANSFER_WORDS)
    ip = ip[~transfer]
    ip["no_encounter"] = ~ip["has_encounter"].astype(bool)
    ip["no_bed"] = ip["bed_id"].isna()
    ip["no_admission"] = ip["admission_date"].isna() & ip["start_time"].isna()
    ip["no_discharge"] = ip["discharge_date_enc"].isna() & ip["end_time"].isna() & ip["discharge_date"].isna()
    flags = ["no_encounter", "no_bed", "no_admission", "no_discharge"]
    bad = ip[ip[flags].any(axis=1)]
    words = {"no_encounter": "no encounter record", "no_bed": "no bed assigned",
             "no_admission": "no admission date", "no_discharge": "no discharge date"}
    out = []
    for r in bad.itertuples(index=False):
        missing = [w for w in flags if getattr(r, w)]
        if "no_encounter" in missing:
            missing = ["no_encounter"]
        setting_txt = (str(r.encounter_type or r.claim_type).lower() or "inpatient").replace("_", "-")
        out.append(_sig(
            ctx, control, subject_type="claim", subject_id=r.claim_sk,
            fact_key=f"admission_evidence:{r.claim_sk}", claim_ids=[r.claim_sk],
            event_time=r.service_date, period=_period(r.service_date), confidence=0.8,
            evidence={
                "provider_sk": r.provider_sk, "member_sk": r.member_sk, "setting_billed": setting_txt,
                "missing_evidence": missing, "gross_amount_aed": round(_f(r.gross), 2),
                "reason_code": control.reason_code, "subject_id": r.claim_sk,
                "plain_language": (f"Billed as {_article(setting_txt)} {setting_txt} stay ({aed(r.gross)}, {plain_date(r.service_date)}) but "
                                   f"the records show {' and '.join(words[m] for m in missing)}."),
                "what_the_reviewer_must_verify": "Ask for the admission and discharge notes; check whether the records "
                                                 "are late or the patient was transferred.",
            },
            exposure=_exp.no_exposure("missing admission evidence pends the claim; whether the stay happened is not "
                                      "yet established"),
        ))
    return out


@_register("ent_05_r03_related_claim_setting_conflict")
def ent_05_r03_related_claim_setting_conflict(ctx, control) -> list:
    """ENT-05-R03 — the same stay at one provider billed as inpatient on one claim and outpatient on another."""
    claims = _claims(ctx)
    if claims.empty:
        return []
    setting = _claim_setting(ctx, claims)
    ip_mask = _is_inpatient(setting["claim_type"]) | _is_inpatient(setting["encounter_type"])
    ip = setting[ip_mask].copy()
    op = setting[~ip_mask & ~_is_tele(setting["encounter_type"]) & ~_is_emergency(setting["encounter_type"])]
    if ip.empty or op.empty:
        return []
    # cross-facility transfer: a stay that began or ended with a transfer is left out
    ip = ip[~(_has_word(ip["admission_type"].fillna(""), _TRANSFER_WORDS)
              | _has_word(ip["discharge_type"].fillna(""), _TRANSFER_WORDS))]
    ip["adm"] = ip["admission_date"].fillna(ip["service_date"])
    ip["dis"] = ip["discharge_date_enc"].fillna(ip["discharge_date"])
    ip = ip.dropna(subset=["adm", "dis"])
    ip = ip[ip["dis"] > ip["adm"]]
    prov = _providers(ctx)
    if "provider_type" in prov.columns or "facility_type" in prov.columns:
        pt = _up(prov.set_index("provider_sk").get("provider_type", pd.Series(dtype=object)).reindex(prov["provider_sk"]).fillna("")
                 ) + " " + _up(prov.set_index("provider_sk").get("facility_type", pd.Series(dtype=object)).reindex(prov["provider_sk"]).fillna(""))
        indep = set(prov["provider_sk"][pt.str.contains("INDEPENDENT|PRACTITIONER", regex=True).values])
        op = op[~op["provider_sk"].isin(indep)]
    j = ip[["claim_sk", "member_sk", "provider_sk", "adm", "dis", "gross", "encounter_sk"]].merge(
        op[["claim_sk", "member_sk", "provider_sk", "service_date", "gross", "encounter_type", "claim_type", "encounter_sk"]],
        on=["member_sk", "provider_sk"], suffixes=("_ip", "_op"))
    same_enc = j["encounter_sk_ip"].notna() & (j["encounter_sk_ip"] == j["encounter_sk_op"])
    inside = (j["service_date"] > j["adm"]) & (j["service_date"] < j["dis"])
    j = j[(inside | same_enc) & (j["claim_sk_ip"] != j["claim_sk_op"])]
    out = []
    for r in j.itertuples(index=False):
        a, b = sorted([r.claim_sk_ip, r.claim_sk_op])
        op_setting = (r.encounter_type or r.claim_type or "outpatient").lower()
        out.append(_sig(
            ctx, control, subject_type="claim", subject_id=r.claim_sk_op,
            fact_key=f"setting_conflict:{a}:{b}", claim_ids=[r.claim_sk_ip, r.claim_sk_op],
            event_time=r.service_date, period=_period(r.service_date), confidence=0.8,
            evidence={
                "provider_sk": r.provider_sk, "member_sk": r.member_sk,
                "inpatient_claim_sk": r.claim_sk_ip, "other_claim_sk": r.claim_sk_op,
                "admission_date": _dstr(r.adm), "discharge_date": _dstr(r.dis),
                "other_claim_date": _dstr(r.service_date), "other_claim_setting": op_setting,
                "inpatient_gross_aed": round(_f(r.gross_ip), 2), "other_gross_aed": round(_f(r.gross_op), 2),
                "reason_code": control.reason_code, "subject_id": r.claim_sk_op,
                "plain_language": (f"The patient was an inpatient at this provider from {plain_date(r.adm)} to "
                                   f"{plain_date(r.dis)} (claim {r.claim_sk_ip}, {aed(r.gross_ip)}), yet claim "
                                   f"{r.claim_sk_op} bills {_article(op_setting)} {op_setting} visit at the same provider on "
                                   f"{plain_date(r.service_date)} ({aed(r.gross_op)})."),
                "what_the_reviewer_must_verify": "Compare both claims side by side and check for a transfer or an "
                                                 "independent practitioner.",
            },
            exposure=_exp.no_exposure("which of the two settings is correct is not established"),
        ))
    return out


@_register("ent_05_r04_setting_shift_anomaly")
def ent_05_r04_setting_shift_anomaly(ctx, control) -> list:
    """ENT-05-R04 — a provider's share of better-paid (inpatient/day-case) settings jumps and tops its peers."""
    claims = _claims(ctx)
    if claims.empty:
        return []
    min_delta = float(ctx.cfg("ent05_setting_shift_min_delta"))
    min_months = int(ctx.cfg("ent05_setting_shift_min_months"))
    min_claims = int(ctx.cfg("ent05_setting_shift_min_claims"))
    q = float(ctx.cfg("ent05_setting_shift_peer_percentile"))
    setting = _claim_setting(ctx, claims)
    setting = setting.dropna(subset=["service_date", "provider_sk"])
    setting["high"] = (_is_inpatient(setting["claim_type"]) | _is_inpatient(setting["encounter_type"])).astype(int)
    setting["month"] = setting["service_date"].dt.to_period("M")
    monthly = setting.groupby(["provider_sk", "month"]).agg(n=("high", "size"), h=("high", "sum")).reset_index()
    prov = _providers(ctx)
    info = prov.set_index("provider_sk") if not prov.empty else pd.DataFrame()
    ftype = info["facility_type"] if "facility_type" in info.columns else pd.Series(dtype=object)
    overall = setting.groupby("provider_sk")["high"].mean()
    mean_by_setting = setting.groupby("high")["gross"].mean()
    rows = []
    for p, g in monthly.groupby("provider_sk", sort=False):
        g = g.sort_values("month")
        if len(g) < 2 * min_months:
            continue
        cn, ch = g["n"].cumsum().values, g["h"].cumsum().values
        tn, th = cn[-1], ch[-1]
        best = None
        for k in range(min_months, len(g) - min_months + 1):
            nb, hb = cn[k - 1], ch[k - 1]
            na, ha = tn - nb, th - hb
            if nb < min_claims or na < min_claims:
                continue
            d = ha / na - hb / nb
            if best is None or d > best[0]:
                best = (d, k, hb / nb, ha / na, int(nb), int(na))
        if best is None:
            continue
        rows.append((p, g["month"].iloc[best[1]], *best))
    if not rows:
        return []
    # peer yardstick: the largest rise each other provider shows at its own best change point
    deltas = pd.Series({r[0]: r[2] for r in rows})
    rows = [r for r in rows if r[2] >= min_delta]
    own_change = _d(info["ownership_changed_on"]) if "ownership_changed_on" in info.columns else pd.Series(dtype="datetime64[ns]")
    contract = _table(ctx, "contract")
    c_start = pd.Series(dtype="datetime64[ns]")
    if not contract.empty and "provider_sk" in contract.columns:
        c_start = pd.DataFrame({"p": _s(contract["provider_sk"]), "vf": _d(_col(contract, "valid_from"))}).groupby("p")["vf"].apply(list)
    min_peer = int(ctx.cfg("min_entity_opportunities"))
    out = []
    flagged = {r[0] for r in rows}
    for p, split_month, delta, k, before, after, nb, na in rows:
        split = split_month.to_timestamp()
        near = lambda dt: dt is not None and not pd.isna(dt) and abs((dt - split).days) <= 60  # noqa: E731
        if near(own_change.get(p) if len(own_change) else None):
            continue
        if len(c_start) and any(near(x) for x in c_start.get(p, [])):
            continue
        ft = ftype.get(p) if len(ftype) else None
        peers = deltas[[x for x in deltas.index if x not in flagged and (ft is None or ftype.get(x) == ft)]]
        if len(peers) < min_peer:
            peers = deltas[[x for x in deltas.index if x not in flagged]]
        if len(peers) < min_peer:
            continue
        cut = float(np.quantile(peers, q))
        if delta <= cut:
            continue
        peer_share = overall[[x for x in overall.index if x not in flagged and (ft is None or ftype.get(x) == ft)]]
        sub = setting[(setting["provider_sk"] == p) & (setting["service_date"] >= split) & (setting["high"] == 1)]
        out.append(_sig(
            ctx, control, subject_type="provider", subject_id=p,
            fact_key=f"setting_shift:{p}", claim_ids=sub["claim_sk"].tolist()[:200],
            event_time=split, period=_period(split), confidence=0.6,
            peer_level_used=f"facility_type={ft}" if ft is not None else "all providers",
            evidence={
                "provider_sk": p, "change_month": str(split_month), "share_before": round(before, 3),
                "share_after": round(after, 3), "claims_before": nb, "claims_after": na, "rise": round(delta, 3),
                "peer_percentile": q, "peer_cut_rise": round(cut, 3), "peer_n": int(len(peers)),
                "peer_median_share": round(float(peer_share.median()), 3) if len(peer_share) else None,
                "mean_gross_inpatient_aed": round(float(mean_by_setting.get(1, np.nan)), 2),
                "mean_gross_outpatient_aed": round(float(mean_by_setting.get(0, np.nan)), 2),
                "reason_code": control.reason_code, "subject_id": p,
                "plain_language": (f"From {split:%B %Y} {pct(after)} of this provider's claims were billed as inpatient "
                                   f"or day-case, up from {pct(before)} before — a rise of {round(delta * 100)} "
                                   f"points; similar providers' mix moved by at most {round(cut * 100)} points "
                                   f"(the {pct(q)} mark)."),
                "what_the_reviewer_must_verify": "Check for a contract or facility change around the shift and sample "
                                                 "claims from after it for record review.",
            },
            exposure=_exp.model_only_exposure(float(sub["gross"].sum()), "setting-shift pattern"),
        ))
    return out


# ===========================================================================
# ENT-06 — telehealth
# ===========================================================================


def _tele_claims(ctx, claims: pd.DataFrame) -> pd.DataFrame:
    setting = _claim_setting(ctx, claims)
    return setting[_is_tele(setting["encounter_type"]) | _is_tele(setting["claim_type"])]


@_register("ent_06_r01_telehealth_eligibility")
def ent_06_r01_telehealth_eligibility(ctx, control) -> list:
    """ENT-06-R01 — a telehealth visit billing a service not eligible for remote delivery, or a provider
    without a telehealth privilege on the date."""
    claims = _claims(ctx)
    tele = _tele_claims(ctx, claims)
    ref = _activity_ref(ctx)
    if tele.empty or ref.empty or "is_telehealth_eligible" not in ref.columns:
        return []
    lines = _lines(ctx, claims[claims["claim_sk"].isin(set(tele["claim_sk"]))])
    if lines.empty:
        return []
    elig = ref.set_index("activity_code")["is_telehealth_eligible"]
    l2 = lines[lines["activity_code"].isin(elig.index)].copy()
    l2["ineligible"] = _falsy(l2["activity_code"].map(elig).fillna(""))
    # non-clinical lines (drugs, supplies) follow the order, not the visit
    l2 = l2[_clinician_required(l2, ref) | l2["ineligible"]]
    bad_lines = l2[l2["ineligible"]]
    # provider privilege limb (only when the register records telehealth privileges at all)
    st = _status_rows(ctx)
    tp = st[st["status_type"].str.contains("TELE") | st["status_value"].str.contains("TELE")]
    no_priv: set[str] = set()
    if not tp.empty:
        good = tp[~_has_word(tp["status_value"], _BAD_LICENCE_WORDS)]
        j = tele[["claim_sk", "provider_sk", "service_date"]].merge(good, on="provider_sk", how="left")
        ok = (j["service_date"] >= j["vf"].fillna(pd.Timestamp.min)) & ((j["service_date"] <= j["vt"]) | j["vt"].isna()) & j["status_type"].notna()
        has = j.assign(ok=ok).groupby("claim_sk")["ok"].any()
        no_priv = set(has[~has].index)
    ids = set(bad_lines["claim_sk"]) | no_priv
    if not ids:
        return []
    desc = ref.set_index("activity_code")["description"] if "description" in ref.columns else pd.Series(dtype=object)
    out = []
    tinfo = tele.set_index("claim_sk")
    for claim_sk in sorted(ids):
        g = bad_lines[bad_lines["claim_sk"] == claim_sk]
        r = tinfo.loc[claim_sk]
        limbs = []
        if not g.empty:
            limbs.append("service_not_eligible_remotely")
        if claim_sk in no_priv:
            limbs.append("provider_without_telehealth_privilege")
        if not g.empty:
            code = g.iloc[0]["activity_code"]
            what = desc.get(code) if len(desc) else None
            what = f"{what} (code {code})" if what and not is_missing(what) else f"code {code}"
            amount = float(g["net_line"].sum())
            text = (f"A telehealth visit on {plain_date(r['service_date'])} billed {what} ({aed(amount)}), a service "
                    f"that cannot be delivered remotely.")
        else:
            amount = _f(r["net"])
            text = (f"A telehealth visit on {plain_date(r['service_date'])} ({aed(amount)}) was billed by a provider "
                    f"with no telehealth privilege on the register for that date.")
        out.append(_sig(
            ctx, control, subject_type="claim", subject_id=claim_sk,
            fact_key=f"telehealth:{claim_sk}", claim_ids=[claim_sk],
            event_time=r["service_date"], period=_period(r["service_date"]),
            evidence={
                "provider_sk": r["provider_sk"], "member_sk": r["member_sk"], "limb_fired": limbs,
                "ineligible_lines": [{"line_sk": x.line_sk, "activity_code": x.activity_code,
                                      "net_amount_aed": round(_f(x.net_line), 2)} for x in g.itertuples(index=False)],
                "reason_code": control.reason_code, "subject_id": claim_sk,
                "plain_language": text,
                "what_the_reviewer_must_verify": "Check the regulator and product telehealth rules and the provider's "
                                                 "telehealth privileges on the date.",
            },
            exposure=_exp.line_edit_exposure(amount, 0.0),
        ))
    return out


def _downstream_orders(ctx, members: set[str]) -> pd.DataFrame:
    """Prescriptions and referrals, as (member, clinician, provider, date, kind, recipient, amount)."""
    parts = []
    rx = _table(ctx, "prescription_dispense")
    if not rx.empty:
        date = _d(_col(rx, "prescribed_date")).fillna(_d(_col(rx, "fill_date")))
        amt = _num(_col(rx, "billed_amount"))
        parts.append(pd.DataFrame({
            "member_sk": _s(_col(rx, "member_sk")), "clinician": _s(_col(rx, "prescriber_id")),
            "referrer_provider": np.nan, "date": date, "kind": "prescription",
            "recipient": _s(_col(rx, "pharmacy_id")), "product": _s(_col(rx, "billed_product")).fillna(_s(_col(rx, "dispensed_product"))),
            "amount": amt, "claim_sk": _s(_col(rx, "claim_sk")),
        }))
    rf = _table(ctx, "referral")
    if not rf.empty:
        parts.append(pd.DataFrame({
            "member_sk": _s(_col(rf, "member_sk")), "clinician": _s(_col(rf, "referrer_id")),
            "referrer_provider": _s(_col(rf, "referrer_provider_sk")), "date": _d(_col(rf, "referral_date")),
            "kind": "referral", "recipient": _s(_col(rf, "recipient_provider_sk")).fillna(_s(_col(rf, "recipient_id"))),
            "product": _s(_col(rf, "specialty")), "amount": np.nan, "claim_sk": _s(_col(rf, "resulting_claim_sk")),
        }))
    if not parts:
        return pd.DataFrame(columns=["member_sk", "clinician", "referrer_provider", "date", "kind", "recipient",
                                     "product", "amount", "claim_sk"])
    out = pd.concat(parts, ignore_index=True)
    return out[out["member_sk"].isin(members)] if members else out


def _encounter_clinician(lines: pd.DataFrame) -> pd.Series:
    """claim_sk -> the treating clinician of the claim (most frequent rendering id)."""
    l2 = lines.dropna(subset=["rendering"])
    if l2.empty:
        return pd.Series(dtype=object)
    cnt = l2.groupby(["claim_sk", "rendering"]).size().reset_index(name="n")
    cnt = cnt.sort_values(["claim_sk", "n", "rendering"], ascending=[True, False, True]).drop_duplicates("claim_sk")
    return cnt.set_index("claim_sk")["rendering"]


def _orders_after(enc: pd.DataFrame, orders: pd.DataFrame, window: int) -> pd.DataFrame:
    """Orders by the encounter's clinician (or provider) for the same member within ``window`` days after."""
    if enc.empty or orders.empty:
        return pd.DataFrame(columns=["claim_sk", "kind", "recipient", "amount", "product", "order_claim_sk"])
    e = enc[["claim_sk", "member_sk", "provider_sk", "clinician", "service_date"]]
    o = orders.rename(columns={"claim_sk": "order_claim_sk"})
    by_clin = e.merge(o.drop(columns=["referrer_provider"]), on=["member_sk", "clinician"], how="inner")
    by_prov = e.merge(o.dropna(subset=["referrer_provider"]).drop(columns=["clinician"]).rename(
        columns={"referrer_provider": "provider_sk"}), on=["member_sk", "provider_sk"], how="inner")
    a = pd.concat([by_clin, by_prov], ignore_index=True)
    delta = (a["date"] - a["service_date"]).dt.days
    a = a[(delta >= 0) & (delta <= window)]
    keys = [k for k in ("claim_sk", "kind", "recipient", "date", "order_claim_sk", "product") if k in a.columns]
    return a.drop_duplicates(subset=keys)


@_register("ent_06_r02_no_meaningful_interaction")
def ent_06_r02_no_meaningful_interaction(ctx, control) -> list:
    """ENT-06-R02 — a remote visit that produced an order but lasted below the policy minimum, or has no note
    where this provider's teleconsults normally carry one."""
    claims = _claims(ctx)
    tele = _tele_claims(ctx, claims)
    if tele.empty:
        return []
    min_minutes = float(ctx.cfg("ent06_min_interaction_minutes"))
    window = int(ctx.cfg("ent06_order_window_days"))
    note_cov = float(ctx.cfg("ent06_note_expected_coverage"))
    tele = tele[~_is_emergency(tele["encounter_type"]) & ~_has_word(tele["admission_type"].fillna(""), ("ASYNC", "STORE"))
                & ~_has_word(tele["observation_status"].fillna(""), ("ASYNC", "STORE"))].copy()
    lines = _lines(ctx, claims[claims["claim_sk"].isin(set(tele["claim_sk"]))])
    tele["clinician"] = tele["claim_sk"].map(_encounter_clinician(lines))
    dur = tele["duration_minutes"]
    missing_dur = dur.isna() & tele["start_time"].notna() & tele["end_time"].notna()
    dur = dur.where(~missing_dur, (tele["end_time"] - tele["start_time"]).dt.total_seconds() / 60.0)
    tele["dur"] = dur
    # notes: a document on the claim or an observation with an attachment
    docs = _table(ctx, "document")
    obs = _table(ctx, "observation")
    noted: set[str] = set()
    if not docs.empty and "claim_sk" in docs.columns:
        noted |= set(_s(docs["claim_sk"]).dropna())
    if not obs.empty and "claim_sk" in obs.columns:
        o = obs[_s(_col(obs, "attachment_ref")).notna()]
        noted |= set(_s(o["claim_sk"]).dropna())
    tele["has_note"] = tele["claim_sk"].isin(noted)
    prov_cov = tele.groupby("provider_sk")["has_note"].mean()
    note_limb_on = tele["provider_sk"].map(prov_cov).fillna(0) >= note_cov
    orders = _downstream_orders(ctx, set(tele["member_sk"].dropna()))
    ordered = _orders_after(tele, orders, window)
    n_orders = ordered.groupby("claim_sk").size()
    # own-claim orders: lab/imaging lines on the same claim
    tele["orders"] = tele["claim_sk"].map(n_orders).fillna(0).astype(int)
    tele["short"] = tele["dur"].notna() & (tele["dur"] < min_minutes)
    tele["no_note"] = note_limb_on & ~tele["has_note"]
    bad = tele[(tele["orders"] > 0) & (tele["short"] | tele["no_note"])]
    out = []
    for r in bad.itertuples(index=False):
        od = ordered[ordered["claim_sk"] == r.claim_sk]
        limbs = []
        facts = []
        if r.short:
            limbs.append("duration_below_minimum")
            facts.append(f"lasted {r.dur:.0f} minute(s) (policy minimum {min_minutes:.0f})")
        if r.no_note:
            limbs.append("no_consultation_note")
            facts.append("has no consultation note, although this provider's other teleconsults do")
        kinds = od["kind"].value_counts().to_dict()
        kinds_txt = ", ".join(count_phrase(v, k) for k, v in kinds.items())
        out.append(_sig(
            ctx, control, subject_type="claim", subject_id=r.claim_sk,
            fact_key=f"tele_interaction:{r.claim_sk}", claim_ids=[r.claim_sk] + _ids(od["order_claim_sk"]),
            event_time=r.service_date, period=_period(r.service_date), confidence=0.7,
            evidence={
                "provider_sk": r.provider_sk, "member_sk": r.member_sk, "clinician_id": r.clinician,
                "duration_minutes": None if pd.isna(r.dur) else round(float(r.dur), 1),
                "minimum_minutes": min_minutes, "has_consultation_note": bool(r.has_note),
                "orders_within_days": window, "orders": kinds, "limb_fired": limbs,
                "reason_code": control.reason_code, "subject_id": r.claim_sk,
                "plain_language": (f"A telehealth visit on {plain_date(r.service_date)} " + " and ".join(facts)
                                   + f", yet it led to {kinds_txt} within {window} days."),
                "what_the_reviewer_must_verify": "Ask for the consultation note and call record and send the case for "
                                                 "clinical review.",
            },
            exposure=_exp.no_exposure("whether the resulting orders were justified is a clinical question, not established"),
        ))
    return out


def _tele_providers(ctx, claims: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    setting = _claim_setting(ctx, claims)
    setting["is_tele"] = _is_tele(setting["encounter_type"]) | _is_tele(setting["claim_type"])
    share = setting.groupby("provider_sk")["is_tele"].mean()
    return setting, share


@_register("ent_06_r03_downstream_referral_concentration")
def ent_06_r03_downstream_referral_concentration(ctx, control) -> list:
    """ENT-06-R03 — a telehealth provider sending most of its downstream orders to one lab or pharmacy."""
    claims = _claims(ctx)
    if claims.empty:
        return []
    min_share = float(ctx.cfg("referral_concentration_threshold"))
    min_orders = int(ctx.cfg("ent06_referral_concentration_min_orders"))
    q = float(ctx.cfg("ent06_referral_peer_percentile"))
    tele_share_min = float(ctx.cfg("ent06_telehealth_provider_min_share"))
    narrow_min = int(ctx.cfg("ent06_narrow_network_min_suppliers"))
    setting, share = _tele_providers(ctx, claims)
    orders = _downstream_orders(ctx, set())
    ros = _roster(ctx)
    if orders.empty:
        return []
    # attribute each order to the ordering provider: the referrer provider, else the prescriber's roster provider
    clin_home = ros.drop_duplicates("clinician_id").set_index("clinician_id")["provider_sk"] if not ros.empty else pd.Series(dtype=object)
    orders = orders.copy()
    orders["from_provider"] = orders["referrer_provider"].fillna(orders["clinician"].map(clin_home))
    # prescribers on several rosters: prefer a telehealth provider they belong to
    if not ros.empty:
        multi = ros.groupby("clinician_id")["provider_sk"].apply(list)
        tele_set = set(share[share >= tele_share_min].index)
        pref = {c: next((p for p in ps if p in tele_set), None) for c, ps in multi.items() if len(ps) > 1}
        repl = orders["clinician"].map(pref)
        orders["from_provider"] = repl.where(repl.notna() & orders["referrer_provider"].isna(), orders["from_provider"])
    orders = orders.dropna(subset=["from_provider", "recipient"])
    # dispensing by the ordering provider's own pharmacy is in-house, not a downstream referral
    orders = orders[orders["from_provider"] != orders["recipient"]]
    counts = orders.groupby(["from_provider", "recipient"]).size().rename("n").reset_index()
    tot = counts.groupby("from_provider")["n"].sum()
    top = counts.sort_values("n", ascending=False).drop_duplicates("from_provider").set_index("from_provider")
    top["total"] = tot
    top["share"] = top["n"] / top["total"]
    eligible = top[top["total"] >= min_orders]
    if eligible.empty:
        return []
    min_peer = int(ctx.cfg("min_entity_opportunities"))
    tele_prov = [p for p in eligible.index if share.get(p, 0) >= tele_share_min]
    prov = _providers(ctx)
    info = prov.set_index("provider_sk") if not prov.empty else pd.DataFrame()
    others = eligible.drop(index=tele_prov, errors="ignore")
    ptype_of = info["provider_type"] if "provider_type" in info.columns else pd.Series(dtype=object)
    owner = info["owner_entity_id"] if "owner_entity_id" in info.columns else pd.Series(dtype=object)
    emir = info["emirate"] if "emirate" in info.columns else pd.Series(dtype=object)
    ftype = _up(info["facility_type"]) if "facility_type" in info.columns else pd.Series(dtype=object)
    out = []
    for p in tele_prov:
        row = eligible.loc[p]
        # peers: other prescribing providers of the same provider type (a clinic is compared with clinics)
        peers = others[others.index.map(ptype_of).astype(str) == str(ptype_of.get(p))]["share"] \
            if len(ptype_of) else others["share"]
        if len(peers) < min_peer:
            peers = others["share"]
        if len(peers) < min_peer:
            continue
        cut = max(min_share, float(np.quantile(peers, q)))
        if row["share"] <= cut:
            continue
        rec = row["recipient"]
        o1, o2 = owner.get(p), owner.get(rec)
        if o1 is not None and o2 is not None and not is_missing(o1) and o1 == o2:
            continue  # corporate ownership: in-group routing is declared legitimate
        if len(ftype) and len(emir):
            same_kind = ftype[(ftype == ftype.get(rec, "")) & (emir == emir.get(rec))]
            if len(same_kind) < narrow_min:
                continue  # narrow network: there are hardly any alternatives
        sub = orders[(orders["from_provider"] == p) & (orders["recipient"] == rec)]
        kind = sub["kind"].mode().iloc[0] if not sub.empty else "order"
        rec_kind = str(ftype.get(rec, "supplier")).lower().replace("_", " ") if len(ftype) else "supplier"
        amount = float(sub["amount"].fillna(0).sum())
        first = sub["date"].min()
        out.append(_sig(
            ctx, control, subject_type="provider", subject_id=p,
            fact_key=f"downstream:{p}:{rec}", claim_ids=_ids(sub["claim_sk"])[:200],
            event_time=first, period=_period(first), confidence=0.6,
            evidence={
                "provider_sk": p, "recipient_provider_sk": rec, "recipient_type": rec_kind,
                "orders_to_recipient": int(row["n"]), "orders_total": int(row["total"]),
                "share_to_recipient": round(float(row["share"]), 3), "peer_percentile": q,
                "peer_cut_share": round(cut, 3), "peer_median_share": round(float(peers.median()), 3) if len(peers) else None,
                "telehealth_share_of_visits": round(float(share.get(p, 0)), 3), "dominant_order_kind": kind,
                "reason_code": control.reason_code, "subject_id": p,
                "plain_language": (f"{pct(row['share'])} of this telehealth provider's {int(row['total'])} downstream "
                                   f"orders went to one {rec_kind} ({rec}); for other prescribing providers the top "
                                   f"supplier's share is typically {pct(float(peers.median()) if len(peers) else 0)}."),
                "what_the_reviewer_must_verify": "Check for shared ownership or a narrow network and review a sample "
                                                 "of orders sent to that supplier.",
            },
            exposure=_exp.model_only_exposure(amount, "downstream referral concentration"),
        ))
    return out


@_register("ent_06_r04_remote_order_conversion_spike")
def ent_06_r04_remote_order_conversion_spike(ctx, control) -> list:
    """ENT-06-R04 — a telehealth provider's visits turn into high-cost orders far more often than peers'."""
    claims = _claims(ctx)
    if claims.empty:
        return []
    window = int(ctx.cfg("ent06_order_window_days"))
    high_cost = float(ctx.cfg("ent06_high_cost_order_aed"))
    min_enc = int(ctx.cfg("ent06_conversion_min_encounters"))
    ratio_min = float(ctx.cfg("ent06_conversion_ratio_to_peer"))
    q = float(ctx.cfg("ent06_conversion_peer_percentile"))
    tele_share_min = float(ctx.cfg("ent06_telehealth_provider_min_share"))
    setting, share = _tele_providers(ctx, claims)
    consult = setting[~(_is_inpatient(setting["claim_type"]) | _is_inpatient(setting["encounter_type"]))].copy()
    lines = _lines(ctx, claims[claims["claim_sk"].isin(set(consult["claim_sk"]))])
    consult["clinician"] = consult["claim_sk"].map(_encounter_clinician(lines))
    consult = consult.dropna(subset=["clinician"])
    orders = _downstream_orders(ctx, set(consult["member_sk"].dropna()))
    if orders.empty or consult.empty:
        return []
    # high-cost: a dispensed product flagged high-cost, a billed amount above the threshold,
    # or a referral whose resulting claim is above the threshold
    drug = _table(ctx, "drug_policy")
    hc_products = set()
    if not drug.empty and "is_high_cost" in drug.columns:
        hc_products = set(_s(drug.loc[_truthy(drug["is_high_cost"]), "product"]).dropna())
    res_amt = orders["claim_sk"].map(claims.set_index("claim_sk")["gross"])
    orders = orders.assign(hc=orders["product"].isin(hc_products) | (orders["amount"].fillna(0) >= high_cost)
                           | (res_amt.fillna(0) >= high_cost))
    hc_orders = orders[orders["hc"]]
    conv = _orders_after(consult, hc_orders, window)
    consult["converted"] = consult["claim_sk"].isin(set(conv["claim_sk"]))
    # rates: telehealth visits of telehealth providers vs all providers' consult visits
    rate_all = consult.groupby("provider_sk").agg(n=("converted", "size"), k=("converted", "sum"))
    rate_all = rate_all[rate_all["n"] >= min_enc]
    tele_v = consult[consult["is_tele"]]
    rate_tele = tele_v.groupby("provider_sk").agg(n=("converted", "size"), k=("converted", "sum"))
    rate_tele = rate_tele[(rate_tele["n"] >= min_enc) & rate_tele.index.isin(share[share >= tele_share_min].index)]
    if rate_tele.empty:
        return []
    peers = (rate_all["k"] / rate_all["n"]).drop(index=rate_tele.index, errors="ignore")
    min_peer = int(ctx.cfg("min_entity_opportunities"))
    if len(peers) < min_peer:
        return []
    med = float(peers.median())
    cut = float(np.quantile(peers, q))
    out = []
    for p, r in rate_tele.iterrows():
        rate = r["k"] / r["n"]
        if rate <= cut or rate < ratio_min * max(med, 1e-9):
            continue
        cv = conv[conv["provider_sk"] == p]
        vis = tele_v[(tele_v["provider_sk"] == p) & tele_v["converted"]]
        first = vis["service_date"].min()
        out.append(_sig(
            ctx, control, subject_type="provider", subject_id=p,
            fact_key=f"conversion:{p}", claim_ids=_ids(list(vis["claim_sk"]) + list(cv["order_claim_sk"]))[:200],
            event_time=first, period=_period(first), confidence=0.6, peer_level_used="all consulting providers",
            evidence={
                "provider_sk": p, "telehealth_visits": int(r["n"]), "converted_visits": int(r["k"]),
                "conversion_rate": round(float(rate), 3), "peer_median_rate": round(med, 3),
                "peer_percentile": q, "peer_cut_rate": round(cut, 3), "peer_n": int(len(peers)),
                "high_cost_threshold_aed": high_cost, "order_window_days": window,
                "reason_code": control.reason_code, "subject_id": p,
                "plain_language": (f"{pct(rate)} of this provider's {int(r['n'])} telehealth visits led to a high-cost "
                                   f"order (over {aed(high_cost)}) within {window} days; for similar providers it is "
                                   f"{pct(med)}."),
                "what_the_reviewer_must_verify": "Check the provider's specialty, patients' diagnoses and any campaign "
                                                 "or new service, then review a sample of the orders.",
            },
            exposure=_exp.model_only_exposure(float(cv["amount"].fillna(0).sum()), "remote-order conversion"),
        ))
    return out
