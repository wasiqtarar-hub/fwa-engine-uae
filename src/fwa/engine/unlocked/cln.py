"""Unlocked implementations for the CLN family.

Each function here runs only when a dataset unlock (`rules/unlocks/CLN.yaml`)
is satisfied by the loaded file. See :mod:`fwa.engine.unlocks`.

These are the clinical and coding controls the claim-header extract cannot
support: they need claim lines, the full ordered diagnosis list, results,
referrals, member confirmations, equipment inventories and the coding
reference tables. Conventions shared with :mod:`fwa.engine.controls`:

* every threshold is a governed ``ctx.cfg`` lookup (``cln..`` keys in
  ``config/parameters.yaml``);
* every signal is built through ``_sig`` so evidence, exposure and the
  effective data-support classification are stamped uniformly;
* a missing table, a missing column or an empty frame returns ``[]`` — the
  ``_register`` wrapper additionally guarantees that no exception escapes to
  the evaluator (it is kept in :data:`LAST_ERRORS` for diagnosis instead);
* statistical and pattern controls never present a gross amount as
  established exposure.
"""

from __future__ import annotations

import functools
import re
import traceback
from typing import Any, Callable

import numpy as np
import pandas as pd

from ...cases import exposure as _exp
from ...presentation import aed, count_phrase, pct, plain_date, times_phrase
from ...statistical.robust import robust_stats
from ...statistical.shrinkage import fit_beta_prior, shrink_rate
from ..controls import _claim_frame, _f, _sig, _ts  # noqa: F401  (shared conventions)
from ..evallib import is_missing, period_bucket

IMPLEMENTATIONS: dict[str, Callable] = {}

#: Implementation name -> the traceback of its last failure on this process.
#: A control must never raise into the evaluator; this is where a failure
#: goes instead so that a developer can still find it.
LAST_ERRORS: dict[str, str] = {}


def _register(name: str):
    def deco(fn):
        @functools.wraps(fn)
        def safe(ctx, control):
            try:
                out = fn(ctx, control) or []
                LAST_ERRORS.pop(name, None)
                return out
            except Exception:  # pragma: no cover - surfaced through LAST_ERRORS
                LAST_ERRORS[name] = traceback.format_exc(limit=8)
                return []

        IMPLEMENTATIONS[name] = safe
        return safe

    return deco


# ===========================================================================
# data access helpers
# ===========================================================================

_EMPTY = pd.DataFrame()


def _table(ctx, name: str) -> pd.DataFrame:
    """A populated table, tenant-scoped, or an empty frame."""
    ds = ctx.dataset
    if ds is None:
        return _EMPTY
    try:
        check = getattr(ds, "is_populated", None)
        if check is not None and not check(name):
            return _EMPTY
        df = ds.get(name)
    except Exception:
        return _EMPTY
    if df is None or len(df) == 0:
        return _EMPTY
    key = f"table:{name}"
    hit = _CACHE.get(key)
    if hit is not None and hit[0] is ctx.dataset and hit[1] is df and hit[2] == ctx.tenant_id:
        return hit[3]
    frame = df
    if "tenant_id" in frame.columns and frame["tenant_id"].notna().any():
        frame = frame[frame["tenant_id"].astype(str) == str(ctx.tenant_id)]
    frame = _objectify(frame)
    _CACHE[key] = (ctx.dataset, df, ctx.tenant_id, frame)
    return frame


def _objectify(df: pd.DataFrame) -> pd.DataFrame:
    """Plain Python-object text columns: the Arrow-backed string dtype makes
    ``isin`` and row iteration an order of magnitude slower on this workload."""
    text = [c for c in df.columns if isinstance(df[c].dtype, pd.StringDtype)]
    if not text:
        return df
    out = df.copy()
    for c in text:
        try:
            out[c] = pd.Series(list(df[c].to_numpy(dtype=object, na_value=None)), index=df.index, dtype=object)
        except Exception:  # pragma: no cover - keep the column as it is rather than fail
            pass
    return out


def _dates(series: pd.Series) -> pd.Series:
    out = pd.to_datetime(series, errors="coerce")
    try:
        if getattr(out.dt, "tz", None) is not None:
            out = out.dt.tz_localize(None)
    except (AttributeError, TypeError):
        pass
    return out


def _str(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series("", index=df.index, dtype=object)
    vals = df[col].to_numpy(dtype=object)
    out = [("" if (v is None or v != v) else (v.strip() if isinstance(v, str) else str(v).strip())) for v in vals]
    return pd.Series(out, index=df.index, dtype=object)


def _num(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(np.nan, index=df.index, dtype=float)
    return pd.to_numeric(df[col], errors="coerce")


def _bool(df: pd.DataFrame, col: str) -> pd.Series:
    """Truthy strings/bools to a clean boolean; missing is False."""
    if col not in df.columns:
        return pd.Series(False, index=df.index)
    s = df[col]
    txt = s.where(s.notna(), "").astype(str).str.strip().str.lower()
    return txt.isin({"true", "1", "yes", "y", "t"})


def _code(series: pd.Series) -> pd.Series:
    return series.where(series.notna(), "").astype(str).str.upper().str.replace(".", "", regex=False).str.strip()


def _prefixes(text: Any) -> list[str]:
    if is_missing(text):
        return []
    parts = re.split(r"[;,|\s]+", str(text).upper().replace(".", ""))
    return [p for p in parts if p]


def _jsonable(v: Any) -> Any:
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return None if np.isnan(v) else float(v)
    if isinstance(v, (pd.Timestamp,)):
        return None if pd.isna(v) else str(v.date())
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    if isinstance(v, dict):
        return {str(k): _jsonable(x) for k, x in v.items()}
    if v is None:
        return None
    try:
        if pd.isna(v):
            return None
    except (TypeError, ValueError):
        pass
    return v


def _day(v: Any) -> str:
    return "" if is_missing(v) or pd.isna(v) else str(pd.Timestamp(v).date())


_CACHE: dict[str, tuple] = {}


def _cached(ctx, key: str, build: Callable[[], pd.DataFrame]) -> pd.DataFrame:
    hit = _CACHE.get(key)
    if hit is not None and hit[0] is ctx.dataset and hit[1] is ctx.claims and hit[2] == ctx.tenant_id:
        return hit[3]
    frame = build()
    _CACHE[key] = (ctx.dataset, ctx.claims, ctx.tenant_id, frame)
    return frame


def _header(ctx) -> pd.DataFrame:
    def build() -> pd.DataFrame:
        df = _claim_frame(ctx)
        if df is None or df.empty or "claim_sk" not in df.columns:
            return _EMPTY
        df = _objectify(df)
        cols = [c for c in ("claim_sk", "member_sk", "provider_sk", "payer_id", "service_date",
                            "discharge_date", "claim_type", "diagnosis_primary", "gross_amount_aed",
                            "approved_amount_aed", "length_of_stay_days", "encounter_sk")
                if c in df.columns]
        out = df[cols].copy()
        out["claim_sk"] = out["claim_sk"].astype(str).astype(object)
        out["sd"] = _dates(out["service_date"]) if "service_date" in out.columns else pd.NaT
        out["gross"] = _num(out, "gross_amount_aed").fillna(0.0)
        out["inpatient"] = _is_inpatient(out)
        return out
    return _cached(ctx, "header", build)


def _is_inpatient(df: pd.DataFrame) -> pd.Series:
    ct = _str(df, "claim_type").str.lower()
    ip = ct.str.contains("inp") | ct.isin({"ip", "i", "in"})
    known = ct != ""
    los = _num(df, "length_of_stay_days").fillna(0)
    return ip | (~known & (los > 0))


#: How a service family (activity_code_reference.service_family) maps to the
#: kinds of service the CLN controls reason about. Vocabulary, not thresholds.
_FAMILY_KIND = {
    "LAB": "lab", "IMAGING": "imaging", "ADVANCED_IMAGING": "imaging",
    "CONSULTATION": "consult", "EMERGENCY": "consult", "TELEHEALTH": "consult", "MENTAL_HEALTH": "consult",
    "PHYSIOTHERAPY": "therapy", "PROCEDURE": "procedure", "DAY_SURGERY": "procedure",
    "MATERNITY": "procedure", "INPATIENT": "procedure", "COSMETIC": "procedure",
    "PHARMACY": "drug", "SPECIALTY_DRUG": "drug", "DEVICE": "device",
}
#: Finer code families that override the service family.
_CODE_FAMILY_KIND = [
    (r"^EM_|^CRITICAL_CARE|^TELEHEALTH|^PSYCHOTHERAPY", "consult"),
    (r"^CASE_RATE|^ROOM_DAY", "package"),
    (r"^SPECIMEN", "collection"),
    (r"^INFUSION|^CHEMO|^INJECTION|^PHYSIO", "therapy"),
    (r"^DRUG|^VACCINE", "drug"),
    (r"^LAB", "lab"),
    (r"^IMAGING", "imaging"),
]
#: Last-resort keyword classification when a code is not in the reference.
_KIND_PATTERNS = [
    ("lab", r"lab|patholog|chemistry|haemat|hemat|microbio|genetic|molecular"),
    ("imaging", r"radiol|imag|scan|mri|ultraso|x-ray|xray|ct|tomograph|mammo"),
    ("drug", r"drug|pharm|medication"),
    ("consult", r"consult|evaluation and management|visit"),
    ("therapy", r"therap|physio|rehab|dialys|chemo"),
    ("procedure", r"proced|surg|operat|endoscop"),
]


def _kind(family: pd.Series, code_family: pd.Series, text: pd.Series) -> pd.Series:
    out = family.str.upper().map(_FAMILY_KIND).astype(object)
    cf = code_family.str.upper()
    for pattern, label in _CODE_FAMILY_KIND:
        out = out.where(~cf.str.contains(pattern, regex=True), label)
    t = text.str.lower()
    for label, pattern in _KIND_PATTERNS:
        out = out.where(out.notna() | ~t.str.contains(pattern, regex=True), label)
    return out.fillna("other")


def _reference(ctx) -> pd.DataFrame:
    def build() -> pd.DataFrame:
        ref = _table(ctx, "activity_code_reference")
        if ref.empty or "activity_code" not in ref.columns:
            return _EMPTY
        ref = ref.copy()
        ref["activity_code"] = ref["activity_code"].astype(str).str.strip()
        return ref.drop_duplicates("activity_code")
    return _cached(ctx, "reference", build)


def _lines(ctx) -> pd.DataFrame:
    """Claim lines joined to their header and to the activity-code reference."""
    def build() -> pd.DataFrame:
        lines = _table(ctx, "claim_line")
        hdr = _header(ctx)
        if lines.empty or hdr.empty or "claim_sk" not in lines.columns:
            return _EMPTY
        lines = lines.copy()
        lines["claim_sk"] = lines["claim_sk"].astype(str).astype(object)
        if "line_sk" not in lines.columns:
            lines["line_sk"] = lines["claim_sk"] + ":" + lines.groupby("claim_sk").cumcount().astype(str)
        lines["line_sk"] = lines["line_sk"].astype(str).astype(object)
        keep = hdr[[c for c in ("claim_sk", "member_sk", "provider_sk", "payer_id", "sd", "claim_type",
                                "diagnosis_primary", "gross", "inpatient") if c in hdr.columns]]
        keep = keep.rename(columns={"sd": "claim_sd", "gross": "claim_gross"})
        work = lines.merge(keep, on="claim_sk", how="inner")
        own_sd = _dates(work["service_date"]) if "service_date" in work.columns else pd.Series(pd.NaT, index=work.index)
        work["sd"] = own_sd.fillna(work["claim_sd"])
        work["day"] = work["sd"].dt.normalize()
        gross = _num(work, "gross_amount")
        net = _num(work, "net_amount")
        work["gross_line"] = gross.fillna(net).fillna(0.0)
        work["amount"] = net.fillna(gross).fillna(0.0)
        work["units_n"] = _num(work, "units").fillna(1.0)
        work["activity_code"] = _str(work, "activity_code")
        ref = _reference(ctx)
        if not ref.empty:
            rcols = [c for c in ("activity_code", "activity_type", "description", "service_family",
                                 "code_family", "code_level", "minutes", "is_time_based",
                                 "complexity", "is_scarce_equipment", "unit_price_reference",
                                 "specialty_required") if c in ref.columns]
            work = work.merge(ref[rcols].add_prefix("ref_").rename(
                columns={"ref_activity_code": "activity_code"}), on="activity_code", how="left")
        # classify each distinct code once, then map (100k lines, a few hundred codes)
        codes = work.drop_duplicates("activity_code")
        desc = (_str(codes, "activity_type") + " " + _str(codes, "ref_description") + " "
                + _str(codes, "activity_description"))
        kinds = _kind(_str(codes, "ref_service_family"), _str(codes, "ref_code_family"), desc)
        work["kind"] = work["activity_code"].map(dict(zip(codes["activity_code"], kinds))).fillna("other")
        work["label"] = _str(work, "ref_description").where(_str(work, "ref_description") != "",
                                                            _str(work, "activity_description"))
        work["label"] = work["label"].where(work["label"] != "", work["activity_code"])
        return work.reset_index(drop=True)
    return _cached(ctx, "lines", build)


def _diagnoses(ctx) -> pd.DataFrame:
    """Every diagnosis on every in-scope claim, with principal/secondary resolved."""
    def build() -> pd.DataFrame:
        dx = _table(ctx, "diagnosis")
        hdr = _header(ctx)
        if dx.empty or hdr.empty or "claim_sk" not in dx.columns or "code" not in dx.columns:
            return _EMPTY
        dx = dx.copy()
        dx["claim_sk"] = dx["claim_sk"].astype(str).astype(object)
        dx = dx.merge(hdr[[c for c in ("claim_sk", "member_sk", "provider_sk", "sd", "inpatient",
                                       "gross", "claim_type") if c in hdr.columns]],
                      on="claim_sk", how="inner")
        dx["dx"] = _code(dx["code"])
        dx = dx[dx["dx"] != ""]
        dx["cat"] = dx["dx"].str[:3]
        dx["seq"] = _num(dx, "sequence")
        dtype = _str(dx, "diagnosis_type").str.lower()
        typed = dtype != ""
        dx["principal"] = np.where(typed, dtype.str.contains("princ|primary|main"), dx["seq"] == 1)
        dx["poa"] = _str(dx, "present_on_admission").str.upper()
        dx["desc"] = _str(dx, "description")
        return dx.reset_index(drop=True)
    return _cached(ctx, "diagnoses", build)


def _grouper(ctx) -> pd.DataFrame:
    g = _table(ctx, "drg_grouper")
    if g.empty or "diagnosis_code" not in g.columns:
        return _EMPTY
    g = g.copy()
    g["dx"] = _code(g["diagnosis_code"])
    g["is_cc"] = _bool(g, "is_cc")
    g["is_mcc"] = _bool(g, "is_mcc")
    g["weight"] = _num(g, "severity_weight")
    g["weight"] = g["weight"].fillna(g["is_cc"].astype(float) + 2.0 * g["is_mcc"].astype(float))
    return g.drop_duplicates("dx")[["dx", "is_cc", "is_mcc", "weight"]]


def _with_grouper(dx: pd.DataFrame, grp: pd.DataFrame) -> pd.DataFrame:
    """Attach CC/MCC flags by exact code, falling back to the 3-character category."""
    exact = dx.merge(grp, on="dx", how="left")
    cat = grp.rename(columns={"dx": "cat", "is_cc": "cc3", "is_mcc": "mcc3", "weight": "w3"})
    cat = cat[cat["cat"].str.len() == 3]
    out = exact.merge(cat, on="cat", how="left")
    out["is_cc"] = out["is_cc"].astype("boolean").fillna(out["cc3"].astype("boolean")).fillna(False).astype(bool)
    out["is_mcc"] = out["is_mcc"].astype("boolean").fillna(out["mcc3"].astype("boolean")).fillna(False).astype(bool)
    out["weight"] = out["weight"].fillna(out["w3"]).fillna(0.0)
    out["severe"] = out["is_cc"] | out["is_mcc"]
    return out.drop(columns=["cc3", "mcc3", "w3"])


def _observations(ctx) -> pd.DataFrame:
    def build() -> pd.DataFrame:
        obs = _table(ctx, "observation")
        if obs.empty:
            return _EMPTY
        obs = obs.copy()
        obs["line_sk"] = _str(obs, "line_sk")
        obs["claim_sk"] = _str(obs, "claim_sk")
        obs["t"] = _dates(obs["event_time"]) if "event_time" in obs.columns else pd.NaT
        obs["val"] = _str(obs, "value")
        obs["ocode"] = _str(obs, "observation_code")
        return obs
    return _cached(ctx, "observations", build)


def _provider_table(ctx) -> pd.DataFrame:
    p = _table(ctx, "provider")
    if p.empty or "provider_sk" not in p.columns:
        return _EMPTY
    return p.drop_duplicates("provider_sk").set_index("provider_sk")


def _month(series: pd.Series) -> pd.Series:
    return series.dt.to_period("M").astype(str)


# ===========================================================================
# shared statistics
# ===========================================================================


def _step_shift(months: list[str], values: np.ndarray, ctx) -> dict[str, Any] | None:
    """The strongest upward two-segment mean shift in a monthly series.

    A deliberately simple, explainable detector: for every admissible split
    (at least ``cln_changepoint_min_segment_periods`` months either side, and
    at least ``min_history_periods`` months in total) it compares the mean
    after the split with the mean before, standardised by the pooled
    within-segment spread. The split with the largest standardised increase is
    returned; the caller decides whether it is large enough.
    """
    min_total = int(ctx.cfg("min_history_periods"))
    min_seg = int(ctx.cfg("cln_changepoint_min_segment_periods"))
    n = len(values)
    if n < max(min_total, 2 * min_seg):
        return None
    best: dict[str, Any] | None = None
    for s in range(min_seg, n - min_seg + 1):
        pre, post = values[:s], values[s:]
        pm, qm = float(np.mean(pre)), float(np.mean(post))
        ss = float(np.sum((pre - pm) ** 2) + np.sum((post - qm) ** 2))
        sd = max(np.sqrt(ss / max(n - 2, 1)), 1e-6)
        z = (qm - pm) / (sd * np.sqrt(1.0 / len(pre) + 1.0 / len(post)))
        if best is None or z > best["z"]:
            best = {"split": s, "change_month": months[s], "pre": pm, "post": qm, "z": float(z),
                    "months_before": len(pre), "months_after": len(post)}
    return best


def _casemix_shift(hdr: pd.DataFrame, provider: Any, change_month: str) -> float | None:
    """Total-variation distance between the diagnosis-chapter mix before and after a month."""
    sub = hdr[(hdr["provider_sk"] == provider) & hdr["sd"].notna()]
    if sub.empty or "diagnosis_primary" not in sub.columns:
        return None
    chapter = _code(sub["diagnosis_primary"]).str[:1]
    after = _month(sub["sd"]) >= change_month
    if after.all() or (~after).all():
        return None
    p = chapter[~after].value_counts(normalize=True)
    q = chapter[after].value_counts(normalize=True)
    both = p.index.union(q.index)
    return float(0.5 * np.abs(p.reindex(both, fill_value=0) - q.reindex(both, fill_value=0)).sum())


def _structural_change(ctx, provider: Any, change_date: pd.Timestamp) -> str:
    """A declared structural change (ownership or contract) near the change date, if any."""
    window = int(ctx.cfg("cln_structural_change_window_days"))
    found: list[str] = []
    prov = _provider_table(ctx)
    if not prov.empty and provider in prov.index and "ownership_changed_on" in prov.columns:
        d = pd.to_datetime(prov.at[provider, "ownership_changed_on"], errors="coerce")
        if not pd.isna(d) and abs((d - change_date).days) <= window:
            found.append(f"ownership changed on {plain_date(d)}")
    contracts = _table(ctx, "contract")
    if not contracts.empty and "provider_sk" in contracts.columns and "valid_from" in contracts.columns:
        c = contracts[contracts["provider_sk"] == provider]
        starts = _dates(c["valid_from"]).dropna()
        near = starts[(starts - change_date).abs().dt.days <= window]
        # the first contract is the provider joining the network, not a change
        if len(starts) > 1 and not near.empty and near.min() > starts.min():
            found.append(f"a new contract started on {plain_date(near.min())}")
    return "; ".join(found)


def _peer_z(values: pd.Series) -> tuple[pd.Series, Any]:
    stats = robust_stats(values)
    if stats.scale <= 0:
        return pd.Series(np.nan, index=values.index), stats
    return (values - stats.median) / stats.scale, stats


# ===========================================================================
# CLN-01-R01 (upgrade) — top-level code share against shrunk peers
# ===========================================================================


def _graded_consult_lines(ctx) -> pd.DataFrame:
    lines = _lines(ctx)
    if lines.empty or "ref_code_level" not in lines.columns or "ref_code_family" not in lines.columns:
        return _EMPTY
    work = lines.assign(lv=_num(lines, "ref_code_level"), fam=_str(lines, "ref_code_family"))
    work = work[work["lv"].notna() & (work["fam"] != "") & (work["kind"] == "consult")]
    if work.empty:
        return _EMPTY
    fmax = work.groupby("fam")["lv"].transform("max")
    fmin = work.groupby("fam")["lv"].transform("min")
    work = work[fmax > fmin].copy()
    work["top"] = (work["lv"] == fmax.loc[work.index]).astype(float)
    return work


@_register("cln_01_r01_top_level_share")
def cln_01_r01_top_level_share(ctx, control) -> list:
    work = _graded_consult_lines(ctx)
    if work.empty:
        return []
    percentile = float(ctx.cfg("coding_mismatch_peer_percentile"))
    min_n = int(ctx.cfg("min_entity_opportunities"))
    mass = float(ctx.cfg("posterior_interval_mass"))
    width = float(ctx.cfg("max_posterior_width"))
    min_peers = int(ctx.cfg("cln03_min_specialty_peers"))
    prov = _provider_table(ctx)
    agg = work.groupby("provider_sk").agg(events=("top", "sum"), n=("top", "size"),
                                          claims=("claim_sk", "nunique")).reset_index()
    agg = agg[agg["n"] >= min_n]
    if agg.empty:
        return []
    spec = None
    if not prov.empty and "specialty" in prov.columns:
        label = prov["specialty"].astype(str)
        if "provider_type" in prov.columns:
            label = prov["provider_type"].astype(str) + " " + label
        spec = agg["provider_sk"].map(label)
    agg["peer"] = spec.fillna("ALL").astype(str) if spec is not None else "ALL"
    sizes = agg.groupby("peer")["provider_sk"].transform("size")
    agg.loc[sizes < min_peers, "peer"] = "ALL"   # back off to all providers when a specialty is thin
    out = []
    for peer, grp in agg.groupby("peer", sort=True):
        if len(grp) < min_peers:
            continue
        prior = fit_beta_prior(grp["events"], grp["n"])
        shrunk = {r.provider_sk: shrink_rate(str(r.provider_sk), r.events, r.n, prior, interval_mass=mass,
                                             max_posterior_width=width) for r in grp.itertuples(index=False)}
        cut = float(np.quantile([v.shrunk_rate for v in shrunk.values()], percentile))
        for r in grp.sort_values("provider_sk").itertuples(index=False):
            sr = shrunk[r.provider_sk]
            if sr.excluded_from_ranking or sr.shrunk_rate <= cut or sr.posterior_low <= prior.peer_mean:
                continue
            if r.events == 0 or (sr.observed_rate or 0.0) <= prior.peer_mean:
                continue
            sub = work[(work["provider_sk"] == r.provider_sk)]
            tops = sub[sub["top"] == 1]
            lower = work[work["top"] == 0].groupby("fam")["amount"].median()
            incr = (tops["amount"] - tops["fam"].map(lower).fillna(tops["amount"])).clip(lower=0)
            excess = max(0.0, 1.0 - prior.peer_mean / max(sr.observed_rate or 0.0, 1e-9))
            last = sub["sd"].max()
            level = "the provider's specialty" if peer != "ALL" else "all providers"
            out.append(_sig(
                ctx, control, subject_type="provider", subject_id=r.provider_sk,
                fact_key=f"levelshare:{r.provider_sk}", claim_ids=sorted(tops["claim_sk"].unique().tolist()),
                event_time=last, period=period_bucket(last), peer_level_used=level,
                evidence={
                    "provider_sk": r.provider_sk, "observed_rate": round(float(sr.observed_rate or 0.0), 4),
                    "shrunk_rate": round(sr.shrunk_rate, 4),
                    "posterior_interval": [round(sr.posterior_low, 4), round(sr.posterior_high, 4)],
                    "peer_mean": round(prior.peer_mean, 4), "peer_level_used": level, "peer_n": int(len(grp)),
                    "claim_count": int(r.claims), "graded_services": int(r.n), "top_level_services": int(r.events),
                    "peer_percentile": percentile, "peer_percentile_cut": round(cut, 4),
                    "shrinkage_explanation": sr.explain(),
                    "plain_language": (
                        f"{pct(sr.observed_rate)} of this provider's graded visits are billed at the top level "
                        f"of their code family ({int(r.events)} of {int(r.n)}); comparable providers average "
                        f"{pct(prior.peer_mean)}."
                    ),
                    "what_the_reviewer_must_verify": (
                        "Whether specialty, setting or case mix explains the higher levels; read a sample of "
                        "top-level visit notes."
                    ),
                    "proxy_note": "Measured on the billed evaluation-and-management levels of each service "
                                  "line; case mix is not risk-adjusted beyond the specialty peer group.",
                },
                exposure=_exp.provider_pattern_exposure((incr * excess).round(2).tolist()),
            ))
    return out


# ===========================================================================
# CLN-01-R04 — level-migration change point
# ===========================================================================


@_register("cln_01_r04_level_migration_changepoint")
def cln_01_r04_level_migration_changepoint(ctx, control) -> list:
    # levels of evaluation-and-management services; case-rate severity bands are
    # driven by diagnoses and are CLN-02's business, not a level choice
    work = _graded_consult_lines(ctx)
    if work.empty:
        return []
    work = work[work["sd"].notna()].copy()
    work["month"] = _month(work["sd"])
    min_lines = int(ctx.cfg("cln01_level_min_lines_per_month"))
    min_increase = float(ctx.cfg("cln01_level_shift_min_increase"))
    min_z = float(ctx.cfg("cln_changepoint_min_z"))
    max_mix = float(ctx.cfg("cln_casemix_max_shift"))
    monthly = work.groupby(["provider_sk", "month"]).agg(n=("top", "size"), top=("top", "mean")).reset_index()
    monthly = monthly[monthly["n"] >= min_lines]
    # what a lower level of the same family typically costs: the incremental
    # amount of a top-level line is measured against it
    lower_price = work[work["top"] == 0].groupby("fam")["amount"].median()
    hdr = _header(ctx)
    out = []
    for provider, sub in monthly.groupby("provider_sk", sort=True):
        sub = sub.sort_values("month")
        shift = _step_shift(sub["month"].tolist(), sub["top"].to_numpy(dtype=float), ctx)
        if shift is None or shift["z"] < min_z or shift["post"] - shift["pre"] < min_increase:
            continue
        change = pd.Timestamp(shift["change_month"] + "-01")
        mix = _casemix_shift(hdr, provider, shift["change_month"])
        if mix is not None and mix > max_mix:
            continue  # the patients changed: not a coding shift
        structural = _structural_change(ctx, provider, change)
        if structural:
            continue  # declared exclusion: segment known structural changes
        pw = work[work["provider_sk"] == provider]
        after = pw[pw["month"] >= shift["change_month"]]
        before = pw[pw["month"] < shift["change_month"]]
        top_after = after[after["top"] == 1]
        excess_share = max(0.0, (shift["post"] - shift["pre"]) / max(shift["post"], 1e-9))
        incr = (top_after["amount"] - top_after["fam"].map(lower_price).fillna(top_after["amount"])).clip(lower=0)
        claims = sorted(top_after["claim_sk"].unique().tolist())
        out.append(_sig(
            ctx, control, subject_type="provider", subject_id=provider,
            fact_key=f"levelshift:{provider}",
            claim_ids=claims, event_time=change, period=period_bucket(change),
            confidence=0.8,
            evidence={
                "provider_sk": provider,
                "change_month": shift["change_month"],
                "top_level_share_before": round(shift["pre"], 4),
                "top_level_share_after": round(shift["post"], 4),
                "months_before": shift["months_before"], "months_after": shift["months_after"],
                "graded_lines_before": int(len(before)), "graded_lines_after": int(len(after)),
                "standardised_shift": round(shift["z"], 2),
                "diagnosis_mix_shift": None if mix is None else round(mix, 3),
                "code_families": sorted(after["fam"].unique().tolist())[:10],
                "plain_language": (
                    f"From {change:%B %Y} this provider billed the top level on "
                    f"{pct(shift['post'])} of graded services, up from {pct(shift['pre'])} over the previous "
                    f"{shift['months_before']} months; its mix of diagnoses moved only "
                    f"{pct(mix or 0.0)} and no ownership or contract change is on record."
                ),
                "what_the_reviewer_must_verify": (
                    "Whether a new service line, clinician or contract explains the jump; if not, "
                    "compare a sample of visits billed before and after the change."
                ),
            },
            exposure=_exp.provider_pattern_exposure((incr * excess_share).round(2).tolist()),
        ))
    return out


# ===========================================================================
# CLN-02 — diagnosis-driven payment
# ===========================================================================


def _obj(series: pd.Series) -> np.ndarray:
    return series.to_numpy(dtype=object)


def _support_index(ctx, dx: pd.DataFrame) -> tuple[dict, dict, dict]:
    """What could support a diagnosis: member history, treating lines, dispensed drugs.

    Returns ``(member_cat_claims, claim_line_prefixes, member_drug_prefixes)``.
    Built once per run (plain dictionaries: this is on the hot path of three controls).
    """
    hit = _CACHE.get("support")
    if hit is not None and hit[0] is ctx.dataset and hit[1] is ctx.claims and hit[2] == ctx.tenant_id:
        return hit[3]
    member_cat_claims = dx.groupby(["member_sk", "cat"])["claim_sk"].nunique().to_dict()
    claim_prefixes: dict[str, set[str]] = {}
    lines = _lines(ctx)
    ind = _table(ctx, "indication_policy")
    drugs = _table(ctx, "drug_policy")
    policy: dict[str, list[str]] = {}
    if not ind.empty and "activity_code" in ind.columns:
        policy.update({str(r.activity_code).strip(): _prefixes(getattr(r, "allowed_diagnosis_prefixes", ""))
                       for r in ind.itertuples(index=False)})
    drug_ind: dict[str, list[str]] = {}
    if not drugs.empty and "product" in drugs.columns:
        drug_ind = {str(r.product): _prefixes(getattr(r, "indication_prefixes", ""))
                    for r in drugs.itertuples(index=False)}
    if not lines.empty and (policy or drug_ind):
        sub = lines[lines["activity_code"].isin(set(policy) | set(drug_ind))]
        for claim, code in zip(_obj(sub["claim_sk"]), _obj(sub["activity_code"])):
            s = claim_prefixes.setdefault(claim, set())
            s.update(policy.get(code, ()))
            s.update(drug_ind.get(code, ()))
    member_drugs: dict[str, set[str]] = {}
    rx = _table(ctx, "prescription_dispense")
    if not rx.empty and drug_ind:
        prod = _str(rx, "dispensed_product").where(_str(rx, "dispensed_product") != "", _str(rx, "billed_product"))
        hdr = _header(ctx)
        by_claim = rx["claim_sk"].astype(str).map(hdr.set_index("claim_sk")["member_sk"])             if "claim_sk" in rx.columns else pd.Series(np.nan, index=rx.index)
        own = rx["member_sk"] if "member_sk" in rx.columns else pd.Series(np.nan, index=rx.index)
        member = own.where(own.notna(), by_claim)
        ok = member.notna()
        for m, p in zip(_obj(member[ok].astype(str)), _obj(prod[ok])):
            member_drugs.setdefault(m, set()).update(drug_ind.get(p, ()))
    out = (member_cat_claims, claim_prefixes, member_drugs)
    _CACHE["support"] = (ctx.dataset, ctx.claims, ctx.tenant_id, out)
    return out


def _prefix_mask(codes: pd.Series, prefixes: list[str] | set[str]) -> pd.Series:
    """Vectorised 'starts with any of these prefixes'."""
    prefixes = sorted({p for p in prefixes if p})
    if not prefixes:
        return pd.Series(False, index=codes.index)
    return codes.str.match("^(?:" + "|".join(re.escape(p) for p in prefixes) + ")")


def _covered(code: str, prefixes: set[str] | list[str]) -> bool:
    return any(code.startswith(p) for p in prefixes)


_BAND_RANK = {"C": 0, "B": 1, "A": 2}


def _claim_document_text(ctx, claims: set[str]) -> dict[str, str]:
    """Clinical documents on each claim, upper-cased with dots removed so codes match either way."""
    docs = _table(ctx, "document")
    if docs.empty or "claim_sk" not in docs.columns or "text" not in docs.columns:
        return {}
    d = docs[docs["claim_sk"].astype(str).isin(claims)]
    if d.empty:
        return {}
    txt = _str(d, "text").str.upper().str.replace(".", "", regex=False)
    return txt.groupby(d["claim_sk"].astype(str)).agg(" ".join).to_dict()


def _case_rate_lines(ctx) -> pd.DataFrame:
    """Case-rate (DRG-like) lines: base code, severity band A/B/C, and the price of each band."""
    lines = _lines(ctx)
    if lines.empty:
        return _EMPTY
    code = lines["activity_code"].str.upper()
    fam = _str(lines, "ref_code_family").str.upper()
    is_cr = (fam == "CASE_RATE") & code.str[-1:].isin(list(_BAND_RANK))
    cr = lines[is_cr].copy()
    if cr.empty:
        return _EMPTY
    cr["base"] = cr["activity_code"].str[:-1]
    cr["band"] = cr["activity_code"].str[-1:].str.upper()
    ref = _reference(ctx)
    price = _num(ref, "unit_price_reference").set_axis(ref["activity_code"]) if not ref.empty else pd.Series(dtype=float)
    cr["price_billed"] = cr["activity_code"].map(price)
    return cr


def _regroup(claim_dx: list[tuple[str, bool, bool]], removed: str) -> str:
    rest = [(cc, mcc) for code, cc, mcc in claim_dx if code != removed]
    if any(mcc for _, mcc in rest):
        return "A"
    if any(cc for cc, _ in rest):
        return "B"
    return "C"


@_register("cln_02_r01_payment_changing_secondary_dx")
def cln_02_r01_payment_changing_secondary_dx(ctx, control) -> list:
    dx = _diagnoses(ctx)
    grp = _grouper(ctx)
    if dx.empty or grp.empty:
        return []
    cr = _case_rate_lines(ctx)
    if cr.empty:
        return []
    g = _with_grouper(dx, grp)
    secondary = g[~g["principal"].astype(bool)]
    cand = secondary[secondary["severe"] & secondary["claim_sk"].isin(set(cr["claim_sk"]))]
    if cand.empty:
        return []
    history, claim_prefixes, member_drugs = _support_index(ctx, dx)
    ref = _reference(ctx)
    price = _num(ref, "unit_price_reference").set_axis(ref["activity_code"]) if not ref.empty else pd.Series(dtype=float)
    in_scope = secondary[secondary["claim_sk"].isin(set(cand["claim_sk"]))]
    by_claim: dict[str, list[tuple[str, bool, bool]]] = {}
    for c, code, cc, mcc in zip(_obj(in_scope["claim_sk"]), _obj(in_scope["dx"]),
                                in_scope["is_cc"].to_numpy(), in_scope["is_mcc"].to_numpy()):
        by_claim.setdefault(c, []).append((code, bool(cc), bool(mcc)))
    doc_text = _claim_document_text(ctx, set(cand["claim_sk"]))
    cr_by_claim = cr.drop_duplicates("claim_sk").set_index("claim_sk")[
        ["band", "base", "activity_code", "amount"]].to_dict("index")
    out = []
    for row in cand.sort_values(["claim_sk", "dx"]).drop_duplicates(["claim_sk", "dx"]).itertuples(index=False):
        line = cr_by_claim[row.claim_sk]
        new_band = _regroup(by_claim[row.claim_sk], row.dx)
        if _BAND_RANK[new_band] >= _BAND_RANK.get(line["band"], 0):
            continue  # removing it would not lower the payment band
        if history.get((row.member_sk, row.cat), 0) >= 2:
            continue  # the condition is on the patient's record elsewhere
        if _covered(row.dx, claim_prefixes.get(row.claim_sk, set())):
            continue  # something billed on the stay treats it
        if _covered(row.dx, member_drugs.get(str(row.member_sk), set())):
            continue  # the patient receives medicine for it
        if row.dx in doc_text.get(row.claim_sk, ""):
            continue  # the discharge summary records it
        billed = float(line["amount"])
        p_old, p_new = price.get(line["activity_code"]), price.get(line["base"] + new_band)
        if p_old is None or p_new is None or pd.isna(p_old) or pd.isna(p_new) or p_old <= 0:
            continue
        correct = round(billed * float(p_new) / float(p_old), 2)
        out.append(_sig(
            ctx, control, subject_type="claim", subject_id=row.claim_sk,
            fact_key=f"secdx:{row.claim_sk}:{row.dx}",
            claim_ids=[row.claim_sk], event_time=row.sd, period=period_bucket(row.sd),
            confidence=0.75,
            evidence={
                "claim_sk": row.claim_sk, "member_sk": row.member_sk, "provider_sk": row.provider_sk,
                "secondary_diagnosis": row.dx, "diagnosis_description": row.desc,
                "grouper_flag": "MCC" if row.is_mcc else "CC",
                "billed_case_rate": line["activity_code"], "case_rate_without_it": line["base"] + new_band,
                "billed_case_rate_amount_aed": round(billed, 2), "regrouped_amount_aed": correct,
                "support_found": {"other_claims_with_condition": 0, "treating_service_billed": False,
                                  "medicine_dispensed_for_condition": False},
                "plain_language": (
                    f"Secondary diagnosis {row.dx}{' (' + row.desc + ')' if row.desc else ''} lifts this stay "
                    f"to case rate {line['activity_code']} ({aed(billed)}); without it the stay groups to "
                    f"{line['base'] + new_band} ({aed(correct)}), and the condition appears on no other claim "
                    f"for this patient with nothing billed or dispensed to treat it."
                ),
                "what_the_reviewer_must_verify": (
                    "Whether the clinical record documents this condition as present and treated during "
                    "the stay; if not, re-group the claim without it."
                ),
                "proxy_note": "The payment band comes from the dataset's stand-in grouper table, and "
                              "'supporting evidence' is read from the patient's other claims, the "
                              "services billed and the medicines dispensed, not from the chart.",
            },
            exposure=_exp.Exposure(
                max(0.0, round(billed - correct, 2)),
                f"Regrouping without {row.dx}: billed case rate AED {billed:,.2f} minus AED {correct:,.2f} "
                f"for {line['base'] + new_band}. Not yet established: a coding audit must confirm the "
                f"diagnosis is unsupported.",
                established=False),
        ))
    return out


@_register("cln_02_r02_comorbidity_prevalence_outlier")
def cln_02_r02_comorbidity_prevalence_outlier(ctx, control) -> list:
    from scipy import stats as _st

    dx = _diagnoses(ctx)
    grp = _grouper(ctx)
    if dx.empty or grp.empty:
        return []
    g = _with_grouper(dx, grp)
    g = g[g["inpatient"]]
    if g.empty:
        return []
    # exclusion: transfers are coded differently
    enc = _table(ctx, "encounter")
    transfer_claims: set[str] = set()
    if not enc.empty and "claim_sk" in enc.columns:
        adm = (_str(enc, "admission_type") + " " + _str(enc, "discharge_type")).str.lower()
        transfer_claims = set(enc.loc[adm.str.contains("transfer"), "claim_sk"].astype(str))
    g = g[~g["claim_sk"].isin(transfer_claims)]
    principal = g[g["principal"].astype(bool)].drop_duplicates("claim_sk")[["claim_sk", "provider_sk", "dx"]]
    if principal.empty:
        return []
    principal["chapter"] = principal["dx"].str[:1]
    has_cc = g[(~g["principal"].astype(bool)) & g["severe"]].groupby("claim_sk").size()
    principal["cc"] = principal["claim_sk"].map(has_cc).fillna(0).gt(0).astype(float)
    chapter_rate = principal.groupby("chapter")["cc"].mean()
    principal["expected"] = principal["chapter"].map(chapter_rate).fillna(principal["cc"].mean())
    agg = principal.groupby("provider_sk").agg(n=("cc", "size"), observed=("cc", "sum"),
                                                expected=("expected", "sum")).reset_index()
    min_n = int(ctx.cfg("cln02_comorbidity_min_claims"))
    ratio_cut = float(ctx.cfg("cln02_comorbidity_oe_ratio"))
    p_cut = float(ctx.cfg("cln02_comorbidity_max_pvalue"))
    out = []
    for row in agg.itertuples(index=False):
        if row.n < min_n or row.expected <= 0:
            continue
        ratio = row.observed / row.expected
        if ratio < ratio_cut:
            continue
        p = float(_st.poisson.sf(row.observed - 1, row.expected))
        if p > p_cut:
            continue
        claims = principal.loc[(principal["provider_sk"] == row.provider_sk) & (principal["cc"] == 1), "claim_sk"]
        hdr = _header(ctx)
        last = hdr.loc[hdr["claim_sk"].isin(claims), "sd"].max()
        out.append(_sig(
            ctx, control, subject_type="provider", subject_id=row.provider_sk,
            fact_key=f"comorbidity:{row.provider_sk}",
            claim_ids=sorted(claims.tolist()), event_time=last, period=period_bucket(last),
            peer_level_used="principal_diagnosis_chapter", confidence=0.8,
            evidence={
                "provider_sk": row.provider_sk, "inpatient_claims": int(row.n),
                "claims_with_complication_codes": int(row.observed),
                "observed_rate": round(row.observed / row.n, 4),
                "expected_rate": round(row.expected / row.n, 4),
                "observed_to_expected": round(ratio, 2),
                "poisson_tail_probability": p,
                "peer_level_used": "principal diagnosis chapter",
                "plain_language": (
                    f"{pct(row.observed / row.n)} of this provider's inpatient stays carry a complication "
                    f"or major-complication code; for the same mix of main diagnoses at other providers "
                    f"it is {pct(row.expected / row.n)} ({times_phrase(ratio, 1.0, 'the expected rate')})."
                ),
                "what_the_reviewer_must_verify": (
                    "Whether age, transfers or a specialist service explain the difference; sample stays "
                    "with complication codes and check them against the notes."
                ),
            },
            exposure=_exp.no_exposure(
                "a provider-level coding pattern; the payment effect of each code needs a coding audit"),
        ))
    return out


@_register("cln_02_r03_prior_history_inconsistency")
def cln_02_r03_prior_history_inconsistency(ctx, control) -> list:
    dx = _diagnoses(ctx)
    grp = _grouper(ctx)
    if dx.empty or grp.empty or "sd" not in dx.columns:
        return []
    min_recur = float(ctx.cfg("cln02_chronic_recurrence_min_share"))
    min_prior = int(ctx.cfg("cln02_min_history_claims"))
    lookback = int(ctx.cfg("cln02_history_lookback_days"))
    # a condition is treated as long-standing when most patients who have it
    # at all have it on more than one claim — learnt from the file itself
    per = dx.groupby(["cat", "member_sk"])["claim_sk"].nunique()
    recurrence = (per >= 2).groupby(level=0).mean()
    chronic = set(recurrence[recurrence >= min_recur].index)
    g = _with_grouper(dx, grp)
    cand = g[g["severe"] & g["cat"].isin(chronic) & (~g["principal"].astype(bool))]
    if cand.empty:
        return []
    once = per[per == 1].reset_index()[["cat", "member_sk"]]
    cand = cand.merge(once, on=["cat", "member_sk"], how="inner")
    if cand.empty:
        return []
    hdr = _header(ctx)
    member_dates = hdr.groupby("member_sk")["sd"].apply(lambda s: np.sort(s.dropna().to_numpy()))
    _, _, member_drugs = _support_index(ctx, dx)
    out = []
    for row in cand.sort_values(["claim_sk", "dx"]).itertuples(index=False):
        dates = member_dates.get(row.member_sk)
        if dates is None or pd.isna(row.sd):
            continue
        t = np.datetime64(row.sd)
        lo, hi = t - np.timedelta64(lookback, "D"), t + np.timedelta64(lookback, "D")
        prior = int(((dates >= lo) & (dates < t)).sum())
        later = int(((dates > t) & (dates <= hi)).sum())
        if prior < min_prior or later < 1:
            continue  # absence is weak evidence without a real history either side
        if _covered(row.dx, member_drugs.get(str(row.member_sk), set())):
            continue
        out.append(_sig(
            ctx, control, subject_type="claim", subject_id=row.claim_sk,
            fact_key=f"secdx:{row.claim_sk}:{row.dx}",
            claim_ids=[row.claim_sk], event_time=row.sd, period=period_bucket(row.sd),
            confidence=0.6,
            evidence={
                "claim_sk": row.claim_sk, "member_sk": row.member_sk, "provider_sk": row.provider_sk,
                "diagnosis": row.dx, "diagnosis_description": row.desc,
                "claims_in_prior_window": prior, "claims_in_following_window": later,
                "window_days": lookback,
                "share_of_patients_with_condition_who_recur": round(float(recurrence.get(row.cat, 0.0)), 3),
                "plain_language": (
                    f"Long-standing condition {row.dx}{' (' + row.desc + ')' if row.desc else ''} appears on "
                    f"this claim only: none of the patient's {prior} earlier or {later} later claims within "
                    f"{lookback} days mention it, and no medicine for it was dispensed."
                ),
                "what_the_reviewer_must_verify": (
                    "Read the patient's earlier and later records; a genuinely new diagnosis is possible, "
                    "so absence of history is weak evidence on its own."
                ),
            },
            exposure=_exp.no_exposure("a record-review priority; the payment effect depends on regrouping"),
        ))
    return out


_POA_NO = {"N", "NO", "0", "FALSE"}


@_register("cln_02_r04_poa_sequencing_conflict")
def cln_02_r04_poa_sequencing_conflict(ctx, control) -> list:
    dx = _diagnoses(ctx)
    if dx.empty:
        return []
    enc = _table(ctx, "encounter")
    transfer: set[str] = set()
    if not enc.empty and "claim_sk" in enc.columns:
        adm = _str(enc, "admission_type").str.lower()
        transfer = set(enc.loc[adm.str.contains("transfer"), "claim_sk"].astype(str))
    grp = dx.groupby("claim_sk")
    n_principal = grp["principal"].sum()
    dup = dx[dx.duplicated(["claim_sk", "dx"], keep=False)].groupby("claim_sk")["dx"].first()
    principal = dx[dx["principal"].astype(bool)]
    ext = principal[principal["dx"].str[:1].isin(list("VWXY"))].groupby("claim_sk")["dx"].first()
    exempt_chapter = principal["dx"].str[:1].isin(["O", "P"]) | principal["dx"].str.startswith("Z38")
    poa_bad = principal[principal["inpatient"] & principal["poa"].isin(_POA_NO) & ~exempt_chapter
                        & ~principal["claim_sk"].isin(transfer)].groupby("claim_sk")["dx"].first()
    info = dx.drop_duplicates("claim_sk").set_index("claim_sk")
    problems: dict[str, list[str]] = {}
    facts: dict[str, dict[str, Any]] = {}
    for claim, k in n_principal.items():
        if k == 0:
            problems.setdefault(claim, []).append("no diagnosis is marked as the main diagnosis")
        elif k > 1:
            problems.setdefault(claim, []).append(f"{int(k)} diagnoses are each marked as the main diagnosis")
        facts.setdefault(claim, {})["main_diagnosis_count"] = int(k)
    for claim, code in dup.items():
        problems.setdefault(claim, []).append(f"{code} is listed twice")
        facts.setdefault(claim, {})["repeated_code"] = code
    for claim, code in ext.items():
        problems.setdefault(claim, []).append(f"the main diagnosis {code} is an external-cause code, "
                                              "which may not be the main diagnosis")
        facts.setdefault(claim, {})["external_cause_principal"] = code
    for claim, code in poa_bad.items():
        problems.setdefault(claim, []).append(f"the main diagnosis {code} is marked as not present on "
                                              "admission, although it is the reason for the admission")
        facts.setdefault(claim, {})["principal_not_present_on_admission"] = code
    out = []
    for claim in sorted(problems):
        row = info.loc[claim]
        issues = problems[claim]
        out.append(_sig(
            ctx, control, subject_type="claim", subject_id=claim,
            fact_key=f"dxseq:{claim}", claim_ids=[claim], event_time=row["sd"],
            period=period_bucket(row["sd"]),
            evidence={
                "claim_sk": claim, "member_sk": row["member_sk"], "provider_sk": row["provider_sk"],
                "issues": issues, **facts.get(claim, {}),
                "diagnoses": dx.loc[dx["claim_sk"] == claim].sort_values("seq")[["dx", "seq", "poa"]]
                .astype(str).to_dict("records"),
                "plain_language": "On this claim " + "; ".join(issues) + ".",
                "what_the_reviewer_must_verify": (
                    "Ask for the corrected diagnosis list; confirm the order and on-admission status "
                    "against the coding policy in force on the admission date."
                ),
            },
            exposure=_exp.no_exposure("a coding pend; the payment effect is known only after correction"),
        ))
    return out


@_register("cln_02_r05_severity_mix_changepoint")
def cln_02_r05_severity_mix_changepoint(ctx, control) -> list:
    dx = _diagnoses(ctx)
    grp = _grouper(ctx)
    if dx.empty or grp.empty:
        return []
    g = _with_grouper(dx, grp)
    g = g[g["inpatient"] & g["sd"].notna()]
    if g.empty:
        return []
    sec = g[(~g["principal"].astype(bool)) & g["severe"]].groupby("claim_sk")["weight"].sum()
    hdr = _header(ctx)
    claims = hdr[hdr["claim_sk"].isin(set(g["claim_sk"]))].copy()
    claims["severity"] = claims["claim_sk"].map(sec).fillna(0.0)
    claims["month"] = _month(claims["sd"])
    claims["los"] = _num(claims, "length_of_stay_days")
    lines = _lines(ctx)
    per_claim_lines = lines.groupby("claim_sk").size() if not lines.empty else pd.Series(dtype=float)
    claims["n_lines"] = claims["claim_sk"].map(per_claim_lines).fillna(0.0)
    min_claims = int(ctx.cfg("cln02_severity_min_claims_per_month"))
    min_ratio = float(ctx.cfg("cln02_severity_shift_min_ratio"))
    min_z = float(ctx.cfg("cln_changepoint_min_z"))
    explains = float(ctx.cfg("cln02_severity_explained_by_ratio"))
    max_mix = float(ctx.cfg("cln_casemix_max_shift"))
    monthly = claims.groupby(["provider_sk", "month"]).agg(
        n=("severity", "size"), sev=("severity", "mean")).reset_index()
    monthly = monthly[monthly["n"] >= min_claims]
    out = []
    for provider, sub in monthly.groupby("provider_sk", sort=True):
        sub = sub.sort_values("month")
        shift = _step_shift(sub["month"].tolist(), sub["sev"].to_numpy(dtype=float), ctx)
        if shift is None or shift["z"] < min_z:
            continue
        if shift["post"] < min_ratio * max(shift["pre"], 1e-9) or shift["post"] <= shift["pre"]:
            continue
        pc = claims[claims["provider_sk"] == provider]
        after = pc["month"] >= shift["change_month"]
        los_ratio = float(pc.loc[after, "los"].mean() / max(pc.loc[~after, "los"].mean(), 1e-9))
        int_ratio = float(pc.loc[after, "n_lines"].mean() / max(pc.loc[~after, "n_lines"].mean(), 1e-9))
        if los_ratio >= explains or int_ratio >= explains:
            continue  # the care changed too: the severity rise is corroborated
        mix = _casemix_shift(hdr, provider, shift["change_month"])
        if mix is not None and mix > max_mix:
            continue  # declared exclusion: service-line expansion
        change = pd.Timestamp(shift["change_month"] + "-01")
        if _structural_change(ctx, provider, change):
            continue
        out.append(_sig(
            ctx, control, subject_type="provider", subject_id=provider,
            fact_key=f"severityshift:{provider}",
            claim_ids=sorted(pc.loc[after & (pc["severity"] > 0), "claim_sk"].tolist()),
            event_time=change, period=period_bucket(change), confidence=0.75,
            evidence={
                "provider_sk": provider, "change_month": shift["change_month"],
                "severity_index_before": round(shift["pre"], 3),
                "severity_index_after": round(shift["post"], 3),
                "standardised_shift": round(shift["z"], 2),
                "length_of_stay_ratio_after_to_before": round(los_ratio, 3),
                "services_per_stay_ratio_after_to_before": round(int_ratio, 3),
                "diagnosis_mix_shift": None if mix is None else round(mix, 3),
                "months_before": shift["months_before"], "months_after": shift["months_after"],
                "plain_language": (
                    f"From {change:%B %Y} the average recorded severity of this provider's stays "
                    f"rose from {shift['pre']:.2f} to {shift['post']:.2f} "
                    f"({times_phrase(shift['post'], shift['pre'], 'its earlier level')}), while length of stay "
                    f"changed by a factor of {los_ratio:.2f} and services per stay by {int_ratio:.2f}."
                ),
                "what_the_reviewer_must_verify": (
                    "Whether a new service line or referral source explains sicker patients; otherwise "
                    "audit a sample of stays coded after the change."
                ),
            },
            exposure=_exp.no_exposure("a provider-level trend; payment effect requires a coding audit"),
        ))
    return out


# ===========================================================================
# CLN-03 — clinical consistency
# ===========================================================================


def _member_dx_index(dx: pd.DataFrame) -> dict[str, list[tuple[Any, str]]]:
    idx: dict[str, list[tuple[Any, str]]] = {}
    d = dx.dropna(subset=["sd"])
    for m, t, c in zip(_obj(d["member_sk"].astype(str)), d["sd"].to_numpy(), _obj(d["dx"])):
        idx.setdefault(m, []).append((t, c))
    return idx


@_register("cln_03_r02_unsupported_indication")
def cln_03_r02_unsupported_indication(ctx, control) -> list:
    lines = _lines(ctx)
    ind = _table(ctx, "indication_policy")
    dx = _diagnoses(ctx)
    if lines.empty or ind.empty or dx.empty or "activity_code" not in ind.columns:
        return []
    policy = {str(r.activity_code).strip(): _prefixes(getattr(r, "allowed_diagnosis_prefixes", ""))
              for r in ind.itertuples(index=False)}
    policy = {k: v for k, v in policy.items() if v}
    cand = lines[lines["activity_code"].isin(policy)]
    if cand.empty:
        return []
    lookback = np.timedelta64(int(ctx.cfg("cln03_indication_lookback_days")), "D")
    claim_dx: dict[str, list[str]] = {}
    for c, code in zip(_obj(dx["claim_sk"]), _obj(dx["dx"])):
        claim_dx.setdefault(c, []).append(code)
    member_dx = _member_dx_index(dx)
    out = []
    for row in cand[["line_sk", "claim_sk", "member_sk", "provider_sk", "activity_code", "label", "amount", "sd"]].sort_values("line_sk").itertuples(index=False):
        prefixes = policy[row.activity_code]
        codes = list(claim_dx.get(row.claim_sk, []))
        if any(_covered(c, prefixes) for c in codes):
            continue
        hist = member_dx.get(str(row.member_sk))
        if hist is not None and not pd.isna(row.sd):
            t = np.datetime64(row.sd)
            if any(t - lookback <= d <= t and _covered(c, prefixes) for d, c in hist):
                continue
        out.append(_sig(
            ctx, control, subject_type="claim", subject_id=row.claim_sk,
            fact_key=f"indication:{row.line_sk}", claim_ids=[row.claim_sk],
            event_time=row.sd, period=period_bucket(row.sd), confidence=0.85,
            evidence={
                "claim_sk": row.claim_sk, "line_sk": row.line_sk, "member_sk": row.member_sk,
                "provider_sk": row.provider_sk, "activity_code": row.activity_code,
                "activity_description": row.label, "line_amount_aed": round(float(row.amount), 2),
                "accepted_diagnosis_prefixes": prefixes, "claim_diagnoses": sorted(set(codes)),
                "lookback_days": int(ctx.cfg("cln03_indication_lookback_days")),
                "plain_language": (
                    f"{row.label} ({aed(row.amount)}) is billed without an accepted reason: the policy "
                    f"accepts diagnoses starting {', '.join(prefixes[:6])}, and neither this claim "
                    f"({', '.join(sorted(set(codes))[:4]) or 'no diagnosis'}) nor the patient's previous "
                    f"{int(ctx.cfg('cln03_indication_lookback_days'))} days of claims carries one."
                ),
                "what_the_reviewer_must_verify": (
                    "Whether the clinical notes record an accepted reason for this service that was not coded."
                ),
            },
            exposure=_exp.line_edit_exposure(float(row.amount), 0.0),
        ))
    return out


@_register("cln_03_r03_procedure_sequence_conflict")
def cln_03_r03_procedure_sequence_conflict(ctx, control) -> list:
    lines = _lines(ctx)
    path = _table(ctx, "care_pathway_policy")
    if lines.empty or path.empty or not {"activity_code", "required_prior_code"} <= set(path.columns):
        return []
    path = path.copy()
    path["activity_code"] = _str(path, "activity_code")
    path["required_prior_code"] = _str(path, "required_prior_code")
    path["within"] = _num(path, "within_days").fillna(int(ctx.cfg("cln03_default_pathway_days")))
    cand = lines[lines["activity_code"].isin(set(path["activity_code"])) & lines["sd"].notna()]
    if cand.empty:
        return []
    member_codes = lines[lines["activity_code"].isin(set(path["required_prior_code"]))]
    acc: dict[tuple[str, str], list] = {}
    mc = member_codes.dropna(subset=["sd"])
    for m, c, d in zip(_obj(mc["member_sk"].astype(str)), _obj(mc["activity_code"]), mc["sd"].to_numpy()):
        acc.setdefault((m, c), []).append(d)
    by_member = {k: np.sort(np.array(v, dtype="datetime64[ns]")) for k, v in acc.items()}
    # exclusion: the prerequisite may have been done elsewhere — a referral into
    # this provider shortly before the service says the patient arrived worked-up
    referrals = _table(ctx, "referral")
    referred: dict[tuple[str, str], np.ndarray] = {}
    if not referrals.empty and {"member_sk", "recipient_provider_sk", "referral_date"} <= set(referrals.columns):
        r = referrals.assign(d=_dates(referrals["referral_date"]))
        r = r.dropna(subset=["d"])
        acc2: dict[tuple[str, str], list] = {}
        for m, p, d in zip(_obj(r["member_sk"].astype(str)), _obj(r["recipient_provider_sk"].astype(str)),
                           r["d"].to_numpy()):
            acc2.setdefault((m, p), []).append(d)
        referred = {k: np.sort(np.array(v, dtype="datetime64[ns]")) for k, v in acc2.items()}
    rules: dict[str, list] = {}
    for req in path[["activity_code", "required_prior_code", "within"]].itertuples(index=False):
        rules.setdefault(req.activity_code, []).append(req)
    out = []
    for row in cand[["line_sk", "claim_sk", "member_sk", "provider_sk", "activity_code", "label", "amount", "sd"]].sort_values("line_sk").itertuples(index=False):
        t = np.datetime64(row.sd)
        for req in rules[row.activity_code]:
            window = np.timedelta64(int(req.within), "D")
            dates = by_member.get((str(row.member_sk), req.required_prior_code), np.array([], dtype="datetime64[ns]"))
            before = dates[(dates <= t) & (dates >= t - window)]
            if before.size:
                continue
            after = dates[(dates > t) & (dates <= t + window)]
            ref = referred.get((str(row.member_sk), str(row.provider_sk)), np.array([], dtype="datetime64[ns]"))
            if ref.size and ((ref <= t) & (ref >= t - window)).any():
                continue  # declared exclusion: external services
            prior_label = req.required_prior_code
            order = "after" if after.size else "missing"
            if order == "after":
                text = (f"{row.label} on {plain_date(row.sd)} was billed before its required earlier step "
                        f"({prior_label}), which appears only afterwards on {plain_date(after[0])}.")
            else:
                text = (f"{row.label} on {plain_date(row.sd)} requires {prior_label} within the previous "
                        f"{int(req.within)} days, and the patient has no such service on record.")
            out.append(_sig(
                ctx, control, subject_type="claim", subject_id=row.claim_sk,
                fact_key=f"pathway:{row.line_sk}:{prior_label}", claim_ids=[row.claim_sk],
                event_time=row.sd, period=period_bucket(row.sd), confidence=0.8,
                evidence={
                    "claim_sk": row.claim_sk, "line_sk": row.line_sk, "member_sk": row.member_sk,
                    "provider_sk": row.provider_sk, "activity_code": row.activity_code,
                    "activity_description": row.label, "required_prior_code": prior_label,
                    "within_days": int(req.within), "prerequisite_status": order,
                    "prerequisite_date_found_after": _day(after[0]) if after.size else None,
                    "line_amount_aed": round(float(row.amount), 2),
                    "plain_language": text,
                    "what_the_reviewer_must_verify": (
                        "Whether the earlier step was performed elsewhere (ask for the outside report); "
                        "if not, whether the billed service could clinically have happened."
                    ),
                },
                exposure=_exp.line_edit_exposure(float(row.amount), 0.0),
            ))
            break
    return out


@_register("cln_03_r04_rare_code_specialty")
def cln_03_r04_rare_code_specialty(ctx, control) -> list:
    lines = _lines(ctx)
    prov = _provider_table(ctx)
    if lines.empty or prov.empty or "specialty" not in prov.columns:
        return []
    work = lines[["provider_sk", "activity_code", "claim_sk", "sd", "label"]].copy()
    ptype = prov["provider_type"].astype(str) + " " if "provider_type" in prov.columns else ""
    peer_label = (ptype + prov["specialty"].astype(str)) if "provider_type" in prov.columns else prov["specialty"]
    work["specialty"] = work["provider_sk"].map(peer_label).fillna("").astype(str).astype(object)
    work = work[(work["specialty"] != "") & (work["activity_code"] != "")]
    if work.empty:
        return []
    max_prev = float(ctx.cfg("cln03_rare_code_max_prevalence"))
    min_lines = int(ctx.cfg("cln03_min_provider_lines"))
    min_peers = int(ctx.cfg("cln03_min_specialty_peers"))
    min_share = float(ctx.cfg("cln03_rare_share_min"))
    multiple = float(ctx.cfg("cln03_rare_share_peer_multiple"))
    spec_code = work.groupby(["specialty", "activity_code"]).size().rename("sc")
    spec_all = work.groupby("specialty").size().rename("s")
    prov_code = work.groupby(["provider_sk", "specialty", "activity_code"]).size().rename("pc").reset_index()
    prov_all = work.groupby("provider_sk").size().rename("p")
    peers = work.groupby("specialty")["provider_sk"].nunique()
    pc = prov_code.join(spec_code, on=["specialty", "activity_code"]).join(spec_all, on="specialty")
    pc = pc.join(prov_all, on="provider_sk")
    # leave-one-out: how common the code is among the OTHER providers of the specialty
    denom = (pc["s"] - pc["p"]).clip(lower=1)
    pc["loo_prev"] = (pc["sc"] - pc["pc"]) / denom
    pc["rare"] = pc["loo_prev"] <= max_prev
    share = pc[pc["rare"]].groupby("provider_sk")["pc"].sum().reindex(prov_all.index, fill_value=0) / prov_all
    frame = pd.DataFrame({"share": share, "n": prov_all})
    frame["specialty"] = frame.index.map(peer_label).astype(str)
    out = []
    for provider, row in frame.sort_index().iterrows():
        if row["n"] < min_lines or peers.get(row["specialty"], 0) - 1 < min_peers:
            continue
        others = frame[(frame["specialty"] == row["specialty"]) & (frame.index != provider)
                       & (frame["n"] >= min_lines)]["share"]
        peer_median = float(others.median()) if len(others) else 0.0
        if row["share"] < min_share or row["share"] < multiple * peer_median:
            continue
        rare = pc[(pc["provider_sk"] == provider) & pc["rare"]].sort_values("pc", ascending=False)
        top = rare.head(5)
        claims = work.loc[(work["provider_sk"] == provider) & work["activity_code"].isin(set(rare["activity_code"])),
                          ["claim_sk", "sd"]]
        last = claims["sd"].max()
        out.append(_sig(
            ctx, control, subject_type="provider", subject_id=provider,
            fact_key=f"rarecode:{provider}", claim_ids=sorted(claims["claim_sk"].unique().tolist()),
            event_time=last, period=period_bucket(last), peer_level_used="provider specialty",
            confidence=0.6,
            evidence={
                "provider_sk": provider, "specialty": row["specialty"],
                "rare_line_share": round(float(row["share"]), 4), "peer_median_share": round(peer_median, 4),
                "lines_billed": int(row["n"]), "specialty_peers": int(peers.get(row["specialty"], 0)) - 1,
                "rare_codes": [{"activity_code": r.activity_code, "lines": int(r.pc),
                                "share_among_peers": round(float(r.loo_prev), 5)} for r in top.itertuples()],
                "plain_language": (
                    f"{pct(row['share'])} of this provider's billed services are codes that other "
                    f"{row['specialty'].lower().replace('_', ' ')} providers almost never bill (under {pct(max_prev)} of "
                    f"their services); for its peers the typical share is {pct(peer_median)}."
                ),
                "what_the_reviewer_must_verify": (
                    "Whether the provider offers a genuine sub-specialty service; this is a monitoring lead "
                    "for policy review, not a claim decision."
                ),
            },
            exposure=_exp.no_exposure("an exploratory monitoring pattern; no claim is shown to be wrong"),
        ))
    return out


# ===========================================================================
# CLN-04 — repeats and cascades
# ===========================================================================

_FAILED = r"fail|inconclus|haemoly|hemoly|reject|insufficient|clotted|repeat requested|unsatisf"


@_register("cln_04_r03_no_result_before_repeat")
def cln_04_r03_no_result_before_repeat(ctx, control) -> list:
    lines = _lines(ctx)
    obs = _observations(ctx)
    pol = _table(ctx, "repeat_interval_policy")
    if lines.empty or obs.empty or pol.empty or "activity_code" not in pol.columns:
        return []
    need = pol[_bool(pol, "requires_result_before_repeat")] if "requires_result_before_repeat" in pol.columns else pol
    codes = set(_str(need, "activity_code"))
    work = lines[lines["activity_code"].isin(codes) & lines["sd"].notna()]
    if work.empty:
        return []
    window = int(ctx.cfg("cln04_repeat_window_days"))
    latency = int(ctx.cfg("cln04_result_latency_days"))
    first_result = obs[obs["line_sk"] != ""].groupby("line_sk")["t"].min()
    failed = obs[obs["val"].str.lower().str.contains(_FAILED, regex=True)]["line_sk"]
    failed = set(failed[failed != ""])
    work = work.sort_values(["member_sk", "activity_code", "sd", "line_sk"])
    prev = work.groupby(["member_sk", "activity_code"]).shift(1)
    pairs = work.assign(prev_line=prev["line_sk"], prev_sd=prev["sd"], prev_claim=prev["claim_sk"],
                        prev_amount=prev["amount"])
    pairs = pairs[pairs["prev_line"].notna()]
    pairs = pairs.assign(gap=(pairs["sd"] - pairs["prev_sd"]).dt.days)
    pairs = pairs[(pairs["gap"] >= 1) & (pairs["gap"] <= window)]
    out = []
    for row in pairs[["line_sk", "claim_sk", "member_sk", "provider_sk", "activity_code", "label", "amount", "sd", "prev_line", "prev_sd", "prev_claim", "gap"]].itertuples(index=False):
        if row.prev_line in failed:
            continue  # declared exclusion: failed or inconclusive earlier test
        res = first_result.get(row.prev_line)
        if res is not None and not pd.isna(res):
            if res <= row.sd:
                continue  # a result existed before the repeat
            if (res - row.sd).days <= latency:
                continue  # declared exclusion: result latency
        status = "never recorded" if res is None or pd.isna(res) else f"recorded only on {plain_date(res)}"
        out.append(_sig(
            ctx, control, subject_type="claim", subject_id=row.claim_sk,
            fact_key=f"repeatresult:{row.prev_line}:{row.line_sk}", claim_ids=[row.prev_claim, row.claim_sk],
            event_time=row.sd, period=period_bucket(row.sd), confidence=0.8,
            evidence={
                "claim_sk": row.claim_sk, "line_sk": row.line_sk, "member_sk": row.member_sk,
                "provider_sk": row.provider_sk, "activity_code": row.activity_code,
                "activity_description": row.label, "earlier_line_sk": row.prev_line,
                "earlier_claim_sk": row.prev_claim, "earlier_service_date": _day(row.prev_sd),
                "repeat_service_date": _day(row.sd), "days_between": int(row.gap),
                "earlier_result": status, "line_amount_aed": round(float(row.amount), 2),
                "plain_language": (
                    f"{row.label} was repeated {int(row.gap)} days after the first test "
                    f"({plain_date(row.prev_sd)}), but the first test's result was {status}."
                ),
                "what_the_reviewer_must_verify": (
                    "Whether the first result was lost, failed or inconclusive; otherwise the repeat was "
                    "ordered before anyone could have acted on the first."
                ),
            },
            exposure=_exp.line_edit_exposure(float(row.amount), 0.0),
        ))
    return out


@_register("cln_04_r05_cascade_pattern")
def cln_04_r05_cascade_pattern(ctx, control) -> list:
    ref = _table(ctx, "referral")
    hdr = _header(ctx)
    if ref.empty or hdr.empty or "referrer_provider_sk" not in ref.columns:
        return []
    ref = ref.copy()
    ref["d"] = _dates(ref["referral_date"]) if "referral_date" in ref.columns else pd.NaT
    ref["resulting_claim_sk"] = _str(ref, "resulting_claim_sk")
    ref["referral_sk"] = _str(ref, "referral_sk")
    ref = ref[_str(ref, "referrer_provider_sk") != ""]
    if ref.empty:
        return []
    window = int(ctx.cfg("cln04_cascade_window_days"))
    amounts = hdr.set_index("claim_sk")["gross"]
    # downstream spend per referral: the claim(s) the referral produced, else
    # every claim of the member at the recipient inside the window
    down = ref["resulting_claim_sk"].map(amounts)
    need = down.isna() & ref["recipient_provider_sk"].notna() if "recipient_provider_sk" in ref.columns else down.isna()
    if need.any():
        m = ref[need].reset_index().merge(hdr[["member_sk", "provider_sk", "sd", "gross"]],
                                          left_on=["member_sk", "recipient_provider_sk"],
                                          right_on=["member_sk", "provider_sk"], how="left")
        ok = (m["sd"] >= m["d"]) & ((m["sd"] - m["d"]).dt.days <= window)
        fill = m[ok].groupby("index")["gross"].sum()
        down = down.fillna(fill)
    ref["downstream"] = down.fillna(0.0)
    # declared exclusion (specialty pathway): each referral is compared with the
    # typical follow-on spend of referrals into the SAME specialty
    ref["pathway"] = _str(ref, "specialty").where(_str(ref, "specialty") != "", "UNSPECIFIED")
    # the expectation for a referral is what OTHER providers' referrals into the
    # same specialty lead to (leave-one-out, so a provider cannot set its own norm)
    min_ref = int(ctx.cfg("cln04_cascade_min_referrals"))
    ref["expected"] = np.nan
    for (pathway, referrer), idx in ref.groupby(["pathway", "referrer_provider_sk"]).groups.items():
        others = ref.loc[(ref["pathway"] == pathway) & (ref["referrer_provider_sk"] != referrer), "downstream"]
        if len(others) < min_ref:
            others = ref.loc[ref["referrer_provider_sk"] != referrer, "downstream"]
        ref.loc[idx, "expected"] = float(others.median()) if len(others) else np.nan
    per = ref.groupby("referrer_provider_sk").agg(
        referrals=("referral_sk", "size"), mean_downstream=("downstream", "mean"),
        total_downstream=("downstream", "sum"), total_expected=("expected", "sum"),
        top_recipient=("recipient_provider_sk", lambda s: s.value_counts(normalize=True).iloc[0]
                       if s.notna().any() else 0.0))
    per = per[(per["referrals"] >= min_ref) & (per["total_expected"] > 0)]
    if len(per) < int(ctx.cfg("cln03_min_specialty_peers")):
        return []
    per["ratio"] = per["total_downstream"] / per["total_expected"]
    z, stats = _peer_z(per["ratio"])
    z_cut = float(ctx.cfg("robust_residual_flag_threshold"))
    multiple = float(ctx.cfg("cln04_cascade_peer_multiple"))
    out = []
    for provider, row in per.sort_index().iterrows():
        zz = z.get(provider)
        if zz is None or pd.isna(zz) or zz <= z_cut or row["ratio"] < multiple:
            continue
        sub = ref[ref["referrer_provider_sk"] == provider]
        claims = sorted(set(sub["resulting_claim_sk"]) - {""})
        excess = (sub["downstream"] - sub["expected"]).clip(lower=0).round(2).tolist()
        typical = float(row["total_expected"] / row["referrals"])
        last = sub["d"].max()
        out.append(_sig(
            ctx, control, subject_type="provider", subject_id=provider,
            fact_key=f"cascade:{provider}", claim_ids=claims, event_time=last, period=period_bucket(last),
            peer_level_used="referrals into the same specialty", confidence=0.7,
            evidence={
                "provider_sk": provider, "referrals": int(row["referrals"]),
                "mean_downstream_aed": round(float(row["mean_downstream"]), 2),
                "expected_downstream_per_referral_aed": round(typical, 2),
                "observed_to_expected": round(float(row["ratio"]), 2),
                "peer_median_ratio": round(float(stats.median), 2),
                "robust_residual": round(float(zz), 2),
                "total_downstream_aed": round(float(row["total_downstream"]), 2),
                "top_recipient_share": round(float(row["top_recipient"]), 3),
                "top_recipients": {str(k): int(v) for k, v in
                                   sub["recipient_provider_sk"].value_counts().head(3).items()},
                "pathways": {str(k): int(v) for k, v in sub["pathway"].value_counts().head(3).items()},
                "plain_language": (
                    f"Each referral from this provider led on average to {aed(row['mean_downstream'])} of "
                    f"follow-on services, where referrals into the same specialties typically lead to "
                    f"{aed(typical)} ({times_phrase(row['ratio'], 1.0, 'the usual amount')}), across "
                    f"{int(row['referrals'])} referrals; {pct(row['top_recipient'])} went to a single recipient."
                ),
                "what_the_reviewer_must_verify": (
                    "Whether a specialty pathway explains the follow-on bundle, and whether the provider and "
                    "the main recipient are related."
                ),
            },
            exposure=_exp.provider_pattern_exposure(excess),
        ))
    return out


# ===========================================================================
# CLN-05-R04 — admission rate
# ===========================================================================


def _encounter_kind(enc: pd.DataFrame) -> pd.Series:
    t = _str(enc, "encounter_type").str.lower()
    out = pd.Series("other", index=enc.index, dtype=object)
    out[t.str.contains(r"emerg|\bed\b|^er$|urgent|a&e")] = "ed"
    out[t.str.contains(r"outpat|^op$|clinic|ambul")] = "op"
    out[t.str.contains(r"inpat|^ip$|admit|admission")] = "ip"
    out[t.str.contains(r"day ?case|daycare|day surg")] = "daycase"
    return out


@_register("cln_05_r04_admission_rate_outlier")
def cln_05_r04_admission_rate_outlier(ctx, control) -> list:
    from scipy import stats as _st

    enc = _table(ctx, "encounter")
    hdr = _header(ctx)
    if enc.empty or hdr.empty or "encounter_type" not in enc.columns:
        return []
    enc = enc.copy()
    enc["claim_sk"] = _str(enc, "claim_sk")
    enc = enc.merge(hdr[["claim_sk", "provider_sk", "member_sk", "sd", "diagnosis_primary", "inpatient"]]
                    .rename(columns={"member_sk": "h_member"}), on="claim_sk", how="inner")
    enc["kind"] = _encounter_kind(enc)
    obs_status = _str(enc, "observation_status").str.lower()
    adm_type = _str(enc, "admission_type").str.lower()
    # presentations: emergency and walk-in outpatient attendances at the hospital, and
    # the unplanned admissions that came through them; planned admissions are left out
    emergency_admit = (enc["kind"] == "ip") & adm_type.str.contains("emerg|urgent|unplanned")
    observation = obs_status.str.contains("observ") & ~obs_status.str.contains("not")
    pres = enc[enc["kind"].isin(["ed", "op"]) | emergency_admit].copy()
    if pres.empty:
        return []
    pres["admitted"] = (emergency_admit.loc[pres.index] & ~observation.loc[pres.index]).astype(float)
    pres["chapter"] = _code(pres["diagnosis_primary"]).str[:1]
    # only providers that can admit at all are compared
    can_admit = set(enc.loc[enc["kind"] == "ip", "provider_sk"])
    pres = pres[pres["provider_sk"].isin(can_admit)]
    if pres.empty:
        return []
    chapter_rate = pres.groupby("chapter")["admitted"].mean()
    pres["expected"] = pres["chapter"].map(chapter_rate)
    agg = pres.groupby("provider_sk").agg(n=("admitted", "size"), observed=("admitted", "sum"),
                                          expected=("expected", "sum"))
    min_n = int(ctx.cfg("cln05_min_presentations"))
    ratio_cut = float(ctx.cfg("cln05_admission_oe_ratio"))
    p_cut = float(ctx.cfg("cln02_comorbidity_max_pvalue"))
    out = []
    for provider, row in agg.sort_index().iterrows():
        if row["n"] < min_n or row["expected"] <= 0:
            continue
        ratio = row["observed"] / row["expected"]
        if ratio < ratio_cut:
            continue
        p = float(_st.binom.sf(row["observed"] - 1, row["n"], min(row["expected"] / row["n"], 1.0)))
        if p > p_cut:
            continue
        sub = pres[pres["provider_sk"] == provider]
        adm_claims = sorted(sub.loc[sub["admitted"] == 1, "claim_sk"].unique().tolist())
        last = sub["sd"].max()
        amounts = hdr.set_index("claim_sk")["gross"].reindex(adm_claims).fillna(0.0)
        excess_share = max(0.0, (ratio - 1.0) / ratio)
        out.append(_sig(
            ctx, control, subject_type="provider", subject_id=provider,
            fact_key=f"admitrate:{provider}", claim_ids=adm_claims, event_time=last,
            period=period_bucket(last), peer_level_used="principal diagnosis chapter", confidence=0.75,
            evidence={
                "provider_sk": provider, "presentations": int(row["n"]), "admitted": int(row["observed"]),
                "observed_rate": round(row["observed"] / row["n"], 4),
                "expected_rate": round(row["expected"] / row["n"], 4),
                "observed_to_expected": round(float(ratio), 2), "binomial_tail_probability": p,
                "plain_language": (
                    f"This hospital admitted {pct(row['observed'] / row['n'])} of its "
                    f"{int(row['n'])} unplanned presentations; hospitals seeing the same mix of conditions "
                    f"admit {pct(row['expected'] / row['n'])}."
                ),
                "what_the_reviewer_must_verify": (
                    "Whether severity, referral mix or use of observation status explains the difference; "
                    "sample short admissions for medical necessity."
                ),
            },
            exposure=_exp.provider_pattern_exposure((amounts * excess_share).round(2).tolist()),
        ))
    return out


# ===========================================================================
# CLN-06 — did the service happen
# ===========================================================================


def _latency_cutoff(ctx, key: str) -> pd.Timestamp | None:
    hdr = _header(ctx)
    if hdr.empty:
        return None
    last = hdr["sd"].max()
    if pd.isna(last):
        return None
    return last - pd.Timedelta(days=int(ctx.cfg(key)))


def _lines_without_result(ctx, kinds: set[str]) -> pd.DataFrame:
    lines = _lines(ctx)
    obs = _observations(ctx)
    if lines.empty or obs.empty:
        return _EMPTY
    work = lines[lines["kind"].isin(kinds)]
    if work.empty:
        return _EMPTY
    by_line = set(obs["line_sk"]) - {""}
    by_claim_code = set(zip(obs["claim_sk"], obs["ocode"]))
    has = work["line_sk"].isin(by_line) | pd.Series(
        [(c, a) in by_claim_code for c, a in zip(work["claim_sk"], work["activity_code"])], index=work.index)
    return work[~has]


@_register("cln_06_r01_missing_order_result_trail")
def cln_06_r01_missing_order_result_trail(ctx, control) -> list:
    missing = _lines_without_result(ctx, {"imaging"})
    if missing.empty:
        return []
    cutoff = _latency_cutoff(ctx, "cln06_artifact_latency_days")
    if cutoff is not None:
        missing = missing[missing["sd"] <= cutoff]  # declared exclusion: artefact latency
    missing = missing.assign(noord=_str(missing, "ordering_clinician_id") == "").sort_values("line_sk")
    out = []
    for row in missing[["line_sk", "claim_sk", "member_sk", "provider_sk", "activity_code", "label", "amount", "sd", "kind", "noord"]].itertuples(index=False):
        noord = bool(row.noord)
        what = "no report or result" + (" and no ordering clinician" if noord else "")
        out.append(_sig(
            ctx, control, subject_type="claim", subject_id=row.claim_sk,
            fact_key=f"result:{row.line_sk}", claim_ids=[row.claim_sk], event_time=row.sd,
            period=period_bucket(row.sd), confidence=0.8,
            evidence={
                "claim_sk": row.claim_sk, "line_sk": row.line_sk, "member_sk": row.member_sk,
                "provider_sk": row.provider_sk, "activity_code": row.activity_code,
                "activity_description": row.label, "service_kind": row.kind,
                "ordering_clinician_recorded": not noord, "result_recorded": False,
                "line_amount_aed": round(float(row.amount), 2),
                "plain_language": (
                    f"{row.label} ({aed(row.amount)}) on {plain_date(row.sd)} has {what} on file, "
                    f"{int(ctx.cfg('cln06_artifact_latency_days'))}+ days after the service."
                ),
                "what_the_reviewer_must_verify": "Request the report or administration record for this service.",
            },
            exposure=_exp.line_edit_exposure(float(row.amount), 0.0),
        ))
    return out


_UNVERIFIED = r"unverif|anonym|unauth|unknown"


@_register("cln_06_r04_member_denial_attendance")
def cln_06_r04_member_denial_attendance(ctx, control) -> list:
    hdr = _header(ctx)
    conf = _table(ctx, "member_confirmation")
    att = _table(ctx, "attendance_record")
    if hdr.empty or (conf.empty and att.empty):
        return []
    recall = int(ctx.cfg("cln06_confirmation_recall_max_days"))
    info = hdr.set_index("claim_sk")
    findings: dict[str, list[str]] = {}
    facts: dict[str, dict[str, Any]] = {}
    if not conf.empty and "claim_sk" in conf.columns:
        c = conf.copy()
        c["claim_sk"] = _str(c, "claim_sk")
        c = c[c["claim_sk"].isin(info.index)]
        denied = c["service_confirmed"].astype(str).str.lower().isin({"false", "0", "no", "n"}) \
            if "service_confirmed" in c.columns else pd.Series(False, index=c.index)
        reliable = ~_str(c, "channel").str.lower().str.contains(_UNVERIFIED, regex=True)
        rd = _dates(c["response_date"]) if "response_date" in c.columns else pd.Series(pd.NaT, index=c.index)
        delay = (rd - c["claim_sk"].map(info["sd"])).dt.days
        timely = delay.isna() | (delay <= recall)
        for r, d in zip(c[denied & reliable & timely].itertuples(index=False), delay[denied & reliable & timely]):
            findings.setdefault(r.claim_sk, []).append(
                f"the patient answered a verification request ({getattr(r, 'channel', '') or 'channel not recorded'}) "
                f"saying the service was not received")
            facts.setdefault(r.claim_sk, {})["member_response"] = {
                "response": getattr(r, "response", None), "channel": getattr(r, "channel", None),
                "response_date": _day(getattr(r, "response_date", None)),
                "days_after_service": None if pd.isna(d) else int(d)}
    if not att.empty and "attended" in att.columns:
        a = att.copy()
        a["claim_sk"] = _str(a, "claim_sk")
        a["d"] = _dates(a["attendance_date"]) if "attendance_date" in a.columns else pd.NaT
        linked = a[a["claim_sk"].isin(info.index)]
        unlinked = a[~a["claim_sk"].isin(info.index)]
        if not unlinked.empty and {"member_sk", "provider_sk"} <= set(unlinked.columns):
            m = unlinked.drop(columns=["claim_sk"]).merge(
                hdr[["claim_sk", "member_sk", "provider_sk", "sd"]], on=["member_sk", "provider_sk"], how="inner")
            m = m[m["sd"].dt.normalize() == m["d"].dt.normalize()]
            linked = pd.concat([linked, m[linked.columns.intersection(m.columns)]], ignore_index=True)
        absent = linked["attended"].astype(str).str.lower().isin({"false", "0", "no", "n"})
        for r in linked[absent].itertuples(index=False):
            findings.setdefault(r.claim_sk, []).append(
                f"the {getattr(r, 'source', '') or 'attendance'} record shows the patient did not attend")
            facts.setdefault(r.claim_sk, {})["attendance_record"] = {
                "source": getattr(r, "source", None), "attendance_date": _day(getattr(r, "d", None)),
                "attended": False}
    out = []
    for claim in sorted(findings):
        row = info.loc[claim]
        reasons = findings[claim]
        out.append(_sig(
            ctx, control, subject_type="claim", subject_id=claim, fact_key=f"notreceived:{claim}",
            claim_ids=[claim], event_time=row["sd"], period=period_bucket(row["sd"]),
            confidence=0.85 if len(reasons) > 1 else 0.7,
            evidence={
                "claim_sk": claim, "member_sk": row["member_sk"], "provider_sk": row["provider_sk"],
                "service_date": _day(row["sd"]), "gross_amount_aed": round(float(row["gross"]), 2),
                **_jsonable(facts.get(claim, {})),
                "plain_language": (
                    f"For this {aed(row['gross'])} claim dated {plain_date(row['sd'])}, " + " and ".join(reasons) + "."
                ),
                "what_the_reviewer_must_verify": (
                    "Re-contact the patient through an authenticated channel and ask the provider for "
                    "sign-in or appointment evidence."
                ),
            },
            exposure=_exp.model_only_exposure(float(row["gross"]), "member statement or attendance record"),
        ))
    return out


@_register("cln_06_r05_no_longitudinal_footprint")
def cln_06_r05_no_longitudinal_footprint(ctx, control) -> list:
    dx = _diagnoses(ctx)
    drugs = _table(ctx, "drug_policy")
    if dx.empty or drugs.empty or "indication_prefixes" not in drugs.columns:
        return []
    # conditions a medicine is indicated for, named at category level (a bare
    # chapter letter such as "M" names a body system, not a condition)
    treated_prefixes = sorted({p for t in drugs["indication_prefixes"] for p in _prefixes(t) if len(p) >= 3})
    if not treated_prefixes:
        return []
    min_claims = int(ctx.cfg("cln06_footprint_min_claims"))
    peer_min = float(ctx.cfg("cln06_footprint_peer_min_rate"))
    work = dx[["member_sk", "claim_sk", "dx", "cat", "sd", "provider_sk", "gross"]].dropna(subset=["sd"])
    work = work[_prefix_mask(work["dx"], treated_prefixes)]
    if work.empty:
        return []
    work = work.assign(month=_month(work["sd"]))
    per = work.groupby(["member_sk", "cat"]).agg(
        claims=("claim_sk", "nunique"), months=("month", "nunique"),
        first=("sd", "min"), last=("sd", "max"), total=("gross", "sum")).reset_index()
    per = per[(per["claims"] >= min_claims) & (per["months"] >= 2)]
    if per.empty:
        return []
    _, _, member_drugs = _support_index(ctx, dx)
    obs = _observations(ctx)
    obs_members: set[str] = set()
    if not obs.empty:
        if "member_sk" in obs.columns and obs["member_sk"].notna().any():
            obs_members = set(obs["member_sk"].dropna().astype(str))
        hdr = _header(ctx)
        obs_members |= set(hdr.loc[hdr["claim_sk"].isin(set(obs["claim_sk"]) - {""}), "member_sk"].astype(str))
    per["drug"] = [_covered(c, member_drugs.get(str(m), set())) for m, c in zip(per["member_sk"], per["cat"])]
    per["result"] = per["member_sk"].astype(str).isin(obs_members)
    per["footprint"] = per["drug"] | per["result"]
    peer_rate = per.groupby("cat")["footprint"].mean()
    out = []
    for row in per[~per["footprint"]].sort_values(["member_sk", "cat"]).itertuples(index=False):
        rate = float(peer_rate.get(row.cat, 0.0))
        if rate < peer_min:
            continue  # peers with this condition do not usually leave a trace either
        claims = sorted(work.loc[(work["member_sk"] == row.member_sk) & (work["cat"] == row.cat),
                                 "claim_sk"].unique().tolist())
        out.append(_sig(
            ctx, control, subject_type="member", subject_id=row.member_sk,
            fact_key=f"footprint:{row.member_sk}:{row.cat}", claim_ids=claims, event_time=row.last,
            period=period_bucket(row.last), confidence=0.6,
            evidence={
                "member_sk": row.member_sk, "condition_category": row.cat, "claims_with_condition": int(row.claims),
                "months_with_condition": int(row.months), "first_claim": _day(row.first), "last_claim": _day(row.last),
                "medicine_dispensed": False, "results_recorded": False,
                "peer_footprint_rate": round(rate, 3), "billed_on_those_claims_aed": round(float(row.total), 2),
                "providers": sorted(work.loc[(work["member_sk"] == row.member_sk) & (work["cat"] == row.cat),
                                             "provider_sk"].astype(str).unique().tolist()),
                "plain_language": (
                    f"This patient's condition {row.cat} was billed on {int(row.claims)} claims over "
                    f"{int(row.months)} months, yet no medicine for it was dispensed and no test result exists; "
                    f"{pct(rate)} of other patients billed for it show one or the other."
                ),
                "what_the_reviewer_must_verify": (
                    "Whether the patient is treated out of network; this is supporting evidence only."
                ),
            },
            exposure=_exp.no_exposure("supporting evidence only; no individual claim is shown to be unpayable"),
        ))
    return out


# ===========================================================================
# CLN-07 — capacity
# ===========================================================================


@_register("cln_07_r01_improbable_service_day")
def cln_07_r01_improbable_service_day(ctx, control) -> list:
    lines = _lines(ctx)
    if lines.empty or "rendering_clinician_id" not in lines.columns:
        return []
    work = lines[(_str(lines, "rendering_clinician_id") != "") & lines["day"].notna()].copy()
    # parallel and team services are excluded: laboratory, imaging and dispensing
    # run without the clinician's continuous presence
    work = work[work["kind"].isin({"consult", "procedure", "therapy"})]
    team = _str(work, "indicator").str.lower().str.contains("team|assist|parallel|supervis")
    work = work[~team]
    if work.empty:
        return []
    ref_min = _num(work, "ref_minutes")
    if "service_start_time" in work.columns and "service_end_time" in work.columns:
        dur = (_dates(work["service_end_time"]) - _dates(work["service_start_time"])).dt.total_seconds() / 60
        dur = dur.where(dur > 0)
    else:
        dur = pd.Series(np.nan, index=work.index)
    # conservative: the reference minimum time per unit, else the recorded duration
    work["minutes"] = (ref_min * work["units_n"]).fillna(dur)
    work = work[work["minutes"].notna()]
    work["clinician"] = _str(work, "rendering_clinician_id")
    capacity = float(ctx.cfg("cln07_clinician_day_capacity_minutes")) * float(ctx.cfg("capacity_tolerance_multiple"))
    daily = work.groupby(["clinician", "day"]).agg(minutes=("minutes", "sum"), services=("line_sk", "size"),
                                                  claims=("claim_sk", "nunique")).reset_index()
    over = daily[daily["minutes"] > capacity]
    out = []
    for row in over.sort_values(["clinician", "day"]).itertuples(index=False):
        sub = work[(work["clinician"] == row.clinician) & (work["day"] == row.day)]
        top = sub.groupby("label")["minutes"].agg(["size", "sum"]).sort_values("sum", ascending=False).head(4)
        providers = sorted(sub["provider_sk"].astype(str).unique().tolist())
        out.append(_sig(
            ctx, control, subject_type="clinician", subject_id=row.clinician,
            fact_key=f"clinday:{row.clinician}:{_day(row.day)}",
            claim_ids=sorted(sub["claim_sk"].unique().tolist()), event_time=row.day,
            period=period_bucket(row.day), confidence=0.8,
            evidence={
                "clinician_id": row.clinician, "service_date": _day(row.day), "provider_sk": providers[0],
                "providers": providers, "billed_minutes": round(float(row.minutes), 1),
                "billed_hours": round(float(row.minutes) / 60.0, 1), "services": int(row.services),
                "claims": int(row.claims), "capacity_minutes_with_tolerance": round(capacity, 1),
                "largest_contributors": [{"service": str(k), "count": int(v["size"]), "minutes": float(v["sum"])}
                                         for k, v in top.iterrows()],
                "plain_language": (
                    f"On {plain_date(row.day)} this clinician billed {int(row.services)} services needing at least "
                    f"{float(row.minutes) / 60.0:.1f} hours, against {capacity / 60.0:.1f} hours available in a "
                    f"working day even with a margin."
                ),
                "what_the_reviewer_must_verify": (
                    "Whether team members, documented extended hours or a billing-identity error explain the day."
                ),
            },
            exposure=_exp.no_exposure("a timeline alone cannot say which of the day's services did not happen"),
        ))
    return out


@_register("cln_07_r03_scarce_equipment_concurrency")
def cln_07_r03_scarce_equipment_concurrency(ctx, control) -> list:
    lines = _lines(ctx)
    inv = _table(ctx, "equipment_inventory")
    if lines.empty or inv.empty or "activity_codes" not in inv.columns:
        return []
    tol = float(ctx.cfg("capacity_tolerance_multiple"))
    rows = []
    for r in inv.itertuples(index=False):
        for code in _prefixes(getattr(r, "activity_codes", "")):
            rows.append({"provider_sk": r.provider_sk, "equipment_type": getattr(r, "equipment_type", ""),
                         "activity_code": code, "eq_units": _f(getattr(r, "units", None), 0.0),
                         "minutes_per_use": _f(getattr(r, "minutes_per_use", None), 0.0),
                         "hours_per_day": _f(getattr(r, "hours_per_day", None), 0.0)})
    if not rows:
        return []
    m = pd.DataFrame(rows)
    work = lines.assign(code_u=lines["activity_code"].str.upper().str.replace(".", "", regex=False))
    work = work.merge(m, left_on=["provider_sk", "code_u"], right_on=["provider_sk", "activity_code"],
                      how="inner", suffixes=("", "_eq"))
    work = work[work["day"].notna() & (work["minutes_per_use"] > 0) & (work["eq_units"] > 0)]
    if work.empty:
        return []
    daily = work.groupby(["provider_sk", "equipment_type", "day"]).agg(
        uses=("units_n", "sum"), lines_n=("line_sk", "size"), eq_units=("eq_units", "first"),
        minutes_per_use=("minutes_per_use", "first"), hours=("hours_per_day", "first")).reset_index()
    daily["capacity"] = np.floor(daily["eq_units"] * daily["hours"] * 60.0 / daily["minutes_per_use"])
    over = daily[daily["uses"] > daily["capacity"] * tol]
    out = []
    for row in over.sort_values(["provider_sk", "day"]).itertuples(index=False):
        sub = work[(work["provider_sk"] == row.provider_sk) & (work["equipment_type"] == row.equipment_type)
                   & (work["day"] == row.day)].sort_values("amount")
        n_excess = int(max(0, row.uses - row.capacity))
        excess_amounts = sub["amount"].head(n_excess).round(2).tolist()
        out.append(_sig(
            ctx, control, subject_type="provider", subject_id=row.provider_sk,
            fact_key=f"equipment:{row.provider_sk}:{row.equipment_type}:{_day(row.day)}",
            claim_ids=sorted(sub["claim_sk"].unique().tolist()), event_time=row.day,
            period=period_bucket(row.day), confidence=0.85,
            evidence={
                "provider_sk": row.provider_sk, "equipment_type": row.equipment_type,
                "service_date": _day(row.day), "uses_billed": float(row.uses),
                "machines": float(row.eq_units), "minutes_per_use": float(row.minutes_per_use),
                "hours_per_day": float(row.hours), "daily_capacity": float(row.capacity),
                "tolerance_multiple": tol, "uses_beyond_capacity": n_excess,
                "plain_language": (
                    f"On {plain_date(row.day)} this provider billed {int(row.uses)} {row.equipment_type} scans, "
                    f"but {int(row.eq_units)} machine(s) at {int(row.minutes_per_use)} minutes a scan over "
                    f"{int(row.hours)} hours can do at most {int(row.capacity)}."
                ),
                "what_the_reviewer_must_verify": (
                    "Whether the equipment inventory is out of date or scans were done at another site."
                ),
            },
            exposure=_exp.provider_pattern_exposure(excess_amounts),
        ))
    return out


# ===========================================================================
# CLN-08 — laboratory
# ===========================================================================


def _screening(dx: pd.DataFrame) -> set[str]:
    if dx.empty:
        return set()
    return set(dx.loc[dx["dx"].str.match(r"Z(0[0-9]|1[0-3])"), "claim_sk"])


@_register("cln_08_r01_no_qualified_order")
def cln_08_r01_no_qualified_order(ctx, control) -> list:
    lines = _lines(ctx)
    roster = _table(ctx, "clinician_roster")
    if lines.empty or roster.empty or "clinician_id" not in roster.columns:
        return []
    labs = lines[lines["kind"] == "lab"].copy()
    if labs.empty:
        return []
    dx = _diagnoses(ctx)
    labs = labs[~labs["claim_sk"].isin(_screening(dx))]  # declared exclusion: screening programmes
    standing = _str(labs, "indicator").str.lower().str.contains("standing")
    labs = labs[~standing]  # declared exclusion: standing orders
    r = roster.copy()
    r["clinician_id"] = _str(r, "clinician_id")
    r["vf"] = _dates(r["licence_valid_from"]) if "licence_valid_from" in r.columns else pd.NaT
    r["vt"] = _dates(r["licence_valid_to"]) if "licence_valid_to" in r.columns else pd.NaT
    r["role_l"] = _str(r, "role").str.lower()
    # a test done during a visit is ordered by the clinician who saw the patient:
    # with no separate order, the visit's clinician is the qualifying relationship
    visit_claims = set(lines.loc[lines["kind"] == "consult", "claim_sk"])
    own = _str(labs, "ordering_clinician_id")
    in_visit = labs["claim_sk"].isin(visit_claims) & (_str(labs, "rendering_clinician_id") != "")
    labs["orderer"] = own.where((own != "") | ~in_visit, _str(labs, "rendering_clinician_id"))
    known = set(r["clinician_id"])
    lic = labs[["line_sk", "orderer", "sd"]].merge(r[["clinician_id", "vf", "vt", "role_l"]],
                                                   left_on="orderer", right_on="clinician_id", how="inner")
    valid = (lic["vf"].isna() | (lic["vf"] <= lic["sd"])) & (lic["vt"].isna() | (lic["vt"] >= lic["sd"]))
    can_order = ~lic["role_l"].str.contains("technician|technologist|receptionist|admin|clerk")
    ok_lines = set(lic.loc[valid & can_order, "line_sk"])
    out = []
    for row in labs[["line_sk", "claim_sk", "member_sk", "provider_sk", "activity_code", "label", "amount", "sd", "orderer"]].sort_values("line_sk").itertuples(index=False):
        if row.orderer == "":
            problem = "has no ordering clinician"
        elif row.orderer not in known:
            problem = f"names an ordering clinician ({row.orderer}) who is not on any clinician roster"
        elif row.line_sk not in ok_lines:
            problem = (f"names an ordering clinician ({row.orderer}) who did not hold a valid licence or "
                       f"ordering role on the service date")
        else:
            continue
        out.append(_sig(
            ctx, control, subject_type="claim", subject_id=row.claim_sk,
            fact_key=f"order:{row.line_sk}", claim_ids=[row.claim_sk], event_time=row.sd,
            period=period_bucket(row.sd), confidence=0.85,
            evidence={
                "claim_sk": row.claim_sk, "line_sk": row.line_sk, "member_sk": row.member_sk,
                "provider_sk": row.provider_sk, "activity_code": row.activity_code,
                "activity_description": row.label, "clinician_id": row.orderer or None,
                "line_amount_aed": round(float(row.amount), 2),
                "plain_language": f"The laboratory test {row.label} ({aed(row.amount)}) {problem}.",
                "what_the_reviewer_must_verify": (
                    "Ask for the signed test order; check whether a screening programme or standing order covers it."
                ),
            },
            exposure=_exp.line_edit_exposure(float(row.amount), 0.0),
        ))
    return out


@_register("cln_08_r02_panel_component_inflation")
def cln_08_r02_panel_component_inflation(ctx, control) -> list:
    lines = _lines(ctx)
    panels = _table(ctx, "panel_definition")
    if lines.empty or panels.empty or not {"panel_code", "component_code"} <= set(panels.columns):
        return []
    panels = panels.assign(panel_code=_str(panels, "panel_code"), component_code=_str(panels, "component_code"))
    comp_of = panels.groupby("panel_code")["component_code"].apply(lambda s: sorted(set(s))).to_dict()
    comp_of = {k: v for k, v in comp_of.items() if len(v) >= 2}
    if not comp_of:
        return []
    all_codes = set(comp_of) | {c for v in comp_of.values() for c in v}
    work = lines[lines["activity_code"].isin(all_codes)]
    reflex = _str(work, "indicator").str.lower().str.contains("reflex")
    work = work[~reflex]  # declared exclusion: reflex testing
    comp_codes = {c for v in comp_of.values() for c in v}
    per_claim = work.assign(comp=work["activity_code"].isin(comp_codes)).groupby("claim_sk")["comp"].sum()
    work = work[work["claim_sk"].isin(set(per_claim[per_claim >= 2].index))]  # a panel needs two or more
    if work.empty:
        return []
    price_ref = _reference(ctx)
    ref_price = pd.Series(dtype=float)
    if not price_ref.empty and "unit_price_reference" in price_ref.columns:
        ref_price = _num(price_ref, "unit_price_reference").set_axis(price_ref["activity_code"])
    tariff = _table(ctx, "tariff")
    if not tariff.empty and {"activity_code", "allowed_price"} <= set(tariff.columns):
        tp = _num(tariff, "allowed_price").groupby(tariff["activity_code"].astype(str)).median()
        ref_price = tp.combine_first(ref_price)
    out = []
    for (claim, day), sub in work.groupby(["claim_sk", "day"], sort=True):
        codes = set(sub["activity_code"])
        for panel, comps in sorted(comp_of.items()):
            billed = [c for c in comps if c in codes]
            if len(billed) < len(comps) and panel not in codes:
                continue
            comp_lines = sub[sub["activity_code"].isin(billed)]
            comp_amount = float(comp_lines["amount"].sum())
            if panel in codes:
                if not billed:
                    continue
                correct = 0.0
                panel_amt = float(sub.loc[sub["activity_code"] == panel, "amount"].sum())
                what = (f"the panel {panel} ({aed(panel_amt)}) and {len(billed)} of its own components "
                        f"({aed(comp_amount)}) are both billed")
                limb = "panel_and_components"
            else:
                price = ref_price.get(panel)
                if price is None or pd.isna(price):
                    continue  # no panel price to reprice to
                correct = float(price)
                if comp_amount <= correct:
                    continue
                what = (f"all {len(comps)} components of panel {panel} are billed one by one for "
                        f"{aed(comp_amount)}, where the panel is priced at {aed(correct)}")
                limb = "components_billed_separately"
            first = sub.iloc[0]
            out.append(_sig(
                ctx, control, subject_type="claim", subject_id=claim,
                fact_key=f"panel:{claim}:{panel}", claim_ids=[claim], event_time=day, period=period_bucket(day),
                evidence={
                    "claim_sk": claim, "member_sk": first["member_sk"], "provider_sk": first["provider_sk"],
                    "panel_code": panel, "component_codes_billed": billed, "limb": limb,
                    "component_amount_aed": round(comp_amount, 2), "correct_payable_aed": round(correct, 2),
                    "line_sks": sorted(comp_lines["line_sk"].tolist()),
                    "plain_language": f"On this claim {what}.",
                    "what_the_reviewer_must_verify": (
                        "Whether a local panel rule or a reflex test justifies separate billing."
                    ),
                },
                exposure=_exp.line_edit_exposure(comp_amount, correct),
            ))
    return out


@_register("cln_08_r03_absent_duplicate_result")
def cln_08_r03_absent_duplicate_result(ctx, control) -> list:
    out = []
    missing = _lines_without_result(ctx, {"lab"})
    obs = _observations(ctx)
    if obs.empty:
        return []
    # limb 1: a paid test with no result, after the latency allowance
    if not missing.empty:
        cutoff = _latency_cutoff(ctx, "cln06_artifact_latency_days")
        if cutoff is not None:
            missing = missing[missing["sd"] <= cutoff]
        rem = _table(ctx, "remittance")
        if not rem.empty and "line_sk" in rem.columns:
            paid_lines = set(rem.loc[_num(rem, "payment_amount").fillna(0) > 0, "line_sk"].astype(str))
            paid_claims = set(rem.loc[(_num(rem, "payment_amount").fillna(0) > 0)
                                      & (_str(rem, "line_sk") == ""), "claim_sk"].astype(str))
            missing = missing[missing["line_sk"].isin(paid_lines) | missing["claim_sk"].isin(paid_claims)]
        for row in missing[["line_sk", "claim_sk", "member_sk", "provider_sk", "activity_code", "label", "amount", "sd"]].sort_values("line_sk").itertuples(index=False):
            out.append(_sig(
                ctx, control, subject_type="claim", subject_id=row.claim_sk,
                fact_key=f"result:{row.line_sk}", claim_ids=[row.claim_sk], event_time=row.sd,
                period=period_bucket(row.sd), confidence=0.8,
                evidence={
                    "claim_sk": row.claim_sk, "line_sk": row.line_sk, "member_sk": row.member_sk,
                    "provider_sk": row.provider_sk, "activity_code": row.activity_code,
                    "activity_description": row.label, "limb": "absent_result",
                    "line_amount_aed": round(float(row.amount), 2),
                    "plain_language": (
                        f"The laboratory test {row.label} ({aed(row.amount)}) on {plain_date(row.sd)} was paid, "
                        f"but no result has been recorded for it."
                    ),
                    "what_the_reviewer_must_verify": "Request the laboratory report for this test.",
                },
                exposure=_exp.line_edit_exposure(float(row.amount), 0.0),
            ))
    # limb 2: an identical set of results reported for unrelated patients
    min_results = int(ctx.cfg("cln08_min_results_for_fingerprint"))
    numeric = obs[pd.to_numeric(obs["val"], errors="coerce").notna() & (obs["claim_sk"] != "")]
    if numeric.empty:
        return out
    hdr = _header(ctx)
    numeric = numeric.assign(pair=numeric["ocode"] + "=" + numeric["val"]).sort_values(["claim_sk", "pair"])
    grouped = numeric.groupby("claim_sk")["pair"]
    sizes = grouped.size()
    fp = grouped.agg("|".join)[sizes >= min_results]
    if fp.empty:
        return out
    frame = fp.rename("fp").reset_index().merge(hdr[["claim_sk", "member_sk", "provider_sk", "sd"]], on="claim_sk")
    member = _table(ctx, "member")
    sponsor = member.set_index("member_sk")["sponsor_id"] if not member.empty and "sponsor_id" in member.columns \
        else pd.Series(dtype=object)
    frame["family"] = frame["member_sk"].map(sponsor).fillna(frame["member_sk"]).astype(str)
    for key, sub in frame.groupby("fp", sort=True):
        if sub["family"].nunique() < 2:
            continue  # declared exclusion: same patient or family
        claims = sorted(sub["claim_sk"].tolist())
        last = sub["sd"].max()
        provider = sub["provider_sk"].mode().iloc[0]
        out.append(_sig(
            ctx, control, subject_type="provider", subject_id=provider,
            fact_key=f"sameresult:{'|'.join(claims)}", claim_ids=claims, event_time=last,
            period=period_bucket(last), confidence=0.9,
            evidence={
                "provider_sk": provider, "limb": "identical_results_across_patients",
                "patients": int(sub["member_sk"].nunique()), "claims": claims,
                "member_sks": sorted(sub["member_sk"].astype(str).unique().tolist()),
                "results_in_each_set": int(key.count("|") + 1), "result_set": key[:300],
                "plain_language": (
                    f"{count_phrase(sub['member_sk'].nunique(), 'unrelated patient')} have exactly the same "
                    f"{int(key.count('|') + 1)} numeric laboratory results, value for value, on claims between "
                    f"{plain_date(sub['sd'].min())} and {plain_date(last)}."
                ),
                "what_the_reviewer_must_verify": (
                    "Request the analyser print-outs; identical multi-value results for different people "
                    "suggest a copied report."
                ),
            },
            exposure=_exp.no_exposure("a document-integrity finding; which payment is affected needs review"),
        ))
    return out


@_register("cln_08_r04_reference_lab_passthrough")
def cln_08_r04_reference_lab_passthrough(ctx, control) -> list:
    lines = _lines(ctx)
    if lines.empty or "performing_entity_id" not in lines.columns:
        return []
    labs = lines[lines["kind"] == "lab"].copy()
    labs["performer"] = _str(labs, "performing_entity_id")
    labs["unit_price"] = labs["gross_line"] / labs["units_n"].where(labs["units_n"] > 0, 1.0)
    direct = labs[(labs["performer"] == "") | (labs["performer"] == labs["provider_sk"].astype(str))]
    passed = labs[(labs["performer"] != "") & (labs["performer"] != labs["provider_sk"].astype(str))]
    if passed.empty or direct.empty:
        return []
    # what the performing laboratory charges when it bills the same test itself,
    # else what everybody charges for the test when billing it directly
    own = direct.groupby(["provider_sk", "activity_code"])["unit_price"].median()
    typical = direct.groupby("activity_code")["unit_price"].median()
    base = pd.Series([own.get((p, c), np.nan) for p, c in zip(passed["performer"], passed["activity_code"])],
                     index=passed.index)
    passed["base_price"] = base.fillna(passed["activity_code"].map(typical))
    passed = passed[passed["base_price"] > 0]
    passed["markup"] = passed["unit_price"] / passed["base_price"]
    min_lines = int(ctx.cfg("cln08_passthrough_min_lines"))
    agg = passed.groupby("provider_sk").agg(n=("markup", "size"), markup=("markup", "median"),
                                             performers=("performer", "nunique"),
                                             top_share=("performer", lambda s: s.value_counts(normalize=True).iloc[0]))
    agg = agg[agg["n"] >= min_lines]
    if agg.empty:
        return []
    peer = float(agg["markup"].median())
    multiple = float(ctx.cfg("cln08_passthrough_peer_multiple"))
    min_markup = float(ctx.cfg("cln08_passthrough_min_markup"))
    out = []
    for provider, row in agg.sort_index().iterrows():
        if row["markup"] < max(multiple * peer, min_markup):
            continue
        sub = passed[passed["provider_sk"] == provider]
        top = sub["performer"].value_counts().index[0]
        excess = ((sub["unit_price"] - sub["base_price"] * max(peer, 1.0)) * sub["units_n"]).clip(lower=0)
        last = sub["sd"].max()
        out.append(_sig(
            ctx, control, subject_type="provider", subject_id=provider,
            fact_key=f"passthrough:{provider}", claim_ids=sorted(sub["claim_sk"].unique().tolist()),
            event_time=last, period=period_bucket(last), peer_level_used="all billing providers", confidence=0.7,
            evidence={
                "provider_sk": provider, "pass_through_lines": int(row["n"]),
                "median_markup": round(float(row["markup"]), 2), "peer_median_markup": round(peer, 2),
                "performing_laboratories": int(row["performers"]), "main_performer": top,
                "main_performer_share": round(float(row["top_share"]), 3),
                "billed_aed": round(float(sub["gross_line"].sum()), 2),
                "plain_language": (
                    f"This provider bills {int(row['n'])} tests that another laboratory performed "
                    f"({pct(row['top_share'])} by one lab) at a median {row['markup']:.1f} times what that "
                    f"laboratory charges itself; other providers passing tests through add "
                    f"{max(peer - 1.0, 0.0):.0%} on average."
                ),
                "what_the_reviewer_must_verify": (
                    "Whether a contract permits this reference-laboratory arrangement and its mark-up."
                ),
            },
            exposure=_exp.provider_pattern_exposure(excess.round(2).tolist()),
        ))
    return out


_HIGH_COMPLEXITY = r"high|genetic|genomic|molecular|sequenc|complex"
_SPECIALIST_CENTRE = r"oncolog|cancer|genetic|rare|haemat|hemat"


@_register("cln_08_r05_high_complexity_test_outlier")
def cln_08_r05_high_complexity_test_outlier(ctx, control) -> list:
    lines = _lines(ctx)
    if lines.empty:
        return []
    labs = lines[lines["kind"] == "lab"].copy()
    if labs.empty:
        return []
    marker = (_str(labs, "ref_complexity") + " " + _str(labs, "ref_code_family") + " "
              + _str(labs, "ref_description")).str.lower()
    labs["hc"] = marker.str.contains(_HIGH_COMPLEXITY, regex=True).astype(float)
    if labs["hc"].sum() == 0:
        return []
    prov = _provider_table(ctx)
    if not prov.empty and "specialty" in prov.columns:
        spec = labs["provider_sk"].map(prov["specialty"]).fillna("").astype(str).str.lower()
        labs = labs[~spec.str.contains(_SPECIALIST_CENTRE, regex=True)]  # declared exclusion
    agg = labs.groupby("provider_sk").agg(events=("hc", "sum"), n=("hc", "size"))
    min_n = int(ctx.cfg("cln08_complexity_min_tests"))
    agg = agg[agg["n"] >= min_n]
    if len(agg) < int(ctx.cfg("cln03_min_specialty_peers")):
        return []
    prior = fit_beta_prior(agg["events"], agg["n"])
    mass = float(ctx.cfg("posterior_interval_mass"))
    width = float(ctx.cfg("max_posterior_width"))
    multiple = float(ctx.cfg("cln08_complexity_peer_multiple"))
    out = []
    for provider, row in agg.sort_index().iterrows():
        sr = shrink_rate(str(provider), row["events"], row["n"], prior, interval_mass=mass,
                         max_posterior_width=width)
        if sr.excluded_from_ranking or sr.posterior_low <= prior.peer_mean:
            continue
        if sr.shrunk_rate < multiple * prior.peer_mean:
            continue
        sub = labs[(labs["provider_sk"] == provider) & (labs["hc"] == 1)]
        last = sub["sd"].max()
        peer_cost = labs.loc[labs["hc"] == 1, "amount"].median()
        out.append(_sig(
            ctx, control, subject_type="provider", subject_id=provider,
            fact_key=f"complexlab:{provider}", claim_ids=sorted(sub["claim_sk"].unique().tolist()),
            event_time=last, period=period_bucket(last), peer_level_used="all non-specialist providers",
            confidence=0.8,
            evidence={
                "provider_sk": provider, "high_complexity_tests": int(row["events"]), "lab_tests": int(row["n"]),
                "observed_rate": round(float(sr.observed_rate or 0.0), 4),
                "shrunk_rate": round(sr.shrunk_rate, 4), "peer_mean": round(prior.peer_mean, 4),
                "posterior_interval": [round(sr.posterior_low, 4), round(sr.posterior_high, 4)],
                "mean_cost_aed": round(float(sub["amount"].mean()), 2),
                "peer_median_cost_aed": round(float(peer_cost), 2),
                "shrinkage_explanation": sr.explain(),
                "plain_language": (
                    f"{pct(sr.observed_rate)} of this provider's laboratory tests are genetic or high-complexity "
                    f"tests ({int(row['events'])} of {int(row['n'])}); across comparable providers it is "
                    f"{pct(prior.peer_mean)}."
                ),
                "what_the_reviewer_must_verify": (
                    "Whether the provider runs a specialist service; sample the orders for clinical indication."
                ),
            },
            exposure=_exp.provider_pattern_exposure(
                (sub["amount"] * max(0.0, 1.0 - prior.peer_mean / max(sr.shrunk_rate, 1e-9))).round(2).tolist()),
        ))
    return out


# ===========================================================================
# Upgrades of claim-header proxies (CLN-04-R01, CLN-04-R02, CLN-06-R03)
# ===========================================================================
#
# On a claim-header file these three controls run a proxy in
# fwa.engine.controls (a claim stands for an admission). On a multi-table
# file a claim is one visit and the proxies fire on routine follow-ups, so the
# unlocks below replace them with the line-level logic the catalogue describes.

_PATHWAY_DX = r"^(C|Z51|N18[56]|Z49|Z99[12]|O|Z3[3-9])"


@_register("cln_04_r01_repeat_within_policy_interval")
def cln_04_r01_repeat_within_policy_interval(ctx, control) -> list:
    lines = _lines(ctx)
    pol = _table(ctx, "repeat_interval_policy")
    if lines.empty or pol.empty or not {"activity_code", "min_interval_days"} <= set(pol.columns):
        return []
    interval = pd.Series(_num(pol, "min_interval_days").to_numpy(), index=_str(pol, "activity_code")).dropna()
    interval = interval[~interval.index.duplicated()]
    work = lines[lines["activity_code"].isin(set(interval.index)) & lines["sd"].notna()]
    if work.empty:
        return []
    cols = ["line_sk", "claim_sk", "member_sk", "provider_sk", "activity_code", "label", "amount", "sd",
            "diagnosis_primary"]
    work = work[cols + (["indicator"] if "indicator" in work.columns else [])]
    work = work.sort_values(["member_sk", "activity_code", "sd", "line_sk"])
    prev = work.groupby(["member_sk", "activity_code"]).shift(1)
    pairs = work.assign(prev_line=prev["line_sk"], prev_sd=prev["sd"], prev_claim=prev["claim_sk"],
                        prev_provider=prev["provider_sk"], prev_amount=prev["amount"])
    pairs = pairs[pairs["prev_line"].notna() & (pairs["prev_claim"] != pairs["claim_sk"])]
    pairs = pairs.assign(gap=(pairs["sd"] - pairs["prev_sd"]).dt.days,
                         limit=pairs["activity_code"].map(interval))
    # same-day copies on another claim are a duplicate-claim matter (PAY-01), not a repeat
    pairs = pairs[(pairs["gap"] >= 1) & (pairs["gap"] < pairs["limit"])]
    if pairs.empty:
        return []
    # declared exclusions: an accepted repeat indicator on the line; a failed or
    # inconclusive earlier result; oncology, dialysis and maternity pathways
    indicator = _str(pairs, "indicator").str.lower()
    pairs = pairs[~indicator.str.contains("repeat|91|clinical")]
    obs = _observations(ctx)
    if not obs.empty:
        failed = set(obs.loc[obs["val"].str.lower().str.contains(_FAILED, regex=True), "line_sk"]) - {""}
        pairs = pairs[~pairs["prev_line"].isin(failed)]
    pairs = pairs[~_code(pairs["diagnosis_primary"]).str.match(_PATHWAY_DX)]
    out = []
    for row in pairs.itertuples(index=False):
        out.append(_sig(
            ctx, control, subject_type="member", subject_id=row.member_sk,
            fact_key=f"repeat:{row.prev_line}:{row.line_sk}", claim_ids=[row.prev_claim, row.claim_sk],
            event_time=row.sd, period=period_bucket(row.sd), confidence=0.85,
            evidence={
                "member_sk": row.member_sk, "prior_claim_sk": row.prev_claim, "claim_sk": row.claim_sk,
                "activity_code": row.activity_code, "activity_description": row.label,
                "days_between": int(row.gap), "interval_threshold": int(row.limit),
                "diagnosis_code": row.diagnosis_primary, "provider_sk": row.provider_sk,
                "prior_provider_sk": row.prev_provider, "prior_service_date": _day(row.prev_sd),
                "plain_language": (
                    f"{row.label} was billed again {int(row.gap)} days after the previous one "
                    f"({plain_date(row.prev_sd)}); the repeat-interval policy expects at least "
                    f"{int(row.limit)} days between them."
                ),
                "what_the_reviewer_must_verify": (
                    "Whether a clinical reason (a failed test, a change in condition) justified the early repeat."
                ),
            },
            exposure=_exp.duplicate_exposure([float(row.prev_amount), float(row.amount)]),
        ))
    return out


_VISIT_FAMILIES = {"CONSULTATION", "TELEHEALTH", "EMERGENCY", "PROCEDURE", "IMAGING", "ADVANCED_IMAGING",
                   "LAB", "MENTAL_HEALTH", "PHYSIOTHERAPY"}


@_register("cln_04_r02_service_frequency_excess")
def cln_04_r02_service_frequency_excess(ctx, control) -> list:
    lines = _lines(ctx)
    if lines.empty or "ref_service_family" not in lines.columns:
        return []
    window = int(ctx.cfg("cln04_frequency_window_days"))
    floor = int(ctx.cfg("member_repeat_service_threshold"))
    percentile = float(ctx.cfg("cln04_frequency_peer_percentile"))
    fam = _str(lines, "ref_service_family")
    # visit-type services only: dispensing, supplies and inpatient components follow their own rules
    keep = fam.isin(_VISIT_FAMILIES) & ~lines["inpatient"].astype(bool) & lines["day"].notna()
    work = lines.loc[keep, ["member_sk", "claim_sk", "provider_sk", "day", "diagnosis_primary", "claim_gross"]]
    work = work.assign(family=fam[keep])
    if work.empty:
        return []
    # declared exclusions: oncology, dialysis and maternity pathways
    work = work[~_code(work["diagnosis_primary"]).str.match(_PATHWAY_DX)]
    visits = work.drop_duplicates(["member_sk", "family", "day"]).sort_values(["member_sk", "family", "day"])
    if visits.empty:
        return []
    # service days in the trailing window ending at each visit, per member and service family
    t = visits["day"].to_numpy(dtype="datetime64[D]").astype(np.int64)
    key = pd.factorize(visits["member_sk"].astype(str) + "|" + visits["family"])[0]
    key_start = np.r_[0, np.flatnonzero(np.diff(key)) + 1]
    group_first = np.repeat(key_start, np.diff(np.r_[key_start, len(key)]))
    # searchsorted on (key, t) composite so a window never crosses into another group
    composite = key.astype(np.int64) * 10_000_000 + t
    lo = np.searchsorted(composite, composite - (window - 1), side="left")
    lo = np.maximum(lo, group_first)
    visits = visits.assign(count=np.arange(len(t)) - lo + 1)
    best = visits.sort_values(["count", "day"]).groupby(["member_sk", "family"]).tail(1)
    cut = best.groupby("family")["count"].quantile(percentile, interpolation="higher")
    best = best.assign(cut=best["family"].map(cut))
    hits = best[(best["count"] > best["cut"]) & (best["count"] > floor)]
    out = []
    for row in hits.sort_values(["member_sk", "family"]).itertuples(index=False):
        lo_day = row.day - pd.Timedelta(days=window - 1)
        sub = work[(work["member_sk"] == row.member_sk) & (work["family"] == row.family)
                   & (work["day"] >= lo_day) & (work["day"] <= row.day)]
        claims = sorted(sub["claim_sk"].unique().tolist())
        per_claim = sub.drop_duplicates("claim_sk")["claim_gross"].astype(float).tolist()
        out.append(_sig(
            ctx, control, subject_type="member", subject_id=row.member_sk,
            fact_key=f"frequency:{row.member_sk}:{row.family}", claim_ids=claims, event_time=row.day,
            period=period_bucket(row.day), confidence=0.75, peer_level_used="all members, same service family",
            evidence={
                "member_sk": row.member_sk, "service_family": row.family, "claims_in_window": int(row.count),
                "window_days": window, "threshold": float(row.cut), "peer_percentile": percentile,
                "diagnoses_in_window": sorted(sub["diagnosis_primary"].dropna().astype(str).unique().tolist())[:8],
                "providers_in_window": sorted(sub["provider_sk"].astype(str).unique().tolist()),
                "total_gross_aed": round(float(sum(per_claim)), 2),
                "plain_language": (
                    f"This patient had {int(row.count)} {row.family.lower().replace('_', ' ')} service days within "
                    f"{window} days; {pct(percentile, 1)} of patients using that service have no more than "
                    f"{float(row.cut):.0f}."
                ),
                "what_the_reviewer_must_verify": (
                    "Whether a care pathway (chronic, rehabilitation) explains the frequency; age is not adjusted for."
                ),
            },
            exposure=_exp.provider_pattern_exposure(per_claim[int(row.cut):]),
        ))
    return out


@_register("cln_06_r03_line_signature_repeat")
def cln_06_r03_line_signature_repeat(ctx, control) -> list:
    lines = _lines(ctx)
    if lines.empty or "service_start_time" not in lines.columns:
        return []
    min_lines = int(ctx.cfg("cln06_signature_min_lines"))
    min_members = int(ctx.cfg("cln06_signature_min_members"))
    work = lines[~lines["inpatient"].astype(bool) & (lines["kind"] != "package")]
    if work.empty:
        return []
    clock = _dates(work["service_start_time"]).dt.strftime("%H:%M").fillna("")
    # declared exclusion: batch timestamps (midnight) carry no time information
    clock = clock.where(clock != "00:00", "")
    item = (work["activity_code"] + "x" + work["units_n"].round(2).astype(str) + "@"
            + work["gross_line"].round(2).astype(str) + "/" + clock).astype(object)
    work = work.assign(item=item)[["claim_sk", "item", "kind", "member_sk", "provider_sk", "sd", "claim_gross"]]
    work = work.sort_values(["claim_sk", "item"])
    per = work.groupby("claim_sk").agg(sig=("item", "|".join), n=("item", "size"),
                                       member_sk=("member_sk", "first"), provider_sk=("provider_sk", "first"),
                                       sd=("sd", "min"), gross=("claim_gross", "first"))
    # declared exclusion: standard packages - laboratory panels, collections and
    # dispensing alone are priced identically for everyone by design
    clinical = work[~work["kind"].isin(["lab", "collection", "drug", "device"])].groupby("claim_sk").size()
    per = per[(per["n"] >= min_lines) & per.index.isin(clinical.index)]
    if per.empty:
        return []
    size = per.groupby(["provider_sk", "sig"])["member_sk"].transform("nunique")
    hit = per[size >= min_members]
    out = []
    for (provider, sig), grp in hit.groupby(["provider_sk", "sig"], sort=True):
        claims = sorted(grp.index.tolist())
        first = grp["sd"].min()
        items = sig.split("|")
        out.append(_sig(
            ctx, control, subject_type="provider", subject_id=provider,
            fact_key=f"signature:{provider}:{sig}", claim_ids=claims, event_time=first,
            period=period_bucket(first), confidence=0.75,
            evidence={
                "provider_sk": provider, "signature": items, "distinct_members": int(grp["member_sk"].nunique()),
                "claim_sks": claims, "repeat_count": int(len(grp)),
                "gross_amount_aed": round(float(grp["gross"].iloc[0]), 2),
                "plain_language": (
                    f"{count_phrase(grp['member_sk'].nunique(), 'different patient')} have claims at this provider "
                    f"with exactly the same {len(items)} services, units, amounts and start times "
                    f"({aed(grp['gross'].iloc[0])} each), between {plain_date(first)} and "
                    f"{plain_date(grp['sd'].max())}."
                ),
                "what_the_reviewer_must_verify": (
                    "Request the appointment log and clinical notes for these visits; identical encounters for "
                    "different people suggest templated or fabricated records."
                ),
            },
            exposure=_exp.provider_pattern_exposure(grp["gross"].astype(float).tolist()[1:]),
        ))
    return out
