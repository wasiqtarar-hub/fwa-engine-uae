#!/usr/bin/env python3
"""Build a synthetic claim file that exercises the whole engine.

WHAT THIS IS FOR, AND WHAT IT IS NOT
====================================
The shipped ``data/claims.csv`` leaves 11 runnable controls silent and both
unsupervised models in shadow. Those are honest findings about that file, not
defects — but they mean large parts of the machinery are never demonstrated
end to end. This generator builds a file that does demonstrate them.

**A gate passed on a dataset built to pass it is a test of the gate, not
evidence about detection performance.** Every figure produced from this file is
a property of a generator whose parameters are in this module. Nothing here
transfers to real claims, and the file labels itself as synthetic in its own
name so the two cannot be confused.

What it is genuinely useful for: proving the promotion gate *can* return PASS
when its conditions are met, which is the other half of proving a gate works.
A gate that has only ever failed is indistinguishable from a gate that is
broken shut.

THE THREE THINGS THAT HAD TO BE ENGINEERED
==========================================

**1. Drift — stationary inputs.** The models fail the drift criterion on the
shipped file because the prior-history features are EXPANDING windows: a
provider's "claims so far" is near-zero at the start of the file and large at
the end, so the training and scoring periods genuinely have different input
distributions and PSI breaches. That is real drift, correctly detected.

The fix is not to relax the threshold. It is to give the population a
structure in which those features are stationary: every provider, agent and
member is active for a BOUNDED window, and those windows are staggered
uniformly across the file. At any date the cross-section then holds a stable
mixture of entities at every stage of their tenure, so "claims so far" has the
same distribution early and late. ``days_since_policy_start`` is drawn
independently of the service date for the same reason.

**2. Lift — a signature the composite cannot see.** The transparent composite
scores providers on seven *provider-mean* metrics: amount, amount per
inpatient day, approval ratio, coding-mismatch rate, pharmacy share,
readmission rate and utilisation velocity. The injected anomaly deliberately
touches NONE of them. It lives in four claim-level dimensions the composite
does not read — submission lag, insurers on the event, policy tenure at
service, and cashless status — and it appears on only a handful of a
provider's claims, so provider means stay at the peer median.

That is the case the design argues for in the abstract: a model earning
promotion by catching something the transparent baseline cannot. Here it is
constructed rather than discovered, which is the whole caveat.

**3. Calibration — a base rate that is not saturated.** On the shipped file
199 of 200 providers have at least one claim labelled fraud, so the
provider-level outcome is a constant, lift is unassessable and calibration is
hopeless. Fraud here is CONCENTRATED: roughly half the providers have no
labelled claim at all.

Labels are assigned in a second pass, from the model scores of a first pass
over the unlabelled file (``--calibrate``). That is legitimate only because
labels are held out of every feature path — adding them cannot change the
scores — and ``verify_demo_dataset.py`` asserts exactly that by comparing the
two passes' scores. Without it the ECE criterion measures the arbitrary shape
of an unsupervised score rather than anything about the model.

Usage::

    python tools/make_demo_dataset.py --out data/claims_demo_synthetic.csv
    python tools/make_demo_dataset.py --out data/claims_demo_synthetic.csv --calibrate
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

SEED = 20260920
#: FIVE WHOLE YEARS, and the reason is `admission_month`.
#:
#: That feature is the month of the year, 1–12. The split puts the first 60% of
#: claims in training and the rest in scoring, so unless each side spans a whole
#: number of years each sees a different mix of months, and the population-
#: stability monitor correctly reports that the input distribution moved. Over
#: five years the 60/40 split falls at three years and two years — both whole,
#: both covering every month evenly, and the feature stops drifting for a reason
#: that has nothing to do with the model.
#: Starting in 2023 keeps every service date inside the governed FX table,
#: which covers 2023 onward and refuses — correctly — to invent a rate for a
#: date it does not cover rather than falling back to a literal.
PERIOD_START = _dt.date(2023, 1, 1)
PERIOD_DAYS = 1826

# ---- population ------------------------------------------------------------
# 340 providers matters: the split is entity-isolated, so only about half are
# ever scored, and the lift criterion refuses to evaluate on fewer entities
# than the review capacity (100). 170 scored providers leaves real headroom.
#: 1400 providers, because only a slice of them is ever SCORED. The split is
#: temporal and entity-isolated at once: training takes early claims from half
#: the providers, scoring takes late claims from the other half. A provider
#: whose bounded window sits entirely in the early period is in neither. At 340
#: providers that left 73 scored entities, and the lift criterion refuses to
#: evaluate on fewer entities than the review capacity of 100 — so the gate
#: could not assess the one criterion the whole design turns on. At 700 it
#: could assess it, but only over 167 entities — and picking the top 100 of 167
#: is so blunt a cut that the model and the composite select largely the same
#: set and no ranking difference can show through. Around 330 scored providers
#: makes the top 100 a genuine selection.
N_PROVIDERS = 1400
#: Enough agents that each one's "claims so far" stays small and bounded. With
#: sixty, agents accumulated well over a hundred claims apiece and their
#: expanding counts were the second-worst drift in the file.
N_AGENTS = 760
PROVIDER_WINDOW = 260          # days a provider is active — bounded, so that
AGENT_WINDOW = 210             # "claims so far" cannot grow without limit
#: Members get a bounded window too, and for the same reason. Draw a member
#: uniformly from a fixed pool and a member's claims spread across the whole
#: file, so "this member's claims so far" climbs steadily and the member
#: history features drift — which is what happened at the first attempt, where
#: they became the four worst features in the file. A policy year is a real
#: bound, and it makes the feature stationary.
N_MEMBERS = 14000
MEMBER_WINDOW = 300
CLAIMS_PER_PROVIDER = (10, 19)

#: Providers in the shared-typology ring. A continuous anomaly intensity gives
#: a smooth gradient, and HDBSCAN finds no cluster in a gradient — which is why
#: the novel-cluster control stayed silent. A discrete group sharing one
#: distinctive residual profile is what a "candidate typology" actually means.
RING_PROVIDERS = 30

#: Documents generated per run. Must match the application, for the reason
#: given in ``measure_provider_probability``.
DOCUMENTS = 800

#: Diagnoses with a TIGHT length-of-stay distribution. On the shipped file
#: length of stay is uniform on [1, 30], which makes its MAD ≈ 7.5 and the
#: largest possible robust residual ≈ 1.4 — CLN-05-R01 tests against 3.0 and
#: therefore cannot fire at any threshold. A tight per-diagnosis distribution
#: is also simply more realistic: a cataract admission is not uniform on a
#: month.
#: (code, description, mean length of stay, mean amount, pharmacy share).
#: The pharmacy share is per-diagnosis and TIGHT for the same reason the length
#: of stay is: a feature derived from an expanding per-diagnosis median settles
#: after a handful of observations when the underlying spread is narrow, and
#: keeps drifting all file long when it is wide.
DIAGNOSES = [
    ("I10",   "Essential hypertension",          2.0, 18_000, 0.08),
    ("E11.9", "Type 2 diabetes complications",   3.5, 34_000, 0.22),
    ("Z51.1", "Chemotherapy session",            1.5, 82_000, 0.41),
    ("S72.0", "Femur fracture",                  9.0, 210_000, 0.06),
    ("J18.9", "Pneumonia, unspecified",          5.0, 46_000, 0.18),
    ("K35.8", "Acute appendicitis",              3.0, 95_000, 0.07),
    ("I21.9", "Acute myocardial infarction",     7.0, 310_000, 0.12),
    ("N18.5", "Chronic kidney disease stage 5",  4.5, 128_000, 0.26),
    ("H25.1", "Age-related cataract",            1.0, 54_000, 0.05),
    ("O80",   "Normal delivery",                 2.5, 62_000, 0.09),
    ("M17.1", "Primary gonarthrosis",            6.0, 188_000, 0.07),
    ("A09",   "Gastroenteritis",                 2.0, 21_000, 0.15),
    ("C50.9", "Malignant neoplasm of breast",    6.5, 265_000, 0.34),
    ("G40.9", "Epilepsy, unspecified",           3.0, 39_000, 0.24),
    ("L03.1", "Cellulitis of limb",              4.0, 33_000, 0.19),
    ("R07.4", "Chest pain, unspecified",         1.5, 26_000, 0.1),
    ("K80.2", "Calculus of gallbladder",         3.5, 112_000, 0.08),
    ("J45.9", "Asthma, unspecified",             2.5, 28_000, 0.21),
    ("N39.0", "Urinary tract infection",         2.5, 23_000, 0.17),
    ("I50.9", "Heart failure, unspecified",      6.0, 97_000, 0.23),
    ("D64.9", "Anaemia, unspecified",            2.0, 24_000, 0.2),
    ("T14.9", "Injury, unspecified",             2.0, 31_000, 0.11),
    ("B34.9", "Viral infection, unspecified",    2.0, 19_000, 0.16),
    ("Z38.0", "Liveborn infant",                 3.0, 44_000, 0.13),
]
TPAS = ["MD India", "Good Health TPA", "Anytime Health", "FHPL (Family Health Plan)",
        "Paramount Health", "Vidal Health", "Heritage Health", "Medi Assist",
        "Raksha Health", "Star Health"]
POLICY_TYPES = ["individual", "family", "group_corporate"]

#: How many of a provider's claims carry the injected signature, at full
#: intensity. Small on purpose: the signature has to be invisible in the
#: provider's MEANS, which is what the transparent composite reads.
ANOMALOUS_CLAIMS_AT_FULL_INTENSITY = 3

FRAUD_TYPES = ["bill_inflation", "phantom_billing", "unnecessary_procedure",
               "identity_misuse", "coordinated_ring", "upcoding"]
GROUND_TRUTH_SOURCES = ["pattern_detection", "expert_review", "rule_engine"]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _d(offset: float) -> _dt.date:
    return PERIOD_START + _dt.timedelta(days=int(offset))


def _fmt(d: _dt.date) -> str:
    return d.strftime("%d/%m/%Y")


def _fx_compensation() -> dict[int, float]:
    """Per-year multipliers that hold the AED value of a claim steady.

    Amounts are written in the source currency and converted at the rate in
    force on the service date — correctly, because converting a 2023 exposure
    at a 2027 rate would make a historical figure irreproducible. But the
    governed table steps the rate down about 3% across the period, and the
    split puts the earlier rates in training and the later one in scoring. The
    result is a systematic shift in every amount-derived feature, and the
    population-stability monitor reports it as drift — which it is, only it is
    drift in the exchange rate rather than in anybody's billing.

    Generating source amounts that inflate exactly as the currency weakens
    leaves the AED value stationary, so the monitor measures what it is for.
    Read from the governed table rather than copied out of it, so the two
    cannot fall out of step.
    """
    import yaml
    raw = yaml.safe_load((ROOT / "config" / "fx.yaml").read_text(encoding="utf-8"))
    rates = {}
    for entry in raw.get("rates", []):
        if entry.get("from_currency") != "INR":
            continue
        start = _dt.date.fromisoformat(str(entry["valid_from"]))
        end = _dt.date.fromisoformat(str(entry["valid_to"]))
        for year in range(start.year, min(end.year, 2100) + 1):
            rates.setdefault(year, float(entry["rate"]))
    base = rates[PERIOD_START.year]
    return {y: base / r for y, r in rates.items()}


def _tenure(rng: np.random.Generator, n: int) -> np.ndarray:
    """Policy tenure at service, drawn INDEPENDENTLY of the service date.

    This is what keeps ``days_since_policy_start`` stationary. Tie tenure to
    the calendar — everyone's policy starting on day zero — and the feature
    climbs steadily through the file, the training and scoring periods stop
    being comparable, and PSI breaches for a reason that has nothing to do
    with the model.
    """
    return np.clip(rng.normal(900, 320, n), 40, 2600).round().astype(int)


# ---------------------------------------------------------------------------
# the base population
# ---------------------------------------------------------------------------


def build_base(rng: np.random.Generator) -> pd.DataFrame:
    dx_codes = [d[0] for d in DIAGNOSES]
    dx_desc = {d[0]: d[1] for d in DIAGNOSES}
    dx_los = {d[0]: d[2] for d in DIAGNOSES}
    dx_amt = {d[0]: d[3] for d in DIAGNOSES}
    dx_pharm = {d[0]: d[4] for d in DIAGNOSES}

    # Staggered, bounded activity windows — the whole drift fix in two lines.
    # Windows may START BEFORE the file opens and END AFTER it closes, with
    # claims outside the period discarded. Without that overhang the first and
    # last months hold only entities at the very beginning or very end of their
    # tenure, and "claims so far" drifts at exactly the two edges the split
    # cares about.
    prov_start = rng.uniform(-PROVIDER_WINDOW, PERIOD_DAYS, N_PROVIDERS)
    prov_intensity = rng.uniform(0, 1, N_PROVIDERS)
    agent_start = rng.uniform(-AGENT_WINDOW, PERIOD_DAYS, N_AGENTS)

    # The shared-typology ring: a discrete group, not a point on a gradient.
    ring = set(rng.choice(N_PROVIDERS, size=RING_PROVIDERS, replace=False).tolist())
    fx = _fx_compensation()

    rows = []
    for p in range(N_PROVIDERS):
        provider = f"H{p + 1:04d}"
        n_claims = int(rng.integers(*CLAIMS_PER_PROVIDER))
        # Each provider draws from a handful of agents whose active windows
        # overlap its own, so agent tenure is staggered for free.
        near = np.argsort(np.abs(agent_start - prov_start[p]))[:6]
        blacklisted = rng.random() < 0.06
        # A provider-level price level, INDEPENDENT of the injected anomaly, so
        # that the transparent composite has something real of its own to find.
        price_level = float(rng.normal(1.0, 0.08))

        in_ring = p in ring
        # Admissions fall on CLINIC DAYS, a handful of dates per provider,
        # rather than on any day of the window. Two reasons. It is how a small
        # facility actually runs, and `provider_same_day_admissions` is a
        # model feature: spread admissions uniformly and almost every provider
        # scores exactly 1, the PSI quantile bins collapse onto that single
        # value, and any burst elsewhere in the file registers as a population
        # shift. Two or three admissions on a clinic day is an ordinary number,
        # and an ordinary number is what gives the feature a distribution.
        clinic_days = prov_start[p] + rng.choice(
            PROVIDER_WINDOW, size=max(2, n_claims // 2), replace=False)
        for _ in range(n_claims):
            offset = float(clinic_days[int(rng.integers(0, len(clinic_days)))])
            if not 0 <= offset < PERIOD_DAYS:
                continue                      # window overhang, outside the file
            dx = dx_codes[int(rng.integers(0, len(dx_codes)))]
            los = int(np.clip(round(rng.normal(dx_los[dx], 0.7)), 1, 45))
            admit = _d(offset)
            discharge = admit + _dt.timedelta(days=los)
            lag = int(np.clip(rng.normal(14, 3.0), 1, 60))
            gross = float(dx_amt[dx] * price_level * rng.lognormal(0, 0.12)
                          * fx.get(admit.year, 1.0))
            ratio = float(np.clip(rng.beta(60, 7), 0.45, 1.0))
            pharm = float(np.clip(rng.normal(dx_pharm[dx], 0.015), 0.02, 0.85))
            if in_ring:
                # One coherent profile shared across the ring: pharmacy-heavy
                # bills, approval below par, stays shorter than the diagnosis
                # warrants. The shift is MODEST and CONSISTENT, and that
                # combination is the point. The transparent composite scores a
                # provider on the mean of twenty-odd claims, so a small
                # consistent shift shows up against a standard error shrunk by
                # the square root of that count. Per claim it is barely
                # remarkable, which is exactly what an isolation forest reads.
                # The two therefore disagree about this group on purpose: the
                # composite is right about it and the models are not.
                pharm = float(np.clip(pharm + rng.normal(0.11, 0.02), 0.02, 0.9))
                ratio = float(np.clip(ratio - rng.normal(0.09, 0.02), 0.2, 1.0))
                los = max(1, int(round(los * 0.78)))
                discharge = admit + _dt.timedelta(days=los)
            rows.append({
                "patient_id": None,                     # filled below
                "hospital_id": provider,
                "diagnosis_primary": dx,
                "diagnosis_description": dx_desc[dx],
                "date_of_admission": admit,
                "date_of_discharge": discharge,
                "date_of_claim": discharge + _dt.timedelta(days=lag),
                "length_of_stay_days": los,
                "claim_amount_requested_inr": int(round(gross)),
                "claim_amount_approved_inr": int(round(gross * ratio)),
                "policy_type": POLICY_TYPES[int(rng.integers(0, 3))],
                "is_cashless": bool(rng.random() < 0.62),
                "tpa": TPAS[int(rng.integers(0, len(TPAS)))],
                "provider_blacklist_flag": blacklisted,
                "previous_fraud_on_policy": bool(rng.random() < 0.05),
                "days_since_policy_start": 0,           # filled below
                "num_insurers_same_event": 1,
                "icd_code_matches_procedure": bool(rng.random() < 0.93),
                "discharge_readmit_gap_days": int(np.clip(rng.normal(180, 45), 35, 400)),
                "pharmacy_bill_ratio": round(pharm, 2),
                "agent_id": f"A{near[int(rng.integers(0, len(near)))] + 1:03d}",
                "_provider_index": p,
                "_intensity": prov_intensity[p],
                "_stratum": "ring" if in_ring else "base",
                "_anomalous": False,
                "_offset": offset,
            })

    df = pd.DataFrame(rows)
    df["days_since_policy_start"] = _tenure(rng, len(df))
    df["patient_id"] = _assign_members(df["_offset"].to_numpy(), rng)
    return df.drop(columns=["_offset"])


def _assign_members(offsets: np.ndarray, rng: np.random.Generator) -> list[str]:
    """Give each claim a member whose policy window contains its service date.

    Members are staggered exactly as providers and agents are, so repeat visits
    stay local in time instead of accumulating across five years. The claim is
    matched to the eligible slice by binary search rather than by filtering per
    claim, which matters at this size.
    """
    starts = np.sort(rng.uniform(-MEMBER_WINDOW, PERIOD_DAYS, N_MEMBERS))
    out: list[str] = []
    for o in offsets:
        lo = np.searchsorted(starts, o - MEMBER_WINDOW, side="left")
        hi = np.searchsorted(starts, o, side="right")
        if hi <= lo:                       # no window covers this date
            out.append(f"P{int(rng.integers(0, N_MEMBERS)):06d}")
            continue
        out.append(f"P{int(rng.integers(lo, hi)):06d}")
    return out


# ---------------------------------------------------------------------------
# the injected signature the models are meant to find
# ---------------------------------------------------------------------------


def inject_anomaly(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Displace a few claims per provider, in dimensions the composite ignores.

    The four dimensions are submission lag, insurers on the event, policy
    tenure at service and cashless status. None of them appears in the
    transparent composite's seven provider-mean metrics, and each displacement
    is individually unremarkable — it is the COMBINATION that is rare, which is
    what an isolation forest and a local outlier factor are good at and a
    univariate peer residual is not.
    """
    df = df.copy()
    for p, sub in df.groupby("_provider_index", sort=False):
        a = float(sub["_intensity"].iloc[0])
        n = int(round(ANOMALOUS_CLAIMS_AT_FULL_INTENSITY * a))
        if n <= 0:
            continue
        idx = rng.choice(sub.index.values, size=min(n, len(sub)), replace=False)
        # QUANTISED, not continuous. Spreading the signature along a smooth
        # gradient leaves its far end thinly populated, and a point with no
        # neighbours gets a local-outlier-factor score several times anyone
        # else's. Min-max rescaling then crushes every other provider toward
        # zero and the calibration criterion measures that one point's
        # loneliness rather than the model. Five discrete profiles put the
        # same claims in five populated pockets instead — which is also closer
        # to how a billing pattern actually recurs.
        level = int(np.clip(round(a * 4), 0, 4))
        for i in idx:
            lag = int(46 + 11 * level + rng.uniform(0, 5))
            discharge = df.at[i, "date_of_discharge"]
            df.at[i, "date_of_claim"] = discharge + _dt.timedelta(days=lag)
            df.at[i, "num_insurers_same_event"] = 3 if level < 3 else 4
            df.at[i, "days_since_policy_start"] = int(52 - 9 * level + rng.uniform(0, 7))
            df.at[i, "is_cashless"] = False
            df.at[i, "_anomalous"] = True
    return df


# ---------------------------------------------------------------------------
# strata: one per control that the shipped file leaves silent
# ---------------------------------------------------------------------------


def build_strata(base: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Targeted rows, each constructed for one named control.

    Every stratum is tagged in ``_stratum`` so that ``verify_demo_dataset.py``
    can report which construction fed which control, and so that nobody has to
    reverse-engineer intent from the data later.
    """
    out: list[pd.DataFrame] = []
    template = base.drop(columns=["_provider_index", "_intensity", "_anomalous", "_stratum"])

    def sample(n: int) -> pd.DataFrame:
        return template.sample(n, random_state=int(rng.integers(0, 2**31))).reset_index(drop=True)

    def window_sample(n: int, start_day: int, width: int) -> pd.DataFrame:
        """Rows whose admission falls inside one bounded window.

        Used for the strata that give a single entity a large block of claims.
        Those entities still have to look like every other entity to the
        expanding-window features, and an entity active across the whole file
        does not.
        """
        lo, hi = _d(start_day), _d(start_day + width)
        pool = template[(template["date_of_admission"] >= lo)
                        & (template["date_of_admission"] < hi)]
        if len(pool) < n:                       # widen rather than fail
            pool = template
        return pool.sample(n, replace=len(pool) < n,
                           random_state=int(rng.integers(0, 2**31))).reset_index(drop=True)

    # ---- PAY-06-R02: approved exceeds gross -------------------------------
    s = sample(30)
    s["claim_amount_approved_inr"] = (s["claim_amount_requested_inr"] * 1.18).round().astype(int)
    out.append(s.assign(_stratum="PAY-06-R02 amount arithmetic"))

    # ---- PAY-01-R01: exact duplicates -------------------------------------
    # Identical payer, member, provider, diagnosis, admission, discharge and
    # gross — the tolerance is 0.0, so "identical" means identical.
    s = sample(24)
    out.append(pd.concat([s, s]).assign(_stratum="PAY-01-R01 exact duplicate"))

    # ---- PAY-01-R02: near duplicates --------------------------------------
    s = sample(28)
    t = s.copy()
    shift = rng.integers(1, 7, len(t))
    t["date_of_admission"] = [d + _dt.timedelta(days=int(k)) for d, k in zip(t["date_of_admission"], shift)]
    t["date_of_discharge"] = [d + _dt.timedelta(days=int(k)) for d, k in zip(t["date_of_discharge"], shift)]
    t["claim_amount_requested_inr"] = (t["claim_amount_requested_inr"] * 1.02).round().astype(int)
    t["claim_amount_approved_inr"] = (t["claim_amount_approved_inr"] * 1.02).round().astype(int)
    out.append(pd.concat([s, t]).assign(_stratum="PAY-01-R02 near duplicate"))

    # ---- PAY-01-R03 + CLN-05-R03: same-episode, same-provider repeats ------
    # One construction serves both: a second admission at the SAME provider
    # with the SAME principal diagnosis, 4–25 days after the first discharge.
    # That is inside the 30-day episode window (PAY-01-R03) and is also a
    # discharge-to-readmission gap inside the same window (CLN-05-R03).
    s = sample(45)
    t = s.copy()
    gap = rng.integers(4, 26, len(t))
    t["date_of_admission"] = [d + _dt.timedelta(days=int(g)) for d, g in zip(s["date_of_discharge"], gap)]
    t["date_of_discharge"] = [a + _dt.timedelta(days=int(l)) for a, l in zip(t["date_of_admission"], t["length_of_stay_days"])]
    t["date_of_claim"] = [d + _dt.timedelta(days=12) for d in t["date_of_discharge"]]
    t["discharge_readmit_gap_days"] = gap
    s = s.copy()
    s["discharge_readmit_gap_days"] = gap
    out.append(pd.concat([s, t]).assign(_stratum="PAY-01-R03 / CLN-05-R03 episode repeat"))

    # ---- CLN-06-R03 + DOC-02-R01: cloned encounters across members --------
    # A (diagnosis, length of stay, gross amount) triple recurring across
    # DISTINCT members at one provider. Because the synthetic note is rendered
    # from exactly those fields, the clones also produce near-identical
    # documents, which is what the cross-patient near-duplicate control reads.
    # Few templates, many copies: the corpus samples only ~800 claims, so a
    # thinly-spread clone set would never land two copies of one template in
    # the sample.
    # Five templates, not six, and 180 copies each rather than 70. The corpus
    # samples only 800 of the file's claims, so what matters is not how many
    # clones exist but how many copies of ONE template survive the sample —
    # near-duplicate detection needs a pair. Thin the templates and thicken
    # each one.
    clones = []
    for k in range(5):
        seed_row = template.sample(1, random_state=1000 + k).iloc[0].to_dict()
        for _ in range(180):
            r = dict(seed_row)
            r["patient_id"] = f"P9{k:01d}{int(rng.integers(0, 10000)):04d}"
            clones.append(r)
    out.append(pd.DataFrame(clones).assign(_stratum="CLN-06-R03 / DOC-02-R01 cloned encounter"))

    # ---- CLN-05-R01: length-of-stay outliers ------------------------------
    # Against a tight per-diagnosis distribution these are genuine robust
    # residuals, not an artefact of a flat prior.
    # Against a per-diagnosis spread of about 0.9 days the robust scale is
    # ≈0.89, so four days out is already a residual above four — comfortably
    # past the threshold of 3.0 without planting points that sit alone in the
    # feature space and distort every model score around them.
    s = sample(40)
    s["length_of_stay_days"] = (s["length_of_stay_days"] + rng.integers(4, 9, len(s))).astype(int)
    s["date_of_discharge"] = [a + _dt.timedelta(days=int(l)) for a, l in zip(s["date_of_admission"], s["length_of_stay_days"])]
    s["date_of_claim"] = [d + _dt.timedelta(days=14) for d in s["date_of_discharge"]]
    out.append(s.assign(_stratum="CLN-05-R01 length-of-stay outlier"))

    # ---- CLN-07-R02: concurrent occupancy beyond capacity -----------------
    # The effective limit is capacity × tolerance = 50 concurrent admissions,
    # so the burst has to be large and genuinely overlapping.
    burst = []
    # Four smaller bursts spread across the file rather than two large ones in
    # the training period. `provider_same_day_admissions` is a feature, and a
    # spike that exists only on one side of the split is drift.
    # EIGHT burst providers, not two, and each burst spread over eight days
    # rather than three.
    #
    # Two reasons, both learned the hard way. Which side of the split a burst
    # lands on is decided by the PROVIDER's hash bucket, not by its date — the
    # split is entity-isolated — so placing bursts by calendar controls
    # nothing, and a handful of them can end up entirely on one side. Eight
    # providers put some in each group. And `provider_same_day_admissions` is
    # itself a feature: sixty admissions crammed into three days gives a value
    # twenty times anything else in the file, so whichever side holds it
    # carries a distribution the other does not. Spread over eight days the
    # peak occupancy still clears the limit of fifty — every stay is longer
    # than the eleven-day admission window, so nobody has left by the last day
    # — while the daily count stays within sight of the rest of the population.
    for k, (provider, day0) in enumerate(
            [("H0001", 180), ("H0002", 400), ("H0003", 620), ("H0004", 840),
             ("H0005", 1060), ("H0006", 1280), ("H0007", 1500), ("H0008", 1700)]):
        for j in range(62):
            admit = _d(day0 + rng.integers(0, 11))
            los = int(rng.integers(13, 19))
            dx = DIAGNOSES[int(rng.integers(0, len(DIAGNOSES)))]
            burst.append({
                "patient_id": f"P8{k}{j:04d}", "hospital_id": provider,
                "diagnosis_primary": dx[0], "diagnosis_description": dx[1],
                "date_of_admission": admit,
                "date_of_discharge": admit + _dt.timedelta(days=los),
                "date_of_claim": admit + _dt.timedelta(days=los + 10),
                "length_of_stay_days": los,
                "claim_amount_requested_inr": int(dx[3] * rng.uniform(0.9, 1.2)),
                "claim_amount_approved_inr": int(dx[3] * rng.uniform(0.75, 0.95)),
                "policy_type": "group_corporate", "is_cashless": True,
                "tpa": TPAS[int(rng.integers(0, len(TPAS)))],
                "provider_blacklist_flag": False, "previous_fraud_on_policy": False,
                "days_since_policy_start": int(rng.uniform(200, 2000)),
                "num_insurers_same_event": 1, "icd_code_matches_procedure": True,
                "discharge_readmit_gap_days": int(rng.uniform(40, 400)),
                "pharmacy_bill_ratio": round(float(rng.uniform(0.05, 0.3)), 2),
                "agent_id": f"A{int(rng.integers(1, N_AGENTS + 1)):03d}",
            })
    out.append(pd.DataFrame(burst).assign(_stratum="CLN-07-R02 capacity breach"))

    # ---- NET-04-R02: an originator funnelling into few providers ----------
    # The control looks for a LOW distinct-providers-per-claim ratio relative
    # to the spread across agents, so this agent needs many claims and almost
    # no provider diversity.
    #
    # Confined to a bounded window like every other agent. Spread the same 150
    # claims across five years and this one agent's "claims so far" climbs from
    # zero to a hundred and fifty over the file — a drift breach manufactured
    # by the fixture built to exercise a different control.
    s = window_sample(150, 900, AGENT_WINDOW)
    s["agent_id"] = "A199"
    s["hospital_id"] = rng.choice(["H0301", "H0302", "H0303"], len(s))
    out.append(s.assign(_stratum="NET-04-R02 originator concentration"))

    # ---- NET-04-R03: early-tenure, above-median-value clustering ----------
    s = window_sample(120, 1250, AGENT_WINDOW)
    s["agent_id"] = np.where(rng.random(len(s)) < 0.5, "A197", "A198")
    s["days_since_policy_start"] = rng.integers(5, 85, len(s))
    # The control tests "above the diagnosis peer median", so a 40% uplift is
    # already decisive. Doubling would only add amount outliers.
    s["claim_amount_requested_inr"] = (s["claim_amount_requested_inr"] * 1.4).round().astype(int)
    s["claim_amount_approved_inr"] = (s["claim_amount_approved_inr"] * 1.4).round().astype(int)
    out.append(s.assign(_stratum="NET-04-R03 early-tenure cluster"))

    # ---- ANL-01-R02 + CLN-07-R04: providers with a history and a step -----
    # Both are CUSUM/EWMA change-point controls and need at least six monthly
    # periods of a provider's own history before a change may be DECLARED. A
    # bounded 260-day window gives about eight months at one or two claims
    # each, which is too sparse to establish a baseline, let alone a departure
    # from it. These providers run for two years and step partway through:
    # throughput up (CLN-07-R04), mean value and approval ratio down
    # (ANL-01-R02).
    steppers = []
    for k in range(26):
        provider = f"H1{k:03d}"
        base_row = template.sample(1, random_state=7000 + k).iloc[0].to_dict()
        start = int(rng.integers(0, PERIOD_DAYS - 430))
        for month in range(14):
            after = month >= 7
            n = int(rng.integers(6, 9)) if after else int(rng.integers(2, 4))
            # Distinct days within the month. Drawing them independently makes
            # same-day collisions commoner after the step than before it,
            # which shows up as drift in `provider_same_day_admissions` rather
            # than as the throughput change these providers exist to carry.
            days = rng.choice(29, size=n, replace=False)
            for offset_in_month in days:
                day = start + month * 30 + int(offset_in_month)
                if day >= PERIOD_DAYS:
                    continue
                r = dict(base_row)
                admit = _d(day)
                los = int(np.clip(rng.normal(3.5, 0.8), 1, 30))
                gross = float(base_row["claim_amount_requested_inr"]) * (
                    rng.normal(1.55, 0.06) if after else rng.normal(1.0, 0.06))
                ratio = float(np.clip(rng.normal(0.62 if after else 0.93, 0.03), 0.3, 1.0))
                r.update({
                    "patient_id": f"P7{k:02d}{int(rng.integers(0, 10000)):04d}",
                    "hospital_id": provider,
                    "date_of_admission": admit,
                    "date_of_discharge": admit + _dt.timedelta(days=los),
                    "date_of_claim": admit + _dt.timedelta(days=los + 12),
                    "length_of_stay_days": los,
                    "claim_amount_requested_inr": int(round(gross)),
                    "claim_amount_approved_inr": int(round(gross * ratio)),
                    "days_since_policy_start": int(np.clip(rng.normal(900, 320), 40, 2600)),
                })
                steppers.append(r)
    out.append(pd.DataFrame(steppers).assign(_stratum="ANL-01-R02 / CLN-07-R04 step change"))

    frame = pd.concat(out, ignore_index=True)
    frame["_provider_index"] = -1
    frame["_intensity"] = 0.0
    frame["_anomalous"] = False
    return frame


# ---------------------------------------------------------------------------
# assembly
# ---------------------------------------------------------------------------


def assemble(rng: np.random.Generator) -> pd.DataFrame:
    base = inject_anomaly(build_base(rng), rng)
    strata = build_strata(base, rng)
    df = pd.concat([base, strata], ignore_index=True)
    df = df.sort_values("date_of_admission", kind="stable").reset_index(drop=True)
    df.insert(0, "claim_id", [f"C{i + 1:07d}" for i in range(len(df))])
    return df


def label(df: pd.DataFrame, provider_probability: dict[str, float] | None,
          rng: np.random.Generator,
          secondary: dict[str, float] | None = None) -> pd.DataFrame:
    """Attach the held-out label columns.

    ``provider_probability`` maps a provider to the probability that it is a
    fraud provider at all. Pass ``None`` for the first, unlabelled pass; pass
    the measured model scores for the second. A provider that comes up positive
    has its injected claims marked, or a random claim if it has none — the
    outcome the gate reads is ``max(fraud_label)`` over a provider's claims, so
    what matters is whether the provider has ANY.
    """
    df = df.copy()
    df["fraud_label"] = 0

    if provider_probability:
        # Providers the split never scored are labelled at the same rate. The
        # gate never sees them — it reads only the scored set — but a file in
        # which fraud appears exclusively among scored providers would be a
        # strange artefact for anyone who opened it.
        rate = float(np.mean(list(provider_probability.values())))
        unscored = [p for p in df["hospital_id"].unique() if p not in provider_probability]
        extra = rng.choice(unscored, size=int(round(rate * len(unscored))), replace=False) \
            if unscored else []
        for provider in list(_positive_providers(provider_probability, rng, secondary)) + list(extra):
            sub = df.index[df["hospital_id"] == provider]
            anomalous = df.index[(df["hospital_id"] == provider)
                                 & df["_anomalous"].fillna(False).astype(bool)]
            pool = anomalous if len(anomalous) else sub
            if not len(pool):
                continue
            n = max(1, int(round(len(pool) * 0.6)))
            df.loc[rng.choice(pool.values, size=min(n, len(pool)), replace=False), "fraud_label"] = 1

    n = len(df)
    df["fraud_type"] = np.where(
        df["fraud_label"] == 1,
        rng.choice(FRAUD_TYPES, n), "legitimate")
    df["fraud_confidence"] = np.where(
        df["fraud_label"] == 1, rng.uniform(0.70, 0.99, n), rng.uniform(0.80, 0.99, n)).round(4)
    df["ground_truth_source"] = rng.choice(GROUND_TRUTH_SOURCES, n)
    return df


def _positive_providers(probability: dict[str, float],
                        rng: np.random.Generator,
                        secondary: dict[str, float] | None = None) -> list[str]:
    """Choose which providers are positive, by STRATIFIED count rather than coin flip.

    Drawing each provider independently from Bernoulli(p) is unbiased but
    noisy, and the calibration criterion is measured on about thirty providers
    per bin — where the standard error of a rate near 0.3 is about 0.08, which
    is most of the error budget spent on nothing but sampling.

    Sorting by probability and labelling exactly the expected NUMBER within
    each decile removes that noise. It is not a trick: matching the observed
    rate to the stated confidence is the definition of calibration, and this
    builds a population that is calibrated rather than one that is calibrated
    on average across resamples nobody will draw.
    """
    series = pd.Series(probability).sort_values()
    if series.empty:
        return []
    chosen: list[str] = []
    bins = np.array_split(series.index.to_numpy(), 10)
    for group in bins:
        if not len(group):
            continue
        target = int(round(float(series.loc[group].mean()) * len(group)))
        if target <= 0:
            continue
        # The bin RATE is fixed by the model being calibrated. Which members
        # of the bin are positive is free, and calibration cannot see it — an
        # expected calibration error depends on per-bin rates alone. So the
        # choice within a bin goes to the OTHER model's ranking, which costs
        # the calibrated model nothing and lets the second model's lift be
        # assessed against labels it has some purchase on. Without this the
        # second model is being ranked against labels drawn purely from its
        # rival, and its lift measures that rivalry rather than the model.
        inner = (pd.Series(secondary).reindex(group).fillna(-np.inf)
                 if secondary else series.loc[group])
        ranked = inner.sort_values(ascending=False).index.to_numpy()
        chosen.extend(ranked[:target].tolist())
    return chosen


COLUMNS = [
    "claim_id", "patient_id", "hospital_id", "diagnosis_primary", "diagnosis_description",
    "date_of_admission", "date_of_discharge", "date_of_claim", "length_of_stay_days",
    "claim_amount_requested_inr", "claim_amount_approved_inr", "policy_type", "is_cashless",
    "tpa", "provider_blacklist_flag", "previous_fraud_on_policy", "days_since_policy_start",
    "num_insurers_same_event", "icd_code_matches_procedure", "discharge_readmit_gap_days",
    "pharmacy_bill_ratio", "agent_id", "fraud_label", "fraud_type", "fraud_confidence",
    "ground_truth_source",
]


def write(df: pd.DataFrame, path: Path) -> None:
    out = df.copy()
    for col in ("date_of_admission", "date_of_discharge", "date_of_claim"):
        out[col] = [_fmt(d) for d in out[col]]
    for col in ("is_cashless", "provider_blacklist_flag", "previous_fraud_on_policy",
                "icd_code_matches_procedure"):
        out[col] = out[col].astype(bool).map({True: "TRUE", False: "FALSE"})
    path.parent.mkdir(parents=True, exist_ok=True)
    out[COLUMNS].to_csv(path, index=False)

    # The stratum map travels with the file. Which construction was intended to
    # feed which control is not recoverable from the CSV, and a demonstration
    # dataset whose intent has to be guessed at is worse than none.
    # Per-claim stratum map, so a diagnostic can ask "which construction
    # produced this score?" without reverse-engineering it from the values.
    df[["claim_id", "_stratum"]].rename(columns={"_stratum": "stratum"}).to_csv(
        path.with_suffix(".strata.csv"), index=False)

    sidecar = path.with_suffix(".strata.json")
    sidecar.write_text(json.dumps({
        "generator": "tools/make_demo_dataset.py",
        "seed": SEED,
        "rows": int(len(out)),
        "providers": int(out["hospital_id"].nunique()),
        "claim_level_fraud_rate": round(float(df["fraud_label"].mean()), 4),
        "strata": {k: int(v) for k, v in df["_stratum"].value_counts().items()},
        "warning": "SYNTHETIC. Generated to exercise the engine. No figure computed "
                   "from this file is evidence about detection performance.",
    }, indent=2), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="data/claims_demo_synthetic.csv")
    ap.add_argument("--calibrate-to", default="local_outlier_factor",
                    choices=["local_outlier_factor", "isolation_forest"],
                    help="Which model's score the outcome rate is matched to. Only one can "
                         "be satisfied; see measure_provider_probability for why.")
    ap.add_argument("--calibrate", action="store_true",
                    help="Second pass: run the engine on the unlabelled file, then set each "
                         "provider's label probability from the model scores it produced.")
    args = ap.parse_args()

    path = (ROOT / args.out) if not Path(args.out).is_absolute() else Path(args.out)
    rng = np.random.default_rng(SEED)
    df = assemble(rng)

    probability: dict[str, float] | None = None
    secondary: dict[str, float] | None = None
    if args.calibrate:
        write(label(df, None, np.random.default_rng(SEED + 1)), path)
        probability, secondary = measure_provider_probability(path, args.calibrate_to)

    write(label(df, probability, np.random.default_rng(SEED + 1), secondary), path)
    print(f"wrote {len(df):,} rows, {df['hospital_id'].nunique()} providers → {path}")
    print(f"strata: {json.dumps({k: int(v) for k, v in df['_stratum'].value_counts().items()}, indent=1)}")
    return 0


def measure_provider_probability(path: Path, target: str) -> tuple[dict[str, float], dict[str, float]]:
    """Run the engine once and turn each provider's model scores into a rate.

    ONE model, not an average of both, and that is forced rather than chosen.

    The calibration criterion min-max rescales the model score and compares it,
    bin by bin, against the observed outcome rate. Summing those bins, the
    error can never be smaller than the distance between the mean rescaled
    score and the overall base rate. Both models are scored against the SAME
    outcome series, so one base rate has to serve both, and the best it can do
    is sit midway between the two means — leaving each model at least half
    their separation.

    On this data those means are about 0.54 and 0.05. The isolation forest's
    score is bounded and roughly symmetric, so rescaling leaves it near the
    middle; the local outlier factor is a density ratio with one point three
    times beyond the 99th percentile, and min-max keys the whole scale off
    that point. Half the gap is ≈0.25, against a configured maximum of 0.15.
    No dataset closes that, because it is a property of the two score shapes
    rather than of the labels.

    So the base rate calibrates ONE of them. The default is the local outlier
    factor, because calibrating to the isolation forest instead would require
    54% of providers to carry a confirmed fraudulent claim — a figure that
    would discredit the file it appeared in.
    """
    import sys
    sys.path.insert(0, str(ROOT / "src"))
    from fwa.config import load_config
    from fwa.pipeline import run_pipeline

    # 800 documents, matching what the application runs. The corpus is
    # generated BEFORE the models are fitted and draws from the same seeded
    # stream, so asking for a different number here changes the model scores —
    # and a calibration pass measured against scores the gate will never see
    # calibrates nothing.
    result = run_pipeline(path, config=load_config(), generate_documents=DOCUMENTS,
                          document_dir=ROOT / "data" / "synthetic_documents", verbose=False)
    scores = result.models.provider_scores()
    series = scores.get(target)
    if series is None or series.empty:
        raise SystemExit(f"no scores for {target!r}; available: {sorted(scores)}")
    lo, hi = series.min(), series.max()
    p = (series - lo) / max(hi - lo, 1e-9)
    other = {n: v for n, v in scores.items() if n != target}
    second = next(iter(other.values()), None)
    return ({str(k): float(v) for k, v in p.items()},
            {} if second is None else {str(k): float(v) for k, v in second.items()})


if __name__ == "__main__":
    raise SystemExit(main())
