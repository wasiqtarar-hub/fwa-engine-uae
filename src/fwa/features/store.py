"""The eight-level feature store (manuscript §4.5).

    "The feature-engineering programme computes behavioural features at eight
     levels, chosen to match the level at which each scenario family in
     Appendix C actually manifests: claim line, claim, encounter, member,
     provider, pharmacy/product, network and episode."     — manuscript §4.5

Two requirements from that section govern every line of this module, and both
are the kind of thing that is easy to satisfy in prose and easy to violate in
code:

**Strictly-before computation.** "Every feature must be computed using data
available strictly before the scored period." A provider's mean billed amount,
computed over the whole file and then attached to every one of that provider's
claims, leaks the future into the past: the claim being scored contributed to
the baseline it is being compared against. Every aggregate here is therefore
computed as an *expanding, time-ordered, shifted* statistic — for claim *i*, the
aggregate is over that entity's claims strictly earlier than *i*. The first
claim for an entity has no history and gets ``NaN`` plus a missingness
indicator, which is the honest answer, not zero.

**Explicit missingness.** "Every feature pipeline must produce an explicit
missingness indicator rather than silently imputing a default value." Handled
by :class:`~fwa.features.frame.FeatureFrame.add`, which creates a companion
``__missing`` column for every feature.

``claim_line`` is the one level that cannot be built: the claim extract is
claim-header level. It is reported as ``NOT_POPULATED`` with the list of
canonical fields that would populate it, rather than being approximated.
"""

from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from ..config import AppConfig
from .frame import FeatureFrame

__all__ = ["FeatureStore", "FeatureLevel", "FEATURE_LEVELS"]

FEATURE_LEVELS = (
    "claim_line",
    "claim",
    "encounter",
    "member",
    "provider",
    "pharmacy_product",
    "network",
    "episode",
)


@dataclass
class FeatureLevel:
    name: str
    status: str                       # POPULATED | PARTIAL | NOT_POPULATED
    reason: str
    frame: FeatureFrame | None = None
    required_canonical_fields: list[str] = field(default_factory=list)


def _prior_expanding(
    df: pd.DataFrame,
    group_col: str,
    value_col: str,
    stat: str = "mean",
    order_col: str = "service_date",
) -> pd.Series:
    """Expanding statistic over an entity's **strictly earlier** rows.

    This is the leakage guard of §4.5 expressed as one function. Sorting by
    ``order_col`` and shifting by one within the group means the value attached
    to row *i* never includes row *i* itself.
    """
    work = df[[group_col, order_col, value_col]].copy()
    if work.empty or work[group_col].notna().sum() == 0:
        # The entity column is absent from this source, so every row groups to
        # nothing and pandas has no arrays to concatenate. An all-missing
        # feature is the correct answer — there is no prior history for an
        # entity the file does not identify — and it propagates as missingness
        # rather than as a zero that would read as "this agent has never
        # claimed before".
        return pd.Series(np.nan, index=df.index, dtype=float)
    work["orig_pos"] = np.arange(len(work))
    work = work.sort_values([group_col, order_col, "orig_pos"], kind="stable")
    grouped = work.groupby(group_col, sort=False)[value_col]
    shifted = grouped.shift(1)
    if stat == "mean":
        out = shifted.groupby(work[group_col], sort=False).expanding().mean().reset_index(level=0, drop=True)
    elif stat == "median":
        out = shifted.groupby(work[group_col], sort=False).expanding().median().reset_index(level=0, drop=True)
    elif stat == "std":
        out = shifted.groupby(work[group_col], sort=False).expanding().std().reset_index(level=0, drop=True)
    elif stat == "sum":
        out = shifted.groupby(work[group_col], sort=False).expanding().sum().reset_index(level=0, drop=True)
    elif stat == "count":
        out = shifted.groupby(work[group_col], sort=False).expanding().count().reset_index(level=0, drop=True)
    else:
        raise ValueError(f"Unsupported statistic {stat!r}")
    work["_out"] = out.values
    return work.sort_values("orig_pos")["_out"].reset_index(drop=True).set_axis(df.index)


def _prior_window_count(
    df: pd.DataFrame, group_col: str, order_col: str, window_days: int
) -> pd.Series:
    """Count of an entity's strictly-earlier rows within a trailing window."""
    work = df[[group_col, order_col]].copy()
    if work.empty or work[group_col].notna().sum() == 0:
        return pd.Series(np.nan, index=df.index, dtype=float)
    work["orig_pos"] = np.arange(len(work))
    work[order_col] = pd.to_datetime(work[order_col])
    out = np.zeros(len(work), dtype=float)
    for _, idx in work.groupby(group_col, sort=False).groups.items():
        sub = work.loc[idx].sort_values([order_col, "orig_pos"], kind="stable")
        times = sub[order_col].to_numpy()
        positions = sub["orig_pos"].to_numpy()
        for j in range(len(sub)):
            if pd.isna(times[j]):
                out[positions[j]] = np.nan
                continue
            lo = times[j] - np.timedelta64(window_days, "D")
            earlier = times[:j]
            out[positions[j]] = int(((earlier > lo) & (earlier <= times[j])).sum())
    return pd.Series(out, index=df.index)


class FeatureStore:
    """Builds all eight levels and reports honestly on the one it cannot."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.levels: dict[str, FeatureLevel] = {}

    # ------------------------------------------------------------------ build

    def build(self, dataset, episode_map: dict[str, str] | None = None) -> dict[str, FeatureLevel]:
        claims = dataset.get("claim_header").copy()
        if claims.empty:
            raise ValueError("Cannot build features from an empty claim_header table.")

        claims["service_date"] = pd.to_datetime(claims["service_date"])
        claims["discharge_date"] = pd.to_datetime(claims["discharge_date"])
        claims["claim_date"] = pd.to_datetime(claims["claim_date"])
        claims = claims.sort_values(["service_date", "claim_sk"], kind="stable").reset_index(drop=True)
        if episode_map:
            claims["episode_id"] = claims["claim_sk"].map(episode_map)
        else:
            claims["episode_id"] = None

        cfg = self.config
        early_tenure_days = cfg.get("early_tenure_days")
        readmit_window = cfg.get("readmit_window_days")
        velocity_window = cfg.get("utilisation_velocity_window_days")

        self._build_claim_line_level()
        self._build_claim_level(claims, early_tenure_days, readmit_window)
        self._build_encounter_level(claims)
        self._build_member_level(claims, velocity_window)
        self._build_provider_level(claims)
        self._build_pharmacy_product_level(claims)
        self._build_network_level(claims, velocity_window)
        self._build_episode_level(claims)
        return self.levels

    # ---------------------------------------------------------------- levels

    def _build_claim_line_level(self) -> None:
        self.levels["claim_line"] = FeatureLevel(
            name="claim_line",
            status="NOT_POPULATED",
            reason=(
                "the claim extract is claim-header level. There are no activity codes, no units and "
                "no line amounts, so unit counts, time-derived duration and code-pair "
                "co-occurrence cannot be computed. The feature design names code-pair "
                "co-occurrence as what PAY-02-R05's novel-unbundling detection depends on, and states that "
                "no claim-level or provider-level aggregate can substitute for it. This level "
                "is therefore empty rather than approximated."
            ),
            required_canonical_fields=[
                "claim_line.activity_code", "claim_line.activity_type", "claim_line.units",
                "claim_line.net_amount", "claim_line.indicator", "claim_line.service_date",
                "claim_line.rendering_clinician_id",
            ],
        )

    def _build_claim_level(self, claims: pd.DataFrame, early_tenure_days: int, readmit_window: int) -> None:
        ff = FeatureFrame.from_frame(
            claims[["claim_sk", "member_sk", "provider_sk", "tpa", "agent_id", "policy_type",
                    "diagnosis_primary", "diagnosis_chapter", "service_date", "claim_date",
                    "discharge_date", "episode_id", "tenant_id"]],
            level="claim",
        )

        gross = claims["gross_amount_aed"].astype(float)
        approved = claims["approved_amount_aed"].astype(float)
        los = claims["length_of_stay_days"].astype(float)

        ff.add("gross_amount_aed", gross, source="claim_header.gross_amount_aed (INR converted at service-date FX)")
        ff.add("approved_amount_aed", approved, source="claim_header.approved_amount_aed")
        ff.add("log_gross_amount_aed", np.log1p(gross.clip(lower=0)),
               source="log1p(gross_amount_aed) — amounts are heavily right-skewed")

        # approval_ratio = approved / requested (build brief §3.2)
        ratio = np.where(gross > 0, approved / gross.replace(0, np.nan), np.nan)
        ff.add("approval_ratio", pd.Series(ratio, index=claims.index),
               source="approved_amount_aed / gross_amount_aed; undefined (not zero) when gross is 0")

        # claim_lag_days = date_of_claim − date_of_discharge
        lag = (claims["claim_date"] - claims["discharge_date"]).dt.days.astype(float)
        ff.add("claim_lag_days", lag, source="date_of_claim − date_of_discharge")

        # amount_per_los_day
        per_day = np.where(los > 0, gross / los.replace(0, np.nan), np.nan)
        ff.add("amount_per_los_day", pd.Series(per_day, index=claims.index),
               source="gross_amount_aed / length_of_stay_days; undefined when LOS is 0")

        ff.add("length_of_stay_days", los, source="encounter.length_of_stay_days")
        ff.add("pharmacy_bill_ratio", claims["pharmacy_bill_ratio"].astype(float),
               source="the claim extract pharmacy_bill_ratio (aggregate share; no line detail behind it)")
        ff.add("num_insurers_same_event", claims["num_insurers_same_event"].astype(float),
               source="the claim extract num_insurers_same_event")
        ff.add("days_since_policy_start", claims["days_since_policy_start"].astype(float),
               source="the claim extract days_since_policy_start")
        ff.add("discharge_readmit_gap_days", claims["discharge_readmit_gap_days"].astype(float),
               source="the claim extract discharge_readmit_gap_days")

        ff.add("is_cashless", claims["is_cashless"].astype(float), source="the claim extract is_cashless")
        ff.add("icd_code_matches_procedure", claims["icd_code_matches_procedure"].astype(float),
               source="the claim extract icd_code_matches_procedure")
        ff.add("coding_mismatch", (~claims["icd_code_matches_procedure"].astype(bool)).astype(float),
               source="NOT icd_code_matches_procedure — the only coding-integrity proxy in the source")
        ff.add("provider_blacklist_flag", claims["provider_blacklist_flag"].astype(float),
               source="the claim extract provider_blacklist_flag (undated; see provider_status_period)")
        ff.add("previous_fraud_on_policy", claims["previous_fraud_on_policy"].astype(float),
               source="the claim extract previous_fraud_on_policy — a POLICY history flag, NOT the outcome label")

        # is_early_tenure = days_since_policy_start < cfg.early_tenure_days
        ff.add("is_early_tenure", (claims["days_since_policy_start"] < early_tenure_days).astype(float),
               source=f"days_since_policy_start < cfg.early_tenure_days ({early_tenure_days})")

        # readmission_flag = discharge_readmit_gap_days < cfg.readmit_window_days
        ff.add("readmission_flag", (claims["discharge_readmit_gap_days"] < readmit_window).astype(float),
               source=f"discharge_readmit_gap_days < cfg.readmit_window_days ({readmit_window})")

        # policy_tenure_band
        bands = pd.cut(
            claims["days_since_policy_start"],
            bins=[-np.inf, early_tenure_days, 365, 730, np.inf],
            labels=["early", "first_year", "second_year", "mature"],
        )
        ff.add("policy_tenure_band_code", bands.cat.codes.astype(float),
               source="banded days_since_policy_start (early/first_year/second_year/mature)")
        ff._df["policy_tenure_band"] = bands.astype(str)

        ff.add("cob_multi_insurer", (claims["num_insurers_same_event"] > 1).astype(float),
               source="num_insurers_same_event > 1 — PAY-10 coordination-of-benefits trigger")

        self.levels["claim"] = FeatureLevel(
            name="claim", status="POPULATED", frame=ff,
            reason="Gross/net arithmetic, derived ratios and tenure/readmission flags, all "
                   "computable from claim-header fields.",
        )

    def _build_encounter_level(self, claims: pd.DataFrame) -> None:
        ff = FeatureFrame.from_frame(claims[["claim_sk", "provider_sk", "service_date"]], level="encounter")
        los = claims["length_of_stay_days"].astype(float)
        ff.add("los_days", los, source="encounter.length_of_stay_days")
        ff.add("admission_month", claims["service_date"].dt.month.astype(float), source="month of admission")
        ff.add("admission_dow", claims["service_date"].dt.dayofweek.astype(float), source="day of week of admission")
        ff.add("los_is_single_day", (los <= 1).astype(float), source="LOS ≤ 1 day")
        ff.add("los_is_extended", (los >= 21).astype(float), source="LOS ≥ 21 days")
        self.levels["encounter"] = FeatureLevel(
            name="encounter", status="PARTIAL", frame=ff,
            reason="Length of stay and admission timing are available. Setting, bed, ICU status "
                   "and the admission/discharge Observation trail are not, so ENT-05-R02 "
                   "(admission evidence conflict) cannot be evaluated.",
            required_canonical_fields=["encounter.encounter_type", "encounter.location", "observation.*"],
        )

    def _build_member_level(self, claims: pd.DataFrame, velocity_window: int) -> None:
        ff = FeatureFrame.from_frame(claims[["claim_sk", "member_sk", "service_date"]], level="member")

        prior_claims = _prior_expanding(claims, "member_sk", "gross_amount_aed", "count")
        ff.add("member_prior_claim_count", prior_claims,
               source="count of this member's STRICTLY EARLIER claims (expanding, shifted — leakage guard)")

        velocity = _prior_window_count(claims, "member_sk", "service_date", velocity_window)
        ff.add("member_utilisation_velocity", velocity,
               source=f"count of this member's earlier claims within the trailing "
                      f"cfg.utilisation_velocity_window_days ({velocity_window}) window")

        prior_spend = _prior_expanding(claims, "member_sk", "gross_amount_aed", "sum")
        ff.add("member_prior_gross_aed", prior_spend,
               source="sum of this member's strictly earlier gross amounts")

        prior_mean = _prior_expanding(claims, "member_sk", "gross_amount_aed", "mean")
        ff.add("member_prior_mean_gross_aed", prior_mean,
               source="mean of this member's strictly earlier gross amounts")

        # distinct prior providers — the ENT-02 / NET-03 shopping signal
        work = claims[["member_sk", "provider_sk", "service_date"]].copy()
        work["orig_pos"] = np.arange(len(work))
        work = work.sort_values(["member_sk", "service_date", "orig_pos"], kind="stable")
        distinct = np.zeros(len(work))
        seen: dict[str, set] = {}
        for pos, row in enumerate(work.itertuples(index=False)):
            s = seen.setdefault(row.member_sk, set())
            distinct[row.orig_pos] = len(s)
            s.add(row.provider_sk)
        ff.add("member_prior_distinct_providers", pd.Series(distinct, index=claims.index),
               source="count of distinct providers this member used BEFORE this claim")

        self.levels["member"] = FeatureLevel(
            name="member", status="PARTIAL", frame=ff,
            reason="Utilisation velocity, prior spend and provider dispersion are computable. "
                   "Concurrent-encounter conflicts are computed at the provider/encounter level; "
                   "demographic-inconsistency patterns (ENT-02-R04) are not, because the source "
                   "has no demographics.",
            required_canonical_fields=["member.date_of_birth", "member.sex", "member.death_date"],
        )

    def _build_provider_level(self, claims: pd.DataFrame) -> None:
        ff = FeatureFrame.from_frame(claims[["claim_sk", "provider_sk", "service_date"]], level="provider")

        work = claims.copy()
        work["approval_ratio"] = np.where(
            work["gross_amount_aed"] > 0,
            work["approved_amount_aed"] / work["gross_amount_aed"].replace(0, np.nan),
            np.nan,
        )
        work["coding_mismatch"] = (~work["icd_code_matches_procedure"].astype(bool)).astype(float)
        work["amount_per_los_day"] = np.where(
            work["length_of_stay_days"] > 0,
            work["gross_amount_aed"] / work["length_of_stay_days"].replace(0, np.nan),
            np.nan,
        )
        work["readmission_flag"] = (
            work["discharge_readmit_gap_days"] < self.config.get("readmit_window_days")
        ).astype(float)

        ff.add("provider_prior_claim_count", _prior_expanding(work, "provider_sk", "gross_amount_aed", "count"),
               source="count of this provider's strictly earlier claims")
        ff.add("provider_prior_mean_gross_aed", _prior_expanding(work, "provider_sk", "gross_amount_aed", "mean"),
               source="mean of this provider's strictly earlier gross amounts")
        ff.add("provider_prior_median_gross_aed", _prior_expanding(work, "provider_sk", "gross_amount_aed", "median"),
               source="median of this provider's strictly earlier gross amounts (robust)")
        ff.add("provider_prior_mean_approval_ratio", _prior_expanding(work, "provider_sk", "approval_ratio", "mean"),
               source="mean approval ratio over this provider's strictly earlier claims")
        ff.add("provider_prior_std_approval_ratio", _prior_expanding(work, "provider_sk", "approval_ratio", "std"),
               source="std of approval ratio over earlier claims — PAY-06-R04 invariance limb")
        ff.add("provider_prior_coding_mismatch_rate", _prior_expanding(work, "provider_sk", "coding_mismatch", "mean"),
               source="share of this provider's strictly earlier claims with a coding mismatch")
        ff.add("provider_prior_readmission_rate", _prior_expanding(work, "provider_sk", "readmission_flag", "mean"),
               source="share of this provider's strictly earlier claims flagged as readmissions")
        ff.add("provider_prior_mean_los", _prior_expanding(work, "provider_sk", "length_of_stay_days", "mean"),
               source="mean length of stay over this provider's strictly earlier claims")
        ff.add("provider_prior_mean_pharmacy_ratio", _prior_expanding(work, "provider_sk", "pharmacy_bill_ratio", "mean"),
               source="mean pharmacy bill ratio over earlier claims")
        ff.add("provider_prior_mean_amount_per_los_day",
               _prior_expanding(work, "provider_sk", "amount_per_los_day", "mean"),
               source="mean amount per LOS day over earlier claims")

        # same-day concurrent admissions — CLN-07 throughput
        same_day = work.groupby(["provider_sk", work["service_date"].dt.date]).transform("size")
        ff.add("provider_same_day_admissions", same_day.iloc[:, 0].astype(float)
               if isinstance(same_day, pd.DataFrame) else same_day.astype(float),
               source="count of admissions at this provider on this calendar day (including this one) "
                      "— CLN-07 capacity input")

        self.levels["provider"] = FeatureLevel(
            name="provider", status="PARTIAL", frame=ff,
            reason="Prior-only behavioural aggregates are computable. Peer-relative CODE-LEVEL "
                   "mix is not, because there are no activity codes, and resubmission behaviour "
                   "is not, because there is no claim lineage.",
            required_canonical_fields=["claim_line.activity_code", "claim_version.*", "remittance.denial_code"],
        )

    def _build_pharmacy_product_level(self, claims: pd.DataFrame) -> None:
        ff = FeatureFrame.from_frame(claims[["claim_sk", "provider_sk", "diagnosis_primary", "service_date"]],
                                     level="pharmacy_product")
        ratio = claims["pharmacy_bill_ratio"].astype(float)
        ff.add("pharmacy_bill_ratio", ratio, source="the claim extract pharmacy_bill_ratio")
        ff.add("pharmacy_amount_aed", ratio * claims["gross_amount_aed"].astype(float),
               source="pharmacy_bill_ratio × gross_amount_aed — an IMPLIED pharmacy amount, not a billed one")

        # prior-only diagnosis-peer median of the pharmacy share
        peer_median = _prior_expanding(
            claims.assign(pharmacy_bill_ratio=ratio), "diagnosis_primary", "pharmacy_bill_ratio", "median"
        )
        ff.add("pharmacy_ratio_diagnosis_peer_median", peer_median,
               source="median pharmacy ratio among STRICTLY EARLIER claims with the same primary diagnosis")
        ff.add("pharmacy_share_residual", ratio - peer_median,
               source="pharmacy_bill_ratio − diagnosis-peer prior median")

        self.levels["pharmacy_product"] = FeatureLevel(
            name="pharmacy_product", status="PARTIAL", frame=ff,
            reason="Only an aggregate pharmacy SHARE exists. There are no products, quantities, "
                   "days supply, prescribers or fill dates, so refill overlap (PHR-02), "
                   "multi-prescriber convergence and wastage (PHR-03-R04) cannot be computed. "
                   "PHR-03-R03 runs on the ratio residual alone.",
            required_canonical_fields=[
                "prescription_dispense.billed_product", "prescription_dispense.days_supply",
                "prescription_dispense.fill_date", "prescription_dispense.prescriber_id",
            ],
        )

    def _build_network_level(self, claims: pd.DataFrame, velocity_window: int) -> None:
        ff = FeatureFrame.from_frame(claims[["claim_sk", "agent_id", "provider_sk", "tpa", "service_date"]],
                                     level="network")
        ff.add("agent_prior_claim_count", _prior_expanding(claims, "agent_id", "gross_amount_aed", "count"),
               source="count of this agent's strictly earlier claims")
        ff.add("agent_prior_mean_gross_aed", _prior_expanding(claims, "agent_id", "gross_amount_aed", "mean"),
               source="mean gross over this agent's strictly earlier claims")
        ff.add("agent_window_claim_count", _prior_window_count(claims, "agent_id", "service_date", velocity_window),
               source=f"this agent's earlier claims within the trailing {velocity_window}-day window")

        # agent→provider concentration, prior-only
        work = claims[["agent_id", "provider_sk", "service_date"]].copy()
        work["orig_pos"] = np.arange(len(work))
        work = work.sort_values(["agent_id", "service_date", "orig_pos"], kind="stable")
        conc = np.full(len(work), np.nan)
        hist: dict[str, dict[str, int]] = {}
        for row in work.itertuples(index=False):
            counts = hist.setdefault(row.agent_id, {})
            total = sum(counts.values())
            if total > 0:
                conc[row.orig_pos] = max(counts.values()) / total
            counts[row.provider_sk] = counts.get(row.provider_sk, 0) + 1
        ff.add("agent_top_provider_share_prior", pd.Series(conc, index=claims.index),
               source="share of this agent's strictly earlier claims going to their single most-used "
                      "provider — NET-04-R02 concentration input")

        self.levels["network"] = FeatureLevel(
            name="network", status="PARTIAL", frame=ff,
            reason="Agent, provider and TPA relationships are derivable, so referral-style "
                   "concentration and ring detection have real inputs. Referral EDGES themselves "
                   "(referrer → recipient) are not in the source and are inferred from "
                   "co-occurrence, which is weaker and is labelled as such on every network case.",
            required_canonical_fields=["provider.owner_entity_id", "provider.bank_account_token",
                                       "provider.phone_token", "provider.address_token"],
        )

    def _build_episode_level(self, claims: pd.DataFrame) -> None:
        ff = FeatureFrame.from_frame(claims[["claim_sk", "episode_id", "member_sk", "provider_sk"]],
                                     level="episode")
        if claims["episode_id"].isna().all():
            self.levels["episode"] = FeatureLevel(
                name="episode", status="NOT_POPULATED",
                reason="No episode map was supplied to the feature store.",
            )
            return
        grp = claims.groupby("episode_id")
        ff.add("episode_claim_count", grp["claim_sk"].transform("size").astype(float),
               source="number of claims in this claim's episode")
        ff.add("episode_total_gross_aed", grp["gross_amount_aed"].transform("sum").astype(float),
               source="total gross across the episode")
        ff.add("episode_total_los_days", grp["length_of_stay_days"].transform("sum").astype(float),
               source="total length of stay across the episode")
        ff.add("episode_is_multi_claim", (grp["claim_sk"].transform("size") > 1).astype(float),
               source="episode contains more than one claim — CLN-05-R03 split-stay input")
        self.levels["episode"] = FeatureLevel(
            name="episode", status="PARTIAL", frame=ff,
            reason="Episodes are built from same-member/same-provider readmission proximity. "
                   "Cross-claim PACKAGE COMPONENT completeness (the cascade feature) cannot "
                   "be computed without activity lines.",
            required_canonical_fields=["claim_line.activity_code", "authorization_line.*"],
        )

    # ------------------------------------------------------------------ views

    def combined_claim_features(self) -> FeatureFrame:
        """One wide, claim-indexed frame joining every claim-aligned level.

        This is what the statistical composite and the model layer consume. It
        is a :class:`FeatureFrame`, so it is structurally incapable of carrying
        a label column into a detector.
        """
        base = self.levels["claim"].frame
        assert base is not None
        wide = base.df.copy()
        prov = {**base.provenance}
        for level in ("encounter", "member", "provider", "pharmacy_product", "network", "episode"):
            lvl = self.levels.get(level)
            if lvl is None or lvl.frame is None:
                continue
            frame = lvl.frame.df
            cols = [c for c in frame.columns if c not in wide.columns and c != "claim_sk"]
            wide = wide.merge(frame[["claim_sk", *cols]], on="claim_sk", how="left")
            prov.update(lvl.frame.provenance)
        return FeatureFrame(wide, level="claim_combined", provenance=prov)

    def coverage_report(self) -> pd.DataFrame:
        rows = []
        for name in FEATURE_LEVELS:
            lvl = self.levels.get(name)
            if lvl is None:
                rows.append({"level": name, "status": "NOT_BUILT", "features": 0, "reason": "", "required_fields": ""})
                continue
            rows.append(
                {
                    "level": name,
                    "status": lvl.status,
                    "features": 0 if lvl.frame is None else len(lvl.frame.feature_columns),
                    "reason": lvl.reason,
                    "required_fields": "; ".join(lvl.required_canonical_fields),
                }
            )
        return pd.DataFrame(rows)

    def missingness_report(self) -> pd.DataFrame:
        frames = [
            lvl.frame.missingness_report()
            for lvl in self.levels.values()
            if lvl.frame is not None
        ]
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
