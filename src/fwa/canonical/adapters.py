"""Source adapters — one interface, three implementations (build brief §3.1).

Manuscript §3.5 and RQ2 require that the system handle **two non-identical UAE
data standards** without assuming field-level equivalence between them, and
Chapter 7 treats that as the answer to RQ2. This module implements that
interface, and is honest about what has actually been exercised:

* :class:`ShafafiyaAdapter` — Abu Dhabi / DoH. Schema-complete against the
  Shafafiya transaction constructs named in §3.5 (Claim, Encounter, Activity,
  Observation, PatientShare, Prior Request/Authorization, Remittance Advice,
  denial codes, Resubmission type). Exercised by fixtures in
  ``tests/unit/test_adapters.py``. Marked ``NOT CERTIFIED — no live regulator
  feed``.
* :class:`EClaimLinkAdapter` — Dubai / DHA. A **materially separate** adapter,
  not a parameterisation of the Shafafiya one, because §3.5 forbids assuming
  field-level identity between the regimes. Also ``NOT CERTIFIED``.
* :class:`GenericIndiaTpaAdapter` — the only adapter that runs in the validation
  pipeline. Maps the claim extract into the canonical model.

Every adapter:

1. writes the original row to the :class:`~fwa.canonical.raw_store.ImmutableRawStore`
   and records its SHA-256 **before parsing** (§3.5, §5.3);
2. records ``source_system`` on every canonical row;
3. retains the original row as ``source_vocabulary`` so a canonical value can
   always be traced back to the literal source value;
4. converts money to AED through :class:`~fwa.config.FxService` at the
   **service date** — never with a hard-coded literal (build brief §3.1).
"""

from __future__ import annotations

import abc
import datetime as _dt
import json
from typing import Any, Mapping

import numpy as np
import pandas as pd

from ..config import AppConfig
from .model import CANONICAL_TABLES, CanonicalDataset, PopulationStatus
from .raw_store import ImmutableRawStore, sha256_record

__all__ = [
    "SourceAdapter",
    "GenericIndiaTpaAdapter",
    "ShafafiyaAdapter",
    "EClaimLinkAdapter",
    "AdapterError",
    "NOT_CERTIFIED",
    "get_adapter",
]

NOT_CERTIFIED = "NOT CERTIFIED — no live regulator feed"


class AdapterError(ValueError):
    """A source file cannot be mapped, with a reason a user can act on.

    Distinct from the pandas ``KeyError`` that an unguarded mapping raises five
    frames down: this one names the columns that are missing and the columns
    that are present, which is the difference between a message a user can do
    something about and one they can only forward to a developer.
    """


class SourceAdapter(abc.ABC):
    """One interface, so a control never learns which regime a claim came from."""

    source_system: str = "UNKNOWN"
    certification: str = NOT_CERTIFIED
    currency: str = "AED"

    def __init__(self, config: AppConfig, raw_store: ImmutableRawStore | None = None, tenant_id: str = "T001") -> None:
        self.config = config
        self.raw_store = raw_store if raw_store is not None else ImmutableRawStore()
        self.tenant_id = tenant_id

    @abc.abstractmethod
    def load(self, source: Any) -> CanonicalDataset:
        """Read the source and return a populated :class:`CanonicalDataset`."""

    # ---- shared helpers -----------------------------------------------------

    def _store_raw(self, rows: list[Mapping[str, Any]], id_field: str) -> list[str]:
        """Hash-and-store every source row BEFORE parsing. Returns the hashes."""
        hashes = []
        for row in rows:
            rec = self.raw_store.put(
                row, source_system=self.source_system, source_record_id=str(row.get(id_field, ""))
            )
            hashes.append(rec.raw_hash)
        return hashes

    def _to_aed(self, amounts: pd.Series, service_dates: pd.Series) -> tuple[pd.Series, pd.Series]:
        """Convert to AED at the service-date rate. Returns (amounts, rates)."""
        rates = []
        cache: dict[_dt.date, float] = {}
        for d in service_dates:
            if pd.isna(d):
                rates.append(np.nan)
                continue
            key = d.date() if hasattr(d, "date") else d
            if key not in cache:
                cache[key] = self.config.fx.rate(self.currency, key).rate
            rates.append(cache[key])
        rate_series = pd.Series(rates, index=amounts.index, dtype=float)
        return amounts.astype(float) * rate_series, rate_series


# =============================================================================
# The adapter that actually runs
# =============================================================================


class GenericIndiaTpaAdapter(SourceAdapter):
    """Maps a claim-header CSV extract into the canonical model.

    This adapter is where the gap between a source schema and the canonical
    model is handled *explicitly* rather than papered over. Three things it
    deliberately does not do:

    * It does not invent claim lines. ``claim_line`` is left ``NOT_POPULATED``
      because the source is claim-header level, and a fabricated line would
      propagate into PAY-02, PAY-03 and CLN-08 as if it were data.
    * It does not fabricate geography. The peer hierarchy's level 5 stays
      ``NOT_POPULATED`` and every geography-dependent control is classified
      ``NOT_EXECUTABLE_ON_THIS_DATASET``.
    * It does not silently impute. Every derived table carries a ``missingness``
      column recording which canonical fields had no source value (§4.5).
    """

    source_system = "GENERIC_INDIA_TPA"
    certification = ("NOT CERTIFIED — claim-header CSV extract mapped by field name. "
                     "No regulator submission schema has been certified against a live feed.")
    currency = "INR"

    #: Label columns. The adapter carries them into a SEPARATE held-out frame,
    #: never into the canonical tables the detection layers read. See
    #: ``fwa.features.frame.FeatureFrame`` for the enforcement.
    LABEL_COLUMNS = ("fraud_label", "fraud_type", "fraud_confidence", "ground_truth_source")

    #: Every source column this adapter reads. Anything here that a file does
    #: not carry becomes an all-missing column, which is what makes the mapping
    #: survive a real extract rather than only the one it was written against.
    SOURCE_COLUMNS = (
        "claim_id", "patient_id", "hospital_id", "agent_id", "tpa", "policy_type",
        "diagnosis_primary", "diagnosis_description",
        "date_of_admission", "date_of_discharge", "date_of_claim",
        "length_of_stay_days", "claim_amount_requested_inr", "claim_amount_approved_inr",
        "is_cashless", "provider_blacklist_flag", "previous_fraud_on_policy",
        "days_since_policy_start", "num_insurers_same_event",
        "icd_code_matches_procedure", "discharge_readmit_gap_days", "pharmacy_bill_ratio",
    )

    #: Columns without which no mapping is possible at all. A file missing one
    #: of these is not a file with a gap in it; it is a different kind of file,
    #: and saying so is more use than mapping it into an unusable shell.
    REQUIRED_COLUMNS = ("claim_id", "patient_id", "hospital_id")

    def _normalise_columns(self, df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
        """Add every source column the mapping reads, empty where it is absent.

        The alternative — indexing straight into whatever the file happens to
        carry — means the first extract with one fewer column raises a
        ``KeyError`` from inside pandas, five frames below anything a user can
        act on. A missing column is an ordinary and expected fact about a
        source, and it is handled the same way every other gap in this system
        is: the field is present and empty, the absence is recorded, and the
        controls that needed it are classified NOT_EXECUTABLE rather than run
        against a substitute.
        """
        missing_required = [c for c in self.REQUIRED_COLUMNS if c not in df.columns]
        if missing_required:
            raise AdapterError(
                f"This file cannot be mapped: it has no {', '.join(missing_required)} column(s). "
                f"A claim needs at least a claim identifier, a member identifier and a provider "
                f"identifier before anything can be said about it. Columns found: "
                f"{', '.join(map(str, df.columns))}."
            )
        absent = [c for c in self.SOURCE_COLUMNS if c not in df.columns]
        for column in absent:
            df[column] = pd.NA
        return df, absent

    def load(self, source: Any) -> CanonicalDataset:
        df = pd.read_csv(source) if not isinstance(source, pd.DataFrame) else source.copy()
        df, self._absent_columns = self._normalise_columns(df)

        # ---- 1. immutable raw store, BEFORE parsing (§3.5) ------------------
        raw_rows = df.to_dict(orient="records")
        raw_hashes = self._store_raw(raw_rows, id_field="claim_id")

        # ---- 2. parse dates with dayfirst=True (build brief §3) -------------
        for col in ("date_of_admission", "date_of_discharge", "date_of_claim"):
            df[col] = pd.to_datetime(df[col], dayfirst=True, errors="coerce")

        ds = CanonicalDataset.empty(source_system=self.source_system, tenant_id=self.tenant_id)
        ds.raw_hashes = dict(zip(df["claim_id"].astype(str), raw_hashes))

        peer_cfg = self.config.peer_groups

        # ---- member ---------------------------------------------------------
        members = (
            df.groupby("patient_id", as_index=False)
            .agg(first_claim=("date_of_claim", "min"))
        )
        member = pd.DataFrame(
            {
                "member_sk": members["patient_id"].astype(str),
                "source_member_id": members["patient_id"].astype(str),
                "protected_id_token": members["patient_id"].astype(str).map(
                    lambda x: sha256_record({"member": x})[:32]
                ),
                "date_of_birth": pd.NaT,
                "sex": None,
                "death_date": pd.NaT,
                "death_source": None,
                "death_source_confidence": np.nan,
                "sponsor_id": None,
                "employer_id": None,
                "tenant_id": self.tenant_id,
                "source_system": self.source_system,
            }
        )
        member["missingness"] = json.dumps(
            {
                "date_of_birth": "NOT_IN_SOURCE",
                "sex": "NOT_IN_SOURCE",
                "death_date": "NOT_IN_SOURCE",
                "sponsor_id": "NOT_IN_SOURCE",
                "employer_id": "NOT_IN_SOURCE",
            }
        )
        ds.set(
            "member", member, PopulationStatus.PARTIAL,
            "Identity key only. the claim extract has no DOB, sex, death status or sponsor, so "
            "ENT-02-R01 (identity conflict) and ENT-02-R02 (post-mortem service) cannot run.",
        )

        # ---- provider -------------------------------------------------------
        counts = df.groupby("hospital_id").size()
        provider = pd.DataFrame(
            {
                "provider_sk": counts.index.astype(str),
                "source_provider_id": counts.index.astype(str),
                "regulator_id": None,
                "provider_type": "FACILITY_UNSPECIFIED",
                "specialty": None,
                "facility_type": None,
                "owner_entity_id": None,
                "bank_account_token": None,
                "phone_token": None,
                "address_token": None,
                "emirate": None,
                "volume_band": [peer_cfg.volume_band(int(c)) for c in counts.values],
                "tenant_id": self.tenant_id,
                "source_system": self.source_system,
            }
        ).reset_index(drop=True)
        provider["claim_count"] = counts.values
        blacklisted = set(
            df.loc[df["provider_blacklist_flag"], "hospital_id"].astype(str).unique()
        )
        provider["missingness"] = json.dumps(
            {
                "regulator_id": "NOT_IN_SOURCE",
                "specialty": "NOT_IN_SOURCE — ICD chapter used as a service-line proxy",
                "emirate": "NOT_IN_SOURCE — peer hierarchy level 5 is NOT_POPULATED",
                "owner_entity_id": "NOT_IN_SOURCE — NET-02-R01 shared-identifier linkage cannot run",
            }
        )
        ds.set(
            "provider", provider, PopulationStatus.PARTIAL,
            "Provider key and derived volume band only. No regulator ID, specialty, "
            "ownership or administrative identifiers, so ENT-03 licence controls and "
            "NET-02-R01 shared-identity linkage are NOT_EXECUTABLE.",
        )

        # ---- provider_status_period (only the blacklist flag exists) --------
        if blacklisted:
            status = pd.DataFrame(
                {
                    "provider_sk": sorted(blacklisted),
                    "status_type": "EXCLUSION_LIST",
                    "status_value": "BLACKLISTED",
                    "valid_from": _dt.date(1900, 1, 1),
                    "valid_to": _dt.date(2099, 12, 31),
                    "source": "the claim extract provider_blacklist_flag (row-level, not a regulator feed)",
                    "match_type": "EXACT",
                    "version_id": "proxy-1.0.0",
                }
            )
            ds.set(
                "provider_status_period", status, PopulationStatus.PARTIAL,
                "Only an exclusion-list status is derivable, from the row-level "
                "provider_blacklist_flag. No licence periods, no privileges, no network "
                "status, and no effective dating — the flag has no date, so it is "
                "recorded as open-ended and every control depending on 'licence active "
                "AT SERVICE TIME' remains NOT_EXECUTABLE.",
            )

        # ---- coverage_period ------------------------------------------------
        cov = df[["patient_id", "policy_type", "days_since_policy_start", "date_of_claim", "tpa"]].copy()
        cov["inception"] = cov["date_of_claim"] - pd.to_timedelta(cov["days_since_policy_start"], unit="D")
        cov = cov.sort_values("date_of_claim").groupby("patient_id", as_index=False).first()
        coverage = pd.DataFrame(
            {
                "coverage_id": "COV-" + cov["patient_id"].astype(str),
                "member_sk": cov["patient_id"].astype(str),
                "product": cov["policy_type"].astype(str),
                "payer_id": cov["tpa"].astype(str),
                "valid_from": cov["inception"].dt.date,
                "valid_to": _dt.date(2099, 12, 31),
                "network": None,
                "status": "ACTIVE_ASSUMED",
                "policy_inception_date": cov["inception"].dt.date,
                "days_since_policy_start": cov["days_since_policy_start"].astype(int),
                "tenant_id": self.tenant_id,
                "source_system": self.source_system,
            }
        )
        ds.set(
            "coverage_period", coverage, PopulationStatus.PARTIAL,
            "Inception derived as date_of_claim − days_since_policy_start. Coverage END "
            "is unknown and is recorded as open-ended rather than guessed, so "
            "ENT-01-R01 (coverage inactive) is NOT_EXECUTABLE: a control that can only "
            "ever evaluate to 'covered' is not a control.",
        )

        # ---- encounter ------------------------------------------------------
        encounter = pd.DataFrame(
            {
                "encounter_sk": "ENC-" + df["claim_id"].astype(str),
                "claim_sk": df["claim_id"].astype(str),
                "encounter_type": "INPATIENT_INFERRED",
                "facility_id": df["hospital_id"].astype(str),
                "start_time": df["date_of_admission"],
                "end_time": df["date_of_discharge"],
                "admission_date": df["date_of_admission"].dt.date,
                "discharge_date": df["date_of_discharge"].dt.date,
                "length_of_stay_days": df["length_of_stay_days"].astype(int),
                "location": None,
                "tenant_id": self.tenant_id,
            }
        )
        encounter["missingness"] = json.dumps(
            {
                "encounter_type": "INFERRED — every row has an admission and discharge date and a LOS, "
                                  "so the encounter is treated as inpatient. Not sourced.",
                "location": "NOT_IN_SOURCE",
                "start_time/end_time": "DATE ONLY — no time component, so overlap and concurrency "
                                       "controls carry a same-day tolerance and reduced confidence.",
            }
        )
        ds.set(
            "encounter", encounter, PopulationStatus.PARTIAL,
            "Admission/discharge/LOS present; encounter TYPE is inferred, not sourced, and "
            "timestamps are date-only.",
        )

        # ---- diagnosis ------------------------------------------------------
        diagnosis = pd.DataFrame(
            {
                "claim_sk": df["claim_id"].astype(str),
                "encounter_sk": "ENC-" + df["claim_id"].astype(str),
                "code": df["diagnosis_primary"].astype(str),
                "code_system": "ICD-10",
                "code_version": peer_cfg.icd_chapter_map.get("version", "1.0.0"),
                "diagnosis_type": "PRINCIPAL",
                "sequence": 1,
                "present_on_admission": None,
                "description": df["diagnosis_description"].astype(str),
                "chapter": [peer_cfg.chapter_for(c) for c in df["diagnosis_primary"].astype(str)],
                "tenant_id": self.tenant_id,
            }
        )
        ds.set(
            "diagnosis", diagnosis, PopulationStatus.PARTIAL,
            "Exactly one principal diagnosis per claim. No secondary diagnoses and no "
            "present-on-admission indicator, so CLN-02 (comorbidity/DRG severity "
            "manipulation) is NOT_EXECUTABLE — its entire premise is the secondary-code set.",
        )

        # ---- claim_header ---------------------------------------------------
        gross_aed, fx_rates = self._to_aed(df["claim_amount_requested_inr"], df["date_of_admission"])
        approved_aed, _ = self._to_aed(df["claim_amount_approved_inr"], df["date_of_admission"])
        source_vocab = df.drop(columns=list(self.LABEL_COLUMNS), errors="ignore").to_dict(orient="records")
        claim_header = pd.DataFrame(
            {
                "claim_sk": df["claim_id"].astype(str),
                "source_claim_id": df["claim_id"].astype(str),
                "tenant_id": self.tenant_id,
                "member_sk": df["patient_id"].astype(str),
                "provider_sk": df["hospital_id"].astype(str),
                "payer_id": df["tpa"].astype(str),
                "tpa": df["tpa"].astype(str),
                "encounter_sk": "ENC-" + df["claim_id"].astype(str),
                "submission_date": df["date_of_claim"].dt.date,
                "settlement_date": pd.NaT,
                "gross_amount": df["claim_amount_requested_inr"].astype(float),
                "net_amount": df["claim_amount_approved_inr"].astype(float),
                "patient_share": np.nan,
                "approved_amount": df["claim_amount_approved_inr"].astype(float),
                "source_currency": self.currency,
                "gross_amount_aed": gross_aed.round(2),
                "approved_amount_aed": approved_aed.round(2),
                "fx_rate_used": fx_rates,
                "fx_rate_source": "config/fx.yaml, resolved at service (admission) date",
                "is_cashless": df["is_cashless"].astype(bool),
                "num_insurers_same_event": df["num_insurers_same_event"].astype(int),
                "agent_id": df["agent_id"].astype(str),
                "source_system": self.source_system,
                "raw_hash": raw_hashes,
                "source_vocabulary": [json.dumps(r, default=str) for r in source_vocab],
            }
        )
        # extra proxy-only columns the feature store needs, kept on the header
        claim_header["service_date"] = df["date_of_admission"]
        claim_header["discharge_date"] = df["date_of_discharge"]
        claim_header["claim_date"] = df["date_of_claim"]
        claim_header["length_of_stay_days"] = df["length_of_stay_days"].astype(int)
        claim_header["policy_type"] = df["policy_type"].astype(str)
        claim_header["diagnosis_primary"] = df["diagnosis_primary"].astype(str)
        claim_header["diagnosis_chapter"] = diagnosis["chapter"].values
        claim_header["pharmacy_bill_ratio"] = df["pharmacy_bill_ratio"].astype(float)
        claim_header["icd_code_matches_procedure"] = df["icd_code_matches_procedure"].astype(bool)
        claim_header["provider_blacklist_flag"] = df["provider_blacklist_flag"].astype(bool)
        claim_header["previous_fraud_on_policy"] = df["previous_fraud_on_policy"].astype(bool)
        claim_header["days_since_policy_start"] = df["days_since_policy_start"].astype(int)
        claim_header["discharge_readmit_gap_days"] = df["discharge_readmit_gap_days"].astype(int)
        claim_header["missingness"] = json.dumps(
            {
                "patient_share": "NOT_IN_SOURCE — PAY-07 patient-share controls NOT_EXECUTABLE",
                "settlement_date": "NOT_IN_SOURCE — pend turnaround NOT_MEASURABLE",
                "sender/receiver": "NOT_IN_SOURCE",
            }
        )
        ds.set("claim_header", claim_header, PopulationStatus.POPULATED,
               "Fully populated from the claim extract, with AED amounts converted at the "
               "service-date FX rate and the original row retained as source_vocabulary.")

        # ---- the tables the source genuinely cannot fill --------------------
        for table, reason in {
            "claim_line": (
                "the claim extract is CLAIM-HEADER level: no activity codes, no units, no line "
                "amounts. Not invented. Every line-level control (PAY-02 unbundling, "
                "PAY-03 units/time, PAY-05 modifiers, CLN-08 panels, PHR-01 products) is "
                "classified NOT_EXECUTABLE_ON_THIS_DATASET rather than run against "
                "fabricated lines."
            ),
            "benefit_rule_version": (
                "No benefit schedule, no limits, no patient-share rules and no "
                "authorisation-required flags in the source. ENT-01-R02/R03 and PAY-04 "
                "cannot be evaluated."
            ),
            "observation": "No results, reports or attachments in the source.",
            "authorization": (
                "No prior-authorisation records. The entire PAY-04 scenario (5 controls) "
                "is NOT_EXECUTABLE — this is the single largest coverage gap in the "
                "dataset relative to a real Shafafiya feed."
            ),
            "authorization_line": "No authorisation lines; see authorization.",
            "claim_version": (
                "No resubmission or correction lineage. Every claim_id is unique and "
                "appears once, so PAY-08 (denial-resubmission mutation) and DOC-02-R03 "
                "(post-denial alteration) are NOT_EXECUTABLE. Note the consequence for "
                "testing: the correction/cancellation/resubmission test category "
                "is exercised against constructed fixtures, not against the claim extract."
            ),
            "remittance": (
                "No payment decisions, denial codes, payment references or settlement "
                "dates. PAY-12 (reversal/refund/remittance leakage) is NOT_EXECUTABLE, "
                "and 'prevented/recovered AED' is NOT_MEASURABLE_ON_THIS_DATASET."
            ),
            "prescription_dispense": (
                "No prescriptions or dispense records. Only an aggregate "
                "pharmacy_bill_ratio exists, which supports PHR-03-R03 as a ratio "
                "anomaly but supports none of PHR-01, PHR-02, PHR-04 or PHR-05."
            ),
            "policy_event": (
                "No enrolment events, actors or timestamps. POL-01-R01/R02/R03 are "
                "NOT_EXECUTABLE; only the tenure-derived POL-01-R04 can run."
            ),
        }.items():
            ds.mark_not_populated(table, reason)

        # review_outcome starts empty but is POPULATED at runtime by reviewers
        ds.mark_not_populated(
            "review_outcome",
            "Empty at load. Populated by reviewer dispositions recorded in the "
            "application; this is the FR7 feedback source, not a source-system table.",
        )
        ds.reasons["__certification__"] = self.certification
        if self._absent_columns:
            # Recorded on the dataset rather than printed to a log, because the
            # interface needs to be able to say WHY a control could not run on
            # a particular file, and "the source had no agent_id column" is the
            # whole answer for a third of them.
            ds.reasons["__absent_source_columns__"] = (
                "Not present in this source file, and therefore empty in every canonical table "
                "that would have carried them: " + ", ".join(self._absent_columns) + "."
            )
        return ds

    def held_out_labels(self, source: Any) -> pd.DataFrame:
        """Return the label columns **separately**, for evaluation only.

        The pipeline never merges this frame into features. ``fraud_label``,
        ``fraud_type``, ``fraud_confidence`` and ``ground_truth_source`` are
        held-out evaluation data, and
        ``tests/governance/test_label_leakage.py`` asserts that every detector's
        input columns are disjoint from this frame's columns.

        A file with no label columns is the ordinary case for real data — there
        is no ground truth to hold out — so this returns an empty frame rather
        than failing. Everything downstream already treats "no labels" as a
        reason to report a metric as unmeasurable rather than to guess at it.
        """
        df = pd.read_csv(source) if not isinstance(source, pd.DataFrame) else source
        present = [c for c in self.LABEL_COLUMNS if c in df.columns]
        if not present or "claim_id" not in df.columns:
            return pd.DataFrame(columns=["claim_sk", *self.LABEL_COLUMNS])
        cols = ["claim_id", *present]
        out = df[cols].copy()
        for column in self.LABEL_COLUMNS:
            if column not in out.columns:
                out[column] = pd.NA
        out = out.rename(columns={"claim_id": "claim_sk"})
        out["claim_sk"] = out["claim_sk"].astype(str)
        return out


# =============================================================================
# The two UAE adapters — schema-complete, NOT CERTIFIED
# =============================================================================


class ShafafiyaAdapter(SourceAdapter):
    """Abu Dhabi / DoH Shafafiya adapter.

    Schema-complete against the transaction constructs manuscript §3.5 names:
    Claim, Encounter, Activity, Observation, PatientShare, Prior
    Request/Authorization, Remittance Advice, denial code and Resubmission type.
    Shafafiya semantics are **preserved, not altered** — the source element name
    is retained in ``source_vocabulary`` on every row.

    Status: ``NOT CERTIFIED — no live regulator feed``. This adapter has been
    exercised against constructed fixtures only. No UAE claims data underlies
    anything in this repository, and nothing here should be read as a claim that
    it has been tested against a real DoH submission.
    """

    source_system = "SHAFAFIYA_DOH_ABU_DHABI"
    certification = NOT_CERTIFIED
    currency = "AED"

    #: Shafafiya element → canonical field. Deliberately explicit rather than
    #: derived, because §3.5 forbids assuming field-level identity with DHA.
    ELEMENT_MAP: dict[str, tuple[str, str]] = {
        "Claim/ID": ("claim_header", "source_claim_id"),
        "Claim/MemberID": ("claim_header", "member_sk"),
        "Claim/PayerID": ("claim_header", "payer_id"),
        "Claim/ProviderID": ("claim_header", "provider_sk"),
        "Claim/Gross": ("claim_header", "gross_amount"),
        "Claim/PatientShare": ("claim_header", "patient_share"),
        "Claim/Net": ("claim_header", "net_amount"),
        "Encounter/FacilityID": ("encounter", "facility_id"),
        "Encounter/Type": ("encounter", "encounter_type"),
        "Encounter/Start": ("encounter", "start_time"),
        "Encounter/End": ("encounter", "end_time"),
        "Encounter/StartType": ("encounter", "admission_type"),
        "Encounter/EndType": ("encounter", "discharge_type"),
        "Activity/ID": ("claim_line", "line_sk"),
        "Activity/Start": ("claim_line", "service_date"),
        "Activity/Type": ("claim_line", "activity_type"),
        "Activity/Code": ("claim_line", "activity_code"),
        "Activity/Quantity": ("claim_line", "units"),
        "Activity/Net": ("claim_line", "net_amount"),
        "Activity/Clinician": ("claim_line", "rendering_clinician_id"),
        "Activity/OrderingClinician": ("claim_line", "ordering_clinician_id"),
        "Activity/PriorAuthorizationID": ("claim_line", "authorization_id"),
        "Observation/Type": ("observation", "observation_type"),
        "Observation/Code": ("observation", "observation_code"),
        "Observation/Value": ("observation", "value"),
        "Observation/ValueType": ("observation", "unit"),
        "Diagnosis/Type": ("diagnosis", "diagnosis_type"),
        "Diagnosis/Code": ("diagnosis", "code"),
        "Authorization/ID": ("authorization", "authorization_sk"),
        "Authorization/Status": ("authorization", "status"),
        "Authorization/Start": ("authorization", "valid_from"),
        "Authorization/End": ("authorization", "valid_to"),
        "Remittance/DenialCode": ("remittance", "denial_code"),
        "Remittance/PaymentReference": ("remittance", "payment_reference"),
        "Remittance/DateSettlement": ("remittance", "settlement_date"),
        "Resubmission/Type": ("claim_version", "resubmission_type"),
        "Resubmission/Comment": ("claim_version", "changed_fields"),
    }

    def load(self, source: Any) -> CanonicalDataset:
        """Map a Shafafiya-shaped record set into the canonical model.

        ``source`` is a mapping of construct name → list of dicts, as produced
        by a Shafafiya XML parse. The fixtures in ``tests/unit/test_adapters.py``
        supply exactly that shape.
        """
        records: Mapping[str, list[Mapping[str, Any]]] = source
        ds = CanonicalDataset.empty(source_system=self.source_system, tenant_id=self.tenant_id)
        ds.reasons["__certification__"] = self.certification

        for construct, rows in records.items():
            self._store_raw(list(rows), id_field="Claim/ID" if construct == "Claim" else "ID")

        staged: dict[str, list[dict[str, Any]]] = {t: [] for t in CANONICAL_TABLES}
        for construct, rows in records.items():
            for row in rows:
                buckets: dict[str, dict[str, Any]] = {}
                for element, value in row.items():
                    target = self.ELEMENT_MAP.get(element)
                    if target is None:
                        continue
                    table, column = target
                    buckets.setdefault(table, {})[column] = value
                for table, payload in buckets.items():
                    payload["tenant_id"] = self.tenant_id
                    payload["source_system"] = self.source_system
                    payload["source_vocabulary"] = json.dumps(dict(row), default=str)
                    staged.setdefault(table, []).append(payload)

        for table, rows in staged.items():
            if not rows:
                ds.mark_not_populated(table, f"No {table} records in this Shafafiya submission.")
                continue
            frame = pd.DataFrame(rows)
            if "claim_sk" not in frame.columns and "source_claim_id" in frame.columns:
                frame["claim_sk"] = frame["source_claim_id"]
            ds.set(table, frame, PopulationStatus.PARTIAL,
                   f"Mapped from Shafafiya constructs. {self.certification}.")
        return ds


class EClaimLinkAdapter(SourceAdapter):
    """Dubai / DHA eClaimLink adapter.

    A **materially separate** implementation, not a configuration of
    :class:`ShafafiyaAdapter`. Manuscript §3.5 and §7.5 are explicit that the
    design must assume **no field-level identity** between the two regimes, and
    collapsing them into one parameterised mapper would quietly re-introduce
    exactly that assumption. The element map below is therefore its own map,
    and where the two regimes happen to agree, they agree by coincidence rather
    than by construction.

    Status: ``NOT CERTIFIED — no live regulator feed``.
    """

    source_system = "ECLAIMLINK_DHA_DUBAI"
    certification = NOT_CERTIFIED
    currency = "AED"

    ELEMENT_MAP: dict[str, tuple[str, str]] = {
        "Claim/ID": ("claim_header", "source_claim_id"),
        "Claim/IDPayer": ("claim_header", "payer_id"),
        "Claim/MemberID": ("claim_header", "member_sk"),
        "Claim/EmiratesIDNumber": ("member", "protected_id_token"),
        "Claim/ProviderID": ("claim_header", "provider_sk"),
        "Claim/Gross": ("claim_header", "gross_amount"),
        "Claim/PatientShare": ("claim_header", "patient_share"),
        "Claim/Net": ("claim_header", "net_amount"),
        "Encounter/FacilityID": ("encounter", "facility_id"),
        "Encounter/Type": ("encounter", "encounter_type"),
        "Encounter/PatientID": ("encounter", "member_sk"),
        "Encounter/Start": ("encounter", "start_time"),
        "Encounter/End": ("encounter", "end_time"),
        "Activity/ID": ("claim_line", "line_sk"),
        "Activity/Start": ("claim_line", "service_date"),
        "Activity/Type": ("claim_line", "activity_type"),
        "Activity/Code": ("claim_line", "activity_code"),
        "Activity/Quantity": ("claim_line", "units"),
        "Activity/Net": ("claim_line", "net_amount"),
        "Activity/Clinician": ("claim_line", "rendering_clinician_id"),
        "Activity/PriorAuthorizationID": ("claim_line", "authorization_id"),
        "Activity/Observation/Type": ("observation", "observation_type"),
        "Activity/Observation/Code": ("observation", "observation_code"),
        "Activity/Observation/Value": ("observation", "value"),
        "Diagnosis/Type": ("diagnosis", "diagnosis_type"),
        "Diagnosis/Code": ("diagnosis", "code"),
        "Diagnosis/DxInfo": ("diagnosis", "present_on_admission"),
        "Authorization/ID": ("authorization", "authorization_sk"),
        "Authorization/Status": ("authorization", "status"),
        "Remittance/Activity/DenialCode": ("remittance", "denial_code"),
        "Remittance/Activity/PaymentAmount": ("remittance", "payment_amount"),
        "Remittance/DateSettlement": ("remittance", "settlement_date"),
        "Claim/Resubmission/Type": ("claim_version", "resubmission_type"),
    }

    #: Fields that exist in one regime and not the other. Recorded rather than
    #: silently dropped, so the difference between the regimes is inspectable.
    REGIME_DIFFERENCES = {
        "EmiratesIDNumber": "Present in eClaimLink, absent from the Shafafiya map used here.",
        "Encounter/StartType": "Present in Shafafiya, absent from this eClaimLink map.",
        "DxInfo": "eClaimLink carries diagnosis info sub-elements the Shafafiya map does not.",
    }

    def load(self, source: Any) -> CanonicalDataset:
        records: Mapping[str, list[Mapping[str, Any]]] = source
        ds = CanonicalDataset.empty(source_system=self.source_system, tenant_id=self.tenant_id)
        ds.reasons["__certification__"] = self.certification

        for construct, rows in records.items():
            self._store_raw(list(rows), id_field="Claim/ID" if construct == "Claim" else "ID")

        staged: dict[str, list[dict[str, Any]]] = {}
        for construct, rows in records.items():
            for row in rows:
                buckets: dict[str, dict[str, Any]] = {}
                for element, value in row.items():
                    target = self.ELEMENT_MAP.get(element)
                    if target is None:
                        continue
                    table, column = target
                    buckets.setdefault(table, {})[column] = value
                for table, payload in buckets.items():
                    payload["tenant_id"] = self.tenant_id
                    payload["source_system"] = self.source_system
                    payload["source_vocabulary"] = json.dumps(dict(row), default=str)
                    staged.setdefault(table, []).append(payload)

        for table in CANONICAL_TABLES:
            rows = staged.get(table, [])
            if not rows:
                ds.mark_not_populated(table, f"No {table} records in this eClaimLink submission.")
                continue
            frame = pd.DataFrame(rows)
            if "claim_sk" not in frame.columns and "source_claim_id" in frame.columns:
                frame["claim_sk"] = frame["source_claim_id"]
            ds.set(table, frame, PopulationStatus.PARTIAL,
                   f"Mapped from eClaimLink constructs. {self.certification}.")
        return ds


_ADAPTERS: dict[str, type[SourceAdapter]] = {
    "generic_india_tpa": GenericIndiaTpaAdapter,
    "shafafiya": ShafafiyaAdapter,
    "eclaimlink": EClaimLinkAdapter,
}


def get_adapter(name: str) -> type[SourceAdapter]:
    try:
        return _ADAPTERS[name]
    except KeyError as exc:
        raise KeyError(f"Unknown adapter {name!r}. Available: {sorted(_ADAPTERS)}") from exc
