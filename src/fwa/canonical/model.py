"""The sixteen-table canonical model.

The canonical model exists so that a control is written once against canonical
fields and runs unchanged whether the claim arrived through Shafafiya,
eClaimLink or a generic claim-header extract. This module is the attribute-level
data dictionary expressed as schemas.

**Honesty rule.** The shipped the claim extract is claim-header level. It
has no activity lines, no authorisations, no remittance, no documents and no
dispense records. Those tables are therefore created *empty and explicitly
marked* :data:`NOT_POPULATED`, with the reason recorded, rather than being
filled with invented line items. A table that is empty because the source has no
such data is a finding; a table that is full of fabricated data is a lie that
propagates into every downstream metric.

``CanonicalDataset.population_report()`` renders exactly that, and it feeds both
the Overview page and ``reports/control_coverage_matrix.csv``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

import pandas as pd

__all__ = [
    "CANONICAL_TABLES",
    "TableSpec",
    "CanonicalDataset",
    "PopulationStatus",
    "NOT_POPULATED",
]

NOT_POPULATED = "NOT_POPULATED"


class PopulationStatus(str):
    POPULATED = "POPULATED"
    PARTIAL = "PARTIAL"
    NOT_POPULATED = "NOT_POPULATED"


@dataclass(frozen=True)
class TableSpec:
    """One canonical table: its key, its required content, its columns."""

    name: str
    primary_key: str
    required_content: str
    columns: dict[str, str]  # column -> dtype hint

    def empty_frame(self) -> pd.DataFrame:
        return pd.DataFrame({c: pd.Series(dtype=object) for c in self.columns})


def _spec(name: str, pk: str, content: str, cols: dict[str, str]) -> TableSpec:
    return TableSpec(name=name, primary_key=pk, required_content=content, columns=cols)


# -----------------------------------------------------------------------------
# The sixteen tables of Table 3.5, plus review_outcome (the seventeenth row of
# that table, which is the feedback-capture table FR7 depends on).
# -----------------------------------------------------------------------------

CANONICAL_TABLES: dict[str, TableSpec] = {
    "member": _spec(
        "member", "member_sk",
        "Protected identity linkage, date of birth, sex, death status/source, sponsor/employer",
        {
            "member_sk": "str", "source_member_id": "str", "protected_id_token": "str",
            "date_of_birth": "date", "sex": "str", "death_date": "date",
            "death_source": "str", "death_source_confidence": "float",
            "sponsor_id": "str", "employer_id": "str", "tenant_id": "str",
            "source_system": "str", "missingness": "json",
            # extensions used by the UAE multi-table adapter
            "relationship": "str", "weight_kg": "float", "height_cm": "float", "emirate": "str",
        },
    ),
    "coverage_period": _spec(
        "coverage_period", "coverage_id",
        "Member, product, payer, effective dates, network, status",
        {
            "coverage_id": "str", "member_sk": "str", "product": "str", "payer_id": "str",
            "valid_from": "date", "valid_to": "date", "network": "str", "status": "str",
            "policy_inception_date": "date", "days_since_policy_start": "int",
            "tenant_id": "str", "source_system": "str",
            "agent_id": "str", "product_tier": "str",
        },
    ),
    "benefit_rule_version": _spec(
        "benefit_rule_version", "(product, service_family, valid_from)",
        "Product, service family, coverage, limits, patient share, exceptions, effective dates",
        {
            "product": "str", "service_family": "str", "covered": "bool",
            "benefit_limit": "float", "patient_share_pct": "float",
            "authorization_required": "bool", "exceptions": "json",
            "valid_from": "date", "valid_to": "date", "version_id": "str",
            "recorded_at": "datetime", "source": "str",
        },
    ),
    "provider": _spec(
        "provider", "provider_sk",
        "Regulator IDs, type, specialty, facility, ownership and administrative identifiers",
        {
            "provider_sk": "str", "source_provider_id": "str", "regulator_id": "str",
            "provider_type": "str", "specialty": "str", "facility_type": "str",
            "owner_entity_id": "str", "bank_account_token": "str", "phone_token": "str",
            "address_token": "str", "emirate": "str", "volume_band": "str",
            "tenant_id": "str", "source_system": "str", "missingness": "json",
            "credentialing_date": "date", "bed_count": "int", "operational_status": "str",
            "ownership_changed_on": "date", "licence_no": "str",
        },
    ),
    "provider_status_period": _spec(
        "provider_status_period", "(provider_sk, status_type, valid_from)",
        "Licence, privilege, network and exclusion status by effective date",
        {
            "provider_sk": "str", "status_type": "str", "status_value": "str",
            "valid_from": "date", "valid_to": "date", "source": "str",
            "match_type": "str", "version_id": "str",
        },
    ),
    "claim_header": _spec(
        "claim_header", "claim_sk",
        "Source IDs, sender/receiver, payer/TPA, encounter, amounts, submission/settlement data",
        {
            "claim_sk": "str", "source_claim_id": "str", "tenant_id": "str",
            "member_sk": "str", "provider_sk": "str", "payer_id": "str", "tpa": "str",
            "encounter_sk": "str", "submission_date": "date", "settlement_date": "date",
            "gross_amount": "float", "net_amount": "float", "patient_share": "float",
            "approved_amount": "float", "source_currency": "str",
            "gross_amount_aed": "float", "approved_amount_aed": "float",
            "fx_rate_used": "float", "fx_rate_source": "str",
            "is_cashless": "bool", "num_insurers_same_event": "int",
            "agent_id": "str", "source_system": "str", "raw_hash": "str",
            "source_vocabulary": "json", "missingness": "json",
            "claim_type": "str", "accident_indicator": "bool", "discount": "float",
            "cross_payer_match_token": "str",
        },
    ),
    "claim_line": _spec(
        "claim_line", "line_sk",
        "Activity type/code, units, amounts, clinician roles, indicator/modifier, authorisation ID",
        {
            "line_sk": "str", "claim_sk": "str", "activity_type": "str", "activity_code": "str",
            "units": "float", "gross_amount": "float", "net_amount": "float",
            "patient_share": "float", "rendering_clinician_id": "str",
            "ordering_clinician_id": "str", "indicator": "str", "authorization_id": "str",
            "service_date": "date", "tenant_id": "str",
            "service_start_time": "datetime", "service_end_time": "datetime",
            "performing_entity_id": "str", "wastage_units": "float", "unit_price": "float",
            "device_serial": "str", "product": "str", "activity_description": "str",
        },
    ),
    "diagnosis": _spec(
        "diagnosis", "(claim_sk, code, sequence)",
        "Claim/encounter, code/version, type, principal/secondary, present-on-admission",
        {
            "claim_sk": "str", "encounter_sk": "str", "code": "str", "code_system": "str",
            "code_version": "str", "diagnosis_type": "str", "sequence": "int",
            "present_on_admission": "str", "description": "str", "chapter": "str",
            "tenant_id": "str",
        },
    ),
    "encounter": _spec(
        "encounter", "encounter_sk",
        "Type, facility, start/end, admission/discharge and location",
        {
            "encounter_sk": "str", "claim_sk": "str", "encounter_type": "str",
            "facility_id": "str", "start_time": "datetime", "end_time": "datetime",
            "admission_date": "date", "discharge_date": "date",
            "length_of_stay_days": "int", "location": "str", "tenant_id": "str",
            "missingness": "json",
            "member_sk": "str", "bed_id": "str", "duration_minutes": "float",
            "observation_status": "str", "admission_type": "str", "discharge_type": "str",
        },
    ),
    "observation": _spec(
        "observation", "observation_sk",
        "Parent activity, type, result/value, attachment and event time",
        {
            "observation_sk": "str", "line_sk": "str", "observation_type": "str",
            "value": "str", "unit": "str", "attachment_ref": "str", "event_time": "datetime",
            "tenant_id": "str",
            "observation_code": "str", "claim_sk": "str", "member_sk": "str",
        },
    ),
    "authorization": _spec(
        "authorization", "authorization_sk",
        "Request/response IDs, status, validity, provider/facility, approved amounts/units",
        {
            "authorization_sk": "str", "request_id": "str", "response_id": "str",
            "status": "str", "valid_from": "datetime", "valid_to": "datetime",
            "provider_sk": "str", "facility_id": "str", "approved_amount": "float",
            "member_sk": "str", "tenant_id": "str",
        },
    ),
    "authorization_line": _spec(
        "authorization_line", "authorization_line_sk",
        "Approved activity, quantity/value, conditions and denial code",
        {
            "authorization_line_sk": "str", "authorization_sk": "str", "activity_code": "str",
            "approved_units": "float", "approved_value": "float", "conditions": "str",
            "denial_code": "str", "tenant_id": "str",
        },
    ),
    "claim_version": _spec(
        "claim_version", "(claim_sk, version_no)",
        "Original/resubmission/correction relationship and field-level version history",
        {
            "claim_sk": "str", "version_no": "int", "relationship": "str",
            "prior_claim_sk": "str", "changed_fields": "json", "recorded_at": "datetime",
            "resubmission_type": "str", "tenant_id": "str",
        },
    ),
    "remittance": _spec(
        "remittance", "remittance_sk",
        "Claim/line decisions, denial/adjustment, payment, reference and settlement dates",
        {
            "remittance_sk": "str", "claim_sk": "str", "line_sk": "str", "decision": "str",
            "denial_code": "str", "adjustment": "float", "payment_amount": "float",
            "payment_reference": "str", "settlement_date": "date", "tenant_id": "str",
        },
    ),
    "prescription_dispense": _spec(
        "prescription_dispense", "rx_sk",
        "Order, authorisation, prescribed/approved/billed/dispensed product and quantity",
        {
            "rx_sk": "str", "claim_sk": "str", "prescriber_id": "str", "pharmacy_id": "str",
            "prescribed_product": "str", "billed_product": "str", "dispensed_product": "str",
            "prescribed_qty": "float", "dispensed_qty": "float", "days_supply": "float",
            "fill_date": "date", "tenant_id": "str",
            "member_sk": "str", "line_sk": "str", "authorization_id": "str",
            "prescribed_date": "date", "billed_qty": "float", "billed_amount": "float",
            "strength_mg": "float", "form": "str", "dose_mg_per_day": "float",
        },
    ),
    "policy_event": _spec(
        "policy_event", "policy_event_sk",
        "Enrolment/add/delete/correction event, actor and timestamp",
        {
            "policy_event_sk": "str", "coverage_id": "str", "event_type": "str",
            "actor": "str", "event_time": "datetime", "effective_date": "date",
            "agent_id": "str", "tenant_id": "str",
        },
    ),
    "review_outcome": _spec(
        "review_outcome", "review_id",
        "Disposition, validated category, confirmed amount, rationale and appeal result",
        {
            "review_id": "str", "case_id": "str", "signal_ids": "str", "reviewer": "str",
            "reviewer_role": "str", "disposition": "str", "validated_category": "str",
            "confirmed_amount_aed": "float", "rationale": "str", "decided_at": "datetime",
            "appeal_raised": "bool", "appeal_result": "str", "appeal_decided_at": "datetime",
            "turnaround_hours": "float", "tenant_id": "str",
        },
    ),
}


@dataclass
class CanonicalDataset:
    """The populated canonical model plus an honest population report."""

    tables: dict[str, pd.DataFrame] = field(default_factory=dict)
    status: dict[str, str] = field(default_factory=dict)
    reasons: dict[str, str] = field(default_factory=dict)
    source_system: str = ""
    tenant_id: str = "T001"
    raw_hashes: dict[str, str] = field(default_factory=dict)

    @classmethod
    def empty(cls, source_system: str = "", tenant_id: str = "T001") -> "CanonicalDataset":
        ds = cls(source_system=source_system, tenant_id=tenant_id)
        for name, spec in CANONICAL_TABLES.items():
            ds.tables[name] = spec.empty_frame()
            ds.status[name] = PopulationStatus.NOT_POPULATED
            ds.reasons[name] = "Not yet populated by any adapter."
        return ds

    def set(self, name: str, frame: pd.DataFrame, status: str, reason: str) -> None:
        if name not in CANONICAL_TABLES and name not in _supplementary():
            raise KeyError(
                f"{name} is neither one of the sixteen canonical tables nor a declared "
                f"supplementary table (fwa.canonical.supplementary)."
            )
        self.tables[name] = frame
        self.status[name] = status
        self.reasons[name] = reason

    def mark_not_populated(self, name: str, reason: str) -> None:
        spec = _spec_for(name)
        self.tables[name] = spec.empty_frame()
        self.status[name] = PopulationStatus.NOT_POPULATED
        self.reasons[name] = reason

    def __getitem__(self, name: str) -> pd.DataFrame:
        return self.tables[name]

    def get(self, name: str) -> pd.DataFrame:
        if name in self.tables:
            return self.tables[name]
        return _spec_for(name).empty_frame()

    def is_populated(self, name: str) -> bool:
        """True when the table holds rows the source actually supplied.

        A supplementary table that no adapter touched has no status at all and
        is therefore not populated — the same answer as an explicit
        ``NOT_POPULATED``, which is what keeps every control that needs one
        classified out on a claim-header file.
        """
        if self.status.get(name) not in (PopulationStatus.POPULATED, PopulationStatus.PARTIAL):
            return False
        frame = self.tables.get(name)
        return frame is not None and not frame.empty

    def supplementary_report(self) -> pd.DataFrame:
        """One row per supplementary (operational or reference) table.

        Kept apart from :meth:`population_report`, whose seventeen rows are
        manuscript Table 3.5 and do not grow.
        """
        rows = []
        for name, spec in _supplementary().items():
            frame = self.tables.get(name)
            rows.append({
                "table": name,
                "status": self.status.get(name, PopulationStatus.NOT_POPULATED),
                "rows": 0 if frame is None else len(frame),
                "reason": self.reasons.get(name, "Not supplied by this source."),
                "required_content": spec.required_content,
            })
        return pd.DataFrame(rows)

    def population_report(self) -> pd.DataFrame:
        """One row per canonical table: status, rows, reason, required content."""
        rows = []
        for name, spec in CANONICAL_TABLES.items():
            frame = self.tables.get(name)
            rows.append(
                {
                    "table": name,
                    "primary_key": spec.primary_key,
                    "status": self.status.get(name, PopulationStatus.NOT_POPULATED),
                    "rows": 0 if frame is None else len(frame),
                    "columns_populated": 0 if frame is None or frame.empty else int(
                        sum(1 for c in frame.columns if frame[c].notna().any())
                    ),
                    "columns_defined": len(spec.columns),
                    "reason": self.reasons.get(name, ""),
                    "required_content": spec.required_content,
                }
            )
        return pd.DataFrame(rows)

    def key_linkage_rate(self) -> float:
        """Data-readiness gate metric (Table 6.1: ≥99.5% key linkage).

        Measured as the share of claim_header rows whose member and provider
        foreign keys both resolve to a row in the corresponding dimension.
        """
        claims = self.get("claim_header")
        if claims.empty:
            return 0.0
        members = set(self.get("member")["member_sk"].dropna().astype(str))
        providers = set(self.get("provider")["provider_sk"].dropna().astype(str))
        if not members or not providers:
            return 0.0
        linked = (
            claims["member_sk"].astype(str).isin(members)
            & claims["provider_sk"].astype(str).isin(providers)
        )
        return float(linked.mean())

    def line_linkage_rate(self) -> float | None:
        """Share of claim_line rows whose claim_sk resolves to a claim_header row.

        ``None`` when the file has no claim lines: the line half of the
        data-readiness gate is then vacuous, and saying so beats reporting 100%.
        """
        lines = self.get("claim_line")
        if lines.empty or "claim_sk" not in lines.columns:
            return None
        claims = set(self.get("claim_header")["claim_sk"].dropna().astype(str))
        if not claims:
            return 0.0
        return float(lines["claim_sk"].astype(str).isin(claims).mean())


def _supplementary() -> dict[str, TableSpec]:
    from .supplementary import SUPPLEMENTARY_TABLES  # local: supplementary imports TableSpec

    return SUPPLEMENTARY_TABLES


def _spec_for(name: str) -> TableSpec:
    if name in CANONICAL_TABLES:
        return CANONICAL_TABLES[name]
    supplementary = _supplementary()
    if name in supplementary:
        return supplementary[name]
    raise KeyError(f"{name} is not a canonical or supplementary table.")
