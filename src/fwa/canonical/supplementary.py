"""Supplementary tables: the operational and reference data many controls need.

The sixteen tables of manuscript Table 3.5 (plus ``review_outcome``) are the
canonical *claim* model. A large part of the control catalogue also depends on
information that is not a claim at all — a bundling edit table, a clinician
roster, a pharmacy's stock records, a regulator's licence register. The
catalogue names these in ``required_canonical_fields`` (``bundling_edit_table.*``,
``clinician_roster.leave_periods`` …), and until now nothing in the code could
hold them, so every control that needed one was classified unrunnable on any
dataset whatever it carried.

They are kept **separate** from :data:`~fwa.canonical.model.CANONICAL_TABLES`
on purpose: the canonical population report is a thesis deliverable with a
fixed seventeen rows, and a reader comparing it with Table 3.5 should not find
it has grown. :meth:`CanonicalDataset.supplementary_report` lists these.

Honesty rule, unchanged: a supplementary table the source does not supply is
``NOT_POPULATED`` with a reason, never filled with invented rows. The only
dataset that populates them is the clearly-labelled SYNTHETIC UAE demo built by
``tools/make_uae_demo_dataset.py``.
"""

from __future__ import annotations

from .model import TableSpec

__all__ = ["SUPPLEMENTARY_TABLES", "OPERATIONAL_TABLES", "REFERENCE_TABLES"]


def _t(name: str, pk: str, content: str, cols: dict[str, str]) -> TableSpec:
    return TableSpec(name=name, primary_key=pk, required_content=content, columns=cols)


#: Operational records produced by payers, providers, regulators and members.
OPERATIONAL_TABLES: dict[str, TableSpec] = {
    "document": _t(
        "document", "document_sk",
        "Clinical and administrative documents attached to a claim, with language and OCR confidence",
        {
            "document_sk": "str", "claim_sk": "str", "member_sk": "str", "provider_sk": "str",
            "doc_type": "str", "language": "str", "text": "str", "ocr_confidence": "float",
            "created_at": "datetime", "version_no": "int", "prior_document_sk": "str",
            "attachment_ref": "str", "is_synthetic": "bool", "tenant_id": "str",
        },
    ),
    "referral": _t(
        "referral", "referral_sk",
        "Referrals from one clinician or facility to another",
        {
            "referral_sk": "str", "referrer_id": "str", "recipient_id": "str",
            "referrer_provider_sk": "str", "recipient_provider_sk": "str", "member_sk": "str",
            "referral_date": "date", "direction": "str", "specialty": "str",
            "reason_code": "str", "resulting_claim_sk": "str", "tenant_id": "str",
        },
    ),
    "contract": _t(
        "contract", "contract_sk",
        "Provider contracts: effective dates, tariff basis and negotiated discount",
        {
            "contract_sk": "str", "provider_sk": "str", "payer_id": "str",
            "valid_from": "date", "valid_to": "date", "tariff_basis": "str",
            "discount_pct": "float", "network_tier": "str", "tenant_id": "str",
        },
    ),
    "tariff": _t(
        "tariff", "(activity_code, network_tier, valid_from)",
        "Allowed price per activity code by network tier and effective date",
        {
            "activity_code": "str", "network_tier": "str", "allowed_price": "float",
            "valid_from": "date", "valid_to": "date", "payer_id": "str",
        },
    ),
    "adjudication_event": _t(
        "adjudication_event", "event_sk",
        "Adjudication actions: who decided, what, when, and any override reason",
        {
            "event_sk": "str", "claim_sk": "str", "actor": "str", "actor_role": "str",
            "event_type": "str", "decision": "str", "override_reason": "str",
            "amount_before": "float", "amount_after": "float", "event_time": "datetime",
            "system_edit_result": "str", "tenant_id": "str",
        },
    ),
    "clinician_roster": _t(
        "clinician_roster", "(clinician_id, provider_sk)",
        "Clinicians, their licence, specialty, privileges and leave",
        {
            "clinician_id": "str", "provider_sk": "str", "full_name_token": "str",
            "specialty": "str", "role": "str", "licence_no": "str",
            "licence_valid_from": "date", "licence_valid_to": "date",
            "privileges": "str", "leave_periods": "json", "emirate": "str", "tenant_id": "str",
        },
    ),
    "member_confirmation": _t(
        "member_confirmation", "confirmation_sk",
        "Member responses to service-verification requests",
        {
            "confirmation_sk": "str", "member_sk": "str", "claim_sk": "str",
            "service_confirmed": "bool", "response": "str", "response_date": "date",
            "channel": "str", "tenant_id": "str",
        },
    ),
    "member_receipt": _t(
        "member_receipt", "receipt_sk",
        "Receipts members hold for what they paid the provider",
        {
            "receipt_sk": "str", "member_sk": "str", "claim_sk": "str", "provider_sk": "str",
            "amount_paid_aed": "float", "item_description": "str", "receipt_date": "date",
            "tenant_id": "str",
        },
    ),
    "complaint": _t(
        "complaint", "complaint_sk",
        "Complaints from members about providers or charges",
        {
            "complaint_sk": "str", "member_sk": "str", "provider_sk": "str", "claim_sk": "str",
            "complaint_type": "str", "complaint_date": "date", "text": "str", "tenant_id": "str",
        },
    ),
    "attendance_record": _t(
        "attendance_record", "record_sk",
        "Independent attendance evidence (e.g. check-in logs) for a member at a provider",
        {
            "record_sk": "str", "member_sk": "str", "provider_sk": "str", "claim_sk": "str",
            "attendance_date": "date", "attended": "bool", "source": "str", "tenant_id": "str",
        },
    ),
    "device_inventory": _t(
        "device_inventory", "serial_number",
        "Devices and implants: serial number, condition, rental/purchase and useful life",
        {
            "serial_number": "str", "device_code": "str", "provider_sk": "str",
            "condition": "str", "acquisition": "str", "useful_life_days": "int",
            "claim_sk": "str", "line_sk": "str", "member_sk": "str", "issued_date": "date",
            "tenant_id": "str",
        },
    ),
    "equipment_inventory": _t(
        "equipment_inventory", "(provider_sk, equipment_type)",
        "Scarce equipment each facility holds, and how long one use takes",
        {
            "provider_sk": "str", "equipment_type": "str", "units": "int",
            "activity_codes": "str", "minutes_per_use": "int", "hours_per_day": "int",
        },
    ),
    "pharmacy_inventory": _t(
        "pharmacy_inventory", "(pharmacy_id, product, period)",
        "Pharmacy stock movements by product and month",
        {
            "pharmacy_id": "str", "product": "str", "period": "str",
            "opening_stock": "float", "purchased_qty": "float", "closing_stock": "float",
            "tenant_id": "str",
        },
    ),
    "other_payer_remittance": _t(
        "other_payer_remittance", "(claim_sk, other_payer_id)",
        "Payments made by other insurers on the same event",
        {
            "claim_sk": "str", "other_payer_id": "str", "payment_amount": "float",
            "settlement_date": "date", "cross_payer_match_token": "str", "tenant_id": "str",
        },
    ),
    "coordination_of_benefits": _t(
        "coordination_of_benefits", "(member_sk, primary_payer)",
        "Which insurer pays first when a member holds more than one policy",
        {
            "member_sk": "str", "primary_payer": "str", "secondary_payer": "str",
            "valid_from": "date", "valid_to": "date", "tenant_id": "str",
        },
    ),
    "third_party_liability": _t(
        "third_party_liability", "claim_sk",
        "Accident and third-party liability records",
        {
            "claim_sk": "str", "member_sk": "str", "accident_type": "str",
            "liable_party": "str", "reported_date": "date", "tenant_id": "str",
        },
    ),
    "third_party_settlement": _t(
        "third_party_settlement", "(claim_sk, settlement_date)",
        "Recoveries already received from a liable third party",
        {
            "claim_sk": "str", "settlement_amount": "float", "settlement_date": "date",
            "payer": "str", "tenant_id": "str",
        },
    ),
    "recovery_ledger": _t(
        "recovery_ledger", "ledger_sk",
        "Provider refunds and credits owed to the payer, and whether they were applied",
        {
            "ledger_sk": "str", "provider_sk": "str", "claim_sk": "str",
            "credit_amount": "float", "credit_date": "date", "applied": "bool",
            "applied_date": "date", "tenant_id": "str",
        },
    ),
    "employer_roster": _t(
        "employer_roster", "(employer_id, member_sk)",
        "Employer-supplied lists of eligible employees and dependants",
        {
            "employer_id": "str", "member_sk": "str", "sponsor_id": "str",
            "relationship": "str", "valid_from": "date", "valid_to": "date", "tenant_id": "str",
        },
    ),
    "policy_application": _t(
        "policy_application", "application_sk",
        "What the applicant declared when the policy was bought",
        {
            "application_sk": "str", "member_sk": "str", "coverage_id": "str",
            "application_date": "date", "declared_conditions": "str",
            "declared_prior_cover": "bool", "agent_id": "str", "tenant_id": "str",
        },
    ),
    "prior_coverage_history": _t(
        "prior_coverage_history", "(member_sk, prior_payer, valid_from)",
        "Cover the member held with earlier insurers, and conditions treated then",
        {
            "member_sk": "str", "prior_payer": "str", "valid_from": "date", "valid_to": "date",
            "conditions_treated": "str", "tenant_id": "str",
        },
    ),
}

#: Maintained policy and coding reference tables.
REFERENCE_TABLES: dict[str, TableSpec] = {
    "activity_code_reference": _t(
        "activity_code_reference", "activity_code",
        "Every activity code: family, level, type, restrictions and time basis",
        {
            "activity_code": "str", "description": "str", "activity_type": "str",
            "service_family": "str", "code_family": "str", "code_level": "int",
            "sex_restriction": "str", "age_min": "int", "age_max": "int",
            "minutes": "int", "is_time_based": "bool", "specialty_required": "str",
            "facility_types": "str", "complexity": "str", "is_telehealth_eligible": "bool",
            "is_scarce_equipment": "bool", "is_implant": "bool", "unit_price_reference": "float",
        },
    ),
    "code_system_version": _t(
        "code_system_version", "(code, code_system)",
        "Effective dates of each code in its code system",
        {"code": "str", "code_system": "str", "valid_from": "date", "valid_to": "date"},
    ),
    "unit_maximum_policy": _t(
        "unit_maximum_policy", "activity_code",
        "Maximum billable units per activity code per day",
        {"activity_code": "str", "max_units_per_day": "float", "rationale": "str"},
    ),
    "bundling_edit_table": _t(
        "bundling_edit_table", "(column_1_code, column_2_code)",
        "Pairs of codes that may not be billed together, and whether a modifier can separate them",
        {
            "column_1_code": "str", "column_2_code": "str", "edit_type": "str",
            "modifier_allowed": "bool", "valid_from": "date", "valid_to": "date",
        },
    ),
    "package_definition": _t(
        "package_definition", "(package_code, component_code)",
        "Package or case-rate codes and the components they already include",
        {"package_code": "str", "component_code": "str", "expected_zero_price": "bool"},
    ),
    "panel_definition": _t(
        "panel_definition", "(panel_code, component_code)",
        "Laboratory panels and their component tests",
        {"panel_code": "str", "component_code": "str"},
    ),
    "dx_proc_prohibition_table": _t(
        "dx_proc_prohibition_table", "(diagnosis_prefix, activity_code)",
        "Diagnosis and procedure pairs that are clinically incompatible",
        {"diagnosis_prefix": "str", "activity_code": "str", "reason": "str"},
    ),
    "indication_policy": _t(
        "indication_policy", "activity_code",
        "Diagnoses that support each activity code",
        {"activity_code": "str", "allowed_diagnosis_prefixes": "str"},
    ),
    "repeat_interval_policy": _t(
        "repeat_interval_policy", "activity_code",
        "Minimum clinically expected interval between repeats of a service",
        {"activity_code": "str", "min_interval_days": "int", "requires_result_before_repeat": "bool"},
    ),
    "care_pathway_policy": _t(
        "care_pathway_policy", "(activity_code, required_prior_code)",
        "Services that must be preceded by another within a window",
        {"activity_code": "str", "required_prior_code": "str", "within_days": "int"},
    ),
    "payment_policy": _t(
        "payment_policy", "(policy_key, service_family)",
        "Payment policies: readmission bundling windows, inpatient-only codes and similar",
        {"policy_key": "str", "service_family": "str", "value": "float", "description": "str"},
    ),
    "document_requirement_policy": _t(
        "document_requirement_policy", "(service_family, doc_type)",
        "Documents that must accompany a service",
        {"service_family": "str", "activity_code": "str", "doc_type": "str"},
    ),
    "medical_necessity_policy": _t(
        "medical_necessity_policy", "activity_code",
        "Facts that must appear in the record to support a high-cost service",
        {"activity_code": "str", "required_terms": "str"},
    ),
    "disguise_risk_policy": _t(
        "disguise_risk_policy", "activity_code",
        "Covered codes commonly used to disguise excluded (cosmetic, non-medical) services",
        {"activity_code": "str", "excluded_service": "str", "risk_diagnosis_prefixes": "str"},
    ),
    "drug_policy": _t(
        "drug_policy", "product",
        "Products: equivalence group, strength, form, maximum duration and dose limits",
        {
            "product": "str", "description": "str", "equivalence_group": "str",
            "therapeutic_class": "str", "strength_mg": "float", "form": "str",
            "unit_price": "float", "max_duration_days": "int",
            "max_mg_per_kg_day": "float", "is_high_cost": "bool", "is_controlled": "bool",
            "vial_size_mg": "float", "indication_prefixes": "str",
        },
    ),
    "code_equivalence_map": _t(
        "code_equivalence_map", "(code, equivalence_group)",
        "Codes and products that are clinically equivalent",
        {"code": "str", "equivalence_group": "str"},
    ),
    "step_therapy_policy": _t(
        "step_therapy_policy", "product",
        "Products that may only be used after a first-line product has been tried",
        {"product": "str", "required_prior_group": "str", "lookback_days": "int"},
    ),
    "drg_grouper": _t(
        "drg_grouper", "diagnosis_code",
        "Stand-in grouper: which secondary diagnoses raise the payment band, and by how much",
        {"diagnosis_code": "str", "is_cc": "bool", "is_mcc": "bool", "severity_weight": "float"},
    ),
    "morbidity_model": _t(
        "morbidity_model", "(principal_chapter, secondary_code)",
        "Expected prevalence of each secondary condition for a given principal diagnosis chapter",
        {"principal_chapter": "str", "secondary_code": "str", "expected_prevalence": "float"},
    ),
}

#: Every supplementary table, operational first.
SUPPLEMENTARY_TABLES: dict[str, TableSpec] = {**OPERATIONAL_TABLES, **REFERENCE_TABLES}
