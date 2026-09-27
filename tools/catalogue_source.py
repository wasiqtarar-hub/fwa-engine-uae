"""Appendix C, transcribed — the source of truth for ``rules/**/*.yaml``.

This module is the *transcription* of manuscript Appendix C: all 39 scenarios
and all 164 atomic controls, with each control's ID, scenario, type/stage,
implementable trigger, configuration/exclusions and output/disposition kept
exactly as the specification states them (the ``trigger``, ``config`` and
``outcome`` fields below are verbatim).

Everything *beside* those verbatim fields is this artefact's own governed
mapping, and is marked as such:

``disp``
    The catalogue's Output/disposition column is prose ("Reject/pend with
    licence period"). A control must declare exactly ONE of the eight governed
    dispositions (§3.3, Table 3.2). The mapping rule is mechanical and stated
    once here so it can be checked: **take the first disposition named in the
    prose**, map it to its Table 3.2 value, and record the rest as
    ``alternate_dispositions`` (advisory; the engine never acts on them). Where
    the first-named disposition would be illegal for the control's type under
    §3.3, it is replaced by the first legal one and a ``governance_note``
    records the substitution. There are two such substitutions in the whole
    catalogue and both are listed in the README.

``support`` / ``why``
    The honest classification of what ``claims.csv`` can carry, and a one-line
    reason. This is a deliverable in its own right (build brief §11.2).

``impl``
    The implementing function in :mod:`fwa.engine.controls`, or ``None``.

``req``
    The canonical fields that would be needed to make the control executable.
    This is what turns ``control_coverage_matrix.csv`` into a concrete answer to
    "what would real Shafafiya/eClaimLink data unlock?".

Regenerate the YAML with::

    python tools/gen_rules.py
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Scenario metadata: id -> (title, priority, added_by_coverage_audit)
# The eight scenarios marked with * in Appendix C were added by the
# specification's own coverage audit beyond the earlier 31-scenario catalogue.
# ---------------------------------------------------------------------------

SCENARIOS: dict[str, tuple[str, str, bool]] = {
    "ENT-01": ("Coverage, benefit and network eligibility", "P0", False),
    "ENT-02": ("Member identity misuse, card sharing and post-mortem billing", "P0", False),
    "ENT-03": ("Provider/facility licence, exclusion and privilege", "P0", False),
    "ENT-04": ("Rendering identity and clinician activity integrity", "P1", False),
    "ENT-05": ("Encounter type/place-of-service misrepresentation", "P0", False),
    "ENT-06": ("Telehealth encounter and downstream-order integrity", "P1", True),
    "PAY-01": ("Exact, near and cross-submission duplicate billing", "P0", False),
    "PAY-02": ("Unbundling and inclusive/package/DRG leakage", "P0", False),
    "PAY-03": ("Code validity, demographics, units and time billing", "P0", False),
    "PAY-04": ("Prior-authorization compliance and scope", "P0", False),
    "PAY-05": ("Modifier, indicator and edit-bypass abuse", "P1", False),
    "PAY-06": ("Tariff, contract and billed-amount manipulation", "P0", False),
    "PAY-07": ("Patient-share, copay, deductible and balance billing", "P0", False),
    "PAY-08": ("Denial-resubmission mutation and gaming", "P0", False),
    "PAY-09": ("Disguised non-covered service or product", "P0", True),
    "PAY-10": ("Coordination of benefits and third-party liability", "P1", True),
    "PAY-11": ("Internal adjudication, override and payment manipulation", "P1", True),
    "PAY-12": ("Reversal, refund, credit-balance and remittance leakage", "P1", True),
    "CLN-01": ("Upcoding and code-intensity inflation", "P0", False),
    "CLN-02": ("Diagnosis, comorbidity and DRG severity manipulation", "P0", False),
    "CLN-03": ("Clinical incompatibility and specialty mismatch", "P1", False),
    "CLN-04": ("Unnecessary, excessive or repeated services", "P0", False),
    "CLN-05": ("Length-of-stay, admission and readmission irregularity", "P1", False),
    "CLN-06": ("Phantom billing/service not rendered", "P0", False),
    "CLN-07": ("Provider capacity and throughput impossibility", "P1", False),
    "CLN-08": ("Laboratory, pathology and genetic-testing integrity", "P1", True),
    "PHR-01": ("Prescribed, authorized, billed and dispensed mismatch", "P0", False),
    "PHR-02": ("Early refill, stockpiling and multi-prescriber convergence", "P0", False),
    "PHR-03": ("High-cost drug/infusion/specialty-product spike", "P1", False),
    "PHR-04": ("Prescriber-pharmacy steering and reciprocal concentration", "P1", False),
    "PHR-05": ("Equipment, implant, consumable and supply integrity", "P1", True),
    "DOC-01": ("Documentation-to-billing/authorization inconsistency", "P1", False),
    "DOC-02": ("Cloned, templated or retrospectively altered documentation", "P2", False),
    "NET-01": ("Referral concentration, reciprocity and downstream conversion", "P1", False),
    "NET-02": ("Collusion ring and shared-identifier linkage", "P1", False),
    "NET-03": ("Provider-member collusion and inducement", "P1", False),
    "NET-04": ("Broker, agent and employer-group concentration", "P2", False),
    "ANL-01": ("Multivariate anomaly and behavioral change", "P1", False),
    "POL-01": ("Enrollment, employer-group and policy manipulation", "P1", True),
}

# Appendix C stage notation -> manuscript Table 3.4 stage
STAGE_MAP = {
    "INGEST": "INGEST",
    "PREPAY": "PREPAY_SYNC",
    "PREPAY_ASYNC": "PREPAY_ASYNC",
    "ASYNC": "PREPAY_ASYNC",
    "DAILY": "POSTPAY_DAILY",
    "WEEKLY": "NETWORK_WEEKLY",
    "MONTHLY": "MODEL_MONTHLY",
    "QUARTERLY": "MODEL_MONTHLY",
}

EXEC = "EXECUTABLE"
PART = "PARTIAL"
NOPE = "NOT_EXECUTABLE_ON_THIS_DATASET"


def c(rid, name, types, stage, trigger, config, outcome, disp, rc, owner,
      support, why, impl=None, alt=(), req=(), note="", score=None,
      population=None, inputs=(), expression=None, exclusions=(),
      grouping=None, evidence=(), params=None):
    """One control record. Positional args mirror the Appendix C columns."""
    return dict(
        rid=rid, name=name, types=types, stage=STAGE_MAP[stage], trigger=trigger,
        config=config, outcome=outcome, disp=disp, rc=rc, owner=owner,
        support=support, why=why, impl=impl, alt=list(alt), req=list(req), note=note,
        score=score, population=population, inputs=list(inputs), expression=expression,
        exclusions=list(exclusions), grouping=grouping, evidence=list(evidence),
        params=params or {},
    )


# Canonical fields that recur as "what would unlock this"
F_LINE = ["claim_line.activity_code", "claim_line.units", "claim_line.net_amount"]
F_AUTH = ["authorization.*", "authorization_line.*", "claim_line.authorization_id"]
F_DOC = ["observation.attachment_ref", "document.text", "document.language"]
F_REMIT = ["remittance.decision", "remittance.denial_code", "remittance.payment_amount"]
F_LINEAGE = ["claim_version.relationship", "claim_version.prior_claim_sk", "claim_version.changed_fields"]
F_BENEFIT = ["benefit_rule_version.covered", "benefit_rule_version.benefit_limit",
             "benefit_rule_version.authorization_required"]
F_LICENCE = ["provider_status_period.status_type", "provider_status_period.valid_from",
             "provider_status_period.valid_to", "provider.regulator_id"]
F_GEO = ["provider.emirate", "encounter.location"]
F_CLINICIAN = ["claim_line.rendering_clinician_id", "claim_line.ordering_clinician_id",
               "provider.specialty"]
F_RX = ["prescription_dispense.billed_product", "prescription_dispense.days_supply",
        "prescription_dispense.prescriber_id", "prescription_dispense.pharmacy_id"]
F_DEMO = ["member.date_of_birth", "member.sex"]
F_IDENT = ["provider.owner_entity_id", "provider.bank_account_token", "provider.phone_token",
           "provider.address_token"]

CATALOGUE: list[dict] = []

# =============================================================================
# ENT — Entitlement, identity, provider status, encounter setting  (25 controls)
# =============================================================================

CATALOGUE += [
    # ---- ENT-01 ------------------------------------------------------------
    c("ENT-01-R01", "Coverage inactive", "H", "PREPAY",
      "service_time NOT BETWEEN coverage.valid_from AND coverage.valid_to",
      "Grace/newborn/emergency and retroactive-update policy",
      "Reject/pend; show coverage interval",
      "REJECT", "COVERAGE_INACTIVE", "claims_policy", NOPE,
      "Coverage END date is unknown in claims.csv — inception is derivable from "
      "days_since_policy_start but termination is not, so the control could only ever "
      "evaluate to 'covered'. A control that cannot fail is not a control.",
      alt=["PREPAY_PEND"],
      req=["coverage_period.valid_to", "coverage_period.status"] + F_BENEFIT),

    c("ENT-01-R02", "Benefit not covered", "H", "PREPAY",
      "expected_benefit(activity_family, product).covered = false",
      "Mandatory benefits, approved exception, medical-tourism/self-pay route",
      "Reject with benefit version",
      "REJECT", "BENEFIT_NOT_COVERED", "claims_policy", NOPE,
      "No benefit schedule exists in the source: benefit_rule_version is NOT_POPULATED, "
      "so there is nothing to evaluate 'covered = false' against.",
      req=F_BENEFIT + F_LINE),

    c("ENT-01-R03", "Benefit limit exceeded", "H", "PREPAY",
      "paid_or_authorized_ytd + requested_payable > benefit_limit",
      "Reset period, family/member basis, reversals, pending authorizations",
      "Reprice or pend excess only",
      "REPRICE", "BENEFIT_LIMIT_EXCEEDED", "claims_policy", NOPE,
      "No benefit limits and no paid-to-date accumulator in the source.",
      alt=["PREPAY_PEND"], req=F_BENEFIT + F_REMIT),

    c("ENT-01-R04", "Network/referral violation", "H", "PREPAY",
      "Provider/route outside effective network or required referral missing",
      "Emergency, access-gap and regulator exceptions",
      "Reject/pend with failed route",
      "REJECT", "NETWORK_VIOLATION", "provider_network", NOPE,
      "No network membership, no referral records and no provider network status in the source.",
      alt=["PREPAY_PEND"], req=["coverage_period.network", "provider_status_period.status_value"]),

    # ---- ENT-02 ------------------------------------------------------------
    c("ENT-02-R01", "Authoritative identity conflict", "H", "INGEST",
      "Protected national-ID token maps to different enrolled member or DOB/sex conflicts "
      "with authoritative record",
      "Placeholder IDs, newborns, corrected identity; names alone prohibited",
      "Pend for identity resolution",
      "PREPAY_PEND", "IDENTITY_CONFLICT", "claims_policy", NOPE,
      "claims.csv has one opaque patient_id and no DOB, sex or national-ID token, so there "
      "is no authoritative record to conflict with.",
      req=F_DEMO + ["member.protected_id_token"]),

    c("ENT-02-R02", "Post-mortem service", "H", "PREPAY",
      "service_start > confirmed_death_date + cfg.reporting_tolerance",
      "Death-source confidence; administrative claims after service permitted",
      "Pend/reject with source confidence",
      "PREPAY_PEND", "POST_MORTEM_SERVICE", "claims_policy", NOPE,
      "No death status or death date in the source.",
      alt=["REJECT"], req=["member.death_date", "member.death_source", "member.death_source_confidence"]),

    c("ENT-02-R03", "Member impossible presence", "S", "PREPAY_ASYNC",
      "Same member has overlapping encounters or travel speed above cfg.max_speed",
      "Inpatient ancillary lines, telehealth, claim date without time",
      "Member lead with encounter pairs",
      "PREPAY_PEND", "MEMBER_IMPOSSIBLE_PRESENCE", "claims_policy", PART,
      "The OVERLAPPING-ENCOUNTER limb runs: admission and discharge dates exist, so two "
      "inpatient stays for one member that overlap in time are detectable. The TRAVEL-SPEED "
      "limb does not run — there is no geography. Dates carry no time component, so a "
      "same-day transfer cannot be distinguished from a genuine overlap and confidence is "
      "reduced accordingly.",
      impl="ent_02_r03_member_impossible_presence",
      population="claim pairs where member_sk is equal and the encounters overlap in time",
      inputs=["encounter.admission_date", "encounter.discharge_date", "claim_header.member_sk",
              "claim_header.provider_sk"],
      expression="overlaps(a.admission_date, a.discharge_date, b.admission_date, b.discharge_date, "
                 "tolerance_days=cfg.concurrent_admission_overlap_tolerance_days) is TRUE "
                 "and a.provider_sk != b.provider_sk",
      exclusions=["inpatient_ancillary_lines", "telehealth_encounter", "claim_date_without_time",
                  "same_provider_transfer_within_facility"],
      grouping=["tenant_id", "payer_id", "member_sk", "scenario_id", "period_bucket"],
      evidence=["member_sk", "overlapping_claims", "admission_dates", "discharge_dates",
                "overlap_days", "providers"],
      params={"tolerance_days": "cfg.concurrent_admission_overlap_tolerance_days",
              "max_concurrent": "cfg.max_concurrent_admissions_per_member"},
      req=F_GEO + ["encounter.start_time (with time of day)"]),

    c("ENT-02-R04", "Card-sharing utilization pattern", "S", "DAILY",
      "Incompatible concurrent geography/provider clusters or abrupt demographic-inconsistent "
      "service pattern",
      "Chronic/rare disease pathways; require minimum independent evidence",
      "SIU lead, never auto-reject",
      "SIU_LEAD", "CARD_SHARING_PATTERN", "siu", NOPE,
      "Both limbs need data the source lacks: geography clusters (no location) and "
      "demographic inconsistency (no demographics).",
      req=F_GEO + F_DEMO),

    # ---- ENT-03 ------------------------------------------------------------
    c("ENT-03-R01", "Inactive/expired licence", "H", "PREPAY",
      "Rendering, ordering, dispensing or facility licence not active at service time",
      "Role-specific licence requirement and approved grace",
      "Reject/pend with licence period",
      "REJECT", "LICENCE_INACTIVE", "provider_network", NOPE,
      "No licence periods in the source. provider_status_period holds only an undated "
      "blacklist flag, and 'active AT SERVICE TIME' cannot be evaluated without dates.",
      alt=["PREPAY_PEND"], req=F_LICENCE),

    c("ENT-03-R02", "Excluded/watch-listed entity", "H", "PREPAY",
      "Effective match on regulator or approved internal exclusion list",
      "Exact ID match required for reject; fuzzy match only pends",
      "Reject or SIU lead",
      "PREPAY_PEND", "EXCLUDED_ENTITY", "provider_network", PART,
      "provider_blacklist_flag is the only exclusion-list signal available and it is "
      "UNDATED, so 'effective match at service time' cannot be evaluated. The catalogue "
      "permits REJECT only on an exact, effective-dated identifier match; with no effective "
      "date the governed disposition is downgraded to PREPAY_PEND, which is the treatment "
      "the catalogue itself prescribes for a non-exact match.",
      impl="ent_03_r02_excluded_entity",
      note="Disposition downgraded from REJECT to PREPAY_PEND because the exclusion source "
           "carries no effective date. Recorded rather than silently applied.",
      alt=["SIU_LEAD"],
      population="claim where provider_blacklist_flag is true",
      inputs=["claim_header.provider_sk", "provider_status_period.status_value",
              "provider_status_period.valid_from", "provider_status_period.valid_to"],
      expression="provider_status_period.status_value == 'BLACKLISTED' "
                 "as_of claim.service_date and match_type == 'EXACT'",
      exclusions=["fuzzy_match_pends_only", "approved_internal_exception", "delisted_before_service_date"],
      grouping=["tenant_id", "payer_id", "provider_sk", "scenario_id", "period_bucket"],
      evidence=["provider_sk", "status_value", "match_type", "exclusion_source",
                "effective_dating_available"],
      params={"requires_exact_match": "cfg.blacklist_requires_exact_match"},
      req=F_LICENCE + ["provider_status_period.valid_from (effective dating for the exclusion)"]),

    c("ENT-03-R03", "Hard privilege violation", "H", "PREPAY",
      "Activity outside regulator/facility privilege explicitly marked prohibited",
      "Effective privilege source and exception",
      "Reject/pend",
      "REJECT", "PRIVILEGE_VIOLATION", "provider_network", NOPE,
      "No privilege table and no activity codes.",
      alt=["PREPAY_PEND"], req=F_LICENCE + F_LINE),

    c("ENT-03-R04", "Soft specialty mismatch", "E", "PREPAY_ASYNC",
      "Activity outside curated specialty map but not legally prohibited",
      "cfg.specialty_map, cross-specialty privileges, sample-based validation",
      "Coding review, not reject",
      "PREPAY_PEND", "SPECIALTY_MISMATCH", "coding_policy", NOPE,
      "No provider specialty and no activity codes; the ICD-chapter proxy used for peer "
      "grouping is a service line, not a licensed specialty, and is too weak to assert a "
      "mismatch against.",
      req=F_CLINICIAN + F_LINE),

    c("ENT-03-R05", "Dormant/new/non-operational provider burst", "S", "DAILY",
      "New/reactivated provider immediately exceeds specialty volume/value percentile or lacks "
      "expected operating footprint",
      "Credentialing date, acquisition/merger and new contract",
      "SIU/provider-enrollment lead",
      "SIU_LEAD", "PROVIDER_BURST", "siu", NOPE,
      "No credentialing or contract-start date, so 'new or reactivated' cannot be "
      "established. First-appearance-in-file is not the same fact and would flag every "
      "provider in the first month of the file.",
      req=["provider.credentialing_date", "provider_status_period.valid_from"]),

    # ---- ENT-04 ------------------------------------------------------------
    c("ENT-04-R01", "Missing/invalid rendering clinician", "H", "PREPAY",
      "Service family requires rendering clinician and ID is null, facility-only, or invalid",
      "Role requirement by activity/encounter",
      "Return/pend",
      "RETURN", "RENDERING_CLINICIAN_MISSING", "claims_policy", NOPE,
      "No clinician identifiers of any kind in the source.",
      alt=["PREPAY_PEND"], req=F_CLINICIAN),

    c("ENT-04-R02", "Billing-rendering role conflict", "H/E", "PREPAY",
      "Billing/rendering/ordering/supervising combination violates explicit role rule",
      "Incident-to/team care and supervision policy",
      "Pend with role conflict",
      "PREPAY_PEND", "ROLE_CONFLICT", "claims_policy", NOPE,
      "No clinician role fields.", req=F_CLINICIAN),

    c("ENT-04-R03", "Concurrent clinician services", "S", "DAILY",
      "Clinician has overlapping time-dependent services beyond cfg.max_concurrency",
      "Team procedures, anesthesia concurrency, date-only records",
      "Provider case with timeline",
      "POSTPAY_AUDIT", "CLINICIAN_CONCURRENCY", "provider_network", NOPE,
      "No clinician identifiers and no service times. The facility-level analogue is "
      "implemented instead as CLN-07-R02.",
      req=F_CLINICIAN + ["claim_line.service_start_time", "claim_line.service_end_time"]),

    c("ENT-04-R04", "Geographic or leave impossibility", "S", "DAILY",
      "Clinician claims during confirmed leave/out-of-country period or impossible facility travel",
      "Timestamp/source reliability",
      "SIU lead if repeated",
      "SIU_LEAD", "CLINICIAN_IMPOSSIBILITY", "siu", NOPE,
      "No clinicians, no rosters, no geography.",
      req=F_CLINICIAN + F_GEO + ["clinician_roster.leave_periods"]),

    # ---- ENT-05 ------------------------------------------------------------
    c("ENT-05-R01", "Facility-type incompatibility", "H", "PREPAY",
      "Claimed encounter/activity requires a facility type not matching licence",
      "Mobile/home/telehealth exceptions",
      "Reject/pend",
      "REJECT", "FACILITY_TYPE_INCOMPATIBLE", "provider_network", NOPE,
      "Facility type is absent and encounter type is INFERRED, not sourced.",
      alt=["PREPAY_PEND"], req=["provider.facility_type", "encounter.encounter_type"] + F_LINE),

    c("ENT-05-R02", "Admission evidence conflict", "H/E", "PREPAY_ASYNC",
      "Inpatient/ICU/day-case billing lacks admission, bed, discharge or required Observation trail",
      "Data latency and transferred cases",
      "Pend for records",
      "PREPAY_PEND", "ADMISSION_EVIDENCE_MISSING", "clinical_policy", NOPE,
      "Every row HAS an admission and discharge date by construction, and there is no "
      "Observation trail to be missing, so the control can never fire. Firing it on the "
      "absence of a table that is absent for every row would be noise, not detection.",
      req=["observation.*", "encounter.bed_id", "encounter.encounter_type"]),

    c("ENT-05-R03", "Related-claim setting conflict", "E", "DAILY",
      "Professional and facility claims for same episode disagree on setting",
      "Cross-facility transfer, independent practitioner",
      "Correlated episode case",
      "POSTPAY_AUDIT", "SETTING_CONFLICT", "clinical_policy", NOPE,
      "No professional/facility claim split and no setting field.",
      req=["encounter.encounter_type", "claim_header.claim_type"]),

    c("ENT-05-R04", "Setting-shift anomaly", "S", "MONTHLY",
      "Provider's better-paid setting share rises beyond changepoint and peer threshold",
      "Contract/facility change segmentation",
      "Postpay audit",
      "POSTPAY_AUDIT", "SETTING_SHIFT", "analytics", NOPE,
      "No setting field, so there is no setting share to observe a shift in.",
      req=["encounter.encounter_type"]),

    # ---- ENT-06 (added by the coverage audit) ------------------------------
    c("ENT-06-R01", "Telehealth eligibility/setting failure", "H", "PREPAY",
      "Telehealth code billed for ineligible service, member geography or provider privilege",
      "Regulator/product telehealth rules",
      "Reject/pend",
      "REJECT", "TELEHEALTH_INELIGIBLE", "claims_policy", NOPE,
      "No telehealth indicator, no activity codes, no geography.",
      alt=["PREPAY_PEND"], req=F_LINE + F_GEO + ["encounter.encounter_type"]),

    c("ENT-06-R02", "No meaningful clinical interaction", "E/T", "ASYNC",
      "Remote order has no qualifying encounter/note or interaction duration below policy minimum",
      "Asynchronous-care rules and emergencies",
      "Clinical review",
      "PREPAY_PEND", "TELEHEALTH_NO_INTERACTION", "clinical_policy", NOPE,
      "No notes, no interaction durations, no telehealth flag.",
      req=F_DOC + ["encounter.duration_minutes"]),

    c("ENT-06-R03", "Downstream referral concentration", "S/N", "WEEKLY",
      "Telehealth provider sends share above peer threshold to one lab/pharmacy/device supplier",
      "Corporate ownership and narrow network",
      "Network lead",
      "SIU_LEAD", "TELEHEALTH_DOWNSTREAM_CONCENTRATION", "siu", NOPE,
      "No referral edges, no downstream supplier identity, no telehealth flag. The general "
      "concentration analogue is implemented as NET-01-R01.",
      req=["referral.referrer_id", "referral.recipient_id"] + F_RX),

    c("ENT-06-R04", "Remote-order conversion spike", "S", "DAILY",
      "Downstream high-cost test/drug/device conversion materially exceeds adjusted peers",
      "Specialty, diagnosis and campaign/new-service change",
      "Postpay/SIU lead",
      "POSTPAY_AUDIT", "REMOTE_ORDER_CONVERSION", "analytics", NOPE,
      "No downstream order linkage and no telehealth flag.",
      req=F_LINE + F_RX + ["referral.*"]),
]

# =============================================================================
# PAY — Payment integrity  (50 controls)
# =============================================================================

CATALOGUE += [
    # ---- PAY-01 duplicates -------------------------------------------------
    c("PAY-01-R01", "Exact line duplicate", "H", "PREPAY",
      "Same payer/member/provider/code/date/units/amount and no valid repeat indicator in "
      "active lineage",
      "Replacement/cancelled line, bilateral/repeat rule",
      "Reject later line",
      "REJECT", "EXACT_DUPLICATE", "claims_policy", PART,
      "Runs at CLAIM-HEADER level, not line level: the match key is payer + member + provider "
      "+ primary diagnosis + admission date + discharge date + exact requested amount. Without "
      "activity codes and units this is a strictly coarser key than the catalogue specifies, "
      "so it can miss a duplicated line inside a differing claim. Because it is an exact, "
      "objective, hard-edit condition it retains the REJECT disposition — but only the later "
      "claim in the active lineage is rejected.",
      impl="pay_01_r01_exact_duplicate",
      population="claim_header within one tenant and payer",
      inputs=["claim_header.payer_id", "claim_header.member_sk", "claim_header.provider_sk",
              "diagnosis.code", "encounter.admission_date", "encounter.discharge_date",
              "claim_header.gross_amount"],
      expression="an earlier active claim exists with identical payer, member, provider, "
                 "principal diagnosis, admission date, discharge date and gross amount "
                 "(tolerance cfg.exact_duplicate_amount_tolerance) and neither claim is "
                 "superseded or cancelled in the lineage index",
      exclusions=["replacement_or_cancelled_line", "valid_repeat_indicator",
                  "bilateral_procedure_rule", "superseded_by_correction"],
      grouping=["tenant_id", "payer_id", "member_sk", "episode_id", "scenario_id"],
      evidence=["duplicate_of_claim_sk", "match_key", "gross_amount_aed", "admission_date",
                "discharge_date", "provider_sk", "lineage_status"],
      params={"amount_tolerance": "cfg.exact_duplicate_amount_tolerance"},
      req=F_LINE + ["claim_line.service_date", "claim_line.units"]),

    c("PAY-01-R02", "Near duplicate", "H", "ASYNC",
      "Same clinical service with changed claim ID, amount, units or code equivalent within "
      "cfg.window",
      "Staged/repeat services and code-equivalence map",
      "Pend with matched line",
      "PREPAY_PEND", "NEAR_DUPLICATE", "claims_policy", PART,
      "Runs on member + provider + primary diagnosis within cfg.duplicate_window_days with "
      "amounts within cfg.duplicate_amount_tolerance. There is no code-equivalence map, so "
      "'code equivalent' is approximated by identical primary diagnosis, which is weaker.",
      impl="pay_01_r02_near_duplicate",
      population="claim_header pairs for the same member and provider inside the duplicate window",
      inputs=["claim_header.member_sk", "claim_header.provider_sk", "diagnosis.code",
              "encounter.admission_date", "claim_header.gross_amount"],
      expression="within_days(a.admission_date, b.admission_date, cfg.duplicate_window_days) is TRUE "
                 "and a.diagnosis_code == b.diagnosis_code "
                 "and abs(a.gross - b.gross) / max(a.gross, b.gross) <= cfg.duplicate_amount_tolerance",
      exclusions=["staged_or_repeat_service", "code_equivalence_map_permits", "planned_series_of_care"],
      grouping=["tenant_id", "payer_id", "member_sk", "episode_id", "scenario_id"],
      evidence=["matched_claim_sk", "days_apart", "amount_difference_aed", "diagnosis_code",
                "provider_sk"],
      params={"window_days": "cfg.duplicate_window_days",
              "amount_tolerance": "cfg.duplicate_amount_tolerance"},
      req=F_LINE + ["code_equivalence_map"]),

    c("PAY-01-R03", "Split-claim duplicate", "H", "DAILY",
      "Component/identical service appears across claim headers for same episode",
      "Facility/professional legitimate split",
      "Reprice/pend episode",
      "PREPAY_PEND", "SPLIT_CLAIM_DUPLICATE", "claims_policy", PART,
      "Episodes are built from same-member readmission proximity, so a multi-claim episode "
      "with a repeated primary diagnosis is detectable. 'Component service' is not — there "
      "are no components. Disposition is downgraded to PREPAY_PEND because a correct "
      "repriced amount cannot be COMPUTED without line-level detail, and permits "
      "REPRICE only where the correct payable amount is computable.",
      impl="pay_01_r03_split_claim_duplicate",
      note="Disposition downgraded from REPRICE to PREPAY_PEND: the correct payable amount "
           "is not computable without claim lines, and defines REPRICE as the case "
           "where it is.",
      alt=["PREPAY_PEND"],
      population="multi-claim episodes",
      inputs=["episode.claim_sks", "diagnosis.code", "claim_header.gross_amount"],
      expression="episode contains two or more claims sharing the same principal diagnosis "
                 "at the same provider",
      exclusions=["facility_professional_legitimate_split", "planned_staged_procedure",
                  "cross_provider_transfer"],
      grouping=["tenant_id", "payer_id", "member_sk", "episode_id", "scenario_id"],
      evidence=["episode_id", "claim_sks", "diagnosis_code", "episode_total_gross_aed",
                "crosses_providers", "linkage_confidence"],
      req=F_LINE + ["claim_header.claim_type (facility vs professional)"]),

    c("PAY-01-R04", "Cross-payer duplicate", "H/S", "DAILY",
      "Privacy-preserving member/service match paid by multiple payers",
      "Only authorized data-sharing/COB data; payment status required",
      "Recovery/coordination case",
      "POSTPAY_AUDIT", "CROSS_PAYER_DUPLICATE", "claims_policy", NOPE,
      "Single-payer extract with no payment status. num_insurers_same_event indicates that "
      "other insurers exist but carries no matching service record to compare against; that "
      "weaker fact is handled by PAY-10-R01 instead.",
      req=F_REMIT + ["cross_payer_match_token", "coordination_of_benefits.*"]),

    # ---- PAY-02 unbundling -------------------------------------------------
    c("PAY-02-R01", "Same-claim component edit", "H", "PREPAY",
      "Parent and separately payable component match effective bundling edit",
      "Permitted indicator and edit version",
      "Reprice/reject component",
      "REPRICE", "BUNDLING_EDIT", "claims_policy", NOPE,
      "Requires claim lines and a bundling edit table; neither exists.",
      alt=["REJECT"], req=F_LINE + ["bundling_edit_table.*"]),

    c("PAY-02-R02", "Cross-claim/cross-provider component", "H/E", "DAILY",
      "Included component billed on related claim in global/episode window",
      "Contractual professional/facility split",
      "Episode reprice/pend",
      "REPRICE", "CROSS_CLAIM_COMPONENT", "claims_policy", NOPE,
      "Requires component-level identification of services within an episode.",
      alt=["PREPAY_PEND"], req=F_LINE + ["bundling_edit_table.*"]),

    c("PAY-02-R03", "Package completeness/zero-price lines", "H", "PREPAY",
      "Required package activities absent, or included activity has non-zero charge",
      "Documented medical omission and local claiming rule",
      "Return/reprice",
      "RETURN", "PACKAGE_INCOMPLETE", "claims_policy", NOPE,
      "No package definitions and no lines.",
      alt=["REPRICE"], req=F_LINE + ["package_definition.*"]),

    c("PAY-02-R04", "DRG/inclusive leakage", "H", "DAILY",
      "FFS line separately paid although included in DRG/package",
      "Effective inclusions/exclusions, carve-outs",
      "Recovery candidate",
      "POSTPAY_AUDIT", "DRG_LEAKAGE", "claims_policy", NOPE,
      "No DRG assignment, no lines, no payment records.",
      req=F_LINE + F_REMIT + ["drg_grouper.*"]),

    c("PAY-02-R05", "Novel unbundling pattern", "S", "MONTHLY",
      "Provider has abnormal profitable component co-occurrence not in known edit table",
      "Minimum volume and peer adjustment",
      "Policy-review candidate, not denial",
      "MONITOR_ONLY", "NOVEL_UNBUNDLING", "policy_owner", NOPE,
      "Depends on the code-pair co-occurrence feature that names explicitly and that "
      "no claim-level aggregate can substitute for. The claim_line feature level is "
      "NOT_POPULATED, so this control has no input at all.",
      req=F_LINE),

    # ---- PAY-03 code validity ---------------------------------------------
    c("PAY-03-R01", "Invalid/effective-date code", "H", "INGEST",
      "Code absent from declared code system/version or inactive at service date",
      "Approved unlisted-code pathway and required Observation",
      "Return",
      "RETURN", "INVALID_CODE", "coding_policy", NOPE,
      "Only a primary ICD-10 diagnosis exists, drawn from a closed set of 20 codes that are "
      "all valid. There are no procedure codes to validate and no code-system version history.",
      req=F_LINE + ["code_system_version.*"]),

    c("PAY-03-R02", "Hard demographic impossibility", "H", "PREPAY",
      "Explicit age/sex restriction fails",
      "Gender/clinical exceptions approved by policy",
      "Reject/pend",
      "REJECT", "DEMOGRAPHIC_IMPOSSIBILITY", "coding_policy", NOPE,
      "No member demographics. Note that O80 (normal delivery) appears in the data but "
      "without a sex field there is nothing to check it against.",
      alt=["PREPAY_PEND"], req=F_DEMO + F_LINE),

    c("PAY-03-R03", "Unit maximum", "H/E", "PREPAY",
      "Daily/episode units exceed administrative or clinical maximum",
      "Repeat/laterality, dose, inpatient rules",
      "Reprice or clinical pend",
      "REPRICE", "UNIT_MAXIMUM_EXCEEDED", "coding_policy", NOPE,
      "claims.csv carries one billed amount per claim and no service lines, so there is no "
      "unit count to compare against a daily or episode maximum.",
      alt=["PREPAY_PEND"], req=F_LINE + ["unit_maximum_policy.*"]),

    c("PAY-03-R04", "Time-unit inconsistency", "H/E", "PREPAY",
      "Time-derived units do not match start/end time or overlap impossible services",
      "Rounding policy, anesthesia concurrency",
      "Reprice/pend",
      "REPRICE", "TIME_UNIT_INCONSISTENT", "coding_policy", NOPE,
      "No service times and no units.",
      alt=["PREPAY_PEND"], req=F_LINE + ["claim_line.service_start_time"]),

    # ---- PAY-04 authorisation ---------------------------------------------
    c("PAY-04-R01", "Missing required authorization", "H", "PREPAY",
      "Activity requires authorization and no valid link exists",
      "Emergency, deemed approval, response-time exception",
      "Pend/reject",
      "PREPAY_PEND", "AUTH_MISSING", "claims_policy", NOPE,
      "The authorization table is NOT_POPULATED and there is no authorization_required flag. "
      "This is the single largest coverage gap relative to a real Shafafiya feed: the whole "
      "PAY-04 scenario (5 controls) is unrunnable, and PAY-04-R01 is the worked example the "
      "atomic-control contract is usually explained through.",
      alt=["REJECT"], req=F_AUTH + F_BENEFIT),

    c("PAY-04-R02", "Invalid timing/status", "H", "PREPAY",
      "Service outside authorization validity or authorization denied/cancelled",
      "Approved extension and urgent retrospective route",
      "Pend/reject",
      "PREPAY_PEND", "AUTH_INVALID_TIMING", "claims_policy", NOPE,
      "No authorisation records.", alt=["REJECT"], req=F_AUTH),

    c("PAY-04-R03", "Code/provider/facility scope mismatch", "H", "PREPAY",
      "Claim dimension not equal/equivalent to approved dimension",
      "Equivalent-code map and approved transfer",
      "Pend",
      "PREPAY_PEND", "AUTH_SCOPE_MISMATCH", "claims_policy", NOPE,
      "No authorisation records.", req=F_AUTH + F_LINE),

    c("PAY-04-R04", "Quantity/value exhaustion", "H", "PREPAY",
      "Cumulative claimed/paid plus current exceeds approved units/value",
      "Cancelled/reversed claims and partial fills",
      "Reprice/pend excess",
      "REPRICE", "AUTH_EXHAUSTED", "claims_policy", NOPE,
      "No authorisation records and no paid-to-date accumulator.",
      alt=["PREPAY_PEND"], req=F_AUTH + F_REMIT),

    c("PAY-04-R05", "Authorization reuse", "H/S", "DAILY",
      "Same authorization consumed by unrelated members/episodes/providers or after exhaustion",
      "Family/shared authorization only if policy permits",
      "SIU/prepay lead",
      "SIU_LEAD", "AUTH_REUSE", "siu", NOPE,
      "No authorisation identifiers to reuse.", req=F_AUTH),

    # ---- PAY-05 modifiers --------------------------------------------------
    c("PAY-05-R01", "Prohibited code-indicator pair", "H", "PREPAY",
      "Pair absent/prohibited in effective policy table",
      "Local indicator rules; never import foreign rules blindly",
      "Return/reject",
      "RETURN", "PROHIBITED_INDICATOR_PAIR", "coding_policy", NOPE,
      "No modifiers or indicators in the source.",
      alt=["REJECT"], req=F_LINE + ["claim_line.indicator"]),

    c("PAY-05-R02", "Required support absent", "E/T", "ASYNC",
      "Enabling indicator requires Observation/report/narrative and artifact is missing",
      "Document arrival latency",
      "Pend",
      "PREPAY_PEND", "INDICATOR_SUPPORT_MISSING", "clinical_policy", NOPE,
      "No indicators and no documents.", req=F_LINE + F_DOC),

    c("PAY-05-R03", "Provider use-rate outlier", "S", "MONTHLY",
      "Indicator rate exceeds shrunk specialty peer percentile",
      "Minimum denominator, facility/payment model",
      "Postpay audit",
      "POSTPAY_AUDIT", "INDICATOR_RATE_OUTLIER", "analytics", NOPE,
      "No indicators, so there is no indicator rate to compare.",
      req=F_LINE + ["claim_line.indicator"]),

    c("PAY-05-R04", "Post-edit migration", "S", "MONTHLY",
      "Usage spikes after new edit and disproportionately converts denials to payment",
      "Rule launch and provider education periods",
      "Payment-integrity case",
      "POSTPAY_AUDIT", "POST_EDIT_MIGRATION", "analytics", NOPE,
      "Requires denial outcomes and an edit-launch timeline; neither exists.",
      req=F_REMIT + F_LINE),

    # ---- PAY-06 tariff and amount -----------------------------------------
    c("PAY-06-R01", "Tariff/contract price variance", "H", "PREPAY",
      "Submitted price basis differs from effective allowed price outside rounding tolerance",
      "Negotiated carve-outs, VAT and currency policy",
      "Reprice",
      "REPRICE", "TARIFF_VARIANCE", "claims_policy", NOPE,
      "No tariff or contract price table, so there is no allowed price to compare against. "
      "The peer-relative analogue is PAY-06-R03, which is a statistical control and "
      "therefore may not reprice.",
      req=["tariff.allowed_price", "contract.*"] + F_LINE),

    c("PAY-06-R02", "Gross/net/share arithmetic failure", "H", "INGEST",
      "Amount relationships violate transaction or contract equation",
      "Approved adjustments and other-insurer amounts",
      "Return",
      "RETURN", "AMOUNT_ARITHMETIC_FAILURE", "claims_policy", PART,
      "The full equation (gross − discount − patient_share = net) cannot be checked because "
      "patient_share is absent. The one limb that CAN be checked is checked: approved must "
      "not exceed requested, and neither may be negative. That is a genuine transaction-"
      "integrity edit and it runs at INGEST.",
      impl="pay_06_r02_amount_arithmetic",
      population="every claim_header at ingest",
      inputs=["claim_header.gross_amount", "claim_header.approved_amount"],
      expression="approved_amount > gross_amount or gross_amount < 0 or approved_amount < 0",
      exclusions=["approved_adjustment_uplift", "other_insurer_amount_included",
                  "contractual_true_up"],
      grouping=["tenant_id", "payer_id", "claim_sk", "scenario_id"],
      evidence=["gross_amount_aed", "approved_amount_aed", "difference_aed", "equation_checked"],
      req=["claim_header.patient_share", "claim_header.discount"]),

    c("PAY-06-R03", "Billed amount peer outlier", "S", "DAILY",
      "Code-level amount above contract-adjusted peer threshold",
      "Case mix, units, implant carve-out",
      "Pend only with material exposure",
      "PREPAY_PEND", "BILLED_AMOUNT_PEER_OUTLIER", "analytics", EXEC,
      "Runs as specified except that the peer comparison is at PRIMARY-DIAGNOSIS level "
      "rather than activity-code level, because there are no activity codes. Robust "
      "median/MAD residuals are used, not mean/SD z-scores, because claim amounts are "
      "heavily right-skewed.",
      impl="pay_06_r03_billed_amount_peer_outlier",
      population="claim_header with a resolvable peer group",
      inputs=["claim_header.gross_amount_aed", "diagnosis.code", "claim_header.tpa",
              "claim_header.policy_type", "provider.volume_band"],
      expression="robust_residual(gross_amount_aed, peer_group) > cfg.robust_residual_flag_threshold "
                 "and exposure is material",
      exclusions=["case_mix_difference", "units_not_available", "implant_carve_out",
                  "insufficient_peer_evidence"],
      grouping=["tenant_id", "payer_id", "provider_sk", "scenario_id", "period_bucket"],
      evidence=["gross_amount_aed", "peer_median_aed", "peer_mad", "robust_residual",
                "peer_level_used", "peer_n", "diagnosis_code"],
      params={"threshold": "cfg.robust_residual_flag_threshold"},
      req=["claim_line.activity_code (for code-level rather than diagnosis-level peers)",
           "tariff.allowed_price"]),

    c("PAY-06-R04", "Approved-to-billed pattern", "S", "MONTHLY",
      "Persistent low approval ratio or suspiciously invariant full approval versus peers",
      "Adjudication model and contract differences",
      "Education, audit or PAY-11 link",
      "PROVIDER_EDUCATION", "APPROVAL_RATIO_PATTERN", "analytics", EXEC,
      "Both limbs run: requested and approved amounts exist for every claim, so a provider's "
      "approval-ratio level and its VARIANCE are both computable against shrunk peers.",
      impl="pay_06_r04_approval_ratio_pattern",
      alt=["POSTPAY_AUDIT"],
      population="provider-month with at least cfg.min_entity_opportunities claims",
      inputs=["claim_header.gross_amount_aed", "claim_header.approved_amount_aed",
              "claim_header.provider_sk"],
      expression="shrunk_mean_approval_ratio < quantile(peer shrunk_approval_ratio, "
                 "cfg.approval_ratio_peer_percentile) or variance(approval_ratio) < "
                 "quantile(peer approval_ratio variance, "
                 "cfg.approval_ratio_invariance_percentile)",
      exclusions=["adjudication_model_difference", "contract_difference",
                  "insufficient_peer_evidence", "single_claim_provider"],
      grouping=["tenant_id", "payer_id", "provider_sk", "scenario_id", "period_bucket"],
      evidence=["provider_sk", "shrunk_approval_ratio", "posterior_interval", "peer_mean",
                "approval_ratio_variance", "claim_count", "peer_level_used", "limb_fired"],
      params={"low_percentile": "cfg.approval_ratio_peer_percentile",
              "invariance_percentile": "cfg.approval_ratio_invariance_percentile",
              "min_claims": "cfg.min_entity_opportunities",
              "interval_mass": "cfg.posterior_interval_mass",
              "max_posterior_width": "cfg.max_posterior_width"},
      req=["remittance.decision (to separate adjudication effects from billing behaviour)"]),

    # ---- PAY-07 patient share ---------------------------------------------
    c("PAY-07-R01", "Expected-share mismatch", "H", "PREPAY",
      "Submitted claim/Activity share differs from benefit calculation",
      "Deductible accumulator, exemptions, rounding",
      "Reprice/return",
      "REPRICE", "PATIENT_SHARE_MISMATCH", "claims_policy", NOPE,
      "patient_share is absent from claims.csv and there is no benefit calculation to "
      "compare it against. NOTE: PAY-07 is sometimes described as an 'approval-ratio "
      "anomaly'; the control catalogue defines it as patient-share/copay/balance billing. "
      "The catalogue is followed, and the approval-ratio control is implemented where the "
      "catalogue actually places it, at PAY-06-R04.",
      alt=["RETURN"], req=["claim_header.patient_share"] + F_BENEFIT),

    c("PAY-07-R02", "Share shifted to payer", "H", "PREPAY",
      "gross - valid_discount - patient_share != net or waived share included in net",
      "Approved discount/hardship",
      "Reprice",
      "REPRICE", "SHARE_SHIFTED_TO_PAYER", "claims_policy", NOPE,
      "The equation needs patient_share and discount, both absent.",
      req=["claim_header.patient_share", "claim_header.discount"]),

    c("PAY-07-R03", "Systematic zero/rounded share", "S", "MONTHLY",
      "Provider zero-share or repeated-rounding rate exceeds product-adjusted peers",
      "Zero-share benefits and regulator programs",
      "Postpay audit",
      "POSTPAY_AUDIT", "SYSTEMATIC_ZERO_SHARE", "analytics", NOPE,
      "No patient share to be systematically zero.", req=["claim_header.patient_share"]),

    c("PAY-07-R04", "Excess patient charge/balance bill", "H/S", "DAILY",
      "Receipt/complaint/collection data exceeds approved patient liability",
      "Non-covered elective items with documented consent",
      "Member protection/provider case",
      "PREPAY_PEND", "BALANCE_BILLING", "claims_policy", NOPE,
      "No receipts, complaints or patient-liability records.",
      alt=["POSTPAY_AUDIT"], req=["member_receipt.*", "complaint.*", "claim_header.patient_share"]),

    # ---- PAY-08 resubmission ----------------------------------------------
    c("PAY-08-R01", "Missing/broken lineage", "H", "INGEST",
      "Correction/resubmission lacks valid prior payer ID/type or references unrelated claim",
      "Legacy route where permitted",
      "Return",
      "RETURN", "BROKEN_LINEAGE", "claims_policy", NOPE,
      "claim_version is NOT_POPULATED — every claim_id in the source is unique and appears "
      "once, so there is no lineage to be broken.",
      req=F_LINEAGE),

    c("PAY-08-R02", "Reimbursement-enabling field mutation", "H/E", "ASYNC",
      "Denied line changes diagnosis, code, units, provider, setting or indicator without "
      "coherent correction evidence",
      "Accepted correction matrix by denial code",
      "Pend with field diff",
      "PREPAY_PEND", "FIELD_MUTATION", "claims_policy", NOPE,
      "No resubmissions and no denial codes. The field-diff rendering this control needs IS "
      "implemented (LineageIndex.field_diff) and is exercised by the integration test "
      "against constructed fixtures.",
      req=F_LINEAGE + F_REMIT),

    c("PAY-08-R03", "Repeated resubmission loop", "S", "DAILY",
      "Attempts per underlying service exceed cfg.max_attempts or alternate variants until paid",
      "Internal complaint/appeal route",
      "Provider case",
      "POSTPAY_AUDIT", "RESUBMISSION_LOOP", "claims_policy", NOPE,
      "No resubmission attempts recorded.", req=F_LINEAGE),

    c("PAY-08-R04", "Edit-learning success anomaly", "S", "MONTHLY",
      "Provider's mutated resubmission success materially exceeds peers and concentrates on "
      "specific denial edits",
      "Biller/vendor and policy-change adjustment",
      "SIU/payment-integrity lead",
      "SIU_LEAD", "EDIT_LEARNING_ANOMALY", "siu", NOPE,
      "No resubmissions, no denial codes, no payment outcomes.",
      req=F_LINEAGE + F_REMIT),

    # ---- PAY-09 disguised non-covered --------------------------------------
    c("PAY-09-R01", "Code-description/document conflict", "E/T", "ASYNC",
      "Note/order/Observation describes an uncovered service while claim uses covered code",
      "NLP confidence and clinician review",
      "Pend",
      "PREPAY_PEND", "DISGUISED_SERVICE_DOC_CONFLICT", "clinical_policy", NOPE,
      "Requires both real documents and a covered/non-covered benefit map. The synthetic "
      "document pipeline built for DOC-01 cannot substitute, because the benefit map that "
      "defines 'uncovered' does not exist.",
      req=F_DOC + F_BENEFIT),

    c("PAY-09-R02", "Covered-code substitution pattern", "S", "MONTHLY",
      "Provider uses a payable proxy code unusually often around denied/non-covered services",
      "Code-family and denial-history map",
      "Postpay audit",
      "POSTPAY_AUDIT", "CODE_SUBSTITUTION_PATTERN", "analytics", NOPE,
      "No procedure codes and no denial history.", req=F_LINE + F_REMIT),

    c("PAY-09-R03", "Cosmetic/alternative/non-medical disguise", "E", "PREPAY",
      "Diagnosis, setting and product combination matches approved disguise-risk policy",
      "Reconstructive/medical-necessity exceptions",
      "Clinical pend",
      "PREPAY_PEND", "DISGUISE_RISK_COMBINATION", "clinical_policy", NOPE,
      "Needs a curated disguise-risk policy table plus setting and product, none of which exist.",
      req=F_LINE + ["encounter.encounter_type", "disguise_risk_policy.*"]),

    c("PAY-09-R04", "Member/provider confirmation mismatch", "S", "DAILY",
      "Member receipt/confirmation describes different item/service than billed",
      "Confirmation reliability and consent",
      "SIU lead",
      "SIU_LEAD", "MEMBER_CONFIRMATION_MISMATCH", "siu", NOPE,
      "No member confirmations or receipts.", req=["member_receipt.*", "member_confirmation.*"]),

    # ---- PAY-10 coordination of benefits -----------------------------------
    c("PAY-10-R01", "Other coverage not coordinated", "H", "PREPAY",
      "Active primary/other coverage exists but claim submitted with wrong payer order",
      "UAE/product COB policy",
      "Pend/reprice",
      "PREPAY_PEND", "COB_NOT_COORDINATED", "claims_policy", PART,
      "num_insurers_same_event > 1 establishes that other coverage EXISTS, which is the "
      "precondition the control needs, but payer ORDER cannot be evaluated: there is no "
      "primary/secondary designation and no other-payer payment record. The signal is "
      "therefore raised as 'coordination evidence required', not as a wrong-order finding, "
      "and its confidence reflects that.",
      impl="pay_10_r01_cob_not_coordinated",
      alt=["REPRICE"],
      population="claim where num_insurers_same_event >= cfg.cob_min_insurers",
      inputs=["claim_header.num_insurers_same_event", "claim_header.gross_amount_aed",
              "coverage_period.*"],
      expression="num_insurers_same_event >= cfg.cob_min_insurers and no coordination record exists",
      exclusions=["uae_product_cob_policy_permits", "lawful_top_up_cover",
                  "coordination_already_evidenced"],
      grouping=["tenant_id", "payer_id", "member_sk", "episode_id", "scenario_id"],
      evidence=["num_insurers_same_event", "gross_amount_aed", "member_sk",
                "coordination_record_present", "payer_order_evaluable"],
      params={"min_insurers": "cfg.cob_min_insurers"},
      req=["coverage_period.payer_order", "other_payer_remittance.*"]),

    c("PAY-10-R02", "Multi-payer overpayment", "H", "DAILY",
      "Sum of payer payments + patient liability exceeds allowable charge",
      "Lawful top-up and benefit coordination",
      "Recovery case",
      "POSTPAY_AUDIT", "MULTI_PAYER_OVERPAYMENT", "finance", NOPE,
      "No other-payer payment amounts and no patient liability.",
      req=["other_payer_remittance.payment_amount", "claim_header.patient_share"]),

    c("PAY-10-R03", "Accident/liability indicator missing", "H/E", "PREPAY",
      "Diagnosis/encounter suggests road/work/third-party event but liable-party fields absent",
      "Clinical false-positive list",
      "Pend for coordination data",
      "PREPAY_PEND", "LIABILITY_INDICATOR_MISSING", "claims_policy", NOPE,
      "There are no liable-party fields to be absent, so every trauma claim would fire. "
      "S06.3 and S72.0 are present in the data but flagging all of them would be a "
      "data-gap alert, not a control.",
      req=["claim_header.accident_indicator", "third_party_liability.*"]),

    c("PAY-10-R04", "Duplicate recovery after settlement", "H", "DAILY",
      "Claim paid by health payer after documented third-party settlement without offset",
      "Subrogation/recovery policy",
      "Recovery workflow",
      "POSTPAY_AUDIT", "DUPLICATE_RECOVERY", "finance", NOPE,
      "No settlements, no payments, no subrogation records.",
      req=F_REMIT + ["third_party_settlement.*"]),

    # ---- PAY-11 internal adjudication --------------------------------------
    c("PAY-11-R01", "Unauthorized/manual override", "H", "DAILY",
      "Denial/edit overridden without required role, approval or reason",
      "Emergency escalation and delegated authority",
      "Internal-control case",
      "POSTPAY_AUDIT", "UNAUTHORIZED_OVERRIDE", "administrator", NOPE,
      "No adjudication decisions or override records exist in claims.csv. NOTE: this control "
      "IS live against the artefact's OWN overrides — every manual adjudication override in "
      "the application is a distinct permission and is written to the append-only audit log "
      "with user, timestamp, before/after and reason. The control cannot "
      "run against the source data; the governance property it exists to protect is "
      "implemented in fwa.auth and fwa.audit.",
      req=F_REMIT + ["adjudication_event.actor", "adjudication_event.override_reason"]),

    c("PAY-11-R02", "Adjudicator-provider concentration", "S/N", "MONTHLY",
      "Employee's override/payment volume to provider materially exceeds adjusted peers",
      "Assigned provider books and specialty",
      "Compliance/SIU lead",
      "SIU_LEAD", "ADJUDICATOR_CONCENTRATION", "administrator", NOPE,
      "No adjudicator identity in the source.", req=["adjudication_event.actor"]),

    c("PAY-11-R03", "Post-settlement upward adjustment", "H/S", "DAILY",
      "Paid amount increased after settlement without valid adjustment reason/workflow",
      "Contract true-up and appeal decision",
      "Internal audit case",
      "POSTPAY_AUDIT", "POST_SETTLEMENT_UPLIFT", "finance", NOPE,
      "No settlement dates and no payment history.", req=F_REMIT),

    c("PAY-11-R04", "Suspicious full-pay/invariant decisions", "S", "MONTHLY",
      "Adjudicator/provider pair shows unusually high approval and low variance despite "
      "comparable edits",
      "Auto-adjudicated claims excluded",
      "Internal audit lead",
      "POSTPAY_AUDIT", "INVARIANT_DECISIONS", "administrator", NOPE,
      "No adjudicator identity. The provider-side half of this pattern — invariant approval "
      "ratio — IS implemented, as the second limb of PAY-06-R04.",
      req=["adjudication_event.actor"] + F_REMIT),

    # ---- PAY-12 reversals --------------------------------------------------
    c("PAY-12-R01", "Paid cancelled/reversed claim", "H", "DAILY",
      "Net payment remains after effective cancellation/reversal",
      "Timing tolerance and netting process",
      "Recovery",
      "POSTPAY_AUDIT", "PAID_CANCELLED_CLAIM", "finance", NOPE,
      "No cancellations, reversals or payments.", req=F_REMIT + F_LINEAGE),

    c("PAY-12-R02", "Duplicate remittance/payment reference", "H", "DAILY",
      "Same payable claim/line paid more than once or payment reference reused inconsistently",
      "Split settlement and batch payments",
      "Finance pend/recovery",
      "POSTPAY_AUDIT", "DUPLICATE_REMITTANCE", "finance", NOPE,
      "The remittance table is NOT_POPULATED.", req=F_REMIT),

    c("PAY-12-R03", "Unapplied provider refund/credit", "H", "DAILY",
      "Received credit/refund not linked to outstanding recovery within SLA",
      "Dispute and unapplied-cash queue",
      "Finance exception",
      "POSTPAY_AUDIT", "UNAPPLIED_CREDIT", "finance", NOPE,
      "No credits, refunds or recovery ledger.", req=F_REMIT + ["recovery_ledger.*"]),

    c("PAY-12-R04", "Negative/offset manipulation", "S", "MONTHLY",
      "Provider repeatedly offsets credits against unrelated claims or delays reversals versus peers",
      "Contractual netting rules",
      "Postpay audit",
      "POSTPAY_AUDIT", "OFFSET_MANIPULATION", "finance", NOPE,
      "claims.csv has no remittance, credit or reversal records, so there is nothing that "
      "could show an offset being applied, delayed or netted against another claim.", req=F_REMIT),
]

# =============================================================================
# CLN — Clinical and coding integrity  (36 controls)
# =============================================================================

CATALOGUE += [
    # ---- CLN-01 upcoding ---------------------------------------------------
    c("CLN-01-R01", "High-level code share", "S", "MONTHLY",
      "Highest-paid level share in code family exceeds shrunk peer percentile",
      "Specialty, setting, case mix, minimum n",
      "Provider coding case",
      "POSTPAY_AUDIT", "HIGH_LEVEL_CODE_SHARE", "coding_policy", PART,
      "There are no code levels to take a share of. The implemented PROXY is the provider's "
      "rate of icd_code_matches_procedure = False, shrunk toward the diagnosis-chapter peer "
      "mean by empirical Bayes. That measures coding-integrity pressure, not level "
      "intensity, and the signal says so on its face; it is a weaker fact than the catalogue "
      "control and is classified PARTIAL for that reason.",
      impl="cln_01_r01_coding_mismatch_rate",
      population="provider with at least cfg.min_entity_opportunities claims",
      inputs=["claim_header.icd_code_matches_procedure", "claim_header.provider_sk",
              "diagnosis.chapter"],
      expression="shrunk_rate(coding_mismatch, provider, peer_group) > quantile(peer "
                 "shrunk_rate, cfg.coding_mismatch_peer_percentile) and "
                 "posterior_interval_width <= cfg.max_posterior_width",
      exclusions=["case_mix_difference", "specialty_difference", "setting_difference",
                  "below_minimum_denominator", "posterior_interval_too_wide"],
      grouping=["tenant_id", "payer_id", "provider_sk", "scenario_id", "period_bucket"],
      evidence=["provider_sk", "observed_rate", "shrunk_rate", "posterior_interval", "peer_mean",
                "peer_level_used", "peer_n", "claim_count", "proxy_note"],
      params={"peer_percentile": "cfg.coding_mismatch_peer_percentile",
              "min_claims": "cfg.min_entity_opportunities",
              "interval_mass": "cfg.posterior_interval_mass",
              "max_posterior_width": "cfg.max_posterior_width"},
      req=["claim_line.activity_code (code family and level)", "provider.specialty"]),

    c("CLN-01-R02", "Expected-level residual", "S/M", "DAILY",
      "Billed level exceeds level predicted from diagnoses, age, setting and prior utilization "
      "by cfg.delta",
      "Interpretable ordinal model; no document features at scoring time if unavailable",
      "Prepay sample",
      "PREPAY_PEND", "EXPECTED_LEVEL_RESIDUAL", "coding_policy", PART,
      "Implemented as an interpretable EXPECTED-AMOUNT residual rather than an expected-LEVEL "
      "residual: the expectation is built from primary diagnosis, length of stay and policy "
      "type (age and setting are unavailable), and the residual is robust (median/MAD). This "
      "is the same shape of control against a coarser target, and it is classified PARTIAL "
      "because 'level' does not exist in the source.",
      impl="cln_01_r02_expected_level_residual",
      population="claim_header with a resolvable peer group and non-null LOS",
      inputs=["claim_header.gross_amount_aed", "encounter.length_of_stay_days", "diagnosis.code",
              "claim_header.policy_type"],
      expression="robust_residual(amount_per_los_day, diagnosis_peer_group) > cfg.robust_residual_flag_threshold",
      exclusions=["case_mix_difference", "no_document_features_available", "los_zero_or_missing",
                  "insufficient_peer_evidence"],
      grouping=["tenant_id", "payer_id", "provider_sk", "episode_id", "scenario_id"],
      evidence=["amount_per_los_day", "peer_median", "peer_mad", "robust_residual",
                "expected_amount_aed", "excess_amount_aed", "peer_level_used", "proxy_note"],
      params={"threshold": "cfg.robust_residual_flag_threshold"},
      req=["claim_line.activity_code", "member.date_of_birth", "encounter.encounter_type"]),

    c("CLN-01-R03", "Documentation-level conflict", "E/T", "ASYNC",
      "Extracted work/severity/time elements do not meet policy for billed level",
      "Validated Arabic/English extraction and human review",
      "Coding pend",
      "PREPAY_PEND", "DOCUMENTATION_LEVEL_CONFLICT", "clinical_policy", PART,
      "Runs ONLY against clearly-labelled SYNTHETIC discharge summaries generated by this "
      "artefact (data/synthetic_documents/, every filename and body marked SYNTHETIC). "
      "claims.csv contains no documents. Every finding carries a source span, and a finding "
      "without a span is discarded by the pipeline rather than shown with a caveat.",
      impl="cln_01_r03_documentation_level_conflict",
      population="claims with a generated synthetic discharge summary",
      inputs=["document.text", "document.language", "encounter.length_of_stay_days",
              "diagnosis.code", "claim_header.gross_amount_aed"],
      expression="extracted stay length or extracted diagnosis conflicts with the billed claim "
                 "and extraction_confidence >= cfg.nlp_min_extraction_confidence "
                 "and a source span exists",
      exclusions=["extraction_confidence_below_threshold", "no_source_span_finding_discarded",
                  "language_not_validated", "human_review_required_before_action"],
      grouping=["tenant_id", "payer_id", "claim_sk", "scenario_id"],
      evidence=["source_span_text", "span_offsets", "extracted_value", "billed_value",
                "extraction_confidence", "ocr_confidence", "document_language",
                "synthetic_document_marker"],
      params={"min_confidence": "cfg.nlp_min_extraction_confidence"},
      req=F_DOC + ["real clinical documentation with OCR confidence"]),

    c("CLN-01-R04", "Level migration changepoint", "S", "MONTHLY",
      "Provider abruptly shifts to higher code levels absent case-mix/contract change",
      "Segment known structural changes",
      "Postpay sample audit",
      "POSTPAY_AUDIT", "LEVEL_MIGRATION_CHANGEPOINT", "analytics", NOPE,
      "No code levels to migrate between. The analogous self-history changepoint on billed "
      "AMOUNT is implemented as ANL-01-R02.",
      req=["claim_line.activity_code"]),

    # ---- CLN-02 severity ---------------------------------------------------
    c("CLN-02-R01", "Payment-changing secondary diagnosis", "H/E", "DAILY",
      "Removing suspect secondary diagnosis lowers DRG/payment and code lacks required "
      "supporting evidence",
      "Current grouper and audit policy",
      "Coding audit candidate",
      "POSTPAY_AUDIT", "SECONDARY_DX_PAYMENT_IMPACT", "coding_policy", NOPE,
      "Exactly one principal diagnosis per claim; there are no secondary diagnoses to remove "
      "and no DRG grouper.",
      req=["diagnosis.sequence (secondary codes)", "drg_grouper.*"]),

    c("CLN-02-R02", "Comorbidity prevalence outlier", "S", "MONTHLY",
      "Provider's CC/MCC or local severity-code rate exceeds risk-adjusted peers",
      "DRG, age, transfer, specialty, minimum volume",
      "Provider case",
      "POSTPAY_AUDIT", "COMORBIDITY_PREVALENCE", "analytics", NOPE,
      "No secondary or comorbidity codes.", req=["diagnosis.sequence", "drg_grouper.*"]),

    c("CLN-02-R03", "Prior-history inconsistency", "E/S", "DAILY",
      "Acute/severe diagnosis materially conflicts with longitudinal history and episode evidence",
      "New diagnoses allowed; absence is weak evidence",
      "Prioritize record review",
      "POSTPAY_AUDIT", "PRIOR_HISTORY_INCONSISTENCY", "clinical_policy", NOPE,
      "A member's longitudinal history here is at most six claims with one code each, which "
      "is far too thin to establish a material conflict. The catalogue itself warns that "
      "'absence is weak evidence'; on this data it would be no evidence.",
      req=["diagnosis.sequence", "longitudinal member history across payers"]),

    c("CLN-02-R04", "POA/sequencing conflict", "H/E", "PREPAY",
      "POA or principal/secondary sequence violates effective coding policy",
      "Transfer and obstetric/newborn rules",
      "Coding pend",
      "PREPAY_PEND", "POA_SEQUENCING_CONFLICT", "coding_policy", NOPE,
      "No present-on-admission indicator and no diagnosis sequence.",
      req=["diagnosis.present_on_admission", "diagnosis.sequence"]),

    c("CLN-02-R05", "Severity-mix changepoint", "S", "MONTHLY",
      "Severity index increases without corresponding procedure, LOS or referral shift",
      "Service-line expansion segmentation",
      "Postpay audit",
      "POSTPAY_AUDIT", "SEVERITY_MIX_CHANGEPOINT", "analytics", NOPE,
      "No severity index can be computed without secondary diagnoses.",
      req=["diagnosis.sequence", "drg_grouper.*"]),

    # ---- CLN-03 incompatibility --------------------------------------------
    c("CLN-03-R01", "Impossible diagnosis-procedure pair", "H", "PREPAY",
      "Pair explicitly prohibited by authoritative rule",
      "Only rules marked hard=true",
      "Reject/pend",
      "PREPAY_PEND", "IMPOSSIBLE_DX_PROC_PAIR", "coding_policy", PART,
      "Runs on icd_code_matches_procedure = False, which is a boolean VERDICT with no "
      "procedure behind it: it records that some upstream process judged a mismatch, without "
      "the evidence for that judgement. The disposition is therefore DOWNGRADED from the "
      "catalogue's first-named REJECT to the pend limb: a denial is permitted only on an "
      "objective, effective-dated condition this system can evaluate itself, and another "
      "system's undocumented verdict is not one. The same field also feeds CLN-01-R01 as a "
      "provider-level rate.",
      impl="cln_03_r01_dx_procedure_mismatch",
      note="Disposition downgraded from REJECT to PREPAY_PEND: the trigger is an upstream "
           "boolean verdict, not an objective condition this system can re-derive, so it "
           "cannot support a denial.",
      alt=["REJECT"],
      population="claim where icd_code_matches_procedure is false",
      inputs=["claim_header.icd_code_matches_procedure", "diagnosis.code"],
      expression="icd_code_matches_procedure == false",
      exclusions=["only_rules_marked_hard_true_may_reject", "upstream_verdict_not_re_derivable",
                  "approved_clinical_exception", "procedure_detail_unavailable"],
      grouping=["tenant_id", "payer_id", "claim_sk", "scenario_id"],
      evidence=["claim_sk", "diagnosis_code", "icd_code_matches_procedure", "gross_amount_aed",
                "provider_sk", "what_the_reviewer_must_verify"],
      req=F_LINE + ["dx_proc_prohibition_table.*"]),

    c("CLN-03-R02", "Unsupported indication", "E", "ASYNC",
      "No same-episode or lookback indication in curated policy",
      "Lookback and accepted indication groups",
      "Clinical review",
      "PREPAY_PEND", "UNSUPPORTED_INDICATION", "clinical_policy", NOPE,
      "Requires procedures and a curated indication policy.",
      req=F_LINE + ["indication_policy.*"]),

    c("CLN-03-R03", "Procedure sequence conflict", "E", "DAILY",
      "Prerequisite procedure/result absent or service occurs in impossible order",
      "External services and data latency",
      "Pend/audit",
      "PREPAY_PEND", "PROCEDURE_SEQUENCE_CONFLICT", "clinical_policy", NOPE,
      "No procedures and no results.", req=F_LINE + ["observation.*"]),

    c("CLN-03-R04", "Rare pair/specialty anomaly", "S", "MONTHLY",
      "Provider's rare combinations exceed specialty peer prevalence",
      "Minimum volume; exploratory only",
      "Monitor/policy review",
      "MONITOR_ONLY", "RARE_PAIR_ANOMALY", "analytics", NOPE,
      "No procedure pairs and no licensed specialty.", req=F_LINE + F_CLINICIAN),

    # ---- CLN-04 unnecessary services ---------------------------------------
    c("CLN-04-R01", "Minimum repeat interval", "E", "PREPAY",
      "Same/equivalent service repeated inside policy interval without accepted indicator",
      "Code family, clinical exception and required Observation",
      "Clinical pend",
      "PREPAY_PEND", "MINIMUM_REPEAT_INTERVAL", "clinical_policy", PART,
      "'Same or equivalent service' is approximated by the same primary diagnosis for the "
      "same member, because there are no service codes. The interval test itself runs "
      "exactly as specified against cfg.min_repeat_interval_days.",
      impl="cln_04_r01_minimum_repeat_interval",
      population="claim pairs for the same member with the same principal diagnosis",
      inputs=["claim_header.member_sk", "diagnosis.code", "encounter.admission_date"],
      expression="a prior claim exists for the same member and principal diagnosis with "
                 "0 <= days_between <= cfg.min_repeat_interval_days and no accepted indicator",
      exclusions=["accepted_clinical_indicator", "code_family_permits_repeat",
                  "required_observation_present", "planned_staged_care",
                  "chemotherapy_or_dialysis_pathway"],
      grouping=["tenant_id", "payer_id", "member_sk", "episode_id", "scenario_id"],
      evidence=["prior_claim_sk", "days_between", "diagnosis_code", "provider_sk",
                "prior_provider_sk", "interval_threshold"],
      params={"interval_days": "cfg.min_repeat_interval_days"},
      req=F_LINE + ["observation.*", "repeat_interval_policy.*"]),

    c("CLN-04-R02", "Episode frequency excess", "E/S", "DAILY",
      "Count per episode/member exceeds policy maximum or risk-adjusted peer threshold",
      "Diagnosis/care pathway and age",
      "Pend/audit",
      "PREPAY_PEND", "EPISODE_FREQUENCY_EXCESS", "clinical_policy", PART,
      "Member-level admission counts within the utilisation window are computable and are "
      "compared against cfg.member_repeat_service_threshold. Risk adjustment for age is not "
      "possible (no demographics), so the peer comparison is diagnosis-based only.",
      impl="cln_04_r02_episode_frequency_excess",
      alt=["POSTPAY_AUDIT"],
      population="member with at least one claim in the utilisation window",
      inputs=["claim_header.member_sk", "encounter.admission_date"],
      expression="count(member claims strictly earlier within cfg.utilisation_velocity_window_days) "
                 ">= cfg.member_repeat_service_threshold",
      exclusions=["chronic_care_pathway", "oncology_or_dialysis_pathway", "age_risk_adjustment_unavailable",
                  "maternity_pathway"],
      grouping=["tenant_id", "payer_id", "member_sk", "scenario_id", "period_bucket"],
      evidence=["member_sk", "claims_in_window", "window_days", "threshold", "diagnoses_in_window",
                "providers_in_window", "total_gross_aed"],
      params={"threshold": "cfg.member_repeat_service_threshold",
              "window_days": "cfg.utilisation_velocity_window_days"},
      req=F_LINE + F_DEMO + ["care_pathway_policy.*"]),

    c("CLN-04-R03", "No result before repeat", "E", "ASYNC",
      "Repeat diagnostic service occurs while prior result/report is absent or unchanged",
      "Result latency, failed/inconclusive test",
      "Review",
      "PREPAY_PEND", "NO_RESULT_BEFORE_REPEAT", "clinical_policy", NOPE,
      "No results or reports exist.", req=["observation.value", "observation.event_time"]),

    c("CLN-04-R04", "Provider utilization outlier", "S", "MONTHLY",
      "Services per comparable episode/member exceed shrunk peers",
      "Case mix, referral role and denominator",
      "Provider case",
      "POSTPAY_AUDIT", "PROVIDER_UTILISATION_OUTLIER", "analytics", PART,
      "Claims per distinct member at a provider is computable and is shrunk toward the peer "
      "mean. 'Services' is substituted by 'claims' because there are no service lines, so "
      "the intensity of each claim is invisible to this control.",
      impl="cln_04_r04_provider_utilisation_outlier",
      population="provider with at least cfg.min_entity_opportunities claims",
      inputs=["claim_header.provider_sk", "claim_header.member_sk"],
      expression="shrunk claims-per-member for the provider exceeds the peer posterior upper bound "
                 "and posterior_interval_width <= cfg.max_posterior_width",
      exclusions=["case_mix_difference", "referral_centre_role", "small_denominator",
                  "posterior_interval_too_wide"],
      grouping=["tenant_id", "payer_id", "provider_sk", "scenario_id", "period_bucket"],
      evidence=["provider_sk", "claims_per_member", "shrunk_value", "posterior_interval",
                "peer_mean", "peer_level_used", "peer_n", "distinct_members"],
      params={"max_posterior_width": "cfg.max_posterior_width"},
      req=F_LINE + ["referral.*"]),

    c("CLN-04-R05", "Cascade pattern", "S/N", "WEEKLY",
      "Initial low-intensity encounter reliably triggers unusually large downstream test/service "
      "bundle",
      "Specialty pathway and network structure",
      "Postpay/network case",
      "POSTPAY_AUDIT", "CASCADE_PATTERN", "analytics", NOPE,
      "Downstream service bundles require claim lines and referral edges; neither exists.",
      req=F_LINE + ["referral.*"]),

    # ---- CLN-05 LOS and readmission ---------------------------------------
    c("CLN-05-R01", "LOS high/low residual", "S", "DAILY",
      "LOS outside risk-adjusted DRG/diagnosis prediction interval",
      "Transfers, deaths, ICU and outlier payment",
      "Utilization review",
      "POSTPAY_AUDIT", "LOS_RESIDUAL", "clinical_policy", EXEC,
      "Runs as specified. Length of stay and primary diagnosis both exist, so the "
      "diagnosis-peer robust residual is exactly the control the catalogue describes. Risk "
      "adjustment is by diagnosis rather than DRG, which the catalogue permits ('DRG/diagnosis').",
      impl="cln_05_r01_los_residual",
      population="claim_header with non-null length of stay and a resolvable diagnosis peer group",
      inputs=["encounter.length_of_stay_days", "diagnosis.code"],
      expression="abs(robust_residual(length_of_stay_days, diagnosis_peer_group)) > cfg.los_residual_threshold",
      exclusions=["transfer_case", "death_in_hospital", "icu_stay", "outlier_payment_case",
                  "insufficient_peer_evidence"],
      grouping=["tenant_id", "payer_id", "provider_sk", "episode_id", "scenario_id"],
      evidence=["length_of_stay_days", "peer_median_los", "peer_mad", "robust_residual",
                "direction", "diagnosis_code", "peer_level_used", "peer_n"],
      params={"threshold": "cfg.los_residual_threshold"},
      req=["drg_grouper.*", "encounter.icu_flag", "member.death_date"]),

    c("CLN-05-R02", "Same/related readmission", "E/S", "DAILY",
      "Related admission inside cfg.readmit_days with payment impact",
      "Planned/staged care, unrelated trauma",
      "Episode review, not fraud label",
      "POSTPAY_AUDIT", "RELATED_READMISSION", "clinical_policy", EXEC,
      "Runs as specified: discharge_readmit_gap_days is present for every claim and is "
      "compared against cfg.readmit_window_days, with the episode's payment impact attached. "
      "The catalogue's own words — 'Episode review, NOT FRAUD LABEL' — are reproduced on the "
      "signal so a reviewer sees them.",
      impl="cln_05_r02_related_readmission",
      population="claim_header with a non-null discharge_readmit_gap_days",
      inputs=["claim_header.discharge_readmit_gap_days", "claim_header.gross_amount_aed",
              "diagnosis.code"],
      expression="discharge_readmit_gap_days < cfg.readmit_window_days",
      exclusions=["planned_or_staged_care", "unrelated_trauma", "obstetric_pathway",
                  "oncology_treatment_cycle"],
      grouping=["tenant_id", "payer_id", "member_sk", "episode_id", "scenario_id"],
      evidence=["discharge_readmit_gap_days", "readmit_window_days", "gross_amount_aed",
                "diagnosis_code", "provider_sk", "not_a_fraud_label_notice"],
      params={"window_days": "cfg.readmit_window_days"},
      req=["diagnosis relatedness map (to distinguish related from unrelated readmission)"]),

    c("CLN-05-R03", "Discharge-readmit split", "H/E", "DAILY",
      "Two stays likely constitute one continuous episode under payment policy",
      "Transfers and genuine clinical deterioration",
      "Reprice/pend",
      "PREPAY_PEND", "DISCHARGE_READMIT_SPLIT", "claims_policy", PART,
      "Multi-claim episodes at the same provider with a short gap are detectable. The "
      "disposition is PENDED rather than REPRICED because the correct combined payable "
      "amount is not computable without a payment policy or tariff, and reserves "
      "REPRICE for the case where it is.",
      impl="cln_05_r03_discharge_readmit_split",
      note="Catalogue says 'Reprice/pend'. The pend limb is used because repricing requires "
           "a payment policy this dataset does not contain.",
      alt=["REPRICE"],
      population="multi-claim episodes at a single provider",
      inputs=["episode.claim_sks", "encounter.discharge_date", "encounter.admission_date",
              "claim_header.gross_amount_aed"],
      expression="episode contains two or more claims at the SAME provider with a gap of "
                 "0 to cfg.readmit_window_days days between discharge and next admission",
      exclusions=["cross_provider_transfer", "genuine_clinical_deterioration",
                  "planned_staged_procedure"],
      grouping=["tenant_id", "payer_id", "member_sk", "episode_id", "scenario_id"],
      evidence=["episode_id", "claim_sks", "gap_days", "combined_gross_aed", "provider_sk",
                "linkage_reason"],
      params={"window_days": "cfg.readmit_window_days"},
      req=["payment_policy.continuous_stay_rule", "encounter.transfer_indicator"]),

    c("CLN-05-R04", "Admission-rate outlier", "S", "MONTHLY",
      "Provider admits unusually high share of comparable presentations",
      "Severity, referral mix and observation status",
      "Postpay audit",
      "POSTPAY_AUDIT", "ADMISSION_RATE_OUTLIER", "analytics", NOPE,
      "The denominator — comparable PRESENTATIONS — does not exist. claims.csv contains only "
      "admitted inpatient claims, so the admission RATE is 100% for every provider by "
      "construction. Computing it anyway would produce a number that means nothing.",
      req=["encounter.encounter_type (outpatient presentations)", "encounter.observation_status"]),

    # ---- CLN-06 phantom billing --------------------------------------------
    c("CLN-06-R01", "Missing expected order/result trail", "E", "ASYNC",
      "Billed diagnostic/therapy lacks required order, result, Observation or administration evidence",
      "Artifact latency and service-specific requirements",
      "Pend",
      "PREPAY_PEND", "MISSING_ORDER_TRAIL", "clinical_policy", NOPE,
      "There is no order or result trail for any claim, so the control would fire on all "
      "5,000 rows. That is a data-completeness finding, which the canonical population "
      "report already states, not a detection signal.",
      req=["observation.*", "claim_line.*"]),

    c("CLN-06-R02", "Closed/non-operational facility service", "H/S", "DAILY",
      "Service occurs outside verified operating period or facility shows no credible "
      "operating footprint",
      "24-hour/emergency and outreach service",
      "SIU lead",
      "SIU_LEAD", "NON_OPERATIONAL_FACILITY", "siu", PART,
      "The 'verified operating period' limb cannot run (no operating periods). The 'no "
      "credible operating footprint' limb is implemented as a composite of the undated "
      "blacklist flag plus a thin, bursty claim footprint, and the signal states which limb "
      "fired. SIU_LEAD is the catalogue disposition and is retained — this control never "
      "denies.",
      impl="cln_06_r02_non_operational_facility",
      population="providers flagged on the exclusion list or with a thin, concentrated footprint",
      inputs=["claim_header.provider_sk", "provider_status_period.status_value",
              "claim_header.service_date", "claim_header.gross_amount_aed"],
      expression="provider is exclusion-listed OR (provider claim count is low AND claims "
                 "concentrate into few days AND mean claim value exceeds the peer median)",
      exclusions=["24_hour_or_emergency_service", "outreach_service", "new_provider_onboarding",
                  "operating_period_unverifiable"],
      grouping=["tenant_id", "payer_id", "provider_sk", "scenario_id", "period_bucket"],
      evidence=["provider_sk", "exclusion_listed", "claim_count", "active_days",
                "claims_per_active_day", "mean_gross_aed", "peer_median_gross_aed", "limb_fired"],
      req=["provider_status_period.valid_from/valid_to (operating periods)", "provider.facility_type"]),

    c("CLN-06-R03", "Repeated synthetic encounter signature", "S", "DAILY",
      "High rate of identical code/time/amount/note patterns across unrelated members",
      "Standard packages and batch timestamps",
      "Provider case",
      "POSTPAY_AUDIT", "SYNTHETIC_ENCOUNTER_SIGNATURE", "analytics", PART,
      "Implemented on the signature available: identical (primary diagnosis, length of stay, "
      "requested amount) triples recurring across DIFFERENT members at one provider. Codes "
      "and notes are unavailable, so the signature is coarser than the catalogue's and can "
      "be produced by a genuine standard package — which is why 'standard_package' is a "
      "declared exclusion and the disposition is an audit, not a denial.",
      impl="cln_06_r03_synthetic_encounter_signature",
      population="providers with at least cfg.min_entity_opportunities claims",
      inputs=["claim_header.provider_sk", "claim_header.member_sk", "diagnosis.code",
              "encounter.length_of_stay_days", "claim_header.gross_amount"],
      expression="a (diagnosis, length_of_stay, gross_amount) triple recurs across two or more "
                 "DISTINCT members at the same provider",
      exclusions=["standard_package_pricing", "batch_timestamp_artefact", "capitated_arrangement",
                  "single_member_repeat"],
      grouping=["tenant_id", "payer_id", "provider_sk", "scenario_id", "period_bucket"],
      evidence=["provider_sk", "signature", "distinct_members", "claim_sks", "repeat_count",
                "gross_amount_aed"],
      req=F_LINE + F_DOC),

    c("CLN-06-R04", "Member denial/attendance mismatch", "S", "DAILY",
      "Reliable member confirmation or attendance record says service not received",
      "Contact/authentication quality and recall",
      "High-priority SIU lead",
      "SIU_LEAD", "MEMBER_ATTENDANCE_MISMATCH", "siu", NOPE,
      "No member confirmations or attendance records.",
      req=["member_confirmation.*", "attendance_record.*"]),

    c("CLN-06-R05", "No longitudinal clinical footprint", "S", "MONTHLY",
      "Claimed high-impact service lacks expected follow-up, medication, result or episode "
      "consequences versus peers",
      "Out-of-network follow-up and data completeness",
      "Supporting evidence only",
      "MONITOR_ONLY", "NO_LONGITUDINAL_FOOTPRINT", "analytics", NOPE,
      "Follow-up medication and results are absent for every claim, so 'expected consequences' "
      "has no observable form. The catalogue already marks this control 'supporting evidence "
      "only'; on this data it would be no evidence.",
      req=F_RX + ["observation.*", "claim_line.*"]),

    # ---- CLN-07 capacity ---------------------------------------------------
    c("CLN-07-R01", "Improbable service day", "E/S", "DAILY",
      "Sum of conservative service minutes/units per clinician exceeds available day capacity",
      "Parallel/team services and documented hours",
      "Provider timeline",
      "POSTPAY_AUDIT", "IMPROBABLE_SERVICE_DAY", "provider_network", NOPE,
      "No clinicians, no service durations. The FACILITY-level analogue is CLN-07-R02.",
      req=F_CLINICIAN + ["claim_line.service_start_time", "claim_line.service_end_time"]),

    c("CLN-07-R02", "Facility/staff capacity", "E/S", "DAILY",
      "Claimed volume exceeds rooms/equipment/licensed staff capacity plus tolerance",
      "Missing roster produces low confidence",
      "Audit lead",
      "POSTPAY_AUDIT", "FACILITY_CAPACITY_EXCEEDED", "provider_network", PART,
      "Concurrent inpatient occupancy per provider per day is computable from admission and "
      "discharge dates and is compared against cfg.provider_daily_capacity_admissions × "
      "cfg.capacity_tolerance_multiple. The capacity figure is a CONFIGURED PLACEHOLDER, not "
      "a licensed bed count — the catalogue's own note that 'missing roster produces low "
      "confidence' applies directly, and the signal's confidence is reduced accordingly.",
      impl="cln_07_r02_facility_capacity",
      population="provider-days with at least one active inpatient stay",
      inputs=["claim_header.provider_sk", "encounter.admission_date", "encounter.discharge_date"],
      expression="concurrent_occupancy(provider, day) > cfg.provider_daily_capacity_admissions "
                 "* cfg.capacity_tolerance_multiple",
      exclusions=["roster_unavailable_low_confidence", "licensed_capacity_not_sourced",
                  "outreach_or_mobile_service", "day_case_turnover"],
      grouping=["tenant_id", "payer_id", "provider_sk", "scenario_id", "period_bucket"],
      evidence=["provider_sk", "peak_day", "peak_concurrent_occupancy", "assumed_capacity",
                "tolerance_multiple", "claims_on_peak_day", "capacity_source"],
      params={"capacity": "cfg.provider_daily_capacity_admissions",
              "tolerance": "cfg.capacity_tolerance_multiple"},
      req=["provider.licensed_bed_count", "provider.room_count", "clinician_roster.*"]),

    c("CLN-07-R03", "Scarce-equipment concurrency", "E", "DAILY",
      "Same device/resource required by overlapping services beyond capacity",
      "Equipment inventory and turnaround",
      "Pend/audit",
      "PREPAY_PEND", "EQUIPMENT_CONCURRENCY", "provider_network", NOPE,
      "No equipment inventory and no procedures.",
      req=["equipment_inventory.*"] + F_LINE),

    c("CLN-07-R04", "Capacity trend discontinuity", "S", "MONTHLY",
      "Throughput jumps without staff, hours, equipment or facility change",
      "Onboarding/data-feed changes",
      "Provider case",
      "POSTPAY_AUDIT", "CAPACITY_TREND_DISCONTINUITY", "analytics", PART,
      "Monthly claim throughput per provider is computable, and a CUSUM change point on it "
      "is exactly the shape of this control. What cannot be done is the EXCLUSION: without "
      "staff, hours or facility-change records there is no way to segment a legitimate "
      "expansion from a suspicious jump, so every signal carries that caveat and the "
      "disposition stays an audit.",
      impl="cln_07_r04_capacity_trend_discontinuity",
      population="providers with at least cfg.min_history_periods monthly periods",
      inputs=["claim_header.provider_sk", "claim_header.service_date"],
      expression="CUSUM on monthly claim throughput crosses cfg.cusum_h after at least "
                 "cfg.min_history_periods periods of history",
      exclusions=["staff_or_hours_change_unknown", "facility_change_unknown",
                  "onboarding_or_data_feed_change", "insufficient_history"],
      grouping=["tenant_id", "payer_id", "provider_sk", "scenario_id", "period_bucket"],
      evidence=["provider_sk", "change_date", "pre_change_mean", "post_change_mean",
                "cusum_statistic", "periods_of_history", "segmentation_caveat"],
      params={"k": "cfg.cusum_k", "h": "cfg.cusum_h", "min_history": "cfg.min_history_periods"},
      req=["clinician_roster.*", "provider.operating_hours", "contract.effective_dates"]),

    # ---- CLN-08 laboratory --------------------------------------------------
    c("CLN-08-R01", "No qualified order/relationship", "E", "PREPAY_ASYNC",
      "Test lacks valid ordering clinician or qualifying encounter/indication",
      "Screening programs, standing orders",
      "Clinical pend",
      "PREPAY_PEND", "LAB_NO_QUALIFIED_ORDER", "clinical_policy", NOPE,
      "No tests, no ordering clinicians.", req=F_LINE + F_CLINICIAN),

    c("CLN-08-R02", "Panel/component inflation", "H/E", "PREPAY",
      "Components billed separately or panel substantially exceeds ordered tests",
      "Reflex testing and local panel rules",
      "Reprice/pend",
      "REPRICE", "PANEL_INFLATION", "claims_policy", NOPE,
      "No panels, components or orders.",
      alt=["PREPAY_PEND"], req=F_LINE + ["panel_definition.*"]),

    c("CLN-08-R03", "Absent/duplicate result", "E/T", "ASYNC",
      "Paid test lacks result/report or identical result appears across unrelated patients",
      "Result latency and normal templates",
      "Audit/SIU lead",
      "POSTPAY_AUDIT", "LAB_RESULT_ABSENT_OR_DUPLICATE", "clinical_policy", NOPE,
      "claims.csv carries no observations or result values, so neither the absence of a "
      "result for a paid test nor a result repeated across patients is visible.", alt=["SIU_LEAD"], req=["observation.value", "observation.event_time"]),

    c("CLN-08-R04", "Reference-lab/pass-through spread", "S/N", "MONTHLY",
      "Billing entity adds abnormal markup or volume while test is performed by concentrated "
      "third party",
      "Contract and permitted reference arrangements",
      "Postpay/network case",
      "POSTPAY_AUDIT", "REFERENCE_LAB_SPREAD", "analytics", NOPE,
      "No performing-entity field and no test-level amounts.",
      req=F_LINE + ["claim_line.performing_entity_id"]),

    c("CLN-08-R05", "High-complexity/genetic test outlier", "S", "MONTHLY",
      "Provider ordering/rendering rate, panel size or cost exceeds risk-adjusted peers",
      "Specialty/oncology/rare-disease centres",
      "Targeted record sample",
      "POSTPAY_AUDIT", "GENETIC_TEST_OUTLIER", "analytics", NOPE,
      "No test identification of any kind.", req=F_LINE),
]

# =============================================================================
# PHR — Pharmacy, products and supplies  (20 controls)
# =============================================================================

CATALOGUE += [
    # ---- PHR-01 ------------------------------------------------------------
    c("PHR-01-R01", "Product identity mismatch", "H", "PREPAY",
      "Billed product differs from prescribed/authorized product and is not approved equivalent",
      "Generic/trade equivalence and substitution policy",
      "Pend/reject",
      "PREPAY_PEND", "PRODUCT_IDENTITY_MISMATCH", "pharmacy_policy", NOPE,
      "prescription_dispense is NOT_POPULATED. claims.csv carries one aggregate "
      "pharmacy_bill_ratio and no products at all.",
      alt=["REJECT"], req=F_RX),

    c("PHR-01-R02", "Quantity/strength/form mismatch", "H/E", "PREPAY",
      "Billed quantity, strength or dosage form exceeds order/authorization",
      "Partial fill, titration and package conversion",
      "Reprice/pend",
      "REPRICE", "QUANTITY_STRENGTH_MISMATCH", "pharmacy_policy", NOPE,
      "No quantities, strengths or dosage forms.",
      alt=["PREPAY_PEND"], req=F_RX + F_AUTH),

    c("PHR-01-R03", "Dispense-claim mismatch", "H", "DAILY",
      "Dispensing/inventory record differs from billed product/quantity",
      "Claim timing and reversals",
      "Pharmacy case",
      "POSTPAY_AUDIT", "DISPENSE_CLAIM_MISMATCH", "pharmacy_policy", NOPE,
      "No dispensing or inventory records.", req=F_RX + ["pharmacy_inventory.*"]),

    c("PHR-01-R04", "Non-medical/uncovered substitution", "E/S", "DAILY",
      "Member/receipt/inventory evidence indicates uncovered or non-medical item supplied",
      "Confirmation reliability",
      "SIU lead",
      "SIU_LEAD", "NON_MEDICAL_SUBSTITUTION", "siu", NOPE,
      "No receipts, inventory or item-level detail.",
      req=F_RX + ["member_receipt.*", "pharmacy_inventory.*"]),

    # ---- PHR-02 ------------------------------------------------------------
    c("PHR-02-R01", "Refill overlap", "H/E", "PREPAY",
      "Remaining supply at next fill exceeds cfg.allowed_overlap_days",
      "Dose change, lost/travel override, inpatient days",
      "Pend",
      "PREPAY_PEND", "REFILL_OVERLAP", "pharmacy_policy", NOPE,
      "No fills, no days supply. Remaining supply is uncomputable.", req=F_RX),

    c("PHR-02-R02", "Therapy-duration excess", "E", "DAILY",
      "Continuous fills exceed diagnosis/drug policy duration without review",
      "Chronic therapy and specialist approval",
      "Clinical review",
      "PREPAY_PEND", "THERAPY_DURATION_EXCESS", "pharmacy_policy", NOPE,
      "No fills or therapy durations.", req=F_RX + ["drug_policy.duration_limits"]),

    c("PHR-02-R03", "Multi-prescriber same-equivalent drug", "E/S", "DAILY",
      "Distinct prescribers within window exceed drug/specialty threshold",
      "Care team and provider-group identity",
      "Member/pharmacy case",
      "POSTPAY_AUDIT", "MULTI_PRESCRIBER_CONVERGENCE", "pharmacy_policy", NOPE,
      "No prescribers and no drugs. The member-level provider-dispersion feature exists "
      "(member_prior_distinct_providers) but counts hospitals, not prescribers of one drug, "
      "and substituting it would change what the control means.",
      req=F_RX),

    c("PHR-02-R04", "Multi-pharmacy convergence", "S/N", "DAILY",
      "Same member obtains overlapping equivalent drug from multiple pharmacies",
      "Stock shortage and partial fills",
      "SIU/clinical lead",
      "SIU_LEAD", "MULTI_PHARMACY_CONVERGENCE", "siu", NOPE,
      "No pharmacies and no drugs in the source.", req=F_RX),

    # ---- PHR-03 ------------------------------------------------------------
    c("PHR-03-R01", "Dose/weight/body-surface conflict", "E", "PREPAY",
      "Claimed dose outside approved clinical range",
      "Wastage, loading dose and rounding",
      "Clinical pend/reprice",
      "PREPAY_PEND", "DOSE_OUT_OF_RANGE", "clinical_policy", NOPE,
      "No doses, weights or body-surface data.",
      alt=["REPRICE"], req=F_RX + ["member.weight", "member.height"]),

    c("PHR-03-R02", "Drug-diagnosis/step-therapy conflict", "E", "PREPAY",
      "Indication or prerequisite therapy absent",
      "Authorization overrides and rare disease",
      "Pend",
      "PREPAY_PEND", "STEP_THERAPY_CONFLICT", "clinical_policy", NOPE,
      "No drugs to check against the diagnosis.", req=F_RX + ["step_therapy_policy.*"]),

    c("PHR-03-R03", "Provider/pharmacy volume spike", "S", "MONTHLY",
      "Risk-adjusted volume/value changepoint without patient-mix shift",
      "New formulary/centre status",
      "Audit",
      "POSTPAY_AUDIT", "PHARMACY_VOLUME_SPIKE", "pharmacy_policy", PART,
      "Implemented on the pharmacy SHARE residual against diagnosis peers — the only "
      "pharmacy quantity in the source — rather than on drug volume or value. A provider "
      "whose pharmacy share sits above cfg.pharmacy_share_peer_multiple × the diagnosis-peer "
      "median is flagged. 'Without patient-mix shift' is partially controlled by the "
      "diagnosis-level peer group, not by a formal mix adjustment.",
      impl="phr_03_r03_pharmacy_volume_spike",
      population="claims with a non-null pharmacy_bill_ratio and a resolvable diagnosis peer group",
      inputs=["claim_header.pharmacy_bill_ratio", "diagnosis.code", "claim_header.provider_sk",
              "claim_header.gross_amount_aed"],
      expression="pharmacy_bill_ratio > cfg.pharmacy_share_peer_multiple * "
                 "diagnosis_peer_median(pharmacy_bill_ratio)",
      exclusions=["new_formulary_status", "centre_of_excellence_status", "patient_mix_shift",
                  "insufficient_peer_evidence"],
      grouping=["tenant_id", "payer_id", "provider_sk", "scenario_id", "period_bucket"],
      evidence=["pharmacy_bill_ratio", "peer_median_ratio", "peer_multiple", "implied_pharmacy_aed",
                "diagnosis_code", "peer_level_used", "peer_n"],
      params={"multiple": "cfg.pharmacy_share_peer_multiple"},
      req=F_RX + ["claim_line.activity_code (drug-level volume and value)"]),

    c("PHR-03-R04", "Wastage/unused-vial anomaly", "S", "MONTHLY",
      "Wastage units or vial rounding exceed comparable dosing peers",
      "Single-use vial policy",
      "Postpay sample",
      "POSTPAY_AUDIT", "WASTAGE_ANOMALY", "pharmacy_policy", NOPE,
      "No wastage units or vial data.", req=F_RX + ["claim_line.wastage_units"]),

    # ---- PHR-04 ------------------------------------------------------------
    c("PHR-04-R01", "Top-pharmacy concentration", "S", "MONTHLY",
      "Prescriber's share to top pharmacy exceeds geographic/specialty peers",
      "On-site, specialty and narrow network",
      "Network lead",
      "SIU_LEAD", "TOP_PHARMACY_CONCENTRATION", "siu", NOPE,
      "No prescribers, no pharmacies and no geography for the peer comparison. The "
      "structurally identical AGENT→PROVIDER concentration control is implemented as "
      "NET-04-R02.",
      req=F_RX + F_GEO),

    c("PHR-04-R02", "Reciprocal value loop", "N", "WEEKLY",
      "Concentrated prescriber-pharmacy flow aligns with referrals/ownership/shared IDs",
      "Known corporate relationships",
      "SIU graph",
      "SIU_LEAD", "RECIPROCAL_VALUE_LOOP", "siu", NOPE,
      "No prescriber-pharmacy edges, no ownership and no shared identifiers.",
      req=F_RX + F_IDENT),

    c("PHR-04-R03", "High-cost steering", "S/N", "WEEKLY",
      "Concentration is materially stronger for profitable/high-cost drugs",
      "Product availability",
      "Prioritized lead",
      "SIU_LEAD", "HIGH_COST_STEERING", "siu", NOPE,
      "claims.csv has a single pharmacy_bill_ratio per claim and no drug-level lines or "
      "prices, so high-cost products cannot be separated from the rest of the basket.", req=F_RX),

    c("PHR-04-R04", "Rapid relationship formation", "N", "WEEKLY",
      "New prescriber-pharmacy edge rapidly becomes dominant",
      "New site/contract launch",
      "Monitor/SIU",
      "MONITOR_ONLY", "RAPID_RELATIONSHIP_FORMATION", "siu", NOPE,
      "No prescriber-pharmacy edges. The analogous agent-provider edge-formation pattern is "
      "covered by NET-02-R04's structural-change logic, which is itself NOT_EXECUTABLE for "
      "want of typed relationship edges.",
      req=F_RX),

    # ---- PHR-05 (added by the coverage audit) ------------------------------
    c("PHR-05-R01", "New/used/rental/purchase mismatch", "H/E", "PREPAY",
      "Billed condition or payment method conflicts with authorization, serial/inventory or contract",
      "Refurbishment and rent-to-own policy",
      "Reprice/pend",
      "REPRICE", "DEVICE_CONDITION_MISMATCH", "claims_policy", NOPE,
      "No devices, serials, inventory or contracts.",
      alt=["PREPAY_PEND"], req=F_LINE + ["device_inventory.*"] + F_AUTH),

    c("PHR-05-R02", "Implant/device not linked to procedure", "E", "PREPAY",
      "Device/implant lacks qualifying procedure, laterality or operative record",
      "Replacement/spare policy",
      "Clinical pend",
      "PREPAY_PEND", "DEVICE_NOT_LINKED", "clinical_policy", NOPE,
      "No devices and no procedures.", req=F_LINE + F_DOC),

    c("PHR-05-R03", "Supply quantity/consumption excess", "E/S", "DAILY",
      "Units exceed procedure/episode norm or repeat too soon for useful life",
      "Member growth/damage/clinical change",
      "Reprice/audit",
      "POSTPAY_AUDIT", "SUPPLY_CONSUMPTION_EXCESS", "clinical_policy", NOPE,
      "claims.csv records no supply or device lines and no quantities, so neither units per "
      "episode nor replacement interval against useful life can be computed.",
      note=" SUBSTITUTION: the catalogue's first-named disposition for this E/S control is "
           "'Reprice'. A control that is part STATISTICAL may never reprice, so the governed "
           "disposition is the catalogue's second option, POSTPAY_AUDIT. This is the ONLY "
           "control in the catalogue whose disposition the type rule forces down; the "
           "other four recorded downgrades are driven by what this DATASET can support, not "
           "by the control's type. All five are listed in the README.",
      req=F_LINE + ["device_inventory.useful_life"]),

    c("PHR-05-R04", "Serial/inventory duplication", "H/N", "DAILY",
      "Same serial/batch billed for multiple members or claimed stock exceeds "
      "inventory/acquisition",
      "Bulk/non-serialized consumables",
      "SIU lead",
      "SIU_LEAD", "SERIAL_DUPLICATION", "siu", NOPE,
      "No serials, batches or inventory.", req=["device_inventory.serial_number"] + F_LINE),
]

# =============================================================================
# DOC — Documentation  (8 controls)
# =============================================================================

CATALOGUE += [
    c("DOC-01-R01", "Required document absent", "H/E", "ASYNC",
      "Rule requires note/report/consent/Observation and it is missing after SLA",
      "Document type and latency",
      "Pend",
      "PREPAY_PEND", "REQUIRED_DOCUMENT_ABSENT", "clinical_policy", NOPE,
      "There is no document-requirement rule set and no document SLA, so 'required' is "
      "undefined. The synthetic corpus does not change that: it supplies documents, not the "
      "policy that makes one mandatory.",
      req=F_DOC + ["document_requirement_policy.*"]),

    c("DOC-01-R02", "Structured fact conflict", "T/E", "ASYNC",
      "Extracted diagnosis, procedure, date, quantity or setting contradicts claim/authorization",
      "Per-field extraction precision threshold",
      "Human review",
      "PREPAY_PEND", "STRUCTURED_FACT_CONFLICT", "clinical_policy", PART,
      "Runs against the clearly-labelled SYNTHETIC discharge summaries this artefact "
      "generates (every filename and body marked SYNTHETIC), never against real "
      "documentation, because claims.csv contains none. Extracted diagnosis and stay length "
      "are compared field-by-field against the claim, each with its own precision threshold, "
      "and every rendered finding carries its exact source span.",
      impl="doc_01_r02_structured_fact_conflict",
      population="claims with a generated synthetic discharge summary",
      inputs=["document.text", "document.language", "diagnosis.code",
              "encounter.length_of_stay_days", "encounter.admission_date"],
      expression="an extracted field conflicts with the corresponding claim field, the "
                 "field-level extraction confidence >= cfg.nlp_min_extraction_confidence, "
                 "and a source span is present",
      exclusions=["extraction_confidence_below_threshold", "no_source_span_finding_discarded",
                  "document_latency", "language_specific_precision_not_validated"],
      grouping=["tenant_id", "payer_id", "claim_sk", "scenario_id"],
      evidence=["conflicting_field", "extracted_value", "claim_value", "source_span_text",
                "span_offsets", "extraction_confidence", "ocr_confidence", "document_language",
                "synthetic_document_marker"],
      params={"min_confidence": "cfg.nlp_min_extraction_confidence"},
      req=F_DOC + ["real clinical documents with OCR confidence"]),

    c("DOC-01-R03", "Medical-necessity support absent", "T/E", "ASYNC",
      "Required indication/severity element cannot be found",
      "Absence is not proof; use qualified reviewer",
      "Clinical review",
      "PREPAY_PEND", "MEDICAL_NECESSITY_UNSUPPORTED", "clinical_policy", NOPE,
      "Requires a medical-necessity policy defining which elements are required. The "
      "catalogue's own caution — 'absence is not proof' — is the reason this is not "
      "approximated on synthetic notes: an absence in a generated document proves only that "
      "the generator did not write it.",
      req=F_DOC + ["medical_necessity_policy.*"]),

    c("DOC-01-R04", "Authorization narrative drift", "T", "ASYNC",
      "Material semantic/entity mismatch between request narrative and billed event",
      "Explain with extracted conflicting facts",
      "Pend",
      "PREPAY_PEND", "AUTH_NARRATIVE_DRIFT", "clinical_policy", NOPE,
      "No authorisation narratives exist.", req=F_AUTH + F_DOC),

    c("DOC-02-R01", "Cross-patient near duplicate", "T", "DAILY",
      "Patient-specific content similarity exceeds threshold across unrelated members",
      "Remove standard boilerplate before comparison",
      "Audit sample",
      "POSTPAY_AUDIT", "CROSS_PATIENT_NEAR_DUPLICATE", "clinical_policy", PART,
      "Runs against the SYNTHETIC corpus only. Boilerplate is stripped before comparison, as "
      "the catalogue requires, and similarity is computed on the patient-specific remainder. "
      "Because the corpus is generated from templates, a baseline similarity is expected and "
      "the threshold (cfg.doc_clone_similarity_threshold) is set above it — a limitation "
      "stated on every signal.",
      impl="doc_02_r01_cross_patient_near_duplicate",
      population="pairs of synthetic documents belonging to different members",
      inputs=["document.text", "claim_header.member_sk", "claim_header.provider_sk"],
      expression="token-set similarity of the boilerplate-stripped text between two documents "
                 "for DIFFERENT members exceeds cfg.doc_clone_similarity_threshold",
      exclusions=["boilerplate_removed_before_comparison", "same_member_documents",
                  "template_generated_corpus_baseline", "specialty_template_in_use"],
      grouping=["tenant_id", "payer_id", "provider_sk", "scenario_id", "period_bucket"],
      evidence=["document_pair", "similarity", "threshold", "member_sks", "provider_sk",
                "boilerplate_removed", "synthetic_document_marker"],
      params={"threshold": "cfg.doc_clone_similarity_threshold"},
      req=F_DOC),

    c("DOC-02-R02", "Contradictory copied facts", "T/E", "DAILY",
      "Note contains wrong patient demographics/date/laterality or impossible copied findings",
      "Extraction confidence",
      "High-priority review",
      "PREPAY_PEND", "CONTRADICTORY_COPIED_FACTS", "clinical_policy", NOPE,
      "Needs demographics and laterality to contradict; the source has neither, so a "
      "'wrong demographic' finding could not be evaluated even on synthetic text.",
      req=F_DOC + F_DEMO),

    c("DOC-02-R03", "Post-denial material alteration", "H/T", "DAILY",
      "Document version changes reimbursement-relevant facts after denial without addendum "
      "provenance",
      "Valid signed addendum policy",
      "PAY-08-linked case",
      "PREPAY_PEND", "POST_DENIAL_ALTERATION", "clinical_policy", NOPE,
      "Needs denials and document version history; neither exists.",
      req=F_DOC + F_REMIT + F_LINEAGE),

    c("DOC-02-R04", "Template-to-complexity mismatch", "T/S", "MONTHLY",
      "High-complexity services supported by unusually generic notes versus peers",
      "Specialty and EHR template",
      "Supporting evidence only",
      "MONITOR_ONLY", "TEMPLATE_COMPLEXITY_MISMATCH", "analytics", NOPE,
      "'High complexity' cannot be established without service codes, so the comparison has "
      "no x-axis.", req=F_DOC + F_LINE),
]

# =============================================================================
# NET / ANL / POL  (25 controls)
# =============================================================================

CATALOGUE += [
    # ---- NET-01 ------------------------------------------------------------
    c("NET-01-R01", "Referral concentration outlier", "S", "MONTHLY",
      "Top-recipient share/HHI exceeds specialty/geographic peers",
      "Group/network and centre-of-excellence",
      "Network case",
      "POSTPAY_AUDIT", "REFERRAL_CONCENTRATION", "siu", PART,
      "There are no referral edges in claims.csv. The implemented edge is AGENT→PROVIDER "
      "co-occurrence: the share of an agent's claims flowing to their single most-used "
      "provider, plus the Herfindahl index of that agent's provider distribution. That is a "
      "distribution-channel concentration, not a clinical referral, and the signal says so. "
      "The peer comparison is by agent portfolio size, not by specialty or geography, "
      "because neither is available.",
      impl="net_01_r01_referral_concentration",
      population="agents with at least cfg.min_entity_opportunities claims",
      inputs=["claim_header.agent_id", "claim_header.provider_sk", "claim_header.gross_amount_aed"],
      expression="top_recipient_share(agent) > cfg.referral_concentration_threshold "
                 "and the agent's claim count is above the minimum denominator",
      exclusions=["corporate_group_or_narrow_network", "centre_of_excellence",
                  "small_denominator", "inferred_edge_not_a_clinical_referral"],
      grouping=["tenant_id", "payer_id", "agent_id", "scenario_id", "period_bucket"],
      evidence=["agent_id", "top_provider_sk", "top_provider_share", "hhi", "claim_count",
                "distinct_providers", "exposure_aed", "edge_type"],
      params={"threshold": "cfg.referral_concentration_threshold"},
      req=["referral.referrer_id", "referral.recipient_id", "provider.specialty"] + F_GEO),

    c("NET-01-R02", "Reciprocal referral loop", "N", "WEEKLY",
      "A→B and B→A edges are unusually strong relative to opportunity",
      "Multidisciplinary teams",
      "Graph evidence",
      "MONITOR_ONLY", "RECIPROCAL_REFERRAL_LOOP", "siu", NOPE,
      "Reciprocity requires DIRECTED referral edges. Agent-provider and patient-provider "
      "edges here are bipartite and undirected by construction, so A→B and B→A cannot exist "
      "and the metric would be identically zero.",
      req=["referral.referrer_id", "referral.recipient_id", "referral.direction"]),

    c("NET-01-R03", "High-cost conversion", "S/N", "WEEKLY",
      "Referred members convert to high-cost service at abnormal rate/value",
      "Case mix and referral indication",
      "Prioritized audit",
      "POSTPAY_AUDIT", "HIGH_COST_CONVERSION", "analytics", NOPE,
      "No referrals to convert from, and no service-level costs to convert to.",
      req=["referral.*"] + F_LINE),

    c("NET-01-R04", "Closed downstream chain", "N", "WEEKLY",
      "Referrer→lab/imaging/pharmacy chain retains members within small dense group",
      "Ownership/narrow network",
      "SIU lead",
      "SIU_LEAD", "CLOSED_DOWNSTREAM_CHAIN", "siu", NOPE,
      "No labs, imaging centres or pharmacies as distinct entities, and no ownership data.",
      req=["referral.*", "provider.provider_type"] + F_IDENT),

    # ---- NET-02 ------------------------------------------------------------
    c("NET-02-R01", "Shared administrative identity", "H/N", "DAILY",
      "Legally distinct entities share bank, phone, address, device or owner identifier",
      "Known groups/shared services; fuzzy matches cannot be hard flags",
      "Graph edge",
      "MONITOR_ONLY", "SHARED_ADMIN_IDENTITY", "siu", NOPE,
      "No bank, phone, address, device or owner identifiers exist. The entity-resolution "
      "service that would consume them IS implemented (fwa.graph.entity_resolution) and runs "
      "on the identifiers that do exist, surfacing every candidate for human confirmation "
      "and auto-merging nothing — but this control's specific inputs are absent.",
      req=F_IDENT),

    c("NET-02-R02", "Dense member circulation", "N", "WEEKLY",
      "Community has abnormal internal member/provider edge density and low external flow",
      "Specialty/geography/size matched communities",
      "SIU graph",
      "SIU_LEAD", "DENSE_MEMBER_CIRCULATION", "siu", PART,
      "Louvain communities are computed on the real member-provider-agent-TPA graph built "
      "from claims.csv, with weekly time-bounded snapshots, and internal density is compared "
      "against SIZE-matched communities. Peer matching on specialty and geography is not "
      "possible, so the comparison is size-only and the signal records that.",
      impl="net_02_r02_dense_member_circulation",
      population="Louvain communities of at least cfg.community_min_size nodes",
      inputs=["claim_header.member_sk", "claim_header.provider_sk", "claim_header.agent_id",
              "claim_header.service_date", "claim_header.gross_amount_aed"],
      expression="internal_edge_density(community) > cfg.community_density_threshold "
                 "and community size >= cfg.community_min_size",
      exclusions=["size_matched_peers_only", "specialty_matching_unavailable",
                  "geography_matching_unavailable", "corporate_group_structure"],
      grouping=["tenant_id", "payer_id", "community_id", "scenario_id", "period_bucket"],
      evidence=["community_id", "size", "internal_density", "peer_density_median",
                "member_count", "provider_count", "agent_count", "claim_count", "exposure_aed",
                "snapshot_window", "each_claim_counted_once"],
      params={"min_size": "cfg.community_min_size", "density": "cfg.community_density_threshold"},
      req=["provider.specialty"] + F_GEO + F_IDENT),

    c("NET-02-R03", "Synchronized billing", "N/S", "DAILY",
      "Community claims share unusual time, amount, code or resubmission signatures",
      "Batch billing systems",
      "Supporting evidence",
      "MONITOR_ONLY", "SYNCHRONIZED_BILLING", "analytics", PART,
      "The TIME and AMOUNT limbs run: claims within a community that share an admission date "
      "and a closely-matching amount are a synchronisation signature. The CODE and "
      "RESUBMISSION limbs do not (no codes, no lineage). The catalogue already assigns this "
      "control 'supporting evidence' status and MONITOR_ONLY, which is retained.",
      impl="net_02_r03_synchronized_billing",
      population="communities identified by NET-02-R02's graph snapshot",
      inputs=["claim_header.service_date", "claim_header.gross_amount_aed",
              "claim_header.provider_sk", "claim_header.member_sk"],
      expression="two or more claims in one community share an admission date and have gross "
                 "amounts within cfg.duplicate_amount_tolerance of each other",
      exclusions=["batch_billing_system", "code_signature_unavailable",
                  "resubmission_signature_unavailable", "standard_package_pricing"],
      grouping=["tenant_id", "payer_id", "community_id", "scenario_id", "period_bucket"],
      evidence=["community_id", "synchronised_claims", "shared_date", "amount_spread",
                "distinct_members", "distinct_providers", "limbs_available"],
      params={"amount_tolerance": "cfg.duplicate_amount_tolerance"},
      req=F_LINE + F_LINEAGE),

    c("NET-02-R04", "Structural-change alert", "N", "WEEKLY",
      "Previously peripheral nodes rapidly become central/dense with material exposure",
      "Merger/new contract",
      "Monitor/SIU",
      "MONITOR_ONLY", "STRUCTURAL_CHANGE_ALERT", "siu", NOPE,
      "Requires a stable node-centrality history across snapshots plus merger and contract "
      "records to exclude legitimate structural change. With no contract data, every "
      "growing provider would alert, so the control is classified honestly rather than run "
      "as a growth detector.",
      req=["contract.effective_dates", "provider.ownership_history"] + F_IDENT),

    # ---- NET-03 ------------------------------------------------------------
    c("NET-03-R01", "Repeated high-value dyad", "S/N", "MONTHLY",
      "Member-provider pair has abnormal frequency/value and benefit exhaustion",
      "Chronic/rare-disease pathways",
      "Case evidence",
      "POSTPAY_AUDIT", "HIGH_VALUE_DYAD", "siu", EXEC,
      "Member-provider pair frequency and total value are directly computable and are "
      "compared against the distribution of all dyads. The BENEFIT-EXHAUSTION limb is "
      "unavailable (no benefit limits) and the signal states that only two of the three "
      "conditions were evaluated.",
      impl="net_03_r01_repeated_high_value_dyad",
      population="member-provider pairs with more than one claim",
      inputs=["claim_header.member_sk", "claim_header.provider_sk",
              "claim_header.gross_amount_aed"],
      expression="dyad claim count >= 2 and dyad total value exceeds the robust upper bound "
                 "of the dyad-value distribution",
      exclusions=["chronic_disease_pathway", "rare_disease_pathway", "oncology_pathway",
                  "benefit_exhaustion_limb_unavailable"],
      grouping=["tenant_id", "payer_id", "member_sk", "provider_sk", "scenario_id"],
      evidence=["member_sk", "provider_sk", "dyad_claim_count", "dyad_total_aed",
                "dyad_value_percentile", "robust_residual", "diagnoses", "limbs_evaluated"],
      req=F_BENEFIT + ["benefit accumulator for exhaustion"]),

    c("NET-03-R02", "Small closed member group", "N", "WEEKLY",
      "Provider activity concentrates in tightly connected member/employer group",
      "Clinic catchment and employer onsite care",
      "SIU graph",
      "SIU_LEAD", "CLOSED_MEMBER_GROUP", "siu", NOPE,
      "No employer or group identifiers, and no catchment data to exclude a legitimate "
      "local clinic. The general community-density control NET-02-R02 covers the "
      "graph-structural part of this pattern.",
      req=["member.employer_id", "member.sponsor_id"] + F_GEO),

    c("NET-03-R03", "Inducement signature", "S", "MONTHLY",
      "Zero patient share, non-medical substitution or repetitive profitable services co-occur",
      "Approved programs",
      "High-priority lead",
      "SIU_LEAD", "INDUCEMENT_SIGNATURE", "siu", NOPE,
      "All three limbs need absent data: patient share, item-level substitution and service "
      "codes.", req=["claim_header.patient_share"] + F_LINE + F_RX),

    c("NET-03-R04", "Complaint/confirmation corroboration", "S", "DAILY",
      "Authenticated member evidence corroborates service/item or charge mismatch",
      "Evidence reliability",
      "Escalate case",
      "SIU_LEAD", "COMPLAINT_CORROBORATION", "siu", NOPE,
      "No complaints or member confirmations.", req=["complaint.*", "member_confirmation.*"]),

    # ---- NET-04 ------------------------------------------------------------
    c("NET-04-R01", "Confirmed-outcome rate", "S", "MONTHLY",
      "Hierarchical model shows excess validated FWA/error outcomes by originator/group",
      "Never use raw system flags as label; control portfolio mix",
      "Distribution review",
      "POSTPAY_AUDIT", "ORIGINATOR_OUTCOME_RATE", "analytics", NOPE,
      "This control requires CONFIRMED REVIEW OUTCOMES as its label, and the catalogue is "
      "explicit that raw system flags must never be used as one. review_outcome is empty at "
      "load, and fraud_label is held-out evaluation data that no detection component may "
      "read. Implementing it against fraud_label would be precisely the "
      "leakage this artefact is built to prevent. It becomes executable once the artefact "
      "has accumulated real reviewer dispositions.",
      req=["review_outcome.validated_category", "review_outcome.confirmed_amount_aed"]),

    c("NET-04-R02", "Provider concentration", "S/N", "MONTHLY",
      "Originator/group members disproportionately use a small provider network",
      "Employer clinic and geography",
      "Network case",
      "POSTPAY_AUDIT", "ORIGINATOR_PROVIDER_CONCENTRATION", "siu", PART,
      "Agent (originator) → provider concentration is computable and is the structural "
      "content of this control. Employer groups do not exist in the source, so only the "
      "agent limb runs, and the employer-clinic exclusion cannot be applied — stated on "
      "every signal.",
      impl="net_04_r02_originator_provider_concentration",
      population="agents with at least cfg.min_entity_opportunities claims",
      inputs=["claim_header.agent_id", "claim_header.provider_sk", "claim_header.member_sk"],
      expression="the agent's members are concentrated into a provider set materially smaller "
                 "than the peer expectation for the agent's portfolio size",
      exclusions=["employer_onsite_clinic_unverifiable", "geographic_catchment_unavailable",
                  "small_portfolio", "corporate_group_structure"],
      grouping=["tenant_id", "payer_id", "agent_id", "scenario_id", "period_bucket"],
      evidence=["agent_id", "distinct_providers", "expected_distinct_providers", "claim_count",
                "concentration_ratio", "top_provider_share", "exposure_aed"],
      req=["member.employer_id"] + F_GEO),

    c("NET-04-R03", "Early-tenure high-risk cluster", "S", "MONTHLY",
      "High-cost claims shortly after enrollment cluster by originator/group beyond morbidity "
      "expectation",
      "Maternity/chronic continuity and guaranteed issue",
      "Review, never individual denial",
      "POSTPAY_AUDIT", "EARLY_TENURE_CLUSTER", "analytics", PART,
      "Early tenure (cfg.early_tenure_days) and claim value both exist, so clustering of "
      "high-value early-tenure claims by AGENT is computable. 'Beyond morbidity expectation' "
      "is approximated by the diagnosis-peer amount distribution rather than a morbidity "
      "model. The catalogue's instruction — 'review, NEVER INDIVIDUAL DENIAL' — is reproduced "
      "on the signal.",
      impl="net_04_r03_early_tenure_cluster",
      population="agents with at least cfg.min_entity_opportunities claims",
      inputs=["claim_header.agent_id", "claim_header.days_since_policy_start",
              "claim_header.gross_amount_aed", "diagnosis.code"],
      expression="the agent's share of early-tenure claims whose value exceeds the "
                 "diagnosis-peer median materially exceeds the cross-agent rate",
      exclusions=["maternity_continuity", "chronic_condition_continuity", "guaranteed_issue_product",
                  "morbidity_model_unavailable", "never_an_individual_denial"],
      grouping=["tenant_id", "payer_id", "agent_id", "scenario_id", "period_bucket"],
      evidence=["agent_id", "early_tenure_high_value_count", "agent_claim_count", "agent_rate",
                "overall_rate", "shrunk_rate", "posterior_interval", "exposure_aed",
                "never_individual_denial_notice"],
      params={"early_tenure_days": "cfg.early_tenure_days"},
      req=["member.employer_id", "morbidity_model.*"]),

    c("NET-04-R04", "Shared identifiers/actors", "N", "WEEKLY",
      "Broker, employer, member and provider share suspicious contact/payment/device links",
      "Lawful data and known corporate links",
      "SIU graph",
      "SIU_LEAD", "SHARED_IDENTIFIERS_ACTORS", "siu", NOPE,
      "No contact, payment or device identifiers.", req=F_IDENT),

    # ---- ANL-01 ------------------------------------------------------------
    c("ANL-01-R01", "Robust peer composite", "S", "MONTHLY",
      "Weighted robust residuals across scenario-aligned features exceed calibrated threshold",
      "Specialty/setting/size peer hierarchy",
      "Explainable entity lead",
      "POSTPAY_AUDIT", "ROBUST_PEER_COMPOSITE", "analytics", EXEC,
      "Fully implemented and fully auditable. This is THE BENCHMARK every model must beat "
      " and the system's explainable fallback when every model is switched off. "
      "Per-feature contributions are stored in ORIGINAL UNITS so a reviewer can reconstruct "
      "the score by hand from the weights in config/parameters.yaml.",
      impl="anl_01_r01_robust_composite",
      population="providers with at least cfg.min_entity_opportunities claims",
      inputs=["amount_residual", "amount_per_los_day_residual", "pharmacy_share_residual",
              "approval_ratio_residual", "utilisation_velocity_residual",
              "readmission_rate_residual", "coding_mismatch_rate_residual"],
      expression="weighted sum of robust (median/MAD) residuals across scenario-aligned "
                 "features > cfg.composite_flag_threshold",
      exclusions=["insufficient_peer_evidence", "posterior_interval_too_wide",
                  "below_minimum_denominator", "specialty_setting_size_peer_mismatch"],
      grouping=["tenant_id", "payer_id", "provider_sk", "scenario_id", "period_bucket"],
      evidence=["provider_sk", "composite_score", "per_feature_contributions",
                "per_feature_values_original_units", "per_feature_peer_medians", "weights",
                "peer_level_used", "peer_n", "threshold"],
      params={"threshold": "cfg.composite_flag_threshold", "weights": "cfg.composite_feature_weights"},
      req=[]),

    c("ANL-01-R02", "Self-history changepoint", "S", "MONTHLY",
      "CUSUM/Bayesian changepoint on value, mix, frequency or denial behavior",
      "Segment tariff/contract/ownership changes",
      "Entity lead",
      "POSTPAY_AUDIT", "SELF_HISTORY_CHANGEPOINT", "analytics", EXEC,
      "CUSUM and EWMA on monthly normalised provider VALUE and FREQUENCY both run, with "
      "cfg.min_history_periods enforced before a change point may be declared, and "
      "pre-change mean, post-change mean, change date and confidence stored for every one. "
      "MIX and DENIAL behaviour are unavailable (no codes, no remittance) and the signal "
      "names which metrics were evaluated.",
      impl="anl_01_r02_self_history_changepoint",
      population="providers with at least cfg.min_history_periods monthly periods",
      inputs=["claim_header.provider_sk", "claim_header.service_date",
              "claim_header.gross_amount_aed"],
      expression="CUSUM statistic on the monthly normalised metric crosses cfg.cusum_h, or "
                 "the EWMA statistic exits ±cfg.ewma_L sigma, after cfg.min_history_periods",
      exclusions=["known_tariff_change_segment", "contract_change_segment",
                  "ownership_change_segment", "insufficient_history"],
      grouping=["tenant_id", "payer_id", "provider_sk", "scenario_id", "period_bucket"],
      evidence=["provider_sk", "metric", "change_date", "pre_change_mean", "post_change_mean",
                "change_magnitude", "cusum_statistic", "ewma_statistic", "confidence",
                "periods_of_history", "declared_segments"],
      params={"k": "cfg.cusum_k", "h": "cfg.cusum_h", "lambda": "cfg.ewma_lambda",
              "L": "cfg.ewma_L", "min_history": "cfg.min_history_periods"},
      req=["contract.effective_dates (to segment known structural change)"]),

    c("ANL-01-R03", "Unsupervised incremental anomaly", "M", "MONTHLY",
      "Benchmarked model flags entity and adds lift beyond R01/R02",
      "Temporal validation, feature attribution, drift",
      "Monitor/SIU only",
      "MONITOR_ONLY", "UNSUPERVISED_INCREMENTAL_ANOMALY", "analytics", EXEC,
      "Isolation Forest and Local Outlier Factor are trained on a strictly earlier period "
      "with entity isolation, scored on a later one, explained with SHAP in original units "
      "beside the peer value, and monitored for drift. The disposition is MONITOR_ONLY — the "
      "catalogue's own choice — and the safety boundary makes it impossible for this control to be anything "
      "else. Whether it adds lift beyond R01/R02 is DECIDED BY THE PROMOTION GATE at run "
      "time and reported honestly either way.",
      impl="anl_01_r03_unsupervised_incremental",
      population="entities in the scoring period that were not used for training",
      inputs=["model_matrix (identifiers and protected attributes excluded)"],
      expression="model score exceeds the capacity-calibrated threshold AND the entity is not "
                 "already surfaced by ANL-01-R01 or ANL-01-R02",
      exclusions=["entity_in_training_period", "already_flagged_by_transparent_composite",
                  "model_in_shadow_status", "drift_breach_active"],
      grouping=["tenant_id", "payer_id", "provider_sk", "scenario_id", "period_bucket"],
      evidence=["provider_sk", "model_name", "model_version", "score", "score_percentile",
                "shap_top_features_original_units", "peer_comparison_values",
                "capacity_threshold", "incremental_over_composite", "training_period",
                "scoring_period", "feature_set_hash"],
      params={"contamination": "cfg.isolation_forest_contamination",
              "n_estimators": "cfg.isolation_forest_n_estimators",
              "n_neighbors": "cfg.lof_n_neighbors"},
      req=[]),

    c("ANL-01-R04", "Novel-cluster discovery", "M", "QUARTERLY",
      "Analyst-reviewed cluster exhibits coherent new typology and material exposure",
      "Exploratory; no production action until converted to rule",
      "Rule-development candidate",
      "MONITOR_ONLY", "NOVEL_CLUSTER_DISCOVERY", "policy_owner", EXEC,
      "HDBSCAN over the residual space produces candidate clusters for the quarterly "
      "typology review. Disposition is MONITOR_ONLY only, and no production action "
      "follows until a human converts a cluster into an authored rule — which itself enters "
      "the registry in shadow status and requires a separate approver.",
      impl="anl_01_r04_novel_cluster",
      population="the residual space of entities scored in the current period",
      inputs=["residual_matrix"],
      expression="HDBSCAN cluster of at least cfg.novel_cluster_min_size members with "
                 "coherent top residual features and material aggregate exposure",
      exclusions=["exploratory_only_no_production_action", "below_minimum_cluster_size",
                  "noise_points_excluded", "requires_analyst_review_before_rule_authoring"],
      grouping=["tenant_id", "payer_id", "cluster_id", "scenario_id", "period_bucket"],
      evidence=["cluster_id", "cluster_size", "top_residual_features", "feature_means",
                "aggregate_exposure_aed", "candidate_typology_name", "unvalidated_hypothesis_notice"],
      params={"min_cluster_size": "cfg.novel_cluster_min_size"},
      req=[]),

    # ---- POL-01 (added by the coverage audit) ------------------------------
    c("POL-01-R01", "Eligibility/roster conflict", "H", "DAILY",
      "Enrolled member lacks valid sponsor/employment/dependent relationship or duplicate "
      "active identity exists",
      "Continuation, newborn and mandated coverage",
      "Enrollment pend",
      "PREPAY_PEND", "ELIGIBILITY_ROSTER_CONFLICT", "underwriting", NOPE,
      "No sponsor, employer or dependent relationships, and no roster to conflict with.",
      req=["member.sponsor_id", "member.employer_id", "employer_roster.*"]),

    c("POL-01-R02", "Retroactive event after service", "H/S", "DAILY",
      "Member add/change occurs after high-cost service with actor/time pattern outside policy",
      "Approved retroactive correction/SLA",
      "Underwriting/compliance review",
      "PREPAY_PEND", "RETROACTIVE_ENROLMENT_EVENT", "underwriting", NOPE,
      "policy_event is NOT_POPULATED: there are no enrolment events, actors or timestamps, "
      "so 'occurs after' has nothing to order.",
      req=["policy_event.event_time", "policy_event.actor", "policy_event.event_type"]),

    c("POL-01-R03", "Application/claim fact conflict", "E", "DAILY",
      "Material application declaration conflicts with authoritative prior coverage/claim history "
      "where legally usable",
      "Contestability, non-discrimination and privacy law",
      "Human review only",
      "PREPAY_PEND", "APPLICATION_FACT_CONFLICT", "underwriting", NOPE,
      "No application declarations. The catalogue's own constraints — contestability, "
      "non-discrimination and privacy law — would in any case gate this control behind a "
      "legal review that no detection artefact can grant itself.",
      req=["policy_application.declarations", "prior_coverage_history.*"]),

    c("POL-01-R04", "Early-tenure utilization anomaly", "S", "MONTHLY",
      "Claim pattern shortly after inception exceeds tenure/risk-adjusted expectation",
      "Not evidence of fraud alone; maternity/chronic continuity",
      "Monitor/distribution signal",
      "MONITOR_ONLY", "EARLY_TENURE_UTILISATION", "underwriting", PART,
      "days_since_policy_start supports the tenure limb exactly, and claim value supports "
      "the utilisation limb. Risk adjustment is by primary diagnosis, not by a morbidity "
      "model. previous_fraud_on_policy is used as a POLICY-HISTORY input (it is a policy "
      "attribute in the source, not the held-out outcome label) and is reported separately "
      "in the evidence so a reviewer can weigh it themselves. The catalogue's caution — "
      "'not evidence of fraud alone' — is reproduced on the signal and the disposition is "
      "MONITOR_ONLY.",
      impl="pol_01_r04_early_tenure_utilisation",
      population="claims within cfg.early_tenure_days of policy inception",
      inputs=["claim_header.days_since_policy_start", "claim_header.gross_amount_aed",
              "diagnosis.code", "claim_header.previous_fraud_on_policy"],
      expression="days_since_policy_start < cfg.early_tenure_days and "
                 "robust_residual(gross_amount_aed, diagnosis_peer_group) > "
                 "cfg.robust_residual_flag_threshold",
      exclusions=["maternity_continuity", "chronic_condition_continuity", "guaranteed_issue_product",
                  "not_evidence_of_fraud_alone", "morbidity_model_unavailable"],
      grouping=["tenant_id", "payer_id", "member_sk", "scenario_id", "period_bucket"],
      evidence=["member_sk", "days_since_policy_start", "early_tenure_days", "gross_amount_aed",
                "peer_median_aed", "robust_residual", "diagnosis_code",
                "previous_fraud_on_policy", "peer_level_used", "not_fraud_evidence_alone_notice"],
      params={"early_tenure_days": "cfg.early_tenure_days",
              "threshold": "cfg.robust_residual_flag_threshold"},
      req=["morbidity_model.*", "policy_event.*"]),

    c("POL-01-R05", "Employer/broker enrollment cluster", "S/N", "MONTHLY",
      "Suspicious member additions, identity links or early claims cluster by employer/originator",
      "Portfolio size and enrollment campaign",
      "NET-04-linked case",
      "POSTPAY_AUDIT", "ENROLMENT_CLUSTER", "siu", NOPE,
      "Member ADDITIONS require policy_event, which is NOT_POPULATED, and identity links "
      "require administrative identifiers that do not exist. The 'early claims cluster by "
      "originator' limb alone is implemented, at NET-04-R03, rather than duplicated here "
      "under a control whose other two limbs cannot run.",
      req=["policy_event.*", "member.employer_id"] + F_IDENT),
]
