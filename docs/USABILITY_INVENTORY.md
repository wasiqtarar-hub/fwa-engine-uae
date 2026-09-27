# Usability inventory (before the plain-language work)

This is the "before" picture the usability brief asks for: every user-facing surface, which parts
speak in code terms, and which controls cannot run and why. It was produced by reading every page
module and extracting widgets, tables, charts and strings with an AST pass, and by reading
`reports/control_coverage_matrix.csv` for the controls.

## 1. Pages, widgets, tables and charts

Counts are from the page source. "Help" is the number of widgets that carry a `help=` tooltip.

| Page (internal name) | Widgets | Help | Tables | Charts | What speaks in code |
|---|---|---|---|---|---|
| Home / sign-in | 8 | 1 | 1 (demo accounts, raw role enum) | 0 | Role shown as `CLAIMS_REVIEWER`, tenant `T001` |
| Overview | 0 | 0 | 5 | 3 | `NOT_MEASURABLE_ON_THIS_DATASET` inline; canonical table names and `POPULATED`/`NOT_POPULATED`; coverage table keyed by `EXECUTABLE`/`PARTIAL`/`NOT_EXECUTABLE_ON_THIS_DATASET`; chart of controls by `rule_id`; `dimension_label`, `priority_band`; timings keyed by internal stage names; seed and fingerprint |
| Data | 9 | 1 | 5 | 0 | Adapter ids, `sha256`, `NOT_EXECUTABLE_ON_THIS_DATASET`, per-control table keyed by `rule_id` with raw outcome/support enums, `by_data_support` |
| Review Queue | 8 | 1 | 2 | 0 | Columns `case_id`, `dimension_label`, `scenario_family` values such as `coding_integrity`, disposition enums (`POSTPAY_AUDIT`), "Precision proxy @ capacity"; sliders "Reviewers on the team", "Alerts per reviewer per day", "Minimum priority" with no tooltip; "by control" table of `rule_id` × disposition |
| Case Evidence | 13 | 2 | 7 | 0 | Headline falls back to "`RULE-ID` fired … with reason code `REASON_CODE`"; signal cards titled `rule_id — REASON_CODE`; "underlying facts" table of raw evidence keys (`peer_level_used`, `robust_residual`, `posterior_interval`, `proxy_note` …); priority term table with `raw_value`/`normalised`; exposure table `exposure_aed`/`established`; SHAP table `shap_value`/`ratio_to_peer`; category options `billing_error`, `utilisation_concern`; adjusted-disposition list of raw enums |
| Provider Analytics | 2 | 0 | 6 | 4 | Tab/section titles "ANL-01-R01 — the transparent composite", "Empirical-Bayes shrinkage", "CUSUM/EWMA"; columns `composite_score`, `peer_level_used`, `shrunk_rate`, `posterior_low/high`, `pre_change_mean`, `cusum_h`, `ewma_lambda` |
| Network | 6 | 0 | 4 | 2 | "Neighbourhood depth" slider with no explanation; columns `community_id`, `internal_density`, `external_ratio`, `matched_fields`; `NET-02`, `NOT_EXECUTABLE`; Louvain named without translation |
| Models | 7 | 0 | 9 | 4 | "Promotion gate", `PROMOTION_BLOCKED`, `NOT_ASSESSABLE`; "Lift over the transparent composite", `review_yield_proxy`; PSI, `warn_threshold`/`breach_threshold`; SHAP `mean_abs_shap`; feature names such as `agent_top_provider_share_prior`; "ANL-01-R04 — novel-cluster discovery"; HDBSCAN; `split_date`, `provider_overlap` |
| Rule Registry | 13 | 0 | 3 | 1 | Every control by id and internal name; `data_support`, `run_outcome`, `signals_this_run`; type letters `S`, `H/E`; stage enums |
| Governance | 8 | 0 | 6 | 3 | Parameter keys (`alert_volume_ceiling_per_rule_per_run`), audit event enums (`KILL_SWITCH_ENGAGED`), stage enums (`PREPAY_SYNC`), `expected_alert_volume` |
| Parameters & Models | 2 | 0 | 11 | 0 | Parameter keys, `min_peer_group_n`, `prior_strength`, `scale_source`, fitted-hyperparameter table (`n_estimators`, `contamination`, `n_neighbors`, `decision_threshold (fitted)`), peer back-off levels |
| Validation Report | 5 | 0 | 1 | 0 | Artefact list by file name (`control_coverage_matrix`, `precision_by_ground_truth_source` …); no plain summary before the full report |
| User Management | 24 | 0 | 3 | 0 | Role and permission enums (`MANUAL_OVERRIDE`, `SIU_LEAD`), audit event enums |
| Help | 2 | 0 | 0 | 0 | Renders `docs/USER_GUIDE.md`, written for a technical reader; no glossary |

Totals: **107 widgets, 5 with a tooltip; 63 tables; 17 charts.** No page opens with a sentence
that says what it is for. No chart title states a finding. Several tables show `True`/`False`,
`None` and `nan` directly.

### Terms used without translation anywhere in the default view

Isolation Forest, Local Outlier Factor, anomaly score, SHAP, PSI / drift, peer group, median and
MAD, empirical-Bayes shrinkage, Beta prior, posterior interval, CUSUM, EWMA, change point,
Louvain community, entity resolution, HDBSCAN, contamination, `n_neighbors`, capacity threshold,
shadow mode, promotion gate, precision, recall, lift, calibration / ECE, evidence capping,
correlation dimension, disposition.

### Enumerations shown raw

Dispositions (8): `RETURN`, `REJECT`, `REPRICE`, `PREPAY_PEND`, `POSTPAY_AUDIT`, `SIU_LEAD`,
`PROVIDER_EDUCATION`, `MONITOR_ONLY`. Data support (3): `EXECUTABLE`, `PARTIAL`,
`NOT_EXECUTABLE_ON_THIS_DATASET`. Run outcomes (7): `TRIGGERED`, `NOT_TRIGGERED`,
`NOT_EFFECTIVE`, `KILL_SWITCHED`, `NULL_INPUT`, `INSUFFICIENT_PEER_EVIDENCE`,
`NOT_EXECUTABLE_ON_THIS_DATASET`. Gate and metric statuses: `PASS`, `FAIL`, `WARN`, `BREACH`,
`PROMOTED`, `SHADOW`, `PROMOTION_BLOCKED`, `NOT_ASSESSABLE`, `NOT_MEASURABLE_ON_THIS_DATASET`,
`INFORMATIONAL`. Table population: `POPULATED`, `PARTIAL`, `NOT_POPULATED`. Control types
(`H`, `E`, `S`, `N`, `T`, `M` and composites), stages (6), rule status (`shadow`, `active`,
`retired`), roles (8), permissions, audit event types, case families (`coding_integrity` …),
correlation dimensions, evidence domains, reason codes (164).

## 2. Controls that cannot run on `claims_demo_synthetic.csv`

9 controls are fully executable, 28 run in a simplified (proxy) form, and **127 cannot run**.
Grouped below by the first canonical table each one is missing; the bracket lists every table the
control needs. Many of the tables named are not part of the sixteen-table canonical model at all
(reference and operational tables such as `bundling_edit_table` or `clinician_roster`), so no
claim file could have supplied them before this work.

| Primary missing table | Controls | Each control: every table it needs |
|---|---|---|
| `claim_line` | 33 | CLN-01-R04 Level migration changepoint (claim_line)<br>CLN-03-R02 Unsupported indication (claim_line, indication_policy)<br>CLN-03-R03 Procedure sequence conflict (claim_line, observation)<br>CLN-03-R04 Rare pair/specialty anomaly (claim_line, provider)<br>CLN-04-R05 Cascade pattern (claim_line, referral)<br>CLN-07-R01 Improbable service day (claim_line, provider)<br>CLN-08-R01 No qualified order/relationship (claim_line, provider)<br>CLN-08-R02 Panel/component inflation (claim_line, panel_definition)<br>CLN-08-R04 Reference-lab/pass-through spread (claim_line)<br>CLN-08-R05 High-complexity/genetic test outlier (claim_line)<br>ENT-03-R04 Soft specialty mismatch (claim_line, provider)<br>ENT-04-R01 Missing/invalid rendering clinician (claim_line, provider)<br>ENT-04-R02 Billing-rendering role conflict (claim_line, provider)<br>ENT-04-R03 Concurrent clinician services (claim_line, provider)<br>ENT-04-R04 Geographic or leave impossibility (claim_line, provider, encounter, clinician_roster)<br>ENT-06-R01 Telehealth eligibility/setting failure (claim_line, provider, encounter)<br>ENT-06-R04 Remote-order conversion spike (claim_line, prescription_dispense, referral)<br>PAY-02-R01 Same-claim component edit (claim_line, bundling_edit_table)<br>PAY-02-R02 Cross-claim/cross-provider component (claim_line, bundling_edit_table)<br>PAY-02-R03 Package completeness/zero-price lines (claim_line, package_definition)<br>PAY-02-R04 DRG/inclusive leakage (claim_line, remittance, drg_grouper)<br>PAY-02-R05 Novel unbundling pattern (claim_line)<br>PAY-03-R01 Invalid/effective-date code (claim_line, code_system_version)<br>PAY-03-R03 Unit maximum (claim_line, unit_maximum_policy)<br>PAY-03-R04 Time-unit inconsistency (claim_line)<br>PAY-05-R01 Prohibited code-indicator pair (claim_line)<br>PAY-05-R02 Required support absent (claim_line, observation, document)<br>PAY-05-R03 Provider use-rate outlier (claim_line)<br>PAY-09-R02 Covered-code substitution pattern (claim_line, remittance)<br>PAY-09-R03 Cosmetic/alternative/non-medical disguise (claim_line, encounter, disguise_risk_policy)<br>PHR-05-R01 New/used/rental/purchase mismatch (claim_line, device_inventory, authorization, authorization_line)<br>PHR-05-R02 Implant/device not linked to procedure (claim_line, observation, document)<br>PHR-05-R03 Supply quantity/consumption excess (claim_line, device_inventory) |
| `prescription_dispense` | 16 | CLN-06-R05 No longitudinal clinical footprint (prescription_dispense, observation, claim_line)<br>PHR-01-R01 Product identity mismatch (prescription_dispense)<br>PHR-01-R02 Quantity/strength/form mismatch (prescription_dispense, authorization, authorization_line, claim_line)<br>PHR-01-R03 Dispense-claim mismatch (prescription_dispense, pharmacy_inventory)<br>PHR-01-R04 Non-medical/uncovered substitution (prescription_dispense, member_receipt, pharmacy_inventory)<br>PHR-02-R01 Refill overlap (prescription_dispense)<br>PHR-02-R02 Therapy-duration excess (prescription_dispense, drug_policy)<br>PHR-02-R03 Multi-prescriber same-equivalent drug (prescription_dispense)<br>PHR-02-R04 Multi-pharmacy convergence (prescription_dispense)<br>PHR-03-R01 Dose/weight/body-surface conflict (prescription_dispense, member)<br>PHR-03-R02 Drug-diagnosis/step-therapy conflict (prescription_dispense, step_therapy_policy)<br>PHR-03-R04 Wastage/unused-vial anomaly (prescription_dispense, claim_line)<br>PHR-04-R01 Top-pharmacy concentration (prescription_dispense, provider, encounter)<br>PHR-04-R02 Reciprocal value loop (prescription_dispense, provider)<br>PHR-04-R03 High-cost steering (prescription_dispense)<br>PHR-04-R04 Rapid relationship formation (prescription_dispense) |
| `observation` | 11 | CLN-04-R03 No result before repeat (observation)<br>CLN-06-R01 Missing expected order/result trail (observation, claim_line)<br>CLN-08-R03 Absent/duplicate result (observation)<br>DOC-01-R01 Required document absent (observation, document, document_requirement_policy)<br>DOC-01-R03 Medical-necessity support absent (observation, document, medical_necessity_policy)<br>DOC-02-R02 Contradictory copied facts (observation, document, member)<br>DOC-02-R03 Post-denial material alteration (observation, document, remittance, claim_version)<br>DOC-02-R04 Template-to-complexity mismatch (observation, document, claim_line)<br>ENT-05-R02 Admission evidence conflict (observation, encounter)<br>ENT-06-R02 No meaningful clinical interaction (observation, document, encounter)<br>PAY-09-R01 Code-description/document conflict (observation, document, benefit_rule_version) |
| `remittance` | 9 | PAY-01-R04 Cross-payer duplicate (remittance, cross_payer_match_token, coordination_of_benefits)<br>PAY-05-R04 Post-edit migration (remittance, claim_line)<br>PAY-10-R04 Duplicate recovery after settlement (remittance, third_party_settlement)<br>PAY-11-R01 Unauthorized/manual override (remittance, adjudication_event)<br>PAY-11-R03 Post-settlement upward adjustment (remittance)<br>PAY-12-R01 Paid cancelled/reversed claim (remittance, claim_version)<br>PAY-12-R02 Duplicate remittance/payment reference (remittance)<br>PAY-12-R03 Unapplied provider refund/credit (remittance, recovery_ledger)<br>PAY-12-R04 Negative/offset manipulation (remittance) |
| `authorization` | 6 | DOC-01-R04 Authorization narrative drift (authorization, authorization_line, claim_line, observation, document)<br>PAY-04-R01 Missing required authorization (authorization, authorization_line, claim_line, benefit_rule_version)<br>PAY-04-R02 Invalid timing/status (authorization, authorization_line, claim_line)<br>PAY-04-R03 Code/provider/facility scope mismatch (authorization, authorization_line, claim_line)<br>PAY-04-R04 Quantity/value exhaustion (authorization, authorization_line, claim_line, remittance)<br>PAY-04-R05 Authorization reuse (authorization, authorization_line, claim_line) |
| `diagnosis` | 5 | CLN-02-R01 Payment-changing secondary diagnosis (diagnosis, drg_grouper)<br>CLN-02-R02 Comorbidity prevalence outlier (diagnosis, drg_grouper)<br>CLN-02-R03 Prior-history inconsistency (diagnosis, longitudinal member history across payers)<br>CLN-02-R04 POA/sequencing conflict (diagnosis)<br>CLN-02-R05 Severity-mix changepoint (diagnosis, drg_grouper) |
| `member` | 5 | ENT-02-R01 Authoritative identity conflict (member)<br>ENT-02-R02 Post-mortem service (member)<br>NET-03-R02 Small closed member group (member, provider, encounter)<br>PAY-03-R02 Hard demographic impossibility (member, claim_line)<br>POL-01-R01 Eligibility/roster conflict (member, employer_roster) |
| `provider` | 5 | ENT-02-R04 Card-sharing utilization pattern (provider, encounter, member)<br>ENT-03-R05 Dormant/new/non-operational provider burst (provider, provider_status_period)<br>ENT-05-R01 Facility-type incompatibility (provider, encounter, claim_line)<br>NET-02-R01 Shared administrative identity (provider)<br>NET-04-R04 Shared identifiers/actors (provider) |
| `claim_header` | 5 | NET-03-R03 Inducement signature (claim_header, claim_line, prescription_dispense)<br>PAY-07-R01 Expected-share mismatch (claim_header, benefit_rule_version)<br>PAY-07-R02 Share shifted to payer (claim_header)<br>PAY-07-R03 Systematic zero/rounded share (claim_header)<br>PAY-10-R03 Accident/liability indicator missing (claim_header, third_party_liability) |
| `referral` | 4 | ENT-06-R03 Downstream referral concentration (referral, prescription_dispense)<br>NET-01-R02 Reciprocal referral loop (referral)<br>NET-01-R03 High-cost conversion (referral, claim_line)<br>NET-01-R04 Closed downstream chain (referral, provider) |
| `claim_version` | 4 | PAY-08-R01 Missing/broken lineage (claim_version)<br>PAY-08-R02 Reimbursement-enabling field mutation (claim_version, remittance)<br>PAY-08-R03 Repeated resubmission loop (claim_version)<br>PAY-08-R04 Edit-learning success anomaly (claim_version, remittance) |
| `encounter` | 3 | CLN-05-R04 Admission-rate outlier (encounter)<br>ENT-05-R03 Related-claim setting conflict (encounter, claim_header)<br>ENT-05-R04 Setting-shift anomaly (encounter) |
| `coverage_period` | 2 | ENT-01-R01 Coverage inactive (coverage_period, benefit_rule_version)<br>ENT-01-R04 Network/referral violation (coverage_period, provider_status_period) |
| `benefit_rule_version` | 2 | ENT-01-R02 Benefit not covered (benefit_rule_version, claim_line)<br>ENT-01-R03 Benefit limit exceeded (benefit_rule_version, remittance) |
| `provider_status_period` | 2 | ENT-03-R01 Inactive/expired licence (provider_status_period, provider)<br>ENT-03-R03 Hard privilege violation (provider_status_period, provider, claim_line) |
| `member_receipt` | 2 | PAY-07-R04 Excess patient charge/balance bill (member_receipt, complaint, claim_header)<br>PAY-09-R04 Member/provider confirmation mismatch (member_receipt, member_confirmation) |
| `adjudication_event` | 2 | PAY-11-R02 Adjudicator-provider concentration (adjudication_event)<br>PAY-11-R04 Suspicious full-pay/invariant decisions (adjudication_event, remittance) |
| `policy_event` | 2 | POL-01-R02 Retroactive event after service (policy_event)<br>POL-01-R05 Employer/broker enrollment cluster (policy_event, member, provider) |
| `member_confirmation` | 1 | CLN-06-R04 Member denial/attendance mismatch (member_confirmation, attendance_record) |
| `equipment_inventory` | 1 | CLN-07-R03 Scarce-equipment concurrency (equipment_inventory, claim_line) |
| `contract` | 1 | NET-02-R04 Structural-change alert (contract, provider) |
| `complaint` | 1 | NET-03-R04 Complaint/confirmation corroboration (complaint, member_confirmation) |
| `review_outcome` | 1 | NET-04-R01 Confirmed-outcome rate (review_outcome) |
| `tariff` | 1 | PAY-06-R01 Tariff/contract price variance (tariff, contract, claim_line) |
| `other_payer_remittance` | 1 | PAY-10-R02 Multi-payer overpayment (other_payer_remittance, claim_header) |
| `device_inventory` | 1 | PHR-05-R04 Serial/inventory duplication (device_inventory, claim_line) |
| `policy_application` | 1 | POL-01-R03 Application/claim fact conflict (policy_application, prior_coverage_history) |

The same list in words, by the kind of information that is missing:

| Missing information (plain words) | Canonical source |
|---|---|
| Line-level service details: which procedures, tests and drugs were billed, how many, by whom | `claim_line` |
| Pharmacy prescriptions and what was actually dispensed | `prescription_dispense`, `pharmacy_inventory` |
| Test results, reports and attachments | `observation`, `document` |
| Hospital and clinic details: licence, specialty, facility type, ownership, bank/phone/address | `provider`, `provider_status_period`, `clinician_roster` |
| Payment decisions, denials and refunds | `remittance`, `adjudication_event`, `recovery_ledger` |
| Corrections and resubmissions of a claim | `claim_version` |
| What the policy covers, limits and patient share | `benefit_rule_version`, `coverage_period`, `payment_policy`, `tariff`, `contract` |
| Prior approvals | `authorization`, `authorization_line` |
| Patient details: date of birth, sex, date of death, employer | `member`, `employer_roster` |
| Referrals between providers | `referral` |
| Enrolment events and applications | `policy_event`, `policy_application`, `prior_coverage_history` |
| Patient confirmations, receipts and complaints | `member_confirmation`, `member_receipt`, `complaint` |
| Clinical coding reference tables | `bundling_edit_table`, `dx_proc_prohibition_table`, `indication_policy`, `repeat_interval_policy`, `care_pathway_policy`, `unit_maximum_policy`, `panel_definition`, `drg_grouper`, `morbidity_model`, `code_system_version` |
| Other payers, accidents and third parties | `other_payer_remittance`, `coordination_of_benefits`, `third_party_liability`, `third_party_settlement` |
| Equipment and device records | `device_inventory`, `equipment_inventory` |
| Attendance records | `attendance_record` |
| Reviewer decisions (only exist once reviewers use the tool) | `review_outcome` |

## 3. One structural finding that shapes the work

Whether a control can run is **static**: it is written into each control's YAML as
`data_support`, and the contract (`src/fwa/engine/contract.py`) forbids a control classified
`NOT_EXECUTABLE_ON_THIS_DATASET` from naming any implementation. Those classifications describe
the claim-header extract the catalogue was classified against. They cannot be flipped to make
controls run on a richer dataset without breaking the contract invariant and
`tests/governance/test_catalogue_completeness.py` (which also pins the static runnable count to
25–60).

The work therefore adds executability **per dataset** alongside the static classification rather
than replacing it: a separately governed declaration says which canonical tables would unlock a
control and which implementation then runs, and the evaluator records, per run, the effective
classification and why. On the claim-header files nothing is unlocked and every existing figure
is unchanged. See `docs/DEMO_DATASET.md` (UAE section) for the result.
