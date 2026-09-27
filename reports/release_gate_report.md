# Release-gate report — `uae-fwa-engine` 1.0.0

> **THIS REPORT DESCRIBES ONE DATASET**
>
> Source `claims_demo_synthetic.csv` · 20,893 claim rows · `GENERIC_INDIA_TPA` adapter · run 2026-09-27 22:24 UTC.
>
> Every figure below is a measured property of that file. None of it transfers to another population without being re-measured there.
>
> **SAFETY BOUNDARY.** A signal is not a fraud finding. This system can establish non-payability, inconsistency or statistical abnormality. It cannot establish intent, and intent is what distinguishes fraud from waste, abuse or honest error. Only a human reviewer, on evidence, may reach a conclusion about conduct.


Each of the six release gates, evaluated against **this build**, with an explicit PASS / FAIL / NOT_ASSESSABLE and its evidence.

`NOT_ASSESSABLE` is **never** a pass. A gate that cannot be evaluated is a reason not to promote, not a reason to shrug — which is why every control in this build is still in shadow.


## Data readiness — **PASS**


**Minimum acceptance evidence:** ≥99.5% key linkage for claims and lines; scenario-specific completeness reported explicitly, not hidden by imputation


**Threshold:** `0.995` (a configuration acceptance criterion quoted from the specification, not a result achieved here)


**Measured:** `1.0`


**Evidence:** Key linkage (claim → member and provider) is 100.0000% against a required 99.5%. Scenario completeness IS reported explicitly: 10 of 17 canonical tables are NOT_POPULATED, each with a stated reason, and no field is imputed to hide the gap. The LINE half of this gate is vacuous here — there are no claim lines to link — so the measured figure covers claim-level linkage only.


## Hard edit — **NOT_ASSESSABLE**


**Minimum acceptance evidence:** Policy-owner sign-off; expected-versus-observed volume comparison; ≥99% decision reproducibility; documented exception coverage


**Threshold:** `0.99` (a configuration acceptance criterion quoted from the specification, not a result achieved here)


**Evidence:** 1 hard/expert control that may deny is executable here, and 0 carry a policy-owner sign-off — because no control in this build has been activated. Decision REPRODUCIBILITY is demonstrable (signal ids are a pure function of rule, version, subject, fact and period, and the idempotency test asserts a replay produces no duplicate), but reproducibility against an EXISTING ADJUDICATION SYSTEM — which is what this gate means — cannot be measured without that system's decisions. Exception coverage IS documented: every control declares its exclusions or fails to register.


## Expert edit — **NOT_ASSESSABLE**


**Minimum acceptance evidence:** Clinical/coding validation sample; documented false-positive rate and an appeal route


**Evidence:** No clinical or coding validation sample has been reviewed, so no false-positive rate exists. An appeal route IS implemented — the Case Evidence page captures an appeal and its result against review_outcome — but it has never been exercised, and an untested route is not evidence.


## Statistical rule — **NOT_ASSESSABLE**


**Minimum acceptance evidence:** A prospective shadow period; minimum peer sizes met; stable volume; review yield above an agreed baseline


**Threshold:** `review yield ≥ 20%, shadow ≥ 30 days` (a configuration acceptance criterion quoted from the specification, not a result achieved here)


**Evidence:** 22 statistical controls execute here. MINIMUM PEER SIZES: every peer group used met the cfg.min_peer_group_n floor of 30. PROSPECTIVE SHADOW PERIOD: not satisfiable by a single retrospective run — the gate requires 30 days of forward observation. REVIEW YIELD: NOT_MEASURABLE (no reviewer dispositions), so the 20% baseline cannot be tested.


## Model — **PASS**


**Minimum acceptance evidence:** Temporal holdout evaluation; calibration check; demonstrated prospective lift over the simple statistical baseline; drift and explanation-quality tests


**Measured:** `1 promoted of 2`


**Evidence:** isolation_forest: REMAINS IN SHADOW; failed Calibration check | local_outlier_factor: PROMOTED


## Production — **NOT_ASSESSABLE**


**Minimum acceptance evidence:** Rollback capability, kill switch, an alert-volume ceiling, audit logs and a named responsible owner on call


**Evidence:** Rollback capability: RuleRegistry.rollback() restores a prior registered version and re-enters it in SHADOW, never straight to active. Kill switch: RuleRegistry.kill_switch() routes a control's signals to MONITOR_ONLY without a code deployment, with a mandatory reason. Alert-volume ceiling: cfg.alert_volume_ceiling_per_rule_per_run = 400; 7 breach(es) this run (CLN-03-R01, CLN-04-R01, CLN-04-R02, CLN-05-R01, PAY-06-R03, PAY-10-R01, POL-01-R04), each routed to MONITOR_ONLY and audited. Audit logs: Append-only and hash-chained. Verified 9 entries; hash chain intact. Named responsible owner: 11 distinct owners across the catalogue; a control with no owner cannot be registered. ON CALL is the one element that cannot be evidenced by software: it is a rota, not a capability, and this artefact has no one on call. The four software elements are demonstrable and demonstrated; the gate as a whole is therefore reported NOT_ASSESSABLE rather than passed.
