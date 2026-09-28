# Limitations — `uae-fwa-engine` 1.0.0

> **THIS REPORT DESCRIBES ONE DATASET**
>
> Source `claims_demo_synthetic.csv` · 20,893 claim rows · `GENERIC_INDIA_TPA` adapter · run 2026-09-28 21:18 UTC.
>
> Every figure below is a measured property of that file. None of it transfers to another population without being re-measured there.
>
> **SAFETY BOUNDARY.** A signal is not a fraud finding. This system can establish non-payability, inconsistency or statistical abnormality. It cannot establish intent, and intent is what distinguishes fraud from waste, abuse or honest error. Only a human reviewer, on evidence, may reach a conclusion about conduct.


## 1. Every figure here is a property of one input file

This run's input is a claim-header extract of 20,893 rows, in a non-AED source currency converted through `config/fx.yaml` at the service-date rate. Which controls can run at all, which are silent, and every rate below are all properties of that file's schema and contents. **Re-measure on your own data before treating any number here as a forecast of what it will do there.**

## 2. The labels were produced by detection processes

`ground_truth_source` takes three values — `pattern_detection`, `expert_review` and `rule_engine`. Every one of them is a detection process, so the labels carry that process's blind spots. Worse, profiling shows each `fraud_type` is encoded through a single dominant field, so a control testing that field recovers the type almost perfectly. **Precision against these labels is close to tautological for rule-based controls and is reported as an upper bound throughout.**

## 3. Most of the catalogue cannot run here

127 of 164 controls are classified `NOT_EXECUTABLE_ON_THIS_DATASET`. The four structural gaps, in order of consequence:

1. **No `authorization` table.** The entire PAY-04 scenario is unrunnable — including PAY-04-R01, the usual worked example of the atomic-control contract.
2. **Claim-header level only.** No activity codes, units or line amounts, so PAY-02, PAY-03, PAY-05, CLN-08 and all of PHR-01/02/04/05 have no input at all. The missingness rule says explicitly that no claim-level or provider-level aggregate can substitute for the code-pair co-occurrence feature.
3. **No `claim_version` lineage and no `remittance`.** PAY-08 and PAY-12 are dead, and 'prevented/recovered AED' is NOT_MEASURABLE.
4. **No geography.** Peer-hierarchy level 5 is `NOT_POPULATED` and every travel-impossibility and geographic-cluster control is classified out rather than run against a fabricated location.

## 4. Proxies are proxies

28 controls run against a PROXY rather than the field the catalogue specifies: ICD chapter for provider specialty, policy type for encounter/facility type, an undated blacklist flag for an effective-dated exclusion list, agent co-occurrence for a clinical referral edge. Each proxy is named on the signal itself, and each weakens the finding.

## 5. Nothing has been reviewed

Precision in the strict sense is *confirmed ÷ reviewed*. Nothing has been reviewed, so precision, review yield, confirmed AED, prevented/recovered AED, net savings, abrasion, turnaround, overturn rate and rule stability are all `NOT_MEASURABLE_ON_THIS_DATASET`. They are implemented and reported as unmeasurable rather than dropped.

## 6. The priority formula saturates on this data

The priority coefficients are reproduced exactly from the design and are not tuned. With `exposure_aed / 1000` inside the log and case exposures reaching six figures, the exposure term alone can contribute enough to push the sigmoid above 99. The UI therefore shows the **term-by-term breakdown and the raw logit**, so the saturation is visible rather than hidden behind a rounded score. A real deployment would set the divisor to the scale of its own exposures — it is a governed parameter, and the fact that it needs setting is itself a finding.

## 7. The adapters are not certified

`ShafafiyaAdapter` and `EClaimLinkAdapter` are schema-complete and exercised by fixtures. Both are marked **NOT CERTIFIED — no live regulator feed**. Neither has ever seen a real DoH or DHA submission.

## 8. This is a demonstration artefact, not a production system

The authentication layer demonstrates the access-control requirement. It uses Argon2 password hashing, enforces separation of duties and PHI masking, and writes an append-only audit log — and it is **not hardened for real PHI**. SQLite stands in for PostgreSQL behind a repository interface. There is no rota, no on-call, no penetration test and no threat model.
