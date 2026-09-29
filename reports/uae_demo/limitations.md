# Limitations — `uae-fwa-engine` 1.0.0

> **THIS REPORT DESCRIBES ONE DATASET**
>
> Source `uae_demo.zip` · 24,855 claim rows · `UAE_MULTITABLE` adapter · run 2026-09-29 02:01 UTC.
>
> Every figure below is a measured property of that file. None of it transfers to another population without being re-measured there.
>
> **SAFETY BOUNDARY.** A signal is not a fraud finding. This system can establish non-payability, inconsistency or statistical abnormality. It cannot establish intent, and intent is what distinguishes fraud from waste, abuse or honest error. Only a human reviewer, on evidence, may reach a conclusion about conduct.


## 1. Every figure here is a property of one input file

This run's input is `uae_demo.zip`: 24,855 claims and 62,009 claim lines across the canonical tables. Amounts are in AED as supplied. Which controls can run, which are silent, and every rate below are properties of that file's schema and contents. **Re-measure on your own data before treating any number here as a forecast of what it will do there.**

## 2. How the labels were produced

Its labels are the answer key of the generator that planted the patterns (`synthetic_injection`). The injectors were written with the checks in mind, so every precision and recall figure below is closer to a self-test than to an estimate of real performance, and an upper bound at best. **Precision and recall against these labels are reported as upper bounds throughout.**

## 3. What cannot run here

1 control cannot run on this file:

- **NET-04-R01** — This control requires CONFIRMED REVIEW OUTCOMES as its label, and the catalogue is explicit that raw system flags must never be used as one. review_outcome is empty at load, and fraud_label is held-out evaluation data that no detection component may read. Implementing it against fraud_label would be precisely the leakage this artefact is built to prevent. It becomes executable once the artefact has accumulated real reviewer dispositions.

## 4. Simplified checks are simplified

86 controls run in a simplified form on this file: a proxy stands in for a field or table the catalogue specifies. Each simplification is named on the signal itself, and each weakens the finding.


## 5. Nothing has been reviewed

Precision in the strict sense is *confirmed ÷ reviewed*. Nothing has been reviewed, so precision, review yield, confirmed AED, prevented/recovered AED, net savings, abrasion, turnaround, overturn rate and rule stability are all `NOT_MEASURABLE_ON_THIS_DATASET`. They are implemented and reported as unmeasurable rather than dropped.

## 6. The priority formula saturates on this data

The priority coefficients are reproduced exactly from the design and are not tuned. With `exposure_aed / 1000` inside the log and case exposures reaching six figures, the exposure term alone can contribute enough to push the sigmoid above 99. The UI therefore shows the **term-by-term breakdown and the raw logit**, so the saturation is visible rather than hidden behind a rounded score. A real deployment would set the divisor to the scale of its own exposures — it is a governed parameter, and the fact that it needs setting is itself a finding.

## 7. The adapters are not certified

`ShafafiyaAdapter` and `EClaimLinkAdapter` are schema-complete and exercised by fixtures. Both are marked **NOT CERTIFIED — no live regulator feed**. Neither has ever seen a real DoH or DHA submission.

## 8. This is a demonstration artefact, not a production system

The authentication layer demonstrates the access-control requirement. It uses Argon2 password hashing, enforces separation of duties and PHI masking, and writes an append-only audit log — and it is **not hardened for real PHI**. SQLite stands in for PostgreSQL behind a repository interface. There is no rota, no on-call, no penetration test and no threat model.
