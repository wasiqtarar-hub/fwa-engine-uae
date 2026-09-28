# Model card — `uae-fwa-engine` 1.0.0

> **THIS REPORT DESCRIBES ONE DATASET**
>
> Source `claims_demo_synthetic.csv` · 20,893 claim rows · `GENERIC_INDIA_TPA` adapter · run 2026-09-28 21:18 UTC.
>
> Every figure below is a measured property of that file. None of it transfers to another population without being re-measured there.
>
> **SAFETY BOUNDARY.** A signal is not a fraud finding. This system can establish non-payability, inconsistency or statistical abnormality. It cannot establish intent, and intent is what distinguishes fraud from waste, abuse or honest error. Only a human reviewer, on evidence, may reach a conclusion about conduct.


## Intended use

These models produce **leads for human review**. They cannot deny a claim, change its price, or set any disposition. The safety boundary makes that structural: a control of type M declaring `REJECT` or `REPRICE` is rejected by the contract validator at registration time, so the system cannot be started with one loaded.

## Out of scope

Supervised propensity modelling. The supervised-prerequisite gate returns `PROMOTION_BLOCKED`; no classifier is trained anywhere in this repository.


## Models

| model                | version   |   training_rows |   scored_rows |   capacity_threshold |   flagged |   features | params                                                                 | notes                                                                                                                                  |
|:---------------------|:----------|----------------:|--------------:|---------------------:|----------:|-----------:|:-----------------------------------------------------------------------|:---------------------------------------------------------------------------------------------------------------------------------------|
| isolation_forest     | 1.0.0     |            6319 |          4173 |               0.5546 |       100 |        104 | {'n_estimators': 300, 'contamination': 0.05, 'random_state': 20260920} | Trained on the earlier period, on providers held out from scoring. Threshold set by reviewer capacity, not by contamination.           |
| local_outlier_factor | 1.0.0     |            6319 |          4173 |               2.2943 |       100 |        104 | {'n_neighbors': 35, 'novelty': True}                                   | Retained because it surfaces anomalies that are only anomalous relative to a LOCAL peer cluster — a pattern Isolation Forest can miss. |


## Training and scoring data

| property         | value                                                                                                                                                                                                                                                                                                                                                                   |
|:-----------------|:------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| split_date       | 2026-02-11                                                                                                                                                                                                                                                                                                                                                              |
| train_claims     | 6319                                                                                                                                                                                                                                                                                                                                                                    |
| score_claims     | 4173                                                                                                                                                                                                                                                                                                                                                                    |
| train_providers  | 675                                                                                                                                                                                                                                                                                                                                                                     |
| score_providers  | 665                                                                                                                                                                                                                                                                                                                                                                     |
| provider_overlap | 0                                                                                                                                                                                                                                                                                                                                                                       |
| excluded_claims  | 10401                                                                                                                                                                                                                                                                                                                                                                   |
| total_claims     | 20893                                                                                                                                                                                                                                                                                                                                                                   |
| coverage_note    | 10,401 of 20,893 claims (50%) are scored by NEITHER model: they belong to a provider held out for training, or fall in the training period. This is the price of combining a temporal split with entity isolation on a single-file dataset, and it is reported rather than absorbed — a model that scored everything would have been trained on the entities it scores. |


## Features


104 numeric features plus their explicit missingness indicators. **Excluded by construction:** the held-out label columns (physically absent from every `FeatureFrame`), direct identifiers (`hospital_id`, `patient_id`, `agent_id`, claim and episode keys) and every protected demographic attribute. Features are computed strictly from data available BEFORE the scored period — provider and member aggregates are expanding, time-ordered and shifted.


## Explanation


SHAP per flagged entity, rendered in **original units beside the peer median** — "Amount per inpatient day: AED 2,649/day, peer median AED 912/day (2.9×)" — not as bare SHAP magnitudes. Methods in use: `isolation_forest` → TreeExplainer, `local_outlier_factor` → KernelExplainer (kmeans-summarised background, 25 centroids).


## Monitoring


**isolation_forest**


| monitor                                |    value | status         | note                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 |
|:---------------------------------------|---------:|:---------------|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Population stability (max feature PSI) |   0.1819 | WARN           | PSI detects that the INPUT distribution moved between the training and scoring periods. It does not say the model got worse. Expect a structural breach on this dataset: the prior-history features (a member's or provider's earlier claim count, spend and rates) are EXPANDING windows, so they are near-empty at the start of the file and well populated later. That is real drift in the input, and the honest consequence is that the model gate fails on the drift criterion and the models stay in shadow. Top drifting features: agent_top_provider_share_prior (PSI 0.1819), agent_prior_claim_count (PSI 0.1461), provider_prior_mean_approval_ratio (PSI 0.1315), provider_prior_mean_pharmacy_ratio (PSI 0.1166), agent_window_claim_count (PSI 0.091) |
| Score-distribution drift (PSI)         | nan      | NOT_MEASURABLE | Requires a stored reference score distribution from a prior run. A single-run artefact has none; it is created on the first run and compared on the second.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          |
| Alert volume vs expectation            | 100      | OK             | No expected_alert_volume is registered for this model.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               |
| Review yield (confirmed ÷ reviewed)    | nan      | NOT_MEASURABLE | NOT_MEASURABLE_ON_THIS_DATASET until reviewers record dispositions. Review yield is the primary signal of whether a model is still earning its place; substituting a label-derived proxy here would be exactly the leakage the artefact forbids.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     |


**local_outlier_factor**


| monitor                                |    value | status         | note                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 |
|:---------------------------------------|---------:|:---------------|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Population stability (max feature PSI) |   0.1819 | WARN           | PSI detects that the INPUT distribution moved between the training and scoring periods. It does not say the model got worse. Expect a structural breach on this dataset: the prior-history features (a member's or provider's earlier claim count, spend and rates) are EXPANDING windows, so they are near-empty at the start of the file and well populated later. That is real drift in the input, and the honest consequence is that the model gate fails on the drift criterion and the models stay in shadow. Top drifting features: agent_top_provider_share_prior (PSI 0.1819), agent_prior_claim_count (PSI 0.1461), provider_prior_mean_approval_ratio (PSI 0.1315), provider_prior_mean_pharmacy_ratio (PSI 0.1166), agent_window_claim_count (PSI 0.091) |
| Score-distribution drift (PSI)         | nan      | NOT_MEASURABLE | Requires a stored reference score distribution from a prior run. A single-run artefact has none; it is created on the first run and compared on the second.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          |
| Alert volume vs expectation            | 100      | OK             | No expected_alert_volume is registered for this model.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               |
| Review yield (confirmed ÷ reviewed)    | nan      | NOT_MEASURABLE | NOT_MEASURABLE_ON_THIS_DATASET until reviewers record dispositions. Review yield is the primary signal of whether a model is still earning its place; substituting a label-derived proxy here would be exactly the leakage the artefact forbids.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     |


## Promotion status


- **isolation_forest: REMAINS IN SHADOW** — isolation_forest REMAINS IN SHADOW. Failed: Calibration check. The system loses nothing by this: ANL-01-R01, the transparent composite, is the explainable fallback and continues to run.


- **local_outlier_factor: PROMOTED TO ACTIVE** — local_outlier_factor clears every model-gate criterion and may be promoted from shadow to active by a policy owner who did not propose it.


## Ethical considerations

A model score here is evidence that an entity is UNUSUAL. It is not evidence of intent, and the difference between the two is the defining requirement of the whole system. The exposure attached to a model-only lead renders as the gross amount plus the literal statement *"exposure not yet established"*, because reporting it as savings would overstate the system's contribution — this system treats that as an ethical requirement, not a measurement nicety.
