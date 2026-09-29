# Validation report — `uae-fwa-engine` 1.0.0

> **THIS REPORT DESCRIBES ONE DATASET**
>
> Source `uae_demo.zip` · 24,855 claim rows · `UAE_MULTITABLE` adapter · run 2026-09-29 02:27 UTC.
>
> Every figure below is a measured property of that file. None of it transfers to another population without being re-measured there.
>
> **SAFETY BOUNDARY.** A signal is not a fraud finding. This system can establish non-payability, inconsistency or statistical abnormality. It cannot establish intent, and intent is what distinguishes fraud from waste, abuse or honest error. Only a human reviewer, on evidence, may reach a conclusion about conduct.


Generated 2026-09-29T02:27:40.444084+00:00 · parameter-registry fingerprint `9dd1afee3e5cccad` · seed `20260920`.

## In short

- **The file.** `uae_demo.zip`, 24,855 claims. Every number in this report describes this file only.
- **The checks.** Of 164 checks, 77 ran fully, 86 ran in a simplified form because the file lacks some of what they need, and 1 could not run. 157 raised at least one flag.
- **What they found.** 5,894 cases for people to review. The most common suggestions: 3,809 watch only, 1,011 hold before paying, 344 pay, but check afterwards.
- **Money.** AED 38,493,901 is at risk where a clear rule was shown to fail. A further AED 17,600,966 is flagged but not yet established. The two are never added together, and neither is money saved: nothing has been reviewed yet.
- **Pattern-finding models.** 2 trained and scored claims. None is ready to be trusted for real decisions; they stay in watch-only mode.
- **What cannot be measured yet.** How often the flags are right. That needs reviewers to record decisions, and none have been recorded, so no accuracy figure is given in this summary.
- **What a flag means.** A reason to look, not proof of fraud. Every check runs in watch-only mode; only a reviewer, looking at the evidence, can decide.


## What this report establishes, and what it does not

**It establishes** that the design can be built, that the safety boundary can be enforced structurally rather than by convention, and that the governance machinery — effective dating, shadow-before-active, evidence capping, alert ceilings, kill switches, an append-only audit log — runs.

**It does not establish** a performance claim that transfers to any other book of business. Its labels are the answer key of the generator that planted the patterns (`synthetic_injection`). The injectors were written with the checks in mind, so every precision and recall figure below is closer to a self-test than to an estimate of real performance, and an upper bound at best. Nothing here has been reviewed by a human, so precision in the strict sense (confirmed ÷ reviewed) is NOT_MEASURABLE and is reported as such.

Every number below is a measured output of this run, a configuration threshold labelled as such, or `NOT_MEASURABLE_ON_THIS_DATASET`.

## 1. Run summary

| item                                           | value          |
|:-----------------------------------------------|:---------------|
| Claims ingested                                | 24,855         |
| Immutable raw records (SHA-256 before parsing) | 86,864         |
| Adapter                                        | UAE_MULTITABLE |
| Controls in the catalogue                      | 164            |
| Controls executed                              | 163            |
| Controls that produced signals                 | 157            |
| Signals                                        | 6,512          |
| Cases after correlation                        | 5,894          |
| Controls active (not shadow)                   | 0              |


Every control in this build is in **shadow**. None has been activated, because activation requires a policy owner who did not author the rule to approve it after a prospective shadow period, and neither exists in a single retrospective run.


## 2. Catalogue coverage — what `uae_demo.zip` can actually carry

| classification                 |   controls on this file | share   |   catalogue classification (claim-header extract) |
|:-------------------------------|------------------------:|:--------|--------------------------------------------------:|
| EXECUTABLE                     |                      77 | 47.0%   |                                                 9 |
| PARTIAL                        |                      86 | 52.4%   |                                                28 |
| NOT_EXECUTABLE_ON_THIS_DATASET |                       1 | 0.6%    |                                               127 |


All 39 scenarios and 164 atomic controls of the catalogue are registered as configuration and validated against the atomic-control contract. 1 of them cannot run here, each with a stated reason and the canonical fields that would unlock it — see `control_coverage_matrix.csv`, which is the concrete answer to *what would real Shafafiya or eClaimLink data unlock?*

Only **21** controls in the entire catalogue may deny or reprice a claim, and every one of them is type H or H/E. The registry refuses to load a statistical, network, document or model control that declares `REJECT` or `REPRICE`.


### Controls that produced signals

| rule_id    | scenario_id   | type_label   | stage         | data_support   |   signal_count |   elapsed_ms |
|:-----------|:--------------|:-------------|:--------------|:---------------|---------------:|-------------:|
| NET-03-R01 | NET-03        | S/N          | MODEL_MONTHLY | EXECUTABLE     |           1236 |      4259.66 |
| PAY-06-R03 | PAY-06        | S            | POSTPAY_DAILY | EXECUTABLE     |           1155 |      1434.98 |
| PAY-10-R01 | PAY-10        | H            | PREPAY_SYNC   | PARTIAL        |            640 |        49.7  |
| PAY-01-R03 | PAY-01        | H            | POSTPAY_DAILY | PARTIAL        |            434 |     25877.8  |
| POL-01-R04 | POL-01        | S            | MODEL_MONTHLY | PARTIAL        |            392 |       408.67 |
| CLN-05-R03 | CLN-05        | H/E          | POSTPAY_DAILY | PARTIAL        |            278 |       730.79 |
| PAY-01-R02 | PAY-01        | H            | PREPAY_ASYNC  | PARTIAL        |            149 |     12589.8  |
| ANL-01-R03 | ANL-01        | M            | MODEL_MONTHLY | EXECUTABLE     |            136 |      3402.8  |
| ENT-04-R01 | ENT-04        | H            | PREPAY_SYNC   | EXECUTABLE     |            111 |       467.46 |
| PAY-07-R01 | PAY-07        | H            | PREPAY_SYNC   | PARTIAL        |            110 |       416.22 |
| CLN-03-R01 | CLN-03        | H            | PREPAY_SYNC   | PARTIAL        |            103 |        10.18 |
| PAY-03-R03 | PAY-03        | H/E          | PREPAY_SYNC   | PARTIAL        |            103 |       575.1  |
| DOC-02-R01 | DOC-02        | T            | POSTPAY_DAILY | PARTIAL        |             99 |       220.03 |
| CLN-08-R01 | CLN-08        | E            | PREPAY_ASYNC  | PARTIAL        |             98 |        70.54 |
| DOC-01-R02 | DOC-01        | T/E          | PREPAY_ASYNC  | PARTIAL        |             87 |       157.89 |
| ANL-01-R01 | ANL-01        | S            | MODEL_MONTHLY | EXECUTABLE     |             64 |       174.73 |
| ANL-01-R02 | ANL-01        | S            | MODEL_MONTHLY | EXECUTABLE     |             61 |      1746.26 |
| PAY-06-R01 | PAY-06        | H            | PREPAY_SYNC   | EXECUTABLE     |             59 |       307.43 |
| NET-02-R03 | NET-02        | N/S          | POSTPAY_DAILY | PARTIAL        |             51 |      5244.79 |
| PAY-07-R02 | PAY-07        | H            | PREPAY_SYNC   | EXECUTABLE     |             47 |       462.82 |
| PAY-02-R01 | PAY-02        | H            | PREPAY_SYNC   | EXECUTABLE     |             42 |       187.28 |
| PAY-03-R01 | PAY-03        | H            | INGEST        | PARTIAL        |             41 |       641.56 |
| PAY-08-R02 | PAY-08        | H/E          | PREPAY_ASYNC  | EXECUTABLE     |             37 |      1117.89 |
| PAY-09-R04 | PAY-09        | S            | POSTPAY_DAILY | EXECUTABLE     |             36 |      1109.21 |
| CLN-03-R02 | CLN-03        | E            | PREPAY_ASYNC  | EXECUTABLE     |             33 |        63.43 |
| PAY-04-R04 | PAY-04        | H            | PREPAY_SYNC   | EXECUTABLE     |             32 |       565.85 |
| CLN-03-R03 | CLN-03        | E            | POSTPAY_DAILY | PARTIAL        |             31 |        36.7  |
| DOC-01-R01 | DOC-01        | H/E          | PREPAY_ASYNC  | EXECUTABLE     |             31 |       543.63 |
| ENT-03-R01 | ENT-03        | H            | PREPAY_SYNC   | EXECUTABLE     |             31 |       385.43 |
| PHR-03-R02 | PHR-03        | E            | PREPAY_SYNC   | EXECUTABLE     |             25 |       598.27 |
| PAY-06-R04 | PAY-06        | S            | MODEL_MONTHLY | EXECUTABLE     |             24 |       130.7  |
| PHR-02-R01 | PHR-02        | H/E          | PREPAY_SYNC   | EXECUTABLE     |             23 |        98.75 |
| CLN-06-R05 | CLN-06        | S            | MODEL_MONTHLY | PARTIAL        |             21 |       178.3  |
| CLN-01-R02 | CLN-01        | S/M          | POSTPAY_DAILY | PARTIAL        |             21 |        86.75 |
| CLN-06-R04 | CLN-06        | S            | POSTPAY_DAILY | EXECUTABLE     |             20 |        16.47 |
| CLN-04-R01 | CLN-04        | E            | PREPAY_SYNC   | PARTIAL        |             19 |        76.19 |
| POL-01-R01 | POL-01        | H            | POSTPAY_DAILY | EXECUTABLE     |             18 |       305.29 |
| PAY-06-R02 | PAY-06        | H            | INGEST        | PARTIAL        |             18 |         5.62 |
| PAY-07-R04 | PAY-07        | H/S          | POSTPAY_DAILY | EXECUTABLE     |             16 |       129.7  |
| ENT-01-R04 | ENT-01        | H            | PREPAY_SYNC   | EXECUTABLE     |             16 |       458.87 |

_(117 further rows in the accompanying CSV.)_


### Controls that ran and found nothing — and why that is a result

| rule_id    | type_label   | data_support   |
|:-----------|:-------------|:---------------|
| CLN-05-R01 | S            | EXECUTABLE     |
| CLN-07-R02 | E/S          | PARTIAL        |
| ENT-03-R02 | H            | PARTIAL        |
| NET-04-R02 | S/N          | PARTIAL        |
| NET-04-R03 | S            | PARTIAL        |
| PHR-03-R03 | S            | PARTIAL        |


A control that runs and finds nothing has found something. Specifically:

- **CLN-05-R01** ran on this file and raised nothing. Whether nothing in the file matches, or its threshold cannot be crossed here, needs a look at the data it reads; the threshold is not lowered to manufacture signals.
- **CLN-07-R02** ran on this file and raised nothing. Whether nothing in the file matches, or its threshold cannot be crossed here, needs a look at the data it reads; the threshold is not lowered to manufacture signals.
- **ENT-03-R02** ran on this file and raised nothing. Whether nothing in the file matches, or its threshold cannot be crossed here, needs a look at the data it reads; the threshold is not lowered to manufacture signals.
- **NET-04-R02** ran on this file and raised nothing. Whether nothing in the file matches, or its threshold cannot be crossed here, needs a look at the data it reads; the threshold is not lowered to manufacture signals.
- **NET-04-R03** ran on this file and raised nothing. Whether nothing in the file matches, or its threshold cannot be crossed here, needs a look at the data it reads; the threshold is not lowered to manufacture signals.
- **PHR-03-R03** ran on this file and raised nothing. Whether nothing in the file matches, or its threshold cannot be crossed here, needs a look at the data it reads; the threshold is not lowered to manufacture signals.


## 3. Canonical model population

| table                  | status        |   rows |   columns_populated |   columns_defined |
|:-----------------------|:--------------|-------:|--------------------:|------------------:|
| member                 | POPULATED     |   9024 |                  16 |                17 |
| coverage_period        | POPULATED     |  16920 |                  14 |                14 |
| benefit_rule_version   | POPULATED     |    128 |                  12 |                12 |
| provider               | POPULATED     |    244 |                  20 |                20 |
| provider_status_period | POPULATED     |    492 |                   8 |                 8 |
| claim_header           | POPULATED     |  24855 |                  45 |                30 |
| claim_line             | POPULATED     |  62009 |                  22 |                22 |
| diagnosis              | POPULATED     |  30985 |                  11 |                11 |
| encounter              | POPULATED     |  24681 |                  17 |                18 |
| observation            | POPULATED     |  37807 |                  11 |                11 |
| authorization          | POPULATED     |   2215 |                  11 |                11 |
| authorization_line     | POPULATED     |  21285 |                   8 |                 8 |
| claim_version          | POPULATED     |  24855 |                   8 |                 8 |
| remittance             | POPULATED     |  62031 |                  10 |                10 |
| prescription_dispense  | POPULATED     |  13023 |                  21 |                21 |
| policy_event           | POPULATED     |  22082 |                   8 |                 8 |
| review_outcome         | NOT_POPULATED |      0 |                   0 |                15 |


No table is filled with invented data. A table that is empty because the source has no such data is a finding; a table full of fabricated rows would be a lie that propagates into every downstream metric. Reasons are in `canonical_population_report.csv`.


## 4. Per-layer results

### Deterministic and expert controls

| control type   |   signals |
|:---------------|----------:|
| E              |       271 |
| E/S            |        34 |
| E/T            |        29 |
| H              |      2023 |
| H/E            |       550 |
| H/N            |         9 |
| H/S            |        47 |
| H/T            |         6 |
| M              |       138 |
| N              |        11 |
| N/S            |        51 |
| S              |      1859 |
| S/M            |        21 |
| S/N            |      1251 |
| T              |       105 |
| T/E            |       106 |
| T/S            |         1 |


### Statistical layer — ANL-01-R01 transparent composite


240 providers scored; 64 above the `cfg.composite_flag_threshold` of 1.2. Score distribution: median 0.43, p95 24.74, max 43.50. The threshold's position in that distribution is visible rather than asserted.

| provider_sk   |   composite_score | peer_level_used   |   peer_n |   claim_count | top_features                                                                                                                                                           |
|:--------------|------------------:|:------------------|---------:|--------------:|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| PRV0153       |           43.4987 | GLOBAL            |      240 |            53 | Pharmacy share of the bill=100.0% vs peer 0.0%; Approved-to-billed ratio=73.8% vs peer 84.1%; Mean claim amount=AED 76 vs peer AED 272                                 |
| PRV0143       |           43.4827 | GLOBAL            |      240 |            80 | Pharmacy share of the bill=100.0% vs peer 0.0%; Approved-to-billed ratio=78.4% vs peer 84.1%; Claims per distinct member=2.16 claims/member vs peer 1.67 claims/member |
| PRV0144       |           37.958  | L1                |       30 |            68 | Pharmacy share of the bill=100.0% vs peer 0.0%; Approved-to-billed ratio=73.2% vs peer 86.2%; Mean claim amount=AED 60 vs peer AED 218                                 |
| PRV0167       |           37.8691 | L1                |       30 |            41 | Pharmacy share of the bill=100.0% vs peer 0.0%; Approved-to-billed ratio=76.9% vs peer 86.2%; Mean claim amount=AED 105 vs peer AED 218                                |
| PRV0149       |           37.8483 | L1                |       30 |            51 | Pharmacy share of the bill=100.0% vs peer 0.0%; Approved-to-billed ratio=77.9% vs peer 86.2%; Claims per distinct member=1.65 claims/member vs peer 1.63 claims/member |
| PRV0171       |           37.6895 | L1                |       30 |            14 | Pharmacy share of the bill=100.0% vs peer 0.0%; Approved-to-billed ratio=87.1% vs peer 86.2%; Claims per distinct member=1.75 claims/member vs peer 1.63 claims/member |
| PRV0161       |           36.379  | L1                |       30 |            45 | Pharmacy share of the bill=96.0% vs peer 0.0%; Approved-to-billed ratio=77.9% vs peer 86.2%; Claims per distinct member=1.80 claims/member vs peer 1.63 claims/member  |
| PRV0155       |           36.3499 | L1                |       30 |            94 | Pharmacy share of the bill=95.6% vs peer 0.0%; Approved-to-billed ratio=71.6% vs peer 86.2%; Mean claim amount=AED 101 vs peer AED 218                                 |
| PRV0025       |           32.0175 | L2                |       51 |           322 | Mean claim amount=AED 5,574 vs peer AED 270; Approved-to-billed ratio=85.7% vs peer 83.7%; Pharmacy share of the bill=3.3% vs peer 0.0%                                |
| PRV0141       |           24.8959 | L1                |       44 |            56 | Pharmacy share of the bill=100.0% vs peer 0.4%; Approved-to-billed ratio=71.3% vs peer 79.7%; Mean claim amount=AED 75 vs peer AED 162                                 |
| PRV0175       |           24.761  | L1                |       44 |           161 | Pharmacy share of the bill=99.5% vs peer 0.4%; Approved-to-billed ratio=71.6% vs peer 79.7%; Mean claim amount=AED 79 vs peer AED 162                                  |
| PRV0154       |           24.7534 | L1                |       44 |            33 | Pharmacy share of the bill=100.0% vs peer 0.4%; Approved-to-billed ratio=78.2% vs peer 79.7%; Mean claim amount=AED 68 vs peer AED 162                                 |

_(52 further rows in the accompanying CSV.)_


### Graph layer


80 weekly, time-bounded snapshots; 5,699 nodes and 23,395 edges in the cumulative graph; 1,071 snapshot communities. 1 entity-resolution candidate surfaced for human confirmation — **none merged automatically**, at any confidence.

| edge_type                        | present   |   edges | observed   | note                                                                                                                                                                                   |
|:---------------------------------|:----------|--------:|:-----------|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| member_provider                  | True      |   13386 | True       | One edge per claim. Directly observed.                                                                                                                                                 |
| agent_member                     | True      |    5409 | True       | Policy origination. Directly observed.                                                                                                                                                 |
| agent_provider                   | True      |    3916 | False      | INFERRED from co-occurrence on the same claim, not an observed referral. Every signal resting on this edge says so.                                                                    |
| provider_tpa                     | True      |     684 | True       | Claim administration. Directly observed.                                                                                                                                               |
| referral (directed A→B)          | False     |       0 | False      | NOT BUILT INTO THE GRAPH — the file has 4,989 referral records, but this graph does not yet build the edge. The controls that need it read those records directly.                     |
| prescriber—pharmacy              | False     |       0 | False      | NOT BUILT INTO THE GRAPH — the file has 13,023 prescription and dispensing records, but this graph does not yet build the edge. The controls that need it read those records directly. |
| shared administrative identifier | False     |       0 | False      | NOT BUILT INTO THE GRAPH — the file has bank or phone tokens for 244 providers, but this graph does not yet build the edge. The controls that need it read those records directly.     |
| ownership                        | False     |       0 | False      | NOT BUILT INTO THE GRAPH — the file has ownership details for 244 providers, but this graph does not yet build the edge. The controls that need it read those records directly.        |


### Model layer

| model                |   training_rows |   scored_rows |   capacity_threshold |   flagged |   features |
|:---------------------|----------------:|--------------:|---------------------:|----------:|-----------:|
| isolation_forest     |            7925 |          4561 |               0.5998 |       100 |        104 |
| local_outlier_factor |            7925 |          4561 |               4.1247 |       100 |        104 |


12,369 of 24,855 claims (50%) are scored by NEITHER model: they belong to a provider held out for training, or fall in the training period. This is the price of combining a temporal split with entity isolation on a single-file dataset, and it is reported rather than absorbed — a model that scored everything would have been trained on the entities it scores.


**Promotion gate verdicts:**


- **isolation_forest: REMAINS IN SHADOW** — isolation_forest REMAINS IN SHADOW. Failed: Prospective lift over ANL-01-R01; Calibration check; Drift monitoring. The system loses nothing by this: ANL-01-R01, the transparent composite, is the explainable fallback and continues to run.


| criterion                        | status   |    value |   threshold | evidence                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  |
|:---------------------------------|:---------|---------:|------------:|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Temporal holdout evaluation      | PASS     | nan      |      nan    | Trained on 7,925 claims before 2025-05-15; scored 4,561 claims after it. Entity isolation holds: 0 providers appear in both sets. 12,369 of 24,855 claims (50%) are scored by NEITHER model: they belong to a provider held out for training, or fall in the training period. This is the price of combining a temporal split with entity isolation on a single-file dataset, and it is reported rather than absorbed — a model that scored everything would have been trained on the entities it scores. |
| Prospective lift over ANL-01-R01 | FAIL     |   1.0361 |        1.1  | At a reviewer capacity of 100, the model's review yield is 86.0% against the transparent composite's 83.0% — a lift of 1.04× against a required 1.10×. THE TRANSPARENT COMPOSITE WAS NOT BEATEN. This is a legitimate finding and the model stays in shadow; it is not a reason to retune until the model wins. NOTE: the outcomes used are investigation-derived, selection-biased labels , so this lift is an upper bound, not an estimate of production behaviour.                                     |
| Calibration check                | FAIL     |   0.48   |        0.15 | Expected calibration error 0.480 against a maximum of 0.150. An unsupervised anomaly score is NOT a probability; it is min-max rescaled here purely so a calibration curve can be drawn, and a poor ECE on an unsupervised score is expected rather than surprising.                                                                                                                                                                                                                                      |
| Drift monitoring                 | FAIL     | nan      |      nan    | Drift breach on: Population stability (max feature PSI)                                                                                                                                                                                                                                                                                                                                                                                                                                                   |
| Explanation quality              | PASS     | nan      |      nan    | 15 of 15 sampled contributions render in ORIGINAL UNITS beside a peer value, as the promotion gate requires.                                                                                                                                                                                                                                                                                                                                                                                              |


- **local_outlier_factor: REMAINS IN SHADOW** — local_outlier_factor REMAINS IN SHADOW. Failed: Prospective lift over ANL-01-R01; Calibration check; Drift monitoring. The system loses nothing by this: ANL-01-R01, the transparent composite, is the explainable fallback and continues to run.


| criterion                        | status   |    value |   threshold | evidence                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  |
|:---------------------------------|:---------|---------:|------------:|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Temporal holdout evaluation      | PASS     | nan      |      nan    | Trained on 7,925 claims before 2025-05-15; scored 4,561 claims after it. Entity isolation holds: 0 providers appear in both sets. 12,369 of 24,855 claims (50%) are scored by NEITHER model: they belong to a provider held out for training, or fall in the training period. This is the price of combining a temporal split with entity isolation on a single-file dataset, and it is reported rather than absorbed — a model that scored everything would have been trained on the entities it scores. |
| Prospective lift over ANL-01-R01 | FAIL     |   1.0361 |        1.1  | At a reviewer capacity of 100, the model's review yield is 86.0% against the transparent composite's 83.0% — a lift of 1.04× against a required 1.10×. THE TRANSPARENT COMPOSITE WAS NOT BEATEN. This is a legitimate finding and the model stays in shadow; it is not a reason to retune until the model wins. NOTE: the outcomes used are investigation-derived, selection-biased labels , so this lift is an upper bound, not an estimate of production behaviour.                                     |
| Calibration check                | FAIL     |   0.7331 |        0.15 | Expected calibration error 0.733 against a maximum of 0.150. An unsupervised anomaly score is NOT a probability; it is min-max rescaled here purely so a calibration curve can be drawn, and a poor ECE on an unsupervised score is expected rather than surprising.                                                                                                                                                                                                                                      |
| Drift monitoring                 | FAIL     | nan      |      nan    | Drift breach on: Population stability (max feature PSI)                                                                                                                                                                                                                                                                                                                                                                                                                                                   |
| Explanation quality              | PASS     | nan      |      nan    | 15 of 15 sampled contributions render in ORIGINAL UNITS beside a peer value, as the promotion gate requires.                                                                                                                                                                                                                                                                                                                                                                                              |


**Supervised gate: `PROMOTION_BLOCKED`.** PROMOTION_BLOCKED. 3 of 5 prerequisites are unmet: Stable outcome capture (review_outcome); Random-audit data to correct investigation-selection bias; Calibration and specialty/payer stability. No supervised model is trained, and none should be: a propensity model fitted to these labels would learn to reproduce the blind spots of whatever process produced them. The promotion gate treats that failure mode as disqualifying, not merely undesirable.


| prerequisite                                              | met   | what_would_satisfy_it                                                                                                                     |
|:----------------------------------------------------------|:------|:------------------------------------------------------------------------------------------------------------------------------------------|
| Stable outcome capture (review_outcome)                   | False | Several hundred reviewer dispositions with validated categories and confirmed amounts, captured consistently over at least two quarters.  |
| Random-audit data to correct investigation-selection bias | False | A genuine random audit: a statistically representative sample of claims reviewed by humans REGARDLESS of whether the system flagged them. |
| Temporal train/validation/test splits                     | True  | A date-ordered split with no future data in training.                                                                                     |
| Entity isolation across splits                            | True  | No provider, member or episode present in both a training and an evaluation split.                                                        |
| Calibration and specialty/payer stability                 | False | Confirmed outcomes for a calibration curve, plus a real provider specialty and payer dimension to check stability across.                 |


**ANL-01-R04 novel-cluster candidates (MONITOR_ONLY):**


|   cluster_id |   size | candidate_typology_name                                                             |   coherence |   aggregate_exposure_aed | top_features                                                                               | members                                                                 |
|-------------:|-------:|:------------------------------------------------------------------------------------|------------:|-------------------------:|:-------------------------------------------------------------------------------------------|:------------------------------------------------------------------------|
|            0 |     11 | Candidate pattern: elevated claim amount with elevated days from discharge to claim |       0.214 |               9.6217e+06 | Claim amount=+43.16σ; Days from discharge to claim=+1.40σ; Approved-to-billed ratio=+0.55σ | PRV0001, PRV0005, PRV0008, PRV0011, PRV0012, PRV0014, PRV0015, PRV0017… |
|            1 |    100 | Candidate pattern: elevated claim amount with elevated early policy tenure          |       0.315 |               4.7925e+06 | Claim amount=+1.04σ; Early policy tenure=+0.26σ; Approved-to-billed ratio=-0.16σ           | PRV0021, PRV0022, PRV0023, PRV0028, PRV0031, PRV0034, PRV0036, PRV0037… |


### Document / NLP layer


The source file carries **13,095 documents**, which the document controls read directly. Separately, the extraction pipeline runs against 800 clearly-labelled SYNTHETIC discharge summaries generated by this artefact ({'en': 554, 'ar': 246}), of which 95 carry a deliberately injected inconsistency and 141 carry poor OCR quality so that confidence degradation is observable. 8 findings were DISCARDED for want of a source span or sufficient confidence — not displayed with a caveat.


**Extraction quality, reported separately by language and document type, as the document design requires:**


| language   | document_type     | field          |   extractions |   mean_confidence |   mean_ocr |   below_threshold |
|:-----------|:------------------|:---------------|--------------:|------------------:|-----------:|------------------:|
| ar         | discharge_summary | admission_date |           246 |             0.829 |      0.873 |                28 |
| ar         | discharge_summary | diagnosis_text |           246 |             0.829 |      0.873 |                28 |
| ar         | discharge_summary | discharge_date |           246 |             0.829 |      0.873 |                28 |
| ar         | discharge_summary | length_of_stay |           246 |             0.829 |      0.873 |                28 |
| ar         | discharge_summary | pharmacy_share |           246 |             0.829 |      0.873 |                28 |
| en         | discharge_summary | admission_date |           554 |             0.837 |      0.881 |                59 |
| en         | discharge_summary | diagnosis_text |           554 |             0.837 |      0.881 |                59 |
| en         | discharge_summary | discharge_date |           554 |             0.837 |      0.881 |                59 |
| en         | discharge_summary | length_of_stay |           554 |             0.837 |      0.881 |                59 |
| en         | discharge_summary | pharmacy_share |           554 |             0.837 |      0.881 |                59 |


No figure in that table is a measurement of clinical-NLP performance. The corpus is template-generated, so extraction accuracy on it measures whether the pipeline's plumbing works — nothing more. Arabic is included because the design requires language-separated reporting, and a pipeline evaluated on one language cannot make that claim.


### AI layer


|   grounded_narrative |   reviewer_copilot |   rule_authoring |   typology_triage |
|---------------------:|-------------------:|-----------------:|------------------:|
|                    1 |                  1 |                1 |                 1 |


Provider: `offline_deterministic` (`template-composer-1.0.0`), deterministic: True. No AI output sets or changes a disposition, a priority or an exposure, under any configuration. Every narrative is passed through the groundedness validator, which drops any uncited or unresolvable sentence before display, and the reviewer copilot refuses conduct questions **before any model call**.


## 5. Operational metrics

| metric                                          | value                          | unit               | status                         | note                                                                                                                                                                                                                                                                                                     |
|:------------------------------------------------|:-------------------------------|:-------------------|:-------------------------------|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Precision proxy — CLAIM-level controls          | 0.2217                         | ratio              | MEASURED                       | 8,636 claims flagged (34.7% of the file) against a base rate of 10.6% — a lift of 2.10×. This is a PROXY for precision, not precision: real precision is confirmed ÷ REVIEWED and nothing has been reviewed. The labels are investigation-derived and selection-biased, so this is an UPPER BOUND.       |
| Precision proxy — ENTITY leads (context claims) | 0.1264                         | ratio              | MEASURED                       | 19,123 claims sit under an entity lead. An entity lead does NOT assert that each of those claims is suspect — it asserts that the entity's pattern warrants review — so this figure is close to the base rate of 10.6% by construction and is reported to make that visible, not as a performance claim. |
| Claim coverage — any control                    | 0.8076                         | ratio              | MEASURED                       | Reported so that the difference between the two figures above is inspectable rather than implicit.                                                                                                                                                                                                       |
| Gross flagged value (established exposure)      | 38493900.8                     | AED                | MEASURED                       | GROSS FLAGGED VALUE IS NOT SAVINGS. Nothing here has been reviewed, confirmed, prevented or recovered.                                                                                                                                                                                                   |
| Gross flagged value (exposure NOT established)  | 17600966.2                     | AED                | MEASURED                       | Reported SEPARATELY and never added to the figure above: a model-only lead shows its gross amount with 'exposure not yet established'.                                                                                                                                                                   |
| Confirmed AED                                   | NOT_MEASURABLE_ON_THIS_DATASET | AED                | NOT_MEASURABLE_ON_THIS_DATASET |                                                                                                                                                                                                                                                                                                          |
| Prevented / recovered AED                       | NOT_MEASURABLE_ON_THIS_DATASET | AED                | NOT_MEASURABLE_ON_THIS_DATASET |                                                                                                                                                                                                                                                                                                          |
| Net savings after review cost                   | NOT_MEASURABLE_ON_THIS_DATASET | AED                | NOT_MEASURABLE_ON_THIS_DATASET | The COST side is computable: 5,894 cases × AED 120.00 = AED 707,280.00 of review effort. The SAVINGS side is not, because nothing has been confirmed. Reporting the cost alone would imply a negative return that is equally unevidenced.                                                                |
| Provider/member abrasion                        | NOT_MEASURABLE_ON_THIS_DATASET | rate               | NOT_MEASURABLE_ON_THIS_DATASET |                                                                                                                                                                                                                                                                                                          |
| Pend turnaround                                 | NOT_MEASURABLE_ON_THIS_DATASET | hours              | NOT_MEASURABLE_ON_THIS_DATASET |                                                                                                                                                                                                                                                                                                          |
| Overturn / appeal rate                          | NOT_MEASURABLE_ON_THIS_DATASET | ratio              | NOT_MEASURABLE_ON_THIS_DATASET |                                                                                                                                                                                                                                                                                                          |
| Rule stability                                  | NOT_MEASURABLE_ON_THIS_DATASET | revisions/rule     | NOT_MEASURABLE_ON_THIS_DATASET | Every control in this build is in SHADOW; none has been activated, so none has had a post-activation revision to count.                                                                                                                                                                                  |
| Alerts per reviewer                             | 1473.5                         | cases/reviewer/run | MEASURED                       | 5,894 cases across 4 reviewers. Daily capacity is 100 alerts, so this run's queue represents 58.9 days of review work.                                                                                                                                                                                   |
| Time to case disposition                        | NOT_MEASURABLE_ON_THIS_DATASET | hours              | NOT_MEASURABLE_ON_THIS_DATASET |                                                                                                                                                                                                                                                                                                          |


## 6. Capacity-aware precision, stratified by label source

|   capacity_cases |   reviewer_days |   cases_reviewed |   precision_proxy |   labelled_cases |   exposure_reviewed_aed |
|-----------------:|----------------:|-----------------:|------------------:|-----------------:|------------------------:|
|               25 |            0.25 |               25 |            1      |               25 |             1.96971e+07 |
|              100 |            1    |              100 |            0.63   |               63 |             2.39381e+07 |
|              500 |            5    |              500 |            0.392  |              196 |             3.01801e+07 |
|             2000 |           20    |             2000 |            0.3445 |              689 |             3.74964e+07 |


Precision is evaluated **at the alert volume a review team can actually process**, not at an arbitrary threshold. The capacity basis is `cfg.alerts_per_reviewer_per_day` × `cfg.reviewer_count` = 100 cases per day.


### Stratified by `ground_truth_source`

| ground_truth_source   |   claims |   labelled_fraud |   prevalence |   flagged |   flagged_and_labelled |   precision_proxy |   recall_proxy | reliability_caveat                                                                                                                                                                                            |
|:----------------------|---------:|-----------------:|-------------:|----------:|-----------------------:|------------------:|---------------:|:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| synthetic_injection   |    24855 |             2627 |       0.1057 |      8636 |                   1915 |            0.2217 |          0.729 | synthetic_injection labels are the answer key of the generator that planted the patterns; the injectors were written with the checks in mind, so precision and recall against them are closer to a self-test. |


Its labels are the answer key of the generator that planted the patterns (`synthetic_injection`). The injectors were written with the checks in mind, so every precision and recall figure below is closer to a self-test than to an estimate of real performance, and an upper bound at best.


### Recall by `fraud_type`, and which control found it

| fraud_type                                                                                            |   labelled_claims |   flagged_by_claim_control |   recall_claim_level |   also_under_an_entity_lead | top_claim_level_controls                                            |
|:------------------------------------------------------------------------------------------------------|------------------:|---------------------------:|---------------------:|----------------------------:|:--------------------------------------------------------------------|
| CLN level migration to top-level visits from April 2025                                               |               279 |                        161 |               0.5771 |                         279 | NET-03-R01 (112); PAY-06-R03 (38); PAY-01-R03 (13); POL-01-R04 (10) |
| PAY modifier 25 appended to most visits                                                               |               278 |                        158 |               0.5683 |                         278 | NET-03-R01 (133); PAY-06-R03 (17); POL-01-R04 (9); PAY-10-R01 (9)   |
| NET-02 quiet clinic becomes one of the busiest right after an ownership change                        |               112 |                          0 |               0      |                         112 | — none —                                                            |
| CLN hospital records (and documents) complication codes on most stays                                 |                86 |                         53 |               0.6163 |                          86 | PAY-06-R03 (43); NET-03-R01 (16); PAY-10-R01 (4); POL-01-R04 (3)    |
| CLN clinician billing more than seventeen hours of visits in one day                                  |                78 |                         52 |               0.6667 |                          78 | PAY-06-R03 (52)                                                     |
| CLN hospital admitting overnight presentations others treat as outpatients                            |                70 |                         70 |               1      |                          70 | CLN-03-R01 (70); PAY-03-R03 (45); NET-03-R01 (1); CLN-05-R02 (1)    |
| ENT-06 telehealth clinic whose calls turn into MRI referrals far more often than peers'               |                57 |                         23 |               0.4035 |                          46 | PAY-06-R03 (23); POL-01-R04 (1)                                     |
| CLN hospital severity index jumps from May 2025 with unchanged stays                                  |                56 |                         54 |               0.9643 |                          56 | PAY-06-R03 (53); NET-03-R01 (21); PAY-01-R03 (7); CLN-01-R02 (7)    |
| CLN a general practice billing electroencephalograms                                                  |                49 |                         49 |               1      |                          49 | ENT-04-R01 (49); ENT-02-R03 (1)                                     |
| PAY clinic systematically records zero or rounded-down patient share                                  |                49 |                         49 |               1      |                          43 | PAY-07-R01 (49); POL-01-R04 (2)                                     |
| ENT-03 newly credentialed clinic billing far above peers at once                                      |                48 |                         41 |               0.8542 |                          48 | PAY-01-R03 (41); PAY-01-R02 (6); CLN-05-R03 (5); NET-03-R01 (5)     |
| POL-01 employer-sponsored member not on the employer's roster                                         |                45 |                         45 |               1      |                          37 | POL-01-R01 (45); NET-03-R01 (17); PAY-01-R03 (10); PAY-10-R01 (10)  |
| NET-03 one small employer's staff all treated by the same two clinics                                 |                42 |                          7 |               0.1667 |                          42 | PAY-01-R03 (6); CLN-05-R03 (6); POL-01-R04 (1)                      |
| ENT-06 telehealth clinic sending nearly every prescription to one pharmacy                            |                36 |                         18 |               0.5    |                          36 | PHR-03-R02 (18)                                                     |
| CLN clinic ordering genetic tests for a large share of its patients                                   |                34 |                         34 |               1      |                          34 | PAY-06-R03 (34)                                                     |
| CLN minor clinic visits reliably followed by a large specialist, CT and echo bundle                   |                32 |                         16 |               0.5    |                          32 | PAY-06-R03 (16); ENT-04-R01 (3); CLN-04-R01 (1)                     |
| NET-03 clinic waiving patient share on repeated knee injections for a small patient group             |                32 |                         32 |               1      |                          32 | PAY-07-R01 (32); PAY-07-R02 (32); NET-03-R01 (32); CLN-06-R05 (32)  |
| PAY clinic bills a payable substitute code after refused services                                     |                32 |                         21 |               0.6562 |                          32 | CLN-03-R01 (16); CLN-03-R02 (16); ENT-04-R01 (16); CLN-05-R03 (4)   |
| PHR-04 prescriber concentrated on one pharmacy                                                        |                30 |                         20 |               0.6667 |                          30 | PAY-01-R03 (20); CLN-05-R03 (20)                                    |
| CLN ten consultations for a cold within one month                                                     |                30 |                         30 |               1      |                          30 | CLN-04-R02 (30); CLN-05-R03 (30); PAY-01-R02 (30); NET-03-R01 (30)  |
| PHR-02 therapy continued beyond the policy duration on one prescription                               |                30 |                         30 |               1      |                          30 | PHR-02-R02 (30); PAY-01-R03 (30); CLN-05-R03 (23); PAY-01-R02 (10)  |
| PHR-04 clinic-to-pharmacy flow with shared bank account and phone                                     |                27 |                         18 |               0.6667 |                          27 | PAY-01-R03 (18); CLN-05-R03 (8)                                     |
| DOC-02 one-line discharge summaries for high-complexity admissions                                    |                27 |                         21 |               0.7778 |                          27 | NET-03-R01 (20); PAY-06-R03 (6); CLN-05-R03 (5); PAY-10-R01 (1)     |
| CLN MRI scans billed beyond one day's scanner capacity (18)                                           |                27 |                          1 |               0.037  |                          27 | CLN-04-R01 (1)                                                      |
| NET-01 radiology centre turning every referral into a contrast CT                                     |                24 |                         24 |               1      |                          24 | PAY-06-R03 (24); POL-01-R04 (1)                                     |
| PHR-04 new prescriber-pharmacy link rapidly dominant                                                  |                24 |                         16 |               0.6667 |                          24 | PAY-01-R03 (16); CLN-05-R03 (16)                                    |
| CLN diabetes billed repeatedly with no medicine or result ever recorded                               |                24 |                         24 |               1      |                          24 | CLN-06-R05 (24)                                                     |
| ENT-02 one Emirates-ID reference on two records with different dates of birth                         |                23 |                         23 |               1      |                          16 | ENT-02-R01 (23); PAY-10-R01 (10); NET-03-R01 (4); POL-01-R04 (1)    |
| PAY clinic's changed resubmissions after one refusal reason always paid                               |                22 |                         22 |               1      |                          22 | PAY-08-R02 (22); NET-03-R01 (20); CLN-05-R03 (4); ENT-02-R03 (2)    |
| NET-01 two clinics referring patients back and forth                                                  |                22 |                          1 |               0.0455 |                          22 | POL-01-R04 (1)                                                      |
| PAY novel unbundling: inflammation markers always billed as a separate pair                           |                22 |                         22 |               1      |                          22 | ENT-04-R01 (22); CLN-08-R01 (22)                                    |
| ENT-05 hospital recording ordinary visits as day cases from August 2025                               |                21 |                         17 |               0.8095 |                          21 | NET-03-R01 (17); PAY-01-R03 (7); PAY-01-R02 (4); CLN-05-R03 (3)     |
| PAY refused claim resubmitted with diagnosis and code swapped                                         |                20 |                         20 |               1      |                          18 | PAY-08-R02 (20); NET-03-R01 (12); PAY-01-R03 (4); CLN-05-R03 (2)    |
| PHR-02 early refills (stockpiling)                                                                    |                20 |                         20 |               1      |                          20 | PHR-02-R01 (20); PAY-01-R03 (20); CLN-05-R03 (16)                   |
| ENT-02 one member's cover used in two distant emirates on the same days                               |                20 |                         20 |               1      |                          17 | ENT-02-R04 (20); NET-03-R01 (1)                                     |
| PHR-03 wastage billed far above vial arithmetic                                                       |                20 |                         20 |               1      |                          20 | PAY-04-R04 (20); NET-03-R01 (20); PAY-01-R03 (2); DOC-01-R02 (1)    |
| PAY service resubmitted four times with a different change each time                                  |                20 |                         20 |               1      |                          20 | PAY-08-R03 (20); NET-03-R01 (20); PAY-08-R02 (20); PAY-01-R03 (9)   |
| PHR-04 high-cost prescriptions steered to one pharmacy                                                |                20 |                         13 |               0.65   |                          20 | PAY-06-R03 (10); NET-03-R01 (6); PAY-01-R03 (2)                     |
| POL-01 application declared no prior cover and no conditions although earlier treatment is on record  |                19 |                         15 |               0.7895 |                          10 | POL-01-R03 (15); POL-01-R04 (3)                                     |
| POL-01 new employer group claiming within weeks at two clinics sharing a bank account                 |                18 |                          6 |               0.3333 |                          18 | POL-01-R04 (6)                                                      |
| ENT-03 clinic billing after its facility licence expired                                              |                16 |                         16 |               1      |                          11 | ENT-03-R01 (16)                                                     |
| CLN HbA1c repeated twelve days later, before the first result was recorded                            |                16 |                         16 |               1      |                           4 | CLN-04-R03 (16); CLN-04-R01 (16); PAY-01-R03 (12); CLN-05-R03 (10)  |
| ENT-01 clinic billing after leaving the payer network                                                 |                16 |                         16 |               1      |                          12 | ENT-01-R04 (16); PAY-10-R01 (4); CLN-06-R05 (1); POL-01-R04 (1)     |
| PAY one adjudicator pays one provider's pended claims in full, uniformly                              |                15 |                          0 |               0      |                          15 | — none —                                                            |
| PHR-02 same controlled medicine from several unrelated prescribers                                    |                14 |                         14 |               1      |                          14 | PAY-03-R01 (14); PHR-02-R03 (14); PAY-01-R03 (14); CLN-05-R03 (14)  |
| NET-01 clinic keeps its lab and imaging referrals inside providers sharing its bank account and phone |                14 |                         10 |               0.7143 |                          14 | CLN-08-R01 (10)                                                     |
| CLN complication code lifts the case rate but is not in the discharge summary                         |                13 |                         13 |               1      |                          12 | CLN-02-R01 (10); PAY-06-R03 (8); CLN-01-R02 (2)                     |
| CLN clinic re-billing a reference laboratory's tests at three times its price                         |                13 |                         13 |               1      |                          13 | PAY-06-R03 (13); PAY-06-R01 (13)                                    |
| ENT-03 clinician billing after their licence expired                                                  |                13 |                         13 |               1      |                          13 | ENT-03-R01 (13); NET-03-R01 (4); PAY-06-R03 (2); CLN-05-R03 (1)     |
| POL-01 one person enrolled twice with overlapping cover                                               |                13 |                          4 |               0.3077 |                           3 | POL-01-R01 (3); PAY-07-R01 (1)                                      |
| CLN ECG billed for an upper respiratory infection with no cardiac reason                              |                12 |                         12 |               1      |                           9 | CLN-03-R02 (12); CLN-03-R01 (12)                                    |
| PHR-02 same medicine collected from several pharmacies                                                |                12 |                         12 |               1      |                           8 | PHR-02-R02 (12); PHR-02-R01 (12); PHR-02-R04 (12)                   |
| PAY patient share below the benefit co-payment                                                        |                11 |                         11 |               1      |                           9 | PAY-07-R01 (11); NET-03-R01 (5)                                     |
| ENT-05 outpatient visit billed at the same hospital during the stay                                   |                10 |                         10 |               1      |                          10 | NET-03-R01 (10); ENT-05-R03 (10); PAY-06-R03 (1)                    |
| ENT-02 services after the member's recorded death                                                     |                10 |                         10 |               1      |                           9 | ENT-02-R02 (10); NET-03-R01 (5); PAY-01-R03 (2); PAY-06-R03 (1)     |
| PAY one adjudicator overrides one hospital's claims far more than colleagues                          |                10 |                          5 |               0.5    |                          10 | NET-03-R01 (5); PAY-06-R03 (3); CLN-02-R03 (1); CLN-05-R03 (1)      |
| CLN member says the visit did not happen (verified app response)                                      |                10 |                         10 |               1      |                           8 | PAY-09-R04 (10); CLN-06-R04 (10); NET-03-R01 (2)                    |
| CLN imaging paid with no report or result on file                                                     |                10 |                         10 |               1      |                           7 | CLN-06-R01 (10); DOC-01-R01 (10); NET-03-R01 (1)                    |
| NET-04 clinic owned by the agent who sold its patients' policies                                      |                10 |                          1 |               0.1    |                          10 | DOC-01-R02 (1)                                                      |
| PAY post-edit migration: modifier 59 after a new edit                                                 |                 9 |                          0 |               0      |                           9 | — none —                                                            |
| PAY member receipt exceeds the approved patient share                                                 |                 9 |                          9 |               1      |                           9 | PAY-07-R04 (9); NET-03-R01 (3)                                      |
| CLN long-standing condition on one claim only, absent from a long history                             |                 9 |                          9 |               1      |                           9 | CLN-02-R03 (8); NET-03-R01 (6); PAY-01-R03 (1)                      |
| PAY road-traffic or work injury with no liable-party record                                           |                 9 |                          9 |               1      |                           9 | PAY-10-R03 (9); NET-03-R01 (4); PAY-01-R03 (3); POL-01-R04 (1)      |
| PAY provider offsets credits against unrelated claims and bills negative lines                        |                 9 |                          9 |               1      |                           9 | PAY-06-R02 (5); NET-03-R01 (5)                                      |
| CLN knee MRI with no knee X-ray first                                                                 |                 9 |                          9 |               1      |                           5 | CLN-03-R03 (9); PAY-06-R03 (6)                                      |
| PAY payment check overridden without a reason or by an unauthorised role                              |                 9 |                          9 |               1      |                           8 | PAY-11-R01 (9); NET-03-R01 (3)                                      |
| CLN all basic metabolic panel components billed one by one                                            |                 8 |                          8 |               1      |                           5 | CLN-08-R02 (8)                                                      |
| PAY clinical note describes a cosmetic service billed under a covered code                            |                 8 |                          8 |               1      |                           7 | PAY-09-R01 (8); NET-03-R01 (5); PAY-01-R03 (2); DOC-01-R02 (1)      |
| CLN main diagnosis marked not present on admission                                                    |                 8 |                          8 |               1      |                           8 | CLN-02-R04 (8); PAY-06-R03 (5); NET-03-R01 (4); CLN-05-R03 (1)      |
| CLN laboratory tests with no ordering clinician, or one on no roster                                  |                 8 |                          8 |               1      |                           1 | CLN-08-R01 (8); ENT-04-R01 (4); POL-01-R01 (1); PAY-10-R01 (1)      |
| CLN laboratory tests paid with no result recorded                                                     |                 8 |                          8 |               1      |                           1 | DOC-01-R01 (8); CLN-08-R03 (7)                                      |
| ENT-01 service after cover ended                                                                      |                 8 |                          8 |               1      |                           7 | ENT-01-R01 (8); POL-01-R01 (7); NET-03-R01 (4)                      |
| DOC-01 required document never sent (operative note, discharge summary or imaging report)             |                 8 |                          8 |               1      |                           6 | DOC-01-R01 (8); PAY-06-R03 (2); PAY-10-R01 (1)                      |
| PAY member says the billed service did not take place                                                 |                 8 |                          8 |               1      |                           7 | PAY-09-R04 (8); CLN-06-R04 (8); NET-03-R01 (2)                      |
| DOC-01 high-cost service without the required indication in the record                                |                 8 |                          8 |               1      |                           7 | DOC-01-R03 (8); NET-03-R01 (5); PAY-06-R03 (2); PAY-10-R01 (1)      |
| DOC-02 discharge summary carries another patient's age or sex                                         |                 8 |                          8 |               1      |                           8 | DOC-02-R02 (8); NET-03-R01 (2); PAY-06-R03 (2)                      |
| PAY same line paid twice                                                                              |                 7 |                          7 |               1      |                           7 | PAY-12-R02 (7); PAY-06-R02 (6); NET-03-R01 (2); PAY-01-R03 (1)      |
| PHR-05 supplies beyond usual consumption or useful life                                               |                 7 |                          7 |               1      |                           7 | PHR-05-R03 (5); PAY-03-R01 (4); PAY-06-R03 (3)                      |
| PAY patient share left inside the amount claimed from the insurer                                     |                 7 |                          7 |               1      |                           7 | PAY-07-R02 (7); NET-03-R01 (4); POL-01-R04 (1); PAY-01-R03 (1)      |
| PAY resubmission with a missing, unknown or unrelated original                                        |                 7 |                          7 |               1      |                           6 | PAY-08-R01 (7); NET-03-R01 (2)                                      |
| CLN identical three-service encounter at 10:00 for different patients                                 |                 6 |                          6 |               1      |                           6 | ENT-05-R01 (6); PAY-06-R01 (1)                                      |
| CLN identical multi-value results reported for unrelated patients                                     |                 6 |                          0 |               0      |                           6 | — none —                                                            |
| PAY covered procedure billed with a cosmetic-risk diagnosis                                           |                 6 |                          6 |               1      |                           6 | PAY-09-R03 (6); CLN-03-R01 (4); CLN-03-R02 (4); ENT-04-R01 (4)      |
| PAY unbundling: panel component billed beside its panel                                               |                 6 |                          6 |               1      |                           3 | PAY-02-R01 (6); CLN-08-R01 (6)                                      |
| PHR-01 billed product differs from prescribed (non-equivalent)                                        |                 6 |                          6 |               1      |                           6 | PHR-01-R01 (6)                                                      |
| PHR-03 second-line drug without first-line history, or drug not indicated                             |                 6 |                          6 |               1      |                           0 | PHR-03-R02 (6)                                                      |
| PHR-05 one device serial billed for two patients                                                      |                 6 |                          6 |               1      |                           6 | PAY-03-R01 (6)                                                      |
| PHR-01 pharmacy billed more of a product in a month than its stock records allow                      |                 6 |                          0 |               0      |                           6 | — none —                                                            |
| PAY payment increased after settlement with no reason on record                                       |                 6 |                          6 |               1      |                           4 | PAY-06-R02 (6); PAY-11-R03 (6); NET-03-R01 (2); PAY-06-R03 (1)      |
| PAY patient share waived and shifted onto the insurer                                                 |                 6 |                          6 |               1      |                           6 | PAY-07-R01 (6); PAY-07-R02 (6); NET-03-R01 (2)                      |
| PAY second insurer paid the bill and we paid too                                                      |                 6 |                          6 |               1      |                           5 | PAY-10-R02 (6); PAY-01-R04 (6); PAY-10-R01 (6); NET-03-R01 (4)      |
| ENT-04 physiotherapist billed for three overlapping timed sessions                                    |                 6 |                          6 |               1      |                           6 | PAY-03-R04 (6); NET-03-R01 (5)                                      |
| ENT-01 physiotherapy course billed past the annual benefit limit                                      |                 6 |                          6 |               1      |                           6 | PAY-06-R03 (6); CLN-05-R03 (6); PAY-01-R02 (6); PAY-03-R03 (6)      |
| ENT-03 GP whose privileges exclude joint injections billing one                                       |                 6 |                          6 |               1      |                           6 | ENT-03-R03 (6)                                                      |
| DOC-02 discharge summary rewritten after a denial (longer stay, severity added)                       |                 6 |                          6 |               1      |                           6 | DOC-02-R03 (6)                                                      |
| DOC-01 approval requested for a plain X-ray or ultrasound, advanced scan billed                       |                 6 |                          6 |               1      |                           4 | DOC-01-R04 (6); NET-03-R01 (2)                                      |
| PAY claim cancelled after payment; payment never reversed                                             |                 6 |                          6 |               1      |                           6 | PAY-12-R01 (6); PAY-06-R03 (1); NET-03-R01 (1)                      |
| PAY provider refund received but never applied (one applied months late)                              |                 6 |                          1 |               0.1667 |                           6 | NET-03-R01 (1)                                                      |
| PAY price above the contracted tariff                                                                 |                 6 |                          6 |               1      |                           4 | PAY-06-R01 (6); NET-03-R01 (2); POL-01-R04 (1)                      |
| ENT-01 cosmetic service billed under a plan that excludes it                                          |                 6 |                          6 |               1      |                           5 | PAY-06-R03 (6); ENT-01-R02 (6); PAY-07-R01 (6); ENT-04-R01 (6)      |
| PHR-03 daily dose above the mg/kg limit                                                               |                 5 |                          5 |               1      |                           0 | PHR-03-R01 (5); PAY-03-R01 (2)                                      |
| PAY demographic impossibility: service outside its sex or age restriction                             |                 5 |                          5 |               1      |                           2 | PAY-03-R02 (5); DOC-02-R02 (3); CLN-08-R01 (3)                      |
| PAY approval scope: another code or another provider                                                  |                 5 |                          5 |               1      |                           5 | PAY-04-R03 (5); CLN-03-R03 (5)                                      |
| PAY approval refused, or service after the approval expired                                           |                 5 |                          5 |               1      |                           5 | PAY-04-R02 (5); CLN-03-R03 (5)                                      |
| PAY cross-payer duplicate: second insurer also paid, no coordination record                           |                 5 |                          5 |               1      |                           5 | PAY-01-R04 (5); NET-03-R01 (3); CLN-05-R03 (1); POL-01-R04 (1)      |
| ENT-04 clinician from another facility billed as the treating clinician                               |                 5 |                          5 |               1      |                           4 | ENT-04-R02 (5); NET-03-R01 (3)                                      |
| ENT-05 inpatient stay billed with no bed assigned                                                     |                 5 |                          5 |               1      |                           5 | ENT-05-R02 (5); PAY-06-R03 (2); NET-03-R01 (2); PAY-10-R01 (1)      |
| POL-01 member added after a costly service with cover backdated over it                               |                 5 |                          5 |               1      |                           5 | POL-01-R02 (5); NET-03-R01 (5); PAY-06-R03 (3)                      |
| PAY paid in full after a documented third-party settlement                                            |                 5 |                          5 |               1      |                           5 | PAY-10-R04 (5); NET-03-R01 (3)                                      |
| PAY prior approval missing for advanced imaging                                                       |                 5 |                          5 |               1      |                           5 | PAY-04-R01 (5); CLN-03-R03 (5)                                      |
| PAY member receipt names a different item from the one billed                                         |                 5 |                          5 |               1      |                           4 | PAY-09-R04 (5); NET-03-R01 (3)                                      |
| PAY modifier 22 without the supporting report                                                         |                 5 |                          5 |               1      |                           2 | PAY-05-R02 (5)                                                      |
| PHR-01 dispensing record differs from billed product or quantity                                      |                 5 |                          5 |               1      |                           5 | PHR-01-R03 (5)                                                      |
| PHR-01 receipt shows a non-medical item billed as a covered medicine                                  |                 5 |                          5 |               1      |                           5 | PAY-07-R04 (5); PAY-09-R04 (5)                                      |
| PHR-01 billed quantity three times the prescription                                                   |                 5 |                          5 |               1      |                           5 | PHR-01-R02 (5)                                                      |
| ENT-03 GP billing specialist consultation codes                                                       |                 4 |                          4 |               1      |                           4 | ENT-03-R04 (4); PAY-06-R03 (2); NET-03-R01 (1)                      |
| PAY modifier 59 used to bypass an edit that allows no modifier                                        |                 4 |                          4 |               1      |                           2 | PAY-02-R01 (4); PAY-05-R01 (4); CLN-08-R01 (3)                      |
| PAY unit maximum: units above the daily maximum                                                       |                 4 |                          4 |               1      |                           3 | PAY-03-R03 (4); PAY-06-R03 (2); DOC-01-R01 (1)                      |
| PAY time units: billed minutes exceed the recorded window                                             |                 4 |                          4 |               1      |                           3 | PAY-03-R04 (4)                                                      |
| PAY package: included component charged, refused at remittance                                        |                 4 |                          4 |               1      |                           4 | PAY-02-R03 (4); PAY-06-R03 (3); NET-03-R01 (1)                      |
| PAY payment reference reused across providers and dates                                               |                 4 |                          4 |               1      |                           4 | PAY-12-R02 (4); NET-03-R01 (2); PAY-01-R03 (1)                      |
| PAY invalid code: deleted code billed after its end date                                              |                 4 |                          4 |               1      |                           4 | PAY-03-R01 (4)                                                      |
| PAY package leakage: included component charged and paid                                              |                 4 |                          4 |               1      |                           4 | PAY-02-R04 (4); PAY-02-R03 (4); PAY-04-R04 (4); NET-03-R01 (3)      |
| PHR-05 implant with no procedure, or on the opposite side                                             |                 4 |                          4 |               1      |                           4 | PAY-06-R03 (4); PHR-05-R02 (4); NET-03-R01 (2); PAY-01-R03 (2)      |
| ENT-06 three-minute teleconsult ending in a prescription                                              |                 4 |                          4 |               1      |                           4 | ENT-06-R02 (4); PHR-03-R02 (1)                                      |
| PAY approval reused for other patients                                                                |                 4 |                          4 |               1      |                           4 | PAY-04-R05 (4); PAY-04-R04 (4); CLN-03-R03 (4)                      |
| ENT-06 telehealth visit billing an ECG                                                                |                 4 |                          4 |               1      |                           3 | ENT-06-R01 (4)                                                      |
| PAY approval exhausted: drug course beyond the approved units                                         |                 4 |                          4 |               1      |                           2 | PAY-06-R03 (4); PAY-04-R04 (4); NET-03-R01 (4); PAY-01-R03 (4)      |
| PAY cross-provider unbundling: ECG interpretation billed again                                        |                 4 |                          4 |               1      |                           4 | PAY-02-R02 (4)                                                      |
| ENT-04 clinician billed while on recorded annual leave                                                |                 3 |                          0 |               0      |                           3 | — none —                                                            |
| CLN the same diagnosis code listed twice on a claim                                                   |                 3 |                          3 |               1      |                           3 | CLN-02-R04 (3); DOC-01-R02 (1); PAY-06-R03 (1); NET-03-R01 (1)      |
| CLN lipid panel billed together with its own cholesterol component                                    |                 3 |                          3 |               1      |                           1 | PAY-02-R01 (3)                                                      |
| PHR-05 used or rented equipment billed as a new purchase                                              |                 3 |                          3 |               1      |                           3 | PAY-03-R01 (3); PHR-05-R01 (3)                                      |
| PAY package incomplete: core activities not itemised                                                  |                 3 |                          3 |               1      |                           3 | PAY-02-R03 (3); PAY-06-R03 (2); NET-03-R01 (2); CLN-05-R03 (1)      |
| NET-03 several patients of one clinic say billed services were never given                            |                 3 |                          2 |               0.6667 |                           3 | PAY-09-R04 (2); CLN-06-R04 (2)                                      |
| ENT-05 service billed by a pharmacy not licensed for it                                               |                 3 |                          3 |               1      |                           2 | ENT-05-R01 (3); POL-01-R01 (1); ENT-01-R01 (1)                      |
| ENT-04 consultation line with no treating clinician                                                   |                 3 |                          3 |               1      |                           2 | ENT-04-R01 (3)                                                      |
| ENT-04 clinician billed in two distant emirates twenty minutes apart                                  |                 2 |                          1 |               0.5    |                           2 | ENT-04-R02 (1)                                                      |
| ENT-05 service billed by a radiology centre not licensed for it                                       |                 2 |                          2 |               1      |                           0 | ENT-05-R01 (2); CLN-08-R01 (2); ENT-01-R01 (1); POL-01-R01 (1)      |
| PAY post-edit: plain component billed with its parent after the edit                                  |                 2 |                          2 |               1      |                           2 | PAY-02-R01 (2)                                                      |
| PAY cross-payer duplicate: secondary insurer paid the full bill                                       |                 2 |                          2 |               1      |                           1 | PAY-10-R02 (2); PAY-10-R01 (2); PAY-01-R04 (2); POL-01-R04 (1)      |
| ENT-04 consultation line with the facility's own code as treating clinician                           |                 2 |                          2 |               1      |                           2 | ENT-04-R01 (2); NET-03-R01 (1); PAY-01-R03 (1)                      |
| PAY invalid code: code absent from the code set                                                       |                 2 |                          2 |               1      |                           2 | PAY-03-R01 (2); PAY-07-R01 (1)                                      |
| PAY unit maximum: daily units split over two claims                                                   |                 2 |                          2 |               1      |                           2 | PAY-01-R01 (2); CLN-05-R03 (2); PAY-01-R02 (2); PAY-03-R03 (2)      |
| PAY member complaint of being charged more than the share                                             |                 2 |                          2 |               1      |                           2 | PAY-07-R04 (2); NET-03-R01 (1)                                      |
| ENT-04 consultation line with a clinician ID on no roster                                             |                 1 |                          1 |               1      |                           1 | ENT-04-R01 (1)                                                      |
| PAY time units: one clinician in two timed sessions at once                                           |                 1 |                          1 |               1      |                           0 | PAY-03-R04 (1)                                                      |


**Read this table with the generator in mind.** Every labelled type in a generated file is encoded through some field, and a control that tests that field will recover the type almost perfectly. That is a fact about the generator rather than evidence the control works on real claims. Measured here:

- `CLN ECG billed for an upper respiratory infection with no cardiac reason` (12 claims) separates most on **`icd_code_matches_procedure`** — 0.00 against 1.00 elsewhere, 16.5 standard deviations apart.
- `CLN HbA1c repeated twelve days later, before the first result was recorded` (16 claims) separates most on **`pharmacy_bill_ratio`** — 0.00 against 0.23 elsewhere, 0.6 standard deviations apart.
- `CLN MRI scans billed beyond one day's scanner capacity (18)` (27 claims) separates most on **`patient_share`** — 219.95 against 60.51 elsewhere, 0.9 standard deviations apart.
- `CLN a general practice billing electroencephalograms` (49 claims) separates most on **`pharmacy_bill_ratio`** — 0.00 against 0.23 elsewhere, 0.6 standard deviations apart.
- `CLN all basic metabolic panel components billed one by one` (8 claims) separates most on **`pharmacy_bill_ratio`** — 0.00 against 0.23 elsewhere, 0.6 standard deviations apart.
- `CLN clinic ordering genetic tests for a large share of its patients` (34 claims) separates most on **`patient_share`** — 241.36 against 60.43 elsewhere, 1.1 standard deviations apart.
- `CLN clinic re-billing a reference laboratory's tests at three times its price` (13 claims) separates most on **`days_since_policy_start`** — 1283.38 against 877.84 elsewhere, 0.7 standard deviations apart.
- `CLN clinician billing more than seventeen hours of visits in one day` (78 claims) separates most on **`pharmacy_bill_ratio`** — 0.00 against 0.23 elsewhere, 0.6 standard deviations apart.
- `CLN complication code lifts the case rate but is not in the discharge summary` (13 claims) separates most on **`length_of_stay_days`** — 3.46 against 0.18 elsewhere, 4.2 standard deviations apart.
- `CLN diabetes billed repeatedly with no medicine or result ever recorded` (24 claims) separates most on **`pharmacy_bill_ratio`** — 0.00 against 0.23 elsewhere, 0.6 standard deviations apart.
- `CLN hospital admitting overnight presentations others treat as outpatients` (70 claims) separates most on **`icd_code_matches_procedure`** — 0.00 against 1.00 elsewhere, 27.4 standard deviations apart.
- `CLN hospital records (and documents) complication codes on most stays` (86 claims) separates most on **`gross_amount`** — 9030.89 against 1002.65 elsewhere, 2.6 standard deviations apart.
- `CLN hospital severity index jumps from May 2025 with unchanged stays` (56 claims) separates most on **`gross_amount`** — 13631.23 against 1001.98 elsewhere, 4.0 standard deviations apart.
- `CLN identical multi-value results reported for unrelated patients` (6 claims) separates most on **`pharmacy_bill_ratio`** — 0.00 against 0.23 elsewhere, 0.6 standard deviations apart.
- `CLN identical three-service encounter at 10:00 for different patients` (6 claims) separates most on **`pharmacy_bill_ratio`** — 0.00 against 0.23 elsewhere, 0.6 standard deviations apart.
- `CLN imaging paid with no report or result on file` (10 claims) separates most on **`pharmacy_bill_ratio`** — 0.00 against 0.23 elsewhere, 0.6 standard deviations apart.
- `CLN knee MRI with no knee X-ray first` (9 claims) separates most on **`patient_share`** — 207.43 against 60.63 elsewhere, 0.9 standard deviations apart.
- `CLN laboratory tests paid with no result recorded` (8 claims) separates most on **`pharmacy_bill_ratio`** — 0.00 against 0.23 elsewhere, 0.6 standard deviations apart.
- `CLN laboratory tests with no ordering clinician, or one on no roster` (8 claims) separates most on **`is_cashless`** — 0.88 against 0.98 elsewhere, 0.7 standard deviations apart.
- `CLN level migration to top-level visits from April 2025` (279 claims) separates most on **`pharmacy_bill_ratio`** — 0.00 against 0.23 elsewhere, 0.6 standard deviations apart.
- `CLN long-standing condition on one claim only, absent from a long history` (9 claims) separates most on **`is_cashless`** — 0.89 against 0.98 elsewhere, 0.6 standard deviations apart.
- `CLN main diagnosis marked not present on admission` (8 claims) separates most on **`net_amount`** — 9076.99 against 967.10 elsewhere, 2.6 standard deviations apart.
- `CLN member says the visit did not happen (verified app response)` (10 claims) separates most on **`pharmacy_bill_ratio`** — 0.00 against 0.23 elsewhere, 0.6 standard deviations apart.
- `CLN minor clinic visits reliably followed by a large specialist, CT and echo bundle` (32 claims) separates most on **`patient_share`** — 214.39 against 60.48 elsewhere, 0.9 standard deviations apart.
- `CLN ten consultations for a cold within one month` (30 claims) separates most on **`days_since_policy_start`** — 520.17 against 878.48 elsewhere, 0.6 standard deviations apart.
- `DOC-01 approval requested for a plain X-ray or ultrasound, advanced scan billed` (6 claims) separates most on **`patient_share`** — 203.37 against 60.64 elsewhere, 0.8 standard deviations apart.
- `DOC-01 high-cost service without the required indication in the record` (8 claims) separates most on **`patient_share`** — 587.64 against 60.51 elsewhere, 3.1 standard deviations apart.
- `DOC-01 required document never sent (operative note, discharge summary or imaging report)` (8 claims) separates most on **`length_of_stay_days`** — 1.12 against 0.18 elsewhere, 1.2 standard deviations apart.
- `DOC-02 discharge summary carries another patient's age or sex` (8 claims) separates most on **`length_of_stay_days`** — 3.38 against 0.18 elsewhere, 4.1 standard deviations apart.
- `DOC-02 discharge summary rewritten after a denial (longer stay, severity added)` (6 claims) separates most on **`length_of_stay_days`** — 4.00 against 0.18 elsewhere, 4.9 standard deviations apart.
- `DOC-02 one-line discharge summaries for high-complexity admissions` (27 claims) separates most on **`discount`** — 1179.68 against 89.48 elsewhere, 3.5 standard deviations apart.
- `ENT-01 clinic billing after leaving the payer network` (16 claims) separates most on **`num_insurers_same_event`** — 1.25 against 1.03 elsewhere, 1.4 standard deviations apart.
- `ENT-01 cosmetic service billed under a plan that excludes it` (6 claims) separates most on **`patient_share`** — 3072.32 against 59.95 elsewhere, 18.4 standard deviations apart.
- `ENT-01 physiotherapy course billed past the annual benefit limit` (6 claims) separates most on **`days_since_policy_start`** — 90.00 against 878.24 elsewhere, 1.3 standard deviations apart.
- `ENT-01 service after cover ended` (8 claims) separates most on **`pharmacy_bill_ratio`** — 0.00 against 0.23 elsewhere, 0.6 standard deviations apart.
- `ENT-02 one Emirates-ID reference on two records with different dates of birth` (23 claims) separates most on **`num_insurers_same_event`** — 1.43 against 1.03 elsewhere, 2.6 standard deviations apart.
- `ENT-02 one member's cover used in two distant emirates on the same days` (20 claims) separates most on **`days_since_policy_start`** — 1303.90 against 877.71 elsewhere, 0.7 standard deviations apart.
- `ENT-02 services after the member's recorded death` (10 claims) separates most on **`net_amount`** — 3960.15 against 968.51 elsewhere, 1.0 standard deviations apart.
- `ENT-03 GP whose privileges exclude joint injections billing one` (6 claims) separates most on **`is_cashless`** — 0.83 against 0.98 elsewhere, 1.0 standard deviations apart.
- `ENT-03 clinic billing after its facility licence expired` (16 claims) separates most on **`pharmacy_bill_ratio`** — 0.00 against 0.23 elsewhere, 0.6 standard deviations apart.
- `ENT-03 clinician billing after their licence expired` (13 claims) separates most on **`discount`** — 771.11 against 90.30 elsewhere, 2.2 standard deviations apart.
- `ENT-03 newly credentialed clinic billing far above peers at once` (48 claims) separates most on **`days_since_policy_start`** — 1290.56 against 877.25 elsewhere, 0.7 standard deviations apart.
- `ENT-04 clinician from another facility billed as the treating clinician` (5 claims) separates most on **`pharmacy_bill_ratio`** — 0.00 against 0.23 elsewhere, 0.6 standard deviations apart.
- `ENT-04 physiotherapist billed for three overlapping timed sessions` (6 claims) separates most on **`pharmacy_bill_ratio`** — 0.00 against 0.23 elsewhere, 0.6 standard deviations apart.
- `ENT-05 hospital recording ordinary visits as day cases from August 2025` (21 claims) separates most on **`pharmacy_bill_ratio`** — 0.00 against 0.23 elsewhere, 0.6 standard deviations apart.
- `ENT-05 inpatient stay billed with no bed assigned` (5 claims) separates most on **`discount`** — 954.87 against 90.49 elsewhere, 2.8 standard deviations apart.
- `ENT-05 outpatient visit billed at the same hospital during the stay` (10 claims) separates most on **`discount`** — 861.05 against 90.35 elsewhere, 2.5 standard deviations apart.
- `ENT-06 telehealth clinic sending nearly every prescription to one pharmacy` (36 claims) separates most on **`pharmacy_bill_ratio`** — 0.50 against 0.23 elsewhere, 0.6 standard deviations apart.
- `ENT-06 telehealth clinic whose calls turn into MRI referrals far more often than peers'` (57 claims) separates most on **`pharmacy_bill_ratio`** — 0.00 against 0.23 elsewhere, 0.6 standard deviations apart.
- `NET-01 clinic keeps its lab and imaging referrals inside providers sharing its bank account and phone` (14 claims) separates most on **`pharmacy_bill_ratio`** — 0.00 against 0.23 elsewhere, 0.6 standard deviations apart.
- `NET-01 radiology centre turning every referral into a contrast CT` (24 claims) separates most on **`patient_share`** — 214.60 against 60.53 elsewhere, 0.9 standard deviations apart.
- `NET-01 two clinics referring patients back and forth` (22 claims) separates most on **`pharmacy_bill_ratio`** — 0.00 against 0.23 elsewhere, 0.6 standard deviations apart.
- `NET-02 quiet clinic becomes one of the busiest right after an ownership change` (112 claims) separates most on **`pharmacy_bill_ratio`** — 0.00 against 0.23 elsewhere, 0.6 standard deviations apart.
- `NET-03 clinic waiving patient share on repeated knee injections for a small patient group` (32 claims) separates most on **`pharmacy_bill_ratio`** — 0.00 against 0.23 elsewhere, 0.6 standard deviations apart.
- `NET-03 one small employer's staff all treated by the same two clinics` (42 claims) separates most on **`is_cashless`** — 0.86 against 0.98 elsewhere, 0.9 standard deviations apart.
- `NET-04 clinic owned by the agent who sold its patients' policies` (10 claims) separates most on **`pharmacy_bill_ratio`** — 0.00 against 0.23 elsewhere, 0.6 standard deviations apart.
- `PAY approval refused, or service after the approval expired` (5 claims) separates most on **`patient_share`** — 208.86 against 60.65 elsewhere, 0.9 standard deviations apart.
- `PAY approval scope: another code or another provider` (5 claims) separates most on **`patient_share`** — 229.76 against 60.64 elsewhere, 1.0 standard deviations apart.
- `PAY claim cancelled after payment; payment never reversed` (6 claims) separates most on **`discount`** — 220.61 against 90.63 elsewhere, 0.4 standard deviations apart.
- `PAY clinic bills a payable substitute code after refused services` (32 claims) separates most on **`icd_code_matches_procedure`** — 0.50 against 1.00 elsewhere, 8.4 standard deviations apart.
- `PAY clinic systematically records zero or rounded-down patient share` (49 claims) separates most on **`pharmacy_bill_ratio`** — 0.00 against 0.23 elsewhere, 0.6 standard deviations apart.
- `PAY clinic's changed resubmissions after one refusal reason always paid` (22 claims) separates most on **`pharmacy_bill_ratio`** — 0.00 against 0.23 elsewhere, 0.6 standard deviations apart.
- `PAY clinical note describes a cosmetic service billed under a covered code` (8 claims) separates most on **`pharmacy_bill_ratio`** — 0.00 against 0.23 elsewhere, 0.6 standard deviations apart.
- `PAY covered procedure billed with a cosmetic-risk diagnosis` (6 claims) separates most on **`icd_code_matches_procedure`** — 0.33 against 1.00 elsewhere, 10.5 standard deviations apart.
- `PAY cross-payer duplicate: second insurer also paid, no coordination record` (5 claims) separates most on **`days_since_policy_start`** — 515.80 against 878.12 elsewhere, 0.6 standard deviations apart.
- `PAY demographic impossibility: service outside its sex or age restriction` (5 claims) separates most on **`pharmacy_bill_ratio`** — 0.00 against 0.23 elsewhere, 0.6 standard deviations apart.
- `PAY member receipt exceeds the approved patient share` (9 claims) separates most on **`pharmacy_bill_ratio`** — 0.00 against 0.23 elsewhere, 0.6 standard deviations apart.
- `PAY member receipt names a different item from the one billed` (5 claims) separates most on **`is_cashless`** — 0.80 against 0.98 elsewhere, 1.2 standard deviations apart.
- `PAY member says the billed service did not take place` (8 claims) separates most on **`pharmacy_bill_ratio`** — 0.00 against 0.23 elsewhere, 0.6 standard deviations apart.
- `PAY modifier 22 without the supporting report` (5 claims) separates most on **`days_since_policy_start`** — 1418.40 against 877.94 elsewhere, 0.9 standard deviations apart.
- `PAY modifier 25 appended to most visits` (278 claims) separates most on **`pharmacy_bill_ratio`** — 0.00 against 0.23 elsewhere, 0.6 standard deviations apart.
- `PAY novel unbundling: inflammation markers always billed as a separate pair` (22 claims) separates most on **`pharmacy_bill_ratio`** — 0.00 against 0.23 elsewhere, 0.6 standard deviations apart.
- `PAY one adjudicator overrides one hospital's claims far more than colleagues` (10 claims) separates most on **`discharge_readmit_gap_days`** — 724.70 against 992.21 elsewhere, 3.5 standard deviations apart.
- `PAY one adjudicator pays one provider's pended claims in full, uniformly` (15 claims) separates most on **`pharmacy_bill_ratio`** — 0.00 against 0.23 elsewhere, 0.6 standard deviations apart.
- `PAY paid in full after a documented third-party settlement` (5 claims) separates most on **`accident_indicator`** — 1.00 against 0.00 elsewhere, 27.4 standard deviations apart.
- `PAY patient share below the benefit co-payment` (11 claims) separates most on **`pharmacy_bill_ratio`** — 0.00 against 0.23 elsewhere, 0.6 standard deviations apart.
- `PAY patient share left inside the amount claimed from the insurer` (7 claims) separates most on **`pharmacy_bill_ratio`** — 0.00 against 0.23 elsewhere, 0.6 standard deviations apart.
- `PAY patient share waived and shifted onto the insurer` (6 claims) separates most on **`days_since_policy_start`** — 491.00 against 878.14 elsewhere, 0.6 standard deviations apart.
- `PAY payment check overridden without a reason or by an unauthorised role` (9 claims) separates most on **`days_since_policy_start`** — 699.78 against 878.12 elsewhere, 0.3 standard deviations apart.
- `PAY payment increased after settlement with no reason on record` (6 claims) separates most on **`days_since_policy_start`** — 1302.33 against 877.95 elsewhere, 0.7 standard deviations apart.
- `PAY post-edit migration: modifier 59 after a new edit` (9 claims) separates most on **`pharmacy_bill_ratio`** — 0.00 against 0.23 elsewhere, 0.6 standard deviations apart.
- `PAY price above the contracted tariff` (6 claims) separates most on **`pharmacy_bill_ratio`** — 0.00 against 0.23 elsewhere, 0.6 standard deviations apart.
- `PAY prior approval missing for advanced imaging` (5 claims) separates most on **`patient_share`** — 203.64 against 60.65 elsewhere, 0.8 standard deviations apart.
- `PAY provider offsets credits against unrelated claims and bills negative lines` (9 claims) separates most on **`approved_amount`** — -806.22 against 909.89 elsewhere, 0.6 standard deviations apart.
- `PAY provider refund received but never applied (one applied months late)` (6 claims) separates most on **`discount`** — 250.60 against 90.62 elsewhere, 0.5 standard deviations apart.
- `PAY refused claim resubmitted with diagnosis and code swapped` (20 claims) separates most on **`pharmacy_bill_ratio`** — 0.00 against 0.23 elsewhere, 0.6 standard deviations apart.
- `PAY resubmission with a missing, unknown or unrelated original` (7 claims) separates most on **`pharmacy_bill_ratio`** — 0.00 against 0.23 elsewhere, 0.6 standard deviations apart.
- `PAY road-traffic or work injury with no liable-party record` (9 claims) separates most on **`accident_indicator`** — 0.22 against 0.00 elsewhere, 5.8 standard deviations apart.
- `PAY same line paid twice` (7 claims) separates most on **`approved_amount`** — 4271.33 against 908.32 elsewhere, 1.1 standard deviations apart.
- `PAY second insurer paid the bill and we paid too` (6 claims) separates most on **`num_insurers_same_event`** — 2.00 against 1.03 elsewhere, 6.2 standard deviations apart.
- `PAY service resubmitted four times with a different change each time` (20 claims) separates most on **`pharmacy_bill_ratio`** — 0.00 against 0.23 elsewhere, 0.6 standard deviations apart.
- `PAY unbundling: panel component billed beside its panel` (6 claims) separates most on **`is_cashless`** — 0.83 against 0.98 elsewhere, 1.0 standard deviations apart.
- `PHR-01 billed product differs from prescribed (non-equivalent)` (6 claims) separates most on **`pharmacy_bill_ratio`** — 1.00 against 0.23 elsewhere, 1.8 standard deviations apart.
- `PHR-01 billed quantity three times the prescription` (5 claims) separates most on **`pharmacy_bill_ratio`** — 1.00 against 0.23 elsewhere, 1.8 standard deviations apart.
- `PHR-01 dispensing record differs from billed product or quantity` (5 claims) separates most on **`pharmacy_bill_ratio`** — 1.00 against 0.23 elsewhere, 1.8 standard deviations apart.
- `PHR-01 pharmacy billed more of a product in a month than its stock records allow` (6 claims) separates most on **`pharmacy_bill_ratio`** — 1.00 against 0.23 elsewhere, 1.8 standard deviations apart.
- `PHR-01 receipt shows a non-medical item billed as a covered medicine` (5 claims) separates most on **`pharmacy_bill_ratio`** — 1.00 against 0.23 elsewhere, 1.8 standard deviations apart.
- `PHR-02 early refills (stockpiling)` (20 claims) separates most on **`pharmacy_bill_ratio`** — 1.00 against 0.23 elsewhere, 1.8 standard deviations apart.
- `PHR-02 same controlled medicine from several unrelated prescribers` (14 claims) separates most on **`pharmacy_bill_ratio`** — 1.00 against 0.23 elsewhere, 1.8 standard deviations apart.
- `PHR-02 same medicine collected from several pharmacies` (12 claims) separates most on **`pharmacy_bill_ratio`** — 1.00 against 0.23 elsewhere, 1.8 standard deviations apart.
- `PHR-02 therapy continued beyond the policy duration on one prescription` (30 claims) separates most on **`pharmacy_bill_ratio`** — 1.00 against 0.23 elsewhere, 1.8 standard deviations apart.
- `PHR-03 daily dose above the mg/kg limit` (5 claims) separates most on **`pharmacy_bill_ratio`** — 1.00 against 0.23 elsewhere, 1.8 standard deviations apart.
- `PHR-03 second-line drug without first-line history, or drug not indicated` (6 claims) separates most on **`pharmacy_bill_ratio`** — 1.00 against 0.23 elsewhere, 1.8 standard deviations apart.
- `PHR-03 wastage billed far above vial arithmetic` (20 claims) separates most on **`patient_share`** — 970.66 against 59.95 elsewhere, 5.4 standard deviations apart.
- `PHR-04 clinic-to-pharmacy flow with shared bank account and phone` (27 claims) separates most on **`pharmacy_bill_ratio`** — 1.00 against 0.23 elsewhere, 1.8 standard deviations apart.
- `PHR-04 high-cost prescriptions steered to one pharmacy` (20 claims) separates most on **`pharmacy_bill_ratio`** — 1.00 against 0.23 elsewhere, 1.8 standard deviations apart.
- `PHR-04 new prescriber-pharmacy link rapidly dominant` (24 claims) separates most on **`pharmacy_bill_ratio`** — 1.00 against 0.23 elsewhere, 1.8 standard deviations apart.
- `PHR-04 prescriber concentrated on one pharmacy` (30 claims) separates most on **`pharmacy_bill_ratio`** — 1.00 against 0.23 elsewhere, 1.8 standard deviations apart.
- `PHR-05 one device serial billed for two patients` (6 claims) separates most on **`patient_share`** — 368.71 against 60.60 elsewhere, 1.8 standard deviations apart.
- `PHR-05 supplies beyond usual consumption or useful life` (7 claims) separates most on **`is_cashless`** — 0.86 against 0.98 elsewhere, 0.8 standard deviations apart.
- `POL-01 application declared no prior cover and no conditions although earlier treatment is on record` (19 claims) separates most on **`days_since_policy_start`** — 130.58 against 878.62 elsewhere, 1.2 standard deviations apart.
- `POL-01 employer-sponsored member not on the employer's roster` (45 claims) separates most on **`num_insurers_same_event`** — 1.22 against 1.03 elsewhere, 1.3 standard deviations apart.
- `POL-01 member added after a costly service with cover backdated over it` (5 claims) separates most on **`approved_amount`** — 11650.21 against 907.11 elsewhere, 3.6 standard deviations apart.
- `POL-01 new employer group claiming within weeks at two clinics sharing a bank account` (18 claims) separates most on **`days_since_policy_start`** — 23.94 against 878.67 elsewhere, 1.4 standard deviations apart.
- `POL-01 one person enrolled twice with overlapping cover` (13 claims) separates most on **`days_since_policy_start`** — 643.38 against 878.17 elsewhere, 0.4 standard deviations apart.

A recall figure against these labels therefore says how faithfully a control reads the encoding, not how well it would find the behaviour the label names.


## 7. Lift over the transparent composite

| layer                             |   entities_ranked |   capacity_k |   review_yield_proxy |   lift_over_base_rate |   lift_over_composite |
|:----------------------------------|------------------:|-------------:|---------------------:|----------------------:|----------------------:|
| ANL-01-R01 transparent composite  |               231 |          100 |                 0.83 |                  1.02 |                  1    |
| model: isolation_forest           |               113 |          100 |                 0.86 |                  1.06 |                  1.04 |
| model: local_outlier_factor       |               113 |          100 |                 0.86 |                  1.06 |                  1.04 |
| deterministic + statistical rules |               189 |          100 |                 0.83 |                  1.02 |                  1    |


Base rate among ranked providers: 81.2%. The promotion gate requires a model to demonstrate **prospective lift over ANL-01-R01, the transparent composite**, using future-period review yield rather than in-sample separation — and review yield needs reviewer outcomes, which do not exist. The table above therefore uses a **label proxy** and is the closest this dataset permits, not the measurement the gate actually requires.


## 8. Calibration of the priority score

| bin                        |   cases |   mean_priority |   observed_rate |   predicted_rate |    gap |
|:---------------------------|--------:|----------------:|----------------:|-----------------:|-------:|
| (49.099000000000004, 65.5] |     747 |         61.5075 |          0.5007 |           0.6151 | 0.1144 |
| (65.5, 67.4]               |     733 |         66.122  |          0.1392 |           0.6612 | 0.5221 |
| (67.4, 73.5]               |     759 |         71.1556 |          0.61   |           0.7116 | 0.1015 |
| (73.5, 76.2]               |     724 |         74.9526 |          0.6851 |           0.7495 | 0.0644 |
| (76.2, 81.212]             |     721 |         78.53   |          0.4494 |           0.7853 | 0.3359 |
| (81.212, 86.2]             |     738 |         83.623  |          0.3943 |           0.8362 | 0.4419 |
| (86.2, 92.9]               |     768 |         89.6501 |          0.3203 |           0.8965 | 0.5762 |
| (92.9, 99.9]               |     704 |         95.0892 |          0.3338 |           0.9509 | 0.6171 |


The priority score is a **queue-ordering score, not a probability**. It is rescaled to [0,1] here only so a calibration curve can be drawn; a poor fit is expected and is not evidence the score is broken.


## 9. Evaluation protocol


### Synthetic-corruption injection — **PASS**

Injected 40 permanently-marked SYNTHETIC_INJECTED records into a copy of the claim frame and re-ran the target controls. exact_duplicate: 20/20 detected by PAY-01-R01; upcoding: 20/20 detected by CLN-03-R01


> This confirms a control detects the pattern it was DESIGNED for. It says nothing about real-world prevalence or precision. Every injected record is marked SYNTHETIC_INJECTED, exists only in a copy, and never enters the real signal store.


| injected_pattern   | expected_control   |   injected |   detected |   detection_rate |
|:-------------------|:-------------------|-----------:|-----------:|-----------------:|
| exact_duplicate    | PAY-01-R01         |         20 |         20 |                1 |
| upcoding           | CLN-03-R01         |         20 |         20 |                1 |


### Random-audit sampler — **INFORMATIONAL**

Stratified random sample of 399 claims drawn by TPA × policy type, IRRESPECTIVE of flag status. Estimated FALSE-NEGATIVE RATE 23.1% (9 of 39 labelled-fraud claims in the sample were not flagged). Random-audit MISS RATE 2.26% of all audited claims.


> This is a SIMULATED audit over labels that were themselves produced by the generator that planted the patterns (synthetic_injection). It therefore cannot correct for that process's blind spots — a claim it never marked is labelled legitimate here whether or not it was. A genuine random audit needs HUMAN review of a representative sample regardless of flag status, and its absence blocks supervised modelling.


| group                             |   n |
|:----------------------------------|----:|
| audited claims                    | 399 |
| labelled fraud in sample          |  39 |
| labelled fraud the system FLAGGED |  30 |
| labelled fraud the system MISSED  |   9 |
| flagged but not labelled          | 110 |


### Temporal holdout — **PASS**

Training data ends at 2025-05-15; scoring data begins after it. 7,925 training claims, 4,561 scored claims.


> 12,369 of 24,855 claims (50%) are scored by NEITHER model: they belong to a provider held out for training, or fall in the training period. This is the price of combining a temporal split with entity isolation on a single-file dataset, and it is reported rather than absorbed — a model that scored everything would have been trained on the entities it scores.


| split_date   |   train_claims |   score_claims |   train_providers |   score_providers |   provider_overlap |   excluded_claims |   total_claims | coverage_note                                                                                                                                                                                                                                                                                                                                                           |
|:-------------|---------------:|---------------:|------------------:|------------------:|-------------------:|------------------:|---------------:|:------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| 2025-05-15   |           7925 |           4561 |               124 |               116 |                  0 |             12369 |          24855 | 12,369 of 24,855 claims (50%) are scored by NEITHER model: they belong to a provider held out for training, or fall in the training period. This is the price of combining a temporal split with entity isolation on a single-file dataset, and it is reported rather than absorbed — a model that scored everything would have been trained on the entities it scores. |


### Entity isolation — **PASS**

0 providers appear in both the training and scoring sets (124 train / 116 score).


> 12,369 of 24,855 claims (50%) are scored by NEITHER model: they belong to a provider held out for training, or fall in the training period. This is the price of combining a temporal split with entity isolation on a single-file dataset, and it is reported rather than absorbed — a model that scored everything would have been trained on the entities it scores.


### Calibration check — **INFORMATIONAL**

Expected calibration error between the priority score (rescaled to [0,1]) and the observed labelled-fraud rate is 0.346 across 8 bins.


> The priority score is NOT a probability — it is a queue-ordering score from the priority formula. A poor calibration figure here is expected and is not evidence that the score is broken; it is evidence that it was never a probability.


| bin                        |   cases |   mean_priority |   observed_rate |   predicted_rate |    gap |
|:---------------------------|--------:|----------------:|----------------:|-----------------:|-------:|
| (49.099000000000004, 65.5] |     747 |         61.5075 |          0.5007 |           0.6151 | 0.1144 |
| (65.5, 67.4]               |     733 |         66.122  |          0.1392 |           0.6612 | 0.5221 |
| (67.4, 73.5]               |     759 |         71.1556 |          0.61   |           0.7116 | 0.1015 |
| (73.5, 76.2]               |     724 |         74.9526 |          0.6851 |           0.7495 | 0.0644 |
| (76.2, 81.212]             |     721 |         78.53   |          0.4494 |           0.7853 | 0.3359 |
| (81.212, 86.2]             |     738 |         83.623  |          0.3943 |           0.8362 | 0.4419 |
| (86.2, 92.9]               |     768 |         89.6501 |          0.3203 |           0.8965 | 0.5762 |
| (92.9, 99.9]               |     704 |         95.0892 |          0.3338 |           0.9509 | 0.6171 |


### Capacity-aware precision — **INFORMATIONAL**

At one day of team capacity (100 cases), the precision proxy is 63.0%.


> Precision proxy, not precision: nothing has been reviewed. Upper bound.


|   capacity_cases |   reviewer_days |   cases_reviewed |   precision_proxy |   labelled_cases |   exposure_reviewed_aed |
|-----------------:|----------------:|-----------------:|------------------:|-----------------:|------------------------:|
|               25 |            0.25 |               25 |            1      |               25 |             1.96971e+07 |
|              100 |            1    |              100 |            0.63   |               63 |             2.39381e+07 |
|              500 |            5    |              500 |            0.392  |              196 |             3.01801e+07 |
|             2000 |           20    |             2000 |            0.3445 |              689 |             3.74964e+07 |


### Review-yield tracking — **NOT_MEASURABLE_ON_THIS_DATASET**

Review yield is the confirmed-to-flagged ratio and needs reviewer dispositions.


> Review yield is 'the primary signal of whether a control or model is still earning its place in production'. Substituting a label-derived proxy here would be exactly the leakage this artefact is built to prevent, and would make the model promotion gate meaningless. It is reported as unmeasurable instead.


### Stability — **INFORMATIONAL**

155 controls produced monthly signals. 136 show a coefficient of variation above 1.0 across periods.


> High variation here does not necessarily mean an unstable RULE: claim volume itself varies by month, and several controls are provider-level and fire once per provider per period by design.


| rule_id    |   periods |   mean_signals_per_period |   coefficient_of_variation |
|:-----------|----------:|--------------------------:|---------------------------:|
| CLN-02-R02 |         1 |                      0.1  |                      4.583 |
| CLN-01-R03 |         1 |                      0.05 |                      4.583 |
| ANL-01-R04 |         1 |                      0.1  |                      4.583 |
| CLN-01-R04 |         1 |                      0.05 |                      4.583 |
| CLN-05-R04 |         1 |                      0.05 |                      4.583 |
| CLN-06-R02 |         1 |                      0.05 |                      4.583 |
| CLN-02-R05 |         1 |                      0.05 |                      4.583 |
| POL-01-R05 |         1 |                      0.1  |                      4.583 |
| PAY-12-R04 |         1 |                      0.05 |                      4.583 |
| PAY-11-R04 |         1 |                      0.05 |                      4.583 |
| NET-03-R02 |         1 |                      0.05 |                      4.583 |
| ENT-05-R04 |         1 |                      0.05 |                      4.583 |

_(143 further rows in the accompanying CSV.)_


### Fairness monitoring — **INFORMATIONAL**

Flag rates compared across TPA, policy type and provider volume band. Largest between-segment spread is on provider_volume_band (18.6% points).


> A spread is not by itself unfairness: segments genuinely differ in case mix, and the volume band in particular is CONSTRUCTED from claim count, so a relationship between it and flag rate is partly definitional. What would indicate unfairness is a segment whose FLAG-TO-LABEL ratio is markedly higher than others', and that column is reported here for exactly that comparison. The appeal half of 'fairness/appeal monitoring' is NOT_MEASURABLE — no appeals exist.


| segment_value   |   claims |   flag_rate |   labelled_rate | dimension            |   flag_to_label_ratio |
|:----------------|---------:|------------:|----------------:|:---------------------|----------------------:|
| SYN-TPA-1       |     8484 |      0.3346 |          0.1023 | tpa                  |                3.2707 |
| SYN-TPA-2       |     8332 |      0.3581 |          0.09   | tpa                  |                3.9787 |
| SYN-TPA-3       |     8039 |      0.3499 |          0.1255 | tpa                  |                2.7879 |
| family          |     3201 |      0.3683 |          0.094  | policy_type          |                3.9169 |
| group_corporate |    19045 |      0.3429 |          0.1062 | policy_type          |                3.23   |
| individual      |     2609 |      0.3549 |          0.1165 | policy_type          |                3.0461 |
| large           |     3939 |      0.2529 |          0.1074 | provider_volume_band |                2.3546 |
| major           |    18487 |      0.3685 |          0.0919 | provider_volume_band |                4.0094 |
| medium          |     1768 |      0.3077 |          0.185  | provider_volume_band |                1.6636 |
| micro           |       89 |      0.3708 |          0.1798 | provider_volume_band |                2.0625 |
| small           |      572 |      0.4388 |          0.2832 | provider_volume_band |                1.5494 |


## 10. Release-acceptance gates

| gate             | status         | measured        | threshold                            |
|:-----------------|:---------------|:----------------|:-------------------------------------|
| Data readiness   | PASS           | 1.0             | 0.995                                |
| Hard edit        | NOT_ASSESSABLE |                 | 0.99                                 |
| Expert edit      | NOT_ASSESSABLE |                 |                                      |
| Statistical rule | NOT_ASSESSABLE |                 | review yield ≥ 20%, shadow ≥ 30 days |
| Model            | FAIL           | 0 promoted of 2 |                                      |
| Production       | NOT_ASSESSABLE |                 |                                      |


Full evidence is in `release_gate_report.md`.


## 11. Reproducibility

This run is deterministic. Seed `20260920` (`cfg.random_seed`), parameter-registry fingerprint `9dd1afee3e5cccad`. Re-running `python -m fwa.run_validation --data data/uae_demo.zip --adapter uae_multitable --reports reports/uae_demo` regenerates every file in `reports/uae_demo/` identically, because signal identity is a pure function of (tenant, rule, version, subject, fact, period) and every model is seeded.

Stage timings for this run:

| stage                                          |   seconds |
|:-----------------------------------------------|----------:|
| adapter and canonical model                    |      4.59 |
| lineage and episodes                           |      1.47 |
| rule registry                                  |      0.43 |
| feature store                                  |      4.86 |
| peers, shrinkage and the transparent composite |      1.71 |
| graph and entity resolution                    |      7.76 |
| synthetic document corpus and NLP pipeline     |      0.45 |
| unsupervised models, SHAP, drift, clusters     |      3.9  |
| AI layer                                       |      0    |
| control evaluation                             |    113.02 |
| case correlation and priority                  |      0.21 |


### Stage latency against the stage ceilings

| stage          |   ceiling_ms |   observed_ms | within_ceiling   | note                                                                                                                                                                                                  |
|:---------------|-------------:|--------------:|:-----------------|:------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| INGEST         |         1000 |        1205.1 | False            | Observed is the BATCH time for this stage across 24,855 claims, not a per-claim online latency. A real PREPAY_SYNC deployment would be measured per claim; this artefact runs as a batch and says so. |
| PREPAY_SYNC    |          500 |       18452.5 | False            | Observed is the BATCH time for this stage across 24,855 claims, not a per-claim online latency. A real PREPAY_SYNC deployment would be measured per claim; this artefact runs as a batch and says so. |
| PREPAY_ASYNC   |       600000 |       25678.2 | True             | Observed is the BATCH time for this stage across 24,855 claims, not a per-claim online latency. A real PREPAY_SYNC deployment would be measured per claim; this artefact runs as a batch and says so. |
| POSTPAY_DAILY  |     86400000 |       46232.4 | True             | Observed is the BATCH time for this stage across 24,855 claims, not a per-claim online latency. A real PREPAY_SYNC deployment would be measured per claim; this artefact runs as a batch and says so. |
| NETWORK_WEEKLY |    604800000 |        2043.2 | True             | Observed is the BATCH time for this stage across 24,855 claims, not a per-claim online latency. A real PREPAY_SYNC deployment would be measured per claim; this artefact runs as a batch and says so. |
| MODEL_MONTHLY  |   2592000000 |       19393.8 | True             | Observed is the BATCH time for this stage across 24,855 claims, not a per-claim online latency. A real PREPAY_SYNC deployment would be measured per claim; this artefact runs as a batch and says so. |
