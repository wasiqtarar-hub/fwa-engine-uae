# uae-fwa-engine

A running, testable implementation of the fraud-, waste- and abuse-detection architecture designed
in *A Governed, Explainable Detection Architecture for Health-Insurance Payment Integrity in the
United Arab Emirates* (Wasiq Ahmed Tarar, M01092789, MSc Data Science & AI, Middlesex University).

The thesis specifies a system. This repository is that system, built and run.

```bash
pip install -e .
streamlit run app/Home.py          # the application
python -m fwa.run_validation       # the reproducible validation run
pytest                             # 740 tests
```

No API key. No network access. No external service of any kind.

---

## The one thing to read before anything else

> **A signal is not a fraud finding.**
>
> This system can establish non-payability, inconsistency or statistical abnormality. It cannot
> establish intent, and intent is what distinguishes fraud from waste, abuse or honest error. Only a
> human reviewer, on evidence, may reach a conclusion about conduct.

That sentence is the constraint everything else is built around, and in this repository it is a
piece of code rather than a paragraph. `src/fwa/engine/contract.py` refuses to construct a control of type `S`, `N`, `T` or `M`
that declares `REJECT` or `REPRICE`, and the refusal happens at **registration time**, before
anything runs. The engine cannot be started with such a rule in the catalogue. There is no override
flag, no `force=True` and no environment variable that relaxes it —
`tests/governance/test_disposition_legality.py` asserts all of that, including by reading the
validator's own source for an escape hatch.

The consequences run through everything else:

- Only **21 of 164** controls may change what is paid, and every one of them is type `H` or `H/E`.
- An anomaly score, a graph metric, an NLP extraction and an LLM sentence can all raise a signal.
  None of them can set or change a disposition.
- **Priority orders the review queue. It never substitutes for a disposition.**
- **Gross flagged value is not savings.** A model-only lead renders its gross amount beside the
  literal string `exposure not yet established`, and that figure is never added into any savings
  total.
- Signals resting on the same underlying fact contribute `max(evidence_strength)`, never the sum.
  Three rules firing because one amount is high is **one** piece of evidence.

---

## What this is not

It is worth being blunt, because an artefact that oversells itself fails the thesis it belongs to.

**This is not a validated fraud detector.** Nothing in it has been reviewed by a human, so
precision in the strict sense — *confirmed ÷ reviewed* — is not measurable and is reported as
`NOT_MEASURABLE_ON_THIS_DATASET` rather than replaced with something that looks like it.

**Every figure is a property of one input file, and this project generated that file.**
`data/claims_demo_synthetic.csv` is a 20,893-row synthetic claim-header extract produced by
`tools/make_demo_dataset.py`, in a non-AED source currency converted to AED through an
effective-dated `config/fx.yaml`. It exists because large parts of the engine — the change-point
detectors, the duplicate and capacity controls, the novel-cluster discovery, and the promotion gate
*in its passing state* — cannot be exercised by an arbitrary extract. **A gate that passes on a
dataset built to pass it is a test of the gate, not evidence about detection performance.**
`docs/DEMO_DATASET.md` says so at the top and sets out what had to be engineered and why. Load your
own data on the **Data** page and the whole engine re-runs on it — re-measure there before treating
any number here as a forecast of what it will do elsewhere.

**The two UAE adapters are not certified.** `ShafafiyaAdapter` and `EClaimLinkAdapter` are
schema-complete against the published transaction constructs and are exercised by tests, but they
have never seen a live regulator feed. Both carry `NOT CERTIFIED — no live regulator feed` in the
code and in their output.

**The models are not promoted.** Both unsupervised models fail the promotion gate and stay in
shadow. That is a result, not a bug: they do not beat the transparent composite baseline at the
capacities that matter, and the gate says so with its evidence.

**No supervised classifier is trained on the labels.** `models/supervised_gate.py` exists to
explain *why not*, and it returns `PROMOTION_BLOCKED` every time. It trains nothing —
`tests/governance/test_label_leakage.py` proves it by parsing every module in the package for a
`.fit(X, y)`.

**No control is active.** All 164 are in `shadow`. Activation requires a prospective shadow period
and a policy owner who did not author the rule; neither exists in a single retrospective run.

---

## What it does

| Layer | What it is | Where |
|---|---|---|
| Canonical model | 16 tables plus `review_outcome`, each POPULATED / PARTIAL / NOT_POPULATED with a reason | `src/fwa/canonical/` |
| Adapters | Three source adapters behind one interface; SHA-256 of the raw record *before* parsing | `src/fwa/canonical/adapters.py` |
| Rule catalogue | All 39 scenarios and all 164 atomic controls of the control catalogue as governed YAML | `rules/` |
| Engine | The contract, the legality validator, the registry, the evaluator | `src/fwa/engine/` |
| Statistics | Six-level peer hierarchy with back-off, empirical-Bayes shrinkage with posterior intervals, robust MAD, CUSUM/EWMA, the ANL-01-R01 transparent composite | `src/fwa/statistical/` |
| Models | Isolation Forest + LOF, temporal split with entity isolation, SHAP in original units, drift, HDBSCAN novel clusters, the promotion gate | `src/fwa/models/` |
| Graph | Bipartite claim graph, Louvain communities, entity-resolution candidates | `src/fwa/graph/` |
| NLP | A clearly-labelled SYNTHETIC EN/AR document corpus, extraction with mandatory source spans, OCR-degraded confidence | `src/fwa/nlp/` |
| AI | Offline deterministic provider by default, groundedness validation, NL rule authoring, a copilot with a hard-coded refusal | `src/fwa/ai/` |
| Cases | Six-dimension correlation with a deterministic fingerprint, the priority formula, conservative exposure by signal type | `src/fwa/cases/` |
| Evaluation | Every metric (or an honest `NOT_MEASURABLE`), the nine-procedure protocol, the six release gates, five reports | `src/fwa/evaluation/` |
| Application | A 13-page, login-protected, role-aware Streamlit app | `app/` |

### Running it on your own data

The **Data** page takes a claim file — dragged in, or named by its path on the machine the app is
running on — and re-runs the pipeline end to end on it. That means the adapter, the feature store,
all 164 controls, the peer groups and their fitted Beta priors, the graph, the unsupervised models
and the promotion gate. Nothing carries over from the previous dataset except the rule catalogue
and the parameter registry, which are governance artefacts rather than properties of the data.

Runs are cached on the **content hash** of the input file, so switching back to a dataset you
loaded earlier is free, and the page can report what changed between two of them: which controls
became runnable, which stopped being runnable, and which found a different number of signals. That
comparison is usually more informative than the signal counts themselves — a file carrying an
authorisation table brings a whole scenario family to life, and a file without `agent_id` kills the
distribution-channel controls outright.

A column the file does not have is not an error. It becomes an empty canonical field, the absence
is recorded and named on screen, and the controls that needed it are classified
`NOT_EXECUTABLE_ON_THIS_DATASET` rather than run against a substitute. A file with no claim, member
or provider identifier is refused, with a message naming the columns that are missing and the
columns that were found instead.

The **Parameters & Models** page is the other half of the same idea: it separates the numbers that
were *measured from this file* — robust medians and MADs, fitted Beta priors, model thresholds,
the split the models were trained under — from the numbers that are *governance decisions* with an
owner, an approver, effective dates and a written rationale. Confusing the two is the most
expensive mistake a reader of this system can make, so the interface keeps them on separate tabs.

### What the validation run produces

`python -m fwa.run_validation` writes 15 files to `reports/`, five of which are the required
deliverables:

- `validation_report.md` (and `.html`) — what ran, what fired, what it means and what it does not
- `control_coverage_matrix.csv` — all 164 controls with `data_support` and a reason
- `traceability_matrix.csv` — requirement → module → test → gate
- `model_card.md` — what the models are, what they are not, why they are in shadow
- `limitations.md` — the honest account of everything this build cannot support

The run prints its seed and its parameter-registry fingerprint. Re-running on the same input
reproduces every **finding** exactly — the same signals, case identities, dispositions, exposures
and priorities — and `tests/integration/test_pipeline.py` asserts that against the generated files.
What does not reproduce byte-for-byte is the measurement of the run itself: per-control
`elapsed_ms`, each case's `opened_at`, and the generation timestamp. Those are wall-clock readings,
and the stage latency ceilings are checked against them, so reproducing them exactly would mean
not measuring them.

---

## Getting in

`streamlit run app/Home.py` seeds one demo account per role on first start and prints the
credentials **once**. Every seeded account is forced to change its password at first sign-in.

| Username | Role | What they see |
|---|---|---|
| `admin` | ADMIN | Everything, plus data loading, user management, kill switch and rollback |
| `policy` | POLICY_OWNER | Rule registry, parameter governance, rule authoring |
| `reviewer` | CLAIMS_REVIEWER | Review queue and case evidence; identities masked |
| `clinical` | CLINICAL_REVIEWER | Clinical dispositions |
| `siu` | SIU_INVESTIGATOR | Network and case linking; identities unmasked **on assigned cases only** |
| `analyst` | ANALYST | Data loading, parameters, provider analytics, models, validation report |
| `qa` | QA | Test evidence and release gates |
| `auditor` | AUDITOR | Read-only across the system, including the audit log |
| `reviewer2` | CLAIMS_REVIEWER (tenant T002) | The second tenant, for the isolation demonstration |

PHI is masked by default. A reviewer sees `PROV-3F2A91C4`, not `H0141` — a **stable pseudonym**, so
two providers stay distinguishable and a case can still be discussed. Unmasking is a separate,
reason-required, audited action. Manual override is a distinct permission held by **no role at all**,
including ADMIN; it is granted to an individual, and it writes a different audit event from an
ordinary disposition.

> The authentication here is real (Argon2id, lockout, forced rotation, session timeout, an
> append-only hash-chained audit log) but it has not been hardened for live PHI. It demonstrates the
> control model; it is not a deployment posture.

---

## Where the catalogue and the code disagree, and why

Five controls carry a `governance_note` recording a disposition that differs from the catalogue's
first-named one. Every one is visible in the Rule Registry page and in
`control_coverage_matrix.csv`, because a silent downgrade would be exactly the kind of undocumented
drift the thesis argues against.

| Control | Catalogue says | Governed as | Why |
|---|---|---|---|
| `PHR-05-R03` | Reprice/audit | `POSTPAY_AUDIT` | ** forces it.** The control is type `E/S`; a control that is part statistical may never reprice. The catalogue's second option is taken. This is the only control in the catalogue whose type forces the change. |
| `CLN-03-R01` | Reject/pend | `PREPAY_PEND` | The trigger is an upstream boolean verdict, not an objective condition this system can re-derive. |
| `ENT-03-R02` | Reject or SIU lead | `PREPAY_PEND` | The exclusion source carries no effective date, and the safety boundary requires an *effective-dated* condition to deny. |
| `PAY-01-R03` | Reprice/pend episode | `PREPAY_PEND` | The correct payable amount is not computable without claim lines, and defines REPRICE as the case where it is. |
| `CLN-05-R03` | Reprice/pend | `PREPAY_PEND` | Repricing requires a payment policy this dataset does not contain. |

**One discrepancy worth recording.** PAY-07 is sometimes described as an "approval-ratio
anomaly". The control catalogue defines it as patient-share, copay, deductible and balance billing,
and places the approval-ratio control at **PAY-06-R04**. The catalogue is the normative source, so
it is followed and the approval-ratio control is implemented where the catalogue puts it. PAY-07's four controls are classified `NOT_EXECUTABLE_ON_THIS_DATASET` because `patient_share`
is absent from the claim extract.

---

## Findings worth reading

These came out of building it, and they are in `reports/limitations.md` in full.

1. **127 of 164 controls cannot run on this file** — not because the design is wrong, but because a
   claim-header-only extract carries neither activity codes nor authorizations nor remittances. Each
   one names the canonical fields that would unlock it, which makes the coverage matrix a concrete
   answer to *what would real Shafafiya data buy?*
2. **The labels are investigation-derived, so every precision figure is an upper bound.** They
   come from three processes (`pattern_detection`, `expert_review`, `rule_engine`), and a
   rule-based detector scoring well against rule-derived labels is close to tautological. Precision
   is reported as a *proxy*, stratified by `ground_truth_source` so the reader can see which
   labels produced which figure.
3. **One control could not fire on any dataset, and nothing revealed it until now.** DOC-02-R01's
   boilerplate stripper compiled its template patterns with `re.DOTALL`, so `DISCHARGE SUMMARY.*`
   ran to the end of every document and every document stripped to the empty string. A control that
   finds nothing looks exactly like a control with nothing to find. Fixed in
   `src/fwa/nlp/controls.py`, with regression tests that assert on the stripped text rather than on
   the signal count.
4. **The priority formula saturates.** The coefficients are reproduced exactly and not tuned;
   with six-figure exposures the exposure term alone can push the sigmoid above 99. The UI shows the
   term-by-term breakdown and the raw logit so the saturation is visible. A real deployment would
   set `priority_exposure_scale_aed` to the scale of its own exposures — it is a governed parameter,
   and the fact that it needs setting is itself a finding.
5. **The promotion gate's calibration criterion cannot pass two unsupervised models at once.**
   `local_outlier_factor` clears all five criteria and is promoted; `isolation_forest` clears four,
   including the higher lift, and fails calibration alone — for a reason that is provable from the
   criterion's own arithmetic rather than a property of any dataset. Published as a result rather
   than tuned away; `docs/DEMO_DATASET.md` gives the proof and the two ways out, both decisions
   rather than fixes.

---

## The dataset

`data/claims_demo_synthetic.csv` (20,893 rows, 1,340 providers, 12,651 members,
five whole years, seed 20260920) is generated by this project to exercise the
engine end to end. On it **all 37 runnable controls fire and none runs silently**,
producing 9,074 signals in 8,502 cases; the maximum feature PSI is 0.18, a warning
rather than a breach; and `local_outlier_factor` clears every promotion-gate
criterion.

A gate passed on a dataset built to pass it is a test of the gate, not evidence
about detection performance — `docs/DEMO_DATASET.md` says so at the top, along
with what had to be engineered and why. Four things keep the exercise honest: the
file names itself `_synthetic` and is banner-labelled everywhere; the promotion
thresholds are unchanged from the specification; the labels are verified
bit-identically held out of every feature path by `tools/verify_demo_dataset.py`;
and the one criterion the dataset could not satisfy is reported as a failure and
analysed, not relaxed.

```bash
python tools/make_demo_dataset.py --out data/claims_demo_synthetic.csv --calibrate
python tools/verify_demo_dataset.py data/claims_demo_synthetic.csv
```

## Light and dark

The interface ships light: white ground, a serif for display headings, a plum
accent. To switch it to dark, open `.streamlit/config.toml`, comment out the
`LIGHT` theme block, uncomment the `DARK` one and restart. Nothing else changes.

That one file is the single source of truth on purpose. Streamlit writes no
marker into the DOM saying which theme is active, so the obvious thing for a
stylesheet to do is ask the operating system — `@media (prefers-color-scheme:
dark)`. That is a different question from *what theme is this application
running*, and the two come apart the moment a base is pinned here: the
stylesheet paints a dark background while Streamlit, still on its light theme,
paints its own headings near-black, and the page becomes dark text on a dark
ground for anyone whose desktop is set the other way from the app. So nothing
in `app/assets/` asks the OS anything. `app.common.active_theme` reads the
configured base and `app.common._stylesheet` loads `style.css`, plus
`style-dark.css` when that base is dark — one decision, both halves.

## Repository layout

```
config/       parameters.yaml (77 governed parameters), peer_groups.yaml, fx.yaml
rules/        39 scenario files, 164 atomic controls
src/fwa/      the package
app/          the Streamlit application (13 pages + login)
tests/        740 tests: unit, integration, governance
tools/        the catalogue source, the rule generator, the screenshot driver
reports/      output of the validation run
docs/         USER_GUIDE.md, DEFINITION_OF_DONE.md and the screenshots in them
data/         claims_demo_synthetic.csv, its construction sidecars, and the
              generated synthetic document corpus
```

## Tests

```bash
pytest                          # everything
pytest -m governance            # the boundary tests alone
pytest -m "not slow"            # skip the full-pipeline integration tests
pytest --cov=fwa.engine --cov=fwa.statistical --cov=fwa.cases --cov=fwa.evaluation
```

Coverage on those four packages is **91%** against a required 85%, enforced in `pyproject.toml`.

`docs/DEFINITION_OF_DONE.md` walks the acceptance checklist item by item, records how each
one was verified against a clean copy of the repository in a fresh virtual environment with no API
keys set, and ends with the three things this build deliberately does **not** do and why.

The ten-category suite in `tests/governance/test_ten_category_suite.py` runs all ten categories
against **every implemented control, uniformly, regardless of type** — an Isolation Forest control
is held to the same idempotency and tenant-isolation standard as a duplicate edit. It runs against a
stratified sample of the real file rather than hand-built fixtures, because a fixture built to make
a control fire only tells you that the control fires on the fixture.

## Docker

```bash
docker compose up               # the app on http://localhost:8501
docker compose run --rm validate  # the validation run, writing into ./reports
docker compose run --rm tests     # the suite
```

## Licence and provenance

The design, the control catalogue and the governance model come from the specification this was
built to. The implementation, the calibrations and the findings above are this repository's, and
every calibration is recorded with its rationale, owner and approval in `config/parameters.yaml` —
no threshold anywhere in the code is a literal.
