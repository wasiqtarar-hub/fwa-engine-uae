# Definition of done — design specification , checked

Each item below was verified against a **clean copy** of this repository: the tree copied without
`reports/`, `data/fwa.db`, `data/synthetic_documents/`, `__pycache__/`, `.pytest_cache/` or any
`*.egg-info`, installed into a fresh virtual environment, and run with `ANTHROPIC_API_KEY` and
`OPENAI_API_KEY` unset.

---

### 1. `pip install -e . && pytest && streamlit run app/Home.py` works from a clean clone, offline, with no API key

**Verified.** In the clean room: `pip install -e ".[dev]"` succeeded, `pytest` ran **740 tests, all
passing**, and the application starts and serves.

The offline property is asserted rather than assumed. `tests/governance/test_ai_disabled.py` parses
every module in `src/fwa/` and fails the build if any of them imports `requests`, `httpx`,
`urllib`, `socket`, `http` or `aiohttp` anywhere, or imports `anthropic` or `openai` at module
scope. It also checks that the default provider needs no key, that a hosted provider without a key
degrades to the offline one rather than crashing, and that no API key is present in the test
environment at all. `docker-compose.yml` runs the suite and the validation under
`network_mode: none`, which is the same claim made by a container with no network interface.

### 2. `python -m fwa.run_validation --data data/claims_demo_synthetic.csv` regenerates every file in `reports/` deterministically

**Verified, with the claim stated precisely.** The run writes 15 files and prints its seed
(`20260920`) and parameter-registry fingerprint (`cc7dd3640521a132`).

Two consecutive runs in the clean room produce **identical findings**: the same signals, the same
case identities, the same dispositions, exposures and priorities, and identical metric,
coverage-matrix and queue content. `tests/integration/test_pipeline.py::
test_the_reports_reproduce_every_finding_across_two_runs` runs the pipeline twice and compares six
generated CSVs column by column.

What does **not** reproduce byte-for-byte is the measurement of the run itself: per-control
`elapsed_ms`, each case's `opened_at`, and the generation timestamp. Those are wall-clock readings,
and 's latency ceilings are checked against them — reproducing them exactly would mean not
measuring them. The tool's footer says exactly this rather than the easier sentence, and
`test_the_run_does_not_claim_more_reproducibility_than_it_has` asserts the easier sentence is not
there.

### 3. The disposition-legality, label-leakage, evidence-capping, idempotency, tenant-isolation and AI-disabled tests all pass

**Verified.** `pytest -m governance` — 207 tests, all passing, across ten files:

| File | What it asserts |
|---|---|
| `test_disposition_legality.py` | A control of type S/N/T/M declaring REJECT or REPRICE cannot be constructed, cannot be registered, and cannot be loaded from YAML in either strict or non-strict mode. The validator's own source is scanned for an escape hatch. |
| `test_label_leakage.py` | The four label columns cannot enter a `FeatureFrame` by any route, are absent from every canonical table, and are referenced nowhere outside `fwa/evaluation/`. No module anywhere calls `.fit(X, y)`. |
| `test_evidence_capping.py` | N signals on one fact contribute `max`, never the sum — including on two shipped controls that deliberately share a `fact_key`. |
| `test_ten_category_suite.py` | The ten categories, including idempotency, against **every** implemented control, uniformly, regardless of type. |
| `test_tenant_isolation.py` | Every read path — signal store, cases, queue, audit log, session — is tenant-filtered, and the same fact in two tenants is two distinct signals. |
| `test_ai_disabled.py` | Every AI feature switches independently; cases, dispositions, exposures and priorities are identical with the layer present and absent; no network import exists. |
| `test_copilot_refusal.py` | Sixteen phrasings of a conduct question are refused **before any provider is called**, proven with a provider that raises on use. |
| `test_ai_cannot_change_disposition.py` | A deep-copied case is byte-identical before and after narration, copilot and triage; no module in `fwa/ai/` assigns to a disposition, priority or exposure. |
| `test_parameter_governance.py` | Every parameter carries its metadata; the engine, statistical and cases packages contain no bare numeric comparison. |
| `test_catalogue_completeness.py` | 39 scenarios, 164 controls, honest `data_support`, and no dishonest classification in either direction. |

### 4. All 39 scenarios exist as YAML with honest `data_support`; the coverage matrix accounts for all 164 controls

**Verified.** 39 scenario files in `rules/`, 164 controls, 164 rows in
`reports/control_coverage_matrix.csv`. Classification: 9 EXECUTABLE, 28 PARTIAL, 127
NOT_EXECUTABLE_ON_THIS_DATASET.

Honesty is tested in both directions: a control classified NOT_EXECUTABLE may not name an
implementation, a control classified EXECUTABLE or PARTIAL must name one that exists in
`CONTROL_IMPLEMENTATIONS`, every classification must carry a reason of substance, and every
unrunnable control must name the canonical fields that would unlock it. There are no orphan
implementations — no code that no catalogue entry can reach.

### 5. Every number in every report is measured, labelled as configuration, or `NOT_MEASURABLE_ON_THIS_DATASET`

**Verified.** `tests/integration/test_metrics_and_gates.py` asserts that every metric carries one of
exactly two statuses, that every unmeasurable one names the data that would supply it, that no
measured metric is a savings figure, and that the six metrics this dataset cannot support —
confirmed AED, prevented/recovered AED, net savings, abrasion, turnaround, overturn rate — are all
present and all honest.

Thresholds quoted in the release-gate report are labelled *"a configuration acceptance criterion
quoted from the specification, not a result achieved here"*. Gross flagged value is reported in two
separate figures, established and not-established, with a note on each that they are never summed.

### 6. Every page is behind the access guard; separation-of-duties, PHI-masking and audit tests pass; no page is reachable by URL without authorisation

**Verified.** All 11 page modules call `require_page(...)` as their first statement. Navigation is
*built* from the signed-in role's permissions, so a page a role may not see does not appear; the
guard re-checks on entry, so a bookmarked or forged reference fails closed and writes an
`ACCESS_DENIED` entry to the audit log.

`tests/unit/test_rbac.py` (35 tests) covers the eight roles, the page gates, the stable-pseudonym
masking, and the rule that `MANUAL_OVERRIDE` is held by **no role including ADMIN**.
`tests/unit/test_review_outcome.py` (24 tests) covers login, lockout, forced rotation, the mandatory
rationale, the override path, the reason-required audited unmask, appeals, and the separation of
duties on model promotion. `tests/unit/test_audit_log.py` (23 tests) covers the hash chain, tamper
detection and the append-only refusals.

### 7. `docs/USER_GUIDE.md` exists, covers all roles, and its screenshots come from a real run

**Verified.** The guide covers the interface page by page — purpose and audience, the
roles, configuration governance, rollback and kill switch, drift, exceptions — and is rendered
in-app on the Help page from the same file, so the two cannot drift apart.

Its 11 screenshots were captured by `tools/capture_screenshots.py` from a running instance, signed
in with the real seeded credentials for each role. The script navigates by **clicking sidebar
links** rather than by URL, because a fresh URL load would start a new Streamlit session and
produce a screenshot of the login page labelled as the page it is not. It waits for the pipeline
and every spinner to settle before capturing. `docs/screenshots/MANIFEST.md` records which role
captured which page.

### 8. A new user can log in, find a case and record a defensible disposition without reading anything first

**Verified by construction, and the reader should judge it from the screenshots.**

The Review Queue carries a **"Walk me through a case"** toggle that annotates the path — queue,
case, evidence, AI panel, decision — in five steps. Without it: the queue's columns are in plain
language, the case card answers *what fired*, *what it's worth* and *what is being asked of you* in
that order, the disposition form is the one visually primary action on the screen, and every field
in it carries help text explaining what it is for. The rationale field's placeholder is the
question it wants answered: *"What did you check, what did you find, and why does that settle it?"*

---

## What is deliberately not done

Three things a reader might expect to find, and the reason each is absent:

**No control is active.** Activation requires a prospective shadow period and a policy owner who
did not author the rule. Neither exists in a single retrospective run, so all 164 controls are in
shadow and the release-gate report says why.

**Neither model is promoted.** Both fail the gate on calibration and drift, and neither can be
assessed for lift because fewer entities carry an outcome than the review capacity requires. The
gate reports `NOT_ASSESSABLE` where it cannot judge, and `NOT_ASSESSABLE` is never a pass.

**The production release gate does not pass.** Four of its five elements are software capabilities
and are demonstrated — rollback, kill switch, alert ceiling, hash-chained audit log. The fifth, a
named responsible owner *on call*, is a rota rather than a capability, and no artefact can evidence
it. The gate is therefore reported `NOT_ASSESSABLE` rather than passed, because passing it would
report this build as meeting a production standard it cannot meet.
