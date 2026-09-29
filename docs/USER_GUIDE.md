# User guide — UAE FWA Engine

This guide is for the people who use the tool day to day: claims reviewers working through a
queue, audit managers deciding what to follow up, and examiners checking what the tool does and
does not claim. You need to know health insurance. You do not need to know statistics or
programming.

The same guide is built into the app: open **Help** in the sidebar. The Help page also has a
searchable glossary of every technical term the tool uses.

---

## Start here

1. **Choose or load data.** Open **Data**. The UAE demo file is already loaded when you sign in,
   so you can skip this step the first time.
2. **Review the most urgent cases.** Open **Review queue**. The most urgent cases are at the top.
3. **Read why they were flagged.** Pick a case and open **Case evidence**. It starts with one
   sentence, then the reasons in order, how strong the evidence is, and what to check next.

---

## The one rule: a reason to look, not proof

Everything in this tool is built around one rule:

> **A flag is a reason to look, not proof of fraud.** The tool can show that a claim breaks a
> dated rule, does not add up, or is unusual compared with similar claims. It cannot show
> intent, and intent is what separates fraud from waste, abuse or an honest mistake. Only a
> reviewer, looking at the evidence, can decide.

In practice this means:

- **Only clear-cut rules can lead to a claim being denied.** For example, "the patient had no
  cover on the date of treatment" can. "This hospital bills more than similar hospitals" cannot,
  however unusual it is. Only 21 of the 164 checks are allowed to suggest denying or repricing a
  claim, and the tool refuses to start if any other check tries.
- **Urgency only decides the order of the queue.** It never changes what should happen to a
  claim.
- **Money flagged is not money saved.** Nothing has been reviewed, confirmed, stopped or
  recovered, so flagged amounts are never added up as savings.
- **Two kinds of amount are always kept apart.** An *established* amount rests on a clear rule
  that failed. An amount *not yet established* is the full value of a claim that looks unusual.
  The two are shown side by side and never added together.
- **Everything runs in watch-only mode.** Every check and both pattern-finding models record and
  measure their results, but none is used for real payment decisions yet.
- **Patient and provider identities are hidden by default.** Only roles that are allowed can
  reveal them, and each reveal needs a reason and is logged.

---

## How do I find the most urgent cases?

Open **Review queue**. Cases are listed most urgent first, with a coloured urgency label
(Urgent, High, Medium, Low, Watch only) and what the rules suggest should happen next, such as
"Hold before paying" or "Pay, but check afterwards".

- To see only the most urgent cases, raise **"Show only cases at least this urgent"**.
- To work on one kind of problem, for example only medicines or only coding, use the
  **"Kind of pattern"** filter.
- The sliders for **team size** and **cases per reviewer per day** tell you how long the whole
  queue would take your team. They do not change the cases or their order.

The **Overview** page also lists the top cases to look at first, each with its one-line summary.

---

## How do I read why a claim was flagged?

Open a case on **Case evidence**. Every explanation has the same six parts, in the same order:

1. **Headline.** One sentence: who, what, how much, and what we suggest. For example, "Hospital
   H1023 billed AED 815,607 across 80 claims in November 2025, with three unusual patterns. We
   suggest checking it after payment."
2. **Reasons**, most important first, at most five. Each reason is a sentence with the actual
   fact and a comparison, such as "It charged AED 2,649 per day of hospital stay. Similar
   hospitals charge about AED 912 per day, so this is roughly 3 times higher."
3. **How strong the evidence is:** Strong, Moderate or Weak, with one line on why. When several
   reasons rest on the same underlying fact, the explanation says so and they count once. Three
   checks noticing the same high amount are one piece of evidence, not three.
4. **What this does not mean.** Always: "This is a reason to look, not proof of fraud." When only
   a pattern-finding model flagged the claim, it also says that no rule was broken and the amount
   at risk has not been established.
5. **What to check next.** Two to four concrete steps, such as "Ask for the discharge summary"
   or "Compare the invoice with the pharmacy dispensing record."
6. **Technical details**, folded away, for auditors and examiners: check ids, scores,
   thresholds, the model's reasons in numbers, and field names.

You can download a **one-page summary** of the case (printable HTML, or plain text) from the same
screen.

---

## How do I record a decision?

At the bottom of **Case evidence**:

1. Choose what you decided: confirm what the rules suggest, change it, clear the case (no issue),
   or refer it to the investigations team.
2. Pick the kind of issue you found, if any, and the amount you actually confirmed. The amount
   you confirm is often less than the amount flagged; that difference is exactly what the tool
   needs to learn from.
3. Write a short reason: what you checked, what you found, and why that settles it. A decision
   without a reason is refused.
4. Press **Submit decision**.

Every decision goes into a permanent audit log that records who decided what, when, and why.
Changing what the rules suggested is a separate permission; not every role has it.

---

## How do I load my own data?

Open **Data** (available to analysts and administrators).

- **To use a demo file**, choose "A demo file that comes with the app", pick one, and press
  **Use this file**.
- **To upload a file**, choose "Upload a file from my computer". Upload either a single **.csv**
  with one row per claim, or a **.zip** holding several tables (claims, service lines,
  providers, patients and so on).
- **For a large file or one on a shared drive**, choose "A file or folder already on this
  machine" and type its full path. The file is read where it is and never copied.

Leave **File format** on "Work it out from the file" unless your file comes from a regulator's
system. A .zip or folder is read as UAE multi-table; a single .csv as a generic claim list. The
Shafafiya and eClaimLink layouts are available but have not been tested against a live
regulator feed.

Loading a file re-runs everything: all 164 checks, the comparisons with similar providers, the
connections, and both pattern-finding models. Every page then describes the new file. If the
file cannot be read, the page says what is wrong and how to fix it, and the file in use before
stays in use. Files you used earlier in the session open again instantly.

---

## What does 'watch only' mean?

"Watch only" appears in two places, and both mean the same thing: **recorded and measured, but
not acted on yet.**

- **For a case**, "Watch only" means there is nothing to act on yet. The pattern is being
  watched, and the case sits at the bottom of the queue.
- **For a check or a model**, watch-only (also called shadow mode) means it runs and its results
  are kept and measured, but they are not used for real payment decisions. Every check in this
  tool starts in watch-only mode. Switching one on for real use needs a policy owner who did not
  write it, and a period of watching to show it works.

A check that raises far more flags than a team could handle is also moved to watch-only
automatically, so it cannot flood the queue.

---

## Why couldn't some checks run?

Each check needs certain information. Some need only the claim summary (who, when, how much);
many need more, such as line-by-line service details, prior approvals, payment records,
provider licences or prescriptions.

The **Data** page tells you, in one sentence, how many checks ran fully, how many ran in a
simplified form, and how many could not run. It then groups the checks that could not run by
what is missing, for example "needs line-level service details (41 checks)".

- **Ran fully:** the file had everything the check needs.
- **Ran in a simplified form:** the file had only a stand-in for part of what the check needs.
  Its flags say so, and its evidence counts for less.
- **Could not run:** the information is missing, so the check was not run rather than guessed.
  That is a fact about the file, not a failure of the check.

On the older one-row-per-claim demo file, only 9 checks run fully, 28 in a simplified form, and
127 cannot run. The UAE demo file carries the extra tables: at the time of writing, 163 of the 164
checks run on it (77 fully and 86 in a simplified form). The one that cannot run needs reviewers'
recorded decisions, which no file can supply until reviewers have worked the queue.

---

## What do the Simple and Advanced views show?

The **Simple view** switch is in the sidebar. It is on by default and the app remembers your
choice.

- **Simple view** shows the pages you need to review cases: **Overview**, **Review queue**,
  **Case evidence**, **Data** and **Help**.
- **Advanced view** adds the analysis and governance pages: **Pattern-finding models**,
  **Compare hospitals**, **Connections between providers, agents and patients**, **Settings and
  thresholds**, **Checks library**, **Validation report** and **Governance**.

The switch never changes what you are allowed to see. Your role decides that; the switch only
decides how many of your pages are listed. On every page, technical detail stays one click away
in sections titled "Technical details" or behind a "Show all columns" tick box.

---

## Can I trust the pattern-finding models?

Not for real decisions yet, and the tool says so. Two automated pattern-finders rank how unusual
each claim is:

- **Isolation Forest** looks for claims that are easy to separate from the crowd.
- **Local Outlier Factor** looks for claims that are unusual compared with their nearest
  neighbours.

To keep the test fair, each model learns from earlier claims and is then tested on later claims
from providers it has never seen. That is why many claims get no score.

Before a model may be used, it must pass a five-question checklist (the promotion gate): Was it
tested fairly? Does it find more real issues than the simple scorecard? When it sounds sure, is
it right that often? Are the claims it sees now like the ones it learned from? Can it explain
each flag in real units? A question that cannot be answered counts as a "no". Even when every
answer is "yes", a second person who did not propose the model must approve it. The
**Pattern-finding models** page shows the checklist with a plain answer to each question.

---

## How do I tell whether an amount is money at risk or money saved?

It is never money saved. Look for one of two labels:

- **Established:** a clear rule failed, for example no cover on the service date, so the amount
  at risk rests on a fact a reviewer can check.
- **Not yet established:** the claim looks unusual, and the figure is simply its full value. It
  is not a loss.

The two are always shown separately and never added together. Measures such as money confirmed,
money recovered and net savings need reviewers' recorded decisions; until those exist, the tool
says "can't be measured yet" instead of showing a number.

---

## The demo files

Three files come with the app. The **Data** page describes each one and lets you switch.

| File | What it is | When to use it |
|---|---|---|
| **UAE demo claims (multi-table)**, `uae_demo.zip` — **SYNTHETIC**, the default | 24,855 claims over eighteen months (July 2024 to December 2025) of generated claims from UAE-style hospitals, clinics, pharmacies, labs and radiology centres, in AED, with the supporting tables most checks need: service lines, diagnoses, visits, prior approvals, payments, resubmissions, provider licences, clinicians, contracts, patients and cover, prescriptions and dispensing, referrals, documents, complaints and reference tables. At least one realistic pattern is planted for every kind of problem the checks look for. | Start here. Most checks and both models can run on it. |
| **Earlier demo claims**, `claims_demo_synthetic.csv` — **SYNTHETIC** | 20,893 generated claims over five years, one row per claim, in Indian rupees converted to AED. | To see how much less the engine can do with claim summaries only. |
| **Sample claim extract**, `claims.csv` | A 5,000-claim sample in the same one-row-per-claim layout, in Indian rupees converted to AED. | To compare results on a small file in the simple layout. |

**About synthetic data.** The two demo files were generated by a program; no row describes a
real person, provider or claim. They are labelled SYNTHETIC wherever their numbers appear. The
record of which patterns were planted where (the answer key) is kept in a separate file that the
detection code never reads. **A check that passes on data built to pass it is a test of the
check, not evidence that it detects fraud.** No figure from these files says anything about how
well the tool would do on real claims.

For reference, on the earlier demo file the tool raised 9,074 flags, grouped into 8,502 cases,
with AED 61,838,054 established and, separately, AED 17,028,427 not yet established. None of it
has been reviewed.

---

## Page by page

| Page | What it is for |
|---|---|
| **Overview** | A one-screen summary of the file: how many cases need review, how much is at risk, and which cases to open first. |
| **Data** | Choose the claim file, load your own, and see which checks could run and what was missing. |
| **Review queue** | Your list of cases, most urgent first, with filters and how long the queue would take your team. |
| **Case evidence** | One case: why it was flagged, how strong the evidence is, what to check next, and where to record your decision. |
| **Compare hospitals** | How each provider compares with similar providers and with its own past. |
| **Connections** | Who is connected to whom through shared patients, referrals and sales agents. |
| **Pattern-finding models** | What the two models found, why, and whether they are ready to be trusted (so far they are not). |
| **Checks library** | All 164 checks: what each looks for, why it matters, and whether it could run on this file. |
| **Settings and thresholds** | Every number the engine uses, split into numbers measured from your data and numbers chosen by policy with a named owner. |
| **Validation report** | A one-screen plain summary of the run, then the full technical report and its files. |
| **Governance** | The audit log, who owns each setting, switching a check off, and whether checks stayed within their expected workload. |
| **Users and access** | Creating users, changing roles and assigning cases (administrators only). |
| **Help** | This guide, a page-by-page list, what each role sees, and the searchable glossary. |

![Overview, signed in as admin](screenshots/overview-admin.png)

*The Overview page.*

![The review queue, signed in as reviewer](screenshots/review-queue-reviewer.png)

*The Review queue, as a claims reviewer sees it. Identities are hidden by default.*

![A case, signed in as reviewer](screenshots/case-evidence-reviewer.png)

*One case: the headline first, then the reasons, the evidence strength, and the decision.*

---

## Who sees what

What you can open depends on your role. Pages you may not open are not listed and cannot be
reached by link.

| Role | Sees |
|---|---|
| Administrator | Everything, including real names and IDs; manages users, access and the emergency switch-off for checks. |
| Policy owner | The checks library, settings and thresholds; drafts, edits and approves checks. |
| Claims reviewer | The review queue and case evidence, with identities hidden by default; records decisions. |
| Clinical reviewer | Clinical cases, documents and coding evidence; records clinical decisions. |
| Investigator (special investigations) | Cases referred to investigations and the connections view; sees real identities only on cases assigned to them. |
| Analyst | Data loading, pattern-finding models, hospital comparisons, settings and the validation report. |
| Quality assurance | Test evidence and the release checks. |
| Auditor | Read-only access to everything, including the audit log. |

Some rules are enforced by the tool itself: the person who writes a check cannot approve it for
real use, and the person who proposes a model cannot approve it either.

---

## Signing in

The tool runs on your own computer with **no internet connection and no API key**; nothing in it
calls out. On first start it creates one demo account per role (reviewer, clinical, siu, policy,
admin, analyst, qa, auditor) and shows their passwords once. Every demo account must change its
password at first sign-in.

The sign-in and access controls demonstrate how access would work. They are **not hardened for
real patient data**.

To switch the app between light and dark, an administrator changes the theme in
`.streamlit/config.toml` and restarts the app.

---

## Glossary

The full glossary, with an everyday comparison for each term, is on the **Help** page and can be
searched. The terms you will meet most often:

- **Flag (signal):** one check's output about one claim, provider or group. A reason to look,
  not a finding of fraud.
- **Case:** related flags about the same thing, grouped so you read one story instead of
  several alerts.
- **What the rules suggest (disposition):** the one next step a check's policy recommends, such
  as hold before paying, pay but check afterwards, refer to investigations, talk to the
  provider, or watch only. Fixed in the check, never changed by a score.
- **Urgency (priority):** decides the order of the queue, nothing else.
- **Established / not yet established:** whether an amount rests on a clear rule that failed,
  or is only the full value of an unusual claim. Never added together.
- **Watch-only (shadow) mode:** results are recorded and measured but not used for real
  decisions.
- **Similar providers (peer group):** the providers a provider is compared with, for example the
  same specialty and size. The comparison is refused when too few similar providers exist.
- **Pattern-finding model:** a program that ranks how unusual a claim is. Isolation Forest and
  Local Outlier Factor are the two used here.
- **Promotion gate:** the five-question checklist a model must pass before it may be used.
- **Can't be measured yet:** the tool will not invent a figure that needs information which
  does not exist yet, usually reviewers' decisions.
- **Synthetic:** generated by a program to test the tool; not real claims.

---

## For technical readers

This guide describes the tool for its users. For the engineering and the evidence:

- **`README.md`** — the architecture, how to install, run and test it (`streamlit run
  app/Home.py`, `python -m fwa.run_validation`, `pytest`), and what the artefact does not claim.
- **`docs/DEMO_DATASET.md`** — how the synthetic datasets were generated, what was planted and
  why, and why no figure derived from them may be used to claim detection performance.
- **`reports/`** — the validation report, release-gate report, model card, limitations, the
  control coverage matrix and the other artefacts, regenerated by `python -m fwa.run_validation`
  and viewable on the **Validation report** page. Two runs from the same input and seed produce
  identical findings; only wall-clock fields differ.

A few facts that sit behind the plain wording above:

- The contract in `src/fwa/engine/contract.py` refuses, at registration time, any control of type
  S, N, T or M that declares a REJECT or REPRICE disposition. There is no override.
- Signals sharing an underlying fact contribute `max(evidence_strength)`, never the sum.
- Data support per control is `EXECUTABLE`, `PARTIAL` or `NOT_EXECUTABLE_ON_THIS_DATASET`,
  computed per run from the canonical tables the dataset populates (`rules/unlocks/` can make a
  control runnable when a dataset carries the tables it needs). Each non-executable control names
  the canonical fields that would unlock it.
- The models are fitted on a temporal split with provider hold-out; a criterion that cannot be
  evaluated is `NOT_ASSESSABLE`, which is never a pass. An optional exploratory mode scores every
  claim, under a banner, and is excluded from the gate and every metric.
- Precision in the strict sense is confirmed ÷ reviewed. With nothing reviewed it is
  `NOT_MEASURABLE_ON_THIS_DATASET`. Precision-like figures computed against labels in a file
  are upper bounds, not review results.
- The audit log is append-only and hash-chained; unmasking, overriding and kill-switching require
  a reason.
- Each run is cached on the content hash of the input file and the adapter, so switching back to
  a dataset is free.
