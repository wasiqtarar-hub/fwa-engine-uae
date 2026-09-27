"""Shared plain-language helpers for the Data, Help and Validation Report pages.

Presentation only. Everything here reads the engine's output and describes it
in words; nothing changes what the engine does, and no internal identifier is
renamed. Wording for things the shared vocabulary lacks lives in
``config/plain_language/data_help_pages.yaml`` (section ``data_help_pages:``,
read directly here; its ``glossary:`` entries are merged by the shared loader).

The one rule: plain wording must carry the safety boundary, never drop it.
Established and not-yet-established amounts are reported side by side and
never added together; a measure that cannot be measured is said in words,
never replaced with a number.
"""

from __future__ import annotations

import dataclasses
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

import dataset as ds
from fwa.presentation import (
    aed, control_text, missing_information, model_label, plain_number, status, table_label,
)
from fwa.presentation.plain_language import VOCAB_DIR

ROOT = Path(__file__).resolve().parents[2]
VOCAB_FILE = VOCAB_DIR / "data_help_pages.yaml"

NOT_RUN = ("NOT_EXECUTABLE_ON_THIS_DATASET", "NOT_EXECUTABLE")


# ---------------------------------------------------------------------------
# vocabulary
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def _vocab() -> dict[str, Any]:
    try:
        raw = yaml.safe_load(VOCAB_FILE.read_text(encoding="utf-8")) or {}
    except Exception:  # a missing wording file must never break a page
        raw = {}
    return raw.get("data_help_pages", {}) or {}


def words(section: str) -> dict[str, Any]:
    return dict(_vocab().get(section, {}) or {})


def number(n: Any) -> str:
    return plain_number(n)


def plural(n: int, one: str, many: str | None = None) -> str:
    return f"{n:,} {one if n == 1 else (many or one + 's')}"


def join_words(items: list[str], last: str = "and") -> str:
    items = [i for i in items if i]
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    return ", ".join(items[:-1]) + f" {last} " + items[-1]


# ---------------------------------------------------------------------------
# datasets and file formats
# ---------------------------------------------------------------------------


def adapter_labels() -> dict[str, str]:
    """Every adapter the app offers, with a plain label (internal key unchanged)."""
    known = words("adapters")
    out: dict[str, str] = {}
    for key, fallback in getattr(ds, "ADAPTER_LABELS", {}).items():
        out[key] = str((known.get(key) or {}).get("label") or fallback)
    return out


def adapter_when(key: str) -> str:
    return str((words("adapters").get(key) or {}).get("when") or "")


def adapter_help() -> str:
    lines: list[str] = []
    for key, label in adapter_labels().items():
        when = adapter_when(key)
        lines.append(f"• {label}: {when}" if when else f"• {label}")
    lines.append("If you pick the wrong one nothing is mis-read: the checks that cannot find what "
                 "they need are reported as unable to run.")
    return "\n\n".join(lines)


def _info_key(spec) -> str:
    path = Path(str(getattr(spec, "path", "")))
    return path.name


def dataset_info(spec) -> dict[str, Any]:
    """Plain name, one-line description and SYNTHETIC flag for a dataset."""
    shipped = words("shipped")
    for candidate in (_info_key(spec), str(getattr(spec, "name", ""))):
        if candidate in shipped:
            info = dict(shipped[candidate])
            break
    else:
        name = str(getattr(spec, "name", "")) or _info_key(spec)
        info = {"name": name, "one_line": "", "use_when": "",
                "synthetic": "synthetic" in name.lower() or "uae_demo" in name.lower(),
                "default": False}
    info.setdefault("name", str(getattr(spec, "name", "")))
    return info


def dataset_title(spec) -> str:
    info = dataset_info(spec)
    return str(info["name"]) + (" — SYNTHETIC" if info.get("synthetic") else "")


def _fallback_shipped() -> list:
    """Build the shipped list ourselves when ``dataset.shipped_datasets`` is absent."""
    out = []
    candidates = [
        (ROOT / "data" / "uae_demo.zip", "uae_multitable"),
        (ROOT / "data" / "claims_demo_synthetic.csv", "generic_india_tpa"),
        (ROOT / "data" / "claims.csv", "generic_india_tpa"),
    ]
    labels = getattr(ds, "ADAPTER_LABELS", {})
    for path, adapter in candidates:
        if not path.exists() or adapter not in labels:
            continue
        try:
            spec = ds.spec_from_path(str(path), adapter)
            try:
                spec = dataclasses.replace(spec, origin="shipped")
            except Exception:
                pass
            out.append(spec)
        except Exception:
            continue
    return out


def shipped_specs() -> list:
    """The datasets that come with the app, default first. Never raises."""
    fn = getattr(ds, "shipped_datasets", None)
    specs: list = []
    if callable(fn):
        try:
            specs = [s for s in (fn() or []) if hasattr(s, "key") and hasattr(s, "path")]
        except Exception:
            specs = []
    if not specs:
        specs = _fallback_shipped()
    specs.sort(key=lambda s: (not dataset_info(s).get("default"),))
    return specs


def friendly_load_problem(exc: BaseException) -> tuple[str, str]:
    """(what is wrong, how to fix it) for a file that could not be loaded."""
    name = type(exc).__name__
    text = str(exc)
    if isinstance(exc, FileNotFoundError) or "No file at" in text:
        return ("We couldn't find anything at that location.",
                "Check the path for typing mistakes, and make sure it is the full path, "
                "starting with the drive letter (for example C:\\claims\\march.csv).")
    if "not a delimited" in text or "suffix" in text.lower() or "extension" in text.lower() \
            or "not supported" in text.lower():
        return ("That is not a file type the engine can read.",
                "Use a .csv file (one row per claim), or a .zip or folder of .csv tables.")
    if "directory" in text.lower() or "folder" in text.lower():
        return ("That location is a folder the chosen file format can't read.",
                "For a folder of several tables, choose 'UAE multi-table' as the file format. "
                "Otherwise, point at the .csv file itself.")
    if name in ("BadZipFile", "BadZipfile") or "zip" in text.lower():
        return ("The .zip file could not be opened.",
                "It may be damaged or only partly downloaded. Create the .zip again and retry.")
    if name == "EmptyDataError" or "No columns to parse" in text:
        return ("The file is empty.",
                "Open it in a spreadsheet to check it holds claims, then save and upload it again.")
    if name == "ParserError" or "tokeniz" in text.lower():
        return ("The file isn't laid out as a table the engine can read.",
                "Save it from your spreadsheet as 'CSV (comma delimited)' and try again.")
    if name == "UnicodeDecodeError":
        return ("The file is saved in a text encoding the engine can't read.",
                "Save it as 'CSV UTF-8' from your spreadsheet and try again.")
    if isinstance(exc, KeyError) or "column" in text.lower() or "missing" in text.lower():
        return ("The file doesn't have the columns this file format expects.",
                "Check that you picked the right file format. A file with one row per claim "
                "needs 'Generic claim list'; a folder or .zip of tables needs 'UAE multi-table'.")
    if isinstance(exc, PermissionError):
        return ("The app isn't allowed to read that file.",
                "Copy it to a folder you own (for example your Documents folder) and point at "
                "the copy, or ask your administrator for read access.")
    return ("The engine couldn't run on this file.",
            "Check the file format setting and that the file opens in a spreadsheet. If it "
            "still fails, give the technical details below to an administrator.")


# ---------------------------------------------------------------------------
# check coverage
# ---------------------------------------------------------------------------


def _tables_of(required) -> list[str]:
    if not required:
        return []
    if isinstance(required, str):
        required = required.split(";")
    tables: list[str] = []
    for f in required:
        t = str(f).strip().split(" (")[0].split(".")[0].strip()
        if t and t not in tables:
            tables.append(t)
    return tables


def _populated(result, table: str) -> bool:
    try:
        return bool(result.dataset.is_populated(table))
    except Exception:
        return False


@dataclass
class Coverage:
    total: int = 0
    full: int = 0
    simplified: int = 0
    not_run: int = 0
    found_something: int = 0
    unlocked: int = 0
    upgraded: int = 0
    #: [(table, "line-level service details (…)", count)], largest first
    groups: list[tuple[str, str, int]] = field(default_factory=list)
    #: one dict per control, internal columns + plain ones
    rows: list[dict[str, Any]] = field(default_factory=list)

    def headline(self) -> str:
        text = (f"Of {self.total:,} checks, {self.full:,} ran fully, {self.simplified:,} ran in a "
                f"simplified form, and {self.not_run:,} could not run")
        if self.not_run and self.groups:
            top = [needs_short(t) for t, _, _ in self.groups[:3]]
            if len(top) == 1:
                lead = top[0]
            else:
                lead = ", ".join(top[:-1]) + (", or " if len(top) > 2 else " or ") + top[-1]
            if self.not_run == 1:
                text += " because the file lacks " + lead
            else:
                text += " because the file lacks information they need, most often " + lead
        return text + "."


def needs_short(table: str) -> str:
    """"line-level service details" — the label, in lower case, for a sentence."""
    label = table_label(table)
    if label and label[:1].isupper() and not label[:2].isupper():
        label = label[:1].lower() + label[1:]
    return label


def coverage(result) -> Coverage:
    """Of 164 checks: how many ran fully, simplified, or not at all, and why not."""
    cov = Coverage()
    group_counts: dict[str, int] = {}
    for rec in result.evaluation.records:
        cov.total += 1
        support = str(rec.data_support)
        if support == "EXECUTABLE":
            cov.full += 1
        elif support == "PARTIAL":
            cov.simplified += 1
        else:
            cov.not_run += 1
        if rec.signal_count:
            cov.found_something += 1
        basis = str(getattr(rec, "support_basis", "catalogue") or "catalogue")
        if basis == "unlocked":
            cov.unlocked += 1
        elif basis == "upgraded":
            cov.upgraded += 1

        try:
            control = result.registry.get(rec.rule_id)
            required = list(getattr(control, "required_canonical_fields", []) or [])
            fallback_name = str(getattr(control, "name", ""))
        except Exception:
            required, fallback_name = [], ""

        missing_tables: list[str] = []
        needs = ""
        if support in NOT_RUN:
            tables = _tables_of(required)
            missing_tables = [t for t in tables if not _populated(result, t)]
            if not missing_tables:
                missing_tables = _tables_of(getattr(rec, "missing_for_unlock", ""))
            if not missing_tables:
                missing_tables = tables
            for t in missing_tables:
                group_counts[t] = group_counts.get(t, 0) + 1
            needs = missing_information(missing_tables) or "information this file does not have"
            needs_text = "Needs " + needs
        elif support == "PARTIAL":
            unlock = _tables_of(getattr(rec, "missing_for_unlock", ""))
            needs_text = ("Ran on a stand-in measure; the full check needs "
                          + missing_information(unlock)) if unlock else "Ran on a stand-in measure"
        else:
            needs_text = "Nothing missing"

        text = control_text(rec.rule_id, fallback_name=fallback_name)
        cov.rows.append({
            "check": text.title,
            "on this file": status(support).label,
            "what happened": status(rec.outcome).label,
            "flags raised": int(rec.signal_count or 0),
            "what it needs": needs_text,
            # ---- technical, behind "Show all columns"
            "check id": rec.rule_id,
            "scenario id": rec.scenario_id,
            "classification in the catalogue": status(getattr(rec, "catalogue_data_support", "")
                                                      or support).label,
            "how it was classified": {
                "catalogue": "As catalogued",
                "unlocked": "Unlocked by this file's tables",
                "upgraded": "Upgraded from a stand-in by this file's tables",
            }.get(basis, basis),
            "required information (technical)": "; ".join(required) or "—",
            "missing for full version (technical)": getattr(rec, "missing_for_unlock", "") or "—",
            # ---- internal, for sorting and filtering (never shown)
            "_support": support,
            "_outcome": str(rec.outcome),
        })
    cov.groups = sorted(
        ((t, missing_information([t]), n) for t, n in group_counts.items()),
        key=lambda g: (-g[2], g[0]),
    )
    return cov


def coverage_frame(cov: Coverage, previous: pd.DataFrame | None = None) -> pd.DataFrame:
    frame = pd.DataFrame(cov.rows)
    if frame.empty:
        return frame
    if previous is not None and not previous.empty and "rule_id" in previous.columns:
        change = dict(zip(previous["rule_id"], previous["change"]))
        frame.insert(5, "compared with the previous file", [
            _change_words(change.get(r, "—")) for r in frame["check id"]])
    order = {"EXECUTABLE": 0, "PARTIAL": 1}
    frame["_o"] = frame["_support"].map(lambda s: order.get(s, 2))
    # Checks that found something first (most flags first), then the rest by
    # how fully they ran, so the checks that could not run sink to the bottom.
    frame = frame.sort_values(["flags raised", "_o", "check"], ascending=[False, True, True])
    return frame.drop(columns=["_o"]).reset_index(drop=True)


def _change_words(change: str) -> str:
    if change in ("—", "", None):
        return "Not on the previous file"
    m = re.fullmatch(r"([+-]\d+) signals", str(change))
    if m:
        n = int(m.group(1))
        return f"{abs(n):,} {'more' if n > 0 else 'fewer'} flag{'s' if abs(n) != 1 else ''}"
    return str(change)


def comparison_sentence(frame: pd.DataFrame, current_name: str, previous_name: str) -> str:
    """The comparison with the previously loaded dataset, in words."""
    if frame is None or frame.empty or "change" not in frame.columns:
        return ""
    newly = int((frame["change"] == "Newly runnable").sum())
    lost = int((frame["change"] == "No longer runnable").sum())
    changed = int(frame["change"].astype(str).str.endswith(" signals").sum())
    same = int((frame["change"] == "Unchanged").sum())
    parts = []
    parts.append(f"{plural(newly, 'check')} can run on {current_name} that could not run on "
                 f"{previous_name}" if newly else f"no check that was unable to run on "
                 f"{previous_name} can run now")
    parts.append(f"{plural(lost, 'check')} could run on {previous_name} but cannot run here"
                 if lost else "no check has stopped being able to run")
    parts.append(f"{plural(changed, 'check')} found a different number of flags" if changed
                 else "every check that ran on both found the same number of flags")
    text = "Compared with the file loaded before, " + join_words(parts) + "."
    if same:
        text += f" {plural(same, 'check')} behaved exactly the same on both."
    return text[:1].upper() + text[1:]


# ---------------------------------------------------------------------------
# models, cases and money, in words
# ---------------------------------------------------------------------------


def gate_question(name: str) -> str:
    return str(words("gate_questions").get(name) or name.lower())


def model_sentences(result) -> list[str]:
    """One paragraph per pattern-finding model: trained, scored, flagged, cleared or not."""
    layer = getattr(result, "models", None)
    anomaly = getattr(layer, "anomaly", None) if layer is not None else None
    models = getattr(anomaly, "models", None) or {}
    if not models:
        reasons = []
        for source in (getattr(anomaly, "messages", None), getattr(layer, "messages", None)):
            reasons += [str(m) for m in (source or []) if m]
        text = ("No pattern-finding model could be trained on this file, so only the checks "
                "and the simple scorecard ranked anything.")
        if reasons:
            text += " " + " ".join(dict.fromkeys(reasons))
        return [text]

    claims = getattr(result, "claims", None)
    total = 0 if claims is None else len(claims)
    out = []
    verdicts = getattr(layer, "gate_verdicts", {}) or {}
    for name, scores in models.items():
        label = model_label(name)
        what = model_label(name, plain=True)
        try:
            scored = int(scores.scores.notna().sum())
        except Exception:
            scored = 0
        flagged_index = getattr(scores, "flagged_index", None)
        flagged = 0 if flagged_index is None else len(flagged_index)
        trained = int(getattr(scores, "training_rows", 0) or 0)
        what = what.rstrip(". ")
        text = (f"**{label}** ({what[:1].lower() + what[1:] if what else 'a pattern-finder'}) "
                f"learned from {plural(trained, 'earlier claim')}, scored "
                f"{plural(scored, 'later claim')} from providers it had never seen, and flagged "
                f"{number(flagged)} of them as unusual.")
        if total and scored < total:
            text += (f" The other {number(total - scored)} claims got no score because they were "
                     "used for learning or come from providers it learned from, which keeps the "
                     "test fair.")
        verdict = verdicts.get(name)
        if verdict is None:
            text += (" Whether it can be trusted for real decisions has not been checked on this "
                     "file yet (the Pattern-finding models page runs that checklist). Until then it "
                     "stays in watch-only mode.")
        elif getattr(verdict, "promoted", False):
            text += (" It met every condition of the promotion checklist, so a policy owner who "
                     "did not propose it may approve it for use. Until someone does, it stays in "
                     "watch-only mode.")
        else:
            failed = [gate_question(c.name) for c in verdict.criteria if c.status == "FAIL"]
            unknown = [gate_question(c.name) for c in verdict.criteria
                       if c.status not in ("PASS", "FAIL", "INFORMATIONAL")]
            why = []
            if failed:
                why.append("it has not shown " + join_words([f"that it {q}" for q in failed],
                                                             "or"))
            if unknown:
                why.append("it can't yet be judged whether it " + join_words(unknown, "or"))
            text += (" It is **not cleared for use**" + (", because " + "; and ".join(why)
                                                        if why else "") +
                     ". It stays in watch-only mode.")
        out.append(text)
    return out


def exposure_split(result) -> tuple[float, float, int, int]:
    """(established AED, not-yet-established AED, cases established, cases not). Never summed."""
    try:
        queue = result.queue()
    except Exception:
        return 0.0, 0.0, 0, 0
    if queue is None or queue.empty or "exposure_established" not in queue.columns:
        return 0.0, 0.0, 0, 0
    est = queue["exposure_established"].fillna(False).astype(bool)
    amount = pd.to_numeric(queue.get("exposure_aed"), errors="coerce").fillna(0.0)
    return (float(amount[est].sum()), float(amount[~est].sum()),
            int(est.sum()), int((~est).sum()))


def disposition_sentence(result, top: int = 3) -> str:
    """"Most suggest checking after payment (3,210), …" — what the cases suggest."""
    try:
        queue = result.queue()
    except Exception:
        return ""
    if queue is None or queue.empty or "disposition" not in queue.columns:
        return ""
    counts = queue["disposition"].astype(str).value_counts()
    suggestions = {}
    try:
        from fwa.presentation import vocabulary
        suggestions = vocabulary().suggestions
    except Exception:
        pass
    parts = []
    for value, n in counts.head(top).items():
        phrase = (suggestions.get(value) or status(value).label.lower()).split(";")[0].strip()
        parts.append(f"{number(n)} suggest {phrase}")
    rest = int(counts.iloc[top:].sum()) if len(counts) > top else 0
    if rest:
        parts.append(f"{number(rest)} suggest something else")
    return "; ".join(parts[:-1]) + ("; and " if len(parts) > 1 else "") + parts[-1] + "."


def unmeasurable(result) -> list[tuple[str, str]]:
    """[(plain name, why)] for every measure this run cannot measure yet."""
    try:
        from fwa.evaluation.metrics import MetricSuite, NOT_MEASURABLE
        metrics = MetricSuite(result).compute()
    except Exception:
        return []
    known = words("metrics")
    out = []
    for m in metrics:
        if m.status != NOT_MEASURABLE:
            continue
        entry = known.get(m.name) or {}
        out.append((str(entry.get("label") or m.name.lower()),
                    str(entry.get("why") or "the information it needs does not exist in this "
                                             "file yet")))
    return out


def money(value: float) -> str:
    return aed(value)
