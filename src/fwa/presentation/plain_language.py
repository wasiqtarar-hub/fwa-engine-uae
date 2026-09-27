"""The central plain-language vocabulary, and the number formatters.

Every page reads its user-facing wording through this module. The wording
itself lives in ``config/plain_language/*.yaml`` (statuses, fields, model
features, tables, glossary, pages) and ``config/plain_language/controls/*.yaml``
(one entry per control), so a policy owner can improve a sentence without a
code change — the same reason the rule catalogue is configuration.

Two properties matter more than the wording:

* **Internal identifiers are never renamed.** Enum values, field names,
  rule ids and model feature names are what the contract, the audit log, the
  CSV reports and the tests are keyed on. This module maps them to words at the
  moment they are shown, and nowhere else.
* **Nothing is silently dropped.** An identifier with no entry falls back to a
  humanised form of itself ("Not executable on this dataset") rather than to
  a blank, and ``tests/unit/test_plain_language.py`` fails when a control, a
  status or a model feature has no entry at all.
"""

from __future__ import annotations

import datetime as _dt
import math
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import yaml

from ..config import CONFIG_DIR
from ..evaluation.prose import plural

__all__ = [
    "PlainVocabulary", "Phrase", "ControlText", "vocabulary", "reload_vocabulary",
    "status", "label", "field_label", "field_meaning", "feature_phrase", "feature_unit",
    "control_text", "control_title", "table_label", "missing_information", "glossary", "parameter_text",
    "glossary_entry", "term_help", "page_info", "standard_text", "humanise", "subject_word",
    "model_label", "aed", "pct", "times_phrase", "ratio_words", "plain_date", "plain_number",
    "plain_value", "yes_no", "count_phrase", "duration_days", "friendly_frame",
    "VOCAB_DIR", "RAW_TOKEN_RE", "SNAKE_TOKEN_RE", "RULE_ID_RE",
]

VOCAB_DIR = CONFIG_DIR / "plain_language"

#: What "speaking in code" looks like. Used by the UI tests and by
#: :func:`humanise` to decide whether a string needs translating at all.
RAW_TOKEN_RE = re.compile(r"\b[A-Z]{2,}(?:_[A-Z0-9]+)+\b")
SNAKE_TOKEN_RE = re.compile(r"\b[a-z][a-z0-9]*(?:_[a-z0-9]+)+\b")
RULE_ID_RE = re.compile(r"\b(?:ENT|PAY|CLN|PHR|DOC|NET|ANL|POL)-\d{2}(?:-R\d{2})?\b")

_DASH = "—"


# =============================================================================
# data classes
# =============================================================================


@dataclass(frozen=True)
class Phrase:
    """A status in words: a short label, a one-sentence meaning, a tone, an icon."""

    value: str
    label: str
    meaning: str = ""
    tone: str = "neutral"          # ok | info | warn | bad | neutral
    icon: str = ""

    def __str__(self) -> str:      # so f"{status('PASS')}" reads as the label
        return self.label

    @property
    def badge(self) -> str:
        return f"{self.icon} {self.label}".strip()


@dataclass(frozen=True)
class ControlText:
    """One control, in words. Every field is presentation only."""

    rule_id: str
    title: str
    finding: str
    what: str = ""
    why: str = ""
    next_steps: tuple[str, ...] = ()
    needs: str = ""
    explanation: str = ""          # optional template with {evidence_key:format} placeholders


@dataclass
class PlainVocabulary:
    statuses: dict[str, dict[str, Any]] = field(default_factory=dict)
    priority_bands: dict[str, dict[str, Any]] = field(default_factory=dict)
    control_types: dict[str, dict[str, Any]] = field(default_factory=dict)
    stages: dict[str, dict[str, Any]] = field(default_factory=dict)
    domains: dict[str, dict[str, Any]] = field(default_factory=dict)
    case_families: dict[str, dict[str, Any]] = field(default_factory=dict)
    dimensions: dict[str, dict[str, Any]] = field(default_factory=dict)
    subjects: dict[str, str] = field(default_factory=dict)
    roles: dict[str, str] = field(default_factory=dict)
    role_sees: dict[str, str] = field(default_factory=dict)
    review_categories: dict[str, str] = field(default_factory=dict)
    audit_events: dict[str, str] = field(default_factory=dict)
    models: dict[str, dict[str, Any]] = field(default_factory=dict)
    fields: dict[str, dict[str, Any]] = field(default_factory=dict)
    features: dict[str, dict[str, Any]] = field(default_factory=dict)
    tables: dict[str, dict[str, Any]] = field(default_factory=dict)
    glossary: dict[str, dict[str, Any]] = field(default_factory=dict)
    pages: dict[str, dict[str, Any]] = field(default_factory=dict)
    start_here: list[dict[str, Any]] = field(default_factory=list)
    standard_text: dict[str, str] = field(default_factory=dict)
    controls: dict[str, dict[str, Any]] = field(default_factory=dict)
    suggestions: dict[str, str] = field(default_factory=dict)
    #: governed parameter key -> {label, meaning, raise, lower}
    parameters: dict[str, dict[str, Any]] = field(default_factory=dict)
    evidence_strength_bands: list[dict[str, Any]] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)


def _read(path: Path) -> dict[str, Any]:
    try:
        return yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except FileNotFoundError:
        return {}


def _load(directory: Path) -> PlainVocabulary:
    vocab = PlainVocabulary()
    if not directory.exists():
        return vocab
    for path in sorted(directory.glob("*.yaml")):
        raw = _read(path)
        vocab.sources.append(path.name)
        for section, content in raw.items():
            if not hasattr(vocab, section) or section in ("controls", "sources"):
                continue
            current = getattr(vocab, section)
            if isinstance(current, dict) and isinstance(content, dict):
                current.update({str(k): v for k, v in content.items()})
            elif isinstance(current, list) and isinstance(content, list):
                current.extend(content)
    controls_dir = directory / "controls"
    if controls_dir.exists():
        for path in sorted(controls_dir.glob("*.yaml")):
            raw = _read(path)
            vocab.sources.append(f"controls/{path.name}")
            for rule_id, entry in raw.items():
                if isinstance(entry, dict):
                    vocab.controls[str(rule_id)] = entry
    return vocab


@lru_cache(maxsize=4)
def _cached(directory: str) -> PlainVocabulary:
    return _load(Path(directory))


def vocabulary(directory: Path | str | None = None) -> PlainVocabulary:
    """The merged vocabulary. Cached; call :func:`reload_vocabulary` after an edit."""
    return _cached(str(directory or VOCAB_DIR))


def reload_vocabulary() -> None:
    _cached.cache_clear()


# =============================================================================
# words for identifiers
# =============================================================================


def humanise(token: Any) -> str:
    """Last-resort readable form of an identifier: ``NOT_EXECUTABLE`` → "Not executable".

    Used only when the vocabulary has no entry, so that the reader sees words
    rather than a code token, and the gap stays visible to a maintainer (the
    vocabulary tests list every identifier that reaches this path).
    """
    if token is None:
        return _DASH
    text = str(token).strip()
    if not text:
        return _DASH
    if "_" not in text and not text.isupper():
        return text
    words = text.replace("_", " ").strip().lower()
    return words[:1].upper() + words[1:]


def status(value: Any) -> Phrase:
    """A status, disposition or outcome in words."""
    key = getattr(value, "value", value)
    key = "" if key is None else str(key)
    entry = vocabulary().statuses.get(key)
    if entry is None and key.upper() in vocabulary().statuses:
        entry = vocabulary().statuses[key.upper()]
    if entry is None:
        return Phrase(value=key, label=humanise(key))
    return Phrase(
        value=key,
        label=str(entry.get("label") or humanise(key)),
        meaning=str(entry.get("meaning") or ""),
        tone=str(entry.get("tone") or "neutral"),
        icon=str(entry.get("icon") or ""),
    )


def _section_label(section: Mapping[str, Any], key: str) -> str | None:
    entry = section.get(key)
    if entry is None:
        return None
    if isinstance(entry, dict):
        return str(entry.get("label") or "") or None
    return str(entry)


def label(value: Any, kind: str | None = None) -> str:
    """The plain label for any identifier the app can display.

    ``kind`` narrows the lookup (``"status"``, ``"role"``, ``"family"``,
    ``"dimension"``, ``"domain"``, ``"stage"``, ``"type"``, ``"band"``,
    ``"audit"``, ``"category"``, ``"subject"``, ``"field"``, ``"model"``);
    without it every section is tried in a fixed order, statuses first.
    """
    key = getattr(value, "value", value)
    if key is None or (isinstance(key, float) and math.isnan(key)):
        return _DASH
    key = str(key)
    v = vocabulary()
    order: dict[str, Mapping[str, Any]] = {
        "status": v.statuses, "role": v.roles, "audit": v.audit_events,
        "family": v.case_families, "dimension": v.dimensions, "domain": v.domains,
        "stage": v.stages, "type": v.control_types, "band": v.priority_bands,
        "category": v.review_categories, "subject": v.subjects, "field": v.fields,
        "model": v.models,
    }
    sections = [order[kind]] if kind in order else list(order.values())
    for section in sections:
        found = _section_label(section, key)
        if found:
            return found
    if kind in (None, "type") and "/" in key and all(p in v.control_types for p in key.split("/")):
        return " + ".join(_section_label(v.control_types, p) or p for p in key.split("/"))
    return humanise(key)


def subject_word(subject_type: Any) -> str:
    return vocabulary().subjects.get(str(subject_type), humanise(subject_type).lower())


def model_label(name: str, *, plain: bool = False) -> str:
    entry = vocabulary().models.get(str(name), {})
    if plain:
        return str(entry.get("plain") or entry.get("label") or humanise(name))
    return str(entry.get("label") or humanise(name))


def field_label(name: Any) -> str:
    key = str(name)
    entry = vocabulary().fields.get(key)
    if isinstance(entry, dict) and entry.get("label"):
        return str(entry["label"])
    return humanise(key)


def field_meaning(name: Any) -> str:
    entry = vocabulary().fields.get(str(name))
    return str(entry.get("meaning", "")) if isinstance(entry, dict) else ""


def feature_phrase(name: str) -> str:
    """A model feature as a phrase that can follow "because of"."""
    entry = vocabulary().features.get(str(name))
    if isinstance(entry, dict) and entry.get("phrase"):
        return str(entry["phrase"])
    if str(name).endswith("__missing"):
        return f"that {feature_phrase(str(name)[: -len('__missing')])} was missing"
    return humanise(name).lower()


def feature_unit(name: str) -> str:
    entry = vocabulary().features.get(str(name))
    return str(entry.get("unit", "other")) if isinstance(entry, dict) else "other"


def control_text(rule_id: str, *, fallback_name: str = "") -> ControlText:
    entry = vocabulary().controls.get(str(rule_id), {})
    title = str(entry.get("title") or "") or (humanise(fallback_name) if fallback_name else "A check")
    return ControlText(
        rule_id=str(rule_id),
        title=title,
        finding=str(entry.get("finding") or entry.get("what") or title),
        what=str(entry.get("what") or ""),
        why=str(entry.get("why") or ""),
        next_steps=tuple(str(s) for s in (entry.get("next_steps") or ())),
        needs=str(entry.get("needs") or ""),
        explanation=str(entry.get("explanation") or ""),
    )


def control_title(rule_id: str, fallback_name: str = "") -> str:
    return control_text(rule_id, fallback_name=fallback_name).title


def table_label(table: str) -> str:
    entry = vocabulary().tables.get(str(table))
    return str(entry.get("label")) if isinstance(entry, dict) and entry.get("label") else humanise(table)


def _table_needs(table: str) -> str:
    entry = vocabulary().tables.get(str(table))
    if isinstance(entry, dict) and entry.get("needs"):
        return str(entry["needs"])
    return humanise(table).lower()


def missing_information(required_fields: Iterable[str] | str | None) -> str:
    """The canonical fields a control needs, as one plain phrase.

    ``["claim_line.activity_code", "provider.specialty"]`` becomes
    "line-level service details (…) and hospital and clinic details (…)".
    """
    if required_fields is None:
        return ""
    if isinstance(required_fields, str):
        required_fields = [p for p in required_fields.split(";")]
    tables: list[str] = []
    for f in required_fields:
        t = str(f).strip().split(" (")[0].split(".")[0].strip()
        if t and t not in tables:
            tables.append(t)
    phrases = [_table_needs(t) for t in tables]
    if not phrases:
        return ""
    if len(phrases) == 1:
        return phrases[0]
    return ", ".join(phrases[:-1]) + " and " + phrases[-1]


def glossary() -> dict[str, dict[str, Any]]:
    return dict(vocabulary().glossary)


def glossary_entry(key: str) -> dict[str, Any]:
    return dict(vocabulary().glossary.get(key, {}))


def term_help(key: str) -> str:
    """The one-line tooltip for a glossary term, for ``help=`` arguments."""
    entry = vocabulary().glossary.get(key, {})
    return str(entry.get("short") or "")


def page_info(name: str) -> dict[str, Any]:
    entry = dict(vocabulary().pages.get(name, {}))
    entry.setdefault("title", name)
    entry.setdefault("purpose", "")
    entry.setdefault("simple", False)
    entry.setdefault("icon", "")
    return entry


def parameter_text(key: str) -> dict[str, str]:
    """Plain name, meaning and the effect of raising/lowering a governed parameter."""
    entry = vocabulary().parameters.get(str(key), {})
    return {
        "label": str(entry.get("label") or humanise(key)),
        "meaning": str(entry.get("meaning") or ""),
        "raise": str(entry.get("raise") or ""),
        "lower": str(entry.get("lower") or ""),
    }


def standard_text(key: str) -> str:
    return str(vocabulary().standard_text.get(key, ""))


# =============================================================================
# numbers, dates and values
# =============================================================================


def _missing(value: Any) -> bool:
    if value is None:
        return True
    try:
        import pandas as pd

        if value is pd.NA or value is pd.NaT:
            return True
    except Exception:  # pragma: no cover
        pass
    if isinstance(value, float) and math.isnan(value):
        return True
    return False


def aed(value: Any, decimals: int = 0, *, missing: str = "not available") -> str:
    """``AED 1,264,948``."""
    if _missing(value):
        return missing
    try:
        v = float(value)
    except (TypeError, ValueError):
        return str(value)
    return f"AED {v:,.{decimals}f}"


def pct(value: Any, decimals: int | None = None, *, missing: str = "not available") -> str:
    """A ratio as a percentage: ``0.0106`` → ``1.1%``; ``0.41`` → ``41%``."""
    if _missing(value):
        return missing
    try:
        v = float(value) * 100.0
    except (TypeError, ValueError):
        return str(value)
    if decimals is None:
        decimals = 0 if abs(v) >= 10 or v == 0 else 1
    text = f"{v:.{decimals}f}"
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return f"{text}%"


def ratio_words(ratio: Any) -> str:
    """A multiplier in words: 2.9 → "about 3 times"; 0.5 → "about half"."""
    if _missing(ratio):
        return "not comparable"
    r = float(ratio)
    if r <= 0:
        return "none of"
    if r < 0.2:
        return "a small fraction of"
    if r < 0.4:
        return "about a third of"
    if r < 0.6:
        return "about half"
    if r < 0.85:
        return "somewhat less than"
    if r < 1.15:
        return "about the same as"
    if r < 1.5:
        return "somewhat more than"
    if r < 1.75:
        return "about one and a half times"
    if r < 10:
        whole = int(round(r))
        return f"about {whole} times"
    return f"more than {int(r)} times"


def times_phrase(value: Any, typical: Any, noun: str = "the usual amount") -> str:
    """"about 3 times the usual amount" — the comparison a reviewer can act on."""
    if _missing(value) or _missing(typical):
        return ""
    try:
        t = float(typical)
        v = float(value)
    except (TypeError, ValueError):
        return ""
    if t == 0:
        return "" if v == 0 else f"well above {noun}"
    words = ratio_words(v / t)
    if words == "about the same as":
        return f"about the same as {noun}"
    if words.endswith(" of") or words in ("about half", "somewhat less than", "somewhat more than"):
        return f"{words} {noun}"
    return f"{words} {noun}"


def plain_date(value: Any, *, missing: str = "date not recorded") -> str:
    """``11 Mar 2024``."""
    if _missing(value):
        return missing
    try:
        import pandas as pd

        ts = pd.Timestamp(value)
        if pd.isna(ts):
            return missing
        return f"{ts.day} {ts:%b %Y}"
    except Exception:
        return str(value)


def plain_month(value: Any) -> str:
    try:
        import pandas as pd

        return f"{pd.Timestamp(value):%B %Y}"
    except Exception:
        return str(value)


def plain_number(value: Any, decimals: int = 0, *, missing: str = "not available") -> str:
    if _missing(value):
        return missing
    if isinstance(value, bool):
        return yes_no(value)
    try:
        v = float(value)
    except (TypeError, ValueError):
        return str(value)
    if decimals == 0 and abs(v - round(v)) < 1e-9:
        return f"{int(round(v)):,}"
    return f"{v:,.{decimals or 2}f}"


def yes_no(value: Any) -> str:
    if _missing(value):
        return "not recorded"
    if isinstance(value, str):
        return {"true": "Yes", "false": "No"}.get(value.strip().lower(), value)
    return "Yes" if bool(value) else "No"


def duration_days(days: Any) -> str:
    if _missing(days):
        return "not recorded"
    d = float(days)
    return f"{d:,.0f} {plural('day', round(d))}"


def count_phrase(n: Any, noun: str) -> str:
    """``count_phrase(3, "claim")`` → "3 claims"."""
    if _missing(n):
        return f"no {plural(noun, 2)}"
    n_int = int(n)
    return f"{n_int:,} {plural(noun, n_int)}"


def plain_value(value: Any, *, name: str | None = None) -> str:
    """Any single value, as it should appear in a default view.

    Never returns ``nan``, ``None``, ``True`` or ``False``. When ``name`` is a
    known field its unit steers the format (money as AED, ratios as %).
    """
    if _missing(value):
        return _DASH
    if isinstance(value, bool):
        return yes_no(value)
    try:
        import numpy as np

        if isinstance(value, np.bool_):
            return yes_no(bool(value))
    except Exception:  # pragma: no cover
        pass
    if isinstance(value, (_dt.date, _dt.datetime)):
        return plain_date(value)
    if isinstance(value, (list, tuple, set)):
        return ", ".join(plain_value(v) for v in value) or _DASH
    if isinstance(value, dict):
        return "; ".join(f"{humanise(k)}: {plain_value(v)}" for k, v in value.items()) or _DASH
    if isinstance(value, str):
        if value.strip().lower() in ("nan", "none", "nat", "null", ""):
            return _DASH
        if value in ("True", "False"):
            return yes_no(value)
        if RAW_TOKEN_RE.fullmatch(value) or value in vocabulary().statuses:
            return label(value)
        return value
    if isinstance(value, (int, float)):
        unit = _unit_for(name)
        if unit == "AED":
            return aed(value)
        if unit == "ratio":
            return pct(value)
        return plain_number(value, 0 if float(value).is_integer() else 2)
    return str(value)


def _unit_for(name: str | None) -> str:
    if not name:
        return ""
    n = str(name)
    feat = vocabulary().features.get(n)
    if isinstance(feat, dict) and feat.get("unit"):
        return str(feat["unit"])
    if n.endswith("_aed") or n.endswith("amount") or "_amount_" in n:
        return "AED"
    if n.endswith(("_ratio", "_rate", "_share")):
        return "ratio"
    return ""


# =============================================================================
# tables
# =============================================================================

#: Columns whose values are internal identifiers that should be translated.
_VALUE_KINDS: dict[str, str] = {
    "disposition": "status", "data_support": "status", "outcome": "status",
    "run_outcome": "status", "status": "status", "verdict": "status", "gate": "status",
    "rule_status": "status", "role": "role", "actor_role": "role", "event_type": "audit",
    "scenario_family": "family", "case_type": "dimension", "domain": "domain",
    "stage": "stage", "type": "type", "priority_band": "band", "validated_category": "category",
    "subject_type": "subject",
}


def friendly_frame(
    frame,
    columns: Sequence[str] | None = None,
    *,
    rename: Mapping[str, str] | None = None,
    value_kinds: Mapping[str, str] | None = None,
    money: Iterable[str] = (),
    ratios: Iterable[str] = (),
    dates: Iterable[str] = (),
    keep_numeric: Iterable[str] = (),
):
    """A copy of ``frame`` ready for a non-technical reader.

    * keeps ``columns`` (in that order) when given;
    * translates identifier values (dispositions, statuses, roles, families…);
    * formats money, ratios and dates; turns booleans into Yes/No;
    * replaces every missing value with "—" so no ``nan``/``None`` is shown;
    * renames columns to their plain labels (``rename`` overrides).

    Numeric columns named in ``keep_numeric`` stay numeric so they still sort,
    with missing values left as blanks rather than the word "nan".
    """
    import numpy as np
    import pandas as pd

    if frame is None:
        return pd.DataFrame()
    df = frame.copy()
    if columns is not None:
        df = df[[c for c in columns if c in df.columns]]
    kinds = dict(_VALUE_KINDS)
    kinds.update(value_kinds or {})
    money, ratios, dates, keep_numeric = set(money), set(ratios), set(dates), set(keep_numeric)

    for col in list(df.columns):
        series = df[col]
        if col in keep_numeric and pd.api.types.is_numeric_dtype(series):
            continue
        if col in money:
            df[col] = [aed(v, missing=_DASH) for v in series]
        elif col in ratios:
            df[col] = [pct(v, missing=_DASH) for v in series]
        elif col in dates:
            df[col] = [plain_date(v, missing=_DASH) for v in series]
        elif col in kinds:
            kind = kinds[col]
            df[col] = [_DASH if _missing(v) else label(v, kind) for v in series]
        elif pd.api.types.is_bool_dtype(series):
            df[col] = [yes_no(v) for v in series]
        elif pd.api.types.is_numeric_dtype(series):
            if series.isna().any():
                df[col] = [plain_value(v, name=col) for v in series]
        else:
            df[col] = [plain_value(v, name=col) for v in series]
    mapping = {c: (rename or {}).get(c) or field_label(c) for c in df.columns}
    df = df.rename(columns=mapping)
    return df.replace({np.nan: _DASH}) if not df.empty else df
