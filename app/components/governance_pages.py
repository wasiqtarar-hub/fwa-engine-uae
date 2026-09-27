"""Plain-language helpers for the four governance pages.

Shared by ``pages/rule_registry.py`` (Checks library), ``pages/governance.py``
(Governance), ``pages/parameters.py`` (Settings and thresholds) and
``pages/user_management.py`` (Users and access). Presentation only: every
function here turns an engine value into words and never changes it.

The wording lives in ``config/plain_language/governance_pages.yaml`` under the
``governance_pages:`` section (read directly here; the shared loader ignores
it) and in ``config/plain_language/parameters.yaml`` (read through
``fwa.presentation.parameter_text``).
"""

from __future__ import annotations

import math
import re
from functools import lru_cache
from typing import Any

import yaml

from fwa.presentation import (
    control_text, humanise, label, missing_information, parameter_text, plain_date, plain_number,
    status,
)
from fwa.presentation.plain_language import RAW_TOKEN_RE, SNAKE_TOKEN_RE, VOCAB_DIR

__all__ = [
    "words", "owner_label", "permission_label", "permission_meaning", "family_label",
    "basis_label", "role_can_do", "actor_role_label", "plain_text", "format_value",
    "setting_label", "check_title", "check_rows", "missing_words", "peer_level_label",
    "availability_label", "scale_source_label", "model_parameter_words", "explainer_label",
]


# ------------------------------------------------------------------ vocabulary


@lru_cache(maxsize=1)
def _load() -> dict[str, Any]:
    try:
        raw = yaml.safe_load((VOCAB_DIR / "governance_pages.yaml").read_text(encoding="utf-8")) or {}
    except Exception:  # a missing wording file must never break a page
        raw = {}
    return raw.get("governance_pages", {}) or {}


def words(section: str) -> dict[str, Any]:
    return dict(_load().get(section, {}) or {})


def _lookup(section: str, value: Any) -> str:
    key = "" if value is None else str(getattr(value, "value", value))
    found = words(section).get(key)
    if found is None:
        return humanise(key)
    return str(found.get("label") if isinstance(found, dict) else found)


def owner_label(value: Any) -> str:
    """A parameter or check owner (``claims_policy``) in words."""
    key = "" if value is None else str(value)
    if key in words("owners"):
        return str(words("owners")[key])
    if key.upper() in {r for r in words("role_can_do")}:
        return label(key.upper(), "role")
    return humanise(key)


def permission_label(value: Any) -> str:
    return _lookup("permissions", value)


def permission_meaning(value: Any) -> str:
    return str(words("permission_meanings").get(str(getattr(value, "value", value)), ""))


def family_label(family: Any) -> str:
    return _lookup("scenario_families", family)


def basis_label(basis: Any) -> str:
    return _lookup("support_basis", basis or "catalogue")


def role_can_do(role: Any) -> str:
    return str(words("role_can_do").get(str(getattr(role, "value", role)), ""))


def actor_role_label(value: Any) -> str:
    key = "" if value is None else str(value)
    if key in words("actor_roles"):
        return str(words("actor_roles")[key])
    return label(key, "role")


def peer_level_label(value: Any) -> str:
    return _lookup("peer_levels", value)


def availability_label(value: Any) -> str:
    return _lookup("peer_availability", value)


def scale_source_label(value: Any) -> str:
    return _lookup("scale_sources", value)


def explainer_label(method: Any) -> str:
    text = str(method or "")
    for key, words_ in words("explainers").items():
        if text.startswith(key):
            return str(words_)
    return humanise(text) if text else "—"


def model_parameter_words(name: Any) -> tuple[str, str]:
    """(plain label, "Chosen by policy" | "Measured from your data") for a model setting."""
    entry = words("model_parameters").get(str(name))
    if isinstance(entry, dict):
        kind = "Measured from your data" if entry.get("kind") == "measured" else "Chosen by policy"
        return str(entry.get("label") or humanise(name)), kind
    return humanise(name), "Chosen by policy"


# ------------------------------------------------------------------ free text


_PATH_RE = re.compile(r"\b[\w./\\-]+\.(?:py|yaml|csv|json)\b")
_CFG_RE = re.compile(r"`?cfg\.([a-z0-9_]+)`?")


def plain_text(text: Any) -> str:
    """Free engine prose with its code tokens put into words.

    ``MONITOR_ONLY`` becomes "Watch only", ``cfg.min_peer_group_n`` becomes the
    setting's plain name and a snake_case field name becomes its label. The
    sentence is otherwise untouched, so its meaning is never rewritten.
    """
    if text is None or (isinstance(text, float) and math.isnan(text)):
        return ""
    s = str(text)
    s = _PATH_RE.sub(lambda m: "the technical reports" if m.group(0).endswith((".csv", ".json"))
                     else "the configuration", s)
    s = _CFG_RE.sub(lambda m: f"the setting “{setting_label(m.group(1))}”", s)
    s = s.replace("`", "")
    s = RAW_TOKEN_RE.sub(lambda m: f"“{label(m.group(0))}”", s)

    def _snake(m: re.Match) -> str:
        token = m.group(0)
        from fwa.presentation import field_label

        return field_label(token).lower() if field_label(token) != humanise(token) else humanise(token).lower()

    return SNAKE_TOKEN_RE.sub(_snake, s)


# ------------------------------------------------------------------ settings


def setting_label(key: str) -> str:
    return parameter_text(key)["label"]


def _num(v: float) -> str:
    if abs(v - round(v)) < 1e-9:
        return f"{int(round(v)):,}"
    text = f"{v:,.4f}".rstrip("0").rstrip(".")
    return text


def _pct(v: float) -> str:
    p = v * 100
    text = f"{p:.2f}".rstrip("0").rstrip(".")
    return f"{text}%"


#: Words in a parameter key that say its value is a share (shown as a %).
_FRACTION_WORDS = {"share", "fraction", "percentile", "quantile", "contamination", "mass",
                   "margin", "uplift", "excess", "increase", "delta", "confidence", "coverage",
                   "rate", "tolerance", "similarity", "density", "flow", "width", "pvalue",
                   "probability", "yield", "linkage", "reproducibility", "prevalence", "pct",
                   "shift", "gap", "concentration"}
#: Words that say its value is a multiple of something (shown as "3 times").
_MULTIPLE_WORDS = {"multiple", "ratio", "lift", "oe", "markup", "expected"}
_UNITS = {"days": "day", "minutes": "minute", "months": "month", "years": "year",
          "periods": "month", "kmh": "km per hour", "ms": "milliseconds", "chars": "characters"}


def format_value(key: str, value: Any) -> str:
    """A governed value as a reader should see it: AED, %, days, "3 times", Yes/No."""
    k = str(key).lower()
    tokens = k.split("_")
    if value is None:
        return "not set"
    if isinstance(value, str) and value.strip()[:1] in ("{", "["):
        # the registry's report rows carry dicts and lists as JSON text
        try:
            import json

            value = json.loads(value)
        except ValueError:
            pass
    if isinstance(value, bool):
        return "Switched on" if value else "Switched off"
    if isinstance(value, dict):
        vals = list(value.values())
        if vals and all(isinstance(v, bool) for v in vals):
            return f"{sum(vals)} of {len(vals)} switched on"
        return f"{len(vals)} separate {'value' if len(vals) == 1 else 'values'} (see details)"
    if isinstance(value, (list, tuple, set)):
        def _item(x):
            if isinstance(x, (list, tuple)):
                return " and ".join(_item(y) for y in x)
            return humanise(x) if "_" in str(x) else str(x).strip()

        items = [_item(x) for x in value]
        if not items:
            return "none listed"
        shown = ", ".join(items[:4])
        more = f" and {len(items) - 4} more" if len(items) > 4 else ""
        return f"{len(items)} listed: {shown}{more}"
    if isinstance(value, str):
        if tokens[-1] == "period" and value.upper() in ("W", "D", "M"):
            return {"W": "Weekly", "D": "Daily", "M": "Monthly"}[value.upper()]
        if "_" in value:
            return humanise(value)
        return value
    try:
        v = float(value)
    except (TypeError, ValueError):
        return str(value)
    if "seed" in tokens:
        return str(int(v)) if v.is_integer() else str(v)
    if "aed" in tokens:
        return f"AED {v:,.2f}" if (v < 100 and not v.is_integer()) else f"AED {v:,.0f}"
    unit = _UNITS.get(tokens[-1]) or ("day" if "days" in tokens else None)
    if unit:
        if unit in ("km per hour", "milliseconds", "characters"):
            return f"{_num(v)} {unit}"
        return f"{_num(v)} {unit}{'' if abs(v) == 1 else 's'}"
    if tokens[-1] in ("percentile", "quantile") or "percentile" in tokens or "quantile" in tokens:
        return _pct(v) if v <= 1 else _num(v)
    if 0 <= v <= 1 and not v.is_integer() and _FRACTION_WORDS & set(tokens):
        return _pct(v)
    if _MULTIPLE_WORDS & set(tokens) and "std" not in tokens and v >= 0.5:
        return f"{_num(v)} times"
    if v == 0 and ({"tolerance", "share", "rate"} & set(tokens)):
        return "0%"
    return _num(v)


# ------------------------------------------------------------------ checks


def check_title(rule_id: str, name: str = "") -> str:
    return control_text(rule_id, fallback_name=name).title


def missing_words(record: dict[str, Any], required: str = "") -> str:
    """What the file lacks, in words, for a check that could not run in full."""
    support = str(record.get("data_support") or "")
    if support == "EXECUTABLE":
        return ""
    source = record.get("missing_for_unlock") or required
    text = missing_information(source) if source else ""
    if not text:
        text = control_text(str(record.get("rule_id", ""))).needs
    if support == "PARTIAL" and text:
        return f"Ran simplified; the full check also needs {text}"
    return text or "not recorded"


def check_rows(result) -> "list[dict[str, Any]]":
    """One row per check: catalogue facts plus what happened on THIS run.

    ``data_support`` is the EFFECTIVE support on this run (a dataset unlock can
    make a catalogue "couldn't run" check run); the catalogue's own
    classification is kept beside it as ``catalogue_data_support``.
    """
    registry = result.registry
    records = {r.rule_id: r for r in getattr(result.evaluation, "records", [])}
    rows: list[dict[str, Any]] = []
    for base in registry.coverage_rows():
        rid = base["rule_id"]
        rec = records.get(rid)
        text = control_text(rid, fallback_name=base.get("name", ""))
        row = dict(base)
        row["catalogue_data_support"] = base["data_support"]
        if rec is not None:
            row["data_support"] = rec.data_support or base["data_support"]
            row["catalogue_data_support"] = rec.catalogue_data_support or base["data_support"]
            row["support_basis"] = rec.support_basis or "catalogue"
            row["support_reason"] = rec.support_reason or base.get("data_support_reason", "")
            row["missing_for_unlock"] = rec.missing_for_unlock or ""
            row["signal_count"] = int(rec.signal_count or 0)
            row["run_outcome"] = rec.outcome
            row["rule_version"] = rec.rule_version
            row["ceiling_breached"] = bool(rec.ceiling_breached)
        else:
            row["support_basis"] = "catalogue"
            row["support_reason"] = base.get("data_support_reason", "")
            row["missing_for_unlock"] = ""
            row["signal_count"] = 0
            row["run_outcome"] = ""
            row["rule_version"] = registry.get(rid).version
            row["ceiling_breached"] = False
        row["check_title"] = text.title
        row["check_what"] = text.what or text.finding
        row["check_why"] = text.why
        row["family_words"] = family_label(base.get("family"))
        row["owner_words"] = owner_label(base.get("owner"))
        row["basis_words"] = basis_label(row["support_basis"])
        row["missing_words"] = missing_words(row, base.get("required_canonical_fields", ""))
        rows.append(row)
    return rows


def plain_count(n: Any, noun: str) -> str:
    from fwa.presentation import count_phrase

    return count_phrase(n, noun)


def plain_when(value: Any) -> str:
    """An ISO timestamp as "27 Sep 2026, 14:10"."""
    try:
        import pandas as pd

        ts = pd.Timestamp(value)
        if pd.isna(ts):
            return "—"
        try:
            ts = ts.tz_convert(None) if ts.tzinfo else ts
        except Exception:
            pass
        return f"{plain_date(ts)}, {ts:%H:%M}"
    except Exception:
        return str(value) if value else "—"


def plain_seconds(ms: Any) -> str:
    try:
        v = float(ms)
    except (TypeError, ValueError):
        return "not measured"
    if math.isnan(v):
        return "not measured"
    seconds = v / 1000.0
    if seconds < 1:
        return f"{v:,.0f} milliseconds"
    if seconds < 120:
        return f"{seconds:,.1f} seconds"
    minutes = seconds / 60
    if minutes < 120:
        return f"{minutes:,.0f} minutes"
    hours = minutes / 60
    if hours < 48:
        return f"{hours:,.0f} hours"
    return f"{hours / 24:,.0f} days"


__all__ += ["plain_count", "plain_when", "plain_seconds", "status", "plain_number"]
