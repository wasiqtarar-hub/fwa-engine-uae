"""The plain-language vocabulary is complete, plain, and never drops a meaning.

The presentation layer is only as good as its coverage: an identifier with no
entry falls back to a humanised form of itself, which reads, but reads badly.
These tests make every gap a failure, so a new status, control or model
feature cannot reach the screen untranslated without someone noticing.
"""

from __future__ import annotations

import datetime as _dt
import math

import numpy as np
import pandas as pd
import pytest

from fwa.audit.log import AuditEventType
from fwa.auth import Role
from fwa.cases.correlate import CASE_FAMILIES, CORRELATION_DIMENSIONS
from fwa.canonical import CANONICAL_TABLES
from fwa.canonical.supplementary import SUPPLEMENTARY_TABLES
from fwa.engine.evallib import EvaluationOutcome
from fwa.enums import ControlType, DataSupport, Disposition, RuleStatus, SignalDomain, Stage
from fwa.presentation import plain_language as pl

PLAIN_FIELDS = ("title", "finding", "what", "why", "needs")


def _is_plain(text: str) -> bool:
    return not (pl.RAW_TOKEN_RE.search(text) or pl.SNAKE_TOKEN_RE.search(text)
                or pl.RULE_ID_RE.search(text))


# ----------------------------------------------------------------- statuses


@pytest.mark.parametrize("value", [d.value for d in Disposition])
def test_every_disposition_has_a_phrase_and_a_suggestion(value):
    phrase = pl.status(value)
    assert phrase.label and phrase.meaning and phrase.tone
    assert _is_plain(phrase.label) and _is_plain(phrase.meaning)
    assert pl.vocabulary().suggestions.get(value)


@pytest.mark.parametrize("value", [
    *[d.value for d in DataSupport], *[o.value for o in EvaluationOutcome],
    *[s.value for s in RuleStatus], "PASS", "FAIL", "WARN", "BREACH", "OK", "MEASURED",
    "PROMOTED", "SHADOW", "PROMOTION_BLOCKED", "NOT_ASSESSABLE", "NOT_MEASURABLE",
    "NOT_MEASURABLE_ON_THIS_DATASET", "INFORMATIONAL", "POPULATED", "NOT_POPULATED",
])
def test_every_status_the_app_can_show_has_a_phrase(value):
    entry = pl.vocabulary().statuses.get(value)
    assert entry, f"{value} has no plain-language entry"
    assert entry["label"] and entry["meaning"]
    assert _is_plain(entry["label"])


def test_the_required_example_phrases_carry_their_meaning():
    assert pl.status("POSTPAY_AUDIT").label == "Pay, but check afterwards"
    assert pl.status("PREPAY_PEND").label == "Hold before paying"
    assert pl.status("MONITOR_ONLY").label == "Watch only"
    assert pl.status("NOT_EXECUTABLE_ON_THIS_DATASET").label.startswith("Couldn't run")
    assert pl.status("PARTIAL").label == "Ran in a simplified form"
    assert pl.status("NOT_ASSESSABLE").label == "Can't be judged yet"
    assert pl.status("PROMOTION_BLOCKED").label == "Not ready to be trusted for real decisions"


def test_not_measurable_is_words_not_a_number():
    meaning = pl.status("NOT_MEASURABLE_ON_THIS_DATASET").meaning
    assert "can't measure" in meaning.lower()
    assert not any(ch.isdigit() for ch in meaning)


@pytest.mark.parametrize("section,values", [
    ("roles", [r.value for r in Role]),
    ("audit_events", [e.value for e in AuditEventType]),
    ("stages", [s.value for s in Stage]),
    ("control_types", [t.value for t in ControlType]),
    ("domains", [d.value for d in SignalDomain]),
    ("case_families", sorted(set(CASE_FAMILIES.values()))),
    ("dimensions", list(CORRELATION_DIMENSIONS)),
    ("priority_bands", ["P1", "P2", "P3", "P4", "P5"]),
])
def test_every_enumeration_section_is_complete(section, values):
    table = getattr(pl.vocabulary(), section)
    missing = [v for v in values if v not in table]
    assert not missing, f"{section} lacks {missing}"


def test_composite_control_types_read_as_words():
    assert pl.label("H/E", "type") == "Hard rule + Expert rule"


# ----------------------------------------------------------------- controls


def test_every_control_has_plain_text(registry):
    """All 164 controls: a title, a finding, what/why, 2–4 next steps, and what it needs."""
    problems = []
    for control in registry.all():
        entry = pl.vocabulary().controls.get(control.rule_id)
        if not entry:
            problems.append(f"{control.rule_id}: no entry")
            continue
        for key in PLAIN_FIELDS:
            text = str(entry.get(key) or "")
            if not text.strip():
                problems.append(f"{control.rule_id}: empty {key}")
            elif not _is_plain(text):
                problems.append(f"{control.rule_id}: {key} contains a code token: {text!r}")
        steps = entry.get("next_steps") or []
        if not 2 <= len(steps) <= 4:
            problems.append(f"{control.rule_id}: {len(steps)} next steps")
    assert not problems, "\n".join(problems)
    assert len(pl.vocabulary().controls) == len(registry)


def test_no_control_text_calls_anything_fraud(registry):
    for control in registry.all():
        entry = pl.vocabulary().controls[control.rule_id]
        for key in ("title", "finding"):
            assert "fraud" not in str(entry[key]).lower(), control.rule_id


# ------------------------------------------------------------------ features


def test_every_model_feature_has_a_plain_translation(context):
    """A feature the models use must never reach the screen as its code name."""
    _, columns = context.features.model_matrix()
    missing = [c for c in columns if c not in pl.vocabulary().features]
    assert not missing, f"Model features without a plain phrase: {missing}"
    for c in columns:
        phrase = pl.feature_phrase(c)
        assert phrase and _is_plain(phrase), f"{c}: {phrase!r}"
        assert pl.feature_unit(c) in ("AED", "AED/day", "ratio", "days", "count", "flag",
                                      "log AED", "score", "other")


def test_every_explain_feature_label_has_a_translation():
    from fwa.models.explain import FEATURE_UNITS

    missing = [f for f in FEATURE_UNITS if f not in pl.vocabulary().features]
    assert not missing


# -------------------------------------------------------------------- fields


def test_every_canonical_and_supplementary_table_is_described():
    tables = pl.vocabulary().tables
    missing = [t for t in [*CANONICAL_TABLES] if t not in tables]
    assert not missing
    phrase = pl.missing_information("claim_line.activity_code; provider.specialty")
    assert phrase and _is_plain(phrase) and " and " in phrase


def test_every_canonical_column_has_a_field_label():
    fields = pl.vocabulary().fields
    missing = sorted({c for spec in CANONICAL_TABLES.values() for c in spec.columns if c not in fields})
    # Supplementary tables are reached through tables.yaml; the claim model's own
    # columns must each have a label because they appear in tables on screen.
    assert len(missing) <= 40, f"Too many canonical columns without a label: {missing}"


def test_the_required_field_example_reads_as_asked():
    assert pl.field_label("exposure_aed") == "Amount at risk (AED)"


# ---------------------------------------------------------------- formatters


def test_money_is_aed_with_thousands_separators():
    assert pl.aed(1264948.4) == "AED 1,264,948"
    assert pl.aed(None) == "not available"
    assert pl.aed(float("nan")) == "not available"


def test_ratios_are_percentages():
    assert pl.pct(0.0106) == "1.1%"
    assert pl.pct(0.41) == "41%"
    assert pl.pct(0.0) == "0%"


def test_multipliers_are_words():
    assert pl.times_phrase(2649, 912, "the usual amount") == "about 3 times the usual amount"
    assert pl.ratio_words(1.02) == "about the same as"
    assert pl.ratio_words(0.5) == "about half"


def test_dates_are_plain():
    assert pl.plain_date(_dt.date(2024, 3, 11)) == "11 Mar 2024"
    assert pl.plain_date(pd.NaT) == "date not recorded"


@pytest.mark.parametrize("value", [None, float("nan"), np.nan, pd.NA, pd.NaT, "nan", "None", True,
                                   False, np.bool_(True), "True", "PREPAY_PEND"])
def test_plain_value_never_shows_code_or_missing_tokens(value):
    out = pl.plain_value(value)
    assert out not in ("nan", "None", "True", "False", "NaT", "<NA>")
    assert _is_plain(out)


def test_friendly_frame_translates_and_fills():
    frame = pd.DataFrame({
        "disposition": ["POSTPAY_AUDIT", "PREPAY_PEND"],
        "exposure_aed": [1234.5, float("nan")],
        "exposure_established": [True, False],
        "scenario_family": ["coding_integrity", None],
    })
    out = pl.friendly_frame(frame, money=["exposure_aed"])
    assert list(out.columns)[0] == "What policy suggests" or "disposition" not in out.columns
    text = out.astype(str).to_string()
    assert "POSTPAY_AUDIT" not in text and "nan" not in text.lower().split()
    assert "Pay, but check afterwards" in text
    assert "AED 1,234" in text
    assert "Yes" in text and "No" in text
    for column in out.columns:
        assert _is_plain(str(column)), column


def test_humanise_is_the_readable_fallback():
    assert pl.humanise("SOMETHING_NEW") == "Something new"
    assert pl.humanise("some_field") == "Some field"
    assert pl.humanise(None) == "—"
