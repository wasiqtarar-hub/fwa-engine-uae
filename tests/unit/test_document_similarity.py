"""Boilerplate stripping for DOC-02-R01, and the bug that made it a no-op.

DOC-02-R01 compares documents belonging to DIFFERENT members at the same
provider and raises a signal when the patient-specific text is a near-
duplicate. Everything it does depends on :func:`_strip_boilerplate` leaving
the patient-specific text behind.

It did not. The template patterns each end in ``.*`` — meaning "the rest of
this line" — but the alternation was compiled with ``re.DOTALL``, so ``.``
matched newlines too and ``DISCHARGE SUMMARY.*`` ran greedily from the first
heading to the end of the document. Every document stripped to the empty
string, every token set was empty, the guard in the control dropped every
document before the comparison, and the control returned no signals on every
dataset it was ever run against.

That failure mode is invisible from the outside: a control that finds nothing
looks exactly like a control with nothing to find, which is why these tests
assert on the stripped text directly rather than on the signal count.
"""

from __future__ import annotations

import re

import pytest

from fwa.nlp.controls import _strip_boilerplate
from fwa.nlp.synthetic import _BODY_EN, _FOOTER_EN, _HEADER_EN

EXTRA = "Vital signs remained within expected limits throughout."
OTHER_EXTRA = "The treating team documented daily progress notes."


def _document(member: str, claim: str, extra: str = EXTRA, dx: str = "Essential hypertension",
              los: int = 4) -> str:
    return (_HEADER_EN + _BODY_EN + _FOOTER_EN).format(
        marker="SYNTHETIC", generated="2026-01-01", provider="H0001",
        member=member, claim=claim, admit="2024-03-11", discharge="2024-03-15",
        los=los, los_text=los, dx_code="I10", dx_text=dx, pharm="31%", extra=extra,
    )


def _tokens(text: str) -> frozenset[str]:
    """The control's own tokenisation, so the tests measure what it measures."""
    return frozenset(t for t in re.findall(r"\w+", text.lower()) if len(t) > 2)


def _similarity(a: str, b: str) -> float:
    ta, tb = _tokens(_strip_boilerplate(a)), _tokens(_strip_boilerplate(b))
    union = len(ta | tb)
    return len(ta & tb) / union if union else 0.0


def test_stripping_leaves_the_narrative_behind():
    """The regression test for the bug itself: the result must not be empty."""
    stripped = _strip_boilerplate(_document("P000001", "C0000001"))
    assert stripped, "every document stripped to nothing; the comparison has no input"
    assert "admitted" in stripped
    assert "Essential hypertension" in stripped


def test_stripping_removes_the_template_lines():
    """The headings and reference lines are the template, not the clinician."""
    stripped = _strip_boilerplate(_document("P000001", "C0000001"))
    for template_line in ("DISCHARGE SUMMARY", "Facility reference", "Patient reference",
                          "Claim reference", "Admission date", "CLINICAL COURSE",
                          "MEDICATION AND SUPPLIES", "DISPOSITION", "END OF"):
        assert template_line not in stripped


def test_identical_narratives_for_different_members_are_near_duplicates():
    """The case the control exists to catch."""
    a = _document("P000001", "C0000001")
    b = _document("P000999", "C0000002")          # different member and claim
    assert _similarity(a, b) > 0.92


def test_a_different_narrative_is_not_a_near_duplicate():
    """And the case it must not catch, or the threshold means nothing.

    Two admissions that genuinely differ in their write-up have to fall below
    the threshold, otherwise "near-duplicate" would just mean "same template"
    and the control would flag the whole corpus.
    """
    a = _document("P000001", "C0000001", extra=EXTRA, dx="Essential hypertension", los=4)
    b = _document("P000999", "C0000002", extra=OTHER_EXTRA, dx="Acute appendicitis", los=9)
    assert _similarity(a, b) <= 0.92


@pytest.mark.parametrize("pattern", ["DISCHARGE SUMMARY.*", "Facility reference.*"])
def test_the_template_patterns_are_line_bounded(pattern):
    """A pattern ending in ``.*`` must stop at the newline, not eat the file.

    Asserting on the compiled behaviour rather than on the flags, so that the
    property survives someone rewriting the patterns in another style.
    """
    compiled = re.compile(pattern)
    text = "DISCHARGE SUMMARY (SYNTHETIC)\nFacility reference: H0001\nThe patient was admitted."
    assert "The patient was admitted." in compiled.sub(" ", text)
