"""Pluralisation in generated report text.

Every report this artefact writes is quoted in the manuscript and read by
people. ``1 hard/expert control(s)`` is log output; the helper exists so that a
sixty-page report reads as prose, and these tests pin the cases that are easy
to get wrong: zero takes the plural, a consonant + ``y`` becomes ``ies`` but a
vowel + ``y`` does not, and a compound noun pluralises on its head.
"""

from __future__ import annotations

import pytest

from fwa.evaluation.prose import count, plural, verb


@pytest.mark.parametrize("n,expected", [(0, "controls"), (1, "control"), (2, "controls")])
def test_zero_takes_the_plural(n, expected):
    """`0 control` is the mistake `(s)` was invented to avoid making."""
    assert plural("control", n) == expected


@pytest.mark.parametrize("noun,expected", [
    ("entity", "entities"),      # consonant + y
    ("policy", "policies"),
    ("anomaly", "anomalies"),
    ("day", "days"),             # vowel + y: must NOT become "daies"
    ("survey", "surveys"),
    ("analysis", "analyses"),    # irregular
    ("match", "matches"),        # sibilant
    ("box", "boxes"),
    ("control", "controls"),
])
def test_plural_forms(noun, expected):
    assert plural(noun, 2) == expected


def test_count_renders_number_and_noun_together():
    assert count(1, "statistical control") == "1 statistical control"
    assert count(22, "statistical control") == "22 statistical controls"


def test_count_groups_thousands():
    """The reports use this for quantities a reader takes in, not for ids."""
    assert count(20893, "claim") == "20,893 claims"
    assert count(20893, "claim", comma=False) == "20893 claims"


def test_a_compound_noun_pluralises_on_its_head():
    assert count(2, "entity-resolution candidate") == "2 entity-resolution candidates"


@pytest.mark.parametrize("v,one,many", [
    ("is", "is", "are"),
    ("has", "has", "have"),
    ("was", "was", "were"),
    ("carries", "carries", "carry"),
    ("executes", "executes", "execute"),
    ("produces", "produces", "produce"),
])
def test_verbs_agree_with_the_count(v, one, many):
    """`1 control are executable` is still log output; agreement finishes the job."""
    assert verb(v, 1) == one
    assert verb(v, 2) == many
    assert verb(v, 0) == many, "zero takes the plural verb"


def test_a_count_and_its_verb_compose_into_a_sentence():
    for n, expected in [(1, "1 statistical control executes here"),
                        (22, "22 statistical controls execute here")]:
        assert f"{count(n, 'statistical control')} {verb('executes', n)} here" == expected
