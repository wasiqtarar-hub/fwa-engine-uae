"""Small helpers that keep generated reports reading as prose.

Every report this package writes is a deliverable in its own right, quoted
directly in the manuscript and read by people rather than parsed by machines.
``1 hard/expert control(s)`` is log output; ``1 hard/expert control`` is a
sentence. The difference is cosmetic in isolation and not cosmetic across a
sixty-page report, so the pluralisation lives in one place instead of being
sidestepped with ``(s)`` at every call site.
"""

from __future__ import annotations

__all__ = ["plural", "count", "verb"]

#: Words whose plural is not formed by appending "s".
_IRREGULAR = {
    "analysis": "analyses",
    "entity": "entities",
    "policy": "policies",
    "property": "properties",
    "summary": "summaries",
    "category": "categories",
    "anomaly": "anomalies",
}


def plural(noun: str, n: float) -> str:
    """The noun as it should read beside ``n``.

    ``n`` is compared against exactly one, so 0 and 2 both take the plural,
    as English requires (``0 controls``, not ``0 control``).
    """
    if n == 1:
        return noun
    if noun in _IRREGULAR:
        return _IRREGULAR[noun]
    if noun.endswith("y") and noun[-2:-1] not in "aeiou":
        return noun[:-1] + "ies"
    if noun.endswith(("s", "x", "z", "ch", "sh")):
        return noun + "es"
    return noun + "s"


def count(n: float, noun: str, *, comma: bool = True) -> str:
    """``count(1, "control") -> '1 control'``; ``count(22, "control") -> '22 controls'``.

    ``comma`` groups thousands, which is what the reports want everywhere the
    number is a quantity a reader is meant to take in rather than an id.
    """
    figure = f"{n:,}" if comma and isinstance(n, int) else f"{n}"
    return f"{figure} {plural(noun, n)}"


#: Verbs the reports use, singular form -> plural form.
_VERBS = {"is": "are", "has": "have", "was": "were", "does": "do"}


def verb(v: str, n: float) -> str:
    """The verb form that agrees with ``n``.

    ``count`` alone is not enough: "1 control **are** executable" is still log
    output. Pass the third-person singular form and this returns the form the
    sentence needs, so a count of one reads as a sentence and a count of zero
    or two still does.
    """
    if n == 1:
        return v
    if v in _VERBS:
        return _VERBS[v]
    if v.endswith("ies"):
        return v[:-3] + "y"
    if v.endswith(("ses", "xes", "zes", "ches", "shes")):
        return v[:-2]
    return v[:-1] if v.endswith("s") else v
