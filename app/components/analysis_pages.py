"""Plain-language helpers for the three analysis pages.

Shared by ``pages/models.py`` (Pattern-finding models), ``pages/provider_analytics.py``
(Compare hospitals) and ``pages/network.py`` (Connections). Presentation only:
every function here turns an engine object into words and never changes it.

The wording lives in ``config/plain_language/analysis_pages.yaml`` under the
``analysis_pages:`` section, so it can be improved without a code change.
"""

from __future__ import annotations

import html
import math
import re
from functools import lru_cache
from typing import Any, Iterable

import yaml

from fwa.presentation import aed, feature_phrase, pct, plain_month, ratio_words, status
from fwa.presentation.plain_language import VOCAB_DIR

__all__ = [
    "words", "model_name", "gate_checklist", "overall_verdict", "drift_rows",
    "fmt_unit", "peer_group_words", "match_sentence", "node_word", "edge_words",
    "change_title", "safe_float", "checklist_html",
]


# ------------------------------------------------------------------ vocabulary


@lru_cache(maxsize=1)
def _load() -> dict[str, Any]:
    try:
        raw = yaml.safe_load((VOCAB_DIR / "analysis_pages.yaml").read_text(encoding="utf-8")) or {}
    except Exception:  # a missing wording file must never break a page
        raw = {}
    return raw.get("analysis_pages", {}) or {}


def words(section: str) -> dict[str, Any]:
    return dict(_load().get(section, {}) or {})


def model_name(name: str) -> str:
    from fwa.presentation import model_label

    return model_label(name)


def safe_float(value: Any) -> float | None:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(v) or math.isinf(v) else v


def fmt_unit(value: Any, unit: str) -> str:
    """A value in the unit a reader expects: AED, AED a day, a percentage…"""
    v = safe_float(value)
    if v is None:
        return "not available"
    if unit == "AED":
        return aed(v)
    if unit == "AED/day":
        return f"{aed(v)} a day"
    if unit == "ratio":
        return pct(v)
    if unit == "claims/member":
        return f"{v:,.1f} claims per patient"
    return f"{v:,.2f}"


# ------------------------------------------------------------------ promotion gate


_ICON = {"PASS": "✅", "FAIL": "❌", "NOT_ASSESSABLE": "❓"}


def _nums(text: str) -> list[int]:
    return [int(n.replace(",", "")) for n in re.findall(r"\d[\d,]*", text or "")]


def _gate_answer(criterion, *, model: str, capacity: int, drift: Iterable[Any] = ()) -> str:
    """One line answering the plain question. Markdown (bold Yes/No)."""
    name, st_, ev = criterion.name, criterion.status, criterion.evidence or ""
    value, thr = safe_float(criterion.value), safe_float(criterion.threshold)

    if name.startswith("Temporal holdout"):
        if st_ == "PASS":
            return ("**Yes.** It learned from earlier claims and was tested on later claims from "
                    "different providers; no provider appears in both.")
        if st_ == "FAIL":
            n = _nums(ev)
            return (f"**No.** {n[0] if n else 'Some'} providers appear in both the claims it learned "
                    "from and the claims it was tested on, so the test is not fair.")
        return "**Can't tell yet:** there is no record of how the claims were split."

    if name.startswith("Prospective lift"):
        need = f"it needed at least {thr:.1f} times the scorecard's" if thr else "it needed to beat it"
        if value is None and criterion.value is not None:          # scorecard found none at all
            share = "higher than the scorecard's, which found none"
        elif (value or 0) <= 0:
            share = "zero: none of its top cases were real issues"
        else:
            share = f"{ratio_words(value)} the simple scorecard's"
        if st_ == "PASS":
            return (f"**Yes.** Among the {capacity:,} cases the team can review, its share of real "
                    f"issues is {share} ({need}). \"Real issues\" here come from past investigation "
                    "labels, so this is a best case, not a promise.")
        if st_ == "FAIL":
            return (f"**No**, not at the capacity that matters. Among the {capacity:,} cases the team "
                    f"can review, its share of real issues is {share}; {need}.")
        if "Only" in ev:
            return ("**Can't tell yet:** too few of the providers it scored have a known outcome — "
                    "fewer than the number the team can review.")
        if "no ranking" in ev:
            return ("**Can't tell yet:** the simple scorecard could not rank these providers, so "
                    "there is nothing to compare against.")
        return ("**Can't tell yet:** we can't measure this yet because no reviewer has recorded a "
                "decision to compare its flags with.")

    if name.startswith("Calibration"):
        if st_ == "PASS":
            return ("**Yes.** How sure its scores look roughly matches how often its flags turned "
                    "out to be real.")
        if st_ == "FAIL":
            return (f"**No.** Its scores are not a reliable guide to how likely a flag is to be real "
                    f"(off by {pct(value)} on average; up to {pct(thr)} is allowed). That is expected: "
                    "a pattern-finder's score says how unusual a claim is, not how likely it is to be "
                    "a problem.")
        if "too few" in ev:
            return "**Can't tell yet:** too few scored providers have a known outcome to check this."
        return ("**Can't tell yet:** we can't measure this yet because no reviewer has recorded a "
                "decision to check its confidence against.")

    if name.startswith("Drift"):
        if st_ == "PASS":
            warn = [d for d in drift if getattr(d, "status", "") == "WARN"]
            return ("**Yes.** The claims it scores look like the ones it learned from"
                    + (", although one measure is close to the limit." if warn else "."))
        if st_ == "FAIL":
            return ("**No.** The claims it scores now differ clearly from the ones it learned from, "
                    "so its sense of \"unusual\" may be out of date. Some of this is built into the "
                    "file: measures based on a provider's or patient's earlier claims are nearly empty "
                    "at the start of the file and fuller later.")
        return "**Can't tell yet:** no stability check ran."

    if name.startswith("Explanation"):
        n = _nums(ev)
        if st_ == "PASS":
            detail = f" ({n[0]} of the {n[1]} reasons sampled)" if len(n) >= 2 else ""
            return ("**Yes.** Its reasons can be shown in real units beside a typical value, so a "
                    f"reviewer can check them{detail}.")
        if st_ == "FAIL":
            detail = f"Only {n[0]} of {n[1]}" if len(n) >= 2 else "Too few"
            return (f"**No.** {detail} of its reasons could be shown in real units beside a typical "
                    "value.")
        return "**Can't tell yet:** it flagged no claims, so there was nothing to explain."

    # an unknown criterion: fall back to the status label, never to raw evidence
    return f"**{status(st_).label}.**"


def gate_checklist(verdict, *, capacity: int, drift: Iterable[Any] = ()) -> list[dict[str, str]]:
    """The five gate criteria as plain questions with ✅/❌/❓ and a one-line answer."""
    questions = words("gate_questions")
    rows = []
    for c in verdict.criteria:
        rows.append({
            "icon": _ICON.get(c.status, "❓"),
            "question": questions.get(c.name, c.name),
            "answer": _gate_answer(c, model=verdict.model, capacity=capacity, drift=drift),
            "status": c.status,
        })
    return rows


def checklist_html(rows: list[dict[str, str]]) -> str:
    """Render checklist rows as HTML (answers may carry **bold** markdown)."""
    def md(s: str) -> str:
        return re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", html.escape(s))

    items = "".join(
        f"<li style='list-style:none;margin:.35rem 0'><span style='margin-right:.4rem'>{r['icon']}</span>"
        f"<strong>{html.escape(r['question'])}</strong><br>"
        f"<span style='margin-left:1.6rem;display:inline-block'>{md(r['answer'])}</span></li>"
        for r in rows
    )
    return f"<ul style='padding-left:0;margin:.3rem 0'>{items}</ul>"


def overall_verdict(verdict) -> tuple[str, str, str]:
    """(icon, headline, sentence) for a model's overall gate verdict."""
    if verdict.promoted:
        return ("✅", "May be proposed for use; a separate person must approve",
                "It passed all five questions. An analyst may propose it for use, and a policy "
                "owner who did not make the proposal must approve it before anything changes.")
    failed = sum(1 for c in verdict.criteria if c.status == "FAIL")
    unknown = sum(1 for c in verdict.criteria if c.status == "NOT_ASSESSABLE")
    parts = []
    if failed:
        parts.append(f"{failed} {'answer is' if failed == 1 else 'answers are'} no")
    if unknown:
        parts.append(f"{unknown} can't be judged yet")
    detail = " and ".join(parts) or "not every question was answered yes"
    return ("🚫", "Not ready to be trusted for real decisions",
            f"{detail[:1].upper() + detail[1:]}. It stays in watch-only mode: its results are "
            "recorded but not used for real payment decisions. Nothing is lost: the simple scorecard "
            "keeps running as the explainable fallback.")


# ------------------------------------------------------------------ drift


def drift_rows(results: Iterable[Any], *, warn: float, breach: float) -> list[dict[str, str]]:
    """Each drift monitor as a question, a status phrase and one plain sentence."""
    monitors = words("drift_monitors")
    rows = []
    for d in results:
        entry = monitors.get(d.monitor, {}) or {}
        question = entry.get("question") or d.monitor
        v = safe_float(d.value)
        name = d.monitor
        if name.startswith("Population"):
            top = ""
            detail = getattr(d, "detail", None)
            if detail is not None and not detail.empty and "feature" in detail.columns:
                top = feature_phrase(str(detail.iloc[0]["feature"]))
            if v is None:
                sentence = "Can't be measured: there were no comparable claim details."
            else:
                band = ("little change" if d.status == "OK" else
                        "a change worth watching" if d.status == "WARN" else "a big change")
                sentence = (f"The biggest shift scores {v:.2f}, which counts as {band} (below "
                            f"{warn:g} is little change, {warn:g}–{breach:g} is worth watching, "
                            f"above {breach:g} is a big change)"
                            + (f". It is in {top}." if top else "."))
        elif name.startswith("Score-distribution"):
            sentence = ("Can't be measured yet: it needs the scores from a previous run to compare "
                        "with, and this is the first run." if v is None else
                        f"The scores moved by {v:.2f} since the previous run.")
        elif name.startswith("Alert volume"):
            if d.warn_threshold is None:
                sentence = (f"It flagged {int(v or 0):,} claims. No expected number has been set for "
                            "this model, so there is nothing to compare against.")
            else:
                sentence = f"It flagged {ratio_words(v)} the expected number of claims."
        elif name.startswith("Review yield"):
            sentence = ("We can't measure this yet because no reviewer has recorded a decision on "
                        "its flags." if v is None else f"{pct(v)} of its reviewed flags were confirmed.")
        else:
            sentence = status(d.status).meaning
        phrase = status(d.status)
        rows.append({"question": question, "status": d.status, "label": phrase.label,
                     "icon": phrase.icon or "❓", "sentence": sentence})
    return rows


# ------------------------------------------------------------------ peers


def peer_group_words(level_name: Any) -> str:
    names = words("peer_group_names")
    key = str(level_name)
    return str(names.get(key) or key.replace("_", " ").lower())


# ------------------------------------------------------------------ network


def node_word(kind: str) -> str:
    return str(words("node_types").get(kind, str(kind).replace("_", " ").capitalize()))


def edge_words(edge_type: str) -> tuple[str, str]:
    entry = words("edge_types").get(edge_type) or {}
    return (str(entry.get("label") or str(edge_type).replace("_", " ").capitalize()),
            str(entry.get("meaning") or ""))


def match_sentence(fields: dict[str, Any]) -> str:
    templates = words("match_fields")
    parts = []
    for key, value in (fields or {}).items():
        tpl = templates.get(key)
        v = safe_float(value)
        if tpl is None or v is None:
            continue
        try:
            parts.append(tpl.format(value=v, pct=pct(v)))
        except Exception:
            continue
    return "; ".join(parts) or "similar billing and connections"


# ------------------------------------------------------------------ change points


def change_title(who: str, cp, metric_words: str = "average claim value") -> str:
    """A chart title that states the finding: "Hospital X's average claim value jumped in Feb 2026"."""
    pre, post = safe_float(cp.pre_change_mean), safe_float(cp.post_change_mean)
    when = plain_month(cp.change_date)
    if pre and post is not None and pre > 0:
        r = post / pre
        verb = ("jumped" if r >= 1.5 else "rose" if r > 1 else "dropped" if r <= 0.67 else "fell")
    else:
        verb = "rose" if cp.direction == "increase" else "fell"
    return f"{who}'s {metric_words} {verb} in {when}"
