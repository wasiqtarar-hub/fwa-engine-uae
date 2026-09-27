"""Shared pieces for the three reviewer pages: Overview, Review queue, Case evidence.

Presentation only. Nothing here changes a disposition, a priority or an
exposure; it words them, links the pages together and runs the guided tour.

* **Links between pages.** The app registers its pages as callables with
  ``st.navigation``, so a page can only be linked by an ``st.Page`` object with
  the same ``url_path``. :func:`link_to` builds one, and only for pages the
  user's role may open (role-based access applies first; a page the role
  cannot open is described in words instead of linked).
* **The guided tour.** "Walk me through a case" sets the ``fwa_guided`` session
  flag that Case evidence already reads; ``fwa_tour_step`` records the step
  the user last saw. The steps run Overview → Review queue → Case evidence.
* **Workload in words.** "With 4 reviewers at 25 cases a day, the queue would
  take about 95 working days."
"""

from __future__ import annotations

import html
import importlib
import math
from typing import Any, Iterable

import pandas as pd
import streamlit as st

from common import chip, guide, mask, pages_for_mode
from fwa.presentation import label, status, subject_word
from fwa.presentation.explain_case import is_plain

__all__ = [
    "TOUR", "TOUR_TOTAL", "days_words", "case_label_func", "tour_toggle", "tour_step", "link_to", "open_case_button",
    "workload_days", "workload_sentence", "subject_for", "case_option_labels",
    "band_icon", "policy_label", "plain_or", "urgency_text", "split_amounts",
]

# ---------------------------------------------------------------------------
# guided tour
# ---------------------------------------------------------------------------

#: (page, title, text). Step numbers are positions in this list, so a step can
#: be added without renumbering by hand.
TOUR: list[tuple[str, str, str]] = [
    ("Overview", "how big the job is",
     "The cards at the top say how many cases need review, how long that would take your team, "
     "and how much money is involved — as two separate figures that are never added together. "
     "Below them are the ten cases to look at first."),
    ("Review Queue", "the queue",
     "Cases are listed most urgent first. Urgency only decides the order; it never changes what "
     "should happen to a claim. The 'What policy suggests' column is what the rules say should "
     "happen next — a suggestion for you to confirm, not a finding of fraud."),
    ("Review Queue", "one case at a glance",
     "Pick a case under 'Read one case'. You see one sentence on what was found, the reasons in "
     "order, and how strong the evidence is. Then open Case evidence to decide."),
    ("Case Evidence", "why it was flagged",
     "The case opens on the plain explanation: headline, reasons most important first, how strong "
     "the evidence is, what it does not mean, and what to check next. You can download it as a "
     "one-page summary."),
    ("Case Evidence", "recording your decision",
     "The highlighted box is where you record what you decided and why. Only roles allowed to "
     "decide see it; everyone else can read the case but not decide it."),
    ("Case Evidence", "the evidence underneath",
     "Each section below is one kind of evidence: what each check found, why the case is urgent, "
     "how the amount was worked out, comparisons with similar providers, and documents. Technical "
     "details (check ids, scores, formulas) are one click away for auditors."),
    ("Case Evidence", "the AI panel",
     "The dashed panel is written by an assistant and is NOT evidence. Every sentence in it must "
     "cite the evidence, and any sentence that could not be traced back was removed. Ask the "
     "assistant whether something is fraud and it will refuse — only a reviewer can decide that."),
]
TOUR_TOTAL = len(TOUR)
GUIDED_KEY = "fwa_guided"
STEP_KEY = "fwa_tour_step"


def tour_toggle(page: str, *, key: str) -> bool:
    """The "Walk me through a case" switch, shared across the three pages.

    The widget key is page-local (Streamlit drops a page's widget state when the
    user leaves it), so it is seeded from ``fwa_guided`` each time and written
    back to it — the flag, not the widget, is what carries across pages.
    """
    if key not in st.session_state:
        st.session_state[key] = bool(st.session_state.get(GUIDED_KEY, False))
    on = st.toggle(
        "Walk me through a case",
        key=key,
        help="Turns on a short guided tour across Overview, Review queue and Case evidence. Each "
             "step explains what you are looking at. Turn it off at any time; it changes nothing "
             "in the data.",
    )
    st.session_state[GUIDED_KEY] = bool(on)
    return bool(on)


def tour_step(page: str, index: int) -> None:
    """Show tour step ``index`` (0-based within this page's steps) if the tour is on."""
    if not st.session_state.get(GUIDED_KEY):
        return
    steps = [i for i, s in enumerate(TOUR) if s[0] == page]
    if index >= len(steps):
        return
    n = steps[index]
    _, title, text = TOUR[n]
    st.session_state[STEP_KEY] = n + 1
    guide(f"Step {n + 1} of {TOUR_TOTAL} — {title}", text)


# ---------------------------------------------------------------------------
# links between pages
# ---------------------------------------------------------------------------


def _can_open(state, page: str) -> bool:
    try:
        allowed = list(state.pages())
    except Exception:
        return False
    return page in allowed and page in pages_for_mode(allowed)


def _page_object(page: str):
    from navigation import PAGE_MODULES, url_path
    from fwa.presentation import page_info

    module = importlib.import_module(PAGE_MODULES[page])
    info = page_info(page)
    return st.Page(module.render, title=info["title"], icon=info.get("icon") or None,
                   url_path=url_path(page))


def link_to(state, page: str, text: str, *, help: str | None = None) -> None:
    """A link to another page, or plain directions when the role can't open it."""
    from fwa.presentation import page_info

    title = page_info(page)["title"]
    if not _can_open(state, page):
        st.caption(f"{text} — the {title} page isn't available to your role or in the current view.")
        return
    try:
        st.page_link(_page_object(page), label=text, icon=page_info(page).get("icon") or None,
                     help=help)
    except Exception:
        st.markdown(f"➜ {html.escape(text)} (open **{html.escape(title)}** in the sidebar).")


def open_case_button(state, case_id: str, *, key: str, text: str = "Open this case") -> None:
    """Make ``case_id`` the current case and go to Case evidence."""
    if not _can_open(state, "Case Evidence"):
        return
    if st.button(text, key=key, help="Makes this the current case and opens it on the Case "
                                     "evidence page, where you can read the evidence and record "
                                     "a decision."):
        st.session_state["fwa_selected_case"] = case_id
        try:
            st.switch_page(_page_object("Case Evidence"))
        except Exception as exc:  # noqa: BLE001
            if type(exc).__name__ in ("StopException", "RerunException", "RerunData"):
                raise
            st.info("This is now the current case. Open **Case evidence** in the sidebar.")


# ---------------------------------------------------------------------------
# workload
# ---------------------------------------------------------------------------


def workload_days(n_cases: int, reviewers: int, per_day: int) -> float:
    capacity = max(int(reviewers) * int(per_day), 1)
    return n_cases / capacity


def days_words(days: float) -> str:
    if days <= 0:
        return "no time at all"
    if days < 1:
        return "less than one working day"
    if days < 10:
        d = round(days, 1)
        d_txt = f"{d:g}"
        return f"about {d_txt} working day" + ("" if d_txt == "1" else "s")
    whole = int(math.ceil(days))
    weeks = whole / 5
    tail = f" (roughly {weeks:.0f} working weeks)" if whole >= 15 else ""
    return f"about {whole:,} working days{tail}"


def workload_sentence(n_cases: int, reviewers: int, per_day: int, *, what: str = "the current queue") -> str:
    days = workload_days(n_cases, reviewers, per_day)
    who = f"{reviewers} reviewer" + ("" if reviewers == 1 else "s")
    return (f"With {who} at {per_day} cases a day ({reviewers * per_day:,} a day in total), "
            f"{what} of {n_cases:,} case{'s' if n_cases != 1 else ''} would take "
            f"**{days_words(days)}**.")


# ---------------------------------------------------------------------------
# wording for cases
# ---------------------------------------------------------------------------

_BAND_ICON = {"P1": "🔴", "P2": "🟠", "P3": "🟡", "P4": "🔵", "P5": "⚪"}


def band_icon(band: Any) -> str:
    return _BAND_ICON.get(str(band), "⚪")


def policy_label(disposition: Any) -> str:
    """Disposition as "icon + plain label" for tables and pickers."""
    phrase = status(disposition)
    return phrase.badge


def urgency_text(band: Any, score: Any) -> str:
    try:
        value = float(score)
        num = f" ({value:.0f}/100)" if not math.isnan(value) else ""
    except (TypeError, ValueError):
        num = ""
    return f"{band_icon(band)} {label(band, 'band')}{num}"


def subject_for(subject_type: Any, subject_id: Any, state, *, assigned: bool = False) -> str:
    return f"{subject_word(subject_type).capitalize()} {mask(subject_id, state, assigned=assigned)}"


def case_option_labels(frame: pd.DataFrame, state) -> dict[str, str]:
    """case_id -> "Provider PROV-1a2b · Coding and documentation · 🔴 Urgent (98/100)"."""
    assigned = set(getattr(state, "assigned_case_ids", []) or [])
    labels: dict[str, str] = {}
    for row in frame.itertuples(index=False):
        labels[row.case_id] = (
            f"{subject_for(row.subject_type, row.subject_id, state, assigned=row.case_id in assigned)}"
            f" · {label(row.scenario_family, 'family')}"
            f" · {urgency_text(row.priority_band, row.priority)}"
        )
    return labels


def case_label_func(result, state):
    """A ``format_func`` that words ANY case id, not only those in the current list.

    A picker whose list shrinks (a filter tightened) can briefly hold a case
    that is no longer listed; formatting it from the case itself keeps its
    label identical before and after, so the widget value stays consistent.
    """
    assigned = set(getattr(state, "assigned_case_ids", []) or [])
    cache: dict[str, str] = {}

    def fmt(case_id: str) -> str:
        if case_id in cache:
            return cache[case_id]
        case = result.cases.cases.get(case_id)
        if case is None:
            text = str(case_id)
        else:
            band = case.priority.band if case.priority is not None else ""
            score = case.priority.value if case.priority is not None else None
            text = (f"{case_id} · "
                    f"{subject_for(case.primary_subject_type, case.primary_subject_id, state, assigned=case_id in assigned)}"
                    f" · {label(case.scenario_family, 'family')} · {urgency_text(band, score)}")
        cache[case_id] = text
        return text

    return fmt


def split_amounts(frame: pd.DataFrame) -> tuple[float, float]:
    """(established, not yet established) — two figures, never added together."""
    if frame is None or frame.empty:
        return 0.0, 0.0
    est = frame["exposure_established"].astype(bool)
    return float(frame.loc[est, "exposure_aed"].sum()), float(frame.loc[~est, "exposure_aed"].sum())


def plain_or(text: Any, fallback: str = "") -> str:
    """``text`` if it reads as plain words (no codes, no rule ids), else ``fallback``."""
    if text is None:
        return fallback
    s = str(text).strip()
    return s if s and is_plain(s) else fallback


def chips_line(items: Iterable[str]) -> None:
    st.markdown(" ".join(items), unsafe_allow_html=True)


def tag(text: str) -> str:
    return f"<span class='fwa-tag'>{html.escape(str(text))}</span>"


__all__ += ["chips_line", "tag", "chip"]
