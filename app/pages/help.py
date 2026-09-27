"""Help — task-based help, a page-by-page guide, who sees what, and a searchable glossary.

Written for a claims reviewer, not an engineer. The answers to "How do I…?"
questions are the task sections of ``docs/USER_GUIDE.md``: one source, two
surfaces, so the guide in the tool cannot drift out of step with the guide on
disk. The page-by-page list and the role descriptions come from the shared
plain-language vocabulary, and the glossary is ``fwa.presentation.glossary()``.

Rendering the guide is not quite as simple as handing the markdown to
Streamlit. Markdown image references are relative paths, and Streamlit resolves
them against the served origin rather than the file's own directory, so every
screenshot would render as a broken image. :func:`_render` walks the markdown,
takes the image references out and renders those through ``st.image`` with the
resolved path, passing everything else through untouched.
"""

from __future__ import annotations

import html
import re

import streamlit as st

from common import (
    ROOT, banner, boundary_note, empty_state, is_simple, page_header, require_page,
    session_banner,
)
from components.data_help import words
from fwa.auth import PAGE_PERMISSIONS, Permission, Role
from fwa.presentation import glossary, label, page_info, vocabulary

GUIDE = ROOT / "docs" / "USER_GUIDE.md"
DOCS = GUIDE.parent

#: ``![alt](path)`` on a line of its own.
_IMAGE = re.compile(r"^\s*!\[(?P<alt>[^\]]*)\]\((?P<src>[^)\s]+)\)\s*$")

#: Guide sections shown as answers on the first tab, in guide order.
_TASK = re.compile(r"^(How do I|What does|What do the|Why couldn|Can I trust|The one rule)", re.I)


# ---------------------------------------------------------------------------
# page
# ---------------------------------------------------------------------------


def render() -> None:
    state = require_page("Help", Permission.VIEW_HELP)
    session_banner(state)
    page_header("Help")
    banner()
    boundary_note()

    _role_card(state)

    text = GUIDE.read_text(encoding="utf-8") if GUIDE.exists() else ""
    sections = _split_sections(text) if text else {}

    tab_tasks, tab_pages, tab_roles, tab_glossary, tab_guide = st.tabs(
        ["How do I…?", "Page by page", "Who sees what", "Glossary", "Full user guide"])
    with tab_tasks:
        _tasks(sections)
    with tab_pages:
        _pages(state)
    with tab_roles:
        _roles(state)
    with tab_glossary:
        _glossary()
    with tab_guide:
        _full_guide(text, sections)


# ---------------------------------------------------------------------------
# sections
# ---------------------------------------------------------------------------


def _role_card(state) -> None:
    role = state.role.value
    sees = vocabulary().role_sees.get(role, "")
    can = words("role_can_do").get(role, "")
    titles = [page_info(p)["title"] for p in state.pages()]
    st.markdown(
        f"<div class='fwa-card'><h4>You are signed in as: {html.escape(label(role, 'role'))}</h4>"
        + (f"<p><strong>You see:</strong> {html.escape(sees)}</p>" if sees else "")
        + (f"<p><strong>You can:</strong> {html.escape(can)}</p>" if can else "")
        + f"<p class='fwa-sub'>Pages you can open: {html.escape(', '.join(titles))}.</p></div>",
        unsafe_allow_html=True,
    )


def _tasks(sections: dict[str, str]) -> None:
    start = vocabulary().start_here
    if start:
        st.markdown("### Start here")
        cols = st.columns(len(start))
        for col, step in zip(cols, start):
            col.markdown(
                f"<div class='fwa-card'><h4>{html.escape(str(step.get('title', '')))}</h4>"
                f"<p>{html.escape(str(step.get('body', '')))}</p></div>",
                unsafe_allow_html=True,
            )

    tasks = [(title, body) for title, body in sections.items() if _TASK.match(title)]
    if not tasks:
        empty_state("The answers are not in this copy of the app",
                    "They come from the user guide file (docs/USER_GUIDE.md), which is missing. "
                    "Ask an administrator to restore it. The glossary tab still works.")
        return
    st.markdown("### Common questions")
    st.caption("Click a question to open the answer.")
    for i, (title, body) in enumerate(tasks):
        with st.expander(title, expanded=(i == 0)):
            # Drop the section's own heading line; the expander carries it.
            _render("\n".join(body.splitlines()[1:]).strip().rstrip("-").strip())


def _pages(state) -> None:
    st.markdown("### What each page is for")
    mine = set(state.pages())
    st.caption("Pages marked 'Simple view' are listed when the Simple view switch in the sidebar "
               "is on. " + ("It is on now." if is_simple() else "It is off now, so all your "
                            "pages are listed."))
    for name in PAGE_PERMISSIONS:
        info = page_info(name)
        tags = ["Simple view" if info.get("simple") else "Advanced view only",
                "you can open it" if name in mine else "not available to your role"]
        st.markdown(
            f"**{html.escape(str(info.get('icon', '')))} {html.escape(str(info['title']))}** "
            f"<span class='fwa-sub'>({html.escape(' · '.join(tags))})</span><br>"
            f"{html.escape(str(info.get('purpose', '')))}",
            unsafe_allow_html=True,
        )


def _roles(state) -> None:
    st.markdown("### What each role sees and can do")
    st.caption("Your role decides which pages you can open. The Simple/Advanced switch only "
               "decides how many of them are listed.")
    sees = vocabulary().role_sees
    can = words("role_can_do")
    for role in Role:
        mark = " (you)" if role == state.role else ""
        st.markdown(
            f"**{html.escape(label(role.value, 'role'))}{mark}**  \n"
            f"Sees: {html.escape(sees.get(role.value, '—'))}  \n"
            f"Can: {html.escape(can.get(role.value, '—'))}"
        )
    st.caption("Some rules are enforced by the tool itself: the person who writes a check cannot "
               "approve it for real use, and the person who proposes a model cannot approve it.")


def _matches(entry: dict, query: str) -> bool:
    if not query:
        return True
    also = entry.get("also_called") or []
    if isinstance(also, str):
        also = [also]
    haystack = " ".join([str(entry.get("term", "")), " ".join(map(str, also)),
                         str(entry.get("short", ""))]).lower()
    return all(word in haystack for word in query.lower().split())


def _glossary() -> None:
    st.markdown("### Glossary")
    entries = sorted(glossary().items(), key=lambda kv: str(kv[1].get("term", kv[0])).lower())
    query = st.text_input(
        "Search the glossary", key="help_glossary_q",
        placeholder="For example: shadow, peer group, drift, Isolation Forest",
        help="What it does: shows only the terms whose name, other names or one-line meaning "
             "contain every word you type. When to use it: whenever a page uses a word you "
             "don't know. Clear the box to see every term.",
    )
    hits = [(k, e) for k, e in entries if _matches(e, query.strip())]
    st.caption(f"Showing {len(hits)} of {len(entries)} terms.")
    if not hits:
        empty_state("No term matches that search",
                    "Try a shorter word, or clear the box to see every term.")
        return
    for _, entry in hits:
        term = str(entry.get("term", ""))
        also = entry.get("also_called") or []
        if isinstance(also, str):
            also = [also]
        with st.container(border=True):
            st.markdown(
                f"**{html.escape(term)}**"
                + (f" <span class='fwa-sub'>(also called: {html.escape(', '.join(map(str, also)))})"
                   f"</span>" if also else "")
                + f"<br>{html.escape(str(entry.get('short', '')))}",
                unsafe_allow_html=True,
            )
            parts = []
            if entry.get("explanation"):
                parts.append(html.escape(str(entry["explanation"])))
            if entry.get("analogy"):
                parts.append(f"<em>Everyday comparison:</em> {html.escape(str(entry['analogy']))}")
            if entry.get("in_this_app"):
                parts.append(f"<em>In this app:</em> {html.escape(str(entry['in_this_app']))}")
            if parts:
                st.markdown("<br><br>".join(parts), unsafe_allow_html=True)


def _full_guide(text: str, sections: dict[str, str]) -> None:
    if not text:
        empty_state("The user guide is not in this copy of the app",
                    "docs/USER_GUIDE.md is missing. The guide is a file, not text built into "
                    "this page, so it can go missing, and saying so is better than an empty page.")
        return
    left, right = st.columns([3, 1], gap="large")
    with left:
        chosen = st.selectbox(
            "Section of the guide", list(sections), key="help_guide_section",
            help="What it does: shows one section of the user guide at a time. When to change "
                 "it: pick the topic you want to read; the whole guide is also downloadable.",
        )
    with right:
        st.download_button("Download the guide", text, file_name="USER_GUIDE.md",
                           mime="text/markdown", width="stretch",
                           help="Saves the whole guide as a text file you can read or print.")
    st.markdown("---")
    _render(sections[chosen])
    st.caption("This tab and docs/USER_GUIDE.md are the same file, so the guide here cannot fall "
               "out of step with the guide on disk.")


# ---------------------------------------------------------------------------
# rendering
# ---------------------------------------------------------------------------


def _render(markdown: str) -> None:
    """Render a section, sending image references through ``st.image``.

    Text is accumulated and flushed in blocks rather than line by line, because
    markdown is not line-oriented: emitting each line separately would break
    tables, fenced code and lists into unrelated paragraphs.
    """
    buffer: list[str] = []

    def flush() -> None:
        if buffer:
            st.markdown("\n".join(buffer))
            buffer.clear()

    lines = markdown.splitlines()
    i = 0
    while i < len(lines):
        match = _IMAGE.match(lines[i])
        if match is None:
            buffer.append(lines[i])
            i += 1
            continue

        flush()
        path = (DOCS / match.group("src")).resolve()
        if path.exists():
            st.image(str(path), caption=match.group("alt") or None, width="stretch")
        else:
            st.info(f"The picture '{match.group('alt') or match.group('src')}' is not in this "
                    "copy of the app. Screenshots are captured from a running copy by a separate "
                    "tool; a missing one is shown as missing rather than silently skipped.")

        # A caption directly beneath an image (an italic line, by convention in
        # this guide) belongs to the image rather than to the prose after it.
        i += 1
        while i < len(lines) and not lines[i].strip():
            i += 1
        if i < len(lines) and lines[i].startswith("*") and not lines[i].startswith("**"):
            caption: list[str] = []
            while i < len(lines) and lines[i].strip():
                caption.append(lines[i])
                i += 1
            st.markdown(
                f"<div class='fwa-shot-caption'>{' '.join(caption).strip('*')}</div>",
                unsafe_allow_html=True,
            )
    flush()


def _split_sections(text: str) -> dict[str, str]:
    """Split on second-level headings so the guide is navigable, not a scroll."""
    sections: dict[str, str] = {}
    current = "Introduction"
    buffer: list[str] = []
    fenced = False
    for line in text.splitlines():
        if line.lstrip().startswith("```"):
            fenced = not fenced
        if not fenced and re.match(r"^##\s+\S", line):
            if buffer and "\n".join(buffer).strip():
                sections[current] = "\n".join(buffer)
            current = line.lstrip("#").strip()
            buffer = [line]
        else:
            buffer.append(line)
    if buffer:
        sections[current] = "\n".join(buffer)
    return sections or {"Guide": text}
