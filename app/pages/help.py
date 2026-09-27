"""Help — the user guide, rendered in the application it documents.

One source, two surfaces: ``docs/USER_GUIDE.md`` on disk and this page are the
same file, so a guide that has gone stale is visibly stale to the person using
the tool rather than quietly wrong in a folder nobody opens.

Rendering it is not quite as simple as handing the markdown to Streamlit.
Markdown image references are relative paths, and Streamlit resolves them
against the served origin rather than the file's own directory — so every
screenshot in the guide renders as a broken-image icon, which is how a page
ends up saying "every screenshot below was captured from a real run" above a
column of missing pictures. :func:`_render` therefore walks the markdown, takes
the image references out, and renders those through ``st.image`` with the
resolved path while passing everything else through untouched.
"""

from __future__ import annotations

import re
from pathlib import Path

import streamlit as st

from common import (
    ROOT, banner, boundary_note, dataframe, empty_state, note, require_page, session_banner,
)
from fwa.auth import Permission, ROLE_DESCRIPTIONS

GUIDE = ROOT / "docs" / "USER_GUIDE.md"
DOCS = GUIDE.parent

#: ``![alt](path)`` on a line of its own.
_IMAGE = re.compile(r"^\s*!\[(?P<alt>[^\]]*)\]\((?P<src>[^)\s]+)\)\s*$")


def render() -> None:
    state = require_page("Help", Permission.VIEW_HELP)
    session_banner(state)

    st.markdown("# Help")
    banner()
    boundary_note()

    role = ROLE_DESCRIPTIONS[state.role]
    st.markdown(
        f"<div class='fwa-card'><h4>You are signed in as {state.role.value}</h4>"
        f"<p><strong>In an insurer's organisation:</strong> {role['job_title']}</p>"
        f"<p><strong>You see:</strong> {role['sees']}</p>"
        f"<p><strong>You can:</strong> {role['can_do']}</p>"
        f"<p class='fwa-sub'>Pages available to you: {', '.join(state.pages())}</p></div>",
        unsafe_allow_html=True,
    )

    if not GUIDE.exists():
        empty_state("The user guide is not in this checkout",
                    "docs/USER_GUIDE.md is missing. The guide is a file in the repository, not "
                    "text embedded in this page, so it can go missing — and saying so is better "
                    "than rendering an empty page.")
        return

    text = GUIDE.read_text(encoding="utf-8")
    sections = _split_sections(text)

    left, right = st.columns([3, 1], gap="large")
    with left:
        chosen = st.selectbox("Section", list(sections), label_visibility="collapsed")
    with right:
        st.download_button("Download the guide", text, file_name="USER_GUIDE.md",
                           mime="text/markdown", width="stretch")

    st.markdown("---")
    _render(sections[chosen])

    note(
        "This page and `docs/USER_GUIDE.md` are the same file. Nothing is duplicated between "
        "them, so the guide you read here cannot fall out of step with the guide on disk."
    )


# ---------------------------------------------------------------------------
# rendering
# ---------------------------------------------------------------------------


def _render(markdown: str) -> None:
    """Render a section, sending image references through ``st.image``.

    Text is accumulated and flushed in blocks rather than line by line, because
    markdown is not a line-oriented format: emitting each line separately would
    break every multi-line construct in the guide — tables, fenced code, lists —
    into a series of unrelated paragraphs.
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
            st.warning(
                f"`{match.group('src')}` is not in this checkout. Screenshots are captured from "
                "a running instance by `tools/capture_screenshots.py`; an image that is missing "
                "is shown as missing rather than silently skipped."
            )

        # A caption directly beneath an image — an italic line, by convention in
        # this guide — belongs to the image rather than to the prose after it.
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
            if buffer:
                sections[current] = "\n".join(buffer)
            current = line.lstrip("#").strip()
            buffer = [line]
        else:
            buffer.append(line)
    if buffer:
        sections[current] = "\n".join(buffer)
    return sections or {"Guide": text}
