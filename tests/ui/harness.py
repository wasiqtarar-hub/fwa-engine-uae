"""Render one app page headlessly and read what a user would see on it.

Each page module exposes ``render()``. The harness runs it through
``streamlit.testing.v1.AppTest`` with a signed-in session for the given role
and a chosen dataset, exactly as the app shell would, and then walks the
rendered element tree collecting the text of the **default view**:

* everything on the page, except
* the contents of collapsed expanders ("Technical details" and similar), and
* every tab but the first (a tab is a click away, like an expander).

Tooltips (``help=``) are not visible by default and are not collected.

``st.cache_resource`` is process-wide under AppTest, so the first page on a
dataset pays for the pipeline run and every later page on that dataset renders
in about a second.
"""

from __future__ import annotations

import datetime as _dt
import html
import json
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
APP = ROOT / "app"
SRC = ROOT / "src"
for p in (str(SRC), str(APP)):
    if p not in sys.path:
        sys.path.insert(0, p)

from fwa.auth import Role, visible_pages  # noqa: E402
from fwa.auth.service import SessionState  # noqa: E402
from fwa.presentation.plain_language import RAW_TOKEN_RE, RULE_ID_RE, SNAKE_TOKEN_RE  # noqa: E402

from navigation import PAGE_MODULES  # noqa: E402

SCRIPT = """
import sys
sys.path.insert(0, r"{app}"); sys.path.insert(0, r"{src}")
from common import boot
boot("Test")
import importlib
importlib.import_module("{module}").render()
"""

#: Things that look like code but are identifiers a user legitimately reads:
#: file names, masked pseudonyms, case/claim ids.
_ALLOW = [
    re.compile(r"[\w.\-]+\.(?:csv|zip|json|md|html|txt|yaml|db|pdf|png)\b", re.I),
    re.compile(r"\b(?:CASE|PROV|MEM|AGT|PHARM|CLIN|EP|CL|COM)-[A-Za-z0-9\-]+\b"),
]

_TAG = re.compile(r"<[^>]+>")
_MISSING_WORD = re.compile(r"(?<![\w-])(?:nan|NaN|NaT|<NA>)(?![\w-])")
#: A whole value that is a missing marker or a raw boolean — never shown by default.
_BARE = {"None", "nan", "NaN", "NaT", "<NA>", "True", "False", "null"}


@dataclass
class Seen:
    page: str
    items: list[tuple[str, str]] = field(default_factory=list)   # (element kind, text)

    def add(self, kind: str, text) -> None:
        if text is None:
            return
        text = html.unescape(_TAG.sub(" ", str(text)))
        if text.strip():
            self.items.append((kind, text))


def session_for(role: Role, username: str | None = None) -> SessionState:
    now = _dt.datetime.now(_dt.timezone.utc)
    name = username or role.value.lower()
    return SessionState(session_id=f"test-{name}", username=name, display_name=f"Test {name}",
                        role=role, tenant_id="T001", started_at=now, last_seen=now)


def pages_for(role: Role) -> list[str]:
    return [p for p in visible_pages(role) if p in PAGE_MODULES]


def render(page: str, role: Role, spec=None, *, mode: str = "simple", timeout: int = 1800):
    from streamlit.testing.v1 import AppTest

    script = SCRIPT.format(app=str(APP), src=str(SRC), module=PAGE_MODULES[page])
    at = AppTest.from_string(script, default_timeout=timeout)
    at.session_state["fwa_session"] = session_for(role)
    at.session_state["fwa_ui_mode"] = mode
    if spec is not None:
        at.session_state["fwa_dataset"] = spec
    at.run()
    return at


# ----------------------------------------------------------------- walking


def _children(node):
    ch = getattr(node, "children", None)
    if isinstance(ch, dict):
        return list(ch.values())
    return []


def _visit(node, seen: Seen) -> None:
    kind = type(node).__name__
    if kind == "Expander":
        seen.add("expander label", getattr(node, "label", ""))
        expanded = dict((f.name, v) for f, v in node.proto.ListFields()).get("expanded", False)
        if not expanded:
            return
    children = _children(node)
    if kind == "Block" and children and all(type(c).__name__ == "Tab" for c in children):
        for tab in children:
            seen.add("tab label", getattr(tab, "label", ""))
        _visit(children[0], seen)       # only the first tab is on screen by default
        return
    _collect(node, kind, seen)
    for child in children:
        _visit(child, seen)


def _collect(node, kind: str, seen: Seen) -> None:
    if kind in ("Markdown", "Caption", "Title", "Header", "Subheader", "Text", "Code", "Latex",
                "Warning", "Info", "Error", "Success", "Toast"):
        seen.add(kind, getattr(node, "value", ""))
    elif kind == "Exception":
        seen.add("EXCEPTION", getattr(node, "value", "") or getattr(node, "message", ""))
    elif kind == "Metric":
        seen.add("metric", f"{node.label} {node.value}")
    elif kind in ("Dataframe", "Table"):
        frame = node.value
        try:
            seen.add("table columns", " | ".join(str(c) for c in frame.columns))
            for column in frame.columns:
                for value in frame[column].tolist():
                    seen.add("table cell", "None" if value is None else (
                        "nan" if isinstance(value, float) and value != value else value))
        except Exception:
            pass
    elif kind in ("Button", "Checkbox", "Toggle", "DownloadButton", "TextInput", "TextArea",
                  "NumberInput", "DateInput", "TimeInput", "ColorPicker", "Slider", "ButtonGroup",
                  "FileUploader"):
        seen.add(f"{kind} label", getattr(node, "label", ""))
    elif kind in ("Radio",):
        seen.add("radio label", node.label)
        for option in node.options:
            seen.add("radio option", option)
    elif kind in ("Selectbox", "SelectSlider"):
        seen.add(f"{kind} label", node.label)
        try:
            index = node.index
            if index is not None:
                seen.add(f"{kind} value", node.options[index])
        except Exception:
            seen.add(f"{kind} value", getattr(node, "value", ""))
    elif kind == "Multiselect":
        seen.add("multiselect label", node.label)
        try:
            for i in node.indices:
                seen.add("multiselect value", node.options[i])
        except Exception:
            for v in node.value or []:
                seen.add("multiselect value", v)
    elif kind == "UnknownElement":
        proto = getattr(node, "proto", None)
        spec = getattr(proto, "spec", None) if proto is not None else None
        if spec:
            try:
                fig = json.loads(spec)
                layout = fig.get("layout", {})
                for path in (("title", "text"), ("xaxis", "title", "text"), ("yaxis", "title", "text"),
                             ("legend", "title", "text")):
                    cur = layout
                    for k in path:
                        cur = cur.get(k, {}) if isinstance(cur, dict) else {}
                    if isinstance(cur, str):
                        seen.add("chart text", cur)
                for trace in fig.get("data", []):
                    if trace.get("name"):
                        seen.add("chart series", trace["name"])
                    for axis in ("x", "y"):
                        values = trace.get(axis)
                        if isinstance(values, list) and values and isinstance(values[0], str):
                            for v in values[:60]:
                                seen.add("chart category", v)
            except Exception:
                pass


def visible_text(at, page: str) -> Seen:
    seen = Seen(page=page)
    for node in _children(at.main):
        _visit(node, seen)
    for exc in at.exception:
        seen.add("EXCEPTION", getattr(exc, "value", "") or str(exc))
    return seen


def code_tokens(text: str, *, rule_ids: bool) -> list[str]:
    """Code-like tokens in ``text`` after removing allowed identifiers."""
    for allow in _ALLOW:
        text = allow.sub(" ", text)
    found = RAW_TOKEN_RE.findall(text) + SNAKE_TOKEN_RE.findall(text)
    if rule_ids:
        found += RULE_ID_RE.findall(text)
    if _MISSING_WORD.search(text):
        found += _MISSING_WORD.findall(text)
    if text.strip() in _BARE:
        found.append(text.strip())
    return found


def violations(seen: Seen, *, rule_ids: bool) -> list[str]:
    out = []
    for kind, text in seen.items:
        if kind == "EXCEPTION":
            out.append(f"{seen.page}: raised an exception: {text[:200]}")
            continue
        tokens = code_tokens(text, rule_ids=rule_ids)
        if tokens:
            out.append(f"{seen.page} [{kind}] {sorted(set(tokens))[:6]} in: {text[:160]!r}")
    return out
