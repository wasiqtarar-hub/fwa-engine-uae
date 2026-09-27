"""Shared UI helpers: caching, guards, chips, panels, chart template.

Three things live here because they must be identical on every page:

* **The pipeline cache.** ``@st.cache_resource`` means a full-file run
  happens once per dataset, not once per page load. The queue has to render in
  well under a second, and a model must never be recomputed on a page load.
* **The access guard.** :func:`require_page` is the single authorisation gate.
  Every page calls it first; no page is reachable without it, because
  navigation itself is built from the role's permissions.
* **The visual language.** The active theme, disposition colour, priority band,
  the AI panel and the chart template are defined once so they cannot drift
  between pages — and consistency achieved by copy-paste is consistency that
  lasts until the first edit.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import html
import sys
from pathlib import Path
from typing import Any, Iterable, Sequence

import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
if str(ROOT / "app") not in sys.path:
    sys.path.insert(0, str(ROOT / "app"))

from fwa import SAFETY_BOUNDARY_STATEMENT, __version__  # noqa: E402
from fwa.audit.log import AuditLog  # noqa: E402
from fwa.auth import (  # noqa: E402
    AccessDenied, AuthService, Permission, Role, SessionState, mask_identity,
)
from fwa.config import load_config  # noqa: E402
from fwa.enums import DISPOSITION_MEANINGS, Disposition, REVIEWER_ASK  # noqa: E402
from fwa.pipeline import run_pipeline  # noqa: E402
from fwa.storage import Database  # noqa: E402

from dataset import DatasetSpec, active_dataset  # noqa: E402

__all__ = [
    "boot", "require_page", "session", "banner", "session_banner", "card", "tiles",
    "chip", "disposition_chip", "priority_chip", "status_chip", "ai_panel", "empty_state",
    "mono", "note", "source_span", "guide", "chart_template", "bar", "line", "histogram",
    "_chart_key",
    "mask", "pipeline", "pipeline_for", "auth_service", "audit_log", "config", "dataframe",
    "active_theme", "disposition_colours",
]

# ---------------------------------------------------------------------------
# palette — mirrors app/assets/style.css so charts match the chips
# ---------------------------------------------------------------------------

#: Disposition colour, per theme, mirroring the two token blocks in
#: `app/assets/`. The mapping is the same in both: a disposition keeps its hue
#: and only its lightness moves, because the colour carries meaning and a
#: reader who has learnt that SIU_LEAD is the crimson one should not have to
#: learn it again after switching theme.
DISPOSITION_COLOURS_BY_THEME = {
    "light": {
        "REJECT": "#a32d2d", "REPRICE": "#9d6210", "RETURN": "#55606c",
        "PREPAY_PEND": "#1f5f9c", "POSTPAY_AUDIT": "#6b4796", "SIU_LEAD": "#8e2b52",
        "PROVIDER_EDUCATION": "#1c6f63", "MONITOR_ONLY": "#6c7580",
    },
    "dark": {
        "REJECT": "#e0716b", "REPRICE": "#e0a35a", "RETURN": "#9aa4af",
        "PREPAY_PEND": "#6aa9e0", "POSTPAY_AUDIT": "#b191dd", "SIU_LEAD": "#e07a9e",
        "PROVIDER_EDUCATION": "#5fc2ae", "MONITOR_ONLY": "#9aa4af",
    },
}

#: Kept as a module-level name because pages import it directly. It holds the
#: light values; :func:`disposition_colours` is what a page should call when it
#: wants the ones that will actually be legible on the current ground.
DISPOSITION_COLOURS = DISPOSITION_COLOURS_BY_THEME["light"]


def disposition_colours(theme: str | None = None) -> dict[str, str]:
    """Disposition colours for the theme currently being rendered."""
    return DISPOSITION_COLOURS_BY_THEME[theme or active_theme()]


#: The chart colorway. Related-but-distinct hues for series that carry no
#: inherent meaning of their own.
SEQUENCE = ["#6d3560", "#1f5f9c", "#1c6f63", "#9d6210", "#8e2b52", "#55606c", "#6b4796"]
#: The same hues, lightened for a dark ground. Same order, so a series keeps its
#: colour when the theme changes and a reader comparing two screenshots is not
#: quietly misled by a swap.
SEQUENCE_DARK = ["#d9a3cc", "#6aa9e0", "#5fc2ae", "#e0a35a", "#e07a9e", "#9aa4af", "#b191dd"]


#: Chart ink, per theme. A chart sits on a transparent background and inherits
#: the page's, so its axis labels and grid lines have to follow the page too —
#: a #6b6470 tick label is perfectly legible on white and close to invisible on
#: #16131a. The colorway stays put: those colours carry meaning and are checked
#: for contrast against both grounds.
CHART_INK = {
    "light": {"text": "#6b6470", "grid": "rgba(33,29,36,0.10)", "axis": "rgba(33,29,36,0.28)"},
    "dark":  {"text": "#a49bab", "grid": "rgba(236,232,238,0.12)", "axis": "rgba(236,232,238,0.26)"},
}


def chart_template(theme: str | None = None) -> str:
    """One plotly template per theme, registered once, so charts look related."""
    theme = theme or active_theme()
    name = f"fwa-{theme}"
    if name not in pio.templates:
        ink = CHART_INK[theme]
        base = pio.templates["plotly_white" if theme == "light" else "plotly_dark"]
        template = base.to_plotly_json()
        template["layout"].update({
            "colorway": SEQUENCE if theme == "light" else SEQUENCE_DARK,
            "font": {"family": "ui-sans-serif, -apple-system, Segoe UI, Roboto, sans-serif",
                     "size": 12, "color": ink["text"]},
            # Transparent, so the chart sits on the page rather than in a box of
            # its own — which is the whole reason the ink above has to follow
            # the theme.
            "paper_bgcolor": "rgba(0,0,0,0)",
            "plot_bgcolor": "rgba(0,0,0,0)",
            "margin": {"l": 56, "r": 18, "t": 34, "b": 44},
            "xaxis": {"gridcolor": ink["grid"], "zeroline": False, "linecolor": ink["axis"]},
            "yaxis": {"gridcolor": ink["grid"], "zeroline": False, "linecolor": ink["axis"]},
            "legend": {"orientation": "h", "y": -0.18, "x": 0},
            "hoverlabel": {"font": {"size": 12}},
        })
        pio.templates[name] = go.layout.Template(template)
    return name


# ---------------------------------------------------------------------------
# boot and caching
# ---------------------------------------------------------------------------


@st.cache_resource(show_spinner=False)
def config():
    return load_config()


@st.cache_resource(show_spinner=False)
def audit_log() -> AuditLog:
    return AuditLog()


@st.cache_resource(show_spinner=False)
def _database() -> Database:
    return Database()


@st.cache_resource(show_spinner=False)
def auth_service() -> AuthService:
    service = AuthService(_database(), audit_log(), config())
    created = service.seed()
    if created:
        # Printed once to the console and documented in the user guide.
        print("\n" + "=" * 72)
        print("uae-fwa-engine — demo accounts seeded (first run only)")
        print("Every account is forced to change its password at first login.")
        print("=" * 72)
        for username, password, role in created:
            print(f" {username:<10} / {password:<10} {role}")
        print("=" * 72 + "\n", flush=True)
    return service


@st.cache_resource(show_spinner="Running every control and model over this dataset…")
def _pipeline_for(path: str, adapter: str, key: str):
    """One full run, memoised on the dataset's identity.

    The cache key is the content hash of the input file plus the adapter, not
    the process. That has two consequences worth stating: loading a second
    dataset does not discard the first one's results, so switching back to it
    is free and a comparison between the two is possible at all; and re-loading
    a file the user has edited in place under an unchanged name correctly
    produces a new run rather than serving the stale one.

    ``key`` is not used inside the function. It is in the signature because
    ``st.cache_resource`` keys on the arguments, and the content hash is what
    makes the identity correct.
    """
    return run_pipeline(
        path,
        config=config(),
        adapter=adapter,
        generate_documents=800,
        document_dir=ROOT / "data" / "synthetic_documents",
        verbose=False,
    )


def pipeline_for(spec: DatasetSpec):
    """Results for one named dataset, whether or not it is the active one."""
    return _pipeline_for(str(spec.path), spec.adapter, spec.key)


def pipeline():
    """Results for whichever dataset is currently loaded.

    Every page calls this. Because it resolves the active dataset on each call
    rather than closing over one at import time, changing the dataset on the
    Data page changes what every other page is describing, with no page needing
    to know that datasets can change at all.
    """
    return pipeline_for(active_dataset())


def boot(page_title: str, icon: str = "🛡️") -> None:
    """Page config, stylesheet, chart template. Called first on every page."""
    if not st.session_state.get("_fwa_page_config"):
        st.set_page_config(
            page_title=f"{page_title} · UAE FWA Engine",
            page_icon=icon,
            layout="wide",
            initial_sidebar_state="expanded",
        )
        st.session_state["_fwa_page_config"] = True
    # Injected on EVERY render, not once per session. With st.navigation each
    # page render replaces the DOM, so a <style> element added on a previous
    # page is gone — and the AI panel would lose the dashed border that makes
    # it visually distinct from an evidence panel, which is non-negotiable.
    st.markdown(f"<style>{_stylesheet(active_theme())}</style>", unsafe_allow_html=True)
    chart_template()


def active_theme() -> str:
    """Which theme Streamlit is ACTUALLY rendering: ``"light"`` or ``"dark"``.

    Asking Streamlit rather than asking the browser matters more than it
    sounds. Streamlit writes no marker into the DOM — no ``data-theme``, no
    class — so a stylesheet has nothing to test and the obvious substitute is
    ``@media (prefers-color-scheme: dark)``. That reads the *operating
    system's* preference, which is a different question from what theme this
    application is running under, and the two part company the moment
    ``.streamlit/config.toml`` pins a base or a user picks a theme in the
    settings menu.

    When they part company the page tears in half: our stylesheet paints a dark
    background while Streamlit, still on its light theme, paints its own
    headings near-black. Dark text on a dark ground — not a subtle
    contrast problem but an unreadable page, and one that only appears for
    users whose desktop is set the other way from the app.

    ``theme.base`` in ``.streamlit/config.toml`` is the answer, because it is
    what Streamlit itself resolves to. ``st.context.theme.type`` is consulted
    as well, so that a Streamlit which does offer an in-app theme picker is
    followed rather than overridden — with one measured caveat. On the **first
    script run of a session** it has not been negotiated with the frontend yet
    and reports the browser's ``prefers-color-scheme`` instead. That is the
    same wrong answer as the media query arriving by a different route, and
    taking it would paint the login screen dark for exactly the users this
    function exists to protect. So the first run uses the configured base and
    every run after it uses the negotiated value; on a Streamlit with no theme
    picker the two are the same value and nothing moves.
    """
    configured = str(_configured_base() or "light").lower()
    try:
        settled = st.session_state.get("_fwa_theme_settled", False)
        st.session_state["_fwa_theme_settled"] = True
        if not settled:
            return "dark" if configured == "dark" else "light"
        return "dark" if st.context.theme.type == "dark" else "light"
    except Exception:
        # No script run at all (an import, a bare `python -c`), or a Streamlit
        # too old to have `st.context.theme`. The configured base is the honest
        # answer in both cases.
        return "dark" if configured == "dark" else "light"


def _configured_base() -> str | None:
    try:
        return st.get_option("theme.base")
    except Exception:
        return None


@st.cache_data(show_spinner=False)
def _stylesheet(theme: str) -> str:
    """The stylesheet for one theme: base tokens, plus dark overrides if dark.

    Cached on the theme rather than on nothing, so a theme change re-reads the
    files and changing back is free.
    """
    assets = ROOT / "app" / "assets"
    css = (assets / "style.css").read_text(encoding="utf-8")
    if theme == "dark":
        css += "\n" + (assets / "style-dark.css").read_text(encoding="utf-8")
    return css


# ---------------------------------------------------------------------------
# session and access
# ---------------------------------------------------------------------------


def session() -> SessionState | None:
    return st.session_state.get("fwa_session")


def require_page(page_name: str, permission: Permission) -> SessionState:
    """The single access guard (build brief §13.3).

    No page is reachable by URL without authorisation: navigation is built from
    the role's permissions, and this guard re-checks on entry so a bookmarked
    or forged page reference fails closed.
    """
    state = session()
    if state is None:
        st.error("Your session has ended. Please sign in again.")
        st.stop()
    service = auth_service()
    if service.enforce_timeout(state):
        st.session_state.pop("fwa_session", None)
        st.warning(
            f"You were signed out after {config().get('session_timeout_minutes')} minutes without "
            f"activity. Please sign in again."
        )
        st.stop()
    if not state.can(permission):
        audit_log().record(
            "ACCESS_DENIED", actor=state.username, actor_role=state.role.value,
            tenant_id=state.tenant_id, subject=page_name,
            reason=f"Role {state.role.value} does not hold {permission.value}.",
        )
        from fwa.presentation import label as _lbl, page_info as _pi

        st.error(
            f"**You don't have access to this page.** The {_lbl(state.role.value, 'role').lower()} "
            f"role can't open {_pi(page_name)['title']}. This attempt has been written to the "
            f"access log; ask an administrator if you need access."
        )
        st.stop()

    # Every page calls this guard exactly once, at the top, which makes it the
    # one reliable "a new script run has started" hook Streamlit gives us.
    # ``_chart_key`` needs that to hand out keys that are stable WITHIN a run
    # and unique within it — see the note there.
    st.session_state["_fwa_chart_keys"] = {}
    return state


# ---------------------------------------------------------------------------
# chrome
# ---------------------------------------------------------------------------


def banner() -> None:
    """The dataset strip: which file produced everything below it.

    Every figure on every page is a property of one input file. When that file
    was a constant the interface could leave it implicit; now that it is a
    choice, a page that does not name its dataset is a page whose numbers
    cannot be attributed to anything — and a screenshot of one is evidence of
    nothing. So the strip is persistent rather than a detail on the Data page.
    """
    spec = active_dataset()
    st.markdown(
        f'<div class="fwa-dataset">'
        f'<span class="fwa-dataset-name">{html.escape(spec.name)}</span>'
        f'<span class="fwa-dataset-meta">{html.escape(spec.meta_line())}</span>'
        f"</div>",
        unsafe_allow_html=True,
    )


def boundary_note() -> None:
    st.markdown(
        f'<div class="fwa-boundary"><strong>Safety boundary.</strong> '
        f"{html.escape(SAFETY_BOUNDARY_STATEMENT)}</div>",
        unsafe_allow_html=True,
    )


def session_banner(state: SessionState) -> None:
    started = state.started_at.astimezone().strftime("%H:%M")
    timeout = config().get("session_timeout_minutes")
    from fwa.presentation import label as _lbl

    st.markdown(
        f'<div class="fwa-session">'
        f'<span class="fwa-who">{html.escape(state.display_name)}</span>'
        f'<span class="fwa-chip s-neutral">{html.escape(_lbl(state.role.value, "role"))}</span>'
        f'<span class="fwa-tag">organisation {html.escape(state.tenant_id)}</span>'
        f"<span>signed in {started} · signs out after {timeout} min without activity</span>"
        f'<span class="fwa-tag">v{__version__}</span>'
        f"</div>",
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# components
# ---------------------------------------------------------------------------


def chip(label: str, css_class: str = "s-neutral") -> str:
    return f'<span class="fwa-chip {css_class}">{html.escape(str(label))}</span>'


def disposition_chip(disposition: str | Disposition) -> str:
    value = disposition.value if isinstance(disposition, Disposition) else str(disposition)
    return chip(value.replace("_", " ").title(), f"d-{value}")


def priority_chip(band: str, label: str, score: float | None = None) -> str:
    text = f"{band} · {label}" + (f" · {score:.0f}" if score is not None else "")
    return chip(text, f"p-{band}")


def status_chip(status: str) -> str:
    mapping = {
        "PASS": "s-ok", "MEASURED": "s-ok", "OK": "s-ok", "PROMOTED": "s-ok",
        "FAIL": "s-bad", "BREACH": "s-bad", "PROMOTION_BLOCKED": "s-bad",
        "WARN": "s-warn", "PARTIAL": "s-warn",
        "NOT_ASSESSABLE": "s-neutral", "NOT_MEASURABLE_ON_THIS_DATASET": "s-neutral",
        "INFORMATIONAL": "s-neutral", "EXECUTABLE": "s-ok",
        "NOT_EXECUTABLE_ON_THIS_DATASET": "s-neutral",
    }
    return chip(status.replace("_", " ").title(), mapping.get(status, "s-neutral"))


def card(title: str, body: str, sub: str = "") -> None:
    st.markdown(
        f'<div class="fwa-card"><h4>{title}</h4><p>{body}</p>'
        + (f'<p class="fwa-sub">{sub}</p>' if sub else "")
        + "</div>",
        unsafe_allow_html=True,
    )


def tiles(items: Sequence[tuple[str, Any, str]]) -> None:
    """A row of metric tiles: (label, value, note)."""
    cells = "".join(
        f'<div class="fwa-tile"><div class="fwa-tile-label">{html.escape(str(label))}</div>'
        f'<div class="fwa-tile-value">{value}</div>'
        + (f'<div class="fwa-tile-note">{note}</div>' if note else "")
        + "</div>"
        for label, value, note in items
    )
    st.markdown(f'<div class="fwa-tiles">{cells}</div>', unsafe_allow_html=True)


def ai_panel(text: str, footer: str = "", label: str = "AI-generated summary — not evidence") -> None:
    """The visually distinct advisory panel (§14, non-negotiable).

    Different background, dashed border, an icon and an explicit label. A
    reviewer must never have to work out whether they are reading evidence or a
    generated summary.
    """
    st.markdown(
        f'<div class="fwa-ai"><div class="fwa-ai-label">◇ {html.escape(label)}</div>'
        f'<div class="fwa-ai-body">{html.escape(text)}</div>'
        + (f'<div class="fwa-ai-foot">{html.escape(footer)}</div>' if footer else "")
        + "</div>",
        unsafe_allow_html=True,
    )


def mono(text: str) -> None:
    st.markdown(f'<div class="fwa-mono">{html.escape(text)}</div>', unsafe_allow_html=True)


def note(text: str) -> None:
    st.markdown(f'<div class="fwa-note">{html.escape(text)}</div>', unsafe_allow_html=True)


def source_span(text: str, caption: str = "") -> None:
    st.markdown(
        f'<div class="fwa-span">“{html.escape(text)}”'
        + (f'<br><span class="fwa-sub">{html.escape(caption)}</span>' if caption else "")
        + "</div>",
        unsafe_allow_html=True,
    )


def guide(step: str, text: str) -> None:
    """A guided-demo annotation (§14.1 'Walk me through a case')."""
    st.markdown(
        f'<div class="fwa-guide"><span class="fwa-guide-step">{html.escape(step)}</span>'
        f"{html.escape(text)}</div>",
        unsafe_allow_html=True,
    )


def empty_state(title: str, body: str) -> None:
    """A designed empty state, not a blank page (§14.1)."""
    st.markdown(
        f'<div class="fwa-empty"><div class="fwa-empty-title">{html.escape(title)}</div>'
        f"<div>{html.escape(body)}</div></div>",
        unsafe_allow_html=True,
    )


def dataframe(frame: pd.DataFrame, height: int | None = None, **kwargs) -> None:
    if frame is None or frame.empty:
        empty_state("Nothing to show here", "No rows matched. Widen the filters above.")
        return
    if height is not None:
        kwargs["height"] = height
    st.dataframe(frame, width="stretch", hide_index=True, **kwargs)


def mask(value: Any, state: SessionState, *, assigned: bool = False) -> str:
    return mask_identity(
        value, state.role,
        unmask_granted=str(value) in state.unmasked_subjects,
        assigned=assigned,
        extra_permissions=state.extra_permissions,
    )


# ---------------------------------------------------------------------------
# charts
# ---------------------------------------------------------------------------


def _chart_key(fig, prefix: str) -> str:
    """A key that is unique within a run and the same on the next one.

    Streamlit derives a chart's identity from its type and parameters, so two
    charts that happen to plot the same thing — a SHAP panel rendered once per
    model, say — collide and raise ``StreamlitDuplicateElementId``. Hashing the
    figure and counting repeats of that hash within the run disambiguates them
    without making the key depend on call order, which would change every time
    a page grew a chart above them.
    """
    digest = hashlib.sha1(fig.to_json().encode("utf-8")).hexdigest()[:12]
    seen = st.session_state.setdefault("_fwa_chart_keys", {})
    index = seen.get(digest, 0)
    seen[digest] = index + 1
    return f"{prefix}-{digest}-{index}"


def _axes(fig, x_label: str | None, y_label: str | None, horizontal: bool = False) -> None:
    """Plain axis titles with units. Horizontal bars swap which axis is which."""
    if horizontal:
        x_label, y_label = y_label, x_label
    if x_label is not None:
        fig.update_xaxes(title_text=x_label)
    if y_label is not None:
        fig.update_yaxes(title_text=y_label)


def bar(frame: pd.DataFrame, x: str, y: str, *, title: str = "", colour: str | None = None,
        colour_map: dict[str, str] | None = None, horizontal: bool = False, height: int = 320,
        x_label: str | None = None, y_label: str | None = None, how_to_read: str = "",
        legend_title: str | None = None):
    """A bar chart. ``title`` should state the finding; ``x_label``/``y_label`` name
    the axes with units (for a horizontal chart, ``x_label`` still names the
    category and ``y_label`` the value); ``how_to_read`` adds the one-line hint."""
    import plotly.express as px

    fig = px.bar(
        frame, x=y if horizontal else x, y=x if horizontal else y,
        orientation="h" if horizontal else "v",
        color=colour, color_discrete_map=colour_map,
        color_discrete_sequence=SEQUENCE, title=title or None,
    )
    fig.update_layout(template=chart_template(), height=height, showlegend=bool(colour))
    _axes(fig, x_label, y_label, horizontal)
    if legend_title is not None:
        fig.update_layout(legend_title_text=legend_title)
    st.plotly_chart(fig, width="stretch", key=_chart_key(fig, "bar"))
    if how_to_read:
        chart_note(how_to_read)
    return fig


def line(frame: pd.DataFrame, x: str, y: str | Sequence[str], *, title: str = "",
         height: int = 300, markers: bool = True, x_label: str | None = None,
         y_label: str | None = None, how_to_read: str = "", legend_title: str | None = None):
    import plotly.express as px

    fig = px.line(frame, x=x, y=y, markers=markers, title=title or None,
                  color_discrete_sequence=SEQUENCE)
    fig.update_layout(template=chart_template(), height=height)
    _axes(fig, x_label, y_label)
    if legend_title is not None:
        fig.update_layout(legend_title_text=legend_title)
    st.plotly_chart(fig, width="stretch", key=_chart_key(fig, "line"))
    if how_to_read:
        chart_note(how_to_read)
    return fig


def histogram(values: Iterable[float], *, title: str = "", nbins: int = 40,
              vline: float | None = None, vline_label: str = "", height: int = 300,
              x_label: str | None = None, y_label: str | None = "Number of items",
              how_to_read: str = ""):
    import plotly.express as px

    fig = px.histogram(pd.DataFrame({"value": list(values)}), x="value", nbins=nbins,
                       title=title or None, color_discrete_sequence=SEQUENCE)
    if vline is not None:
        fig.add_vline(x=vline, line_dash="dash", line_color=SEQUENCE[0],
                      annotation_text=vline_label or f"{vline:g}", annotation_position="top")
    fig.update_layout(template=chart_template(), height=height, bargap=0.04)
    _axes(fig, x_label if x_label is not None else "Value", y_label)
    st.plotly_chart(fig, width="stretch", key=_chart_key(fig, "hist"))
    if how_to_read:
        chart_note(how_to_read)
    return fig


# =============================================================================
# PLAIN-LANGUAGE LAYER
# =============================================================================
# Everything below renders the engine's output for a non-technical reader. It
# is presentation only: internal identifiers are translated through
# ``fwa.presentation`` at the moment they are shown, and never renamed.

import json as _json  # noqa: E402
import logging as _logging  # noqa: E402
from contextlib import contextmanager  # noqa: E402

from fwa.presentation import (  # noqa: E402
    aed as _aed, field_label as _field_label, field_meaning as _field_meaning,
    friendly_frame as _friendly_frame, label as _plain_label, page_info as _page_info,
    standard_text as _standard_text, status as _status, term_help as _term_help,
)
from fwa.presentation.explain_case import (  # noqa: E402
    CaseExplanation, diagnosis_names, explain_case, explain_model_claim,
)

_log = _logging.getLogger("fwa.app")

#: Per-user interface preferences (Simple/Advanced). A small JSON file rather
#: than a database table because it is a convenience, not governed state: it
#: never affects what the engine does, only which pages are listed.
PREFS_PATH = Path(__import__("os").environ.get("FWA_UI_PREFS", str(ROOT / "data" / "ui_prefs.json")))
MODE_KEY = "fwa_ui_mode"

__all__ += [
    "ui_mode", "set_ui_mode", "is_simple", "mode_toggle", "pages_for_mode", "page_header",
    "status_badge", "friendly_table", "headline_cards", "chart_note", "case_explanation",
    "render_explanation", "render_model_claim", "friendly_failure", "friendly_error", "tone_class",
    "help_text", "synthetic_banner", "PREFS_PATH", "explain_model_claim", "is_synthetic_dataset",
]


# ---------------------------------------------------------------- mode


def _read_prefs() -> dict:
    try:
        return _json.loads(PREFS_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _write_prefs(prefs: dict) -> None:
    try:
        PREFS_PATH.parent.mkdir(parents=True, exist_ok=True)
        PREFS_PATH.write_text(_json.dumps(prefs, indent=1, sort_keys=True), encoding="utf-8")
    except Exception:  # a preference that cannot be saved is not an error worth showing
        _log.warning("Could not save interface preferences to %s", PREFS_PATH)


def ui_mode() -> str:
    """``"simple"`` (the default) or ``"advanced"``, remembered per user."""
    mode = st.session_state.get(MODE_KEY)
    if mode in ("simple", "advanced"):
        return mode
    state = session()
    stored = _read_prefs().get(state.username, {}).get("mode") if state else None
    mode = stored if stored in ("simple", "advanced") else "simple"
    st.session_state[MODE_KEY] = mode
    return mode


def set_ui_mode(mode: str) -> None:
    mode = "advanced" if mode == "advanced" else "simple"
    st.session_state[MODE_KEY] = mode
    state = session()
    if state is not None:
        prefs = _read_prefs()
        prefs.setdefault(state.username, {})["mode"] = mode
        _write_prefs(prefs)


def is_simple() -> bool:
    return ui_mode() == "simple"


def mode_toggle() -> None:
    """The sidebar switch. On by default; remembered for this user."""
    simple = st.toggle(
        "Simple view", value=is_simple(), key="fwa_mode_toggle",
        help="Simple view shows the pages you need to review cases: Overview, Review queue, "
             "Case evidence, Data and Help. Turn it off to add the analysis and governance "
             "pages (models, hospital comparisons, connections, settings, the checks library, "
             "validation and governance). Your choice is remembered next time you sign in. "
             "It never changes what you are allowed to see: access still depends on your role.",
    )
    wanted = "simple" if simple else "advanced"
    if wanted != ui_mode():
        set_ui_mode(wanted)
        st.rerun()


def pages_for_mode(allowed: Sequence[str]) -> list[str]:
    """Filter a role's pages by the current mode (role-based access applies first)."""
    if not is_simple():
        return list(allowed)
    return [p for p in allowed if _page_info(p).get("simple")]


# ---------------------------------------------------------------- frame


def page_header(name: str) -> None:
    """Plain page title plus its two- or three-line "what this page is for" caption."""
    info = _page_info(name)
    st.markdown(f"# {info['title']}")
    if info.get("purpose"):
        st.markdown(f"<div class='fwa-intro'>{html.escape(str(info['purpose']))}</div>",
                    unsafe_allow_html=True)


def is_synthetic_dataset() -> bool:
    spec = active_dataset()
    name = f"{spec.name} {spec.path}".lower()
    return "synthetic" in name or "uae_demo" in name


def synthetic_banner() -> None:
    """The SYNTHETIC notice, shown whenever the active file is one of the generated datasets."""
    if is_synthetic_dataset():
        body = _standard_text("synthetic").replace("SYNTHETIC data. ", "")
        st.markdown(
            f"<div class='fwa-banner'><strong>SYNTHETIC data.</strong> {html.escape(body)}</div>",
            unsafe_allow_html=True,
        )


def help_text(term: str, fallback: str = "") -> str:
    """Tooltip text for a glossary term (``help=`` argument)."""
    return _term_help(term) or fallback


# ---------------------------------------------------------------- chips


_TONE_CLASS = {"ok": "s-ok", "warn": "s-warn", "bad": "s-bad", "info": "s-info", "neutral": "s-neutral"}


def tone_class(tone: str) -> str:
    return _TONE_CLASS.get(str(tone), "s-neutral")


def status_badge(value, kind: str | None = None) -> str:
    """A coloured chip carrying the plain label of any status or enum value."""
    if kind in (None, "status"):
        phrase = _status(value)
        return chip(phrase.label, tone_class(phrase.tone))
    return chip(_plain_label(value, kind), "s-neutral")


def disposition_chip(disposition) -> str:  # noqa: F811 - plain wording, same colour classes
    value = disposition.value if isinstance(disposition, Disposition) else str(disposition)
    return chip(_status(value).label, f"d-{value}")


def priority_chip(band: str, label: str = "", score: float | None = None) -> str:  # noqa: F811
    text = _plain_label(band, "band") + (f" · {score:.0f}/100" if score is not None else "")
    return chip(text, f"p-{band}")


def status_chip(status: str) -> str:  # noqa: F811
    return status_badge(status)


# ---------------------------------------------------------------- tables


def friendly_table(
    frame: pd.DataFrame,
    *,
    key: str,
    columns: Sequence[str] | None = None,
    max_default: int = 8,
    money: Iterable[str] = (),
    ratios: Iterable[str] = (),
    dates: Iterable[str] = (),
    keep_numeric: Iterable[str] = (),
    rename: dict[str, str] | None = None,
    value_kinds: dict[str, str] | None = None,
    height: int | None = None,
    empty_title: str = "Nothing to show here",
    empty_body: str = "No rows matched. Widen the filters above.",
) -> None:
    """A table for a non-technical reader: plain column names, translated values,
    at most ``max_default`` columns until "Show all columns" is ticked."""
    if frame is None or frame.empty:
        empty_state(empty_title, empty_body)
        return
    ordered = [c for c in (columns or list(frame.columns)) if c in frame.columns]
    extra = [c for c in frame.columns if c not in ordered]
    show_all = False
    if len(ordered) > max_default or extra:
        show_all = st.checkbox(
            "Show all columns", key=f"{key}_all",
            help="The table starts with the columns most people need. Tick this to see every "
                 "column, including technical ones.",
        )
    cols = (ordered + extra) if show_all else ordered[:max_default]
    view = _friendly_frame(frame, cols, money=money, ratios=ratios, dates=dates,
                           keep_numeric=keep_numeric, rename=rename, value_kinds=value_kinds)
    config = {}
    for original, shown in zip(cols, view.columns):
        meaning = _field_meaning(original)
        if meaning:
            config[shown] = st.column_config.Column(help=meaning)
    kwargs = {"height": height} if height else {}
    st.dataframe(view, width="stretch", hide_index=True, column_config=config, key=key, **kwargs)


def headline_cards(items: Sequence[tuple[str, str]]) -> None:
    """Cards with a big number and a sentence underneath: ``[(big, sentence), ...]``."""
    cells = "".join(
        f"<div class='fwa-hcard'><div class='fwa-big'>{big}</div>"
        f"<div class='fwa-say'>{say}</div></div>"
        for big, say in items
    )
    st.markdown(f"<div class='fwa-cards'>{cells}</div>", unsafe_allow_html=True)


def chart_note(how_to_read: str) -> None:
    """The one-line "How to read this chart" under every chart."""
    st.markdown(f"<div class='fwa-howto'><strong>How to read this chart:</strong> "
                f"{html.escape(how_to_read)}</div>", unsafe_allow_html=True)


# ---------------------------------------------------------------- explanations


def case_explanation(result, case, state) -> CaseExplanation:
    """The plain explanation of one case, with identities masked for this user."""
    signals = result.signals_for_case(case.case_id)
    assigned = case.case_id in state.assigned_case_ids
    cache_key = f"_fwa_dx_names_{id(result)}"
    names = st.session_state.get(cache_key)
    if names is None:
        names = diagnosis_names(result.dataset)
        st.session_state[cache_key] = names
    return explain_case(
        case, signals, registry=result.registry, claims=result.claims,
        mask=lambda v: mask(v, state, assigned=assigned), dx_names=names,
    )


def render_explanation(expl: CaseExplanation, *, key: str, compact: bool = False) -> None:
    """Headline, ranked reasons, strength, what it does not mean, next steps, details."""
    reasons = "".join(
        f"<li>{html.escape(r.sentence)}"
        + (f" <span class='fwa-sub'>({r.more_like_it} more like it)</span>" if r.more_like_it else "")
        + (" <span class='fwa-sub'>(simplified check)</span>" if r.simplified else "")
        + "</li>"
        for r in expl.reasons
    )
    strength = (
        f"<div class='fwa-strength'>{chip(expl.strength.label + ' evidence', tone_class(expl.strength.tone))} "
        f"{html.escape(expl.strength.why)}"
        + (f" {html.escape(expl.strength.shared_fact_note)}" if expl.strength.shared_fact_note else "")
        + "</div>"
    )
    st.markdown(
        f"<div class='fwa-explain'><div class='fwa-headline'>{html.escape(expl.headline)}</div>"
        f"<div class='fwa-sub' style='margin-top:.5rem'><strong>Why it was flagged</strong></div>"
        f"<ol>{reasons}</ol>{strength}</div>",
        unsafe_allow_html=True,
    )
    st.markdown("<div class='fwa-caveat'><strong>What this does not mean.</strong> "
                + " ".join(html.escape(c) for c in expl.caveats) + "</div>",
                unsafe_allow_html=True)
    if expl.next_steps:
        st.markdown("**What to check next**\n\n" + "\n".join(
            f"{i}. {s}" for i, s in enumerate(expl.next_steps, start=1)))
    if compact:
        return
    with st.expander("Technical details (for auditors and examiners)", expanded=False):
        dataframe(pd.DataFrame(expl.technical).drop(columns=["evidence"], errors="ignore"))
        st.caption("Rule id and version, the control's score, confidence, evidence strength and the "
                   "underlying fact each signal rests on. Signals sharing an underlying fact count "
                   "once: their evidence is capped at the strongest, never added.")
        for t in expl.technical:
            st.markdown(f"**{t['rule_id']}@{t['rule_version']}** · `{t['reason_code']}`")
            st.code(t["evidence"], language="json")
    c1, c2 = st.columns(2)
    with c1:
        st.download_button(
            "Download one-page summary (HTML, printable)",
            expl.to_html(synthetic_notice=_standard_text("synthetic") if is_synthetic_dataset() else ""),
            file_name=f"{expl.case_id}_summary.html", mime="text/html", key=f"{key}_html",
            help="A printable one-page summary of this case: headline, reasons, strength, "
                 "caveats, next steps and the technical details. Open it and print to PDF.",
        )
    with c2:
        st.download_button(
            "Download summary (Markdown)", expl.to_markdown(),
            file_name=f"{expl.case_id}_summary.md", mime="text/markdown", key=f"{key}_md",
            help="The same summary as plain text with simple formatting, for pasting into a "
                 "report or a case-management system.",
        )


def render_model_claim(expl) -> None:
    """A pattern-finder's reasons for ranking one claim as unusual."""
    if expl is None:
        empty_state("No explanation available",
                    "This claim was not scored by the pattern-finder, so there is nothing to explain.")
        return
    if expl.exploratory:
        st.markdown(
            "<div class='fwa-banner'><strong>Exploratory scores.</strong> The model has seen some of "
            "these hospitals during training, so these scores are not used for the promotion gate."
            "</div>", unsafe_allow_html=True)
    st.markdown(
        f"<div class='fwa-explain'><div class='fwa-headline'>{html.escape(expl.headline)}</div>"
        + ("<ol>" + "".join(f"<li>{html.escape(r)}</li>" for r in expl.reasons) + "</ol>"
           if expl.reasons else "")
        + "</div>", unsafe_allow_html=True)
    st.markdown("<div class='fwa-caveat'>" + " ".join(html.escape(c) for c in expl.caveats)
                + "</div>", unsafe_allow_html=True)


# ---------------------------------------------------------------- failures


def friendly_error(what: str, how_to_fix: str, exc: BaseException | None = None) -> None:
    """A plain message instead of a traceback: what is missing and how to fix it."""
    st.warning(f"**{what}** {how_to_fix}")
    if exc is not None:
        _log.error("UI section failed: %s: %s: %s", what, type(exc).__name__, exc)
        with st.expander("Technical details", expanded=False):
            st.code(f"{type(exc).__name__}: {exc}")


@contextmanager
def friendly_failure(what: str, how_to_fix: str = "Try another dataset, or ask an administrator "
                                                   "to check the application log."):
    """``with friendly_failure("the model charts"):`` - never a raw traceback on screen."""
    try:
        yield
    except Exception as exc:  # noqa: BLE001 - the whole point is to catch everything
        # Streamlit's own control flow (st.stop, st.rerun) must pass through.
        if type(exc).__name__ in ("StopException", "RerunException", "RerunData"):
            raise
        friendly_error(f"Couldn't show {what}.", how_to_fix, exc)
