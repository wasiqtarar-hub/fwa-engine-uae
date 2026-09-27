"""Data — choose which claim file the engine works on, and see what it could do with it.

Loading a file here does not re-render the pages with new numbers in them. It
re-runs the pipeline end to end: the adapter maps the source layout into the
canonical model, the feature store rebuilds, all 164 controls are re-evaluated,
the peer groups and Beta priors are refitted, the graph is rebuilt, the
unsupervised models are retrained on the new temporal split, and the promotion
gate can be re-run against the transparent composite. Nothing is carried over
from the previous dataset except the rule catalogue and the parameter registry,
which are governance artefacts and not properties of the data.

The page speaks to a claims reviewer, not a data engineer. The default view
says, in words, which checks ran fully, which ran in a simplified form and
which could not run, grouped by the information the file lacks; the rule ids,
the canonical field names and the raw comparison table are one click away.

Two datasets differ far more interestingly in *which controls can run at all*
than in how many signals they produce, so the comparison with the previously
loaded file is reported in a sentence rather than left for the reader to diff.
"""

from __future__ import annotations

import html

import pandas as pd
import streamlit as st

from common import (
    banner, dataframe, empty_state, friendly_error, friendly_failure, friendly_table,
    headline_cards, help_text, note, page_header, pipeline_for, require_page, session_banner,
    synthetic_banner,
)
from components.data_help import (
    adapter_help, adapter_labels, comparison_sentence, coverage, coverage_frame, dataset_info,
    dataset_title, friendly_load_problem, model_sentences, needs_short, plural, shipped_specs,
)
from dataset import (
    active_dataset, compare_runs, loaded_datasets, model_parameters, model_summary,
    set_active_dataset, spec_from_path, spec_from_upload,
)
from fwa.auth import Permission
from fwa.presentation import standard_text

RESULT_KEY = "fwa_previous_run_key"
FLASH_KEY = "fwa_data_flash"
AUTO = "__auto__"


# ---------------------------------------------------------------------------
# widgets
# ---------------------------------------------------------------------------


def _adapter_picker(key: str) -> str:
    labels = adapter_labels()
    options = [AUTO] + list(labels)
    names = {AUTO: "Work it out from the file (recommended)", **labels}
    return st.selectbox(
        "File format",
        options,
        format_func=lambda a: names.get(a, a),
        key=key,
        help=("What it does: " + (help_text("source_format") or "Tells the engine how the "
              "columns in your file are laid out.") + "\n\n" + adapter_help() + "\n\nWhen to change it: leave it on 'Work it "
              "out from the file' unless your file comes from a regulator's system (Shafafiya "
              "or eClaimLink); a .zip or folder is read as UAE multi-table and a single .csv as "
              "a generic claim list."),
    )


def _resolve_adapter(choice: str, name: str) -> str:
    if choice != AUTO:
        return choice
    labels = adapter_labels()
    lowered = str(name).lower().rstrip("\\/")
    looks_multi = lowered.endswith(".zip") or "." not in lowered.rsplit("\\", 1)[-1].rsplit("/", 1)[-1]
    if looks_multi and "uae_multitable" in labels:
        return "uae_multitable"
    return "generic_india_tpa"


def _activate(spec) -> None:
    """Run the pipeline on ``spec`` first; only switch to it if the run succeeds."""
    previous = active_dataset()
    try:
        with st.spinner(f"Running every check and model over {spec.name}. This can take a "
                        "minute or two for a large file…"):
            pipeline_for(spec)
    except Exception as exc:  # noqa: BLE001 - a bad file must never show a traceback
        what, how = friendly_load_problem(exc)
        friendly_error(what, how + " The file that was in use before is still in use.", exc)
        return
    st.session_state[RESULT_KEY] = previous.key if previous.key != spec.key else None
    set_active_dataset(spec)
    st.session_state[FLASH_KEY] = (
        f"**{dataset_title(spec)}** is now the file in use. Every page now describes it.")
    st.rerun()


def _load_failed(exc: BaseException) -> None:
    what, how = friendly_load_problem(exc)
    friendly_error(what, how, exc)


# ---------------------------------------------------------------------------
# choosing a file
# ---------------------------------------------------------------------------


def _choose_section() -> None:
    st.markdown("## Choose the claim file")
    current = active_dataset()
    source = st.radio(
        "Where should the claims come from?",
        ["shipped", "upload", "path"],
        format_func={
            "shipped": "A demo file that comes with the app",
            "upload": "Upload a file from my computer (.csv or .zip)",
            "path": "A file or folder already on this machine (type its full path)",
        }.get,
        horizontal=True,
        key="data_source_choice",
        help="What it does: picks how you give the engine a claim file. When to change it: keep "
             "the demo files to explore the app; upload a file of your own to see what the "
             "engine finds in it; type a path for a large file, or one on a shared drive that "
             "the browser cannot open.",
    )

    if source == "shipped":
        specs = shipped_specs()
        if not specs:
            empty_state("No demo files were found",
                        "The demo files live in the data folder of the app. Ask an administrator "
                        "to restore them, or upload a file of your own instead.")
            return
        keys = [s.key for s in specs]
        index = keys.index(current.key) if current.key in keys else 0
        captions = []
        for s in specs:
            info = dataset_info(s)
            rows = f"{s.rows:,} claims. " if getattr(s, "rows", 0) else ""
            tail = " (Default.)" if info.get("default") else ""
            captions.append(f"{rows}{info.get('one_line', '')} {info.get('use_when', '')}{tail}"
                            .strip())
        choice = st.radio(
            "Demo files",
            list(range(len(specs))),
            index=index,
            format_func=lambda i: dataset_title(specs[i]),
            captions=captions,
            key="data_shipped_choice",
            help="What it does: chooses one of the files that come with the app. When to "
                 "change it: the UAE demo is the default because most checks and both "
                 "pattern-finding models can run on it; the two older files show how much less "
                 "the engine can do with a file that has only claim summaries. Files marked "
                 "SYNTHETIC were generated by a program; no row is a real person or claim.",
        )
        chosen = specs[choice]
        if chosen.key == current.key:
            st.caption("This is the file in use now.")
        elif st.button("Use this file", type="primary", key="use_shipped",
                       help="Runs every check and model over the chosen file and makes it the "
                            "file every page describes. A file used earlier in this session "
                            "opens instantly."):
            _activate(chosen)

    elif source == "upload":
        st.caption("A .csv file with one row per claim, or a .zip holding several tables "
                   "(claims, service lines, providers, patients and so on). The file is kept in "
                   "the app's uploads folder under a name based on its contents, so uploading "
                   "the same file again opens instantly.")
        uploaded = st.file_uploader(
            "Claim file", type=["csv", "txt", "tsv", "zip"],
            help="What it does: sends a file from your computer to the app. When to use it: for "
                 "a file of your own up to a few hundred megabytes; for anything larger, or on a "
                 "shared drive, use the 'already on this machine' option instead.",
        )
        choice = _adapter_picker("upload_adapter")
        if uploaded is not None and choice == AUTO:
            st.caption("It will be read as: **" + adapter_labels().get(
                _resolve_adapter(AUTO, uploaded.name), "") + "**.")
        if st.button("Run the engine on this file", type="primary", disabled=uploaded is None,
                     key="run_upload",
                     help="Runs all 164 checks and both pattern-finding models over the file. "
                          "The file in use now stays in use if anything goes wrong."):
            try:
                spec = spec_from_upload(uploaded, _resolve_adapter(choice, uploaded.name))
            except Exception as exc:  # noqa: BLE001
                _load_failed(exc)
            else:
                _activate(spec)

    else:
        st.caption("Type the full location of a .csv file, a .zip, or a folder of tables. The "
                   "file is read where it is and never copied.")
        raw = st.text_input(
            "Full path to the file or folder", placeholder=r"C:\claims\september.csv",
            help="What it does: tells the app where the file is on this computer or a shared "
                 "drive. When to use it: for large files, or files the browser is not allowed "
                 "to open. Tip: in Windows Explorer, hold Shift, right-click the file and "
                 "choose 'Copy as path'.",
        )
        choice = _adapter_picker("path_adapter")
        # Deliberately NOT disabled on an empty box: a text input does not tell
        # Streamlit it has changed until it loses focus, so a gated button would
        # look like the tool had rejected the path. Validate on the click.
        if st.button("Run the engine on this file", type="primary", key="run_path",
                     help="Runs all 164 checks and both pattern-finding models over the file. "
                          "The file in use now stays in use if anything goes wrong."):
            if not raw.strip():
                friendly_error("No location was entered.",
                               "Type the full path to a .csv file, a .zip or a folder, then "
                               "press the button again.")
            else:
                try:
                    spec = spec_from_path(raw, _resolve_adapter(choice, raw.strip().strip('"')))
                except Exception as exc:  # noqa: BLE001
                    _load_failed(exc)
                else:
                    _activate(spec)

    _history_section(current)


def _history_section(current) -> None:
    history = loaded_datasets()
    if not any(s.key == current.key for s in history):
        history = history + [current]
    with st.expander(f"Files used in this session ({len(history)})", expanded=False):
        st.caption("Each file's results are kept while the app is running, so switching back "
                   "to one costs nothing.")
        for spec in history:
            cols = st.columns([5, 1])
            rows = f"{spec.rows:,} claims" if getattr(spec, "rows", 0) else "size not yet known"
            cols[0].markdown(f"**{html.escape(dataset_title(spec))}** · {rows} · read as "
                             f"{html.escape(adapter_labels().get(spec.adapter, spec.adapter))}")
            if spec.key == current.key:
                cols[1].markdown("_in use_")
            elif cols[1].button("Switch", key=f"switch_{spec.key}",
                                help="Make this the file every page describes."):
                _activate(spec)
        dataframe(pd.DataFrame([s.to_row() for s in history]))


# ---------------------------------------------------------------------------
# what the engine could do with this file
# ---------------------------------------------------------------------------


def _previous_result():
    previous_key = st.session_state.get(RESULT_KEY)
    if not previous_key:
        return None, None
    for candidate in loaded_datasets():
        if candidate.key == previous_key:
            try:
                return candidate, pipeline_for(candidate)
            except Exception:  # noqa: BLE001
                return None, None
    return None, None


def _coverage_section(spec, result, cov, prev_spec, previous) -> None:
    st.markdown("## What the engine could do with this file")
    headline_cards([
        (f"{cov.full:,}", ("check" if cov.full == 1 else "checks")
         + " ran fully: the file had everything needed."),
        (f"{cov.simplified:,}", ("check" if cov.simplified == 1 else "checks")
         + " ran in a simplified form, on a stand-in measure. Their flags say so."),
        (f"{cov.not_run:,}", ("check" if cov.not_run == 1 else "checks")
         + " could not run because the file lacks information needed. None was guessed."),
        (f"{len(result.signals):,}", "flags raised. Each is a reason to look, not proof of fraud."),
        (f"{len(result.cases.cases):,}", "cases for review, after related flags were grouped."),
    ])
    st.markdown(f"**{cov.headline()}**")

    if cov.unlocked or cov.upgraded:
        bits = []
        if cov.unlocked:
            bits.append(f"{plural(cov.unlocked, 'check')} that cannot run on a file of claim "
                        "summaries alone could run here, because this file has the extra tables")
        if cov.upgraded:
            bits.append(f"{plural(cov.upgraded, 'check')} ran in full instead of on a stand-in")
        st.markdown("On this file, " + "; and ".join(bits) + ".")

    if cov.not_run and cov.groups:
        st.markdown("### Why some checks could not run")
        st.caption("Grouped by the information the file is missing. A check that needs more "
                   "than one kind of information is counted under each, so the numbers can add "
                   "up to more than the checks that could not run.")
        lines = [f"- Needs **{html.escape(needs_short(t))}** ({plural(n, 'check')})"
                 for t, _, n in cov.groups[:8]]
        st.markdown("\n".join(lines))
        if len(cov.groups) > 8:
            with st.expander(f"The other {len(cov.groups) - 8} kinds of missing information"):
                st.markdown("\n".join(
                    f"- Needs **{html.escape(needs_short(t))}** ({plural(n, 'check')})"
                    for t, _, n in cov.groups[8:]))
        with st.expander("What each kind of missing information means", expanded=False):
            st.markdown("\n".join(f"- {html.escape(needs)[:1].upper() + html.escape(needs)[1:]}"
                                  for _, needs, _ in cov.groups))
    note("A check that could not run on this file is not a check that failed. It needs "
         "information this file does not carry, and that is recorded as a fact about the data "
         "rather than shown as a silent zero.")

    if previous is not None and prev_spec is not None:
        st.markdown("### Compared with the file loaded before")
        with friendly_failure("the comparison with the previous file"):
            frame_all = compare_runs(result, previous)
            st.markdown(comparison_sentence(frame_all, dataset_info(spec)["name"],
                                            dataset_info(prev_spec)["name"]))


def _table_section(spec, result, cov, previous) -> None:
    st.markdown("## Check by check")
    prev_frame = None
    if previous is not None:
        with friendly_failure("the comparison column"):
            prev_frame = compare_runs(result, previous)
    frame = coverage_frame(cov, prev_frame)
    if frame.empty:
        empty_state("No checks to show", "The checks library is empty on this run.")
        return

    show = st.selectbox(
        "Which checks to show",
        ["all", "found", "full", "partial", "not_run"],
        format_func={
            "all": "All checks",
            "found": "Only checks that found something",
            "full": "Only checks that ran fully",
            "partial": "Only checks that ran in a simplified form",
            "not_run": "Only checks that could not run",
        }.get,
        key="data_check_filter",
        help="What it does: narrows the table below. When to change it: pick 'could not run' "
             "to see what extra information would unlock more checks, or 'found something' to "
             "see which checks produced the flags behind the review queue.",
    )
    view = frame
    if show == "found":
        view = frame[frame["flags raised"] > 0]
    elif show == "full":
        view = frame[frame["_support"] == "EXECUTABLE"]
    elif show == "partial":
        view = frame[frame["_support"] == "PARTIAL"]
    elif show == "not_run":
        view = frame[~frame["_support"].isin(["EXECUTABLE", "PARTIAL"])]
    view = view.drop(columns=["_support", "_outcome"])

    columns = ["check", "on this file", "what happened", "flags raised", "what it needs"]
    if "compared with the previous file" in view.columns:
        columns.append("compared with the previous file")
    friendly_table(
        view, key="data_checks", columns=columns, keep_numeric=["flags raised"], height=460,
        rename={c: c[:1].upper() + c[1:] for c in view.columns},
        empty_title="No checks match this choice",
        empty_body="Pick 'All checks' above to see every check.",
    )
    st.caption("Sorted with the checks that raised the most flags first. Tick 'Show all columns' "
               "for the check ids, the catalogue classification and the exact fields each check "
               "needs.")
    st.download_button(
        "Download this table (CSV)",
        view.to_csv(index=False).encode("utf-8"),
        file_name=f"checks_on_this_file_{spec.short_sha}.csv", mime="text/csv",
        help="Saves the table as shown, with every column, for a spreadsheet.",
    )
    with st.expander("Technical details: the raw control-by-control comparison", expanded=False):
        raw = compare_runs(result, previous)
        dataframe(raw, height=360)
        st.download_button(
            "Download the raw comparison (CSV)", raw.to_csv(index=False).encode("utf-8"),
            file_name=f"control_outcomes_{spec.short_sha}.csv", mime="text/csv",
            key="data_raw_download",
        )


def _models_section(result) -> None:
    st.markdown("## Pattern-finding models on this file")
    st.caption("Both models are refitted on every file; a model fitted on one file is never "
               "carried over to another. " + standard_text("shadow"))
    for sentence in model_sentences(result):
        st.markdown(sentence)
    with st.expander("Technical details: model fit and settings", expanded=False):
        dataframe(model_summary(result))
        params = model_parameters(result)
        if not params.empty:
            dataframe(params, height=300)
        layer = getattr(result, "models", None)
        if layer is not None and not getattr(layer, "gate_verdicts", None):
            st.caption("The promotion gate has not yet been run on this file. The Pattern-finding "
                       "models page runs it.")


def _technical_section(result) -> None:
    with st.expander("Technical details: how the file was read", expanded=False):
        st.markdown(
            "Loading a file re-runs the pipeline end to end: the adapter maps the source layout "
            "into the canonical model, the feature store rebuilds, all 164 controls are "
            "re-evaluated, the peer groups and their Beta priors are refitted, the graph is "
            "rebuilt, the unsupervised models are retrained on the new temporal and "
            "entity-isolated split, and the promotion gate can be re-run against the "
            "transparent composite. Nothing carries over except the rule catalogue and the "
            "parameter registry. Each run is cached on the content hash of its input.")
        absent = result.dataset.reasons.get("__absent_source_columns__")
        if absent:
            st.markdown(f"**Columns this file does not have:** {absent}")
        support = result.registry.summary().get("by_data_support", {})
        dataframe(pd.DataFrame([{"data_support": k, "controls": v} for k, v in support.items()]))
        try:
            dataframe(result.dataset.population_report())
        except Exception:  # noqa: BLE001 - optional detail
            pass


# ---------------------------------------------------------------------------
# page
# ---------------------------------------------------------------------------


def render() -> None:
    state = require_page("Data", Permission.LOAD_DATASET)
    session_banner(state)
    page_header("Data")
    banner()
    synthetic_banner()

    flash = st.session_state.pop(FLASH_KEY, None)
    if flash:
        st.success(flash)

    with friendly_failure("the file chooser", "Reload the page. If it keeps happening, ask an "
                                              "administrator to check the application log."):
        _choose_section()

    spec = active_dataset()
    try:
        result = pipeline_for(spec)
    except Exception as exc:  # noqa: BLE001
        what, how = friendly_load_problem(exc)
        friendly_error(what, how + " Choose one of the demo files above to carry on.", exc)
        return

    cov = coverage(result)
    prev_spec, previous = _previous_result()

    with friendly_failure("the summary of which checks ran"):
        _coverage_section(spec, result, cov, prev_spec, previous)
    with friendly_failure("the check-by-check table"):
        _table_section(spec, result, cov, previous)
    with friendly_failure("the model summary"):
        _models_section(result)
    _technical_section(result)
