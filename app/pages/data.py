"""Data — load a claim file and re-run every control and every model on it.

This page exists because the rest of the application makes a claim it could not
previously support. Every page describes "the dataset", the reports say what
the run found, and the Rule Registry marks a control
``NOT_EXECUTABLE_ON_THIS_DATASET`` — all of which is only meaningful if the
dataset can actually be something other than the one file that shipped.

Loading a file here does not re-render the pages with new numbers in them. It
re-runs the pipeline end to end: the adapter maps the source schema into the
canonical model, the feature store rebuilds, all 164 controls are re-evaluated,
the peer groups and Beta priors are refitted, the graph is rebuilt, the
unsupervised models are retrained on the new temporal split, and the promotion
gate is re-run against the transparent composite. Nothing is carried over from
the previous dataset except the rule catalogue and the parameter registry,
which are governance artefacts and not properties of the data.

The comparison table below is the part worth reading. Two datasets differ far
more interestingly in *which controls can run at all* than in how many signals
they produce: a file that carries an authorisation table brings a whole
scenario family to life, and a file without ``agent_id`` kills the
distribution-channel controls outright. Those transitions are reported
explicitly rather than left for the reader to diff by eye.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from common import (
    banner, dataframe, empty_state, note, pipeline_for, require_page, session_banner, tiles,
)
from dataset import (
    ADAPTER_LABELS, DatasetSpec, active_dataset, compare_runs, default_spec, loaded_datasets,
    model_summary, set_active_dataset, spec_from_path, spec_from_upload,
)
from fwa.auth import Permission

RESULT_KEY = "fwa_previous_run_key"


def _adapter_picker(key: str) -> str:
    options = list(ADAPTER_LABELS)
    return st.selectbox(
        "Source schema",
        options,
        format_func=lambda a: ADAPTER_LABELS[a],
        key=key,
        help=(
            "Which adapter maps this file's columns into the canonical model. The wrong adapter "
            "does not silently mis-map: columns it cannot find are recorded as missing and the "
            "controls that need them are classified NOT_EXECUTABLE_ON_THIS_DATASET."
        ),
    )


def _activate(spec: DatasetSpec, state) -> None:
    """Make ``spec`` the active dataset and run the pipeline over it."""
    previous = active_dataset()
    st.session_state[RESULT_KEY] = previous.key if previous.key != spec.key else None
    set_active_dataset(spec)
    with st.spinner(f"Running every control and model over {spec.name}…"):
        pipeline_for(spec)
    st.success(f"**{spec.name}** is now the active dataset. Every page below reflects it.")
    st.rerun()


def render() -> None:
    state = require_page("Data", Permission.LOAD_DATASET)
    session_banner(state)

    st.markdown("# Data")
    banner()

    st.markdown(
        "Load a claim file and the whole engine re-runs on it: the adapter, the feature store, "
        "all 164 controls, the peer groups and their fitted priors, the graph, the unsupervised "
        "models and the promotion gate. Nothing is reused from the previous dataset except the "
        "rule catalogue and the parameter registry, which are governance artefacts rather than "
        "properties of the data."
    )

    # ---------------------------------------------------------------- load
    st.markdown("## Load a dataset")
    tab_upload, tab_path, tab_loaded = st.tabs(
        ["Upload a file", "Point at a file on this machine", "Datasets loaded this session"]
    )

    with tab_upload:
        st.caption(
            "A delimited claim-header extract, one row per claim. The file is written to "
            "`data/uploads/` under a content-addressed name so that re-uploading the same file "
            "resolves to the same cached run rather than repeating it."
        )
        uploaded = st.file_uploader("Claim file", type=["csv", "txt", "tsv"],
                                    label_visibility="collapsed")
        adapter = _adapter_picker("upload_adapter")
        if st.button("Run the engine on this file", type="primary", disabled=uploaded is None,
                     key="run_upload"):
            try:
                spec = spec_from_upload(uploaded, adapter)
            except Exception as exc:
                st.error(f"That file could not be read: {exc}")
            else:
                _activate(spec, state)

    with tab_path:
        st.caption(
            "A path avoids pushing a large extract through the browser, and is the only route "
            "that can reach a file the browser is not permitted to read — which is the ordinary "
            "case for anything held on a work share. The file is read in place and not copied."
        )
        raw = st.text_input("Full path to the file", placeholder=r"C:\claims\september.csv")
        adapter_p = _adapter_picker("path_adapter")
        # Deliberately NOT disabled on an empty box. A text input does not tell
        # Streamlit it has changed until it loses focus, so a button gated on
        # its contents stays greyed out while the path sits in it — which reads
        # as the tool having rejected the path rather than as not having seen it
        # yet. Validating on the click, and saying what is wrong, is honest in a
        # way that an inert button is not.
        if st.button("Run the engine on this file", type="primary", key="run_path"):
            if not raw.strip():
                st.error("Enter the full path to a delimited claim file.")
            else:
                try:
                    spec = spec_from_path(raw, adapter_p)
                except Exception as exc:
                    st.error(str(exc))
                else:
                    _activate(spec, state)

    with tab_loaded:
        history = loaded_datasets()
        st.caption(
            "Each run is cached on the content hash of its input file, so switching back to a "
            "dataset loaded earlier costs nothing and does not discard what is already computed."
        )
        dataframe(pd.DataFrame([s.to_row() for s in history]))
        current = active_dataset()
        for spec in history:
            cols = st.columns([4, 1])
            cols[0].markdown(f"**{spec.name}** — {spec.meta_line()}")
            if spec.key == current.key:
                cols[1].markdown("_active_")
            elif cols[1].button("Switch", key=f"switch_{spec.key}"):
                _activate(spec, state)
        if len(history) > 1 or history[0].origin != "shipped":
            if st.button("Back to the shipped dataset", key="reset_default"):
                _activate(default_spec(), state)

    # ------------------------------------------------------- what this run did
    spec = active_dataset()
    result = pipeline_for(spec)

    st.markdown("## What this dataset supports")
    summary = result.evaluation.summary()
    registry_summary = result.registry.summary()
    tiles([
        ("Claim rows", f"{len(result.claims):,}", "as mapped into the canonical model"),
        ("Controls in the catalogue", f"{len(result.registry):,}",
         "governance artefact — the same on every dataset"),
        ("Controls that could run", f"{summary['controls_that_ran']:,}",
         "the rest lack a required canonical field on this file"),
        ("Signals raised", f"{len(result.signals):,}", "a signal is not a fraud finding"),
        ("Cases correlated", f"{len(result.cases.cases):,}", "after de-duplication and capping"),
    ])

    note(
        "A control that did not run on this file is not a control that failed. It is a control "
        "whose required canonical fields this source does not carry — which is a fact about the "
        "data, recorded as such, rather than a silent zero."
    )

    absent = result.dataset.reasons.get("__absent_source_columns__")
    if absent:
        st.warning(f"**Columns this file does not have.** {absent}")

    support = registry_summary["by_data_support"]
    st.markdown("### Data support across the catalogue on this file")
    dataframe(pd.DataFrame([
        {"data support": "EXECUTABLE", "controls": support.get("EXECUTABLE", 0),
         "means": "Every canonical field this control needs is present."},
        {"data support": "PARTIAL", "controls": support.get("PARTIAL", 0),
         "means": "Runs against a stand-in for the field it actually wants, and says so on "
                  "every signal it raises."},
        {"data support": "NOT_EXECUTABLE_ON_THIS_DATASET",
         "controls": support.get("NOT_EXECUTABLE_ON_THIS_DATASET", 0),
         "means": "A required canonical field is absent from this source. Recorded with the "
                  "field that would unlock it, rather than run on a substitute."},
    ]))

    # -------------------------------------------------------- model re-fitting
    st.markdown("## Unsupervised models on this dataset")
    st.markdown(
        "Every model is refitted on this file's own temporal and entity-isolated split. A model "
        "fitted on one dataset is never carried across to another: its thresholds, its peer "
        "baselines and its calibration are all properties of the population it was fitted on."
    )
    models = model_summary(result)
    dataframe(models)
    if result.models is not None and not result.models.gate_verdicts:
        note("The promotion gate has not yet been run on this dataset — open the Models page "
             "to run it.")

    # ---------------------------------------------------------- the comparison
    previous_key = st.session_state.get(RESULT_KEY)
    previous = None
    if previous_key:
        for candidate in loaded_datasets():
            if candidate.key == previous_key:
                previous = pipeline_for(candidate)
                break

    st.markdown("## Control-by-control outcome")
    if previous is None:
        st.caption(
            "Load a second dataset and this table gains a comparison column showing which "
            "controls became runnable, which stopped being runnable, and which found a "
            "different number of signals."
        )
    else:
        frame_all = compare_runs(result, previous)
        changed = frame_all[~frame_all["change"].isin(["Unchanged", "—"])]
        tiles([
            ("Newly runnable", f"{(frame_all['change'] == 'Newly runnable').sum():,}",
             "this file carries fields the previous one did not"),
            ("No longer runnable", f"{(frame_all['change'] == 'No longer runnable').sum():,}",
             "fields the previous file carried and this one does not"),
            ("Changed signal count", f"{len(changed):,}", "of the controls that ran on both"),
        ])
        if not changed.empty:
            st.markdown("**What changed**")
            dataframe(changed, height=320)

    frame = compare_runs(result, previous)
    only_ran = st.checkbox("Show only the controls that ran on this dataset", value=False)
    if only_ran:
        frame = frame[~frame["status on this dataset"].str.startswith("Cannot run")]
    if frame.empty:
        empty_state("No controls to show", "No control in the catalogue matched this filter.")
    else:
        dataframe(frame, height=460)

    st.download_button(
        "Download this table as CSV",
        frame.to_csv(index=False).encode("utf-8"),
        file_name=f"control_outcomes_{spec.short_sha}.csv",
        mime="text/csv",
    )
