"""Validation Report — the §11.2 artefacts rendered in-app and downloadable."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from common import (
    ROOT, banner, dataframe, empty_state, note, pipeline, require_page, session_banner,
    status_chip, tiles,
)
from fwa.auth import Permission

REPORTS = ROOT / "reports"

MARKDOWN_ARTEFACTS = {
    "Validation report": "validation_report.md",
    "Release-gate report": "release_gate_report.md",
    "Model card": "model_card.md",
    "Limitations": "limitations.md",
}

CSV_ARTEFACTS = {
    "Control coverage matrix (all 164 controls)": "control_coverage_matrix.csv",
    "Traceability matrix (Appendix D, regenerated from code)": "traceability_matrix.csv",
    "Operational metrics": "metrics.csv",
    "Evaluation protocol": "evaluation_protocol.csv",
    "Precision by ground-truth source": "precision_by_ground_truth_source.csv",
    "Recall by fraud type": "recall_by_fraud_type.csv",
    "Parameter registry": "parameter_registry.csv",
    "Canonical population report": "canonical_population_report.csv",
    "Control run report": "control_run_report.csv",
    "Case queue": "case_queue.csv",
}


def render() -> None:
    state = require_page("Validation Report", Permission.VIEW_VALIDATION_REPORT)
    session_banner(state)

    st.markdown("# Validation report")
    banner()

    result = pipeline()

    if not REPORTS.exists() or not (REPORTS / "validation_report.md").exists():
        empty_state(
            "No reports have been generated yet",
            "Run `python -m fwa.run_validation --data data/claims_demo_synthetic.csv` to regenerate every "
            "artefact in reports/. The run is deterministic: fixed seed, recorded fingerprint.",
        )
        if state.can(Permission.RUN_PIPELINE):
            if st.button("Generate the reports now", type="primary"):
                from fwa.evaluation import ReportBuilder

                with st.spinner("Generating…"):
                    ReportBuilder(result, REPORTS).build_all()
                st.rerun()
        return

    tiles([
        ("Seed", f"{result.seed}", "cfg.random_seed"),
        ("Parameter fingerprint", result.config.fingerprint(), "stamped onto every signal"),
        ("Artefacts", f"{len(list(REPORTS.glob('*')))}", "in reports/"),
    ])
    note(
        "Every number in every artefact here is a measured output of this run, a configuration "
        "threshold labelled as such, or NOT_MEASURABLE_ON_THIS_DATASET. Nothing is "
        "illustrative-but-unlabelled."
    )

    chosen = st.selectbox("Artefact", list(MARKDOWN_ARTEFACTS) + list(CSV_ARTEFACTS))

    if chosen in MARKDOWN_ARTEFACTS:
        path = REPORTS / MARKDOWN_ARTEFACTS[chosen]
        text = path.read_text(encoding="utf-8")
        st.download_button(f"Download {path.name}", text, file_name=path.name,
                           mime="text/markdown")
        st.markdown("---")
        st.markdown(text)
    else:
        path = REPORTS / CSV_ARTEFACTS[chosen]
        if not path.exists():
            empty_state("Not generated", f"{path.name} was not produced by the last run.")
            return
        frame = pd.read_csv(path)
        st.download_button(f"Download {path.name}", path.read_bytes(), file_name=path.name,
                           mime="text/csv")
        if "status" in frame.columns:
            counts = frame["status"].value_counts()
            st.markdown(
                " ".join(status_chip(str(k)) + f" {v}" for k, v in counts.items()),
                unsafe_allow_html=True,
            )
        dataframe(frame, height=520)

    if state.can(Permission.RUN_PIPELINE):
        st.markdown("---")
        if st.button("Regenerate all reports from the current run"):
            from fwa.evaluation import ReportBuilder

            with st.spinner("Regenerating…"):
                written = ReportBuilder(result, REPORTS).build_all()
            st.success(f"Regenerated {len(written)} artefact(s).")
