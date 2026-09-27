"""Fixtures for the integration tests: one real pipeline run, shared.

The run is the whole system — adapter, features, catalogue, statistics, graph,
models, documents, AI, correlation — over the real ``claims.csv``. It takes
well under a minute and is built once per session, because the properties these
tests assert (reproducibility, report generation, the §6.3 release gates) are
properties *of a run*, and asserting them against different runs would prove
nothing about any of them.
"""

from __future__ import annotations

import pytest

from fwa.pipeline import run_pipeline

#: Enough synthetic documents for the NLP controls to have something to read,
#: far fewer than the default, because generating them is the slowest step and
#: none of these tests depend on the corpus size.
DOCUMENTS = 150


@pytest.fixture(scope="session")
def pipeline(project_root):
    return run_pipeline(
        project_root / "data" / "claims.csv",
        generate_documents=DOCUMENTS,
        document_dir=project_root / "data" / "synthetic_documents",
        enable_ai=True,
        verbose=False,
    )


@pytest.fixture(scope="session")
def metrics(pipeline):
    from fwa.evaluation.metrics import MetricSuite

    return MetricSuite(pipeline)


@pytest.fixture(scope="session")
def gate_results(pipeline, metrics):
    from fwa.evaluation.gates import ReleaseGates

    return ReleaseGates(pipeline, metrics).evaluate()
