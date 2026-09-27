"""Evaluation layer — §6.5 metrics, §6.6 protocol, Table 6.1 gates, reports.

THIS IS THE ONLY PACKAGE PERMITTED TO READ THE HELD-OUT LABELS (build brief
§3.3). Everything upstream loads features through a FeatureFrame, which cannot
contain them, and tests/governance/test_label_leakage.py enforces the boundary.
"""

from .metrics import MetricSuite, Metric, NOT_MEASURABLE
from .protocol import EvaluationProtocol, ProtocolResult
from .gates import ReleaseGates, GateResult
from .traceability import TraceabilityMatrix, REQUIREMENTS
from .reports import ReportBuilder

__all__ = [
    "MetricSuite", "Metric", "NOT_MEASURABLE",
    "EvaluationProtocol", "ProtocolResult",
    "ReleaseGates", "GateResult",
    "TraceabilityMatrix", "REQUIREMENTS",
    "ReportBuilder",
]
