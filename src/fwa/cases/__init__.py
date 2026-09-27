"""Case correlation, priority and exposure — §4.11, §4.12."""

from .exposure import (
    Exposure, line_edit_exposure, duplicate_exposure, provider_pattern_exposure,
    network_exposure, model_only_exposure, no_exposure,
)
from .priority import PriorityScorer, PriorityScore, PriorityTerm
from .correlate import Case, CaseCorrelator, CORRELATION_DIMENSIONS, case_fingerprint

__all__ = [
    "Exposure", "line_edit_exposure", "duplicate_exposure", "provider_pattern_exposure",
    "network_exposure", "model_only_exposure", "no_exposure",
    "PriorityScorer", "PriorityScore", "PriorityTerm",
    "Case", "CaseCorrelator", "CORRELATION_DIMENSIONS", "case_fingerprint",
]
