"""Statistical layer (Type S) — §4.6 peers and shrinkage, §4.7 change detection,
§4.8 transparent composite."""

from .robust import RobustStats, robust_stats, modified_z, robust_residual_frame, percentile_rank
from .shrinkage import BetaPrior, fit_beta_prior, ShrunkRate, shrink_rate, shrink_frame, shrinkage_demonstration
from .peers import PeerGroup, PeerService, INSUFFICIENT_PEER_EVIDENCE
from .changepoint import ChangePoint, SegmentRegistry, cusum, ewma, detect_changepoints, monthly_provider_metrics
from .composite import TransparentComposite, CompositeResult, CompositeFeature, COMPOSITE_FEATURES

__all__ = [
    "RobustStats", "robust_stats", "modified_z", "robust_residual_frame", "percentile_rank",
    "BetaPrior", "fit_beta_prior", "ShrunkRate", "shrink_rate", "shrink_frame", "shrinkage_demonstration",
    "PeerGroup", "PeerService", "INSUFFICIENT_PEER_EVIDENCE",
    "ChangePoint", "SegmentRegistry", "cusum", "ewma", "detect_changepoints", "monthly_provider_metrics",
    "TransparentComposite", "CompositeResult", "CompositeFeature", "COMPOSITE_FEATURES",
]
