"""Model layer (Type M) — §4.8 unsupervised models, SHAP, drift, gates."""

from .anomaly import AnomalyModels, ModelScores, SplitPlan
from .explain import ShapExplainer, humanise_feature, FEATURE_UNITS
from .drift import DriftMonitor, DriftResult, population_stability_index
from .gate import ModelPromotionGate, GateVerdict, GateCriterion
from .supervised_gate import SupervisedGate, SupervisedGateVerdict, PROMOTION_BLOCKED
from .clusters import NovelClusterDiscovery, NovelCluster
from .layer import ModelLayer

__all__ = [
    "AnomalyModels", "ModelScores", "SplitPlan",
    "ShapExplainer", "humanise_feature", "FEATURE_UNITS",
    "DriftMonitor", "DriftResult", "population_stability_index",
    "ModelPromotionGate", "GateVerdict", "GateCriterion",
    "SupervisedGate", "SupervisedGateVerdict", "PROMOTION_BLOCKED",
    "NovelClusterDiscovery", "NovelCluster", "ModelLayer",
]
