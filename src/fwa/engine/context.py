"""Everything a control is allowed to see when it executes.

A control receives a :class:`ControlContext` and nothing else. That is a
deliberate narrowing, and it is what makes two governance properties checkable
rather than hoped-for:

* **No labels.** The context exposes ``features`` as a
  :class:`~fwa.features.frame.FeatureFrame`, which cannot contain label columns,
  and exposes no path at all to the held-out label frame. A control physically
  cannot read ``fraud_label``.
* **Tenant scoping.** The context carries exactly one ``tenant_id`` and its
  frames are already filtered to it. A control has no way to reach another
  tenant's rows, which is what the tenant-isolation test category (§6.2
  category 9) verifies for every control uniformly.

The context also carries the reference versions every signal must stamp
(§3.10 reproducibility): the parameter-registry fingerprint, the FX table, the
peer baseline version and the rule's own fingerprint.
"""

from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass, field
from typing import Any, Callable

import pandas as pd

from ..config import AppConfig
from ..features.frame import FeatureFrame

__all__ = ["ControlContext"]


@dataclass
class ControlContext:
    """The execution environment for one control against one tenant."""

    config: AppConfig
    tenant_id: str

    #: Claim-header frame, already tenant-scoped, with derived columns attached.
    claims: pd.DataFrame

    #: Claim-aligned features, label-free by construction.
    features: FeatureFrame

    #: Canonical dataset (for population reports and table availability checks).
    dataset: Any = None

    #: Peer service over the claim frame (§4.6).
    peers: Any = None

    #: Provider-level peer service, used by provider-subject controls.
    provider_peers: Any = None

    #: Provider-level aggregate metrics (one row per provider).
    provider_metrics: pd.DataFrame | None = None

    #: Episode frame and claim → episode map (§4.12).
    episodes: pd.DataFrame | None = None
    episode_map: dict[str, str] = field(default_factory=dict)

    #: Claim lineage index (§3.6, §6.3). Empty on the claim extract — see LineageIndex.
    lineage: Any = None

    #: Graph snapshots and community assignments (§4.10), when the graph layer ran.
    graph: Any = None

    #: Synthetic document corpus (§4.9), when the NLP layer ran.
    documents: Any = None

    #: Model outputs (scores, SHAP), when the model layer ran.
    models: Any = None

    #: Statistical helpers.
    composite: Any = None
    changepoints: list[Any] = field(default_factory=list)

    #: Reference-version stamps applied to every signal this run produces.
    reference_versions: dict[str, str] = field(default_factory=dict)

    #: The run's as-of date for parameter resolution when a row has no service date.
    run_date: _dt.date = field(default_factory=_dt.date.today)

    #: Audit log, for controls that must record a governance event.
    audit: Any = None

    # ------------------------------------------------------------------ utils

    def cfg(self, key: str, as_of: Any = None) -> Any:
        """Resolve a governed parameter as of a service date (§3.6, §3.9)."""
        return self.config.get(key, as_of=as_of or self.run_date)

    def to_aed(self, amount: float, service_date: Any) -> float:
        return self.config.fx.convert(amount, "INR", service_date)

    def versions(self, control, extra: dict[str, str] | None = None) -> dict[str, str]:
        """The reference-version bundle stamped onto a signal."""
        out = {
            "rule": f"{control.rule_id}@{control.version}",
            "rule_fingerprint": control.fingerprint(),
            "parameters": self.config.fingerprint(),
            "adapter": getattr(self.dataset, "source_system", "unknown"),
            **self.reference_versions,
        }
        if extra:
            out.update(extra)
        return out

    def period_of(self, value: Any, freq: str = "M") -> str:
        from .evallib import period_bucket

        return period_bucket(value, freq)
