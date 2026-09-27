"""SHAP explanation in original units, beside the peer value (manuscript §4.8).

    "produces TOP CONTRIBUTING FEATURES IN ORIGINAL UNITS WITH PEER COMPARISON,
     via SHAP (Lundberg and Lee, 2017), for every flagged entity"
                                                            — manuscript §4.8

The build brief sharpens the same point: "top contributing features **in
original units, alongside the peer comparison value**, not bare shap
magnitudes. This is what the reviewer sees."

That distinction is the whole design of this module. A raw SHAP value is a
number on the model's internal scale; it tells a reviewer that a feature
mattered but not what the feature *said*. What a reviewer can act on is:

    Amount per inpatient day — AED 2,649/day, peer median AED 912/day (2.9×)
    → pushed this claim's anomaly score up the most

So every contribution is returned with the claim's value in its natural unit,
the peer median in the same unit, the ratio, a plain-language sentence, and the
SHAP magnitude retained for the analyst who wants it.

A model whose contributions cannot be rendered this way fails the model gate's
explanation-quality check (Table 6.1), which is checked in
:mod:`fwa.models.gate`.
"""

from __future__ import annotations

import concurrent.futures as _futures
import logging
import threading
from typing import Any, Callable, Iterable

import numpy as np
import pandas as pd

__all__ = ["ShapExplainer", "FEATURE_UNITS", "humanise_feature", "ExplanationTimeout"]

_log = logging.getLogger(__name__)


class ExplanationTimeout(RuntimeError):
    """A SHAP call exceeded its wall-clock budget."""


#: Stack size for the SHAP worker thread (bytes). A platform limit, not a threshold.
_SHAP_THREAD_STACK_BYTES = 64 * 1024 * 1024


def _bounded(fn: Callable[[], Any], seconds: float | None) -> Any:
    """Run ``fn`` with a wall-clock budget; raise :class:`ExplanationTimeout` past it.

    SHAP cannot be interrupted from outside, so a timed-out call is abandoned
    (its worker thread finishes in the background and its result is discarded)
    rather than killed. What matters for the UI is that the page does not wait
    on it. ``None`` or a non-positive budget runs ``fn`` inline.
    """
    if not seconds or seconds <= 0:
        return fn()
    # Worker threads on Windows get a small default stack, and SHAP's tree
    # traversal recurses deeply; after a long test session that surfaced as a
    # "Windows fatal exception: stack overflow" inside this worker. The thread
    # is created at submit(), so the larger size applies to it alone and the
    # process-wide default is restored immediately afterwards.
    previous = threading.stack_size()
    try:
        threading.stack_size(_SHAP_THREAD_STACK_BYTES)
    except (ValueError, RuntimeError):  # platform refuses a custom size
        previous = None
    pool = _futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="shap")
    try:
        future = pool.submit(fn)
    finally:
        if previous is not None:
            threading.stack_size(previous)
    try:
        return future.result(timeout=seconds)
    except _futures.TimeoutError as exc:
        future.cancel()
        raise ExplanationTimeout(f"SHAP call exceeded its {seconds:g}s budget") from exc
    finally:
        pool.shutdown(wait=False)

#: feature name -> (label, unit). Anything not listed renders in its raw name
#: with a blank unit, which is a prompt to add it here rather than a silent gap.
FEATURE_UNITS: dict[str, tuple[str, str]] = {
    "gross_amount_aed": ("Claim amount", "AED"),
    "approved_amount_aed": ("Approved amount", "AED"),
    "log_gross_amount_aed": ("Claim amount (log scale)", "log AED"),
    "approval_ratio": ("Approved-to-billed ratio", "ratio"),
    "claim_lag_days": ("Days from discharge to claim", "days"),
    "amount_per_los_day": ("Amount per inpatient day", "AED/day"),
    "length_of_stay_days": ("Length of stay", "days"),
    "pharmacy_bill_ratio": ("Pharmacy share of the bill", "ratio"),
    "pharmacy_share_residual": ("Pharmacy share vs diagnosis peers", "ratio"),
    "num_insurers_same_event": ("Insurers on the same event", "count"),
    "days_since_policy_start": ("Policy tenure at service", "days"),
    "discharge_readmit_gap_days": ("Gap to next admission", "days"),
    "is_cashless": ("Cashless claim", "flag"),
    "icd_code_matches_procedure": ("Diagnosis matches procedure", "flag"),
    "coding_mismatch": ("Coding mismatch", "flag"),
    "provider_blacklist_flag": ("Provider exclusion flag", "flag"),
    "previous_fraud_on_policy": ("Prior policy history flag", "flag"),
    "is_early_tenure": ("Early policy tenure", "flag"),
    "readmission_flag": ("Readmission inside the window", "flag"),
    "cob_multi_insurer": ("Multiple insurers", "flag"),
    "member_prior_claim_count": ("Member's prior claims", "count"),
    "member_utilisation_velocity": ("Member claims in the window", "count"),
    "member_prior_gross_aed": ("Member's prior spend", "AED"),
    "member_prior_distinct_providers": ("Member's prior providers", "count"),
    "provider_prior_claim_count": ("Provider's prior claims", "count"),
    "provider_prior_mean_gross_aed": ("Provider's prior mean amount", "AED"),
    "provider_prior_median_gross_aed": ("Provider's prior median amount", "AED"),
    "provider_prior_mean_approval_ratio": ("Provider's prior approval ratio", "ratio"),
    "provider_prior_coding_mismatch_rate": ("Provider's prior mismatch rate", "ratio"),
    "provider_prior_readmission_rate": ("Provider's prior readmission rate", "ratio"),
    "provider_same_day_admissions": ("Admissions at this provider that day", "count"),
    "agent_prior_claim_count": ("Agent's prior claims", "count"),
    "agent_top_provider_share_prior": ("Agent's top-provider share", "ratio"),
    "episode_claim_count": ("Claims in this episode", "count"),
    "episode_total_gross_aed": ("Episode total", "AED"),
    "pharmacy_amount_aed": ("Implied pharmacy amount", "AED"),
}


def humanise_feature(name: str) -> tuple[str, str]:
    if name.endswith("__missing"):
        base = name[: -len("__missing")]
        label, _ = FEATURE_UNITS.get(base, (base.replace("_", " ").capitalize(), ""))
        return (f"{label} — value was MISSING", "flag")
    return FEATURE_UNITS.get(name, (name.replace("_", " ").capitalize(), ""))


def _fmt(value: float | None, unit: str) -> str:
    if value is None or (isinstance(value, float) and value != value):
        return "not available"
    v = float(value)
    if unit == "AED":
        return f"AED {v:,.0f}"
    if unit == "AED/day":
        return f"AED {v:,.0f}/day"
    if unit == "ratio":
        return f"{v:.0%}"
    if unit == "days":
        return f"{v:,.0f} day(s)"
    if unit == "count":
        return f"{v:,.0f}"
    if unit == "flag":
        return "yes" if v >= 0.5 else "no"
    if unit == "log AED":
        return f"{v:,.2f} (log)"
    return f"{v:,.2f}"


class ShapExplainer:
    """Wraps SHAP and renders contributions the way §4.8 requires.

    Never raises: construction failures leave :attr:`available` False with a
    :attr:`failure_reason`, and every per-call failure or timeout comes back as
    an ``EXPLANATION_FAILED`` row (which the promotion gate already counts as a
    failed explanation-quality check) with the reason in :attr:`last_failure`.
    """

    #: Wall-clock budget for one claim's explanation. KernelExplainer work is
    #: already bounded by the kmeans background and ``nsamples``; this is the
    #: backstop for a pathological input so a page never hangs on SHAP.
    explain_timeout_seconds: float = 30.0
    #: Budget for the Models page's global-importance panel.
    importance_timeout_seconds: float = 180.0

    def __init__(self, model, feature_columns: list[str], background: pd.DataFrame) -> None:
        self.model = model
        self.feature_columns = list(feature_columns)
        self.background = background
        self._explainer = None
        self.method = "unavailable"
        self.failure_reason = ""
        #: technical reason for the most recent per-call failure ("" if none).
        self.last_failure = ""
        # SIGN CONVENTION. The system's anomaly score is -score_samples(x), so
        # that HIGHER means MORE ANOMALOUS. A TreeExplainer built on
        # IsolationForest explains the model's own output, where higher means
        # more NORMAL. Without this flip every explanation would read backwards
        # — the features driving a claim's anomaly would be described as having
        # "lowered" its score, which is worse than no explanation at all.
        self._sign = -1.0 if type(model).__name__ == "IsolationForest" else 1.0
        self._kernel_nsamples: int | None = None
        try:
            self._build()
        except Exception as exc:  # noqa: BLE001 - never let construction crash a page
            self._explainer = None
            self.method = "unavailable"
            self.failure_reason = f"SHAP explainer construction failed: {type(exc).__name__}: {exc}"
        if self.failure_reason:
            _log.warning("SHAP unavailable for %s: %s", type(model).__name__, self.failure_reason)

    def _build(self) -> None:
        try:
            import shap
        except Exception as exc:  # pragma: no cover
            self.failure_reason = f"shap is not installed: {exc}"
            return
        if self.background is None or len(self.background) < 2:
            # Both explainers below need a background; KernelExplainer's kmeans
            # summary needs at least two rows to form two centroids.
            try:
                self._explainer = shap.TreeExplainer(self.model)
                self.method = "TreeExplainer"
            except Exception as exc:
                self.failure_reason = (
                    f"No SHAP explainer could be constructed for {type(self.model).__name__}: "
                    f"the background has {0 if self.background is None else len(self.background)} "
                    f"row(s), fewer than the 2 KernelExplainer needs ({exc})."
                )
            return
        try:
            self._explainer = shap.TreeExplainer(self.model)
            self.method = "TreeExplainer"
        except Exception:
            try:
                # KernelExplainer cost is O(background × nsamples) per explanation
                # and LOF has no tree structure to exploit. A 100-row background
                # costs ~2.7s per claim, which is ~9 minutes across a capacity-
                # sized queue — unusable, and the §14.1 requirement is that the
                # queue renders in well under a second. A kmeans-summarised
                # background of 25 centroids with a bounded sample count brings
                # it to ~0.2s with a materially similar attribution ranking.
                # The approximation is recorded in `method` so it appears on the
                # Models page rather than being a silent substitution.
                summarised = shap.kmeans(self.background, min(25, max(2, len(self.background) // 4)))
                columns = list(self.background.columns)

                def _score(X):
                    frame = pd.DataFrame(X, columns=columns)
                    return -self.model.score_samples(frame)

                self._explainer = shap.KernelExplainer(_score, summarised)
                self._kernel_nsamples = 100
                self.method = "KernelExplainer (kmeans-summarised background, 25 centroids)"
            except Exception as exc:  # pragma: no cover
                self.failure_reason = (
                    f"No SHAP explainer could be constructed for {type(self.model).__name__}: {exc}"
                )

    @property
    def available(self) -> bool:
        return self._explainer is not None

    def _shap_kwargs(self) -> dict[str, Any]:
        """The keyword arguments the explainer that was actually built accepts.

        ``silent`` and ``nsamples`` belong to KernelExplainer alone.
        TreeExplainer rejects them outright, which is how the Isolation
        Forest's explanations came back as ``EXPLANATION_FAILED`` while the
        LOF's rendered fine — and, because the promotion gate reads a failed
        sample as a failed explanation-quality check, the model was being held
        in shadow partly for a keyword-argument mistake rather than for a
        property of the model. Both call sites go through here so the two
        cannot drift apart again.
        """
        if self._kernel_nsamples:
            return {"silent": True, "nsamples": self._kernel_nsamples}
        return {}

    def explain(
        self,
        row: pd.Series,
        raw_values: pd.Series,
        peer_values: pd.Series | None = None,
        top_n: int = 5,
    ) -> list[dict[str, Any]]:
        """Return the top contributions, in original units, beside peer medians.

        ``row`` is the scaled model input; ``raw_values`` is the same claim's
        values in their natural units; ``peer_values`` is a Series of peer
        medians keyed by feature name.
        """
        if not self.available:
            return [
                {
                    "feature": "EXPLANATION_UNAVAILABLE",
                    "label": "Explanation unavailable",
                    "note": self.failure_reason
                    or "No SHAP explainer available for this model. The promotion gate requires feature "
                       "attribution for every flagged entity; a model that cannot supply it "
                       "fails the model gate's explanation-quality check.",
                    "plain_language": (
                        "No explanation is available for this model's scores, so they have no "
                        "feature breakdown. The technical reason is in the log."
                    ),
                }
            ]
        try:
            frame = row.to_frame().T
            kwargs = self._shap_kwargs()
            values = _bounded(
                lambda: self._explainer.shap_values(frame, **kwargs), self.explain_timeout_seconds
            )
            arr = np.asarray(values).reshape(-1) * self._sign
            if arr.shape[0] != len(self.feature_columns):
                raise ValueError(
                    f"SHAP returned {arr.shape[0]} attributions for "
                    f"{len(self.feature_columns)} features"
                )
            return self._render(arr, raw_values, peer_values, top_n)
        except Exception as exc:  # noqa: BLE001 - a failed explanation, never a crash
            return [self._failed(exc)]

    def _failed(self, exc: BaseException) -> dict[str, Any]:
        """One ``EXPLANATION_FAILED`` row. The gate counts it as a failed check."""
        timed_out = isinstance(exc, ExplanationTimeout)
        self.last_failure = f"{type(exc).__name__}: {exc}"
        _log.warning("SHAP explanation failed (%s): %s", self.method, self.last_failure)
        return {
            "feature": "EXPLANATION_FAILED",
            "label": "Explanation failed",
            "note": str(exc),
            "plain_language": (
                f"The explanation took longer than {self.explain_timeout_seconds:g} seconds and "
                f"was stopped, so this score has no feature breakdown."
                if timed_out else
                "The explanation for this score could not be calculated, so it has no feature "
                "breakdown. The technical reason is in the log."
            ),
        }

    def _render(
        self,
        arr: np.ndarray,
        raw_values: pd.Series,
        peer_values: pd.Series | None,
        top_n: int,
    ) -> list[dict[str, Any]]:
        order = np.argsort(-np.abs(arr))[:top_n]
        out: list[dict[str, Any]] = []
        for i in order:
            name = self.feature_columns[i]
            label, unit = humanise_feature(name)
            value = raw_values.get(name)
            peer = None if peer_values is None else peer_values.get(name)
            ratio = None
            if peer not in (None, 0) and value is not None and not pd.isna(value) and not pd.isna(peer):
                try:
                    ratio = float(value) / float(peer)
                except ZeroDivisionError:  # pragma: no cover
                    ratio = None
            out.append(
                {
                    "feature": name,
                    "label": label,
                    "unit": unit,
                    "value": None if value is None or pd.isna(value) else float(value),
                    "value_display": _fmt(value, unit),
                    "peer_median": None if peer is None or pd.isna(peer) else float(peer),
                    "peer_display": _fmt(peer, unit),
                    "ratio_to_peer": None if ratio is None else round(ratio, 2),
                    "shap_value": float(arr[i]),
                    "direction": "raised the score" if arr[i] > 0 else "lowered the score",
                    "plain_language": (
                        f"{label}: {_fmt(value, unit)}"
                        + (f", peer median {_fmt(peer, unit)}" if peer is not None and not pd.isna(peer) else "")
                        + (f" ({ratio:.1f}×)" if ratio is not None else "")
                        + f" — {'raised' if arr[i] > 0 else 'lowered'} this claim's anomaly score"
                    ),
                }
            )
        return out

    def global_importance(self, matrix: pd.DataFrame, sample: int = 300) -> pd.DataFrame:
        """Mean |SHAP| per feature — the Models page's summary panel."""
        if not self.available or matrix.empty:
            return pd.DataFrame(columns=["feature", "label", "mean_abs_shap"])
        sub = matrix.sample(min(sample, len(matrix)), random_state=0)
        try:
            kwargs = self._shap_kwargs()
            values = np.asarray(_bounded(
                lambda: self._explainer.shap_values(sub, **kwargs), self.importance_timeout_seconds
            )) * self._sign
            if values.ndim == 3:
                values = values[..., 0]
            mean_abs = np.abs(values).mean(axis=0)
            if len(mean_abs) != len(self.feature_columns):
                raise ValueError(
                    f"SHAP returned {len(mean_abs)} attributions for "
                    f"{len(self.feature_columns)} features"
                )
        except Exception as exc:  # noqa: BLE001
            self.last_failure = f"{type(exc).__name__}: {exc}"
            _log.warning("SHAP global importance failed (%s): %s", self.method, self.last_failure)
            return pd.DataFrame(columns=["feature", "label", "mean_abs_shap"])
        rows = []
        for name, v in zip(self.feature_columns, mean_abs):
            label, unit = humanise_feature(name)
            rows.append({"feature": name, "label": label, "unit": unit, "mean_abs_shap": float(v)})
        return pd.DataFrame(rows).sort_values("mean_abs_shap", ascending=False)
