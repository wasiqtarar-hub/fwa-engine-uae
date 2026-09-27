"""The signal contract (manuscript §3.8) and the signal store.

A *signal* is the output of one control execution against one subject. It is
explicitly **not** a finding about conduct — see :data:`fwa.SAFETY_BOUNDARY_STATEMENT`.

Two fields deserve comment because they carry governance weight that is easy to
lose in implementation:

``fact_key``
    The identifier of the *underlying fact* a signal rests on. §3.8 requires
    that "three separately triggered signals resting on the same underlying fact
    do not triple-count the risk score". Evidence capping is implemented by
    grouping on this key and taking ``max(evidence_strength)`` — never the sum.
    A control that forgets to set a meaningful ``fact_key`` therefore degrades
    to "each signal is its own fact", which is the conservative direction only
    when the facts really are independent. The ten-category test suite includes
    an evidence-capping fixture for exactly this reason.

``reference_versions``
    Every version of every piece of reference data the control consulted —
    rule text, parameter-registry fingerprint, peer-baseline version, FX rate,
    model version. This is what makes the §3.10 reproducibility NFR real: a
    peer comparison must be reconstructable from its stored baseline version at
    any later date.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import uuid
from dataclasses import dataclass, field, asdict
from typing import Any, Iterable, Sequence

from ..enums import Disposition, RuleStatus, SignalDomain, Stage

__all__ = ["Signal", "SignalStore", "cap_evidence", "EXPOSURE_NOT_ESTABLISHED"]

#: The literal string §4.11 and §6.5 require beside a model-only lead's gross
#: amount. Gross flagged value is not savings, and this string is how the system
#: is prevented from ever silently implying otherwise.
EXPOSURE_NOT_ESTABLISHED = "exposure not yet established"


def _stable_id(*parts: Any) -> str:
    """Deterministic signal id.

    Deliberately *not* random: idempotency (§6.2 category 8) requires that
    replaying the same input produces no duplicate signal, and the cheapest way
    to guarantee that is to make the identity of a signal a pure function of
    (rule, version, subject, period, fact).
    """
    payload = "|".join("" if p is None else str(p) for p in parts)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]


@dataclass
class Signal:
    """One control execution against one subject (manuscript §3.8)."""

    # ---- identity -----------------------------------------------------------
    signal_id: str
    rule_id: str
    rule_version: str
    scenario_id: str
    tenant_id: str

    # ---- subject ------------------------------------------------------------
    subject_type: str              # claim | member | provider | agent | community | pharmacy
    subject_id: str
    claim_ids: list[str] = field(default_factory=list)

    # ---- timing -------------------------------------------------------------
    event_time: _dt.datetime | None = None       # when the underlying event happened
    detection_time: _dt.datetime | None = None   # when this system detected it
    period_bucket: str = ""                      # e.g. "2024-03" — part of case fingerprinting

    # ---- assessment ---------------------------------------------------------
    score: float = 0.0                 # the control's declared score, [0,100]
    confidence: float = 1.0            # [0,1]; degraded by data quality, OCR, sparse peers
    evidence_strength: float = 0.0     # [0,1]; the input to the §4.11 priority term
    disposition: Disposition = Disposition.MONITOR_ONLY
    reason_code: str = ""
    domain: SignalDomain = SignalDomain.RULE
    stage: Stage = Stage.POSTPAY_DAILY
    rule_status: RuleStatus = RuleStatus.SHADOW

    # ---- money --------------------------------------------------------------
    exposure_aed: float = 0.0
    exposure_basis: str = ""           # human-readable derivation, shown in the UI
    exposure_established: bool = True  # False → render EXPOSURE_NOT_ESTABLISHED

    # ---- evidence -----------------------------------------------------------
    evidence: dict[str, Any] = field(default_factory=dict)
    fact_key: str = ""                 # see module docstring — evidence capping
    reference_versions: dict[str, str] = field(default_factory=dict)
    peer_level_used: str | None = None
    related_signal_ids: list[str] = field(default_factory=list)

    # ---- quality ------------------------------------------------------------
    data_quality_penalty: float = 0.0        # [0,1], subtracted in the priority formula
    known_exception_strength: float = 0.0    # [0,1], subtracted in the priority formula
    suppressed_by_exclusion: str | None = None

    # ---- provenance ---------------------------------------------------------
    is_synthetic: bool = False         # injected corruption fixtures are permanently marked
    notes: str = ""

    # ------------------------------------------------------------------ build

    @classmethod
    def create(
        cls,
        *,
        control,  # AtomicControl; untyped to avoid a circular import
        tenant_id: str,
        subject_type: str,
        subject_id: str,
        evidence: dict[str, Any],
        fact_key: str,
        claim_ids: Sequence[str] = (),
        event_time: _dt.datetime | None = None,
        detection_time: _dt.datetime | None = None,
        period_bucket: str = "",
        score: float | None = None,
        confidence: float = 1.0,
        evidence_strength: float | None = None,
        exposure_aed: float = 0.0,
        exposure_basis: str = "",
        exposure_established: bool = True,
        reference_versions: dict[str, str] | None = None,
        peer_level_used: str | None = None,
        data_quality_penalty: float = 0.0,
        known_exception_strength: float = 0.0,
        is_synthetic: bool = False,
        notes: str = "",
    ) -> "Signal":
        score = control.score if score is None else score
        if evidence_strength is None:
            # Evidence strength is the control's score scaled to [0,1] and
            # attenuated by confidence. A low-confidence finding contributes
            # less to priority than a high-confidence one at the same score —
            # which is what makes the OCR/sparse-peer degradation in §6.3 have
            # an actual consequence rather than being a displayed number.
            evidence_strength = (score / 100.0) * max(0.0, min(1.0, confidence))
        sid = _stable_id(
            tenant_id, control.rule_id, control.version, subject_type, subject_id, fact_key, period_bucket
        )
        return cls(
            signal_id=sid,
            rule_id=control.rule_id,
            rule_version=control.version,
            scenario_id=control.scenario_id,
            tenant_id=tenant_id,
            subject_type=subject_type,
            subject_id=subject_id,
            claim_ids=list(claim_ids),
            event_time=event_time,
            detection_time=detection_time or _dt.datetime.now(_dt.timezone.utc),
            period_bucket=period_bucket,
            score=float(score),
            confidence=float(confidence),
            evidence_strength=float(max(0.0, min(1.0, evidence_strength))),
            disposition=control.effective_disposition(),
            reason_code=control.reason_code,
            domain=control.domain,
            stage=control.stage,
            rule_status=control.status,
            exposure_aed=float(exposure_aed),
            exposure_basis=exposure_basis,
            exposure_established=exposure_established,
            evidence=dict(evidence),
            fact_key=fact_key,
            reference_versions=dict(reference_versions or {}),
            peer_level_used=peer_level_used,
            data_quality_penalty=float(data_quality_penalty),
            known_exception_strength=float(known_exception_strength),
            is_synthetic=is_synthetic,
            notes=notes,
        )

    # ------------------------------------------------------------------ views

    @property
    def qualified_rule(self) -> str:
        return f"{self.rule_id}@{self.rule_version}"

    @property
    def is_shadow(self) -> bool:
        return self.rule_status is RuleStatus.SHADOW

    def exposure_display(self) -> str:
        """What the UI prints for money.

        §4.11: an uncertain model-only lead shows its gross amount together with
        the explicit statement "exposure not yet established" rather than
        presenting that gross amount as if it were savings.
        """
        if not self.exposure_established:
            return f"AED {self.exposure_aed:,.2f} — {EXPOSURE_NOT_ESTABLISHED}"
        return f"AED {self.exposure_aed:,.2f}"

    def to_row(self) -> dict[str, Any]:
        d = asdict(self)
        d["disposition"] = self.disposition.value
        d["domain"] = self.domain.value
        d["stage"] = self.stage.value
        d["rule_status"] = self.rule_status.value
        d["event_time"] = self.event_time.isoformat() if self.event_time else None
        d["detection_time"] = self.detection_time.isoformat() if self.detection_time else None
        d["evidence"] = json.dumps(self.evidence, default=str)
        d["reference_versions"] = json.dumps(self.reference_versions, default=str)
        d["claim_ids"] = ",".join(self.claim_ids)
        d["related_signal_ids"] = ",".join(self.related_signal_ids)
        return d


def cap_evidence(signals: Iterable[Signal]) -> float:
    """Evidence capping — manuscript §3.8, applied at the scoring layer (§4.11).

    Signals resting on the *same underlying fact* contribute
    ``max(evidence_strength)``; distinct facts are combined across facts. The
    combination across distinct facts uses a saturating (noisy-OR) form rather
    than a sum, so that ten weak independent facts cannot manufacture certainty
    the evidence does not support, while genuinely independent strong facts do
    accumulate.

    Returns a value in [0, 1] suitable for the ``evidence_strength`` term of the
    priority formula.
    """
    by_fact: dict[str, float] = {}
    for s in signals:
        key = s.fact_key or s.signal_id
        by_fact[key] = max(by_fact.get(key, 0.0), float(s.evidence_strength))
    if not by_fact:
        return 0.0
    complement = 1.0
    for v in by_fact.values():
        complement *= (1.0 - max(0.0, min(1.0, v)))
    return 1.0 - complement


class SignalStore:
    """In-memory signal store with idempotent upsert.

    Idempotency is a §6.2 test category for *every* control, including
    statistical and model controls: "replaying the same input produces no
    duplicate signal". Because :meth:`Signal.create` derives the id
    deterministically from (tenant, rule, version, subject, fact, period), a
    replay overwrites rather than appends, and the test asserts the store's
    length is unchanged.

    The persistent equivalent lives behind the same interface in
    :mod:`fwa.storage.repository`, so swapping SQLite for PostgreSQL is a
    configuration change (build brief §2).
    """

    def __init__(self) -> None:
        self._signals: dict[str, Signal] = {}

    def add(self, signal: Signal) -> Signal:
        self._signals[signal.signal_id] = signal
        return signal

    def extend(self, signals: Iterable[Signal]) -> None:
        for s in signals:
            self.add(s)

    def get(self, signal_id: str) -> Signal | None:
        return self._signals.get(signal_id)

    def all(self, tenant_id: str | None = None) -> list[Signal]:
        """Tenant-scoped read.

        Every query in this system is filtered by tenant (§13.2 of the build
        brief). Passing ``None`` returns everything and is only used by the
        pipeline itself and by tenant-isolation tests, never by the UI, which
        always passes the session user's tenant.
        """
        vals = list(self._signals.values())
        if tenant_id is None:
            return vals
        return [s for s in vals if s.tenant_id == tenant_id]

    def by_rule(self, rule_id: str, tenant_id: str | None = None) -> list[Signal]:
        return [s for s in self.all(tenant_id) if s.rule_id == rule_id]

    def by_subject(self, subject_type: str, subject_id: str, tenant_id: str | None = None) -> list[Signal]:
        return [
            s for s in self.all(tenant_id)
            if s.subject_type == subject_type and s.subject_id == subject_id
        ]

    def by_claim(self, claim_id: str, tenant_id: str | None = None) -> list[Signal]:
        return [s for s in self.all(tenant_id) if claim_id in s.claim_ids]

    def counts_by_rule(self, tenant_id: str | None = None) -> dict[str, int]:
        out: dict[str, int] = {}
        for s in self.all(tenant_id):
            out[s.rule_id] = out.get(s.rule_id, 0) + 1
        return out

    def to_dataframe(self, tenant_id: str | None = None):
        import pandas as pd

        rows = [s.to_row() for s in self.all(tenant_id)]
        if not rows:
            return pd.DataFrame(columns=[f.name for f in Signal.__dataclass_fields__.values()])
        return pd.DataFrame(rows)

    def __len__(self) -> int:
        return len(self._signals)

    def __iter__(self):
        return iter(self._signals.values())
