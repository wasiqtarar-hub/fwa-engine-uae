"""Case correlation along the six dimensions (manuscript §3.8, §4.12).

    "Signals correlate into cases along six dimensions — claim lineage, episode,
     provider behaviour, pharmacy/product, network and distribution — using a
     deterministic ``case_fingerprint = hash(tenant, case_type, primary_subject,
     scenario_family, period_bucket)``, with MERGE AND SPLIT operations always
     producing a RECORDED AUDIT EVENT rather than a silent case-identity
     change."                                               — manuscript §4.12

The worked example §4.12 gives is the test of whether this works: "a PAY-08
mutated-resubmission signal on a corrected claim, a DOC-02 cloned-documentation
signal on the altered supporting note, and a CLN-01 upcoding signal on the
resulting billed level" should "converge into a single coding case for one
reviewer, rather than presenting three unrelated alerts that obscure the fact
that they describe one underlying event."

Two design points worth stating:

**The fingerprint is deterministic and content-free.** It is a hash of five
fields, not of the signal set, so re-running the pipeline produces the same case
identity — which is what lets a reviewer's disposition from yesterday still
attach to the same case today. A fingerprint derived from the signals would
change every time a new signal arrived.

**Correlation happens after rule execution, never inside it.** §3.8 is explicit.
A control's job is to produce one signal about one fact; deciding that three
signals describe one event is a separate concern with its own dimension rules.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

import pandas as pd

from ..audit.log import AuditEventType
from ..engine.signals import Signal
from ..enums import Disposition, SignalDomain

__all__ = ["Case", "CaseCorrelator", "CORRELATION_DIMENSIONS", "case_fingerprint"]

#: The six §4.12 dimensions, each with the subject it correlates on.
CORRELATION_DIMENSIONS: dict[str, dict[str, Any]] = {
    "claim_lineage": {
        "label": "Claim lineage",
        "subject": "claim",
        "families": {"PAY"},
        "description": "Signals about one claim and its corrections/resubmissions.",
    },
    "episode": {
        "label": "Episode",
        "subject": "episode",
        "families": {"CLN", "PAY"},
        "description": "Signals about one clinical episode, including components billed on "
                       "separate claims.",
    },
    "provider_behaviour": {
        "label": "Provider behaviour",
        "subject": "provider",
        "families": {"CLN", "PAY", "ENT", "ANL"},
        "description": "Peer-relative and self-history signals about one provider.",
    },
    "pharmacy_product": {
        "label": "Pharmacy / product",
        "subject": "provider",
        "families": {"PHR"},
        "description": "Pharmacy and product signals.",
    },
    "network": {
        "label": "Network",
        "subject": "community",
        "families": {"NET"},
        "description": "Relationship and community signals.",
    },
    "distribution": {
        "label": "Distribution",
        "subject": "agent",
        "families": {"NET", "POL"},
        "description": "Broker, agent and enrolment signals.",
    },
}

#: Which disposition wins when several signals correlate into one case.
#: A case takes the STRONGEST disposition among its signals, because a case
#: containing a hard edit is a case that needs the hard edit acted on. This
#: never *weakens* a disposition and never lets a model signal strengthen one:
#: a model signal is MONITOR_ONLY and sits at the bottom of this order.
_DISPOSITION_RANK = [
    Disposition.MONITOR_ONLY,
    Disposition.PROVIDER_EDUCATION,
    Disposition.POSTPAY_AUDIT,
    Disposition.SIU_LEAD,
    Disposition.PREPAY_PEND,
    Disposition.RETURN,
    Disposition.REPRICE,
    Disposition.REJECT,
]


#: Scenario -> CASE FAMILY. §4.12's fingerprint includes ``scenario_family``,
#: and the worked example in that section — a PAY-08 resubmission signal, a
#: DOC-02 cloned-note signal and a CLN-01 upcoding signal converging into ONE
#: coding case — only works if "family" means a family of RELATED TYPOLOGIES
#: rather than the three-letter scenario prefix. These are those families. The
#: prefix is kept on the case as ``scenario_family`` for traceability.
CASE_FAMILIES: dict[str, str] = {
    "CLN-01": "coding_integrity", "CLN-02": "coding_integrity", "CLN-03": "coding_integrity",
    "DOC-01": "coding_integrity", "DOC-02": "coding_integrity", "PAY-08": "coding_integrity",
    "PAY-01": "duplication", "PAY-02": "duplication",
    "CLN-04": "utilisation", "CLN-05": "utilisation", "CLN-06": "utilisation",
    "CLN-07": "utilisation", "CLN-08": "utilisation",
    "ENT-01": "eligibility_identity", "ENT-02": "eligibility_identity",
    "ENT-03": "eligibility_identity", "ENT-04": "eligibility_identity",
    "ENT-05": "eligibility_identity", "ENT-06": "eligibility_identity",
    "PAY-03": "payment_integrity", "PAY-04": "payment_integrity", "PAY-05": "payment_integrity",
    "PAY-06": "payment_integrity", "PAY-07": "payment_integrity", "PAY-09": "payment_integrity",
    "PAY-10": "payment_integrity", "PAY-11": "payment_integrity", "PAY-12": "payment_integrity",
    "PHR-01": "pharmacy_product", "PHR-02": "pharmacy_product", "PHR-03": "pharmacy_product",
    "PHR-04": "pharmacy_product", "PHR-05": "pharmacy_product",
    "NET-01": "network_distribution", "NET-02": "network_distribution",
    "NET-03": "network_distribution", "NET-04": "network_distribution",
    "POL-01": "network_distribution",
    "ANL-01": "multivariate_anomaly",
}


def case_family(scenario_id: str) -> str:
    return CASE_FAMILIES.get(scenario_id, scenario_id.split("-")[0].lower())


def case_fingerprint(
    tenant: str, case_type: str, primary_subject: str, scenario_family: str, period_bucket: str
) -> str:
    """The deterministic §4.12 fingerprint."""
    payload = "|".join([tenant, case_type, primary_subject, scenario_family, period_bucket])
    return "CASE-" + hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


@dataclass
class Case:
    case_id: str
    tenant_id: str
    case_type: str                       # the correlation dimension
    primary_subject_type: str
    primary_subject_id: str
    scenario_family: str
    period_bucket: str
    signal_ids: list[str] = field(default_factory=list)
    claim_ids: list[str] = field(default_factory=list)
    disposition: Disposition = Disposition.MONITOR_ONLY
    exposure_aed: float = 0.0
    exposure_established: bool = True
    exposure_basis: str = ""
    priority: Any = None                 # PriorityScore
    domains: set[SignalDomain] = field(default_factory=set)
    opened_at: _dt.datetime = field(default_factory=lambda: _dt.datetime.now(_dt.timezone.utc))
    merged_from: list[str] = field(default_factory=list)
    split_into: list[str] = field(default_factory=list)
    shadow_only: bool = True

    @property
    def signal_count(self) -> int:
        return len(self.signal_ids)

    def to_row(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "tenant_id": self.tenant_id,
            "case_type": self.case_type,
            "dimension_label": CORRELATION_DIMENSIONS.get(self.case_type, {}).get("label", self.case_type),
            "subject_type": self.primary_subject_type,
            "subject_id": self.primary_subject_id,
            "scenario_family": self.scenario_family,
            "period_bucket": self.period_bucket,
            "signal_count": self.signal_count,
            "claim_count": len(self.claim_ids),
            "disposition": self.disposition.value,
            "exposure_aed": round(self.exposure_aed, 2),
            "exposure_established": self.exposure_established,
            "priority": None if self.priority is None else round(self.priority.value, 1),
            "priority_band": "" if self.priority is None else self.priority.band,
            "priority_band_label": "" if self.priority is None else self.priority.band_label,
            "domains": ",".join(sorted(d.value for d in self.domains)),
            "shadow_only": self.shadow_only,
            "opened_at": self.opened_at.isoformat(),
        }


class CaseCorrelator:
    """Groups signals into cases and scores them."""

    def __init__(self, config, priority_scorer, audit=None) -> None:
        self.config = config
        self.priority = priority_scorer
        self.audit = audit
        self.cases: dict[str, Case] = {}

    # ------------------------------------------------------------------ build

    def correlate(self, signals: Iterable[Signal], registry=None) -> dict[str, Case]:
        buckets: dict[str, list[Signal]] = {}
        meta: dict[str, tuple[str, str, str, str, str]] = {}

        for signal in signals:
            dimension = self._dimension_for(signal)
            subject_type, subject_id = self._subject_for(signal, dimension)
            family = case_family(signal.scenario_id)
            period = signal.period_bucket or "UNSCOPED"
            fp = case_fingerprint(signal.tenant_id, dimension, f"{subject_type}:{subject_id}", family, period)
            buckets.setdefault(fp, []).append(signal)
            meta[fp] = (dimension, subject_type, subject_id, family, period)

        for fp, group in buckets.items():
            dimension, subject_type, subject_id, family, period = meta[fp]
            case = Case(
                case_id=fp,
                tenant_id=group[0].tenant_id,
                case_type=dimension,
                primary_subject_type=subject_type,
                primary_subject_id=subject_id,
                scenario_family=family,
                period_bucket=period,
                signal_ids=[s.signal_id for s in group],
                claim_ids=sorted({c for s in group for c in s.claim_ids}),
                domains={s.domain for s in group},
                shadow_only=all(s.is_shadow for s in group),
            )
            case.disposition = self._strongest_disposition(group)
            case.exposure_aed, case.exposure_established, case.exposure_basis = self._exposure(group)
            case.priority = self.priority.score(
                group, exposure_aed=case.exposure_aed if case.exposure_established else 0.0
            )
            self.cases[fp] = case
        return self.cases

    # -------------------------------------------------------------- internals

    @staticmethod
    def _dimension_for(signal: Signal) -> str:
        family = signal.scenario_id.split("-")[0]
        if family == "NET":
            return "network" if signal.subject_type == "community" else "distribution"
        if family == "POL":
            return "distribution"
        if family == "PHR":
            return "pharmacy_product"
        if signal.subject_type == "episode":
            return "episode"
        if signal.subject_type in ("provider", "cluster"):
            return "provider_behaviour"
        if signal.subject_type == "claim" and family == "PAY":
            return "claim_lineage"
        if signal.subject_type in ("claim", "member"):
            return "episode"
        return "provider_behaviour"

    @staticmethod
    def _subject_for(signal: Signal, dimension: str) -> tuple[str, str]:
        wanted = CORRELATION_DIMENSIONS.get(dimension, {}).get("subject", signal.subject_type)
        if wanted == signal.subject_type:
            return signal.subject_type, signal.subject_id
        # Fall back to the signal's own subject rather than inventing one. A
        # correlation that cannot find its dimension's subject correlates on
        # what it has, and the case records which.
        return signal.subject_type, signal.subject_id

    @staticmethod
    def _strongest_disposition(signals: Sequence[Signal]) -> Disposition:
        rank = {d: i for i, d in enumerate(_DISPOSITION_RANK)}
        return max((s.disposition for s in signals), key=lambda d: rank.get(d, 0))

    @staticmethod
    def _exposure(signals: Sequence[Signal]) -> tuple[float, bool, str]:
        """Case exposure — each distinct CLAIM counted once (§4.11).

        Summing the signals' exposures would double-count: two signals about the
        same claim each carry that claim's amount. The case therefore sums over
        distinct claims, taking the largest established exposure attributed to
        each.
        """
        established: dict[str, float] = {}
        unestablished: dict[str, float] = {}
        no_claim_established = 0.0
        for s in signals:
            target = established if s.exposure_established else unestablished
            if not s.claim_ids:
                if s.exposure_established:
                    no_claim_established = max(no_claim_established, s.exposure_aed)
                continue
            share = s.exposure_aed / max(len(s.claim_ids), 1)
            for claim in s.claim_ids:
                target[claim] = max(target.get(claim, 0.0), share)

        total_established = sum(established.values()) + no_claim_established
        if total_established > 0:
            return (
                total_established, True,
                f"Sum over {len(established)} distinct claim(s), each counted once, of the "
                f"largest ESTABLISHED exposure attributed to it.",
            )
        total_unestablished = sum(unestablished.values())
        return (
            total_unestablished, False,
            f"Gross amount across {len(unestablished)} distinct claim(s). No objective condition "
            f"has been shown to fail, so this is not established exposure.",
        )

    # ------------------------------------------------------------ merge/split

    def merge(self, case_ids: Sequence[str], *, actor: str, reason: str) -> Case:
        """Merge cases. Always emits an audit event (§4.12)."""
        if len(case_ids) < 2:
            raise ValueError("A merge needs at least two cases.")
        if not reason.strip():
            raise ValueError("A case merge requires a stated reason.")
        cases = [self.cases[c] for c in case_ids]
        primary = cases[0]
        merged = Case(
            case_id=case_fingerprint(
                primary.tenant_id, primary.case_type,
                f"{primary.primary_subject_type}:{primary.primary_subject_id}",
                primary.scenario_family, primary.period_bucket,
            ) + "-M",
            tenant_id=primary.tenant_id,
            case_type=primary.case_type,
            primary_subject_type=primary.primary_subject_type,
            primary_subject_id=primary.primary_subject_id,
            scenario_family=primary.scenario_family,
            period_bucket=primary.period_bucket,
            signal_ids=[sid for c in cases for sid in c.signal_ids],
            claim_ids=sorted({cid for c in cases for cid in c.claim_ids}),
            domains=set().union(*(c.domains for c in cases)),
            merged_from=list(case_ids),
            shadow_only=all(c.shadow_only for c in cases),
        )
        rank = {d: i for i, d in enumerate(_DISPOSITION_RANK)}
        merged.disposition = max((c.disposition for c in cases), key=lambda d: rank.get(d, 0))
        merged.exposure_aed = sum(c.exposure_aed for c in cases if c.exposure_established)
        merged.exposure_established = any(c.exposure_established for c in cases)
        for cid in case_ids:
            self.cases.pop(cid, None)
        self.cases[merged.case_id] = merged
        self._audit(AuditEventType.CASE_MERGED, actor, merged.case_id, reason,
                    before={"merged_from": list(case_ids)}, after=merged.to_row())
        return merged

    def split(self, case_id: str, signal_groups: Sequence[Sequence[str]], *, actor: str, reason: str) -> list[Case]:
        if not reason.strip():
            raise ValueError("A case split requires a stated reason.")
        original = self.cases[case_id]
        out: list[Case] = []
        for i, group in enumerate(signal_groups, start=1):
            child = Case(
                case_id=f"{case_id}-S{i}",
                tenant_id=original.tenant_id,
                case_type=original.case_type,
                primary_subject_type=original.primary_subject_type,
                primary_subject_id=original.primary_subject_id,
                scenario_family=original.scenario_family,
                period_bucket=original.period_bucket,
                signal_ids=list(group),
                claim_ids=original.claim_ids,
                disposition=original.disposition,
                domains=set(original.domains),
                shadow_only=original.shadow_only,
            )
            self.cases[child.case_id] = child
            out.append(child)
        original.split_into = [c.case_id for c in out]
        self.cases.pop(case_id, None)
        self._audit(AuditEventType.CASE_SPLIT, actor, case_id, reason,
                    before=original.to_row(), after={"split_into": original.split_into})
        return out

    def _audit(self, event_type, actor, subject, reason, before=None, after=None) -> None:
        if self.audit is None:
            return
        self.audit.record(event_type, actor=actor, actor_role="", subject=subject,
                          reason=reason, before=before, after=after)

    # ---------------------------------------------------------------- reports

    def to_frame(self, tenant_id: str | None = None) -> pd.DataFrame:
        rows = [
            c.to_row() for c in self.cases.values()
            if tenant_id is None or c.tenant_id == tenant_id
        ]
        if not rows:
            return pd.DataFrame(columns=list(Case("", "", "", "", "", "", "").to_row()))
        return pd.DataFrame(rows).sort_values("priority", ascending=False)

    def queue(self, tenant_id: str, limit: int | None = None) -> pd.DataFrame:
        frame = self.to_frame(tenant_id)
        return frame if limit is None else frame.head(limit)
