"""The governed vocabularies of the system.

Everything in this module is quoted from the manuscript. None of it is a design
choice made here, and none of it may be extended without a corresponding change
to the manuscript, because each enumeration is load-bearing for a governance
guarantee:

* :class:`Disposition` — manuscript Table 3.2. Eight values, no more. The
  boundary in §3.3 is expressed as *which of these eight a control may declare*.
* :class:`ControlType` — manuscript Table 3.3. Six values. The type is what
  decides whether a control is allowed to deny a claim.
* :class:`Stage` — manuscript Table 3.4. Six values, each with a latency ceiling.
* :class:`RuleStatus` — manuscript §3.7. ``shadow`` before ``active``, always.
"""

from __future__ import annotations

from enum import Enum

__all__ = [
    "Disposition",
    "ControlType",
    "Stage",
    "RuleStatus",
    "DataSupport",
    "SignalDomain",
    "DENYING_DISPOSITIONS",
    "TYPES_ALLOWED_TO_DENY",
    "DISPOSITION_MEANINGS",
    "STAGE_LATENCY_DESCRIPTION",
]


class Disposition(str, Enum):
    """The eight governed dispositions of manuscript Table 3.2.

    A control declares exactly one. There is no ninth value and no "other".
    """

    RETURN = "RETURN"
    REJECT = "REJECT"
    REPRICE = "REPRICE"
    PREPAY_PEND = "PREPAY_PEND"
    POSTPAY_AUDIT = "POSTPAY_AUDIT"
    SIU_LEAD = "SIU_LEAD"
    PROVIDER_EDUCATION = "PROVIDER_EDUCATION"
    MONITOR_ONLY = "MONITOR_ONLY"


DISPOSITION_MEANINGS: dict[Disposition, str] = {
    Disposition.RETURN: "The transaction cannot be processed because required data or format is invalid.",
    Disposition.REJECT: "An objective, effective-dated coverage or payment condition fails and policy authorises rejection.",
    Disposition.REPRICE: "The correct payable amount is computable under the applicable tariff or contract.",
    Disposition.PREPAY_PEND: "Human coding, clinical or payment review is required before payment.",
    Disposition.POSTPAY_AUDIT: "The pattern requires records, sampling or recovery review after payment.",
    Disposition.SIU_LEAD: "Coordinated or intentional conduct is plausible and warrants Special Investigation Unit assessment.",
    Disposition.PROVIDER_EDUCATION: "The pattern is more consistent with error or poor billing practice than misconduct.",
    Disposition.MONITOR_ONLY: "The evidence is insufficient for any operational intervention at this time.",
}

#: Dispositions that change what a payer pays. Only these two are restricted.
DENYING_DISPOSITIONS: frozenset[Disposition] = frozenset(
    {Disposition.REJECT, Disposition.REPRICE}
)

#: Plain-language description of what each disposition asks of a reviewer.
REVIEWER_ASK: dict[Disposition, str] = {
    Disposition.RETURN: "Send back to the submitter — the data is not processable as submitted.",
    Disposition.REJECT: "Confirm the objective condition failed, then deny.",
    Disposition.REPRICE: "Confirm the correct payable amount and adjust.",
    Disposition.PREPAY_PEND: "Review before payment and decide: pay, adjust or deny.",
    Disposition.POSTPAY_AUDIT: "Request records after payment and decide whether to recover.",
    Disposition.SIU_LEAD: "Assess whether this warrants a formal investigation.",
    Disposition.PROVIDER_EDUCATION: "Decide whether to open a billing-practice conversation with the provider.",
    Disposition.MONITOR_ONLY: "No action. Watch the pattern; it is not yet actionable.",
}


class ControlType(str, Enum):
    """Manuscript Table 3.3 — what a control *checks*."""

    H = "H"  # Hard: exact validation against an authoritative, effective-dated rule
    E = "E"  # Expert: deterministic execution of maintained clinical/coding policy
    S = "S"  # Statistical: peer-relative or self-history deviation
    N = "N"  # Network: relationship concentration, reciprocity, community
    T = "T"  # Text/document: structured facts extracted from documents
    M = "M"  # Model: a trained unsupervised (or, later, supervised) score

    @property
    def label(self) -> str:
        return {
            "H": "Hard edit",
            "E": "Expert edit",
            "S": "Statistical",
            "N": "Network",
            "T": "Document/NLP",
            "M": "Model",
        }[self.value]


#: §3.3, enforced by ``validate_disposition_legality``.
#:
#: "Only REJECT may deny a claim outright, and only where an objective,
#:  effective-dated condition, **not a statistical or model score**, fails."
#:
#: Type E is included because manuscript Appendix C assigns REPRICE to several
#: ``H/E`` composite controls (PAY-03-R03, CLN-05-R03, PHR-01-R02, PHR-05-R01),
#: which execute a *maintained, deterministic* policy table — the same class of
#: objective condition as a hard edit, not a score. Types S, N, T and M can
#: never deny, under any circumstance, at any threshold.
TYPES_ALLOWED_TO_DENY: frozenset[ControlType] = frozenset({ControlType.H, ControlType.E})


class Stage(str, Enum):
    """Manuscript Table 3.4 — *where in the lifecycle* a control executes."""

    INGEST = "INGEST"
    PREPAY_SYNC = "PREPAY_SYNC"
    PREPAY_ASYNC = "PREPAY_ASYNC"
    POSTPAY_DAILY = "POSTPAY_DAILY"
    NETWORK_WEEKLY = "NETWORK_WEEKLY"
    MODEL_MONTHLY = "MODEL_MONTHLY"


STAGE_LATENCY_DESCRIPTION: dict[Stage, str] = {
    Stage.INGEST: "Under 1 second per file-validation batch",
    Stage.PREPAY_SYNC: "Under 500 ms per claim once features are available",
    Stage.PREPAY_ASYNC: "Minutes",
    Stage.POSTPAY_DAILY: "Daily",
    Stage.NETWORK_WEEKLY: "Weekly",
    Stage.MODEL_MONTHLY: "Monthly, or when drift triggers",
}


class RuleStatus(str, Enum):
    """Manuscript §3.7. Every rule starts in ``shadow``.

    The shadow→active transition is a governed, separation-of-duties-controlled
    action (§13.2 of the build brief): the author of a rule may not approve it.
    """

    SHADOW = "shadow"
    ACTIVE = "active"
    RETIRED = "retired"


class DataSupport(str, Enum):
    """How far the claim extract can actually carry a control.

    This is an honesty artefact in its own right (build brief §4, §11.2), not a
    workaround. Every one of the 164 controls carries one of these values and a
    one-line reason, and ``reports/control_coverage_matrix.csv`` is the
    concrete answer to "what would real Shafafiya/eClaimLink data unlock?".
    """

    EXECUTABLE = "EXECUTABLE"
    PARTIAL = "PARTIAL"
    NOT_EXECUTABLE_ON_THIS_DATASET = "NOT_EXECUTABLE_ON_THIS_DATASET"


class SignalDomain(str, Enum):
    """The independent evidence domains counted by the §4.11 priority formula.

    ``independent_domain_count`` is a term in the priority score precisely
    because two findings from *different* domains are stronger than two findings
    from the same one.
    """

    RULE = "rule"
    STATISTICAL = "statistical"
    NETWORK = "network"
    DOCUMENT = "document"
    MODEL = "model"


DOMAIN_FOR_TYPE: dict[ControlType, SignalDomain] = {
    ControlType.H: SignalDomain.RULE,
    ControlType.E: SignalDomain.RULE,
    ControlType.S: SignalDomain.STATISTICAL,
    ControlType.N: SignalDomain.NETWORK,
    ControlType.T: SignalDomain.DOCUMENT,
    ControlType.M: SignalDomain.MODEL,
}
