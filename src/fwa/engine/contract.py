"""The atomic-control contract (manuscript §3.7) and the safety-boundary validator.

This module contains the single most important piece of code in the repository:
:func:`validate_disposition_legality`. Manuscript §3.3 says that it must be
*structurally impossible* for an anomaly score to autonomously deny a claim.
The way that is made structural, rather than aspirational, is that every control
is parsed into an :class:`AtomicControl` at **registration time**, and a control
of type ``S``, ``N``, ``T`` or ``M`` that declares ``REJECT`` or ``REPRICE``
raises :class:`DispositionLegalityError` before the registry will accept it.

The engine cannot be started with an illegal rule loaded. There is no override
flag, no ``force=True``, and no environment variable that relaxes this.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import re
from typing import Any, Iterable, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ..enums import (
    DENYING_DISPOSITIONS,
    DOMAIN_FOR_TYPE,
    ControlType,
    DataSupport,
    Disposition,
    RuleStatus,
    SignalDomain,
    Stage,
    TYPES_ALLOWED_TO_DENY,
)

__all__ = [
    "AtomicControl",
    "ControlContractError",
    "DispositionLegalityError",
    "validate_disposition_legality",
    "parse_types",
]

_RULE_ID_RE = re.compile(r"^(ENT|PAY|CLN|PHR|DOC|NET|ANL|POL)-\d{2}-R\d{2}$")
_SCENARIO_ID_RE = re.compile(r"^(ENT|PAY|CLN|PHR|DOC|NET|ANL|POL)-\d{2}$")
_SEMVER_RE = re.compile(r"^\d+\.\d+\.\d+$")


class ControlContractError(ValueError):
    """A control does not satisfy the §3.7 contract."""


class DispositionLegalityError(Exception):
    """A control attempts a disposition its type is not permitted to declare.

    This is the §3.3 safety boundary firing. Raising it at registration time is
    the whole point: the system refuses to start rather than run with a control
    that could let a score deny a claim.

    It deliberately does **not** inherit from :class:`ValueError`, and therefore
    not from :class:`ControlContractError` either. Two reasons, both structural:

    1. pydantic converts a ``ValueError`` raised inside a validator into a
       :class:`pydantic.ValidationError`. That would bury the safety boundary
       inside the same exception type as a missing field or a bad date, and any
       caller doing ``except ValidationError: skip this control`` would be
       silently discarding the one violation that must never be skipped. A
       non-``ValueError`` propagates out of pydantic untouched.
    2. A legality violation is not "this control is malformed". It is "this
       catalogue is unsafe to run". Keeping it off the contract-error hierarchy
       means no existing ``except ControlContractError`` handler can downgrade
       it into a warning without someone deliberately naming this class.
    """


def parse_types(raw: str | Iterable[str]) -> list[ControlType]:
    """Parse the catalogue's ``H``, ``H/E``, ``S/N``… type notation.

    Appendix C frequently assigns a control a *composite* type (``E/T``,
    ``H/S``) because a single control may combine a deterministic check with a
    statistical or document-derived one. The legality rule treats a composite
    strictly: the control is only permitted to deny if **every** component type
    is permitted to deny. A control that is part statistical is, for the purpose
    of §3.3, statistical.
    """
    if isinstance(raw, str):
        parts = [p.strip().upper() for p in raw.replace("|", "/").split("/") if p.strip()]
    else:
        parts = [str(p).strip().upper() for p in raw]
    if not parts:
        raise ControlContractError("Control declares no type; §3.3 legality cannot be evaluated.")
    out: list[ControlType] = []
    for p in parts:
        try:
            out.append(ControlType(p))
        except ValueError as exc:  # pragma: no cover - defensive
            raise ControlContractError(
                f"Unknown control type {p!r}. Manuscript Table 3.3 defines exactly H, E, S, N, T, M."
            ) from exc
    return out


def validate_disposition_legality(
    rule_id: str,
    types: Iterable[ControlType],
    disposition: Disposition,
) -> None:
    """Enforce manuscript §3.3.

    Raises
    ------
    DispositionLegalityError
        If ``disposition`` is ``REJECT`` or ``REPRICE`` and any of ``types`` is
        ``S``, ``N``, ``T`` or ``M``.

    Notes
    -----
    This function is deliberately free of configuration. There is no threshold
    at which a statistical control becomes allowed to deny, because the
    manuscript's claim is not that scores are *usually* too weak to deny on —
    it is that a score is the wrong *kind* of evidence for a denial. A score can
    establish that a claim is unusual. Only an objective, effective-dated
    condition can establish that it is not payable.
    """
    types = list(types)
    if disposition not in DENYING_DISPOSITIONS:
        return
    offending = [t for t in types if t not in TYPES_ALLOWED_TO_DENY]
    if offending:
        raise DispositionLegalityError(
            f"§3.3 SAFETY BOUNDARY VIOLATION in {rule_id}: a control of type "
            f"{'/'.join(t.value for t in types)} declares disposition "
            f"{disposition.value}, but type(s) {', '.join(t.value for t in offending)} "
            f"may never deny or reprice a claim. Only an objective, effective-dated "
            f"condition evaluated by a hard (H) or expert (E) control may do that. "
            f"A signal is not a fraud finding."
        )


class AtomicControl(BaseModel):
    """One atomic control, exactly as manuscript §3.7 defines the contract.

    Every field below is a *requirement*, not documentation. In particular the
    model refuses to construct a control that omits ``exclusions`` or ``owner``,
    because §3.7 says those two fields are what make the governance model real:
    ``exclusions`` forces a rule author to declare the legitimate exceptions up
    front rather than leaving a reviewer to discover them, and ``owner`` is what
    makes parameter governance auditable rather than aspirational.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", use_enum_values=False)

    # ---- §3.7 contract fields ----------------------------------------------
    rule_id: str
    scenario_id: str
    name: str
    version: str = "1.0.0"
    status: RuleStatus = RuleStatus.SHADOW
    type: list[ControlType]
    stage: Stage
    population: str
    inputs: list[str]
    expression: str
    parameters: dict[str, Any] = Field(default_factory=dict)
    exclusions: list[str]
    grouping_key: list[str]
    score: float
    disposition: Disposition
    reason_code: str
    evidence_fields: list[str]
    owner: str
    effective_from: _dt.date
    effective_to: _dt.date | None = None

    # ---- artefact extensions (documented as extensions, not §3.7) ----------
    priority: Literal["P0", "P1", "P2"] = "P1"
    data_support: DataSupport = DataSupport.NOT_EXECUTABLE_ON_THIS_DATASET
    data_support_reason: str = ""
    required_canonical_fields: list[str] = Field(default_factory=list)
    implementation: str | None = None
    catalogue_trigger: str = ""
    catalogue_config_exclusions: str = ""
    catalogue_disposition: str = ""
    alternate_dispositions: list[Disposition] = Field(default_factory=list)
    governance_note: str = ""
    authored_by: str | None = None
    approved_by: str | None = None
    approved_at: _dt.datetime | None = None
    shadow_since: _dt.date | None = None
    kill_switched: bool = False
    kill_switch_reason: str = ""

    # ---- validators ---------------------------------------------------------

    @field_validator("rule_id")
    @classmethod
    def _rule_id_shape(cls, v: str) -> str:
        if not _RULE_ID_RE.match(v):
            raise ControlContractError(
                f"rule_id {v!r} does not match the catalogue convention FAM-NN-RNN "
                "(e.g. PAY-04-R01). Appendix C identifiers are load-bearing: the "
                "traceability matrix is generated from them."
            )
        return v

    @field_validator("scenario_id")
    @classmethod
    def _scenario_id_shape(cls, v: str) -> str:
        if not _SCENARIO_ID_RE.match(v):
            raise ControlContractError(f"scenario_id {v!r} does not match FAM-NN (e.g. PAY-04).")
        return v

    @field_validator("version")
    @classmethod
    def _semver(cls, v: str) -> str:
        if not _SEMVER_RE.match(v):
            raise ControlContractError(
                f"version {v!r} must be semantic (MAJOR.MINOR.PATCH). Rule versions are "
                "stamped onto every signal so a finding can be reproduced against the "
                "exact rule text that produced it (§3.10 reproducibility)."
            )
        return v

    @field_validator("type", mode="before")
    @classmethod
    def _coerce_types(cls, v: Any) -> list[ControlType]:
        return parse_types(v)

    @field_validator("exclusions")
    @classmethod
    def _exclusions_declared(cls, v: list[str]) -> list[str]:
        if not v:
            raise ControlContractError(
                "A control with no declared exclusions is rejected. §3.7: 'exclusions "
                "requires every hard edit to declare its known legitimate exceptions "
                "rather than relying on a reviewer to catch them after the fact'. If a "
                "control genuinely has none, declare the single exclusion "
                "'none_declared_by_policy_owner' so the absence is an owned decision."
            )
        return v

    @field_validator("owner")
    @classmethod
    def _owner_present(cls, v: str) -> str:
        if not v or not v.strip():
            raise ControlContractError("A control with no owner is rejected (§3.7, §3.9).")
        return v.strip()

    @field_validator("grouping_key")
    @classmethod
    def _grouping_key_present(cls, v: list[str]) -> list[str]:
        if not v:
            raise ControlContractError(
                "grouping_key is required: it is what allows case correlation (§3.8) to "
                "work without a bespoke correlation rule per scenario."
            )
        return v

    @field_validator("evidence_fields")
    @classmethod
    def _evidence_present(cls, v: list[str]) -> list[str]:
        if not v:
            raise ControlContractError(
                "evidence_fields is required. A signal that cannot render human-readable "
                "evidence fails the evidence-snapshot test (§6.2) and the explainability "
                "NFR (§3.10)."
            )
        return v

    @field_validator("score")
    @classmethod
    def _score_range(cls, v: float) -> float:
        if not 0 <= v <= 100:
            raise ControlContractError("score must lie in [0, 100].")
        return v

    @model_validator(mode="after")
    def _safety_boundary(self) -> "AtomicControl":
        # ---- THE boundary check (§3.3). Runs on every construction. ----
        validate_disposition_legality(self.rule_id, self.type, self.disposition)
        for alt in self.alternate_dispositions:
            validate_disposition_legality(self.rule_id, self.type, alt)

        if not self.rule_id.startswith(self.scenario_id + "-R"):
            raise ControlContractError(
                f"rule_id {self.rule_id} does not belong to scenario {self.scenario_id}."
            )
        # ---- honesty invariants on the data_support classification ----------
        # EXECUTABLE  : runs as the catalogue specifies it.
        # PARTIAL     : runs, but against a PROXY input, so its evidentiary
        #               force is reduced and the signal says so.
        # NOT_EXECUTABLE_ON_THIS_DATASET : does not run at all.
        #
        # The two failure modes this guards against are a control that claims to
        # run and does not, and a control that quietly runs while classified as
        # unrunnable. Both would make control_coverage_matrix.csv — a
        # deliverable in its own right (build brief §11.2) — untrue.
        if self.data_support is DataSupport.NOT_EXECUTABLE_ON_THIS_DATASET and self.implementation:
            raise ControlContractError(
                f"{self.rule_id} declares data_support=NOT_EXECUTABLE_ON_THIS_DATASET but names an "
                f"implementation ({self.implementation}). A control that cannot run on this "
                "dataset must not pretend to. Classify it honestly or implement it."
            )
        if self.data_support is not DataSupport.NOT_EXECUTABLE_ON_THIS_DATASET and not self.implementation:
            raise ControlContractError(
                f"{self.rule_id} is classified {self.data_support.value} but names no "
                "implementation. The coverage matrix would then overstate what this build "
                "actually runs; classify it NOT_EXECUTABLE_ON_THIS_DATASET instead."
            )
        if not self.data_support_reason.strip():
            raise ControlContractError(
                f"{self.rule_id} must state a one-line reason for its data_support "
                "classification. The coverage matrix is a deliverable in its own right "
                "(build brief §11.2) and an unexplained classification is useless in it."
            )
        return self

    # ---- derived ------------------------------------------------------------

    @property
    def primary_type(self) -> ControlType:
        return self.type[0]

    @property
    def type_label(self) -> str:
        return "/".join(t.value for t in self.type)

    @property
    def domain(self) -> SignalDomain:
        """Which of the five evidence domains this control contributes to.

        Used by the §4.11 priority score's ``independent_domain_count`` term.
        A composite control is attributed to the domain of its *most* advanced
        component, because that is the domain whose evidence it actually adds.
        """
        order = [ControlType.M, ControlType.N, ControlType.T, ControlType.S, ControlType.E, ControlType.H]
        for t in order:
            if t in self.type:
                return DOMAIN_FOR_TYPE[t]
        return SignalDomain.RULE

    @property
    def can_deny(self) -> bool:
        return self.disposition in DENYING_DISPOSITIONS

    @property
    def qualified_id(self) -> str:
        """``rule_id@version`` — what the UI shows beside every finding."""
        return f"{self.rule_id}@{self.version}"

    def is_effective(self, as_of: _dt.date) -> bool:
        if as_of < self.effective_from:
            return False
        if self.effective_to and as_of > self.effective_to:
            return False
        return True

    def is_runnable(self, as_of: _dt.date) -> bool:
        """Runnable means: effective, not retired, not kill-switched.

        A ``shadow`` control IS runnable — that is the point of shadow mode. It
        produces signals that are recorded and measured but that carry a
        shadow marker and never reach the live queue as an actionable item.
        """
        if self.kill_switched:
            return False
        if self.status is RuleStatus.RETIRED:
            return False
        return self.is_effective(as_of)

    def effective_disposition(self) -> Disposition:
        """The disposition after kill-switch routing (§9.7).

        A kill-switched control does not vanish — its signals are routed to
        ``MONITOR_ONLY`` so the loss of coverage is visible rather than silent.
        """
        return Disposition.MONITOR_ONLY if self.kill_switched else self.disposition

    def fingerprint(self) -> str:
        """Stable hash of the rule text, for the evidence snapshot."""
        payload = json.dumps(
            {
                "rule_id": self.rule_id,
                "version": self.version,
                "type": self.type_label,
                "stage": self.stage.value,
                "population": self.population,
                "expression": self.expression,
                "parameters": self.parameters,
                "exclusions": self.exclusions,
                "disposition": self.disposition.value,
                "score": self.score,
            },
            sort_keys=True,
            default=str,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]

    def to_yaml_dict(self) -> dict[str, Any]:
        d = self.model_dump(mode="json")
        d["type"] = self.type_label
        return d
