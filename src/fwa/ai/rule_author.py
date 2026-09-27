"""Natural-language rule authoring (manuscript §9.6, build brief §9.2).

    "The policy owner types *'pend any claim where the pharmacy share exceeds
     twice the diagnosis-peer median and the stay is under two days.'* The LLM
     drafts a **YAML atomic-control contract**; the deterministic validator then
     checks schema, type/disposition legality (§1), declared exclusions, owner
     and effective date; the draft enters the registry in **``shadow`` status
     only**, with its ten-category test-fixture stubs generated alongside. A
     human must approve activation. **Never auto-activate.** Show the diff."

The division of labour here is the point, and it is worth being explicit about
which half is trusted:

* The **LLM drafts**. It turns prose into a candidate contract. It is allowed to
  be wrong.
* The **deterministic validator decides**. :class:`~fwa.engine.contract.AtomicControl`
  construction runs the full §3.7 contract check including the §3.3
  disposition-legality rule, so a drafted control that would let a statistical
  test deny a claim is rejected before it can be registered — by the same code
  path that validates a hand-written rule, not by a separate LLM-specific check.
* A **human approves**. The draft enters in ``shadow``, and
  :meth:`~fwa.engine.registry.RuleRegistry.activate` refuses an approver who is
  also the author.

The offline provider drafts by parsing the request with deterministic patterns,
so the whole flow is demonstrable with no API key — and, more usefully, so the
governance machinery can be tested against a draft whose contents are known.
"""

from __future__ import annotations

import datetime as _dt
import difflib
import re
from dataclasses import dataclass, field
from typing import Any

import yaml

from ..audit.log import AuditEventType
from ..engine.contract import AtomicControl, ControlContractError, DispositionLegalityError
from ..enums import ControlType, DataSupport, Disposition, RuleStatus, Stage
from .provider import LLMProvider

__all__ = ["RuleDraft", "NaturalLanguageRuleAuthor"]


@dataclass
class RuleDraft:
    request: str
    yaml_text: str
    payload: dict[str, Any]
    control: AtomicControl | None
    valid: bool
    errors: list[str] = field(default_factory=list)
    test_fixture_stubs: dict[str, str] = field(default_factory=dict)
    diff: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "request": self.request,
            "yaml": self.yaml_text,
            "valid": self.valid,
            "errors": self.errors,
            "rule_id": self.payload.get("rule_id"),
            "status": "shadow",
            "test_fixture_stubs": sorted(self.test_fixture_stubs),
            "diff": self.diff,
        }


#: The ten mandatory test categories (§6.2), generated as stubs alongside the
#: draft so a new rule cannot reach activation without its suite existing.
TEN_CATEGORIES = (
    "positive_fixture", "clean_negative_fixture", "boundary_case", "null_missing_input",
    "effective_date", "correction_cancellation_resubmission", "approved_exception",
    "idempotency", "tenant_isolation", "evidence_snapshot",
)

_DISPOSITION_WORDS = {
    "pend": Disposition.PREPAY_PEND,
    "hold": Disposition.PREPAY_PEND,
    "review before pay": Disposition.PREPAY_PEND,
    "audit": Disposition.POSTPAY_AUDIT,
    "post-pay": Disposition.POSTPAY_AUDIT,
    "recover": Disposition.POSTPAY_AUDIT,
    "reject": Disposition.REJECT,
    "deny": Disposition.REJECT,
    "reprice": Disposition.REPRICE,
    "return": Disposition.RETURN,
    "educate": Disposition.PROVIDER_EDUCATION,
    "education": Disposition.PROVIDER_EDUCATION,
    "investigate": Disposition.SIU_LEAD,
    "siu": Disposition.SIU_LEAD,
    "monitor": Disposition.MONITOR_ONLY,
    "watch": Disposition.MONITOR_ONLY,
}

#: Phrases that imply a peer-relative or self-history comparison, i.e. a
#: STATISTICAL control — which under §3.3 may never deny or reprice.
_STATISTICAL_CUES = (
    "peer", "median", "average", "percentile", "outlier", "unusual", "compared to",
    "more than twice", "exceeds .* times", "standard deviation", "anomal",
    "relative to", "changepoint", "trend",
)


class NaturalLanguageRuleAuthor:
    def __init__(self, provider: LLMProvider, registry, config, audit=None) -> None:
        self.provider = provider
        self.registry = registry
        self.config = config
        self.audit = audit

    # ------------------------------------------------------------------ draft

    def draft(
        self,
        request: str,
        *,
        author: str,
        scenario_id: str = "ANL-01",
        owner: str = "policy_owner",
    ) -> RuleDraft:
        payload = self._compose(request, scenario_id=scenario_id, owner=owner)
        response = self.provider.generate(
            "Draft an atomic-control contract for this policy request.",
            {"request": request, "draft": payload}, task="rule_draft",
        )
        yaml_text = yaml.safe_dump(payload, sort_keys=False, allow_unicode=True, width=100)

        control: AtomicControl | None = None
        errors: list[str] = []
        try:
            control = AtomicControl(**payload)
        except (DispositionLegalityError, ControlContractError) as exc:
            errors.append(str(exc))
        except Exception as exc:  # pydantic validation errors
            errors.append(str(exc))

        draft = RuleDraft(
            request=request,
            yaml_text=yaml_text,
            payload=payload,
            control=control,
            valid=control is not None,
            errors=errors,
            test_fixture_stubs=self._fixture_stubs(payload),
            diff=self._diff(payload),
        )
        if self.audit is not None:
            self.audit.record(
                AuditEventType.AI_TEXT_GENERATED,
                actor=author, actor_role="POLICY_OWNER",
                subject=payload.get("rule_id", "draft"),
                reason="Natural-language rule draft generated.",
                after={**response.to_audit_payload(), "valid": draft.valid, "errors": errors},
            )
        return draft

    def register(self, draft: RuleDraft, *, author: str) -> AtomicControl:
        """Place a valid draft in the registry — in ``shadow`` status only."""
        if not draft.valid or draft.control is None:
            raise ControlContractError(
                "A draft that does not satisfy the atomic-control contract cannot be registered. "
                + " ".join(draft.errors)
            )
        control = self.registry.draft(draft.control, author=author)
        if self.audit is not None:
            self.audit.record(
                AuditEventType.RULE_DRAFTED,
                actor=author, actor_role="POLICY_OWNER", subject=control.rule_id,
                reason="Rule drafted from natural language and entered in SHADOW status. "
                       "Activation requires a different approver (separation of duties).",
                after=control.to_yaml_dict(),
            )
        return control

    # ---------------------------------------------------------------- compose

    def _compose(self, request: str, *, scenario_id: str, owner: str) -> dict[str, Any]:
        text = (request or "").lower()

        disposition = Disposition.MONITOR_ONLY
        for word, disp in _DISPOSITION_WORDS.items():
            if re.search(rf"\b{re.escape(word)}", text):
                disposition = disp
                break

        statistical = any(re.search(cue, text) for cue in _STATISTICAL_CUES)
        control_type = "S" if statistical else "E"
        stage = Stage.MODEL_MONTHLY.value if statistical else Stage.PREPAY_SYNC.value

        # NOTE: the draft is composed HONESTLY, including when that makes it
        # invalid. If the request asks for a statistical test to deny a claim,
        # the draft says S + REJECT and the validator rejects it — which is
        # exactly the demonstration §9.6 is for. Quietly rewriting the request
        # into something legal would hide the boundary rather than show it.
        rule_id = self._next_rule_id(scenario_id)

        multiples = re.findall(r"(\d+(?:\.\d+)?)\s*(?:x|times|×)", text)
        thresholds = re.findall(r"(?:exceeds?|above|over|under|below)\s+(\d+(?:\.\d+)?)", text)
        parameters: dict[str, Any] = {}
        if multiples:
            parameters["peer_multiple"] = float(multiples[0])
        if thresholds:
            parameters["threshold"] = float(thresholds[0])

        inputs = [f for f in (
            "claim_header.pharmacy_bill_ratio" if "pharmacy" in text else None,
            "encounter.length_of_stay_days" if "stay" in text or "los" in text else None,
            "claim_header.gross_amount_aed" if "amount" in text or "cost" in text or "bill" in text else None,
            "diagnosis.code" if "diagnosis" in text or "peer" in text else None,
            "claim_header.provider_sk" if "provider" in text or "hospital" in text else None,
            "claim_header.member_sk" if "member" in text or "patient" in text else None,
            "claim_header.days_since_policy_start" if "tenure" in text or "policy start" in text else None,
        ) if f] or ["claim_header.*"]

        return {
            "rule_id": rule_id,
            "scenario_id": scenario_id,
            "name": self._name(request),
            "version": "1.0.0",
            "status": RuleStatus.SHADOW.value,
            "type": control_type,
            "stage": stage,
            "population": f"Drafted from the request: {request.strip()}",
            "inputs": inputs,
            "expression": request.strip(),
            "parameters": parameters,
            "exclusions": [
                "drafted_from_natural_language_requires_policy_owner_review",
                "no_domain_exclusions_declared_yet",
            ],
            "grouping_key": ["tenant_id", "payer_id", "provider_sk", "scenario_id", "period_bucket"],
            "score": 50.0,
            "disposition": disposition.value,
            "reason_code": self._reason_code(request),
            "evidence_fields": inputs + ["threshold", "peer_median", "peer_level_used"],
            "owner": owner,
            "effective_from": _dt.date.today().isoformat(),
            "priority": "P1",
            "data_support": DataSupport.NOT_EXECUTABLE_ON_THIS_DATASET.value,
            "data_support_reason": (
                "Drafted from natural language and not yet implemented. A drafted control has no "
                "implementation until an engineer writes one, and classifying it EXECUTABLE "
                "before that would overstate the catalogue."
            ),
            "implementation": None,
            "catalogue_trigger": request.strip(),
            "catalogue_config_exclusions": "To be declared by the policy owner before activation.",
            "catalogue_disposition": disposition.value,
            "governance_note": (
                "AUTHORED FROM NATURAL LANGUAGE. Enters the registry in SHADOW. The LLM drafted "
                "it; the deterministic contract validator decided whether it is legal; a human "
                "who did not author it must approve activation."
            ),
        }

    def _next_rule_id(self, scenario_id: str) -> str:
        existing = [c.rule_id for c in self.registry.by_scenario(scenario_id)]
        numbers = [int(r.split("-R")[-1]) for r in existing if "-R" in r]
        return f"{scenario_id}-R{(max(numbers) + 1) if numbers else 1:02d}"

    @staticmethod
    def _name(request: str) -> str:
        words = re.sub(r"[^a-zA-Z0-9 ]", " ", request).split()
        return " ".join(words[:8]).strip().capitalize() or "Drafted control"

    @staticmethod
    def _reason_code(request: str) -> str:
        words = [w.upper() for w in re.findall(r"[a-zA-Z]{4,}", request)][:3]
        return "_".join(words) or "DRAFTED_RULE"

    def _fixture_stubs(self, payload: dict[str, Any]) -> dict[str, str]:
        rule_id = payload.get("rule_id", "DRAFT")
        stubs = {}
        for category in TEN_CATEGORIES:
            stubs[category] = (
                f"def test_{rule_id.lower().replace('-', '_')}_{category}():\n"
                f'    """Test category: {category.replace("_", " ")}.\n\n'
                f"    STUB — generated alongside the draft of {rule_id}. The ten-category suite\n"
                f"    is mandatory for EVERY control regardless of type, so the stubs exist\n"
                f"    before activation is even proposed.\n"
                f'    """\n'
                f"    raise NotImplementedError(\n"
                f'        "{rule_id} cannot be activated until its {category} fixture is written."\n'
                f"    )\n"
            )
        return stubs

    def _diff(self, payload: dict[str, Any]) -> str:
        """Diff against the existing rule of the same id, if any. §9.6: 'show the diff'."""
        rule_id = payload.get("rule_id", "")
        try:
            existing = self.registry.get(rule_id)
        except Exception:
            return (
                f"NEW CONTROL — {rule_id} does not exist in the registry. The whole contract "
                f"below is new."
            )
        before = yaml.safe_dump(existing.to_yaml_dict(), sort_keys=False).splitlines()
        after = yaml.safe_dump(payload, sort_keys=False).splitlines()
        return "\n".join(
            difflib.unified_diff(before, after, fromfile=f"{rule_id} (current)",
                                 tofile=f"{rule_id} (draft)", lineterm="")
        )
