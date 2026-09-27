"""The rule registry — controls as *configuration*, not application logic.

Manuscript §3.7: "Rules are required to be configuration, not hard-coded
application logic, so that a policy owner can add, retire or version a control
without a code deployment."

This registry is what makes that true. It reads ``rules/**/*.yaml``, validates
every control against the §3.7 contract (which includes the §3.3 disposition
legality check), and exposes them by id, scenario, stage, type and status. The
Streamlit Rule Registry page writes new YAML through :meth:`RuleRegistry.draft`
and :meth:`RuleRegistry.activate` — no code deployment is involved in adding a
rule, which is the property §9.6 of the manuscript asks the artefact to
demonstrate.

Separation of duties (build brief §13.2) is enforced here rather than in the UI,
because a rule that can be activated by its own author through a different
entry point is not actually governed:

    >>> registry.activate("PAY-01-R01", actor="alice")   # alice authored it
    SeparationOfDutiesError: ...
"""

from __future__ import annotations

import datetime as _dt
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

import yaml

from ..enums import ControlType, DataSupport, Disposition, RuleStatus, Stage
from .contract import AtomicControl, ControlContractError, DispositionLegalityError

__all__ = ["RuleRegistry", "SeparationOfDutiesError", "RuleNotFound", "DEFAULT_RULES_DIR"]

DEFAULT_RULES_DIR = Path(__file__).resolve().parents[3] / "rules"


class RuleNotFound(KeyError):
    pass


class SeparationOfDutiesError(PermissionError):
    """The author of a rule may not approve its own shadow→active transition.

    Build brief §13.2, derived from manuscript §9.3/§9.4. Enforced in code and
    tested, not left to UI discipline.
    """


class RuleRegistry:
    """Versioned, owned, effective-dated store of atomic controls."""

    def __init__(self, controls: Iterable[AtomicControl] | None = None) -> None:
        self._by_id: dict[str, AtomicControl] = {}
        self._history: dict[str, list[AtomicControl]] = {}
        self._load_errors: list[tuple[str, str]] = []
        for c in controls or []:
            self.register(c)

    # ------------------------------------------------------------------ load

    @classmethod
    def from_directory(
        cls, rules_dir: Path | str | None = None, *, strict: bool = True
    ) -> "RuleRegistry":
        """Load every scenario YAML under ``rules/``.

        ``strict=True`` (the default, and what the application uses) means a
        contract violation anywhere aborts the load. The engine does not start
        with an invalid or illegal control in the catalogue. ``strict=False``
        exists only so that the Rule Registry UI page can *show* a broken draft
        to its author with its error message instead of crashing the app.
        """
        rules_dir = Path(rules_dir) if rules_dir else DEFAULT_RULES_DIR
        registry = cls()
        paths = sorted(rules_dir.rglob("*.yaml")) + sorted(rules_dir.rglob("*.yml"))
        for path in paths:
            raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
            scenario_defaults = {
                k: v for k, v in raw.items() if k not in ("controls", "scenario")
            }
            scenario = raw.get("scenario", {}) or {}
            for entry in raw.get("controls", []) or []:
                merged: dict[str, Any] = {
                    "scenario_id": scenario.get("scenario_id"),
                    "priority": scenario.get("priority", "P1"),
                    **scenario_defaults,
                    **entry,
                }
                merged.pop("name_of_scenario", None)
                try:
                    control = AtomicControl(**merged)
                except DispositionLegalityError:
                    # Never tolerated, in either mode. ``strict=False`` exists so
                    # the Rule Registry page can show an author a *malformed*
                    # draft instead of crashing; a §3.3 violation is a different
                    # thing entirely and aborts the load unconditionally.
                    raise
                except (ControlContractError, Exception) as exc:
                    message = f"{path.name}:{entry.get('rule_id', '?')}: {exc}"
                    if strict:
                        raise
                    registry._load_errors.append((entry.get("rule_id", "?"), message))
                    continue
                registry.register(control)
        return registry

    # -------------------------------------------------------------- register

    def register(self, control: AtomicControl) -> AtomicControl:
        """Register a control.

        Construction of :class:`AtomicControl` already ran the §3.3 legality
        check; this re-runs it explicitly so that a control built by any other
        route (a test fixture, an LLM draft, a hand-made dict) cannot slip past.
        Defence in depth on the one constraint that must not fail.
        """
        from .contract import validate_disposition_legality

        validate_disposition_legality(control.rule_id, control.type, control.disposition)

        existing = self._by_id.get(control.rule_id)
        if existing is not None and existing.version != control.version:
            self._history.setdefault(control.rule_id, []).append(existing)
        elif existing is not None:
            # Same id, same version, different content is a governance error:
            # a rule change must bump the version so stored signals remain
            # reproducible against the exact text that produced them.
            if existing.fingerprint() != control.fingerprint():
                raise ControlContractError(
                    f"{control.rule_id} v{control.version} is being registered with different "
                    f"content than the already-registered v{control.version}. Bump the version: "
                    "signals store rule_id@version and must remain reproducible (§3.10)."
                )
        self._by_id[control.rule_id] = control
        return control

    # ------------------------------------------------------------------ read

    def get(self, rule_id: str) -> AtomicControl:
        try:
            return self._by_id[rule_id]
        except KeyError as exc:
            raise RuleNotFound(rule_id) from exc

    def __contains__(self, rule_id: object) -> bool:
        return rule_id in self._by_id

    def __len__(self) -> int:
        return len(self._by_id)

    def __iter__(self) -> Iterator[AtomicControl]:
        return iter(self.all())

    def all(self) -> list[AtomicControl]:
        return [self._by_id[k] for k in sorted(self._by_id)]

    def history(self, rule_id: str) -> list[AtomicControl]:
        return list(self._history.get(rule_id, []))

    @property
    def load_errors(self) -> list[tuple[str, str]]:
        return list(self._load_errors)

    # ---- selectors ----------------------------------------------------------

    def by_scenario(self, scenario_id: str) -> list[AtomicControl]:
        return [c for c in self.all() if c.scenario_id == scenario_id]

    def scenarios(self) -> list[str]:
        return sorted({c.scenario_id for c in self.all()})

    def families(self) -> list[str]:
        return sorted({c.scenario_id.split("-")[0] for c in self.all()})

    def by_stage(self, stage: Stage) -> list[AtomicControl]:
        return [c for c in self.all() if c.stage is stage]

    def by_type(self, ctype: ControlType) -> list[AtomicControl]:
        return [c for c in self.all() if ctype in c.type]

    def by_status(self, status: RuleStatus) -> list[AtomicControl]:
        return [c for c in self.all() if c.status is status]

    def executable(self) -> list[AtomicControl]:
        """Controls this build genuinely runs against the claim extract."""
        return [c for c in self.all() if c.data_support is DataSupport.EXECUTABLE]

    def runnable(self, as_of: _dt.date) -> list[AtomicControl]:
        return [c for c in self.executable() if c.is_runnable(as_of)]

    def denying_controls(self) -> list[AtomicControl]:
        """Every control in the catalogue that may change what is paid.

        Surfaced in the Governance page. The list should be short, and every
        entry should be type H or H/E — if it is not, the registry would not
        have loaded.
        """
        return [c for c in self.all() if c.can_deny]

    # ---------------------------------------------------------------- mutate

    def draft(self, control: AtomicControl, *, author: str) -> AtomicControl:
        """Enter a new control into the registry in ``shadow`` status only.

        Used by the natural-language rule-authoring flow (§9.6). A drafted rule
        is never active, whoever drafted it and however confident the draft.
        """
        drafted = control.model_copy(
            update={
                "status": RuleStatus.SHADOW,
                "authored_by": author,
                "approved_by": None,
                "approved_at": None,
                "shadow_since": _dt.date.today(),
            }
        )
        return self.register(drafted)

    def activate(self, rule_id: str, *, actor: str, note: str = "") -> AtomicControl:
        """Promote shadow → active.

        Raises :class:`SeparationOfDutiesError` if ``actor`` authored the rule.
        """
        control = self.get(rule_id)
        if control.status is RuleStatus.ACTIVE:
            return control
        if control.authored_by and control.authored_by == actor:
            raise SeparationOfDutiesError(
                f"{actor} authored {rule_id} and may not approve its own activation. "
                "Separation of duties (build brief §13.2, manuscript §9.3): the user who "
                "authors a rule may not approve its shadow→active transition."
            )
        promoted = control.model_copy(
            update={
                "status": RuleStatus.ACTIVE,
                "approved_by": actor,
                "approved_at": _dt.datetime.now(_dt.timezone.utc),
                "governance_note": (control.governance_note + " " + note).strip(),
            }
        )
        self._by_id[rule_id] = promoted
        return promoted

    def retire(self, rule_id: str, *, actor: str, reason: str) -> AtomicControl:
        control = self.get(rule_id)
        retired = control.model_copy(
            update={
                "status": RuleStatus.RETIRED,
                "governance_note": f"{control.governance_note} Retired by {actor}: {reason}".strip(),
            }
        )
        self._history.setdefault(rule_id, []).append(control)
        self._by_id[rule_id] = retired
        return retired

    def kill_switch(self, rule_id: str, *, actor: str, reason: str) -> AtomicControl:
        """Deactivate a control without a code deployment (§3.10 NFR, §9.7).

        The control's signals are routed to ``MONITOR_ONLY`` rather than
        suppressed, so the coverage gap is visible in the queue instead of
        silently disappearing.
        """
        if not reason.strip():
            raise ValueError("A kill switch requires a stated reason (§9.7).")
        control = self.get(rule_id)
        killed = control.model_copy(
            update={"kill_switched": True, "kill_switch_reason": f"{actor}: {reason}"}
        )
        self._by_id[rule_id] = killed
        return killed

    def restore(self, rule_id: str, *, actor: str, reason: str) -> AtomicControl:
        control = self.get(rule_id)
        restored = control.model_copy(
            update={
                "kill_switched": False,
                "kill_switch_reason": "",
                "governance_note": f"{control.governance_note} Restored by {actor}: {reason}".strip(),
            }
        )
        self._by_id[rule_id] = restored
        return restored

    def rollback(self, rule_id: str, *, actor: str, to_version: str | None = None) -> AtomicControl:
        """Roll a control back to a previous registered version (§9.7)."""
        versions = self.history(rule_id)
        if not versions:
            raise RuleNotFound(f"No prior version of {rule_id} is registered.")
        if to_version:
            match = [v for v in versions if v.version == to_version]
            if not match:
                raise RuleNotFound(f"{rule_id} has no registered version {to_version}.")
            target = match[-1]
        else:
            target = versions[-1]
        current = self.get(rule_id)
        self._history.setdefault(rule_id, []).append(current)
        restored = target.model_copy(
            update={
                "status": RuleStatus.SHADOW,  # a rollback re-enters shadow, never straight to active
                "governance_note": f"Rolled back by {actor} from v{current.version} to v{target.version}.",
            }
        )
        self._by_id[rule_id] = restored
        return restored

    # ----------------------------------------------------------------- audit

    def coverage_rows(self) -> list[dict[str, Any]]:
        """Rows for ``reports/control_coverage_matrix.csv`` (build brief §11.2)."""
        rows = []
        for c in self.all():
            rows.append(
                {
                    "rule_id": c.rule_id,
                    "scenario_id": c.scenario_id,
                    "family": c.scenario_id.split("-")[0],
                    "name": c.name,
                    "priority": c.priority,
                    "type": c.type_label,
                    "stage": c.stage.value,
                    "status": c.status.value,
                    "disposition": c.disposition.value,
                    "can_deny": c.can_deny,
                    "data_support": c.data_support.value,
                    "data_support_reason": c.data_support_reason,
                    "required_canonical_fields": "; ".join(c.required_canonical_fields),
                    "implementation": c.implementation or "",
                    "owner": c.owner,
                    "exclusions": "; ".join(c.exclusions),
                    "catalogue_trigger": c.catalogue_trigger,
                    "catalogue_disposition": c.catalogue_disposition,
                    "governance_note": c.governance_note,
                    "rule_fingerprint": c.fingerprint(),
                }
            )
        return rows

    def summary(self) -> dict[str, Any]:
        rows = self.coverage_rows()
        by_support: dict[str, int] = {}
        for r in rows:
            by_support[r["data_support"]] = by_support.get(r["data_support"], 0) + 1
        return {
            "total_controls": len(rows),
            "total_scenarios": len(self.scenarios()),
            "by_data_support": by_support,
            "by_status": {s.value: len(self.by_status(s)) for s in RuleStatus},
            "controls_that_may_deny": len(self.denying_controls()),
        }
