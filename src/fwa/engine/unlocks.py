"""Dataset unlocks — executability that depends on what the loaded file carries.

Every control in ``rules/`` declares a static ``data_support`` classification.
That classification is a statement about the **claim-header extract the
catalogue was classified against**, and the contract enforces it in both
directions: a control classified ``NOT_EXECUTABLE_ON_THIS_DATASET`` may not
name an implementation, and a runnable one must. Nothing here weakens that.

What it adds is the other half of the same honesty. A control that cannot run
on a claim-header file *can* run on a file that carries claim lines, a
licence register or a bundling edit table — and saying "not executable" about
such a file would be as untrue as claiming to run on one that lacks them. A
:class:`DatasetUnlock` is a separately governed declaration that says:

    "When the loaded dataset populates *these* tables (and, optionally, *these*
     columns), run *this* implementation, and classify the control *so* on that
     run, for *this* reason."

The evaluator consults it per run. On a file that does not populate every
named table the unlock is inert and the static classification stands, so on
the shipped claim-header files every existing figure is unchanged — a
governance test asserts exactly that.

What an unlock can never do:

* change a control's type, disposition, score, stage or owner — it carries none
  of them, so the §3.3 legality check that ran at registration still governs
  every signal the unlocked implementation raises;
* classify a control ``NOT_EXECUTABLE`` (that is the absence of an unlock);
* apply to a table the dataset does not actually populate with rows.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ..enums import DataSupport

__all__ = ["DatasetUnlock", "UnlockDecision", "load_unlocks", "UNLOCKS_SUBDIR"]

#: Unlock declarations live beside the catalogue, under ``rules/unlocks/``.
UNLOCKS_SUBDIR = "unlocks"


class DatasetUnlock(BaseModel):
    """One governed unlock declaration."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    rule_id: str
    implementation: str
    requires_tables: list[str]
    requires_columns: list[str] = Field(default_factory=list)   # "table.column"
    data_support: DataSupport
    reason: str
    #: When the static classification is PARTIAL (a proxy runs), an unlock may
    #: replace the proxy with the full implementation on a richer dataset. It
    #: must say so explicitly; an unlock never silently replaces running code.
    upgrades_partial: bool = False

    @field_validator("data_support")
    @classmethod
    def _runnable(cls, v: DataSupport) -> DataSupport:
        if v is DataSupport.NOT_EXECUTABLE_ON_THIS_DATASET:
            raise ValueError(
                "An unlock that classifies its control NOT_EXECUTABLE is not an unlock. "
                "Declare EXECUTABLE or PARTIAL, or remove the declaration."
            )
        return v

    @field_validator("requires_tables")
    @classmethod
    def _names_tables(cls, v: list[str]) -> list[str]:
        if not v:
            raise ValueError("An unlock must name at least one table the dataset must populate.")
        return v

    @field_validator("reason")
    @classmethod
    def _substantive(cls, v: str) -> str:
        if len(v.strip()) < 25:
            raise ValueError("An unlock must state a substantive reason (at least 25 characters).")
        return v.strip()

    @model_validator(mode="after")
    def _columns_belong_to_named_tables(self) -> "DatasetUnlock":
        for col in self.requires_columns:
            table = col.split(".")[0]
            if table not in self.requires_tables:
                raise ValueError(
                    f"{self.rule_id}: requires_columns entry {col!r} names table {table!r}, "
                    f"which is not in requires_tables."
                )
        return self

    # ------------------------------------------------------------------ check

    def missing(self, dataset: Any) -> list[str]:
        """What the dataset lacks for this unlock, in the order declared. Empty = satisfied."""
        if dataset is None:
            return list(self.requires_tables)
        gaps: list[str] = []
        for table in self.requires_tables:
            if not _populated(dataset, table):
                gaps.append(table)
        for col in self.requires_columns:
            table, _, column = col.partition(".")
            if table in gaps:
                continue
            frame = dataset.get(table) if _populated(dataset, table) else None
            if frame is None or column not in frame.columns or frame[column].isna().all():
                gaps.append(col)
        return gaps

    def satisfied_by(self, dataset: Any) -> bool:
        return not self.missing(dataset)


def _populated(dataset: Any, table: str) -> bool:
    check = getattr(dataset, "is_populated", None)
    if check is None:
        return False
    try:
        return bool(check(table))
    except KeyError:
        return False


class UnlockDecision(BaseModel):
    """What the evaluator decided for one control on one run."""

    model_config = ConfigDict(frozen=True)

    rule_id: str
    catalogue_support: DataSupport
    effective_support: DataSupport
    basis: str                      # "catalogue" | "unlocked" | "upgraded"
    implementation: str | None
    reason: str
    missing: list[str] = Field(default_factory=list)


def load_unlocks(rules_dir: Path | str) -> dict[str, DatasetUnlock]:
    """Read every ``rules/unlocks/*.yaml``. A malformed declaration aborts the load."""
    directory = Path(rules_dir) / UNLOCKS_SUBDIR
    out: dict[str, DatasetUnlock] = {}
    if not directory.exists():
        return out
    for path in sorted(directory.glob("*.yaml")):
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        for entry in raw.get("unlocks", []) or []:
            unlock = DatasetUnlock(**entry)
            if unlock.rule_id in out:
                raise ValueError(f"{unlock.rule_id} has more than one unlock declaration ({path.name}).")
            out[unlock.rule_id] = unlock
    return out


def decide(control, unlock: DatasetUnlock | None, dataset: Any) -> UnlockDecision:
    """The effective classification of ``control`` on ``dataset``."""
    static = control.data_support
    if unlock is None:
        return UnlockDecision(
            rule_id=control.rule_id, catalogue_support=static, effective_support=static,
            basis="catalogue", implementation=control.implementation,
            reason=control.data_support_reason,
        )
    gaps = unlock.missing(dataset)
    eligible = (
        static is DataSupport.NOT_EXECUTABLE_ON_THIS_DATASET
        or (static is DataSupport.PARTIAL and unlock.upgrades_partial)
    )
    if eligible and not gaps:
        return UnlockDecision(
            rule_id=control.rule_id, catalogue_support=static,
            effective_support=unlock.data_support,
            basis="unlocked" if static is DataSupport.NOT_EXECUTABLE_ON_THIS_DATASET else "upgraded",
            implementation=unlock.implementation, reason=unlock.reason,
        )
    return UnlockDecision(
        rule_id=control.rule_id, catalogue_support=static, effective_support=static,
        basis="catalogue", implementation=control.implementation,
        reason=control.data_support_reason, missing=gaps,
    )


def runtime_view(control, decision: UnlockDecision):
    """The control as it runs on this dataset, for the signals it raises.

    A copy with the effective classification, so evidence says "ran fully"
    rather than repeating the claim-header classification. Everything the
    safety boundary depends on — type, disposition, score, status — is carried
    over untouched from the registered control.
    """
    if decision.basis == "catalogue":
        return control
    return control.model_copy(update={
        "data_support": decision.effective_support,
        "data_support_reason": decision.reason,
        "implementation": decision.implementation,
    })


def unlock_rows(unlocks: Iterable[DatasetUnlock]) -> list[dict[str, Any]]:
    return [
        {
            "rule_id": u.rule_id, "implementation": u.implementation,
            "requires_tables": "; ".join(u.requires_tables),
            "requires_columns": "; ".join(u.requires_columns),
            "data_support_when_unlocked": u.data_support.value,
            "unlock_reason": u.reason, "upgrades_partial": u.upgrades_partial,
        }
        for u in unlocks
    ]
