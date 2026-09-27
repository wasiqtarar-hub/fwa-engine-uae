"""Financial exposure, computed conservatively by signal type (manuscript §4.11).

    "Financial exposure is computed differently by signal type, and the design is
     DELIBERATELY CONSERVATIVE about what may be called 'exposure': a line
     edit's exposure is max(0, submitted_payable − correctly_repriced_payable);
     a duplicate's exposure is the LOWER of the duplicated paid or requested
     amounts after valid patient share; a provider-pattern exposure is the sum
     of sampled suspect incremental amounts, with extrapolation shown ONLY as an
     explicitly labelled interval and only once audit policy permits it; a
     network case sums distinct paid or requested lines ONCE, never
     double-counting a claim that touches multiple graph nodes; and an uncertain
     MODEL-ONLY LEAD shows its gross amount together with the explicit statement
     'exposure not yet established' rather than presenting that gross amount as
     if it were savings. This last rule is one of the more consequential design
     decisions in the whole system, because it is the rule that prevents the
     system from ever silently overstating its own value, a risk this
     dissertation treats as an ETHICAL REQUIREMENT, not only a
     measurement-accuracy one."                            — manuscript §4.11

Each function below returns an :class:`Exposure` carrying the amount, a
human-readable derivation shown verbatim in the Case Evidence panel, and an
``established`` flag. When ``established`` is ``False`` the UI and every report
render the amount with :data:`fwa.engine.signals.EXPOSURE_NOT_ESTABLISHED`
beside it. Nothing in this system adds an unestablished exposure into a savings
total; :mod:`fwa.evaluation.metrics` keeps the two apart by construction.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Sequence

from ..engine.signals import EXPOSURE_NOT_ESTABLISHED

__all__ = [
    "Exposure",
    "line_edit_exposure",
    "duplicate_exposure",
    "provider_pattern_exposure",
    "network_exposure",
    "model_only_exposure",
    "no_exposure",
]


@dataclass(frozen=True)
class Exposure:
    amount_aed: float
    basis: str
    established: bool = True
    interval_low_aed: float | None = None
    interval_high_aed: float | None = None
    extrapolated: bool = False

    def display(self) -> str:
        if not self.established:
            return f"AED {self.amount_aed:,.2f} — {EXPOSURE_NOT_ESTABLISHED}"
        if self.extrapolated and self.interval_low_aed is not None:
            return (
                f"AED {self.amount_aed:,.2f} "
                f"(EXTRAPOLATED, labelled interval AED {self.interval_low_aed:,.2f}–"
                f"{self.interval_high_aed:,.2f})"
            )
        return f"AED {self.amount_aed:,.2f}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "exposure_aed": round(self.amount_aed, 2),
            "exposure_basis": self.basis,
            "exposure_established": self.established,
            "exposure_interval_low_aed": self.interval_low_aed,
            "exposure_interval_high_aed": self.interval_high_aed,
            "exposure_extrapolated": self.extrapolated,
            "exposure_display": self.display(),
        }


def no_exposure(reason: str) -> Exposure:
    """Zero exposure with a stated reason.

    Used where a control finds something real that has no computable monetary
    consequence — a data-integrity RETURN, for example. Zero is the honest
    figure; carrying the claim's gross amount forward would inflate the queue's
    apparent value.
    """
    return Exposure(0.0, f"No monetary exposure: {reason}", established=True)


def line_edit_exposure(submitted_payable: float, correctly_repriced_payable: float) -> Exposure:
    """``max(0, submitted − correctly_repriced)`` — §4.11 line-edit rule."""
    amount = max(0.0, float(submitted_payable) - float(correctly_repriced_payable))
    return Exposure(
        amount,
        f"Line edit: submitted payable AED {float(submitted_payable):,.2f} minus correctly "
        f"repriced payable AED {float(correctly_repriced_payable):,.2f}, floored at zero.",
    )


def duplicate_exposure(
    amounts: Sequence[float], valid_patient_share: float = 0.0, *, count_note: str = ""
) -> Exposure:
    """Lower of the duplicated amounts, after valid patient share — §4.11.

    Taking the *lower* is the conservative choice the manuscript specifies: if
    two claims duplicate one service, at most the smaller of them is recoverable,
    because the larger may be the legitimate one.
    """
    vals = [float(a) for a in amounts if a is not None]
    if len(vals) < 2:
        return no_exposure("fewer than two amounts were supplied to a duplicate exposure")
    lower = min(vals)
    amount = max(0.0, lower - float(valid_patient_share))
    return Exposure(
        amount,
        f"Duplicate: the LOWER of the duplicated amounts (AED {lower:,.2f} of "
        f"{', '.join(f'AED {v:,.2f}' for v in sorted(vals))}), less valid patient share "
        f"AED {float(valid_patient_share):,.2f}. The lower amount is taken because the larger "
        f"claim may be the legitimate one." + (f" {count_note}" if count_note else ""),
    )


def provider_pattern_exposure(
    sampled_incremental_amounts: Sequence[float],
    *,
    population_size: int | None = None,
    sample_size: int | None = None,
    audit_policy_permits_extrapolation: bool = False,
) -> Exposure:
    """Sum of sampled suspect incremental amounts — §4.11.

    Extrapolation to the full population is shown **only** as an explicitly
    labelled interval, and **only** once audit policy permits it. The default
    here is that it does not: no audit policy exists for a dissertation
    artefact, so the honest figure is the sampled sum alone.
    """
    sampled = [max(0.0, float(a)) for a in sampled_incremental_amounts if a is not None]
    base = sum(sampled)
    basis = (
        f"Provider pattern: sum of {len(sampled)} sampled suspect incremental amount(s) "
        f"= AED {base:,.2f}."
    )
    if not audit_policy_permits_extrapolation or not population_size or not sample_size:
        return Exposure(
            base,
            basis + " NOT extrapolated to the provider's full population: extrapolation is "
                    "permitted only once audit policy allows it, and no such policy exists "
                    "for this artefact.",
        )
    factor = population_size / max(sample_size, 1)
    point = base * factor
    # A crude but honest interval: the sampled sum is the floor, the naive
    # extrapolation the ceiling. Presented as an interval precisely so it cannot
    # be mistaken for a measured amount.
    return Exposure(
        base,
        basis + f" Extrapolation permitted by audit policy: ×{factor:.2f} to "
                f"AED {point:,.2f}, shown as a LABELLED INTERVAL only.",
        interval_low_aed=base,
        interval_high_aed=point,
        extrapolated=True,
    )


def network_exposure(claim_amounts: dict[str, float]) -> Exposure:
    """Each distinct claim counted exactly once — §4.11, §6.3.

    The argument is a mapping keyed by ``claim_sk`` precisely so that a claim
    touching several graph nodes cannot be added twice. The integration test
    ``test_graph_case_counts_each_claim_once`` passes a claim that appears under
    three nodes and asserts the total is the single claim's amount.
    """
    total = sum(float(v) for v in claim_amounts.values())
    return Exposure(
        total,
        f"Network case: {len(claim_amounts)} DISTINCT claim(s) summed exactly once "
        f"(keyed by claim_sk), total AED {total:,.2f}. A claim touching several nodes in the "
        f"community contributes once.",
    )


def model_only_exposure(gross_amount_aed: float, model_name: str = "model") -> Exposure:
    """Gross amount plus the literal §4.11 statement. Never savings.

    This is the rule the manuscript calls one of the most consequential in the
    system. The amount is carried so a reviewer can see the scale of what is at
    stake; ``established=False`` is what stops it from ever being counted.
    """
    return Exposure(
        float(gross_amount_aed),
        f"Model-only lead ({model_name}): gross amount AED {float(gross_amount_aed):,.2f}. "
        f"{EXPOSURE_NOT_ESTABLISHED} — no objective condition has been shown to fail, so no "
        f"part of this amount has been established as recoverable or preventable.",
        established=False,
    )
