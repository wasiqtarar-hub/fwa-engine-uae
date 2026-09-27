"""``fwa`` — governed fraud, waste and abuse detection engine.

Executable artefact for the MSc Data Science & AI dissertation *"Automated
Fraud, Waste and Abuse Detection in UAE Medical Claims Using Explainable
Unsupervised Learning and Governed Payment-Integrity Controls"* (Tarar, 2026).

The controlling specification is the manuscript. Where this code and the
manuscript disagree, the manuscript wins and the discrepancy is a bug.

The one constraint that outranks every other design consideration here:

    **A signal is not a fraud finding.** (manuscript §3.3)

That is not a comment. It is enforced structurally — see
:mod:`fwa.enums` for the eight governed dispositions and
:func:`fwa.engine.contract.validate_disposition_legality`, which raises at
**rule-registration time** if a statistical, network, text or model control
attempts to declare ``REJECT`` or ``REPRICE``. It is impossible to configure an
anomaly score into a denial in this system.
"""

from __future__ import annotations

__version__ = "1.0.0"

SAFETY_BOUNDARY_STATEMENT = (
    "A signal is not a fraud finding. This system can establish non-payability, "
    "inconsistency or statistical abnormality. It cannot establish intent, and "
    "intent is what distinguishes fraud from waste, abuse or honest error. "
    "Only a human reviewer, on evidence, may reach a conclusion about conduct."
)

__all__ = ["__version__", "SAFETY_BOUNDARY_STATEMENT"]
