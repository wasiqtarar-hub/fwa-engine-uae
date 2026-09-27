"""The default view of every page speaks plain English.

For every role, every page that role can open is rendered in Simple mode on
the app's default dataset, and the text a user sees without clicking anything
(collapsed expanders and non-first tabs excluded) is scanned for:

* raw enumeration tokens (``POSTPAY_AUDIT``, ``NOT_EXECUTABLE_ON_THIS_DATASET``),
* snake_case field names (``exposure_aed``, ``scenario_family``),
* bare ``nan`` / ``None`` / ``NaT``,
* and, on the Simple-mode pages, rule ids (``CLN-01-R01``).

File names, claim and case ids and masked pseudonyms are allowed: they are
identifiers a user reads, not code.
"""

from __future__ import annotations

import pytest

from fwa.auth import Role
from fwa.presentation import page_info

from .conftest import dataset_spec
from .harness import pages_for, render, violations, visible_text

pytestmark = pytest.mark.slow


@pytest.mark.parametrize("role", list(Role), ids=lambda r: r.value.lower())
def test_default_view_has_no_code_tokens(role):
    spec = dataset_spec("uae_demo")
    problems: list[str] = []
    for page in pages_for(role):
        at = render(page, role, spec, mode="simple")
        seen = visible_text(at, page)
        problems += violations(seen, rule_ids=bool(page_info(page).get("simple")))
    assert not problems, f"{len(problems)} problem(s):\n" + "\n".join(problems[:80])
