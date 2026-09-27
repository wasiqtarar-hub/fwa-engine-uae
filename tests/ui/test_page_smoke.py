"""Every page opens, for every role, on all three datasets, without an error.

A page that raises shows the user a traceback, which is exactly what the
usability work exists to prevent. Failures inside a page are expected to be
turned into a friendly message by ``common.friendly_failure``; this test proves
none escapes.
"""

from __future__ import annotations

import pytest

from fwa.auth import Role

from .conftest import DATASETS, dataset_spec
from .harness import pages_for, render, visible_text

pytestmark = pytest.mark.slow


@pytest.mark.parametrize("dataset_key", list(DATASETS))
@pytest.mark.parametrize("role", list(Role), ids=lambda r: r.value.lower())
def test_every_page_opens_without_error(dataset_key, role):
    spec = dataset_spec(dataset_key)
    failures = []
    for page in pages_for(role):
        at = render(page, role, spec, mode="advanced")
        if at.exception:
            failures.append(f"{page}: {at.exception[0].value[:300]}")
            continue
        text = " ".join(t for _, t in visible_text(at, page).items)
        if "Traceback (most recent call last)" in text:
            failures.append(f"{page}: a raw traceback is on screen")
    assert not failures, "\n".join(failures)
