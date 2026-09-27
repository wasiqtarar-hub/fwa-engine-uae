#!/usr/bin/env python3
"""Probe app pages for code-speak in their default view (developer tool).

Renders pages headlessly with the same harness as ``tests/ui`` and prints every
raw enum token, snake_case name, rule id (on Simple-mode pages) or bare
nan/None it finds in what a user sees without clicking.

    python tools/probe_plain_views.py <dataset> <adapter> [ROLE] [Page,Page]

Uses a throwaway database and preferences file, never ``data/fwa.db``.
"""
import os, sys, tempfile
os.environ["FWA_DATABASE_URL"] = "sqlite:///" + tempfile.mkdtemp().replace("\\", "/") + "/t.db"
os.environ["FWA_UI_PREFS"] = tempfile.mkdtemp() + "/p.json"
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1] / "tests"))
from ui.harness import pages_for, render, violations, visible_text
from fwa.auth import Role
from fwa.presentation import page_info
import dataset as ds
spec = ds.spec_from_path(sys.argv[1], sys.argv[2])
role = Role(sys.argv[3]) if len(sys.argv) > 3 else Role.ADMIN
only = set(sys.argv[4].split(",")) if len(sys.argv) > 4 else None
total = 0
for page in pages_for(role):
    if only and page not in only:
        continue
    at = render(page, role, spec, mode="simple")
    v = violations(visible_text(at, page), rule_ids=bool(page_info(page).get("simple")))
    total += len(v)
    print(f"== {page}: {len(v)} violations; exception={bool(at.exception)}")
    for line in v[:4]: print("   ", line[:220])
print("TOTAL", total)
