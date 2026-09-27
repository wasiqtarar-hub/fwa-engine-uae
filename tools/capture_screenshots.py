#!/usr/bin/env python3
"""Capture the USER_GUIDE screenshots FROM A REAL RUN.

This script starts nothing and mocks nothing. It drives a running instance of
the application with Playwright, signs in as each role with that role's real
credentials, visits each page that role can reach, and writes a PNG. If the
application is not running, it says so and exits — a screenshot of a page that
did not load is worse than no screenshot.

Usage::

    streamlit run app/Home.py           # in one terminal
    python tools/capture_screenshots.py # in another
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "screenshots"

# Each seeded account, its post-change password, and the pages worth capturing.
PLAN = [
    ("admin", "admin", "AdminPass2026!", [
        ("overview", "Overview"),
        ("user-management", "User Management"),
        ("governance", "Governance"),
    ]),
    ("reviewer", "reviewer", "ReviewPass2026!", [
        ("review-queue", "Review Queue"),
        ("case-evidence", "Case Evidence"),
    ]),
    ("policy", "policy", "PolicyPass2026!", [
        ("rule-registry", "Rule Registry"),
    ]),
    ("analyst", "analyst", "AnalystPass2026!", [
        ("data", "Data"),
        ("parameters-models", "Parameters & Models"),
        ("provider-analytics", "Provider Analytics"),
        ("models", "Models"),
        ("network", "Network"),
        ("validation-report", "Validation Report"),
    ]),
    ("auditor", "auditor", "AuditPass2026!", [
        ("help", "Help"),
    ]),
]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8501")
    parser.add_argument("--out", default=str(OUT))
    parser.add_argument("--width", type=int, default=1560)
    parser.add_argument("--height", type=int, default=1400)
    parser.add_argument("--only", default="",
                        help="Comma-separated usernames to capture, for re-shooting one "
                             "page without paying the whole set's cost again.")
    args = parser.parse_args()
    wanted = {u.strip() for u in args.only.split(",") if u.strip()}

    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print("playwright is not installed: pip install playwright && playwright install chromium")
        return 1

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    captured = []

    with sync_playwright() as p:
        browser = p.chromium.launch()
        for username, seed_password, new_password, pages in PLAN:
            if wanted and username not in wanted:
                continue
            context = browser.new_context(
                viewport={"width": args.width, "height": args.height}, device_scale_factor=2)
            page = context.new_page()
            page.goto(args.url, wait_until="networkidle", timeout=120_000)
            page.wait_for_timeout(2500)

            _wait_until_settled(page)
            if not _sign_in(page, username, seed_password, new_password):
                print(f"  ! could not sign in as {username}")
                context.close()
                continue

            for url_path, label in pages:
                # Navigate by CLICKING the sidebar link, never by page.goto.
                # Streamlit session state lives on the websocket connection, so
                # a fresh URL load starts a new session and loses the sign-in —
                # which would produce a screenshot of the login page labelled as
                # the page it is not.
                link = page.get_by_role("link", name=label, exact=False).first
                if link.count() == 0:
                    print(f"  ! {label} is not in the sidebar for {username}")
                    continue
                link.click()
                page.wait_for_timeout(2000)
                _wait_until_settled(page)
                _expand_all(page)
                _scroll_to_top(page)
                errors = page.locator("[data-testid='stException']").count()
                if errors:
                    detail = page.locator("[data-testid='stException']").first.inner_text()[:400]
                    print(f"  ! {label} rendered {errors} exception(s): {detail}")
                path = out / f"{url_path}-{username}.png"
                page.screenshot(path=str(path), full_page=True)
                captured.append(path)
                print(f"  ✓ {label} as {username} → {path.name}")
            context.close()
        browser.close()

    if wanted:
        print(f"\nRe-captured {len(captured)} screenshot(s); MANIFEST left as it was.")
        return 0

    manifest = out / "MANIFEST.md"
    manifest.write_text(
        "# Screenshots\n\n"
        "Every image in this directory was captured by `tools/capture_screenshots.py` from a "
        "**running instance** of the application, signed in with the real seeded credentials for "
        "the stated role. None is a mock-up.\n\n"
        + "\n".join(f"- `{p.name}`" for p in sorted(captured)) + "\n",
        encoding="utf-8",
    )
    print(f"\nCaptured {len(captured)} screenshot(s) to {out}")
    return 0


def _wait_until_settled(page, timeout_ms: int = 300_000) -> None:
    """Wait for every spinner and running indicator to clear.

    The first page a session opens triggers the cached pipeline — a full run
    over 5,000 claims, which takes the better part of a minute. A fixed sleep
    that is shorter than that produces a screenshot of a loading spinner
    labelled as the Overview page, which is precisely the kind of image the
    build brief means when it says do not mock them up.
    """
    deadline = timeout_ms
    step = 1000
    while deadline > 0:
        busy = (
            page.locator("[data-testid='stSpinner']").count()
            or page.locator("[data-testid='stStatusWidget']").count()
        )
        if not busy:
            page.wait_for_timeout(1500)
            return
        page.wait_for_timeout(step)
        deadline -= step
    print("  ! still busy after waiting; capturing anyway")


def _scroll_to_top(page) -> None:
    """Start the image at the top of the page.

    Opening the expanders leaves the viewport part-way down whichever one was
    clicked last, and a guide illustrated with pages that all begin mid-table
    is harder to follow than one that begins where the reader would. Streamlit
    scrolls an inner container rather than the window, so both are reset.
    """
    try:
        page.evaluate(
            "() => { window.scrollTo(0, 0);"
            " document.querySelectorAll('section.main, [data-testid=\"stMain\"],"
            " [data-testid=\"stAppViewContainer\"]').forEach(e => e.scrollTop = 0); }"
        )
        page.wait_for_timeout(800)
    except Exception:
        pass


def _expand_all(page) -> None:
    """Open every expander so the screenshot shows the panels, not the headings."""
    try:
        summaries = page.locator("details summary")
        for i in range(min(summaries.count(), 12)):
            try:
                summaries.nth(i).click(timeout=2000)
                page.wait_for_timeout(250)
            except Exception:
                continue
        page.wait_for_timeout(1500)
    except Exception:
        pass


def _sign_in(page, username: str, seed_password: str, new_password: str) -> bool:
    """Sign in, handling the forced first-login password change."""
    for password in (seed_password, new_password):
        try:
            inputs = page.locator("input")
            inputs.nth(0).fill(username)
            inputs.nth(1).fill(password)
            page.get_by_role("button", name="Sign in").click()
            page.wait_for_timeout(5000)
        except Exception:
            return False

        if page.get_by_text("Choose a new password").count():
            fields = page.locator("input[type=password]")
            fields.nth(0).fill(password)
            fields.nth(1).fill(new_password)
            fields.nth(2).fill(new_password)
            page.get_by_role("button", name="Change password").click()
            page.wait_for_timeout(6000)

        if page.locator("section[data-testid='stSidebar']").locator("a").count():
            return True
    return False


if __name__ == "__main__":
    sys.exit(main())
