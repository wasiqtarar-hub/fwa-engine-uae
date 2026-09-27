#!/usr/bin/env python3
"""Generate ``rules/**/*.yaml`` from the Appendix C transcription.

Rules are *configuration* (§3.7), and the YAML under ``rules/`` is the
configuration the engine actually loads. This script is how that configuration
is produced from the transcribed catalogue, so that the 164 controls cannot
drift away from Appendix C by hand-editing.

Run::

    python tools/gen_rules.py

It is idempotent. Re-running it after editing ``catalogue_source.py``
regenerates every file; the engine then validates each control against the §3.7
contract and the §3.3 disposition-legality rule at load time.

Note that hand-editing a generated file is legitimate — a policy owner adding an
exclusion is exactly the workflow §9.6 describes — but the edit must be carried
back into ``catalogue_source.py``, or the next regeneration will overwrite it.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent))
import catalogue_source as cat  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RULES_DIR = ROOT / "rules"

DEFAULT_SCORES = {
    ("P0", "H"): 90, ("P0", "E"): 78, ("P0", "S"): 62, ("P0", "N"): 58,
    ("P0", "T"): 60, ("P0", "M"): 50,
    ("P1", "H"): 82, ("P1", "E"): 70, ("P1", "S"): 58, ("P1", "N"): 55,
    ("P1", "T"): 56, ("P1", "M"): 48,
    ("P2", "H"): 70, ("P2", "E"): 60, ("P2", "S"): 48, ("P2", "N"): 46,
    ("P2", "T"): 46, ("P2", "M"): 40,
}

FAMILY_GROUPING = {
    "ENT": ["tenant_id", "payer_id", "member_sk", "scenario_id", "period_bucket"],
    "PAY": ["tenant_id", "payer_id", "member_sk", "episode_id", "scenario_id"],
    "CLN": ["tenant_id", "payer_id", "provider_sk", "episode_id", "scenario_id"],
    "PHR": ["tenant_id", "payer_id", "member_sk", "scenario_id", "period_bucket"],
    "DOC": ["tenant_id", "payer_id", "claim_sk", "scenario_id"],
    "NET": ["tenant_id", "payer_id", "community_id", "scenario_id", "period_bucket"],
    "ANL": ["tenant_id", "payer_id", "provider_sk", "scenario_id", "period_bucket"],
    "POL": ["tenant_id", "payer_id", "member_sk", "scenario_id", "period_bucket"],
}

_SPLIT = re.compile(r"[;,]| and | or |/")


def slugify(text: str) -> str:
    s = text.strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    return s.strip("_")[:60] or "unspecified"


def derive_exclusions(config_text: str) -> list[str]:
    """Turn Appendix C's 'Configuration and exclusions' prose into tokens.

    The catalogue column mixes configuration with exclusions, so this mapping is
    mechanical and is declared as such in every generated file's header. For the
    36 implemented controls the exclusions are hand-authored instead, because
    those are the ones whose exclusion logic actually executes and is tested by
    the approved-exception test category (§6.2 category 7).
    """
    parts = [p for p in _SPLIT.split(config_text) if p and p.strip()]
    tokens, seen = [], set()
    for p in parts:
        tok = slugify(p)
        if tok and tok not in seen and len(tok) > 2:
            seen.add(tok)
            tokens.append(tok)
    return tokens or ["none_declared_by_policy_owner"]


def build_control(rec: dict) -> dict:
    fam = rec["rid"][:3]
    scenario_id = rec["rid"][:6]
    title, priority, starred = cat.SCENARIOS[scenario_id]
    primary_type = rec["types"].split("/")[0]
    score = rec["score"] or DEFAULT_SCORES.get((priority, primary_type), 50)

    exclusions = rec["exclusions"] or derive_exclusions(rec["config"])
    inputs = rec["inputs"] or rec["req"] or ["<canonical fields unavailable on this dataset>"]
    population = rec["population"] or rec["trigger"][:120]
    expression = rec["expression"] or rec["trigger"]
    grouping = rec["grouping"] or FAMILY_GROUPING[fam]
    evidence = rec["evidence"] or ["rule_id", "scenario_id", "subject_id", "reason_code"]

    out = {
        "rule_id": rec["rid"],
        "name": rec["name"],
        "version": "1.0.0",
        "status": "shadow",
        "type": rec["types"],
        "stage": rec["stage"],
        "population": population,
        "inputs": inputs,
        "expression": expression,
        "parameters": rec["params"],
        "exclusions": exclusions,
        "grouping_key": grouping,
        "score": score,
        "disposition": rec["disp"],
        "reason_code": rec["rc"],
        "evidence_fields": evidence,
        "owner": rec["owner"],
        "effective_from": "2026-01-01",
        "priority": priority,
        "data_support": rec["support"],
        "data_support_reason": " ".join(rec["why"].split()),
        "required_canonical_fields": rec["req"],
        "implementation": rec["impl"],
        "catalogue_trigger": rec["trigger"],
        "catalogue_config_exclusions": rec["config"],
        "catalogue_disposition": rec["outcome"],
        "alternate_dispositions": rec["alt"],
        "governance_note": " ".join(rec["note"].split()),
        "shadow_since": "2026-01-01",
    }
    return out


HEADER = """\
# =============================================================================
# {scenario_id}: {title} · {priority}{star}
# =============================================================================
# Scenario transcribed from the control catalogue. Every control below carries
# the catalogue's verbatim trigger, configuration/exclusions and
# output/disposition alongside this artefact's governed mapping.
#
# GENERATED FILE — regenerate with `python tools/gen_rules.py`. Hand edits are a
# legitimate policy-owner workflow but must be carried back into
# tools/catalogue_source.py or they will be overwritten.
#
# `disposition` is ONE of the eight governed dispositions. Where the catalogue
# names several ("Reject/pend"), the first is taken and the rest are recorded
# as `alternate_dispositions`, which are ADVISORY ONLY — the engine never acts
# on them. Where the first would be illegal under the safety boundary for the
# control's type, the substitution is recorded in `governance_note`.
#
# `exclusions` for controls that do not execute here are derived mechanically
# from the catalogue's "Configuration and exclusions" column; for the controls
# this build actually runs they are hand-authored and are exercised by the
# approved-exception test category.
#
# Every control starts in `shadow`. Nothing in this catalogue is active until a
# policy owner who did not author it approves the transition.
# =============================================================================
"""


def main() -> int:
    by_scenario: dict[str, list[dict]] = {}
    for rec in cat.CATALOGUE:
        by_scenario.setdefault(rec["rid"][:6], []).append(rec)

    missing = set(cat.SCENARIOS) - set(by_scenario)
    if missing:
        raise SystemExit(f"Scenarios in SCENARIOS but with no controls: {sorted(missing)}")

    written = 0
    total_controls = 0
    for scenario_id, records in sorted(by_scenario.items()):
        title, priority, starred = cat.SCENARIOS[scenario_id]
        family = scenario_id[:3]
        controls = [build_control(r) for r in sorted(records, key=lambda r: r["rid"])]
        total_controls += len(controls)
        doc = {
            "scenario": {
                "scenario_id": scenario_id,
                "title": title,
                "priority": priority,
                "added_by_coverage_audit": starred,
                "control_count": len(controls),
                "source": "Control catalogue v3.0, transcribed in full.",
            },
            "controls": controls,
        }
        out_dir = RULES_DIR / family
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"{scenario_id}.yaml"
        header = HEADER.format(
            scenario_id=scenario_id, title=title, priority=priority,
            star=" *  (added by the specification's own coverage audit)" if starred else "",
        )
        body = yaml.safe_dump(doc, sort_keys=False, allow_unicode=True, width=100, default_flow_style=False)
        path.write_text(header + body, encoding="utf-8")
        written += 1

    print(f"Wrote {written} scenario files covering {total_controls} atomic controls to {RULES_DIR}")
    if total_controls != 164:
        raise SystemExit(
            f"Expected 164 controls (Appendix C completeness statement), generated {total_controls}."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
