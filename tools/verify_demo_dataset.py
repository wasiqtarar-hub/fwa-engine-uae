#!/usr/bin/env python3
"""Check what the demonstration dataset actually achieves — and that it is honest.

Three things are verified, and the first is the one that matters.

**1. Labels cannot have influenced the model scores.** The dataset's labels are
derived from model scores measured on an earlier, unlabelled pass. That is only
legitimate if labels are genuinely held out of every feature path — otherwise
it is circular, and the promotion that follows means nothing. This script runs
the engine twice, once on the file as shipped and once on a copy with every
label column blanked, and asserts the model scores are IDENTICAL. If they are
not, the labels reached a feature and the whole exercise is void.

**2. Which controls fire**, so that "the engine is exercised end to end" is a
count rather than a claim.

**3. What the promotion gate says**, criterion by criterion, including what it
refuses to pass and why.

Usage::

    python tools/verify_demo_dataset.py data/claims_demo_synthetic.csv
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

warnings.filterwarnings("ignore")

from fwa.canonical.adapters import GenericIndiaTpaAdapter  # noqa: E402
from fwa.config import load_config  # noqa: E402
from fwa.evaluation import MetricSuite  # noqa: E402
from fwa.pipeline import run_pipeline  # noqa: E402

DOCUMENTS = 800


def _run(path: Path):
    return run_pipeline(path, config=load_config(), generate_documents=DOCUMENTS,
                        document_dir=ROOT / "data" / "synthetic_documents", verbose=False)


def check_labels_are_held_out(path: Path, scores: dict[str, pd.Series]) -> bool:
    """Blank every label column, re-run, and require the scores not to move."""
    frame = pd.read_csv(path)
    stripped = frame.drop(columns=[c for c in GenericIndiaTpaAdapter.LABEL_COLUMNS
                                   if c in frame.columns])
    scratch = path.with_name(path.stem + "__unlabelled.csv")
    stripped.to_csv(scratch, index=False)
    try:
        other = _run(scratch).models.provider_scores()
    finally:
        scratch.unlink(missing_ok=True)

    ok = True
    for name, series in scores.items():
        twin = other.get(name)
        if twin is None:
            print(f"  ✗ {name}: no scores on the unlabelled copy")
            ok = False
            continue
        aligned = series.align(twin, join="inner")
        drift = float((aligned[0] - aligned[1]).abs().max()) if len(aligned[0]) else float("nan")
        same_index = len(aligned[0]) == len(series) == len(twin)
        good = same_index and drift < 1e-9
        print(f"  {'✓' if good else '✗'} {name}: {len(series)} providers, "
              f"largest score difference {drift:.2e}")
        ok = ok and good
    return ok


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("path", nargs="?", default="data/claims_demo_synthetic.csv")
    ap.add_argument("--skip-holdout-check", action="store_true",
                    help="Skip the second pipeline run. Halves the runtime and removes the "
                         "only check that makes the labelling defensible.")
    args = ap.parse_args()
    path = (ROOT / args.path) if not Path(args.path).is_absolute() else Path(args.path)

    config = load_config()
    result = _run(path)
    metrics = MetricSuite(result)
    outcomes = metrics.provider_outcomes()
    capacity = int(config.get("alerts_per_reviewer_per_day")) * int(config.get("reviewer_count"))
    result.models.run_gate(outcomes=outcomes, capacity=capacity)
    scores = result.models.provider_scores()

    print(f"\n{'=' * 74}\n{path.name}\n{'=' * 74}")
    print(f"rows {len(result.claims):,} · providers {result.claims['provider_sk'].nunique():,} "
          f"· members {result.claims['member_sk'].nunique():,}")
    print(f"provider-level outcome rate {outcomes.mean():.1%} "
          f"({int(outcomes.sum())} of {len(outcomes)} providers carry a labelled claim)")

    print("\n-- 1. labels are held out of every feature path " + "-" * 27)
    if args.skip_holdout_check:
        print("  (skipped)")
        holdout_ok = None
    else:
        holdout_ok = check_labels_are_held_out(path, scores)
        print("  " + ("Labels cannot have influenced the scores."
                      if holdout_ok else
                      "LABELS REACHED A FEATURE. The calibration is circular and void."))

    print("\n-- 2. controls " + "-" * 59)
    records = pd.DataFrame([r.to_row() for r in result.evaluation.records])
    ran = records[records.outcome.isin(["TRIGGERED", "NOT_TRIGGERED"])]
    fired = ran[ran.signal_count > 0]
    print(f"  in the catalogue      {len(records):,}")
    print(f"  runnable on this file {len(ran):,}")
    print(f"  fired                 {len(fired):,}")
    print(f"  runnable but silent   {len(ran) - len(fired):,}")
    if len(ran) > len(fired):
        print(ran[ran.signal_count == 0][["rule_id", "type_label", "data_support"]]
              .to_string(index=False))
    print(f"  signals               {int(records.signal_count.sum()):,}")
    print(f"  cases                 {len(result.cases.cases):,}")

    strata = path.with_suffix(".strata.csv")
    if strata.exists():
        print("\n  rows by construction:")
        counts = pd.read_csv(strata)["stratum"].value_counts()
        for name, n in counts.items():
            print(f"    {n:6,}  {name}")

    print("\n-- 3. model promotion gate " + "-" * 47)
    promoted = []
    for name, verdict in result.models.gate_verdicts.items():
        print(f"\n  {name} — {'PROMOTED' if verdict.promoted else 'stays in shadow'}")
        for criterion in verdict.criteria:
            value = "" if criterion.value is None else f"  ({criterion.value:.4f})"
            print(f"    [{criterion.status:14s}] {criterion.name}{value}")
        if verdict.promoted:
            promoted.append(name)

    print(f"\n{'=' * 74}")
    print(f"controls fired : {len(fired)}/{len(ran)} runnable")
    print(f"models promoted: {', '.join(promoted) if promoted else 'none'}")
    if holdout_ok is False:
        print("HOLD-OUT CHECK FAILED — do not use this dataset")
        return 1
    print("=" * 74)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
