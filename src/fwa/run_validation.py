"""``python -m fwa.run_validation`` — the reproducible validation run.

Definition of Done §17: *"``python -m fwa.run_validation --data data/claims_demo_synthetic.csv``
regenerates every file in ``reports/`` deterministically (fixed seeds, recorded
in the report)."*

Everything is seeded from ``cfg.random_seed``, and the seed and the
parameter-registry fingerprint are printed here and written into every report,
so two runs can be compared field by field.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from .config import load_config
from .evaluation import ReportBuilder
from .pipeline import run_pipeline


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="fwa-validate",
        description="Run the full pipeline and regenerate every report in reports/.",
    )
    parser.add_argument("--data", default="data/claims_demo_synthetic.csv", help="Path to the claims extract.")
    parser.add_argument("--config", default=None, help="Path to a config/ directory.")
    parser.add_argument("--reports", default="reports", help="Output directory.")
    parser.add_argument("--adapter", default="generic_india_tpa",
                        choices=["generic_india_tpa", "shafafiya", "eclaimlink", "uae_multitable"])
    parser.add_argument("--tenant", default="T001")
    parser.add_argument("--documents", type=int, default=800,
                        help="Number of SYNTHETIC discharge summaries to generate (0 to skip).")
    parser.add_argument("--no-ai", action="store_true",
                        help="Disable the entire AI layer. The tool must remain fully "
                             "functional — that property is asserted by a governance test.")
    parser.add_argument("--enable-experimental", action="store_true",
                        help="Allow the clearly-labelled experimental model variants of §4.8.")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    config = load_config(args.config) if args.config else load_config()
    started = time.perf_counter()

    if not args.quiet:
        print("=" * 78)
        print("uae-fwa-engine — validation run")
        print("=" * 78)
        print(f"Data      : {args.data}")
        print(f"Adapter   : {args.adapter}")
        print(f"Seed      : {config.get('random_seed')}  (cfg.random_seed)")
        print(f"Parameters: fingerprint {config.fingerprint()}")
        print(f"AI layer  : {'DISABLED' if args.no_ai else 'enabled (offline provider)'}")

    result = run_pipeline(
        args.data, config=config, tenant_id=args.tenant, adapter=args.adapter,
        generate_documents=args.documents or None,
        enable_ai=False if args.no_ai else None,
        enable_experimental=args.enable_experimental,
        verbose=not args.quiet,
    )

    if not args.quiet:
        print("\nGenerating reports …")
    written = ReportBuilder(result, args.reports).build_all()

    elapsed = time.perf_counter() - started
    if not args.quiet:
        print(f"\nWrote {len(written)} file(s) to {Path(args.reports).resolve()}:")
        for path in written:
            print(f"  - {path.name}")
        # Precise rather than flattering. Every FINDING reproduces: the same
        # seed and the same parameter fingerprint give the same signals, the
        # same case identities, the same dispositions, exposures and priorities
        # — which is what §6.3's reproducibility requirement is about, and what
        # tests/integration/test_pipeline.py asserts. What does NOT reproduce
        # byte-for-byte is the wall-clock measurement: the per-control
        # ``elapsed_ms`` that Table 3.4's latency ceilings are checked against,
        # each case's ``opened_at``, and the generation timestamp. Claiming
        # those reproduce exactly would be an easy sentence to write and a
        # false one.
        print(f"\nTotal {elapsed:.1f}s. Seed {result.seed}, parameter fingerprint "
              f"{result.config.fingerprint()}.")
        print("Re-running on the same input reproduces every finding exactly — the same signals, "
              "case identities, dispositions, exposures and priorities. Measured latencies and "
              "timestamps differ between runs, as measurements do.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
