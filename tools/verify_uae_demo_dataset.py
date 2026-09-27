#!/usr/bin/env python3
"""Check the SYNTHETIC multi-table UAE demo dataset: honest, intact, reproducible — then run it.

Checks, in order (any integrity failure → non-zero exit):

1. **Reproducible** — rebuild with the recorded seed and size into a scratch
   folder and compare every file's SHA-256 with the shipped one.
2. **Complete** — every canonical and supplementary table (except
   ``review_outcome``, which is populated at runtime) is present and non-empty.
3. **Referentially intact** — every claim_line → claim_header; every claim →
   member, provider, and a coverage row active on the service date (claims an
   injector recorded in the answer key as planted are exempt).
4. **Labelled** — the SYNTHETIC notice is in the README; the answer key is not
   inside the folder or the zip.
5. **Firewalled** — a static scan: no module under ``src/`` or ``app/``
   refers to the answer key (``strata``) or ``evaluation_labels`` except
   ``fwa/canonical/uae_adapter.py`` and ``fwa/evaluation``.

Then it runs the full pipeline and prints controls by effective support,
controls that could run but raised nothing, controls that errored, the model
split and the maximum feature PSI.

Usage::

    python tools/verify_uae_demo_dataset.py                    # data/uae_demo(.zip)
    python tools/verify_uae_demo_dataset.py --skip-rebuild     # faster
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import sys
import tempfile
import time
import warnings
import zipfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

warnings.filterwarnings("ignore")

from fwa.canonical.model import CANONICAL_TABLES  # noqa: E402
from fwa.canonical.supplementary import SUPPLEMENTARY_TABLES  # noqa: E402

RUNTIME_ONLY = {"review_outcome"}
ALLOWED_LABEL_READERS = ("src/fwa/canonical/uae_adapter.py", "src/fwa/evaluation/")
_FORBIDDEN = re.compile(r"strata\.(csv|json)|\.strata\b|evaluation_labels")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_reproducible(folder: Path, zip_path: Path) -> bool:
    from tools.make_uae_demo_dataset import generate

    meta = json.loads((folder.parent / f"{folder.name}.strata.json").read_text(encoding="utf-8"))
    scratch = Path(tempfile.mkdtemp(prefix="uae_rebuild_"))
    try:
        started = time.perf_counter()
        generate(scratch / folder.name, scratch / zip_path.name, n_claims=int(meta["requested_claims"]),
                 seed=int(meta["seed"]), verbose=False)
        took = time.perf_counter() - started
        ok = True
        for f in sorted(folder.iterdir()):
            twin = scratch / folder.name / f.name
            if not twin.exists() or _sha(twin) != _sha(f):
                print(f"  ✗ {f.name} differs on rebuild")
                ok = False
        if _sha(scratch / zip_path.name) != _sha(zip_path):
            print(f"  ✗ {zip_path.name} differs on rebuild")
            ok = False
        print(f"  {'✓' if ok else '✗'} rebuild with seed {meta['seed']} is byte-identical ({took:.0f}s)")
        return ok
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


def check_complete(folder: Path) -> bool:
    ok = True
    for name in list(CANONICAL_TABLES) + list(SUPPLEMENTARY_TABLES):
        if name in RUNTIME_ONLY:
            continue
        f = folder / f"{name}.csv"
        if not f.exists():
            print(f"  ✗ {name}.csv missing")
            ok = False
            continue
        rows = sum(1 for _ in pd.read_csv(f, usecols=[0], dtype=str).itertuples())
        if rows == 0:
            print(f"  ✗ {name}.csv is empty")
            ok = False
    print(f"  {'✓' if ok else '✗'} every canonical and supplementary table present and non-empty")
    return ok


def check_integrity(folder: Path) -> bool:
    ok = True
    read = lambda n, **kw: pd.read_csv(folder / f"{n}.csv", dtype=str, keep_default_na=False, **kw)  # noqa: E731
    header = read("claim_header", usecols=["claim_sk", "member_sk", "provider_sk", "encounter_sk"])
    lines = read("claim_line", usecols=["claim_sk"])
    members = set(read("member", usecols=["member_sk"])["member_sk"])
    providers = set(read("provider", usecols=["provider_sk"])["provider_sk"])
    enc = read("encounter", usecols=["encounter_sk", "start_time", "admission_date"])
    cov = read("coverage_period", usecols=["member_sk", "valid_from", "valid_to"])
    key_path = folder.parent / f"{folder.name}.strata.csv"
    planted = set()
    if key_path.exists():
        planted = set(pd.read_csv(key_path, dtype=str, keep_default_na=False)["claim_sk"]) - {""}

    def report(label: str, bad: pd.Series | set, exempt: bool = False) -> None:
        nonlocal ok
        bad = set(bad) - (planted if exempt else set())
        mark = "✓" if not bad else "✗"
        print(f"  {mark} {label}" + (f": {len(bad)} failing (e.g. {sorted(bad)[:3]})" if bad else ""))
        ok = ok and not bad

    report("every claim_line has a claim_header", set(lines["claim_sk"]) - set(header["claim_sk"]))
    report("every claim has a member", header.loc[~header["member_sk"].isin(members), "claim_sk"], exempt=True)
    report("every claim has a provider", header.loc[~header["provider_sk"].isin(providers), "claim_sk"], exempt=True)
    e = enc.drop_duplicates("encounter_sk").set_index("encounter_sk")
    sd = pd.to_datetime(header["encounter_sk"].map(e["admission_date"]).replace("", None)).fillna(
        pd.to_datetime(header["encounter_sk"].map(e["start_time"]).replace("", None)).dt.normalize())
    report("every claim has an encounter", header.loc[sd.isna(), "claim_sk"], exempt=True)
    q = header.assign(sd=sd)[["claim_sk", "member_sk", "sd"]].merge(cov, on="member_sk", how="left")
    q["vf"] = pd.to_datetime(q["valid_from"].replace("", None))
    q["vt"] = pd.to_datetime(q["valid_to"].replace("", None))
    active = q[(q["vf"] <= q["sd"]) & (q["vt"].isna() | (q["sd"] <= q["vt"]))]["claim_sk"]
    report("every claim is inside an active coverage period (planted exceptions exempt)",
           set(header["claim_sk"]) - set(active), exempt=True)
    return ok


def check_labelled(folder: Path, zip_path: Path) -> bool:
    from tools.uae_demo.world import SYNTHETIC_NOTICE

    ok = True
    readme = folder / "README_SYNTHETIC.txt"
    has = readme.exists() and SYNTHETIC_NOTICE in readme.read_text(encoding="utf-8")
    print(f"  {'✓' if has else '✗'} README_SYNTHETIC.txt carries the SYNTHETIC notice")
    ok &= has
    inside = [f.name for f in folder.iterdir() if "strata" in f.name]
    with zipfile.ZipFile(zip_path) as zf:
        inside += [n for n in zf.namelist() if "strata" in n]
    print(f"  {'✓' if not inside else '✗'} answer key is outside the folder and the zip"
          + (f" (found {inside})" if inside else ""))
    ok &= not inside
    outside = (folder.parent / f"{folder.name}.strata.csv").exists()
    print(f"  {'✓' if outside else '✗'} answer key present beside the dataset")
    return ok and outside


def check_firewall() -> bool:
    offenders = []
    for base in (ROOT / "src", ROOT / "app"):
        for f in base.rglob("*.py"):
            rel = f.relative_to(ROOT).as_posix()
            if any(rel.startswith(a) or rel == a for a in ALLOWED_LABEL_READERS):
                continue
            text = f.read_text(encoding="utf-8", errors="ignore")
            for n, line in enumerate(text.splitlines(), start=1):
                if _FORBIDDEN.search(line):
                    offenders.append(f"{rel}:{n}: {line.strip()[:90]}")
    print(f"  {'✓' if not offenders else '✗'} no detection or app module reads the answer key or evaluation labels")
    for o in offenders[:10]:
        print(f"      {o}")
    return not offenders


def run_engine(zip_path: Path) -> None:
    from fwa.pipeline import run_pipeline

    started = time.perf_counter()
    doc_dir = Path(tempfile.mkdtemp(prefix="uae_docs_"))
    try:
        result = run_pipeline(zip_path, adapter="uae_multitable", generate_documents=200, verbose=False,
                              document_dir=doc_dir)
    finally:
        shutil.rmtree(doc_dir, ignore_errors=True)
    took = time.perf_counter() - started
    records = pd.DataFrame([{"rule_id": r.rule_id, "outcome": r.outcome, "data_support": r.data_support,
                             "support_basis": getattr(r, "support_basis", "catalogue"),
                             "signals": r.signal_count} for r in result.evaluation.records])
    print(f"  pipeline ran in {took:.0f}s on {len(result.claims):,} claims")
    print("\n  controls by effective support (data_support × support_basis):")
    tab = records.groupby(["data_support", "support_basis"]).size()
    for (support, basis), n in tab.items():
        print(f"    {n:4d}  {support:<34s} {basis}")
    ran = records[~records["outcome"].isin(["NOT_EXECUTABLE_ON_THIS_DATASET", "NOT_EFFECTIVE"])]
    silent = ran[ran["signals"] == 0]
    print(f"\n  controls that could run: {len(ran)} of {len(records)}; raised signals: {int((ran['signals'] > 0).sum())}")
    print(f"  could run but raised nothing ({len(silent)}): " + ", ".join(silent["rule_id"].tolist()))
    errors = result.evaluation.errors
    print(f"  errored ({len(errors)}): " + ("; ".join(f"{rid}: {str(e)[:80]}" for rid, e in errors[:15]) or "none"))
    layer = result.models
    if layer is not None and layer.anomaly is not None and layer.anomaly.models:
        cov = layer.anomaly.coverage()
        print(f"\n  model split: train {cov['train_claims']:,} claims / {cov['train_providers']} providers; "
              f"score {cov['score_claims']:,} claims / {cov['score_providers']} providers; "
              f"excluded {cov['excluded_claims']:,} of {cov['total_claims']:,}; split date {cov['split_date']}")
        for name, m in layer.anomaly.models.items():
            print(f"    {name}: trained on {m.training_rows:,}, scored {len(m.scores):,}, flagged {len(m.flagged_index):,}")
        worst = []
        for name, results in layer.drift.items():
            for d in results:
                if "feature PSI" in d.monitor:
                    worst.append((name, d.value, d.detail.head(5) if d.detail is not None else None))
        for name, value, detail in worst[:1]:
            print(f"  max feature PSI: {value:.3f}")
            if detail is not None and not detail.empty:
                print("    top: " + ", ".join(f"{r.feature} {r.psi}" for r in detail.itertuples()))
    else:
        print("  models: not trained — " + " ".join(getattr(layer, "notes", []) or []))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--folder", default="data/uae_demo")
    ap.add_argument("--zip", default="data/uae_demo.zip")
    ap.add_argument("--skip-rebuild", action="store_true", help="skip the reproducibility rebuild")
    ap.add_argument("--skip-pipeline", action="store_true", help="integrity checks only")
    args = ap.parse_args()
    folder = Path(args.folder) if Path(args.folder).is_absolute() else ROOT / args.folder
    zip_path = Path(args.zip) if Path(args.zip).is_absolute() else ROOT / args.zip
    if not folder.exists() or not zip_path.exists():
        print(f"Dataset not found ({folder}, {zip_path}). Build it: python tools/make_uae_demo_dataset.py")
        return 2
    print(f"{'=' * 74}\n{folder.name} — SYNTHETIC multi-table UAE demo\n{'=' * 74}")
    ok = True
    print("\n-- 1. reproducible " + "-" * 55)
    if args.skip_rebuild:
        print("  (skipped)")
    else:
        ok &= check_reproducible(folder, zip_path)
    print("\n-- 2. complete " + "-" * 59)
    ok &= check_complete(folder)
    print("\n-- 3. referential integrity " + "-" * 46)
    ok &= check_integrity(folder)
    print("\n-- 4. labelled as SYNTHETIC, answer key outside " + "-" * 25)
    ok &= check_labelled(folder, zip_path)
    print("\n-- 5. label firewall (static scan) " + "-" * 39)
    ok &= check_firewall()
    if not args.skip_pipeline:
        print("\n-- 6. the engine on this dataset " + "-" * 41)
        run_engine(zip_path)
    print(f"\n{'=' * 74}\n{'ALL INTEGRITY CHECKS PASSED' if ok else 'INTEGRITY FAILURES — see above'}\n{'=' * 74}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
