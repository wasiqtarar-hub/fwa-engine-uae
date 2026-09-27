#!/usr/bin/env python3
"""Build the SYNTHETIC multi-table UAE demo dataset.

Runs, in a fixed order, from one seeded ``numpy.random.Generator``:

1. ``tools/uae_demo/reference.py`` — every reference table;
2. ``tools/uae_demo/base.py`` — a clean, plausible population and the claim factory;
3. each injector that exists, in this order: ``inject_ent``, ``inject_pol``,
   ``inject_pay_a``, ``inject_pay_b``, ``inject_cln``, ``inject_phr``,
   ``inject_doc``, ``inject_net``, ``inject_anl``. A missing module is skipped
   with a note; an injector that raises fails the build.

Then writes, into ``--out`` (a folder), one ``<table>.csv`` per canonical and
supplementary table, ``README_SYNTHETIC.txt`` and ``evaluation_labels.csv``
(held-out labels — read only by ``UaeMultiTableAdapter.held_out_labels``);
the same files into ``--zip``; and the ANSWER KEY *outside* the folder:
``<out>.strata.csv`` and ``<out>.strata.json``.

Nothing the adapter derives is written (no ``service_date``, ``agent_id``,
``approved_amount`` … on ``claim_header``): the files are source-style tables.

Deterministic: the same seed gives byte-identical CSVs and zip.

Usage::

    python tools/make_uae_demo_dataset.py --out data/uae_demo --zip data/uae_demo.zip
    python tools/make_uae_demo_dataset.py --claims 1500 --out /tmp/uae_small --zip /tmp/uae_small.zip
"""

from __future__ import annotations

import argparse
import datetime as _dt
import importlib
import io
import json
import sys
import time
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from fwa.canonical.model import CANONICAL_TABLES  # noqa: E402
from fwa.canonical.supplementary import SUPPLEMENTARY_TABLES  # noqa: E402
from tools.uae_demo import base, reference  # noqa: E402
from tools.uae_demo.world import SEED, SYNTHETIC_NOTICE, World  # noqa: E402

INJECTORS = ("inject_ent", "inject_pol", "inject_pay_a", "inject_pay_b", "inject_cln", "inject_phr",
             "inject_doc", "inject_net", "inject_anl")
ALL_SPECS = {**CANONICAL_TABLES, **SUPPLEMENTARY_TABLES}
LABELS_FILE = "evaluation_labels.csv"
README_FILE = "README_SYNTHETIC.txt"
KEY_SENTENCE = ("A check that passes on data built to pass it is a test of the check, "
                "not evidence that it detects fraud.")
_FIXED_ZIP_TIME = (2026, 1, 1, 0, 0, 0)


# ----------------------------------------------------------------------------- build


def build_world(n_claims: int = base.DEFAULT_CLAIMS, seed: int = SEED, *, inject: bool = True,
                verbose: bool = True) -> tuple[World, dict[str, Any]]:
    """Reference → base → injectors. Returns the world and a run log."""
    log: dict[str, Any] = {"timings": {}, "injectors": {}}
    world = World(rng=np.random.default_rng(seed))
    t = time.perf_counter()
    reference.build_reference(world)
    log["timings"]["reference"] = time.perf_counter() - t
    t = time.perf_counter()
    base.build(world, n_claims, verbose=verbose)
    log["timings"]["base"] = time.perf_counter() - t
    for name in INJECTORS:
        path = ROOT / "tools" / "uae_demo" / f"{name}.py"
        if not inject:
            log["injectors"][name] = "skipped (--no-inject)"
            continue
        if not path.exists():
            log["injectors"][name] = "not present"
            if verbose:
                print(f"  · {name}: module not present — skipped")
            continue
        module = importlib.import_module(f"tools.uae_demo.{name}")
        if not hasattr(module, "inject"):
            raise RuntimeError(f"tools/uae_demo/{name}.py has no inject(world) function.")
        t = time.perf_counter()
        before = len(world.strata)
        try:
            module.inject(world)
        except Exception as exc:  # fail loudly, naming the injector
            raise RuntimeError(f"Injector {name} failed: {exc!r}") from exc
        log["timings"][name] = time.perf_counter() - t
        log["injectors"][name] = f"{len(world.strata) - before} answer-key rows"
        if verbose:
            print(f"  ✓ {name}: {len(world.strata) - before} answer-key rows ({log['timings'][name]:.1f}s)")
    return world, log


# ----------------------------------------------------------------------------- write


def _format(value: Any, hint: str | None = None) -> Any:
    """One cell as written. Dates as YYYY-MM-DD; datetimes always with a time (a column never mixes)."""
    if value is None:
        return None
    if isinstance(value, float) and np.isnan(value):
        return None
    if isinstance(value, (pd.Timestamp, _dt.datetime)):
        if pd.isna(value):
            return None
        return value.strftime("%Y-%m-%d") if hint == "date" else value.strftime("%Y-%m-%d %H:%M:%S")
    if isinstance(value, _dt.date):
        return value.isoformat() if hint != "datetime" else value.strftime("%Y-%m-%d 00:00:00")
    return value


def _sort_keys(name: str, frame: pd.DataFrame) -> list[str]:
    spec = ALL_SPECS.get(name)
    keys: list[str] = []
    if spec is not None:
        pk = spec.primary_key.strip("()")
        keys = [k.strip() for k in pk.split(",") if k.strip() in frame.columns]
    return keys or [c for c in frame.columns[:3]]


def table_csv(name: str, frame: pd.DataFrame) -> bytes:
    """One table as deterministic CSV bytes: spec columns first, extras after, rows sorted."""
    spec = ALL_SPECS.get(name)
    # only columns the generator actually produced: nothing the adapter derives is written
    cols = [c for c in spec.columns if c in frame.columns] if spec is not None else []
    extras = sorted(c for c in frame.columns if c not in cols and not str(c).startswith("_"))
    out = frame.reindex(columns=cols + extras).copy()
    hints = spec.columns if spec is not None else {}
    for c in out.columns:
        hint = hints.get(c)
        is_dt = str(out[c].dtype).startswith("datetime")
        if hint in ("date", "datetime") or is_dt:
            conv = pd.to_datetime(out[c], errors="coerce")
            fmt = "%Y-%m-%d" if hint == "date" else "%Y-%m-%d %H:%M:%S"
            text = conv.dt.strftime(fmt)
            out[c] = text.where(conv.notna(), None)
        elif out[c].dtype == object and out[c].map(lambda v: isinstance(v, (_dt.date, pd.Timestamp))).any():
            out[c] = [_format(v, hint) for v in out[c].tolist()]
    keys = _sort_keys(name, out)
    order = out[keys].astype(str).fillna("")
    out = out.iloc[np.lexsort([order[k].to_numpy() for k in reversed(keys)])] if len(out) else out
    buf = io.StringIO()
    out.to_csv(buf, index=False, lineterminator="\n")
    return buf.getvalue().encode("utf-8")


def evaluation_labels(world: World) -> pd.DataFrame:
    claims = sorted(world.tables["claim_header"]["claim_sk"].astype(str))
    stratum: dict[str, str] = {}
    for row in world.strata:
        c = row.get("claim_sk")
        if c and c in world.positive_claims and c not in stratum:
            stratum[c] = row["stratum"]
    return pd.DataFrame({
        "claim_sk": claims,
        "fraud_label": [1 if c in world.positive_claims else 0 for c in claims],
        "fraud_type": [stratum.get(c, "legitimate") if c in world.positive_claims else "legitimate" for c in claims],
        "fraud_confidence": 1.0,
        "ground_truth_source": "synthetic_injection",
    })


def readme(world: World, seed: int, n_claims: int, files: dict[str, int]) -> str:
    h = world.tables["claim_header"]
    mix = h["claim_type"].value_counts(normalize=True)
    lines = [
        "SYNTHETIC UAE MULTI-TABLE DEMO DATASET",
        "=" * 60,
        "",
        SYNTHETIC_NOTICE,
        "",
        "What this is",
        "------------",
        "A clearly fictional, AED-denominated, multi-table claims dataset modelled on UAE",
        "claim constructs (claim header and activity lines, diagnoses, encounters, prior",
        "authorisations, remittance, resubmissions, prescriptions and dispensing, clinician",
        "rosters, licence and network periods, documents, referrals, inventories and the",
        "policy/coding reference tables). Every payer, TPA, provider, clinician, member,",
        "employer and agent is invented; identifiers and tokens are random. Documents are",
        "template-generated and marked SYNTHETIC in their text and attachment names.",
        "",
        "Load it with the adapter 'uae_multitable' (fwa.canonical.uae_adapter.UaeMultiTableAdapter).",
        "NOT CERTIFIED — never run against a live regulator feed.",
        "",
        f"Generator: tools/make_uae_demo_dataset.py   seed: {seed}   requested claims: {n_claims:,}",
        f"Period: {world.start.isoformat()} to {world.end.isoformat()}   currency: AED",
        f"Claims: {len(h):,}   mix: " + ", ".join(f"{k} {v:.1%}" for k, v in mix.items()),
        "",
        f"{LABELS_FILE} holds HELD-OUT evaluation labels (one row per claim). They are for",
        "evaluation only and are never loaded into a canonical table or a feature.",
        "The answer key (which construction produced which claim) is deliberately NOT in",
        "this folder or zip; it sits beside it as <name>.strata.csv / .strata.json.",
        "",
        "Files",
        "-----",
    ]
    lines += [f"  {name:<40s} {rows:>9,} rows" for name, rows in sorted(files.items())]
    return "\n".join(lines) + "\n"


def write_outputs(world: World, out_dir: Path, zip_path: Path | None, *, seed: int, n_claims: int,
                  log: dict[str, Any] | None = None) -> dict[str, Any]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for stale in out_dir.glob("*.csv"):
        stale.unlink()
    payload: dict[str, bytes] = {}
    counts: dict[str, int] = {}
    unknown = sorted(n for n in world.tables if n not in ALL_SPECS)
    for name in sorted(world.tables):
        if name not in ALL_SPECS:
            continue
        frame = world.tables[name]
        payload[f"{name}.csv"] = table_csv(name, frame)
        counts[f"{name}.csv"] = len(frame)
    labels = evaluation_labels(world)
    buf = io.StringIO()
    labels.to_csv(buf, index=False, lineterminator="\n")
    payload[LABELS_FILE] = buf.getvalue().encode("utf-8")
    counts[LABELS_FILE] = len(labels)
    payload[README_FILE] = readme(world, seed, n_claims, counts).encode("utf-8")
    for fname, data in payload.items():
        (out_dir / fname).write_bytes(data)
    if zip_path is not None:
        zip_path = Path(zip_path)
        zip_path.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for fname in sorted(payload):
                info = zipfile.ZipInfo(fname, date_time=_FIXED_ZIP_TIME)
                info.compress_type = zipfile.ZIP_DEFLATED
                info.external_attr = 0o644 << 16
                zf.writestr(info, payload[fname])
    # ------------------------------------------------------------- answer key (OUTSIDE the folder)
    key_csv = out_dir.parent / f"{out_dir.name}.strata.csv"
    key_json = out_dir.parent / f"{out_dir.name}.strata.json"
    strata = pd.DataFrame(world.strata, columns=["stratum", "family", "rule_ids", "claim_sk", "subject_type",
                                                 "subject_ids", "note"])
    strata.to_csv(key_csv, index=False, lineterminator="\n")
    per_stratum = strata.groupby("stratum").agg(rows=("stratum", "size"),
                                                claims=("claim_sk", lambda s: int((s.astype(str) != "").sum())))
    key = {
        "warning": "SYNTHETIC ANSWER KEY. " + KEY_SENTENCE + " Never read by detection code.",
        "generator": "tools/make_uae_demo_dataset.py",
        "seed": seed,
        "requested_claims": n_claims,
        "claims": int(len(world.tables["claim_header"])),
        "positive_claims": int(len(world.positive_claims)),
        "strata": {k: {"rows": int(v.rows), "claims": int(v.claims)} for k, v in per_stratum.iterrows()},
        "injectors": (log or {}).get("injectors", {}),
        "notes": list(world.notes),
    }
    key_json.write_text(json.dumps(key, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"counts": counts, "unknown_tables": unknown, "answer_key": [str(key_csv), str(key_json)],
            "out_dir": str(out_dir), "zip": str(zip_path) if zip_path else None}


def generate(out_dir: Path | str, zip_path: Path | str | None = None, *, n_claims: int = base.DEFAULT_CLAIMS,
             seed: int = SEED, inject: bool = True, verbose: bool = True) -> dict[str, Any]:
    """Build and write everything. Returns a summary dict (also used by the tests)."""
    started = time.perf_counter()
    world, log = build_world(n_claims, seed, inject=inject, verbose=verbose)
    t = time.perf_counter()
    summary = write_outputs(world, Path(out_dir), Path(zip_path) if zip_path else None, seed=seed,
                            n_claims=n_claims, log=log)
    log["timings"]["write"] = time.perf_counter() - t
    log["timings"]["total"] = time.perf_counter() - started
    summary.update(log)
    summary["claim_mix"] = world.tables["claim_header"]["claim_type"].value_counts().to_dict()
    summary["positives"] = len(world.positive_claims)
    return summary


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="data/uae_demo", help="output folder (one CSV per table)")
    ap.add_argument("--zip", default="data/uae_demo.zip", help="zip of the same files ('' to skip)")
    ap.add_argument("--claims", type=int, default=base.DEFAULT_CLAIMS, help="approximate number of claims")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--no-inject", action="store_true", help="build the clean base only")
    args = ap.parse_args()
    out = Path(args.out) if Path(args.out).is_absolute() else ROOT / args.out
    zp = None if not args.zip else (Path(args.zip) if Path(args.zip).is_absolute() else ROOT / args.zip)
    print(f"Building the SYNTHETIC UAE demo dataset — {args.claims:,} claims, seed {args.seed}")
    s = generate(out, zp, n_claims=args.claims, seed=args.seed, inject=not args.no_inject)
    print(f"\nWrote {len(s['counts'])} files to {s['out_dir']}" + (f" and {s['zip']}" if s["zip"] else ""))
    for name, rows in sorted(s["counts"].items()):
        print(f"  {name:<40s} {rows:>9,}")
    if s["unknown_tables"]:
        print("  (not written — not a canonical or supplementary table: " + ", ".join(s["unknown_tables"]) + ")")
    print("claim mix: " + ", ".join(f"{k} {v:,}" for k, v in sorted(s["claim_mix"].items())))
    print(f"positive (planted) claims: {s['positives']:,}")
    print("answer key (outside the dataset): " + " , ".join(s["answer_key"]))
    print("timings: " + ", ".join(f"{k} {v:.1f}s" for k, v in s["timings"].items()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
