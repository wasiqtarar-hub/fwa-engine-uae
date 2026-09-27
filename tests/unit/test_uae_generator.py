"""The SYNTHETIC multi-table UAE demo generator: reproducible, labelled, answer key kept outside.

Built at ~1,500 claims with the injectors switched off, so the test exercises the
reference tables, the clean base and the writer — the parts this generator owns —
and stays fast and independent of injector work in progress.
"""

from __future__ import annotations

import hashlib
import json
import sys
import zipfile
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT, ROOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from tools.make_uae_demo_dataset import KEY_SENTENCE, LABELS_FILE, generate  # noqa: E402
from tools.uae_demo.world import SYNTHETIC_NOTICE  # noqa: E402

N_CLAIMS = 1500
SEED = 4242


@pytest.fixture(scope="module")
def two_builds(tmp_path_factory):
    a = tmp_path_factory.mktemp("uae_a")
    b = tmp_path_factory.mktemp("uae_b")
    sa = generate(a / "uae", a / "uae.zip", n_claims=N_CLAIMS, seed=SEED, inject=False, verbose=False)
    sb = generate(b / "uae", b / "uae.zip", n_claims=N_CLAIMS, seed=SEED, inject=False, verbose=False)
    return (a, sa), (b, sb)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_two_builds_are_byte_identical(two_builds):
    (a, _), (b, _) = two_builds
    files_a = sorted(f.name for f in (a / "uae").iterdir())
    files_b = sorted(f.name for f in (b / "uae").iterdir())
    assert files_a == files_b and len(files_a) > 50
    for name in files_a:
        assert _sha(a / "uae" / name) == _sha(b / "uae" / name), name
    assert _sha(a / "uae.zip") == _sha(b / "uae.zip")


def test_build_is_small_and_plausible(two_builds):
    (a, summary), _ = two_builds
    header = pd.read_csv(a / "uae" / "claim_header.csv", dtype=str)
    assert 0.8 * N_CLAIMS <= len(header) <= 1.2 * N_CLAIMS
    assert set(header["claim_type"]) == {"OUTPATIENT", "PHARMACY", "LAB", "RADIOLOGY", "INPATIENT"}
    assert header["source_currency"].eq("AED").all()
    assert summary["timings"]["total"] < 60


def test_synthetic_notice_everywhere_it_matters(two_builds):
    (a, _), _ = two_builds
    assert SYNTHETIC_NOTICE in (a / "uae" / "README_SYNTHETIC.txt").read_text(encoding="utf-8")
    docs = pd.read_csv(a / "uae" / "document.csv", dtype=str)
    assert len(docs) > 0
    assert docs["text"].str.contains("SYNTHETIC").all()
    assert docs["attachment_ref"].str.startswith("SYNTHETIC_").all()
    assert docs["is_synthetic"].eq("True").all()
    names = " ".join(pd.read_csv(a / "uae" / "provider.csv", dtype=str).columns)
    assert "name" not in names  # no provider names at all, fictional or otherwise


def test_answer_key_is_written_outside_the_dataset(two_builds):
    (a, _), _ = two_builds
    assert (a / "uae.strata.csv").exists() and (a / "uae.strata.json").exists()
    assert not any("strata" in f.name for f in (a / "uae").iterdir())
    with zipfile.ZipFile(a / "uae.zip") as zf:
        names = zf.namelist()
    assert not any("strata" in n for n in names)
    assert LABELS_FILE in names and "claim_header.csv" in names
    meta = json.loads((a / "uae.strata.json").read_text(encoding="utf-8"))
    assert KEY_SENTENCE in meta["warning"] and "SYNTHETIC" in meta["warning"]
    assert meta["seed"] == SEED


def test_evaluation_labels_cover_every_claim(two_builds):
    (a, _), _ = two_builds
    labels = pd.read_csv(a / "uae" / LABELS_FILE, dtype={"claim_sk": str})
    header = pd.read_csv(a / "uae" / "claim_header.csv", dtype=str)
    assert list(labels.columns) == ["claim_sk", "fraud_label", "fraud_type", "fraud_confidence",
                                    "ground_truth_source"]
    assert set(labels["claim_sk"]) == set(header["claim_sk"])
    # the clean base plants nothing
    assert labels["fraud_label"].eq(0).all()
    assert labels["ground_truth_source"].eq("synthetic_injection").all()
