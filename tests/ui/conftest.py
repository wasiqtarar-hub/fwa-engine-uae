"""UI tests run against a throwaway database and preferences file.

The app's own database (``data/fwa.db``) holds a user's changed passwords and
recorded decisions; a test must never touch it. Both paths are redirected
before any app module is imported.
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import pytest

_TMP = Path(tempfile.mkdtemp(prefix="fwa-ui-tests-"))
os.environ.setdefault("FWA_DATABASE_URL", f"sqlite:///{(_TMP / 'fwa.db').as_posix()}")
os.environ.setdefault("FWA_UI_PREFS", str(_TMP / "ui_prefs.json"))

ROOT = Path(__file__).resolve().parents[2]
for p in (str(ROOT / "src"), str(ROOT / "app")):
    if p not in sys.path:
        sys.path.insert(0, p)

#: The three datasets every page must survive: the new multi-table UAE demo
#: (the app's default), the claim-header synthetic demo, and the sample extract.
DATASETS = {
    "uae_demo": ROOT / "data" / "uae_demo.zip",
    "claims_demo_synthetic": ROOT / "data" / "claims_demo_synthetic.csv",
    "claims": ROOT / "data" / "claims.csv",
}


def dataset_spec(key: str):
    import dataset as ds

    path = DATASETS[key]
    if not path.exists():
        pytest.skip(f"{path.name} has not been generated yet")
    if path.suffix == ".zip" or path.is_dir():
        return ds.spec_from_path(str(path), "uae_multitable")
    return ds.spec_from_path(str(path), "generic_india_tpa")
