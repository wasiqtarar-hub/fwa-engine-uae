"""The active dataset: which file the engine is currently running on.

Every number in this application is a property of one input file. Before this
module existed that file was a constant, and the interface could get away with
describing it in prose. It is now a choice, which changes three things:

1. **The pipeline cache has to be keyed on it.** ``st.cache_resource`` keys on
   the arguments, so the run is memoised per dataset rather than per process —
   switch back to a file you loaded earlier and its results are still there,
   with no recomputation.
2. **Every page has to say which file it is describing.** A screenshot of a
   queue that does not name its dataset is evidence of nothing.
3. **Switching should be legible.** Loading a new file changes which controls
   can run at all, so :func:`compare_runs` reports what became runnable, what
   stopped being runnable, and what found something different.

An uploaded file is written under ``data/uploads/`` with a content-addressed
name. That is not tidiness: it means two uploads of the same file resolve to
the same cache key and the same run, and it means the name a user sees can be
their original filename while the identity the cache uses is the content.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
UPLOAD_DIR = ROOT / "data" / "uploads"
#: The dataset a fresh session opens on. It is the synthetic demonstration
#: file, because it is the one that exercises every runnable control and the
#: promotion gate in its passing state (``docs/DEMO_DATASET.md``). It is
#: clearly named ``_synthetic`` and carries its own warning banner: a gate
#: passed on a file built to pass it is a test of the gate, not evidence about
#: detection performance. Any other file can be loaded from the Data page.
DEFAULT_PATH = ROOT / "data" / "claims_demo_synthetic.csv"

SESSION_KEY = "fwa_dataset"
HISTORY_KEY = "fwa_dataset_history"

__all__ = [
    "DatasetSpec", "active_dataset", "set_active_dataset", "default_spec",
    "spec_from_upload", "spec_from_path", "loaded_datasets", "compare_runs",
    "model_summary", "model_parameters", "outcome_label", "ADAPTER_LABELS",
]

#: The adapters a user may pick, with what each one expects. The two UAE
#: adapters are offered because they exist and are tested; the label says
#: plainly that neither has been run against a live regulator feed.
ADAPTER_LABELS: dict[str, str] = {
    "generic_india_tpa": "Generic TPA — one row per claim header (CSV)",
    "shafafiya": "Shafafiya (DoH) — not certified against a live feed",
    "eclaimlink": "eClaimLink (DHA) — not certified against a live feed",
}


@dataclass(frozen=True)
class DatasetSpec:
    """One input file, identified by its content rather than its name."""

    name: str
    path: Path
    adapter: str = "generic_india_tpa"
    sha256: str = ""
    rows: int = 0
    columns: tuple[str, ...] = ()
    loaded_at: _dt.datetime = field(default_factory=lambda: _dt.datetime.now())
    origin: str = "shipped"          # shipped | upload | path

    @property
    def key(self) -> str:
        """What the pipeline cache keys on.

        The content hash, not the path: the same file uploaded twice under two
        names is one run, and a file edited in place under an unchanged name is
        correctly a different one.
        """
        return f"{self.sha256[:16]}:{self.adapter}"

    @property
    def short_sha(self) -> str:
        return self.sha256[:12] if self.sha256 else "—"

    def meta_line(self) -> str:
        """Everything except the name, for the persistent dataset strip."""
        stamp = self.loaded_at.strftime("%d %b, %H:%M")
        rows = f"{self.rows:,} claims" if self.rows else "row count pending"
        return f"{rows} · {_adapter_short(self.adapter)} adapter · sha256 {self.short_sha} · loaded {stamp}"

    def line(self) -> str:
        """The one-line description, name included."""
        return f"{self.name} · {self.meta_line()}"

    def to_row(self) -> dict[str, Any]:
        return {
            "dataset": self.name,
            "rows": self.rows,
            "columns": len(self.columns),
            "adapter": self.adapter,
            "sha256": self.short_sha,
            "loaded_at": self.loaded_at.strftime("%Y-%m-%d %H:%M:%S"),
            "origin": self.origin,
        }


def _adapter_short(adapter: str) -> str:
    return {
        "generic_india_tpa": "Generic TPA",
        "shafafiya": "Shafafiya",
        "eclaimlink": "eClaimLink",
    }.get(adapter, adapter)


# ---------------------------------------------------------------------------
# building a spec
# ---------------------------------------------------------------------------


def _profile(path: Path) -> tuple[str, int, tuple[str, ...]]:
    """Hash the file and read its shape, without loading it twice.

    The hash is taken over the bytes on disk, before any parsing, for the same
    reason the engine hashes each raw record before parsing it: the identity of
    an input should not depend on how successfully it was interpreted.
    """
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    header = pd.read_csv(path, nrows=0)
    with path.open("rb") as handle:
        rows = max(sum(1 for _ in handle) - 1, 0)
    return digest.hexdigest(), rows, tuple(header.columns)


def default_spec() -> DatasetSpec:
    """The dataset the application starts on."""
    sha, rows, columns = _profile(DEFAULT_PATH)
    return DatasetSpec(
        name=DEFAULT_PATH.name, path=DEFAULT_PATH, sha256=sha, rows=rows,
        columns=columns, origin="shipped",
    )


def _safe_stem(name: str) -> str:
    stem = Path(name).stem
    return re.sub(r"[^A-Za-z0-9._-]+", "-", stem)[:60] or "dataset"


def spec_from_upload(uploaded, adapter: str) -> DatasetSpec:
    """Persist an uploaded file and describe it.

    Streamlit hands us the bytes in memory; the pipeline wants a path, and the
    cache wants a stable identity. Writing the bytes under a content-addressed
    name satisfies both and makes a re-upload of the same file free.
    """
    payload = uploaded.getvalue()
    sha = hashlib.sha256(payload).hexdigest()
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    path = UPLOAD_DIR / f"{_safe_stem(uploaded.name)}-{sha[:12]}.csv"
    if not path.exists():
        path.write_bytes(payload)
    _, rows, columns = _profile(path)
    return DatasetSpec(
        name=uploaded.name, path=path, adapter=adapter, sha256=sha, rows=rows,
        columns=columns, origin="upload",
    )


def spec_from_path(raw: str, adapter: str) -> DatasetSpec:
    """Describe a file already on this machine, without copying it.

    A path avoids pushing a large extract through the browser. It is also the
    only route that can reach a file the browser is not allowed to read, which
    is the common case for anything sitting on a work share.
    """
    path = Path(raw.strip().strip('"').strip("'")).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"No file at {path}")
    if not path.is_file():
        raise ValueError(f"{path} is a directory, not a file")
    if path.suffix.lower() not in (".csv", ".txt", ".tsv"):
        raise ValueError(f"{path.suffix or 'That file'} is not a delimited text file")
    sha, rows, columns = _profile(path)
    return DatasetSpec(
        name=path.name, path=path, adapter=adapter, sha256=sha, rows=rows,
        columns=columns, origin="path",
    )


# ---------------------------------------------------------------------------
# session state
# ---------------------------------------------------------------------------


def active_dataset() -> DatasetSpec:
    spec = st.session_state.get(SESSION_KEY)
    if spec is None:
        spec = default_spec()
        st.session_state[SESSION_KEY] = spec
        _remember(spec)
    return spec


def set_active_dataset(spec: DatasetSpec) -> None:
    st.session_state[SESSION_KEY] = spec
    _remember(spec)


def _remember(spec: DatasetSpec) -> None:
    history: list[DatasetSpec] = st.session_state.setdefault(HISTORY_KEY, [])
    if not any(s.key == spec.key for s in history):
        history.append(spec)


def loaded_datasets() -> list[DatasetSpec]:
    return list(st.session_state.get(HISTORY_KEY, []))


# ---------------------------------------------------------------------------
# what changed between two runs
# ---------------------------------------------------------------------------

#: A control's fate on one dataset, in words a reader does not have to decode.
_OUTCOME_LABEL = {
    "TRIGGERED": "Ran and found something",
    "NOT_TRIGGERED": "Ran, found nothing",
    "NOT_EXECUTABLE_ON_THIS_DATASET": "Cannot run — required fields absent",
    "NOT_EFFECTIVE": "Not in force on the run date",
    "KILL_SWITCHED": "Suspended by kill switch",
    "NULL_INPUT": "Stopped on a null input",
    "INSUFFICIENT_PEER_EVIDENCE": "Ran, but no peer group was large enough",
}


def outcome_label(outcome: str) -> str:
    return _OUTCOME_LABEL.get(outcome, outcome.replace("_", " ").capitalize())


def compare_runs(current, previous=None) -> pd.DataFrame:
    """Per-control comparison of this run against the one before it.

    Loading a new file is not just new numbers: a dataset that carries an
    authorization table makes a whole scenario family runnable that was dead
    before, and a dataset missing ``agent_id`` kills the distribution controls.
    That is the most interesting thing about switching datasets, so it is
    reported first-class rather than left for the reader to diff by eye.
    """
    now = {r.rule_id: r for r in current.evaluation.records}
    before = {r.rule_id: r for r in previous.evaluation.records} if previous is not None else {}

    rows = []
    for rule_id, record in sorted(now.items()):
        prior = before.get(rule_id)
        ran_now = record.outcome not in ("NOT_EXECUTABLE_ON_THIS_DATASET", "NOT_EFFECTIVE")
        ran_before = (
            prior is not None
            and prior.outcome not in ("NOT_EXECUTABLE_ON_THIS_DATASET", "NOT_EFFECTIVE")
        )
        if prior is None:
            change = "—"
        elif ran_now and not ran_before:
            change = "Newly runnable"
        elif ran_before and not ran_now:
            change = "No longer runnable"
        elif record.signal_count != prior.signal_count:
            delta = record.signal_count - prior.signal_count
            change = f"{delta:+d} signals"
        else:
            change = "Unchanged"

        rows.append({
            "rule_id": rule_id,
            "scenario": record.scenario_id,
            "type": record.type_label,
            "status on this dataset": outcome_label(record.outcome),
            "data support": record.data_support,
            "signals": record.signal_count,
            "previous": "—" if prior is None else prior.signal_count,
            "change": change,
        })
    return pd.DataFrame(rows)


def model_summary(result) -> pd.DataFrame:
    """Which unsupervised models were refitted on this dataset, and to what end.

    Deliberately reports the fit and the gate as two separate facts. A model
    that trained is not a model that may be used: promotion requires
    demonstrated prospective lift over the transparent composite plus
    calibration, drift and explanation-quality checks, and collapsing "it
    trained" into "it works" is the conflation the gate exists to prevent.
    """
    layer = getattr(result, "models", None)
    columns = ["model", "trained on", "flagged", "gate verdict", "what happened"]
    if layer is None or layer.anomaly is None or not layer.anomaly.models:
        return pd.DataFrame(
            [{"model": "—", "trained on": "—", "flagged": "—", "gate verdict": "not run",
              "what happened": "No unsupervised model could be trained on this dataset. "
                               + " ".join(getattr(layer, "notes", []) or [])}],
            columns=columns,
        )

    rows = []
    for name, scores in layer.anomaly.models.items():
        verdict = (layer.gate_verdicts or {}).get(name)
        if verdict is None:
            gate, story = "not yet run", "Refitted on this dataset. The promotion gate has not " \
                                         "been run — open the Models page to run it."
        else:
            failed = [c.name for c in verdict.criteria if c.status == "FAIL"]
            unknown = [c.name for c in verdict.criteria if c.status == "NOT_ASSESSABLE"]
            gate = "PROMOTED" if verdict.promoted else "SHADOW"
            story = "Refitted on this dataset. " + (
                f"Failed: {'; '.join(failed)}. " if failed else ""
            ) + (
                f"Not assessable: {'; '.join(unknown)}. " if unknown else ""
            ) + ("" if failed or unknown else "Every gate criterion passed. ")
        rows.append({
            "model": name,
            "trained on": f"{scores.training_rows:,} claims",
            "flagged": f"{len(scores.flagged_index):,}",
            "gate verdict": gate,
            "what happened": story.strip(),
        })
    return pd.DataFrame(rows, columns=columns)


def model_parameters(result) -> pd.DataFrame:
    """The hyperparameters each model was actually fitted with on this run."""
    layer = getattr(result, "models", None)
    if layer is None or layer.anomaly is None or not layer.anomaly.models:
        return pd.DataFrame(columns=["model", "version", "parameter", "value"])
    rows = []
    for name, scores in layer.anomaly.models.items():
        for key, value in sorted((scores.params or {}).items()):
            rows.append({"model": name, "version": scores.version,
                         "parameter": key, "value": str(value)})
        rows.append({"model": name, "version": scores.version,
                     "parameter": "decision_threshold (fitted)", "value": f"{scores.threshold:.6f}"})
        rows.append({"model": name, "version": scores.version,
                     "parameter": "feature_count", "value": str(len(scores.feature_columns))})
        rows.append({"model": name, "version": scores.version,
                     "parameter": "training_rows", "value": f"{scores.training_rows:,}"})
    return pd.DataFrame(rows)
