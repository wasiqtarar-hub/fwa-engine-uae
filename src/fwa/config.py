"""Effective-dated parameter registry and FX service.

Manuscript §3.9 (Parameter governance) requires that *every* threshold, peer
grouping parameter and configuration value live in an effective-dated table
scoped by regulator, payer, product, contract, provider type and code family,
and that every value record its owner, rationale, source, approval, expected
alert volume and expiry.

This module is the single implementation of that requirement. Nothing else in
``src/fwa`` is permitted to contain a bare numeric threshold; the governance
test ``tests/governance/test_parameter_governance.py`` enforces that by scanning
the source tree.

Two design points worth stating explicitly, because both are manuscript
requirements rather than conveniences:

1. **As-of resolution.** :meth:`ParameterRegistry.get` takes an ``as_of`` date
   and resolves against ``valid_from`` / ``valid_to``. The caller is expected to
   pass the *service date* of the claim being evaluated, not today's date
   (§3.6). Evaluating a 2023 claim against a 2026 threshold silently misapplies
   policy, which is exactly the failure §3.6 exists to prevent.

2. **No hidden defaults.** A missing key raises :class:`ParameterNotFound`.
   There is no ``registry.get(key, default=...)`` convenience, because a default
   supplied at the call site is a threshold living outside the registry.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

import yaml

__all__ = [
    "ParameterNotFound",
    "ParameterRecord",
    "ParameterRegistry",
    "FxService",
    "PeerGroupConfig",
    "load_config",
    "AppConfig",
    "CONFIG_DIR",
    "PROJECT_ROOT",
]

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "config"

_DATE_MIN = _dt.date(1900, 1, 1)
_DATE_MAX = _dt.date(2099, 12, 31)


class ParameterNotFound(KeyError):
    """Raised when a configuration value is requested that is not registered.

    Deliberately fatal. A payment-integrity control that silently falls back to
    a hard-coded default is a control whose behaviour cannot be reconstructed
    from the parameter registry at audit time.
    """


def _as_date(value: Any) -> _dt.date:
    if isinstance(value, _dt.datetime):
        return value.date()
    if isinstance(value, _dt.date):
        return value
    if isinstance(value, str):
        return _dt.date.fromisoformat(value)
    raise TypeError(f"Cannot interpret {value!r} as a date")


@dataclass(frozen=True)
class ParameterRecord:
    """One governed configuration value, with its full §3.9 provenance."""

    key: str
    value: Any
    owner: str
    rationale: str
    source: str
    approved_by: str
    valid_from: _dt.date
    valid_to: _dt.date
    regulator: str = "*"
    payer: str = "*"
    product: str = "*"
    contract: str = "*"
    provider_type: str = "*"
    code_family: str = "*"
    expected_alert_volume: int | None = None
    review_date: _dt.date | None = None
    expires_on: _dt.date | None = None

    # ---- scope matching -----------------------------------------------------

    _SCOPE_FIELDS = ("regulator", "payer", "product", "contract", "provider_type", "code_family")

    #: Values that mean "this record is not narrowed on this dimension".
    #: ``*`` is the ordinary wildcard. ``NOT_APPLICABLE`` is the honesty marker
    #: the shipped defaults use on ``regulator``: no regulator's rule book
    #: governs these values, and nobody should be able to read a regulator's
    #: endorsement into a threshold that was set here. That is a statement about
    #: provenance rather than a narrowing, and treating it as a narrowing would
    #: make every parameter unresolvable.
    _UNSCOPED = frozenset({"*", "NOT_APPLICABLE", "NONE_PROXY_DATASET"})

    def matches_scope(self, scope: Mapping[str, str] | None) -> bool:
        """A record applies to a query only if the query is *within* its scope.

        A record that pins ``payer=PAYER-A`` is a statement about that payer and
        nothing else. It therefore does not answer an unscoped question ("what
        is the general value?"), and it does not answer a question scoped to a
        different dimension either. Any other reading lets a narrow override
        leak out as though it were the default — which is the failure mode
        §3.9's scoping exists to prevent, and a particularly quiet one, because
        the returned number looks perfectly ordinary.
        """
        for dim in self._SCOPE_FIELDS:
            mine = getattr(self, dim)
            if mine in self._UNSCOPED:
                continue  # applies to every value of this dimension
            if not scope or scope.get(dim) != mine:
                return False
        return True

    def specificity(self) -> int:
        """How many scope dimensions this record pins down. More specific wins."""
        return sum(
            1 for dim in self._SCOPE_FIELDS if getattr(self, dim) not in self._UNSCOPED
        )

    def covers(self, as_of: _dt.date) -> bool:
        return self.valid_from <= as_of <= self.valid_to

    def is_expired(self, as_of: _dt.date) -> bool:
        """§3.9 requires an expiry/review date. An expired value is still
        returned (so the system does not silently break) but is reported by
        :meth:`ParameterRegistry.governance_report` so it shows up on the
        Governance page rather than rotting unnoticed."""
        horizon = self.expires_on or self.review_date
        return horizon is not None and as_of > horizon

    def to_row(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "value": json.dumps(self.value) if isinstance(self.value, (dict, list)) else self.value,
            "owner": self.owner,
            "rationale": self.rationale,
            "source": self.source,
            "approved_by": self.approved_by,
            "valid_from": self.valid_from.isoformat(),
            "valid_to": self.valid_to.isoformat(),
            "regulator": self.regulator,
            "payer": self.payer,
            "product": self.product,
            "contract": self.contract,
            "provider_type": self.provider_type,
            "code_family": self.code_family,
            "expected_alert_volume": self.expected_alert_volume,
            "review_date": self.review_date.isoformat() if self.review_date else None,
        }


class ParameterRegistry:
    """Effective-dated, scoped, auditable store of every ``cfg.*`` value."""

    def __init__(self, records: Iterable[ParameterRecord]) -> None:
        self._records: dict[str, list[ParameterRecord]] = {}
        for rec in records:
            self._records.setdefault(rec.key, []).append(rec)
        self._lock = threading.RLock()

    # ---- construction -------------------------------------------------------

    @classmethod
    def from_yaml(cls, path: Path | str | None = None) -> "ParameterRegistry":
        path = Path(path) if path else CONFIG_DIR / "parameters.yaml"
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        defaults = raw.get("defaults", {}) or {}
        records: list[ParameterRecord] = []
        for entry in raw.get("parameters", []) or []:
            merged = {**defaults, **entry}
            missing = [f for f in ("key", "value", "owner", "rationale", "source") if f not in merged]
            if missing:
                raise ValueError(
                    f"parameters.yaml entry {entry.get('key', '<unnamed>')!r} is missing "
                    f"required §3.9 governance fields: {missing}"
                )
            records.append(
                ParameterRecord(
                    key=merged["key"],
                    value=merged["value"],
                    owner=merged["owner"],
                    rationale=merged["rationale"],
                    source=merged["source"],
                    approved_by=merged.get("approved_by", "UNAPPROVED"),
                    valid_from=_as_date(merged.get("valid_from", _DATE_MIN)),
                    valid_to=_as_date(merged.get("valid_to", _DATE_MAX)),
                    regulator=str(merged.get("regulator", "*")),
                    payer=str(merged.get("payer", "*")),
                    product=str(merged.get("product", "*")),
                    contract=str(merged.get("contract", "*")),
                    provider_type=str(merged.get("provider_type", "*")),
                    code_family=str(merged.get("code_family", "*")),
                    expected_alert_volume=merged.get("expected_alert_volume"),
                    review_date=_as_date(merged["review_date"]) if merged.get("review_date") else None,
                    expires_on=_as_date(merged["expires_on"]) if merged.get("expires_on") else None,
                )
            )
        return cls(records)

    # ---- lookup -------------------------------------------------------------

    def get(
        self,
        key: str,
        as_of: _dt.date | str | None = None,
        scope: Mapping[str, str] | None = None,
    ) -> Any:
        """Resolve one parameter as of a date, within a scope.

        Parameters
        ----------
        key
            The registry key, e.g. ``"readmit_window_days"``.
        as_of
            The **service date** of the thing being evaluated. Defaults to the
            widest-valid record when omitted, which is only appropriate for
            platform parameters (session timeout, seeds) that are not
            claim-scoped.
        scope
            Optional mapping of ``regulator/payer/product/contract/
            provider_type/code_family``. The most specific matching record wins.
        """
        with self._lock:
            candidates = self._records.get(key)
        if not candidates:
            raise ParameterNotFound(
                f"Parameter {key!r} is not registered in config/parameters.yaml. "
                "Manuscript §3.9 forbids a threshold living anywhere else; add it "
                "to the registry with an owner, rationale, source and review date."
            )
        as_of_date = _as_date(as_of) if as_of is not None else _dt.date.today()
        in_scope = [r for r in candidates if r.matches_scope(scope)]
        if not in_scope:
            raise ParameterNotFound(
                f"Parameter {key!r} exists but no record matches scope {dict(scope or {})!r}."
            )
        viable = [r for r in in_scope if r.covers(as_of_date)]
        if not viable:
            # Deliberately a failure rather than a fallback. Returning the
            # nearest out-of-window value would silently apply a threshold that
            # was not in force when the claim was incurred, and §3.6's whole
            # point is that the version in force on the service date is the one
            # that applies. A visible error is recoverable; a quietly wrong
            # threshold applied to a real adjudication is not.
            windows = ", ".join(
                f"{r.valid_from.isoformat()}→{r.valid_to.isoformat()}" for r in in_scope
            )
            raise ParameterNotFound(
                f"Parameter {key!r} has no version in force on {as_of_date.isoformat()}. "
                f"Registered windows: {windows}. §3.9 requires an effective-dated value; "
                f"add or extend a record rather than defaulting."
            )
        viable.sort(key=lambda r: (r.specificity(), r.valid_from), reverse=True)
        return viable[0].value

    def record(
        self,
        key: str,
        as_of: _dt.date | str | None = None,
        scope: Mapping[str, str] | None = None,
    ) -> ParameterRecord:
        """Same resolution as :meth:`get` but returns the full provenance record."""
        with self._lock:
            candidates = self._records.get(key)
        if not candidates:
            raise ParameterNotFound(key)
        as_of_date = _as_date(as_of) if as_of is not None else _dt.date.today()
        viable = [r for r in candidates if r.covers(as_of_date) and r.matches_scope(scope)] or [
            r for r in candidates if r.matches_scope(scope)
        ]
        viable.sort(key=lambda r: (r.specificity(), r.valid_from), reverse=True)
        return viable[0]

    def keys(self) -> list[str]:
        return sorted(self._records)

    def all_records(self) -> list[ParameterRecord]:
        return [r for recs in self._records.values() for r in recs]

    def governance_report(self, as_of: _dt.date | str | None = None) -> list[dict[str, Any]]:
        """Rows for the Governance page: every parameter with its provenance,
        flagged where the review date has passed or approval is missing."""
        as_of_date = _as_date(as_of) if as_of is not None else _dt.date.today()
        rows = []
        for rec in sorted(self.all_records(), key=lambda r: r.key):
            row = rec.to_row()
            row["expired"] = rec.is_expired(as_of_date)
            row["unapproved"] = rec.approved_by == "UNAPPROVED"
            rows.append(row)
        return rows

    def fingerprint(self) -> str:
        """Stable hash of the whole registry.

        Stamped onto every signal so that a peer comparison or model result can
        be reproduced from its stored baseline version at any later date
        (§3.10 reproducibility NFR).
        """
        payload = json.dumps(
            [r.to_row() for r in sorted(self.all_records(), key=lambda r: (r.key, r.valid_from))],
            sort_keys=True,
            default=str,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


# ---------------------------------------------------------------------------
# FX
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FxRate:
    from_currency: str
    to_currency: str
    rate: float
    valid_from: _dt.date
    valid_to: _dt.date
    source: str
    owner: str


class FxService:
    """Currency conversion through ``config/fx.yaml``, never a literal.

    AED is the internal exposure currency. A dataset denominated in anything
    else is converted at the rate in force on the **service date** — an as-of
    join like any other reference lookup. Converting at today's spot rate would
    restate a 2023 exposure at a 2026 rate, which would make a historical
    figure irreproducible.
    """

    def __init__(self, rates: Iterable[FxRate], base_currency: str = "AED") -> None:
        self._rates = list(rates)
        self.base_currency = base_currency

    @classmethod
    def from_yaml(cls, path: Path | str | None = None) -> "FxService":
        path = Path(path) if path else CONFIG_DIR / "fx.yaml"
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        rates = [
            FxRate(
                from_currency=r["from_currency"],
                to_currency=r["to_currency"],
                rate=float(r["rate"]),
                valid_from=_as_date(r["valid_from"]),
                valid_to=_as_date(r["valid_to"]),
                source=r.get("source", "UNDOCUMENTED"),
                owner=r.get("owner", "UNOWNED"),
            )
            for r in raw.get("rates", [])
        ]
        return cls(rates, base_currency=raw.get("base_currency", "AED"))

    def rate(self, from_currency: str, as_of: _dt.date | str, to_currency: str | None = None) -> FxRate:
        to_currency = to_currency or self.base_currency
        as_of_date = _as_date(as_of)
        viable = [
            r
            for r in self._rates
            if r.from_currency == from_currency
            and r.to_currency == to_currency
            and r.valid_from <= as_of_date <= r.valid_to
        ]
        if not viable:
            raise ParameterNotFound(
                f"No {from_currency}->{to_currency} FX rate effective on {as_of_date}. "
                "config/fx.yaml must cover every service date in the data; the system "
                "will not fall back to a hard-coded literal."
            )
        viable.sort(key=lambda r: r.valid_from, reverse=True)
        return viable[0]

    def convert(self, amount: float, from_currency: str, as_of: _dt.date | str, to_currency: str | None = None) -> float:
        if amount is None:
            return None  # type: ignore[return-value]
        return float(amount) * self.rate(from_currency, as_of, to_currency).rate

    def convert_series(self, amounts, from_currency: str, dates, to_currency: str | None = None):
        """Vectorised conversion for a pandas Series of amounts and dates."""
        import pandas as pd

        dates = pd.to_datetime(dates)
        out = []
        cache: dict[_dt.date, float] = {}
        for amt, dt in zip(amounts, dates):
            if dt is None or (hasattr(dt, "__class__") and str(dt) == "NaT"):
                out.append(None)
                continue
            d = dt.date()
            if d not in cache:
                cache[d] = self.rate(from_currency, d, to_currency).rate
            out.append(None if amt is None else float(amt) * cache[d])
        return pd.Series(out, index=getattr(amounts, "index", None))


# ---------------------------------------------------------------------------
# Peer-group configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PeerLevel:
    level: int
    canonical_name: str
    dataset_mapping: str | None
    availability: str
    description: str
    proxy_note: str | None = None
    bands: list[dict[str, Any]] = field(default_factory=list)
    required_canonical_fields: list[str] = field(default_factory=list)

    @property
    def populated(self) -> bool:
        return self.availability != "NOT_POPULATED"


@dataclass(frozen=True)
class PeerGroupConfig:
    levels: list[PeerLevel]
    backoff_order: list[dict[str, Any]]
    icd_chapter_map: dict[str, Any]

    @classmethod
    def from_yaml(cls, path: Path | str | None = None) -> "PeerGroupConfig":
        path = Path(path) if path else CONFIG_DIR / "peer_groups.yaml"
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        levels = [
            PeerLevel(
                level=int(l["level"]),
                canonical_name=l["canonical_name"],
                dataset_mapping=l.get("dataset_mapping"),
                availability=l["availability"],
                description=l.get("description", "").strip(),
                proxy_note=l.get("proxy_note"),
                bands=l.get("bands", []) or [],
                required_canonical_fields=l.get("required_canonical_fields", []) or [],
            )
            for l in raw.get("hierarchy", [])
        ]
        return cls(
            levels=levels,
            backoff_order=raw.get("backoff_order", []),
            icd_chapter_map=raw.get("icd_chapter_map", {}),
        )

    def level(self, canonical_name: str) -> PeerLevel:
        for l in self.levels:
            if l.canonical_name == canonical_name:
                return l
        raise KeyError(canonical_name)

    def chapter_for(self, icd_code: str) -> str:
        codes = self.icd_chapter_map.get("codes", {})
        entry = codes.get(icd_code)
        if entry is None:
            # Honest fallback: an unmapped code is UNMAPPED, not silently binned.
            return "UNMAPPED"
        return entry["chapter"]

    def volume_band(self, claim_count: int) -> str:
        for band in self.level("provider_volume_band").bands:
            if band["min_claims"] <= claim_count <= band["max_claims"]:
                return band["name"]
        return "unbanded"


# ---------------------------------------------------------------------------
# Aggregate handle
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AppConfig:
    """Everything the pipeline reads from ``config/``, resolved once."""

    parameters: ParameterRegistry
    fx: FxService
    peer_groups: PeerGroupConfig
    config_dir: Path

    def get(self, key: str, as_of: _dt.date | str | None = None, **scope: str) -> Any:
        return self.parameters.get(key, as_of=as_of, scope=scope or None)

    def fingerprint(self) -> str:
        return self.parameters.fingerprint()


_CACHE: dict[str, AppConfig] = {}
_CACHE_LOCK = threading.Lock()


def load_config(config_dir: Path | str | None = None, *, refresh: bool = False) -> AppConfig:
    """Load (and cache) the configuration bundle."""
    directory = Path(config_dir) if config_dir else CONFIG_DIR
    key = str(directory.resolve())
    with _CACHE_LOCK:
        if refresh or key not in _CACHE:
            _CACHE[key] = AppConfig(
                parameters=ParameterRegistry.from_yaml(directory / "parameters.yaml"),
                fx=FxService.from_yaml(directory / "fx.yaml"),
                peer_groups=PeerGroupConfig.from_yaml(directory / "peer_groups.yaml"),
                config_dir=directory,
            )
        return _CACHE[key]
