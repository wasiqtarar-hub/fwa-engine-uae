"""Canonical data model, immutable raw store and source adapters."""

from .model import CANONICAL_TABLES, CanonicalDataset, PopulationStatus, TableSpec, NOT_POPULATED
from .raw_store import ImmutableRawStore, RawRecord, sha256_record
from .adapters import (
    SourceAdapter,
    GenericIndiaTpaAdapter,
    ShafafiyaAdapter,
    EClaimLinkAdapter,
    get_adapter,
    AdapterError,
    NOT_CERTIFIED,
)
from .uae_adapter import UaeMultiTableAdapter
from .adapters import _ADAPTERS as _REGISTRY

_REGISTRY.setdefault("uae_multitable", UaeMultiTableAdapter)

__all__ = [
    "CANONICAL_TABLES", "CanonicalDataset", "PopulationStatus", "TableSpec", "NOT_POPULATED",
    "ImmutableRawStore", "RawRecord", "sha256_record",
    "SourceAdapter", "GenericIndiaTpaAdapter", "ShafafiyaAdapter", "EClaimLinkAdapter", "UaeMultiTableAdapter",
    "get_adapter", "AdapterError", "NOT_CERTIFIED",
]
