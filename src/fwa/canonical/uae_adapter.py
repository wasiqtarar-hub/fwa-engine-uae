"""Adapter for the SYNTHETIC multi-table UAE demo layout (folder or .zip of CSVs).

One ``<table>.csv`` per canonical or supplementary table, named exactly as in
:data:`~fwa.canonical.model.CANONICAL_TABLES` and
:data:`~fwa.canonical.supplementary.SUPPLEMENTARY_TABLES` — the layout written
by ``tools/make_uae_demo_dataset.py``. The adapter:

1. hashes every raw ``claim_header`` and ``claim_line`` row into the
   :class:`~fwa.canonical.raw_store.ImmutableRawStore` **before parsing**
   (and, for every other table, records the SHA-256 of the file bytes in
   ``ds.raw_hashes["__file__:<table>"]``; per-row hashing of all tables is
   available with ``hash_all_rows=True``);
2. parses dates, datetimes, numbers and booleans by the table spec's dtype hints;
3. converts money through :meth:`SourceAdapter._to_aed` (rate 1.0 for AED — the
   call is kept so the FX path stays uniform);
4. derives, from the underlying tables, every ``claim_header`` column the rest
   of the engine reads (``service_date``, ``discharge_date``, ``claim_date``,
   ``length_of_stay_days``, ``policy_type``, ``diagnosis_primary``/
   ``_description``/``_chapter``, ``pharmacy_bill_ratio``,
   ``icd_code_matches_procedure``, ``provider_blacklist_flag``,
   ``previous_fraud_on_policy``, ``days_since_policy_start``,
   ``discharge_readmit_gap_days``, ``gross_amount_aed``/``approved_amount_aed``,
   ``fx_rate_used``, ``is_cashless``, ``num_insurers_same_event``, ``agent_id``,
   ``tpa``, ``payer_id``, ``raw_hash``, ``source_vocabulary``, ``missingness``);
5. sets every table POPULATED (with a reason saying it is SYNTHETIC) or
   NOT_POPULATED (with a plain reason), and records the certification.

``evaluation_labels.csv`` is **never** loaded into a table. Only
:meth:`UaeMultiTableAdapter.held_out_labels` reads it, for evaluation.
"""

from __future__ import annotations

import hashlib
import io
import json
import zipfile
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from .adapters import AdapterError, GenericIndiaTpaAdapter, SourceAdapter
from .model import CANONICAL_TABLES, CanonicalDataset, PopulationStatus
from .supplementary import SUPPLEMENTARY_TABLES

__all__ = ["UaeMultiTableAdapter", "READMIT_SENTINEL_DAYS", "LABELS_FILE"]

LABELS_FILE = "evaluation_labels.csv"
#: ``discharge_readmit_gap_days`` for a claim with no later admission of the same member.
READMIT_SENTINEL_DAYS = 999
_ALL_SPECS = {**CANONICAL_TABLES, **SUPPLEMENTARY_TABLES}
_LABELS = tuple(GenericIndiaTpaAdapter.LABEL_COLUMNS)
_TRUE = {"true", "1", "yes", "y", "t"}
_FALSE = {"false", "0", "no", "n", "f"}
_EXCLUSION_TYPES = {"EXCLUSION", "EXCLUSION_LIST"}


def _read_sources(source: Any) -> tuple[dict[str, pd.DataFrame], dict[str, str], str]:
    """Return ({table: raw string frame}, {table: file sha256}, description)."""
    frames: dict[str, pd.DataFrame] = {}
    digests: dict[str, str] = {}
    if isinstance(source, Mapping):
        for name, frame in source.items():
            if name == LABELS_FILE.removesuffix(".csv"):
                continue
            df = frame if isinstance(frame, pd.DataFrame) else pd.DataFrame(frame)
            frames[str(name)] = df.astype(object).where(df.notna(), "").map(str).astype(object)
        return frames, digests, "in-memory mapping"
    path = Path(str(source))
    if not path.exists():
        raise AdapterError(f"No UAE multi-table dataset at {path}. Point at the folder or the .zip that "
                           f"tools/make_uae_demo_dataset.py wrote.")

    def _take(name: str, data: bytes) -> None:
        table = Path(name).name
        if not table.lower().endswith(".csv") or table.upper().startswith("README") or table == LABELS_FILE:
            return
        table = table[:-4]
        if table not in _ALL_SPECS:
            return
        digests[table] = hashlib.sha256(data).hexdigest()
        frames[table] = pd.read_csv(io.BytesIO(data), dtype=object, keep_default_na=False, encoding="utf-8")

    if path.is_dir():
        for f in sorted(path.glob("*.csv")):
            _take(f.name, f.read_bytes())
        return frames, digests, f"folder {path.name}"
    if path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path) as zf:
            for name in sorted(zf.namelist()):
                if name.endswith("/"):
                    continue
                _take(name, zf.read(name))
        return frames, digests, f"zip {path.name}"
    raise AdapterError(f"{path.name} is neither a folder nor a .zip. The UAE multi-table adapter reads one CSV "
                       f"per table from a folder or a zip; for a single claim-header CSV use 'generic_india_tpa'.")


def _records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    """Row dicts of a raw string frame (faster than ``to_dict`` on large string frames)."""
    cols = [str(c) for c in frame.columns]
    values = [frame[c].tolist() for c in frame.columns]
    return [dict(zip(cols, row)) for row in zip(*values)]


def _parse(frame: pd.DataFrame, table: str) -> pd.DataFrame:
    """Typed copy of a raw string frame, by the table spec's dtype hints. Extra columns are kept."""
    spec = _ALL_SPECS[table]
    out = frame.drop(columns=[c for c in _LABELS if c in frame.columns])
    out = out.replace({"": None})
    for col, hint in spec.columns.items():
        if col not in out.columns:
            out[col] = None
            continue
        s = out[col]
        if hint == "date":
            out[col] = pd.to_datetime(s, errors="coerce", format="ISO8601").dt.normalize()
        elif hint == "datetime":
            out[col] = pd.to_datetime(s, errors="coerce", format="ISO8601")
        elif hint == "float":
            out[col] = pd.to_numeric(s, errors="coerce").astype(float)
        elif hint == "int":
            out[col] = pd.to_numeric(s, errors="coerce").round().astype("Int64")
        elif hint == "bool":
            low = s.astype(object).map(lambda v: None if v is None else str(v).strip().lower())
            out[col] = pd.Series([True if v in _TRUE else (False if v in _FALSE else None) for v in low],
                                 index=out.index, dtype=object)
        else:
            out[col] = s.astype(object).where(s.notna(), None)
    ordered = list(spec.columns) + [c for c in out.columns if c not in spec.columns]
    return out[ordered].reset_index(drop=True)


class UaeMultiTableAdapter(SourceAdapter):
    """Maps the SYNTHETIC multi-table UAE demo layout into the canonical model."""

    source_system = "UAE_MULTITABLE"
    currency = "AED"
    certification = ("NOT CERTIFIED — synthetic multi-table layout modelled on UAE claim constructs; "
                     "never run against a live regulator feed")

    #: Tables whose rows are hashed individually into the raw store before parsing.
    ROW_HASHED_TABLES = ("claim_header", "claim_line")

    def __init__(self, config, raw_store=None, tenant_id: str = "T001", *, hash_all_rows: bool = False) -> None:
        super().__init__(config, raw_store, tenant_id)
        self.hash_all_rows = hash_all_rows

    # ------------------------------------------------------------------ load

    def load(self, source: Any) -> CanonicalDataset:
        raw, digests, described = _read_sources(source)
        if "claim_header" not in raw:
            raise AdapterError(
                f"This UAE multi-table dataset ({described}) has no claim_header.csv. The adapter needs at least "
                f"claim_header.csv (one row per claim); every other table is optional and is reported "
                f"NOT_POPULATED when absent. Tables found: {', '.join(sorted(raw)) or 'none'}."
            )
        missing_keys = [c for c in ("claim_sk", "member_sk", "provider_sk") if c not in raw["claim_header"].columns]
        if missing_keys:
            raise AdapterError(f"claim_header.csv has no {', '.join(missing_keys)} column(s); found "
                               f"{', '.join(raw['claim_header'].columns)}.")

        ds = CanonicalDataset.empty(source_system=self.source_system, tenant_id=self.tenant_id)
        ds.reasons["__certification__"] = self.certification

        # ---- 1. raw store, BEFORE parsing ------------------------------------
        header_hashes: list[str] = []
        for table, frame in raw.items():
            if digests.get(table):
                ds.raw_hashes[f"__file__:{table}"] = digests[table]
            if table in self.ROW_HASHED_TABLES or self.hash_all_rows:
                key = _ALL_SPECS[table].primary_key.strip("()").split(",")[0].strip()
                rows = _records(frame)
                hashes = []
                for row in rows:
                    rec = self.raw_store.put(row, source_system=self.source_system,
                                             source_record_id=f"{table}:{row.get(key, '')}")
                    hashes.append(rec.raw_hash)
                if table == "claim_header":
                    header_hashes = hashes
                    ds.raw_hashes.update(zip(frame["claim_sk"].astype(str), hashes))
        source_vocab = [json.dumps(r, ensure_ascii=False, sort_keys=True)
                        for r in _records(raw["claim_header"].drop(
                            columns=[c for c in _LABELS if c in raw["claim_header"].columns]))]

        # ---- 2. parse ---------------------------------------------------------
        tables = {name: _parse(frame, name) for name, frame in raw.items()}
        for name, frame in tables.items():
            if "tenant_id" in _ALL_SPECS[name].columns:
                frame["tenant_id"] = frame["tenant_id"].where(frame["tenant_id"].notna(), self.tenant_id)
            if "source_system" in _ALL_SPECS[name].columns:
                frame["source_system"] = self.source_system

        # ---- 3. derived claim_header ------------------------------------------
        header, notes = self._derive_header(tables, header_hashes, source_vocab)
        tables["claim_header"] = header
        if "provider" in tables:
            counts = header.groupby("provider_sk").size()
            prov = tables["provider"]
            prov["claim_count"] = prov["provider_sk"].map(counts).fillna(0).astype(int)
            prov["volume_band"] = [self.config.peer_groups.volume_band(int(c)) for c in prov["claim_count"]]
        if "coverage_period" in tables:
            cov = tables["coverage_period"]
            cov["days_since_policy_start"] = (cov["valid_to"].clip(upper=header["service_date"].max())
                                              - cov["policy_inception_date"]).dt.days.astype("Int64")

        # ---- 4. status and reasons --------------------------------------------
        for name in _ALL_SPECS:
            if name == "review_outcome" and name not in tables:
                ds.mark_not_populated(
                    name, "Empty at load. Populated by reviewer dispositions recorded in the application; this "
                          "is the feedback source, not a source-system table.")
                continue
            if name not in tables:
                ds.mark_not_populated(name, f"No {name}.csv in this dataset ({described}), so nothing that needs "
                                            f"{name} can run on it.")
                continue
            frame = tables[name]
            if frame.empty:
                ds.mark_not_populated(name, f"{name}.csv is present in {described} but has no rows.")
                continue
            reason = (f"SYNTHETIC — {len(frame):,} rows from {name}.csv ({described}), generated by "
                      f"tools/make_uae_demo_dataset.py. Modelled on UAE constructs; describes no real person, "
                      f"provider or payer. {self.certification}.")
            if name == "claim_header":
                reason += " Derived columns: " + "; ".join(notes)
            ds.set(name, frame, PopulationStatus.POPULATED, reason)
        return ds

    # ------------------------------------------------------------------ derive

    def _derive_header(self, t: dict[str, pd.DataFrame], hashes: list[str],
                       vocab: list[str]) -> tuple[pd.DataFrame, list[str]]:
        h = t["claim_header"].copy()
        h["claim_sk"] = h["claim_sk"].astype(str)
        n = len(h)
        notes: list[str] = []
        flags: list[dict[str, str]] = [dict() for _ in range(n)]

        def flag(mask: pd.Series, field: str, why: str) -> None:
            for i in np.flatnonzero(np.asarray(mask, dtype=bool)):
                flags[i][field] = why

        lines = t.get("claim_line", _ALL_SPECS["claim_line"].empty_frame())
        # ---- service / discharge dates and LOS from the encounter
        enc = t.get("encounter")
        if enc is not None and not enc.empty and "encounter_sk" in h.columns:
            e = enc.drop_duplicates("encounter_sk").set_index("encounter_sk")
            key = h["encounter_sk"]
            adm = key.map(e["admission_date"])
            start = key.map(e["start_time"]).dt.normalize()
            dis = key.map(e["discharge_date"])
            end = key.map(e["end_time"]).dt.normalize()
            enc_los = key.map(e["length_of_stay_days"])
            enc_type = key.map(e["encounter_type"])
        else:
            adm = start = dis = end = pd.Series(pd.NaT, index=h.index)
            enc_los = pd.Series(np.nan, index=h.index)
            enc_type = pd.Series(None, index=h.index, dtype=object)
        line_first = lines.groupby("claim_sk")["service_date"].min() if not lines.empty else pd.Series(dtype="datetime64[ns]")
        line_last = lines.groupby("claim_sk")["service_date"].max() if not lines.empty else pd.Series(dtype="datetime64[ns]")
        service = adm.fillna(start)
        missing_service = service.isna()
        service = service.fillna(h["claim_sk"].map(line_first))
        flag(missing_service, "service_date", "no encounter; earliest claim_line.service_date used")
        discharge = dis.fillna(end)
        missing_dis = discharge.isna()
        discharge = discharge.fillna(h["claim_sk"].map(line_last)).fillna(service)
        flag(missing_dis, "discharge_date", "no encounter end; latest claim_line.service_date used")
        los = pd.to_numeric(enc_los, errors="coerce")
        los = los.where(los.notna(), (discharge - service).dt.days).fillna(0).clip(lower=0).astype(int)
        h["service_date"] = pd.to_datetime(service)
        h["discharge_date"] = pd.to_datetime(discharge)
        h["claim_date"] = pd.to_datetime(h["submission_date"]).fillna(h["discharge_date"])
        h["length_of_stay_days"] = los.values
        h["encounter_type"] = enc_type.values
        notes.append("service/discharge dates and LOS from encounter (claim_line dates when absent)")

        # ---- cover at the service date
        cov = t.get("coverage_period")
        agent = pd.Series(None, index=h.index, dtype=object)
        product = pd.Series(None, index=h.index, dtype=object)
        inception = pd.Series(pd.NaT, index=h.index)
        cov_payer = pd.Series(None, index=h.index, dtype=object)
        if cov is not None and not cov.empty:
            c = cov[["member_sk", "valid_from", "valid_to", "product", "payer_id", "agent_id",
                     "policy_inception_date"]].copy()
            q = pd.DataFrame({"_i": np.arange(n), "member_sk": h["member_sk"].astype(str).values,
                              "sd": h["service_date"].values})
            m = q.merge(c.assign(member_sk=c["member_sk"].astype(str)), on="member_sk", how="left")
            active = (m["valid_from"] <= m["sd"]) & ((m["valid_to"].isna()) | (m["sd"] <= m["valid_to"]))
            m["_rank"] = np.where(active, 0, 1)
            m["_dist"] = (m["sd"] - m["valid_from"]).abs().dt.days
            m = m.sort_values(["_i", "_rank", "_dist"], kind="stable").drop_duplicates("_i").set_index("_i")
            m = m.reindex(np.arange(n))
            agent = pd.Series(m["agent_id"].values, index=h.index)
            product = pd.Series(m["product"].values, index=h.index)
            inception = pd.Series(pd.to_datetime(m["policy_inception_date"]).fillna(m["valid_from"]).values, index=h.index)
            cov_payer = pd.Series(m["payer_id"].values, index=h.index)
            flag(~pd.Series(m["_rank"].values == 0, index=h.index), "coverage", "no coverage row active on the service date")
        h["agent_id"] = agent.values
        h["payer_id"] = h["payer_id"].where(h["payer_id"].notna(), cov_payer) if "payer_id" in h.columns else cov_payer
        h["tpa"] = h["tpa"].where(h["tpa"].notna(), h["payer_id"]) if "tpa" in h.columns else h["payer_id"]
        h["policy_type"] = [self._policy_type(p) for p in product]
        dsp = (h["service_date"] - pd.to_datetime(inception)).dt.days
        flag(dsp.isna(), "days_since_policy_start", "no policy inception date")
        h["days_since_policy_start"] = dsp.fillna(0).astype(int).values
        notes.append("agent_id, policy_type and days_since_policy_start from the coverage active on the service date")

        # ---- principal diagnosis
        dx = t.get("diagnosis")
        h["diagnosis_primary"] = None
        h["diagnosis_description"] = None
        h["diagnosis_chapter"] = None
        if dx is not None and not dx.empty:
            d = dx.copy()
            d["_p"] = np.where(d["diagnosis_type"].astype(str).str.upper() == "PRINCIPAL", 0, 1)
            d = d.sort_values(["claim_sk", "_p", "sequence"], kind="stable").drop_duplicates("claim_sk").set_index("claim_sk")
            h["diagnosis_primary"] = h["claim_sk"].map(d["code"])
            h["diagnosis_description"] = h["claim_sk"].map(d["description"])
            chap = h["claim_sk"].map(d["chapter"])
            h["diagnosis_chapter"] = [c if isinstance(c, str) and c else
                                      (self.config.peer_groups.chapter_for(str(p)) if isinstance(p, str) else "UNMAPPED")
                                      for c, p in zip(chap, h["diagnosis_primary"])]
        flag(h["diagnosis_primary"].isna(), "diagnosis_primary", "no diagnosis rows")
        h["diagnosis_primary"] = h["diagnosis_primary"].fillna("UNKNOWN")
        h["diagnosis_description"] = h["diagnosis_description"].fillna("")
        h["diagnosis_chapter"] = h["diagnosis_chapter"].fillna("UNMAPPED")
        notes.append("diagnosis_primary/description/chapter from the PRINCIPAL diagnosis")

        # ---- money
        gross = pd.to_numeric(h["gross_amount"], errors="coerce")
        rem = t.get("remittance")
        if rem is not None and not rem.empty:
            paid = rem.groupby("claim_sk")["payment_amount"].sum(min_count=1)
            approved = h["claim_sk"].map(paid)
            last_settle = rem.groupby("claim_sk")["settlement_date"].max()
            h["settlement_date"] = pd.to_datetime(h["settlement_date"]).fillna(h["claim_sk"].map(last_settle))
        else:
            approved = pd.Series(np.nan, index=h.index)
        flag(approved.isna(), "approved_amount", "no remittance rows: nothing paid yet, recorded as 0")
        approved = approved.fillna(0.0)
        h["approved_amount"] = approved.round(2).values
        gross_aed, rates = self._to_aed(gross.fillna(0.0), h["service_date"])
        approved_aed, _ = self._to_aed(approved, h["service_date"])
        h["source_currency"] = h["source_currency"].fillna(self.currency) if "source_currency" in h.columns else self.currency
        h["gross_amount_aed"] = gross_aed.round(2).values
        h["approved_amount_aed"] = approved_aed.round(2).values
        h["fx_rate_used"] = rates.values
        h["fx_rate_source"] = "config/fx.yaml, resolved at the service date (AED→AED 1.0)"
        notes.append("approved amount = sum of remittance payment_amount; AED through the FX service")

        # ---- pharmacy share
        if not lines.empty:
            is_drug = lines["activity_type"].astype(str).str.upper().eq("DRUG") | lines["product"].notna()
            pharm = lines.loc[is_drug].groupby("claim_sk")["gross_amount"].sum()
            pharm_amt = h["claim_sk"].map(pharm).fillna(0.0)
        else:
            pharm_amt = pd.Series(0.0, index=h.index)
        h["pharmacy_bill_ratio"] = np.where(gross > 0, pharm_amt / gross.replace(0, np.nan), 0.0).round(4)

        # ---- coding consistency
        h["icd_code_matches_procedure"] = self._coding_consistent(h, lines, t).values
        notes.append("icd_code_matches_procedure: principal diagnosis vs billed codes per "
                     "dx_proc_prohibition_table and indication_policy")

        # ---- exclusion status at the service date
        h["provider_blacklist_flag"] = self._excluded(h, t.get("provider_status_period")).values
        notes.append("provider_blacklist_flag: an EXCLUSION status active on the service date")
        if "previous_fraud_on_policy" in h.columns:
            h["previous_fraud_on_policy"] = h["previous_fraud_on_policy"].map(lambda v: bool(v) if v is not None else False)
        else:
            h["previous_fraud_on_policy"] = False
        notes.append("previous_fraud_on_policy: False — no source table records it")

        # ---- readmission gap
        h["discharge_readmit_gap_days"] = self._readmit_gap(h).values
        notes.append(f"discharge_readmit_gap_days: to the member's next inpatient admission "
                     f"({READMIT_SENTINEL_DAYS} when there is none, and for non-inpatient claims)")

        # ---- other payers on the event
        opr = t.get("other_payer_remittance")
        if opr is not None and not opr.empty:
            others = opr.groupby("claim_sk")["other_payer_id"].nunique()
            h["num_insurers_same_event"] = (1 + h["claim_sk"].map(others).fillna(0)).astype(int).values
        else:
            h["num_insurers_same_event"] = 1
        if "is_cashless" in h.columns:
            h["is_cashless"] = [True if v is None else bool(v) for v in h["is_cashless"]]
        else:
            h["is_cashless"] = True
        for col in ("accident_indicator",):
            if col in h.columns:
                h[col] = [False if v is None else bool(v) for v in h[col]]
        for col in ("gross_amount", "net_amount", "patient_share", "discount"):
            if col in h.columns:
                h[col] = pd.to_numeric(h[col], errors="coerce")

        h["raw_hash"] = hashes if len(hashes) == n else None
        h["source_vocabulary"] = vocab
        h["source_system"] = self.source_system
        h["missingness"] = [json.dumps(f, sort_keys=True) for f in flags]
        return h, notes

    @staticmethod
    def _policy_type(product: Any) -> str:
        if not isinstance(product, str) or not product:
            return "unknown"
        p = product.upper()
        if p.startswith("GROUP"):
            return "group_corporate"
        if p.startswith("FAMILY"):
            return "family"
        if p.startswith("INDIVIDUAL"):
            return "individual"
        return product.lower()

    @staticmethod
    def _coding_consistent(h: pd.DataFrame, lines: pd.DataFrame, t: dict[str, pd.DataFrame]) -> pd.Series:
        ok = pd.Series(True, index=h.index)
        if lines.empty:
            return ok
        principal = h.set_index("claim_sk")["diagnosis_primary"].astype(str)
        codes = lines[["claim_sk", "activity_code"]].dropna().drop_duplicates()
        codes = codes.assign(dx=codes["claim_sk"].map(principal).fillna(""))
        bad: set[str] = set()
        pro = t.get("dx_proc_prohibition_table")
        if pro is not None and not pro.empty:
            j = codes.merge(pro[["diagnosis_prefix", "activity_code"]], on="activity_code")
            hit = [str(d).startswith(str(p)) for d, p in zip(j["dx"], j["diagnosis_prefix"])]
            bad |= set(j.loc[hit, "claim_sk"])
        ind = t.get("indication_policy")
        if ind is not None and not ind.empty:
            pol = ind[["activity_code", "allowed_diagnosis_prefixes"]].dropna()
            pol = pol.assign(prefix=pol["allowed_diagnosis_prefixes"].astype(str).str.split(";")).explode("prefix")
            pol["prefix"] = pol["prefix"].str.strip()
            j = codes.merge(pol[["activity_code", "prefix"]], on="activity_code")
            j["match"] = [str(d).startswith(p) for d, p in zip(j["dx"], j["prefix"])]
            per_line = j.groupby(["claim_sk", "activity_code"])["match"].any()
            bad |= set(per_line[~per_line].index.get_level_values(0))
        return pd.Series(~h["claim_sk"].isin(bad), index=h.index)

    @staticmethod
    def _excluded(h: pd.DataFrame, status: pd.DataFrame | None) -> pd.Series:
        out = pd.Series(False, index=h.index)
        if status is None or status.empty:
            return out
        ex = status[status["status_type"].astype(str).str.upper().isin(_EXCLUSION_TYPES)]
        if ex.empty:
            return out
        q = pd.DataFrame({"_i": np.arange(len(h)), "provider_sk": h["provider_sk"].astype(str).values,
                          "sd": h["service_date"].values})
        m = q.merge(ex.assign(provider_sk=ex["provider_sk"].astype(str))[["provider_sk", "valid_from", "valid_to"]],
                    on="provider_sk")
        hit = ((m["valid_from"].isna()) | (m["valid_from"] <= m["sd"])) & ((m["valid_to"].isna()) | (m["sd"] <= m["valid_to"]))
        idx = set(m.loc[hit, "_i"])
        return pd.Series([i in idx for i in range(len(h))], index=h.index)

    @staticmethod
    def _readmit_gap(h: pd.DataFrame) -> pd.Series:
        gap = pd.Series(READMIT_SENTINEL_DAYS, index=h.index, dtype=int)
        ip_mask = h["claim_type"].astype(str).str.upper().eq("INPATIENT") if "claim_type" in h.columns else \
            h["encounter_type"].astype(str).str.upper().isin(["INPATIENT", "DAY_CASE"])
        ip = h.loc[ip_mask, ["member_sk", "service_date", "discharge_date"]]
        if ip.empty:
            return gap
        adm = ip[["member_sk", "service_date"]].drop_duplicates().sort_values(["member_sk", "service_date"])
        adm["next_admission"] = adm.groupby("member_sk")["service_date"].shift(-1)
        nxt = ip.reset_index().merge(adm, on=["member_sk", "service_date"], how="left").set_index("index")
        days = (nxt["next_admission"] - nxt["discharge_date"]).dt.days
        gap.loc[days.index] = days.fillna(READMIT_SENTINEL_DAYS).astype(int).values
        return gap

    # ------------------------------------------------------------------ labels

    def held_out_labels(self, source: Any) -> pd.DataFrame:
        """``evaluation_labels.csv`` as a separate frame, for evaluation only (empty when absent)."""
        cols = ["claim_sk", *_LABELS]
        frame: pd.DataFrame | None = None
        if isinstance(source, Mapping):
            raw = source.get("evaluation_labels")
            frame = None if raw is None else pd.DataFrame(raw)
        else:
            path = Path(str(source))
            if path.is_dir() and (path / LABELS_FILE).exists():
                frame = pd.read_csv(path / LABELS_FILE, dtype={"claim_sk": str})
            elif path.suffix.lower() == ".zip" and path.exists():
                with zipfile.ZipFile(path) as zf:
                    names = [n for n in zf.namelist() if Path(n).name == LABELS_FILE]
                    if names:
                        frame = pd.read_csv(io.BytesIO(zf.read(names[0])), dtype={"claim_sk": str})
        if frame is None or frame.empty or "claim_sk" not in frame.columns:
            return pd.DataFrame(columns=cols)
        for c in _LABELS:
            if c not in frame.columns:
                frame[c] = pd.NA
        out = frame[cols].copy()
        out["claim_sk"] = out["claim_sk"].astype(str)
        return out
