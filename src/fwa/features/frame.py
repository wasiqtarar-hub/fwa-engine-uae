"""``FeatureFrame`` — the physical barrier between features and labels.

Build brief §3.3, implementing manuscript RQ3 and §6.4:

    "**No detection component may read them.** Enforce this: the pipeline loads
     features through a ``FeatureFrame`` that physically excludes the label
     columns, and a test asserts that every detector's input columns are
     disjoint from the label set. A leakage test that fails must break the
     build."

A convention ("please don't use ``fraud_label``") is not enforcement. This class
is enforcement: the label columns are not *hidden* in the frame, they are *not
in* the frame, and any attempt to add one raises :class:`LabelLeakageError`.
Attribute and item access for a label name raises with a message pointing at the
evaluation module, which is the only place labels are legitimately read.

The same class also carries the **missingness indicators** §4.5 requires. Every
feature column ``x`` may have a companion ``x__missing`` boolean column, and
:meth:`FeatureFrame.add` creates it automatically. Nothing in this system
silently imputes: an absent authorisation ID, an absent result, an undefined
ratio are all *visible* states that a control can branch on, because in claims
data "the absence of a field is frequently itself the signal".
"""

from __future__ import annotations

from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

__all__ = ["FeatureFrame", "LabelLeakageError", "LABEL_COLUMNS", "IDENTIFIER_COLUMNS", "PROTECTED_COLUMNS"]

#: Held-out evaluation columns. Never features. Never model inputs. Never
#: readable by any control.
LABEL_COLUMNS: frozenset[str] = frozenset(
    {"fraud_label", "fraud_type", "fraud_confidence", "ground_truth_source"}
)

#: Direct identifiers. §4.8: "excludes direct identifiers ... unless a specific,
#: legally approved and clinically necessary adjustment requires them." No such
#: adjustment is approved here, so they are excluded from every model matrix.
#: They remain available as *grouping keys* — a model may not learn from
#: ``hospital_id``, but a peer comparison must obviously group by it.
IDENTIFIER_COLUMNS: frozenset[str] = frozenset(
    {"claim_sk", "claim_id", "member_sk", "patient_id", "provider_sk", "hospital_id",
     "agent_id", "protected_id_token", "encounter_sk", "episode_id"}
)

#: Protected demographic attributes. Excluded from model features outright
#: (§4.8). None are present in the claim extract; the set is declared so that a
#: future feed carrying age or sex cannot quietly acquire them as features.
PROTECTED_COLUMNS: frozenset[str] = frozenset(
    {"sex", "gender", "date_of_birth", "age", "nationality", "ethnicity", "religion", "marital_status"}
)


class LabelLeakageError(RuntimeError):
    """A detection component attempted to read held-out evaluation labels.

    Fatal by design. Build brief §3.3: "A leakage test that fails must break the
    build." A model that has seen the labels cannot be evaluated against them,
    and the resulting metric would be meaningless in a way that is very hard to
    notice after the fact.
    """


class FeatureFrame:
    """A pandas frame that cannot contain labels.

    Use :meth:`model_matrix` to get the numeric matrix a detector trains or
    scores on — it additionally strips identifiers and protected attributes and
    returns the column list it used, which the pipeline stamps onto the signal's
    ``reference_versions`` so the feature set is reproducible.
    """

    __slots__ = ("_df", "_feature_columns", "_provenance", "_level")

    def __init__(
        self,
        df: pd.DataFrame,
        *,
        level: str = "claim",
        provenance: dict[str, str] | None = None,
    ) -> None:
        offending = LABEL_COLUMNS.intersection(df.columns)
        if offending:
            raise LabelLeakageError(
                f"Refusing to build a FeatureFrame containing held-out label column(s) "
                f"{sorted(offending)}. Labels are evaluation-only; they "
                f"belong in fwa.evaluation, never in a feature path."
            )
        object.__setattr__(self, "_df", df.copy())
        object.__setattr__(self, "_level", level)
        object.__setattr__(self, "_provenance", dict(provenance or {}))
        object.__setattr__(
            self, "_feature_columns", [c for c in df.columns if not c.endswith("__missing")]
        )

    # ------------------------------------------------------------------ build

    @classmethod
    def from_frame(
        cls,
        df: pd.DataFrame,
        *,
        level: str = "claim",
        drop_labels: bool = True,
        provenance: dict[str, str] | None = None,
    ) -> "FeatureFrame":
        """Build from an arbitrary frame, dropping label columns if present.

        ``drop_labels=False`` is not a back door — it makes the constructor
        raise, which is what a caller wants when it believes the frame is
        already clean and wants to be told loudly if it is not.
        """
        work = df.drop(columns=[c for c in LABEL_COLUMNS if c in df.columns], errors="ignore") if drop_labels else df
        return cls(work, level=level, provenance=provenance)

    def add(
        self,
        name: str,
        values: pd.Series | np.ndarray | Sequence[Any],
        *,
        missing: pd.Series | None = None,
        source: str = "",
    ) -> "FeatureFrame":
        """Add a feature, plus its explicit missingness indicator (§4.5)."""
        if name in LABEL_COLUMNS:
            raise LabelLeakageError(
                f"{name!r} is a held-out evaluation label and may not be added as a feature."
            )
        series = pd.Series(values, index=self._df.index) if not isinstance(values, pd.Series) else values
        self._df[name] = series
        miss = series.isna() if missing is None else missing.astype(bool)
        self._df[f"{name}__missing"] = miss.values
        if name not in self._feature_columns:
            self._feature_columns.append(name)
        if source:
            self._provenance[name] = source
        return self

    # ------------------------------------------------------------------- read

    @property
    def df(self) -> pd.DataFrame:
        """The underlying frame. Guaranteed label-free."""
        return self._df

    @property
    def level(self) -> str:
        return self._level

    @property
    def provenance(self) -> dict[str, str]:
        """feature name → how it was derived. Rendered in the UI's SHAP panel."""
        return dict(self._provenance)

    @property
    def columns(self) -> list[str]:
        return list(self._df.columns)

    @property
    def feature_columns(self) -> list[str]:
        return list(self._feature_columns)

    def __len__(self) -> int:
        return len(self._df)

    def __contains__(self, item: object) -> bool:
        return item in self._df.columns

    def __getitem__(self, key: Any) -> Any:
        if isinstance(key, str) and key in LABEL_COLUMNS:
            raise LabelLeakageError(
                f"{key!r} is a held-out evaluation label. It is not present in this frame and "
                "must not be read by a detection component. Labels are read only in "
                "fwa.evaluation, to compute metrics."
            )
        return self._df[key]

    def __getattr__(self, name: str) -> Any:
        if name in LABEL_COLUMNS:
            raise LabelLeakageError(
                f"{name!r} is a held-out evaluation label and is not accessible from a FeatureFrame."
            )
        raise AttributeError(name)

    def missingness_report(self) -> pd.DataFrame:
        """Per-feature missingness. Shown on the Overview page.

        §4.5 requires an explicit missingness indicator for every feature; this
        is the report that makes those indicators visible rather than merely
        present.
        """
        rows = []
        for col in self._feature_columns:
            flag = f"{col}__missing"
            if flag in self._df.columns:
                n_missing = int(self._df[flag].sum())
            else:
                n_missing = int(self._df[col].isna().sum())
            rows.append(
                {
                    "feature": col,
                    "level": self._level,
                    "missing_count": n_missing,
                    "missing_rate": round(n_missing / max(len(self._df), 1), 4),
                    "derivation": self._provenance.get(col, ""),
                }
            )
        return pd.DataFrame(rows).sort_values("missing_rate", ascending=False)

    def model_matrix(
        self,
        columns: Sequence[str] | None = None,
        *,
        include_missing_indicators: bool = True,
    ) -> tuple[pd.DataFrame, list[str]]:
        """Numeric matrix for a detector, with identifiers and protected
        attributes stripped (§4.8).

        Returns ``(matrix, columns_used)``. The column list is stamped onto the
        model's reference version so a score can be reproduced later against
        exactly the feature set that produced it.
        """
        candidates = list(columns) if columns else [
            c for c in self._feature_columns
            if c not in IDENTIFIER_COLUMNS and c not in PROTECTED_COLUMNS
        ]
        numeric = [
            c for c in candidates
            if c in self._df.columns and pd.api.types.is_numeric_dtype(self._df[c])
        ]
        used = list(numeric)
        if include_missing_indicators:
            used += [f"{c}__missing" for c in numeric if f"{c}__missing" in self._df.columns]
        matrix = self._df[used].copy()
        for c in numeric:
            # Explicit, recorded fill: the companion __missing column preserves
            # the fact that the value was absent, so this is not a silent
            # imputation — the model can see both the fill and the flag.
            median = matrix[c].median()
            matrix[c] = matrix[c].fillna(0.0 if pd.isna(median) else median)
        for c in used:
            if c.endswith("__missing"):
                matrix[c] = matrix[c].astype(float)
        return matrix.replace([np.inf, -np.inf], 0.0), used

    def assert_no_labels(self) -> None:
        """Used by the governance test and by every detector entry point."""
        offending = LABEL_COLUMNS.intersection(self._df.columns)
        if offending:
            raise LabelLeakageError(f"Label column(s) {sorted(offending)} present in a FeatureFrame.")

    def filter(self, mask: pd.Series) -> "FeatureFrame":
        return FeatureFrame(self._df.loc[mask].copy(), level=self._level, provenance=self._provenance)

    def head(self, n: int = 5) -> pd.DataFrame:
        return self._df.head(n)
