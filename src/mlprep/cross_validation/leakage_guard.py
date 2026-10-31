"""Leakage checks and deterministic fold fingerprints."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from datetime import date, datetime
from enum import Enum
from typing import Any

import numpy as np
import pandas as pd


class LeakageGuard:
    """Verify fold separation and fingerprint fold data and fitted state."""

    @classmethod
    def partition_fingerprint(
        cls,
        *,
        features: pd.DataFrame,
        target: pd.Series,
        row_ids: pd.Series,
        groups: pd.Series | None = None,
    ) -> str:
        """Fingerprint values, schemas, indexes, IDs, and optional groups."""

        payload = {
            "features": cls._canonical(features),
            "target": cls._canonical(target),
            "row_ids": cls._canonical(row_ids),
            "groups": cls._canonical(groups),
        }
        return cls._digest(payload)

    @classmethod
    def fitted_state_fingerprint(
        cls,
        *,
        training_fingerprint: str,
        component_states: Sequence[tuple[str, Mapping[str, Any]]],
    ) -> str:
        """Fingerprint only state fitted from the fold training partition."""

        return cls._digest(
            {
                "training_fingerprint": training_fingerprint,
                "component_states": [
                    {
                        "name": name,
                        "state": cls._canonical(state),
                    }
                    for name, state in component_states
                ],
            }
        )

    @classmethod
    def assert_validation_unchanged(
        cls,
        *,
        expected_fingerprint: str,
        features: pd.DataFrame,
        target: pd.Series,
        row_ids: pd.Series,
        groups: pd.Series | None = None,
    ) -> None:
        """Raise if a transform operation mutated its validation input."""

        actual = cls.partition_fingerprint(
            features=features,
            target=target,
            row_ids=row_ids,
            groups=groups,
        )
        if actual != expected_fingerprint:
            raise ValueError(
                "A fold transformer mutated its validation input."
            )

    @classmethod
    def assert_disjoint_row_ids(
        cls,
        *,
        training_row_ids: pd.Series,
        validation_row_ids: pd.Series,
    ) -> None:
        """Raise if train and validation share row identifiers."""

        training = {cls._token(value) for value in training_row_ids}
        validation = {cls._token(value) for value in validation_row_ids}
        if training.intersection(validation):
            raise ValueError("Fold train and validation row IDs overlap.")

    @classmethod
    def assert_disjoint_groups(
        cls,
        *,
        training_groups: pd.Series,
        validation_groups: pd.Series,
    ) -> None:
        """Raise if a group occurs in both fold partitions."""

        training = {cls._token(value) for value in training_groups}
        validation = {cls._token(value) for value in validation_groups}
        if training.intersection(validation):
            raise ValueError("Groups overlap between fold train and validation.")

    @classmethod
    def _token(cls, value: Any) -> str:
        return json.dumps(
            cls._canonical(value),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        )

    @classmethod
    def _digest(cls, value: Any) -> str:
        serialized = json.dumps(
            cls._canonical(value),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        )
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    @classmethod
    def _canonical(cls, value: Any) -> Any:
        if value is None or value is pd.NA or value is pd.NaT:
            return {"type": type(value).__name__, "value": None}
        if isinstance(value, Enum):
            return {
                "type": type(value).__qualname__,
                "value": cls._canonical(value.value),
            }
        if isinstance(value, np.generic):
            return {
                "type": type(value).__qualname__,
                "dtype": str(value.dtype),
                "value": cls._canonical(value.item()),
            }
        if isinstance(value, np.ndarray):
            return {
                "type": "ndarray",
                "dtype": str(value.dtype),
                "shape": list(value.shape),
                "values": cls._canonical(value.tolist()),
            }
        if isinstance(value, pd.DataFrame):
            return {
                "type": "DataFrame",
                "columns": cls._canonical(list(value.columns)),
                "dtypes": [str(dtype) for dtype in value.dtypes],
                "index": cls._canonical(value.index),
                "values": cls._canonical(
                    [list(row) for row in value.itertuples(index=False, name=None)]
                ),
            }
        if isinstance(value, pd.Series):
            return {
                "type": "Series",
                "name": cls._canonical(value.name),
                "dtype": str(value.dtype),
                "index": cls._canonical(value.index),
                "values": cls._canonical(value.tolist()),
            }
        if isinstance(value, pd.Index):
            return {
                "type": type(value).__qualname__,
                "dtype": str(value.dtype),
                "name": cls._canonical(value.name),
                "values": cls._canonical(value.tolist()),
            }
        if isinstance(value, Mapping):
            entries = [
                (cls._canonical(key), cls._canonical(item))
                for key, item in value.items()
            ]
            return {
                "type": "mapping",
                "entries": sorted(
                    entries,
                    key=lambda entry: json.dumps(
                        entry[0],
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                ),
            }
        if isinstance(value, (list, tuple)):
            return {
                "type": type(value).__name__,
                "values": [cls._canonical(item) for item in value],
            }
        if isinstance(value, (set, frozenset)):
            values = [cls._canonical(item) for item in value]
            return {
                "type": type(value).__name__,
                "values": sorted(
                    values,
                    key=lambda item: json.dumps(
                        item,
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                ),
            }
        if isinstance(value, (datetime, date)):
            return {
                "type": type(value).__qualname__,
                "value": value.isoformat(),
            }
        if isinstance(value, bool):
            return {"type": "bool", "value": value}
        if isinstance(value, int):
            return {"type": "int", "value": value}
        if isinstance(value, float):
            return {
                "type": "float",
                "value": repr(value) if not math.isfinite(value) else value,
            }
        if isinstance(value, str):
            return {"type": "str", "value": value}
        raise TypeError(
            "Fold state fingerprints require JSON-compatible values; "
            f"unsupported value type: {type(value).__qualname__}."
        )
