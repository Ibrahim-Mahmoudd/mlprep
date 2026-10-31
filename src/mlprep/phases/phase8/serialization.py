"""Shared JSON serialization helpers for Phase 8 models."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from enum import Enum
from typing import Any

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, field_serializer


class Phase8SerializableModel(BaseModel):
    """Pydantic model with normalized JSON serialization."""

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        arbitrary_types_allowed=True,
    )

    @field_serializer(
        "*",
        when_used="json",
        check_fields=False,
    )
    def serialize_json_value(
        self,
        value: Any,
    ) -> Any:
        """Normalize nested values before Pydantic's JSON encoding."""

        return normalize_phase8_json_value(value)


def normalize_phase8_json_value(value: Any) -> Any:
    """Convert nested Phase 8 values into JSON-compatible Python values."""

    if isinstance(value, np.generic):
        return normalize_phase8_json_value(value.item())

    if value is pd.NA or value is pd.NaT:
        return None

    if isinstance(value, pd.Series):
        serialized = {
            "__mlprep_type__": "pandas.Series",
            "values": normalize_phase8_json_value(value.tolist()),
            "dtype": str(value.dtype),
            "index": normalize_phase8_json_value(value.index),
            "name": normalize_phase8_json_value(value.name),
        }
        if isinstance(value.dtype, pd.CategoricalDtype):
            serialized["categories"] = normalize_phase8_json_value(
                value.cat.categories.tolist()
            )
            serialized["ordered"] = value.dtype.ordered
        return serialized

    if isinstance(value, pd.MultiIndex):
        return {
            "__mlprep_type__": "pandas.MultiIndex",
            "values": normalize_phase8_json_value(value.to_list()),
            "names": normalize_phase8_json_value(value.names),
        }

    if isinstance(value, pd.Index):
        serialized = {
            "__mlprep_type__": "pandas.Index",
            "values": normalize_phase8_json_value(value.tolist()),
            "dtype": str(value.dtype),
            "name": normalize_phase8_json_value(value.name),
        }
        if isinstance(value.dtype, pd.CategoricalDtype):
            serialized["categories"] = normalize_phase8_json_value(
                value.categories.tolist()
            )
            serialized["ordered"] = value.dtype.ordered
        return serialized

    if isinstance(value, BaseModel):
        return normalize_phase8_json_value(value.model_dump(mode="python"))

    if is_dataclass(value) and not isinstance(value, type):
        return normalize_phase8_json_value(asdict(value))

    if isinstance(value, dict):
        return {
            str(key): normalize_phase8_json_value(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [normalize_phase8_json_value(item) for item in value]

    if isinstance(value, set):
        normalized = [normalize_phase8_json_value(item) for item in value]
        return sorted(normalized, key=repr)

    if isinstance(value, Enum):
        return normalize_phase8_json_value(value.value)

    if hasattr(value, "isoformat") and callable(value.isoformat):
        return value.isoformat()

    return value


def restore_phase8_series(value: Any) -> Any:
    """Restore a pandas Series produced by the Phase 8 JSON serializer."""

    if isinstance(value, pd.Series):
        return value

    if not isinstance(value, dict):
        return value

    if value.get("__mlprep_type__") != "pandas.Series":
        if {"values", "index"} <= value.keys():
            return pd.Series(
                value["values"],
                index=value["index"],
                name=value.get("name"),
            )
        return value

    serialized_index = value["index"]
    if serialized_index.get("__mlprep_type__") == "pandas.MultiIndex":
        index = pd.MultiIndex.from_tuples(
            [tuple(item) for item in serialized_index["values"]],
            names=serialized_index["names"],
        )
    else:
        index_options: dict[str, Any] = {
            "name": serialized_index.get("name"),
        }
        if "categories" in serialized_index:
            index_options["dtype"] = pd.CategoricalDtype(
                categories=serialized_index["categories"],
                ordered=serialized_index.get("ordered", False),
            )
        elif serialized_index.get("dtype") is not None:
            index_options["dtype"] = serialized_index["dtype"]
        index = pd.Index(serialized_index["values"], **index_options)

    dtype = value.get("dtype")
    if dtype == "category" and "categories" in value:
        series_values: Any = pd.Categorical(
            value["values"],
            categories=value["categories"],
            ordered=value.get("ordered", False),
        )
    else:
        series_values = value["values"]

    return pd.Series(
        series_values,
        index=index,
        dtype=dtype,
        name=value.get("name"),
    )
