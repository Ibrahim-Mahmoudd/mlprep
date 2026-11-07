from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum

import numpy as np
import pandas as pd
from pydantic import BaseModel

from mlprep.phases.phase8.serialization import (
    Phase8SerializableModel,
    normalize_phase8_json_value,
    restore_phase8_series,
)


class _Kind(StrEnum):
    SAMPLE = "sample"


class _NestedModel(BaseModel):
    value: int


@dataclass
class _NestedData:
    value: str


class _Serializable(Phase8SerializableModel):
    payload: object


def test_normalize_phase8_values_covers_nested_supported_types() -> None:
    index = pd.Index(
        pd.Categorical(["a", "b"], categories=["a", "b"], ordered=True),
        name="category",
    )
    series = pd.Series(
        pd.Categorical(["a", "b"], categories=["a", "b"], ordered=True),
        index=index,
        name="values",
    )
    multi_index = pd.MultiIndex.from_tuples(
        [("a", 1), ("b", 2)],
        names=["letter", "number"],
    )
    normalized = normalize_phase8_json_value(
        {
            np.int64(1): [
                pd.NA,
                pd.NaT,
                series,
                multi_index,
                index,
                _NestedModel(value=2),
                _NestedData(value="nested"),
                {_Kind.SAMPLE},
                datetime(2024, 1, 2),
            ]
        }
    )

    assert normalized["1"][0:2] == [None, None]
    assert normalized["1"][2]["__mlprep_type__"] == "pandas.Series"
    assert normalized["1"][2]["categories"] == ["a", "b"]
    assert normalized["1"][3]["__mlprep_type__"] == "pandas.MultiIndex"
    assert normalized["1"][4]["__mlprep_type__"] == "pandas.Index"
    assert normalized["1"][5] == {"value": 2}
    assert normalized["1"][6] == {"value": "nested"}
    assert normalized["1"][7] == ["sample"]
    assert normalized["1"][8] == "2024-01-02T00:00:00"


def test_serializable_model_normalizes_payload_for_json() -> None:
    model = _Serializable(payload={"value": np.int64(3)})

    assert model.model_dump(mode="json") == {"payload": {"value": 3}}


def test_restore_series_handles_passthrough_and_plain_values() -> None:
    series = pd.Series([1, 2], name="values")

    assert restore_phase8_series(series) is series
    assert restore_phase8_series("value") == "value"
    restored = restore_phase8_series(
        {"values": [1, 2], "index": [10, 11], "name": "values"}
    )
    assert isinstance(restored, pd.Series)
    assert restored.index.tolist() == [10, 11]
    assert restored.name == "values"
    assert restore_phase8_series({"other": "value"}) == {"other": "value"}


def test_restore_series_handles_multiindex_and_categorical_values() -> None:
    restored = restore_phase8_series(
        {
            "__mlprep_type__": "pandas.Series",
            "values": ["a", "b"],
            "dtype": "category",
            "categories": ["a", "b"],
            "ordered": True,
            "index": {
                "__mlprep_type__": "pandas.MultiIndex",
                "values": [["x", 1], ["y", 2]],
                "names": ["letter", "number"],
            },
            "name": "values",
        }
    )

    assert isinstance(restored, pd.Series)
    assert isinstance(restored.index, pd.MultiIndex)
    assert restored.dtype == "category"
    assert restored.cat.ordered
    assert restored.index.names == ["letter", "number"]


def test_restore_series_handles_categorical_index_and_regular_index() -> None:
    categorical = restore_phase8_series(
        {
            "__mlprep_type__": "pandas.Series",
            "values": [1, 2],
            "dtype": "int64",
            "index": {
                "__mlprep_type__": "pandas.Index",
                "values": ["a", "b"],
                "dtype": "category",
                "categories": ["a", "b"],
                "ordered": False,
                "name": "key",
            },
        }
    )
    regular = restore_phase8_series(
        {
            "__mlprep_type__": "pandas.Series",
            "values": [1, 2],
            "dtype": "int64",
            "index": {"values": [1, 2], "dtype": "int64"},
        }
    )

    assert isinstance(categorical.index.dtype, pd.CategoricalDtype)
    assert categorical.index.name == "key"
    assert regular.index.dtype == "int64"
