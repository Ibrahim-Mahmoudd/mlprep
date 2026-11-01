from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest

from mlprep.phases.phase3.contract.column_selection import (
    AnalysisColumnGroups,
)
from mlprep.phases.phase3.diagnostics.high_cardinality import (
    HighCardinalityDiagnosticAnalyzer,
    HighCardinalityDiagnosticOptions,
)


def _contract(features: pd.DataFrame) -> SimpleNamespace:
    return SimpleNamespace(
        training_features=lambda: features.copy(deep=True),
    )


def test_analyzer_classifies_cardinality_and_recommends_encodings() -> None:
    values = list(range(20))
    features = pd.DataFrame(
        {
            "low": ["a", "b"] * 10,
            "frequency": ["a", "b", "c", "a", "b", "c"] * 3
            + ["a", "b"],
            "target": ["a", "b", "c", "d"] * 5,
            "hash": [f"category-{i % 10}" for i in values],
            "identifier": [f"id-{i}" for i in values],
            "not_identifier": ["a", "b", "c", "d", "a"]
            + [None] * 15,
            "text": ["short phrase", "another phrase"] * 10,
            "empty": [None] * 20,
            "text_identifier": [f"phrase {i}" for i in values],
        }
    )
    before = features.copy(deep=True)
    options = HighCardinalityDiagnosticOptions(
        low_cardinality_limit=2,
        medium_cardinality_limit=5,
        identifier_uniqueness_threshold=0.95,
        target_encoding_minimum_cardinality=4,
        prefer_hashing_above=10,
    )

    result = HighCardinalityDiagnosticAnalyzer(options).analyze(
        contract=_contract(features),
        columns=AnalysisColumnGroups(
            categorical=[
                "low",
                "frequency",
                "target",
                "hash",
                "identifier",
                "not_identifier",
                "empty",
                "missing_column",
                "text",
            ],
            text=["text", "text_identifier"],
        ),
    )

    pd.testing.assert_frame_equal(features, before)
    assert list(result.diagnostics) == [
        "low",
        "frequency",
        "target",
        "hash",
        "identifier",
        "not_identifier",
        "empty",
        "text",
        "text_identifier",
    ]
    assert result.diagnostics["low"].level == "low"
    assert result.diagnostics["low"].suggested_encoding == "one_hot"
    assert result.diagnostics["frequency"].level == "medium"
    assert (
        result.diagnostics["frequency"].suggested_encoding
        == "frequency"
    )
    assert result.diagnostics["target"].unique_count == 4
    assert (
        result.diagnostics["target"].suggested_encoding
        == "target_encoding"
    )
    assert result.diagnostics["hash"].level == "high"
    assert result.diagnostics["hash"].uniqueness_ratio == 0.5
    assert result.diagnostics["hash"].suggested_encoding == "hashing"
    assert result.diagnostics["identifier"].identifier_like
    assert (
        result.diagnostics["identifier"].suggested_encoding
        == "exclude"
    )
    assert result.diagnostics["not_identifier"].uniqueness_ratio == 0.8
    assert (
        result.diagnostics["not_identifier"].suggested_encoding
        == "target_encoding"
    )
    assert result.diagnostics["empty"].nonmissing_count == 0
    assert result.diagnostics["empty"].uniqueness_ratio == 0.0
    assert result.diagnostics["empty"].suggested_encoding == "one_hot"
    assert (
        result.diagnostics["text"].suggested_encoding
        == "text_vectorization"
    )
    assert result.diagnostics["text_identifier"].identifier_like
    assert (
        result.diagnostics["text_identifier"].suggested_encoding
        == "exclude"
    )
    assert result.high_cardinality_columns == [
        "hash",
        "identifier",
        "text_identifier",
    ]
    assert result.identifier_like_columns == [
        "identifier",
        "text_identifier",
    ]


def test_identifier_threshold_boundary_and_selected_column_deduplication() -> None:
    features = pd.DataFrame(
        {
            "boundary": ["a", "b", "c", "d", "a"],
            "text": ["one", "two", "three", "four", "five"],
        }
    )

    result = HighCardinalityDiagnosticAnalyzer(
        HighCardinalityDiagnosticOptions(
            low_cardinality_limit=2,
            medium_cardinality_limit=5,
            identifier_uniqueness_threshold=0.8,
            target_encoding_minimum_cardinality=4,
        )
    ).analyze(
        contract=_contract(features),
        columns=AnalysisColumnGroups(
            categorical=["boundary", "text"],
            text=["text"],
        ),
    )

    assert list(result.diagnostics) == ["boundary", "text"]
    assert result.diagnostics["boundary"].identifier_like
    assert result.diagnostics["boundary"].level == "identifier_like"
    assert result.diagnostics["text"].suggested_encoding == "exclude"
    assert result.identifier_like_columns == ["boundary", "text"]


def test_disabled_analyzer_and_invalid_options() -> None:
    disabled = HighCardinalityDiagnosticAnalyzer(
        HighCardinalityDiagnosticOptions(enabled=False)
    ).analyze(
        contract=_contract(pd.DataFrame({"category": ["a", "b"]})),
        columns=AnalysisColumnGroups(categorical=["category"]),
    )

    assert disabled.diagnostics == {}
    assert disabled.high_cardinality_columns == []
    assert disabled.identifier_like_columns == []

    with pytest.raises(ValueError, match="low_cardinality_limit"):
        HighCardinalityDiagnosticOptions(low_cardinality_limit=1)
    with pytest.raises(ValueError, match="medium_cardinality_limit"):
        HighCardinalityDiagnosticOptions(
            low_cardinality_limit=10,
            medium_cardinality_limit=9,
        )
    with pytest.raises(ValueError, match="identifier_uniqueness_threshold"):
        HighCardinalityDiagnosticOptions(
            identifier_uniqueness_threshold=1.1
        )
