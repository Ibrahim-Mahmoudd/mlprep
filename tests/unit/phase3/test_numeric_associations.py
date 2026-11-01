from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from mlprep.phases.phase3.contract.column_selection import (
    AnalysisColumnGroups,
)
from mlprep.phases.phase3.relationships.numeric_associations import (
    NumericAssociationAnalyzer,
    NumericAssociationOptions,
)


def _contract(features: pd.DataFrame) -> SimpleNamespace:
    return SimpleNamespace(
        train=SimpleNamespace(X=features),
    )


def test_analyzer_calculates_correlations_and_serializes_missing_values() -> None:
    x = np.arange(1, 9, dtype=float)
    features = pd.DataFrame(
        {
            "x": x,
            "positive": 2 * x,
            "negative": -x,
            "constant": np.ones(8),
            "sparse": [1.0, 2.0, 3.0, np.nan, np.nan, np.nan, np.nan, np.nan],
            "numeric_text": [str(value) for value in x],
        }
    )
    before = features.copy(deep=True)

    result = NumericAssociationAnalyzer(
        NumericAssociationOptions(
            methods=("pearson", "spearman", "kendall"),
            primary_method="spearman",
            minimum_valid_pairs=3,
            high_correlation_threshold=0.95,
            perfect_correlation_threshold=0.999,
        )
    ).analyze(
        contract=_contract(features),
        columns=AnalysisColumnGroups(
            numeric=[
                "x",
                "positive",
                "negative",
                "constant",
                "sparse",
                "numeric_text",
                "missing_column",
            ]
        ),
    )

    pd.testing.assert_frame_equal(features, before)
    assert result.analyzed_columns == [
        "x",
        "positive",
        "negative",
        "constant",
        "sparse",
        "numeric_text",
    ]
    assert result.primary_method == "spearman"
    assert len(result.pairs) == 15
    positive_pair = next(
        pair
        for pair in result.pairs
        if (pair.left_column, pair.right_column)
        == ("x", "positive")
    )
    negative_pair = next(
        pair
        for pair in result.pairs
        if (pair.left_column, pair.right_column)
        == ("x", "negative")
    )
    sparse_pair = next(
        pair
        for pair in result.pairs
        if (pair.left_column, pair.right_column)
        == ("x", "sparse")
    )
    assert positive_pair.pearson == 1.0
    assert positive_pair.spearman == 1.0
    assert positive_pair.kendall == pytest.approx(1.0)
    assert positive_pair.absolute_primary_correlation == 1.0
    assert positive_pair.high_correlation
    assert positive_pair.perfect_correlation
    assert negative_pair.spearman == -1.0
    assert negative_pair.high_correlation
    assert negative_pair.perfect_correlation
    assert sparse_pair.valid_pair_count == 3
    assert sparse_pair.spearman == 1.0
    assert result.correlation_matrices["pearson"]["x"]["constant"] is None
    assert result.correlation_matrices["spearman"]["x"]["numeric_text"] == 1.0
    assert result.highly_correlated_pairs
    assert all(
        pair.high_correlation
        for pair in result.highly_correlated_pairs
    )
    assert [
        pair.absolute_primary_correlation
        for pair in result.highly_correlated_pairs
    ] == sorted(
        (
            pair.absolute_primary_correlation
            for pair in result.highly_correlated_pairs
        ),
        reverse=True,
    )


def test_pair_filtering_column_limit_and_disabled_all_pairs() -> None:
    features = pd.DataFrame(
        {
            "first": [1, 2, 3, 4, 5],
            "second": [2, 4, 6, 8, 10],
            "third": [5, 4, 3, 2, 1],
        }
    )
    groups = AnalysisColumnGroups(
        numeric=["first", "second", "third"]
    )

    result = NumericAssociationAnalyzer(
        NumericAssociationOptions(
            maximum_columns=2,
            include_all_pairs=False,
            high_correlation_threshold=0.9,
        )
    ).analyze(
        contract=_contract(features),
        columns=groups,
    )

    assert result.analyzed_columns == ["first", "second"]
    assert result.pairs == []
    assert len(result.highly_correlated_pairs) == 1
    assert result.highly_correlated_pairs[0].left_column == "first"
    assert result.highly_correlated_pairs[0].right_column == "second"

    disabled = NumericAssociationAnalyzer(
        NumericAssociationOptions(enabled=False)
    ).analyze(
        contract=_contract(features),
        columns=groups,
    )
    assert disabled.pairs == []
    assert disabled.correlation_matrices == {}
    assert disabled.analyzed_columns == []


def test_insufficient_pairs_and_invalid_options() -> None:
    features = pd.DataFrame(
        {
            "sparse": [1.0, 2.0, np.nan],
            "other": [1.0, 2.0, 3.0],
        }
    )
    result = NumericAssociationAnalyzer(
        NumericAssociationOptions(minimum_valid_pairs=3)
    ).analyze(
        contract=_contract(features),
        columns=AnalysisColumnGroups(
            numeric=["sparse", "other"]
        ),
    )

    assert len(result.pairs) == 1
    assert result.pairs[0].valid_pair_count == 2
    assert result.pairs[0].pearson is None
    assert result.pairs[0].absolute_primary_correlation is None
    assert not result.pairs[0].high_correlation

    with pytest.raises(ValueError, match="At least one"):
        NumericAssociationOptions(methods=())
    with pytest.raises(ValueError, match="Unsupported"):
        NumericAssociationOptions(methods=("invalid",))
    with pytest.raises(ValueError, match="primary_method"):
        NumericAssociationOptions(
            methods=("pearson",),
            primary_method="spearman",
        )
    with pytest.raises(ValueError, match="minimum_valid_pairs"):
        NumericAssociationOptions(minimum_valid_pairs=1)
    with pytest.raises(ValueError, match="high_correlation_threshold"):
        NumericAssociationOptions(high_correlation_threshold=1.1)
    with pytest.raises(ValueError, match="perfect_correlation_threshold"):
        NumericAssociationOptions(
            high_correlation_threshold=0.9,
            perfect_correlation_threshold=0.8,
        )
    with pytest.raises(ValueError, match="maximum_columns"):
        NumericAssociationOptions(maximum_columns=1)
