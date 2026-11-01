from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from mlprep.phases.phase3.contract.column_selection import (
    AnalysisColumnGroups,
)
from mlprep.phases.phase3.relationships.mixed_associations import (
    MixedAssociationAnalyzer,
    MixedAssociationOptions,
)


def _contract(features: pd.DataFrame) -> SimpleNamespace:
    return SimpleNamespace(
        train=SimpleNamespace(X=features),
    )


def test_analyzer_reports_group_statistics_and_sorted_strong_pairs() -> None:
    features = pd.DataFrame(
        {
            "strong_numeric": [0.0] * 4 + [10.0] * 4,
            "weak_numeric": [0.0, 2.0] * 4,
            "constant_numeric": [1.0] * 8,
            "category": ["A"] * 4 + ["B"] * 4,
            "flag": [True] * 4 + [False] * 4,
        }
    )
    before = features.copy(deep=True)

    result = MixedAssociationAnalyzer(
        MixedAssociationOptions(
            strong_eta_squared_threshold=0.5,
        )
    ).analyze(
        contract=_contract(features),
        columns=AnalysisColumnGroups(
            numeric=[
                "strong_numeric",
                "weak_numeric",
                "constant_numeric",
                "missing_numeric",
            ],
            categorical=["category"],
            boolean=["flag"],
        ),
    )

    pd.testing.assert_frame_equal(features, before)
    assert [
        (pair.numeric_column, pair.categorical_column)
        for pair in result.pairs
    ] == [
        ("strong_numeric", "category"),
        ("strong_numeric", "flag"),
        ("weak_numeric", "category"),
        ("weak_numeric", "flag"),
        ("constant_numeric", "category"),
        ("constant_numeric", "flag"),
    ]
    assert [
        (pair.numeric_column, pair.categorical_column)
        for pair in result.strong_pairs
    ] == [
        ("strong_numeric", "category"),
        ("strong_numeric", "flag"),
    ]
    strong_pair = result.pairs[0]
    assert strong_pair.eta_squared == 1.0
    assert strong_pair.strong_association
    assert strong_pair.valid_pair_count == 8
    assert strong_pair.category_count == 2
    assert strong_pair.group_statistics == {
        "A": {
            "count": 4,
            "mean": 0.0,
            "median": 0.0,
            "standard_deviation": 0.0,
            "minimum": 0.0,
            "maximum": 0.0,
        },
        "B": {
            "count": 4,
            "mean": 10.0,
            "median": 10.0,
            "standard_deviation": 0.0,
            "minimum": 10.0,
            "maximum": 10.0,
        },
    }
    assert result.pairs[2].eta_squared == 0.0
    assert result.pairs[4].eta_squared is None
    assert not result.pairs[4].strong_association


@pytest.mark.parametrize(
    ("include_missing", "expected_count", "expected_categories"),
    [
        (False, 4, 2),
        (True, 6, 3),
    ],
)
def test_missing_values_are_dropped_or_counted_as_a_category(
    include_missing: bool,
    expected_count: int,
    expected_categories: int,
) -> None:
    features = pd.DataFrame(
        {
            "numeric": [1.0, 2.0, 10.0, 11.0, 20.0, 21.0],
            "category": ["A", "A", "B", "B", None, None],
        }
    )
    before = features.copy(deep=True)

    result = MixedAssociationAnalyzer(
        MixedAssociationOptions(
            minimum_valid_pairs=4,
            include_missing_as_category=include_missing,
            minimum_group_count=1,
        )
    ).analyze(
        contract=_contract(features),
        columns=AnalysisColumnGroups(
            numeric=["numeric"],
            categorical=["category"],
        ),
    )

    pd.testing.assert_frame_equal(features, before)
    assert len(result.pairs) == 1
    pair = result.pairs[0]
    assert pair.valid_pair_count == expected_count
    assert pair.category_count == expected_categories
    assert ("__MISSING__" in pair.group_statistics) is include_missing


def test_pair_filters_and_numeric_coercion() -> None:
    features = pd.DataFrame(
        {
            "numeric": ["1", "2", "3", "4", "5", "bad", np.inf],
            "category": ["A", "A", "B", "B", "B", "A", "B"],
            "too_many": list("ABCDEFG"),
            "rare_groups": ["A", "A", "A", "B", "B", "C", "C"],
            "constant": [1.0] * 7,
        }
    )
    groups = AnalysisColumnGroups(
        numeric=["numeric"],
        categorical=["category", "too_many", "rare_groups"],
    )

    result = MixedAssociationAnalyzer(
        MixedAssociationOptions(
            minimum_valid_pairs=5,
            minimum_group_count=3,
            maximum_categories=2,
        )
    ).analyze(
        contract=_contract(features),
        columns=groups,
    )

    assert result.pairs == []
    assert result.strong_pairs == []

    valid_pair = MixedAssociationAnalyzer(
        MixedAssociationOptions(
            minimum_valid_pairs=5,
            minimum_group_count=2,
            maximum_categories=2,
        )
    ).analyze(
        contract=_contract(features),
        columns=AnalysisColumnGroups(
            numeric=["numeric"],
            categorical=["category"],
        ),
    ).pairs[0]
    assert valid_pair.valid_pair_count == 5
    assert valid_pair.category_count == 2
    assert valid_pair.group_statistics["A"]["count"] == 2


def test_disabled_analyzer_excludes_boolean_and_validates_options() -> None:
    features = pd.DataFrame(
        {
            "numeric": [1, 2, 3, 4],
            "category": ["A", "A", "B", "B"],
            "flag": [True, True, False, False],
        }
    )
    columns = AnalysisColumnGroups(
        numeric=["numeric"],
        categorical=["category"],
        boolean=["flag"],
    )
    disabled = MixedAssociationAnalyzer(
        MixedAssociationOptions(
            enabled=False,
            minimum_valid_pairs=4,
        )
    ).analyze(
        contract=_contract(features),
        columns=columns,
    )
    without_boolean = MixedAssociationAnalyzer(
        MixedAssociationOptions(
            include_boolean_columns=False,
            minimum_valid_pairs=4,
        )
    ).analyze(
        contract=_contract(features),
        columns=columns,
    )

    assert disabled.pairs == []
    assert disabled.strong_pairs == []
    assert [pair.categorical_column for pair in without_boolean.pairs] == [
        "category"
    ]

    with pytest.raises(ValueError, match="minimum_valid_pairs"):
        MixedAssociationOptions(minimum_valid_pairs=1)
    with pytest.raises(ValueError, match="minimum_group_count"):
        MixedAssociationOptions(minimum_group_count=0)
    with pytest.raises(ValueError, match="maximum_categories"):
        MixedAssociationOptions(maximum_categories=1)
    with pytest.raises(ValueError, match="strong_eta_squared_threshold"):
        MixedAssociationOptions(strong_eta_squared_threshold=1.1)
