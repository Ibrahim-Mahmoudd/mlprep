from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest

from mlprep.phases.phase3.contract.column_selection import (
    AnalysisColumnGroups,
)
from mlprep.phases.phase3.relationships.categorical_associations import (
    CategoricalAssociationAnalyzer,
    CategoricalAssociationOptions,
)


def _contract(frame: pd.DataFrame) -> SimpleNamespace:
    return SimpleNamespace(
        training_features=lambda: frame,
    )


def test_perfect_and_independent_pairs_are_ranked_with_metadata() -> None:
    frame = pd.DataFrame(
        {
            "left": ["a"] * 4 + ["b"] * 4,
            "right": ["a"] * 4 + ["b"] * 4,
            "balanced": ["x", "y"] * 4,
        }
    )
    before = frame.copy(deep=True)
    analyzer = CategoricalAssociationAnalyzer(
        CategoricalAssociationOptions(
            minimum_expected_frequency=5.0,
        )
    )
    columns = AnalysisColumnGroups(
        categorical=["left", "right", "balanced"]
    )

    result = analyzer.analyze(
        contract=_contract(frame),
        columns=columns,
    )

    assert result.analyzed_columns == ["left", "right", "balanced"]
    assert result.analyzed_pair_count == 3
    assert result.pairs[0].left_column == "left"
    assert result.pairs[0].right_column == "right"
    assert result.pairs[0].cramers_v == pytest.approx(1.0)
    assert result.pairs[0].strong_association
    assert result.pairs[0].perfect_association
    assert result.pairs[0].degrees_of_freedom == 1
    assert result.pairs[0].metadata["contingency_shape"] == [2, 2]
    assert result.pairs[0].low_expected_frequency_cell_count == 4
    assert "low_expected_frequency_warning" in result.pairs[0].metadata
    assert [pair.left_column for pair in result.strong_pairs] == ["left"]
    assert [pair.left_column for pair in result.perfect_pairs] == ["left"]
    pd.testing.assert_frame_equal(frame, before)


def test_missing_values_are_dropped_or_counted_as_a_category() -> None:
    frame = pd.DataFrame(
        {
            "left": ["a", "a", "b", "b", None, "a"],
            "right": ["x", "x", "y", "y", "x", None],
        }
    )
    columns = AnalysisColumnGroups(categorical=["left", "right"])

    dropped = CategoricalAssociationAnalyzer(
        CategoricalAssociationOptions(
            minimum_valid_pairs=5,
        )
    ).analyze(
        contract=_contract(frame),
        columns=columns,
    )
    included = CategoricalAssociationAnalyzer(
        CategoricalAssociationOptions(
            minimum_valid_pairs=5,
            include_missing_as_category=True,
            missing_category_label="<missing>",
        )
    ).analyze(
        contract=_contract(frame),
        columns=columns,
    )

    assert dropped.skipped_pair_count == 1
    assert dropped.pairs == []
    assert included.analyzed_pair_count == 1
    assert included.pairs[0].valid_pair_count == 6
    assert included.pairs[0].left_unique_count == 3
    assert included.pairs[0].right_unique_count == 3
    assert included.metadata["include_missing_as_category"]


def test_unusable_columns_and_boolean_exclusion_are_reported() -> None:
    frame = pd.DataFrame(
        {
            "too_few": ["a", "b", None, None, None, None],
            "constant": ["same"] * 6,
            "high_cardinality": list("abcdef"),
            "usable": ["x", "y"] * 3,
            "flag": [True, False] * 3,
        }
    )
    result = CategoricalAssociationAnalyzer(
        CategoricalAssociationOptions(
            minimum_valid_pairs=4,
            maximum_categories_per_column=3,
            include_boolean_columns=False,
        )
    ).analyze(
        contract=_contract(frame),
        columns=AnalysisColumnGroups(
            categorical=[
                "too_few",
                "constant",
                "high_cardinality",
                "usable",
            ],
            boolean=["flag"],
        ),
    )

    assert result.analyzed_columns == ["usable"]
    assert set(result.skipped_columns) == {
        "too_few",
        "constant",
        "high_cardinality",
    }
    assert result.skipped_columns["too_few"] == (
        "The column contains too few valid values."
    )
    assert result.skipped_columns["constant"].startswith(
        "The column is constant"
    )
    assert result.skipped_columns["high_cardinality"].startswith(
        "The column exceeds"
    )


def test_jointly_insufficient_rows_skip_a_pair() -> None:
    frame = pd.DataFrame(
        {
            "left": ["a", "b", "a", "b", None, None],
            "right": [None, None, "x", "y", "x", "y"],
        }
    )

    result = CategoricalAssociationAnalyzer(
        CategoricalAssociationOptions(
            minimum_valid_pairs=3,
        )
    ).analyze(
        contract=_contract(frame),
        columns=AnalysisColumnGroups(
            categorical=["left", "right"]
        ),
    )

    assert result.analyzed_pair_count == 0
    assert result.skipped_pair_count == 1


def test_pair_limit_and_disabled_analysis_are_explicit() -> None:
    frame = pd.DataFrame(
        {
            "first": ["a", "a", "b", "b", "a", "b"],
            "second": ["a", "a", "b", "b", "a", "b"],
            "third": ["x", "y", "x", "y", "x", "y"],
        }
    )
    columns = AnalysisColumnGroups(
        categorical=["first", "second", "third"]
    )
    limited = CategoricalAssociationAnalyzer(
        CategoricalAssociationOptions(maximum_pairs=1)
    ).analyze(
        contract=_contract(frame),
        columns=columns,
    )
    disabled = CategoricalAssociationAnalyzer(
        CategoricalAssociationOptions(enabled=False)
    ).analyze(
        contract=SimpleNamespace(),
        columns=columns,
    )

    assert limited.analyzed_pair_count == 1
    assert limited.metadata["pair_limit_reached"]
    assert disabled.pairs == []
    assert disabled.metadata == {"enabled": False}


def test_association_options_reject_invalid_limits_and_thresholds() -> None:
    with pytest.raises(ValueError, match="minimum_valid_pairs"):
        CategoricalAssociationOptions(minimum_valid_pairs=0)

    with pytest.raises(ValueError, match="maximum_categories_per_column"):
        CategoricalAssociationOptions(maximum_categories_per_column=1)

    with pytest.raises(ValueError, match="strong_cramers_v_threshold"):
        CategoricalAssociationOptions(strong_cramers_v_threshold=1.1)

    with pytest.raises(ValueError, match="perfect_cramers_v_threshold"):
        CategoricalAssociationOptions(
            strong_cramers_v_threshold=0.8,
            perfect_cramers_v_threshold=0.7,
        )

    with pytest.raises(ValueError, match="minimum_expected_frequency"):
        CategoricalAssociationOptions(minimum_expected_frequency=-1.0)

    with pytest.raises(ValueError, match="missing_category_label"):
        CategoricalAssociationOptions(missing_category_label="")

    with pytest.raises(ValueError, match="maximum_pairs"):
        CategoricalAssociationOptions(maximum_pairs=0)
