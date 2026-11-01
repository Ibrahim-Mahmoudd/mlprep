from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from mlprep.phases.phase3.contract.column_selection import (
    AnalysisColumnGroups,
)
from mlprep.phases.phase3.relationships.multicollinearity import (
    MulticollinearityAnalyzer,
    MulticollinearityOptions,
)


def _contract(features: pd.DataFrame) -> SimpleNamespace:
    return SimpleNamespace(
        train=SimpleNamespace(X=features),
    )


def test_perfect_linear_dependence_is_flagged_and_inputs_are_unchanged() -> None:
    values = np.arange(20, dtype=float)
    features = pd.DataFrame(
        {
            "first": values,
            "duplicate_scale": values * 2,
            "independent": np.tile([-1.0, 1.0], 10),
        },
        index=np.arange(100, 120),
    )
    before = features.copy(deep=True)

    result = MulticollinearityAnalyzer().analyze(
        contract=_contract(features),
        columns=AnalysisColumnGroups(
            numeric=[
                "first",
                "duplicate_scale",
                "independent",
            ]
        ),
    )

    vif_by_column = {
        item.column: item for item in result.vif_results
    }
    assert result.analyzed_columns == [
        "first",
        "duplicate_scale",
        "independent",
    ]
    assert result.singular
    assert result.matrix_rank == 2
    assert result.condition_number is None
    assert result.high_vif_columns == ["first", "duplicate_scale"]
    assert vif_by_column["first"].perfect_multicollinearity
    assert vif_by_column["duplicate_scale"].perfect_multicollinearity
    assert vif_by_column["first"].vif is None
    assert features.equals(before)


def test_noncollinear_features_have_finite_condition_and_low_vif() -> None:
    features = pd.DataFrame(
        {
            "linear": np.arange(12, dtype=float),
            "alternating": np.tile([0.0, 1.0, -1.0], 4),
        }
    )

    result = MulticollinearityAnalyzer().analyze(
        contract=_contract(features),
        columns=AnalysisColumnGroups(numeric=list(features.columns)),
    )

    assert not result.singular
    assert result.matrix_rank == 2
    assert result.condition_number is not None
    assert np.isfinite(result.condition_number)
    assert result.high_vif_columns == []
    assert all(
        item.vif is not None and item.vif < 10
        for item in result.vif_results
    )


def test_missing_values_are_median_filled_or_rows_are_dropped() -> None:
    features = pd.DataFrame(
        {
            "increasing": [
                1.0,
                2.0,
                3.0,
                4.0,
                5.0,
                6.0,
                np.nan,
                8.0,
                9.0,
                10.0,
                11.0,
                12.0,
            ],
            "alternating": [
                0.0,
                1.0,
                -1.0,
                0.0,
                1.0,
                -1.0,
                0.0,
                np.inf,
                -1.0,
                0.0,
                1.0,
                -1.0,
            ],
        }
    )
    columns = AnalysisColumnGroups(numeric=list(features.columns))

    filled = MulticollinearityAnalyzer().analyze(
        contract=_contract(features),
        columns=columns,
    )
    dropped = MulticollinearityAnalyzer(
        MulticollinearityOptions(
            fill_missing_with_median=False,
            minimum_complete_rows=10,
        )
    ).analyze(
        contract=_contract(features),
        columns=columns,
    )

    assert filled.metadata["row_count_used"] == 12
    assert dropped.metadata["row_count_used"] == 10
    assert len(filled.vif_results) == 2
    assert len(dropped.vif_results) == 2


def test_constants_and_insufficient_complete_rows_are_reported() -> None:
    features = pd.DataFrame(
        {
            "varying_a": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "varying_b": [0.0, 1.0, -1.0, 0.0, 1.0, -1.0],
            "constant": [5.0] * 6,
        }
    )

    after_constant_removal = MulticollinearityAnalyzer(
        MulticollinearityOptions(minimum_complete_rows=5)
    ).analyze(
        contract=_contract(features),
        columns=AnalysisColumnGroups(numeric=list(features.columns)),
    )
    insufficient = MulticollinearityAnalyzer(
        MulticollinearityOptions(
            fill_missing_with_median=False,
            minimum_complete_rows=7,
        )
    ).analyze(
        contract=_contract(features.assign(varying_a=[1, 2, np.nan, 4, 5, 6])),
        columns=AnalysisColumnGroups(
            numeric=["varying_a", "varying_b"],
        ),
    )

    assert after_constant_removal.analyzed_columns == [
        "varying_a",
        "varying_b",
    ]
    assert after_constant_removal.metadata["excluded_constant_columns"] == [
        "constant"
    ]
    assert insufficient.vif_results == []
    assert insufficient.metadata == {
        "excluded_constant_columns": [],
    }


def test_disabled_and_short_column_selections_return_without_analysis() -> None:
    features = pd.DataFrame({"first": [1.0, 2.0], "second": [2.0, 4.0]})

    disabled = MulticollinearityAnalyzer(
        MulticollinearityOptions(enabled=False)
    ).analyze(
        contract=_contract(features),
        columns=AnalysisColumnGroups(numeric=["first", "second"]),
    )
    short_selection = MulticollinearityAnalyzer().analyze(
        contract=_contract(features),
        columns=AnalysisColumnGroups(numeric=["missing", "first"]),
    )

    assert disabled.analyzed_columns == []
    assert disabled.vif_results == []
    assert short_selection.analyzed_columns == ["first"]
    assert short_selection.vif_results == []


@pytest.mark.parametrize(
    ("options", "message"),
    [
        ({"maximum_columns": 1}, "maximum_columns"),
        ({"minimum_complete_rows": 1}, "minimum_complete_rows"),
        ({"high_vif_threshold": 1.0}, "high_vif_threshold"),
        (
            {
                "high_vif_threshold": 10.0,
                "perfect_vif_threshold": 5.0,
            },
            "perfect_vif_threshold",
        ),
    ],
)
def test_invalid_options_are_rejected(
    options: dict[str, float | int],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        MulticollinearityOptions(**options)
