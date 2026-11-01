from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from mlprep.phases.phase3.contract.column_selection import (
    AnalysisColumnGroups,
)
from mlprep.phases.phase3.diagnostics.skewness import (
    SkewnessDiagnosticAnalyzer,
    SkewnessDiagnosticOptions,
)


def _contract(features: pd.DataFrame) -> SimpleNamespace:
    return SimpleNamespace(
        training_features=lambda: features.copy(deep=True),
    )


def _analyze(
    series: pd.Series,
    options: SkewnessDiagnosticOptions | None = None,
):
    features = series.to_frame(name="feature")
    before = features.copy(deep=True)
    result = SkewnessDiagnosticAnalyzer(options).analyze(
        contract=_contract(features),
        columns=AnalysisColumnGroups(numeric=["feature"]),
    )
    pd.testing.assert_frame_equal(features, before)
    return result.diagnostics["feature"]


def test_symmetric_moderate_and_high_skewness_levels_are_classified() -> None:
    symmetric = _analyze(
        pd.Series([-5.5, -4.5, -3.5, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5])
    )
    moderate = _analyze(
        pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15])
    )
    high = _analyze(
        pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 50])
    )

    assert symmetric.level == "symmetric"
    assert symmetric.suggested_transformation == "none"
    assert moderate.level == "moderate"
    assert moderate.suggested_transformation == "box_cox"
    assert high.level == "high"
    assert high.suggested_transformation == "box_cox"


@pytest.mark.parametrize(
    ("values", "expected", "contains_zero", "contains_negative"),
    [
        ([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 50], "log1p", True, False),
        ([-50, -10, -9, -8, -7, -6, -5, -4, -3, -2, -1, 0], "yeo_johnson", True, True),
    ],
)
def test_zero_and_negative_values_select_domain_safe_transforms(
    values: list[int],
    expected: str,
    contains_zero: bool,
    contains_negative: bool,
) -> None:
    diagnostic = _analyze(
        pd.Series(values),
        SkewnessDiagnosticOptions(
            recommend_quantile_above=4.0,
        ),
    )

    assert diagnostic.level == "high"
    assert diagnostic.suggested_transformation == expected
    assert diagnostic.contains_zero is contains_zero
    assert diagnostic.contains_negative is contains_negative


def test_extreme_skewness_recommends_quantile_transform() -> None:
    diagnostic = _analyze(
        pd.Series([0] * 10 + [1, 2, 1000]),
    )

    assert diagnostic.absolute_skewness >= 3.0
    assert diagnostic.suggested_transformation == "quantile"
    assert diagnostic.reason == "The distribution is extremely skewed."


def test_missing_infinite_and_non_numeric_values_are_safely_coerced() -> None:
    diagnostic = _analyze(
        pd.Series(
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, np.nan, np.inf, "bad"],
            dtype=object,
        )
    )

    assert diagnostic.valid_count == 10
    assert diagnostic.unique_count == 10
    assert diagnostic.minimum == 1.0
    assert diagnostic.maximum == 10.0
    assert diagnostic.skewness is not None
    assert np.isfinite(diagnostic.skewness)


def test_insufficient_values_and_unavailable_columns_are_reported() -> None:
    features = pd.DataFrame(
        {
            "too_short": [1.0, 2.0, np.inf],
            "too_few_unique": [2.0, 2.0, 2.0],
        }
    )
    result = SkewnessDiagnosticAnalyzer().analyze(
        contract=_contract(features),
        columns=AnalysisColumnGroups(
            numeric=["too_short", "too_few_unique", "missing"]
        ),
    )

    assert list(result.diagnostics) == ["too_short", "too_few_unique"]
    assert result.diagnostics["too_short"].level == "undefined"
    assert result.diagnostics["too_short"].valid_count == 2
    assert result.diagnostics["too_short"].minimum == 1.0
    assert result.diagnostics["too_few_unique"].level == "undefined"
    assert result.diagnostics["too_few_unique"].unique_count == 1
    assert result.diagnostics["too_short"].suggested_transformation == "review"
    assert result.transformation_candidates == {
        "too_short": "review",
        "too_few_unique": "review",
    }


def test_disabled_analyzer_and_invalid_options() -> None:
    disabled = SkewnessDiagnosticAnalyzer(
        SkewnessDiagnosticOptions(enabled=False)
    ).analyze(
        contract=_contract(pd.DataFrame({"feature": [1, 2, 3]})),
        columns=AnalysisColumnGroups(numeric=["feature"]),
    )

    assert disabled.diagnostics == {}
    assert disabled.transformation_candidates == {}

    with pytest.raises(ValueError, match="minimum_valid_count"):
        SkewnessDiagnosticOptions(minimum_valid_count=2)
    with pytest.raises(ValueError, match="minimum_unique_count"):
        SkewnessDiagnosticOptions(minimum_unique_count=1)
    with pytest.raises(ValueError, match="moderate_threshold"):
        SkewnessDiagnosticOptions(moderate_threshold=-0.1)
    with pytest.raises(ValueError, match="high_threshold"):
        SkewnessDiagnosticOptions(high_threshold=0.4)
    with pytest.raises(ValueError, match="recommend_quantile_above"):
        SkewnessDiagnosticOptions(recommend_quantile_above=0.8)
