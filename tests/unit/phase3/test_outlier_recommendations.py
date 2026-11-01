from __future__ import annotations

import pytest

from mlprep.phases.phase3.profiling.numeric_profile import (
    NumericColumnProfile,
)
from mlprep.phases.phase3.recommendations.outlier_recommendations import (
    OutlierRecommendationBuilder,
    OutlierRecommendationOptions,
)


def _profile(
    column: str,
    *,
    skewness: float | None = 0.0,
    infinite_count: int = 0,
    first_quartile: float | None = 2.0,
    third_quartile: float | None = 6.0,
    interquartile_range: float | None = 4.0,
) -> NumericColumnProfile:
    return NumericColumnProfile(
        column=column,
        dtype="float64",
        row_count=10,
        nonmissing_count=10,
        missing_count=0,
        missing_percentage=0.0,
        unique_count=10,
        uniqueness_ratio=1.0,
        minimum=0.0,
        maximum=10.0,
        mean=5.0,
        median=5.0,
        standard_deviation=2.0,
        variance=4.0,
        first_quartile=first_quartile,
        third_quartile=third_quartile,
        interquartile_range=interquartile_range,
        median_absolute_deviation=1.0,
        skewness=skewness,
        kurtosis=0.0,
        zero_count=1,
        zero_percentage=10.0,
        negative_count=0,
        negative_percentage=0.0,
        positive_count=9,
        infinite_count=infinite_count,
    )


def test_builder_selects_treatment_by_skew_and_invalid_values() -> None:
    result = OutlierRecommendationBuilder().build(
        numeric_profiles={
            "infinite": _profile(
                "infinite",
                skewness=0.2,
                infinite_count=1,
            ),
            "extreme": _profile(
                "extreme",
                skewness=5.0,
            ),
            "skewed": _profile(
                "skewed",
                skewness=-1.5,
            ),
            "ordinary": _profile(
                "ordinary",
                skewness=0.8,
            ),
            "no_quartiles": _profile(
                "no_quartiles",
                first_quartile=None,
                third_quartile=None,
                interquartile_range=None,
            ),
        }
    )

    recommendations = result.recommendations
    assert list(recommendations) == [
        "infinite",
        "extreme",
        "skewed",
        "ordinary",
    ]
    assert recommendations["infinite"].treatment == "set_missing"
    assert recommendations["infinite"].priority == 1.0
    assert "infinite values" in recommendations["infinite"].reason
    assert recommendations["extreme"].treatment == "winsorize"
    assert recommendations["extreme"].priority == 1.0
    assert recommendations["extreme"].warnings == [
        "Do not remove validation or test rows."
    ]
    assert recommendations["skewed"].treatment == "robust_scale"
    assert recommendations["skewed"].priority == 0.5
    assert recommendations["ordinary"].treatment == "report_only"
    assert recommendations["ordinary"].priority == 0.3
    assert recommendations["ordinary"].add_indicator is False
    assert recommendations["extreme"].add_indicator
    assert recommendations["infinite"].suggested_lower_bound == -4.0
    assert recommendations["infinite"].suggested_upper_bound == 12.0
    assert recommendations["infinite"].metadata["infinite_count"] == 1
    assert result.treatment_columns == [
        "infinite",
        "extreme",
        "skewed",
    ]
    assert result.domain_review_columns == ["infinite"]


def test_custom_bounds_indicator_policy_and_disabled_builder() -> None:
    result = OutlierRecommendationBuilder(
        OutlierRecommendationOptions(
            iqr_multiplier=2.0,
            recommend_indicator=False,
        )
    ).build(
        numeric_profiles={
            "feature": _profile(
                "feature",
                skewness=2.0,
            )
        }
    )

    recommendation = result.recommendations["feature"]
    assert recommendation.treatment == "robust_scale"
    assert recommendation.priority == pytest.approx(2.0 / 3.0)
    assert recommendation.suggested_lower_bound == -6.0
    assert recommendation.suggested_upper_bound == 14.0
    assert recommendation.add_indicator is False
    assert result.treatment_columns == ["feature"]

    disabled = OutlierRecommendationBuilder(
        OutlierRecommendationOptions(enabled=False)
    ).build(numeric_profiles={"feature": _profile("feature")})
    assert disabled.recommendations == {}
    assert disabled.treatment_columns == []
    assert disabled.domain_review_columns == []


def test_option_validation() -> None:
    with pytest.raises(ValueError, match="iqr_multiplier"):
        OutlierRecommendationOptions(iqr_multiplier=0.0)
    with pytest.raises(ValueError, match="high_skewness_threshold"):
        OutlierRecommendationOptions(high_skewness_threshold=-0.1)
    with pytest.raises(ValueError, match="extreme_skewness_threshold"):
        OutlierRecommendationOptions(
            high_skewness_threshold=2.0,
            extreme_skewness_threshold=1.0,
        )
