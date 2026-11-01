from __future__ import annotations

import pytest

from mlprep.phases.phase3.diagnostics.skewness import (
    SkewnessDiagnostic,
    SkewnessDiagnosticResult,
)
from mlprep.phases.phase3.recommendations.transformation_recommendations import (
    TransformationRecommendationBuilder,
    TransformationRecommendationOptions,
)


def _diagnostic(
    column: str,
    *,
    level: str,
    suggestion: str,
    skewness: float | None,
    absolute_skewness: float | None,
) -> SkewnessDiagnostic:
    return SkewnessDiagnostic(
        column=column,
        valid_count=20,
        unique_count=10,
        minimum=-3.0,
        maximum=10.0,
        skewness=skewness,
        absolute_skewness=absolute_skewness,
        level=level,
        contains_zero=True,
        contains_negative=True,
        suggested_transformation=suggestion,
        reason=f"{column} diagnostic reason",
    )


def test_builder_maps_skewness_suggestions_and_result_groups() -> None:
    skewness = SkewnessDiagnosticResult(
        diagnostics={
            "symmetric": _diagnostic(
                "symmetric",
                level="symmetric",
                suggestion="none",
                skewness=0.2,
                absolute_skewness=0.2,
            ),
            "log": _diagnostic(
                "log",
                level="moderate",
                suggestion="log",
                skewness=0.8,
                absolute_skewness=0.8,
            ),
            "log1p": _diagnostic(
                "log1p",
                level="moderate",
                suggestion="log1p",
                skewness=0.9,
                absolute_skewness=0.9,
            ),
            "box_cox": _diagnostic(
                "box_cox",
                level="high",
                suggestion="box_cox",
                skewness=1.5,
                absolute_skewness=1.5,
            ),
            "yeo_johnson": _diagnostic(
                "yeo_johnson",
                level="high",
                suggestion="yeo_johnson",
                skewness=-2.0,
                absolute_skewness=2.0,
            ),
            "quantile": _diagnostic(
                "quantile",
                level="high",
                suggestion="quantile",
                skewness=4.0,
                absolute_skewness=4.0,
            ),
            "review": _diagnostic(
                "review",
                level="undefined",
                suggestion="review",
                skewness=None,
                absolute_skewness=None,
            ),
        }
    )

    result = TransformationRecommendationBuilder(
        TransformationRecommendationOptions(
            preserve_original_features=True
        )
    ).build(skewness=skewness)

    recommendations = result.recommendations
    assert {
        column: recommendation.method
        for column, recommendation in recommendations.items()
    } == {
        "symmetric": "none",
        "log": "log",
        "log1p": "log1p",
        "box_cox": "box_cox",
        "yeo_johnson": "yeo_johnson",
        "quantile": "quantile",
        "review": "review",
    }
    assert recommendations["symmetric"].priority == 0.1
    assert recommendations["log"].priority == 0.5
    assert recommendations["box_cox"].priority == 0.8
    assert recommendations["quantile"].priority == 1.0
    assert recommendations["review"].priority == 0.4
    assert result.transformation_columns == [
        "log",
        "log1p",
        "box_cox",
        "yeo_johnson",
        "quantile",
    ]
    assert result.review_columns == ["review"]
    assert result.no_transformation_columns == ["symmetric"]
    assert result.fitted_transformation_columns == [
        "box_cox",
        "yeo_johnson",
        "quantile",
    ]
    assert all(
        recommendation.preserve_original_feature
        for recommendation in recommendations.values()
    )
    assert recommendations["box_cox"].requires_training_fit
    assert not recommendations["log"].requires_training_fit
    assert recommendations["box_cox"].prerequisites
    assert recommendations["box_cox"].warnings
    assert recommendations["review"].warnings
    assert result.metadata == {
        "enabled": True,
        "recommendation_count": 7,
        "transformation_count": 5,
        "review_count": 1,
        "no_transformation_count": 1,
    }


@pytest.mark.parametrize(
    "method",
    ["standard_scale", "robust_scale"],
)
def test_symmetric_features_can_receive_configured_scaling(
    method: str,
) -> None:
    result = TransformationRecommendationBuilder(
        TransformationRecommendationOptions(
            recommend_scaling_for_symmetric_numeric=True,
            symmetric_scaling_method=method,
        )
    ).build(
        skewness=SkewnessDiagnosticResult(
            diagnostics={
                "feature": _diagnostic(
                    "feature",
                    level="symmetric",
                    suggestion="none",
                    skewness=0.1,
                    absolute_skewness=0.1,
                )
            }
        )
    )

    recommendation = result.recommendations["feature"]
    assert recommendation.method == method
    assert recommendation.priority == 0.3
    assert recommendation.requires_training_fit
    assert result.transformation_columns == ["feature"]
    assert result.fitted_transformation_columns == ["feature"]
    assert recommendation.prerequisites
    assert recommendation.warnings


def test_priority_filter_and_no_transformation_option() -> None:
    skewness = SkewnessDiagnosticResult(
        diagnostics={
            "symmetric": _diagnostic(
                "symmetric",
                level="symmetric",
                suggestion="none",
                skewness=0.1,
                absolute_skewness=0.1,
            ),
            "high": _diagnostic(
                "high",
                level="high",
                suggestion="quantile",
                skewness=4.0,
                absolute_skewness=4.0,
            ),
        }
    )

    result = TransformationRecommendationBuilder(
        TransformationRecommendationOptions(
            minimum_priority=1.0,
            include_no_transformation_recommendations=False,
        )
    ).build(skewness=skewness)

    assert list(result.recommendations) == ["high"]
    assert result.recommendations["high"].priority == 1.0
    assert result.no_transformation_columns == []


def test_disabled_builder_and_invalid_options() -> None:
    disabled = TransformationRecommendationBuilder(
        TransformationRecommendationOptions(enabled=False)
    ).build(skewness=SkewnessDiagnosticResult())

    assert disabled.recommendations == {}
    assert disabled.metadata == {"enabled": False}

    with pytest.raises(ValueError, match="minimum_priority"):
        TransformationRecommendationOptions(minimum_priority=1.1)
    with pytest.raises(ValueError, match="symmetric_priority"):
        TransformationRecommendationOptions(symmetric_priority=-0.1)
    with pytest.raises(ValueError, match="scaling_priority"):
        TransformationRecommendationOptions(scaling_priority=1.1)
    with pytest.raises(ValueError, match="moderate_skewness_priority"):
        TransformationRecommendationOptions(
            moderate_skewness_priority=-0.1
        )
    with pytest.raises(ValueError, match="high_skewness_priority"):
        TransformationRecommendationOptions(
            high_skewness_priority=1.1
        )
    with pytest.raises(ValueError, match="undefined_skewness_priority"):
        TransformationRecommendationOptions(
            undefined_skewness_priority=-0.1
        )
    with pytest.raises(ValueError, match="extreme_skewness_reference"):
        TransformationRecommendationOptions(
            extreme_skewness_reference=0.0
        )
