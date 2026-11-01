from __future__ import annotations

import pytest

from mlprep.phases.phase3.diagnostics.high_cardinality import (
    HighCardinalityDiagnostic,
    HighCardinalityDiagnosticResult,
)
from mlprep.phases.phase3.diagnostics.rare_categories import (
    RareCategoryDiagnostic,
    RareCategoryDiagnosticResult,
)
from mlprep.phases.phase3.recommendations.encoding_recommendations import (
    EncodingRecommendationBuilder,
    EncodingRecommendationOptions,
)


def _cardinality(
    column: str,
    *,
    level: str,
    encoding: str,
    unique_count: int,
    uniqueness_ratio: float,
) -> HighCardinalityDiagnostic:
    return HighCardinalityDiagnostic(
        column=column,
        row_count=100,
        nonmissing_count=100,
        unique_count=unique_count,
        uniqueness_ratio=uniqueness_ratio,
        level=level,
        identifier_like=level == "identifier_like",
        suggested_encoding=encoding,
        reason=f"{column} diagnostic reason",
    )


def test_builder_maps_encoding_advice_rare_grouping_and_fit_requirements() -> None:
    cardinality = HighCardinalityDiagnosticResult(
        diagnostics={
            "low": _cardinality(
                "low",
                level="low",
                encoding="one_hot",
                unique_count=3,
                uniqueness_ratio=0.03,
            ),
            "medium": _cardinality(
                "medium",
                level="medium",
                encoding="ordinal",
                unique_count=20,
                uniqueness_ratio=0.2,
            ),
            "high": _cardinality(
                "high",
                level="high",
                encoding="hashing",
                unique_count=200,
                uniqueness_ratio=1.0,
            ),
            "target": _cardinality(
                "target",
                level="high",
                encoding="target_encoding",
                unique_count=50,
                uniqueness_ratio=0.5,
            ),
            "text": _cardinality(
                "text",
                level="medium",
                encoding="text_vectorization",
                unique_count=30,
                uniqueness_ratio=0.3,
            ),
            "identifier": _cardinality(
                "identifier",
                level="identifier_like",
                encoding="exclude",
                unique_count=100,
                uniqueness_ratio=1.0,
            ),
            "review": _cardinality(
                "review",
                level="medium",
                encoding="review",
                unique_count=20,
                uniqueness_ratio=0.2,
            ),
        }
    )
    rare_categories = RareCategoryDiagnosticResult(
        diagnostics={
            "high": RareCategoryDiagnostic(
                column="high",
                recommend_grouping=True,
                rare_categories=["rare"],
            ),
            "target": RareCategoryDiagnostic(
                column="target",
                recommend_grouping=True,
                rare_categories=["other"],
            ),
            "text": RareCategoryDiagnostic(
                column="text",
                recommend_grouping=False,
            ),
        },
        grouping_candidates=["high", "target"],
    )

    result = EncodingRecommendationBuilder(
        EncodingRecommendationOptions(
            rare_category_label="<rare>",
            unknown_category_policy="ignore",
        )
    ).build(
        cardinality=cardinality,
        rare_categories=rare_categories,
    )
    recommendations = result.recommendations

    assert {
        column: recommendation.method
        for column, recommendation in recommendations.items()
    } == {
        "low": "one_hot",
        "medium": "ordinal",
        "high": "hashing",
        "target": "target_encoding",
        "text": "tfidf",
        "identifier": "exclude",
        "review": "review",
    }
    assert recommendations["low"].priority == 0.4
    assert recommendations["medium"].priority == 0.65
    assert recommendations["high"].priority == 0.85
    assert recommendations["identifier"].priority == 1.0
    assert recommendations["high"].group_rare_categories
    assert recommendations["target"].group_rare_categories
    assert not recommendations["text"].group_rare_categories
    assert recommendations["high"].rare_category_label == "<rare>"
    assert recommendations["high"].unknown_category_policy == "ignore"
    assert recommendations["target"].requires_cross_fitting
    assert recommendations["target"].warnings
    assert recommendations["identifier"].requires_training_fit is False
    assert recommendations["review"].requires_training_fit
    assert result.cross_fitting_columns == ["target"]
    assert result.rare_grouping_columns == ["high", "target"]
    assert result.excluded_columns == ["identifier"]


@pytest.mark.parametrize(
    ("options", "expected_reason"),
    [
        (
            EncodingRecommendationOptions(
                allow_target_encoding=False
            ),
            "Target encoding is disabled",
        ),
        (
            EncodingRecommendationOptions(
                prefer_frequency_over_target_encoding=True
            ),
            "Frequency encoding is preferred",
        ),
    ],
)
def test_target_encoding_can_be_replaced_by_frequency(
    options: EncodingRecommendationOptions,
    expected_reason: str,
) -> None:
    result = EncodingRecommendationBuilder(options).build(
        cardinality=HighCardinalityDiagnosticResult(
            diagnostics={
                "category": _cardinality(
                    "category",
                    level="high",
                    encoding="target_encoding",
                    unique_count=30,
                    uniqueness_ratio=0.3,
                )
            }
        ),
        rare_categories=RareCategoryDiagnosticResult(),
    )

    recommendation = result.recommendations["category"]
    assert recommendation.method == "frequency"
    assert expected_reason in recommendation.reason
    assert not recommendation.requires_cross_fitting
    assert recommendation.warnings == []
    assert result.cross_fitting_columns == []


def test_disabled_builder_and_option_validation() -> None:
    disabled = EncodingRecommendationBuilder(
        EncodingRecommendationOptions(enabled=False)
    ).build(
        cardinality=HighCardinalityDiagnosticResult(),
        rare_categories=RareCategoryDiagnosticResult(),
    )

    assert disabled.recommendations == {}
    assert disabled.cross_fitting_columns == []
    assert disabled.rare_grouping_columns == []
    assert disabled.excluded_columns == []

    with pytest.raises(ValueError, match="rare_category_label"):
        EncodingRecommendationOptions(rare_category_label="")
    with pytest.raises(ValueError, match="unknown_category_policy"):
        EncodingRecommendationOptions(unknown_category_policy="")
