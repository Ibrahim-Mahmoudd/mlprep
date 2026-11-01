from __future__ import annotations

import pytest

from mlprep.phases.phase3.contract.column_selection import (
    AnalysisColumnGroups,
)
from mlprep.phases.phase3.profiling.missingness_profile import (
    ColumnMissingnessProfile,
    MissingnessProfile,
)
from mlprep.phases.phase3.profiling.numeric_profile import (
    NumericColumnProfile,
)
from mlprep.phases.phase3.recommendations.imputation_recommendations import (
    ImputationRecommendationBuilder,
    ImputationRecommendationOptions,
)
from mlprep.phases.phase3.relationships.missingness_associations import (
    MissingnessAssociationResult,
)


def _numeric_profile(
    column: str,
    skewness: float,
) -> NumericColumnProfile:
    return NumericColumnProfile(
        column=column,
        dtype="float64",
        row_count=10,
        nonmissing_count=8,
        missing_count=2,
        missing_percentage=20.0,
        unique_count=8,
        uniqueness_ratio=0.8,
        zero_count=0,
        zero_percentage=0.0,
        negative_count=0,
        negative_percentage=0.0,
        positive_count=8,
        infinite_count=0,
        skewness=skewness,
    )


def _missingness_profile(
    percentages: dict[str, float],
    completely_missing: set[str] | None = None,
) -> MissingnessProfile:
    all_missing = completely_missing or set()
    profiles = {
        column: ColumnMissingnessProfile(
            column=column,
            missing_count=int(percentage),
            missing_percentage=percentage,
            completely_missing=column in all_missing,
            has_missing_values=percentage > 0.0,
        )
        for column, percentage in percentages.items()
    }
    return MissingnessProfile(
        total_missing_cells=sum(
            profile.missing_count
            for profile in profiles.values()
        ),
        total_missing_percentage=0.0,
        rows_with_missing_values=0,
        rows_with_missing_percentage=0.0,
        maximum_missing_values_per_row=0,
        average_missing_values_per_row=0.0,
        columns=profiles,
    )


def test_builder_selects_methods_by_missingness_and_feature_family() -> None:
    missingness = _missingness_profile(
        {
            "complete": 0.0,
            "all_missing": 100.0,
            "high_missing": 80.0,
            "numeric_skewed": 40.0,
            "numeric_associated": 5.0,
            "numeric_ordinary": 20.0,
            "datetime": 10.0,
            "categorical": 20.0,
            "boolean": 15.0,
            "text": 20.0,
            "unknown": 10.0,
        },
        completely_missing={"all_missing"},
    )
    columns = AnalysisColumnGroups(
        numeric=[
            "numeric_skewed",
            "numeric_associated",
            "numeric_ordinary",
        ],
        datetime=["datetime"],
        categorical=["categorical"],
        boolean=["boolean"],
        text=["text"],
    )
    result = ImputationRecommendationBuilder(
        ImputationRecommendationOptions(
            high_missingness_threshold=0.8,
            indicator_missingness_threshold=0.1,
            missing_category_value="<missing>",
        )
    ).build(
        columns=columns,
        missingness=missingness,
        numeric_profiles={
            "numeric_skewed": _numeric_profile(
                "numeric_skewed",
                -1.5,
            ),
            "numeric_ordinary": _numeric_profile(
                "numeric_ordinary",
                0.2,
            ),
        },
        missingness_associations=MissingnessAssociationResult(
            strongly_associated_features=["numeric_associated"]
        ),
    )

    recommendations = result.recommendations
    assert list(recommendations) == [
        "all_missing",
        "high_missing",
        "numeric_skewed",
        "numeric_associated",
        "numeric_ordinary",
        "datetime",
        "categorical",
        "boolean",
        "text",
        "unknown",
    ]
    assert recommendations["all_missing"].method == "drop_feature"
    assert recommendations["all_missing"].priority == 1.0
    assert recommendations["high_missing"].method == "review"
    assert recommendations["high_missing"].priority == 0.95
    assert recommendations["numeric_skewed"].method == "median"
    assert "skewed" in recommendations["numeric_skewed"].reason
    assert recommendations["numeric_associated"].add_missing_indicator
    assert recommendations["numeric_associated"].constant_value is None
    assert recommendations["numeric_ordinary"].method == "median"
    assert "robust" in recommendations["numeric_ordinary"].reason
    assert recommendations["datetime"].method == "time_aware"
    assert recommendations["categorical"].method == "missing_category"
    assert recommendations["boolean"].method == "missing_category"
    assert recommendations["categorical"].constant_value == "<missing>"
    assert recommendations["text"].method == "constant"
    assert recommendations["text"].constant_value == "<missing>"
    assert recommendations["unknown"].method == "review"
    assert recommendations["unknown"].requires_training_fit is False
    assert result.missing_indicator_columns == [
        "all_missing",
        "high_missing",
        "numeric_skewed",
        "numeric_associated",
        "numeric_ordinary",
        "datetime",
        "categorical",
        "boolean",
        "text",
        "unknown",
    ]
    assert result.drop_review_columns == [
        "all_missing",
        "high_missing",
    ]
    assert recommendations["numeric_associated"].metadata[
        "target_associated"
    ]


def test_indicator_and_high_missingness_threshold_boundaries() -> None:
    missingness = _missingness_profile(
        {
            "high_boundary": 80.0,
            "indicator_boundary": 5.0,
            "below_indicator": 4.9,
        }
    )
    result = ImputationRecommendationBuilder(
        ImputationRecommendationOptions(
            high_missingness_threshold=0.8,
            indicator_missingness_threshold=0.05,
        )
    ).build(
        columns=AnalysisColumnGroups(numeric=["indicator_boundary"]),
        missingness=missingness,
        numeric_profiles={},
        missingness_associations=MissingnessAssociationResult(),
    )

    assert result.recommendations["high_boundary"].method == "review"
    assert result.recommendations["indicator_boundary"].add_missing_indicator
    assert not result.recommendations["below_indicator"].add_missing_indicator
    assert result.missing_indicator_columns == [
        "high_boundary",
        "indicator_boundary",
    ]


def test_disabled_builder_and_option_validation() -> None:
    disabled = ImputationRecommendationBuilder(
        ImputationRecommendationOptions(enabled=False)
    ).build(
        columns=AnalysisColumnGroups(),
        missingness=_missingness_profile({"feature": 50.0}),
        numeric_profiles={},
        missingness_associations=MissingnessAssociationResult(),
    )

    assert disabled.recommendations == {}
    assert disabled.missing_indicator_columns == []
    assert disabled.drop_review_columns == []

    with pytest.raises(ValueError, match="between zero and one"):
        ImputationRecommendationOptions(high_missingness_threshold=1.1)
    with pytest.raises(ValueError, match="between zero and one"):
        ImputationRecommendationOptions(indicator_missingness_threshold=-0.1)
    with pytest.raises(ValueError, match="skewness_median_threshold"):
        ImputationRecommendationOptions(skewness_median_threshold=-0.1)
    with pytest.raises(ValueError, match="missing_category_value"):
        ImputationRecommendationOptions(missing_category_value="")
