from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest

from mlprep.phases.phase3.contract.column_selection import (
    AnalysisColumnGroups,
)
from mlprep.phases.phase3.diagnostics.rare_categories import (
    RareCategoryDiagnosticAnalyzer,
    RareCategoryDiagnosticOptions,
)


def _contract(
    training: pd.DataFrame,
    validation: pd.DataFrame | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        training_features=lambda: training.copy(deep=True),
        validation_features=lambda: (
            validation.copy(deep=True)
            if validation is not None
            else None
        ),
    )


def test_analyzer_reports_rare_and_partition_only_categories() -> None:
    training = pd.DataFrame(
        {
            "category": ["common"] * 5
            + ["usual"] * 3
            + ["rare", "training_only"],
            "flag": [True, False] * 5,
        }
    )
    validation = pd.DataFrame(
        {
            "category": [
                "common",
                "usual",
                "validation_only",
            ],
            "flag": [True, False, None],
        }
    )
    training_before = training.copy(deep=True)
    validation_before = validation.copy(deep=True)

    result = RareCategoryDiagnosticAnalyzer(
        RareCategoryDiagnosticOptions(
            minimum_count=2,
            maximum_ratio=0.0,
            grouping_category_count_threshold=2,
            grouping_observation_percentage_threshold=20.0,
            compare_validation=True,
            maximum_reported_categories=1,
            include_boolean_columns=True,
        )
    ).analyze(
        contract=_contract(training, validation),
        columns=AnalysisColumnGroups(
            categorical=["category", "missing_column"],
            boolean=["flag", "category"],
        ),
    )

    pd.testing.assert_frame_equal(training, training_before)
    pd.testing.assert_frame_equal(validation, validation_before)
    assert list(result.diagnostics) == ["category", "flag"]
    category = result.diagnostics["category"]
    assert category.valid_count == 10
    assert category.unique_count == 4
    assert category.rare_category_count == 2
    assert category.rare_observation_count == 2
    assert category.rare_observation_percentage == 20.0
    assert category.rare_categories == ["rare"]
    assert category.validation_only_categories == ["validation_only"]
    assert category.training_only_categories == ["rare"]
    assert category.recommend_grouping
    assert category.metadata["reported_rare_categories_truncated"]
    assert result.grouping_candidates == ["category"]
    assert result.unknown_category_risk_columns == ["category"]


def test_validation_comparison_can_be_skipped_and_boolean_selection_is_configurable(
) -> None:
    training = pd.DataFrame(
        {
            "category": ["a", "a", "rare"],
            "flag": [True, False, True],
        }
    )
    validation = pd.DataFrame(
        {
            "category": ["validation_only"],
            "flag": [False],
        }
    )

    result = RareCategoryDiagnosticAnalyzer(
        RareCategoryDiagnosticOptions(
            minimum_count=2,
            compare_validation=False,
            include_boolean_columns=False,
        )
    ).analyze(
        contract=_contract(training, validation),
        columns=AnalysisColumnGroups(
            categorical=["category"],
            boolean=["flag"],
        ),
    )

    assert list(result.diagnostics) == ["category"]
    assert result.diagnostics["category"].rare_categories == ["rare"]
    assert result.diagnostics["category"].validation_only_categories == []
    assert result.diagnostics["category"].training_only_categories == []
    assert result.unknown_category_risk_columns == []


def test_empty_data_disabled_analysis_and_option_validation() -> None:
    empty = RareCategoryDiagnosticAnalyzer().analyze(
        contract=_contract(
            pd.DataFrame({"category": pd.Series([], dtype=object)})
        ),
        columns=AnalysisColumnGroups(categorical=["category"]),
    )
    disabled = RareCategoryDiagnosticAnalyzer(
        RareCategoryDiagnosticOptions(enabled=False)
    ).analyze(
        contract=_contract(pd.DataFrame({"category": ["a"]})),
        columns=AnalysisColumnGroups(categorical=["category"]),
    )

    assert empty.diagnostics["category"].valid_count == 0
    assert empty.diagnostics["category"].unique_count == 0
    assert empty.diagnostics["category"].rare_categories == []
    assert empty.grouping_candidates == []
    assert disabled.diagnostics == {}

    with pytest.raises(ValueError, match="minimum_count"):
        RareCategoryDiagnosticOptions(minimum_count=0)
    with pytest.raises(ValueError, match="maximum_ratio"):
        RareCategoryDiagnosticOptions(maximum_ratio=1.1)
    with pytest.raises(ValueError, match="grouping_category_count_threshold"):
        RareCategoryDiagnosticOptions(grouping_category_count_threshold=0)
    with pytest.raises(
        ValueError,
        match="grouping_observation_percentage_threshold",
    ):
        RareCategoryDiagnosticOptions(
            grouping_observation_percentage_threshold=100.1
        )
