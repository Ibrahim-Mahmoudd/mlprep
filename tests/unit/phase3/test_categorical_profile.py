from __future__ import annotations

from math import log2
from types import SimpleNamespace

import pandas as pd
import pytest

from mlprep.phases.phase3.contract.column_selection import (
    AnalysisColumnGroups,
)
from mlprep.phases.phase3.profiling.categorical_profile import (
    CategoricalProfiler,
    CategoricalProfilerOptions,
)


def _contract(features: pd.DataFrame) -> SimpleNamespace:
    return SimpleNamespace(
        training_features=lambda: features.copy(deep=True),
    )


def test_profile_reports_distribution_rare_categories_and_limits() -> None:
    features = pd.DataFrame(
        {
            "category": ["A"] * 6 + ["B"] * 3 + ["C", None],
            "flag": [True, False] * 5 + [True],
            "ignored": list(range(11)),
        }
    )
    before = features.copy(deep=True)

    profiles = CategoricalProfiler(
        CategoricalProfilerOptions(
            maximum_categories_in_profile=2,
            rare_category_minimum_count=2,
            rare_category_maximum_ratio=0.1,
        )
    ).profile(
        contract=_contract(features),
        columns=AnalysisColumnGroups(
            categorical=["category", "missing_column"],
            boolean=["flag", "category"],
        ),
    )

    pd.testing.assert_frame_equal(features, before)
    assert list(profiles) == ["category", "flag"]
    profile = profiles["category"]
    assert profile.row_count == 11
    assert profile.nonmissing_count == 10
    assert profile.missing_count == 1
    assert profile.missing_percentage == pytest.approx(100 / 11)
    assert profile.unique_count == 3
    assert profile.uniqueness_ratio == pytest.approx(0.3)
    assert profile.dominant_value == "A"
    assert profile.dominant_value_count == 6
    assert profile.dominant_value_percentage == 60.0
    assert profile.entropy == pytest.approx(
        -(
            0.6 * log2(0.6)
            + 0.3 * log2(0.3)
            + 0.1 * log2(0.1)
        )
    )
    assert 0.0 <= profile.normalized_entropy <= 1.0
    assert profile.rare_category_count == 1
    assert profile.rare_categories == ["C"]
    assert profile.category_frequencies == [
        {"value": "A", "count": 6, "percentage": 60.0},
        {"value": "B", "count": 3, "percentage": 30.0},
    ]
    assert not profile.high_cardinality
    assert not profile.identifier_like
    assert profiles["flag"].unique_count == 2


def test_missing_values_can_be_profiled_as_a_category() -> None:
    features = pd.DataFrame(
        {"category": ["a", "a", "b", None]}
    )

    profile = CategoricalProfiler(
        CategoricalProfilerOptions(
            include_missing_as_category=True,
            rare_category_minimum_count=2,
            rare_category_maximum_ratio=0.0,
        )
    ).profile(
        contract=_contract(features),
        columns=AnalysisColumnGroups(categorical=["category"]),
    )["category"]

    assert profile.unique_count == 2
    assert profile.missing_count == 1
    assert profile.dominant_value == "a"
    assert profile.dominant_value_count == 2
    assert profile.dominant_value_percentage == 50.0
    assert profile.rare_categories == ["b", "__MISSING__"]
    missing_frequency = next(
        item
        for item in profile.category_frequencies
        if item["value"] == "__MISSING__"
    )
    assert missing_frequency == {
        "value": "__MISSING__",
        "count": 1,
        "percentage": 25.0,
    }


def test_empty_and_all_missing_values_have_finite_zero_entropy() -> None:
    empty = CategoricalProfiler()._profile_column(
        column="empty",
        series=pd.Series([], dtype=object),
    )
    all_missing = CategoricalProfiler(
        CategoricalProfilerOptions(
            include_missing_as_category=True
        )
    )._profile_column(
        column="missing",
        series=pd.Series([None, None], dtype=object),
    )

    assert empty.row_count == 0
    assert empty.missing_percentage == 0.0
    assert empty.dominant_value is None
    assert empty.entropy == 0.0
    assert empty.normalized_entropy == 0.0
    assert empty.category_frequencies == []

    assert all_missing.row_count == 2
    assert all_missing.nonmissing_count == 0
    assert all_missing.missing_percentage == 100.0
    assert all_missing.dominant_value == "__MISSING__"
    assert all_missing.dominant_value_count == 2
    assert all_missing.normalized_entropy == 0.0
    assert all_missing.rare_categories == ["__MISSING__"]


def test_cardinality_thresholds_disabled_profiler_and_invalid_options() -> None:
    features = pd.DataFrame(
        {
            "identifier": [f"id-{index}" for index in range(10)],
            "category": ["a", "b"] * 5,
        }
    )
    result = CategoricalProfiler(
        CategoricalProfilerOptions(
            high_cardinality_threshold=10,
            identifier_uniqueness_threshold=1.0,
        )
    ).profile(
        contract=_contract(features),
        columns=AnalysisColumnGroups(
            categorical=["identifier", "category"]
        ),
    )

    assert result["identifier"].high_cardinality
    assert result["identifier"].identifier_like
    assert result["identifier"].normalized_entropy == 1.0
    assert not result["category"].high_cardinality
    assert not result["category"].identifier_like

    disabled = CategoricalProfiler(
        CategoricalProfilerOptions(enabled=False)
    ).profile(
        contract=_contract(features),
        columns=AnalysisColumnGroups(categorical=["category"]),
    )
    assert disabled == {}

    with pytest.raises(ValueError, match="maximum_categories_in_profile"):
        CategoricalProfilerOptions(maximum_categories_in_profile=0)
    with pytest.raises(ValueError, match="rare_category_minimum_count"):
        CategoricalProfilerOptions(rare_category_minimum_count=0)
    with pytest.raises(ValueError, match="rare_category_maximum_ratio"):
        CategoricalProfilerOptions(rare_category_maximum_ratio=1.1)
    with pytest.raises(ValueError, match="identifier_uniqueness_threshold"):
        CategoricalProfilerOptions(identifier_uniqueness_threshold=-0.1)
