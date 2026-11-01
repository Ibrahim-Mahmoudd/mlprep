from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest

from mlprep.phases.phase3.recommendations.imputation_recommendations import (
    ImputationRecommendation,
)
from mlprep.phases.phase4.contract.imputation_plan import (
    ImputationFeatureFamily,
    ImputationMethodName,
    ImputationPlan,
    ImputationPlanBuilder,
    ImputationPlanEntry,
    ImputationPlanOptions,
    PlanEntryStatus,
)


def _recommendation(
    column: str,
    method: str,
    **values: object,
) -> ImputationRecommendation:
    return ImputationRecommendation(
        column=column,
        method=method,  # type: ignore[arg-type]
        priority=0.8,
        reason=f"Recommended {method} for {column}.",
        **values,
    )


def _contract(
    frame: pd.DataFrame,
    *,
    recommendations: dict[str, ImputationRecommendation] | None = None,
    families: dict[str, list[str]] | None = None,
) -> SimpleNamespace:
    family_columns = families or {}
    return SimpleNamespace(
        train=SimpleNamespace(X=frame),
        feature_columns=list(frame.columns),
        target_column="target",
        internal_row_id_column="__row_id__",
        group_columns=["group"],
        timestamp_column="timestamp",
        phase3_state=SimpleNamespace(
            columns=SimpleNamespace(
                numeric=family_columns.get("numeric", []),
                categorical=family_columns.get("categorical", []),
                boolean=family_columns.get("boolean", []),
                datetime=family_columns.get("datetime", []),
                text=family_columns.get("text", []),
                timedelta=family_columns.get("timedelta", []),
            )
        ),
        phase3_recommendations=SimpleNamespace(
            imputation=SimpleNamespace(
                recommendations=recommendations or {},
            )
        ),
    )


def test_recommendations_map_to_executable_training_fit_entries() -> None:
    frame = pd.DataFrame(
        {
            "numeric": [1.0, None],
            "category": ["a", None],
            "boolean": [True, None],
            "datetime": [pd.Timestamp("2025-01-01"), pd.NaT],
            "grouped": ["x", None],
            "modeled": [1.0, None],
            "drop": [None, None],
            "review": [None, None],
            "target": [0, 1],
            "__row_id__": ["r1", "r2"],
            "blocked": [1.0, None],
        }
    )
    before = frame.copy(deep=True)
    recommendations = {
        "numeric": _recommendation(
            "numeric",
            "median",
            add_missing_indicator=True,
        ),
        "category": _recommendation("category", "missing_category"),
        "boolean": _recommendation("boolean", "constant"),
        "datetime": _recommendation("datetime", "time_aware"),
        "grouped": _recommendation("grouped", "group_based"),
        "modeled": _recommendation("modeled", "model_based"),
        "drop": _recommendation("drop", "drop_feature"),
        "review": _recommendation("review", "review"),
    }
    contract = _contract(
        frame,
        recommendations=recommendations,
        families={
            "numeric": ["numeric", "modeled"],
            "categorical": ["category"],
            "boolean": ["boolean"],
            "datetime": ["datetime"],
        },
    )

    plan = ImputationPlanBuilder(
        ImputationPlanOptions(blocked_columns=["blocked"])
    ).build(contract=contract)

    assert list(plan.entries) == [
        "numeric",
        "category",
        "boolean",
        "datetime",
        "grouped",
        "modeled",
        "drop",
        "review",
    ]
    assert plan.entry("numeric").feature_family == ImputationFeatureFamily.NUMERIC
    assert plan.entry("numeric").add_missing_indicator
    assert plan.entry("category").constant_value == "__MISSING__"
    assert plan.entry("boolean").constant_value is False
    assert plan.entry("datetime").timestamp_column == "timestamp"
    assert plan.entry("grouped").group_columns == ["group"]
    assert plan.entry("drop").status == PlanEntryStatus.BLOCKED
    assert plan.entry("review").status == PlanEntryStatus.BLOCKED
    assert plan.indicator_columns == ["numeric"]
    assert plan.drop_feature_columns == ["drop"]
    assert plan.review_columns == ["review"]
    assert plan.skipped_columns == ["target", "__row_id__", "blocked"]
    assert len(plan.executable_entries()) == 6
    assert {entry.column for entry in plan.fitting_entries()} == {
        "numeric",
        "datetime",
        "grouped",
        "modeled",
    }
    assert plan.metadata == {
        "entry_count": 8,
        "training_fit_entry_count": 4,
        "indicator_count": 1,
        "drop_feature_count": 1,
        "review_count": 1,
    }
    pd.testing.assert_frame_equal(frame, before)


def test_unspecified_missing_features_use_family_defaults_in_order() -> None:
    frame = pd.DataFrame(
        {
            "numeric": [1.0, None],
            "categorical": ["a", None],
            "boolean": [True, None],
            "datetime": [pd.Timestamp("2025-01-01"), pd.NaT],
            "text": ["hello", None],
            "timedelta": [pd.Timedelta(days=1), pd.NaT],
            "unknown": ["x", None],
            "complete": [1.0, 2.0],
            "target": [0, 1],
            "__row_id__": ["a", "b"],
        }
    )
    contract = _contract(
        frame,
        families={
            "numeric": ["numeric"],
            "categorical": ["categorical"],
            "boolean": ["boolean"],
            "datetime": ["datetime"],
            "text": ["text"],
            "timedelta": ["timedelta"],
        },
    )

    plan = ImputationPlanBuilder().build(contract=contract)

    assert list(plan.entries) == [
        "numeric",
        "categorical",
        "boolean",
        "datetime",
        "text",
        "timedelta",
        "unknown",
    ]
    assert plan.entry("numeric").method == ImputationMethodName.MEDIAN
    assert (
        plan.entry("categorical").method
        == ImputationMethodName.MISSING_CATEGORY
    )
    assert plan.entry("categorical").constant_value == "__MISSING__"
    assert (
        plan.entry("boolean").method
        == ImputationMethodName.MOST_FREQUENT
    )
    assert plan.entry("datetime").method == ImputationMethodName.MEDIAN
    assert plan.entry("text").method == ImputationMethodName.CONSTANT
    assert plan.entry("text").constant_value == ""
    assert plan.entry("timedelta").method == ImputationMethodName.MEDIAN
    assert plan.entry("unknown").method == ImputationMethodName.REVIEW
    assert plan.review_columns == ["unknown"]
    assert "complete" not in plan.entries
    assert plan.skipped_columns == ["target", "__row_id__"]


def test_unspecified_defaults_can_be_disabled_and_overridden() -> None:
    frame = pd.DataFrame(
        {
            "numeric": [1.0, None],
            "unplanned": [None, 2.0],
            "target": [0, 1],
            "__row_id__": ["a", "b"],
        }
    )
    recommendations = {
        "numeric": _recommendation(
            "numeric",
            "median",
        )
    }
    contract = _contract(
        frame,
        recommendations=recommendations,
        families={"numeric": ["numeric"]},
    )

    plan = ImputationPlanBuilder(
        ImputationPlanOptions(
            create_unspecified_missing_plan_entries=False,
            method_overrides={
                "numeric": ImputationMethodName.CONSTANT,
            },
            constant_overrides={"numeric": 7},
            indicator_overrides={"numeric": True},
        )
    ).build(contract=contract)

    assert list(plan.entries) == ["numeric"]
    assert plan.entry("numeric").method == ImputationMethodName.CONSTANT
    assert plan.entry("numeric").constant_value == 7
    assert plan.entry("numeric").add_missing_indicator
    assert not plan.entry("numeric").requires_training_fit
    assert plan.indicator_columns == ["numeric"]


def test_drop_and_review_recommendations_can_be_explicitly_enabled() -> None:
    frame = pd.DataFrame(
        {"drop": [None, None], "review": [1.0, None], "target": [0, 1]}
    )
    contract = _contract(
        frame,
        recommendations={
            "drop": _recommendation("drop", "drop_feature"),
            "review": _recommendation("review", "review"),
        },
    )

    plan = ImputationPlanBuilder(
        ImputationPlanOptions(
            allow_drop_feature=True,
            allow_review_entries=True,
        )
    ).build(contract=contract)

    assert plan.entry("drop").status == PlanEntryStatus.PLANNED
    assert plan.entry("review").status == PlanEntryStatus.PLANNED
    assert plan.drop_feature_columns == ["drop"]
    assert plan.review_columns == ["review"]
    assert plan.entry("drop").warnings == []
    assert plan.entry("review").warnings == []


def test_plan_queries_and_entry_validation_enforce_method_requirements() -> None:
    median = ImputationPlanEntry(
        column="numeric",
        feature_family=ImputationFeatureFamily.NUMERIC,
        method=ImputationMethodName.MEDIAN,
        reason="Fit median on training data.",
    )
    constant = ImputationPlanEntry(
        column="category",
        feature_family=ImputationFeatureFamily.CATEGORICAL,
        method=ImputationMethodName.CONSTANT,
        constant_value="missing",
        requires_training_fit=False,
        reason="Use a fixed token.",
    )
    blocked = ImputationPlanEntry(
        column="review",
        feature_family=ImputationFeatureFamily.UNKNOWN,
        method=ImputationMethodName.REVIEW,
        status=PlanEntryStatus.BLOCKED,
        reason="Review the feature.",
    )
    plan = ImputationPlan(
        entries={
            "numeric": median,
            "category": constant,
            "review": blocked,
        }
    )

    assert plan.entry("numeric") is median
    assert [entry.column for entry in plan.executable_entries()] == [
        "numeric",
        "category",
    ]
    assert [entry.column for entry in plan.fitting_entries()] == ["numeric"]
    with pytest.raises(KeyError, match="missing"):
        plan.entry("missing")

    with pytest.raises(ValueError, match="constant_value"):
        ImputationPlanEntry(
            column="feature",
            feature_family=ImputationFeatureFamily.NUMERIC,
            method=ImputationMethodName.CONSTANT,
            reason="Missing constant.",
        )
    with pytest.raises(ValueError, match="group columns"):
        ImputationPlanEntry(
            column="feature",
            feature_family=ImputationFeatureFamily.NUMERIC,
            method=ImputationMethodName.GROUP_BASED,
            reason="Missing group columns.",
        )
    with pytest.raises(ValueError, match="timestamp column"):
        ImputationPlanEntry(
            column="feature",
            feature_family=ImputationFeatureFamily.DATETIME,
            method=ImputationMethodName.TIME_AWARE,
            reason="Missing timestamp.",
        )


def test_unknown_methods_fall_back_to_review_and_options_validate() -> None:
    assert ImputationPlanBuilder._map_method("future_method") == (
        ImputationMethodName.REVIEW
    )

    with pytest.raises(ValueError, match="categorical_missing_value"):
        ImputationPlanOptions(categorical_missing_value="")
