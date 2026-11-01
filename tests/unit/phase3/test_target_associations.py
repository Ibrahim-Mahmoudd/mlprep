from __future__ import annotations

from types import SimpleNamespace

import pandas as pd

from mlprep.phases.phase3.contract.column_selection import (
    AnalysisColumnGroups,
)
from mlprep.phases.phase3.relationships.target_associations import (
    TargetAssociationAnalyzer,
    TargetAssociationOptions,
)
from mlprep.types import TaskType


def _contract(
    features: pd.DataFrame,
    target: pd.Series,
    task_type: TaskType,
):
    return SimpleNamespace(
        train=SimpleNamespace(X=features),
        training_target=lambda: target.copy(deep=True),
        target_column="target",
        phase1_state=SimpleNamespace(
            problem_contract=SimpleNamespace(task_type=task_type)
        ),
    )


def test_binary_classification_analyzes_numeric_and_categorical_features() -> None:
    features = pd.DataFrame(
        {
            "numeric": [0.0, 0.0, 0.0, 0.0, 5.0, 5.0, 5.0, 5.0],
            "category": ["a", "a", "a", "a", "b", "b", "b", "b"],
        }
    )
    target = pd.Series([0, 0, 0, 0, 1, 1, 1, 1], name="target")
    features_before = features.copy(deep=True)
    target_before = target.copy(deep=True)

    result = TargetAssociationAnalyzer().analyze(
        contract=_contract(
            features,
            target,
            TaskType.BINARY_CLASSIFICATION,
        ),
        columns=AnalysisColumnGroups(
            numeric=["numeric"],
            categorical=["category"],
        ),
    )

    associations = {item.feature: item for item in result.associations}
    assert result.task_type == "binary_classification"
    assert associations["numeric"].kind == "numeric_classification"
    assert associations["numeric"].association_score == 1.0
    assert associations["category"].kind == "categorical_classification"
    assert associations["category"].cramers_v == 1.0
    assert result.suspicious_features == ["numeric", "category"]
    assert features.equals(features_before)
    assert target.equals(target_before)


def test_multiclass_classification_reports_categorical_association() -> None:
    features = pd.DataFrame(
        {"group": ["a", "a", "b", "b", "c", "c"]}
    )
    target = pd.Series([0, 0, 1, 1, 2, 2], name="target")
    before = features.copy(deep=True)

    result = TargetAssociationAnalyzer().analyze(
        contract=_contract(
            features,
            target,
            TaskType.MULTICLASS_CLASSIFICATION,
        ),
        columns=AnalysisColumnGroups(categorical=["group"]),
    )

    assert result.task_type == "multiclass_classification"
    assert len(result.associations) == 1
    assert result.associations[0].kind == "categorical_classification"
    assert result.associations[0].association_score == 1.0
    assert features.equals(before)


def test_regression_analyzes_numeric_and_categorical_features() -> None:
    features = pd.DataFrame(
        {
            "numeric": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "category": ["low", "low", "mid", "mid", "high", "high"],
        }
    )
    target = pd.Series([2.0, 4.0, 6.0, 8.0, 10.0, 12.0], name="target")

    result = TargetAssociationAnalyzer().analyze(
        contract=_contract(features, target, TaskType.REGRESSION),
        columns=AnalysisColumnGroups(
            numeric=["numeric"],
            categorical=["category"],
        ),
    )

    associations = {item.feature: item for item in result.associations}
    assert result.task_type == "regression"
    assert associations["numeric"].kind == "numeric_regression"
    assert associations["numeric"].pearson == 1.0
    assert associations["numeric"].spearman == 1.0
    assert associations["category"].kind == "categorical_regression"
    assert associations["category"].eta_squared == 32 / 35


def test_constant_features_and_missing_values_have_safe_scores() -> None:
    features = pd.DataFrame(
        {
            "constant": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            "with_missing": [0.0, 1.0, None, 3.0, 4.0, None],
        }
    )
    target = pd.Series([0, 0, 1, 1, 0, 1], name="target")

    result = TargetAssociationAnalyzer(
        TargetAssociationOptions(minimum_valid_pairs=4)
    ).analyze(
        contract=_contract(features, target, TaskType.BINARY_CLASSIFICATION),
        columns=AnalysisColumnGroups(numeric=["constant", "with_missing"]),
    )

    associations = {item.feature: item for item in result.associations}
    assert associations["constant"].association_score is None
    assert not associations["constant"].suspiciously_strong
    assert associations["with_missing"].valid_pair_count == 4


def test_insufficient_observations_and_unsupported_numeric_dtype_are_omitted() -> None:
    features = pd.DataFrame(
        {
            "too_short": [1.0, 2.0, 3.0, None],
            "not_numeric": ["a", "b", "c", "d"],
        }
    )
    target = pd.Series([0, 1, 0, 1], name="target")
    analyzer = TargetAssociationAnalyzer(
        TargetAssociationOptions(minimum_valid_pairs=5)
    )

    result = analyzer.analyze(
        contract=_contract(features, target, TaskType.BINARY_CLASSIFICATION),
        columns=AnalysisColumnGroups(
            numeric=["too_short", "not_numeric"]
        ),
    )

    assert result.associations == []
    assert result.strongest_features == []


def test_constant_feature_ties_keep_input_order_deterministically() -> None:
    features = pd.DataFrame(
        {
            "second": [2.0] * 6,
            "first": [1.0] * 6,
        }
    )
    target = pd.Series([0, 1, 0, 1, 0, 1], name="target")
    contract = _contract(
        features,
        target,
        TaskType.BINARY_CLASSIFICATION,
    )
    columns = AnalysisColumnGroups(numeric=["second", "first"])
    analyzer = TargetAssociationAnalyzer()

    first = analyzer.analyze(contract=contract, columns=columns)
    second = analyzer.analyze(contract=contract, columns=columns)

    assert [item.feature for item in first.associations] == ["second", "first"]
    assert first.model_dump(mode="json") == second.model_dump(mode="json")
