from __future__ import annotations

import pandas as pd
import pytest

from mlprep.phases.phase1.output_governance.result import Phase1Result
from mlprep.phases.phase1.output_governance.state import Phase1State
from mlprep.phases.phase1.validation_and_risk.target_validation import (
    TargetValidationOptions,
    TargetValidator,
)
from mlprep.types import Severity, TaskType


def _validate(
    frame: pd.DataFrame,
    options: TargetValidationOptions,
) -> tuple[bool, Phase1Result]:
    result = Phase1Result(
        data=frame,
        internal_row_id_column="__row_id__",
    )
    is_valid = TargetValidator(options).validate(
        frame,
        result,
        Phase1State(),
    )
    return is_valid, result


def test_missing_target_column_is_a_critical_blocking_finding() -> None:
    frame = pd.DataFrame({"feature": [1, 2]})

    valid, result = _validate(
        frame,
        TargetValidationOptions(
            target_column="target",
            task_type=TaskType.REGRESSION,
        ),
    )

    finding = result.findings[0]
    assert not valid
    assert finding.code == "P1-TARGET-001"
    assert finding.severity == Severity.CRITICAL
    assert finding.columns == ["target"]
    assert finding.blocks_readiness


def test_missing_classification_targets_follow_allow_and_policy_options() -> None:
    frame = pd.DataFrame(
        {"target": [0, 0, 1, 1, None]},
        index=[10, 11, 12, 13, 14],
    )
    allowed, allowed_result = _validate(
        frame,
        TargetValidationOptions(
            target_column="target",
            task_type=TaskType.BINARY_CLASSIFICATION,
            positive_class=1,
            minimum_class_count=2,
            allow_missing_target=True,
        ),
    )
    rejected, rejected_result = _validate(
        frame,
        TargetValidationOptions(
            target_column="target",
            task_type=TaskType.BINARY_CLASSIFICATION,
            positive_class=1,
            minimum_class_count=2,
            missing_target_policy="raise",
            maximum_examples=1,
        ),
    )

    assert allowed
    assert allowed_result.findings[0].code == "P1-TARGET-002"
    assert allowed_result.findings[0].severity == Severity.WARNING
    assert not allowed_result.findings[0].blocks_readiness
    assert not rejected
    assert rejected_result.findings[0].severity == Severity.ERROR
    assert rejected_result.findings[0].affected_row_count == 1
    assert rejected_result.findings[0].example_row_ids == [14]
    assert rejected_result.findings[0].blocks_readiness


def test_binary_classification_reports_count_rare_and_unexpected_classes() -> None:
    frame = pd.DataFrame(
        {"target": [0, 0, 1, 1, 2]},
        index=[20, 21, 22, 23, 24],
    )

    valid, result = _validate(
        frame,
        TargetValidationOptions(
            target_column="target",
            task_type=TaskType.BINARY_CLASSIFICATION,
            positive_class=1,
            allowed_class_values=[0, 1],
            minimum_class_count=2,
        ),
    )

    findings = {finding.code: finding for finding in result.findings}
    assert not valid
    assert set(findings) == {
        "P1-TARGET-003",
        "P1-TARGET-005",
        "P1-TARGET-006",
    }
    assert findings["P1-TARGET-003"].details["class_count"] == 3
    assert findings["P1-TARGET-005"].details["small_classes"] == {"2": 1}
    assert findings["P1-TARGET-006"].details["unexpected_values"] == ["2"]
    assert all(finding.blocks_readiness for finding in result.findings)


def test_single_class_and_missing_positive_class_are_reported() -> None:
    frame = pd.DataFrame({"target": [0, 0, 0]})

    valid, result = _validate(
        frame,
        TargetValidationOptions(
            target_column="target",
            task_type=TaskType.BINARY_CLASSIFICATION,
            positive_class=1,
        ),
    )

    assert not valid
    assert {finding.code for finding in result.findings} == {
        "P1-TARGET-003",
        "P1-TARGET-004",
        "P1-TARGET-007",
    }


def test_regression_target_requires_numeric_nonconstant_finite_values() -> None:
    nonnumeric, nonnumeric_result = _validate(
        pd.DataFrame({"target": ["low", "high"]}),
        TargetValidationOptions(
            target_column="target",
            task_type=TaskType.REGRESSION,
        ),
    )
    infinite, infinite_result = _validate(
        pd.DataFrame({"target": [1.0, float("inf")]}),
        TargetValidationOptions(
            target_column="target",
            task_type=TaskType.REGRESSION,
        ),
    )
    constant, constant_result = _validate(
        pd.DataFrame({"target": [4.0, 4.0]}),
        TargetValidationOptions(
            target_column="target",
            task_type=TaskType.REGRESSION,
        ),
    )
    infinite_allowed, allowed_result = _validate(
        pd.DataFrame({"target": [1.0, float("inf")]}),
        TargetValidationOptions(
            target_column="target",
            task_type=TaskType.REGRESSION,
            allow_infinite_values=True,
        ),
    )

    assert not nonnumeric
    assert nonnumeric_result.findings[0].code == "P1-TARGET-008"
    assert not infinite
    assert infinite_result.findings[0].code == "P1-TARGET-009"
    assert not constant
    assert constant_result.findings[0].code == "P1-TARGET-010"
    assert infinite_allowed
    assert allowed_result.findings == []


def test_regression_bounds_report_each_violation_count() -> None:
    valid, result = _validate(
        pd.DataFrame({"target": [-1.0, 0.0, 10.0, 11.0]}),
        TargetValidationOptions(
            target_column="target",
            task_type=TaskType.REGRESSION,
            regression_minimum=0.0,
            regression_maximum=10.0,
        ),
    )

    findings = {finding.code: finding for finding in result.findings}
    assert not valid
    assert findings["P1-TARGET-011"].affected_row_count == 1
    assert findings["P1-TARGET-012"].affected_row_count == 1


def test_target_options_reject_invalid_bounds_and_counts() -> None:
    with pytest.raises(ValueError, match="target_column"):
        TargetValidationOptions(
            target_column=" ",
            task_type=TaskType.REGRESSION,
        )

    with pytest.raises(ValueError, match="minimum_class_count"):
        TargetValidationOptions(
            target_column="target",
            task_type=TaskType.BINARY_CLASSIFICATION,
            minimum_class_count=0,
        )

    with pytest.raises(ValueError, match="regression_minimum"):
        TargetValidationOptions(
            target_column="target",
            task_type=TaskType.REGRESSION,
            regression_minimum=10,
            regression_maximum=0,
        )
