"""Supervised target validation for Phase 1."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype

from mlprep.phases.phase1.output_governance.result import (
    Finding,
    Phase1Result,
)
from mlprep.phases.phase1.output_governance.state import Phase1State
from mlprep.types import (
    FindingAction,
    FindingCategory,
    Severity,
    TaskType,
)

MissingTargetPolicy = Literal[
    "report",
    "raise",
    "quarantine",
]


@dataclass(slots=True)
class TargetValidationOptions:
    """Options controlling supervised target validation."""

    target_column: str
    task_type: TaskType

    positive_class: str | int | bool | None = None
    allowed_class_values: list[str | int | bool] | None = None

    allow_missing_target: bool = False
    missing_target_policy: MissingTargetPolicy = "raise"

    allow_single_class: bool = False
    minimum_class_count: int = 2

    regression_minimum: float | None = None
    regression_maximum: float | None = None
    allow_infinite_values: bool = False

    maximum_examples: int = 5

    def __post_init__(self) -> None:
        """Validate target-validation options."""

        if not self.target_column.strip():
            raise ValueError(
                "target_column cannot be empty."
            )

        if self.minimum_class_count < 1:
            raise ValueError(
                "minimum_class_count must be at least 1."
            )

        if (
            self.regression_minimum is not None
            and self.regression_maximum is not None
            and self.regression_minimum
            > self.regression_maximum
        ):
            raise ValueError(
                "regression_minimum cannot exceed regression_maximum."
            )


class TargetValidator:
    """Validate a classification or regression target."""

    component_name = "target_validation"

    def __init__(
        self,
        options: TargetValidationOptions,
    ) -> None:
        """Initialize the target validator."""

        self.options = options

    def validate(
        self,
        data: pd.DataFrame,
        result: Phase1Result,
        state: Phase1State,
    ) -> bool:
        """Validate the configured target."""

        if not isinstance(data, pd.DataFrame):
            raise TypeError(
                "Target validation requires a pandas DataFrame."
            )

        target = self.options.target_column

        if target not in data.columns:
            self._add_finding(
                result=result,
                code="P1-TARGET-001",
                severity=Severity.CRITICAL,
                message=f"Target column {target!r} does not exist.",
                blocks_readiness=True,
            )

            return False

        target_series = data[target]

        is_valid = self._validate_missing_values(
            data=data,
            series=target_series,
            result=result,
        )

        if self.options.task_type in {
            TaskType.BINARY_CLASSIFICATION,
            TaskType.MULTICLASS_CLASSIFICATION,
        }:
            classification_valid = self._validate_classification(
                series=target_series,
                result=result,
            )

            is_valid = is_valid and classification_valid

        elif (
            self.options.task_type
            == TaskType.REGRESSION
        ):
            regression_valid = self._validate_regression(
                series=target_series,
                result=result,
            )

            is_valid = is_valid and regression_valid

        return is_valid

    def _validate_missing_values(
        self,
        *,
        data: pd.DataFrame,
        series: pd.Series,
        result: Phase1Result,
    ) -> bool:
        """Validate missing target values."""

        missing_mask = series.isna()
        missing_count = int(missing_mask.sum())

        if missing_count == 0:
            return True

        blocks = (
            not self.options.allow_missing_target
            and self.options.missing_target_policy
            == "raise"
        )

        self._add_finding(
            result=result,
            code="P1-TARGET-002",
            severity=(
                Severity.ERROR
                if blocks
                else Severity.WARNING
            ),
            message="The target contains missing values.",
            affected_row_count=missing_count,
            affected_row_percentage=self._percentage(
                missing_count,
                len(data),
            ),
            example_row_ids=[
                self._safe_row_id(index)
                for index in data.index[
                    missing_mask
                ].tolist()[
                    : self.options.maximum_examples
                ]
            ],
            blocks_readiness=blocks,
        )

        return not blocks

    def _validate_classification(
        self,
        *,
        series: pd.Series,
        result: Phase1Result,
    ) -> bool:
        """Validate a classification target."""

        non_missing = series.dropna()

        class_counts = non_missing.value_counts(
            dropna=False
        )

        class_count = len(class_counts)
        is_valid = True

        expected_class_count = (
            2
            if self.options.task_type
            == TaskType.BINARY_CLASSIFICATION
            else None
        )

        if (
            expected_class_count is not None
            and class_count != expected_class_count
        ):
            self._add_finding(
                result=result,
                code="P1-TARGET-003",
                severity=Severity.ERROR,
                message=(
                    "A binary classification target must contain "
                    "exactly two nonmissing classes."
                ),
                blocks_readiness=True,
                details={
                    "class_count": class_count,
                    "class_counts": {
                        str(key): int(value)
                        for key, value
                        in class_counts.items()
                    },
                },
            )

            is_valid = False

        if (
            class_count <= 1
            and not self.options.allow_single_class
        ):
            self._add_finding(
                result=result,
                code="P1-TARGET-004",
                severity=Severity.CRITICAL,
                message="The target contains only one usable class.",
                blocks_readiness=True,
            )

            is_valid = False

        small_classes = {
            str(label): int(count)
            for label, count in class_counts.items()
            if count < self.options.minimum_class_count
        }

        if small_classes:
            self._add_finding(
                result=result,
                code="P1-TARGET-005",
                severity=Severity.ERROR,
                message=(
                    "One or more target classes contain too few observations."
                ),
                blocks_readiness=True,
                details={
                    "small_classes": small_classes,
                    "minimum_class_count": (
                        self.options.minimum_class_count
                    ),
                },
            )

            is_valid = False

        if self.options.allowed_class_values is not None:
            allowed = set(
                self.options.allowed_class_values
            )

            observed = set(
                non_missing.unique().tolist()
            )

            unexpected = observed.difference(allowed)

            if unexpected:
                self._add_finding(
                    result=result,
                    code="P1-TARGET-006",
                    severity=Severity.ERROR,
                    message=(
                        "The target contains unexpected class labels."
                    ),
                    blocks_readiness=True,
                    details={
                        "unexpected_values": [
                            str(value)
                            for value in unexpected
                        ],
                    },
                )

                is_valid = False

        if (
            self.options.task_type
            == TaskType.BINARY_CLASSIFICATION
            and self.options.positive_class
            not in set(non_missing.unique())
        ):
            self._add_finding(
                result=result,
                code="P1-TARGET-007",
                severity=Severity.ERROR,
                message=(
                    "The configured positive class is absent "
                    "from the target."
                ),
                blocks_readiness=True,
            )

            is_valid = False

        return is_valid

    def _validate_regression(
        self,
        *,
        series: pd.Series,
        result: Phase1Result,
    ) -> bool:
        """Validate a regression target."""

        if not is_numeric_dtype(series.dtype):
            self._add_finding(
                result=result,
                code="P1-TARGET-008",
                severity=Severity.CRITICAL,
                message="A regression target must be numeric.",
                blocks_readiness=True,
            )

            return False

        non_missing = series.dropna()

        is_valid = True

        values = non_missing.to_numpy(
            dtype=float
        )

        infinite_mask = np.isinf(values)
        infinite_count = int(infinite_mask.sum())

        if (
            infinite_count > 0
            and not self.options.allow_infinite_values
        ):
            self._add_finding(
                result=result,
                code="P1-TARGET-009",
                severity=Severity.ERROR,
                message="The regression target contains infinite values.",
                affected_row_count=infinite_count,
                blocks_readiness=True,
            )

            is_valid = False

        if non_missing.nunique() <= 1:
            self._add_finding(
                result=result,
                code="P1-TARGET-010",
                severity=Severity.CRITICAL,
                message="The regression target is constant.",
                blocks_readiness=True,
            )

            is_valid = False

        if self.options.regression_minimum is not None:
            below_count = int(
                (
                    non_missing
                    < self.options.regression_minimum
                ).sum()
            )

            if below_count > 0:
                self._add_finding(
                    result=result,
                    code="P1-TARGET-011",
                    severity=Severity.ERROR,
                    message=(
                        "The regression target contains values "
                        "below the configured minimum."
                    ),
                    affected_row_count=below_count,
                    blocks_readiness=True,
                )

                is_valid = False

        if self.options.regression_maximum is not None:
            above_count = int(
                (
                    non_missing
                    > self.options.regression_maximum
                ).sum()
            )

            if above_count > 0:
                self._add_finding(
                    result=result,
                    code="P1-TARGET-012",
                    severity=Severity.ERROR,
                    message=(
                        "The regression target contains values "
                        "above the configured maximum."
                    ),
                    affected_row_count=above_count,
                    blocks_readiness=True,
                )

                is_valid = False

        return is_valid

    def _add_finding(
        self,
        *,
        result: Phase1Result,
        code: str,
        severity: Severity,
        message: str,
        affected_row_count: int = 0,
        affected_row_percentage: float = 0.0,
        example_row_ids: list[str | int] | None = None,
        blocks_readiness: bool,
        details: dict[str, object] | None = None,
    ) -> None:
        """Create one target finding."""

        result.add_finding(
            Finding(
                code=code,
                category=FindingCategory.TARGET,
                severity=severity,
                component=self.component_name,
                message=message,
                columns=[self.options.target_column],
                affected_row_count=affected_row_count,
                affected_row_percentage=(
                    affected_row_percentage
                ),
                example_row_ids=example_row_ids or [],
                recommended_action=FindingAction.RAISE,
                applied_action=FindingAction.REPORT,
                blocks_readiness=blocks_readiness,
                details=details or {},
            )
        )

    @staticmethod
    def _safe_row_id(
        value: object,
    ) -> str | int:
        """Convert an index into a supported identifier."""

        if isinstance(value, (str, int)):
            return value

        return str(value)

    @staticmethod
    def _percentage(
        affected_count: int,
        total_count: int,
    ) -> float:
        """Calculate a percentage safely."""

        if total_count <= 0:
            return 0.0

        return round(
            affected_count / total_count * 100.0,
            6,
        )