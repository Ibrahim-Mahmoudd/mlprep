"""Problem-contract definition and validation for Phase 1."""

from __future__ import annotations

from collections.abc import Hashable, Iterable
from dataclasses import dataclass, field
from typing import Literal

import pandas as pd

from mlprep.phases.phase1.output_governance.result import (
    Finding,
    Phase1Result,
)
from mlprep.phases.phase1.output_governance.state import (
    Phase1State,
    ProblemContractState,
)
from mlprep.types import (
    FindingAction,
    FindingCategory,
    Severity,
    TaskType,
)

MissingObservationKeyPolicy = Literal[
    "allow",
    "warn",
    "error",
]


@dataclass(slots=True)
class ProblemContractOptions:
    """Options required to define a supervised-learning problem."""

    task_type: TaskType
    target_column: str
    unit_of_observation: str

    positive_class: str | int | bool | None = None

    observation_key: list[str] = field(default_factory=list)
    entity_columns: list[str] = field(default_factory=list)
    group_columns: list[str] = field(default_factory=list)

    prediction_timestamp_column: str | None = None
    outcome_timestamp_column: str | None = None

    target_horizon: str | None = None

    require_unique_observation_key: bool = True

    missing_observation_key_policy: MissingObservationKeyPolicy = (
        "warn"
    )

    maximum_examples: int = 5

    def __post_init__(self) -> None:
        """Validate problem-contract options."""

        if not self.target_column.strip():
            raise ValueError(
                "target_column cannot be empty."
            )

        if not self.unit_of_observation.strip():
            raise ValueError(
                "unit_of_observation cannot be empty."
            )

        if self.maximum_examples < 0:
            raise ValueError(
                "maximum_examples cannot be negative."
            )

        if (
            self.task_type
            == TaskType.BINARY_CLASSIFICATION
            and self.positive_class is None
        ):
            raise ValueError(
                "positive_class is required for binary "
                "classification."
            )

        if (
            self.task_type
            != TaskType.BINARY_CLASSIFICATION
            and self.positive_class is not None
        ):
            raise ValueError(
                "positive_class is supported only for binary "
                "classification."
            )

        if self.target_column in self.observation_key:
            raise ValueError(
                "The target column cannot be part of the "
                "observation key."
            )

        if self.target_column in self.entity_columns:
            raise ValueError(
                "The target column cannot be an entity column."
            )

        if self.target_column in self.group_columns:
            raise ValueError(
                "The target column cannot be a group column."
            )

        if (
            self.prediction_timestamp_column
            == self.target_column
        ):
            raise ValueError(
                "The target column cannot also be the prediction "
                "timestamp column."
            )

        if (
            self.outcome_timestamp_column
            == self.target_column
        ):
            raise ValueError(
                "The target column cannot also be the outcome "
                "timestamp column."
            )

        if (
            self.prediction_timestamp_column is not None
            and self.outcome_timestamp_column is not None
            and self.prediction_timestamp_column
            == self.outcome_timestamp_column
        ):
            raise ValueError(
                "Prediction and outcome timestamp columns must "
                "be different."
            )

        duplicate_declarations = list(
            dict.fromkeys(
                duplicate
                for declarations in (
                    self.observation_key,
                    self.entity_columns,
                    self.group_columns,
                )
                for duplicate in self._find_duplicates(
                    declarations
                )
            )
        )

        if duplicate_declarations:
            formatted_columns = ", ".join(
                duplicate_declarations
            )

            raise ValueError(
                "Problem-contract column declarations contain "
                f"duplicates: {formatted_columns}"
            )

    @staticmethod
    def _find_duplicates(
        values: Iterable[str],
    ) -> list[str]:
        """Return duplicated values while preserving order."""

        seen: set[str] = set()
        duplicates: list[str] = []

        for value in values:
            if value in seen and value not in duplicates:
                duplicates.append(value)

            seen.add(value)

        return duplicates


class ProblemContractResolver:
    """Validate and resolve a machine-learning problem contract."""

    component_name = "problem_contract"

    def __init__(
        self,
        options: ProblemContractOptions,
    ) -> None:
        """Initialize the problem-contract resolver."""

        self.options = options

    def resolve(
        self,
        data: object,
        result: Phase1Result,
        state: Phase1State,
    ) -> ProblemContractState | None:
        """Validate the dataset and build the resolved contract."""

        if not isinstance(data, pd.DataFrame):
            self._add_finding(
                result=result,
                code="P1-CONTRACT-001",
                severity=Severity.CRITICAL,
                message=(
                    "Problem-contract resolution requires a "
                    "pandas DataFrame."
                ),
                recommended_action=FindingAction.RAISE,
                blocks_readiness=True,
                details={
                    "received_type": type(data).__name__,
                    "expected_type": "DataFrame",
                },
            )

            return None

        if not self._validate_dataset_shape(
            data=data,
            result=result,
        ):
            return None

        target_exists = self._validate_target_column(
            data=data,
            result=result,
        )

        required_columns_exist = (
            self._validate_required_columns(
                data=data,
                result=result,
            )
        )

        observation_key_valid = (
            self._validate_observation_key(
                data=data,
                result=result,
            )
        )

        timestamps_valid = self._validate_timestamps(
            data=data,
            result=result,
        )

        if not target_exists:
            return None

        contract = self._build_contract(data)

        state.problem_contract = contract

        result.problem_contract = contract.model_dump(
            mode="json"
        )

        if (
            required_columns_exist
            and observation_key_valid
            and timestamps_valid
        ):
            contract.resolved = True

        return contract

    def _validate_dataset_shape(
        self,
        *,
        data: pd.DataFrame,
        result: Phase1Result,
    ) -> bool:
        """Validate that the dataset contains rows and columns."""

        is_valid = True

        if data.shape[1] == 0:
            self._add_finding(
                result=result,
                code="P1-CONTRACT-002",
                severity=Severity.CRITICAL,
                message="The dataset contains no columns.",
                recommended_action=FindingAction.RAISE,
                blocks_readiness=True,
            )

            is_valid = False

        if data.shape[0] == 0:
            self._add_finding(
                result=result,
                code="P1-CONTRACT-003",
                severity=Severity.CRITICAL,
                message="The dataset contains no rows.",
                recommended_action=FindingAction.RAISE,
                blocks_readiness=True,
            )

            is_valid = False

        return is_valid

    def _validate_target_column(
        self,
        *,
        data: pd.DataFrame,
        result: Phase1Result,
    ) -> bool:
        """Validate that the target column exists."""

        target_column = self.options.target_column

        if target_column in data.columns:
            return True

        self._add_finding(
            result=result,
            code="P1-CONTRACT-004",
            severity=Severity.CRITICAL,
            message=(
                f"The target column {target_column!r} does not "
                "exist in the dataset."
            ),
            columns=[target_column],
            recommended_action=FindingAction.RAISE,
            blocks_readiness=True,
            details={
                "target_column": target_column,
                "available_columns": [
                    str(column)
                    for column in data.columns
                ],
            },
        )

        return False

    def _validate_required_columns(
        self,
        *,
        data: pd.DataFrame,
        result: Phase1Result,
    ) -> bool:
        """Validate all columns required by the contract."""

        required_columns = self._required_columns()

        missing_columns = self._find_missing_columns(
            required_columns=required_columns,
            available_columns=data.columns,
        )

        if not missing_columns:
            return True

        self._add_finding(
            result=result,
            code="P1-CONTRACT-005",
            severity=Severity.ERROR,
            message=(
                "The problem contract references columns that "
                "do not exist in the dataset."
            ),
            columns=missing_columns,
            recommended_action=FindingAction.RAISE,
            blocks_readiness=True,
            details={
                "missing_columns": missing_columns,
                "required_columns": required_columns,
            },
        )

        return False

    def _validate_observation_key(
        self,
        *,
        data: pd.DataFrame,
        result: Phase1Result,
    ) -> bool:
        """Validate completeness and uniqueness of the observation key."""

        key_columns = self.options.observation_key

        if not key_columns:
            return self._handle_missing_observation_key(
                result=result
            )

        missing_columns = self._find_missing_columns(
            required_columns=key_columns,
            available_columns=data.columns,
        )

        if missing_columns:
            return False

        is_valid = True

        missing_mask = data[key_columns].isna().any(
            axis=1
        )

        missing_count = int(missing_mask.sum())

        if missing_count > 0:
            self._add_finding(
                result=result,
                code="P1-CONTRACT-006",
                severity=Severity.ERROR,
                message=(
                    "The observation key contains missing values."
                ),
                columns=key_columns,
                affected_row_count=missing_count,
                affected_row_percentage=self._percentage(
                    affected_count=missing_count,
                    total_count=len(data),
                ),
                example_row_ids=self._example_indices(
                    data=data,
                    mask=missing_mask,
                ),
                recommended_action=(
                    FindingAction.QUARANTINE
                ),
                blocks_readiness=True,
            )

            is_valid = False

        if not self.options.require_unique_observation_key:
            return is_valid

        duplicate_mask = data.duplicated(
            subset=key_columns,
            keep=False,
        )

        duplicate_count = int(
            duplicate_mask.sum()
        )

        if duplicate_count > 0:
            self._add_finding(
                result=result,
                code="P1-CONTRACT-007",
                severity=Severity.ERROR,
                message=(
                    "The observation key is not unique."
                ),
                columns=key_columns,
                affected_row_count=duplicate_count,
                affected_row_percentage=self._percentage(
                    affected_count=duplicate_count,
                    total_count=len(data),
                ),
                example_row_ids=self._example_indices(
                    data=data,
                    mask=duplicate_mask,
                ),
                recommended_action=FindingAction.RAISE,
                blocks_readiness=True,
                details={
                    "observation_key": key_columns,
                },
            )

            is_valid = False

        return is_valid

    def _handle_missing_observation_key(
        self,
        *,
        result: Phase1Result,
    ) -> bool:
        """Apply the configured policy for a missing observation key."""

        policy = (
            self.options.missing_observation_key_policy
        )

        if policy == "allow":
            return True

        if policy == "warn":
            self._add_finding(
                result=result,
                code="P1-CONTRACT-008",
                severity=Severity.WARNING,
                message=(
                    "No observation key was configured. "
                    "Business-level row uniqueness cannot be "
                    "validated."
                ),
                recommended_action=FindingAction.WARN,
                blocks_readiness=False,
            )

            return True

        self._add_finding(
            result=result,
            code="P1-CONTRACT-009",
            severity=Severity.ERROR,
            message=(
                "An observation key is required by the current "
                "problem-contract policy."
            ),
            recommended_action=FindingAction.RAISE,
            blocks_readiness=True,
        )

        return False

    def _validate_timestamps(
        self,
        *,
        data: pd.DataFrame,
        result: Phase1Result,
    ) -> bool:
        """Validate prediction and outcome timestamp ordering."""

        prediction_column = (
            self.options.prediction_timestamp_column
        )

        outcome_column = (
            self.options.outcome_timestamp_column
        )

        if (
            prediction_column is None
            or outcome_column is None
        ):
            return True

        if (
            prediction_column not in data.columns
            or outcome_column not in data.columns
        ):
            return False

        prediction_values = pd.to_datetime(
            data[prediction_column],
            errors="coerce",
        )

        outcome_values = pd.to_datetime(
            data[outcome_column],
            errors="coerce",
        )

        prediction_parse_failures = (
            data[prediction_column].notna()
            & prediction_values.isna()
        )

        outcome_parse_failures = (
            data[outcome_column].notna()
            & outcome_values.isna()
        )

        is_valid = True

        if bool(prediction_parse_failures.any()):
            failure_count = int(
                prediction_parse_failures.sum()
            )

            self._add_finding(
                result=result,
                code="P1-CONTRACT-010",
                severity=Severity.ERROR,
                message=(
                    "Some prediction timestamp values could not "
                    "be parsed."
                ),
                columns=[prediction_column],
                affected_row_count=failure_count,
                affected_row_percentage=self._percentage(
                    affected_count=failure_count,
                    total_count=len(data),
                ),
                example_row_ids=self._example_indices(
                    data=data,
                    mask=prediction_parse_failures,
                ),
                recommended_action=FindingAction.COERCE,
                blocks_readiness=True,
            )

            is_valid = False

        if bool(outcome_parse_failures.any()):
            failure_count = int(
                outcome_parse_failures.sum()
            )

            self._add_finding(
                result=result,
                code="P1-CONTRACT-011",
                severity=Severity.ERROR,
                message=(
                    "Some outcome timestamp values could not "
                    "be parsed."
                ),
                columns=[outcome_column],
                affected_row_count=failure_count,
                affected_row_percentage=self._percentage(
                    affected_count=failure_count,
                    total_count=len(data),
                ),
                example_row_ids=self._example_indices(
                    data=data,
                    mask=outcome_parse_failures,
                ),
                recommended_action=FindingAction.COERCE,
                blocks_readiness=True,
            )

            is_valid = False

        valid_pair_mask = (
            prediction_values.notna()
            & outcome_values.notna()
        )

        invalid_order_mask = (
            valid_pair_mask
            & (outcome_values < prediction_values)
        )

        invalid_order_count = int(
            invalid_order_mask.sum()
        )

        if invalid_order_count > 0:
            self._add_finding(
                result=result,
                code="P1-CONTRACT-012",
                severity=Severity.ERROR,
                message=(
                    "Some outcome timestamps occur before their "
                    "prediction timestamps."
                ),
                columns=[
                    prediction_column,
                    outcome_column,
                ],
                affected_row_count=invalid_order_count,
                affected_row_percentage=self._percentage(
                    affected_count=invalid_order_count,
                    total_count=len(data),
                ),
                example_row_ids=self._example_indices(
                    data=data,
                    mask=invalid_order_mask,
                ),
                recommended_action=FindingAction.RAISE,
                blocks_readiness=True,
            )

            is_valid = False

        return is_valid

    def _build_contract(
        self,
        data: pd.DataFrame,
    ) -> ProblemContractState:
        """Build the persistent problem-contract state."""

        target_column = self.options.target_column

        feature_columns = [
            str(column)
            for column in data.columns
            if str(column) != target_column
        ]

        return ProblemContractState(
            task_type=self.options.task_type,
            target_column=target_column,
            positive_class=self.options.positive_class,
            unit_of_observation=(
                self.options.unit_of_observation
            ),
            observation_key=list(
                self.options.observation_key
            ),
            entity_columns=list(
                self.options.entity_columns
            ),
            group_columns=list(
                self.options.group_columns
            ),
            prediction_timestamp_column=(
                self.options.prediction_timestamp_column
            ),
            outcome_timestamp_column=(
                self.options.outcome_timestamp_column
            ),
            target_horizon=self.options.target_horizon,
            feature_columns=feature_columns,
            inference_columns=list(feature_columns),
            resolved=False,
        )

    def _required_columns(self) -> list[str]:
        """Return all columns required by the problem contract."""

        required_columns = [
            self.options.target_column,
            *self.options.observation_key,
            *self.options.entity_columns,
            *self.options.group_columns,
        ]

        if (
            self.options.prediction_timestamp_column
            is not None
        ):
            required_columns.append(
                self.options.prediction_timestamp_column
            )

        if self.options.outcome_timestamp_column is not None:
            required_columns.append(
                self.options.outcome_timestamp_column
            )

        return list(
            dict.fromkeys(required_columns)
        )

    @staticmethod
    def _find_missing_columns(
        *,
        required_columns: Iterable[str],
        available_columns: Iterable[Hashable],
    ) -> list[str]:
        """Return required columns absent from the dataset."""

        available_names = {
            str(column)
            for column in available_columns
        }

        return [
            str(column)
            for column in required_columns
            if column not in available_names
        ]

    def _example_indices(
        self,
        *,
        data: pd.DataFrame,
        mask: pd.Series,
    ) -> list[str | int]:
        """Return example indices for affected rows."""

        if self.options.maximum_examples == 0:
            return []

        selected_indices = data.index[mask].tolist()[
            : self.options.maximum_examples
        ]

        examples: list[str | int] = []

        for index in selected_indices:
            if isinstance(index, (str, int)):
                examples.append(index)
            else:
                examples.append(str(index))

        return examples

    def _add_finding(
        self,
        *,
        result: Phase1Result,
        code: str,
        severity: Severity,
        message: str,
        columns: list[str] | None = None,
        affected_row_count: int = 0,
        affected_row_percentage: float = 0.0,
        example_row_ids: list[str | int] | None = None,
        recommended_action: FindingAction,
        blocks_readiness: bool,
        details: dict[str, object] | None = None,
    ) -> None:
        """Create and add a problem-contract finding."""

        result.add_finding(
            Finding(
                code=code,
                category=(
                    FindingCategory.PROBLEM_CONTRACT
                ),
                severity=severity,
                component=self.component_name,
                message=message,
                columns=columns or [],
                affected_row_count=affected_row_count,
                affected_row_percentage=(
                    affected_row_percentage
                ),
                example_row_ids=example_row_ids or [],
                recommended_action=recommended_action,
                applied_action=FindingAction.REPORT,
                blocks_readiness=blocks_readiness,
                details=details or {},
            )
        )

    @staticmethod
    def _percentage(
        *,
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
