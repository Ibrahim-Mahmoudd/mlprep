"""Column-role assignment for Phase 1."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from mlprep.phases.phase1.output_governance.result import (
    Finding,
    Phase1Result,
)
from mlprep.phases.phase1.output_governance.state import (
    ColumnState,
    Phase1State,
)
from mlprep.types import (
    ColumnRole,
    FindingAction,
    FindingCategory,
    Severity,
)


@dataclass(slots=True)
class ColumnRoleOptions:
    """User-declared column roles."""

    target_column: str

    feature_columns: list[str] | None = None

    identifier_columns: list[str] = field(
        default_factory=list
    )

    group_columns: list[str] = field(
        default_factory=list
    )

    timestamp_columns: list[str] = field(
        default_factory=list
    )

    prediction_timestamp_column: str | None = None
    outcome_timestamp_column: str | None = None

    sensitive_columns: list[str] = field(
        default_factory=list
    )

    metadata_columns: list[str] = field(
        default_factory=list
    )

    ignored_columns: list[str] = field(
        default_factory=list
    )

    inference_unavailable_columns: list[str] = field(
        default_factory=list
    )

    sample_weight_column: str | None = None
    fold_column: str | None = None

    allow_unassigned_columns_as_features: bool = True

    maximum_examples: int = 5

    def __post_init__(self) -> None:
        """Validate role-assignment options."""

        if not self.target_column.strip():
            raise ValueError(
                "target_column cannot be empty."
            )

        if self.maximum_examples < 0:
            raise ValueError(
                "maximum_examples cannot be negative."
            )

        if (
            self.feature_columns is not None
            and self.target_column
            in self.feature_columns
        ):
            raise ValueError(
                "The target column cannot appear in "
                "feature_columns."
            )

        if self.target_column in self.identifier_columns:
            raise ValueError(
                "The target column cannot be an identifier."
            )

        if self.target_column in self.metadata_columns:
            raise ValueError(
                "The target column cannot be metadata."
            )

        if self.target_column in self.ignored_columns:
            raise ValueError(
                "The target column cannot be ignored."
            )

        if (
            self.sample_weight_column
            == self.target_column
        ):
            raise ValueError(
                "The target column cannot also be the sample "
                "weight column."
            )

        if self.fold_column == self.target_column:
            raise ValueError(
                "The target column cannot also be the fold column."
            )


class ColumnRoleAssigner:
    """Assign functional roles to every dataset column."""

    component_name = "column_roles"

    def __init__(
        self,
        options: ColumnRoleOptions,
    ) -> None:
        """Initialize the role assigner."""

        self.options = options

    def assign(
        self,
        data: pd.DataFrame,
        result: Phase1Result,
        state: Phase1State,
    ) -> dict[str, set[ColumnRole]]:
        """Assign roles and register column state objects."""

        if not isinstance(data, pd.DataFrame):
            raise TypeError(
                "Column-role assignment requires a "
                "pandas DataFrame."
            )

        declared_columns = self._declared_columns()

        missing_columns = [
            column
            for column in declared_columns
            if column not in data.columns
        ]

        if missing_columns:
            result.add_finding(
                Finding(
                    code="P1-ROLE-001",
                    category=FindingCategory.COLUMN_ROLE,
                    severity=Severity.ERROR,
                    component=self.component_name,
                    message=(
                        "Role declarations reference columns that "
                        "do not exist in the dataset."
                    ),
                    columns=missing_columns,
                    recommended_action=FindingAction.RAISE,
                    applied_action=FindingAction.REPORT,
                    blocks_readiness=True,
                    details={
                        "missing_columns": missing_columns,
                    },
                )
            )

        role_map: dict[str, set[ColumnRole]] = {
            str(column): set()
            for column in data.columns
        }

        self._assign_single_role(
            role_map=role_map,
            column=self.options.target_column,
            role=ColumnRole.TARGET,
        )

        self._assign_multiple_roles(
            role_map=role_map,
            columns=self.options.identifier_columns,
            role=ColumnRole.IDENTIFIER,
        )

        self._assign_multiple_roles(
            role_map=role_map,
            columns=self.options.group_columns,
            role=ColumnRole.GROUP,
        )

        self._assign_multiple_roles(
            role_map=role_map,
            columns=self.options.timestamp_columns,
            role=ColumnRole.TIMESTAMP,
        )

        self._assign_single_role(
            role_map=role_map,
            column=(
                self.options.prediction_timestamp_column
            ),
            role=ColumnRole.PREDICTION_TIMESTAMP,
        )

        self._assign_single_role(
            role_map=role_map,
            column=self.options.outcome_timestamp_column,
            role=ColumnRole.OUTCOME_TIMESTAMP,
        )

        self._assign_multiple_roles(
            role_map=role_map,
            columns=self.options.sensitive_columns,
            role=ColumnRole.SENSITIVE,
        )

        self._assign_multiple_roles(
            role_map=role_map,
            columns=self.options.metadata_columns,
            role=ColumnRole.METADATA,
        )

        self._assign_multiple_roles(
            role_map=role_map,
            columns=self.options.ignored_columns,
            role=ColumnRole.IGNORED,
        )

        self._assign_single_role(
            role_map=role_map,
            column=self.options.sample_weight_column,
            role=ColumnRole.SAMPLE_WEIGHT,
        )

        self._assign_single_role(
            role_map=role_map,
            column=self.options.fold_column,
            role=ColumnRole.FOLD,
        )

        self._assign_feature_roles(
            role_map=role_map,
        )

        self._validate_role_conflicts(
            role_map=role_map,
            result=result,
        )

        self._register_column_states(
            data=data,
            role_map=role_map,
            state=state,
        )

        self._update_problem_contract(
            role_map=role_map,
            state=state,
        )

        self._update_dataset_summary(
            role_map=role_map,
            result=result,
        )

        return role_map

    def _assign_feature_roles(
        self,
        *,
        role_map: dict[str, set[ColumnRole]],
    ) -> None:
        """Assign declared or automatically inferred feature roles."""

        if self.options.feature_columns is not None:
            self._assign_multiple_roles(
                role_map=role_map,
                columns=self.options.feature_columns,
                role=ColumnRole.FEATURE,
            )

            return

        if not self.options.allow_unassigned_columns_as_features:
            return

        non_feature_roles = {
            ColumnRole.TARGET,
            ColumnRole.IDENTIFIER,
            ColumnRole.METADATA,
            ColumnRole.IGNORED,
            ColumnRole.SAMPLE_WEIGHT,
            ColumnRole.FOLD,
            ColumnRole.OUTCOME_TIMESTAMP,
        }

        for column, roles in role_map.items():
            if roles.intersection(non_feature_roles):
                continue

            roles.add(ColumnRole.FEATURE)

    def _validate_role_conflicts(
        self,
        *,
        role_map: dict[str, set[ColumnRole]],
        result: Phase1Result,
    ) -> None:
        """Detect incompatible role combinations."""

        incompatible_pairs = [
            (
                ColumnRole.TARGET,
                ColumnRole.FEATURE,
            ),
            (
                ColumnRole.TARGET,
                ColumnRole.IDENTIFIER,
            ),
            (
                ColumnRole.TARGET,
                ColumnRole.IGNORED,
            ),
            (
                ColumnRole.FEATURE,
                ColumnRole.IGNORED,
            ),
            (
                ColumnRole.FEATURE,
                ColumnRole.METADATA,
            ),
            (
                ColumnRole.FEATURE,
                ColumnRole.OUTCOME_TIMESTAMP,
            ),
            (
                ColumnRole.SAMPLE_WEIGHT,
                ColumnRole.FEATURE,
            ),
            (
                ColumnRole.FOLD,
                ColumnRole.FEATURE,
            ),
        ]

        for column, roles in role_map.items():
            conflicting_pairs = [
                (
                    first_role.value,
                    second_role.value,
                )
                for first_role, second_role
                in incompatible_pairs
                if (
                    first_role in roles
                    and second_role in roles
                )
            ]

            if not conflicting_pairs:
                continue

            result.add_finding(
                Finding(
                    code="P1-ROLE-002",
                    category=FindingCategory.COLUMN_ROLE,
                    severity=Severity.ERROR,
                    component=self.component_name,
                    message=(
                        f"Column {column!r} has incompatible "
                        "functional roles."
                    ),
                    columns=[column],
                    recommended_action=FindingAction.RAISE,
                    applied_action=FindingAction.REPORT,
                    blocks_readiness=True,
                    details={
                        "assigned_roles": sorted(
                            role.value
                            for role in roles
                        ),
                        "conflicting_pairs": (
                            conflicting_pairs
                        ),
                    },
                )
            )

        for column, roles in role_map.items():
            if roles:
                continue

            result.add_finding(
                Finding(
                    code="P1-ROLE-003",
                    category=FindingCategory.COLUMN_ROLE,
                    severity=Severity.WARNING,
                    component=self.component_name,
                    message=(
                        f"Column {column!r} has no assigned role."
                    ),
                    columns=[column],
                    recommended_action=FindingAction.WARN,
                    applied_action=FindingAction.REPORT,
                    blocks_readiness=False,
                )
            )

    def _register_column_states(
        self,
        *,
        data: pd.DataFrame,
        role_map: dict[str, set[ColumnRole]],
        state: Phase1State,
    ) -> None:
        """Create or update persistent state for every column."""

        unavailable_columns = set(
            self.options.inference_unavailable_columns
        )

        for column in data.columns:
            normalized_name = str(column)

            column_name_state = (
                state.column_name_states.get(
                    normalized_name
                )
            )

            if column_name_state is None:
                original_name = normalized_name
            else:
                original_name = (
                    column_name_state.original_name
                )

            roles = role_map[normalized_name]

            existing_state = state.column_states.get(
                normalized_name
            )

            if existing_state is None:
                column_state = ColumnState(
                    original_name=original_name,
                    normalized_name=normalized_name,
                    roles=set(roles),
                    required=True,
                    nullable=True,
                    use_as_model_feature=(
                        ColumnRole.FEATURE in roles
                    ),
                    available_at_inference=(
                        normalized_name
                        not in unavailable_columns
                        and ColumnRole.TARGET not in roles
                        and ColumnRole.OUTCOME_TIMESTAMP
                        not in roles
                        and ColumnRole.IGNORED not in roles
                    ),
                    is_sensitive=(
                        ColumnRole.SENSITIVE in roles
                    ),
                    dtype_before=str(
                        data[column].dtype
                    ),
                    dtype_after=str(
                        data[column].dtype
                    ),
                )
            else:
                existing_state.roles = set(roles)
                existing_state.use_as_model_feature = (
                    ColumnRole.FEATURE in roles
                )
                existing_state.available_at_inference = (
                    normalized_name
                    not in unavailable_columns
                    and ColumnRole.TARGET not in roles
                    and ColumnRole.OUTCOME_TIMESTAMP
                    not in roles
                    and ColumnRole.IGNORED not in roles
                )
                existing_state.is_sensitive = (
                    ColumnRole.SENSITIVE in roles
                )

                column_state = existing_state

            state.register_column_state(
                column_state
            )

    @staticmethod
    def _update_problem_contract(
        *,
        role_map: dict[str, set[ColumnRole]],
        state: Phase1State,
    ) -> None:
        """Update the resolved contract with role-derived columns."""

        contract = state.problem_contract

        if contract is None:
            return

        contract.feature_columns = [
            column
            for column, roles in role_map.items()
            if ColumnRole.FEATURE in roles
        ]

        contract.inference_columns = [
            column
            for column, roles in role_map.items()
            if ColumnRole.TARGET not in roles
            and ColumnRole.INTERNAL not in roles
            and ColumnRole.IGNORED not in roles
            and ColumnRole.OUTCOME_TIMESTAMP not in roles
        ]

    @staticmethod
    def _update_dataset_summary(
        *,
        role_map: dict[str, set[ColumnRole]],
        result: Phase1Result,
    ) -> None:
        """Update role counts in the dataset summary."""

        result.dataset_summary.feature_count = sum(
            ColumnRole.FEATURE in roles
            for roles in role_map.values()
        )

        result.dataset_summary.identifier_count = sum(
            ColumnRole.IDENTIFIER in roles
            for roles in role_map.values()
        )

        result.dataset_summary.timestamp_count = sum(
            bool(
                roles.intersection(
                    {
                        ColumnRole.TIMESTAMP,
                        ColumnRole.PREDICTION_TIMESTAMP,
                        ColumnRole.OUTCOME_TIMESTAMP,
                    }
                )
            )
            for roles in role_map.values()
        )

        result.dataset_summary.sensitive_column_count = sum(
            ColumnRole.SENSITIVE in roles
            for roles in role_map.values()
        )

        result.dataset_summary.ignored_column_count = sum(
            ColumnRole.IGNORED in roles
            for roles in role_map.values()
        )

    def _declared_columns(self) -> list[object]:
        """Return every explicitly declared column."""

        columns = [
            self.options.target_column,
            *(
                self.options.feature_columns
                if self.options.feature_columns is not None
                else []
            ),
            *self.options.identifier_columns,
            *self.options.group_columns,
            *self.options.timestamp_columns,
            *self.options.sensitive_columns,
            *self.options.metadata_columns,
            *self.options.ignored_columns,
            *self.options.inference_unavailable_columns,
        ]

        optional_columns = [
            self.options.prediction_timestamp_column,
            self.options.outcome_timestamp_column,
            self.options.sample_weight_column,
            self.options.fold_column,
        ]

        columns.extend(
            column
            for column in optional_columns
            if column is not None
        )

        return list(dict.fromkeys(columns))

    @staticmethod
    def _assign_single_role(
        *,
        role_map: dict[str, set[ColumnRole]],
        column: str | None,
        role: ColumnRole,
    ) -> None:
        """Assign one role to one column if the column exists."""

        if column is None:
            return

        if column not in role_map:
            return

        role_map[column].add(role)

    def _assign_multiple_roles(
        self,
        *,
        role_map: dict[str, set[ColumnRole]],
        columns: list[str],
        role: ColumnRole,
    ) -> None:
        """Assign one role to multiple columns."""

        for column in columns:
            self._assign_single_role(
                role_map=role_map,
                column=column,
                role=role,
            )