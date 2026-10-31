"""Column-role resolution for Phase 2 splitting."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from mlprep.phases.phase1.output_governance.state import Phase1State
from mlprep.phases.phase2.output_governance.metadata import (
    SplitFindingAction,
    SplitFindingCategory,
    SplitFindingSeverity,
    SplitStrategyName,
)
from mlprep.phases.phase2.output_governance.result import SplitFinding
from mlprep.phases.phase2.output_governance.state import (
    SplitColumnState,
)
from mlprep.types import ColumnRole


@dataclass(slots=True)
class SplitRoleOptions:
    """Options controlling Phase 2 column-role resolution."""

    target_column: str | None = None

    feature_columns: list[str] | None = None

    identifier_columns: list[str] = field(
        default_factory=list
    )

    group_columns: list[str] = field(
        default_factory=list
    )

    observation_key: list[str] = field(
        default_factory=list
    )

    timestamp_column: str | None = None

    excluded_feature_columns: list[str] = field(
        default_factory=list
    )

    internal_row_id_column: str = "__mlprep_row_id__"

    allow_phase1_role_fallback: bool = True

    remove_identifiers_from_features: bool = True
    remove_groups_from_features: bool = False
    remove_timestamps_from_features: bool = False

    require_at_least_one_feature: bool = True

    def __post_init__(self) -> None:
        """Validate split-role options."""

        if not self.internal_row_id_column.strip():
            raise ValueError(
                "internal_row_id_column cannot be empty."
            )

        if (
            self.target_column is not None
            and not self.target_column.strip()
        ):
            raise ValueError(
                "target_column cannot be empty."
            )

        duplicated_features = self._find_duplicates(
            self.feature_columns or []
        )

        if duplicated_features:
            raise ValueError(
                "feature_columns contains duplicate names: "
                f"{duplicated_features}"
            )

        duplicated_exclusions = self._find_duplicates(
            self.excluded_feature_columns
        )

        if duplicated_exclusions:
            raise ValueError(
                "excluded_feature_columns contains duplicate "
                f"names: {duplicated_exclusions}"
            )

    @staticmethod
    def _find_duplicates(
        values: list[str],
    ) -> list[str]:
        """Return duplicate values while preserving order."""

        seen: set[str] = set()
        duplicates: list[str] = []

        for value in values:
            if value in seen and value not in duplicates:
                duplicates.append(value)

            seen.add(value)

        return duplicates


class SplitRoleResolver:
    """Resolve columns used by Phase 2 split strategies."""

    component_name = "split_roles"

    def __init__(
        self,
        options: SplitRoleOptions | None = None,
    ) -> None:
        """Initialize the split-role resolver."""

        self.options = options or SplitRoleOptions()

    def resolve(
        self,
        *,
        data: pd.DataFrame,
        phase1_state: Phase1State,
        strategy: SplitStrategyName,
    ) -> tuple[
        SplitColumnState,
        list[SplitFinding],
    ]:
        """Resolve Phase 2 column roles and findings."""

        if not isinstance(data, pd.DataFrame):
            raise TypeError(
                "Split-role resolution requires a "
                "pandas DataFrame."
            )

        findings: list[SplitFinding] = []

        target_column = self._resolve_target_column(
            phase1_state=phase1_state
        )

        identifier_columns = (
            self._resolve_identifier_columns(
                phase1_state=phase1_state
            )
        )

        group_columns = self._resolve_group_columns(
            phase1_state=phase1_state
        )

        observation_key = (
            self._resolve_observation_key(
                phase1_state=phase1_state
            )
        )

        timestamp_column = (
            self._resolve_timestamp_column(
                phase1_state=phase1_state
            )
        )

        feature_columns = self._resolve_feature_columns(
            data=data,
            phase1_state=phase1_state,
            target_column=target_column,
            identifier_columns=identifier_columns,
            group_columns=group_columns,
            timestamp_column=timestamp_column,
        )

        columns = SplitColumnState(
            target_column=target_column,
            feature_columns=feature_columns,
            identifier_columns=identifier_columns,
            group_columns=group_columns,
            timestamp_column=timestamp_column,
            observation_key=observation_key,
            excluded_feature_columns=list(
                self.options.excluded_feature_columns
            ),
            internal_row_id_column=(
                self.options.internal_row_id_column
            ),
        )

        findings.extend(
            self._validate_resolved_roles(
                data=data,
                columns=columns,
                strategy=strategy,
            )
        )

        return columns, findings

    def _resolve_target_column(
        self,
        *,
        phase1_state: Phase1State,
    ) -> str:
        """Resolve the target column."""

        if self.options.target_column is not None:
            return self.options.target_column

        if (
            self.options.allow_phase1_role_fallback
            and phase1_state.problem_contract is not None
        ):
            return (
                phase1_state
                .problem_contract
                .target_column
            )

        target_columns = (
            phase1_state.get_columns_by_role(
                ColumnRole.TARGET
            )
        )

        if len(target_columns) == 1:
            return target_columns[0]

        if not target_columns:
            raise ValueError(
                "No target column could be resolved."
            )

        raise ValueError(
            "Multiple target columns were resolved: "
            f"{target_columns}"
        )

    def _resolve_identifier_columns(
        self,
        *,
        phase1_state: Phase1State,
    ) -> list[object]:
        """Resolve identifier columns."""

        if self.options.identifier_columns:
            return list(
                dict.fromkeys(
                    self.options.identifier_columns
                )
            )

        if not self.options.allow_phase1_role_fallback:
            return []

        return phase1_state.get_columns_by_role(
            ColumnRole.IDENTIFIER
        )

    def _resolve_group_columns(
        self,
        *,
        phase1_state: Phase1State,
    ) -> list[object]:
        """Resolve group columns."""

        if self.options.group_columns:
            return list(
                dict.fromkeys(
                    self.options.group_columns
                )
            )

        if (
            self.options.allow_phase1_role_fallback
            and phase1_state.problem_contract is not None
            and phase1_state
            .problem_contract
            .group_columns
        ):
            return list(
                phase1_state
                .problem_contract
                .group_columns
            )

        if not self.options.allow_phase1_role_fallback:
            return []

        return phase1_state.get_columns_by_role(
            ColumnRole.GROUP
        )

    def _resolve_observation_key(
        self,
        *,
        phase1_state: Phase1State,
    ) -> list[object]:
        """Resolve the business observation key."""

        if self.options.observation_key:
            return list(
                dict.fromkeys(
                    self.options.observation_key
                )
            )

        if (
            self.options.allow_phase1_role_fallback
            and phase1_state.problem_contract is not None
        ):
            return list(
                phase1_state
                .problem_contract
                .observation_key
            )

        return []

    def _resolve_timestamp_column(
        self,
        *,
        phase1_state: Phase1State,
    ) -> str | None:
        """Resolve the timestamp used for temporal splitting."""

        if self.options.timestamp_column is not None:
            return self.options.timestamp_column

        if (
            self.options.allow_phase1_role_fallback
            and phase1_state.problem_contract is not None
            and phase1_state
            .problem_contract
            .prediction_timestamp_column
            is not None
        ):
            return (
                phase1_state
                .problem_contract
                .prediction_timestamp_column
            )

        prediction_timestamps = (
            phase1_state.get_columns_by_role(
                ColumnRole.PREDICTION_TIMESTAMP
            )
        )

        if len(prediction_timestamps) == 1:
            return prediction_timestamps[0]

        timestamp_columns = (
            phase1_state.get_columns_by_role(
                ColumnRole.TIMESTAMP
            )
        )

        if len(timestamp_columns) == 1:
            return timestamp_columns[0]

        return None

    def _resolve_feature_columns(
        self,
        *,
        data: pd.DataFrame,
        phase1_state: Phase1State,
        target_column: str,
        identifier_columns: list[str],
        group_columns: list[str],
        timestamp_column: str | None,
    ) -> list[object]:
        """Resolve ordered model feature columns."""

        if self.options.feature_columns is not None:
            candidate_features = list(
                self.options.feature_columns
            )

        elif self.options.allow_phase1_role_fallback:
            candidate_features = (
                phase1_state
                .get_model_feature_columns()
            )

        else:
            candidate_features = [
                str(column)
                for column in data.columns
            ]

        excluded_columns = {
            target_column,
            self.options.internal_row_id_column,
            *self.options.excluded_feature_columns,
        }

        if self.options.remove_identifiers_from_features:
            excluded_columns.update(
                identifier_columns
            )

        if self.options.remove_groups_from_features:
            excluded_columns.update(
                group_columns
            )

        if (
            self.options.remove_timestamps_from_features
            and timestamp_column is not None
        ):
            excluded_columns.add(
                timestamp_column
            )

        feature_columns: list[str] = []

        for feature in candidate_features:
            if feature in excluded_columns:
                continue

            if feature not in data.columns:
                continue

            if feature in feature_columns:
                continue

            feature_columns.append(feature)

        return feature_columns

    def _validate_resolved_roles(
        self,
        *,
        data: pd.DataFrame,
        columns: SplitColumnState,
        strategy: SplitStrategyName,
    ) -> list[object]:
        """Validate resolved column roles."""

        findings: list[SplitFinding] = []

        available_columns = {
            str(column)
            for column in data.columns
        }

        referenced_columns = [
            columns.target_column,
            columns.internal_row_id_column,
            *columns.feature_columns,
            *columns.identifier_columns,
            *columns.group_columns,
            *columns.observation_key,
        ]

        if columns.timestamp_column is not None:
            referenced_columns.append(
                columns.timestamp_column
            )

        missing_columns = [
            column
            for column in dict.fromkeys(
                referenced_columns
            )
            if column not in available_columns
        ]

        if missing_columns:
            findings.append(
                SplitFinding(
                    code="P2-ROLE-001",
                    category=(
                        SplitFindingCategory.CONTRACT
                    ),
                    severity=(
                        SplitFindingSeverity.CRITICAL
                    ),
                    component=self.component_name,
                    message=(
                        "Resolved Phase 2 column roles reference "
                        "columns that do not exist."
                    ),
                    columns=missing_columns,
                    recommended_action=(
                        SplitFindingAction.RAISE
                    ),
                    applied_action=(
                        SplitFindingAction.REPORT
                    ),
                    blocks_readiness=True,
                )
            )

        if (
            columns.target_column
            in columns.feature_columns
        ):
            findings.append(
                SplitFinding(
                    code="P2-ROLE-002",
                    category=(
                        SplitFindingCategory.TARGET_LEAKAGE
                    ),
                    severity=(
                        SplitFindingSeverity.CRITICAL
                    ),
                    component=self.component_name,
                    message=(
                        "The target column is included in the "
                        "resolved feature list."
                    ),
                    columns=[
                        columns.target_column
                    ],
                    recommended_action=(
                        SplitFindingAction.DROP_FEATURE
                    ),
                    applied_action=(
                        SplitFindingAction.REPORT
                    ),
                    blocks_readiness=True,
                )
            )

        if (
            columns.internal_row_id_column
            in columns.feature_columns
        ):
            findings.append(
                SplitFinding(
                    code="P2-ROLE-003",
                    category=(
                        SplitFindingCategory.CONTRACT
                    ),
                    severity=(
                        SplitFindingSeverity.CRITICAL
                    ),
                    component=self.component_name,
                    message=(
                        "The internal row-ID column is included "
                        "in the resolved feature list."
                    ),
                    columns=[
                        columns.internal_row_id_column
                    ],
                    recommended_action=(
                        SplitFindingAction.DROP_FEATURE
                    ),
                    applied_action=(
                        SplitFindingAction.REPORT
                    ),
                    blocks_readiness=True,
                )
            )

        if (
            self.options.require_at_least_one_feature
            and not columns.feature_columns
        ):
            findings.append(
                SplitFinding(
                    code="P2-ROLE-004",
                    category=(
                        SplitFindingCategory.CONTRACT
                    ),
                    severity=(
                        SplitFindingSeverity.CRITICAL
                    ),
                    component=self.component_name,
                    message=(
                        "No model-eligible features remain after "
                        "Phase 2 role resolution."
                    ),
                    recommended_action=(
                        SplitFindingAction.RAISE
                    ),
                    applied_action=(
                        SplitFindingAction.REPORT
                    ),
                    blocks_readiness=True,
                )
            )

        group_strategies = {
            SplitStrategyName.GROUP,
            SplitStrategyName.GROUP_STRATIFIED,
        }

        if (
            strategy in group_strategies
            and not columns.group_columns
        ):
            findings.append(
                SplitFinding(
                    code="P2-ROLE-005",
                    category=(
                        SplitFindingCategory.CONTRACT
                    ),
                    severity=(
                        SplitFindingSeverity.CRITICAL
                    ),
                    component=self.component_name,
                    message=(
                        f"Strategy {strategy.value!r} requires "
                        "at least one group column."
                    ),
                    recommended_action=(
                        SplitFindingAction.RAISE
                    ),
                    applied_action=(
                        SplitFindingAction.REPORT
                    ),
                    blocks_readiness=True,
                )
            )

        temporal_strategies = {
            SplitStrategyName.TIME,
            SplitStrategyName.PURGED_TIME,
        }

        if (
            strategy in temporal_strategies
            and columns.timestamp_column is None
        ):
            findings.append(
                SplitFinding(
                    code="P2-ROLE-006",
                    category=(
                        SplitFindingCategory.CONTRACT
                    ),
                    severity=(
                        SplitFindingSeverity.CRITICAL
                    ),
                    component=self.component_name,
                    message=(
                        f"Strategy {strategy.value!r} requires "
                        "a timestamp column."
                    ),
                    recommended_action=(
                        SplitFindingAction.RAISE
                    ),
                    applied_action=(
                        SplitFindingAction.REPORT
                    ),
                    blocks_readiness=True,
                )
            )

        duplicated_features = self._find_duplicates(
            columns.feature_columns
        )

        if duplicated_features:
            findings.append(
                SplitFinding(
                    code="P2-ROLE-007",
                    category=(
                        SplitFindingCategory.CONTRACT
                    ),
                    severity=(
                        SplitFindingSeverity.CRITICAL
                    ),
                    component=self.component_name,
                    message=(
                        "The resolved feature list contains "
                        "duplicate names."
                    ),
                    columns=duplicated_features,
                    recommended_action=(
                        SplitFindingAction.RAISE
                    ),
                    applied_action=(
                        SplitFindingAction.REPORT
                    ),
                    blocks_readiness=True,
                )
            )

        identifier_features = set(
            columns.identifier_columns
        ).intersection(
            columns.feature_columns
        )

        if identifier_features:
            findings.append(
                SplitFinding(
                    code="P2-ROLE-008",
                    category=(
                        SplitFindingCategory.CONTRACT
                    ),
                    severity=(
                        SplitFindingSeverity.WARNING
                    ),
                    component=self.component_name,
                    message=(
                        "One or more identifier columns remain "
                        "in the feature list."
                    ),
                    columns=sorted(
                        identifier_features
                    ),
                    recommended_action=(
                        SplitFindingAction.WARN
                    ),
                    applied_action=(
                        SplitFindingAction.REPORT
                    ),
                    blocks_readiness=False,
                )
            )

        ignored_features = set(
            columns.excluded_feature_columns
        ).intersection(
            columns.feature_columns
        )

        if ignored_features:
            findings.append(
                SplitFinding(
                    code="P2-ROLE-009",
                    category=(
                        SplitFindingCategory.CONTRACT
                    ),
                    severity=(
                        SplitFindingSeverity.CRITICAL
                    ),
                    component=self.component_name,
                    message=(
                        "Excluded columns remain in the resolved "
                        "feature list."
                    ),
                    columns=sorted(
                        ignored_features
                    ),
                    recommended_action=(
                        SplitFindingAction.DROP_FEATURE
                    ),
                    applied_action=(
                        SplitFindingAction.REPORT
                    ),
                    blocks_readiness=True,
                )
            )

        return findings

    @staticmethod
    def _find_duplicates(
        values: list[str],
    ) -> list[object]:
        """Return duplicate values while preserving order."""

        seen: set[str] = set()
        duplicates: list[str] = []

        for value in values:
            if value in seen and value not in duplicates:
                duplicates.append(value)

            seen.add(value)

        return duplicates