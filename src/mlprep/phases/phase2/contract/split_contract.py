"""Input and output contract validation for Phase 2."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import pandas as pd

from mlprep.phases.phase1.output_governance.result import (
    Phase1Result,
)
from mlprep.phases.phase1.output_governance.state import (
    Phase1State,
)
from mlprep.phases.phase2.output_governance.metadata import (
    SplitFindingAction,
    SplitFindingCategory,
    SplitFindingSeverity,
    SplitRatios,
    SplitStrategyName,
)
from mlprep.phases.phase2.output_governance.result import (
    SplitFinding,
)
from mlprep.phases.phase2.output_governance.state import (
    SplitColumnState,
)

MissingGroupPolicy = Literal[
    "raise",
    "warn",
    "ignore",
]


@dataclass(slots=True)
class SplitContractOptions:
    """Options defining the Phase 2 split contract."""

    strategy: SplitStrategyName

    ratios: SplitRatios = field(
        default_factory=SplitRatios
    )

    random_seed: int = 42
    shuffle: bool = True

    target_column: str | None = None

    feature_columns: list[str] | None = None

    identifier_columns: list[str] = field(
        default_factory=list
    )

    group_columns: list[str] = field(
        default_factory=list
    )

    timestamp_column: str | None = None

    observation_key: list[str] = field(
        default_factory=list
    )

    excluded_feature_columns: list[str] = field(
        default_factory=list
    )

    internal_row_id_column: str = (
        "__mlprep_row_id__"
    )

    require_phase1_readiness: bool = True
    require_unique_row_ids: bool = True
    require_nonmissing_target: bool = True

    missing_group_policy: MissingGroupPolicy = "raise"

    lock_test_set: bool = True

    def __post_init__(self) -> None:
        """Validate split-contract options."""

        self.ratios.validate_total()

        if not self.internal_row_id_column.strip():
            raise ValueError(
                "internal_row_id_column cannot be empty."
            )

        strategies_requiring_groups = {
            SplitStrategyName.GROUP,
            SplitStrategyName.GROUP_STRATIFIED,
        }

        if (
            self.strategy
            in strategies_requiring_groups
            and not self.group_columns
        ):
            raise ValueError(
                f"Split strategy {self.strategy.value!r} "
                "requires at least one group column."
            )

        temporal_strategies = {
            SplitStrategyName.TIME,
            SplitStrategyName.PURGED_TIME,
        }

        if (
            self.strategy in temporal_strategies
            and self.timestamp_column is None
        ):
            raise ValueError(
                f"Split strategy {self.strategy.value!r} "
                "requires a timestamp column."
            )


class SplitContract:
    """Resolved contract used by Phase 2 split strategies."""

    def __init__(
        self,
        *,
        data: pd.DataFrame,
        phase1_result: Phase1Result,
        phase1_state: Phase1State,
        columns: SplitColumnState,
        options: SplitContractOptions,
        findings: list[SplitFinding],
    ) -> None:
        """Initialize the resolved split contract."""

        self.data = data
        self.phase1_result = phase1_result
        self.phase1_state = phase1_state
        self.columns = columns
        self.options = options
        self.findings = findings

    @property
    def target_column(
        self,
    ) -> str:
        """Return the resolved target column."""

        return self.columns.target_column

    @property
    def feature_columns(
        self,
    ) -> list[object]:
        """Return ordered model feature columns."""

        return list(
            self.columns.feature_columns
        )

    @property
    def internal_row_id_column(
        self,
    ) -> str:
        """Return the internal row-ID column."""

        return self.columns.internal_row_id_column

    def feature_frame(
        self,
    ) -> pd.DataFrame:
        """Return the complete unsplit feature frame."""

        return self.data.loc[
            :,
            self.columns.feature_columns,
        ].copy()

    def target_series(
        self,
    ) -> pd.Series:
        """Return the complete unsplit target series."""

        return self.data[
            self.columns.target_column
        ].copy()

    def row_id_series(
        self,
    ) -> pd.Series:
        """Return the complete internal row-ID series."""

        return self.data[
            self.columns.internal_row_id_column
        ].copy()


class SplitContractResolver:
    """Validate Phase 1 output and build the Phase 2 contract."""

    component_name = "split_contract"

    def __init__(
        self,
        options: SplitContractOptions,
    ) -> None:
        """Initialize the split-contract resolver."""

        self.options = options

    def resolve(
        self,
        *,
        phase1_result: Phase1Result,
        phase1_state: Phase1State,
    ) -> SplitContract:
        """Resolve and validate the complete Phase 2 input contract."""

        findings: list[SplitFinding] = []

        self._validate_phase1_readiness(
            phase1_result=phase1_result,
            findings=findings,
        )

        data = phase1_result.data

        if not isinstance(data, pd.DataFrame):
            raise TypeError(
                "Phase 1 result data must be a pandas DataFrame."
            )

        target_column = self._resolve_target_column(
            phase1_state=phase1_state,
        )

        feature_columns = self._resolve_feature_columns(
            data=data,
            target_column=target_column,
            phase1_state=phase1_state,
        )

        group_columns = self._resolve_group_columns(
            phase1_state=phase1_state,
        )

        observation_key = self._resolve_observation_key(
            phase1_state=phase1_state,
        )

        timestamp_column = self._resolve_timestamp_column(
            phase1_state=phase1_state,
        )

        columns = SplitColumnState(
            target_column=target_column,
            feature_columns=feature_columns,
            identifier_columns=list(
                self.options.identifier_columns
            ),
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

        self._validate_required_columns(
            data=data,
            columns=columns,
            findings=findings,
        )

        self._validate_target(
            data=data,
            target_column=target_column,
            findings=findings,
        )

        self._validate_row_identity(
            data=data,
            row_id_column=(
                columns.internal_row_id_column
            ),
            findings=findings,
        )

        self._validate_features(
            data=data,
            columns=columns,
            findings=findings,
        )

        self._validate_groups(
            data=data,
            group_columns=group_columns,
            findings=findings,
        )

        self._validate_timestamp(
            data=data,
            timestamp_column=timestamp_column,
            findings=findings,
        )

        return SplitContract(
            data=data.copy(deep=True),
            phase1_result=phase1_result,
            phase1_state=phase1_state,
            columns=columns,
            options=self.options,
            findings=findings,
        )

    def _validate_phase1_readiness(
        self,
        *,
        phase1_result: Phase1Result,
        findings: list[SplitFinding],
    ) -> None:
        """Require a successful Phase 1 handoff."""

        if not self.options.require_phase1_readiness:
            return

        readiness = phase1_result.readiness

        if readiness is None:
            findings.append(
                self._finding(
                    code="P2-CONTRACT-001",
                    severity=SplitFindingSeverity.CRITICAL,
                    message=(
                        "Phase 1 readiness has not been evaluated."
                    ),
                    blocks_readiness=True,
                )
            )

            return

        if not readiness.ready:
            findings.append(
                self._finding(
                    code="P2-CONTRACT-002",
                    severity=SplitFindingSeverity.CRITICAL,
                    message=(
                        "Phase 1 did not approve the dataset "
                        "for Phase 2."
                    ),
                    blocks_readiness=True,
                    details={
                        "phase1_status": (
                            readiness.status.value
                        ),
                        "phase1_reasons": (
                            readiness.reasons
                        ),
                    },
                )
            )

    def _resolve_target_column(
        self,
        *,
        phase1_state: Phase1State,
    ) -> str:
        """Resolve the target from options or Phase 1 state."""

        if self.options.target_column is not None:
            return self.options.target_column

        if phase1_state.problem_contract is None:
            raise ValueError(
                "Cannot resolve the target column because "
                "Phase 1 has no problem contract."
            )

        return (
            phase1_state.problem_contract.target_column
        )

    def _resolve_feature_columns(
        self,
        *,
        data: pd.DataFrame,
        target_column: str,
        phase1_state: Phase1State,
    ) -> list[object]:
        """Resolve ordered Phase 2 feature columns."""

        if self.options.feature_columns is not None:
            candidate_features = list(
                self.options.feature_columns
            )
        else:
            candidate_features = (
                phase1_state.get_model_feature_columns()
            )

        excluded_columns = {
            target_column,
            self.options.internal_row_id_column,
            *self.options.excluded_feature_columns,
        }

        return [
            column
            for column in candidate_features
            if column in data.columns
            and column not in excluded_columns
        ]

    def _resolve_group_columns(
        self,
        *,
        phase1_state: Phase1State,
    ) -> list[object]:
        """Resolve group columns."""

        if self.options.group_columns:
            return list(
                self.options.group_columns
            )

        if phase1_state.problem_contract is None:
            return []

        return list(
            phase1_state.problem_contract.group_columns
        )

    def _resolve_observation_key(
        self,
        *,
        phase1_state: Phase1State,
    ) -> list[object]:
        """Resolve the business observation key."""

        if self.options.observation_key:
            return list(
                self.options.observation_key
            )

        if phase1_state.problem_contract is None:
            return []

        return list(
            phase1_state.problem_contract.observation_key
        )

    def _resolve_timestamp_column(
        self,
        *,
        phase1_state: Phase1State,
    ) -> str | None:
        """Resolve the timestamp used by temporal strategies."""

        if self.options.timestamp_column is not None:
            return self.options.timestamp_column

        if phase1_state.problem_contract is None:
            return None

        return (
            phase1_state
            .problem_contract
            .prediction_timestamp_column
        )

    def _validate_required_columns(
        self,
        *,
        data: pd.DataFrame,
        columns: SplitColumnState,
        findings: list[SplitFinding],
    ) -> None:
        """Validate columns required to execute the split."""

        missing_columns = [
            column
            for column in columns.all_required_columns()
            if column not in data.columns
        ]

        if not missing_columns:
            return

        findings.append(
            self._finding(
                code="P2-CONTRACT-003",
                severity=SplitFindingSeverity.CRITICAL,
                message=(
                    "Columns required by the split contract "
                    "are missing."
                ),
                columns=missing_columns,
                blocks_readiness=True,
            )
        )

    def _validate_target(
        self,
        *,
        data: pd.DataFrame,
        target_column: str,
        findings: list[SplitFinding],
    ) -> None:
        """Validate the target before splitting."""

        if target_column not in data.columns:
            findings.append(
                self._finding(
                    code="P2-CONTRACT-004",
                    severity=SplitFindingSeverity.CRITICAL,
                    message=(
                        f"Target column {target_column!r} "
                        "does not exist."
                    ),
                    columns=[target_column],
                    blocks_readiness=True,
                )
            )

            return

        missing_count = int(
            data[target_column].isna().sum()
        )

        if (
            missing_count > 0
            and self.options.require_nonmissing_target
        ):
            findings.append(
                self._finding(
                    code="P2-CONTRACT-005",
                    severity=SplitFindingSeverity.ERROR,
                    message=(
                        "The target contains missing values "
                        "before splitting."
                    ),
                    columns=[target_column],
                    affected_row_count=missing_count,
                    blocks_readiness=True,
                )
            )

    def _validate_row_identity(
        self,
        *,
        data: pd.DataFrame,
        row_id_column: str,
        findings: list[SplitFinding],
    ) -> None:
        """Validate the internal row identity."""

        if row_id_column not in data.columns:
            findings.append(
                self._finding(
                    code="P2-CONTRACT-006",
                    severity=SplitFindingSeverity.CRITICAL,
                    message=(
                        f"Internal row-ID column "
                        f"{row_id_column!r} does not exist."
                    ),
                    columns=[row_id_column],
                    blocks_readiness=True,
                )
            )

            return

        missing_count = int(
            data[row_id_column].isna().sum()
        )

        if missing_count > 0:
            findings.append(
                self._finding(
                    code="P2-CONTRACT-007",
                    severity=SplitFindingSeverity.CRITICAL,
                    message=(
                        "The internal row-ID column contains "
                        "missing values."
                    ),
                    columns=[row_id_column],
                    affected_row_count=missing_count,
                    blocks_readiness=True,
                )
            )

        if (
            self.options.require_unique_row_ids
            and not data[row_id_column].is_unique
        ):
            duplicate_count = int(
                data[row_id_column].duplicated(
                    keep=False
                ).sum()
            )

            findings.append(
                self._finding(
                    code="P2-CONTRACT-008",
                    severity=SplitFindingSeverity.CRITICAL,
                    message=(
                        "The internal row-ID column is not unique."
                    ),
                    columns=[row_id_column],
                    affected_row_count=duplicate_count,
                    blocks_readiness=True,
                )
            )

    def _validate_features(
        self,
        *,
        data: pd.DataFrame,
        columns: SplitColumnState,
        findings: list[SplitFinding],
    ) -> None:
        """Validate model feature columns."""

        if not columns.feature_columns:
            findings.append(
                self._finding(
                    code="P2-CONTRACT-009",
                    severity=SplitFindingSeverity.CRITICAL,
                    message=(
                        "The split contract contains no "
                        "model-eligible features."
                    ),
                    blocks_readiness=True,
                )
            )

            return

        missing_features = [
            column
            for column in columns.feature_columns
            if column not in data.columns
        ]

        if missing_features:
            findings.append(
                self._finding(
                    code="P2-CONTRACT-010",
                    severity=SplitFindingSeverity.CRITICAL,
                    message=(
                        "One or more resolved feature columns "
                        "do not exist."
                    ),
                    columns=missing_features,
                    blocks_readiness=True,
                )
            )

        if (
            columns.target_column
            in columns.feature_columns
        ):
            findings.append(
                self._finding(
                    code="P2-CONTRACT-011",
                    severity=SplitFindingSeverity.CRITICAL,
                    message=(
                        "The target column is included in "
                        "the feature list."
                    ),
                    columns=[
                        columns.target_column
                    ],
                    blocks_readiness=True,
                )
            )

    def _validate_groups(
        self,
        *,
        data: pd.DataFrame,
        group_columns: list[str],
        findings: list[SplitFinding],
    ) -> None:
        """Validate configured group columns."""

        strategies_requiring_groups = {
            SplitStrategyName.GROUP,
            SplitStrategyName.GROUP_STRATIFIED,
        }

        if (
            self.options.strategy
            not in strategies_requiring_groups
        ):
            return

        missing_columns = [
            column
            for column in group_columns
            if column not in data.columns
        ]

        if missing_columns:
            findings.append(
                self._finding(
                    code="P2-CONTRACT-012",
                    severity=SplitFindingSeverity.CRITICAL,
                    message=(
                        "Required group columns are missing."
                    ),
                    columns=missing_columns,
                    blocks_readiness=True,
                )
            )

            return

        group_missing_mask = data[
            group_columns
        ].isna().any(axis=1)

        missing_group_count = int(
            group_missing_mask.sum()
        )

        if missing_group_count == 0:
            return

        severity_by_policy = {
            "raise": SplitFindingSeverity.ERROR,
            "warn": SplitFindingSeverity.WARNING,
            "ignore": SplitFindingSeverity.INFO,
        }

        findings.append(
            self._finding(
                code="P2-CONTRACT-013",
                severity=severity_by_policy[
                    self.options.missing_group_policy
                ],
                message=(
                    "One or more rows contain missing "
                    "group values."
                ),
                columns=group_columns,
                affected_row_count=missing_group_count,
                blocks_readiness=(
                    self.options.missing_group_policy
                    == "raise"
                ),
            )
        )

    def _validate_timestamp(
        self,
        *,
        data: pd.DataFrame,
        timestamp_column: str | None,
        findings: list[SplitFinding],
    ) -> None:
        """Validate the timestamp for temporal strategies."""

        temporal_strategies = {
            SplitStrategyName.TIME,
            SplitStrategyName.PURGED_TIME,
        }

        if (
            self.options.strategy
            not in temporal_strategies
        ):
            return

        if (
            timestamp_column is None
            or timestamp_column not in data.columns
        ):
            findings.append(
                self._finding(
                    code="P2-CONTRACT-014",
                    severity=SplitFindingSeverity.CRITICAL,
                    message=(
                        "The temporal split timestamp "
                        "column is missing."
                    ),
                    columns=(
                        [timestamp_column]
                        if timestamp_column
                        is not None
                        else []
                    ),
                    blocks_readiness=True,
                )
            )

            return

        parsed_timestamps = pd.to_datetime(
            data[timestamp_column],
            errors="coerce",
        )

        invalid_count = int(
            (
                data[timestamp_column].notna()
                & parsed_timestamps.isna()
            ).sum()
        )

        if invalid_count > 0:
            findings.append(
                self._finding(
                    code="P2-CONTRACT-015",
                    severity=SplitFindingSeverity.ERROR,
                    message=(
                        "The split timestamp contains "
                        "unparseable values."
                    ),
                    columns=[timestamp_column],
                    affected_row_count=invalid_count,
                    blocks_readiness=True,
                )
            )

    def _finding(
        self,
        *,
        code: str,
        severity: SplitFindingSeverity,
        message: str,
        columns: list[str] | None = None,
        affected_row_count: int = 0,
        blocks_readiness: bool,
        details: dict[str, object] | None = None,
    ) -> SplitFinding:
        """Build one split-contract finding."""

        return SplitFinding(
            code=code,
            category=SplitFindingCategory.CONTRACT,
            severity=severity,
            component=self.component_name,
            message=message,
            columns=columns or [],
            affected_row_count=affected_row_count,
            recommended_action=(
                SplitFindingAction.RAISE
                if blocks_readiness
                else SplitFindingAction.REPORT
            ),
            applied_action=(
                SplitFindingAction.REPORT
            ),
            blocks_readiness=blocks_readiness,
            details=details or {},
        )
