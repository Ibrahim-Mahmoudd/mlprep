"""Input-contract validation for Phase 3."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import pandas as pd

from mlprep.phases.phase1.output_governance.state import (
    Phase1State,
)
from mlprep.phases.phase2.output_governance.metadata import (
    Phase2Status,
)
from mlprep.phases.phase2.output_governance.result import (
    DatasetPartition,
    LockedTestPartition,
    Phase2Result,
)
from mlprep.phases.phase2.output_governance.state import (
    Phase2State,
)

ContractSeverity = Literal[
    "info",
    "warning",
    "error",
    "critical",
]


@dataclass(slots=True)
class EDAContractIssue:
    """One issue detected while resolving the Phase 3 contract."""

    code: str
    severity: ContractSeverity
    message: str

    columns: list[str] = field(
        default_factory=list
    )

    blocks_execution: bool = False

    details: dict[str, object] = field(
        default_factory=dict
    )


@dataclass(slots=True)
class EDAContractOptions:
    """Options controlling the Phase 3 input contract."""

    require_phase2_readiness: bool = True
    require_locked_test_partition: bool = True

    require_nonempty_training_partition: bool = True
    require_training_target: bool = True
    require_nonmissing_training_target: bool = True

    require_unique_feature_names: bool = True
    require_unique_row_ids: bool = True
    require_aligned_indices: bool = True

    prohibit_target_in_features: bool = True
    prohibit_internal_row_id_in_features: bool = True

    require_phase1_state: bool = True
    require_phase2_state: bool = True

    minimum_training_rows: int = 2
    minimum_feature_count: int = 1

    allow_validation_analysis: bool = True
    allow_test_analysis: bool = False

    copy_training_data: bool = True
    copy_validation_data: bool = True

    def __post_init__(self) -> None:
        """Validate EDA contract options."""

        if self.minimum_training_rows < 1:
            raise ValueError(
                "minimum_training_rows must be at least one."
            )

        if self.minimum_feature_count < 1:
            raise ValueError(
                "minimum_feature_count must be at least one."
            )

        if self.allow_test_analysis:
            raise ValueError(
                "Phase 3 must not enable test-set analysis."
            )


class EDAContract:
    """Resolved input contract used by Phase 3."""

    def __init__(
        self,
        *,
        train: DatasetPartition,
        validation: DatasetPartition | None,
        phase1_state: Phase1State,
        phase2_state: Phase2State,
        issues: list[EDAContractIssue],
        options: EDAContractOptions,
    ) -> None:
        """Initialize the resolved EDA contract."""

        self.train = train
        self.validation = validation

        self.phase1_state = phase1_state
        self.phase2_state = phase2_state

        self.issues = issues
        self.options = options

    @property
    def target_column(self) -> str:
        """Return the target-column name."""

        return self.phase2_state.columns.target_column

    @property
    def feature_columns(self) -> list[str]:
        """Return ordered training feature names."""

        return self.train.feature_names()

    @property
    def internal_row_id_column(self) -> str:
        """Return the internal row-ID column name."""

        return (
            self.phase2_state
            .columns
            .internal_row_id_column
        )

    @property
    def training_row_count(self) -> int:
        """Return the number of training rows."""

        return self.train.row_count()

    @property
    def training_feature_count(self) -> int:
        """Return the number of training features."""

        return self.train.feature_count()

    @property
    def has_blocking_issues(self) -> bool:
        """Return whether the contract has blocking issues."""

        return any(
            issue.blocks_execution
            or issue.severity == "critical"
            for issue in self.issues
        )

    def training_features(self) -> pd.DataFrame:
        """Return a copy of the training features."""

        return self.train.X.copy(
            deep=self.options.copy_training_data
        )

    def training_target(self) -> pd.Series:
        """Return a copy of the training target."""

        return self.train.y.copy(
            deep=self.options.copy_training_data
        )

    def training_row_ids(self) -> pd.Series:
        """Return a copy of the training row IDs."""

        return self.train.row_ids.copy(
            deep=self.options.copy_training_data
        )

    def validation_features(
        self,
    ) -> pd.DataFrame | None:
        """Return validation features when permitted."""

        if (
            not self.options.allow_validation_analysis
            or self.validation is None
        ):
            return None

        return self.validation.X.copy(
            deep=self.options.copy_validation_data
        )

    def validation_target(
        self,
    ) -> pd.Series | None:
        """Return the validation target when permitted."""

        if (
            not self.options.allow_validation_analysis
            or self.validation is None
        ):
            return None

        return self.validation.y.copy(
            deep=self.options.copy_validation_data
        )

    def assert_ready(self) -> None:
        """Raise when blocking contract issues exist."""

        blocking_issues = [
            issue
            for issue in self.issues
            if (
                issue.blocks_execution
                or issue.severity == "critical"
            )
        ]

        if not blocking_issues:
            return

        codes = [
            issue.code
            for issue in blocking_issues
        ]

        raise ValueError(
            "The Phase 3 EDA contract contains blocking "
            f"issues: {codes}"
        )


class EDAContractResolver:
    """Validate Phase 2 output and build the Phase 3 contract."""

    component_name = "eda_contract"

    def __init__(
        self,
        options: EDAContractOptions | None = None,
    ) -> None:
        """Initialize the EDA contract resolver."""

        self.options = options or EDAContractOptions()

    def resolve(
        self,
        *,
        phase2_result: Phase2Result,
        phase1_state: Phase1State | None = None,
        phase2_state: Phase2State | None = None,
    ) -> EDAContract:
        """Resolve and validate the complete Phase 3 contract."""

        if not isinstance(
            phase2_result,
            Phase2Result,
        ):
            raise TypeError(
                "phase2_result must be an instance "
                "of Phase2Result."
            )

        resolved_phase2_state = (
            phase2_state
            or phase2_result.state
        )

        if not isinstance(
            resolved_phase2_state,
            Phase2State,
        ):
            raise TypeError(
                "A valid Phase2State is required."
            )

        resolved_phase1_state = (
            phase1_state
            or self._resolve_phase1_state(
                phase2_state=resolved_phase2_state
            )
        )

        issues: list[EDAContractIssue] = []

        self._validate_phase2_readiness(
            phase2_result=phase2_result,
            issues=issues,
        )

        self._validate_test_lock(
            phase2_result=phase2_result,
            phase2_state=resolved_phase2_state,
            issues=issues,
        )

        train = phase2_result.train

        validation = (
            phase2_result.validation
            if self.options.allow_validation_analysis
            else None
        )

        self._validate_training_partition(
            train=train,
            phase2_state=resolved_phase2_state,
            issues=issues,
        )

        if validation is not None:
            self._validate_validation_partition(
                train=train,
                validation=validation,
                issues=issues,
            )

        return EDAContract(
            train=train.copy(
                deep=self.options.copy_training_data
            ),
            validation=(
                validation.copy(
                    deep=self.options
                    .copy_validation_data
                )
                if validation is not None
                else None
            ),
            phase1_state=resolved_phase1_state,
            phase2_state=resolved_phase2_state,
            issues=issues,
            options=self.options,
        )

    def _validate_phase2_readiness(
        self,
        *,
        phase2_result: Phase2Result,
        issues: list[EDAContractIssue],
    ) -> None:
        """Validate the Phase 2 readiness decision."""

        if not self.options.require_phase2_readiness:
            return

        readiness = phase2_result.readiness

        if readiness is None:
            issues.append(
                EDAContractIssue(
                    code="P3-CONTRACT-001",
                    severity="critical",
                    message=(
                        "Phase 2 readiness has not been evaluated."
                    ),
                    blocks_execution=True,
                )
            )

            return

        if not readiness.ready:
            issues.append(
                EDAContractIssue(
                    code="P3-CONTRACT-002",
                    severity="critical",
                    message=(
                        "Phase 2 did not approve the dataset "
                        "for Phase 3."
                    ),
                    blocks_execution=True,
                    details={
                        "phase2_status": (
                            readiness.status.value
                        ),
                        "phase2_reasons": list(
                            readiness.reasons
                        ),
                    },
                )
            )

        if readiness.status == Phase2Status.FAILED:
            issues.append(
                EDAContractIssue(
                    code="P3-CONTRACT-003",
                    severity="critical",
                    message=(
                        "Phase 2 completed with failed status."
                    ),
                    blocks_execution=True,
                )
            )

    def _validate_test_lock(
        self,
        *,
        phase2_result: Phase2Result,
        phase2_state: Phase2State,
        issues: list[EDAContractIssue],
    ) -> None:
        """Ensure the test partition remains protected."""

        if not self.options.require_locked_test_partition:
            return

        if not isinstance(
            phase2_result.test,
            LockedTestPartition,
        ):
            issues.append(
                EDAContractIssue(
                    code="P3-CONTRACT-004",
                    severity="critical",
                    message=(
                        "The Phase 2 test partition is not "
                        "protected by LockedTestPartition."
                    ),
                    blocks_execution=True,
                )
            )

        if not phase2_state.test_lock.locked:
            issues.append(
                EDAContractIssue(
                    code="P3-CONTRACT-005",
                    severity="critical",
                    message=(
                        "The Phase 2 test-lock state is not locked."
                    ),
                    blocks_execution=True,
                )
            )

        if phase2_state.test_lock.access_count > 0:
            issues.append(
                EDAContractIssue(
                    code="P3-CONTRACT-006",
                    severity="warning",
                    message=(
                        "The test partition has an existing "
                        "authorized-access history."
                    ),
                    blocks_execution=False,
                    details={
                        "access_count": (
                            phase2_state
                            .test_lock
                            .access_count
                        ),
                    },
                )
            )

    def _validate_training_partition(
        self,
        *,
        train: DatasetPartition,
        phase2_state: Phase2State,
        issues: list[EDAContractIssue],
    ) -> None:
        """Validate training data for analytical profiling."""

        if not isinstance(
            train,
            DatasetPartition,
        ):
            issues.append(
                EDAContractIssue(
                    code="P3-CONTRACT-007",
                    severity="critical",
                    message=(
                        "The training partition is invalid."
                    ),
                    blocks_execution=True,
                )
            )

            return

        if (
            self.options
            .require_nonempty_training_partition
            and train.row_count() == 0
        ):
            issues.append(
                EDAContractIssue(
                    code="P3-CONTRACT-008",
                    severity="critical",
                    message=(
                        "The training partition is empty."
                    ),
                    blocks_execution=True,
                )
            )

        if (
            train.row_count()
            < self.options.minimum_training_rows
        ):
            issues.append(
                EDAContractIssue(
                    code="P3-CONTRACT-009",
                    severity="critical",
                    message=(
                        "The training partition contains fewer "
                        "rows than the configured minimum."
                    ),
                    blocks_execution=True,
                    details={
                        "training_row_count": (
                            train.row_count()
                        ),
                        "minimum_training_rows": (
                            self.options
                            .minimum_training_rows
                        ),
                    },
                )
            )

        if (
            train.feature_count()
            < self.options.minimum_feature_count
        ):
            issues.append(
                EDAContractIssue(
                    code="P3-CONTRACT-010",
                    severity="critical",
                    message=(
                        "The training partition contains fewer "
                        "featuresgured minimum."
                    ),
                    blocks_execution=True,
                    details={
                        "feature_count": (
                            train.feature_count()
                        ),
                        "minimum_feature_count": (
                            self.options
                            .minimum_feature_count
                        ),
                    },
                )
            )

        if (
            self.options.require_training_target
            and train.y is None
        ):
            issues.append(
                EDAContractIssue(
                    code="P3-CONTRACT-011",
                    severity="critical",
                    message=(
                        "The training target is missing."
                    ),
                    blocks_execution=True,
                )
            )

            return

        if (
            self.options
            .require_nonmissing_training_target
            and train.y.isna().any()
        ):
            issues.append(
                EDAContractIssue(
                    code="P3-CONTRACT-012",
                    severity="critical",
                    message=(
                        "The training target contains "
                        "missing values."
                    ),
                    blocks_execution=True,
                    details={
                        "missing_target_count": int(
                            train.y.isna().sum()
                        ),
                    },
                )
            )

        if (
            self.options.require_aligned_indices
            and (
                not train.X.index.equals(
                    train.y.index
                )
                or not train.X.index.equals(
                    train.row_ids.index
                )
            )
        ):
            issues.append(
                EDAContractIssue(
                    code="P3-CONTRACT-013",
                    severity="critical",
                    message=(
                        "Training features, target, and row IDs "
                        "are not index-aligned."
                    ),
                    blocks_execution=True,
                )
            )

        if (
            self.options.require_unique_row_ids
            and not train.row_ids.is_unique
        ):
            issues.append(
                EDAContractIssue(
                    code="P3-CONTRACT-014",
                    severity="critical",
                    message=(
                        "The training partition contains "
                        "duplicate internal row IDs."
                    ),
                    blocks_execution=True,
                )
            )

        feature_names = train.feature_names()

        if (
            self.options.require_unique_feature_names
            and len(feature_names)
            != len(set(feature_names))
        ):
            issues.append(
                EDAContractIssue(
                    code="P3-CONTRACT-015",
                    severity="critical",
                    message=(
                        "The training feature matrix contains "
                        "duplicate feature names."
                    ),
                    blocks_execution=True,
                )
            )

        target_column = (
            phase2_state.columns.target_column
        )

        if (
            self.options.prohibit_target_in_features
            and target_column in feature_names
        ):
            issues.append(
                EDAContractIssue(
                    code="P3-CONTRACT-016",
                    severity="critical",
                    message=(
                        "The target column is included in the "
                        "Phase 3 feature matrix."
                    ),
                    columns=[target_column],
                    blocks_execution=True,
                )
            )

        internal_row_id_column = (
            phase2_state
            .columns
            .internal_row_id_column
        )

        if (
            self.options
            .prohibit_internal_row_id_in_features
            and internal_row_id_column
            in feature_names
        ):
            issues.append(
                EDAContractIssue(
                    code="P3-CONTRACT-017",
                    severity="critical",
                    message=(
                        "The internal row-ID column is included "
                        "in the Phase 3 feature matrix."
                    ),
                    columns=[
                        internal_row_id_column
                    ],
                    blocks_execution=True,
                )
            )

    def _validate_validation_partition(
        self,
        *,
        train: DatasetPartition,
        validation: DatasetPartition,
        issues: list[EDAContractIssue],
    ) -> None:
        """Validate validation data for drift comparison."""

        if validation.row_count() == 0:
            issues.append(
                EDAContractIssue(
                    code="P3-CONTRACT-018",
                    severity="warning",
                    message=(
                        "The validation partition is empty, so "
                        "distribution-shift diagnostics cannot run."
                    ),
                    blocks_execution=False,
                )
            )

            return

        if (
            train.feature_names()
            != validation.feature_names()
        ):
            issues.append(
                EDAContractIssue(
                    code="P3-CONTRACT-019",
                    severity="critical",
                    message=(
                        "Training and validation feature orders "
                        "do not match."
                    ),
                    blocks_execution=True,
                )
            )

        if (
            not validation.X.index.equals(
                validation.y.index
            )
            or not validation.X.index.equals(
                validation.row_ids.index
            )
        ):
            issues.append(
                EDAContractIssue(
                    code="P3-CONTRACT-020",
                    severity="critical",
                    message=(
                        "Validation features, target, and row IDs "
                        "are not index-aligned."
                    ),
                    blocks_execution=True,
                )
            )

    def _resolve_phase1_state(
        self,
        *,
        phase2_state: Phase2State,
    ) -> Phase1State:
        """Resolve Phase 1 state from Phase 2 metadata."""

        if self.options.require_phase1_state:
            serialized_state = (
                phase2_state.metadata.get(
                    "phase1_state"
                )
            )

            if serialized_state is None:
                raise ValueError(
                    "Phase 1 state is required for Phase 3. "
                    "Pass phase1_state explicitly or store it "
                    "inside Phase2State.metadata."
                )

            return Phase1State.model_validate(
                serialized_state
            )

        return Phase1State()
