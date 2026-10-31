"""Input-contract validation for Phase 4."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from mlprep.phases.phase1.output_governance.state import (
    Phase1State,
)
from mlprep.phases.phase2.output_governance.metadata import (
    PartitionName,
)
from mlprep.phases.phase2.output_governance.result import (
    DatasetPartition,
    LockedTestPartition,
    Phase2Result,
)
from mlprep.phases.phase2.output_governance.state import (
    Phase2State,
)
from mlprep.phases.phase3.output_governance.result import (
    Phase3Result,
)
from mlprep.phases.phase3.output_governance.state import (
    Phase3State,
)
from mlprep.phases.phase3.recommendations.recommendation_engine import (
    Phase3Recommendations,
)

ContractSeverity = Literal[
    "info",
    "warning",
    "error",
    "critical",
]


@dataclass(slots=True)
class ImputationContractIssue:
    """One issue detected in the Phase 4 contract."""

    code: str
    severity: ContractSeverity
    message: str

    columns: list[str] = field(
        default_factory=list
    )

    partitions: list[PartitionName] = field(
        default_factory=list
    )

    blocks_execution: bool = False

    details: dict[str, object] = field(
        default_factory=dict
    )


@dataclass(slots=True)
class ImputationContractOptions:
    """Options controlling the Phase 4 input contract."""

    require_phase3_readiness: bool = True
    require_phase3_recommendations: bool = True

    require_locked_test_partition: bool = True
    prohibit_existing_test_access: bool = False

    require_nonempty_training_partition: bool = True
    require_validation_partition: bool = True
    require_test_partition: bool = True

    require_aligned_indices: bool = True
    require_identical_feature_schema: bool = True
    require_identical_feature_order: bool = True

    require_unique_row_ids: bool = True
    require_disjoint_row_ids: bool = True

    require_training_fingerprint_match: bool = True
    require_validation_fingerprint_match: bool = True

    prohibit_target_imputation: bool = True
    prohibit_internal_row_id_imputation: bool = True

    copy_input_partitions: bool = True

    def __post_init__(self) -> None:
        """Validate contract options."""

        boolean_fields = {
            "require_phase3_readiness": (
                self.require_phase3_readiness
            ),
            "require_phase3_recommendations": (
                self.require_phase3_recommendations
            ),
            "require_locked_test_partition": (
                self.require_locked_test_partition
            ),
            "prohibit_existing_test_access": (
                self.prohibit_existing_test_access
            ),
            "require_nonempty_training_partition": (
                self.require_nonempty_training_partition
            ),
            "require_validation_partition": (
                self.require_validation_partition
            ),
            "require_test_partition": (
                self.require_test_partition
            ),
            "require_aligned_indices": (
                self.require_aligned_indices
            ),
            "require_identical_feature_schema": (
                self.require_identical_feature_schema
            ),
            "require_identical_feature_order": (
                self.require_identical_feature_order
            ),
            "require_unique_row_ids": (
                self.require_unique_row_ids
            ),
            "require_disjoint_row_ids": (
                self.require_disjoint_row_ids
            ),
            "require_training_fingerprint_match": (
                self.require_training_fingerprint_match
            ),
            "require_validation_fingerprint_match": (
                self.require_validation_fingerprint_match
            ),
            "prohibit_target_imputation": (
                self.prohibit_target_imputation
            ),
            "prohibit_internal_row_id_imputation": (
                self.prohibit_internal_row_id_imputation
            ),
            "copy_input_partitions": (
                self.copy_input_partitions
            ),
        }

        for name, value in boolean_fields.items():
            if not isinstance(value, bool):
                raise TypeError(
                    f"{name} must be a boolean."
                )


class ImputationContract:
    """Resolved input contract used by Phase 4."""

    def __init__(
        self,
        *,
        train: DatasetPartition,
        validation: DatasetPartition,
        test: DatasetPartition,
        phase1_state: Phase1State,
        phase2_state: Phase2State,
        phase3_state: Phase3State,
        phase3_recommendations: Phase3Recommendations,
        issues: list[ImputationContractIssue],
        options: ImputationContractOptions,
    ) -> None:
        """Initialize the resolved imputation contract."""

        self.train = train
        self.validation = validation
        self.test = test

        self.phase1_state = phase1_state
        self.phase2_state = phase2_state
        self.phase3_state = phase3_state

        self.phase3_recommendations = (
            phase3_recommendations
        )

        self.issues = issues
        self.options = options

    @property
    def target_column(self) -> str:
        """Return the target column."""

        return self.phase2_state.columns.target_column

    @property
    def feature_columns(self) -> list[str]:
        """Return ordered model feature columns."""

        return self.train.feature_names()

    @property
    def internal_row_id_column(self) -> str:
        """Return the internal row-ID column."""

        return (
            self.phase2_state
            .columns
            .internal_row_id_column
        )

    @property
    def group_columns(self) -> list[str]:
        """Return configured group columns."""

        return list(
            self.phase2_state
            .columns
            .group_columns
        )

    @property
    def timestamp_column(self) -> str | None:
        """Return the configured timestamp column."""

        return (
            self.phase2_state
            .columns
            .timestamp_column
        )

    @property
    def has_blocking_issues(self) -> bool:
        """Return whether blocking issues exist."""

        return any(
            issue.blocks_execution
            or issue.severity == "critical"
            for issue in self.issues
        )

    def assert_ready(self) -> None:
        """Raise when contract issues block execution."""

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
            "The Phase 4 imputation contract contains "
            f"blocking issues: {codes}"
        )

    def partition(
        self,
        name: PartitionName,
    ) -> DatasetPartition:
        """Return one input partition."""

        if name == PartitionName.TRAIN:
            return self.train

        if name == PartitionName.VALIDATION:
            return self.validation

        if name == PartitionName.TEST:
            return self.test

        raise KeyError(
            f"Partition {name.value!r} is unavailable."
        )


class ImputationContractResolver:
    """Resolve and validate the Phase 4 input contract."""

    component_name = "imputation_contract"

    def __init__(
        self,
        options: ImputationContractOptions | None = None,
    ) -> None:
        """Initialize the contract resolver."""

        self.options = (
            options or ImputationContractOptions()
        )

    def resolve(
        self,
        *,
        phase2_result: Phase2Result,
        phase3_result: Phase3Result,
        phase1_state: Phase1State,
        phase2_state: Phase2State | None = None,
        phase3_state: Phase3State | None = None,
        test_partition: DatasetPartition | None = None,
    ) -> ImputationContract:
        """Resolve the complete Phase 4 contract."""

        self._validate_input_types(
            phase2_result=phase2_result,
            phase3_result=phase3_result,
            phase1_state=phase1_state,
        )

        resolved_phase2_state = (
            phase2_state
            or phase2_result.state
        )

        resolved_phase3_state = (
            phase3_state
            or self._resolve_phase3_state(
                phase3_result
            )
        )

        issues: list[
            ImputationContractIssue
        ] = []

        self._validate_phase3_readiness(
            phase3_result=phase3_result,
            issues=issues,
        )

        recommendations = (
            self._resolve_recommendations(
                phase3_result=phase3_result,
                issues=issues,
            )
        )

        self._validate_test_lock(
            phase2_result=phase2_result,
            phase2_state=resolved_phase2_state,
            test_partition=test_partition,
            issues=issues,
        )

        resolved_test = self._resolve_test_partition(
            phase2_result=phase2_result,
            test_partition=test_partition,
        )

        train = self._copy_partition(
            phase2_result.train
        )

        validation = self._copy_partition(
            phase2_result.validation
        )

        test = self._copy_partition(
            resolved_test
        )

        self._validate_partitions(
            train=train,
            validation=validation,
            test=test,
            issues=issues,
        )

        self._validate_protected_columns(
            train=train,
            phase2_state=resolved_phase2_state,
            recommendations=recommendations,
            issues=issues,
        )

        self._validate_fingerprints(
            train=train,
            validation=validation,
            phase3_state=resolved_phase3_state,
            issues=issues,
        )

        return ImputationContract(
            train=train,
            validation=validation,
            test=test,
            phase1_state=phase1_state,
            phase2_state=resolved_phase2_state,
            phase3_state=resolved_phase3_state,
            phase3_recommendations=recommendations,
            issues=issues,
            options=self.options,
        )

    def _validate_phase3_readiness(
        self,
        *,
        phase3_result: Phase3Result,
        issues: list[ImputationContractIssue],
    ) -> None:
        """Validate Phase 3 readiness."""

        if not self.options.require_phase3_readiness:
            return

        readiness = phase3_result.readiness

        if readiness is None:
            issues.append(
                ImputationContractIssue(
                    code="P4-CONTRACT-001",
                    severity="critical",
                    message=(
                        "Phase 3 readiness has not been evaluated."
                    ),
                    blocks_execution=True,
                )
            )

            return

        if not readiness.ready:
            issues.append(
                ImputationContractIssue(
                    code="P4-CONTRACT-002",
                    severity="critical",
                    message=(
                        "Phase 3 did not approve the dataset "
                        "for Phase 4."
                    ),
                    blocks_execution=True,
                    details={
                        "phase3_status": (
                            readiness.status.value
                        ),
                        "phase3_reasons": list(
                            readiness.reasons
                        ),
                    },
                )
            )

    def _resolve_recommendations(
        self,
        *,
        phase3_result: Phase3Result,
        issues: list[ImputationContractIssue],
    ) -> Phase3Recommendations:
        """Resolve Phase 3 recommendations."""

        recommendations = (
            phase3_result.recommendations
        )

        if recommendations is not None:
            return recommendations

        issues.append(
            ImputationContractIssue(
                code="P4-CONTRACT-003",
                severity=(
                    "critical"
                    if self.options
                    .require_phase3_recommendations
                    else "warning"
                ),
                message=(
                    "Phase 3 imputation recommendations "
                    "are unavailable."
                ),
                blocks_execution=(
                    self.options
                    .require_phase3_recommendations
                ),
            )
        )

        return Phase3Recommendations.model_validate(
            {
                "transformations": {},
                "imputation": {},
                "outliers": {},
                "encoding": {},
                "interactions": {},
            }
        )

    def _validate_test_lock(
        self,
        *,
        phase2_result: Phase2Result,
        phase2_state: Phase2State,
        test_partition: DatasetPartition | None,
        issues: list[ImputationContractIssue],
    ) -> None:
        """Validate test-lock governance."""

        if not self.options.require_locked_test_partition:
            return

        if not isinstance(
            phase2_result.test,
            LockedTestPartition,
        ):
            issues.append(
                ImputationContractIssue(
                    code="P4-CONTRACT-004",
                    severity="critical",
                    message=(
                        "The Phase 2 test partition is not locked."
                    ),
                    partitions=[
                        PartitionName.TEST
                    ],
                    blocks_execution=True,
                )
            )

        if not phase2_state.test_lock.locked:
            issues.append(
                ImputationContractIssue(
                    code="P4-CONTRACT-005",
                    severity="critical",
                    message=(
                        "The Phase 2 test-lock state is not locked."
                    ),
                    partitions=[
                        PartitionName.TEST
                    ],
                    blocks_execution=True,
                )
            )

        if (
            self.options.prohibit_existing_test_access
            and phase2_state
            .test_lock
            .access_count
            > 0
        ):
            issues.append(
                ImputationContractIssue(
                    code="P4-CONTRACT-006",
                    severity="critical",
                    message=(
                        "The test partition has an existing "
                        "access history."
                    ),
                    partitions=[
                        PartitionName.TEST
                    ],
                    blocks_execution=True,
                    details={
                        "access_count": (
                            phase2_state
                            .test_lock
                            .access_count
                        ),
                    },
                )
            )

        if test_partition is None:
            issues.append(
                ImputationContractIssue(
                    code="P4-CONTRACT-007",
                    severity="critical",
                    message=(
                        "Phase 4 requires an authorized test "
                        "partition for transformation."
                    ),
                    partitions=[
                        PartitionName.TEST
                    ],
                    blocks_execution=True,
                )
            )

    def _validate_partitions(
        self,
        *,
        train: DatasetPartition,
        validation: DatasetPartition,
        test: DatasetPartition,
        issues: list[ImputationContractIssue],
    ) -> None:
        """Validate input partition integrity."""

        partitions = {
            PartitionName.TRAIN: train,
            PartitionName.VALIDATION: validation,
            PartitionName.TEST: test,
        }

        if (
            self.options
            .require_nonempty_training_partition
            and train.row_count() == 0
        ):
            issues.append(
                ImputationContractIssue(
                    code="P4-CONTRACT-008",
                    severity="critical",
                    message=(
                        "The training partition is empty."
                    ),
                    partitions=[
                        PartitionName.TRAIN
                    ],
                    blocks_execution=True,
                )
            )

        if (
            self.options.require_validation_partition
            and validation.row_count() == 0
        ):
            issues.append(
                ImputationContractIssue(
                    code="P4-CONTRACT-009",
                    severity="critical",
                    message=(
                        "The validation partition is empty."
                    ),
                    partitions=[
                        PartitionName.VALIDATION
                    ],
                    blocks_execution=True,
                )
            )

        if (
            self.options.require_test_partition
            and test.row_count() == 0
        ):
            issues.append(
                ImputationContractIssue(
                    code="P4-CONTRACT-010",
                    severity="critical",
                    message=(
                        "The test partition is empty."
                    ),
                    partitions=[
                        PartitionName.TEST
                    ],
                    blocks_execution=True,
                )
            )

        reference_features = train.feature_names()

        for name, partition in partitions.items():
            if (
                self.options.require_aligned_indices
                and (
                    not partition.X.index.equals(
                        partition.y.index
                    )
                    or not partition.X.index.equals(
                        partition.row_ids.index
                    )
                )
            ):
                issues.append(
                    ImputationContractIssue(
                        code="P4-CONTRACT-011",
                        severity="critical",
                        message=(
                            f"Partition {name.value!r} has "
                            "misaligned features, target, "
                            "or row IDs."
                        ),
                        partitions=[name],
                        blocks_execution=True,
                    )
                )

            if (
                self.options.require_unique_row_ids
                and not partition.row_ids.is_unique
            ):
                issues.append(
                    ImputationContractIssue(
                        code="P4-CONTRACT-012",
                        severity="critical",
                        message=(
                            f"Partition {name.value!r} "
                            "contains duplicate row IDs."
                        ),
                        partitions=[name],
                        blocks_execution=True,
                    )
                )

            if (
                name != PartitionName.TRAIN
                and self.options
                .require_identical_feature_order
                and partition.feature_names()
                != reference_features
            ):
                issues.append(
                    ImputationContractIssue(
                        code="P4-CONTRACT-013",
                        severity="critical",
                        message=(
                            f"Partition {name.value!r} has "
                            "a different feature order from "
                            "the training partition."
                        ),
                        partitions=[
                            PartitionName.TRAIN,
                            name,
                        ],
                        blocks_execution=True,
                    )
                )

            if (
                name != PartitionName.TRAIN
                and self.options
                .require_identical_feature_schema
            ):
                training_dtypes = {
                    column: str(
                        train.X[column].dtype
                    )
                    for column
                    in reference_features
                }

                candidate_dtypes = {
                    column: str(
                        partition.X[column].dtype
                    )
                    for column
                    in partition.feature_names()
                }

                if training_dtypes != candidate_dtypes:
                    issues.append(
                        ImputationContractIssue(
                            code="P4-CONTRACT-014",
                            severity="error",
                            message=(
                                f"Partition {name.value!r} "
                                "has feature dtypes that differ "
                                "from training."
                            ),
                            partitions=[
                                PartitionName.TRAIN,
                                name,
                            ],
                            blocks_execution=True,
                            details={
                                "training_dtypes": (
                                    training_dtypes
                                ),
                                "partition_dtypes": (
                                    candidate_dtypes
                                ),
                            },
                        )
                    )

        if self.options.require_disjoint_row_ids:
            partition_names = list(
                partitions
            )

            for position, left_name in enumerate(
                partition_names
            ):
                left_ids = set(
                    partitions[
                        left_name
                    ].row_ids.tolist()
                )

                for right_name in partition_names[
                    position + 1:
                ]:
                    overlap = left_ids.intersection(
                        partitions[
                            right_name
                        ].row_ids.tolist()
                    )

                    if not overlap:
                        continue

                    issues.append(
                        ImputationContractIssue(
                            code="P4-CONTRACT-015",
                            severity="critical",
                            message=(
                                f"Partitions "
                                f"{left_name.value!r} and "
                                f"{right_name.value!r} share "
                                "internal row IDs."
                            ),
                            partitions=[
                                left_name,
                                right_name,
                            ],
                            blocks_execution=True,
                            details={
                                "overlap_count": (
                                    len(overlap)
                                ),
                            },
                        )
                    )

    def _validate_protected_columns(
        self,
        *,
        train: DatasetPartition,
        phase2_state: Phase2State,
        recommendations: Phase3Recommendations,
        issues: list[ImputationContractIssue],
    ) -> None:
        """Ensure protected columns are not imputed."""

        protected_columns: set[str] = set()

        if self.options.prohibit_target_imputation:
            protected_columns.add(
                phase2_state.columns.target_column
            )

        if (
            self.options
            .prohibit_internal_row_id_imputation
        ):
            protected_columns.add(
                phase2_state
                .columns
                .internal_row_id_column
            )

        recommendation_columns = set(
            recommendations
            .imputation
            .recommendations
        )

        invalid_columns = (
            protected_columns
            .intersection(
                recommendation_columns
            )
        )

        if invalid_columns:
            issues.append(
                ImputationContractIssue(
                    code="P4-CONTRACT-016",
                    severity="critical",
                    message=(
                        "Protected columns appear in the "
                        "imputation recommendations."
                    ),
                    columns=sorted(
                        invalid_columns
                    ),
                    blocks_execution=True,
                )
            )

        missing_recommended_columns = (
            recommendation_columns
            .difference(
                train.X.columns
            )
        )

        if missing_recommended_columns:
            issues.append(
                ImputationContractIssue(
                    code="P4-CONTRACT-017",
                    severity="warning",
                    message=(
                        "Some imputation recommendations "
                        "reference unavailable features."
                    ),
                    columns=sorted(
                        missing_recommended_columns
                    ),
                    blocks_execution=False,
                )
            )

    def _validate_fingerprints(
        self,
        *,
        train: DatasetPartition,
        validation: DatasetPartition,
        phase3_state: Phase3State,
        issues: list[ImputationContractIssue],
    ) -> None:
        """Validate Phase 3 input fingerprints."""

        from mlprep.phases.phase3.output_governance.fingerprint import (
            Phase3FingerprintManager,
        )

        manager = Phase3FingerprintManager()

        if self.options.require_training_fingerprint_match:
            current_train_fingerprint = (
                manager.partition_fingerprint(
                    X=train.X,
                    y=train.y,
                    row_ids=train.row_ids,
                )
            )

            expected_train_fingerprint = (
                phase3_state
                .training_fingerprint_after
            )

            if (
                expected_train_fingerprint is None
                or current_train_fingerprint
                != expected_train_fingerprint
            ):
                issues.append(
                    ImputationContractIssue(
                        code="P4-CONTRACT-018",
                        severity="critical",
                        message=(
                            "The training partition fingerprint "
                            "does not match Phase 3."
                        ),
                        partitions=[
                            PartitionName.TRAIN
                        ],
                        blocks_execution=True,
                    )
                )

        if (
            self.options
            .require_validation_fingerprint_match
        ):
            current_validation_fingerprint = (
                manager.partition_fingerprint(
                    X=validation.X,
                    y=validation.y,
                    row_ids=validation.row_ids,
                )
            )

            expected_validation_fingerprint = (
                phase3_state
                .validation_fingerprint_after
            )

            if (
                expected_validation_fingerprint is None
                or current_validation_fingerprint
                != expected_validation_fingerprint
            ):
                issues.append(
                    ImputationContractIssue(
                        code="P4-CONTRACT-019",
                        severity="critical",
                        message=(
                            "The validation partition fingerprint "
                            "does not match Phase 3."
                        ),
                        partitions=[
                            PartitionName.VALIDATION
                        ],
                        blocks_execution=True,
                    )
                )

    def _copy_partition(
        self,
        partition: DatasetPartition,
    ) -> DatasetPartition:
        """Copy one input partition according to policy."""

        return partition.copy(
            deep=self.options.copy_input_partitions
        )

    @staticmethod
    def _resolve_test_partition(
        *,
        phase2_result: Phase2Result,
        test_partition: DatasetPartition | None,
    ) -> DatasetPartition:
        """Resolve the authorized test partition."""

        if test_partition is not None:
            if test_partition.name != PartitionName.TEST:
                raise ValueError(
                    "test_partition must be the test partition."
                )

            return test_partition

        if isinstance(
            phase2_result.test,
            DatasetPartition,
        ):
            return phase2_result.test

        raise PermissionError(
            "The locked test partition must be accessed "
            "through the authorized Phase 2 interface before "
            "running Phase 4."
        )

    @staticmethod
    def _resolve_phase3_state(
        phase3_result: Phase3Result,
    ) -> Phase3State:
        """Resolve Phase 3 state from the result."""

        if not phase3_result.phase_state:
            raise ValueError(
                "Phase 3 state is unavailable. Pass "
                "phase3_state explicitly or enable Phase 3 "
                "state storage."
            )

        return Phase3State.model_validate(
            phase3_result.phase_state
        )

    @staticmethod
    def _validate_input_types(
        *,
        phase2_result: Phase2Result,
        phase3_result: Phase3Result,
        phase1_state: Phase1State,
    ) -> None:
        """Validate required input objects."""

        if not isinstance(
            phase2_result,
            Phase2Result,
        ):
            raise TypeError(
                "phase2_result must be an instance "
                "of Phase2Result."
            )

        if not isinstance(
            phase3_result,
            Phase3Result,
        ):
            raise TypeError(
                "phase3_result must be an instance "
                "of Phase3Result."
            )

        if not isinstance(
            phase1_state,
            Phase1State,
        ):
            raise TypeError(
                "phase1_state must be an instance "
                "of Phase1State."
            )
