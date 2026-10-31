"""Input-contract validation for Phase 8."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd

from mlprep.phases.phase2.output_governance.metadata import (
    PartitionName,
)
from mlprep.phases.phase2.output_governance.result import (
    DatasetPartition,
)
from mlprep.phases.phase7.output_governance.result import (
    Phase7Result,
)
from mlprep.phases.phase7.output_governance.state import (
    Phase7State,
)
from mlprep.phases.phase8.utils import (
    ClassIdentityMapping,
    build_class_identity_mapping,
    normalize_class_value,
)

ContractSeverity = Literal[
    "info",
    "warning",
    "error",
    "critical",
]


@dataclass(slots=True)
class BalancingContractIssue:
    """One issue detected in the Phase 8 input contract."""

    code: str
    severity: ContractSeverity
    message: str

    columns: list[str] = field(
        default_factory=list,
    )

    partitions: list[PartitionName] = field(
        default_factory=list,
    )

    classes: list[str] = field(
        default_factory=list,
    )

    blocks_execution: bool = False

    details: dict[str, object] = field(
        default_factory=dict,
    )


@dataclass(slots=True)
class BalancingContractOptions:
    """Options controlling the Phase 8 input contract."""

    require_phase7_readiness: bool = True
    require_phase7_state: bool = True

    require_phase7_output_fingerprint: bool = True
    require_training_only_phase7_fit: bool = True
    require_phase7_selected_lineage: bool = True

    require_nonempty_training_partition: bool = True
    require_validation_partition: bool = True
    require_test_partition: bool = True

    require_identical_feature_schema: bool = True
    require_identical_feature_order: bool = True
    require_compatible_feature_dtypes: bool = True

    require_numeric_features: bool = True
    require_no_missing_feature_values: bool = True
    require_finite_feature_values: bool = True

    require_aligned_indices: bool = True

    require_unique_row_ids: bool = True
    require_nonmissing_row_ids: bool = True
    require_disjoint_row_ids: bool = True

    require_nonmissing_training_target: bool = True
    require_at_least_two_target_classes: bool = True

    minimum_training_class_count: int = 2

    require_unchanged_phase7_targets: bool = True
    require_unchanged_phase7_row_ids: bool = True
    require_unchanged_phase7_feature_values: bool = True
    require_consistent_phase7_schemas: bool = True

    require_protected_test_partition: bool = True

    copy_input_partitions: bool = True

    maximum_reported_columns: int = 100
    maximum_reported_row_ids: int = 20
    maximum_reported_classes: int = 50

    def __post_init__(self) -> None:
        """Validate contract options."""

        boolean_options = {
            "require_phase7_readiness": (
                self.require_phase7_readiness
            ),
            "require_phase7_state": (
                self.require_phase7_state
            ),
            "require_phase7_output_fingerprint": (
                self.require_phase7_output_fingerprint
            ),
            "require_training_only_phase7_fit": (
                self.require_training_only_phase7_fit
            ),
            "require_phase7_selected_lineage": (
                self.require_phase7_selected_lineage
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
            "require_identical_feature_schema": (
                self.require_identical_feature_schema
            ),
            "require_identical_feature_order": (
                self.require_identical_feature_order
            ),
            "require_compatible_feature_dtypes": (
                self.require_compatible_feature_dtypes
            ),
            "require_numeric_features": (
                self.require_numeric_features
            ),
            "require_no_missing_feature_values": (
                self.require_no_missing_feature_values
            ),
            "require_finite_feature_values": (
                self.require_finite_feature_values
            ),
            "require_aligned_indices": (
                self.require_aligned_indices
            ),
            "require_unique_row_ids": (
                self.require_unique_row_ids
            ),
            "require_nonmissing_row_ids": (
                self.require_nonmissing_row_ids
            ),
            "require_disjoint_row_ids": (
                self.require_disjoint_row_ids
            ),
            "require_nonmissing_training_target": (
                self.require_nonmissing_training_target
            ),
            "require_at_least_two_target_classes": (
                self.require_at_least_two_target_classes
            ),
            "require_unchanged_phase7_targets": (
                self.require_unchanged_phase7_targets
            ),
            "require_unchanged_phase7_row_ids": (
                self.require_unchanged_phase7_row_ids
            ),
            "require_unchanged_phase7_feature_values": (
                self.require_unchanged_phase7_feature_values
            ),
            "require_consistent_phase7_schemas": (
                self.require_consistent_phase7_schemas
            ),
            "require_protected_test_partition": (
                self.require_protected_test_partition
            ),
            "copy_input_partitions": (
                self.copy_input_partitions
            ),
        }

        for name, value in boolean_options.items():
            if not isinstance(
                value,
                bool,
            ):
                raise TypeError(
                    f"{name} must be a Boolean."
                )

        if self.minimum_training_class_count < 1:
            raise ValueError(
                "minimum_training_class_count must be positive."
            )

        if self.maximum_reported_columns < 1:
            raise ValueError(
                "maximum_reported_columns must be positive."
            )

        if self.maximum_reported_row_ids < 1:
            raise ValueError(
                "maximum_reported_row_ids must be positive."
            )

        if self.maximum_reported_classes < 1:
            raise ValueError(
                "maximum_reported_classes must be positive."
            )


class BalancingContract:
    """Resolved input contract used by Phase 8."""

    def __init__(
        self,
        *,
        train: DatasetPartition,
        validation: DatasetPartition,
        test: DatasetPartition,
        phase7_state: Phase7State,
        training_class_counts: dict[str, int],
        class_identity_mapping: ClassIdentityMapping,
        target_dtype: str,
        issues: list[BalancingContractIssue],
        options: BalancingContractOptions,
    ) -> None:
        """Initialize the resolved balancing contract."""

        self.train = train
        self.validation = validation
        self.test = test

        self.phase7_state = phase7_state

        self.training_class_counts = dict(
            training_class_counts
        )
        self.class_identity_mapping = (
            class_identity_mapping.model_copy(
                deep=True
            )
        )

        self.target_dtype = target_dtype

        self.issues = list(
            issues
        )

        self.options = options

    @property
    def feature_columns(
        self,
    ) -> list[str]:
        """Return ordered selected feature columns."""

        return [
            str(column)
            for column in self.train.X.columns
        ]

    @property
    def target_classes(
        self,
    ) -> list[str]:
        """Return normalized training class labels."""

        return list(
            self.training_class_counts
        )

    @property
    def class_count(
        self,
    ) -> int:
        """Return the number of training classes."""

        return len(
            self.training_class_counts
        )

    @property
    def smallest_class_count(
        self,
    ) -> int:
        """Return the smallest training class count."""

        if not self.training_class_counts:
            return 0

        return min(
            self.training_class_counts.values()
        )

    @property
    def largest_class_count(
        self,
    ) -> int:
        """Return the largest training class count."""

        if not self.training_class_counts:
            return 0

        return max(
            self.training_class_counts.values()
        )

    @property
    def has_blocking_issues(
        self,
    ) -> bool:
        """Return whether blocking issues exist."""

        return any(
            issue.blocks_execution
            or issue.severity == "critical"
            for issue in self.issues
        )

    def assert_ready(
        self,
    ) -> None:
        """Raise when contract issues block Phase 8."""

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
            "The Phase 8 balancing contract contains "
            f"blocking issues: {codes}"
        )

    def partition(
        self,
        name: PartitionName,
    ) -> DatasetPartition:
        """Return one resolved partition."""

        if name == PartitionName.TRAIN:
            return self.train

        if name == PartitionName.VALIDATION:
            return self.validation

        if name == PartitionName.TEST:
            return self.test

        raise KeyError(
            f"Partition {name.value!r} is unavailable."
        )


class BalancingContractResolver:
    """Resolve and validate the Phase 8 input contract."""

    component_name = "balancing_contract"

    def __init__(
        self,
        options: BalancingContractOptions | None = None,
    ) -> None:
        """Initialize the resolver."""

        self.options = (
            options
            or BalancingContractOptions()
        )

    def resolve(
        self,
        *,
        phase7_result: Phase7Result,
        phase7_state: Phase7State | None = None,
        authorized_test_partition: DatasetPartition | None = None,
    ) -> BalancingContract:
        """Resolve the complete Phase 8 input contract."""

        if not isinstance(
            phase7_result,
            Phase7Result,
        ):
            raise TypeError(
                "phase7_result must be a Phase7Result."
            )

        resolved_state = (
            phase7_state
            or self._resolve_phase7_state(
                phase7_result
            )
        )

        issues: list[
            BalancingContractIssue
        ] = []

        self._validate_phase7_readiness(
            result=phase7_result,
            issues=issues,
        )

        self._validate_phase7_state(
            state=resolved_state,
            issues=issues,
        )

        test_partition = (
            authorized_test_partition
            if authorized_test_partition is not None
            else phase7_result.test
        )

        if not isinstance(
            test_partition,
            DatasetPartition,
        ):
            raise PermissionError(
                "Phase 8 requires an authorized selected "
                "test partition from Phase 7."
            )

        train = self._copy_partition(
            phase7_result.train
        )

        validation = self._copy_partition(
            phase7_result.validation
        )

        test = self._copy_partition(
            test_partition
        )

        self._validate_partition_names(
            train=train,
            validation=validation,
            test=test,
            issues=issues,
        )

        self._validate_partition_sizes(
            train=train,
            validation=validation,
            test=test,
            issues=issues,
        )

        self._validate_indices_and_row_ids(
            train=train,
            validation=validation,
            test=test,
            issues=issues,
        )

        self._validate_feature_schemas(
            train=train,
            validation=validation,
            test=test,
            issues=issues,
        )

        self._validate_feature_values(
            train=train,
            validation=validation,
            test=test,
            issues=issues,
        )

        class_identity_mapping = build_class_identity_mapping(
            train.y,
            reject_missing=False,
        )
        training_class_counts = self._training_class_counts(
            target=train.y,
            class_identity_mapping=class_identity_mapping,
        )

        self._validate_training_target(
            train=train,
            class_counts=training_class_counts,
            issues=issues,
        )

        return BalancingContract(
            train=train,
            validation=validation,
            test=test,
            phase7_state=resolved_state,
            training_class_counts=(
                training_class_counts
            ),
            class_identity_mapping=(
                class_identity_mapping
            ),
            target_dtype=str(
                train.y.dtype
            ),
            issues=issues,
            options=self.options,
        )

    def _validate_phase7_readiness(
        self,
        *,
        result: Phase7Result,
        issues: list[BalancingContractIssue],
    ) -> None:
        """Validate Phase 7 readiness."""

        if not self.options.require_phase7_readiness:
            return

        readiness = result.readiness

        if readiness is None:
            issues.append(
                BalancingContractIssue(
                    code="P8-CONTRACT-001",
                    severity="critical",
                    message=(
                        "Phase 7 readiness was not evaluated."
                    ),
                    blocks_execution=True,
                )
            )

            return

        if not readiness.ready:
            issues.append(
                BalancingContractIssue(
                    code="P8-CONTRACT-002",
                    severity="critical",
                    message=(
                        "Phase 7 did not approve progression "
                        "to Phase 8."
                    ),
                    blocks_execution=True,
                    details={
                        "phase7_status": (
                            readiness.status.value
                        ),
                        "phase7_reasons": list(
                            readiness.reasons
                        ),
                    },
                )
            )

    def _validate_phase7_state(
        self,
        *,
        state: Phase7State,
        issues: list[BalancingContractIssue],
    ) -> None:
        """Validate required Phase 7 governance state."""

        checks = [
            (
                self.options
                .require_phase7_output_fingerprint
                and not state.output_fingerprint,
                "P8-CONTRACT-003",
                (
                    "The Phase 7 output fingerprint "
                    "is unavailable."
                ),
            ),
            (
                self.options
                .require_training_only_phase7_fit
                and not state.fitted_on_training_only,
                "P8-CONTRACT-004",
                (
                    "Phase 7 selection state was not fitted "
                    "exclusively on training data."
                ),
            ),
            (
                self.options
                .require_phase7_selected_lineage
                and state.selected_lineage is None,
                "P8-CONTRACT-005",
                (
                    "Phase 7 selected feature lineage "
                    "is unavailable."
                ),
            ),
            (
                self.options
                .require_unchanged_phase7_targets
                and not state.targets_unchanged,
                "P8-CONTRACT-006",
                (
                    "Phase 7 did not verify unchanged targets."
                ),
            ),
            (
                self.options
                .require_unchanged_phase7_row_ids
                and not state.row_ids_unchanged,
                "P8-CONTRACT-007",
                (
                    "Phase 7 did not verify unchanged row IDs."
                ),
            ),
            (
                self.options
                .require_unchanged_phase7_feature_values
                and not state.selected_values_unchanged,
                "P8-CONTRACT-008",
                (
                    "Phase 7 did not verify unchanged selected "
                    "feature values."
                ),
            ),
            (
                self.options
                .require_consistent_phase7_schemas
                and not state.schemas_consistent,
                "P8-CONTRACT-009",
                (
                    "Phase 7 selected schemas are inconsistent."
                ),
            ),
            (
                self.options
                .require_protected_test_partition
                and not state
                .test_partition_remained_protected,
                "P8-CONTRACT-010",
                (
                    "The test partition did not remain "
                    "protected during Phase 7."
                ),
            ),
        ]

        for condition, code, message in checks:
            if condition:
                issues.append(
                    BalancingContractIssue(
                        code=code,
                        severity="critical",
                        message=message,
                        blocks_execution=True,
                    )
                )

    def _validate_partition_names(
        self,
        *,
        train: DatasetPartition,
        validation: DatasetPartition,
        test: DatasetPartition,
        issues: list[BalancingContractIssue],
    ) -> None:
        """Validate partition identity metadata."""

        expected = {
            PartitionName.TRAIN: train,
            PartitionName.VALIDATION: validation,
            PartitionName.TEST: test,
        }

        for expected_name, partition in expected.items():
            if partition.name == expected_name:
                continue

            issues.append(
                BalancingContractIssue(
                    code="P8-CONTRACT-011",
                    severity="critical",
                    message=(
                        f"Expected partition "
                        f"{expected_name.value!r}, received "
                        f"{partition.name.value!r}."
                    ),
                    partitions=[
                        expected_name
                    ],
                    blocks_execution=True,
                )
            )

    def _validate_partition_sizes(
        self,
        *,
        train: DatasetPartition,
        validation: DatasetPartition,
        test: DatasetPartition,
        issues: list[BalancingContractIssue],
    ) -> None:
        """Validate required partition sizes."""

        checks = [
            (
                self.options
                .require_nonempty_training_partition,
                train,
                "P8-CONTRACT-012",
                "The training partition is empty.",
            ),
            (
                self.options
                .require_validation_partition,
                validation,
                "P8-CONTRACT-013",
                "The validation partition is empty.",
            ),
            (
                self.options.require_test_partition,
                test,
                "P8-CONTRACT-014",
                "The test partition is empty.",
            ),
        ]

        for required, partition, code, message in checks:
            if (
                required
                and partition.row_count() == 0
            ):
                issues.append(
                    BalancingContractIssue(
                        code=code,
                        severity="critical",
                        message=message,
                        partitions=[
                            partition.name
                        ],
                        blocks_execution=True,
                    )
                )

    def _validate_indices_and_row_ids(
        self,
        *,
        train: DatasetPartition,
        validation: DatasetPartition,
        test: DatasetPartition,
        issues: list[BalancingContractIssue],
    ) -> None:
        """Validate indices and row IDs."""

        partitions = {
            PartitionName.TRAIN: train,
            PartitionName.VALIDATION: validation,
            PartitionName.TEST: test,
        }

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
                    BalancingContractIssue(
                        code="P8-CONTRACT-015",
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
                    BalancingContractIssue(
                        code="P8-CONTRACT-016",
                        severity="critical",
                        message=(
                            f"Partition {name.value!r} contains "
                            "duplicate row IDs."
                        ),
                        partitions=[name],
                        blocks_execution=True,
                    )
                )

            if (
                self.options.require_nonmissing_row_ids
                and partition.row_ids.isna().any()
            ):
                issues.append(
                    BalancingContractIssue(
                        code="P8-CONTRACT-017",
                        severity="critical",
                        message=(
                            f"Partition {name.value!r} contains "
                            "missing row IDs."
                        ),
                        partitions=[name],
                        blocks_execution=True,
                    )
                )

        if not self.options.require_disjoint_row_ids:
            return

        partition_items = list(
            partitions.items()
        )

        for position, (
            left_name,
            left_partition,
        ) in enumerate(
            partition_items
        ):
            left_ids = set(
                left_partition
                .row_ids
                .dropna()
                .tolist()
            )

            for (
                right_name,
                right_partition,
            ) in partition_items[
                position + 1:
            ]:
                right_ids = set(
                    right_partition
                    .row_ids
                    .dropna()
                    .tolist()
                )

                overlap = left_ids.intersection(
                    right_ids
                )

                if not overlap:
                    continue

                issues.append(
                    BalancingContractIssue(
                        code="P8-CONTRACT-018",
                        severity="critical",
                        message=(
                            f"Partitions {left_name.value!r} "
                            f"and {right_name.value!r} share "
                            "row IDs."
                        ),
                        partitions=[
                            left_name,
                            right_name,
                        ],
                        blocks_execution=True,
                        details={
                            "overlap_count": len(
                                overlap
                            ),
                            "example_row_ids": [
                                str(value)
                                for value in list(
                                    overlap
                                )[
                                    : self.options
                                    .maximum_reported_row_ids
                                ]
                            ],
                        },
                    )
                )

    def _validate_feature_schemas(
        self,
        *,
        train: DatasetPartition,
        validation: DatasetPartition,
        test: DatasetPartition,
        issues: list[BalancingContractIssue],
    ) -> None:
        """Validate selected feature schemas."""

        training_columns = (
            train.feature_names()
        )

        if not training_columns:
            issues.append(
                BalancingContractIssue(
                    code="P8-CONTRACT-019",
                    severity="critical",
                    message=(
                        "The selected training feature schema "
                        "is empty."
                    ),
                    partitions=[
                        PartitionName.TRAIN
                    ],
                    blocks_execution=True,
                )
            )

            return

        duplicate_training_columns = (
            self._duplicates(
                training_columns
            )
        )

        if duplicate_training_columns:
            issues.append(
                BalancingContractIssue(
                    code="P8-CONTRACT-020",
                    severity="critical",
                    message=(
                        "The training partition contains "
                        "duplicate feature names."
                    ),
                    columns=(
                        duplicate_training_columns
                    ),
                    partitions=[
                        PartitionName.TRAIN
                    ],
                    blocks_execution=True,
                )
            )

        training_set = set(
            training_columns
        )

        for name, partition in {
            PartitionName.VALIDATION: validation,
            PartitionName.TEST: test,
        }.items():
            candidate_columns = (
                partition.feature_names()
            )

            candidate_set = set(
                candidate_columns
            )

            missing_columns = sorted(
                training_set.difference(
                    candidate_set
                )
            )

            extra_columns = sorted(
                candidate_set.difference(
                    training_set
                )
            )

            if (
                self.options
                .require_identical_feature_schema
                and (
                    missing_columns
                    or extra_columns
                )
            ):
                issues.append(
                    BalancingContractIssue(
                        code="P8-CONTRACT-021",
                        severity="critical",
                        message=(
                            f"Partition {name.value!r} has "
                            "a different selected schema."
                        ),
                        columns=(
                            missing_columns
                            + extra_columns
                        )[
                            : self.options
                            .maximum_reported_columns
                        ],
                        partitions=[
                            PartitionName.TRAIN,
                            name,
                        ],
                        blocks_execution=True,
                        details={
                            "missing_columns": (
                                missing_columns
                            ),
                            "extra_columns": (
                                extra_columns
                            ),
                        },
                    )
                )

            if (
                self.options
                .require_identical_feature_order
                and candidate_columns
                != training_columns
            ):
                issues.append(
                    BalancingContractIssue(
                        code="P8-CONTRACT-022",
                        severity="critical",
                        message=(
                            f"Partition {name.value!r} has "
                            "a different selected feature order."
                        ),
                        partitions=[
                            PartitionName.TRAIN,
                            name,
                        ],
                        blocks_execution=True,
                    )
                )

            if not (
                self.options
                .require_compatible_feature_dtypes
            ):
                continue

            dtype_mismatches: dict[
                str,
                dict[str, str],
            ] = {}

            for column in training_columns:
                if column not in candidate_set:
                    continue

                training_dtype = str(
                    train.X[
                        column
                    ].dtype
                )

                candidate_dtype = str(
                    partition.X[
                        column
                    ].dtype
                )

                if training_dtype != candidate_dtype:
                    dtype_mismatches[
                        column
                    ] = {
                        "training": training_dtype,
                        name.value: candidate_dtype,
                    }

            if dtype_mismatches:
                issues.append(
                    BalancingContractIssue(
                        code="P8-CONTRACT-023",
                        severity="error",
                        message=(
                            f"Partition {name.value!r} has "
                            "incompatible selected-feature "
                            "dtypes."
                        ),
                        columns=list(
                            dtype_mismatches
                        )[
                            : self.options
                            .maximum_reported_columns
                        ],
                        partitions=[
                            PartitionName.TRAIN,
                            name,
                        ],
                        blocks_execution=True,
                        details={
                            "dtype_mismatches": (
                                dtype_mismatches
                            ),
                        },
                    )
                )

    def _validate_feature_values(
        self,
        *,
        train: DatasetPartition,
        validation: DatasetPartition,
        test: DatasetPartition,
        issues: list[BalancingContractIssue],
    ) -> None:
        """Validate numeric, missing, and finite features."""

        partitions = {
            PartitionName.TRAIN: train,
            PartitionName.VALIDATION: validation,
            PartitionName.TEST: test,
        }

        for name, partition in partitions.items():
            nonnumeric_columns: list[str] = []
            missing_columns: list[str] = []
            nonfinite_columns: list[str] = []

            for column in partition.X.columns:
                column_name = str(
                    column
                )

                series = partition.X[
                    column
                ]

                if not pd.api.types.is_numeric_dtype(
                    series.dtype
                ):
                    nonnumeric_columns.append(
                        column_name
                    )

                    continue

                numeric = pd.to_numeric(
                    series,
                    errors="coerce",
                )

                if bool(
                    numeric.isna().any()
                ):
                    missing_columns.append(
                        column_name
                    )

                nonfinite_mask = (
                    numeric.notna()
                    & ~np.isfinite(
                        numeric.astype(
                            float
                        )
                    )
                )

                if bool(
                    nonfinite_mask.any()
                ):
                    nonfinite_columns.append(
                        column_name
                    )

            if (
                self.options.require_numeric_features
                and nonnumeric_columns
            ):
                issues.append(
                    BalancingContractIssue(
                        code="P8-CONTRACT-024",
                        severity="critical",
                        message=(
                            f"Partition {name.value!r} contains "
                            "nonnumeric selected features."
                        ),
                        columns=(
                            nonnumeric_columns[
                                : self.options
                                .maximum_reported_columns
                            ]
                        ),
                        partitions=[name],
                        blocks_execution=True,
                    )
                )

            if (
                self.options
                .require_no_missing_feature_values
                and missing_columns
            ):
                issues.append(
                    BalancingContractIssue(
                        code="P8-CONTRACT-025",
                        severity="critical",
                        message=(
                            f"Partition {name.value!r} contains "
                            "missing selected-feature values."
                        ),
                        columns=(
                            missing_columns[
                                : self.options
                                .maximum_reported_columns
                            ]
                        ),
                        partitions=[name],
                        blocks_execution=True,
                    )
                )

            if (
                self.options
                .require_finite_feature_values
                and nonfinite_columns
            ):
                issues.append(
                    BalancingContractIssue(
                        code="P8-CONTRACT-026",
                        severity="critical",
                        message=(
                            f"Partition {name.value!r} contains "
                            "nonfinite selected-feature values."
                        ),
                        columns=(
                            nonfinite_columns[
                                : self.options
                                .maximum_reported_columns
                            ]
                        ),
                        partitions=[name],
                        blocks_execution=True,
                    )
                )

    def _validate_training_target(
        self,
        *,
        train: DatasetPartition,
        class_counts: dict[str, int],
        issues: list[BalancingContractIssue],
    ) -> None:
        """Validate the training target for balancing."""

        missing_count = int(
            train.y.isna().sum()
        )

        if (
            self.options.require_nonmissing_training_target
            and missing_count > 0
        ):
            issues.append(
                BalancingContractIssue(
                    code="P8-CONTRACT-027",
                    severity="critical",
                    message=(
                        "The training target contains "
                        "missing values."
                    ),
                    partitions=[
                        PartitionName.TRAIN
                    ],
                    blocks_execution=True,
                    details={
                        "missing_target_count": (
                            missing_count
                        ),
                    },
                )
            )

        if (
            self.options.require_at_least_two_target_classes
            and len(
                class_counts
            )
            < 2
        ):
            issues.append(
                BalancingContractIssue(
                    code="P8-CONTRACT-028",
                    severity="critical",
                    message=(
                        "The training target must contain at "
                        "least two distinct classes."
                    ),
                    partitions=[
                        PartitionName.TRAIN
                    ],
                    classes=list(
                        class_counts
                    ),
                    blocks_execution=True,
                )
            )

        undersized_classes = {
            class_label: count
            for class_label, count
            in class_counts.items()
            if count
            < self.options
            .minimum_training_class_count
        }

        if undersized_classes:
            issues.append(
                BalancingContractIssue(
                    code="P8-CONTRACT-029",
                    severity="error",
                    message=(
                        "One or more training classes contain "
                        "fewer rows than the configured minimum."
                    ),
                    partitions=[
                        PartitionName.TRAIN
                    ],
                    classes=list(
                        undersized_classes
                    )[
                        : self.options
                        .maximum_reported_classes
                    ],
                    blocks_execution=True,
                    details={
                        "minimum_training_class_count": (
                            self.options
                            .minimum_training_class_count
                        ),
                        "undersized_classes": (
                            undersized_classes
                        ),
                    },
                )
            )

    def _copy_partition(
        self,
        partition: DatasetPartition,
    ) -> DatasetPartition:
        """Copy one partition according to policy."""

        return partition.copy(
            deep=self.options.copy_input_partitions
        )

    def _resolve_phase7_state(
        self,
        result: Phase7Result,
    ) -> Phase7State:
        """Resolve Phase 7 state."""

        if not self.options.require_phase7_state:
            return Phase7State()

        direct_state = getattr(
            result,
            "state",
            None,
        )

        if isinstance(
            direct_state,
            Phase7State,
        ):
            return direct_state

        if result.phase_state:
            return Phase7State.model_validate(
                result.phase_state
            )

        raise ValueError(
            "Phase 7 state is unavailable. Pass phase7_state "
            "explicitly or store it in the Phase 7 result."
        )

    @staticmethod
    def _training_class_counts(
        *,
        target: pd.Series,
        class_identity_mapping: ClassIdentityMapping,
    ) -> dict[str, int]:
        """Calculate counts without pandas equality-based class merging."""

        counts: dict[str, int] = {}
        for value in target:
            if normalize_class_value(value) is None:
                continue
            key = class_identity_mapping.key_for(value)
            counts[key] = counts.get(key, 0) + 1
        return counts

    @staticmethod
    def _duplicates(
        values: list[str],
    ) -> list[object]:
        """Return duplicate names while preserving order."""

        seen: set[str] = set()
        duplicates: list[str] = []

        for value in values:
            if (
                value in seen
                and value not in duplicates
            ):
                duplicates.append(
                    value
                )

            seen.add(
                value
            )

        return duplicates
