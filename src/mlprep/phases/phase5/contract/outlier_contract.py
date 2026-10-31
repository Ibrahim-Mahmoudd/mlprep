"""Input-contract validation for Phase 5."""

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
from mlprep.phases.phase4.output_governance.result import (
    Phase4Result,
)
from mlprep.phases.phase4.output_governance.state import (
    Phase4State,
)


ContractSeverity = Literal[
    "info",
    "warning",
    "error",
    "critical",
]


@dataclass(slots=True)
class OutlierContractIssue:
    """One issue detected in the Phase 5 input contract."""

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
class OutlierContractOptions:
    """Options controlling the Phase 5 input contract."""

    require_phase4_readiness: bool = True
    require_phase4_state: bool = True

    require_nonempty_training_partition: bool = True
    require_validation_partition: bool = True
    require_test_partition: bool = True

    require_identical_feature_schema: bool = True
    require_identical_feature_order: bool = True
    require_compatible_dtypes: bool = True

    require_aligned_indices: bool = True
    require_unique_row_ids: bool = True
    require_nonmissing_row_ids: bool = True
    require_disjoint_row_ids: bool = True

    require_no_missing_numeric_values: bool = True
    require_finite_numeric_values: bool = True

    require_phase4_output_fingerprint: bool = True

    prohibit_target_treatment: bool = True
    prohibit_internal_row_id_treatment: bool = True

    copy_input_partitions: bool = True

    maximum_reported_columns: int = 100
    maximum_reported_row_ids: int = 20

    def __post_init__(self) -> None:
        """Validate contract options."""

        boolean_options = {
            "require_phase4_readiness": (
                self.require_phase4_readiness
            ),
            "require_phase4_state": (
                self.require_phase4_state
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
            "require_compatible_dtypes": (
                self.require_compatible_dtypes
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
            "require_no_missing_numeric_values": (
                self.require_no_missing_numeric_values
            ),
            "require_finite_numeric_values": (
                self.require_finite_numeric_values
            ),
            "require_phase4_output_fingerprint": (
                self.require_phase4_output_fingerprint
            ),
            "prohibit_target_treatment": (
                self.prohibit_target_treatment
            ),
            "prohibit_internal_row_id_treatment": (
                self.prohibit_internal_row_id_treatment
            ),
            "copy_input_partitions": (
                self.copy_input_partitions
            ),
        }

        for name, value in boolean_options.items():
            if not isinstance(value, bool):
                raise TypeError(
                    f"{name} must be a boolean."
                )

        if self.maximum_reported_columns < 1:
            raise ValueError(
                "maximum_reported_columns must be positive."
            )

        if self.maximum_reported_row_ids < 1:
            raise ValueError(
                "maximum_reported_row_ids must be positive."
            )


class OutlierContract:
    """Resolved input contract used by Phase 5."""

    def __init__(
        self,
        *,
        train: DatasetPartition,
        validation: DatasetPartition,
        test: DatasetPartition,
        phase4_state: Phase4State,
        issues: list[OutlierContractIssue],
        options: OutlierContractOptions,
    ) -> None:
        """Initialize the resolved Phase 5 contract."""

        self.train = train
        self.validation = validation
        self.test = test

        self.phase4_state = phase4_state

        self.issues = issues
        self.options = options

    @property
    def feature_columns(
        self,
    ) -> list[str]:
        """Return ordered model feature columns."""

        return [
            str(column)
            for column in self.train.X.columns
        ]

    @property
    def numeric_columns(
        self,
    ) -> list[str]:
        """Return numeric features eligible for outlier analysis."""

        return [
            str(column)
            for column in self.train.X.columns
            if (
                pd.api.types.is_numeric_dtype(
                    self.train.X[column].dtype
                )
                and not pd.api.types.is_bool_dtype(
                    self.train.X[column].dtype
                )
            )
        ]

    @property
    def has_blocking_issues(
        self,
    ) -> bool:
        """Return whether blocking contract issues exist."""

        return any(
            issue.blocks_execution
            or issue.severity == "critical"
            for issue in self.issues
        )

    def assert_ready(
        self,
    ) -> None:
        """Raise when contract issues block Phase 5."""

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
            "The Phase 5 outlier contract contains "
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


class OutlierContractResolver:
    """Resolve and validate the Phase 5 input contract."""

    component_name = "outlier_contract"

    def __init__(
        self,
        options: OutlierContractOptions | None = None,
    ) -> None:
        """Initialize the contract resolver."""

        self.options = (
            options or OutlierContractOptions()
        )

    def resolve(
        self,
        *,
        phase4_result: Phase4Result,
        phase4_state: Phase4State | None = None,
        authorized_test_partition: (
            DatasetPartition | None
        ) = None,
    ) -> OutlierContract:
        """Resolve the complete Phase 5 contract."""

        if not isinstance(
            phase4_result,
            Phase4Result,
        ):
            raise TypeError(
                "phase4_result must be a Phase4Result."
            )

        resolved_state = (
            phase4_state
            or self._resolve_phase4_state(
                phase4_result
            )
        )

        issues: list[
            OutlierContractIssue
        ] = []

        self._validate_phase4_readiness(
            phase4_result=phase4_result,
            issues=issues,
        )

        self._validate_phase4_state(
            state=resolved_state,
            issues=issues,
        )

        test_partition = (
            authorized_test_partition
            if authorized_test_partition is not None
            else phase4_result.test
        )

        if not isinstance(
            test_partition,
            DatasetPartition,
        ):
            issues.append(
                OutlierContractIssue(
                    code="P5-CONTRACT-001",
                    severity="critical",
                    message=(
                        "An authorized transformed test "
                        "partition is required."
                    ),
                    partitions=[
                        PartitionName.TEST
                    ],
                    blocks_execution=True,
                )
            )

            raise PermissionError(
                "Phase 5 requires an authorized transformed "
                "test partition from Phase 4."
            )

        train = self._copy_partition(
            phase4_result.train
        )

        validation = self._copy_partition(
            phase4_result.validation
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

        self._validate_schemas(
            train=train,
            validation=validation,
            test=test,
            issues=issues,
        )

        self._validate_numeric_values(
            train=train,
            validation=validation,
            test=test,
            issues=issues,
        )

        return OutlierContract(
            train=train,
            validation=validation,
            test=test,
            phase4_state=resolved_state,
            issues=issues,
            options=self.options,
        )

    def _validate_phase4_readiness(
        self,
        *,
        phase4_result: Phase4Result,
        issues: list[OutlierContractIssue],
    ) -> None:
        """Validate Phase 4 readiness."""

        if not self.options.require_phase4_readiness:
            return

        readiness = phase4_result.readiness

        if readiness is None:
            issues.append(
                OutlierContractIssue(
                    code="P5-CONTRACT-002",
                    severity="critical",
                    message=(
                        "Phase 4 readiness was not evaluated."
                    ),
                    blocks_execution=True,
                )
            )

            return

        if not readiness.ready:
            issues.append(
                OutlierContractIssue(
                    code="P5-CONTRACT-003",
                    severity="critical",
                    message=(
                        "Phase 4 did not approve progression "
                        "to Phase 5."
                    ),
                    blocks_execution=True,
                    details={
                        "phase4_status": (
                            readiness.status.value
                        ),
                        "phase4_reasons": list(
                            readiness.reasons
                        ),
                    },
                )
            )

    def _validate_phase4_state(
        self,
        *,
        state: Phase4State,
        issues: list[OutlierContractIssue],
    ) -> None:
        """Validate required Phase 4 governance state."""

        if (
            self.options.require_phase4_output_fingerprint
            and not state.output_fingerprint
        ):
            issues.append(
                OutlierContractIssue(
                    code="P5-CONTRACT-004",
                    severity="critical",
                    message=(
                        "The Phase 4 output fingerprint "
                        "is unavailable."
                    ),
                    blocks_execution=True,
                )
            )

        if not state.fitted_on_training_only:
            issues.append(
                OutlierContractIssue(
                    code="P5-CONTRACT-005",
                    severity="critical",
                    message=(
                        "Phase 4 imputation was not fitted "
                        "exclusively on training data."
                    ),
                    blocks_execution=True,
                )
            )

        if not state.targets_unchanged:
            issues.append(
                OutlierContractIssue(
                    code="P5-CONTRACT-006",
                    severity="critical",
                    message=(
                        "Phase 4 did not verify unchanged targets."
                    ),
                    blocks_execution=True,
                )
            )

        if not state.row_ids_unchanged:
            issues.append(
                OutlierContractIssue(
                    code="P5-CONTRACT-007",
                    severity="critical",
                    message=(
                        "Phase 4 did not verify unchanged row IDs."
                    ),
                    blocks_execution=True,
                )
            )

        if not state.schemas_consistent:
            issues.append(
                OutlierContractIssue(
                    code="P5-CONTRACT-008",
                    severity="critical",
                    message=(
                        "Phase 4 output schemas are inconsistent."
                    ),
                    blocks_execution=True,
                )
            )

    def _validate_partition_names(
        self,
        *,
        train: DatasetPartition,
        validation: DatasetPartition,
        test: DatasetPartition,
        issues: list[OutlierContractIssue],
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
                OutlierContractIssue(
                    code="P5-CONTRACT-009",
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
        issues: list[OutlierContractIssue],
    ) -> None:
        """Validate required partition sizes."""

        if (
            self.options.require_nonempty_training_partition
            and train.row_count() == 0
        ):
            issues.append(
                OutlierContractIssue(
                    code="P5-CONTRACT-010",
                    severity="critical",
                    message="The training partition is empty.",
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
                OutlierContractIssue(
                    code="P5-CONTRACT-011",
                    severity="critical",
                    message="The validation partition is empty.",
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
                OutlierContractIssue(
                    code="P5-CONTRACT-012",
                    severity="critical",
                    message="The test partition is empty.",
                    partitions=[
                        PartitionName.TEST
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
        issues: list[OutlierContractIssue],
    ) -> None:
        """Validate indices and partition row IDs."""

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
                    OutlierContractIssue(
                        code="P5-CONTRACT-013",
                        severity="critical",
                        message=(
                            f"Partition {name.value!r} has "
                            "misaligned features, targets, "
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
                    OutlierContractIssue(
                        code="P5-CONTRACT-014",
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
                    OutlierContractIssue(
                        code="P5-CONTRACT-015",
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
        ) in enumerate(partition_items):
            left_ids = set(
                left_partition.row_ids.tolist()
            )

            for (
                right_name,
                right_partition,
            ) in partition_items[
                position + 1:
            ]:
                overlap = left_ids.intersection(
                    right_partition
                    .row_ids
                    .tolist()
                )

                if not overlap:
                    continue

                issues.append(
                    OutlierContractIssue(
                        code="P5-CONTRACT-016",
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

    def _validate_schemas(
        self,
        *,
        train: DatasetPartition,
        validation: DatasetPartition,
        test: DatasetPartition,
        issues: list[OutlierContractIssue],
    ) -> None:
        """Validate feature schemas across partitions."""

        training_columns = (
            train.feature_names()
        )

        training_column_set = set(
            training_columns
        )

        for name, partition in {
            PartitionName.VALIDATION: validation,
            PartitionName.TEST: test,
        }.items():
            candidate_columns = (
                partition.feature_names()
            )

            candidate_column_set = set(
                candidate_columns
            )

            missing_columns = sorted(
                training_column_set.difference(
                    candidate_column_set
                )
            )

            extra_columns = sorted(
                candidate_column_set.difference(
                    training_column_set
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
                    OutlierContractIssue(
                        code="P5-CONTRACT-017",
                        severity="critical",
                        message=(
                            f"Partition {name.value!r} has "
                            "a different feature schema."
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
                    OutlierContractIssue(
                        code="P5-CONTRACT-018",
                        severity="critical",
                        message=(
                            f"Partition {name.value!r} "
                            "has a different feature order."
                        ),
                        partitions=[
                            PartitionName.TRAIN,
                            name,
                        ],
                        blocks_execution=True,
                    )
                )

            if not self.options.require_compatible_dtypes:
                continue

            common_columns = [
                column
                for column in training_columns
                if column in candidate_column_set
            ]

            dtype_mismatches: dict[
                str,
                dict[str, str],
            ] = {}

            for column in common_columns:
                training_dtype = str(
                    train.X[column].dtype
                )

                candidate_dtype = str(
                    partition.X[column].dtype
                )

                if (
                    training_dtype
                    != candidate_dtype
                ):
                    dtype_mismatches[column] = {
                        "training": training_dtype,
                        name.value: candidate_dtype,
                    }

            if dtype_mismatches:
                issues.append(
                    OutlierContractIssue(
                        code="P5-CONTRACT-019",
                        severity="error",
                        message=(
                            f"Partition {name.value!r} "
                            "contains incompatible dtypes."
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

    def _validate_numeric_values(
        self,
        *,
        train: DatasetPartition,
        validation: DatasetPartition,
        test: DatasetPartition,
        issues: list[OutlierContractIssue],
    ) -> None:
        """Validate numeric values before outlier treatment."""

        numeric_columns = [
            str(column)
            for column in train.X.columns
            if (
                pd.api.types.is_numeric_dtype(
                    train.X[column].dtype
                )
                and not pd.api.types.is_bool_dtype(
                    train.X[column].dtype
                )
            )
        ]

        partitions = {
            PartitionName.TRAIN: train,
            PartitionName.VALIDATION: validation,
            PartitionName.TEST: test,
        }

        for name, partition in partitions.items():
            for column in numeric_columns:
                if column not in partition.X.columns:
                    continue

                numeric = pd.to_numeric(
                    partition.X[column],
                    errors="coerce",
                )

                if (
                    self.options
                    .require_no_missing_numeric_values
                    and numeric.isna().any()
                ):
                    issues.append(
                        OutlierContractIssue(
                            code="P5-CONTRACT-020",
                            severity="critical",
                            message=(
                                f"Numeric feature {column!r} "
                                f"contains missing values in "
                                f"{name.value!r}."
                            ),
                            columns=[column],
                            partitions=[name],
                            blocks_execution=True,
                            details={
                                "missing_count": int(
                                    numeric.isna().sum()
                                ),
                            },
                        )
                    )

                nonfinite_mask = (
                    numeric.notna()
                    & ~np.isfinite(
                        numeric.astype(float)
                    )
                )

                if (
                    self.options
                    .require_finite_numeric_values
                    and bool(
                        nonfinite_mask.any()
                    )
                ):
                    issues.append(
                        OutlierContractIssue(
                            code="P5-CONTRACT-021",
                            severity="critical",
                            message=(
                                f"Numeric feature {column!r} "
                                f"contains nonfinite values in "
                                f"{name.value!r}."
                            ),
                            columns=[column],
                            partitions=[name],
                            blocks_execution=True,
                            details={
                                "nonfinite_count": int(
                                    nonfinite_mask.sum()
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
            deep=self.options
            .copy_input_partitions
        )

    def _resolve_phase4_state(
        self,
        phase4_result: Phase4Result,
    ) -> Phase4State:
        """Resolve Phase 4 state from its result."""

        if not self.options.require_phase4_state:
            return Phase4State()

        stored_state = getattr(
            phase4_result,
            "phase_state",
            None,
        )

        if stored_state:
            return Phase4State.model_validate(
                stored_state
            )

        direct_state = getattr(
            phase4_result,
            "state",
            None,
        )

        if isinstance(
            direct_state,
            Phase4State,
        ):
            return direct_state

        raise ValueError(
            "Phase 4 state is unavailable. Pass "
            "phase4_state explicitly or store the state "
            "inside the Phase 4 result."
        )