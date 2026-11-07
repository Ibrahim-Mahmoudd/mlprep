"""Input-contract validation for Phase 7."""

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
from mlprep.phases.phase6.output_governance.result import (
    Phase6Result,
)
from mlprep.phases.phase6.output_governance.state import (
    Phase6State,
)

ContractSeverity = Literal[
    "info",
    "warning",
    "error",
    "critical",
]


@dataclass(slots=True)
class SelectionContractIssue:
    """One issue detected in the Phase 7 input contract."""

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
class SelectionContractOptions:
    """Options controlling the Phase 7 input contract."""

    require_phase6_readiness: bool = True
    require_phase6_state: bool = True

    require_phase6_output_fingerprint: bool = True
    require_training_only_phase6_fit: bool = True

    require_nonempty_training_partition: bool = True
    require_validation_partition: bool = True
    require_test_partition: bool = True

    require_identical_feature_schema: bool = True
    require_identical_feature_order: bool = True
    require_compatible_dtypes: bool = True
    require_numeric_features: bool = True

    require_no_missing_feature_values: bool = True
    require_finite_feature_values: bool = True

    require_aligned_indices: bool = True
    require_unique_row_ids: bool = True
    require_nonmissing_row_ids: bool = True
    require_disjoint_row_ids: bool = True

    require_unchanged_targets: bool = True
    require_unchanged_row_ids: bool = True
    require_consistent_phase6_schemas: bool = True
    require_phase6_lineage: bool = True

    require_protected_test_partition: bool = True

    copy_input_partitions: bool = True

    required_features: list[str] = field(
        default_factory=list
    )

    protected_features: list[str] = field(
        default_factory=list
    )

    prohibited_features: list[str] = field(
        default_factory=list
    )

    maximum_reported_columns: int = 100
    maximum_reported_row_ids: int = 20

    def __post_init__(self) -> None:
        """Validate contract options."""

        boolean_options = {
            "require_phase6_readiness": (
                self.require_phase6_readiness
            ),
            "require_phase6_state": (
                self.require_phase6_state
            ),
            "require_phase6_output_fingerprint": (
                self.require_phase6_output_fingerprint
            ),
            "require_training_only_phase6_fit": (
                self.require_training_only_phase6_fit
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
            "require_unchanged_targets": (
                self.require_unchanged_targets
            ),
            "require_unchanged_row_ids": (
                self.require_unchanged_row_ids
            ),
            "require_consistent_phase6_schemas": (
                self.require_consistent_phase6_schemas
            ),
            "require_phase6_lineage": (
                self.require_phase6_lineage
            ),
            "require_protected_test_partition": (
                self.require_protected_test_partition
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

        self.required_features = self._normalize_columns(
            self.required_features,
            name="required_features",
        )

        self.protected_features = self._normalize_columns(
            self.protected_features,
            name="protected_features",
        )

        self.prohibited_features = self._normalize_columns(
            self.prohibited_features,
            name="prohibited_features",
        )

        required = set(
            self.required_features
        )

        protected = set(
            self.protected_features
        )

        prohibited = set(
            self.prohibited_features
        )

        conflicting_required = required.intersection(
            prohibited
        )

        if conflicting_required:
            raise ValueError(
                "Features cannot be both required and "
                f"prohibited: {sorted(conflicting_required)}"
            )

        conflicting_protected = protected.intersection(
            prohibited
        )

        if conflicting_protected:
            raise ValueError(
                "Features cannot be both protected and "
                f"prohibited: {sorted(conflicting_protected)}"
            )

    @staticmethod
    def _normalize_columns(
        values: list[str],
        *,
        name: str,
    ) -> list[str]:
        """Normalize one feature-name list."""

        normalized = [
            str(value).strip()
            for value in values
        ]

        if any(
            not value
            for value in normalized
        ):
            raise ValueError(
                f"{name} cannot contain empty names."
            )

        if len(normalized) != len(
            set(normalized)
        ):
            raise ValueError(
                f"{name} contains duplicate feature names."
            )

        return normalized


class SelectionContract:
    """Resolved input contract used by Phase 7."""

    def __init__(
        self,
        *,
        train: DatasetPartition,
        validation: DatasetPartition,
        test: DatasetPartition,
        phase6_state: Phase6State,
        issues: list[SelectionContractIssue],
        options: SelectionContractOptions,
    ) -> None:
        """Initialize the resolved selection contract."""

        self.train = train
        self.validation = validation
        self.test = test

        self.phase6_state = phase6_state

        self.issues = issues
        self.options = options

    @property
    def feature_columns(
        self,
    ) -> list[str]:
        """Return ordered input feature columns."""

        return [
            str(column)
            for column in self.train.X.columns
        ]

    @property
    def required_features(
        self,
    ) -> list[str]:
        """Return required features."""

        return list(
            self.options.required_features
        )

    @property
    def protected_features(
        self,
    ) -> list[str]:
        """Return protected features."""

        return list(
            self.options.protected_features
        )

    @property
    def prohibited_features(
        self,
    ) -> list[str]:
        """Return prohibited features."""

        return list(
            self.options.prohibited_features
        )

    @property
    def selectable_features(
        self,
    ) -> list[str]:
        """Return features eligible for selection."""

        prohibited = set(
            self.prohibited_features
        )

        return [
            column
            for column in self.feature_columns
            if column not in prohibited
        ]

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
        """Raise if contract issues block Phase 7."""

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
            "The Phase 7 selection contract contains "
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


class SelectionContractResolver:
    """Resolve and validate the Phase 7 input contract."""

    component_name = "selection_contract"

    def __init__(
        self,
        options: SelectionContractOptions | None = None,
    ) -> None:
        """Initialize the contract resolver."""

        self.options = (
            options
            or SelectionContractOptions()
        )

    def resolve(
        self,
        *,
        phase6_result: Phase6Result,
        phase6_state: Phase6State | None = None,
        authorized_test_partition: (
            DatasetPartition | None
        ) = None,
    ) -> SelectionContract:
        """Resolve the complete Phase 7 contract."""

        if not isinstance(
            phase6_result,
            Phase6Result,
        ):
            raise TypeError(
                "phase6_result must be a Phase6Result."
            )

        resolved_state = (
            phase6_state
            or self._resolve_phase6_state(
                phase6_result
            )
        )

        issues: list[
            SelectionContractIssue
        ] = []

        self._validate_phase6_readiness(
            result=phase6_result,
            issues=issues,
        )

        self._validate_phase6_state(
            state=resolved_state,
            issues=issues,
        )

        test_partition = (
            authorized_test_partition
            if authorized_test_partition is not None
            else phase6_result.test
        )

        if not isinstance(
            test_partition,
            DatasetPartition,
        ):
            raise PermissionError(
                "Phase 7 requires an authorized transformed "
                "test partition from Phase 6."
            )

        train = self._copy_partition(
            phase6_result.train
        )

        validation = self._copy_partition(
            phase6_result.validation
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

        self._validate_feature_values(
            train=train,
            validation=validation,
            test=test,
            issues=issues,
        )

        self._validate_configured_features(
            feature_columns=[
                str(column)
                for column in train.X.columns
            ],
            issues=issues,
        )

        return SelectionContract(
            train=train,
            validation=validation,
            test=test,
            phase6_state=resolved_state,
            issues=issues,
            options=self.options,
        )

    def _validate_phase6_readiness(
        self,
        *,
        result: Phase6Result,
        issues: list[SelectionContractIssue],
    ) -> None:
        """Validate Phase 6 readiness."""

        if not self.options.require_phase6_readiness:
            return

        readiness = result.readiness

        if readiness is None:
            issues.append(
                SelectionContractIssue(
                    code="P7-CONTRACT-001",
                    severity="critical",
                    message=(
                        "Phase 6 readiness was not evaluated."
                    ),
                    blocks_execution=True,
                )
            )

            return

        if not readiness.ready:
            issues.append(
                SelectionContractIssue(
                    code="P7-CONTRACT-002",
                    severity="critical",
                    message=(
                        "Phase 6 did not approve progression "
                        "to Phase 7."
                    ),
                    blocks_execution=True,
                    details={
                        "phase6_status": (
                            readiness.status.value
                        ),
                        "phase6_reasons": list(
                            readiness.reasons
                        ),
                    },
                )
            )

    def _validate_phase6_state(
        self,
        *,
        state: Phase6State,
        issues: list[SelectionContractIssue],
    ) -> None:
        """Validate required Phase 6 governance state."""

        state_checks = [
            (
                self.options
                .require_phase6_output_fingerprint
                and not state.output_fingerprint,
                "P7-CONTRACT-003",
                (
                    "The Phase 6 output fingerprint "
                    "is unavailable."
                ),
            ),
            (
                self.options
                .require_training_only_phase6_fit
                and not state
                .fitted_on_training_only,
                "P7-CONTRACT-004",
                (
                    "Phase 6 transformation state was not fitted "
                    "exclusively on training data."
                ),
            ),
            (
                self.options.require_unchanged_targets
                and not state.targets_unchanged,
                "P7-CONTRACT-005",
                (
                    "Phase 6 did not verify unchanged targets."
                ),
            ),
            (
                self.options.require_unchanged_row_ids
                and not state.row_ids_unchanged,
                "P7-CONTRACT-006",
                (
                    "Phase 6 did not verify unchanged row IDs."
                ),
            ),
            (
                self.options
                .require_consistent_phase6_schemas
                and not state.schemas_consistent,
                "P7-CONTRACT-007",
                (
                    "Phase 6 output schemas are inconsistent."
                ),
            ),
            (
                self.options.require_phase6_lineage
                and state.lineage is None,
                "P7-CONTRACT-008",
                "Phase 6 feature lineage is unavailable.",
            ),
            (
                self.options
                .require_protected_test_partition
                and not state
                .test_partition_remained_protected,
                "P7-CONTRACT-009",
                (
                    "The test partition did not remain "
                    "protected during Phase 6."
                ),
            ),
        ]

        for condition, code, message in state_checks:
            if condition:
                issues.append(
                    SelectionContractIssue(
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
        issues: list[SelectionContractIssue],
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
                SelectionContractIssue(
                    code="P7-CONTRACT-010",
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
        issues: list[SelectionContractIssue],
    ) -> None:
        """Validate required partition sizes."""

        checks = [
            (
                self.options
                .require_nonempty_training_partition,
                train,
                "P7-CONTRACT-011",
                "The training partition is empty.",
            ),
            (
                self.options
                .require_validation_partition,
                validation,
                "P7-CONTRACT-012",
                "The validation partition is empty.",
            ),
            (
                self.options.require_test_partition,
                test,
                "P7-CONTRACT-013",
                "The test partition is empty.",
            ),
        ]

        for required, partition, code, message in checks:
            if required and partition.row_count() == 0:
                issues.append(
                    SelectionContractIssue(
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
        issues: list[SelectionContractIssue],
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
                    SelectionContractIssue(
                        code="P7-CONTRACT-014",
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
                    SelectionContractIssue(
                        code="P7-CONTRACT-015",
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
                    SelectionContractIssue(
                        code="P7-CONTRACT-016",
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
                    SelectionContractIssue(
                        code="P7-CONTRACT-017",
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
        issues: list[SelectionContractIssue],
    ) -> None:
        """Validate Phase 6 output schemas."""

        training_columns = (
            train.feature_names()
        )

        training_set = set(
            training_columns
        )

        duplicate_training_columns = self._duplicates(
            training_columns
        )

        if duplicate_training_columns:
            issues.append(
                SelectionContractIssue(
                    code="P7-CONTRACT-018",
                    severity="critical",
                    message=(
                        "The training partition contains "
                        "duplicate feature names."
                    ),
                    columns=duplicate_training_columns,
                    partitions=[
                        PartitionName.TRAIN
                    ],
                    blocks_execution=True,
                )
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
                    SelectionContractIssue(
                        code="P7-CONTRACT-019",
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
                    SelectionContractIssue(
                        code="P7-CONTRACT-020",
                        severity="critical",
                        message=(
                            f"Partition {name.value!r} has "
                            "a different feature order."
                        ),
                        partitions=[
                            PartitionName.TRAIN,
                            name,
                        ],
                        blocks_execution=True,
                    )
                )

            dtype_mismatches: dict[str, dict[str, str]] = {}

            for column in training_columns:
                if column not in candidate_set:
                    continue

                training_dtype = str(
                    train.X[column].dtype
                )

                candidate_dtype = str(
                    partition.X[column].dtype
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
                    SelectionContractIssue(
                        code="P7-CONTRACT-021",
                        severity="error",
                        message=(
                            f"Partition {name.value!r} has "
                            "incompatible feature dtypes."
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
        issues: list[SelectionContractIssue],
    ) -> None:
        """Validate numeric, missing, and finite values."""

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
                        numeric.astype(float)
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
                    SelectionContractIssue(
                        code="P7-CONTRACT-022",
                        severity="critical",
                        message=(
                            f"Partition {name.value!r} contains "
                            "nonnumeric model features."
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
                    SelectionContractIssue(
                        code="P7-CONTRACT-023",
                        severity="critical",
                        message=(
                            f"Partition {name.value!r} contains "
                            "missing feature values."
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
                    SelectionContractIssue(
                        code="P7-CONTRACT-024",
                        severity="critical",
                        message=(
                            f"Partition {name.value!r} contains "
                            "nonfinite feature values."
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

    def _validate_configured_features(
        self,
        *,
        feature_columns: list[str],
        issues: list[SelectionContractIssue],
    ) -> None:
        """Validate required, protected, and prohibited features."""

        available = set(
            feature_columns
        )

        missing_required = sorted(
            set(
                self.options.required_features
            ).difference(
                available
            )
        )

        if missing_required:
            issues.append(
                SelectionContractIssue(
                    code="P7-CONTRACT-025",
                    severity="critical",
                    message=(
                        "One or more required features are "
                        "unavailable."
                    ),
                    columns=missing_required,
                    blocks_execution=True,
                )
            )

        missing_protected = sorted(
            set(
                self.options.protected_features
            ).difference(
                available
            )
        )

        if missing_protected:
            issues.append(
                SelectionContractIssue(
                    code="P7-CONTRACT-026",
                    severity="critical",
                    message=(
                        "One or more protected features are "
                        "unavailable."
                    ),
                    columns=missing_protected,
                    blocks_execution=True,
                )
            )

        present_prohibited = sorted(
            set(
                self.options.prohibited_features
            ).intersection(
                available
            )
        )

        if present_prohibited:
            issues.append(
                SelectionContractIssue(
                    code="P7-CONTRACT-027",
                    severity="warning",
                    message=(
                        "Prohibited features are present and "
                        "must be removed by Phase 7."
                    ),
                    columns=present_prohibited,
                    blocks_execution=False,
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

    def _resolve_phase6_state(
        self,
        result: Phase6Result,
    ) -> Phase6State:
        """Resolve Phase 6 state."""

        if not self.options.require_phase6_state:
            return Phase6State()

        direct_state = getattr(
            result,
            "state",
            None,
        )

        if isinstance(
            direct_state,
            Phase6State,
        ):
            return direct_state

        if result.phase_state:
            return Phase6State.model_validate(
                result.phase_state
            )

        raise ValueError(
            "Phase 6 state is unavailable. Pass phase6_state "
            "explicitly or store it in the Phase 6 result."
        )

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
