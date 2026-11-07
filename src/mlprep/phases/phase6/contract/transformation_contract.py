"""Input-contract validation for Phase 6."""

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
from mlprep.phases.phase5.output_governance.result import (
    Phase5Result,
)
from mlprep.phases.phase5.output_governance.state import (
    Phase5State,
)

ContractSeverity = Literal[
    "info",
    "warning",
    "error",
    "critical",
]


@dataclass(slots=True)
class TransformationContractIssue:
    """One issue detected in the Phase 6 input contract."""

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
class TransformationContractOptions:
    """Options controlling the Phase 6 input contract."""

    require_phase5_readiness: bool = True
    require_phase5_state: bool = True

    require_phase5_output_fingerprint: bool = True
    require_training_only_phase5_fit: bool = True

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

    require_no_missing_feature_values: bool = True
    require_finite_numeric_values: bool = True

    require_unchanged_targets: bool = True
    require_unchanged_row_ids: bool = True
    require_consistent_phase5_schemas: bool = True

    require_protected_test_partition: bool = True

    copy_input_partitions: bool = True

    maximum_reported_columns: int = 100
    maximum_reported_row_ids: int = 20

    def __post_init__(self) -> None:
        """Validate contract options."""

        boolean_options = {
            "require_phase5_readiness": (
                self.require_phase5_readiness
            ),
            "require_phase5_state": (
                self.require_phase5_state
            ),
            "require_phase5_output_fingerprint": (
                self.require_phase5_output_fingerprint
            ),
            "require_training_only_phase5_fit": (
                self.require_training_only_phase5_fit
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
            "require_no_missing_feature_values": (
                self.require_no_missing_feature_values
            ),
            "require_finite_numeric_values": (
                self.require_finite_numeric_values
            ),
            "require_unchanged_targets": (
                self.require_unchanged_targets
            ),
            "require_unchanged_row_ids": (
                self.require_unchanged_row_ids
            ),
            "require_consistent_phase5_schemas": (
                self.require_consistent_phase5_schemas
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


@dataclass(slots=True)
class TransformationFeatureGroups:
    """Feature families available to Phase 6."""

    numeric: list[str] = field(
        default_factory=list
    )

    categorical: list[str] = field(
        default_factory=list
    )

    boolean: list[str] = field(
        default_factory=list
    )

    datetime: list[str] = field(
        default_factory=list
    )

    text: list[str] = field(
        default_factory=list
    )

    timedelta: list[str] = field(
        default_factory=list
    )

    unsupported: list[str] = field(
        default_factory=list
    )

    @property
    def all_supported(
        self,
    ) -> list[object]:
        """Return all supported feature columns."""

        return list(
            dict.fromkeys(
                [
                    *self.numeric,
                    *self.categorical,
                    *self.boolean,
                    *self.datetime,
                    *self.text,
                    *self.timedelta,
                ]
            )
        )

    def family_for(
        self,
        column: str,
    ) -> str:
        """Return the feature family for one column."""

        family_mapping = {
            "numeric": self.numeric,
            "categorical": self.categorical,
            "boolean": self.boolean,
            "datetime": self.datetime,
            "text": self.text,
            "timedelta": self.timedelta,
            "unsupported": self.unsupported,
        }

        for family, columns in family_mapping.items():
            if column in columns:
                return family

        raise KeyError(
            f"Feature family is unavailable for {column!r}."
        )


class TransformationContract:
    """Resolved input contract used by Phase 6."""

    def __init__(
        self,
        *,
        train: DatasetPartition,
        validation: DatasetPartition,
        test: DatasetPartition,
        phase5_state: Phase5State,
        feature_groups: TransformationFeatureGroups,
        issues: list[TransformationContractIssue],
        options: TransformationContractOptions,
    ) -> None:
        """Initialize the resolved contract."""

        self.train = train
        self.validation = validation
        self.test = test

        self.phase5_state = phase5_state
        self.feature_groups = feature_groups

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
        """Raise if contract issues block Phase 6."""

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
            "The Phase 6 transformation contract contains "
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


class TransformationContractResolver:
    """Resolve and validate the Phase 6 input contract."""

    component_name = "transformation_contract"

    def __init__(
        self,
        options: TransformationContractOptions | None = None,
    ) -> None:
        """Initialize the resolver."""

        self.options = (
            options
            or TransformationContractOptions()
        )

    def resolve(
        self,
        *,
        phase5_result: Phase5Result,
        phase5_state: Phase5State | None = None,
        authorized_test_partition: (
            DatasetPartition | None
        ) = None,
    ) -> TransformationContract:
        """Resolve the complete Phase 6 contract."""

        if not isinstance(
            phase5_result,
            Phase5Result,
        ):
            raise TypeError(
                "phase5_result must be a Phase5Result."
            )

        resolved_state = (
            phase5_state
            or self._resolve_phase5_state(
                phase5_result
            )
        )

        issues: list[
            TransformationContractIssue
        ] = []

        self._validate_phase5_readiness(
            result=phase5_result,
            issues=issues,
        )

        self._validate_phase5_state(
            state=resolved_state,
            issues=issues,
        )

        test_partition = (
            authorized_test_partition
            if authorized_test_partition is not None
            else phase5_result.test
        )

        if not isinstance(
            test_partition,
            DatasetPartition,
        ):
            raise PermissionError(
                "Phase 6 requires an authorized transformed "
                "test partition from Phase 5."
            )

        train = self._copy_partition(
            phase5_result.train
        )

        validation = self._copy_partition(
            phase5_result.validation
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

        feature_groups = self._resolve_feature_groups(
            train=train
        )

        if feature_groups.unsupported:
            issues.append(
                TransformationContractIssue(
                    code="P6-CONTRACT-001",
                    severity="warning",
                    message=(
                        "Some features have unsupported dtypes "
                        "and require explicit configuration."
                    ),
                    columns=(
                        feature_groups.unsupported[
                            : self.options
                            .maximum_reported_columns
                        ]
                    ),
                    blocks_execution=False,
                )
            )

        return TransformationContract(
            train=train,
            validation=validation,
            test=test,
            phase5_state=resolved_state,
            feature_groups=feature_groups,
            issues=issues,
            options=self.options,
        )

    def _validate_phase5_readiness(
        self,
        *,
        result: Phase5Result,
        issues: list[TransformationContractIssue],
    ) -> None:
        """Validate Phase 5 readiness."""

        if not self.options.require_phase5_readiness:
            return

        readiness = result.readiness

        if readiness is None:
            issues.append(
                TransformationContractIssue(
                    code="P6-CONTRACT-002",
                    severity="critical",
                    message=(
                        "Phase 5 readiness was not evaluated."
                    ),
                    blocks_execution=True,
                )
            )

            return

        if not readiness.ready:
            issues.append(
                TransformationContractIssue(
                    code="P6-CONTRACT-003",
                    severity="critical",
                    message=(
                        "Phase 5 did not approve progression "
                        "to Phase 6."
                    ),
                    blocks_execution=True,
                    details={
                        "phase5_status": (
                            readiness.status.value
                        ),
                        "phase5_reasons": list(
                            readiness.reasons
                        ),
                    },
                )
            )

    def _validate_phase5_state(
        self,
        *,
        state: Phase5State,
        issues: list[TransformationContractIssue],
    ) -> None:
        """Validate Phase 5 governance state."""

        if (
            self.options.require_phase5_output_fingerprint
            and not state.output_fingerprint
        ):
            issues.append(
                TransformationContractIssue(
                    code="P6-CONTRACT-004",
                    severity="critical",
                    message=(
                        "The Phase 5 output fingerprint "
                        "is unavailable."
                    ),
                    blocks_execution=True,
                )
            )

        if (
            self.options.require_training_only_phase5_fit
            and not state.fitted_on_training_only
        ):
            issues.append(
                TransformationContractIssue(
                    code="P6-CONTRACT-005",
                    severity="critical",
                    message=(
                        "Phase 5 was not fitted exclusively "
                        "on training data."
                    ),
                    blocks_execution=True,
                )
            )

        if (
            self.options.require_unchanged_targets
            and not state.targets_unchanged
        ):
            issues.append(
                TransformationContractIssue(
                    code="P6-CONTRACT-006",
                    severity="critical",
                    message=(
                        "Phase 5 did not verify unchanged targets."
                    ),
                    blocks_execution=True,
                )
            )

        if (
            self.options.require_unchanged_row_ids
            and not state.row_ids_unchanged
        ):
            issues.append(
                TransformationContractIssue(
                    code="P6-CONTRACT-007",
                    severity="critical",
                    message=(
                        "Phase 5 did not verify unchanged row IDs."
                    ),
                    blocks_execution=True,
                )
            )

        if (
            self.options.require_consistent_phase5_schemas
            and not state.schemas_consistent
        ):
            issues.append(
                TransformationContractIssue(
                    code="P6-CONTRACT-008",
                    severity="critical",
                    message=(
                        "Phase 5 output schemas are inconsistent."
                    ),
                    blocks_execution=True,
                )
            )

        if (
            self.options.require_protected_test_partition
            and not state
            .test_partition_remained_protected
        ):
            issues.append(
                TransformationContractIssue(
                    code="P6-CONTRACT-009",
                    severity="critical",
                    message=(
                        "The test partition did not remain "
                        "protected during Phase 5."
                    ),
                    partitions=[
                        PartitionName.TEST
                    ],
                    blocks_execution=True,
                )
            )

    def _validate_partition_names(
        self,
        *,
        train: DatasetPartition,
        validation: DatasetPartition,
        test: DatasetPartition,
        issues: list[TransformationContractIssue],
    ) -> None:
        """Validate partition identity metadata."""

        expected_partitions = {
            PartitionName.TRAIN: train,
            PartitionName.VALIDATION: validation,
            PartitionName.TEST: test,
        }

        for expected_name, partition in (
            expected_partitions.items()
        ):
            if partition.name == expected_name:
                continue

            issues.append(
                TransformationContractIssue(
                    code="P6-CONTRACT-010",
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
        issues: list[TransformationContractIssue],
    ) -> None:
        """Validate required partition sizes."""

        checks = [
            (
                self.options
                .require_nonempty_training_partition,
                train,
                "P6-CONTRACT-011",
                "The training partition is empty.",
            ),
            (
                self.options
                .require_validation_partition,
                validation,
                "P6-CONTRACT-012",
                "The validation partition is empty.",
            ),
            (
                self.options.require_test_partition,
                test,
                "P6-CONTRACT-013",
                "The test partition is empty.",
            ),
        ]

        for required, partition, code, message in checks:
            if required and partition.row_count() == 0:
                issues.append(
                    TransformationContractIssue(
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
        issues: list[TransformationContractIssue],
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
                    TransformationContractIssue(
                        code="P6-CONTRACT-014",
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
                    TransformationContractIssue(
                        code="P6-CONTRACT-015",
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
                self.options.require_nonmissing_row_ids
                and partition.row_ids.isna().any()
            ):
                issues.append(
                    TransformationContractIssue(
                        code="P6-CONTRACT-016",
                        severity="critical",
                        message=(
                            f"Partition {name.value!r} "
                            "contains missing row IDs."
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
                    TransformationContractIssue(
                        code="P6-CONTRACT-017",
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
        issues: list[TransformationContractIssue],
    ) -> None:
        """Validate input feature schemas."""

        training_columns = (
            train.feature_names()
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
                self.options.require_identical_feature_schema
                and (
                    missing_columns
                    or extra_columns
                )
            ):
                issues.append(
                    TransformationContractIssue(
                        code="P6-CONTRACT-018",
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
                self.options.require_identical_feature_order
                and candidate_columns
                != training_columns
            ):
                issues.append(
                    TransformationContractIssue(
                        code="P6-CONTRACT-019",
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

            if not self.options.require_compatible_dtypes:
                continue

            dtype_mismatches: dict[
                str,
                dict[str, str],
            ] = {}

            for column in training_columns:
                if column not in candidate_set:
                    continue

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
                    TransformationContractIssue(
                        code="P6-CONTRACT-020",
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
        issues: list[TransformationContractIssue],
    ) -> None:
        """Validate missing and nonfinite feature values."""

        partitions = {
            PartitionName.TRAIN: train,
            PartitionName.VALIDATION: validation,
            PartitionName.TEST: test,
        }

        for name, partition in partitions.items():
            if (
                self.options
                .require_no_missing_feature_values
            ):
                missing_counts = (
                    partition.X.isna().sum()
                )

                missing_columns = [
                    str(column)
                    for column, count
                    in missing_counts.items()
                    if int(count) > 0
                ]

                if missing_columns:
                    issues.append(
                        TransformationContractIssue(
                            code="P6-CONTRACT-021",
                            severity="critical",
                            message=(
                                f"Partition {name.value!r} "
                                "contains missing feature values."
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

            if not self.options.require_finite_numeric_values:
                continue

            nonfinite_columns: list[str] = []

            for column in partition.X.columns:
                series = partition.X[column]

                if not pd.api.types.is_numeric_dtype(
                    series.dtype
                ):
                    continue

                numeric = pd.to_numeric(
                    series,
                    errors="coerce",
                )

                mask = (
                    numeric.notna()
                    & ~np.isfinite(
                        numeric.astype(float)
                    )
                )

                if bool(mask.any()):
                    nonfinite_columns.append(
                        str(column)
                    )

            if nonfinite_columns:
                issues.append(
                    TransformationContractIssue(
                        code="P6-CONTRACT-022",
                        severity="critical",
                        message=(
                            f"Partition {name.value!r} "
                            "contains nonfinite numeric values."
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

    @staticmethod
    def _resolve_feature_groups(
        *,
        train: DatasetPartition,
    ) -> TransformationFeatureGroups:
        """Infer transformation feature families."""

        groups = TransformationFeatureGroups()

        for column in train.X.columns:
            column_name = str(
                column
            )

            series = train.X[column]

            if pd.api.types.is_bool_dtype(
                series.dtype
            ):
                groups.boolean.append(
                    column_name
                )

            elif pd.api.types.is_numeric_dtype(
                series.dtype
            ):
                groups.numeric.append(
                    column_name
                )

            elif pd.api.types.is_datetime64_any_dtype(
                series.dtype
            ):
                groups.datetime.append(
                    column_name
                )

            elif pd.api.types.is_timedelta64_dtype(
                series.dtype
            ):
                groups.timedelta.append(
                    column_name
                )

            elif isinstance(
                series.dtype,
                pd.CategoricalDtype,
            ):
                groups.categorical.append(
                    column_name
                )

            elif (
                pd.api.types.is_string_dtype(
                    series.dtype
                )
                or pd.api.types.is_object_dtype(
                    series.dtype
                )
            ):
                unique_ratio = (
                    series.nunique(
                        dropna=True
                    )
                    / max(
                        len(series),
                        1,
                    )
                )

                if unique_ratio <= 0.50:
                    groups.categorical.append(
                        column_name
                    )

                else:
                    groups.text.append(
                        column_name
                    )

            else:
                groups.unsupported.append(
                    column_name
                )

        return groups

    def _copy_partition(
        self,
        partition: DatasetPartition,
    ) -> DatasetPartition:
        """Copy one partition according to policy."""

        return partition.copy(
            deep=self.options.copy_input_partitions
        )

    def _resolve_phase5_state(
        self,
        result: Phase5Result,
    ) -> Phase5State:
        """Resolve the Phase 5 state."""

        if not self.options.require_phase5_state:
            return Phase5State()

        direct_state = getattr(
            result,
            "state",
            None,
        )

        if isinstance(
            direct_state,
            Phase5State,
        ):
            return direct_state

        if result.phase_state:
            return Phase5State.model_validate(
                result.phase_state
            )

        raise ValueError(
            "Phase 5 state is unavailable. Pass the state "
            "explicitly or store it in the Phase 5 result."
        )
