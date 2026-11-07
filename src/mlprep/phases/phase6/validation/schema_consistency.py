"""Output-schema consistency validation for Phase 6."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from mlprep.phases.phase2.output_governance.metadata import (
    PartitionName,
)
from mlprep.phases.phase2.output_governance.result import (
    DatasetPartition,
)


class SchemaMismatch(BaseModel):
    """One output-schema mismatch."""

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
    )

    reference_partition: PartitionName
    candidate_partition: PartitionName

    missing_columns: list[str] = Field(
        default_factory=list
    )

    extra_columns: list[str] = Field(
        default_factory=list
    )

    order_mismatch: bool = False

    dtype_mismatches: dict[
        str,
        dict[str, str],
    ] = Field(default_factory=dict)

    duplicate_columns: list[str] = Field(
        default_factory=list
    )

    nonnumeric_columns: list[str] = Field(
        default_factory=list
    )

    expected_schema_mismatch: bool = False

    blocks_readiness: bool = True

    metadata: dict[str, Any] = Field(
        default_factory=dict
    )


class SchemaConsistencyResult(BaseModel):
    """Complete Phase 6 schema-validation output."""

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
    )

    valid: bool = True

    reference_columns: list[str] = Field(
        default_factory=list
    )

    expected_columns: list[str] = Field(
        default_factory=list
    )

    mismatches: list[
        SchemaMismatch
    ] = Field(default_factory=list)

    feature_count_by_partition: dict[
        PartitionName,
        int,
    ] = Field(default_factory=dict)

    metadata: dict[str, Any] = Field(
        default_factory=dict
    )


@dataclass(slots=True)
class SchemaConsistencyOptions:
    """Options controlling output-schema validation."""

    enabled: bool = True

    require_identical_columns: bool = True
    require_identical_order: bool = True
    require_compatible_dtypes: bool = True

    require_unique_column_names: bool = True
    require_numeric_output: bool = True

    require_expected_schema: bool = True

    allow_nullable_dtype_equivalence: bool = True
    allow_numeric_width_equivalence: bool = True

    def __post_init__(self) -> None:
        """Validate schema-consistency options."""

        boolean_options = {
            "enabled": self.enabled,
            "require_identical_columns": (
                self.require_identical_columns
            ),
            "require_identical_order": (
                self.require_identical_order
            ),
            "require_compatible_dtypes": (
                self.require_compatible_dtypes
            ),
            "require_unique_column_names": (
                self.require_unique_column_names
            ),
            "require_numeric_output": (
                self.require_numeric_output
            ),
            "require_expected_schema": (
                self.require_expected_schema
            ),
            "allow_nullable_dtype_equivalence": (
                self.allow_nullable_dtype_equivalence
            ),
            "allow_numeric_width_equivalence": (
                self.allow_numeric_width_equivalence
            ),
        }

        for name, value in boolean_options.items():
            if not isinstance(value, bool):
                raise TypeError(
                    f"{name} must be a boolean."
                )


class SchemaConsistencyValidator:
    """Validate Phase 6 output schemas."""

    component_name = "schema_consistency"

    def __init__(
        self,
        options: SchemaConsistencyOptions | None = None,
    ) -> None:
        """Initialize the validator."""

        self.options = (
            options
            or SchemaConsistencyOptions()
        )

    def validate(
        self,
        *,
        train: DatasetPartition,
        validation: DatasetPartition,
        test: DatasetPartition,
        expected_output_columns: list[str],
    ) -> SchemaConsistencyResult:
        """Validate transformed schemas across partitions."""

        if not self.options.enabled:
            return SchemaConsistencyResult(
                metadata={
                    "enabled": False,
                }
            )

        expected_columns = [
            str(column)
            for column in expected_output_columns
        ]

        reference_columns = (
            train.feature_names()
        )

        mismatches: list[
            SchemaMismatch
        ] = []

        training_mismatch = self._validate_reference(
            train=train,
            expected_columns=expected_columns,
        )

        if training_mismatch is not None:
            mismatches.append(
                training_mismatch
            )

        for candidate in (
            validation,
            test,
        ):
            mismatch = self._compare(
                reference=train,
                candidate=candidate,
                expected_columns=expected_columns,
            )

            if mismatch is not None:
                mismatches.append(
                    mismatch
                )

        return SchemaConsistencyResult(
            valid=not any(
                mismatch.blocks_readiness
                for mismatch in mismatches
            ),
            reference_columns=(
                reference_columns
            ),
            expected_columns=(
                expected_columns
            ),
            mismatches=mismatches,
            feature_count_by_partition={
                PartitionName.TRAIN: (
                    train.feature_count()
                ),
                PartitionName.VALIDATION: (
                    validation.feature_count()
                ),
                PartitionName.TEST: (
                    test.feature_count()
                ),
            },
            metadata={
                "enabled": True,
                "expected_feature_count": len(
                    expected_columns
                ),
            },
        )

    def _validate_reference(
        self,
        *,
        train: DatasetPartition,
        expected_columns: list[str],
    ) -> SchemaMismatch | None:
        """Validate the training schema itself."""

        training_columns = (
            train.feature_names()
        )

        duplicate_columns = self._duplicates(
            training_columns
        )

        nonnumeric_columns = self._nonnumeric_columns(
            train
        )

        expected_schema_mismatch = bool(
            training_columns
            != expected_columns
        )

        blocks_readiness = bool(
            (
                self.options
                .require_unique_column_names
                and duplicate_columns
            )
            or (
                self.options
                .require_numeric_output
                and nonnumeric_columns
            )
            or (
                self.options
                .require_expected_schema
                and expected_schema_mismatch
            )
        )

        has_mismatch = bool(
            duplicate_columns
            or nonnumeric_columns
            or expected_schema_mismatch
        )

        if not has_mismatch:
            return None

        return SchemaMismatch(
            reference_partition=(
                PartitionName.TRAIN
            ),
            candidate_partition=(
                PartitionName.TRAIN
            ),
            duplicate_columns=(
                duplicate_columns
            ),
            nonnumeric_columns=(
                nonnumeric_columns
            ),
            expected_schema_mismatch=(
                expected_schema_mismatch
            ),
            blocks_readiness=(
                blocks_readiness
            ),
            metadata={
                "self_schema_check": True,
            },
        )

    def _compare(
        self,
        *,
        reference: DatasetPartition,
        candidate: DatasetPartition,
        expected_columns: list[str],
    ) -> SchemaMismatch | None:
        """Compare one partition against training."""

        reference_columns = (
            reference.feature_names()
        )

        candidate_columns = (
            candidate.feature_names()
        )

        reference_set = set(
            reference_columns
        )

        candidate_set = set(
            candidate_columns
        )

        missing_columns = sorted(
            reference_set.difference(
                candidate_set
            )
        )

        extra_columns = sorted(
            candidate_set.difference(
                reference_set
            )
        )

        order_mismatch = bool(
            reference_columns
            != candidate_columns
        )

        duplicate_columns = self._duplicates(
            candidate_columns
        )

        nonnumeric_columns = self._nonnumeric_columns(
            candidate
        )

        expected_schema_mismatch = bool(
            candidate_columns
            != expected_columns
        )

        dtype_mismatches: dict[
            str,
            dict[str, str],
        ] = {}

        for column in reference_columns:
            if column not in candidate_set:
                continue

            reference_dtype = str(
                reference.X[
                    column
                ].dtype
            )

            candidate_dtype = str(
                candidate.X[
                    column
                ].dtype
            )

            if self._dtypes_compatible(
                reference_dtype=reference_dtype,
                candidate_dtype=candidate_dtype,
            ):
                continue

            dtype_mismatches[
                column
            ] = {
                "training": reference_dtype,
                candidate.name.value: (
                    candidate_dtype
                ),
            }

        blocks_readiness = bool(
            (
                self.options
                .require_identical_columns
                and (
                    missing_columns
                    or extra_columns
                )
            )
            or (
                self.options
                .require_identical_order
                and order_mismatch
            )
            or (
                self.options
                .require_compatible_dtypes
                and dtype_mismatches
            )
            or (
                self.options
                .require_unique_column_names
                and duplicate_columns
            )
            or (
                self.options
                .require_numeric_output
                and nonnumeric_columns
            )
            or (
                self.options
                .require_expected_schema
                and expected_schema_mismatch
            )
        )

        has_mismatch = bool(
            missing_columns
            or extra_columns
            or order_mismatch
            or dtype_mismatches
            or duplicate_columns
            or nonnumeric_columns
            or expected_schema_mismatch
        )

        if not has_mismatch:
            return None

        return SchemaMismatch(
            reference_partition=(
                PartitionName.TRAIN
            ),
            candidate_partition=(
                candidate.name
            ),
            missing_columns=missing_columns,
            extra_columns=extra_columns,
            order_mismatch=order_mismatch,
            dtype_mismatches=(
                dtype_mismatches
            ),
            duplicate_columns=(
                duplicate_columns
            ),
            nonnumeric_columns=(
                nonnumeric_columns
            ),
            expected_schema_mismatch=(
                expected_schema_mismatch
            ),
            blocks_readiness=(
                blocks_readiness
            ),
        )

    def _dtypes_compatible(
        self,
        *,
        reference_dtype: str,
        candidate_dtype: str,
    ) -> bool:
        """Return whether two output dtypes are compatible."""

        if reference_dtype == candidate_dtype:
            return True

        normalized_reference = self._normalize_dtype(
            reference_dtype
        )

        normalized_candidate = self._normalize_dtype(
            candidate_dtype
        )

        if (
            self.options
            .allow_nullable_dtype_equivalence
            and normalized_reference
            == normalized_candidate
        ):
            return True

        return bool(self.options.allow_numeric_width_equivalence and normalized_reference in {"integer", "floating"} and normalized_reference == normalized_candidate)

    @staticmethod
    def _normalize_dtype(
        dtype: str,
    ) -> str:
        """Normalize related dtype names."""

        normalized = dtype.casefold()

        if (
            normalized.startswith(
                "int"
            )
            or normalized.startswith(
                "uint"
            )
        ):
            return "integer"

        if normalized.startswith(
            "float"
        ):
            return "floating"

        if normalized in {
            "bool",
            "boolean",
        }:
            return "boolean"

        return normalized

    @staticmethod
    def _duplicates(
        values: list[str],
    ) -> list[object]:
        """Return duplicate values while preserving order."""

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

    @staticmethod
    def _nonnumeric_columns(
        partition: DatasetPartition,
    ) -> list[object]:
        """Return nonnumeric output columns."""

        return [
            str(column)
            for column in partition.X.columns
            if not pd.api.types.is_numeric_dtype(
                partition.X[
                    column
                ].dtype
            )
        ]
