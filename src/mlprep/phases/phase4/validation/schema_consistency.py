"""Feature-schema consistency validation for Phase 4."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from mlprep.phases.phase2.output_governance.metadata import (
    PartitionName,
)
from mlprep.phases.phase2.output_governance.result import (
    DatasetPartition,
)


class SchemaMismatch(BaseModel):
    """One schema mismatch between two partitions."""

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

    blocks_readiness: bool = True

    metadata: dict[str, Any] = Field(
        default_factory=dict
    )


class SchemaConsistencyResult(BaseModel):
    """Complete schema-consistency validation output."""

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
    )

    valid: bool = True

    reference_columns: list[str] = Field(
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
    """Options controlling schema-consistency validation."""

    enabled: bool = True

    require_identical_columns: bool = True
    require_identical_order: bool = True
    require_compatible_dtypes: bool = True
    require_unique_column_names: bool = True

    allow_nullable_dtype_equivalence: bool = True

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
            "allow_nullable_dtype_equivalence": (
                self.allow_nullable_dtype_equivalence
            ),
        }

        for name, value in boolean_options.items():
            if not isinstance(value, bool):
                raise TypeError(
                    f"{name} must be a boolean."
                )


class SchemaConsistencyValidator:
    """Validate transformed feature schemas."""

    component_name = "schema_consistency"

    def __init__(
        self,
        options: (
            SchemaConsistencyOptions | None
        ) = None,
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
    ) -> SchemaConsistencyResult:
        """Validate schemas across transformed partitions."""

        if not self.options.enabled:
            return SchemaConsistencyResult(
                metadata={
                    "enabled": False,
                }
            )

        partitions = {
            PartitionName.TRAIN: train,
            PartitionName.VALIDATION: validation,
            PartitionName.TEST: test,
        }

        reference_columns = (
            train.feature_names()
        )

        mismatches: list[
            SchemaMismatch
        ] = []

        for candidate_name in (
            PartitionName.VALIDATION,
            PartitionName.TEST,
        ):
            candidate = partitions[
                candidate_name
            ]

            mismatch = self._compare(
                reference=train,
                candidate=candidate,
            )

            if mismatch is not None:
                mismatches.append(
                    mismatch
                )

        training_duplicates = (
            self._find_duplicates(
                reference_columns
            )
        )

        if (
            training_duplicates
            and self.options
            .require_unique_column_names
        ):
            mismatches.insert(
                0,
                SchemaMismatch(
                    reference_partition=(
                        PartitionName.TRAIN
                    ),
                    candidate_partition=(
                        PartitionName.TRAIN
                    ),
                    duplicate_columns=(
                        training_duplicates
                    ),
                    blocks_readiness=True,
                    metadata={
                        "self_schema_check": True,
                    },
                ),
            )

        return SchemaConsistencyResult(
            valid=not any(
                mismatch.blocks_readiness
                for mismatch in mismatches
            ),
            reference_columns=reference_columns,
            mismatches=mismatches,
            feature_count_by_partition={
                name: partition.feature_count()
                for name, partition
                in partitions.items()
            },
            metadata={
                "enabled": True,
            },
        )

    def _compare(
        self,
        *,
        reference: DatasetPartition,
        candidate: DatasetPartition,
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

        duplicate_columns = (
            self._find_duplicates(
                candidate_columns
            )
        )

        dtype_mismatches: dict[
            str,
            dict[str, str],
        ] = {}

        common_columns = [
            column
            for column in reference_columns
            if column in candidate_set
        ]

        for column in common_columns:
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
                reference_dtype,
                candidate_dtype,
            ):
                continue

            dtype_mismatches[column] = {
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
        )

        has_mismatch = bool(
            missing_columns
            or extra_columns
            or order_mismatch
            or dtype_mismatches
            or duplicate_columns
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
            blocks_readiness=(
                blocks_readiness
            ),
        )

    def _dtypes_compatible(
        self,
        reference_dtype: str,
        candidate_dtype: str,
    ) -> bool:
        """Return whether two dtype names are compatible."""

        if reference_dtype == candidate_dtype:
            return True

        if not self.options.allow_nullable_dtype_equivalence:
            return False

        normalized_reference = (
            self._normalize_dtype(
                reference_dtype
            )
        )

        normalized_candidate = (
            self._normalize_dtype(
                candidate_dtype
            )
        )

        return (
            normalized_reference
            == normalized_candidate
        )

    @staticmethod
    def _normalize_dtype(
        dtype: str,
    ) -> str:
        """Normalize nullable and non-nullable dtype names."""

        normalized = dtype.casefold()

        equivalences = {
            "int64": "integer",
            "int32": "integer",
            "int16": "integer",
            "int8": "integer",
            "uint64": "integer",
            "uint32": "integer",
            "uint16": "integer",
            "uint8": "integer",
            "float64": "floating",
            "float32": "floating",
            "float16": "floating",
            "boolean": "boolean",
            "bool": "boolean",
            "string": "string",
            "object": "object",
        }

        if normalized.startswith(
            "int"
        ) or normalized.startswith(
            "uint"
        ):
            return "integer"

        if normalized.startswith(
            "float"
        ):
            return "floating"

        return equivalences.get(
            normalized,
            normalized,
        )

    @staticmethod
    def _find_duplicates(
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
                duplicates.append(value)

            seen.add(value)

        return duplicates
