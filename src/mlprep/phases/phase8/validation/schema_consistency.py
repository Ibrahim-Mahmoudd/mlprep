"""Feature-schema validation for Phase 8."""

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


class PartitionSchemaValidation(BaseModel):
    """Schema validation for one Phase 8 partition."""

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
    )

    partition: PartitionName

    columns_match: bool = True
    order_matches: bool = True
    dtypes_compatible: bool = True
    duplicate_columns_absent: bool = True

    missing_columns: list[str] = Field(
        default_factory=list,
    )

    extra_columns: list[str] = Field(
        default_factory=list,
    )

    duplicate_columns: list[str] = Field(
        default_factory=list,
    )

    dtype_mismatches: dict[
        str,
        dict[str, str],
    ] = Field(
        default_factory=dict,
    )

    metadata: dict[str, Any] = Field(
        default_factory=dict,
    )


class SchemaConsistencyResult(BaseModel):
    """Complete Phase 8 schema validation."""

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
    )

    valid: bool = True

    expected_columns: list[str] = Field(
        default_factory=list,
    )

    partitions: dict[
        PartitionName,
        PartitionSchemaValidation,
    ] = Field(
        default_factory=dict,
    )

    blocking_reasons: list[str] = Field(
        default_factory=list,
    )

    metadata: dict[str, Any] = Field(
        default_factory=dict,
    )


@dataclass(slots=True)
class SchemaConsistencyOptions:
    """Options controlling Phase 8 schema validation."""

    enabled: bool = True

    require_exact_columns: bool = True
    require_exact_order: bool = True
    require_compatible_dtypes: bool = True
    require_unique_columns: bool = True

    allow_numeric_width_equivalence: bool = True
    allow_synthetic_float_promotion: bool = True


class SchemaConsistencyValidator:
    """Validate training and protected output schemas."""

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
        expected_columns: list[str],
        source_dtypes: dict[str, str],
    ) -> SchemaConsistencyResult:
        """Validate every Phase 8 output schema."""

        expected = self._normalize_columns(
            expected_columns
        )

        if not self.options.enabled:
            return SchemaConsistencyResult(
                expected_columns=expected,
                metadata={
                    "enabled": False,
                },
            )

        partitions: dict[
            PartitionName,
            PartitionSchemaValidation,
        ] = {}

        blocking_reasons: list[str] = []

        for partition in (
            train,
            validation,
            test,
        ):
            result = self._validate_partition(
                partition=partition,
                expected_columns=expected,
                source_dtypes=source_dtypes,
            )

            partitions[
                partition.name
            ] = result

            if (
                self.options.require_exact_columns
                and not result.columns_match
            ):
                blocking_reasons.append(
                    f"{partition.name.value}: feature columns "
                    "do not match the Phase 7 selected schema."
                )

            if (
                self.options.require_exact_order
                and not result.order_matches
            ):
                blocking_reasons.append(
                    f"{partition.name.value}: feature order "
                    "does not match the Phase 7 schema."
                )

            if (
                self.options.require_compatible_dtypes
                and not result.dtypes_compatible
            ):
                blocking_reasons.append(
                    f"{partition.name.value}: feature dtypes "
                    "are incompatible."
                )

            if (
                self.options.require_unique_columns
                and not result
                .duplicate_columns_absent
            ):
                blocking_reasons.append(
                    f"{partition.name.value}: duplicate "
                    "feature columns are present."
                )

        return SchemaConsistencyResult(
            valid=not blocking_reasons,
            expected_columns=expected,
            partitions=partitions,
            blocking_reasons=(
                blocking_reasons
            ),
            metadata={
                "enabled": True,
                "expected_feature_count": len(
                    expected
                ),
            },
        )

    def _validate_partition(
        self,
        *,
        partition: DatasetPartition,
        expected_columns: list[str],
        source_dtypes: dict[str, str],
    ) -> PartitionSchemaValidation:
        """Validate one partition schema."""

        actual_columns = (
            partition.feature_names()
        )

        expected_set = set(
            expected_columns
        )

        actual_set = set(
            actual_columns
        )

        missing_columns = [
            column
            for column in expected_columns
            if column not in actual_set
        ]

        extra_columns = [
            column
            for column in actual_columns
            if column not in expected_set
        ]

        duplicate_columns = (
            self._duplicates(
                actual_columns
            )
        )

        dtype_mismatches: dict[
            str,
            dict[str, str],
        ] = {}

        for column in expected_columns:
            if column not in actual_set:
                continue

            expected_dtype = (
                source_dtypes.get(
                    column
                )
            )

            actual_dtype = str(
                partition.X[
                    column
                ].dtype
            )

            if expected_dtype is None:
                continue

            if not self._compatible_dtype(
                expected=expected_dtype,
                actual=actual_dtype,
                partition=partition.name,
            ):
                dtype_mismatches[
                    column
                ] = {
                    "expected": expected_dtype,
                    "actual": actual_dtype,
                }

        return PartitionSchemaValidation(
            partition=partition.name,
            columns_match=bool(
                not missing_columns
                and not extra_columns
            ),
            order_matches=bool(
                actual_columns
                == expected_columns
            ),
            dtypes_compatible=bool(
                not dtype_mismatches
            ),
            duplicate_columns_absent=bool(
                not duplicate_columns
            ),
            missing_columns=missing_columns,
            extra_columns=extra_columns,
            duplicate_columns=(
                duplicate_columns
            ),
            dtype_mismatches=(
                dtype_mismatches
            ),
            metadata={
                "feature_count": (
                    partition.feature_count()
                ),
            },
        )

    def _compatible_dtype(
        self,
        *,
        expected: str,
        actual: str,
        partition: PartitionName,
    ) -> bool:
        """Return whether two feature dtypes are compatible."""

        if expected == actual:
            return True

        expected_family = self._dtype_family(
            expected
        )

        actual_family = self._dtype_family(
            actual
        )

        if (
            self.options
            .allow_numeric_width_equivalence
            and expected_family
            == actual_family
            and expected_family
            in {
                "integer",
                "floating",
            }
        ):
            return True

        return bool(
            self.options
            .allow_synthetic_float_promotion
            and partition
            == PartitionName.TRAIN
            and expected_family
            == "integer"
            and actual_family
            == "floating"
        )

    @staticmethod
    def _dtype_family(
        dtype: str,
    ) -> str:
        """Return a normalized dtype family."""

        normalized = dtype.casefold()

        if (
            normalized.startswith("int")
            or normalized.startswith("uint")
        ):
            return "integer"

        if normalized.startswith("float"):
            return "floating"

        if normalized in {
            "bool",
            "boolean",
        }:
            return "boolean"

        return normalized

    @staticmethod
    def _normalize_columns(
        columns: list[str],
    ) -> list[object]:
        """Normalize expected feature columns."""

        normalized = [
            str(column).strip()
            for column in columns
        ]

        if not normalized:
            raise ValueError(
                "expected_columns cannot be empty."
            )

        if any(
            not column
            for column in normalized
        ):
            raise ValueError(
                "expected_columns cannot contain empty names."
            )

        if len(normalized) != len(
            set(normalized)
        ):
            raise ValueError(
                "expected_columns contains duplicate names."
            )

        return normalized

    @staticmethod
    def _duplicates(
        columns: list[str],
    ) -> list[object]:
        """Return duplicate column names in source order."""

        seen: set[str] = set()
        duplicates: list[str] = []

        for column in columns:
            if (
                column in seen
                and column not in duplicates
            ):
                duplicates.append(
                    column
                )

            seen.add(
                column
            )

        return duplicates
