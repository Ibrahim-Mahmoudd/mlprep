"""Selected-schema consistency validation for Phase 7."""

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
    """One selected-schema mismatch."""

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
    )

    reference_partition: PartitionName
    candidate_partition: PartitionName

    missing_columns: list[str] = Field(
        default_factory=list,
    )

    extra_columns: list[str] = Field(
        default_factory=list,
    )

    order_mismatch: bool = False
    expected_schema_mismatch: bool = False

    dtype_mismatches: dict[
        str,
        dict[str, str],
    ] = Field(
        default_factory=dict,
    )

    duplicate_columns: list[str] = Field(
        default_factory=list,
    )

    blocks_readiness: bool = True

    metadata: dict[str, Any] = Field(
        default_factory=dict,
    )


class SchemaConsistencyResult(BaseModel):
    """Complete selected-schema validation output."""

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
    )

    valid: bool = True

    expected_columns: list[str] = Field(
        default_factory=list,
    )

    reference_columns: list[str] = Field(
        default_factory=list,
    )

    mismatches: list[
        SchemaMismatch
    ] = Field(
        default_factory=list,
    )

    feature_count_by_partition: dict[
        PartitionName,
        int,
    ] = Field(
        default_factory=dict,
    )

    metadata: dict[str, Any] = Field(
        default_factory=dict,
    )


@dataclass(slots=True)
class SchemaConsistencyOptions:
    """Options controlling selected-schema validation."""

    enabled: bool = True

    require_exact_selected_schema: bool = True
    require_identical_columns: bool = True
    require_identical_order: bool = True
    require_compatible_dtypes: bool = True
    require_unique_column_names: bool = True

    allow_numeric_width_equivalence: bool = True
    allow_nullable_dtype_equivalence: bool = True


class SchemaConsistencyValidator:
    """Validate selected schemas across all partitions."""

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
        selected_features: list[str],
    ) -> SchemaConsistencyResult:
        """Validate selected feature schemas."""

        if not self.options.enabled:
            return SchemaConsistencyResult(
                metadata={
                    "enabled": False,
                },
            )

        expected_columns = self._normalize_columns(
            selected_features
        )

        mismatches: list[
            SchemaMismatch
        ] = []

        training_mismatch = self._validate_training(
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
            expected_columns=expected_columns,
            reference_columns=(
                train.feature_names()
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

    def _validate_training(
        self,
        *,
        train: DatasetPartition,
        expected_columns: list[str],
    ) -> SchemaMismatch | None:
        """Validate the selected training schema."""

        training_columns = (
            train.feature_names()
        )

        duplicate_columns = self._duplicates(
            training_columns
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
                .require_exact_selected_schema
                and expected_schema_mismatch
            )
        )

        if not (
            duplicate_columns
            or expected_schema_mismatch
        ):
            return None

        return SchemaMismatch(
            reference_partition=(
                PartitionName.TRAIN
            ),
            candidate_partition=(
                PartitionName.TRAIN
            ),
            duplicate_columns=duplicate_columns,
            expected_schema_mismatch=(
                expected_schema_mismatch
            ),
            blocks_readiness=(
                blocks_readiness
            ),
            metadata={
                "training_schema_validation": True,
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

        expected_schema_mismatch = bool(
            candidate_columns
            != expected_columns
        )

        duplicate_columns = self._duplicates(
            candidate_columns
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
                .require_exact_selected_schema
                and expected_schema_mismatch
            )
        )

        has_mismatch = bool(
            missing_columns
            or extra_columns
            or order_mismatch
            or expected_schema_mismatch
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
            expected_schema_mismatch=(
                expected_schema_mismatch
            ),
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
        *,
        reference_dtype: str,
        candidate_dtype: str,
    ) -> bool:
        """Return whether two feature dtypes are compatible."""

        if reference_dtype == candidate_dtype:
            return True

        reference_family = self._dtype_family(
            reference_dtype
        )

        candidate_family = self._dtype_family(
            candidate_dtype
        )

        if (
            self.options
            .allow_numeric_width_equivalence
            and reference_family
            == candidate_family
            and reference_family
            in {
                "integer",
                "floating",
            }
        ):
            return True

        return bool(
            self.options
            .allow_nullable_dtype_equivalence
            and reference_family
            == candidate_family
        )

    @staticmethod
    def _dtype_family(
        dtype: str,
    ) -> str:
        """Normalize related dtype names."""

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
    ) -> list[str]:
        """Normalize and validate selected feature names."""

        normalized = [
            str(column).strip()
            for column in columns
        ]

        if not normalized:
            raise ValueError(
                "selected_features cannot be empty."
            )

        if any(
            not column
            for column in normalized
        ):
            raise ValueError(
                "Feature names cannot contain empty names."
            )

        if len(normalized) != len(
            set(normalized)
        ):
            raise ValueError(
                "selected_features contains duplicate names."
            )

        return normalized

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
