"""Synthetic training-row integrity validation for Phase 8."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from mlprep.phases.phase2.output_governance.result import (
    DatasetPartition,
)
from mlprep.phases.phase8.lineage import (
    RowLineageState,
)
from mlprep.phases.phase8.samplers import (
    ResampledRowType,
)


class SyntheticIntegrityResult(BaseModel):
    """Complete synthetic and duplicate integrity validation."""

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
    )

    valid: bool = True

    synthetic_row_count: int = Field(
        default=0,
        ge=0,
    )

    duplicate_row_count: int = Field(
        default=0,
        ge=0,
    )

    synthetic_ids_unique: bool = True
    synthetic_ids_nonmissing: bool = True
    synthetic_ids_collision_free: bool = True

    lineage_complete: bool = True

    synthetic_targets_valid: bool = True
    synthetic_features_numeric: bool = True
    synthetic_features_complete: bool = True
    synthetic_features_finite: bool = True

    expansion_policy_satisfied: bool = True

    invalid_synthetic_row_ids: list[str] = Field(
        default_factory=list,
    )

    invalid_target_row_ids: list[str] = Field(
        default_factory=list,
    )

    missing_lineage_row_ids: list[str] = Field(
        default_factory=list,
    )

    nonnumeric_columns: list[str] = Field(
        default_factory=list,
    )

    missing_value_columns: list[str] = Field(
        default_factory=list,
    )

    nonfinite_columns: list[str] = Field(
        default_factory=list,
    )

    blocking_reasons: list[str] = Field(
        default_factory=list,
    )

    metadata: dict[str, Any] = Field(
        default_factory=dict,
    )


@dataclass(slots=True)
class SyntheticIntegrityOptions:
    """Options controlling synthetic-row validation."""

    enabled: bool = True

    require_unique_synthetic_ids: bool = True
    require_nonmissing_synthetic_ids: bool = True
    reject_original_id_collisions: bool = True

    require_complete_lineage: bool = True

    require_numeric_features: bool = True
    require_complete_features: bool = True
    require_finite_features: bool = True

    require_valid_target_labels: bool = True

    maximum_synthetic_expansion_ratio: float | None = None

    def __post_init__(self) -> None:
        """Validate synthetic-integrity options."""

        if (
            self.maximum_synthetic_expansion_ratio
            is not None
            and self.maximum_synthetic_expansion_ratio
            < 0.0
        ):
            raise ValueError(
                "maximum_synthetic_expansion_ratio cannot "
                "be negative."
            )


class SyntheticIntegrityValidator:
    """Validate generated training rows and lineage."""

    component_name = "synthetic_integrity"

    def __init__(
        self,
        options: SyntheticIntegrityOptions | None = None,
    ) -> None:
        """Initialize the validator."""

        self.options = (
            options
            or SyntheticIntegrityOptions()
        )

    def validate(
        self,
        *,
        source_train: DatasetPartition,
        output_train: DatasetPartition,
        lineage: RowLineageState,
        valid_class_values: list[Any],
    ) -> SyntheticIntegrityResult:
        """Validate synthetic and duplicate output rows."""

        if not self.options.enabled:
            return SyntheticIntegrityResult(
                metadata={
                    "enabled": False,
                },
            )

        synthetic_entries = [
            entry
            for entry in lineage.entries
            if entry.row_type
            == ResampledRowType.SYNTHETIC
        ]

        duplicate_entries = [
            entry
            for entry in lineage.entries
            if entry.row_type
            == ResampledRowType.DUPLICATE
        ]

        synthetic_ids = [
            entry.output_row_id
            for entry in synthetic_entries
        ]

        source_ids = {
            str(value)
            for value
            in source_train.row_ids.tolist()
        }

        synthetic_ids_unique = bool(
            len(
                synthetic_ids
            )
            == len(
                set(
                    synthetic_ids
                )
            )
        )

        synthetic_ids_nonmissing = all(
            bool(
                row_id.strip()
            )
            for row_id in synthetic_ids
        )

        collisions = sorted(
            set(
                synthetic_ids
            ).intersection(
                source_ids
            )
        )

        collision_free = bool(
            not collisions
        )

        output_ids = {
            str(value)
            for value
            in output_train.row_ids.tolist()
        }

        missing_lineage_ids = [
            row_id
            for row_id in synthetic_ids
            if row_id not in output_ids
        ]

        lineage_complete = bool(
            not missing_lineage_ids
            and lineage.complete_output_coverage
        )

        synthetic_positions = [
            entry.output_position
            for entry in synthetic_entries
        ]

        synthetic_frame = (
            output_train.X.iloc[
                synthetic_positions
            ]
            if synthetic_positions
            else output_train.X.iloc[
                0:0
            ]
        )

        nonnumeric_columns = [
            str(column)
            for column
            in synthetic_frame.columns
            if not pd.api.types.is_numeric_dtype(
                synthetic_frame[
                    column
                ].dtype
            )
        ]

        missing_columns = [
            str(column)
            for column
            in synthetic_frame.columns
            if synthetic_frame[
                column
            ].isna().any()
        ]

        nonfinite_columns: list[str] = []

        for column in synthetic_frame.columns:
            if column in nonnumeric_columns:
                continue

            numeric = pd.to_numeric(
                synthetic_frame[
                    column
                ],
                errors="coerce",
            )

            mask = (
                numeric.notna()
                & ~np.isfinite(
                    numeric.astype(
                        float
                    )
                )
            )

            if bool(
                mask.any()
            ):
                nonfinite_columns.append(
                    str(
                        column
                    )
                )

        invalid_target_ids: list[str] = []

        for entry in synthetic_entries:
            target_value = output_train.y.iloc[
                entry.output_position
            ]

            if not any(
                self._values_equal(
                    target_value,
                    valid_value,
                )
                for valid_value
                in valid_class_values
            ):
                invalid_target_ids.append(
                    entry.output_row_id
                )

        source_count = source_train.row_count()

        synthetic_ratio = (
            len(
                synthetic_entries
            )
            / source_count
            if source_count > 0
            else 0.0
        )

        expansion_satisfied = bool(
            self.options
            .maximum_synthetic_expansion_ratio
            is None
            or synthetic_ratio
            <= self.options
            .maximum_synthetic_expansion_ratio
        )

        blocking_reasons: list[str] = []

        if (
            self.options.require_unique_synthetic_ids
            and not synthetic_ids_unique
        ):
            blocking_reasons.append(
                "Synthetic row IDs are not unique."
            )

        if (
            self.options.require_nonmissing_synthetic_ids
            and not synthetic_ids_nonmissing
        ):
            blocking_reasons.append(
                "Synthetic row IDs contain empty values."
            )

        if (
            self.options.reject_original_id_collisions
            and not collision_free
        ):
            blocking_reasons.append(
                "Synthetic row IDs collide with original "
                "training row IDs."
            )

        if (
            self.options.require_complete_lineage
            and not lineage_complete
        ):
            blocking_reasons.append(
                "Synthetic row lineage is incomplete."
            )

        if (
            self.options.require_numeric_features
            and nonnumeric_columns
        ):
            blocking_reasons.append(
                "Synthetic rows contain nonnumeric features."
            )

        if (
            self.options.require_complete_features
            and missing_columns
        ):
            blocking_reasons.append(
                "Synthetic rows contain missing feature values."
            )

        if (
            self.options.require_finite_features
            and nonfinite_columns
        ):
            blocking_reasons.append(
                "Synthetic rows contain nonfinite feature values."
            )

        if (
            self.options.require_valid_target_labels
            and invalid_target_ids
        ):
            blocking_reasons.append(
                "Synthetic rows contain invalid target labels."
            )

        if not expansion_satisfied:
            blocking_reasons.append(
                "Synthetic expansion exceeds the configured "
                "maximum."
            )

        return SyntheticIntegrityResult(
            valid=not blocking_reasons,
            synthetic_row_count=len(
                synthetic_entries
            ),
            duplicate_row_count=len(
                duplicate_entries
            ),
            synthetic_ids_unique=(
                synthetic_ids_unique
            ),
            synthetic_ids_nonmissing=(
                synthetic_ids_nonmissing
            ),
            synthetic_ids_collision_free=(
                collision_free
            ),
            lineage_complete=(
                lineage_complete
            ),
            synthetic_targets_valid=bool(
                not invalid_target_ids
            ),
            synthetic_features_numeric=bool(
                not nonnumeric_columns
            ),
            synthetic_features_complete=bool(
                not missing_columns
            ),
            synthetic_features_finite=bool(
                not nonfinite_columns
            ),
            expansion_policy_satisfied=(
                expansion_satisfied
            ),
            invalid_synthetic_row_ids=(
                collisions
            ),
            invalid_target_row_ids=(
                invalid_target_ids
            ),
            missing_lineage_row_ids=(
                missing_lineage_ids
            ),
            nonnumeric_columns=(
                nonnumeric_columns
            ),
            missing_value_columns=(
                missing_columns
            ),
            nonfinite_columns=(
                nonfinite_columns
            ),
            blocking_reasons=(
                blocking_reasons
            ),
            metadata={
                "enabled": True,
                "synthetic_expansion_ratio": (
                    synthetic_ratio
                ),
            },
        )

    @staticmethod
    def _values_equal(
        left: object,
        right: object,
    ) -> bool:
        """Safely compare two target values."""

        try:
            comparison = (
                left == right
            )

        except Exception:
            return False

        if isinstance(
            comparison,
            (bool, np.bool_),
        ):
            return bool(
                comparison
            )

        return False
