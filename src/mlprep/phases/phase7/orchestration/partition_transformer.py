"""Partition-level selected-schema application for Phase 7."""

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
from mlprep.phases.phase7.orchestration.fit_context import (
    SelectionFitContext,
)


class SelectedFeatureMetadata(BaseModel):
    """Selection metadata for one output feature."""

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
    )

    column: str

    source_position: int = Field(
        ge=0,
    )

    selected_position: int = Field(
        ge=0,
    )

    dtype: str

    required: bool = False
    protected: bool = False

    selection_reasons: list[str] = Field(
        default_factory=list,
    )

    metadata: dict[str, Any] = Field(
        default_factory=dict,
    )


class PartitionSelectionMetadata(BaseModel):
    """Complete selected-schema metadata for one partition."""

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
    )

    partition: PartitionName

    row_count_before: int = Field(
        default=0,
        ge=0,
    )

    row_count_after: int = Field(
        default=0,
        ge=0,
    )

    feature_count_before: int = Field(
        default=0,
        ge=0,
    )

    feature_count_after: int = Field(
        default=0,
        ge=0,
    )

    selected_columns: list[str] = Field(
        default_factory=list,
    )

    removed_columns: list[str] = Field(
        default_factory=list,
    )

    features: dict[
        str,
        SelectedFeatureMetadata,
    ] = Field(
        default_factory=dict,
    )

    feature_values_unchanged: bool = True
    target_unchanged: bool = True
    row_ids_unchanged: bool = True
    row_order_unchanged: bool = True

    metadata: dict[str, Any] = Field(
        default_factory=dict,
    )


@dataclass(slots=True)
class PartitionTransformerOptions:
    """Options controlling selected-schema application."""

    copy_partition: bool = True

    preserve_feature_values: bool = True
    preserve_feature_dtypes: bool = True

    preserve_target: bool = True
    preserve_row_ids: bool = True

    preserve_row_count: bool = True
    preserve_row_order: bool = True

    require_exact_selected_schema: bool = True
    require_unique_selected_columns: bool = True

    reject_unexpected_source_columns: bool = False


class PartitionTransformer:
    """Select and order fitted features without modifying values."""

    component_name = "partition_transformer"

    def __init__(
        self,
        options: PartitionTransformerOptions | None = None,
    ) -> None:
        """Initialize the partition transformer."""

        self.options = (
            options
            or PartitionTransformerOptions()
        )

    def transform(
        self,
        *,
        partition: DatasetPartition,
        context: SelectionFitContext,
        source_feature_order: list[str],
        selected_features: list[str],
        required_features: list[str] | None = None,
        protected_features: list[str] | None = None,
        selection_reasons: dict[
            str,
            list[str],
        ] | None = None,
    ) -> tuple[
        DatasetPartition,
        PartitionSelectionMetadata,
    ]:
        """Apply the fitted selected schema to one partition."""

        context.validate_for_transform()

        if partition.name != context.partition:
            raise ValueError(
                "The partition does not match the "
                "selection context."
            )

        normalized_source_order = self._normalize_columns(
            source_feature_order,
            name="source_feature_order",
        )

        normalized_selected = self._normalize_columns(
            selected_features,
            name="selected_features",
        )

        if (
            self.options.require_unique_selected_columns
            and len(normalized_selected)
            != len(
                set(normalized_selected)
            )
        ):
            raise ValueError(
                "Selected feature names must be unique."
            )

        self._validate_selected_schema(
            partition=partition,
            source_feature_order=(
                normalized_source_order
            ),
            selected_features=(
                normalized_selected
            ),
        )

        required = set(
            required_features or []
        )

        protected = set(
            protected_features or []
        )

        reasons = {
            str(column): list(
                column_reasons
            )
            for column, column_reasons
            in (
                selection_reasons or {}
            ).items()
        }

        source_partition = (
            partition.copy(
                deep=True
            )
            if self.options.copy_partition
            else partition
        )

        source_frame = source_partition.X.copy(
            deep=True
        )

        source_target = source_partition.y.copy(
            deep=True
        )

        source_row_ids = (
            source_partition.row_ids.copy(
                deep=True
            )
        )

        original_index = source_frame.index.copy()

        selected_frame = source_frame.loc[
            :,
            normalized_selected,
        ].copy(
            deep=True
        )

        if self.options.preserve_feature_values:
            self._validate_feature_values(
                source_frame=source_frame,
                selected_frame=selected_frame,
                selected_features=(
                    normalized_selected
                ),
            )

        if self.options.preserve_feature_dtypes:
            self._validate_feature_dtypes(
                source_frame=source_frame,
                selected_frame=selected_frame,
                selected_features=(
                    normalized_selected
                ),
            )

        transformed_partition = DatasetPartition(
            name=partition.name,
            X=selected_frame,
            y=source_target,
            row_ids=source_row_ids,
            metadata=(
                source_partition
                .metadata
                .model_copy(
                    deep=True
                )
            ),
        )

        self._validate_partition_integrity(
            source=source_partition,
            transformed=transformed_partition,
            original_index=original_index,
            selected_features=(
                normalized_selected
            ),
        )

        source_position = {
            column: position
            for position, column
            in enumerate(
                normalized_source_order
            )
        }

        selected_metadata = {
            column: SelectedFeatureMetadata(
                column=column,
                source_position=(
                    source_position[
                        column
                    ]
                ),
                selected_position=position,
                dtype=str(
                    selected_frame[
                        column
                    ].dtype
                ),
                required=(
                    column in required
                ),
                protected=(
                    column in protected
                ),
                selection_reasons=list(
                    reasons.get(
                        column,
                        [],
                    )
                ),
            )
            for position, column
            in enumerate(
                normalized_selected
            )
        }

        selected_set = set(
            normalized_selected
        )

        removed_columns = [
            column
            for column in normalized_source_order
            if column not in selected_set
        ]

        metadata = PartitionSelectionMetadata(
            partition=partition.name,
            row_count_before=len(
                source_frame
            ),
            row_count_after=len(
                selected_frame
            ),
            feature_count_before=len(
                source_frame.columns
            ),
            feature_count_after=len(
                selected_frame.columns
            ),
            selected_columns=(
                normalized_selected
            ),
            removed_columns=(
                removed_columns
            ),
            features=selected_metadata,
            feature_values_unchanged=True,
            target_unchanged=True,
            row_ids_unchanged=True,
            row_order_unchanged=True,
            metadata={
                "source_schema_matches_fitted_schema": (
                    [
                        str(column)
                        for column
                        in source_frame.columns
                    ]
                    == normalized_source_order
                ),
            },
        )

        return (
            transformed_partition,
            metadata,
        )

    def _validate_selected_schema(
        self,
        *,
        partition: DatasetPartition,
        source_feature_order: list[str],
        selected_features: list[str],
    ) -> None:
        """Validate source and selected schemas."""

        actual_columns = [
            str(column)
            for column in partition.X.columns
        ]

        actual_set = set(
            actual_columns
        )

        source_set = set(
            source_feature_order
        )

        if self.options.reject_unexpected_source_columns:
            unexpected = sorted(
                actual_set.difference(
                    source_set
                )
            )

            if unexpected:
                raise ValueError(
                    "The partition contains unexpected "
                    f"source features: {unexpected}"
                )

        if self.options.require_exact_selected_schema:
            missing_source = sorted(
                source_set.difference(
                    actual_set
                )
            )

            if missing_source:
                raise ValueError(
                    "The partition is missing fitted source "
                    f"features: {missing_source}"
                )

            if actual_columns != source_feature_order:
                raise ValueError(
                    "The partition source-feature order differs "
                    "from the fitted Phase 7 schema."
                )

        missing_selected = [
            column
            for column in selected_features
            if column not in actual_set
        ]

        if missing_selected:
            raise KeyError(
                "Selected features are missing from the "
                f"partition: {missing_selected}"
            )

    @staticmethod
    def _validate_feature_values(
        *,
        source_frame: pd.DataFrame,
        selected_frame: pd.DataFrame,
        selected_features: list[str],
    ) -> None:
        """Ensure selected feature values were not modified."""

        for column in selected_features:
            if not source_frame[
                column
            ].equals(
                selected_frame[
                    column
                ]
            ):
                raise ValueError(
                    f"Selected feature {column!r} changed "
                    "during schema application."
                )

    @staticmethod
    def _validate_feature_dtypes(
        *,
        source_frame: pd.DataFrame,
        selected_frame: pd.DataFrame,
        selected_features: list[str],
    ) -> None:
        """Ensure selected feature dtypes were not modified."""

        changed_dtypes = [
            column
            for column in selected_features
            if str(
                source_frame[
                    column
                ].dtype
            )
            != str(
                selected_frame[
                    column
                ].dtype
            )
        ]

        if changed_dtypes:
            raise TypeError(
                "Selected feature dtypes changed during "
                f"schema application: {changed_dtypes}"
            )

    def _validate_partition_integrity(
        self,
        *,
        source: DatasetPartition,
        transformed: DatasetPartition,
        original_index: pd.Index,
        selected_features: list[str],
    ) -> None:
        """Validate protected partition content."""

        if (
            self.options.preserve_row_count
            and source.row_count()
            != transformed.row_count()
        ):
            raise ValueError(
                "Feature selection changed the row count."
            )

        if (
            self.options.preserve_row_order
            and not transformed.X.index.equals(
                original_index
            )
        ):
            raise ValueError(
                "Feature selection changed the row order."
            )

        if (
            self.options.preserve_target
            and not source.y.equals(
                transformed.y
            )
        ):
            raise ValueError(
                "Feature selection changed target values."
            )

        if (
            self.options.preserve_row_ids
            and not source.row_ids.equals(
                transformed.row_ids
            )
        ):
            raise ValueError(
                "Feature selection changed row IDs."
            )

        if transformed.feature_names() != selected_features:
            raise ValueError(
                "The transformed feature schema differs "
                "from the fitted selected schema."
            )

    @staticmethod
    def _normalize_columns(
        columns: list[str],
        *,
        name: str,
    ) -> list[object]:
        """Normalize one list of feature names."""

        normalized = [
            str(column).strip()
            for column in columns
        ]

        if any(
            not column
            for column in normalized
        ):
            raise ValueError(
                f"{name} cannot contain empty names."
            )

        if len(normalized) != len(
            set(normalized)
        ):
            raise ValueError(
                f"{name} contains duplicate names."
            )

        return normalized