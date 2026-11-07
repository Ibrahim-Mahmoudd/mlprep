"""Partition transformation using fitted Phase 4 imputers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from mlprep.phases.phase2.output_governance.metadata import PartitionName
from mlprep.phases.phase2.output_governance.result import DatasetPartition
from mlprep.phases.phase4.contract.imputation_plan import (
    ImputationMethodName,
    ImputationPlan,
    ImputationPlanEntry,
    PlanEntryStatus,
)
from mlprep.phases.phase4.indicators.missing_indicator import (
    MissingIndicatorGenerator,
    MissingIndicatorResult,
)
from mlprep.phases.phase4.orchestration.fit_context import FitContext
from mlprep.phases.phase4.strategies.base import BaseImputer
from mlprep.phases.phase4.strategies.group_imputer import GroupImputer
from mlprep.phases.phase4.strategies.model_imputer import ModelImputer
from mlprep.phases.phase4.strategies.temporal_imputer import TemporalImputer


class ColumnTransformationMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    column: str
    partition: PartitionName
    method: str
    missing_before: int = Field(default=0, ge=0)
    missing_after: int = Field(default=0, ge=0)
    imputed_count: int = Field(default=0, ge=0)
    input_dtype: str
    output_dtype: str
    transformed: bool = False
    skipped: bool = False
    skip_reason: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class PartitionTransformationMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    partition: PartitionName
    row_count_before: int = Field(default=0, ge=0)
    row_count_after: int = Field(default=0, ge=0)
    feature_count_before: int = Field(default=0, ge=0)
    feature_count_after: int = Field(default=0, ge=0)
    missing_before: int = Field(default=0, ge=0)
    missing_after: int = Field(default=0, ge=0)
    total_imputed_count: int = Field(default=0, ge=0)
    generated_indicator_columns: list[str] = Field(default_factory=list)
    dropped_columns: list[str] = Field(default_factory=list)
    columns: dict[str, ColumnTransformationMetadata] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


@dataclass(slots=True)
class PartitionTransformerOptions:
    copy_partition: bool = True
    apply_missing_indicators: bool = True
    allow_drop_feature_entries: bool = False
    allow_unresolved_missing_values: bool = False
    enforce_feature_order: bool = True
    preserve_target: bool = True
    preserve_row_ids: bool = True


class PartitionTransformer:
    """Apply fitted imputers to one partition."""

    component_name = "partition_transformer"

    def __init__(self, options: PartitionTransformerOptions | None = None) -> None:
        self.options = options or PartitionTransformerOptions()

    def transform(
        self,
        *,
        partition: DatasetPartition,
        context: FitContext,
        plan: ImputationPlan,
        imputers: dict[str, BaseImputer],
        indicator_generator: MissingIndicatorGenerator | None = None,
        normalized_frame: pd.DataFrame | None = None,
        original_missing_mask: pd.DataFrame | None = None,
    ) -> tuple[
        DatasetPartition,
        PartitionTransformationMetadata,
        MissingIndicatorResult | None,
    ]:
        context.validate()
        if partition.name != context.partition:
            raise ValueError("Partition does not match transformation context.")

        source_partition = (
            partition.copy(deep=True) if self.options.copy_partition else partition
        )
        frame = (
            source_partition.X.copy(deep=True)
            if normalized_frame is None
            else normalized_frame.copy(deep=True)
        )
        self._validate_frame(source_partition.X, frame)

        initial_columns = [str(column) for column in frame.columns]
        original_target = source_partition.y.copy(deep=True)
        original_row_ids = source_partition.row_ids.copy(deep=True)
        row_count_before = len(frame)
        feature_count_before = len(frame.columns)
        missing_before = int(frame.isna().sum().sum())

        column_metadata: dict[str, ColumnTransformationMetadata] = {}
        dropped_columns: list[str] = []

        for column, entry in plan.entries.items():
            if column not in frame.columns:
                column_metadata[column] = ColumnTransformationMetadata(
                    column=column,
                    partition=partition.name,
                    method=entry.method.value,
                    input_dtype="unavailable",
                    output_dtype="unavailable",
                    skipped=True,
                    skip_reason="Feature does not exist in the partition.",
                )
                continue

            metadata = self._transform_column(
                frame=frame,
                partition=partition.name,
                context=context,
                entry=entry,
                imputers=imputers,
            )
            column_metadata[column] = metadata

            if (
                entry.method == ImputationMethodName.DROP_FEATURE
                and self.options.allow_drop_feature_entries
            ):
                frame = frame.drop(columns=[column])
                dropped_columns.append(column)

        indicator_result = None
        if self.options.apply_missing_indicators and indicator_generator is not None:
            frame, indicator_result = indicator_generator.transform(
                frame=frame,
                partition=partition.name,
                original_missing_mask=original_missing_mask,
            )
            indicator_generator.validate_partition_mapping(indicator_result)

        if self.options.enforce_feature_order:
            expected_columns = [
                column for column in initial_columns if column not in dropped_columns
            ]
            if indicator_generator is not None:
                for indicator in indicator_generator.mapping().values():
                    if indicator not in expected_columns:
                        expected_columns.append(indicator)
            missing = set(expected_columns).difference(frame.columns)
            if missing:
                raise ValueError(f"Missing expected columns: {sorted(missing)}")
            frame = frame.loc[:, expected_columns]

        if (
            not self.options.allow_unresolved_missing_values
            and frame.isna().any().any()
        ):
            unresolved = [
                str(column) for column in frame.columns if frame[column].isna().any()
            ]
            raise ValueError(f"Missing values remain in columns: {unresolved}")

        transformed = DatasetPartition(
            name=partition.name,
            X=frame,
            y=original_target,
            row_ids=original_row_ids,
            metadata=source_partition.metadata.model_copy(deep=True),
        )
        self._validate_integrity(source_partition, transformed)

        metadata = PartitionTransformationMetadata(
            partition=partition.name,
            row_count_before=row_count_before,
            row_count_after=len(frame),
            feature_count_before=feature_count_before,
            feature_count_after=len(frame.columns),
            missing_before=missing_before,
            missing_after=int(frame.isna().sum().sum()),
            total_imputed_count=sum(
                item.imputed_count for item in column_metadata.values()
            ),
            generated_indicator_columns=(
                list(indicator_result.generated_columns)
                if indicator_result is not None
                else []
            ),
            dropped_columns=dropped_columns,
            columns=column_metadata,
        )
        return transformed, metadata, indicator_result

    def _transform_column(
        self,
        *,
        frame: pd.DataFrame,
        partition: PartitionName,
        context: FitContext,
        entry: ImputationPlanEntry,
        imputers: dict[str, BaseImputer],
    ) -> ColumnTransformationMetadata:
        column = entry.column
        series = frame[column]
        input_dtype = str(series.dtype)
        missing_before = int(series.isna().sum())

        if entry.status != PlanEntryStatus.PLANNED:
            return ColumnTransformationMetadata(
                column=column,
                partition=partition,
                method=entry.method.value,
                missing_before=missing_before,
                missing_after=missing_before,
                input_dtype=input_dtype,
                output_dtype=input_dtype,
                skipped=True,
                skip_reason=f"Plan status is {entry.status.value!r}.",
            )

        if entry.method in {ImputationMethodName.NONE, ImputationMethodName.REVIEW}:
            return ColumnTransformationMetadata(
                column=column,
                partition=partition,
                method=entry.method.value,
                missing_before=missing_before,
                missing_after=missing_before,
                input_dtype=input_dtype,
                output_dtype=input_dtype,
                skipped=True,
                skip_reason="Plan entry does not execute imputation.",
            )

        if entry.method == ImputationMethodName.DROP_FEATURE:
            return ColumnTransformationMetadata(
                column=column,
                partition=partition,
                method=entry.method.value,
                missing_before=missing_before,
                missing_after=(
                    0
                    if self.options.allow_drop_feature_entries
                    else missing_before
                ),
                input_dtype=input_dtype,
                output_dtype=(
                    "dropped"
                    if self.options.allow_drop_feature_entries
                    else input_dtype
                ),
                transformed=self.options.allow_drop_feature_entries,
                skipped=not self.options.allow_drop_feature_entries,
                skip_reason=(
                    None if self.options.allow_drop_feature_entries
                    else "Feature dropping is disabled."
                ),
            )

        if column not in imputers:
            raise KeyError(f"No fitted imputer exists for {column!r}.")

        imputer = imputers[column]
        if isinstance(imputer, (GroupImputer, TemporalImputer, ModelImputer)):
            transformed = imputer.transform_frame(frame, mode=context.execution_mode)
        else:
            transformed = imputer.transform(series, mode=context.execution_mode)

        frame[column] = transformed
        missing_after = int(transformed.isna().sum())
        return ColumnTransformationMetadata(
            column=column,
            partition=partition,
            method=entry.method.value,
            missing_before=missing_before,
            missing_after=missing_after,
            imputed_count=max(0, missing_before - missing_after),
            input_dtype=input_dtype,
            output_dtype=str(transformed.dtype),
            transformed=True,
        )

    def _validate_integrity(
        self,
        original: DatasetPartition,
        transformed: DatasetPartition,
    ) -> None:
        if original.row_count() != transformed.row_count():
            raise ValueError("Transformation changed row count.")
        if not original.X.index.equals(transformed.X.index):
            raise ValueError("Transformation changed row order.")
        if self.options.preserve_target and not original.y.equals(transformed.y):
            raise ValueError("Transformation changed target values.")
        if self.options.preserve_row_ids and not original.row_ids.equals(
            transformed.row_ids
        ):
            raise ValueError("Transformation changed row IDs.")

    @staticmethod
    def _validate_frame(original: pd.DataFrame, normalized: pd.DataFrame) -> None:
        if not isinstance(normalized, pd.DataFrame):
            raise TypeError("normalized_frame must be a pandas DataFrame.")
        if len(original) != len(normalized):
            raise ValueError("Normalization changed row count.")
        if not original.index.equals(normalized.index):
            raise ValueError("Normalization changed row order.")
        if list(original.columns) != list(normalized.columns):
            raise ValueError("Normalization changed feature schema.")
