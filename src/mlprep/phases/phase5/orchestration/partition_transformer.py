"""Partition-level outlier transformation for Phase 5."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from mlprep.phases.phase2.output_governance.metadata import (
    PartitionName,
)
from mlprep.phases.phase2.output_governance.result import (
    DatasetPartition,
)
from mlprep.phases.phase5.contract.treatment_plan import (
    TreatmentPlan,
)
from mlprep.phases.phase5.detection.detector_engine import (
    DetectionResult,
    DetectorEngine,
)
from mlprep.phases.phase5.orchestration.fit_context import (
    OutlierFitContext,
)
from mlprep.phases.phase5.strategies.base import (
    BaseTreatmentStrategy,
)
from mlprep.phases.phase5.strategies.indicator_strategy import (
    IndicatorStrategy,
    OutlierIndicatorMetadata,
)


class ColumnTreatmentMetadata(BaseModel):
    """Treatment metadata for one feature and partition."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    column: str
    partition: PartitionName
    detector: str
    treatment: str
    row_count: int = Field(default=0, ge=0)
    outlier_count: int = Field(default=0, ge=0)
    outlier_percentage: float = Field(default=0.0, ge=0.0, le=100.0)
    changed_value_count: int = Field(default=0, ge=0)
    remaining_outlier_count: int = Field(default=0, ge=0)
    missing_before: int = Field(default=0, ge=0)
    missing_after: int = Field(default=0, ge=0)
    nonfinite_before: int = Field(default=0, ge=0)
    nonfinite_after: int = Field(default=0, ge=0)
    lower_bound: float | None = None
    upper_bound: float | None = None
    input_dtype: str
    output_dtype: str
    transformed: bool = False
    skipped: bool = False
    skip_reason: str | None = None
    indicator_column: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class PartitionTreatmentMetadata(BaseModel):
    """Complete treatment metadata for one partition."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    partition: PartitionName
    row_count_before: int = Field(default=0, ge=0)
    row_count_after: int = Field(default=0, ge=0)
    feature_count_before: int = Field(default=0, ge=0)
    feature_count_after: int = Field(default=0, ge=0)
    detected_outlier_count: int = Field(default=0, ge=0)
    changed_value_count: int = Field(default=0, ge=0)
    remaining_outlier_count: int = Field(default=0, ge=0)
    missing_before: int = Field(default=0, ge=0)
    missing_after: int = Field(default=0, ge=0)
    generated_indicator_columns: list[str] = Field(default_factory=list)
    columns: dict[str, ColumnTreatmentMetadata] = Field(default_factory=dict)
    indicators: dict[str, OutlierIndicatorMetadata] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


@dataclass(slots=True)
class PartitionTransformerOptions:
    """Options controlling partition outlier transformation."""

    copy_partition: bool = True
    add_outlier_indicators: bool = True
    preserve_target: bool = True
    preserve_row_ids: bool = True
    preserve_row_count: bool = True
    preserve_row_order: bool = True
    enforce_partition_integrity: bool = True


class PartitionTransformer:
    """Transform feature partitions with training-fitted state."""

    component_name = "partition_transformer"

    def __init__(
        self,
        options: PartitionTransformerOptions | None = None,
    ) -> None:
        """Initialize the transformer."""

        self.options = options or PartitionTransformerOptions()
        self.indicator_strategy = IndicatorStrategy()

    def transform(
        self,
        *,
        partition: DatasetPartition,
        context: OutlierFitContext,
        plan: TreatmentPlan,
        detector_engine: DetectorEngine,
        strategies: dict[str, BaseTreatmentStrategy],
    ) -> tuple[DatasetPartition, PartitionTreatmentMetadata, DetectionResult]:
        """Transform one partition without fitting any state."""

        context.validate()
        context.assert_transform_allowed()
        if context.partition != partition.name:
            raise ValueError("Execution context partition does not match input.")
        if context.detector_fitting_allowed or context.treatment_fitting_allowed:
            raise PermissionError("Partition transformation cannot fit state.")

        source = partition.copy(deep=True) if self.options.copy_partition else partition
        original_x = source.X.copy(deep=True)
        original_y = source.y.copy(deep=True)
        original_row_ids = source.row_ids.copy(deep=True)
        detection = detector_engine.detect(
            frame=source.X,
            partition=partition.name,
        )
        transformed_x = source.X.copy(deep=True)
        metadata = PartitionTreatmentMetadata(
            partition=partition.name,
            row_count_before=len(source.X),
            row_count_after=len(source.X),
            feature_count_before=len(source.X.columns),
            detected_outlier_count=detection.total_outlier_count,
            missing_before=int(source.X.isna().sum().sum()),
        )

        for column, entry in plan.entries.items():
            if column not in transformed_x.columns:
                raise KeyError(f"Planned feature {column!r} is missing.")
            if column not in strategies:
                raise KeyError(f"No fitted treatment exists for {column!r}.")
            mask = detection.mask(column)
            before = transformed_x[column].copy(deep=True)
            strategy = strategies[column]
            transformed = strategy.transform(
                series=before,
                outlier_mask=mask,
                mode=context.execution_mode,
            )
            if len(transformed) != len(before) or not transformed.index.equals(
                before.index
            ):
                raise ValueError(
                    f"Treatment for {column!r} changed row count or index."
                )
            transformed_x[column] = transformed
            changed = ~(before.eq(transformed) | (before.isna() & transformed.isna()))
            changed_count = int(changed.sum())
            detection_summary = detection.summaries[column]
            indicator_column: str | None = None
            if self.options.add_outlier_indicators and entry.add_indicator:
                indicator_column = plan.indicator_columns.get(column)
                transformed_x, indicator_metadata = self.indicator_strategy.create(
                    frame=transformed_x,
                    source_column=column,
                    outlier_mask=mask,
                    indicator_column=indicator_column,
                )
                indicator_column = indicator_metadata.indicator_column
                metadata.indicators[column] = indicator_metadata
                metadata.generated_indicator_columns.append(indicator_column)

            metadata.columns[column] = ColumnTreatmentMetadata(
                column=column,
                partition=partition.name,
                detector=entry.detector.value,
                treatment=entry.treatment.value,
                row_count=len(before),
                outlier_count=int(mask.sum()),
                outlier_percentage=(
                    float(mask.mean() * 100.0) if len(mask) else 0.0
                ),
                changed_value_count=changed_count,
                remaining_outlier_count=0,
                missing_before=int(before.isna().sum()),
                missing_after=int(transformed.isna().sum()),
                nonfinite_before=self._nonfinite_count(before),
                nonfinite_after=self._nonfinite_count(transformed),
                lower_bound=detection_summary.lower_bound,
                upper_bound=detection_summary.upper_bound,
                input_dtype=str(before.dtype),
                output_dtype=str(transformed.dtype),
                transformed=changed_count > 0,
                indicator_column=indicator_column,
            )
            metadata.changed_value_count += changed_count
            metadata.remaining_outlier_count += (
                metadata.columns[column].remaining_outlier_count
            )

        remaining_detection = detector_engine.detect(
            frame=transformed_x,
            partition=partition.name,
        )
        metadata.remaining_outlier_count = 0
        for column, column_metadata in metadata.columns.items():
            remaining_count = int(
                remaining_detection.mask(column).sum()
            )
            column_metadata.remaining_outlier_count = remaining_count
            metadata.remaining_outlier_count += remaining_count

        output = DatasetPartition(
            name=partition.name,
            X=transformed_x,
            y=source.y.copy(deep=True),
            row_ids=source.row_ids.copy(deep=True),
            metadata=source.metadata.model_copy(deep=True),
        )
        metadata.feature_count_after = len(output.X.columns)
        metadata.missing_after = int(output.X.isna().sum().sum())
        self._validate_integrity(
            original_x=original_x,
            original_y=original_y,
            original_row_ids=original_row_ids,
            output=output,
        )
        return output, metadata, detection

    @staticmethod
    def _nonfinite_count(series: pd.Series) -> int:
        """Count infinite numeric values without coercion failures."""

        numeric = pd.to_numeric(series, errors="coerce")
        return int(
            (numeric.notna() & ~np.isfinite(numeric.astype(float))).sum()
        )

    def _validate_integrity(
        self,
        *,
        original_x: pd.DataFrame,
        original_y: pd.Series,
        original_row_ids: pd.Series,
        output: DatasetPartition,
    ) -> None:
        """Verify row and target identity after transformation."""

        if self.options.enforce_partition_integrity:
            if self.options.preserve_row_count and len(output.X) != len(original_x):
                raise ValueError("Outlier transformation changed the row count.")
            if self.options.preserve_row_order and not output.X.index.equals(
                original_x.index
            ):
                raise ValueError("Outlier transformation changed row order.")
            if self.options.preserve_target and not output.y.equals(original_y):
                raise ValueError("Outlier transformation changed target values.")
            if self.options.preserve_row_ids and not output.row_ids.equals(
                original_row_ids
            ):
                raise ValueError("Outlier transformation changed row IDs.")
