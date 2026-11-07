"""Partition-level feature transformation for Phase 6."""

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
from mlprep.phases.phase6.categorical.categorical_pipeline import (
    CategoricalPipeline,
    CategoricalPipelineResult,
)
from mlprep.phases.phase6.contract.transformation_plan import (
    PlanEntryStatus,
    SourceColumnPolicy,
    TransformationPlan,
)
from mlprep.phases.phase6.feature_engineering.boolean_features import (
    BooleanFeatureTransformer,
)
from mlprep.phases.phase6.feature_engineering.datetime_features import (
    DatetimeFeatureExtractor,
)
from mlprep.phases.phase6.feature_engineering.feature_lineage import (
    FeatureLineageTracker,
)
from mlprep.phases.phase6.feature_engineering.text_features import (
    TextFeatureExtractor,
)
from mlprep.phases.phase6.numeric.numeric_pipeline import (
    NumericPipeline,
    NumericPipelineResult,
)
from mlprep.phases.phase6.orchestration.fit_context import (
    TransformationFitContext,
)


class SourceFeatureTransformationMetadata(BaseModel):
    """Transformation metadata for one source feature."""

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
    )

    source_column: str
    partition: PartitionName
    feature_family: str

    input_dtype: str
    output_columns: list[str] = Field(
        default_factory=list
    )

    output_dtypes: dict[
        str,
        str,
    ] = Field(default_factory=dict)

    source_preserved: bool = False
    source_dropped: bool = False

    transformed: bool = False
    skipped: bool = False

    skip_reason: str | None = None

    unknown_category_count: int = Field(
        default=0,
        ge=0,
    )

    unknown_category_percentage: float = Field(
        default=0.0,
        ge=0.0,
        le=100.0,
    )

    metadata: dict[str, Any] = Field(
        default_factory=dict
    )


class PartitionTransformationMetadata(BaseModel):
    """Complete Phase 6 metadata for one partition."""

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

    missing_count_before: int = Field(
        default=0,
        ge=0,
    )

    missing_count_after: int = Field(
        default=0,
        ge=0,
    )

    nonfinite_count_after: int = Field(
        default=0,
        ge=0,
    )

    generated_columns: list[str] = Field(
        default_factory=list
    )

    preserved_source_columns: list[str] = Field(
        default_factory=list
    )

    dropped_source_columns: list[str] = Field(
        default_factory=list
    )

    columns: dict[
        str,
        SourceFeatureTransformationMetadata,
    ] = Field(default_factory=dict)

    metadata: dict[str, Any] = Field(
        default_factory=dict
    )


@dataclass(slots=True)
class PartitionTransformerOptions:
    """Options controlling transformed partitions."""

    copy_partition: bool = True

    preserve_target: bool = True
    preserve_row_ids: bool = True

    preserve_row_count: bool = True
    preserve_row_order: bool = True

    enforce_output_schema: bool = True
    require_unique_output_columns: bool = True

    require_numeric_output: bool = True

    allow_missing_output_values: bool = False
    allow_nonfinite_output_values: bool = False

    include_skipped_source_columns: bool = True


class PartitionTransformer:
    """Assemble all transformed Phase 6 feature groups."""

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
        context: TransformationFitContext,
        plan: TransformationPlan,
        numeric_pipeline: NumericPipeline,
        categorical_pipeline: CategoricalPipeline,
        boolean_transformers: dict[
            str,
            BooleanFeatureTransformer,
        ],
        datetime_extractors: dict[
            str,
            DatetimeFeatureExtractor,
        ],
        text_extractors: dict[
            str,
            TextFeatureExtractor,
        ],
        timedelta_output_columns: dict[
            str,
            str,
        ],
        lineage: FeatureLineageTracker,
        expected_output_columns: list[str],
    ) -> tuple[
        DatasetPartition,
        PartitionTransformationMetadata,
        NumericPipelineResult,
        CategoricalPipelineResult,
    ]:
        """Transform and assemble one partition."""

        context.validate()

        if partition.name != context.partition:
            raise ValueError(
                "The partition does not match the "
                "transformation context."
            )

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

        original_index = source_frame.index.copy()

        original_target = source_partition.y.copy(
            deep=True
        )

        original_row_ids = (
            source_partition.row_ids.copy(
                deep=True
            )
        )

        row_count_before = len(
            source_frame
        )

        feature_count_before = len(
            source_frame.columns
        )

        missing_count_before = int(
            source_frame.isna().sum().sum()
        )

        numeric_result = (
            numeric_pipeline.transform(
                frame=source_frame,
                mode=(
                    context
                    .numeric_execution_mode
                ),
            )
        )

        categorical_result = (
            categorical_pipeline.transform(
                frame=source_frame,
            )
        )

        generated_parts: list[
            pd.DataFrame
        ] = []

        if len(
            numeric_result.frame.columns
        ) > 0:
            generated_parts.append(
                numeric_result.frame
            )

        if len(
            categorical_result.frame.columns
        ) > 0:
            generated_parts.append(
                categorical_result.frame
            )

        column_metadata: dict[
            str,
            SourceFeatureTransformationMetadata,
        ] = {}

        generated_by_source: dict[
            str,
            list[str],
        ] = {}

        for column in plan.numeric_columns:
            state = numeric_result.states[
                column
            ]

            generated_by_source[
                column
            ] = [
                state.output_column
            ]

        for column in plan.categorical_columns:
            state = categorical_result.states[
                column
            ]

            generated_by_source[
                column
            ] = list(
                state.output_columns
            )

        boolean_frames: list[
            pd.DataFrame
        ] = []

        for column in plan.boolean_columns:
            if column not in source_frame.columns:
                raise KeyError(
                    f"Boolean feature {column!r} "
                    "does not exist."
                )

            transformer = (
                boolean_transformers[
                    column
                ]
            )

            encoded = transformer.transform(
                source_frame[column]
            )

            boolean_frames.append(
                encoded
            )

            generated_by_source[
                column
            ] = [
                str(output_column)
                for output_column
                in encoded.columns
            ]

        if boolean_frames:
            generated_parts.append(
                pd.concat(
                    boolean_frames,
                    axis=1,
                )
            )

        datetime_frames: list[
            pd.DataFrame
        ] = []

        for column in plan.datetime_columns:
            if column not in source_frame.columns:
                raise KeyError(
                    f"Datetime feature {column!r} "
                    "does not exist."
                )

            extractor = (
                datetime_extractors[
                    column
                ]
            )

            extracted = extractor.transform(
                source_frame[column]
            )

            datetime_frames.append(
                extracted
            )

            generated_by_source[
                column
            ] = [
                str(output_column)
                for output_column
                in extracted.columns
            ]

        if datetime_frames:
            generated_parts.append(
                pd.concat(
                    datetime_frames,
                    axis=1,
                )
            )

        text_frames: list[
            pd.DataFrame
        ] = []

        for column in plan.text_columns:
            if column not in source_frame.columns:
                raise KeyError(
                    f"Text feature {column!r} "
                    "does not exist."
                )

            extractor = text_extractors[
                column
            ]

            extracted = extractor.transform(
                source_frame[column]
            )

            text_frames.append(
                extracted
            )

            generated_by_source[
                column
            ] = [
                str(output_column)
                for output_column
                in extracted.columns
            ]

        if text_frames:
            generated_parts.append(
                pd.concat(
                    text_frames,
                    axis=1,
                )
            )

        timedelta_frames: list[
            pd.DataFrame
        ] = []

        for column in plan.timedelta_columns:
            if column not in source_frame.columns:
                raise KeyError(
                    f"Timedelta feature {column!r} "
                    "does not exist."
                )

            output_column = (
                timedelta_output_columns[
                    column
                ]
            )

            values = pd.to_timedelta(
                source_frame[column],
                errors="coerce",
            ).dt.total_seconds()

            timedelta_frame = pd.DataFrame(
                {
                    output_column: (
                        values.astype(
                            "float64"
                        )
                    )
                },
                index=source_frame.index,
            )

            timedelta_frames.append(
                timedelta_frame
            )

            generated_by_source[
                column
            ] = [
                output_column
            ]

        if timedelta_frames:
            generated_parts.append(
                pd.concat(
                    timedelta_frames,
                    axis=1,
                )
            )

        preserved_source_columns: list[str] = []
        dropped_source_columns: list[str] = []

        passthrough_parts: list[
            pd.DataFrame
        ] = []

        for column in source_frame.columns:
            column_name = str(
                column
            )

            entry = plan.entries.get(
                column_name
            )

            if entry is None:
                if (
                    self.options
                    .include_skipped_source_columns
                ):
                    passthrough_parts.append(
                        source_frame[
                            [column]
                        ].copy(
                            deep=True
                        )
                    )

                    preserved_source_columns.append(
                        column_name
                    )

                continue

            if entry.status != PlanEntryStatus.PLANNED:
                if (
                    self.options
                    .include_skipped_source_columns
                ):
                    passthrough_parts.append(
                        source_frame[
                            [column]
                        ].copy(
                            deep=True
                        )
                    )

                    preserved_source_columns.append(
                        column_name
                    )

                continue

            if (
                entry.source_column_policy
                == SourceColumnPolicy.KEEP
            ):
                passthrough_parts.append(
                    source_frame[
                        [column]
                    ].copy(
                        deep=True
                    )
                )

                preserved_source_columns.append(
                    column_name
                )

            else:
                dropped_source_columns.append(
                    column_name
                )

        if passthrough_parts:
            generated_parts.insert(
                0,
                pd.concat(
                    passthrough_parts,
                    axis=1,
                ),
            )

        if generated_parts:
            transformed_frame = pd.concat(
                generated_parts,
                axis=1,
            )

        else:
            transformed_frame = pd.DataFrame(
                index=source_frame.index
            )

        if (
            self.options
            .require_unique_output_columns
            and transformed_frame
            .columns
            .duplicated()
            .any()
        ):
            duplicates = [
                str(column)
                for column
                in transformed_frame.columns[
                    transformed_frame
                    .columns
                    .duplicated(
                        keep=False
                    )
                ].unique()
            ]

            raise ValueError(
                "Duplicate transformed feature names: "
                f"{duplicates}"
            )

        if self.options.enforce_output_schema:
            missing_expected = set(
                expected_output_columns
            ).difference(
                transformed_frame.columns
            )

            unexpected = set(
                transformed_frame.columns
            ).difference(
                expected_output_columns
            )

            if missing_expected:
                raise ValueError(
                    "The transformed partition is missing "
                    f"expected columns: "
                    f"{sorted(missing_expected)}"
                )

            if unexpected:
                raise ValueError(
                    "The transformed partition contains "
                    f"unexpected columns: "
                    f"{sorted(unexpected)}"
                )

            transformed_frame = (
                transformed_frame.loc[
                    :,
                    expected_output_columns,
                ]
            )

        self._validate_output_frame(
            source_frame=source_frame,
            transformed_frame=(
                transformed_frame
            ),
            original_index=original_index,
        )

        for column_name, entry in (
            plan.entries.items()
        ):
            if column_name not in source_frame.columns:
                column_metadata[
                    column_name
                ] = (
                    SourceFeatureTransformationMetadata(
                        source_column=(
                            column_name
                        ),
                        partition=partition.name,
                        feature_family=(
                            entry
                            .feature_family
                            .value
                        ),
                        input_dtype=(
                            "unavailable"
                        ),
                        skipped=True,
                        skip_reason=(
                            "The source feature is unavailable."
                        ),
                    )
                )

                continue

            input_dtype = str(
                source_frame[
                    column_name
                ].dtype
            )

            output_columns = (
                generated_by_source.get(
                    column_name,
                    [],
                )
            )

            source_preserved = (
                column_name
                in preserved_source_columns
            )

            source_dropped = (
                column_name
                in dropped_source_columns
            )

            unknown_count = 0
            unknown_percentage = 0.0

            if (
                column_name
                in categorical_result
                .unknown_categories
            ):
                unknown_summary = (
                    categorical_result
                    .unknown_categories[
                        column_name
                    ]
                )

                unknown_count = (
                    unknown_summary
                    .unknown_count
                )

                unknown_percentage = (
                    unknown_summary
                    .unknown_percentage
                )

            output_dtypes = {
                output_column: str(
                    transformed_frame[
                        output_column
                    ].dtype
                )
                for output_column
                in output_columns
                if output_column
                in transformed_frame.columns
            }

            column_metadata[
                column_name
            ] = SourceFeatureTransformationMetadata(
                source_column=(
                    column_name
                ),
                partition=partition.name,
                feature_family=(
                    entry
                    .feature_family
                    .value
                ),
                input_dtype=input_dtype,
                output_columns=(
                    output_columns
                ),
                output_dtypes=(
                    output_dtypes
                ),
                source_preserved=(
                    source_preserved
                ),
                source_dropped=(
                    source_dropped
                ),
                transformed=(
                    entry.status
                    == PlanEntryStatus.PLANNED
                ),
                skipped=(
                    entry.status
                    != PlanEntryStatus.PLANNED
                ),
                skip_reason=(
                    None
                    if entry.status
                    == PlanEntryStatus.PLANNED
                    else (
                        f"Plan status is "
                        f"{entry.status.value!r}."
                    )
                ),
                unknown_category_count=(
                    unknown_count
                ),
                unknown_category_percentage=(
                    unknown_percentage
                ),
            )

        transformed_partition = DatasetPartition(
            name=partition.name,
            X=transformed_frame,
            y=original_target,
            row_ids=original_row_ids,
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
        )

        lineage.validate_output_schema(
            [
                str(column)
                for column
                in transformed_frame.columns
            ]
        )

        metadata = PartitionTransformationMetadata(
            partition=partition.name,
            row_count_before=(
                row_count_before
            ),
            row_count_after=len(
                transformed_frame
            ),
            feature_count_before=(
                feature_count_before
            ),
            feature_count_after=len(
                transformed_frame.columns
            ),
            missing_count_before=(
                missing_count_before
            ),
            missing_count_after=int(
                transformed_frame
                .isna()
                .sum()
                .sum()
            ),
            nonfinite_count_after=(
                self._nonfinite_count(
                    transformed_frame
                )
            ),
            generated_columns=[
                str(column)
                for column
                in transformed_frame.columns
                if str(column)
                not in preserved_source_columns
            ],
            preserved_source_columns=(
                preserved_source_columns
            ),
            dropped_source_columns=(
                dropped_source_columns
            ),
            columns=column_metadata,
            metadata={
                "expected_output_count": len(
                    expected_output_columns
                ),
            },
        )

        return (
            transformed_partition,
            metadata,
            numeric_result,
            categorical_result,
        )

    def _validate_output_frame(
        self,
        *,
        source_frame: pd.DataFrame,
        transformed_frame: pd.DataFrame,
        original_index: pd.Index,
    ) -> None:
        """Validate transformed feature-frame integrity."""

        if (
            self.options.preserve_row_count
            and len(source_frame)
            != len(transformed_frame)
        ):
            raise ValueError(
                "Feature transformation changed row count."
            )

        if (
            self.options.preserve_row_order
            and not transformed_frame.index.equals(
                original_index
            )
        ):
            raise ValueError(
                "Feature transformation changed row order."
            )

        missing_count = int(
            transformed_frame.isna().sum().sum()
        )

        if (
            not self.options
            .allow_missing_output_values
            and missing_count > 0
        ):
            missing_columns = [
                str(column)
                for column
                in transformed_frame.columns
                if transformed_frame[
                    column
                ].isna().any()
            ]

            raise ValueError(
                "Missing values remain in transformed "
                f"features: {missing_columns}"
            )

        nonfinite_count = self._nonfinite_count(
            transformed_frame
        )

        if (
            not self.options
            .allow_nonfinite_output_values
            and nonfinite_count > 0
        ):
            raise ValueError(
                "Nonfinite values remain in the "
                "transformed feature matrix."
            )

        if self.options.require_numeric_output:
            nonnumeric_columns = [
                str(column)
                for column
                in transformed_frame.columns
                if not pd.api.types
                .is_numeric_dtype(
                    transformed_frame[
                        column
                    ].dtype
                )
            ]

            if nonnumeric_columns:
                raise TypeError(
                    "Transformed output contains nonnumeric "
                    f"features: {nonnumeric_columns}"
                )

    def _validate_partition_integrity(
        self,
        *,
        source: DatasetPartition,
        transformed: DatasetPartition,
    ) -> None:
        """Validate protected partition data."""

        if (
            self.options.preserve_target
            and not source.y.equals(
                transformed.y
            )
        ):
            raise ValueError(
                "Feature transformation changed target values."
            )

        if (
            self.options.preserve_row_ids
            and not source.row_ids.equals(
                transformed.row_ids
            )
        ):
            raise ValueError(
                "Feature transformation changed row IDs."
            )

    @staticmethod
    def _nonfinite_count(
        frame: pd.DataFrame,
    ) -> int:
        """Count infinite values in numeric output columns."""

        total = 0

        for column in frame.columns:
            series = frame[
                column
            ]

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

            total += int(
                mask.sum()
            )

        return total
