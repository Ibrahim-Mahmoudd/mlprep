"""Unified fitted transformation pipeline for Phase 6."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from mlprep.phases.phase2.output_governance.metadata import (
    PartitionName,
)
from mlprep.phases.phase2.output_governance.result import (
    DatasetPartition,
)
from mlprep.phases.phase6.categorical.categorical_pipeline import (
    CategoricalColumnPipelineState,
    CategoricalPipeline,
    CategoricalPipelineOptions,
    CategoricalPipelineResult,
)
from mlprep.phases.phase6.contract.transformation_contract import (
    TransformationContract,
)
from mlprep.phases.phase6.contract.transformation_plan import (
    FeatureFamily,
    PlanEntryStatus,
    SourceColumnPolicy,
    TransformationPlan,
)
from mlprep.phases.phase6.feature_engineering.boolean_features import (
    BooleanFeatureOptions,
    BooleanFeatureState,
    BooleanFeatureTransformer,
)
from mlprep.phases.phase6.feature_engineering.datetime_features import (
    DatetimeFeatureExtractor,
    DatetimeFeatureOptions,
    DatetimeFeatureState,
)
from mlprep.phases.phase6.feature_engineering.feature_lineage import (
    FeatureLineageState,
    FeatureLineageTracker,
)
from mlprep.phases.phase6.feature_engineering.text_features import (
    TextFeatureExtractor,
    TextFeatureOptions,
    TextFeatureState,
)
from mlprep.phases.phase6.numeric.numeric_pipeline import (
    NumericColumnPipelineState,
    NumericPipeline,
    NumericPipelineOptions,
    NumericPipelineResult,
)
from mlprep.phases.phase6.orchestration.fit_context import (
    TransformationFitContext,
)
from mlprep.phases.phase6.orchestration.partition_transformer import (
    PartitionTransformationMetadata,
    PartitionTransformer,
    PartitionTransformerOptions,
)


class PartitionPipelineResult(BaseModel):
    """Pipeline metadata for one transformed partition."""

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        arbitrary_types_allowed=True,
    )

    partition: PartitionName

    transformation: PartitionTransformationMetadata

    numeric: NumericPipelineResult

    categorical: CategoricalPipelineResult

    metadata: dict[str, Any] = Field(
        default_factory=dict
    )


class TransformationPipelineResult(BaseModel):
    """Complete fitted Phase 6 pipeline metadata."""

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        arbitrary_types_allowed=True,
    )

    partitions: dict[
        PartitionName,
        PartitionPipelineResult,
    ] = Field(default_factory=dict)

    numeric_states: dict[
        str,
        NumericColumnPipelineState,
    ] = Field(default_factory=dict)

    categorical_states: dict[
        str,
        CategoricalColumnPipelineState,
    ] = Field(default_factory=dict)

    boolean_states: dict[
        str,
        BooleanFeatureState,
    ] = Field(default_factory=dict)

    datetime_states: dict[
        str,
        DatetimeFeatureState,
    ] = Field(default_factory=dict)

    text_states: dict[
        str,
        TextFeatureState,
    ] = Field(default_factory=dict)

    timedelta_output_columns: dict[
        str,
        str,
    ] = Field(default_factory=dict)

    output_columns: list[str] = Field(
        default_factory=list
    )

    lineage: FeatureLineageState

    fitted_on_training_only: bool = True

    metadata: dict[str, Any] = Field(
        default_factory=dict
    )


@dataclass(slots=True)
class TransformationPipelineOptions:
    """Options controlling the fitted Phase 6 pipeline."""

    numeric: NumericPipelineOptions = field(
        default_factory=NumericPipelineOptions
    )

    categorical: CategoricalPipelineOptions = field(
        default_factory=CategoricalPipelineOptions
    )

    boolean: BooleanFeatureOptions = field(
        default_factory=BooleanFeatureOptions
    )

    datetime: DatetimeFeatureOptions = field(
        default_factory=DatetimeFeatureOptions
    )

    text: TextFeatureOptions = field(
        default_factory=TextFeatureOptions
    )

    partition_transformer: PartitionTransformerOptions = field(
        default_factory=PartitionTransformerOptions
    )

    timedelta_separator: str = "__timedelta__"

    test_transform_authorized: bool = True

    require_nonempty_output: bool = True

    def __post_init__(self) -> None:
        """Validate pipeline options."""

        if not self.timedelta_separator:
            raise ValueError(
                "timedelta_separator cannot be empty."
            )


class TransformationPipeline:
    """Fit on training and transform all partitions."""

    component_name = "transformation_pipeline"

    def __init__(
        self,
        options: TransformationPipelineOptions | None = None,
    ) -> None:
        """Initialize the transformation pipeline."""

        self.options = (
            options
            or TransformationPipelineOptions()
        )

        self.numeric_pipeline = NumericPipeline(
            options=self.options.numeric
        )

        self.categorical_pipeline = (
            CategoricalPipeline(
                options=(
                    self.options.categorical
                )
            )
        )

        self.partition_transformer = (
            PartitionTransformer(
                options=(
                    self.options
                    .partition_transformer
                )
            )
        )

        self.boolean_transformers: dict[
            str,
            BooleanFeatureTransformer,
        ] = {}

        self.datetime_extractors: dict[
            str,
            DatetimeFeatureExtractor,
        ] = {}

        self.text_extractors: dict[
            str,
            TextFeatureExtractor,
        ] = {}

        self.timedelta_output_columns: dict[
            str,
            str,
        ] = {}

        self.lineage: (
            FeatureLineageTracker | None
        ) = None

        self.expected_output_columns: list[str] = []

        self.plan: TransformationPlan | None = None

        self._fitted = False

    @property
    def is_fitted(
        self,
    ) -> bool:
        """Return whether the pipeline is fitted."""

        return self._fitted

    def fit(
        self,
        *,
        contract: TransformationContract,
        plan: TransformationPlan,
    ) -> TransformationPipeline:
        """Fit all Phase 6 components on training data."""

        if self._fitted:
            raise RuntimeError(
                "The transformation pipeline is already fitted."
            )

        context = (
            TransformationFitContext.training_fit()
        )

        context.validate()
        context.assert_fitting_allowed()

        training_frame = contract.train.X

        self.numeric_pipeline.fit(
            frame=training_frame,
            plan=plan,
        )

        self.categorical_pipeline.fit(
            frame=training_frame,
            plan=plan,
        )

        self._fit_boolean_transformers(
            frame=training_frame,
            plan=plan,
        )

        self._fit_datetime_extractors(
            frame=training_frame,
            plan=plan,
        )

        self._fit_text_extractors(
            frame=training_frame,
            plan=plan,
        )

        self._configure_timedelta_outputs(
            plan=plan
        )

        self.lineage = FeatureLineageTracker(
            source_columns=(
                contract.feature_columns
            )
        )

        self.plan = plan
        self.expected_output_columns = (
            self._build_output_schema(
                contract=contract,
                plan=plan,
            )
        )

        if (
            self.options.require_nonempty_output
            and not self.expected_output_columns
        ):
            raise ValueError(
                "Phase 6 produced an empty output schema."
            )

        self.lineage.validate_output_schema(
            self.expected_output_columns
        )

        self._fitted = True

        return self

    def fit_transform(
        self,
        *,
        contract: TransformationContract,
        plan: TransformationPlan,
    ) -> tuple[
        DatasetPartition,
        DatasetPartition,
        DatasetPartition,
        TransformationPipelineResult,
    ]:
        """Fit on training and transform all partitions."""

        self.fit(
            contract=contract,
            plan=plan,
        )

        (
            transformed_train,
            training_result,
        ) = self._transform_partition(
            partition=contract.train,
            context=(
                TransformationFitContext
                .training_transform()
            ),
        )

        (
            transformed_validation,
            validation_result,
        ) = self._transform_partition(
            partition=contract.validation,
            context=(
                TransformationFitContext
                .validation_transform()
            ),
        )

        (
            transformed_test,
            test_result,
        ) = self._transform_partition(
            partition=contract.test,
            context=(
                TransformationFitContext
                .test_transform(
                    authorized=(
                        self.options
                        .test_transform_authorized
                    )
                )
            ),
        )

        if self.lineage is None:
            raise RuntimeError(
                "Feature lineage is unavailable."
            )

        pipeline_result = (
            TransformationPipelineResult(
                partitions={
                    PartitionName.TRAIN: (
                        training_result
                    ),
                    PartitionName.VALIDATION: (
                        validation_result
                    ),
                    PartitionName.TEST: (
                        test_result
                    ),
                },
                numeric_states=(
                    self.numeric_pipeline
                    .get_states()
                ),
                categorical_states=(
                    self.categorical_pipeline
                    .get_states()
                ),
                boolean_states={
                    column: transformer.get_state()
                    for column, transformer
                    in self.boolean_transformers.items()
                },
                datetime_states={
                    column: extractor.get_state()
                    for column, extractor
                    in self.datetime_extractors.items()
                },
                text_states={
                    column: extractor.get_state()
                    for column, extractor
                    in self.text_extractors.items()
                },
                timedelta_output_columns=dict(
                    self.timedelta_output_columns
                ),
                output_columns=list(
                    self.expected_output_columns
                ),
                lineage=self.lineage.get_state(),
                fitted_on_training_only=True,
                metadata={
                    "source_feature_count": len(
                        contract.feature_columns
                    ),
                    "output_feature_count": len(
                        self.expected_output_columns
                    ),
                    "numeric_source_count": len(
                        plan.numeric_columns
                    ),
                    "categorical_source_count": len(
                        plan.categorical_columns
                    ),
                    "boolean_source_count": len(
                        plan.boolean_columns
                    ),
                    "datetime_source_count": len(
                        plan.datetime_columns
                    ),
                    "text_source_count": len(
                        plan.text_columns
                    ),
                    "timedelta_source_count": len(
                        plan.timedelta_columns
                    ),
                },
            )
        )

        return (
            transformed_train,
            transformed_validation,
            transformed_test,
            pipeline_result,
        )

    def transform_partition(
        self,
        *,
        partition: DatasetPartition,
        context: TransformationFitContext,
    ) -> tuple[
        DatasetPartition,
        PartitionPipelineResult,
    ]:
        """Transform an additional partition."""

        if not self._fitted:
            raise RuntimeError(
                "The transformation pipeline must be fitted "
                "before transforming another partition."
            )

        return self._transform_partition(
            partition=partition,
            context=context,
        )

    def _transform_partition(
        self,
        *,
        partition: DatasetPartition,
        context: TransformationFitContext,
    ) -> tuple[
        DatasetPartition,
        PartitionPipelineResult,
    ]:
        """Transform one partition with fitted state."""

        if not self._fitted:
            raise RuntimeError(
                "The transformation pipeline is not fitted."
            )

        if self.plan is None:
            raise RuntimeError(
                "The fitted transformation plan is unavailable."
            )

        if self.lineage is None:
            raise RuntimeError(
                "Feature lineage is unavailable."
            )

        (
            transformed,
            transformation_metadata,
            numeric_result,
            categorical_result,
        ) = self.partition_transformer.transform(
            partition=partition,
            context=context,
            plan=self.plan,
            numeric_pipeline=(
                self.numeric_pipeline
            ),
            categorical_pipeline=(
                self.categorical_pipeline
            ),
            boolean_transformers=(
                self.boolean_transformers
            ),
            datetime_extractors=(
                self.datetime_extractors
            ),
            text_extractors=(
                self.text_extractors
            ),
            timedelta_output_columns=(
                self.timedelta_output_columns
            ),
            lineage=self.lineage,
            expected_output_columns=(
                self.expected_output_columns
            ),
        )

        result = PartitionPipelineResult(
            partition=partition.name,
            transformation=(
                transformation_metadata
            ),
            numeric=numeric_result,
            categorical=(
                categorical_result
            ),
            metadata={
                "context_operation": (
                    context.operation.value
                ),
                "training_data": (
                    context.training_data
                ),
                "test_authorized": (
                    context
                    .test_partition_authorized
                ),
            },
        )

        return (
            transformed,
            result,
        )

    def _fit_boolean_transformers(
        self,
        *,
        frame: pd.DataFrame,
        plan: TransformationPlan,
    ) -> None:
        """Fit Boolean transformers on training data."""

        for column in plan.boolean_columns:
            entry = plan.entries[
                column
            ]

            transformer = (
                BooleanFeatureTransformer(
                    column=column,
                    output_prefix=(
                        entry.output_prefix
                    ),
                    options=self.options.boolean,
                )
            )

            transformer.fit(
                frame[column]
            )

            self.boolean_transformers[
                column
            ] = transformer

    def _fit_datetime_extractors(
        self,
        *,
        frame: pd.DataFrame,
        plan: TransformationPlan,
    ) -> None:
        """Fit datetime extractors on training data."""

        for column in plan.datetime_columns:
            entry = plan.entries[
                column
            ]

            extractor = (
                DatetimeFeatureExtractor(
                    column=column,
                    output_prefix=(
                        entry.output_prefix
                    ),
                    options=self.options.datetime,
                )
            )

            extractor.fit(
                frame[column]
            )

            self.datetime_extractors[
                column
            ] = extractor

    def _fit_text_extractors(
        self,
        *,
        frame: pd.DataFrame,
        plan: TransformationPlan,
    ) -> None:
        """Fit text extractors on training data."""

        for column in plan.text_columns:
            entry = plan.entries[
                column
            ]

            extractor = TextFeatureExtractor(
                column=column,
                output_prefix=(
                    entry.output_prefix
                ),
                options=self.options.text,
            )

            extractor.fit(
                frame[column]
            )

            self.text_extractors[
                column
            ] = extractor

    def _configure_timedelta_outputs(
        self,
        *,
        plan: TransformationPlan,
    ) -> None:
        """Build deterministic timedelta output names."""

        for column in plan.timedelta_columns:
            entry = plan.entries[
                column
            ]

            prefix = (
                entry.output_prefix
                or column
            )

            self.timedelta_output_columns[
                column
            ] = (
                f"{prefix}"
                f"{self.options.timedelta_separator}"
                "seconds"
            )

    def _build_output_schema(
        self,
        *,
        contract: TransformationContract,
        plan: TransformationPlan,
    ) -> list[str]:
        """Build and register the final output schema."""

        if self.lineage is None:
            raise RuntimeError(
                "Feature lineage must be initialized first."
            )

        output_columns: list[str] = []

        for source_column in contract.feature_columns:
            entry = plan.entries.get(
                source_column
            )

            if (
                entry is None
                or entry.status
                != PlanEntryStatus.PLANNED
            ):
                output_columns.append(
                    source_column
                )

                family = (
                    contract
                    .feature_groups
                    .family_for(
                        source_column
                    )
                )

                self.lineage.register_passthrough(
                    column=source_column,
                    feature_family=family,
                )

                continue

            source_preserved = (
                entry.source_column_policy
                == SourceColumnPolicy.KEEP
            )

            if source_preserved:
                output_columns.append(
                    source_column
                )

                self.lineage.register_passthrough(
                    column=source_column,
                    feature_family=(
                        entry.feature_family.value
                    ),
                )

            generated_columns = (
                self._generated_columns_for(
                    source_column=source_column,
                    family=entry.feature_family,
                )
            )

            for generated_column in generated_columns:
                if generated_column in output_columns:
                    raise ValueError(
                        "Output-column collision: "
                        f"{generated_column!r}"
                    )

                output_columns.append(
                    generated_column
                )

            self.lineage.register_many(
                output_columns=(
                    generated_columns
                ),
                source_column=(
                    source_column
                ),
                transformation_steps=(
                    self._transformation_steps(
                        source_column
                    )
                ),
                feature_family=(
                    entry.feature_family.value
                ),
                source_preserved=(
                    source_preserved
                ),
            )

        if len(output_columns) != len(
            set(output_columns)
        ):
            duplicates = [
                column
                for column in output_columns
                if output_columns.count(
                    column
                )
                > 1
            ]

            raise ValueError(
                "Duplicate columns in final output schema: "
                f"{sorted(set(duplicates))}"
            )

        return output_columns

    def _generated_columns_for(
        self,
        *,
        source_column: str,
        family: FeatureFamily,
    ) -> list[str]:
        """Return generated columns for one source feature."""

        if family == FeatureFamily.NUMERIC:
            return [
                self.numeric_pipeline
                .states[
                    source_column
                ]
                .output_column
            ]

        if family == FeatureFamily.CATEGORICAL:
            return list(
                self.categorical_pipeline
                .states[
                    source_column
                ]
                .output_columns
            )

        if family == FeatureFamily.BOOLEAN:
            return [
                self.boolean_transformers[
                    source_column
                ].state.output_column
            ]

        if family == FeatureFamily.DATETIME:
            return list(
                self.datetime_extractors[
                    source_column
                ].state.output_columns
            )

        if family == FeatureFamily.TEXT:
            return list(
                self.text_extractors[
                    source_column
                ].state.output_columns
            )

        if family == FeatureFamily.TIMEDELTA:
            return [
                self.timedelta_output_columns[
                    source_column
                ]
            ]

        return []

    def _transformation_steps(
        self,
        source_column: str,
    ) -> list[str]:
        """Return transformation steps for lineage."""

        if self.plan is None:
            raise RuntimeError(
                "The transformation plan is unavailable."
            )

        entry = self.plan.entries[
            source_column
        ]

        steps: list[str] = []

        if entry.feature_family == FeatureFamily.NUMERIC:
            if (
                entry.numeric_transformation is None
                or entry.numeric_scaling is None
            ):
                raise ValueError(
                    f"Numeric feature {source_column!r} has an incomplete plan."
                )
            steps.extend(
                [
                    entry.numeric_transformation.value,
                    entry.numeric_scaling.value,
                ]
            )

        elif (
            entry.feature_family
            == FeatureFamily.CATEGORICAL
        ):
            if entry.group_rare_categories:
                steps.append(
                    "rare_category_grouping"
                )

            if entry.categorical_encoding is None:
                raise ValueError(
                    f"Categorical feature {source_column!r} has no encoder."
                )
            steps.append(
                entry.categorical_encoding.value
            )

        elif entry.feature_engineering is not None:
            steps.append(
                entry.feature_engineering.value
            )

        return steps
