"""Unified fitted imputation pipeline for Phase 4."""

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
from mlprep.phases.phase4.contract.imputation_contract import (
    ImputationContract,
)
from mlprep.phases.phase4.contract.imputation_plan import (
    ImputationFeatureFamily,
    ImputationMethodName,
    ImputationPlan,
    ImputationPlanEntry,
    PlanEntryStatus,
)
from mlprep.phases.phase4.detection.invalid_value_scanner import (
    InvalidValueScanner,
    InvalidValueScannerOptions,
    InvalidValueScanResult,
)
from mlprep.phases.phase4.detection.missing_token_detector import (
    MissingTokenDetectionResult,
    MissingTokenDetector,
    MissingTokenDetectorOptions,
)
from mlprep.phases.phase4.indicators.missing_indicator import (
    MissingIndicatorGenerator,
    MissingIndicatorOptions,
    MissingIndicatorResult,
)
from mlprep.phases.phase4.orchestration.fit_context import (
    FitContext,
)
from mlprep.phases.phase4.orchestration.partition_transformer import (
    PartitionTransformationMetadata,
    PartitionTransformer,
    PartitionTransformerOptions,
)
from mlprep.phases.phase4.strategies.base import (
    BaseImputer,
    ImputerState,
)
from mlprep.phases.phase4.strategies.boolean_imputer import (
    BooleanImputer,
    BooleanImputerOptions,
)
from mlprep.phases.phase4.strategies.categorical_imputer import (
    CategoricalImputer,
    CategoricalImputerOptions,
)
from mlprep.phases.phase4.strategies.constant_imputer import (
    ConstantImputer,
)
from mlprep.phases.phase4.strategies.datetime_imputer import (
    DatetimeImputer,
    DatetimeImputerOptions,
)
from mlprep.phases.phase4.strategies.group_imputer import (
    GroupImputer,
    GroupImputerOptions,
)
from mlprep.phases.phase4.strategies.model_imputer import (
    ModelImputer,
    ModelImputerOptions,
)
from mlprep.phases.phase4.strategies.numeric_imputer import (
    NumericImputer,
    NumericImputerOptions,
)
from mlprep.phases.phase4.strategies.temporal_imputer import (
    TemporalImputer,
    TemporalImputerOptions,
)
from mlprep.phases.phase4.strategies.text_imputer import (
    TextImputer,
    TextImputerOptions,
)


class PartitionPipelineResult(BaseModel):
    """Pipeline output metadata for one partition."""

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        arbitrary_types_allowed=True,
    )

    partition: PartitionName

    invalid_value_scan: InvalidValueScanResult

    missing_token_scan: MissingTokenDetectionResult

    transformation: PartitionTransformationMetadata

    indicators: MissingIndicatorResult | None = None

    metadata: dict[str, Any] = Field(
        default_factory=dict
    )


class ImputationPipelineResult(BaseModel):
    """Complete output metadata from the imputation pipeline."""

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        arbitrary_types_allowed=True,
    )

    partitions: dict[
        PartitionName,
        PartitionPipelineResult,
    ] = Field(default_factory=dict)

    imputer_states: dict[
        str,
        ImputerState,
    ] = Field(default_factory=dict)

    indicator_mapping: dict[
        str,
        str,
    ] = Field(default_factory=dict)

    fitted_on_training_only: bool = True

    metadata: dict[str, Any] = Field(
        default_factory=dict
    )


@dataclass(slots=True)
class ImputationPipelineOptions:
    """Options controlling the fitted imputation pipeline."""

    invalid_values: InvalidValueScannerOptions = field(
        default_factory=InvalidValueScannerOptions
    )

    missing_tokens: MissingTokenDetectorOptions = field(
        default_factory=MissingTokenDetectorOptions
    )

    indicators: MissingIndicatorOptions = field(
        default_factory=MissingIndicatorOptions
    )

    partition_transformer: (
        PartitionTransformerOptions
    ) = field(
        default_factory=PartitionTransformerOptions
    )

    numeric: NumericImputerOptions = field(
        default_factory=NumericImputerOptions
    )

    categorical: CategoricalImputerOptions = field(
        default_factory=CategoricalImputerOptions
    )

    boolean: BooleanImputerOptions = field(
        default_factory=BooleanImputerOptions
    )

    datetime: DatetimeImputerOptions = field(
        default_factory=DatetimeImputerOptions
    )

    text: TextImputerOptions = field(
        default_factory=TextImputerOptions
    )

    group: GroupImputerOptions = field(
        default_factory=GroupImputerOptions
    )

    temporal: TemporalImputerOptions = field(
        default_factory=TemporalImputerOptions
    )

    model: ModelImputerOptions = field(
        default_factory=ModelImputerOptions
    )


class ImputationPipeline:
    """Fit on training and transform every partition."""

    component_name = "imputation_pipeline"

    def __init__(
        self,
        options: ImputationPipelineOptions | None = None,
    ) -> None:
        """Initialize the imputation pipeline."""

        self.options = (
            options or ImputationPipelineOptions()
        )

        self.invalid_value_scanner = (
            InvalidValueScanner(
                self.options.invalid_values
            )
        )

        self.missing_token_detector = (
            MissingTokenDetector(
                self.options.missing_tokens
            )
        )

        self.indicator_generator = (
            MissingIndicatorGenerator(
                self.options.indicators
            )
        )

        self.partition_transformer = (
            PartitionTransformer(
                self.options
                .partition_transformer
            )
        )

        self.imputers: dict[
            str,
            BaseImputer,
        ] = {}

        self._fitted = False
        self._plan: ImputationPlan | None = None

    @property
    def is_fitted(
        self,
    ) -> bool:
        """Return whether the pipeline is fitted."""

        return self._fitted

    def fit_transform(
        self,
        *,
        contract: ImputationContract,
        plan: ImputationPlan,
        test_authorized: bool = True,
    ) -> tuple[
        DatasetPartition,
        DatasetPartition,
        DatasetPartition,
        ImputationPipelineResult,
    ]:
        """Fit on training and transform all partitions."""

        if self._fitted:
            raise RuntimeError(
                "The imputation pipeline is already fitted."
            )

        self._plan = plan

        (
            training_invalid_scan,
            training_invalid_normalized,
        ) = self.invalid_value_scanner.scan_partition(
            contract=contract,
            partition=PartitionName.TRAIN,
        )

        (
            training_token_scan,
            normalized_training,
        ) = self.missing_token_detector.detect_partition(
            contract=contract,
            partition=PartitionName.TRAIN,
            frame=training_invalid_normalized,
        )

        original_training_missing_mask = (
            normalized_training.isna().copy(
                deep=True
            )
        )

        self.indicator_generator.fit(
            frame=normalized_training,
            plan=plan,
        )

        self._fit_imputers(
            frame=normalized_training,
            plan=plan,
        )

        self._fitted = True

        (
            transformed_train,
            training_metadata,
            training_indicators,
        ) = self.partition_transformer.transform(
            partition=contract.train,
            context=FitContext.training_transform(),
            plan=plan,
            imputers=self.imputers,
            indicator_generator=(
                self.indicator_generator
            ),
            normalized_frame=(
                normalized_training
            ),
            original_missing_mask=(
                original_training_missing_mask
            ),
        )

        (
            transformed_validation,
            validation_pipeline_result,
        ) = self._transform_partition(
            contract=contract,
            partition=contract.validation,
            context=FitContext.validation_transform(),
            plan=plan,
        )

        (
            transformed_test,
            test_pipeline_result,
        ) = self._transform_partition(
            contract=contract,
            partition=contract.test,
            context=FitContext.test_transform(
                authorized=test_authorized
            ),
            plan=plan,
        )

        training_pipeline_result = (
            PartitionPipelineResult(
                partition=PartitionName.TRAIN,
                invalid_value_scan=(
                    training_invalid_scan
                ),
                missing_token_scan=(
                    training_token_scan
                ),
                transformation=(
                    training_metadata
                ),
                indicators=(
                    training_indicators
                ),
            )
        )

        pipeline_result = ImputationPipelineResult(
            partitions={
                PartitionName.TRAIN: (
                    training_pipeline_result
                ),
                PartitionName.VALIDATION: (
                    validation_pipeline_result
                ),
                PartitionName.TEST: (
                    test_pipeline_result
                ),
            },
            imputer_states={
                column: imputer.get_state()
                for column, imputer
                in self.imputers.items()
            },
            indicator_mapping=(
                self.indicator_generator.mapping()
            ),
            fitted_on_training_only=True,
            metadata={
                "plan_entry_count": len(
                    plan.entries
                ),
                "fitted_imputer_count": len(
                    self.imputers
                ),
            },
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
        contract: ImputationContract,
        partition: DatasetPartition,
        context: FitContext,
    ) -> tuple[
        DatasetPartition,
        PartitionPipelineResult,
    ]:
        """Transform an additional partition after fitting."""

        if not self._fitted:
            raise RuntimeError(
                "The pipeline must be fitted before "
                "transforming another partition."
            )

        if self._plan is None:
            raise RuntimeError(
                "The fitted imputation plan is unavailable."
            )

        return self._transform_partition(
            contract=contract,
            partition=partition,
            context=context,
            plan=self._plan,
        )

    def _transform_partition(
        self,
        *,
        contract: ImputationContract,
        partition: DatasetPartition,
        context: FitContext,
        plan: ImputationPlan,
    ) -> tuple[
        DatasetPartition,
        PartitionPipelineResult,
    ]:
        """Normalize and transform one non-fitting partition."""

        (
            invalid_scan,
            invalid_normalized,
        ) = self.invalid_value_scanner.scan_partition(
            contract=contract,
            partition=partition.name,
        )

        (
            token_scan,
            normalized_frame,
        ) = self.missing_token_detector.detect_partition(
            contract=contract,
            partition=partition.name,
            frame=invalid_normalized,
        )

        original_missing_mask = (
            normalized_frame.isna().copy(
                deep=True
            )
        )

        (
            transformed,
            transformation_metadata,
            indicator_result,
        ) = self.partition_transformer.transform(
            partition=partition,
            context=context,
            plan=plan,
            imputers=self.imputers,
            indicator_generator=(
                self.indicator_generator
            ),
            normalized_frame=normalized_frame,
            original_missing_mask=(
                original_missing_mask
            ),
        )

        pipeline_result = PartitionPipelineResult(
            partition=partition.name,
            invalid_value_scan=invalid_scan,
            missing_token_scan=token_scan,
            transformation=(
                transformation_metadata
            ),
            indicators=indicator_result,
        )

        return (
            transformed,
            pipeline_result,
        )

    def _fit_imputers(
        self,
        *,
        frame: pd.DataFrame,
        plan: ImputationPlan,
    ) -> None:
        """Fit every executable imputer on training data."""

        for entry in plan.entries.values():
            if not self._entry_requires_imputer(
                entry
            ):
                continue

            if entry.column not in frame.columns:
                raise ValueError(
                    f"Training feature {entry.column!r} "
                    "does not exist."
                )

            imputer = self._build_imputer(
                entry
            )

            if isinstance(
                imputer,
                (GroupImputer, TemporalImputer, ModelImputer),
            ):
                imputer.fit_frame(
                    frame
                )

            else:
                imputer.fit(
                    frame[entry.column]
                )

            self.imputers[
                entry.column
            ] = imputer

    def _build_imputer(
        self,
        entry: ImputationPlanEntry,
    ) -> BaseImputer:
        """Build one imputer from a plan entry."""

        method = entry.method
        family = entry.feature_family

        if method == ImputationMethodName.CONSTANT:
            return ConstantImputer(
                column=entry.column,
                value=entry.constant_value,
            )

        if (
            method
            == ImputationMethodName.MISSING_CATEGORY
        ):
            return CategoricalImputer(
                column=entry.column,
                options=CategoricalImputerOptions(
                    method="missing_category",
                    constant_value=str(
                        entry.constant_value
                        or "__MISSING__"
                    ),
                ),
            )

        if method == ImputationMethodName.GROUP_BASED:
            return GroupImputer(
                column=entry.column,
                group_columns=entry.group_columns,
                options=self.options.group,
            )

        if method == ImputationMethodName.TIME_AWARE:
            if entry.timestamp_column is None:
                raise ValueError(
                    "Time-aware imputation requires "
                    "a timestamp column."
                )

            return TemporalImputer(
                column=entry.column,
                timestamp_column=(
                    entry.timestamp_column
                ),
                options=self.options.temporal,
            )

        if method == ImputationMethodName.MODEL_BASED:
            return ModelImputer(
                column=entry.column,
                options=self.options.model,
            )

        if family == ImputationFeatureFamily.NUMERIC:
            numeric_method = {
                ImputationMethodName.MEAN: "mean",
                ImputationMethodName.MEDIAN: "median",
                ImputationMethodName.MOST_FREQUENT: (
                    "median"
                ),
            }.get(
                method,
                self.options.numeric.method,
            )

            return NumericImputer(
                column=entry.column,
                options=NumericImputerOptions(
                    method=numeric_method,
                    constant_value=(
                        self.options
                        .numeric
                        .constant_value
                    ),
                    quantile=(
                        self.options
                        .numeric
                        .quantile
                    ),
                    trim_fraction=(
                        self.options
                        .numeric
                        .trim_fraction
                    ),
                    preserve_nullable_dtype=(
                        self.options
                        .numeric
                        .preserve_nullable_dtype
                    ),
                ),
            )

        if (
            family
            == ImputationFeatureFamily.BOOLEAN
        ):
            return BooleanImputer(
                column=entry.column,
                options=self.options.boolean,
            )

        if (
            family
            == ImputationFeatureFamily.DATETIME
        ):
            datetime_method = {
                ImputationMethodName.MEDIAN: "median",
                ImputationMethodName.MOST_FREQUENT: (
                    "most_frequent"
                ),
            }.get(
                method,
                self.options.datetime.method,
            )

            return DatetimeImputer(
                column=entry.column,
                options=DatetimeImputerOptions(
                    method=datetime_method,
                    constant_value=(
                        self.options
                        .datetime
                        .constant_value
                    ),
                    day_first=(
                        self.options
                        .datetime
                        .day_first
                    ),
                    year_first=(
                        self.options
                        .datetime
                        .year_first
                    ),
                ),
            )

        if family == ImputationFeatureFamily.TEXT:
            return TextImputer(
                column=entry.column,
                options=self.options.text,
            )

        categorical_method = {
            ImputationMethodName.MOST_FREQUENT: (
                "most_frequent"
            ),
            ImputationMethodName.MISSING_CATEGORY: (
                "missing_category"
            ),
        }.get(
            method,
            self.options.categorical.method,
        )

        return CategoricalImputer(
            column=entry.column,
            options=CategoricalImputerOptions(
                method=categorical_method,
                constant_value=(
                    str(
                        entry.constant_value
                    )
                    if entry.constant_value
                    is not None
                    else self.options
                    .categorical
                    .constant_value
                ),
                preserve_category_dtype=(
                    self.options
                    .categorical
                    .preserve_category_dtype
                ),
            ),
        )

    @staticmethod
    def _entry_requires_imputer(
        entry: ImputationPlanEntry,
    ) -> bool:
        """Return whether a plan entry needs an imputer."""

        return bool(
            entry.status == PlanEntryStatus.PLANNED
            and entry.method
            not in {
                ImputationMethodName.NONE,
                ImputationMethodName.REVIEW,
                ImputationMethodName.DROP_FEATURE,
            }
        )
