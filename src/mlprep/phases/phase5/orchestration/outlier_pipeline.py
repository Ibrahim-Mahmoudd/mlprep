"""Unified fitted outlier-treatment pipeline for Phase 5."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from mlprep.phases.phase2.output_governance.metadata import (
    PartitionName,
)
from mlprep.phases.phase2.output_governance.result import (
    DatasetPartition,
)
from mlprep.phases.phase5.contract.outlier_contract import (
    OutlierContract,
)
from mlprep.phases.phase5.contract.treatment_plan import (
    TreatmentPlan,
)
from mlprep.phases.phase5.detection.detector_engine import (
    DetectionResult,
    DetectorEngine,
    DetectorEngineOptions,
)
from mlprep.phases.phase5.orchestration.fit_context import (
    OutlierFitContext,
)
from mlprep.phases.phase5.orchestration.partition_transformer import (
    PartitionTransformer,
    PartitionTransformerOptions,
    PartitionTreatmentMetadata,
)
from mlprep.phases.phase5.strategies.base import (
    BaseTreatmentStrategy,
    TreatmentExecutionMode,
    TreatmentState,
)
from mlprep.phases.phase5.strategies.strategy_factory import (
    StrategyFactory,
)
from mlprep.phases.phase5.strategies.winsorize_strategy import (
    WinsorizeStrategyOptions,
)


class PartitionPipelineResult(BaseModel):
    """Pipeline output metadata for one partition."""

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        arbitrary_types_allowed=True,
    )

    partition: PartitionName

    detection: DetectionResult

    treatment: PartitionTreatmentMetadata

    metadata: dict[str, Any] = Field(
        default_factory=dict
    )


class OutlierPipelineResult(BaseModel):
    """Complete output metadata from the Phase 5 pipeline."""

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        arbitrary_types_allowed=True,
    )

    partitions: dict[
        PartitionName,
        PartitionPipelineResult,
    ] = Field(default_factory=dict)

    detector_states: dict[
        str,
        dict[str, Any],
    ] = Field(default_factory=dict)

    treatment_states: dict[
        str,
        TreatmentState,
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
class OutlierPipelineOptions:
    """Options controlling the fitted Phase 5 pipeline."""

    detector_engine: DetectorEngineOptions = field(
        default_factory=DetectorEngineOptions
    )

    partition_transformer: (
        PartitionTransformerOptions
    ) = field(
        default_factory=PartitionTransformerOptions
    )

    winsorization: WinsorizeStrategyOptions = field(
        default_factory=WinsorizeStrategyOptions
    )

    require_nonempty_plan: bool = False

    require_all_strategies_fitted: bool = True

    test_transform_authorized: bool = True


class OutlierPipeline:
    """Fit on training and transform every partition."""

    component_name = "outlier_pipeline"

    def __init__(
        self,
        options: OutlierPipelineOptions | None = None,
    ) -> None:
        """Initialize the pipeline."""

        self.options = (
            options
            or OutlierPipelineOptions()
        )

        self.detector_engine = (
            DetectorEngine(
                options=(
                    self.options
                    .detector_engine
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

        self.strategies: dict[
            str,
            BaseTreatmentStrategy,
        ] = {}

        self.plan: TreatmentPlan | None = None

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
        contract: OutlierContract,
        plan: TreatmentPlan,
    ) -> OutlierPipeline:
        """Fit detectors and treatment strategies on training."""

        if self._fitted:
            raise RuntimeError(
                "The outlier pipeline is already fitted."
            )

        context = (
            OutlierFitContext.training_fit()
        )

        context.validate()
        context.assert_detector_fitting_allowed()
        context.assert_treatment_fitting_allowed()

        executable_entries = (
            plan.executable_entries()
        )

        if (
            not executable_entries
            and self.options.require_nonempty_plan
        ):
            raise ValueError(
                "The outlier-treatment plan is empty."
            )

        self.detector_engine.fit(
            frame=contract.train.X,
            plan=plan,
        )

        detector_states = self._detector_states()

        self.strategies = (
            StrategyFactory.create_many(
                entries=executable_entries,
                detector_states=detector_states,
                winsorize_options=(
                    self.options
                    .winsorization
                ),
            )
        )

        training_detection = (
            self.detector_engine.detect(
                frame=contract.train.X,
                partition=PartitionName.TRAIN,
            )
        )

        for entry in executable_entries:
            if entry.column not in self.strategies:
                raise KeyError(
                    "Treatment strategy is unavailable for "
                    f"feature {entry.column!r}."
                )

            strategy = self.strategies[
                entry.column
            ]

            strategy.fit(
                series=contract.train.X[
                    entry.column
                ],
                outlier_mask=(
                    training_detection.masks[
                        entry.column
                    ]
                ),
                mode=TreatmentExecutionMode.FIT,
            )

        if self.options.require_all_strategies_fitted:
            unfitted_columns = [
                column
                for column, strategy
                in self.strategies.items()
                if not strategy.is_fitted
            ]

            if unfitted_columns:
                raise RuntimeError(
                    "Some treatment strategies were not fitted: "
                    f"{unfitted_columns}"
                )

        self.plan = plan
        self._fitted = True

        return self

    def fit_transform(
        self,
        *,
        contract: OutlierContract,
        plan: TreatmentPlan,
    ) -> tuple[
        DatasetPartition,
        DatasetPartition,
        DatasetPartition,
        OutlierPipelineResult,
    ]:
        """Fit on training and transform every partition."""

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
                OutlierFitContext
                .training_transform()
            ),
        )

        (
            transformed_validation,
            validation_result,
        ) = self._transform_partition(
            partition=contract.validation,
            context=(
                OutlierFitContext
                .validation_transform()
            ),
        )

        (
            transformed_test,
            test_result,
        ) = self._transform_partition(
            partition=contract.test,
            context=(
                OutlierFitContext
                .test_transform(
                    authorized=(
                        self.options
                        .test_transform_authorized
                    )
                )
            ),
        )

        pipeline_result = OutlierPipelineResult(
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
            detector_states=(
                self._detector_states()
            ),
            treatment_states={
                column: strategy.get_state()
                for column, strategy
                in self.strategies.items()
            },
            indicator_mapping=(
                dict(
                    self.plan.indicator_columns
                )
                if self.plan is not None
                else {}
            ),
            fitted_on_training_only=True,
            metadata={
                "detector_count": len(
                    self.detector_engine.detectors
                ),
                "strategy_count": len(
                    self.strategies
                ),
                "partition_count": 3,
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
        partition: DatasetPartition,
        context: OutlierFitContext,
    ) -> tuple[
        DatasetPartition,
        PartitionPipelineResult,
    ]:
        """Transform an additional partition after fitting."""

        if not self._fitted:
            raise RuntimeError(
                "The outlier pipeline must be fitted before "
                "transforming additional partitions."
            )

        return self._transform_partition(
            partition=partition,
            context=context,
        )

    def _transform_partition(
        self,
        *,
        partition: DatasetPartition,
        context: OutlierFitContext,
    ) -> tuple[
        DatasetPartition,
        PartitionPipelineResult,
    ]:
        """Transform one partition with fitted state."""

        if not self._fitted:
            raise RuntimeError(
                "The outlier pipeline is not fitted."
            )

        if self.plan is None:
            raise RuntimeError(
                "The fitted treatment plan is unavailable."
            )

        (
            transformed,
            treatment_metadata,
            detection,
        ) = self.partition_transformer.transform(
            partition=partition,
            context=context,
            plan=self.plan,
            detector_engine=(
                self.detector_engine
            ),
            strategies=self.strategies,
        )

        partition_result = PartitionPipelineResult(
            partition=partition.name,
            detection=detection,
            treatment=treatment_metadata,
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
            partition_result,
        )

    def detector_states(
        self,
    ) -> dict[
        str,
        dict[str, Any],
    ]:
        """Return fitted detector states."""

        if not self._fitted:
            raise RuntimeError(
                "Detector states are unavailable before fitting."
            )

        return self._detector_states()

    def treatment_states(
        self,
    ) -> dict[
        str,
        TreatmentState,
    ]:
        """Return fitted treatment states."""

        if not self._fitted:
            raise RuntimeError(
                "Treatment states are unavailable before fitting."
            )

        return {
            column: strategy.get_state()
            for column, strategy
            in self.strategies.items()
        }

    def _detector_states(
        self,
    ) -> dict[
        str,
        dict[str, Any],
    ]:
        """Return detector states without requiring final fit flag."""

        return self.detector_engine.states()
