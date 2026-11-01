"""Fitted numeric transformation pipeline for Phase 6."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from mlprep.phases.phase6.contract.transformation_plan import (
    NumericScalingKind,
    NumericTransformationKind,
    TransformationPlan,
    TransformationPlanEntry,
)
from mlprep.phases.phase6.numeric.base import (
    BaseNumericTransformer,
    NumericExecutionMode,
    NumericTransformerState,
)
from mlprep.phases.phase6.numeric.box_cox_transformer import (
    BoxCoxTransformer,
    BoxCoxTransformerOptions,
)
from mlprep.phases.phase6.numeric.log_transformer import (
    LogTransformer,
    LogTransformerOptions,
)
from mlprep.phases.phase6.numeric.minmax_scaler import (
    MinMaxScaler,
    MinMaxScalerOptions,
)
from mlprep.phases.phase6.numeric.power_transformer import (
    PowerTransformer,
    PowerTransformerOptions,
)
from mlprep.phases.phase6.numeric.robust_scaler import (
    RobustScaler,
    RobustScalerOptions,
)
from mlprep.phases.phase6.numeric.standard_scaler import (
    StandardScaler,
    StandardScalerOptions,
)


class NumericColumnPipelineState(BaseModel):
    """Fitted numeric pipeline state for one feature."""

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        arbitrary_types_allowed=True,
    )

    column: str

    transformation: str
    scaling: str

    transformation_state: NumericTransformerState | None = None
    scaler_state: NumericTransformerState | None = None

    output_column: str

    fitted: bool = False
    fitted_on_training_only: bool = False

    metadata: dict[str, Any] = Field(
        default_factory=dict
    )


class NumericPipelineResult(BaseModel):
    """Output from transforming one numeric frame."""

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        arbitrary_types_allowed=True,
    )

    frame: pd.DataFrame

    output_columns: list[str] = Field(
        default_factory=list
    )

    states: dict[
        str,
        NumericColumnPipelineState,
    ] = Field(default_factory=dict)

    metadata: dict[str, Any] = Field(
        default_factory=dict
    )


@dataclass(slots=True)
class NumericPipelineOptions:
    """Options controlling the numeric pipeline."""

    log: LogTransformerOptions = field(
        default_factory=LogTransformerOptions
    )

    power: PowerTransformerOptions = field(
        default_factory=PowerTransformerOptions
    )

    box_cox: BoxCoxTransformerOptions = field(
        default_factory=BoxCoxTransformerOptions
    )

    standard: StandardScalerOptions = field(
        default_factory=StandardScalerOptions
    )

    robust: RobustScalerOptions = field(
        default_factory=RobustScalerOptions
    )

    minmax: MinMaxScalerOptions = field(
        default_factory=MinMaxScalerOptions
    )

    output_suffix: str = "__numeric"

    require_nonempty_numeric_plan: bool = False

    def __post_init__(self) -> None:
        """Validate numeric-pipeline options."""

        if not self.output_suffix:
            raise ValueError(
                "output_suffix cannot be empty."
            )


class NumericPipeline:
    """Fit and apply numeric transformation chains."""

    component_name = "numeric_pipeline"

    def __init__(
        self,
        options: NumericPipelineOptions | None = None,
    ) -> None:
        """Initialize the numeric pipeline."""

        self.options = (
            options
            or NumericPipelineOptions()
        )

        self.transformers: dict[
            str,
            BaseNumericTransformer | None,
        ] = {}

        self.scalers: dict[
            str,
            BaseNumericTransformer | None,
        ] = {}

        self.states: dict[
            str,
            NumericColumnPipelineState,
        ] = {}

        self.entries: dict[
            str,
            TransformationPlanEntry,
        ] = {}

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
        frame: pd.DataFrame,
        plan: TransformationPlan,
    ) -> NumericPipeline:
        """Fit every numeric chain on training data."""

        if self._fitted:
            raise RuntimeError(
                "The numeric pipeline is already fitted."
            )

        numeric_entries = [
            plan.entries[column]
            for column in plan.numeric_columns
        ]

        if (
            not numeric_entries
            and self.options
            .require_nonempty_numeric_plan
        ):
            raise ValueError(
                "The numeric transformation plan is empty."
            )

        for entry in numeric_entries:
            column = entry.column

            if column not in frame.columns:
                raise KeyError(
                    f"Numeric feature {column!r} "
                    "does not exist."
                )

            transformer = (
                self._build_transformer(
                    entry
                )
            )

            scaler = self._build_scaler(
                entry
            )

            training_values = frame[
                column
            ]

            transformed_training = (
                training_values.astype(
                    float
                )
            )

            transformation_state = None
            scaler_state = None

            if transformer is not None:
                transformed_training = (
                    transformer.fit_transform(
                        training_values
                    )
                )

                transformation_state = (
                    transformer.get_state()
                )

            if scaler is not None:
                scaler.fit(
                    transformed_training,
                    mode=NumericExecutionMode.FIT,
                )

                scaler_state = (
                    scaler.get_state()
                )

            output_column = self._output_column(
                entry
            )

            self.transformers[
                column
            ] = transformer

            self.scalers[
                column
            ] = scaler

            self.entries[
                column
            ] = entry

            self.states[
                column
            ] = NumericColumnPipelineState(
                column=column,
                transformation=(
                    entry
                    .numeric_transformation
                    .value
                ),
                scaling=(
                    entry.numeric_scaling.value
                ),
                transformation_state=(
                    transformation_state
                ),
                scaler_state=scaler_state,
                output_column=output_column,
                fitted=True,
                fitted_on_training_only=True,
            )

        self._fitted = True

        return self

    def transform(
        self,
        *,
        frame: pd.DataFrame,
        mode: NumericExecutionMode,
    ) -> NumericPipelineResult:
        """Transform numeric features using fitted state."""

        if not self._fitted:
            raise RuntimeError(
                "The numeric pipeline must be fitted "
                "before transformation."
            )

        if mode == NumericExecutionMode.FIT:
            raise ValueError(
                "Use fit() before transform()."
            )

        output = pd.DataFrame(
            index=frame.index
        )

        for column, _entry in (
            self.entries.items()
        ):
            if column not in frame.columns:
                raise KeyError(
                    f"Numeric feature {column!r} "
                    "does not exist."
                )

            values = frame[
                column
            ]

            transformer = self.transformers[
                column
            ]

            scaler = self.scalers[
                column
            ]

            transformed = values.astype(
                float
            )

            if transformer is not None:
                transformed = transformer.transform(
                    transformed,
                    mode=mode,
                )

            if scaler is not None:
                transformed = scaler.transform(
                    transformed,
                    mode=mode,
                )

            output_column = (
                self.states[
                    column
                ].output_column
            )

            output[
                output_column
            ] = transformed

        return NumericPipelineResult(
            frame=output,
            output_columns=[
                str(column)
                for column in output.columns
            ],
            states={
                column: state.model_copy(
                    deep=True
                )
                for column, state
                in self.states.items()
            },
            metadata={
                "feature_count": len(
                    output.columns
                ),
                "input_numeric_count": len(
                    self.entries
                ),
            },
        )

    def fit_transform(
        self,
        *,
        frame: pd.DataFrame,
        plan: TransformationPlan,
    ) -> NumericPipelineResult:
        """Fit and transform training numeric features."""

        self.fit(
            frame=frame,
            plan=plan,
        )

        return self.transform(
            frame=frame,
            mode=(
                NumericExecutionMode
                .TRAINING_TRANSFORM
            ),
        )

    def output_columns(
        self,
    ) -> list[object]:
        """Return fitted numeric output columns."""

        if not self._fitted:
            raise RuntimeError(
                "Output columns are unavailable before fitting."
            )

        return [
            state.output_column
            for state in self.states.values()
        ]

    def get_states(
        self,
    ) -> dict[
        str,
        NumericColumnPipelineState,
    ]:
        """Return fitted numeric pipeline states."""

        if not self._fitted:
            raise RuntimeError(
                "Numeric states are unavailable before fitting."
            )

        return {
            column: state.model_copy(
                deep=True
            )
            for column, state
            in self.states.items()
        }

    def _build_transformer(
        self,
        entry: TransformationPlanEntry,
    ) -> BaseNumericTransformer | None:
        """Build the configured numeric transformer."""

        method = entry.numeric_transformation

        if method == NumericTransformationKind.NONE:
            return None

        if method == NumericTransformationKind.LOG1P:
            return LogTransformer(
                column=entry.column,
                options=LogTransformerOptions(
                    method="log1p",
                    domain_tolerance=(
                        self.options
                        .log
                        .domain_tolerance
                    ),
                    preserve_float32=(
                        self.options
                        .log
                        .preserve_float32
                    ),
                ),
            )

        if (
            method
            == NumericTransformationKind.SIGNED_LOG1P
        ):
            return LogTransformer(
                column=entry.column,
                options=LogTransformerOptions(
                    method="signed_log1p",
                    domain_tolerance=(
                        self.options
                        .log
                        .domain_tolerance
                    ),
                    preserve_float32=(
                        self.options
                        .log
                        .preserve_float32
                    ),
                ),
            )

        if (
            method
            == NumericTransformationKind.YEO_JOHNSON
        ):
            return PowerTransformer(
                column=entry.column,
                options=self.options.power,
            )

        if method == NumericTransformationKind.BOX_COX:
            return BoxCoxTransformer(
                column=entry.column,
                options=self.options.box_cox,
            )

        raise ValueError(
            "Unsupported numeric transformation: "
            f"{method!r}"
        )

    def _build_scaler(
        self,
        entry: TransformationPlanEntry,
    ) -> BaseNumericTransformer | None:
        """Build the configured numeric scaler."""

        scaling = entry.numeric_scaling

        if scaling == NumericScalingKind.NONE:
            return None

        if scaling == NumericScalingKind.STANDARD:
            return StandardScaler(
                column=entry.column,
                options=self.options.standard,
            )

        if scaling == NumericScalingKind.ROBUST:
            return RobustScaler(
                column=entry.column,
                options=self.options.robust,
            )

        if scaling == NumericScalingKind.MINMAX:
            return MinMaxScaler(
                column=entry.column,
                options=self.options.minmax,
            )

        raise ValueError(
            f"Unsupported numeric scaler: {scaling!r}"
        )

    def _output_column(
        self,
        entry: TransformationPlanEntry,
    ) -> str:
        """Build one numeric output-column name."""

        prefix = (
            entry.output_prefix
            or entry.column
        )

        transformation = (
            entry
            .numeric_transformation
            .value
        )

        scaling = (
            entry.numeric_scaling.value
        )

        return (
            f"{prefix}"
            f"{self.options.output_suffix}"
            f"__{transformation}"
            f"__{scaling}"
        )
