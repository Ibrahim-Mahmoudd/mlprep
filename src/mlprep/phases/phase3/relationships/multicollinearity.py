"""Multicollinearity diagnostics for Phase 3."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from mlprep.phases.phase3.contract.column_selection import (
    AnalysisColumnGroups,
)
from mlprep.phases.phase3.contract.eda_contract import (
    EDAContract,
)


class VIFResult(BaseModel):
    """Variance inflation factor for one feature."""

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
    )

    column: str

    r_squared: float | None = None
    vif: float | None = None

    high_multicollinearity: bool = False
    perfect_multicollinearity: bool = False


class MulticollinearityResult(BaseModel):
    """Complete multicollinearity output."""

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
    )

    analyzed_columns: list[str] = Field(
        default_factory=list
    )

    vif_results: list[VIFResult] = Field(
        default_factory=list
    )

    condition_number: float | None = None
    matrix_rank: int | None = None

    singular: bool = False

    high_vif_columns: list[str] = Field(
        default_factory=list
    )

    metadata: dict[str, object] = Field(
        default_factory=dict
    )


@dataclass(slots=True)
class MulticollinearityOptions:
    """Options controlling multicollinearity diagnostics."""

    enabled: bool = True

    maximum_columns: int = 100

    minimum_complete_rows: int = 10

    high_vif_threshold: float = 10.0
    perfect_vif_threshold: float = 1_000_000.0

    fill_missing_with_median: bool = True
    standardize_before_condition_number: bool = True

    def __post_init__(self) -> None:
        """Validate multicollinearity options."""

        if self.maximum_columns < 2:
            raise ValueError(
                "maximum_columns must be at least two."
            )

        if self.minimum_complete_rows < 2:
            raise ValueError(
                "minimum_complete_rows must be at least two."
            )

        if self.high_vif_threshold <= 1.0:
            raise ValueError(
                "high_vif_threshold must be greater than one."
            )

        if (
            self.perfect_vif_threshold
            < self.high_vif_threshold
        ):
            raise ValueError(
                "perfect_vif_threshold must not be below "
                "high_vif_threshold."
            )


class MulticollinearityAnalyzer:
    """Analyze linear redundancy among numeric features."""

    component_name = "multicollinearity"

    def __init__(
        self,
        options: MulticollinearityOptions | None = None,
    ) -> None:
        """Initialize the analyzer."""

        self.options = (
            options or MulticollinearityOptions()
        )

    def analyze(
        self,
        *,
        contract: EDAContract,
        columns: AnalysisColumnGroups,
    ) -> MulticollinearityResult:
        """Calculate VIF and matrix-condition diagnostics."""

        if not self.options.enabled:
            return MulticollinearityResult()

        selected_columns = [
            column
            for column in columns.numeric
            if column in contract.train.X.columns
        ][: self.options.maximum_columns]

        if len(selected_columns) < 2:
            return MulticollinearityResult(
                analyzed_columns=selected_columns
            )

        frame = contract.train.X.loc[
            :,
            selected_columns,
        ].apply(
            pd.to_numeric,
            errors="coerce",
        )

        frame = frame.replace(
            [np.inf, -np.inf],
            np.nan,
        )

        if self.options.fill_missing_with_median:
            frame = frame.fillna(
                frame.median(
                    numeric_only=True
                )
            )
        else:
            frame = frame.dropna()

        constant_columns = [
            column
            for column in frame.columns
            if frame[column].nunique() <= 1
        ]

        frame = frame.drop(
            columns=constant_columns
        )

        if (
            len(frame) < self.options.minimum_complete_rows
            or len(frame.columns) < 2
        ):
            return MulticollinearityResult(
                analyzed_columns=[
                    str(column)
                    for column in frame.columns
                ],
                metadata={
                    "excluded_constant_columns": (
                        constant_columns
                    ),
                },
            )

        values = frame.to_numpy(
            dtype=float
        )

        vif_results: list[VIFResult] = []

        for column_position, column in enumerate(
            frame.columns
        ):
            y = values[
                :,
                column_position
            ]

            X = np.delete(
                values,
                column_position,
                axis=1,
            )

            X = np.column_stack(
                [
                    np.ones(len(X)),
                    X,
                ]
            )

            coefficients, _, _, _ = (
                np.linalg.lstsq(
                    X,
                    y,
                    rcond=None,
                )
            )

            predictions = X @ coefficients

            residual_sum_squares = float(
                (
                    (y - predictions) ** 2
                ).sum()
            )

            total_sum_squares = float(
                (
                    (y - y.mean()) ** 2
                ).sum()
            )

            if total_sum_squares <= 0:
                r_squared = None
                vif = None
            else:
                r_squared = max(
                    0.0,
                    min(
                        1.0,
                        1.0
                        - residual_sum_squares
                        / total_sum_squares,
                    ),
                )

                if r_squared >= 1.0 - 1e-12:
                    vif = float("inf")
                else:
                    vif = (
                        1.0
                        / (1.0 - r_squared)
                    )

            vif_results.append(
                VIFResult(
                    column=str(column),
                    r_squared=r_squared,
                    vif=(
                        None
                        if vif is None
                        or not np.isfinite(vif)
                        else float(vif)
                    ),
                    high_multicollinearity=bool(
                        vif is not None
                        and vif
                        >= self.options
                        .high_vif_threshold
                    ),
                    perfect_multicollinearity=bool(
                        vif is not None
                        and (
                            not np.isfinite(vif)
                            or vif
                            >= self.options
                            .perfect_vif_threshold
                        )
                    ),
                )
            )

        condition_values = values.copy()

        if (
            self.options
            .standardize_before_condition_number
        ):
            means = condition_values.mean(
                axis=0
            )

            standard_deviations = (
                condition_values.std(
                    axis=0,
                    ddof=0,
                )
            )

            standard_deviations[
                standard_deviations == 0
            ] = 1.0

            condition_values = (
                condition_values - means
            ) / standard_deviations

        singular_values = np.linalg.svd(
            condition_values,
            compute_uv=False,
        )

        if (
            singular_values.size == 0
            or singular_values[-1] <= 1e-15
        ):
            condition_number = None
        else:
            condition_number = float(
                singular_values[0]
                / singular_values[-1]
            )

        matrix_rank = int(
            np.linalg.matrix_rank(
                condition_values
            )
        )

        singular = (
            matrix_rank
            < condition_values.shape[1]
        )

        return MulticollinearityResult(
            analyzed_columns=[
                str(column)
                for column in frame.columns
            ],
            vif_results=vif_results,
            condition_number=condition_number,
            matrix_rank=matrix_rank,
            singular=singular,
            high_vif_columns=[
                result.column
                for result in vif_results
                if result.high_multicollinearity
            ],
            metadata={
                "excluded_constant_columns": (
                    constant_columns
                ),
                "row_count_used": len(frame),
            },
        )