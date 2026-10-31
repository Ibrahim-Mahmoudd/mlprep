"""Frequency and count encoding for Phase 6."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field


FrequencyEncodingMode = Literal[
    "frequency",
    "count",
]


class FrequencyEncoderState(BaseModel):
    """Serializable frequency encoder state."""

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
    )

    column: str

    fitted: bool = False
    fitted_on_training_only: bool = False

    mode: str

    mapping: dict[
        str,
        float,
    ] = Field(default_factory=dict)

    unknown_value: float = 0.0
    missing_token: str = "__MISSING__"

    output_column: str

    training_row_count: int = Field(
        default=0,
        ge=0,
    )

    metadata: dict[str, Any] = Field(
        default_factory=dict
    )


@dataclass(slots=True)
class FrequencyEncoderOptions:
    """Options controlling frequency encoding."""

    mode: FrequencyEncodingMode = "frequency"

    unknown_value: float = 0.0

    missing_token: str = "__MISSING__"

    output_dtype: str = "float64"

    suffix: str = "__frequency"

    def __post_init__(self) -> None:
        """Validate frequency encoding options."""

        if self.mode not in {
            "frequency",
            "count",
        }:
            raise ValueError(
                "mode must be 'frequency' or 'count'."
            )

        if not self.missing_token:
            raise ValueError(
                "missing_token cannot be empty."
            )

        if not self.suffix:
            raise ValueError(
                "suffix cannot be empty."
            )


class FrequencyEncoder:
    """Map categories to training counts or frequencies."""

    component_name = "frequency_encoder"

    def __init__(
        self,
        *,
        column: str,
        output_prefix: str | None = None,
        options: FrequencyEncoderOptions | None = None,
    ) -> None:
        """Initialize the frequency encoder."""

        if not column.strip():
            raise ValueError(
                "column cannot be empty."
            )

        self.column = column
        self.output_prefix = (
            output_prefix or column
        )

        self.options = (
            options
            or FrequencyEncoderOptions()
        )

        self.state = FrequencyEncoderState(
            column=column,
            mode=self.options.mode,
            unknown_value=(
                self.options.unknown_value
            ),
            missing_token=(
                self.options.missing_token
            ),
            output_column=(
                f"{self.output_prefix}"
                f"{self.options.suffix}"
            ),
        )

    @property
    def is_fitted(
        self,
    ) -> bool:
        """Return whether the encoder is fitted."""

        return self.state.fitted

    def fit(
        self,
        series: pd.Series,
    ) -> FrequencyEncoder:
        """Learn category frequencies from training."""

        normalized = self._normalize(
            series
        )

        counts = normalized.value_counts(
            dropna=False
        )

        if self.options.mode == "frequency":
            denominator = max(
                len(normalized),
                1,
            )

            mapping = {
                str(category): (
                    float(count)
                    / denominator
                )
                for category, count
                in counts.items()
            }

        else:
            mapping = {
                str(category): float(
                    count
                )
                for category, count
                in counts.items()
            }

        self.state.mapping = mapping

        self.state.training_row_count = len(
            normalized
        )

        self.state.fitted = True
        self.state.fitted_on_training_only = True

        return self

    def transform(
        self,
        series: pd.Series,
    ) -> pd.DataFrame:
        """Apply fitted frequency mapping."""

        if not self.is_fitted:
            raise RuntimeError(
                "The frequency encoder is not fitted."
            )

        normalized = self._normalize(
            series
        )

        encoded = normalized.map(
            self.state.mapping
        ).fillna(
            self.options.unknown_value
        )

        return pd.DataFrame(
            {
                self.state.output_column: (
                    encoded.astype(
                        self.options.output_dtype
                    )
                )
            },
            index=series.index,
        )

    def fit_transform(
        self,
        series: pd.Series,
    ) -> pd.DataFrame:
        """Fit and transform training categories."""

        self.fit(
            series
        )

        return self.transform(
            series
        )

    def unknown_mask(
        self,
        series: pd.Series,
    ) -> pd.Series:
        """Return categories absent from fitted mapping."""

        if not self.is_fitted:
            raise RuntimeError(
                "The frequency encoder is not fitted."
            )

        normalized = self._normalize(
            series
        )

        return ~normalized.isin(
            set(
                self.state.mapping
            )
        )

    def get_state(
        self,
    ) -> FrequencyEncoderState:
        """Return a copy of fitted state."""

        return self.state.model_copy(
            deep=True
        )

    def _normalize(
        self,
        series: pd.Series,
    ) -> pd.Series:
        """Normalize categorical values."""

        if not isinstance(
            series,
            pd.Series,
        ):
            raise TypeError(
                "FrequencyEncoder input must be "
                "a pandas Series."
            )

        if (
            series.name is not None
            and str(series.name) != self.column
        ):
            raise ValueError(
                f"Expected column {self.column!r}, "
                f"received {series.name!r}."
            )

        return (
            series.astype(
                "string"
            )
            .fillna(
                self.options.missing_token
            )
        )