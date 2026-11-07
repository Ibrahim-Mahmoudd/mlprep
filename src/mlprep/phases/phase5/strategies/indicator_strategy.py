"""Outlier-indicator generation for Phase 5."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field


class OutlierIndicatorMetadata(BaseModel):
    """Metadata for one generated outlier indicator."""

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
    )

    source_column: str
    indicator_column: str

    row_count: int = Field(
        default=0,
        ge=0,
    )

    outlier_count: int = Field(
        default=0,
        ge=0,
    )

    outlier_percentage: float = Field(
        default=0.0,
        ge=0.0,
        le=100.0,
    )

    metadata: dict[str, Any] = Field(
        default_factory=dict
    )


@dataclass(slots=True)
class IndicatorStrategyOptions:
    """Options controlling outlier indicators."""

    suffix: str = "__was_outlier"

    dtype: str = "boolean"

    allow_existing_identical_indicator: bool = False

    def __post_init__(self) -> None:
        """Validate indicator options."""

        if not self.suffix:
            raise ValueError(
                "suffix cannot be empty."
            )

        if self.dtype not in {
            "boolean",
            "bool",
            "int8",
            "uint8",
        }:
            raise ValueError(
                "dtype must be 'boolean', 'bool', "
                "'int8', or 'uint8'."
            )


class IndicatorStrategy:
    """Create deterministic outlier indicator features."""

    strategy_name = "indicator"

    def __init__(
        self,
        options: IndicatorStrategyOptions | None = None,
    ) -> None:
        """Initialize the indicator strategy."""

        self.options = (
            options
            or IndicatorStrategyOptions()
        )

    def indicator_name(
        self,
        column: str,
        *,
        configured_name: str | None = None,
    ) -> str:
        """Return the indicator name for one feature."""

        if configured_name is not None:
            if not configured_name.strip():
                raise ValueError(
                    "configured_name cannot be empty."
                )

            return configured_name

        if not column.strip():
            raise ValueError(
                "column cannot be empty."
            )

        return (
            f"{column}"
            f"{self.options.suffix}"
        )

    def create(
        self,
        *,
        frame: pd.DataFrame,
        source_column: str,
        outlier_mask: pd.Series,
        indicator_column: str | None = None,
    ) -> tuple[
        pd.DataFrame,
        OutlierIndicatorMetadata,
    ]:
        """Add one outlier indicator to a feature frame."""

        if source_column not in frame.columns:
            raise KeyError(
                f"Source feature {source_column!r} "
                "does not exist."
            )

        if not frame.index.equals(
            outlier_mask.index
        ):
            raise ValueError(
                "The outlier mask index differs from "
                "the feature-frame index."
            )

        if outlier_mask.isna().any():
            raise ValueError(
                "The outlier mask contains missing values."
            )

        resolved_name = self.indicator_name(
            source_column,
            configured_name=indicator_column,
        )

        indicator_values = (
            outlier_mask.astype(bool)
        )

        transformed = frame.copy(
            deep=True
        )

        if resolved_name in transformed.columns:
            if not (
                self.options
                .allow_existing_identical_indicator
            ):
                raise ValueError(
                    "Outlier-indicator name collision: "
                    f"{resolved_name!r}"
                )

            existing_values = (
                transformed[
                    resolved_name
                ]
                .fillna(False)
                .astype(bool)
            )

            if not existing_values.equals(
                indicator_values
            ):
                raise ValueError(
                    "Existing outlier indicator does not "
                    "match the detected outlier mask."
                )

        else:
            transformed[
                resolved_name
            ] = self._cast(
                indicator_values
            )

        outlier_count = int(
            indicator_values.sum()
        )

        row_count = len(
            indicator_values
        )

        metadata = OutlierIndicatorMetadata(
            source_column=source_column,
            indicator_column=resolved_name,
            row_count=row_count,
            outlier_count=outlier_count,
            outlier_percentage=(
                outlier_count
                / row_count
                * 100.0
                if row_count > 0
                else 0.0
            ),
        )

        return (
            transformed,
            metadata,
        )

    def _cast(
        self,
        values: pd.Series,
    ) -> pd.Series:
        """Cast indicator values to the configured dtype."""

        if self.options.dtype == "boolean":
            return values.astype(
                "boolean"
            )

        if self.options.dtype == "bool":
            return values.astype(
                bool
            )

        if self.options.dtype == "int8":
            return values.astype(
                "int8"
            )

        return values.astype(
            "uint8"
        )
