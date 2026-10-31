"""Serializable plot specifications for Phase 3."""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field


class PlotType(StrEnum):
    """Supported Phase 3 plot types."""

    NUMERIC_HISTOGRAM = "numeric_histogram"
    NUMERIC_BOX = "numeric_box"
    NUMERIC_VIOLIN = "numeric_violin"
    NUMERIC_QUANTILE = "numeric_quantile"

    CATEGORICAL_FREQUENCY = (
        "categorical_frequency"
    )

    CATEGORICAL_PERCENTAGE = (
        "categorical_percentage"
    )

    CLASS_DISTRIBUTION = (
        "class_distribution"
    )

    REGRESSION_TARGET_HISTOGRAM = (
        "regression_target_histogram"
    )

    REGRESSION_TARGET_BOX = (
        "regression_target_box"
    )

    MISSINGNESS_BAR = "missingness_bar"
    MISSINGNESS_MATRIX = "missingness_matrix"
    MISSINGNESS_HEATMAP = (
        "missingness_heatmap"
    )

    CORRELATION_HEATMAP = (
        "correlation_heatmap"
    )

    SCATTER = "scatter"

    NUMERIC_BY_CATEGORY = (
        "numeric_by_category"
    )

    NUMERIC_BY_TARGET = (
        "numeric_by_target"
    )

    CATEGORY_BY_TARGET = (
        "category_by_target"
    )

    TARGET_OVER_TIME = "target_over_time"


class PlotStatus(StrEnum):
    """Generation status of one plot."""

    PLANNED = "planned"
    GENERATED = "generated"
    SKIPPED = "skipped"
    FAILED = "failed"


class PlotSpec(BaseModel):
    """Serializable specification for one plot."""

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
    )

    plot_id: str = Field(
        default_factory=lambda: str(uuid4())
    )

    plot_type: PlotType

    title: str

    columns: list[str] = Field(
        default_factory=list
    )

    target_column: str | None = None

    x_label: str | None = None
    y_label: str | None = None

    sample_size: int | None = Field(
        default=None,
        ge=0,
    )

    output_path: str | None = None

    image_format: str = "png"

    status: PlotStatus = PlotStatus.PLANNED

    sensitive: bool = False

    skip_reason: str | None = None
    error_message: str | None = None

    metadata: dict[str, Any] = Field(
        default_factory=dict
    )

    def mark_generated(
        self,
        path: str | Path,
    ) -> None:
        """Mark the plot as generated."""

        self.status = PlotStatus.GENERATED
        self.output_path = str(path)
        self.skip_reason = None
        self.error_message = None

    def mark_skipped(
        self,
        reason: str,
    ) -> None:
        """Mark the plot as skipped."""

        if not reason.strip():
            raise ValueError(
                "A plot skip reason cannot be empty."
            )

        self.status = PlotStatus.SKIPPED
        self.skip_reason = reason
        self.error_message = None

    def mark_failed(
        self,
        error: Exception | str,
    ) -> None:
        """Mark the plot as failed."""

        self.status = PlotStatus.FAILED
        self.error_message = str(error)


class PlotCollection(BaseModel):
    """Collection of Phase 3 plot specifications."""

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
    )

    plots: list[PlotSpec] = Field(
        default_factory=list
    )

    generated_count: int = Field(
        default=0,
        ge=0,
    )

    skipped_count: int = Field(
        default=0,
        ge=0,
    )

    failed_count: int = Field(
        default=0,
        ge=0,
    )

    metadata: dict[str, Any] = Field(
        default_factory=dict
    )

    def add(
        self,
        plot: PlotSpec,
    ) -> None:
        """Add one plot specification."""

        self.plots.append(plot)

    def refresh_counts(
        self,
    ) -> None:
        """Refresh generation-status counts."""

        self.generated_count = sum(
            plot.status == PlotStatus.GENERATED
            for plot in self.plots
        )

        self.skipped_count = sum(
            plot.status == PlotStatus.SKIPPED
            for plot in self.plots
        )

        self.failed_count = sum(
            plot.status == PlotStatus.FAILED
            for plot in self.plots
        )

    def generated_plots(
        self,
    ) -> list[PlotSpec]:
        """Return generated plots."""

        return [
            plot
            for plot in self.plots
            if plot.status == PlotStatus.GENERATED
        ]