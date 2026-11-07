"""Numeric plot generation for Phase 3."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from mlprep.phases.phase3.contract.column_selection import (
    AnalysisColumnGroups,
)
from mlprep.phases.phase3.contract.eda_contract import (
    EDAContract,
)
from mlprep.phases.phase3.visualization.plot_spec import (
    PlotSpec,
    PlotType,
)


@dataclass(slots=True)
class NumericPlotOptions:
    """Options controlling numeric plots."""

    enabled: bool = True
    create_histograms: bool = True
    create_box_plots: bool = True
    create_quantile_plots: bool = False
    histogram_bins: int = 30
    maximum_columns: int = 50
    maximum_sample_rows: int = 50_000
    image_format: str = "png"
    dpi: int = 120

    def __post_init__(self) -> None:
        """Validate numeric plot options."""

        if self.histogram_bins < 2:
            raise ValueError("histogram_bins must be at least two.")
        if self.maximum_columns < 1:
            raise ValueError("maximum_columns must be positive.")
        if self.maximum_sample_rows < 1:
            raise ValueError("maximum_sample_rows must be positive.")
        if self.dpi < 72:
            raise ValueError("dpi must be at least 72.")
        if not self.image_format.strip():
            raise ValueError("image_format cannot be empty.")


class NumericPlotGenerator:
    """Generate numeric plots from training features only."""

    component_name = "numeric_plots"

    def __init__(
        self,
        options: NumericPlotOptions | None = None,
    ) -> None:
        """Initialize the numeric plot generator."""

        self.options = options or NumericPlotOptions()

    def generate(
        self,
        *,
        contract: EDAContract,
        columns: AnalysisColumnGroups,
        output_directory: Path,
    ) -> list[PlotSpec]:
        """Generate configured numeric plots."""

        if not self.options.enabled:
            return []

        frame = contract.training_features()
        selected_columns = [
            column
            for column in columns.numeric
            if column in frame.columns
        ][: self.options.maximum_columns]
        plots: list[PlotSpec] = []

        for column in selected_columns:
            numeric = pd.to_numeric(
                frame[column],
                errors="coerce",
            ).replace([np.inf, -np.inf], np.nan).dropna()
            if len(numeric) > self.options.maximum_sample_rows:
                numeric = numeric.iloc[: self.options.maximum_sample_rows]
            if numeric.empty:
                continue
            if self.options.create_histograms:
                plots.append(
                    self._histogram(column, numeric, output_directory)
                )
            if self.options.create_box_plots:
                plots.append(self._box_plot(column, numeric, output_directory))
            if self.options.create_quantile_plots:
                plots.append(
                    self._quantile_plot(column, numeric, output_directory)
                )
        return plots

    def _histogram(
        self,
        column: str,
        values: pd.Series,
        output_directory: Path,
    ) -> PlotSpec:
        """Generate one numeric histogram."""

        import matplotlib.pyplot as plt

        spec = PlotSpec(
            plot_type=PlotType.NUMERIC_HISTOGRAM,
            title=f"Distribution of {column}",
            columns=[column],
            x_label=column,
            y_label="Frequency",
            sample_size=len(values),
            image_format=self.options.image_format,
        )
        figure, axis = plt.subplots()
        axis.hist(values, bins=self.options.histogram_bins)
        axis.set_title(spec.title)
        axis.set_xlabel(column)
        axis.set_ylabel("Frequency")
        return self._save(figure, spec, output_directory, column, "histogram")

    def _box_plot(
        self,
        column: str,
        values: pd.Series,
        output_directory: Path,
    ) -> PlotSpec:
        """Generate one numeric box plot."""

        import matplotlib.pyplot as plt

        spec = PlotSpec(
            plot_type=PlotType.NUMERIC_BOX,
            title=f"Box plot of {column}",
            columns=[column],
            x_label=column,
            sample_size=len(values),
            image_format=self.options.image_format,
        )
        figure, axis = plt.subplots()
        axis.boxplot(values, orientation="horizontal")
        axis.set_title(spec.title)
        axis.set_xlabel(column)
        axis.set_yticks([])
        return self._save(figure, spec, output_directory, column, "box")

    def _quantile_plot(
        self,
        column: str,
        values: pd.Series,
        output_directory: Path,
    ) -> PlotSpec:
        """Generate a quantile plot for one numeric feature."""

        import matplotlib.pyplot as plt

        spec = PlotSpec(
            plot_type=PlotType.NUMERIC_QUANTILE,
            title=f"Quantiles of {column}",
            columns=[column],
            x_label="Quantile",
            y_label=column,
            sample_size=len(values),
            image_format=self.options.image_format,
        )
        probabilities = np.linspace(0.0, 1.0, min(101, len(values)))
        quantiles = np.quantile(values.to_numpy(dtype=float), probabilities)
        figure, axis = plt.subplots()
        axis.plot(probabilities, quantiles)
        axis.set_title(spec.title)
        axis.set_xlabel("Quantile")
        axis.set_ylabel(column)
        return self._save(figure, spec, output_directory, column, "quantile")

    def _save(
        self,
        figure: object,
        spec: PlotSpec,
        output_directory: Path,
        column: str,
        suffix: str,
    ) -> PlotSpec:
        """Save a plot and return its completed specification."""

        import matplotlib.pyplot as plt

        output_path = output_directory / (
            f"{self._safe_name(column)}__{suffix}."
            f"{self.options.image_format}"
        )
        figure.tight_layout()
        figure.savefig(output_path, dpi=self.options.dpi)
        plt.close(figure)
        spec.mark_generated(output_path)
        return spec

    @staticmethod
    def _safe_name(value: str) -> str:
        """Convert a feature name into a stable filename component."""

        return "".join(
            character
            if character.isalnum() or character in {"_", "-"}
            else "_"
            for character in value
        )
