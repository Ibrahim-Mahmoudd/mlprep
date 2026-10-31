"""Categorical plot generation for Phase 3."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

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
class CategoricalPlotOptions:
    """Options controlling categorical plots."""

    enabled: bool = True
    maximum_columns: int = 50
    maximum_categories: int = 30
    include_boolean_columns: bool = True
    include_missing_as_category: bool = True
    suppress_category_labels_for_sensitive_columns: bool = True
    image_format: str = "png"
    dpi: int = 120

    def __post_init__(self) -> None:
        """Validate categorical plot options."""

        if self.maximum_columns < 1:
            raise ValueError("maximum_columns must be positive.")
        if self.maximum_categories < 2:
            raise ValueError("maximum_categories must be at least two.")
        if self.dpi < 72:
            raise ValueError("dpi must be at least 72.")
        if not self.image_format.strip():
            raise ValueError("image_format cannot be empty.")


class CategoricalPlotGenerator:
    """Generate training-only categorical frequency plots."""

    component_name = "categorical_plots"

    def __init__(
        self,
        options: CategoricalPlotOptions | None = None,
    ) -> None:
        """Initialize the categorical plot generator."""

        self.options = options or CategoricalPlotOptions()

    def generate(
        self,
        *,
        contract: EDAContract,
        columns: AnalysisColumnGroups,
        output_directory: Path,
    ) -> list[PlotSpec]:
        """Generate categorical frequency charts."""

        if not self.options.enabled:
            return []

        frame = contract.training_features()
        selected = list(columns.categorical)
        if self.options.include_boolean_columns:
            selected.extend(columns.boolean)
        selected = [
            column
            for column in dict.fromkeys(selected)
            if column in frame.columns
        ][: self.options.maximum_columns]

        plots: list[PlotSpec] = []
        for column in selected:
            sensitive = column in columns.sensitive
            spec = PlotSpec(
                plot_type=PlotType.CATEGORICAL_FREQUENCY,
                title=f"Category frequencies for {column}",
                columns=[column],
                x_label="Category",
                y_label="Frequency",
                sensitive=sensitive,
                image_format=self.options.image_format,
            )
            values = frame[column].astype("object")
            if self.options.include_missing_as_category:
                values = values.where(values.notna(), "__MISSING__")
            else:
                values = values.dropna()
            counts = values.value_counts(dropna=False).head(
                self.options.maximum_categories
            )
            if counts.empty:
                spec.mark_skipped("The feature contains no usable values.")
                plots.append(spec)
                continue

            import matplotlib.pyplot as plt

            positions = list(range(len(counts)))
            figure, axis = plt.subplots()
            axis.bar(positions, counts.to_numpy())
            hide_labels = (
                sensitive
                and self.options.suppress_category_labels_for_sensitive_columns
            )
            labels = (
                [f"Category {index + 1}" for index in positions]
                if hide_labels
                else [str(value) for value in counts.index]
            )
            axis.set_xticks(positions, labels, rotation=45, ha="right")
            axis.set_title(spec.title)
            axis.set_xlabel("Category")
            axis.set_ylabel("Frequency")
            figure.tight_layout()
            output_path = self._output_path(output_directory, column)
            figure.savefig(output_path, dpi=self.options.dpi)
            plt.close(figure)
            spec.sample_size = int(counts.sum())
            spec.mark_generated(output_path)
            plots.append(spec)
        return plots

    def _output_path(self, output_directory: Path, column: str) -> Path:
        """Build a filesystem-safe categorical plot path."""

        safe_column = "".join(
            character
            if character.isalnum() or character in {"_", "-"}
            else "_"
            for character in column
        )
        return output_directory / (
            f"{safe_column}__categories.{self.options.image_format}"
        )
