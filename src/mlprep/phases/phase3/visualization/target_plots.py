"""Target plot generation for Phase 3."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from mlprep.phases.phase3.contract.eda_contract import (
    EDAContract,
)
from mlprep.phases.phase3.visualization.plot_spec import (
    PlotSpec,
    PlotType,
)
from mlprep.types import TaskType


@dataclass(slots=True)
class TargetPlotOptions:
    """Options controlling target plots."""

    enabled: bool = True

    histogram_bins: int = 30

    create_regression_box_plot: bool = True

    maximum_class_labels: int = 100

    image_format: str = "png"
    dpi: int = 120

    def __post_init__(self) -> None:
        """Validate target plot options."""

        if self.histogram_bins < 2:
            raise ValueError(
                "histogram_bins must be at least two."
            )

        if self.maximum_class_labels < 2:
            raise ValueError(
                "maximum_class_labels must be at least two."
            )

        if self.dpi < 72:
            raise ValueError(
                "dpi must be at least 72."
            )


class TargetPlotGenerator:
    """Generate plots for the training target."""

    component_name = "target_plots"

    def __init__(
        self,
        options: TargetPlotOptions | None = None,
    ) -> None:
        """Initialize the target plot generator."""

        self.options = (
            options or TargetPlotOptions()
        )

    def generate(
        self,
        *,
        contract: EDAContract,
        output_directory: Path,
    ) -> list[object]:
        """Generate target plots from training data."""

        if not self.options.enabled:
            return []

        problem_contract = (
            contract.phase1_state
            .problem_contract
        )

        if problem_contract is None:
            return []

        target = contract.training_target()

        if (
            problem_contract.task_type
            == TaskType.REGRESSION
        ):
            return self._regression_plots(
                target=target,
                target_column=(
                    contract.target_column
                ),
                output_directory=(
                    output_directory
                ),
            )

        return [
            self._classification_plot(
                target=target,
                target_column=(
                    contract.target_column
                ),
                output_directory=(
                    output_directory
                ),
            )
        ]

    def _classification_plot(
        self,
        *,
        target: pd.Series,
        target_column: str,
        output_directory: Path,
    ) -> PlotSpec:
        """Generate a classification target chart."""

        spec = PlotSpec(
            plot_type=(
                PlotType.CLASS_DISTRIBUTION
            ),
            title=(
                f"Class distribution of "
                f"{target_column}"
            ),
            columns=[],
            target_column=target_column,
            x_label="Class",
            y_label="Frequency",
            image_format=self.options.image_format,
        )

        counts = target.value_counts(
            dropna=False
        ).head(
            self.options.maximum_class_labels
        )

        output_path = output_directory / (
            f"{self._safe_name(target_column)}"
            f"__class_distribution."
            f"{self.options.image_format}"
        )

        try:
            figure, axis = plt.subplots()

            positions = list(
                range(len(counts))
            )

            axis.bar(
                positions,
                counts.to_numpy(),
            )

            axis.set_xticks(
                positions
            )

            axis.set_xticklabels(
                [
                    str(value)
                    for value in counts.index
                ],
                rotation=45,
                ha="right",
            )

            axis.set_title(spec.title)
            axis.set_xlabel("Class")
            axis.set_ylabel("Frequency")

            figure.tight_layout()

            figure.savefig(
                output_path,
                dpi=self.options.dpi,
            )

            plt.close(figure)

            spec.sample_size = int(
                counts.sum()
            )

            spec.mark_generated(
                output_path
            )

        except Exception as error:
            plt.close("all")
            spec.mark_failed(error)

        return spec

    def _regression_plots(
        self,
        *,
        target: pd.Series,
        target_column: str,
        output_directory: Path,
    ) -> list[object]:
        """Generate regression target plots."""

        numeric = pd.to_numeric(
            target,
            errors="coerce",
        ).replace(
            [np.inf, -np.inf],
            np.nan,
        ).dropna()

        if numeric.empty:
            return []

        plots = [
            self._regression_histogram(
                target=numeric,
                target_column=target_column,
                output_directory=(
                    output_directory
                ),
            )
        ]

        if self.options.create_regression_box_plot:
            plots.append(
                self._regression_box(
                    target=numeric,
                    target_column=target_column,
                    output_directory=(
                        output_directory
                    ),
                )
            )

        return plots

    def _regression_histogram(
        self,
        *,
        target: pd.Series,
        target_column: str,
        output_directory: Path,
    ) -> PlotSpec:
        """Generate a regression target histogram."""

        spec = PlotSpec(
            plot_type=(
                PlotType
                .REGRESSION_TARGET_HISTOGRAM
            ),
            title=(
                f"Distribution of "
                f"{target_column}"
            ),
            target_column=target_column,
            x_label=target_column,
            y_label="Frequency",
            sample_size=len(target),
            image_format=self.options.image_format,
        )

        output_path = output_directory / (
            f"{self._safe_name(target_column)}"
            f"__target_histogram."
            f"{self.options.image_format}"
        )

        try:
            figure, axis = plt.subplots()

            axis.hist(
                target,
                bins=self.options.histogram_bins,
            )

            axis.set_title(spec.title)
            axis.set_xlabel(target_column)
            axis.set_ylabel("Frequency")

            figure.tight_layout()

            figure.savefig(
                output_path,
                dpi=self.options.dpi,
            )

            plt.close(figure)

            spec.mark_generated(
                output_path
            )

        except Exception as error:
            plt.close("all")
            spec.mark_failed(error)

        return spec

    def _regression_box(
        self,
        *,
        target: pd.Series,
        target_column: str,
        output_directory: Path,
    ) -> PlotSpec:
        """Generate a regression target box plot."""

        spec = PlotSpec(
            plot_type=(
                PlotType.REGRESSION_TARGET_BOX
            ),
            title=(
                f"Box plot of "
                f"{target_column}"
            ),
            target_column=target_column,
            x_label=target_column,
            sample_size=len(target),
            image_format=self.options.image_format,
        )

        output_path = output_directory / (
            f"{self._safe_name(target_column)}"
            f"__target_box."
            f"{self.options.image_format}"
        )

        try:
            figure, axis = plt.subplots()

            axis.boxplot(
                target,
                vert=False,
            )

            axis.set_title(spec.title)
            axis.set_xlabel(target_column)
            axis.set_yticks([])

            figure.tight_layout()

            figure.savefig(
                output_path,
                dpi=self.options.dpi,
            )

            plt.close(figure)

            spec.mark_generated(
                output_path
            )

        except Exception as error:
            plt.close("all")
            spec.mark_failed(error)

        return spec

    @staticmethod
    def _safe_name(
        value: str,
    ) -> str:
        """Convert text into a safe file name."""

        return "".join(
            character
            if character.isalnum()
            or character in {
                "_",
                "-",
            }
            else "_"
            for character in value
        )