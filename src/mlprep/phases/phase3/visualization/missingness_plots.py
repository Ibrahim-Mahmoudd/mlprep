"""Missingness plot generation for Phase 3."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt

from mlprep.phases.phase3.contract.eda_contract import (
    EDAContract,
)
from mlprep.phases.phase3.visualization.plot_spec import (
    PlotSpec,
    PlotType,
)


@dataclass(slots=True)
class MissingnessPlotOptions:
    """Options controlling missingness plots."""

    enabled: bool = True

    create_bar_chart: bool = True
    create_matrix: bool = True
    create_correlation_heatmap: bool = True

    maximum_columns: int = 100
    maximum_matrix_rows: int = 2_000

    image_format: str = "png"
    dpi: int = 120

    def __post_init__(self) -> None:
        """Validate missingness plot options."""

        if self.maximum_columns < 1:
            raise ValueError(
                "maximum_columns must be positive."
            )

        if self.maximum_matrix_rows < 1:
            raise ValueError(
                "maximum_matrix_rows must be positive."
            )

        if self.dpi < 72:
            raise ValueError(
                "dpi must be at least 72."
            )


class MissingnessPlotGenerator:
    """Generate missingness plots from training data."""

    component_name = "missingness_plots"

    def __init__(
        self,
        options: (
            MissingnessPlotOptions | None
        ) = None,
    ) -> None:
        """Initialize the missingness plot generator."""

        self.options = (
            options or MissingnessPlotOptions()
        )

    def generate(
        self,
        *,
        contract: EDAContract,
        output_directory: Path,
    ) -> list[object]:
        """Generate configured missingness plots."""

        if not self.options.enabled:
            return []

        frame = contract.training_features()

        selected_columns = list(
            frame.columns[
                frame.isna().any()
            ]
        )[
            : self.options.maximum_columns
        ]

        if not selected_columns:
            return []

        missing_mask = frame.loc[
            :,
            selected_columns,
        ].isna()

        plots: list[PlotSpec] = []

        if self.options.create_bar_chart:
            plots.append(
                self._bar_plot(
                    mask=missing_mask,
                    output_directory=(
                        output_directory
                    ),
                )
            )

        if self.options.create_matrix:
            plots.append(
                self._matrix_plot(
                    mask=missing_mask,
                    output_directory=(
                        output_directory
                    ),
                )
            )

        if (
            self.options
            .create_correlation_heatmap
            and len(selected_columns) > 1
        ):
            plots.append(
                self._heatmap(
                    mask=missing_mask,
                    output_directory=(
                        output_directory
                    ),
                )
            )

        return plots

    def _bar_plot(
        self,
        *,
        mask,
        output_directory: Path,
    ) -> PlotSpec:
        """Generate a missingness bar chart."""

        spec = PlotSpec(
            plot_type=PlotType.MISSINGNESS_BAR,
            title="Training feature missingness",
            columns=[
                str(column)
                for column in mask.columns
            ],
            x_label="Feature",
            y_label="Missing percentage",
            sample_size=len(mask),
            image_format=self.options.image_format,
        )

        output_path = output_directory / (
            f"missingness__bar."
            f"{self.options.image_format}"
        )

        try:
            percentages = (
                mask.mean() * 100.0
            ).sort_values(
                ascending=False
            )

            figure, axis = plt.subplots()

            positions = list(
                range(len(percentages))
            )

            axis.bar(
                positions,
                percentages.to_numpy(),
            )

            axis.set_xticks(
                positions
            )

            axis.set_xticklabels(
                [
                    str(column)
                    for column in percentages.index
                ],
                rotation=45,
                ha="right",
            )

            axis.set_title(spec.title)
            axis.set_xlabel("Feature")
            axis.set_ylabel("Missing percentage")

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

    def _matrix_plot(
        self,
        *,
        mask,
        output_directory: Path,
    ) -> PlotSpec:
        """Generate a row-level missingness matrix."""

        sampled_mask = mask

        if (
            len(mask)
            > self.options.maximum_matrix_rows
        ):
            sampled_mask = mask.sample(
                n=self.options.maximum_matrix_rows,
                random_state=42,
            )

        spec = PlotSpec(
            plot_type=(
                PlotType.MISSINGNESS_MATRIX
            ),
            title="Training missingness matrix",
            columns=[
                str(column)
                for column in mask.columns
            ],
            x_label="Feature",
            y_label="Sampled row",
            sample_size=len(sampled_mask),
            image_format=self.options.image_format,
        )

        output_path = output_directory / (
            f"missingness__matrix."
            f"{self.options.image_format}"
        )

        try:
            figure, axis = plt.subplots()

            axis.imshow(
                sampled_mask.to_numpy(
                    dtype=int
                ),
                aspect="auto",
                interpolation="nearest",
            )

            axis.set_xticks(
                list(
                    range(
                        len(
                            sampled_mask.columns
                        )
                    )
                )
            )

            axis.set_xticklabels(
                [
                    str(column)
                    for column
                    in sampled_mask.columns
                ],
                rotation=45,
                ha="right",
            )

            axis.set_title(spec.title)
            axis.set_xlabel("Feature")
            axis.set_ylabel("Sampled row")

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

    def _heatmap(
        self,
        *,
        mask,
        output_directory: Path,
    ) -> PlotSpec:
        """Generate a missingness correlation heatmap."""

        useful_columns = [
            column
            for column in mask.columns
            if mask[column].nunique() > 1
        ]

        spec = PlotSpec(
            plot_type=(
                PlotType.MISSINGNESS_HEATMAP
            ),
            title=(
                "Missingness indicator correlations"
            ),
            columns=[
                str(column)
                for column in useful_columns
            ],
            image_format=self.options.image_format,
        )

        if len(useful_columns) < 2:
            spec.mark_skipped(
                "Fewer than two variable missingness "
                "indicators are available."
            )

            return spec

        output_path = output_directory / (
            f"missingness__heatmap."
            f"{self.options.image_format}"
        )

        try:
            correlation = (
                mask.loc[
                    :,
                    useful_columns,
                ]
                .astype(int)
                .corr()
            )

            figure, axis = plt.subplots()

            image = axis.imshow(
                correlation.to_numpy(),
                vmin=-1.0,
                vmax=1.0,
                aspect="auto",
            )

            axis.set_xticks(
                list(
                    range(
                        len(useful_columns)
                    )
                )
            )

            axis.set_yticks(
                list(
                    range(
                        len(useful_columns)
                    )
                )
            )

            axis.set_xticklabels(
                useful_columns,
                rotation=45,
                ha="right",
            )

            axis.set_yticklabels(
                useful_columns
            )

            axis.set_title(spec.title)

            figure.colorbar(
                image,
                ax=axis,
            )

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