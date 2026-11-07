"""Relationship plot generation for Phase 3."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from mlprep.phases.phase3.contract.column_selection import (
    AnalysisColumnGroups,
)
from mlprep.phases.phase3.contract.eda_contract import (
    EDAContract,
)
from mlprep.phases.phase3.relationships.numeric_associations import (
    NumericAssociationResult,
)
from mlprep.phases.phase3.relationships.target_associations import (
    TargetAssociationResult,
)
from mlprep.phases.phase3.visualization.plot_spec import (
    PlotSpec,
    PlotType,
)


@dataclass(slots=True)
class RelationshipPlotOptions:
    """Options controlling relationship plots."""

    enabled: bool = True

    create_correlation_heatmap: bool = True
    create_numeric_scatter_plots: bool = True
    create_target_association_plots: bool = True

    maximum_scatter_plots: int = 20
    maximum_target_plots: int = 20
    maximum_sample_rows: int = 10_000

    minimum_scatter_correlation: float = 0.50

    image_format: str = "png"
    dpi: int = 120

    def __post_init__(self) -> None:
        """Validate relationship plot options."""

        if self.maximum_scatter_plots < 0:
            raise ValueError(
                "maximum_scatter_plots cannot be negative."
            )

        if self.maximum_target_plots < 0:
            raise ValueError(
                "maximum_target_plots cannot be negative."
            )

        if self.maximum_sample_rows < 1:
            raise ValueError(
                "maximum_sample_rows must be positive."
            )

        if not (
            0.0
            <= self.minimum_scatter_correlation
            <= 1.0
        ):
            raise ValueError(
                "minimum_scatter_correlation must be "
                "between zero and one."
            )


class RelationshipPlotGenerator:
    """Generate plots from calculated relationships."""

    component_name = "relationship_plots"

    def __init__(
        self,
        options: (
            RelationshipPlotOptions | None
        ) = None,
    ) -> None:
        """Initialize the relationship plot generator."""

        self.options = (
            options or RelationshipPlotOptions()
        )

    def generate(
        self,
        *,
        contract: EDAContract,
        columns: AnalysisColumnGroups,
        numeric_associations: NumericAssociationResult,
        target_associations: TargetAssociationResult,
        output_directory: Path,
    ) -> list[object]:
        """Generate configured relationship plots."""

        if not self.options.enabled:
            return []

        plots: list[PlotSpec] = []

        if (
            self.options
            .create_correlation_heatmap
        ):
            correlation_plot = (
                self._correlation_heatmap(
                    numeric_associations=(
                        numeric_associations
                    ),
                    output_directory=(
                        output_directory
                    ),
                )
            )

            if correlation_plot is not None:
                plots.append(
                    correlation_plot
                )

        if (
            self.options
            .create_numeric_scatter_plots
        ):
            plots.extend(
                self._scatter_plots(
                    contract=contract,
                    numeric_associations=(
                        numeric_associations
                    ),
                    output_directory=(
                        output_directory
                    ),
                )
            )

        if (
            self.options
            .create_target_association_plots
        ):
            plots.extend(
                self._target_plots(
                    contract=contract,
                    columns=columns,
                    target_associations=(
                        target_associations
                    ),
                    output_directory=(
                        output_directory
                    ),
                )
            )

        return plots

    def _correlation_heatmap(
        self,
        *,
        numeric_associations: NumericAssociationResult,
        output_directory: Path,
    ) -> PlotSpec | None:
        """Generate a numeric correlation heatmap."""

        matrix_values = (
            numeric_associations
            .correlation_matrices
            .get(
                numeric_associations.primary_method
            )
        )

        if not matrix_values:
            return None

        matrix = pd.DataFrame(
            matrix_values
        ).T

        spec = PlotSpec(
            plot_type=(
                PlotType.CORRELATION_HEATMAP
            ),
            title=(
                f"{numeric_associations.primary_method.title()} "
                "correlation heatmap"
            ),
            columns=list(
                matrix.columns
            ),
            image_format=self.options.image_format,
        )

        output_path = output_directory / (
            f"numeric__correlation_heatmap."
            f"{self.options.image_format}"
        )

        try:
            numeric_matrix = matrix.astype(
                float
            )

            figure, axis = plt.subplots()

            image = axis.imshow(
                numeric_matrix.to_numpy(),
                vmin=-1.0,
                vmax=1.0,
                aspect="auto",
            )

            axis.set_xticks(
                list(
                    range(
                        len(matrix.columns)
                    )
                )
            )

            axis.set_yticks(
                list(
                    range(
                        len(matrix.index)
                    )
                )
            )

            axis.set_xticklabels(
                matrix.columns,
                rotation=45,
                ha="right",
            )

            axis.set_yticklabels(
                matrix.index
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

    def _scatter_plots(
        self,
        *,
        contract: EDAContract,
        numeric_associations: NumericAssociationResult,
        output_directory: Path,
    ) -> list[object]:
        """Generate scatter plots for strong numeric pairs."""

        frame = contract.training_features()

        eligible_pairs = [
            pair
            for pair
            in numeric_associations.pairs
            if (
                pair.absolute_primary_correlation
                is not None
                and pair.absolute_primary_correlation
                >= self.options
                .minimum_scatter_correlation
            )
        ][
            : self.options.maximum_scatter_plots
        ]

        plots: list[PlotSpec] = []

        for pair in eligible_pairs:
            pair_frame = frame.loc[
                :,
                [
                    pair.left_column,
                    pair.right_column,
                ],
            ].apply(
                pd.to_numeric,
                errors="coerce",
            ).replace(
                [np.inf, -np.inf],
                np.nan,
            ).dropna()

            if (
                len(pair_frame)
                > self.options.maximum_sample_rows
            ):
                pair_frame = pair_frame.sample(
                    n=self.options.maximum_sample_rows,
                    random_state=42,
                )

            spec = PlotSpec(
                plot_type=PlotType.SCATTER,
                title=(
                    f"{pair.left_column} versus "
                    f"{pair.right_column}"
                ),
                columns=[
                    pair.left_column,
                    pair.right_column,
                ],
                x_label=pair.left_column,
                y_label=pair.right_column,
                sample_size=len(pair_frame),
                image_format=self.options.image_format,
            )

            output_path = output_directory / (
                f"{self._safe_name(pair.left_column)}"
                f"__vs__"
                f"{self._safe_name(pair.right_column)}."
                f"{self.options.image_format}"
            )

            try:
                figure, axis = plt.subplots()

                axis.scatter(
                    pair_frame[
                        pair.left_column
                    ],
                    pair_frame[
                        pair.right_column
                    ],
                    s=10,
                    alpha=0.6,
                )

                axis.set_title(spec.title)
                axis.set_xlabel(
                    pair.left_column
                )
                axis.set_ylabel(
                    pair.right_column
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

            plots.append(spec)

        return plots

    def _target_plots(
        self,
        *,
        contract: EDAContract,
        columns: AnalysisColumnGroups,
        target_associations: TargetAssociationResult,
        output_directory: Path,
    ) -> list[object]:
        """Generate plots for strongest target associations."""

        frame = contract.training_features()
        target = contract.training_target()

        associations = (
            target_associations.associations[
                : self.options.maximum_target_plots
            ]
        )

        plots: list[PlotSpec] = []

        for association in associations:
            feature = association.feature

            if feature not in frame.columns:
                continue

            if feature in columns.numeric:
                plot = self._numeric_target_plot(
                    feature=feature,
                    values=frame[feature],
                    target=target,
                    target_column=(
                        contract.target_column
                    ),
                    output_directory=(
                        output_directory
                    ),
                )

            else:
                plot = self._categorical_target_plot(
                    feature=feature,
                    values=frame[feature],
                    target=target,
                    target_column=(
                        contract.target_column
                    ),
                    output_directory=(
                        output_directory
                    ),
                )

            plots.append(plot)

        return plots

    def _numeric_target_plot(
        self,
        *,
        feature: str,
        values: pd.Series,
        target: pd.Series,
        target_column: str,
        output_directory: Path,
    ) -> PlotSpec:
        """Generate a numeric feature versus target plot."""

        frame = pd.DataFrame(
            {
                feature: pd.to_numeric(
                    values,
                    errors="coerce",
                ),
                target_column: (
                    pd.to_numeric(
                        target,
                        errors="coerce",
                    )
                ),
            }
        ).replace(
            [np.inf, -np.inf],
            np.nan,
        ).dropna()

        if (
            len(frame)
            > self.options.maximum_sample_rows
        ):
            frame = frame.sample(
                n=self.options.maximum_sample_rows,
                random_state=42,
            )

        spec = PlotSpec(
            plot_type=PlotType.NUMERIC_BY_TARGET,
            title=(
                f"{feature} versus {target_column}"
            ),
            columns=[feature],
            target_column=target_column,
            x_label=feature,
            y_label=target_column,
            sample_size=len(frame),
            image_format=self.options.image_format,
        )

        output_path = output_directory / (
            f"{self._safe_name(feature)}"
            f"__target."
            f"{self.options.image_format}"
        )

        try:
            figure, axis = plt.subplots()

            axis.scatter(
                frame[feature],
                frame[target_column],
                s=10,
                alpha=0.6,
            )

            axis.set_title(spec.title)
            axis.set_xlabel(feature)
            axis.set_ylabel(target_column)

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

    def _categorical_target_plot(
        self,
        *,
        feature: str,
        values: pd.Series,
        target: pd.Series,
        target_column: str,
        output_directory: Path,
    ) -> PlotSpec:
        """Generate a categorical feature versus target plot."""

        frame = pd.DataFrame(
            {
                feature: values,
                target_column: pd.to_numeric(
                    target,
                    errors="coerce",
                ),
            }
        ).dropna()

        grouped = (
            frame.groupby(
                feature,
                dropna=False,
            )[target_column]
            .mean()
            .sort_values(
                ascending=False
            )
            .head(30)
        )

        spec = PlotSpec(
            plot_type=PlotType.CATEGORY_BY_TARGET,
            title=(
                f"Average {target_column} by "
                f"{feature}"
            ),
            columns=[feature],
            target_column=target_column,
            x_label=feature,
            y_label=f"Average {target_column}",
            sample_size=len(frame),
            image_format=self.options.image_format,
        )

        output_path = output_directory / (
            f"{self._safe_name(feature)}"
            f"__target_categories."
            f"{self.options.image_format}"
        )

        try:
            figure, axis = plt.subplots()

            positions = list(
                range(len(grouped))
            )

            axis.bar(
                positions,
                grouped.to_numpy(),
            )

            axis.set_xticks(
                positions
            )

            axis.set_xticklabels(
                [
                    str(value)
                    for value in grouped.index
                ],
                rotation=45,
                ha="right",
            )

            axis.set_title(spec.title)
            axis.set_xlabel(feature)
            axis.set_ylabel(
                f"Average {target_column}"
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
