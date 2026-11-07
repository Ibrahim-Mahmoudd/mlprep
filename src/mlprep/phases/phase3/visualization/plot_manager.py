"""Unified plot management for Phase 3."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

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
from mlprep.phases.phase3.visualization.categorical_plots import (
    CategoricalPlotGenerator,
    CategoricalPlotOptions,
)
from mlprep.phases.phase3.visualization.missingness_plots import (
    MissingnessPlotGenerator,
    MissingnessPlotOptions,
)
from mlprep.phases.phase3.visualization.numeric_plots import (
    NumericPlotGenerator,
    NumericPlotOptions,
)
from mlprep.phases.phase3.visualization.plot_spec import (
    PlotCollection,
    PlotSpec,
)
from mlprep.phases.phase3.visualization.relationship_plots import (
    RelationshipPlotGenerator,
    RelationshipPlotOptions,
)
from mlprep.phases.phase3.visualization.target_plots import (
    TargetPlotGenerator,
    TargetPlotOptions,
)


@dataclass(slots=True)
class PlotManagerOptions:
    """Complete Phase 3 visualization configuration."""

    enabled: bool = True

    output_directory: str = (
        "artifacts/phase3/plots"
    )

    create_output_directory: bool = True

    maximum_total_plots: int = 250

    fail_on_plot_error: bool = False

    numeric: NumericPlotOptions = field(
        default_factory=NumericPlotOptions
    )

    categorical: CategoricalPlotOptions = field(
        default_factory=CategoricalPlotOptions
    )

    target: TargetPlotOptions = field(
        default_factory=TargetPlotOptions
    )

    missingness: MissingnessPlotOptions = field(
        default_factory=MissingnessPlotOptions
    )

    relationships: RelationshipPlotOptions = field(
        default_factory=RelationshipPlotOptions
    )

    def __post_init__(self) -> None:
        """Validate plot-manager options."""

        if self.maximum_total_plots < 1:
            raise ValueError(
                "maximum_total_plots must be positive."
            )

        if not self.output_directory.strip():
            raise ValueError(
                "output_directory cannot be empty."
            )


class PlotManager:
    """Coordinate all Phase 3 plot generators."""

    component_name = "plot_manager"

    def __init__(
        self,
        options: PlotManagerOptions | None = None,
    ) -> None:
        """Initialize the plot manager."""

        self.options = (
            options or PlotManagerOptions()
        )

    def generate(
        self,
        *,
        contract: EDAContract,
        columns: AnalysisColumnGroups,
        numeric_associations: NumericAssociationResult,
        target_associations: TargetAssociationResult,
    ) -> PlotCollection:
        """Generate all configured Phase 3 plots."""

        if not self.options.enabled:
            return PlotCollection(
                metadata={
                    "enabled": False,
                }
            )

        output_directory = Path(
            self.options.output_directory
        )

        if self.options.create_output_directory:
            output_directory.mkdir(
                parents=True,
                exist_ok=True,
            )

        plots: list[PlotSpec] = []

        generators = [
            lambda: NumericPlotGenerator(
                self.options.numeric
            ).generate(
                contract=contract,
                columns=columns,
                output_directory=(
                    output_directory
                ),
            ),
            lambda: CategoricalPlotGenerator(
                self.options.categorical
            ).generate(
                contract=contract,
                columns=columns,
                output_directory=(
                    output_directory
                ),
            ),
            lambda: TargetPlotGenerator(
                self.options.target
            ).generate(
                contract=contract,
                output_directory=(
                    output_directory
                ),
            ),
            lambda: MissingnessPlotGenerator(
                self.options.missingness
            ).generate(
                contract=contract,
                output_directory=(
                    output_directory
                ),
            ),
            lambda: RelationshipPlotGenerator(
                self.options.relationships
            ).generate(
                contract=contract,
                columns=columns,
                numeric_associations=(
                    numeric_associations
                ),
                target_associations=(
                    target_associations
                ),
                output_directory=(
                    output_directory
                ),
            ),
        ]

        for generator in generators:
            try:
                generated_plots = generator()

            except Exception:
                if self.options.fail_on_plot_error:
                    raise

                continue

            remaining_capacity = (
                self.options.maximum_total_plots
                - len(plots)
            )

            if remaining_capacity <= 0:
                break

            plots.extend(
                generated_plots[
                    :remaining_capacity
                ]
            )

        collection = PlotCollection(
            plots=plots,
            metadata={
                "enabled": True,
                "output_directory": str(
                    output_directory
                ),
                "maximum_total_plots": (
                    self.options
                    .maximum_total_plots
                ),
                "plot_limit_reached": (
                    len(plots)
                    >= self.options
                    .maximum_total_plots
                ),
            },
        )

        collection.refresh_counts()

        return collection
