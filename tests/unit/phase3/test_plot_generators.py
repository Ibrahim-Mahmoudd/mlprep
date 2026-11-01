from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt
import pandas as pd
import pytest

from mlprep.phases.phase3.contract.column_selection import (
    AnalysisColumnGroups,
)
from mlprep.phases.phase3.relationships.numeric_associations import (
    NumericAssociationPair,
    NumericAssociationResult,
)
from mlprep.phases.phase3.relationships.target_associations import (
    TargetAssociation,
    TargetAssociationResult,
)
from mlprep.phases.phase3.visualization.missingness_plots import (
    MissingnessPlotGenerator,
)
from mlprep.phases.phase3.visualization.numeric_plots import (
    NumericPlotGenerator,
    NumericPlotOptions,
)
from mlprep.phases.phase3.visualization.plot_spec import (
    PlotStatus,
)
from mlprep.phases.phase3.visualization.relationship_plots import (
    RelationshipPlotGenerator,
    RelationshipPlotOptions,
)


@pytest.fixture(autouse=True)
def close_all_figures():
    plt.close("all")
    yield
    plt.close("all")
    assert plt.get_fignums() == []


def _plot_contract(frame: pd.DataFrame, target: pd.Series | None = None):
    features = frame.copy(deep=True)
    labels = (
        target.copy(deep=True)
        if target is not None
        else pd.Series(index=features.index, dtype="object")
    )
    return SimpleNamespace(
        training_features=lambda: features.copy(deep=True),
        training_target=lambda: labels.copy(deep=True),
        target_column="target",
    )


def _assert_generated_files(specs, output_directory: Path, expected: set[str]):
    assert {Path(spec.output_path).name for spec in specs} == expected
    for spec in specs:
        path = Path(spec.output_path)
        assert path.parent == output_directory
        assert spec.status == PlotStatus.GENERATED
        assert path.is_file()
        assert path.stat().st_size > 0


def test_missingness_plots_write_expected_files_without_mutating_input(
    tmp_path: Path,
) -> None:
    frame = pd.DataFrame(
        {
            "first value": [1.0, None, 3.0, None],
            "second": [None, "b", "c", None],
            "complete": [10, 20, 30, 40],
        }
    )
    original = frame.copy(deep=True)
    generator = MissingnessPlotGenerator()

    plots = generator.generate(
        contract=_plot_contract(frame),
        output_directory=tmp_path,
    )

    _assert_generated_files(
        plots,
        tmp_path,
        {
            "missingness__bar.png",
            "missingness__matrix.png",
            "missingness__heatmap.png",
        },
    )
    assert frame.equals(original)
    assert all(spec.columns == ["first value", "second"] for spec in plots)


def test_missingness_plots_skip_complete_or_insufficient_data(
    tmp_path: Path,
) -> None:
    complete = pd.DataFrame({"value": [1, 2]})
    assert MissingnessPlotGenerator().generate(
        contract=_plot_contract(complete),
        output_directory=tmp_path,
    ) == []

    frame = pd.DataFrame({"mostly_missing": [None, 1, None]})
    plots = MissingnessPlotGenerator().generate(
        contract=_plot_contract(frame),
        output_directory=tmp_path,
    )

    assert {Path(spec.output_path).name for spec in plots} == {
        "missingness__bar.png",
        "missingness__matrix.png",
    }


def test_numeric_plots_generate_histogram_box_and_quantile_files(
    tmp_path: Path,
) -> None:
    frame = pd.DataFrame(
        {
            "sales amount": [1.0, 2.0, None, 4.0, float("inf")],
            "invalid": ["x", "y", "z", "w", "v"],
        }
    )
    original = frame.copy(deep=True)
    generator = NumericPlotGenerator(
        NumericPlotOptions(
            create_quantile_plots=True,
            histogram_bins=4,
        )
    )

    plots = generator.generate(
        contract=_plot_contract(frame),
        columns=AnalysisColumnGroups(
            numeric=["sales amount", "invalid", "absent"]
        ),
        output_directory=tmp_path,
    )

    _assert_generated_files(
        plots,
        tmp_path,
        {
            "sales_amount__histogram.png",
            "sales_amount__box.png",
            "sales_amount__quantile.png",
        },
    )
    assert [spec.sample_size for spec in plots] == [3, 3, 3]
    assert frame.equals(original)


def test_numeric_plots_skip_empty_columns_and_honor_disabled_options(
    tmp_path: Path,
) -> None:
    frame = pd.DataFrame({"empty": [None, float("inf"), "not numeric"]})
    assert NumericPlotGenerator().generate(
        contract=_plot_contract(frame),
        columns=AnalysisColumnGroups(numeric=["empty"]),
        output_directory=tmp_path,
    ) == []

    disabled = NumericPlotGenerator(
        NumericPlotOptions(enabled=False)
    )
    assert disabled.generate(
        contract=_plot_contract(pd.DataFrame({"x": [1, 2]})),
        columns=AnalysisColumnGroups(numeric=["x"]),
        output_directory=tmp_path,
    ) == []


def test_relationship_plots_generate_heatmap_scatter_and_target_plots(
    tmp_path: Path,
) -> None:
    frame = pd.DataFrame(
        {
            "numeric x": [1.0, 2.0, 3.0, 4.0],
            "numeric y": [2.0, 4.0, 6.0, 8.0],
            "segment": ["a", "a", "b", "b"],
        }
    )
    target = pd.Series([0, 1, 0, 1], name="target")
    original = frame.copy(deep=True)
    original_target = target.copy(deep=True)
    numeric_associations = NumericAssociationResult(
        primary_method="pearson",
        correlation_matrices={
            "pearson": {
                "numeric x": {"numeric x": 1.0, "numeric y": 1.0},
                "numeric y": {"numeric x": 1.0, "numeric y": 1.0},
            }
        },
        pairs=[
            NumericAssociationPair(
                left_column="numeric x",
                right_column="numeric y",
                absolute_primary_correlation=1.0,
            )
        ],
    )
    target_associations = TargetAssociationResult(
        target_column="target",
        task_type="binary_classification",
        associations=[
            TargetAssociation(
                feature="numeric x",
                kind="numeric_classification",
                association_score=0.8,
            ),
            TargetAssociation(
                feature="segment",
                kind="categorical_classification",
                association_score=0.6,
            ),
        ],
    )
    columns = AnalysisColumnGroups(
        numeric=["numeric x", "numeric y"],
        categorical=["segment"],
    )
    generator = RelationshipPlotGenerator()

    plots = generator.generate(
        contract=_plot_contract(frame, target),
        columns=columns,
        numeric_associations=numeric_associations,
        target_associations=target_associations,
        output_directory=tmp_path,
    )

    _assert_generated_files(
        plots,
        tmp_path,
        {
            "numeric__correlation_heatmap.png",
            "numeric_x__vs__numeric_y.png",
            "numeric_x__target.png",
            "segment__target_categories.png",
        },
    )
    assert frame.equals(original)
    assert target.equals(original_target)


def test_relationship_plots_return_empty_for_insufficient_inputs(
    tmp_path: Path,
) -> None:
    frame = pd.DataFrame({"x": [1.0, 2.0]})
    numeric_associations = NumericAssociationResult(
        primary_method="pearson"
    )
    target_associations = TargetAssociationResult(
        target_column="target",
        task_type="regression",
    )

    assert RelationshipPlotGenerator().generate(
        contract=_plot_contract(frame),
        columns=AnalysisColumnGroups(numeric=["x"]),
        numeric_associations=numeric_associations,
        target_associations=target_associations,
        output_directory=tmp_path,
    ) == []

    disabled = RelationshipPlotGenerator(
        RelationshipPlotOptions(enabled=False)
    )
    assert disabled.generate(
        contract=_plot_contract(frame),
        columns=AnalysisColumnGroups(),
        numeric_associations=numeric_associations,
        target_associations=target_associations,
        output_directory=tmp_path,
    ) == []
