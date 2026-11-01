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
from mlprep.phases.phase3.visualization.categorical_plots import (
    CategoricalPlotGenerator,
    CategoricalPlotOptions,
)
from mlprep.phases.phase3.visualization.plot_spec import (
    PlotSpec,
    PlotStatus,
)


@pytest.fixture(autouse=True)
def close_all_figures():
    plt.close("all")
    yield
    plt.close("all")
    assert plt.get_fignums() == []


def _contract(frame: pd.DataFrame) -> SimpleNamespace:
    features = frame.copy(deep=True)
    return SimpleNamespace(
        training_features=lambda: features.copy(deep=True),
    )


def _assert_nonempty_plot_files(
    specs: list[PlotSpec],
    output_directory: Path,
    expected_names: set[str],
) -> None:
    assert {Path(spec.output_path).name for spec in specs} == expected_names
    for spec in specs:
        path = Path(spec.output_path)
        assert path.parent == output_directory
        assert spec.status == PlotStatus.GENERATED
        assert path.is_file()
        assert path.stat().st_size > 0


def test_categorical_and_boolean_plots_include_missing_and_preserve_inputs(
    tmp_path: Path,
) -> None:
    frame = pd.DataFrame(
        {
            "customer type": ["retail", "business", None, "retail"],
            "active": [True, False, True, None],
        },
        index=[10, 11, 12, 13],
    )
    before = frame.copy(deep=True)
    plots = CategoricalPlotGenerator().generate(
        contract=_contract(frame),
        columns=AnalysisColumnGroups(
            categorical=["customer type"],
            boolean=["active"],
            sensitive=["customer type"],
        ),
        output_directory=tmp_path,
    )

    _assert_nonempty_plot_files(
        plots,
        tmp_path,
        {
            "customer_type__categories.png",
            "active__categories.png",
        },
    )
    by_column = {spec.columns[0]: spec for spec in plots}
    assert by_column["customer type"].sensitive
    assert by_column["customer type"].sample_size == 4
    assert by_column["active"].sample_size == 4
    assert frame.equals(before)


def test_missing_values_can_be_excluded_and_top_categories_are_capped(
    tmp_path: Path,
) -> None:
    frame = pd.DataFrame(
        {
            "category": ["missing"] * 2
            + [f"value-{index}" for index in range(20)]
            + [None]
        }
    )
    plots = CategoricalPlotGenerator(
        CategoricalPlotOptions(
            include_missing_as_category=False,
            maximum_categories=3,
        )
    ).generate(
        contract=_contract(frame),
        columns=AnalysisColumnGroups(categorical=["category"]),
        output_directory=tmp_path,
    )

    assert len(plots) == 1
    assert plots[0].sample_size == 4
    _assert_nonempty_plot_files(
        plots,
        tmp_path,
        {"category__categories.png"},
    )


def test_selection_deduplicates_columns_and_respects_limits(
    tmp_path: Path,
) -> None:
    frame = pd.DataFrame(
        {
            "category": ["a", "b", "a"],
            "flag": [True, False, True],
        }
    )
    plots = CategoricalPlotGenerator(
        CategoricalPlotOptions(maximum_columns=1)
    ).generate(
        contract=_contract(frame),
        columns=AnalysisColumnGroups(
            categorical=["category", "category", "missing"],
            boolean=["category", "flag"],
        ),
        output_directory=tmp_path,
    )

    assert len(plots) == 1
    assert plots[0].columns == ["category"]
    _assert_nonempty_plot_files(
        plots,
        tmp_path,
        {"category__categories.png"},
    )

    without_booleans = CategoricalPlotGenerator(
        CategoricalPlotOptions(include_boolean_columns=False)
    ).generate(
        contract=_contract(frame),
        columns=AnalysisColumnGroups(boolean=["flag"]),
        output_directory=tmp_path,
    )
    assert without_booleans == []


def test_empty_usable_category_data_produces_a_skipped_spec(
    tmp_path: Path,
) -> None:
    plots = CategoricalPlotGenerator(
        CategoricalPlotOptions(include_missing_as_category=False)
    ).generate(
        contract=_contract(pd.DataFrame({"empty": [None, None]})),
        columns=AnalysisColumnGroups(categorical=["empty"]),
        output_directory=tmp_path,
    )

    assert len(plots) == 1
    assert plots[0].status == PlotStatus.SKIPPED
    assert plots[0].skip_reason == "The feature contains no usable values."
    assert plots[0].output_path is None
    assert list(tmp_path.iterdir()) == []


def test_disabled_generator_and_invalid_options(
    tmp_path: Path,
) -> None:
    disabled = CategoricalPlotGenerator(
        CategoricalPlotOptions(enabled=False)
    )
    assert disabled.generate(
        contract=_contract(pd.DataFrame({"category": ["a"]})),
        columns=AnalysisColumnGroups(categorical=["category"]),
        output_directory=tmp_path,
    ) == []

    with pytest.raises(ValueError, match="maximum_columns"):
        CategoricalPlotOptions(maximum_columns=0)
    with pytest.raises(ValueError, match="maximum_categories"):
        CategoricalPlotOptions(maximum_categories=1)
    with pytest.raises(ValueError, match="dpi"):
        CategoricalPlotOptions(dpi=71)
    with pytest.raises(ValueError, match="image_format"):
        CategoricalPlotOptions(image_format=" ")
