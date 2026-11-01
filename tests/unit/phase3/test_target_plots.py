from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from mlprep.phases.phase3.visualization.plot_spec import (
    PlotSpec,
    PlotStatus,
    PlotType,
)
from mlprep.phases.phase3.visualization.target_plots import (
    TargetPlotGenerator,
    TargetPlotOptions,
)
from mlprep.types import TaskType


@pytest.fixture(autouse=True)
def close_all_figures():
    plt.close("all")
    yield
    plt.close("all")
    assert plt.get_fignums() == []


def _contract(
    target: pd.Series,
    task_type: TaskType | None,
    target_column: str = "target",
) -> SimpleNamespace:
    problem_contract = (
        None
        if task_type is None
        else SimpleNamespace(task_type=task_type)
    )
    return SimpleNamespace(
        phase1_state=SimpleNamespace(
            problem_contract=problem_contract,
        ),
        training_target=lambda: target.copy(deep=True),
        target_column=target_column,
    )


def _assert_generated_files(
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


def test_classification_plot_uses_training_target_and_sanitized_filename(
    tmp_path: Path,
) -> None:
    target = pd.Series(
        ["yes", "no", "yes", None, "no"],
        index=[10, 11, 12, 13, 14],
        name="label",
    )
    original = target.copy(deep=True)
    plots = TargetPlotGenerator().generate(
        contract=_contract(
            target,
            TaskType.BINARY_CLASSIFICATION,
            target_column="label value",
        ),
        output_directory=tmp_path,
    )

    _assert_generated_files(
        plots,
        tmp_path,
        {"label_value__class_distribution.png"},
    )
    assert len(plots) == 1
    assert plots[0].plot_type == PlotType.CLASS_DISTRIBUTION
    assert plots[0].target_column == "label value"
    assert plots[0].sample_size == 5
    pd.testing.assert_series_equal(target, original)


def test_regression_plots_ignore_missing_and_nonfinite_target_values(
    tmp_path: Path,
) -> None:
    target = pd.Series(
        [1.0, "2", None, np.inf, "invalid", 5.0],
        name="amount",
    )
    original = target.copy(deep=True)
    plots = TargetPlotGenerator(
        TargetPlotOptions(histogram_bins=4)
    ).generate(
        contract=_contract(
            target,
            TaskType.REGRESSION,
            target_column="target amount",
        ),
        output_directory=tmp_path,
    )

    _assert_generated_files(
        plots,
        tmp_path,
        {
            "target_amount__target_histogram.png",
            "target_amount__target_box.png",
        },
    )
    assert {spec.plot_type for spec in plots} == {
        PlotType.REGRESSION_TARGET_HISTOGRAM,
        PlotType.REGRESSION_TARGET_BOX,
    }
    assert all(spec.sample_size == 3 for spec in plots)
    pd.testing.assert_series_equal(target, original)


def test_regression_box_plot_can_be_disabled_and_empty_numeric_target_skips(
    tmp_path: Path,
) -> None:
    plots = TargetPlotGenerator(
        TargetPlotOptions(create_regression_box_plot=False)
    ).generate(
        contract=_contract(
            pd.Series([1.0, 2.0, 3.0]),
            TaskType.REGRESSION,
        ),
        output_directory=tmp_path,
    )

    _assert_generated_files(
        plots,
        tmp_path,
        {"target__target_histogram.png"},
    )
    assert len(plots) == 1
    assert plots[0].plot_type == PlotType.REGRESSION_TARGET_HISTOGRAM

    empty = TargetPlotGenerator().generate(
        contract=_contract(
            pd.Series([None, np.inf, "not numeric"]),
            TaskType.REGRESSION,
        ),
        output_directory=tmp_path,
    )
    assert empty == []


def test_disabled_and_unresolved_problem_contract_return_no_plots(
    tmp_path: Path,
) -> None:
    contract = _contract(pd.Series([0, 1]), TaskType.BINARY_CLASSIFICATION)
    disabled = TargetPlotGenerator(
        TargetPlotOptions(enabled=False)
    ).generate(
        contract=contract,
        output_directory=tmp_path,
    )
    unresolved = TargetPlotGenerator().generate(
        contract=_contract(pd.Series([0, 1]), None),
        output_directory=tmp_path,
    )

    assert disabled == []
    assert unresolved == []
    assert list(tmp_path.iterdir()) == []


def test_target_plot_options_validate_limits() -> None:
    with pytest.raises(ValueError, match="histogram_bins"):
        TargetPlotOptions(histogram_bins=1)
    with pytest.raises(ValueError, match="maximum_class_labels"):
        TargetPlotOptions(maximum_class_labels=1)
    with pytest.raises(ValueError, match="dpi"):
        TargetPlotOptions(dpi=71)
