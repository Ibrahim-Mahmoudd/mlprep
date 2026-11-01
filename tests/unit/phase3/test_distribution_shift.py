from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from mlprep.phases.phase3.contract.column_selection import (
    AnalysisColumnGroups,
)
from mlprep.phases.phase3.diagnostics.distribution_shift import (
    DistributionShiftAnalyzer,
    DistributionShiftOptions,
)


def _contract(
    training: pd.DataFrame,
    validation: pd.DataFrame,
):
    return SimpleNamespace(
        training_features=lambda: training.copy(deep=True),
        validation_features=lambda: validation.copy(deep=True),
    )


def test_identical_numeric_and_categorical_data_has_no_shift() -> None:
    training = pd.DataFrame(
        {
            "amount": [1.0, 2.0, 3.0, 4.0],
            "category": ["a", "b", "a", "b"],
        }
    )
    validation = training.copy(deep=True)

    result = DistributionShiftAnalyzer().analyze(
        contract=_contract(training, validation),
        columns=AnalysisColumnGroups(
            numeric=["amount"],
            categorical=["category"],
        ),
    )

    assert result.validation_available
    assert result.shifted_columns == []
    assert result.high_shift_columns == []
    assert {item.level for item in result.diagnostics.values()} == {"none"}
    assert result.diagnostics["amount"].psi == 0.0
    assert result.diagnostics["category"].total_variation_distance == 0.0


def test_numeric_and_categorical_distribution_shifts_are_reported() -> None:
    training = pd.DataFrame(
        {
            "amount": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
            "category": ["known"] * 6,
        }
    )
    validation = pd.DataFrame(
        {
            "amount": [100.0, 101.0, 102.0, 103.0],
            "category": ["new"] * 4,
        }
    )

    result = DistributionShiftAnalyzer().analyze(
        contract=_contract(training, validation),
        columns=AnalysisColumnGroups(
            numeric=["amount"],
            categorical=["category"],
        ),
    )

    assert result.shifted_columns == ["amount", "category"]
    assert result.high_shift_columns == ["amount", "category"]
    assert result.diagnostics["amount"].psi is not None
    assert result.diagnostics["amount"].ks_statistic == 1.0
    category_shift = result.diagnostics["category"]
    assert category_shift.validation_only_category_count == 1
    assert category_shift.total_variation_distance == pytest.approx(1.0)


def test_missingness_shift_is_measured_for_numeric_and_categorical_features() -> None:
    training = pd.DataFrame(
        {
            "amount": [1.0, 2.0, 3.0, 4.0],
            "category": ["a", "b", "a", "b"],
        }
    )
    validation = pd.DataFrame(
        {
            "amount": [None, None, 3.0, 4.0],
            "category": [None, None, "a", "b"],
        }
    )

    result = DistributionShiftAnalyzer().analyze(
        contract=_contract(training, validation),
        columns=AnalysisColumnGroups(
            numeric=["amount"],
            categorical=["category"],
        ),
    )

    assert result.diagnostics["amount"].missingness_difference == 0.5
    assert result.diagnostics["category"].missingness_difference == 0.5
    assert result.diagnostics["category"].validation_only_category_count == 1
    assert "category" in result.shifted_columns


def test_missing_or_additional_validation_columns_are_not_analyzed() -> None:
    training = pd.DataFrame(
        {"present": [1.0, 2.0], "validation_missing": [3.0, 4.0]}
    )
    validation = pd.DataFrame(
        {"present": [1.0, 2.0], "validation_only": [8.0, 9.0]}
    )

    result = DistributionShiftAnalyzer().analyze(
        contract=_contract(training, validation),
        columns=AnalysisColumnGroups(
            numeric=["present", "validation_missing", "validation_only"]
        ),
    )

    assert list(result.diagnostics) == ["present"]
    assert result.diagnostics["present"].level == "none"


def test_numeric_shift_threshold_is_inclusive_at_the_boundary() -> None:
    training = pd.DataFrame({"amount": [0.0, 0.0, 0.0, 0.0]})
    validation = pd.DataFrame({"amount": [0.0, 0.0, 0.0, 1.0]})
    columns = AnalysisColumnGroups(numeric=["amount"])
    baseline = DistributionShiftAnalyzer().analyze(
        contract=_contract(training, validation),
        columns=columns,
    ).diagnostics["amount"]
    assert baseline.psi is not None

    analyzer = DistributionShiftAnalyzer(
        DistributionShiftOptions(
            moderate_psi_threshold=baseline.psi,
            high_psi_threshold=baseline.psi + 1.0,
            moderate_ks_threshold=1.0,
            high_ks_threshold=1.0,
        )
    )
    boundary = analyzer.analyze(
        contract=_contract(training, validation),
        columns=columns,
    ).diagnostics["amount"]

    assert boundary.psi == baseline.psi
    assert boundary.level == "moderate"
    assert boundary.shifted


def test_numeric_shift_bins_are_derived_from_training_only() -> None:
    training = pd.DataFrame({"amount": [0.0, 1.0, 2.0, 3.0]})
    validation = pd.DataFrame({"amount": [0.0, 100.0, 100.0, 100.0]})
    options = DistributionShiftOptions(numeric_bin_count=4)
    result = DistributionShiftAnalyzer(options).analyze(
        contract=_contract(training, validation),
        columns=AnalysisColumnGroups(numeric=["amount"]),
    )
    diagnostic = result.diagnostics["amount"]

    boundaries = np.unique(
        training["amount"].quantile(np.linspace(0.0, 1.0, 5)).to_numpy()
    )
    boundaries[0] = -np.inf
    boundaries[-1] = np.inf
    train_counts, _ = np.histogram(training["amount"], bins=boundaries)
    validation_counts, _ = np.histogram(
        validation["amount"],
        bins=boundaries,
    )
    smoothing = options.smoothing
    train_distribution = (train_counts + smoothing) / (
        train_counts + smoothing
    ).sum()
    validation_distribution = (validation_counts + smoothing) / (
        validation_counts + smoothing
    ).sum()
    expected_psi = float(
        (
            (validation_distribution - train_distribution)
            * np.log(validation_distribution / train_distribution)
        ).sum()
    )

    assert diagnostic.psi == expected_psi
    assert diagnostic.training_count == len(training)
    assert diagnostic.validation_count == len(validation)


def test_distribution_shift_is_deterministic_and_does_not_mutate_partitions() -> None:
    training = pd.DataFrame(
        {
            "amount": [0.0, 1.0, 2.0, 3.0, np.nan],
            "category": ["a", "b", "a", None, "b"],
        }
    )
    validation = pd.DataFrame(
        {
            "amount": [2.0, 3.0, 4.0, np.nan],
            "category": ["b", "c", None, "c"],
        }
    )
    training_before = training.copy(deep=True)
    validation_before = validation.copy(deep=True)
    contract = _contract(training, validation)
    columns = AnalysisColumnGroups(
        numeric=["amount"],
        categorical=["category"],
    )
    analyzer = DistributionShiftAnalyzer()

    first = analyzer.analyze(contract=contract, columns=columns)
    second = analyzer.analyze(contract=contract, columns=columns)

    assert first.model_dump(mode="json") == second.model_dump(mode="json")
    assert first.diagnostics["amount"].training_count == 4
    assert first.diagnostics["category"].training_count == 5
    assert training.equals(training_before)
    assert validation.equals(validation_before)


def test_empty_validation_and_disabled_analysis_are_unavailable() -> None:
    training = pd.DataFrame({"amount": [1.0, 2.0]})
    empty_validation = pd.DataFrame({"amount": pd.Series(dtype=float)})
    result = DistributionShiftAnalyzer().analyze(
        contract=_contract(training, empty_validation),
        columns=AnalysisColumnGroups(numeric=["amount"]),
    )
    assert not result.validation_available
    assert result.diagnostics == {}

    disabled = DistributionShiftAnalyzer(
        DistributionShiftOptions(enabled=False)
    )
    disabled_result = disabled.analyze(
        contract=_contract(training, training),
        columns=AnalysisColumnGroups(numeric=["amount"]),
    )
    assert not disabled_result.validation_available
    assert disabled_result.diagnostics == {}
