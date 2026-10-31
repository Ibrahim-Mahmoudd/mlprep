from __future__ import annotations

import pandas as pd
import pytest

from mlprep.phases.phase6.numeric.base import (
    NumericExecutionMode,
)
from mlprep.phases.phase6.numeric.standard_scaler import (
    StandardScaler,
)


def test_standard_scaler_uses_training_mean_and_population_std() -> None:
    training = pd.Series(
        [1.0, 2.0, 3.0],
        index=[4, 5, 6],
        name="value",
    )
    scaler = StandardScaler(column="value")

    transformed = scaler.fit_transform(training)

    assert transformed.index.equals(training.index)
    assert transformed.mean() == pytest.approx(0.0)
    assert transformed.std(ddof=0) == pytest.approx(1.0)
    assert scaler.get_state().fitted_on_training_only


def test_standard_scaler_handles_constant_training_feature() -> None:
    scaler = StandardScaler(column="value")
    transformed = scaler.fit_transform(
        pd.Series([5.0, 5.0, 5.0], name="value")
    )

    assert transformed.tolist() == [0.0, 0.0, 0.0]
    assert scaler.get_state().constant_feature
    assert scaler.get_state().learned_parameters["scale"] == 1.0


def test_standard_scaler_rejects_non_training_fit_context() -> None:
    scaler = StandardScaler(column="value")

    with pytest.raises(PermissionError, match="FIT mode"):
        scaler.fit(
            pd.Series([1.0, 2.0], name="value"),
            mode=NumericExecutionMode.TEST_TRANSFORM,
        )
