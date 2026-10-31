from __future__ import annotations

import pandas as pd
import pytest

from mlprep.phases.phase4.strategies.base import (
    ImputerExecutionMode,
)
from mlprep.phases.phase4.strategies.numeric_imputer import (
    NumericImputer,
    NumericImputerOptions,
)


def test_numeric_imputer_learns_from_training_and_preserves_index() -> None:
    training = pd.Series(
        [1.0, None, 9.0],
        index=[10, 11, 12],
        name="value",
    )
    imputer = NumericImputer(
        column="value",
        options=NumericImputerOptions(method="median"),
    )

    transformed = imputer.fit_transform(training)

    assert transformed.tolist() == [1.0, 5.0, 9.0]
    assert transformed.index.equals(training.index)
    assert imputer.get_state().learned_value == 5.0
    assert imputer.get_state().fitted


def test_numeric_imputer_does_not_fit_in_validation_mode() -> None:
    imputer = NumericImputer(column="value")

    with pytest.raises(PermissionError, match="FIT mode"):
        imputer.fit(
            pd.Series([1.0, 2.0], name="value"),
            mode=ImputerExecutionMode.VALIDATION_TRANSFORM,
        )


def test_numeric_imputer_applies_training_statistic_to_validation() -> None:
    imputer = NumericImputer(column="value").fit(
        pd.Series([2.0, 4.0, None], name="value")
    )

    validation = imputer.transform(
        pd.Series([None, 100.0], index=[7, 8], name="value"),
        mode=ImputerExecutionMode.VALIDATION_TRANSFORM,
    )

    assert validation.tolist() == [3.0, 100.0]
    assert validation.index.tolist() == [7, 8]
    assert imputer.get_state().learned_value == 3.0
