from __future__ import annotations

from types import SimpleNamespace

import pandas as pd

from mlprep.phases.phase3.contract.column_selection import (
    AnalysisColumnGroups,
)
from mlprep.phases.phase3.profiling.numeric_profile import (
    NumericProfiler,
)


def test_numeric_profiler_calculates_statistics_and_missingness() -> None:
    frame = pd.DataFrame(
        {"amount": [1.0, 2.0, None, 5.0, 0.0]}
    )
    contract = SimpleNamespace(
        training_features=lambda: frame.copy(deep=True)
    )
    columns = AnalysisColumnGroups(numeric=["amount"])

    profiles = NumericProfiler().profile(
        contract=contract,
        columns=columns,
    )

    profile = profiles["amount"]
    assert profile.row_count == 5
    assert profile.nonmissing_count == 4
    assert profile.missing_count == 1
    assert profile.minimum == 0.0
    assert profile.maximum == 5.0
    assert profile.zero_count == 1


def test_numeric_profiler_can_be_disabled() -> None:
    profiler = NumericProfiler()
    profiler.options.enabled = False

    assert profiler.profile(
        contract=SimpleNamespace(
            training_features=lambda: pd.DataFrame({"x": [1]})
        ),
        columns=AnalysisColumnGroups(numeric=["x"]),
    ) == {}
