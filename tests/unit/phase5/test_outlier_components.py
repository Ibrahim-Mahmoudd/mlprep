from __future__ import annotations

import pandas as pd
import pytest

from mlprep.phases.phase5.detection.iqr_detector import (
    IQRDetector,
)
from mlprep.phases.phase5.strategies.base import (
    TreatmentExecutionMode,
)
from mlprep.phases.phase5.strategies.clip_strategy import (
    ClipStrategy,
)


def test_iqr_detector_fits_training_bounds_and_detects_extremes() -> None:
    training = pd.Series(
        [1.0, 2.0, 3.0, 4.0, 5.0, 100.0],
        name="value",
    )
    detector = IQRDetector(column="value").fit(training)

    mask = detector.detect(training)

    assert mask.tolist() == [False, False, False, False, False, True]
    assert detector.state.training_outlier_count == 1
    assert detector.state.upper_bound is not None
    assert detector.state.upper_bound < 100.0


def test_iqr_detector_rejects_repeated_fit() -> None:
    detector = IQRDetector(column="value").fit(
        pd.Series([1, 2, 3, 4], name="value")
    )

    with pytest.raises(RuntimeError, match="already fitted"):
        detector.fit(pd.Series([1, 2, 3, 4], name="value"))


def test_clip_strategy_changes_only_masked_values_and_keeps_index() -> None:
    series = pd.Series(
        [1.0, 2.0, 100.0],
        index=[3, 4, 5],
        name="value",
    )
    mask = pd.Series([False, False, True], index=series.index)
    strategy = ClipStrategy(
        column="value",
        lower_bound=0.0,
        upper_bound=10.0,
    )

    result = strategy.fit_transform(
        series=series,
        outlier_mask=mask,
    )

    assert result.tolist() == [1.0, 2.0, 10.0]
    assert result.index.equals(series.index)
    assert series.iloc[-1] == 100.0
    assert strategy.get_state().fitted_on_training_only


def test_clip_strategy_rejects_transform_before_fit() -> None:
    strategy = ClipStrategy(column="value", upper_bound=10.0)
    series = pd.Series([100.0], name="value")
    mask = pd.Series([True])

    with pytest.raises(RuntimeError, match="not fitted"):
        strategy.transform(
            series=series,
            outlier_mask=mask,
            mode=TreatmentExecutionMode.VALIDATION_TRANSFORM,
        )
