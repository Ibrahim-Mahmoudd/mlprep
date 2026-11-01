from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlprep.phases.phase2.output_governance.metadata import (
    PartitionName,
)
from mlprep.phases.phase5.contract.treatment_plan import (
    DetectorKind,
    TreatmentKind,
    TreatmentPlan,
    TreatmentPlanEntry,
)
from mlprep.phases.phase5.detection.detector_engine import (
    DetectorEngine,
    DetectorEngineOptions,
)
from mlprep.phases.phase5.detection.domain_detector import (
    DomainDetector,
    DomainDetectorOptions,
)
from mlprep.phases.phase5.detection.quantile_detector import (
    QuantileDetector,
    QuantileDetectorOptions,
)
from mlprep.phases.phase5.detection.robust_zscore_detector import (
    RobustZScoreDetector,
    RobustZScoreDetectorOptions,
)
from mlprep.phases.phase5.strategies.base import (
    TreatmentExecutionMode,
)
from mlprep.phases.phase5.strategies.indicator_strategy import (
    IndicatorStrategy,
    IndicatorStrategyOptions,
)
from mlprep.phases.phase5.strategies.report_only_strategy import (
    ReportOnlyStrategy,
)
from mlprep.phases.phase5.strategies.set_missing_strategy import (
    SetMissingStrategy,
)
from mlprep.phases.phase5.strategies.winsorize_strategy import (
    WinsorizeStrategy,
    WinsorizeStrategyOptions,
)


def test_quantile_detector_uses_training_bounds_and_keeps_index() -> None:
    training = pd.Series(
        np.arange(100, dtype=float),
        index=pd.Index(range(100, 200), name="row_id"),
        name="value",
    )
    original = training.copy(deep=True)
    detector = QuantileDetector(
        column="value",
        options=QuantileDetectorOptions(
            lower_quantile=0.1,
            upper_quantile=0.9,
            minimum_valid_count=5,
        ),
    ).fit(training)
    validation = pd.Series(
        [-100.0, 50.0, 200.0],
        index=[210, 211, 212],
        name="value",
    )
    mask = detector.detect(validation)

    assert mask.tolist() == [True, False, True]
    assert mask.index.equals(validation.index)
    assert detector.state.lower_bound == pytest.approx(9.9)
    assert detector.state.upper_bound == pytest.approx(89.1)
    assert training.equals(original)
    with pytest.raises(RuntimeError, match="already fitted"):
        detector.fit(training)


def test_domain_and_robust_zscore_handle_boundaries_and_zero_mad() -> None:
    domain = DomainDetector(
        column="amount",
        options=DomainDetectorOptions(
            minimum=0.0,
            maximum=10.0,
            inclusive_minimum=False,
            inclusive_maximum=False,
        ),
    ).fit(
        pd.Series([1.0, 2.0, 3.0], name="amount")
    )
    domain_mask = domain.detect(
        pd.Series(
            [-1.0, 0.0, 1.0, 10.0, 11.0, np.inf, np.nan],
            index=range(7),
            name="amount",
        )
    )
    robust = RobustZScoreDetector(
        column="value",
        options=RobustZScoreDetectorOptions(
            zero_mad_policy="no_outliers",
        ),
    ).fit(
        pd.Series([1.0, 1.0, 1.0, 2.0, 3.0], name="value")
    )

    assert domain_mask.tolist() == [
        True,
        True,
        False,
        True,
        True,
        True,
        False,
    ]
    assert robust.state.median_absolute_deviation == 0.0
    assert robust.detect(
        pd.Series([1.0, 100.0], name="value")
    ).tolist() == [False, False]
    with pytest.raises(ValueError, match="MAD is zero"):
        RobustZScoreDetector(
            column="value",
            options=RobustZScoreDetectorOptions(
                zero_mad_policy="raise",
            ),
        ).fit(pd.Series([1.0, 1.0, 1.0, 2.0, 3.0], name="value"))


def test_detector_engine_fits_once_and_never_uses_validation_for_bounds() -> None:
    training = pd.DataFrame(
        {"value": np.arange(10, dtype=float)},
        index=pd.Index(range(10), name="row_id"),
    )
    original = training.copy(deep=True)
    plan = TreatmentPlan(
        entries={
            "value": TreatmentPlanEntry(
                column="value",
                detector=DetectorKind.IQR,
                treatment=TreatmentKind.CLIP,
                reason="detect training extremes",
            )
        }
    )
    engine = DetectorEngine(
        DetectorEngineOptions()
    ).fit(
        frame=training,
        plan=plan,
    )
    fitted_state = engine.states()
    validation = pd.DataFrame(
        {"value": [1.0, 100.0]},
        index=pd.Index([20, 21], name="row_id"),
    )
    result = engine.detect(
        frame=validation,
        partition=PartitionName.VALIDATION,
    )

    assert result.mask("value").tolist() == [False, True]
    assert result.summaries["value"].partition == PartitionName.VALIDATION
    assert result.summaries["value"].outlier_count == 1
    assert engine.states() == fitted_state
    assert result.metadata["fitted_on_training_only"]
    pd.testing.assert_frame_equal(training, original)
    with pytest.raises(RuntimeError, match="already fitted"):
        engine.fit(frame=training, plan=plan)


def test_winsorize_and_set_missing_use_explicit_fitted_behavior() -> None:
    training = pd.Series(
        [1.0, 2.0, 3.0, 4.0, 100.0],
        index=[2, 4, 6, 8, 10],
        name="amount",
    )
    original = training.copy(deep=True)
    mask = pd.Series(
        [False, False, False, False, True],
        index=training.index,
    )
    winsor = WinsorizeStrategy(
        column="amount",
        options=WinsorizeStrategyOptions(
            learn_bounds_from_training=True,
            lower_quantile=0.2,
            upper_quantile=0.8,
        ),
    )
    winsor.fit(
        series=training,
        outlier_mask=mask,
    )
    winsorized = winsor.transform(
        series=training,
        outlier_mask=mask,
        mode=TreatmentExecutionMode.TRAINING_TRANSFORM,
    )
    set_missing = SetMissingStrategy(
        column="amount",
        fallback_value=0.0,
    ).fit(
        series=training,
        outlier_mask=mask,
    )
    replaced = set_missing.transform(
        series=training,
        outlier_mask=mask,
        mode=TreatmentExecutionMode.VALIDATION_TRANSFORM,
    )

    assert winsorized.iloc[-1] == pytest.approx(23.2)
    assert winsor.get_state().fitted_on_training_only
    assert replaced.iloc[-1] == 0.0
    assert replaced.index.equals(training.index)
    assert training.equals(original)
    with pytest.raises(PermissionError, match="only be fitted"):
        SetMissingStrategy(
            column="amount",
            fallback_value=0.0,
        ).fit(
            series=training,
            outlier_mask=mask,
            mode=TreatmentExecutionMode.VALIDATION_TRANSFORM,
        )


def test_report_only_and_indicator_preserve_values_and_reject_collisions() -> None:
    frame = pd.DataFrame(
        {"amount": [1.0, 100.0]},
        index=[5, 6],
    )
    original = frame.copy(deep=True)
    mask = pd.Series([False, True], index=frame.index)
    report_only = ReportOnlyStrategy(
        column="amount",
        upper_bound=10.0,
    ).fit(
        series=frame["amount"],
        outlier_mask=mask,
    )
    unchanged = report_only.transform(
        series=frame["amount"],
        outlier_mask=mask,
        mode=TreatmentExecutionMode.TEST_TRANSFORM,
    )
    transformed, metadata = IndicatorStrategy(
        IndicatorStrategyOptions(dtype="int8")
    ).create(
        frame=frame,
        source_column="amount",
        outlier_mask=mask,
    )

    assert unchanged.tolist() == [1.0, 100.0]
    assert report_only.state.metadata["values_modified"] is False
    assert transformed["amount__was_outlier"].tolist() == [0, 1]
    assert metadata.outlier_count == 1
    assert frame.equals(original)
    with pytest.raises(ValueError, match="collision"):
        IndicatorStrategy().create(
            frame=transformed,
            source_column="amount",
            outlier_mask=mask,
        )
