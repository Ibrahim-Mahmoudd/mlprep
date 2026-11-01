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


def test_robust_zscore_uses_finite_training_values_and_scores_new_rows() -> None:
    training = pd.Series(
        [1.0, 2.0, 3.0, 4.0, 5.0, np.inf, np.nan],
        index=range(7),
        name="value",
    )
    original = training.copy(deep=True)
    detector = RobustZScoreDetector(
        column="value",
        options=RobustZScoreDetectorOptions(
            threshold=2.0,
            minimum_valid_count=5,
        ),
    ).fit(training)
    validation = pd.Series(
        [-10.0, 3.0, 12.0, np.inf, np.nan],
        index=[20, 21, 22, 23, 24],
        name="value",
    )

    mask = detector.detect(validation)
    scores = detector.robust_zscores(
        pd.Series([3.0, 5.0, np.nan], name="value")
    )

    assert detector.state.training_valid_count == 5
    assert detector.state.median == 3.0
    assert detector.state.median_absolute_deviation == 1.0
    assert mask.tolist() == [True, False, True, True, False]
    assert mask.index.equals(validation.index)
    assert scores.iloc[:2].tolist() == pytest.approx([0.0, 1.3489795003921634])
    assert np.isnan(scores.iloc[2])
    assert training.equals(original)


def test_domain_detector_supports_one_sided_bounds_and_nonfinite_policy() -> None:
    detector = DomainDetector(
        column="value",
        options=DomainDetectorOptions(
            minimum=0.0,
            treat_nonfinite_as_violation=False,
        ),
    ).fit(
        pd.Series([0.0, 1.0, np.nan], name="value")
    )

    result = detector.detect(
        pd.Series([-1.0, 0.0, np.inf, np.nan], name="value")
    )

    assert result.tolist() == [True, False, False, False]
    assert detector.state.training_valid_count == 2
    assert detector.get_state().metadata["treat_nonfinite_as_violation"] is False


def test_detector_engine_builds_each_detector_and_preserves_fitted_state() -> None:
    training = pd.DataFrame(
        {
            "iqr": [*np.arange(9, dtype=float), 100.0],
            "quantile": np.arange(10, dtype=float),
            "robust": np.arange(10, dtype=float),
            "domain": np.arange(10, dtype=float),
        }
    )
    plan = TreatmentPlan(
        entries={
            "iqr": TreatmentPlanEntry(
                column="iqr",
                detector=DetectorKind.IQR,
                treatment=TreatmentKind.CLIP,
                reason="check IQR path",
            ),
            "quantile": TreatmentPlanEntry(
                column="quantile",
                detector=DetectorKind.QUANTILE,
                treatment=TreatmentKind.CLIP,
                reason="check quantile path",
            ),
            "robust": TreatmentPlanEntry(
                column="robust",
                detector=DetectorKind.ROBUST_ZSCORE,
                treatment=TreatmentKind.CLIP,
                reason="check robust path",
            ),
            "domain": TreatmentPlanEntry(
                column="domain",
                detector=DetectorKind.DOMAIN,
                treatment=TreatmentKind.CLIP,
                domain_minimum=1.0,
                domain_maximum=8.0,
                reason="check domain path",
            ),
        }
    )
    engine = DetectorEngine().fit(
        frame=training,
        plan=plan,
    )
    fitted_states = engine.states()
    validation = pd.DataFrame(
        {
            "iqr": [-100.0, 2.0, 100.0],
            "quantile": [-100.0, 5.0, 100.0],
            "robust": [-100.0, 5.0, 100.0],
            "domain": [-1.0, 5.0, 9.0],
        },
        index=[20, 21, 22],
    )

    result = engine.detect(
        frame=validation,
        partition=PartitionName.TEST,
    )

    assert set(result.masks) == {"iqr", "quantile", "robust", "domain"}
    assert result.mask("domain").tolist() == [True, False, True]
    assert result.summaries["quantile"].partition == PartitionName.TEST
    assert result.metadata["detector_count"] == 4
    assert engine.states() == fitted_states
    with pytest.raises(KeyError, match="No detector exists"):
        engine.detector("missing")


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


def test_winsorize_requires_bounds_or_sufficient_training_values() -> None:
    training = pd.Series(
        [1.0, 2.0, 10.0],
        index=[1, 2, 3],
        name="value",
    )
    mask = pd.Series([False, False, True], index=training.index)

    fixed = WinsorizeStrategy(
        column="value",
        lower_bound=0.0,
        upper_bound=5.0,
    ).fit(
        series=training,
        outlier_mask=mask,
    )
    transformed = fixed.transform(
        series=training,
        outlier_mask=mask,
        mode=TreatmentExecutionMode.TEST_TRANSFORM,
    )

    assert transformed.tolist() == [1.0, 2.0, 5.0]
    with pytest.raises(ValueError, match="requires detector bounds"):
        WinsorizeStrategy(
            column="value",
        ).fit(
            series=training,
            outlier_mask=mask,
        )
    with pytest.raises(ValueError, match="Too few valid training values"):
        WinsorizeStrategy(
            column="value",
            options=WinsorizeStrategyOptions(
                learn_bounds_from_training=True,
                minimum_valid_count=4,
            ),
        ).fit(
            series=training,
            outlier_mask=mask,
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


def test_indicator_accepts_an_identical_existing_column_when_configured() -> None:
    frame = pd.DataFrame(
        {
            "amount": [1.0, 100.0],
            "is_outlier": [False, True],
        },
        index=[5, 6],
    )
    mask = pd.Series([False, True], index=frame.index)
    original = frame.copy(deep=True)

    transformed, metadata = IndicatorStrategy(
        IndicatorStrategyOptions(
            allow_existing_identical_indicator=True,
        )
    ).create(
        frame=frame,
        source_column="amount",
        outlier_mask=mask,
        indicator_column="is_outlier",
    )

    pd.testing.assert_frame_equal(transformed, frame)
    assert metadata.indicator_column == "is_outlier"
    assert metadata.outlier_percentage == 50.0
    pd.testing.assert_frame_equal(frame, original)
