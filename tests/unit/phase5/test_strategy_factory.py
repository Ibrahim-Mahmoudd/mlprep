from __future__ import annotations

import pandas as pd
import pytest

from mlprep.phases.phase5.contract.treatment_plan import (
    DetectorKind,
    TreatmentKind,
    TreatmentPlanEntry,
    TreatmentPlanStatus,
)
from mlprep.phases.phase5.strategies.base import (
    BaseTreatmentStrategy,
    TreatmentExecutionMode,
)
from mlprep.phases.phase5.strategies.clip_strategy import (
    ClipStrategy,
)
from mlprep.phases.phase5.strategies.report_only_strategy import (
    ReportOnlyStrategy,
)
from mlprep.phases.phase5.strategies.set_missing_strategy import (
    SetMissingStrategy,
)
from mlprep.phases.phase5.strategies.strategy_factory import (
    StrategyFactory,
)
from mlprep.phases.phase5.strategies.winsorize_strategy import (
    WinsorizeStrategy,
    WinsorizeStrategyOptions,
)


def _entry(
    column: str,
    treatment: TreatmentKind,
    *,
    fallback: float | None = None,
    status: TreatmentPlanStatus = TreatmentPlanStatus.PLANNED,
) -> TreatmentPlanEntry:
    return TreatmentPlanEntry(
        column=column,
        detector=DetectorKind.IQR,
        treatment=treatment,
        set_missing_fallback=fallback,
        status=status,
        reason="test treatment strategy construction",
    )


@pytest.mark.parametrize(
    ("treatment", "expected_type", "fallback"),
    [
        (TreatmentKind.CLIP, ClipStrategy, None),
        (TreatmentKind.WINSORIZE, WinsorizeStrategy, None),
        (TreatmentKind.REPORT_ONLY, ReportOnlyStrategy, None),
        (TreatmentKind.SET_MISSING, SetMissingStrategy, 0.0),
    ],
)
def test_strategy_factory_builds_treatment_with_normalized_bounds(
    treatment: TreatmentKind,
    expected_type: type[BaseTreatmentStrategy],
    fallback: float | None,
) -> None:
    strategy = StrategyFactory.create(
        entry=_entry(
            "amount",
            treatment,
            fallback=fallback,
        ),
        detector_state={
            "lower_bound": "1.25",
            "upper_bound": 10,
        },
    )

    assert isinstance(strategy, expected_type)
    assert strategy.column == "amount"
    assert strategy.state.lower_bound == 1.25
    assert strategy.state.upper_bound == 10.0
    if isinstance(strategy, SetMissingStrategy):
        assert strategy.fallback_value == 0.0


def test_strategy_factory_passes_winsor_options_and_allows_missing_bounds() -> None:
    options = WinsorizeStrategyOptions(
        learn_bounds_from_training=False,
        lower_quantile=0.1,
        upper_quantile=0.9,
    )
    winsor = StrategyFactory.create(
        entry=_entry("value", TreatmentKind.WINSORIZE),
        detector_state={},
        winsorize_options=options,
    )
    clip = StrategyFactory.create(
        entry=_entry("value", TreatmentKind.CLIP),
        detector_state={
            "lower_bound": None,
            "upper_bound": None,
        },
    )

    assert isinstance(winsor, WinsorizeStrategy)
    assert winsor.options == options
    assert clip.state.lower_bound is None
    assert clip.state.upper_bound is None


def test_strategy_factory_creates_multiple_strategies_and_requires_detector_state(
) -> None:
    entries = [
        _entry("lower", TreatmentKind.CLIP),
        _entry("upper", TreatmentKind.REPORT_ONLY),
    ]
    strategies = StrategyFactory.create_many(
        entries=entries,
        detector_states={
            "lower": {"lower_bound": 0.0},
            "upper": {"upper_bound": 5.0},
        },
    )

    assert list(strategies) == ["lower", "upper"]
    assert isinstance(strategies["lower"], ClipStrategy)
    assert isinstance(strategies["upper"], ReportOnlyStrategy)
    assert strategies["lower"].state.lower_bound == 0.0
    assert strategies["upper"].state.upper_bound == 5.0
    with pytest.raises(KeyError, match="state is unavailable"):
        StrategyFactory.create_many(
            entries=entries,
            detector_states={"lower": {}},
        )


def test_strategy_factory_rejects_nonplanned_treatments() -> None:
    with pytest.raises(ValueError, match="status 'skipped'"):
        StrategyFactory.create(
            entry=_entry(
                "amount",
                TreatmentKind.CLIP,
                status=TreatmentPlanStatus.SKIPPED,
            ),
            detector_state={},
        )


def test_strategy_factory_outputs_integrate_with_transformations() -> None:
    series = pd.Series(
        [0.0, 5.0, 15.0],
        index=pd.Index([10, 11, 12], name="row_id"),
        name="amount",
    )
    mask = pd.Series([True, False, True], index=series.index)
    clip = StrategyFactory.create(
        entry=_entry("amount", TreatmentKind.CLIP),
        detector_state={
            "lower_bound": 1.0,
            "upper_bound": 10.0,
        },
    )
    clip.fit(series=series, outlier_mask=mask)
    transformed = clip.transform(
        series=series,
        outlier_mask=mask,
        mode=TreatmentExecutionMode.VALIDATION_TRANSFORM,
    )

    assert transformed.tolist() == [1.0, 5.0, 10.0]
    assert transformed.index.equals(series.index)
