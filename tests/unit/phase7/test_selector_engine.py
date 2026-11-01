from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlprep.phases.phase7.contract.selection_plan import (
    SelectionPlan,
    SelectorAggregationKind,
)
from mlprep.phases.phase7.selectors.base import (
    FeatureScore,
    SelectorState,
)
from mlprep.phases.phase7.selectors.selector_engine import (
    SelectorEngine,
    SelectorEngineOptions,
)


def _plan(
    *,
    selector: str | None = None,
    authorized: bool = False,
    aggregation: SelectorAggregationKind = SelectorAggregationKind.UNION,
    minimum_votes: int = 1,
    required_features: list[str] | None = None,
    protected_features: list[str] | None = None,
    selector_weights: dict[str, float] | None = None,
) -> SelectionPlan:
    plan_data: dict[str, object] = {
        "task": "classification",
        "constant_filter": {},
        "variance_filter": {},
        "duplicate_filter": {},
        "correlation_filter": {},
        "multicollinearity_filter": {},
        "univariate_selector": {},
        "mutual_information_selector": {},
        "model_importance_selector": {},
        "recursive_selector": {},
        "aggregation": aggregation.value,
        "minimum_votes": minimum_votes,
        "selector_weights": selector_weights or {},
        "required_features": required_features or [],
        "protected_features": protected_features or [],
        "supervised_selection_authorized": authorized,
    }

    if selector is not None:
        specification: dict[str, object] = {
            "enabled": True,
            "status": "enabled",
        }
        selector_config = {
            "univariate_selector": {
                "method": "anova",
                "top_k": 1,
                **specification,
            },
            "mutual_information_selector": {
                "discrete_features": "none",
                "random_seed": 7,
                "top_k": 1,
                **specification,
            },
            "model_importance_selector": {
                "number_of_estimators": 16,
                "random_seed": 7,
                "top_k": 1,
                **specification,
            },
            "recursive_selector": {
                "target_feature_count": 1,
                "maximum_iterations": 10,
                **specification,
            },
        }
        plan_data[selector] = selector_config[selector]

    return SelectionPlan.model_validate(plan_data)


def _training_data() -> tuple[pd.DataFrame, pd.Series]:
    random = np.random.default_rng(26)
    signal = np.linspace(-2.0, 2.0, 80)
    features = pd.DataFrame(
        {
            "signal": signal,
            "noise": random.normal(size=len(signal)),
        },
        index=pd.Index(range(200, 280), name="row_id"),
    )
    target = pd.Series(
        (signal > 0).astype(int),
        index=features.index,
        name="target",
    )
    return features, target


def _state(
    selector: str,
    scores: dict[str, tuple[float, bool]],
) -> SelectorState:
    return SelectorState(
        selector=selector,
        task="classification",
        fitted=True,
        feature_scores={
            column: FeatureScore(
                column=column,
                score=score,
                rank=rank,
                selected=selected,
            )
            for rank, (column, (score, selected)) in enumerate(
                scores.items(),
                start=1,
            )
        },
        selected_features=[
            column
            for column, (_, selected) in scores.items()
            if selected
        ],
    )


def test_selector_engine_without_supervised_selectors_keeps_source_order() -> None:
    features, target = _training_data()
    engine = SelectorEngine().fit(
        features=features,
        target=target,
        plan=_plan(),
    )
    result = engine.get_result()

    assert result.source_features == ["signal", "noise"]
    assert result.selected_features == ["signal", "noise"]
    assert result.rejected_features == []
    assert result.target_access_used is False
    assert result.target_access_authorized is False
    assert result.fitted_on_training_only
    assert result.metadata["enabled_selector_count"] == 0
    result.selected_features.clear()
    assert engine.get_result().selected_features == ["signal", "noise"]


@pytest.mark.parametrize(
    "selector",
    [
        "univariate_selector",
        "mutual_information_selector",
        "model_importance_selector",
        "recursive_selector",
    ],
)
def test_selector_engine_fits_each_authorized_selector_on_training_data(
    selector: str,
) -> None:
    features, target = _training_data()
    original_features = features.copy(deep=True)
    original_target = target.copy(deep=True)
    result = SelectorEngine().fit(
        features=features,
        target=target,
        plan=_plan(selector=selector, authorized=True),
    ).get_result()

    assert result.source_features == ["signal", "noise"]
    assert len(result.selected_features) == 1
    assert result.target_access_used and result.target_access_authorized
    assert result.selector_states[selector].fitted_on_training_only
    assert result.selector_states[selector].target_access_authorized
    pd.testing.assert_frame_equal(features, original_features)
    pd.testing.assert_series_equal(target, original_target)


def test_selector_engine_rejects_unauthorized_supervised_selection() -> None:
    features, target = _training_data()

    with pytest.raises(PermissionError, match="without authorization"):
        SelectorEngine().fit(
            features=features,
            target=target,
            plan=_plan(
                selector="univariate_selector",
                authorized=False,
            ),
        )


def test_selector_engine_restores_required_and_protected_features() -> None:
    features, target = _training_data()
    result = SelectorEngine(
        SelectorEngineOptions(
            weighted_score_threshold=2.0,
        )
    ).fit(
        features=features,
        target=target,
        plan=_plan(
            selector="univariate_selector",
            authorized=True,
            aggregation=SelectorAggregationKind.WEIGHTED_SCORE,
            required_features=["noise"],
            protected_features=["signal"],
        ),
    ).get_result()

    assert result.selected_features == ["signal", "noise"]
    assert result.aggregated_scores["noise"].required_override
    assert result.aggregated_scores["signal"].protected_override


@pytest.mark.parametrize(
    ("aggregation", "minimum_votes", "expected"),
    [
        (SelectorAggregationKind.UNION, 1, ["first", "second"]),
        (SelectorAggregationKind.INTERSECTION, 1, ["second"]),
        (SelectorAggregationKind.VOTING, 2, ["second"]),
        (SelectorAggregationKind.PRIORITY, 1, ["second"]),
    ],
)
def test_selector_engine_aggregates_selector_votes(
    aggregation: SelectorAggregationKind,
    minimum_votes: int,
    expected: list[str],
) -> None:
    engine = SelectorEngine(
        SelectorEngineOptions(normalize_selector_scores=False)
    )
    states = {
        "univariate_selector": _state(
            "univariate_selector",
            {
                "first": (0.8, True),
                "second": (0.3, True),
            },
        ),
        "mutual_information_selector": _state(
            "mutual_information_selector",
            {
                "first": (0.6, False),
                "second": (0.7, True),
            },
        ),
    }

    selected, scores = engine._aggregate(
        states=states,
        plan=_plan(
            aggregation=aggregation,
            minimum_votes=minimum_votes,
        ),
        source_features=["first", "second"],
    )

    assert selected == expected
    assert scores["first"].vote_count == 1
    assert scores["second"].vote_count == 2
    assert scores["first"].selected_by == ["univariate_selector"]


def test_selector_engine_weighted_scores_handle_zero_weights_and_constant_scores(
) -> None:
    engine = SelectorEngine(
        SelectorEngineOptions(
            weighted_score_threshold=0.1,
            normalize_selector_scores=True,
        )
    )
    plan = _plan(
        aggregation=SelectorAggregationKind.WEIGHTED_SCORE,
        selector_weights={"univariate_selector": 0.0},
    )
    selected, scores = engine._aggregate(
        states={
            "univariate_selector": _state(
                "univariate_selector",
                {
                    "zero": (0.0, False),
                    "positive": (3.0, True),
                },
            )
        },
        plan=plan,
        source_features=["zero", "positive"],
    )

    assert selected == []
    assert scores["positive"].weighted_score == 0.0
    assert scores["positive"].normalized_score == 0.0
    assert SelectorEngine._normalized_score(
        2.0,
        state=_state(
            "constant_selector",
            {
                "constant": (2.0, True),
                "same": (2.0, False),
            },
        ),
    ) == 1.0


def test_selector_engine_rejects_empty_selection_and_unknown_selector() -> None:
    features, target = _training_data()
    plan = _plan(
        selector="univariate_selector",
        authorized=True,
        aggregation=SelectorAggregationKind.WEIGHTED_SCORE,
    )

    with pytest.raises(ValueError, match="rejected every feature"):
        SelectorEngine(
            SelectorEngineOptions(
                weighted_score_threshold=2.0,
                require_nonempty_selection=True,
            )
        ).fit(features=features, target=target, plan=plan)

    with pytest.raises(ValueError, match="Unsupported selector"):
        SelectorEngine()._build_selector(
            selector_name="unknown",
            plan=plan,
        )
    with pytest.raises(RuntimeError, match="not fitted"):
        SelectorEngine().get_result()
