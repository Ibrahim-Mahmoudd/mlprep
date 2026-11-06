from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlprep.phases.phase7.contract.selection_plan import RecursiveEstimator
from mlprep.phases.phase7.selectors.base import SelectorFitContext
from mlprep.phases.phase7.selectors.recursive_selector import RecursiveSelector


def _data() -> tuple[pd.DataFrame, pd.Series]:
    random = np.random.default_rng(42)
    signal = np.linspace(-2.0, 2.0, 60)
    features = pd.DataFrame(
        {
            "signal": signal,
            "noise_a": random.normal(size=60),
            "noise_b": random.normal(size=60),
        }
    )
    target = pd.Series(3.0 * signal + 0.05 * random.normal(size=60))
    return features, target


def test_recursive_linear_selector_eliminates_noise_and_keeps_protected() -> None:
    features, target = _data()
    selector = RecursiveSelector(
        task="regression",
        estimator=RecursiveEstimator.LINEAR,
        target_feature_count=1,
        required_features=["noise_b"],
        random_seed=5,
    ).fit(
        features=features,
        target=target,
        context=SelectorFitContext.TRAINING_SUPERVISED,
        target_access_authorized=True,
    )

    selected = selector.selected_features()

    assert "noise_b" in selected
    assert len(selected) == 1
    assert selector.state.metadata["final_features"] == selected
    assert selector.state.metadata["converged"] is True
    assert selector.state.fitted_on_training_only


def test_recursive_forest_selector_handles_classification() -> None:
    features, regression_target = _data()
    target = (regression_target > 0).astype(int)
    selector = RecursiveSelector(
        task="classification",
        estimator=RecursiveEstimator.RANDOM_FOREST,
        target_feature_count=2,
        removal_step=2,
        maximum_iterations=1,
        random_seed=7,
    ).fit(
        features=features,
        target=target,
        context="training_supervised",
        target_access_authorized=True,
    )

    assert len(selector.selected_features()) == 2
    assert selector.state.metadata["estimator"] == "random_forest"
    assert selector.state.metadata["iterations"]


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"minimum_feature_count": 0}, "minimum_feature_count"),
        ({"removal_step": 0}, "removal_step"),
        ({"maximum_iterations": 0}, "maximum_iterations"),
    ],
)
def test_recursive_selector_validates_limits(kwargs, message) -> None:
    with pytest.raises(ValueError, match=message):
        RecursiveSelector(
            task="regression",
            estimator=RecursiveEstimator.LINEAR,
            **kwargs,
        )
