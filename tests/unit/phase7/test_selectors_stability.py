from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlprep.phases.phase7.contract.selection_plan import (
    UnivariateMethod,
)
from mlprep.phases.phase7.selectors.base import (
    SelectorFitContext,
)
from mlprep.phases.phase7.selectors.model_importance_selector import (
    ModelImportanceSelector,
)
from mlprep.phases.phase7.selectors.mutual_information_selector import (
    MutualInformationSelector,
)
from mlprep.phases.phase7.selectors.univariate_selector import (
    UnivariateSelector,
)
from mlprep.phases.phase7.stability.distribution_stability import (
    DistributionStabilityEvaluator,
    DistributionStabilityOptions,
)
from mlprep.phases.phase7.stability.selection_stability import (
    SelectionStabilityEvaluator,
    SelectionStabilityOptions,
)
from mlprep.phases.phase7.stability.stability_engine import (
    StabilityEngine,
    StabilityEngineOptions,
)


def _classification_data() -> tuple[pd.DataFrame, pd.Series]:
    random = np.random.default_rng(72)
    signal = np.linspace(-3.0, 3.0, 120)
    features = pd.DataFrame(
        {
            "signal": signal,
            "noise": random.normal(size=len(signal)),
        },
        index=pd.Index(range(1000, 1120), name="row_id"),
    )
    target = pd.Series(
        (signal > 0.0).astype(int),
        index=features.index,
        name="target",
    )
    return features, target


def test_mutual_information_and_model_importance_are_deterministic() -> None:
    features, target = _classification_data()
    original_features = features.copy(deep=True)
    original_target = target.copy(deep=True)
    context = SelectorFitContext.TRAINING_SUPERVISED

    mutual_information = MutualInformationSelector(
        task="classification",
        top_k=1,
        discrete_features="none",
        random_seed=17,
    ).fit(
        features=features,
        target=target,
        context=context,
        target_access_authorized=True,
    )
    first_importance = ModelImportanceSelector(
        task="classification",
        estimator="random_forest",
        number_of_estimators=24,
        top_k=1,
        random_seed=17,
    ).fit(
        features=features,
        target=target,
        context=context,
        target_access_authorized=True,
    )
    repeated_importance = ModelImportanceSelector(
        task="classification",
        estimator="random_forest",
        number_of_estimators=24,
        top_k=1,
        random_seed=17,
    ).fit(
        features=features,
        target=target,
        context=context,
        target_access_authorized=True,
    )

    assert mutual_information.selected_features() == ["signal"]
    assert first_importance.selected_features() == ["signal"]
    assert first_importance.feature_scores() == repeated_importance.feature_scores()
    assert mutual_information.state.fitted_on_training_only
    assert mutual_information.state.target_access_authorized
    pd.testing.assert_frame_equal(features, original_features)
    pd.testing.assert_series_equal(target, original_target)


def test_distribution_stability_uses_training_bins_and_flags_shift() -> None:
    training = pd.DataFrame(
        {
            "shifted": np.linspace(0.0, 1.0, 100),
            "stable": np.tile([0.0, 1.0], 50),
        }
    )
    comparison = pd.DataFrame(
        {
            "shifted": np.linspace(10.0, 11.0, 100),
            "stable": np.tile([0.0, 1.0], 50),
        }
    )
    original_training = training.copy(deep=True)
    original_comparison = comparison.copy(deep=True)
    evaluator = DistributionStabilityEvaluator(
        DistributionStabilityOptions(
            block_unstable_features=True,
        )
    )

    result = evaluator.evaluate(
        training=training,
        comparison=comparison,
        features=["shifted", "stable"],
    )

    shifted = result.features["shifted"]
    stable = result.features["stable"]
    assert shifted.unstable
    assert shifted.blocks_readiness
    assert shifted.population_stability_index > 0.25
    assert shifted.bin_edges[0] == -np.inf
    assert shifted.bin_edges[-1] == np.inf
    assert stable.unstable is False
    assert result.comparison_used_for_evaluation_only
    assert result.blocking_features == ["shifted"]
    pd.testing.assert_frame_equal(training, original_training)
    pd.testing.assert_frame_equal(comparison, original_comparison)


def test_distribution_stability_tracks_missingness_and_validates_domains() -> None:
    training = pd.DataFrame(
        {"feature": [1.0, 2.0, np.nan, 3.0]}
    )
    comparison = pd.DataFrame(
        {"feature": [1.0, np.nan, np.nan, np.nan]}
    )
    evaluator = DistributionStabilityEvaluator(
        DistributionStabilityOptions(
            allow_missing_values=True,
            missing_rate_shift_warning_threshold=0.1,
            missing_rate_shift_blocking_threshold=0.3,
            block_unstable_features=True,
        )
    )
    result = evaluator.evaluate(
        training=training,
        comparison=comparison,
    )

    feature = result.features["feature"]
    assert feature.missing_rate_training == pytest.approx(0.25)
    assert feature.missing_rate_comparison == pytest.approx(0.75)
    assert feature.missing_rate_shift == pytest.approx(0.5)
    assert feature.blocks_readiness
    with pytest.raises(ValueError, match="nonfinite"):
        evaluator.evaluate(
            training=pd.DataFrame({"feature": [1.0, np.inf]}),
            comparison=pd.DataFrame({"feature": [1.0, 2.0]}),
        )
    with pytest.raises(KeyError, match="missing"):
        evaluator.evaluate(
            training=pd.DataFrame({"feature": [1.0, 2.0]}),
            comparison=pd.DataFrame({"other": [1.0, 2.0]}),
            features=["feature"],
        )


def test_selection_stability_is_reproducible_and_requires_authorization() -> None:
    features, target = _classification_data()
    original_features = features.copy(deep=True)
    original_target = target.copy(deep=True)
    options = SelectionStabilityOptions(
        enabled=True,
        bootstrap_iterations=5,
        sample_fraction=0.8,
        random_seed=91,
        minimum_selection_frequency=0.8,
        block_unstable_features=True,
    )
    def factory() -> UnivariateSelector:
        return UnivariateSelector(
            task="classification",
            method=UnivariateMethod.ANOVA,
            top_k=1,
        )

    first = SelectionStabilityEvaluator(options).evaluate(
        features=features,
        target=target,
        selector_factory=factory,
        task="classification",
        target_access_authorized=True,
    )
    repeated = SelectionStabilityEvaluator(options).evaluate(
        features=features,
        target=target,
        selector_factory=factory,
        task="classification",
        target_access_authorized=True,
    )

    assert first.bootstrap_success_count == 5
    assert first.bootstrap_failure_count == 0
    assert first.bootstrap_selected_features == repeated.bootstrap_selected_features
    assert first.features["signal"].selection_frequency == 1.0
    assert first.features["signal"].stable
    assert first.fitted_on_training_only
    assert first.target_access_authorized
    pd.testing.assert_frame_equal(features, original_features)
    pd.testing.assert_series_equal(target, original_target)
    with pytest.raises(PermissionError, match="authorized"):
        SelectionStabilityEvaluator(options).evaluate(
            features=features,
            target=target,
            selector_factory=factory,
            task="classification",
            target_access_authorized=False,
        )


def test_stability_engine_removes_unstable_features_but_keeps_protected() -> None:
    training = pd.DataFrame(
        {
            "protected_shift": np.linspace(0.0, 1.0, 100),
            "removable_shift": np.linspace(0.0, 1.0, 100),
            "stable": np.tile([0.0, 1.0], 50),
        }
    )
    validation = pd.DataFrame(
        {
            "protected_shift": np.linspace(10.0, 11.0, 100),
            "removable_shift": np.linspace(10.0, 11.0, 100),
            "stable": np.tile([0.0, 1.0], 50),
        }
    )
    target = pd.Series(np.tile([0, 1], 50))
    engine = StabilityEngine(
        StabilityEngineOptions(
            distribution=DistributionStabilityOptions(
                block_unstable_features=True,
            ),
            selection=SelectionStabilityOptions(
                enabled=False,
            ),
            remove_unstable_features=True,
        )
    ).evaluate(
        training_features=training,
        validation_features=validation,
        training_target=target,
        selected_features=[
            "protected_shift",
            "removable_shift",
            "stable",
        ],
        task="classification",
        target_access_authorized=False,
        protected_features=["protected_shift"],
    )
    result = engine.result

    assert result is not None
    assert result.unstable_features == [
        "protected_shift",
        "removable_shift",
    ]
    assert result.removed_unstable_features == ["removable_shift"]
    assert result.retained_features == ["protected_shift", "stable"]
    assert result.decisions["protected_shift"].protected_override
    assert result.decisions["protected_shift"].blocks_readiness is False
    assert result.validation_used_for_evaluation_only
