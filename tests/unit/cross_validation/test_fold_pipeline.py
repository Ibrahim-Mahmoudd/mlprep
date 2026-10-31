from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlprep.cross_validation import (
    CrossValidationConfig,
    CrossValidationRunner,
    CrossValidationStrategy,
    FoldPipeline,
    FoldStep,
)
from mlprep.phases.phase8.samplers import (
    RandomOverSampler,
)
from mlprep.phases.phase8.weighting.class_weights import (
    ClassWeightOptions,
)


class TrainingMeanImputer:
    def __init__(self) -> None:
        self.means: dict[str, float] = {}
        self.fit_row_count = 0

    def fit(
        self,
        features: pd.DataFrame,
        target: pd.Series,
    ) -> None:
        self.means = features.mean().to_dict()
        self.fit_row_count = len(target)

    def transform(self, features: pd.DataFrame) -> pd.DataFrame:
        return features.fillna(self.means)

    def get_state(self) -> dict[str, object]:
        return {
            "means": self.means,
            "fit_row_count": self.fit_row_count,
        }


class MutatingTransformer:
    def fit(
        self,
        features: pd.DataFrame,
        target: pd.Series,
    ) -> None:
        return None

    def transform(self, features: pd.DataFrame) -> pd.DataFrame:
        features.iloc[0, 0] = 999.0
        return features

    def get_state(self) -> dict[str, object]:
        return {"fitted": True}


def _classification_data(
    *,
    row_count: int = 60,
) -> tuple[pd.DataFrame, pd.Series]:
    target = pd.Series(
        np.tile([0, 1], row_count // 2),
        name="target",
    )
    features = pd.DataFrame(
        {
            "feature_a": np.linspace(0.0, 2.0, row_count),
            "feature_b": np.linspace(3.0, 5.0, row_count),
        }
    )
    features.loc[::7, "feature_a"] = np.nan
    return features, target


def _mean_imputation_pipeline() -> FoldPipeline:
    return FoldPipeline(
        steps=[
            FoldStep(
                name="train_mean_imputer",
                factory=TrainingMeanImputer,
            )
        ],
        class_weight_options=ClassWeightOptions(),
        target_access_authorized=True,
    )


def test_stratified_runner_fits_transformers_on_training_rows_only() -> None:
    features, target = _classification_data()
    original_features = features.copy(deep=True)
    runner = CrossValidationRunner(
        CrossValidationConfig(
            strategy=CrossValidationStrategy.STRATIFIED,
            n_splits=3,
            random_seed=37,
        ),
        fold_pipeline=_mean_imputation_pipeline(),
    )

    result = runner.run(features=features, target=target)

    assert len(result.folds) == 3
    for fold in result.folds:
        assert set(fold.context.training_indices).isdisjoint(
            fold.context.validation_indices
        )
        assert len(fold.training_features) == len(fold.context.training_indices)
        assert len(fold.validation_features) == len(
            fold.context.validation_indices
        )
        assert not fold.training_features.isna().any().any()
        assert not fold.validation_features.isna().any().any()
        assert fold.class_weights is not None
        assert fold.class_weights.total_count == len(
            fold.context.training_target
        )
        assert fold.fitted_state_fingerprint
        assert fold.training_output_fingerprint
        assert fold.validation_output_fingerprint
        assert len(fold.row_lineage) == len(fold.training_features)
    pd.testing.assert_frame_equal(features, original_features)


def test_k_fold_runner_supports_continuous_regression_targets() -> None:
    features = pd.DataFrame(
        {"feature": np.linspace(0.0, 1.0, 30)}
    )
    target = pd.Series(
        np.linspace(10.0, 20.0, 30),
        name="target",
    )
    result = CrossValidationRunner(
        CrossValidationConfig(
            strategy=CrossValidationStrategy.K_FOLD,
            n_splits=3,
            random_seed=5,
            shuffle=True,
        )
    ).run(features=features, target=target)

    assert len(result.folds) == 3
    for fold in result.folds:
        assert set(fold.context.training_indices).isdisjoint(
            fold.context.validation_indices
        )
        assert len(fold.training_features) + len(
            fold.validation_features
        ) == len(features)


def test_changing_validation_does_not_change_fitted_training_state() -> None:
    features, target = _classification_data()
    config = CrossValidationConfig(
        strategy=CrossValidationStrategy.STRATIFIED,
        n_splits=3,
        random_seed=11,
    )
    baseline = CrossValidationRunner(
        config,
        fold_pipeline=_mean_imputation_pipeline(),
    ).run(features=features, target=target)
    validation_positions = baseline.folds[0].context.validation_indices
    changed_features = features.copy(deep=True)
    changed_features.iloc[list(validation_positions), 1] += 1000.0

    changed = CrossValidationRunner(
        config,
        fold_pipeline=_mean_imputation_pipeline(),
    ).run(features=changed_features, target=target)

    assert (
        baseline.folds[0].fitted_state_fingerprint
        == changed.folds[0].fitted_state_fingerprint
    )
    assert (
        baseline.folds[0].validation_output_fingerprint
        != changed.folds[0].validation_output_fingerprint
    )
    pd.testing.assert_frame_equal(
        baseline.folds[0].training_features,
        changed.folds[0].training_features,
    )


def test_group_cross_validation_keeps_groups_disjoint() -> None:
    features, target = _classification_data(row_count=48)
    groups = pd.Series(
        np.repeat(np.arange(12), 4),
        index=features.index,
        name="group",
    )
    result = CrossValidationRunner(
        CrossValidationConfig(
            strategy=CrossValidationStrategy.GROUP,
            n_splits=4,
        )
    ).run(
        features=features,
        target=target,
        groups=groups,
    )

    assert len(result.folds) == 4
    for fold in result.folds:
        train_groups = set(fold.context.training_groups)
        validation_groups = set(fold.context.validation_groups)
        assert train_groups.isdisjoint(validation_groups)


def test_group_cross_validation_requires_groups() -> None:
    features, target = _classification_data()
    runner = CrossValidationRunner(
        CrossValidationConfig(strategy=CrossValidationStrategy.GROUP)
    )

    with pytest.raises(ValueError, match="groups are required"):
        runner.run(features=features, target=target)


def test_time_series_cross_validation_never_trains_on_later_rows() -> None:
    features, target = _classification_data(row_count=50)
    result = CrossValidationRunner(
        CrossValidationConfig(
            strategy=CrossValidationStrategy.TIME_SERIES,
            n_splits=4,
        )
    ).run(features=features, target=target)

    assert len(result.folds) == 4
    for fold in result.folds:
        assert max(fold.context.training_indices) < min(
            fold.context.validation_indices
        )


def test_time_series_configuration_rejects_shuffle() -> None:
    with pytest.raises(ValueError, match="shuffle=True"):
        CrossValidationConfig(
            strategy=CrossValidationStrategy.TIME_SERIES,
            shuffle=True,
        )


def test_validation_transform_that_mutates_its_input_is_rejected() -> None:
    features, target = _classification_data()
    pipeline = FoldPipeline(
        steps=[
            FoldStep(
                name="mutating",
                factory=MutatingTransformer,
            )
        ]
    )
    runner = CrossValidationRunner(
        CrossValidationConfig(n_splits=3),
        fold_pipeline=pipeline,
    )

    with pytest.raises(ValueError, match="mutated its validation input"):
        runner.run(features=features, target=target)


def test_class_weight_calculation_requires_target_authorization() -> None:
    features, target = _classification_data()
    runner = CrossValidationRunner(
        CrossValidationConfig(n_splits=3),
        fold_pipeline=FoldPipeline(
            class_weight_options=ClassWeightOptions(),
        ),
    )

    with pytest.raises(PermissionError, match="target-access authorization"):
        runner.run(features=features, target=target)


def test_resampling_and_class_weights_use_fold_training_only() -> None:
    features, target = _classification_data()
    target.iloc[:45] = 0
    target.iloc[45:] = 1
    pipeline = FoldPipeline(
        steps=[
            FoldStep(
                name="train_mean_imputer",
                factory=TrainingMeanImputer,
            )
        ],
        class_weight_options=ClassWeightOptions(),
        sampler_factory=lambda: RandomOverSampler(random_seed=13),
        resampling_authorized=True,
        target_access_authorized=True,
    )
    result = CrossValidationRunner(
        CrossValidationConfig(n_splits=3, random_seed=13),
        fold_pipeline=pipeline,
    ).run(features=features, target=target)

    for fold in result.folds:
        assert fold.sampler_state is not None
        assert fold.class_weights is not None
        assert fold.class_weights.total_count == len(
            fold.context.training_target
        )
        assert len(fold.training_features) >= len(
            fold.context.training_features
        )
        assert len(fold.validation_features) == len(
            fold.context.validation_features
        )
        assert fold.sampler_state.fitted_on_training_only
