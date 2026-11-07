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


class InvalidOutputTransformer(TrainingMeanImputer):
    def __init__(self, output: object, state: object) -> None:
        super().__init__()
        self.output = output
        self.state = state

    def transform(self, features: pd.DataFrame) -> object:
        return self.output

    def get_state(self) -> object:
        return self.state


class DuplicateColumnTransformer(TrainingMeanImputer):
    def transform(self, features: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(
            [[1.0, 2.0]] * len(features),
            index=features.index,
            columns=["value", "value"],
        )


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


def test_fold_pipeline_rejects_invalid_step_definitions() -> None:
    with pytest.raises(ValueError, match="cannot be empty"):
        FoldStep(name=" ", factory=TrainingMeanImputer)
    with pytest.raises(TypeError, match="must be callable"):
        FoldStep(name="invalid", factory=object())  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="must be unique"):
        FoldPipeline(
            steps=[
                FoldStep(name="duplicate", factory=TrainingMeanImputer),
                FoldStep(name="duplicate", factory=TrainingMeanImputer),
            ]
        )
    with pytest.raises(TypeError, match="sampler_factory"):
        FoldPipeline(sampler_factory=object())  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="Boolean"):
        FoldPipeline(resampling_authorized=1)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("factory", "message"),
    [
        (lambda: object(), "does not implement"),
        (
            lambda: InvalidOutputTransformer(
                output=pd.Series([1]),
                state={},
            ),
            "must return a DataFrame",
        ),
        (
            lambda: InvalidOutputTransformer(
                output=pd.DataFrame({"value": [1]}),
                state={},
            ),
            "preserve row count and index",
        ),
        (
            DuplicateColumnTransformer,
            "duplicate feature names",
        ),
        (
            lambda: InvalidOutputTransformer(
                output=pd.DataFrame({"feature_a": [1.0] * 40}),
                state=[],
            ),
            "get_state",
        ),
    ],
)
def test_fold_pipeline_rejects_invalid_transformer_contracts(
    factory,
    message: str,
) -> None:
    features, target = _classification_data()
    runner = CrossValidationRunner(
        CrossValidationConfig(n_splits=3),
        fold_pipeline=FoldPipeline(
            steps=[FoldStep(name="invalid", factory=factory)]
        ),
    )

    with pytest.raises((TypeError, ValueError), match=message):
        runner.run(features=features, target=target)


def test_fold_pipeline_rejects_unauthorized_resampling() -> None:
    features, target = _classification_data()
    with pytest.raises(PermissionError, match="not authorized"):
        CrossValidationRunner(
            CrossValidationConfig(n_splits=3),
            fold_pipeline=FoldPipeline(
                sampler_factory=lambda: object(),
            ),
        ).run(features=features, target=target)

    with pytest.raises(PermissionError, match="target-access"):
        CrossValidationRunner(
            CrossValidationConfig(n_splits=3),
            fold_pipeline=FoldPipeline(
                sampler_factory=lambda: object(),
                resampling_authorized=True,
            ),
        ).run(features=features, target=target)


@pytest.mark.parametrize(
    ("features", "target", "message"),
    [
        (None, pd.Series([1]), "features must"),
        (pd.DataFrame({"x": [1]}), None, "target must"),
        (pd.DataFrame(), pd.Series(dtype=int), "cannot be empty"),
        (
            pd.DataFrame({"x": [1, 2]}),
            pd.Series([1], index=[0]),
            "aligned rows",
        ),
        (
            pd.DataFrame([[1, 2]], columns=["x", "x"]),
            pd.Series([1]),
            "unique column",
        ),
    ],
)
def test_runner_rejects_invalid_inputs(
    features,
    target,
    message: str,
) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        CrossValidationRunner().run(features=features, target=target)


def test_runner_rejects_misaligned_or_duplicate_row_ids() -> None:
    features, target = _classification_data()
    runner = CrossValidationRunner(CrossValidationConfig(n_splits=3))

    with pytest.raises(ValueError, match="row_id must align"):
        runner.run(
            features=features,
            target=target,
            row_ids=pd.Series(range(len(features)), index=features.index + 1),
        )
    with pytest.raises(ValueError, match="unique and nonmissing"):
        runner.run(
            features=features,
            target=target,
            row_ids=["same"] * len(features),
        )
