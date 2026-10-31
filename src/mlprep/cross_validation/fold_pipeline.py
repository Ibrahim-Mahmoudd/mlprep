"""Train-only transformation pipeline and fold generation."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import pandas as pd
from sklearn.model_selection import (
    GroupKFold,
    KFold,
    StratifiedKFold,
    TimeSeriesSplit,
)

from mlprep.cross_validation.config import (
    CrossValidationConfig,
    CrossValidationStrategy,
)
from mlprep.cross_validation.fold_context import (
    FoldContext,
)
from mlprep.cross_validation.fold_result import (
    CrossValidationResult,
    FoldResult,
)
from mlprep.cross_validation.leakage_guard import (
    LeakageGuard,
)
from mlprep.phases.phase8.samplers import (
    BaseTrainingSampler,
    RemovedRowRecord,
    ResampledRowType,
    SamplerExecutionContext,
    SamplerRowRecord,
)
from mlprep.phases.phase8.weighting.class_weights import (
    ClassWeightCalculator,
    ClassWeightOptions,
    ClassWeightResult,
)


@runtime_checkable
class FoldTransformer(Protocol):
    """Transformer fitted on fold train and applied to both partitions."""

    def fit(
        self,
        features: pd.DataFrame,
        target: pd.Series,
    ) -> object: ...

    def transform(
        self,
        features: pd.DataFrame,
    ) -> pd.DataFrame: ...

    def get_state(self) -> Mapping[str, Any]: ...


FoldTransformerFactory = Callable[[], FoldTransformer]
SamplerFactory = Callable[[], BaseTrainingSampler]


@dataclass(frozen=True, slots=True)
class FoldStep:
    """Named factory for a fresh train-only transformer."""

    name: str
    factory: FoldTransformerFactory

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("Fold step name cannot be empty.")
        if not callable(self.factory):
            raise TypeError("Fold step factory must be callable.")


class FoldPipeline:
    """Fit transformers on fold train and transform validation without fitting."""

    def __init__(
        self,
        *,
        steps: Sequence[FoldStep] = (),
        class_weight_options: ClassWeightOptions | None = None,
        sampler_factory: SamplerFactory | None = None,
        resampling_authorized: bool = False,
        target_access_authorized: bool = False,
        synthetic_generation_authorized: bool = False,
        leakage_guard: LeakageGuard | None = None,
    ) -> None:
        self.steps = tuple(steps)
        if len({step.name for step in self.steps}) != len(self.steps):
            raise ValueError("Fold step names must be unique.")
        if sampler_factory is not None and not callable(sampler_factory):
            raise TypeError("sampler_factory must be callable.")
        for name, value in (
            ("resampling_authorized", resampling_authorized),
            ("target_access_authorized", target_access_authorized),
            ("synthetic_generation_authorized", synthetic_generation_authorized),
        ):
            if not isinstance(value, bool):
                raise TypeError(f"{name} must be a Boolean.")
        self.class_weight_options = class_weight_options
        self.sampler_factory = sampler_factory
        self.resampling_authorized = resampling_authorized
        self.target_access_authorized = target_access_authorized
        self.synthetic_generation_authorized = (
            synthetic_generation_authorized
        )
        self.leakage_guard = leakage_guard or LeakageGuard()

    def run_fold(self, context: FoldContext) -> FoldResult:
        """Fit each step on training rows and transform validation rows."""

        self.leakage_guard.assert_disjoint_row_ids(
            training_row_ids=context.training_row_ids,
            validation_row_ids=context.validation_row_ids,
        )
        if (
            context.training_groups is not None
            and context.validation_groups is not None
        ):
            self.leakage_guard.assert_disjoint_groups(
                training_groups=context.training_groups,
                validation_groups=context.validation_groups,
            )

        training_features = context.training_features.copy(deep=True)
        training_target = context.training_target.copy(deep=True)
        training_row_ids = context.training_row_ids.copy(deep=True)
        validation_features = context.validation_features.copy(deep=True)
        validation_target = context.validation_target.copy(deep=True)
        validation_row_ids = context.validation_row_ids.copy(deep=True)
        component_states: list[tuple[str, Mapping[str, Any]]] = []

        for step in self.steps:
            transformer = step.factory()
            if not isinstance(transformer, FoldTransformer):
                raise TypeError(
                    f"Fold step {step.name!r} does not implement the "
                    "fit/transform/get_state contract."
                )
            transformer.fit(
                training_features.copy(deep=True),
                training_target.copy(deep=True),
            )
            state = transformer.get_state()
            if not isinstance(state, Mapping):
                raise TypeError(
                    f"Fold step {step.name!r} get_state() must return a mapping."
                )
            component_states.append((step.name, state))

            training_features = self._transform(
                transformer=transformer,
                features=training_features,
                step_name=step.name,
            )
            validation_before = self.leakage_guard.partition_fingerprint(
                features=validation_features,
                target=validation_target,
                row_ids=validation_row_ids,
                groups=context.validation_groups,
            )
            validation_input = validation_features.copy(deep=True)
            validation_output = transformer.transform(validation_input)
            self.leakage_guard.assert_validation_unchanged(
                expected_fingerprint=validation_before,
                features=validation_input,
                target=validation_target,
                row_ids=validation_row_ids,
                groups=context.validation_groups,
            )
            validation_features = self._validate_transform_output(
                validation_output,
                input_features=validation_features,
                step_name=step.name,
            )

        class_weights = self._fit_class_weights(training_target)
        sampler_state = None
        removed_row_lineage: list[RemovedRowRecord] = []
        if self.sampler_factory is None:
            row_lineage = self._identity_lineage(
                target=training_target,
                row_ids=training_row_ids,
            )
        else:
            if not self.resampling_authorized:
                raise PermissionError("Fold resampling is not authorized.")
            if not self.target_access_authorized:
                raise PermissionError(
                    "Fold resampling requires target-access authorization."
                )
            sampler = self.sampler_factory()
            if not isinstance(sampler, BaseTrainingSampler):
                raise TypeError(
                    "sampler_factory must return a BaseTrainingSampler."
                )
            sampler_result = sampler.fit_resample(
                features=training_features,
                target=training_target,
                row_ids=training_row_ids,
                context=SamplerExecutionContext.TRAINING_RESAMPLING,
                resampling_authorized=self.resampling_authorized,
                target_access_authorized=self.target_access_authorized,
                synthetic_generation_authorized=(
                    self.synthetic_generation_authorized
                ),
            )
            training_features = sampler_result.features
            training_target = sampler_result.target
            training_row_ids = sampler_result.row_ids
            row_lineage = sampler_result.row_lineage
            removed_row_lineage = sampler_result.removed_rows
            sampler_state = sampler_result.state

        self.leakage_guard.assert_validation_unchanged(
            expected_fingerprint=context.validation_fingerprint,
            features=context.validation_features,
            target=context.validation_target,
            row_ids=context.validation_row_ids,
            groups=context.validation_groups,
        )
        if class_weights is not None:
            component_states.append(
                (
                    "class_weights",
                    class_weights.model_dump(mode="json"),
                )
            )
        if sampler_state is not None:
            component_states.append(
                (
                    "sampler",
                    sampler_state.model_dump(mode="json"),
                )
            )

        return FoldResult(
            context=context,
            training_features=training_features.copy(deep=True),
            training_target=training_target.copy(deep=True),
            training_row_ids=training_row_ids.copy(deep=True),
            validation_features=validation_features.copy(deep=True),
            validation_target=validation_target,
            validation_row_ids=validation_row_ids,
            class_weights=class_weights,
            row_lineage=row_lineage,
            removed_row_lineage=removed_row_lineage,
            sampler_state=sampler_state,
            fitted_state_fingerprint=(
                self.leakage_guard.fitted_state_fingerprint(
                    training_fingerprint=context.training_fingerprint,
                    component_states=component_states,
                )
            ),
            training_output_fingerprint=(
                self.leakage_guard.partition_fingerprint(
                    features=training_features,
                    target=training_target,
                    row_ids=training_row_ids,
                )
            ),
            validation_output_fingerprint=(
                self.leakage_guard.partition_fingerprint(
                    features=validation_features,
                    target=validation_target,
                    row_ids=validation_row_ids,
                    groups=context.validation_groups,
                )
            ),
            transformer_names=tuple(step.name for step in self.steps),
        )

    def _fit_class_weights(
        self,
        target: pd.Series,
    ) -> ClassWeightResult | None:
        if self.class_weight_options is None:
            return None
        if not self.target_access_authorized:
            raise PermissionError(
                "Fold class-weight calculation requires target-access authorization."
            )
        return ClassWeightCalculator(
            options=self.class_weight_options,
        ).fit(
            target=target,
            target_access_authorized=self.target_access_authorized,
        ).get_result()

    @staticmethod
    def _transform(
        *,
        transformer: FoldTransformer,
        features: pd.DataFrame,
        step_name: str,
    ) -> pd.DataFrame:
        output = transformer.transform(features.copy(deep=True))
        return FoldPipeline._validate_transform_output(
            output,
            input_features=features,
            step_name=step_name,
        )

    @staticmethod
    def _validate_transform_output(
        output: pd.DataFrame,
        *,
        input_features: pd.DataFrame,
        step_name: str,
    ) -> pd.DataFrame:
        if not isinstance(output, pd.DataFrame):
            raise TypeError(
                f"Fold step {step_name!r} must return a DataFrame."
            )
        if len(output) != len(input_features) or not output.index.equals(
            input_features.index
        ):
            raise ValueError(
                f"Fold step {step_name!r} must preserve row count and index."
            )
        if output.columns.duplicated().any():
            raise ValueError(
                f"Fold step {step_name!r} returned duplicate feature names."
            )
        return output.copy(deep=True)

    @staticmethod
    def _identity_lineage(
        *,
        target: pd.Series,
        row_ids: pd.Series,
    ) -> list[SamplerRowRecord]:
        return [
            SamplerRowRecord(
                output_position=position,
                output_row_id=str(row_ids.iloc[position]),
                row_type=ResampledRowType.ORIGINAL,
                class_value=target.iloc[position],
                source_row_ids=[str(row_ids.iloc[position])],
                source_positions=[position],
                original=True,
            )
            for position in range(len(target))
        ]


class CrossValidationRunner:
    """Generate folds and execute a train-only fold pipeline."""

    def __init__(
        self,
        config: CrossValidationConfig | None = None,
        *,
        fold_pipeline: FoldPipeline | None = None,
    ) -> None:
        self.config = config or CrossValidationConfig()
        if not isinstance(self.config, CrossValidationConfig):
            raise TypeError("config must be a CrossValidationConfig.")
        self.fold_pipeline = fold_pipeline or FoldPipeline()

    def run(
        self,
        *,
        features: pd.DataFrame,
        target: pd.Series,
        groups: pd.Series | Sequence[Any] | None = None,
        row_ids: pd.Series | Sequence[Any] | None = None,
    ) -> CrossValidationResult:
        """Execute all folds; no test partition is accepted or accessed."""

        self._validate_inputs(features=features, target=target)
        resolved_groups = self._aligned_series(
            values=groups,
            features=features,
            name="groups",
        )
        resolved_row_ids = self._aligned_series(
            values=row_ids,
            features=features,
            name="row_id",
            default_values=[
                f"cv-row-{position}" for position in range(len(features))
            ],
        ).astype("string")
        if resolved_row_ids.isna().any() or not resolved_row_ids.is_unique:
            raise ValueError("row_ids must be unique and nonmissing.")
        splitter = self._splitter(
            features=features,
            target=target,
            groups=resolved_groups,
        )

        results: list[FoldResult] = []
        for fold_index, (train_positions, validation_positions) in enumerate(
            splitter
        ):
            train_indices = tuple(int(value) for value in train_positions)
            validation_indices = tuple(
                int(value) for value in validation_positions
            )
            if set(train_indices).intersection(validation_indices):
                raise ValueError("Fold train and validation positions overlap.")
            if (
                self.config.strategy == CrossValidationStrategy.TIME_SERIES
                and max(train_indices) >= min(validation_indices)
            ):
                raise ValueError(
                    "Time-series validation must follow all training rows."
                )

            training_groups = (
                resolved_groups.iloc[list(train_indices)].copy(deep=True)
                if resolved_groups is not None
                else None
            )
            validation_groups = (
                resolved_groups.iloc[list(validation_indices)].copy(deep=True)
                if resolved_groups is not None
                else None
            )
            if training_groups is not None and validation_groups is not None:
                self.fold_pipeline.leakage_guard.assert_disjoint_groups(
                    training_groups=training_groups,
                    validation_groups=validation_groups,
                )

            training_features = features.iloc[list(train_indices)].copy(deep=True)
            training_target = target.iloc[list(train_indices)].copy(deep=True)
            training_row_ids = resolved_row_ids.iloc[
                list(train_indices)
            ].copy(deep=True)
            validation_features = features.iloc[
                list(validation_indices)
            ].copy(deep=True)
            validation_target = target.iloc[
                list(validation_indices)
            ].copy(deep=True)
            validation_row_ids = resolved_row_ids.iloc[
                list(validation_indices)
            ].copy(deep=True)
            guard = self.fold_pipeline.leakage_guard
            context = FoldContext(
                fold_index=fold_index,
                training_indices=train_indices,
                validation_indices=validation_indices,
                training_features=training_features,
                training_target=training_target,
                training_row_ids=training_row_ids,
                validation_features=validation_features,
                validation_target=validation_target,
                validation_row_ids=validation_row_ids,
                training_groups=training_groups,
                validation_groups=validation_groups,
                training_fingerprint=guard.partition_fingerprint(
                    features=training_features,
                    target=training_target,
                    row_ids=training_row_ids,
                    groups=training_groups,
                ),
                validation_fingerprint=guard.partition_fingerprint(
                    features=validation_features,
                    target=validation_target,
                    row_ids=validation_row_ids,
                    groups=validation_groups,
                ),
            )
            results.append(self.fold_pipeline.run_fold(context))

        return CrossValidationResult(
            config=self.config,
            folds=tuple(results),
        )

    def _splitter(
        self,
        *,
        features: pd.DataFrame,
        target: pd.Series,
        groups: pd.Series | None,
    ) -> Any:
        strategy = self.config.strategy
        if strategy == CrossValidationStrategy.K_FOLD:
            if groups is not None:
                raise ValueError(
                    "groups are only accepted for group cross-validation."
                )
            shuffle = (
                self.config.shuffle
                if self.config.shuffle is not None
                else False
            )
            splitter = KFold(
                n_splits=self.config.n_splits,
                shuffle=shuffle,
                random_state=(
                    self.config.random_seed if shuffle else None
                ),
            )
            return splitter.split(features, target)
        if strategy == CrossValidationStrategy.STRATIFIED:
            if groups is not None:
                raise ValueError(
                    "groups are only accepted for group cross-validation."
                )
            shuffle = (
                self.config.shuffle
                if self.config.shuffle is not None
                else True
            )
            splitter = StratifiedKFold(
                n_splits=self.config.n_splits,
                shuffle=shuffle,
                random_state=(
                    self.config.random_seed if shuffle else None
                ),
            )
            return splitter.split(features, target)
        if strategy == CrossValidationStrategy.GROUP:
            if groups is None:
                raise ValueError(
                    "groups are required for group cross-validation."
                )
            return GroupKFold(n_splits=self.config.n_splits).split(
                features,
                target,
                groups,
            )
        if groups is not None:
            raise ValueError(
                "groups are only accepted for group cross-validation."
            )
        return TimeSeriesSplit(n_splits=self.config.n_splits).split(features)

    @staticmethod
    def _aligned_series(
        *,
        values: pd.Series | Sequence[Any] | None,
        features: pd.DataFrame,
        name: str,
        default_values: Sequence[Any] | None = None,
    ) -> pd.Series | None:
        if values is None:
            if default_values is None:
                return None
            return pd.Series(
                list(default_values),
                index=features.index,
                name=name,
            )
        if isinstance(values, pd.Series):
            if len(values) != len(features) or not values.index.equals(
                features.index
            ):
                raise ValueError(f"{name} must align with features.")
            return values.copy(deep=True).rename(name)
        if len(values) != len(features):
            raise ValueError(f"{name} must align with features.")
        return pd.Series(list(values), index=features.index, name=name)

    @staticmethod
    def _validate_inputs(
        *,
        features: pd.DataFrame,
        target: pd.Series,
    ) -> None:
        if not isinstance(features, pd.DataFrame):
            raise TypeError("features must be a pandas DataFrame.")
        if not isinstance(target, pd.Series):
            raise TypeError("target must be a pandas Series.")
        if features.empty or target.empty:
            raise ValueError("features and target cannot be empty.")
        if len(features) != len(target) or not features.index.equals(
            target.index
        ):
            raise ValueError("features and target must have aligned rows.")
        if features.columns.duplicated().any():
            raise ValueError("features must have unique column names.")
