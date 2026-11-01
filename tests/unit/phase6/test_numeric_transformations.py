from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlprep.phases.phase6.config_loader import Phase6ConfigLoader
from mlprep.phases.phase6.contract.transformation_plan import (
    FeatureFamily,
    NumericScalingKind,
    NumericTransformationKind,
    TransformationPlan,
    TransformationPlanEntry,
)
from mlprep.phases.phase6.numeric.base import (
    NumericExecutionMode,
    NumericTransformerState,
)
from mlprep.phases.phase6.numeric.box_cox_transformer import (
    BoxCoxTransformer,
    BoxCoxTransformerOptions,
)
from mlprep.phases.phase6.numeric.log_transformer import (
    LogTransformer,
    LogTransformerOptions,
)
from mlprep.phases.phase6.numeric.minmax_scaler import (
    MinMaxScaler,
    MinMaxScalerOptions,
)
from mlprep.phases.phase6.numeric.numeric_pipeline import (
    NumericPipeline,
    NumericPipelineOptions,
)
from mlprep.phases.phase6.numeric.power_transformer import (
    PowerTransformer,
    PowerTransformerOptions,
)
from mlprep.phases.phase6.numeric.robust_scaler import RobustScaler
from mlprep.phases.phase6.numeric.standard_scaler import (
    StandardScaler,
    StandardScalerOptions,
)


def _entry(
    column: str,
    transformation: NumericTransformationKind,
    scaling: NumericScalingKind,
) -> TransformationPlanEntry:
    return TransformationPlanEntry(
        column=column,
        feature_family=FeatureFamily.NUMERIC,
        numeric_transformation=transformation,
        numeric_scaling=scaling,
        reason="test numeric transformation",
    )


def _transformer_state_round_trip(transformer) -> None:
    state = transformer.get_state()
    restored_state = NumericTransformerState.model_validate(
        state.model_dump(mode="json")
    )
    restored = type(transformer)(column=transformer.column)
    restored.state = restored_state

    values = pd.Series(
        [0.5, 2.0],
        index=[31, 14],
        name=transformer.column,
    )
    expected = transformer.transform(
        values,
        mode=NumericExecutionMode.VALIDATION_TRANSFORM,
    )
    actual = restored.transform(
        values,
        mode=NumericExecutionMode.VALIDATION_TRANSFORM,
    )

    pd.testing.assert_series_equal(actual, expected)


def test_log1p_handles_positive_zero_and_lower_domain_boundary() -> None:
    values = pd.Series(
        [-0.5, 0.0, 1.0],
        index=["lower", "zero", "positive"],
        name="value",
    )
    original = values.copy(deep=True)
    transformer = LogTransformer(column="value")

    transformed = transformer.fit_transform(values)

    assert transformed.iloc[0] == pytest.approx(np.log(0.5))
    assert transformed.iloc[1] == pytest.approx(0.0)
    assert transformed.iloc[2] == pytest.approx(np.log(2.0))
    assert transformed.index.equals(values.index)
    assert transformed.name == values.name
    assert values.equals(original)
    with pytest.raises(ValueError, match="below -1"):
        transformer.transform(
            pd.Series([-1.01], name="value"),
            mode=NumericExecutionMode.VALIDATION_TRANSFORM,
        )
    with pytest.raises(ValueError, match="nonfinite values"):
        transformer.transform(
            pd.Series([-1.0], name="value"),
            mode=NumericExecutionMode.VALIDATION_TRANSFORM,
        )


def test_signed_log1p_supports_negative_zero_and_positive_values() -> None:
    values = pd.Series([-10.0, 0.0, 10.0], name="value")
    transformer = LogTransformer(
        column="value",
        options=LogTransformerOptions(method="signed_log1p"),
    )

    transformed = transformer.fit_transform(values)

    assert transformed.tolist() == pytest.approx(
        [-np.log(11.0), 0.0, np.log(11.0)]
    )
    assert np.isfinite(transformed).all()


def test_log1p_rejects_training_values_outside_the_domain() -> None:
    with pytest.raises(ValueError, match="training values below -1"):
        LogTransformer(column="value").fit(
            pd.Series([-2.0, 1.0], name="value")
        )


def test_power_transformer_fits_yeo_johnson_on_mixed_sign_values() -> None:
    values = pd.Series(
        [-8.0, -2.0, 0.0, 1.0, 5.0],
        index=[7, 9, 12, 14, 20],
        name="value",
    )
    original = values.copy(deep=True)
    transformer = PowerTransformer(
        column="value",
        options=PowerTransformerOptions(search_steps=31),
    )

    transformed = transformer.fit_transform(values)

    assert transformed.index.equals(values.index)
    assert transformed.name == "value"
    assert np.isfinite(transformed).all()
    assert transformer.lambda_value is not None
    assert -5.0 <= transformer.lambda_value <= 5.0
    assert values.equals(original)


def test_power_transformer_constant_column_uses_identity_lambda() -> None:
    transformer = PowerTransformer(column="value")

    transformed = transformer.fit_transform(
        pd.Series([-3.0, -3.0, -3.0], name="value")
    )

    assert transformed.tolist() == [-3.0, -3.0, -3.0]
    assert transformer.get_state().constant_feature
    assert transformer.get_state().learned_parameters["lambda"] == 1.0


def test_box_cox_transformer_requires_positive_training_and_transform_values() -> None:
    values = pd.Series(
        [0.25, 1.0, 2.0, 8.0],
        index=[9, 3, 5, 1],
        name="value",
    )
    transformer = BoxCoxTransformer(
        column="value",
        options=BoxCoxTransformerOptions(search_steps=31),
    )
    transformed = transformer.fit_transform(values)

    assert transformed.index.equals(values.index)
    assert transformed.name == "value"
    assert np.isfinite(transformed).all()
    assert transformer.lambda_value is not None
    with pytest.raises(ValueError, match="strictly positive"):
        BoxCoxTransformer(column="value").fit(
            pd.Series([0.0, 1.0, 2.0], name="value")
        )
    with pytest.raises(ValueError, match="strictly positive"):
        transformer.transform(
            pd.Series([-0.5], name="value"),
            mode=NumericExecutionMode.VALIDATION_TRANSFORM,
        )


def test_minmax_scaler_training_bounds_clipping_and_constant_columns() -> None:
    values = pd.Series([2.0, 4.0, 6.0], index=[4, 5, 6], name="amount")
    scaler = MinMaxScaler(
        column="amount",
        options=MinMaxScalerOptions(
            output_minimum=-1.0,
            output_maximum=1.0,
            clip_to_output_range=True,
        ),
    )
    scaled = scaler.fit_transform(values)
    validation = scaler.transform(
        pd.Series([-10.0, 10.0], index=[8, 3], name="amount"),
        mode=NumericExecutionMode.VALIDATION_TRANSFORM,
    )

    assert scaled.tolist() == pytest.approx([-1.0, 0.0, 1.0])
    assert validation.tolist() == [-1.0, 1.0]
    assert validation.index.tolist() == [8, 3]
    assert scaler.get_state().learned_parameters["training_minimum"] == 2.0
    assert scaler.get_state().learned_parameters["training_maximum"] == 6.0

    constant = MinMaxScaler(column="constant")
    assert constant.fit_transform(
        pd.Series([5.0, 5.0, 5.0], name="constant")
    ).tolist() == [0.0, 0.0, 0.0]
    assert constant.get_state().constant_feature


def test_standard_scaler_options_zero_variance_and_float32() -> None:
    training = pd.Series([1.0, 3.0, 5.0], name="value")
    scaler = StandardScaler(
        column="value",
        options=StandardScalerOptions(
            with_mean=False,
            with_standard_deviation=True,
            degrees_of_freedom=1,
            preserve_float32=True,
        ),
    )
    scaled = scaler.fit_transform(training)

    assert str(scaled.dtype) == "float32"
    assert scaled.tolist() == pytest.approx(
        [1.0 / 2.0, 3.0 / 2.0, 5.0 / 2.0]
    )
    assert scaler.get_state().learned_parameters["mean"] == 3.0
    assert scaler.get_state().learned_parameters["scale"] == pytest.approx(2.0)

    constant = StandardScaler(column="constant")
    assert constant.fit_transform(
        pd.Series([3.0, 3.0, 3.0], name="constant")
    ).tolist() == [0.0, 0.0, 0.0]
    assert constant.get_state().constant_feature


def test_robust_scaler_handles_zero_iqr_and_custom_quantiles() -> None:
    values = pd.Series([0.0, 0.0, 0.0, 0.0, 10.0], name="value")
    scaler = RobustScaler(column="value")

    transformed = scaler.fit_transform(values)

    assert np.isfinite(transformed).all()
    assert scaler.get_state().constant_feature
    assert scaler.get_state().learned_parameters["scale"] == 1.0
    assert transformed.tolist() == [0.0, 0.0, 0.0, 0.0, 10.0]


@pytest.mark.parametrize(
    "transformer",
    [
        LogTransformer(column="value"),
        PowerTransformer(column="value"),
        BoxCoxTransformer(column="value"),
        MinMaxScaler(column="value"),
        StandardScaler(column="value"),
        RobustScaler(column="value"),
    ],
)
def test_numeric_transformers_require_fit_and_reject_missing_nonfinite_values(
    transformer,
) -> None:
    with pytest.raises(RuntimeError, match="not fitted"):
        transformer.transform(
            pd.Series([1.0], name="value"),
            mode=NumericExecutionMode.VALIDATION_TRANSFORM,
        )

    for invalid_values, message in (
        ([1.0, np.nan], "missing numeric values"),
        ([1.0, np.inf], "nonfinite values"),
    ):
        with pytest.raises(ValueError, match=message):
            type(transformer)(column="value").fit(
                pd.Series(invalid_values, name="value")
            )


def test_numeric_transformer_rejects_fit_on_nontraining_partitions() -> None:
    with pytest.raises(PermissionError, match="FIT mode"):
        MinMaxScaler(column="value").fit(
            pd.Series([1.0, 2.0], name="value"),
            mode=NumericExecutionMode.VALIDATION_TRANSFORM,
        )


def test_fitted_numeric_statistics_are_training_only_and_stable() -> None:
    train = pd.Series([2.0, 4.0, 8.0], name="value")
    validation = pd.Series([100.0, 200.0], name="value")
    scaler = StandardScaler(column="value").fit(train)
    learned_before = scaler.get_state().learned_parameters.copy()
    scaler.transform(
        validation,
        mode=NumericExecutionMode.VALIDATION_TRANSFORM,
    )
    learned_after = scaler.get_state().learned_parameters.copy()
    changed_validation = pd.Series([-5000.0, 9000.0], name="value")
    scaler.transform(
        changed_validation,
        mode=NumericExecutionMode.VALIDATION_TRANSFORM,
    )

    assert learned_after == learned_before
    assert scaler.get_state().learned_parameters == learned_before
    assert scaler.get_state().fitted_on_training_only


@pytest.mark.parametrize(
    "factory",
    [
        lambda: LogTransformer(column="value"),
        lambda: PowerTransformer(column="value"),
        lambda: MinMaxScaler(column="value"),
        lambda: StandardScaler(column="value"),
        lambda: RobustScaler(column="value"),
    ],
)
def test_numeric_transformer_state_round_trip(factory) -> None:
    transformer = factory()
    transformer.fit(
        pd.Series([0.5, 1.0, 2.0, 4.0], name="value")
    )

    _transformer_state_round_trip(transformer)


def test_numeric_pipeline_preserves_schema_order_and_transform_order() -> None:
    frame = pd.DataFrame(
        {
            "first": [0.0, 1.0, 3.0, 7.0],
            "second": [-3.0, -1.0, 0.0, 5.0],
            "untouched": ["a", "b", "c", "d"],
        },
        index=[10, 4, 12, 8],
    )
    original = frame.copy(deep=True)
    plan = TransformationPlan(
        entries={
            "first": _entry(
                "first",
                NumericTransformationKind.LOG1P,
                NumericScalingKind.MINMAX,
            ),
            "second": _entry(
                "second",
                NumericTransformationKind.SIGNED_LOG1P,
                NumericScalingKind.STANDARD,
            ),
        },
        numeric_columns=["first", "second"],
    )
    pipeline = NumericPipeline()

    output = pipeline.fit_transform(frame=frame, plan=plan)
    expected_first = np.log1p(frame["first"])
    expected_first = (expected_first - expected_first.min()) / (
        expected_first.max() - expected_first.min()
    )
    signed_log = np.sign(frame["second"]) * np.log1p(
        np.abs(frame["second"])
    )
    expected_second = (signed_log - signed_log.mean()) / signed_log.std(ddof=0)

    assert output.output_columns == [
        "first__numeric__log1p__minmax",
        "second__numeric__signed_log1p__standard",
    ]
    assert output.frame.columns.tolist() == output.output_columns
    assert output.frame.index.equals(frame.index)
    assert output.frame.iloc[:, 0].tolist() == pytest.approx(
        expected_first.tolist()
    )
    assert output.frame.iloc[:, 1].tolist() == pytest.approx(
        expected_second.tolist()
    )
    assert frame.equals(original)
    assert output.states["first"].fitted_on_training_only
    assert output.states["first"].transformation_state is not None
    assert output.states["first"].scaler_state is not None
    assert output.states["first"].output_column == output.output_columns[0]


def test_numeric_pipeline_ignores_extra_and_rejects_missing_planned_column() -> None:
    plan = TransformationPlan(
        entries={
            "amount": _entry(
                "amount",
                NumericTransformationKind.NONE,
                NumericScalingKind.MINMAX,
            )
        },
        numeric_columns=["amount"],
    )
    pipeline = NumericPipeline().fit(
        frame=pd.DataFrame(
            {"amount": [1.0, 2.0, 3.0], "extra": ["x", "y", "z"]}
        ),
        plan=plan,
    )
    result = pipeline.transform(
        frame=pd.DataFrame(
            {"amount": [2.0, 4.0], "unknown": [99, 100]},
            index=[3, 5],
        ),
        mode=NumericExecutionMode.VALIDATION_TRANSFORM,
    )

    assert result.frame.columns.tolist() == [
        "amount__numeric__none__minmax"
    ]
    assert result.frame.index.tolist() == [3, 5]
    with pytest.raises(KeyError, match="does not exist"):
        pipeline.transform(
            frame=pd.DataFrame({"extra": [1.0]}),
            mode=NumericExecutionMode.VALIDATION_TRANSFORM,
        )


def test_numeric_pipeline_rejects_transform_before_fit_and_repeat_fit() -> None:
    pipeline = NumericPipeline()
    frame = pd.DataFrame({"value": [1.0, 2.0, 3.0]})
    plan = TransformationPlan(
        entries={
            "value": _entry(
                "value",
                NumericTransformationKind.NONE,
                NumericScalingKind.NONE,
            )
        },
        numeric_columns=["value"],
    )
    with pytest.raises(RuntimeError, match="must be fitted"):
        pipeline.transform(
            frame=frame,
            mode=NumericExecutionMode.VALIDATION_TRANSFORM,
        )

    pipeline.fit(frame=frame, plan=plan)
    with pytest.raises(RuntimeError, match="already fitted"):
        pipeline.fit(frame=frame, plan=plan)
    with pytest.raises(ValueError, match="Use fit"):
        pipeline.transform(
            frame=frame,
            mode=NumericExecutionMode.FIT,
        )


def test_numeric_pipeline_states_round_trip_and_transform_deterministically() -> None:
    frame = pd.DataFrame({"value": [0.0, 1.0, 2.0, 4.0]}, index=[9, 4, 7, 2])
    plan = TransformationPlan(
        entries={
            "value": _entry(
                "value",
                NumericTransformationKind.LOG1P,
                NumericScalingKind.ROBUST,
            )
        },
        numeric_columns=["value"],
    )
    pipeline = NumericPipeline().fit(frame=frame, plan=plan)
    first = pipeline.transform(
        frame=frame,
        mode=NumericExecutionMode.VALIDATION_TRANSFORM,
    )
    second = pipeline.transform(
        frame=frame.copy(deep=True),
        mode=NumericExecutionMode.VALIDATION_TRANSFORM,
    )
    restored_states = {
        column: type(state).model_validate(state.model_dump(mode="json"))
        for column, state in pipeline.get_states().items()
    }

    pd.testing.assert_frame_equal(first.frame, second.frame)
    assert first.states == second.states
    assert restored_states["value"].model_dump(mode="json") == (
        pipeline.get_states()["value"].model_dump(mode="json")
    )


def test_box_cox_plan_and_phase6_configuration_are_supported() -> None:
    config = Phase6ConfigLoader.from_dict(
        {
            "plan": {
                "numeric_transformation_overrides": {
                    "positive": "box_cox"
                }
            },
            "numeric": {
                "box_cox": {
                    "lambda_minimum": -2.0,
                    "lambda_maximum": 3.0,
                    "search_steps": 21,
                }
            },
        }
    )

    assert config.plan.numeric_transformation_overrides[
        "positive"
    ] == NumericTransformationKind.BOX_COX
    assert config.numeric.box_cox.search_steps == 21
    assert config.pipeline.numeric.box_cox.search_steps == 21
    assert NumericTransformationKind.BOX_COX.value == "box_cox"

    plan = TransformationPlan(
        entries={
            "positive": _entry(
                "positive",
                NumericTransformationKind.BOX_COX,
                NumericScalingKind.NONE,
            )
        },
        numeric_columns=["positive"],
    )
    training = pd.DataFrame(
        {"positive": [0.25, 0.5, 1.0, 2.0, 8.0]}
    )
    pipeline = NumericPipeline(config.numeric).fit(
        frame=training,
        plan=plan,
    )
    result = pipeline.transform(
        frame=training,
        mode=NumericExecutionMode.VALIDATION_TRANSFORM,
    )

    assert np.isfinite(result.frame.to_numpy()).all()
    assert (
        result.states["positive"]
        .transformation_state
        .transformer
        == "box_cox"
    )


def test_numeric_pipeline_requires_nonempty_plan_when_configured() -> None:
    with pytest.raises(ValueError, match="plan is empty"):
        NumericPipeline(
            NumericPipelineOptions(require_nonempty_numeric_plan=True)
        ).fit(
            frame=pd.DataFrame(index=[1, 2]),
            plan=TransformationPlan(),
        )
