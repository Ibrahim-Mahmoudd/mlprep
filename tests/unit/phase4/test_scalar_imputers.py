from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlprep.phases.phase4.strategies.base import ImputerExecutionMode
from mlprep.phases.phase4.strategies.boolean_imputer import (
    BooleanImputer,
    BooleanImputerOptions,
)
from mlprep.phases.phase4.strategies.categorical_imputer import (
    CategoricalImputer,
    CategoricalImputerOptions,
)
from mlprep.phases.phase4.strategies.constant_imputer import ConstantImputer
from mlprep.phases.phase4.strategies.datetime_imputer import (
    DatetimeImputer,
    DatetimeImputerOptions,
)
from mlprep.phases.phase4.strategies.numeric_imputer import (
    NumericImputer,
    NumericImputerOptions,
)
from mlprep.phases.phase4.strategies.temporal_imputer import (
    TemporalImputer,
    TemporalImputerOptions,
)
from mlprep.phases.phase4.strategies.text_imputer import (
    TextImputer,
    TextImputerOptions,
)


def _transform(imputer, values: pd.Series) -> pd.Series:
    imputer.fit(values)
    return imputer.transform(
        values,
        mode=ImputerExecutionMode.VALIDATION_TRANSFORM,
    )


@pytest.mark.parametrize(
    ("method", "expected"),
    [
        ("constant_true", True),
        ("constant_false", False),
        ("most_frequent", True),
    ],
)
def test_boolean_imputer_methods_preserve_nullable_boolean_dtype(
    method: str,
    expected: bool,
) -> None:
    values = pd.Series([True, None, False, True], name="flag", dtype=object)
    result = _transform(
        BooleanImputer(
            column="flag",
            options=BooleanImputerOptions(method=method),
        ),
        values,
    )

    assert result.tolist() == [True, expected, False, True]
    assert str(result.dtype) == "boolean"


def test_boolean_imputer_normalizes_common_labels_and_rejects_bad_values() -> None:
    values = pd.Series(["yes", "1", "no", None], name="flag")
    imputer = BooleanImputer(
        column="flag",
        options=BooleanImputerOptions(method="constant_true"),
    )
    transformed = _transform(imputer, values)

    assert transformed.tolist() == [True, True, False, True]
    assert BooleanImputer._normalize_boolean(" Y ") is True
    assert BooleanImputer._normalize_boolean("F") is False
    with pytest.raises(ValueError, match="not Boolean-like"):
        BooleanImputer._normalize_boolean("maybe")
    with pytest.raises(ValueError, match="cannot be empty"):
        BooleanImputerOptions(
            method="missing_category",
            missing_category_value="",
        )


def test_boolean_imputer_missing_category_uses_configured_label() -> None:
    values = pd.Series([True, None, False], name="flag")
    result = _transform(
        BooleanImputer(
            column="flag",
            options=BooleanImputerOptions(
                method="missing_category",
                missing_category_value="unknown",
            ),
        ),
        values,
    )

    assert result.tolist() == [True, "unknown", False]
    assert result.dtype == object


def test_categorical_imputer_modes_constants_and_empty_training() -> None:
    values = pd.Series(["z", "a", None, "a"], name="kind")
    assert _transform(
        CategoricalImputer(
            column="kind",
            options=CategoricalImputerOptions(method="most_frequent"),
        ),
        values,
    ).tolist() == ["z", "a", "a", "a"]

    categorical = pd.Series(
        pd.Categorical(["low", None], categories=["low", "high"]),
        name="kind",
    )
    result = _transform(
        CategoricalImputer(
            column="kind",
            options=CategoricalImputerOptions(
                method="constant",
                constant_value="unknown",
                preserve_category_dtype=False,
            ),
        ),
        categorical,
    )
    assert result.tolist() == ["low", "unknown"]
    assert "unknown" in result.cat.categories
    with pytest.raises(ValueError, match="no valid training values"):
        CategoricalImputer(
            column="kind",
            options=CategoricalImputerOptions(method="most_frequent"),
        ).fit(pd.Series([None, None], name="kind"))
    with pytest.raises(ValueError, match="cannot be empty"):
        CategoricalImputerOptions(method="constant", constant_value="")


def test_constant_imputer_handles_categories_and_rejects_missing_constant() -> None:
    values = pd.Series(
        pd.Categorical(["a", None], categories=["a", "b"]),
        name="kind",
    )
    result = _transform(ConstantImputer(column="kind", value="other"), values)
    assert result.tolist() == ["a", "other"]
    assert "other" in result.cat.categories
    assert ConstantImputer(column="count", value=3).fit(
        pd.Series([None, 1], name="count")
    ).state.learned_value == 3
    with pytest.raises(ValueError, match="cannot be missing"):
        ConstantImputer(column="value", value=np.nan)


@pytest.mark.parametrize("resolution", ["us", "ns"])
def test_datetime_imputer_methods_and_invalid_training_data(
    resolution: str,
) -> None:
    values = pd.Series(
        pd.to_datetime(["2024-01-01", None, "2024-01-03"]).as_unit(
            resolution
        ),
        name="timestamp",
    )
    median = _transform(
        DatetimeImputer(
            column="timestamp",
            options=DatetimeImputerOptions(method="median"),
        ),
        values,
    )
    assert median.iloc[1] == pd.Timestamp("2024-01-02")

    constant = _transform(
        DatetimeImputer(
            column="timestamp",
            options=DatetimeImputerOptions(
                method="constant",
                constant_value="2020-05-06",
            ),
        ),
        values,
    )
    assert constant.iloc[1] == pd.Timestamp("2020-05-06")

    frequent = _transform(
        DatetimeImputer(
            column="timestamp",
            options=DatetimeImputerOptions(method="most_frequent"),
        ),
        pd.Series(
            ["2024-01-01", "2024-01-01", "2024-01-03", None],
            name="timestamp",
        ),
    )
    assert frequent.iloc[-1] == pd.Timestamp("2024-01-01")
    with pytest.raises(ValueError, match="no valid training timestamps"):
        DatetimeImputer(column="timestamp").fit(
            pd.Series(["invalid", None], name="timestamp")
        )
    with pytest.raises(ValueError, match="requires constant_value"):
        DatetimeImputerOptions(method="constant")


def test_text_imputer_methods_and_dtype_control() -> None:
    values = pd.Series(["z", "a", None, "a"], name="comment")
    assert _transform(
        TextImputer(
            column="comment",
            options=TextImputerOptions(method="most_frequent"),
        ),
        values,
    ).tolist() == ["z", "a", "a", "a"]

    for method, expected in (
        ("constant", "constant"),
        ("empty_string", ""),
        ("missing_token", "__MISSING_TEXT__"),
    ):
        output = _transform(
            TextImputer(
                column="comment",
                options=TextImputerOptions(
                    method=method,
                    constant_value="constant",
                ),
            ),
            pd.Series(["text", None], name="comment"),
        )
        assert output.iloc[1] == expected

    object_result = _transform(
        TextImputer(
            column="comment",
            options=TextImputerOptions(
                method="missing_token",
                preserve_string_dtype=False,
            ),
        ),
        pd.Series(["text", None], name="comment"),
    )
    assert object_result.dtype == object
    with pytest.raises(ValueError, match="cannot be empty"):
        TextImputerOptions(method="missing_token", missing_token="")
    with pytest.raises(ValueError, match="no valid training values"):
        TextImputer(
            column="comment",
            options=TextImputerOptions(method="most_frequent"),
        ).fit(pd.Series([None], name="comment"))


def test_numeric_imputer_quantile_trimmed_mean_and_nullable_preservation() -> None:
    values = pd.Series([0.0, 1.0, 2.0, 3.0, 100.0, np.nan], name="amount")
    trimmed = NumericImputer(
        column="amount",
        options=NumericImputerOptions(
            method="trimmed_mean",
            trim_fraction=0.2,
        ),
    )
    result = _transform(trimmed, values)
    assert trimmed.state.learned_value == pytest.approx(2.0)
    assert result.iloc[-1] == pytest.approx(2.0)

    quantile = NumericImputer(
        column="amount",
        options=NumericImputerOptions(method="quantile", quantile=0.75),
    )
    quantile.fit(pd.Series([0.0, 10.0], name="amount"))
    assert quantile.state.learned_value == pytest.approx(7.5)

    nullable = NumericImputer(
        column="amount",
        options=NumericImputerOptions(
            method="constant",
            constant_value=2.0,
            preserve_nullable_dtype=True,
        ),
    )
    nullable_result = _transform(
        nullable,
        pd.Series([1, None], name="amount", dtype="Int64"),
    )
    assert nullable_result.tolist() == [1, 2]
    assert str(nullable_result.dtype) == "Int64"
    with pytest.raises(ValueError, match="quantile"):
        NumericImputerOptions(quantile=2.0)
    with pytest.raises(ValueError, match="trim_fraction"):
        NumericImputerOptions(trim_fraction=0.5)


def test_temporal_imputer_forward_fills_in_time_order_without_future_values() -> None:
    training = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                ["2024-01-01", "2024-01-02", "2024-01-03"]
            ),
            "value": [10.0, None, 30.0],
        }
    )
    imputer = TemporalImputer(
        column="value",
        timestamp_column="timestamp",
        options=TemporalImputerOptions(
            method="forward_fill_then_constant",
        ),
    ).fit_frame(training)

    validation = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                ["2024-01-03", "2024-01-01", "2024-01-02"]
            ),
            "value": [None, None, 20.0],
        },
        index=[9, 7, 8],
    )
    output = imputer.transform_frame(
        validation,
        mode=ImputerExecutionMode.VALIDATION_TRANSFORM,
    )

    assert output.tolist() == [20.0, 10.0, 20.0]
    assert output.index.equals(validation.index)
    assert imputer.state.metadata["future_values_allowed"] is False
    assert imputer.state.transformed_missing_count == 0


def test_temporal_imputer_rejects_invalid_modes_and_timestamp_contracts() -> None:
    frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2024-01-01", "2024-01-02"]),
            "value": [1.0, None],
        }
    )
    imputer = TemporalImputer(
        column="value",
        timestamp_column="timestamp",
        options=TemporalImputerOptions(method="forward_fill"),
    )
    with pytest.raises(ValueError, match="Backward filling is disabled"):
        TemporalImputerOptions(allow_backward_fill=True)
    with pytest.raises(ValueError, match="timestamp_column"):
        TemporalImputer(column="value", timestamp_column=" ")
    with pytest.raises(ValueError, match="Use fit_frame"):
        imputer.transform_frame(frame, mode=ImputerExecutionMode.FIT)
    with pytest.raises(RuntimeError, match="not fitted"):
        imputer.transform_frame(
            frame,
            mode=ImputerExecutionMode.TEST_TRANSFORM,
        )

    imputer.fit_frame(frame)
    assert imputer.transform_frame(
        frame,
        mode=ImputerExecutionMode.TEST_TRANSFORM,
    ).tolist() == [1.0, 1.0]
    with pytest.raises(RuntimeError, match="already fitted"):
        imputer.fit_frame(frame)
    with pytest.raises(ValueError, match="valid timestamps"):
        imputer.transform_frame(
            pd.DataFrame({"timestamp": ["invalid"], "value": [None]}),
            mode=ImputerExecutionMode.TEST_TRANSFORM,
        )
    with pytest.raises(ValueError, match="Equal timestamps"):
        TemporalImputer(
            column="value",
            timestamp_column="timestamp",
            options=TemporalImputerOptions(allow_equal_timestamps=False),
        ).fit_frame(
            pd.DataFrame(
                {
                    "timestamp": pd.to_datetime(
                        ["2024-01-01", "2024-01-01"]
                    ),
                    "value": [1.0, 2.0],
                }
            )
        )
