from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlprep.phases.phase4.strategies.base import ImputerExecutionMode
from mlprep.phases.phase4.strategies.group_imputer import (
    GroupImputer,
    GroupImputerOptions,
)
from mlprep.phases.phase4.strategies.model_imputer import (
    ModelImputer,
    ModelImputerOptions,
)


def test_group_imputer_uses_group_statistic_and_global_fallback() -> None:
    training = pd.DataFrame(
        {
            "group": ["a", "a", "b", "b"],
            "value": [1.0, np.nan, 9.0, np.nan],
        },
        index=[10, 11, 12, 13],
    )
    imputer = GroupImputer(
        column="value",
        group_columns=["group"],
        options=GroupImputerOptions(method="mean", fallback_method="median"),
    ).fit_frame(training)

    validation = pd.DataFrame(
        {
            "group": ["a", "b", "unseen"],
            "value": [np.nan, np.nan, np.nan],
        },
        index=[20, 21, 22],
    )
    transformed = imputer.transform_frame(
        validation,
        mode=ImputerExecutionMode.VALIDATION_TRANSFORM,
    )

    assert transformed.tolist() == [1.0, 9.0, 5.0]
    assert transformed.index.equals(validation.index)
    assert imputer.state.metadata["group_mapping_count"] == 2
    assert imputer.state.transformed_missing_count == 0


def test_group_imputer_most_frequent_tie_is_deterministic() -> None:
    imputer = GroupImputer(
        column="value",
        group_columns=["group"],
        options=GroupImputerOptions(
            method="most_frequent",
            fallback_method="most_frequent",
        ),
    ).fit_frame(
        pd.DataFrame(
            {"group": ["g", "g", "g"], "value": ["z", "a", None]}
        )
    )

    result = imputer.transform_frame(
        pd.DataFrame({"group": ["g"], "value": [None]}),
        mode=ImputerExecutionMode.TEST_TRANSFORM,
    )

    assert result.iloc[0] == "a"


def test_group_imputer_rejects_bad_fit_and_transform_usage() -> None:
    with pytest.raises(ValueError, match="requires group columns"):
        GroupImputer(column="value", group_columns=[])

    imputer = GroupImputer(column="value", group_columns=["group"])
    frame = pd.DataFrame({"value": [1.0]})
    with pytest.raises(ValueError, match="missing columns"):
        imputer.fit_frame(frame)

    valid = pd.DataFrame({"group": ["g"], "value": [1.0]})
    with pytest.raises(ValueError, match="Use fit_frame"):
        imputer.transform_frame(valid, mode=ImputerExecutionMode.FIT)
    with pytest.raises(RuntimeError, match="not fitted"):
        imputer.transform_frame(
            valid,
            mode=ImputerExecutionMode.VALIDATION_TRANSFORM,
        )

    imputer.fit_frame(valid)
    with pytest.raises(RuntimeError, match="already fitted"):
        imputer.fit_frame(valid)
    with pytest.raises(ValueError, match="missing"):
        imputer.transform_frame(
            frame,
            mode=ImputerExecutionMode.VALIDATION_TRANSFORM,
        )


def test_model_imputer_regression_predicts_missing_values_from_train_fit() -> None:
    training = pd.DataFrame(
        {
            "feature": [0.0, 1.0, 2.0, 3.0, 4.0, np.nan],
            "value": [1.0, 3.0, 5.0, 7.0, 9.0, np.nan],
        },
        index=[10, 11, 12, 13, 14, 15],
    )
    imputer = ModelImputer(
        column="value",
        options=ModelImputerOptions(
            enabled=True,
            minimum_training_rows=4,
            predictor_columns=["feature"],
            ridge_alpha=0.0,
        ),
    ).fit_frame(training)

    transformed = imputer.transform_frame(
        pd.DataFrame(
            {"feature": [5.0, np.nan], "value": [np.nan, np.nan]},
            index=[20, 21],
        ),
        mode=ImputerExecutionMode.VALIDATION_TRANSFORM,
    )

    assert transformed.iloc[0] == pytest.approx(11.0)
    assert np.isfinite(transformed.iloc[1])
    assert transformed.index.tolist() == [20, 21]
    assert imputer.state.metadata["training_observed_count"] == 5


def test_model_imputer_classification_uses_training_majority_fallback() -> None:
    training = pd.DataFrame(
        {
            "feature": [0.0, 1.0, 2.0, 3.0],
            "value": ["yes", "yes", "no", None],
        }
    )
    imputer = ModelImputer(
        column="value",
        options=ModelImputerOptions(
            enabled=True,
            task="classification",
            minimum_training_rows=3,
            predictor_columns=["feature"],
        ),
    ).fit_frame(training)

    result = imputer.transform_frame(
        pd.DataFrame({"feature": [5.0], "value": [None]}),
        mode=ImputerExecutionMode.TEST_TRANSFORM,
    )

    assert result.iloc[0] == "yes"


def test_model_imputer_blocks_disabled_and_invalid_operations() -> None:
    frame = pd.DataFrame({"feature": [1.0, 2.0], "value": [1.0, np.nan]})
    disabled = ModelImputer(column="value")
    with pytest.raises(PermissionError, match="disabled"):
        disabled.fit_frame(frame)

    enabled = ModelImputer(
        column="value",
        options=ModelImputerOptions(enabled=True, minimum_training_rows=2),
    )
    with pytest.raises(ValueError, match="does not exist"):
        ModelImputer(
            column="missing",
            options=ModelImputerOptions(enabled=True),
        ).fit_frame(frame)
    with pytest.raises(ValueError, match="Too few observed"):
        enabled.fit_frame(frame)

    no_predictors = ModelImputer(
        column="value",
        options=ModelImputerOptions(
            enabled=True,
            minimum_training_rows=2,
            predictor_columns=[],
        ),
    )
    with pytest.raises(ValueError, match="requires numeric predictors"):
        no_predictors.fit_frame(frame)
