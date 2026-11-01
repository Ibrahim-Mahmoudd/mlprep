from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlprep.phases.phase7.filters.multicollinearity_filter import (
    MulticollinearityFilter,
    MulticollinearityFilterOptions,
)


def _independent_features() -> pd.DataFrame:
    values = np.linspace(-3.0, 3.0, 80)
    return pd.DataFrame(
        {
            "linear": values,
            "curved": np.sin(values),
            "binary": np.tile([0.0, 1.0], 40),
        }
    )


def test_multicollinearity_filter_keeps_independent_features_deterministically(
) -> None:
    features = _independent_features()
    original = features.copy(deep=True)
    options = MulticollinearityFilterOptions(
        enabled=True,
        maximum_vif=10.0,
        maximum_condition_number=100.0,
    )

    first = MulticollinearityFilter(options).fit(frame=features).get_result()
    second = MulticollinearityFilter(options).fit(frame=features).get_result()

    assert first.retained_features == list(features.columns)
    assert first.removed_features == []
    assert first.converged
    assert first.fitted_on_training_only
    assert first.vif_scores == second.vif_scores
    assert first.condition_number == second.condition_number
    assert first.condition_number_exceeded is False
    pd.testing.assert_frame_equal(features, original)


def test_multicollinearity_filter_removes_highest_vif_and_preserves_feature_order(
) -> None:
    values = np.linspace(-4.0, 4.0, 100)
    features = pd.DataFrame(
        {
            "signal": values,
            "duplicate": values * 2.0,
            "independent": np.cos(values),
        }
    )
    result = MulticollinearityFilter(
        MulticollinearityFilterOptions(
            enabled=True,
            maximum_vif=5.0,
        )
    ).fit(
        frame=features,
        source_feature_order=["signal", "duplicate", "independent"],
    ).get_result()

    assert result.removed_features == ["duplicate"]
    assert result.retained_features == ["signal", "independent"]
    assert result.converged
    assert result.iterations[0].removed_feature == "duplicate"
    assert result.vif_scores["signal"] <= 5.0
    assert result.metadata["removed_feature_count"] == 1


def test_multicollinearity_filter_overrides_required_and_protected_features() -> None:
    values = np.linspace(-2.0, 2.0, 50)
    features = pd.DataFrame(
        {
            "required": values,
            "protected": values * 3.0,
        }
    )
    result = MulticollinearityFilter(
        MulticollinearityFilterOptions(
            enabled=True,
            maximum_vif=5.0,
            maximum_condition_number=10.0,
        )
    ).fit(
        frame=features,
        required_features=["required"],
        protected_features=["protected"],
    ).get_result()

    assert result.removed_features == []
    assert result.overridden_features == ["required", "protected"]
    assert result.retained_features == ["required", "protected"]
    assert result.converged is False
    assert result.condition_number_exceeded
    assert "required or protected" in result.iterations[0].removal_reason


def test_multicollinearity_filter_respects_iteration_limit_and_single_feature() -> None:
    values = np.linspace(-2.0, 2.0, 40)
    features = pd.DataFrame(
        {
            "first": values,
            "second": values * 2.0,
            "third": values * -4.0,
        }
    )
    limited = MulticollinearityFilter(
        MulticollinearityFilterOptions(
            enabled=True,
            maximum_vif=5.0,
            maximum_iterations=1,
        )
    ).fit(frame=features).get_result()
    single = MulticollinearityFilter(
        MulticollinearityFilterOptions(enabled=True)
    ).fit(frame=features, candidate_features=["first"]).get_result()

    assert len(limited.iterations) == 1
    assert limited.converged is False
    assert len(limited.removed_features) == 1
    assert single.retained_features == ["first"]
    assert single.converged
    assert single.vif_scores == {"first": 1.0}


def test_multicollinearity_filter_disabled_and_unfitted_results() -> None:
    features = _independent_features()
    disabled = MulticollinearityFilter().fit(
        frame=features,
        candidate_features=["curved", "linear"],
    )

    assert disabled.get_result().retained_features == [
        "curved",
        "linear",
    ]
    assert disabled.get_result().metadata == {"enabled": False}
    assert disabled.get_result() is not disabled.get_result()
    with pytest.raises(RuntimeError, match="not fitted"):
        MulticollinearityFilter().get_result()


@pytest.mark.parametrize(
    ("options", "message"),
    [
        ({"maximum_vif": 1.0}, "must exceed one"),
        ({"maximum_condition_number": 1.0}, "must exceed one"),
        ({"maximum_iterations": 0}, "must be positive"),
        ({"numerical_tolerance": 0.0}, "must be positive"),
    ],
)
def test_multicollinearity_filter_rejects_invalid_options(
    options: dict[str, float | int],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        MulticollinearityFilterOptions(**options)


@pytest.mark.parametrize(
    ("frame", "features", "error", "message"),
    [
        (
            pd.DataFrame({"present": [1.0, 2.0]}),
            ["missing"],
            KeyError,
            "features are missing",
        ),
        (
            pd.DataFrame({"text": ["a", "b"]}),
            ["text"],
            TypeError,
            "requires numeric features",
        ),
        (
            pd.DataFrame({"missing": [1.0, np.nan]}),
            ["missing"],
            ValueError,
            "requires complete finite features",
        ),
        (
            pd.DataFrame({"infinite": [1.0, np.inf]}),
            ["infinite"],
            ValueError,
            "requires complete finite features",
        ),
    ],
)
def test_multicollinearity_filter_rejects_invalid_feature_frames(
    frame: pd.DataFrame,
    features: list[str],
    error: type[Exception],
    message: str,
) -> None:
    with pytest.raises(error, match=message):
        MulticollinearityFilter(
            MulticollinearityFilterOptions(enabled=True)
        ).fit(frame=frame, candidate_features=features)
