from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlprep.phases.phase7.filters.correlation_filter import (
    CorrelationFilter,
    CorrelationFilterOptions,
)


def _correlated_frame() -> pd.DataFrame:
    signal = np.linspace(-3.0, 3.0, 40)
    return pd.DataFrame(
        {
            "signal": signal,
            "duplicate": -2.0 * signal,
        }
    )


def test_high_negative_correlation_removes_lower_priority_feature() -> None:
    frame = _correlated_frame()
    before = frame.copy(deep=True)

    filter_ = CorrelationFilter(
        CorrelationFilterOptions(
            maximum_absolute_correlation=0.9,
            include_correlation_matrix=True,
        )
    ).fit(
        frame=frame,
        feature_scores={"duplicate": 0.8},
        variance_scores={"signal": 100.0},
    )
    result = filter_.get_result()

    assert len(result.pairs) == 1
    pair = result.pairs[0]
    assert pair.left_feature == "signal"
    assert pair.right_feature == "duplicate"
    assert pair.correlation == pytest.approx(-1.0)
    assert pair.absolute_correlation == pytest.approx(1.0)
    assert pair.retained_feature == "duplicate"
    assert pair.removal_candidate == "signal"
    assert "higher deterministic retention priority" in pair.retention_reason
    assert result.removal_candidates == ["signal"]
    assert result.retained_features == ["duplicate"]
    assert result.correlation_matrix["signal"]["duplicate"] == pytest.approx(-1.0)
    assert result.fitted_on_training_only
    assert frame.equals(before)


def test_required_and_protected_features_override_removal() -> None:
    frame = _correlated_frame()
    required = CorrelationFilter().fit(
        frame=frame,
        required_features=["signal"],
    ).get_result()
    protected = CorrelationFilter().fit(
        frame=frame,
        protected_features=["duplicate"],
    ).get_result()
    both_required = CorrelationFilter().fit(
        frame=frame,
        required_features=["signal", "duplicate"],
    ).get_result()
    both_protected = CorrelationFilter().fit(
        frame=frame,
        protected_features=["signal", "duplicate"],
    ).get_result()

    assert required.retained_features == ["signal"]
    assert required.pairs[0].retention_reason == "The left feature is required."
    assert protected.retained_features == ["duplicate"]
    assert protected.pairs[0].retention_reason == "The right feature is protected."
    assert both_required.removal_candidates == []
    assert both_required.overridden_features == ["signal", "duplicate"]
    assert both_required.pairs[0].required_override
    assert both_protected.removal_candidates == []
    assert both_protected.overridden_features == ["signal", "duplicate"]
    assert both_protected.pairs[0].protected_override


def test_threshold_is_strict_and_method_selects_correlation_measure() -> None:
    frame = pd.DataFrame(
        {
            "linear": np.arange(1, 21, dtype=float),
            "squared": np.arange(1, 21, dtype=float) ** 2,
        }
    )
    at_boundary = CorrelationFilter(
        CorrelationFilterOptions(
            maximum_absolute_correlation=1.0,
        )
    ).fit(frame=frame).get_result()
    pearson = CorrelationFilter(
        CorrelationFilterOptions(
            maximum_absolute_correlation=0.99,
            method="pearson",
        )
    ).fit(frame=frame).get_result()
    spearman = CorrelationFilter(
        CorrelationFilterOptions(
            maximum_absolute_correlation=0.99,
            method="spearman",
        )
    ).fit(frame=frame).get_result()

    assert at_boundary.pairs == []
    assert pearson.pairs == []
    assert len(spearman.pairs) == 1
    assert spearman.pairs[0].correlation == pytest.approx(1.0)


def test_disabled_and_single_candidate_fit_preserve_requested_order() -> None:
    frame = _correlated_frame()
    disabled = CorrelationFilter(
        CorrelationFilterOptions(enabled=False)
    ).fit(
        frame=frame,
        candidate_features=["duplicate", "signal"],
    )
    single = CorrelationFilter().fit(
        frame=frame,
        candidate_features=["duplicate"],
    )

    assert disabled.get_result().retained_features == [
        "duplicate",
        "signal",
    ]
    assert disabled.get_result().metadata == {"enabled": False}
    assert single.get_result().retained_features == ["duplicate"]
    assert single.get_result().pairs == []


def test_matrix_omits_undefined_correlations_and_result_is_copied() -> None:
    frame = pd.DataFrame(
        {
            "constant": [1.0, 1.0, 1.0, 1.0],
            "varying": [1.0, 2.0, 3.0, 4.0],
        }
    )
    filter_ = CorrelationFilter(
        CorrelationFilterOptions(include_correlation_matrix=True)
    ).fit(frame=frame)
    first = filter_.get_result()
    first.retained_features.clear()
    second = filter_.get_result()

    assert second.retained_features == ["constant", "varying"]
    assert second.correlation_matrix == {
        "constant": {},
        "varying": {"varying": 1.0},
    }


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
            pd.DataFrame({"category": ["a", "b"]}),
            ["category"],
            TypeError,
            "requires numeric features",
        ),
    ],
)
def test_invalid_candidate_features_are_rejected(
    frame: pd.DataFrame,
    features: list[str],
    error: type[Exception],
    message: str,
) -> None:
    with pytest.raises(error, match=message):
        CorrelationFilter().fit(
            frame=frame,
            candidate_features=features,
        )


@pytest.mark.parametrize(
    ("options", "message"),
    [
        ({"maximum_absolute_correlation": -0.1}, "between zero and one"),
        ({"maximum_absolute_correlation": 1.1}, "between zero and one"),
        ({"method": "unsupported"}, "method must be"),
        ({"minimum_periods": 1}, "minimum_periods"),
    ],
)
def test_invalid_correlation_filter_options_are_rejected(
    options: dict[str, float | int | str],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        CorrelationFilterOptions(**options)


def test_get_result_rejects_unfitted_filter() -> None:
    with pytest.raises(RuntimeError, match="not fitted"):
        CorrelationFilter().get_result()
