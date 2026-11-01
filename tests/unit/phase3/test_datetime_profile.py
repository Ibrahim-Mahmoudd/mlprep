from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest

from mlprep.phases.phase3.contract.column_selection import (
    AnalysisColumnGroups,
)
from mlprep.phases.phase3.profiling.datetime_profile import (
    DatetimeProfiler,
    DatetimeProfilerOptions,
)


def _contract(features: pd.DataFrame) -> SimpleNamespace:
    return SimpleNamespace(
        training_features=lambda: features.copy(deep=True),
    )


def test_profile_counts_missing_invalid_dates_and_time_frequencies() -> None:
    features = pd.DataFrame(
        {
            "date_text": [
                "2024-01-03",
                "invalid",
                None,
                "2024-01-01",
                "2024-01-02",
            ],
            "unselected": ["x", "y", "x", "y", "x"],
        }
    )
    before = features.copy(deep=True)

    profiles = DatetimeProfiler(
        DatetimeProfilerOptions(maximum_frequency_values=1)
    ).profile(
        contract=_contract(features),
        columns=AnalysisColumnGroups(
            datetime=["date_text", "missing_column"]
        ),
    )

    pd.testing.assert_frame_equal(features, before)
    assert list(profiles) == ["date_text"]
    profile = profiles["date_text"]
    assert profile.dtype == str(features["date_text"].dtype)
    assert profile.row_count == 5
    assert profile.nonmissing_count == 3
    assert profile.missing_count == 1
    assert profile.invalid_count == 1
    assert profile.missing_percentage == 20.0
    assert profile.unique_count == 3
    assert profile.minimum_timestamp == "2024-01-01T00:00:00"
    assert profile.maximum_timestamp == "2024-01-03T00:00:00"
    assert profile.duration_seconds == 172800.0
    assert profile.median_interval_seconds == 86400.0
    assert profile.most_frequent_years == [
        {"value": "2024", "count": 3}
    ]
    assert profile.most_frequent_months == [
        {"value": "1", "count": 3}
    ]
    assert len(profile.most_frequent_weekdays) == 1
    assert profile.most_frequent_weekdays[0]["count"] == 1
    assert profile.most_frequent_hours == [
        {"value": "0", "count": 3}
    ]
    assert profile.timezone is None


def test_profile_supports_day_first_and_timezone_aware_values() -> None:
    features = pd.DataFrame(
        {
            "day_first": ["02/01/2024", "03/01/2024"],
            "utc": pd.to_datetime(
                ["2024-02-01T00:00:00Z", "2024-02-02T00:00:00Z"],
                utc=True,
            ),
        }
    )

    profiles = DatetimeProfiler(
        DatetimeProfilerOptions(
            day_first=True,
            maximum_frequency_values=2,
        )
    ).profile(
        contract=_contract(features),
        columns=AnalysisColumnGroups(
            datetime=["day_first", "utc"]
        ),
    )

    assert profiles["day_first"].minimum_timestamp.startswith(
        "2024-01-02"
    )
    assert profiles["day_first"].maximum_timestamp.startswith(
        "2024-01-03"
    )
    assert profiles["utc"].timezone == "UTC"
    assert profiles["utc"].duration_seconds == 86400.0
    assert profiles["utc"].median_interval_seconds == 86400.0


def test_empty_and_all_missing_profiles_have_no_timestamp_statistics() -> None:
    empty = DatetimeProfiler()._profile_column(
        column="empty",
        series=pd.Series([], dtype="datetime64[ns]"),
    )
    all_missing = DatetimeProfiler()._profile_column(
        column="missing",
        series=pd.Series([None, None], dtype=object),
    )

    assert empty.row_count == 0
    assert empty.missing_percentage == 0.0
    assert empty.minimum_timestamp is None
    assert empty.maximum_timestamp is None
    assert empty.duration_seconds is None
    assert empty.median_interval_seconds is None
    assert empty.most_frequent_years == []

    assert all_missing.row_count == 2
    assert all_missing.missing_count == 2
    assert all_missing.nonmissing_count == 0
    assert all_missing.invalid_count == 0
    assert all_missing.missing_percentage == 100.0
    assert all_missing.minimum_timestamp is None
    assert all_missing.most_frequent_months == []


def test_interval_calculation_can_be_disabled_and_options_are_validated() -> None:
    profile = DatetimeProfiler(
        DatetimeProfilerOptions(calculate_intervals=False)
    )._profile_column(
        column="date",
        series=pd.Series(
            pd.date_range("2024-01-01", periods=3, freq="D")
        ),
    )
    disabled = DatetimeProfiler(
        DatetimeProfilerOptions(enabled=False)
    ).profile(
        contract=_contract(
            pd.DataFrame({"date": pd.date_range("2024-01-01", periods=2)})
        ),
        columns=AnalysisColumnGroups(datetime=["date"]),
    )

    assert profile.duration_seconds == 172800.0
    assert profile.median_interval_seconds is None
    assert disabled == {}

    with pytest.raises(ValueError, match="maximum_frequency_values"):
        DatetimeProfilerOptions(maximum_frequency_values=0)
