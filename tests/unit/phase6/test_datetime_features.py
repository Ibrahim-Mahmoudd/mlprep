from __future__ import annotations

import pandas as pd
import pytest

from mlprep.phases.phase6.feature_engineering.datetime_features import (
    DatetimeFeatureExtractor,
    DatetimeFeatureOptions,
)


def test_datetime_features_use_training_reference_and_preserve_index() -> None:
    training = pd.Series(
        pd.to_datetime(["2024-01-01 12:00", "2024-01-02 12:00"]),
        index=[10, 11],
        name="event_time",
    )
    extractor = DatetimeFeatureExtractor(
        column="event_time",
        options=DatetimeFeatureOptions(
            features=[
                "year",
                "quarter",
                "month",
                "week",
                "day",
                "day_of_week",
                "day_of_year",
                "hour",
                "minute",
                "second",
                "is_weekend",
                "is_month_start",
                "is_month_end",
                "elapsed_seconds",
                "elapsed_days",
            ]
        ),
    )
    training_features = extractor.fit_transform(training)

    validation = pd.Series(
        pd.to_datetime(["2024-01-04 12:00"]),
        index=[20],
        name="event_time",
    )
    validation_features = extractor.transform(validation)

    assert training_features.loc[10, "event_time__datetime__elapsed_seconds"] == 0
    assert validation_features.loc[20, "event_time__datetime__elapsed_days"] == 3
    assert validation_features.index.tolist() == [20]
    assert extractor.state.fitted_on_training_only
    assert extractor.state.reference_timestamp == "2024-01-01T12:00:00"


def test_datetime_feature_options_support_explicit_reference_and_dtype() -> None:
    extractor = DatetimeFeatureExtractor(
        column="date",
        output_prefix="calendar",
        options=DatetimeFeatureOptions(
            features=["year", "is_weekend", "elapsed_days"],
            reference_method="explicit",
            explicit_reference="2024-01-01",
            output_dtype="int64",
        ),
    )
    features = extractor.fit_transform(
        pd.Series(pd.to_datetime(["2024-01-06", "2024-01-08"]), name="date")
    )

    assert list(features.columns) == [
        "calendar__datetime__year",
        "calendar__datetime__is_weekend",
        "calendar__datetime__elapsed_days",
    ]
    assert features.iloc[0, 1] == 1
    assert features.iloc[1, 2] == 7
    assert str(features["calendar__datetime__year"].dtype) == "int64"


def test_datetime_extractor_validates_data_and_fitted_state() -> None:
    extractor = DatetimeFeatureExtractor(
        column="date",
        options=DatetimeFeatureOptions(features=["year"]),
    )
    with pytest.raises(RuntimeError, match="not fitted"):
        extractor.transform(pd.Series(["2024-01-01"], name="date"))
    with pytest.raises(ValueError, match="missing values"):
        extractor.fit(pd.Series([None], name="date"))
    with pytest.raises(ValueError, match="invalid values"):
        extractor.fit(pd.Series(["not-a-date"], name="date"))

    permissive = DatetimeFeatureExtractor(
        column="date",
        options=DatetimeFeatureOptions(
            features=["year", "is_month_start"],
            allow_missing_values=True,
            allow_invalid_values=True,
        ),
    ).fit(pd.Series(["2024-01-01", None, "invalid"], name="date"))
    result = permissive.transform(
        pd.Series(["2024-01-01", None, "invalid"], name="date")
    )
    assert result["date__datetime__year"].isna().sum() == 2
    assert permissive.state.training_missing_count == 1
    assert permissive.state.training_invalid_count == 1


def test_datetime_extractor_rejects_invalid_options_and_input_schema() -> None:
    with pytest.raises(ValueError, match="Unsupported datetime features"):
        DatetimeFeatureOptions(features=["unsupported"])
    with pytest.raises(ValueError, match="duplicates"):
        DatetimeFeatureOptions(features=["year", "year"])
    with pytest.raises(ValueError, match="explicit_reference is required"):
        DatetimeFeatureOptions(reference_method="explicit")
    with pytest.raises(ValueError, match="output dtype"):
        DatetimeFeatureOptions(output_dtype="object")

    with pytest.raises(ValueError, match="column cannot be empty"):
        DatetimeFeatureExtractor(column=" ")
    with pytest.raises(TypeError, match="pandas Series"):
        DatetimeFeatureExtractor(column="date").fit(pd.DataFrame({"date": []}))

    extractor = DatetimeFeatureExtractor(
        column="date",
        options=DatetimeFeatureOptions(features=["year"]),
    )
    with pytest.raises(ValueError, match="received"):
        extractor.fit(pd.Series(["2024-01-01"], name="other"))
