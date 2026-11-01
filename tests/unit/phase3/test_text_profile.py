from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest

from mlprep.phases.phase3.contract.column_selection import (
    AnalysisColumnGroups,
)
from mlprep.phases.phase3.profiling.text_profile import (
    TextProfiler,
    TextProfilerOptions,
)


def _contract(features: pd.DataFrame) -> SimpleNamespace:
    return SimpleNamespace(
        training_features=lambda: features.copy(deep=True),
    )


def test_profile_counts_text_properties_without_mutating_input() -> None:
    features = pd.DataFrame(
        {
            "text": [
                "  Hello  ",
                "",
                "   ",
                "a1!",
                "line\nbreak",
                None,
                "  Hello  ",
            ],
            "ignored": list(range(7)),
        }
    )
    before = features.copy(deep=True)

    profile = TextProfiler(
        TextProfilerOptions(maximum_frequency_values=2)
    ).profile(
        contract=_contract(features),
        columns=AnalysisColumnGroups(
            text=["text", "missing_column"]
        ),
    )["text"]

    pd.testing.assert_frame_equal(features, before)
    assert profile.row_count == 7
    assert profile.nonmissing_count == 6
    assert profile.missing_count == 1
    assert profile.missing_percentage == pytest.approx(100 / 7)
    assert profile.empty_string_count == 1
    assert profile.whitespace_only_count == 1
    assert profile.unique_count == 5
    assert profile.uniqueness_ratio == pytest.approx(5 / 6)
    assert profile.minimum_character_count == 0
    assert profile.maximum_character_count == 10
    assert profile.average_character_count == pytest.approx(34 / 6)
    assert profile.median_character_count == 6.0
    assert profile.minimum_word_count == 0
    assert profile.maximum_word_count == 2
    assert profile.average_word_count == pytest.approx(5 / 6)
    assert profile.median_word_count == 1.0
    assert profile.values_with_digits_count == 1
    assert profile.values_with_special_characters_count == 1
    assert profile.values_with_line_breaks_count == 1
    assert profile.duplicate_text_count == 2
    assert len(profile.most_frequent_text_hashes) == 2
    assert profile.most_frequent_text_hashes[0]["count"] == 2
    assert "value_hash" in profile.most_frequent_text_hashes[0]
    assert "value" not in profile.most_frequent_text_hashes[0]


def test_frequency_records_support_opt_in_raw_values_and_hash_disabling() -> None:
    features = pd.DataFrame(
        {"text": ["secret text", "secret text", "other"]}
    )

    raw_profile = TextProfiler(
        TextProfilerOptions(
            include_raw_values=True,
            maximum_raw_value_length=6,
        )
    ).profile(
        contract=_contract(features),
        columns=AnalysisColumnGroups(text=["text"]),
    )["text"]
    unencoded_profile = TextProfiler(
        TextProfilerOptions(hash_frequent_values=False)
    ).profile(
        contract=_contract(features),
        columns=AnalysisColumnGroups(text=["text"]),
    )["text"]

    assert raw_profile.most_frequent_text_hashes[0] == {
        "count": 2,
        "length": 11,
        "value": "secret",
    }
    assert "value_hash" not in raw_profile.most_frequent_text_hashes[0]
    assert unencoded_profile.most_frequent_text_hashes[0] == {
        "count": 2,
        "length": 11,
    }


def test_empty_and_all_missing_text_have_no_derived_statistics() -> None:
    empty = TextProfiler()._profile_column(
        column="empty",
        series=pd.Series([], dtype=object),
    )
    missing = TextProfiler()._profile_column(
        column="missing",
        series=pd.Series([None, None], dtype=object),
    )

    assert empty.row_count == 0
    assert empty.missing_percentage == 0.0
    assert empty.minimum_character_count is None
    assert empty.average_word_count is None
    assert empty.unique_count == 0
    assert empty.duplicate_text_count == 0
    assert empty.most_frequent_text_hashes == []

    assert missing.row_count == 2
    assert missing.missing_count == 2
    assert missing.nonmissing_count == 0
    assert missing.missing_percentage == 100.0
    assert missing.minimum_character_count is None
    assert missing.minimum_word_count is None
    assert missing.most_frequent_text_hashes == []


def test_disabled_profiler_and_option_validation() -> None:
    disabled = TextProfiler(
        TextProfilerOptions(enabled=False)
    ).profile(
        contract=_contract(pd.DataFrame({"text": ["value"]})),
        columns=AnalysisColumnGroups(text=["text"]),
    )

    assert disabled == {}

    with pytest.raises(ValueError, match="maximum_frequency_values"):
        TextProfilerOptions(maximum_frequency_values=-1)
    with pytest.raises(ValueError, match="maximum_raw_value_length"):
        TextProfilerOptions(maximum_raw_value_length=0)
