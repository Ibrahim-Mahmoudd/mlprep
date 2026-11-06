from __future__ import annotations

import pandas as pd
import pytest

from mlprep.phases.phase1.deterministic_normalization.missing_tokens import (
    MissingTokenColumnOptions,
    MissingTokenNormalizer,
    MissingTokenOptions,
)
from mlprep.phases.phase1.deterministic_normalization.text_normalization import (
    TextColumnOptions,
    TextNormalizationOptions,
    TextNormalizer,
)
from mlprep.phases.phase1.output_governance.result import Phase1Result
from mlprep.phases.phase1.output_governance.state import Phase1State


def _result(data: pd.DataFrame) -> Phase1Result:
    return Phase1Result(
        data=data,
        internal_row_id_column="__mlprep_row_id__",
    )


def _state() -> Phase1State:
    return Phase1State(
        internal_row_id_column="__mlprep_row_id__",
    )


def test_missing_token_normalizer_replaces_tokens_and_records_governance() -> None:
    original = pd.DataFrame(
        {
            "status": ["NA", " n/a ", "ready", "", "   ", None],
            "category": ["None", "NULL", "valid", "?", "ok", "valid"],
            "numeric": [1, 2, 0, 3, 4, 5],
        }
    )
    result = _result(original)
    state = _state()

    normalized = MissingTokenNormalizer().normalize(
        data=original,
        result=result,
        state=state,
    )

    assert normalized["status"].isna().tolist() == [
        True,
        True,
        False,
        True,
        True,
        True,
    ]
    assert normalized["category"].isna().tolist() == [
        True,
        True,
        False,
        True,
        False,
        False,
    ]
    assert normalized["numeric"].tolist() == [1, 2, 0, 3, 4, 5]
    assert original["status"].iloc[0] == "NA"
    assert len(result.normalization_events) == 2
    assert {event.column for event in result.normalization_events} == {
        "status",
        "category",
    }
    assert {finding.code for finding in result.findings} == {
        "P1-MISSING-001",
    }
    assert state.missing_token_states["status"].total_normalized_count == 4
    assert state.missing_token_states["category"].total_normalized_count == 3
    assert result.dataset_summary.missing_cell_count == 8


def test_missing_token_options_control_case_whitespace_and_column_rules() -> None:
    data = pd.DataFrame(
        {
            "strict": ["MISSING", "missing", " MISSING ", " "],
            "custom": ["N/A", "absent", "valid", None],
            "excluded": ["NA", "ok", "NA", "ok"],
        }
    )
    options = MissingTokenOptions(
        global_tokens=["NA"],
        include_columns=["strict", "custom", "absent"],
        exclude_columns=["excluded"],
        case_sensitive=True,
        strip_before_comparison=False,
        treat_empty_string_as_missing=False,
        treat_whitespace_only_as_missing=False,
        column_options={
            "strict": MissingTokenColumnOptions(
                tokens=["MISSING"],
                replace_global_tokens=True,
            ),
            "custom": MissingTokenColumnOptions(
                tokens=["absent"],
                replace_global_tokens=True,
            ),
        },
        maximum_examples=1,
    )
    result = _result(data)
    state = _state()

    normalized = MissingTokenNormalizer(options).normalize(
        data=data,
        result=result,
        state=state,
    )

    assert normalized["strict"].isna().tolist() == [
        True,
        False,
        False,
        False,
    ]
    assert normalized["custom"].isna().tolist() == [False, True, False, True]
    assert normalized["excluded"].tolist() == data["excluded"].tolist()
    assert state.missing_token_states["strict"].configured_tokens == ["MISSING"]
    assert result.findings[0].code == "P1-MISSING-002"
    assert result.findings[0].columns == ["absent"]
    assert result.findings[1].affected_row_count == 1
    assert result.findings[1].example_values == ["'MISSING'"]


def test_missing_token_normalizer_supports_numeric_tokens_and_empty_selection() -> None:
    data = pd.DataFrame(
        {
            "number": [0, 1, 2],
            "text": pd.Series(["a", "b", "c"], dtype="string"),
        }
    )
    result = _result(data)
    normalized = MissingTokenNormalizer(
        MissingTokenOptions(
            global_tokens=[0],
            exclude_columns=["text"],
        )
    ).normalize(
        data=data,
        result=result,
        state=_state(),
    )

    assert pd.isna(normalized["number"].iloc[0])
    assert normalized["number"].iloc[1:].tolist() == [1, 2]
    assert normalized["text"].tolist() == ["a", "b", "c"]
    assert MissingTokenNormalizer(
        MissingTokenOptions(enabled=False)
    ).normalize(data, _result(data), _state()).equals(data)
    with pytest.raises(TypeError, match="DataFrame"):
        MissingTokenNormalizer().normalize(
            data=["NA"],
            result=result,
            state=_state(),
        )


def test_missing_token_configuration_validation() -> None:
    with pytest.raises(ValueError, match="maximum_examples"):
        MissingTokenOptions(maximum_examples=-1)
    with pytest.raises(ValueError, match="both included and excluded"):
        MissingTokenOptions(
            include_columns=["target"],
            exclude_columns=["target"],
        )


def test_text_normalizer_applies_unicode_whitespace_case_and_aliases() -> None:
    data = pd.DataFrame(
        {
            "name": [
                "  \uFF21\uFF23\uFF2D\uFF25\t  Corp  ",
                "Straße",
                "line\nbreak",
                None,
                17,
            ],
            "region": ["north", "south", "west", "east", "central"],
            "ignored": ["  Keep  ", "x", "x", "x", "x"],
        }
    )
    options = TextNormalizationOptions(
        include_columns=["name", "region", "absent"],
        exclude_columns=["ignored"],
        case_style="casefold",
        global_aliases={"acme corp": "ACME"},
        column_options={
            "region": TextColumnOptions(
                case_style="upper",
                aliases={"NORTH": "N"},
            )
        },
        maximum_examples=1,
    )
    result = _result(data)

    normalized = TextNormalizer(options).normalize(
        data=data,
        result=result,
    )

    assert normalized["name"].iloc[0] == "ACME"
    assert normalized["name"].iloc[1] == "strasse"
    assert normalized["name"].iloc[2] == "line break"
    assert normalized["name"].iloc[3] is None
    assert normalized["name"].iloc[4] == 17
    assert normalized["region"].iloc[0] == "N"
    assert normalized["region"].iloc[1] == "SOUTH"
    assert normalized["ignored"].iloc[0] == "  Keep  "
    assert len(result.normalization_events) == 2
    assert {event.component for event in result.normalization_events} == {
        "text_normalization",
    }
    assert {finding.code for finding in result.findings} == {
        "P1-TEXT-001",
        "P1-TEXT-002",
    }
    assert result.findings[0].columns == ["absent"]
    assert result.findings[1].affected_row_count == 3
    assert result.findings[1].affected_row_percentage == 60.0
    assert data["name"].iloc[0] == "  \uFF21\uFF23\uFF2D\uFF25\t  Corp  "


def test_text_normalizer_supports_local_overrides_and_disabled_mode() -> None:
    data = pd.DataFrame(
        {
            "first": ["  A\t B  ", "Keep"],
            "second": ["  C  ", "  D  "],
        }
    )
    options = TextNormalizationOptions(
        include_columns=["first", "second", "absent"],
        strip_whitespace=False,
        collapse_whitespace=False,
        normalize_unicode=False,
        remove_control_characters=False,
        case_style="preserve",
        column_options={
            "first": TextColumnOptions(
                strip_whitespace=True,
                collapse_whitespace=True,
                normalize_unicode=True,
                remove_control_characters=True,
                case_style="upper",
                aliases={"A B": "alpha"},
            )
        },
    )
    result = _result(data)
    normalized = TextNormalizer(options).normalize(data, result)

    assert normalized["first"].tolist() == ["alpha", "KEEP"]
    assert normalized["second"].tolist() == ["  C  ", "  D  "]
    assert any(finding.code == "P1-TEXT-002" for finding in result.findings)
    assert TextNormalizer(
        TextNormalizationOptions(enabled=False)
    ).normalize(data, _result(data)).equals(data)
    with pytest.raises(TypeError, match="DataFrame"):
        TextNormalizer().normalize(["value"], result)


def test_text_normalization_options_reject_invalid_limits_and_overlap() -> None:
    with pytest.raises(ValueError, match="maximum_examples"):
        TextNormalizationOptions(maximum_examples=-1)
    with pytest.raises(ValueError, match="both included and excluded"):
        TextNormalizationOptions(
            include_columns=["name"],
            exclude_columns=["name"],
        )
