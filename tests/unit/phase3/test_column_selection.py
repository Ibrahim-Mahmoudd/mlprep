from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from mlprep.phases.phase1.output_governance.state import (
    ColumnState,
    Phase1State,
)
from mlprep.phases.phase3.contract.column_selection import (
    AnalysisColumnGroups,
    ColumnSelectionOptions,
    ColumnSelector,
)
from mlprep.types import ColumnRole, SemanticType


def _contract(
    frame: pd.DataFrame,
    *,
    column_states: dict[str, ColumnState] | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        train=SimpleNamespace(X=frame),
        phase1_state=Phase1State(
            column_states=column_states or {}
        ),
        target_column="target",
    )


def test_column_selector_resolves_pandas_dtypes_and_preserves_input() -> None:
    frame = pd.DataFrame(
        {
            "integer": [1, 2, 3],
            "number": [1.0, 2.0, 3.0],
            "flag": [True, False, True],
            "date": pd.date_range("2024-01-01", periods=3),
            "duration": pd.to_timedelta([1, 2, 3], unit="D"),
            "labels": pd.Series(["a", "b", "a"], dtype="string"),
            "category": pd.Series(
                pd.Categorical(["low", "high", "low"])
            ),
        }
    )
    original = frame.copy(deep=True)

    groups = ColumnSelector().resolve(
        contract=_contract(frame)
    )

    assert groups.all_features == list(frame.columns)
    assert groups.numeric == ["integer", "number"]
    assert groups.boolean == ["flag"]
    assert groups.datetime == ["date"]
    assert groups.timedelta == ["duration"]
    assert groups.text == []
    assert groups.categorical == ["labels", "category"]
    assert groups.analyzable_columns() == [
        "integer",
        "number",
        "labels",
        "category",
        "date",
        "flag",
        "duration",
    ]
    assert groups.target_column == "target"
    pd.testing.assert_frame_equal(frame, original)


def test_column_selector_uses_phase1_semantics_and_role_exclusions() -> None:
    frame = pd.DataFrame(
        {
            "semantic_numeric": ["1", "2"],
            "sensitive": ["a@b.test", "c@d.test"],
            "identifier": [1, 2],
            "internal": [3, 4],
            "ignored": ["x", "y"],
        }
    )
    states = {
        "semantic_numeric": ColumnState(
            original_name="semantic_numeric",
            normalized_name="semantic_numeric",
            semantic_type=SemanticType.CONTINUOUS,
        ),
        "sensitive": ColumnState(
            original_name="sensitive",
            normalized_name="sensitive",
            is_sensitive=True,
        ),
        "identifier": ColumnState(
            original_name="identifier",
            normalized_name="identifier",
            roles={ColumnRole.IDENTIFIER},
        ),
        "internal": ColumnState(
            original_name="internal",
            normalized_name="internal",
            roles={ColumnRole.INTERNAL},
        ),
        "ignored": ColumnState(
            original_name="ignored",
            normalized_name="ignored",
            roles={ColumnRole.IGNORED},
        ),
    }

    default = ColumnSelector().resolve(
        contract=_contract(frame, column_states=states)
    )
    included = ColumnSelector(
        ColumnSelectionOptions(
            include_sensitive_columns=True,
            include_identifier_columns=True,
            include_internal_columns=True,
        )
    ).resolve(contract=_contract(frame, column_states=states))

    assert default.numeric == ["semantic_numeric"]
    assert default.sensitive == ["sensitive"]
    assert default.identifiers == ["identifier"]
    assert default.ignored == [
        "sensitive",
        "identifier",
        "internal",
        "ignored",
    ]
    assert included.sensitive == ["sensitive"]
    assert included.identifiers == ["identifier"]
    assert included.ignored == ["ignored"]
    assert included.categorical == ["sensitive"]
    assert included.numeric == [
        "semantic_numeric",
        "identifier",
        "internal",
    ]


def test_column_selector_respects_include_exclude_and_explicit_family() -> None:
    frame = pd.DataFrame(
        {
            "number": [1, 2],
            "word": ["one", "two"],
            "also_number": [3, 4],
        }
    )
    options = ColumnSelectionOptions(
        include_columns=["word", "number", "missing", "also_number"],
        exclude_columns=["also_number"],
        explicit_categorical_columns=["number"],
        explicit_text_columns=["word"],
    )

    groups = ColumnSelector(options).resolve(
        contract=_contract(frame)
    )

    assert groups.all_features == ["word", "number"]
    assert groups.categorical == ["number"]
    assert groups.text == ["word"]
    assert groups.numeric == []


def test_column_selector_infers_object_boolean_numeric_categorical_and_text() -> None:
    long_values = [
        f"record-{index:03d}-" + ("x" * 50)
        for index in range(10)
    ]
    frame = pd.DataFrame(
        {
            "bool_objects": pd.Series(
                [True, False] * 5,
                dtype=object,
            ),
            "numeric_objects": pd.Series(
                [1, 2.5, 3, 4, 5] * 2,
                dtype=object,
            ),
            "small_categories": ["a", "b", "a", "c", "a"] * 2,
            "long_text": long_values,
            "many_short_values": [
                f"v{index}"
                for index in range(10)
            ],
            "all_missing": pd.Series([None] * 10, dtype=object),
        }
    )
    selector = ColumnSelector(
        ColumnSelectionOptions(
            categorical_unique_limit=3,
            free_text_unique_ratio_threshold=0.8,
            free_text_average_length_threshold=30,
        )
    )

    groups = selector.resolve(contract=_contract(frame))

    assert groups.boolean == ["bool_objects"]
    assert groups.numeric == ["numeric_objects"]
    assert groups.categorical == ["small_categories"]
    assert groups.text == ["long_text", "many_short_values"]
    assert groups.unknown == ["all_missing"]
    assert groups.family_for("long_text") == "text"
    with pytest.raises(KeyError, match="not registered"):
        groups.family_for("absent")


@pytest.mark.parametrize(
    ("options", "message"),
    [
        (
            {"free_text_unique_ratio_threshold": -0.1},
            "between zero and one",
        ),
        (
            {"free_text_unique_ratio_threshold": 1.1},
            "between zero and one",
        ),
        (
            {"free_text_average_length_threshold": -1.0},
            "cannot be negative",
        ),
        (
            {"categorical_unique_limit": 1},
            "at least two",
        ),
        (
            {
                "explicit_numeric_columns": ["same"],
                "explicit_text_columns": ["same"],
            },
            "multiple analysis families",
        ),
    ],
)
def test_column_selection_options_reject_invalid_configuration(
    options: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        ColumnSelectionOptions(**options)


def test_column_selector_can_skip_semantic_types_and_reject_unknown_families() -> None:
    frame = pd.DataFrame({"value": ["1", "2"]})
    state = {
        "value": ColumnState(
            original_name="value",
            normalized_name="value",
            semantic_type=SemanticType.CONTINUOUS,
        )
    }
    without_semantics = ColumnSelector(
        ColumnSelectionOptions(use_phase1_semantic_types=False)
    ).resolve(contract=_contract(frame, column_states=state))

    assert without_semantics.categorical == ["value"]
    with pytest.raises(ValueError, match="Unsupported analysis family"):
        ColumnSelector._assign_family(
            groups=AnalysisColumnGroups(),
            column="value",
            family="unsupported",
        )


def test_column_selector_handles_numpy_scalars_and_deduplicates_family_memberships(
) -> None:
    frame = pd.DataFrame(
        {
            "object_numeric": pd.Series(
                [np.int64(1), np.float64(2.5)],
                dtype=object,
            )
        }
    )
    groups = ColumnSelector().resolve(contract=_contract(frame))

    assert groups.numeric == ["object_numeric"]
    assert groups.analyzable_columns() == ["object_numeric"]
    duplicate_groups = AnalysisColumnGroups(
        numeric=["x", "x"],
        categorical=["x"],
        unknown=["y", "y"],
    )
    ColumnSelector._remove_duplicate_memberships(duplicate_groups)

    assert duplicate_groups.numeric == ["x"]
    assert duplicate_groups.categorical == []
    assert duplicate_groups.unknown == ["y"]
