from __future__ import annotations

import pandas as pd
import pytest

from mlprep.phases.phase1.output_governance.result import Phase1Result
from mlprep.phases.phase1.output_governance.state import Phase1State
from mlprep.phases.phase1.validation_and_risk.duplicate_rows import (
    DuplicateRowDetector,
    DuplicateRowOptions,
)


def _inspect(
    frame: pd.DataFrame,
    options: DuplicateRowOptions | None = None,
) -> tuple[pd.DataFrame, Phase1Result]:
    detector_options = options or DuplicateRowOptions(
        internal_row_id_column="rid",
    )
    result = Phase1Result(
        data=frame,
        internal_row_id_column="rid",
    )
    transformed = DuplicateRowDetector(detector_options).inspect(
        frame,
        result,
        Phase1State(),
    )
    return transformed, result


def test_exact_duplicate_report_ignores_internal_ids_and_preserves_input() -> None:
    frame = pd.DataFrame(
        {
            "rid": ["row-a", "row-b", "row-c"],
            "value": [1, 1, 2],
        },
        index=[10, 11, 12],
    )
    before = frame.copy(deep=True)

    transformed, result = _inspect(frame)

    assert transformed.equals(frame)
    assert transformed is not frame
    assert result.dataset_summary.duplicate_row_count == 2
    assert len(result.findings) == 1
    assert result.findings[0].columns == ["value"]
    assert result.findings[0].example_row_ids == [10, 11]
    assert result.findings[0].affected_row_percentage == pytest.approx(66.666667)
    assert not result.findings[0].blocks_readiness
    pd.testing.assert_frame_equal(frame, before)


@pytest.mark.parametrize(
    ("policy", "expected_ids", "rejected_ids"),
    [
        ("keep_first", ["row-a", "row-c"], ["row-b"]),
        ("keep_last", ["row-b", "row-c"], ["row-a"]),
        ("remove_exact", ["row-a", "row-c"], ["row-b"]),
        ("quarantine", ["row-a", "row-c"], ["row-b"]),
    ],
)
def test_exact_duplicate_resolution_policies(
    policy: str,
    expected_ids: list[str],
    rejected_ids: list[str],
) -> None:
    frame = pd.DataFrame(
        {
            "rid": ["row-a", "row-b", "row-c"],
            "value": ["same", "same", "unique"],
        },
        index=[20, 21, 22],
    )

    transformed, result = _inspect(
        frame,
        DuplicateRowOptions(
            policy=policy,  # type: ignore[arg-type]
            internal_row_id_column="rid",
        ),
    )

    assert transformed["rid"].tolist() == expected_ids
    assert [record.row_id for record in result.rejected_records] == rejected_ids
    assert result.dataset_summary.duplicate_row_count == 2


def test_observation_key_duplicates_report_conflicting_targets() -> None:
    frame = pd.DataFrame(
        {
            "entity": ["a", "a", "b", "b"],
            "target": [0, 1, 0, 0],
        },
        index=[30, 31, 32, 33],
    )

    _, result = _inspect(
        frame,
        DuplicateRowOptions(
            detect_exact_duplicates=False,
            observation_key=["entity"],
            target_column="target",
        ),
    )

    findings = {finding.code: finding for finding in result.findings}
    assert set(findings) == {"P1-DUPROW-003", "P1-DUPROW-004"}
    assert findings["P1-DUPROW-003"].affected_row_count == 4
    assert findings["P1-DUPROW-003"].example_row_ids == [
        30,
        31,
        32,
        33,
    ]
    assert findings["P1-DUPROW-004"].affected_row_count == 1
    assert findings["P1-DUPROW-004"].columns == ["entity", "target"]
    assert findings["P1-DUPROW-004"].blocks_readiness


def test_exact_duplicate_comparison_can_include_only_selected_columns() -> None:
    frame = pd.DataFrame(
        {
            "left": [1, 1, 2],
            "right": ["first", "second", "third"],
        }
    )

    transformed, result = _inspect(
        frame,
        DuplicateRowOptions(
            include_columns=["left"],
            detect_observation_key_duplicates=False,
        ),
    )

    assert transformed.equals(frame)
    assert result.dataset_summary.duplicate_row_count == 2
    assert result.findings[0].columns == ["left"]


def test_missing_observation_key_columns_are_blocking_findings() -> None:
    frame = pd.DataFrame({"value": [1, 2]})

    _, result = _inspect(
        frame,
        DuplicateRowOptions(
            detect_exact_duplicates=False,
            observation_key=["missing"],
        ),
    )

    assert len(result.findings) == 1
    assert result.findings[0].code == "P1-DUPROW-002"
    assert result.findings[0].columns == ["missing"]
    assert result.findings[0].blocks_readiness


def test_row_id_examples_are_bounded_and_normalized() -> None:
    index = pd.Index(
        [("group", 1), ("group", 1), ("other", 2)],
        tupleize_cols=False,
    )
    frame = pd.DataFrame(
        {"value": ["same", "same", "unique"]},
        index=index,
    )

    _, result = _inspect(
        frame,
        DuplicateRowOptions(maximum_examples=1),
    )

    assert result.findings[0].example_row_ids == ["('group', 1)"]


def test_disabled_detector_returns_copy_without_findings() -> None:
    frame = pd.DataFrame({"value": [1, 1]})
    transformed, result = _inspect(
        frame,
        DuplicateRowOptions(enabled=False),
    )

    assert transformed.equals(frame)
    assert transformed is not frame
    assert result.findings == []
    assert result.dataset_summary.duplicate_row_count == 0


def test_duplicate_row_options_and_input_type_are_validated() -> None:
    with pytest.raises(ValueError, match="maximum_examples"):
        DuplicateRowOptions(maximum_examples=-1)

    with pytest.raises(TypeError, match="pandas DataFrame"):
        DuplicateRowDetector().inspect(
            pd.Series([1, 1]),  # type: ignore[arg-type]
            Phase1Result(
                data=pd.DataFrame(),
                internal_row_id_column="rid",
            ),
            Phase1State(),
        )
