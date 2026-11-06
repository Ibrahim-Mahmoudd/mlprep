from __future__ import annotations

import pandas as pd
import pytest

from mlprep.phases.phase1.inspection.mixed_types import (
    MixedTypeDetectionOptions,
    MixedTypeDetector,
)
from mlprep.phases.phase1.inspection.physical_types import (
    PhysicalTypeInspectionOptions,
    PhysicalTypeInspector,
)
from mlprep.phases.phase1.inspection.temporal_detection import (
    TemporalDetectionOptions,
    TemporalDetector,
)
from mlprep.phases.phase1.output_governance.result import Phase1Result
from mlprep.phases.phase1.output_governance.state import (
    ColumnState,
    Phase1State,
    TypeResolutionState,
)
from mlprep.types import PhysicalType, SemanticType


def _result(data: pd.DataFrame) -> Phase1Result:
    return Phase1Result(
        data=data,
        internal_row_id_column="__mlprep_row_id__",
    )


def _state() -> Phase1State:
    return Phase1State(
        internal_row_id_column="__mlprep_row_id__",
    )


def test_mixed_type_detector_groups_compatible_types_and_registers_evidence() -> None:
    data = pd.DataFrame(
        {
            "mixed": pd.Series([1, 2.5, "three", True], dtype=object),
            "compatible": pd.Series([1, 2.5, 3], dtype=object),
            "excluded": pd.Series(["x", 1, "y", 2], dtype=object),
            "numeric": [1, 2, 3, 4],
        }
    )
    state = _state()
    result = _result(data)

    findings = MixedTypeDetector(
        MixedTypeDetectionOptions(
            exclude_columns=["excluded"],
            compatible_numeric_types=True,
            compatible_string_types=False,
            minimum_secondary_type_ratio=0.25,
        )
    ).inspect(data, result, state)

    assert findings == {
        "mixed": {
            "numeric": 2,
            "str": 1,
            "bool": 1,
        }
    }
    assert [finding.code for finding in result.findings] == [
        "P1-MIXED-001",
    ]
    assert result.findings[0].affected_row_percentage == 100.0
    assert result.findings[0].example_values == [1, "three", True]
    assert "compatible" not in findings
    assert "excluded" not in findings


def test_mixed_type_detector_threshold_sampling_and_options() -> None:
    data = pd.DataFrame(
        {
            "mixed": pd.Series(["a", "b", 1, 2], dtype=object),
            "empty": pd.Series([None, None, None, None], dtype=object),
        }
    )
    result = _result(data)
    findings = MixedTypeDetector(
        MixedTypeDetectionOptions(
            include_columns=["mixed", "empty", "missing"],
            sample_size=2,
            minimum_secondary_type_count=2,
            minimum_secondary_type_ratio=0.5,
            maximum_examples=0,
        )
    ).inspect(data, result, _state())

    assert findings == {}
    assert result.findings == []
    assert MixedTypeDetector(
        MixedTypeDetectionOptions(enabled=False)
    ).inspect(data, result, _state()) == {}
    with pytest.raises(TypeError, match="DataFrame"):
        MixedTypeDetector().inspect(["value"], result, _state())
    with pytest.raises(ValueError, match="sample_size"):
        MixedTypeDetectionOptions(sample_size=0)
    with pytest.raises(ValueError, match="minimum_secondary_type_count"):
        MixedTypeDetectionOptions(minimum_secondary_type_count=0)
    with pytest.raises(ValueError, match="ratio"):
        MixedTypeDetectionOptions(minimum_secondary_type_ratio=2.0)
    with pytest.raises(ValueError, match="maximum_examples"):
        MixedTypeDetectionOptions(maximum_examples=-1)


def test_physical_type_inspector_classifies_dtypes_and_registers_column_state() -> None:
    data = pd.DataFrame(
        {
            "boolean": pd.Series([True, False], dtype="bool"),
            "integer": pd.Series([1, 2], dtype="int64"),
            "float": pd.Series([1.0, 2.0], dtype="float64"),
            "complex": pd.Series([1j, 2j], dtype="complex128"),
            "datetime": pd.to_datetime(["2024-01-01", "2024-01-02"]),
            "timedelta": pd.to_timedelta([1, 2], unit="D"),
            "category": pd.Series(["a", "b"], dtype="category"),
            "string": pd.Series(["a", "b"], dtype="string"),
            "object": pd.Series([object(), object()], dtype=object),
        }
    )
    result = _result(data)
    state = _state()
    state.register_column_state(
        ColumnState(
            original_name="integer",
            normalized_name="integer",
        )
    )

    detected = PhysicalTypeInspector().inspect(data, result, state)

    assert detected == {
        "boolean": PhysicalType.BOOLEAN,
        "integer": PhysicalType.INTEGER,
        "float": PhysicalType.FLOAT,
        "complex": PhysicalType.COMPLEX,
        "datetime": PhysicalType.DATETIME,
        "timedelta": PhysicalType.TIMEDELTA,
        "category": PhysicalType.CATEGORY,
        "string": PhysicalType.STRING,
        "object": PhysicalType.STRING,
    }
    assert state.type_resolution_states["integer"].resolved_physical_type is (
        PhysicalType.INTEGER
    )
    assert state.column_states["integer"].dtype_before == "int64"
    assert state.column_states["integer"].dtype_after == "int64"
    assert result.findings == []


def test_physical_type_inspector_respects_selection_and_disabled_options() -> None:
    data = pd.DataFrame({"first": ["a"], "second": [1]})
    result = _result(data)

    inspected = PhysicalTypeInspector(
        PhysicalTypeInspectionOptions(
            include_columns=["first", "missing"],
            exclude_columns=["second"],
            report_object_columns=False,
        )
    ).inspect(data, result, _state())

    assert inspected == {"first": PhysicalType.STRING}
    assert [finding.code for finding in result.findings] == ["P1-PHYSICAL-003"]
    assert PhysicalTypeInspector(
        PhysicalTypeInspectionOptions(enabled=False)
    ).inspect(data, result, _state()) == {}
    with pytest.raises(ValueError, match="maximum_examples"):
        PhysicalTypeInspectionOptions(maximum_examples=-1)
    with pytest.raises(ValueError, match="both included and excluded"):
        PhysicalTypeInspectionOptions(
            include_columns=["first"],
            exclude_columns=["first"],
        )
    with pytest.raises(TypeError, match="DataFrame"):
        PhysicalTypeInspector().inspect(["x"], result, _state())


def test_temporal_detector_reports_ambiguity_parse_failures_and_year_limits() -> None:
    data = pd.DataFrame(
        {
            "when": ["2019-01-02", "2024-05-06", "invalid"],
            "ambiguous": ["01/02/2020", "03/04/2021", "05/06/2022"],
            "not_date": ["alpha", "beta", "gamma"],
        }
    )
    result = _result(data)
    state = _state()
    state.register_type_resolution(
        TypeResolutionState(
            column="when",
            original_physical_type=PhysicalType.STRING,
            resolved_physical_type=PhysicalType.STRING,
            semantic_type=SemanticType.UNKNOWN,
        )
    )
    normalized = TemporalDetector(
        TemporalDetectionOptions(
            include_columns=["when", "ambiguous", "not_date"],
            parse_threshold=0.6,
            convert_detected_columns=True,
            minimum_year=2020,
            maximum_year=2023,
        )
    ).inspect(data, result, state)

    assert pd.api.types.is_datetime64_any_dtype(normalized["when"])
    assert normalized["when"].isna().tolist() == [False, False, True]
    assert data["when"].iloc[0] == "2019-01-02"
    assert state.type_resolution_states["when"].semantic_type is SemanticType.DATETIME
    assert state.type_resolution_states["when"].parse_success_ratio == 2 / 3
    assert state.type_resolution_states["when"].failed_parse_count == 1
    assert state.type_resolution_states["when"].coercion_was_applied is True
    finding_codes = [finding.code for finding in result.findings]
    assert finding_codes.count("P1-TEMPORAL-001") == 1
    assert finding_codes.count("P1-TEMPORAL-002") == 2
    assert "P1-TEMPORAL-003" in finding_codes
    assert "P1-TEMPORAL-004" in finding_codes


def test_temporal_detector_handles_datetime_skip_disabled_and_invalid_options() -> None:
    data = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2020-01-01", "2021-01-01"]),
            "text": ["not a date", "still not"],
        }
    )
    state = _state()
    result = _result(data)
    normalized = TemporalDetector(
        TemporalDetectionOptions(
            include_columns=["timestamp", "text", "missing"],
            parse_threshold=1.0,
            minimum_year=2021,
        )
    ).inspect(data, result, state)

    assert normalized.equals(data)
    assert [finding.code for finding in result.findings] == ["P1-TEMPORAL-004"]
    assert TemporalDetector(
        TemporalDetectionOptions(enabled=False)
    ).inspect(data, result, state).equals(data)
    with pytest.raises(TypeError, match="DataFrame"):
        TemporalDetector().inspect(["x"], result, state)
    with pytest.raises(ValueError, match="parse_threshold"):
        TemporalDetectionOptions(parse_threshold=1.1)
    with pytest.raises(ValueError, match="sample_size"):
        TemporalDetectionOptions(sample_size=0)
    with pytest.raises(ValueError, match="minimum_year"):
        TemporalDetectionOptions(minimum_year=2025, maximum_year=2020)
    with pytest.raises(ValueError, match="maximum_examples"):
        TemporalDetectionOptions(maximum_examples=-1)
