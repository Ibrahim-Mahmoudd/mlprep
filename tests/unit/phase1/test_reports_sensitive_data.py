from __future__ import annotations

import json

import pandas as pd
import pytest

from mlprep.phases.phase1.output_governance.report import (
    Phase1ReportBuilder,
    ReportOptions,
)
from mlprep.phases.phase1.output_governance.result import (
    ColumnInventoryItem,
    Finding,
    NormalizationEvent,
    Phase1Result,
)
from mlprep.phases.phase1.output_governance.state import (
    ColumnState,
    Phase1State,
)
from mlprep.phases.phase1.validation_and_risk.sensitive_data import (
    SensitiveDataDetector,
    SensitiveDataOptions,
)
from mlprep.types import (
    FindingAction,
    FindingCategory,
    Severity,
)


def test_sensitive_detector_marks_pattern_columns_without_exposing_values() -> None:
    data = pd.DataFrame(
        {
            "email": ["alice@example.test", "bob@example.test"],
            "phone": ["+1-202-555-0100", "+1-202-555-0101"],
            "ip": ["192.0.2.1", "2001:db8::1"],
            "url": ["https://example.test/a", "www.example.test/b"],
            "card": ["4111111111111111", "4012888888881881"],
            "ordinary": ["north", "south"],
            "declared": ["internal-a", "internal-b"],
        }
    )
    result = Phase1Result(
        data=data,
        internal_row_id_column="__mlprep_row_id__",
    )
    state = Phase1State(
        column_states={
            name: ColumnState(
                original_name=name,
                normalized_name=name,
            )
            for name in [
                "email",
                "phone",
                "ip",
                "url",
                "card",
                "declared",
            ]
        }
    )
    detector = SensitiveDataDetector(
        SensitiveDataOptions(
            declared_sensitive_columns=["declared"],
            detection_threshold=1.0,
            maximum_examples=0,
        )
    )

    detected = detector.inspect(
        data=data,
        result=result,
        state=state,
    )
    report_json = Phase1ReportBuilder().build_json(
        result=result,
        state=state,
    )

    assert detected == {
        "declared": "declared_sensitive",
        "email": "email",
        "phone": "phone_number",
        "ip": "ip_address",
        "url": "url",
        "card": "payment_card",
    }
    assert state.column_states["email"].is_sensitive
    assert state.column_states["email"].sensitive_type == "email"
    assert len(result.findings) == len(detected)
    assert all(
        finding.category == FindingCategory.SENSITIVE_DATA
        and not finding.blocks_readiness
        for finding in result.findings
    )
    assert "alice@example.test" not in report_json
    assert "4111111111111111" not in report_json


def test_sensitive_detector_respects_inclusion_exclusion_threshold_and_disabled(
) -> None:
    data = pd.DataFrame(
        {
            "contact": [
                "alice@example.test",
                "not-an-email",
            ],
            "notes": [
                "bob@example.test",
                "carol@example.test",
            ],
        }
    )
    result = Phase1Result(
        data=data,
        internal_row_id_column="__mlprep_row_id__",
    )
    state = Phase1State()
    detector = SensitiveDataDetector(
        SensitiveDataOptions(
            include_columns=["contact", "notes", "missing"],
            exclude_columns=["notes"],
            detection_threshold=0.75,
        )
    )

    detected = detector.inspect(data, result, state)
    disabled = SensitiveDataDetector(
        SensitiveDataOptions(enabled=False)
    ).inspect(data, result, state)

    assert detected == {}
    assert disabled == {}
    assert result.findings == []


def test_phase1_report_filters_sections_and_saves_json_and_text(tmp_path) -> None:
    data = pd.DataFrame(
        {"feature": [1, 2]},
        index=pd.Index([10, 11], name="row_id"),
    )
    result = Phase1Result(
        data=data,
        internal_row_id_column="__mlprep_row_id__",
    )
    result.dataset_summary.feature_count = 1
    result.dataset_summary.sensitive_column_count = 1
    result.add_finding(
        Finding(
            code="INFO-001",
            category=FindingCategory.INPUT,
            severity=Severity.INFO,
            component="inspection",
            message="Informational finding.",
        )
    )
    result.add_finding(
        Finding(
            code="WARN-001",
            category=FindingCategory.SENSITIVE_DATA,
            severity=Severity.WARNING,
            component="sensitive_data",
            message="Sensitive pattern detected.",
            columns=["email"],
            recommended_action=FindingAction.WARN,
            applied_action=FindingAction.REPORT,
        )
    )
    result.add_normalization_event(
        NormalizationEvent(
            component="normalization",
            operation="normalize_text_values",
            description="Normalized label values.",
            column="feature",
            affected_row_count=2,
        )
    )
    result.add_column_inventory_item(
        ColumnInventoryItem(
            original_name="feature",
            normalized_name="feature",
            physical_type="integer",
            semantic_type="discrete",
            unique_count=2,
        )
    )
    state = Phase1State(
        dataset_fingerprint="dataset-fingerprint",
        schema_fingerprint="schema-fingerprint",
        column_states={
            "feature": ColumnState(
                original_name="feature",
                normalized_name="feature",
            )
        },
    )
    builder = Phase1ReportBuilder(
        ReportOptions(
            include_info_findings=False,
            include_normalization_events=False,
            include_column_inventory=False,
            include_state=True,
            indent=4,
        )
    )

    report = builder.build_dictionary(
        result=result,
        state=state,
    )
    json_path = builder.save_json(
        tmp_path / "nested" / "phase1-report.json",
        result,
        state,
    )
    text_path = builder.save_text(
        tmp_path / "nested" / "phase1-report.txt",
        result,
        state,
    )
    saved_report = json.loads(json_path.read_text(encoding="utf-8"))
    text_report = text_path.read_text(encoding="utf-8")

    assert [finding["code"] for finding in report["findings"]] == [
        "WARN-001"
    ]
    assert "normalization_events" not in report
    assert "column_inventory" not in report
    assert report["fingerprints"] == {
        "dataset": "dataset-fingerprint",
        "schema": "schema-fingerprint",
    }
    assert report["phase_state"]["dataset_fingerprint"] == "dataset-fingerprint"
    assert saved_report == report
    assert "Sensitive columns: 1" in text_report
    assert "warning: 1" in text_report
    assert json_path.is_file() and json_path.stat().st_size > 0
    assert text_path.is_file() and text_path.stat().st_size > 0


def test_phase1_report_covers_unevaluated_readiness_and_blocking_findings() -> None:
    data = pd.DataFrame({"feature": [1, 2]})
    result = Phase1Result(
        data=data,
        internal_row_id_column="__mlprep_row_id__",
    )
    result.add_finding(
        Finding(
            code="BLOCK-001",
            category=FindingCategory.INPUT,
            severity=Severity.CRITICAL,
            component="inspection",
            message="A required input is unavailable.",
        )
    )
    state = Phase1State()
    builder = Phase1ReportBuilder(
        ReportOptions(
            include_state=True,
            indent=0,
        )
    )

    report = builder.build_dictionary(result, state)
    serialized = builder.build_json(result, state)
    text = builder.build_text(result, state)

    assert report["findings"][0]["code"] == "BLOCK-001"
    assert "phase_state" in report
    assert json.loads(serialized) == report
    assert "\n  " not in serialized
    assert "Status: not_evaluated" in text
    assert "Ready for Phase 2: False" in text
    assert "critical: 1" in text
    assert "Blocking findings" in text
    assert "- [BLOCK-001] A required input is unavailable." in text
    assert "Dataset: not generated" in text
    assert "Schema: not generated" in text


def test_sensitive_detector_uses_sampled_match_ratio_and_luhn_validation() -> None:
    data = pd.DataFrame(
        {
            "contact": [
                "alice@example.test",
                "bob@example.test",
                "carol@example.test",
                "not-an-email",
            ],
            "card": [
                "4111111111111111",
                "4111111111111112",
                "ordinary",
                "ordinary",
            ],
        }
    )
    result = Phase1Result(
        data=data,
        internal_row_id_column="__mlprep_row_id__",
    )
    state = Phase1State()
    detector = SensitiveDataDetector(
        SensitiveDataOptions(
            include_columns=["contact", "card"],
            detect_phone_numbers=False,
            detect_ip_addresses=False,
            detect_urls=False,
            sample_size=4,
            detection_threshold=0.75,
        )
    )

    detected = detector.inspect(data, result, state)
    contact_finding = next(
        finding
        for finding in result.findings
        if finding.columns == ["contact"]
    )

    assert detected == {"contact": "email"}
    assert contact_finding.details["match_ratio"] == 0.75
    assert "alice@example.test" not in contact_finding.message
    assert "4111111111111111" not in contact_finding.message

    card_result = Phase1Result(
        data=data,
        internal_row_id_column="__mlprep_row_id__",
    )
    card_detection = SensitiveDataDetector(
        SensitiveDataOptions(
            include_columns=["card"],
            detect_emails=False,
            detect_phone_numbers=False,
            detect_ip_addresses=False,
            detect_urls=False,
            detection_threshold=0.25,
        )
    ).inspect(data, card_result, Phase1State())

    assert card_detection == {"card": "payment_card"}
    assert card_result.findings[0].details["match_ratio"] == 0.25


def test_sensitive_detector_validates_options_and_input_type() -> None:
    with pytest.raises(ValueError, match="sample_size"):
        SensitiveDataOptions(sample_size=0)
    with pytest.raises(ValueError, match="detection_threshold"):
        SensitiveDataOptions(detection_threshold=1.1)

    data = pd.DataFrame({"email": ["alice@example.test"]})
    result = Phase1Result(
        data=data,
        internal_row_id_column="__mlprep_row_id__",
    )
    with pytest.raises(TypeError, match="pandas DataFrame"):
        SensitiveDataDetector().inspect(
            pd.Series(["alice@example.test"]),  # type: ignore[arg-type]
            result,
            Phase1State(),
        )
