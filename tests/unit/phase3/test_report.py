from __future__ import annotations

import json

from mlprep.phases.phase3.output_governance.report import (
    Phase3ReportBuilder,
    Phase3ReportOptions,
)
from mlprep.phases.phase3.output_governance.result import (
    Phase3Finding,
    Phase3FindingCategory,
    Phase3FindingSeverity,
    Phase3Result,
)
from mlprep.phases.phase3.output_governance.state import Phase3State


def test_report_includes_governance_and_omits_disabled_details(tmp_path) -> None:
    result = Phase3Result()
    result.add_finding(
        Phase3Finding(
            code="INFO-001",
            category=Phase3FindingCategory.PROFILING,
            severity=Phase3FindingSeverity.INFO,
            component="profile",
            message="Informational detail",
        )
    )
    result.add_finding(
        Phase3Finding(
            code="WARN-001",
            category=Phase3FindingCategory.DIAGNOSTIC,
            severity=Phase3FindingSeverity.WARNING,
            component="diagnostic",
            message="Review recommended",
        )
    )
    state = Phase3State(
        training_fingerprint_before="before",
        training_fingerprint_after="after",
        training_data_unchanged=True,
        validation_data_unchanged=True,
        completed_components=["profile"],
    )
    builder = Phase3ReportBuilder(
        Phase3ReportOptions(
            include_info_findings=False,
            include_state=True,
            include_column_profiles=False,
            include_relationship_details=False,
            include_diagnostic_details=False,
            include_plot_index=False,
        )
    )

    report = builder.build_dictionary(result=result, state=state)

    assert [finding["code"] for finding in report["findings"]] == ["WARN-001"]
    assert "numeric_profiles" not in report
    assert "relationships" not in report
    assert "diagnostics" not in report
    assert "plots" not in report
    assert report["fingerprints"]["training_before"] == "before"
    assert report["governance"]["training_data_unchanged"] is True
    assert report["phase_state"]["completed_components"] == ["profile"]

    text = builder.build_text(result=result, state=state)
    assert "Status: not_evaluated" in text
    assert "Transformations: 0" in text
    assert "warning: 1" in text

    json_path = builder.save_json(
        path=tmp_path / "nested" / "report.json",
        result=result,
        state=state,
    )
    text_path = builder.save_text(
        path=tmp_path / "nested" / "report.txt",
        result=result,
        state=state,
    )
    assert json.loads(json_path.read_text(encoding="utf-8"))["phase"] == "phase_3"
    assert "Phase 3 Report" in text_path.read_text(encoding="utf-8")
