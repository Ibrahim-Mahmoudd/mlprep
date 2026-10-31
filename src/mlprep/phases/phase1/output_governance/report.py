"""Human-readable and JSON Phase 1 reporting."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mlprep.phases.phase1.output_governance.result import (
    Phase1Result,
)
from mlprep.phases.phase1.output_governance.state import (
    Phase1State,
)


@dataclass(slots=True)
class ReportOptions:
    """Options controlling Phase 1 report generation."""

    include_info_findings: bool = True
    include_normalization_events: bool = True
    include_column_inventory: bool = True
    include_state: bool = False

    indent: int = 2

    def __post_init__(self) -> None:
        """Validate report options."""

        if self.indent < 0:
            raise ValueError(
                "indent cannot be negative."
            )


class Phase1ReportBuilder:
    """Build machine-readable and human-readable reports."""

    def __init__(
        self,
        options: ReportOptions | None = None,
    ) -> None:
        """Initialize the report builder."""

        self.options = options or ReportOptions()

    def build_dictionary(
        self,
        result: Phase1Result,
        state: Phase1State,
    ) -> dict[str, Any]:
        """Build the complete machine-readable report."""

        report = result.to_report_dict()

        if not self.options.include_info_findings:
            report["findings"] = [
                finding
                for finding in report["findings"]
                if finding["severity"] != "info"
            ]

        if not self.options.include_normalization_events:
            report.pop(
                "normalization_events",
                None,
            )

        if not self.options.include_column_inventory:
            report.pop(
                "column_inventory",
                None,
            )

        report["fingerprints"] = {
            "dataset": state.dataset_fingerprint,
            "schema": state.schema_fingerprint,
        }

        if self.options.include_state:
            report["phase_state"] = state.model_dump(
                mode="json"
            )

        return report

    def build_json(
        self,
        result: Phase1Result,
        state: Phase1State,
    ) -> str:
        """Build the Phase 1 report as JSON text."""

        report = self.build_dictionary(
            result=result,
            state=state,
        )

        return json.dumps(
            report,
            indent=self.options.indent,
            ensure_ascii=False,
            sort_keys=False,
            default=str,
        )

    def build_text(
        self,
        result: Phase1Result,
        state: Phase1State,
    ) -> str:
        """Build a concise human-readable Phase 1 summary."""

        readiness = result.readiness

        status = (
            readiness.status.value
            if readiness is not None
            else "not_evaluated"
        )

        ready = (
            readiness.ready
            if readiness is not None
            else False
        )

        summary = result.dataset_summary

        lines = [
            "Phase 1 Report",
            "==============",
            f"Status: {status}",
            f"Ready for Phase 2: {ready}",
            "",
            "Dataset",
            "-------",
            (
                f"Rows: {summary.row_count_before} "
                f"-> {summary.row_count_after}"
            ),
            (
                f"Columns: {summary.column_count_before} "
                f"-> {summary.column_count_after}"
            ),
            f"Features: {summary.feature_count}",
            f"Identifiers: {summary.identifier_count}",
            f"Timestamps: {summary.timestamp_count}",
            (
                "Sensitive columns: "
                f"{summary.sensitive_column_count}"
            ),
            (
                "Missing cells: "
                f"{summary.missing_cell_count}"
            ),
            (
                "Duplicate rows: "
                f"{summary.duplicate_row_count}"
            ),
            (
                "Rejected rows: "
                f"{summary.rejected_row_count}"
            ),
            (
                "Memory saved: "
                f"{summary.memory_saved_bytes} bytes"
            ),
            "",
            "Findings",
            "--------",
        ]

        severity_counts = (
            result.finding_counts_by_severity()
        )

        for severity, count in severity_counts.items():
            lines.append(
                f"{severity}: {count}"
            )

        lines.extend(
            [
                "",
                "Fingerprints",
                "------------",
                (
                    "Dataset: "
                    f"{state.dataset_fingerprint or 'not generated'}"
                ),
                (
                    "Schema: "
                    f"{state.schema_fingerprint or 'not generated'}"
                ),
            ]
        )

        blocking_findings = (
            result.get_blocking_findings()
        )

        if blocking_findings:
            lines.extend(
                [
                    "",
                    "Blocking findings",
                    "-----------------",
                ]
            )

            for finding in blocking_findings:
                lines.append(
                    f"- [{finding.code}] {finding.message}"
                )

        return "\n".join(lines)

    def save_json(
        self,
        path: str | Path,
        result: Phase1Result,
        state: Phase1State,
    ) -> Path:
        """Save the JSON report to disk."""

        output_path = Path(path)

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        output_path.write_text(
            self.build_json(
                result=result,
                state=state,
            ),
            encoding="utf-8",
        )

        return output_path

    def save_text(
        self,
        path: str | Path,
        result: Phase1Result,
        state: Phase1State,
    ) -> Path:
        """Save the human-readable report to disk."""

        output_path = Path(path)

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        output_path.write_text(
            self.build_text(
                result=result,
                state=state,
            ),
            encoding="utf-8",
        )

        return output_path