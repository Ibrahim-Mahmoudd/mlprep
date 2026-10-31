"""JSON and text reporting for Phase 2."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mlprep.phases.phase2.output_governance.result import (
    Phase2Result,
)
from mlprep.phases.phase2.output_governance.state import (
    Phase2State,
)


@dataclass(slots=True)
class Phase2ReportOptions:
    """Options controlling Phase 2 reporting."""

    include_info_findings: bool = True
    include_state: bool = False
    include_membership_row_ids: bool = False

    indent: int = 2

    def __post_init__(self) -> None:
        """Validate reporting options."""

        if self.indent < 0:
            raise ValueError(
                "indent cannot be negative."
            )


class Phase2ReportBuilder:
    """Build machine-readable and text Phase 2 reports."""

    def __init__(
        self,
        options: Phase2ReportOptions | None = None,
    ) -> None:
        """Initialize the report builder."""

        self.options = (
            options or Phase2ReportOptions()
        )

    def build_dictionary(
        self,
        *,
        result: Phase2Result,
        state: Phase2State,
    ) -> dict[str, Any]:
        """Build the complete report dictionary."""

        report = result.to_report_dict()

        if not self.options.include_info_findings:
            report["findings"] = [
                finding
                for finding in report["findings"]
                if finding["severity"] != "info"
            ]

        report["fingerprints"] = {
            "source": (
                state.source_dataset_fingerprint
            ),
            "train": state.train_fingerprint,
            "validation": (
                state.validation_fingerprint
            ),
            "test": state.test_fingerprint,
            "split": state.split_fingerprint,
            "configuration": (
                state.configuration_fingerprint
            ),
        }

        report["leakage_firewall"] = (
            state.leakage_firewall.model_dump(
                mode="json"
            )
        )

        report["test_lock"] = (
            state.test_lock.model_dump(
                mode="json"
            )
        )

        if self.options.include_state:
            state_data = state.model_dump(
                mode="json"
            )

            if not (
                self.options
                .include_membership_row_ids
            ):
                memberships = state_data.get(
                    "memberships",
                    {}
                )

                for membership in memberships.values():
                    membership.pop(
                        "row_ids",
                        None,
                    )

            report["phase_state"] = state_data

        return report

    def build_json(
        self,
        *,
        result: Phase2Result,
        state: Phase2State,
    ) -> str:
        """Build a JSON report."""

        return json.dumps(
            self.build_dictionary(
                result=result,
                state=state,
            ),
            indent=self.options.indent,
            ensure_ascii=False,
            default=str,
        )

    def build_text(
        self,
        *,
        result: Phase2Result,
        state: Phase2State,
    ) -> str:
        """Build a human-readable report."""

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

        lines = [
            "Phase 2 Report",
            "==============",
            f"Strategy: {state.strategy.value}",
            f"Status: {status}",
            f"Ready for Phase 3: {ready}",
            "",
            "Partitions",
            "----------",
            (
                f"Train rows: "
                f"{result.train.row_count()}"
            ),
            (
                f"Validation rows: "
                f"{result.validation.row_count()}"
            ),
            (
                f"Test rows: "
                f"{result.test.row_count}"
            ),
            (
                f"Feature count: "
                f"{result.train.feature_count()}"
            ),
            "",
            "Leakage firewall",
            "----------------",
            (
                f"Passed: "
                f"{state.leakage_firewall.passed}"
            ),
            (
                "Row overlap checked: "
                f"{state.leakage_firewall.row_overlap_checked}"
            ),
            (
                "Group overlap checked: "
                f"{state.leakage_firewall.group_overlap_checked}"
            ),
            (
                "Temporal ordering checked: "
                f"{state.leakage_firewall.temporal_order_checked}"
            ),
            "",
            "Findings",
            "--------",
        ]

        for severity, count in (
            result.finding_counts_by_severity().items()
        ):
            lines.append(
                f"{severity}: {count}"
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
                    f"- [{finding.code}] "
                    f"{finding.message}"
                )

        return "\n".join(lines)

    def save_json(
        self,
        *,
        path: str | Path,
        result: Phase2Result,
        state: Phase2State,
    ) -> Path:
        """Save a JSON report."""

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
        *,
        path: str | Path,
        result: Phase2Result,
        state: Phase2State,
    ) -> Path:
        """Save a text report."""

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