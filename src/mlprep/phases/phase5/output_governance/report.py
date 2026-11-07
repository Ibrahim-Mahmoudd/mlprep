"""JSON and text reporting for Phase 5."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mlprep.phases.phase5.output_governance.metadata import (
    Phase5FindingSeverity,
)
from mlprep.phases.phase5.output_governance.result import (
    Phase5Result,
)
from mlprep.phases.phase5.output_governance.state import (
    Phase5State,
)


@dataclass(slots=True)
class Phase5ReportOptions:
    """Options controlling Phase 5 reports."""

    include_info_findings: bool = True

    include_phase_state: bool = False

    include_validation_details: bool = True

    include_execution_metadata: bool = True

    include_feature_treatments: bool = True

    include_fingerprints: bool = True

    expose_fitted_bounds: bool = True

    indent: int = 2

    def __post_init__(self) -> None:
        """Validate reporting options."""

        if self.indent < 0:
            raise ValueError(
                "indent cannot be negative."
            )


class Phase5ReportBuilder:
    """Build JSON and text Phase 5 reports."""

    def __init__(
        self,
        options: Phase5ReportOptions | None = None,
    ) -> None:
        """Initialize the report builder."""

        self.options = (
            options
            or Phase5ReportOptions()
        )

    def build_dictionary(
        self,
        *,
        result: Phase5Result,
        state: Phase5State,
    ) -> dict[str, Any]:
        """Build the complete report dictionary."""

        report = result.to_report_dict()

        if not self.options.include_info_findings:
            report["findings"] = [
                finding
                for finding in report["findings"]
                if finding["severity"]
                != Phase5FindingSeverity.INFO.value
            ]

        if not self.options.include_validation_details:
            report.pop(
                "validation",
                None,
            )

        if not self.options.include_execution_metadata:
            report.pop(
                "execution_metadata",
                None,
            )

        elif (
            not self.options.include_feature_treatments
            and report.get(
                "execution_metadata"
            )
            is not None
        ):
            report[
                "execution_metadata"
            ].pop(
                "features",
                None,
            )

        report["governance"] = {
            "fitted_on_training_only": (
                state.fitted_on_training_only
            ),
            "test_transform_authorized": (
                state.test_transform_authorized
            ),
            "test_partition_remained_protected": (
                state
                .test_partition_remained_protected
            ),
            "targets_unchanged": (
                state.targets_unchanged
            ),
            "row_ids_unchanged": (
                state.row_ids_unchanged
            ),
            "schemas_consistent": (
                state.schemas_consistent
            ),
            "completed_components": list(
                state.completed_components
            ),
            "failed_components": list(
                state.failed_components
            ),
            "skipped_components": list(
                state.skipped_components
            ),
        }

        report["treatment_summary"] = {
            "detected_outliers": (
                state
                .total_detected_outlier_count
            ),
            "changed_values": (
                state.total_changed_value_count
            ),
            "remaining_outliers": (
                state
                .total_remaining_outlier_count
            ),
            "generated_indicators": (
                state.generated_indicator_count
            ),
            "indicator_mapping": dict(
                state.indicator_mapping
            ),
        }

        if self.options.expose_fitted_bounds:
            report["detector_states"] = (
                state.detector_states
            )

        else:
            report["detector_states"] = {
                column: {
                    key: value
                    for key, value
                    in detector_state.items()
                    if key not in {
                        "lower_bound",
                        "upper_bound",
                        "q1",
                        "q3",
                        "median",
                        "mad",
                    }
                }
                for column, detector_state
                in state.detector_states.items()
            }

        report["treatment_states"] = {
            column: treatment_state.model_dump(
                mode="json"
            )
            for column, treatment_state
            in state.treatment_states.items()
        }

        if self.options.include_fingerprints:
            report["fingerprints"] = {
                "phase4_output": (
                    state.phase4_output_fingerprint
                ),
                "inputs": {
                    partition.value: fingerprint
                    for partition, fingerprint
                    in state
                    .input_partition_fingerprints
                    .items()
                },
                "outputs": {
                    partition.value: fingerprint
                    for partition, fingerprint
                    in state
                    .output_partition_fingerprints
                    .items()
                },
                "configuration": (
                    state.configuration_fingerprint
                ),
                "plan": (
                    state
                    .treatment_plan_fingerprint
                ),
                "fitted_state": (
                    state
                    .fitted_state_fingerprint
                ),
                "combined_output": (
                    state.output_fingerprint
                ),
            }

        if self.options.include_phase_state:
            report["phase_state"] = (
                state.model_dump(
                    mode="json"
                )
            )

        return report

    def build_json(
        self,
        *,
        result: Phase5Result,
        state: Phase5State,
    ) -> str:
        """Build a JSON report."""

        return json.dumps(
            self.build_dictionary(
                result=result,
                state=state,
            ),
            ensure_ascii=False,
            indent=self.options.indent,
            default=str,
        )

    def build_text(
        self,
        *,
        result: Phase5Result,
        state: Phase5State,
    ) -> str:
        """Build a readable text report."""

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
            "Phase 5 Report",
            "==============",
            f"Status: {status}",
            f"Ready for Phase 6: {ready}",
            "",
            "Partition summary",
            "-----------------",
            (
                "Training rows: "
                f"{result.train.row_count()}"
            ),
            (
                "Validation rows: "
                f"{result.validation.row_count()}"
            ),
            (
                "Test rows: "
                f"{result.test.row_count()}"
            ),
            (
                "Output features: "
                f"{result.train.feature_count()}"
            ),
            "",
            "Treatment summary",
            "-----------------",
            (
                "Fitted detectors: "
                f"{len(state.detector_states)}"
            ),
            (
                "Fitted treatments: "
                f"{len(state.treatment_states)}"
            ),
            (
                "Detected outliers: "
                f"{state.total_detected_outlier_count}"
            ),
            (
                "Changed values: "
                f"{state.total_changed_value_count}"
            ),
            (
                "Remaining outliers: "
                f"{state.total_remaining_outlier_count}"
            ),
            (
                "Generated indicators: "
                f"{state.generated_indicator_count}"
            ),
            "",
            "Governance",
            "----------",
            (
                "Fitted on training only: "
                f"{state.fitted_on_training_only}"
            ),
            (
                "Test transform authorized: "
                f"{state.test_transform_authorized}"
            ),
            (
                "Test remained protected: "
                f"{state.test_partition_remained_protected}"
            ),
            (
                "Targets unchanged: "
                f"{state.targets_unchanged}"
            ),
            (
                "Row IDs unchanged: "
                f"{state.row_ids_unchanged}"
            ),
            (
                "Schemas consistent: "
                f"{state.schemas_consistent}"
            ),
            "",
            "Findings",
            "--------",
        ]

        for severity, count in (
            result.finding_counts().items()
        ):
            lines.append(
                f"{severity}: {count}"
            )

        blocking = (
            result.blocking_findings()
        )

        if blocking:
            lines.extend(
                [
                    "",
                    "Blocking findings",
                    "-----------------",
                ]
            )

            for finding in blocking:
                lines.append(
                    f"- [{finding.code}] "
                    f"{finding.message}"
                )

        if state.indicator_mapping:
            lines.extend(
                [
                    "",
                    "Outlier indicators",
                    "------------------",
                ]
            )

            for source, indicator in sorted(
                state.indicator_mapping.items()
            ):
                lines.append(
                    f"- {source}: {indicator}"
                )

        return "\n".join(
            lines
        )

    def save_json(
        self,
        *,
        path: str | Path,
        result: Phase5Result,
        state: Phase5State,
    ) -> Path:
        """Save the JSON report."""

        output_path = Path(
            path
        )

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
        result: Phase5Result,
        state: Phase5State,
    ) -> Path:
        """Save the text report."""

        output_path = Path(
            path
        )

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
