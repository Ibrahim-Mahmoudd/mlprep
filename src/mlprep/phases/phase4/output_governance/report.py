"""JSON and text reporting for Phase 4."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mlprep.phases.phase4.output_governance.metadata import (
    Phase4FindingSeverity,
)
from mlprep.phases.phase4.output_governance.result import (
    Phase4Result,
)
from mlprep.phases.phase4.output_governance.state import (
    Phase4State,
)


@dataclass(slots=True)
class Phase4ReportOptions:
    """Options controlling Phase 4 reports."""

    include_info_findings: bool = True
    include_phase_state: bool = False

    include_validation_details: bool = True
    include_execution_metadata: bool = True
    include_feature_methods: bool = True
    include_fingerprints: bool = True

    expose_learned_values: bool = False

    indent: int = 2

    def __post_init__(self) -> None:
        """Validate report options."""

        if self.indent < 0:
            raise ValueError(
                "indent cannot be negative."
            )


class Phase4ReportBuilder:
    """Build JSON and text reports for Phase 4."""

    def __init__(
        self,
        options: Phase4ReportOptions | None = None,
    ) -> None:
        """Initialize the report builder."""

        self.options = (
            options or Phase4ReportOptions()
        )

    def build_dictionary(
        self,
        *,
        result: Phase4Result,
        state: Phase4State,
    ) -> dict[str, Any]:
        """Build a complete Phase 4 report dictionary."""

        report = result.to_report_dict()

        if not self.options.include_info_findings:
            report["findings"] = [
                finding
                for finding in report["findings"]
                if finding["severity"]
                != Phase4FindingSeverity.INFO.value
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
            not self.options.include_feature_methods
            and report.get(
                "execution_metadata"
            )
            is not None
        ):
            report[
                "execution_metadata"
            ].pop(
                "feature_methods",
                None,
            )

        if self.options.include_fingerprints:
            report["fingerprints"] = {
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
                    .imputation_plan_fingerprint
                ),
                "fitted_state": (
                    state
                    .fitted_state_fingerprint
                ),
                "combined_output": (
                    state.output_fingerprint
                ),
            }

        report["governance"] = {
            "fitted_on_training_only": (
                state.fitted_on_training_only
            ),
            "test_transform_authorized": (
                state.test_transform_authorized
            ),
            "test_partition_remained_locked": (
                state
                .test_partition_remained_locked
            ),
            "test_access_count_before": (
                state.test_access_count_before
            ),
            "test_access_count_after": (
                state.test_access_count_after
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
            "remaining_missing_count": (
                state.remaining_missing_count
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

        report["indicator_mapping"] = dict(
            state.indicator_mapping
        )

        if self.options.expose_learned_values:
            report["imputer_states"] = {
                column: imputer_state.model_dump(
                    mode="json"
                )
                for column, imputer_state
                in state.imputer_states.items()
            }

        else:
            report["imputer_states"] = {
                column: self._safe_imputer_state(
                    imputer_state.model_dump(
                        mode="json"
                    )
                )
                for column, imputer_state
                in state.imputer_states.items()
            }

        if self.options.include_phase_state:
            report["phase_state"] = state.model_dump(
                mode="json"
            )

        return report

    def build_json(
        self,
        *,
        result: Phase4Result,
        state: Phase4State,
    ) -> str:
        """Build a JSON Phase 4 report."""

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
        result: Phase4Result,
        state: Phase4State,
    ) -> str:
        """Build a readable Phase 4 text report."""

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
            "Phase 4 Report",
            "==============",
            f"Status: {status}",
            f"Ready for Phase 5: {ready}",
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
            "Imputation summary",
            "------------------",
            (
                "Fitted imputers: "
                f"{len(state.imputer_states)}"
            ),
            (
                "Imputed values: "
                f"{state.imputed_value_count}"
            ),
            (
                "Generated indicators: "
                f"{state.generated_indicator_count}"
            ),
            (
                "Remaining missing values: "
                f"{state.remaining_missing_count}"
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
                f"{state.test_partition_remained_locked}"
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
                    "Missing indicators",
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
        result: Phase4Result,
        state: Phase4State,
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
        result: Phase4Result,
        state: Phase4State,
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

    @staticmethod
    def _safe_imputer_state(
        state: dict[str, Any],
    ) -> dict[str, Any]:
        """Hide potentially sensitive learned values."""

        safe_state = dict(
            state
        )

        if "learned_value" in safe_state:
            safe_state[
                "learned_value"
            ] = "<redacted>"

        metadata = safe_state.get(
            "metadata"
        )

        if isinstance(metadata, dict):
            safe_metadata = dict(
                metadata
            )

            sensitive_keys = {
                "group_mapping",
                "group_values",
                "classes",
                "coefficients",
                "feature_means",
            }

            for key in sensitive_keys:
                if key in safe_metadata:
                    safe_metadata[
                        key
                    ] = "<redacted>"

            safe_state[
                "metadata"
            ] = safe_metadata

        return safe_state