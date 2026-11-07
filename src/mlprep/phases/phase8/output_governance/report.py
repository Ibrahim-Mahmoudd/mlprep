"""JSON and text reporting for Phase 8."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mlprep.phases.phase8.output_governance.metadata import (
    Phase8FindingSeverity,
)
from mlprep.phases.phase8.output_governance.result import (
    Phase8Result,
)
from mlprep.phases.phase8.output_governance.state import (
    Phase8State,
)


@dataclass(slots=True)
class Phase8ReportOptions:
    """Options controlling Phase 8 reports."""

    include_info_findings: bool = True
    include_phase_state: bool = False

    include_validation_details: bool = True
    include_execution_metadata: bool = True

    include_distribution_details: bool = True
    include_sampler_state: bool = True
    include_class_weights: bool = True
    include_sample_weights: bool = False
    include_row_lineage: bool = True
    include_removed_row_lineage: bool = True

    include_feature_lineage: bool = True
    include_fingerprints: bool = True

    maximum_reported_rows: int | None = 1000

    indent: int = 2

    def __post_init__(self) -> None:
        """Validate report options."""

        if self.indent < 0:
            raise ValueError(
                "indent cannot be negative."
            )

        if (
            self.maximum_reported_rows is not None
            and self.maximum_reported_rows < 1
        ):
            raise ValueError(
                "maximum_reported_rows must be positive."
            )


class Phase8ReportBuilder:
    """Build JSON and text reports for Phase 8."""

    def __init__(
        self,
        options: Phase8ReportOptions | None = None,
    ) -> None:
        """Initialize the report builder."""

        self.options = (
            options
            or Phase8ReportOptions()
        )

    def build_dictionary(
        self,
        *,
        result: Phase8Result,
        state: Phase8State,
    ) -> dict[str, Any]:
        """Build the complete report dictionary."""

        report = result.to_report_dict()

        if not self.options.include_info_findings:
            report["findings"] = [
                finding
                for finding in report[
                    "findings"
                ]
                if finding[
                    "severity"
                ]
                != Phase8FindingSeverity.INFO.value
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

        report["balancing_summary"] = {
            "configured_strategy": (
                state.configured_strategy
            ),
            "executed_strategy": (
                state.executed_strategy
            ),
            "source_training_rows": (
                state.source_training_row_count
            ),
            "output_training_rows": (
                state.output_training_row_count
            ),
            "retained_original_rows": len(
                state.retained_original_row_ids
            ),
            "duplicate_rows": len(
                state.duplicate_row_ids
            ),
            "synthetic_rows": len(
                state.synthetic_row_ids
            ),
            "removed_original_rows": len(
                state.removed_original_row_ids
            ),
        }

        report["governance"] = {
            "fitted_on_training_only": (
                state.fitted_on_training_only
            ),
            "target_access_authorized": (
                state.target_access_authorized
            ),
            "resampling_authorized": (
                state.resampling_authorized
            ),
            "synthetic_generation_authorized": (
                state
                .synthetic_generation_authorized
            ),
            "validation_resampled": (
                state.validation_resampled
            ),
            "test_resampled": (
                state.test_resampled
            ),
            "validation_remained_protected": (
                state
                .validation_remained_protected
            ),
            "test_remained_protected": (
                state.test_remained_protected
            ),
            "lineage_complete": (
                state.lineage_complete
            ),
        }

        if self.options.include_distribution_details:
            report["distribution"] = {
                "before": (
                    self._model_dump(
                        state
                        .source_class_distribution
                    )
                ),
                "after": (
                    self._model_dump(
                        state
                        .output_class_distribution
                    )
                ),
                "metrics_before": (
                    self._model_dump(
                        state
                        .imbalance_metrics_before
                    )
                ),
                "metrics_after": (
                    self._model_dump(
                        state
                        .imbalance_metrics_after
                    )
                ),
                "recommendation": (
                    self._model_dump(
                        state
                        .strategy_recommendation
                    )
                ),
            }

        if self.options.include_sampler_state:
            report["sampler_state"] = (
                self._model_dump(
                    state.sampler_state
                )
            )

        if self.options.include_class_weights:
            report["class_weight_state"] = (
                self._model_dump(
                    state.class_weights
                )
            )

        if self.options.include_sample_weights:
            report["sample_weight_state"] = (
                self._model_dump(
                    state.sample_weights
                )
            )

        if (
            self.options.include_row_lineage
            and state.row_lineage is not None
        ):
            lineage = state.row_lineage.model_dump(
                mode="json"
            )

            maximum = (
                self.options
                .maximum_reported_rows
            )

            if maximum is not None:
                lineage[
                    "entries"
                ] = lineage[
                    "entries"
                ][
                    :maximum
                ]

                if (
                    self.options
                    .include_removed_row_lineage
                ):
                    lineage[
                        "removed_rows"
                    ] = lineage[
                        "removed_rows"
                    ][
                        :maximum
                    ]

            if not self.options.include_removed_row_lineage:
                lineage.pop(
                    "removed_rows",
                    None,
                )

            report["row_lineage"] = lineage

        if self.options.include_feature_lineage:
            report["selected_feature_lineage"] = (
                self._model_dump(
                    state.selected_feature_lineage
                )
            )

        if self.options.include_fingerprints:
            report["fingerprints"] = {
                "phase7_output": (
                    state.phase7_output_fingerprint
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
                "balancing_plan": (
                    state.balancing_plan_fingerprint
                ),
                "fitted_state": (
                    state.fitted_state_fingerprint
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
        result: Phase8Result,
        state: Phase8State,
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
        result: Phase8Result,
        state: Phase8State,
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
            "Phase 8 Report",
            "==============",
            f"Status: {status}",
            f"Ready for Phase 9: {ready}",
            "",
            "Balancing",
            "---------",
            (
                "Configured strategy: "
                f"{state.configured_strategy}"
            ),
            (
                "Executed strategy: "
                f"{state.executed_strategy}"
            ),
            (
                "Source training rows: "
                f"{state.source_training_row_count}"
            ),
            (
                "Output training rows: "
                f"{state.output_training_row_count}"
            ),
            (
                "Synthetic rows: "
                f"{len(state.synthetic_row_ids)}"
            ),
            (
                "Duplicate rows: "
                f"{len(state.duplicate_row_ids)}"
            ),
            (
                "Removed original rows: "
                f"{len(state.removed_original_row_ids)}"
            ),
            "",
            "Protection",
            "----------",
            (
                "Validation remained protected: "
                f"{state.validation_remained_protected}"
            ),
            (
                "Test remained protected: "
                f"{state.test_remained_protected}"
            ),
            (
                "Lineage complete: "
                f"{state.lineage_complete}"
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

        blocking = result.blocking_findings()

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

        return "\n".join(
            lines
        )

    def save_json(
        self,
        *,
        path: str | Path,
        result: Phase8Result,
        state: Phase8State,
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
        result: Phase8Result,
        state: Phase8State,
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
    def _model_dump(
        value: Any,
    ) -> dict[str, Any] | None:
        """Serialize an optional Pydantic model."""

        if value is None:
            return None

        return value.model_dump(
            mode="json"
        )
