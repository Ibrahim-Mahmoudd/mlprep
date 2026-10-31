"""JSON and text reporting for Phase 6."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mlprep.phases.phase6.output_governance.metadata import (
    Phase6FindingSeverity,
)
from mlprep.phases.phase6.output_governance.result import (
    Phase6Result,
)
from mlprep.phases.phase6.output_governance.state import (
    Phase6State,
)


@dataclass(slots=True)
class Phase6ReportOptions:
    """Options controlling Phase 6 reports."""

    include_info_findings: bool = True
    include_phase_state: bool = False

    include_validation_details: bool = True
    include_execution_metadata: bool = True

    include_feature_states: bool = True
    include_feature_lineage: bool = True
    include_fingerprints: bool = True

    expose_category_vocabularies: bool = False
    expose_numeric_parameters: bool = True

    indent: int = 2

    def __post_init__(self) -> None:
        """Validate report options."""

        if self.indent < 0:
            raise ValueError(
                "indent cannot be negative."
            )


class Phase6ReportBuilder:
    """Build JSON and text Phase 6 reports."""

    def __init__(
        self,
        options: Phase6ReportOptions | None = None,
    ) -> None:
        """Initialize the report builder."""

        self.options = (
            options
            or Phase6ReportOptions()
        )

    def build_dictionary(
        self,
        *,
        result: Phase6Result,
        state: Phase6State,
    ) -> dict[str, Any]:
        """Build a complete report dictionary."""

        report = result.to_report_dict()

        if not self.options.include_info_findings:
            report["findings"] = [
                finding
                for finding
                in report["findings"]
                if finding["severity"]
                != Phase6FindingSeverity.INFO.value
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
            "numeric_integrity_valid": (
                state.numeric_integrity_valid
            ),
            "unknown_categories_valid": (
                state.unknown_categories_valid
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

        report["feature_summary"] = {
            "source_feature_count": len(
                state.source_feature_columns
            ),
            "output_feature_count": len(
                state.output_feature_columns
            ),
            "numeric_source_count": len(
                state.numeric_states
            ),
            "categorical_source_count": len(
                state.categorical_states
            ),
            "boolean_source_count": len(
                state.boolean_states
            ),
            "datetime_source_count": len(
                state.datetime_states
            ),
            "text_source_count": len(
                state.text_states
            ),
            "timedelta_source_count": len(
                state.timedelta_output_columns
            ),
            "unknown_category_count": (
                state
                .total_unknown_category_count
            ),
            "missing_output_count": (
                state.total_missing_output_count
            ),
            "nonfinite_output_count": (
                state
                .total_nonfinite_output_count
            ),
        }

        report["output_columns"] = list(
            state.output_feature_columns
        )

        if self.options.include_feature_states:
            report["numeric_states"] = (
                self._numeric_states(
                    state
                )
            )

            report["categorical_states"] = (
                self._categorical_states(
                    state
                )
            )

            report["boolean_states"] = {
                column: value.model_dump(
                    mode="json"
                )
                for column, value
                in state.boolean_states.items()
            }

            report["datetime_states"] = {
                column: value.model_dump(
                    mode="json"
                )
                for column, value
                in state.datetime_states.items()
            }

            report["text_states"] = {
                column: value.model_dump(
                    mode="json"
                )
                for column, value
                in state.text_states.items()
            }

            report["timedelta_outputs"] = dict(
                state.timedelta_output_columns
            )

        if (
            self.options.include_feature_lineage
            and state.lineage is not None
        ):
            report["feature_lineage"] = (
                state.lineage.model_dump(
                    mode="json"
                )
            )

        if self.options.include_fingerprints:
            report["fingerprints"] = {
                "phase5_output": (
                    state.phase5_output_fingerprint
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
                    .transformation_plan_fingerprint
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
        result: Phase6Result,
        state: Phase6State,
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
        result: Phase6Result,
        state: Phase6State,
    ) -> str:
        """Build a readable Phase 6 text report."""

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
            "Phase 6 Report",
            "==============",
            f"Status: {status}",
            f"Ready for Phase 7: {ready}",
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
                "Source features: "
                f"{len(state.source_feature_columns)}"
            ),
            (
                "Output features: "
                f"{len(state.output_feature_columns)}"
            ),
            "",
            "Transformation summary",
            "----------------------",
            (
                "Numeric sources: "
                f"{len(state.numeric_states)}"
            ),
            (
                "Categorical sources: "
                f"{len(state.categorical_states)}"
            ),
            (
                "Boolean sources: "
                f"{len(state.boolean_states)}"
            ),
            (
                "Datetime sources: "
                f"{len(state.datetime_states)}"
            ),
            (
                "Text sources: "
                f"{len(state.text_states)}"
            ),
            (
                "Timedelta sources: "
                f"{len(state.timedelta_output_columns)}"
            ),
            (
                "Unknown categories: "
                f"{state.total_unknown_category_count}"
            ),
            (
                "Missing output values: "
                f"{state.total_missing_output_count}"
            ),
            (
                "Nonfinite output values: "
                f"{state.total_nonfinite_output_count}"
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
            (
                "Numeric integrity valid: "
                f"{state.numeric_integrity_valid}"
            ),
            (
                "Unknown-category policy valid: "
                f"{state.unknown_categories_valid}"
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

        return "\n".join(
            lines
        )

    def save_json(
        self,
        *,
        path: str | Path,
        result: Phase6Result,
        state: Phase6State,
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
        result: Phase6Result,
        state: Phase6State,
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

    def _numeric_states(
        self,
        state: Phase6State,
    ) -> dict[str, Any]:
        """Build report-safe numeric states."""

        values = {
            column: item.model_dump(
                mode="json"
            )
            for column, item
            in state.numeric_states.items()
        }

        if self.options.expose_numeric_parameters:
            return values

        return {
            column: self._redact_keys(
                value,
                keys={
                    "learned_parameters",
                },
            )
            for column, value
            in values.items()
        }

    def _categorical_states(
        self,
        state: Phase6State,
    ) -> dict[str, Any]:
        """Build report-safe categorical states."""

        values = {
            column: item.model_dump(
                mode="json"
            )
            for column, item
            in state.categorical_states.items()
        }

        if self.options.expose_category_vocabularies:
            return values

        sensitive_keys = {
            "categories",
            "encoded_categories",
            "category_mapping",
            "mapping",
            "retained_categories",
            "rare_categories",
            "category_counts",
            "category_order",
        }

        return {
            column: self._redact_keys(
                value,
                keys=sensitive_keys,
            )
            for column, value
            in values.items()
        }

    @classmethod
    def _redact_keys(
        cls,
        value: Any,
        *,
        keys: set[str],
    ) -> Any:
        """Recursively redact selected dictionary keys."""

        if isinstance(
            value,
            dict,
        ):
            return {
                key: (
                    "<redacted>"
                    if key in keys
                    else cls._redact_keys(
                        item,
                        keys=keys,
                    )
                )
                for key, item in value.items()
            }

        if isinstance(
            value,
            list,
        ):
            return [
                cls._redact_keys(
                    item,
                    keys=keys,
                )
                for item in value
            ]

        return value