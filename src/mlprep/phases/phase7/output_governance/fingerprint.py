"""Fingerprint generation and verification for Phase 7."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, is_dataclass
from typing import Any

import pandas as pd
from pydantic import BaseModel

from mlprep.phases.phase2.output_governance.metadata import (
    PartitionName,
)
from mlprep.phases.phase2.output_governance.result import (
    DatasetPartition,
)
from mlprep.phases.phase7.contract.selection_plan import (
    SelectionPlan,
)
from mlprep.phases.phase7.output_governance.metadata import (
    Phase7Finding,
    Phase7FindingAction,
    Phase7FindingCategory,
    Phase7FindingSeverity,
)
from mlprep.phases.phase7.output_governance.state import (
    Phase7State,
)


class Phase7FingerprintManager:
    """Generate and verify Phase 7 fingerprints."""

    component_name = "fingerprint"

    def capture_inputs(
        self,
        *,
        train: DatasetPartition,
        validation: DatasetPartition,
        test: DatasetPartition,
        state: Phase7State,
    ) -> None:
        """Capture fingerprints before selection."""

        partitions = {
            PartitionName.TRAIN: train,
            PartitionName.VALIDATION: validation,
            PartitionName.TEST: test,
        }

        for name, partition in partitions.items():
            state.input_partition_fingerprints[
                name
            ] = self.partition_fingerprint(
                partition
            )

            state.source_target_fingerprints[
                name
            ] = self.series_fingerprint(
                partition.y
            )

            state.source_row_id_fingerprints[
                name
            ] = self.series_fingerprint(
                partition.row_ids
            )

    def capture_outputs(
        self,
        *,
        train: DatasetPartition,
        validation: DatasetPartition,
        test: DatasetPartition,
        state: Phase7State,
    ) -> None:
        """Capture fingerprints after selection."""

        partitions = {
            PartitionName.TRAIN: train,
            PartitionName.VALIDATION: validation,
            PartitionName.TEST: test,
        }

        for name, partition in partitions.items():
            state.output_partition_fingerprints[
                name
            ] = self.partition_fingerprint(
                partition
            )

            state.output_target_fingerprints[
                name
            ] = self.series_fingerprint(
                partition.y
            )

            state.output_row_id_fingerprints[
                name
            ] = self.series_fingerprint(
                partition.row_ids
            )

            state.selected_feature_value_fingerprints[
                name
            ] = {
                str(column): self.series_fingerprint(
                    partition.X[column]
                )
                for column in partition.X.columns
            }

    def verify_protected_data(
        self,
        *,
        source_partitions: dict[
            PartitionName,
            DatasetPartition,
        ],
        output_partitions: dict[
            PartitionName,
            DatasetPartition,
        ],
        state: Phase7State,
    ) -> list[object]:
        """Verify targets, row IDs, and selected values."""

        findings: list[
            Phase7Finding
        ] = []

        changed_targets = [
            partition
            for partition, fingerprint
            in state.source_target_fingerprints.items()
            if (
                state.output_target_fingerprints.get(
                    partition
                )
                != fingerprint
            )
        ]

        state.targets_unchanged = (
            not changed_targets
        )

        if changed_targets:
            findings.append(
                self._critical_finding(
                    code="P7-FINGERPRINT-001",
                    category=(
                        Phase7FindingCategory
                        .TARGET_INTEGRITY
                    ),
                    message=(
                        "One or more target partitions changed "
                        "during Phase 7."
                    ),
                    partitions=changed_targets,
                )
            )

        changed_row_ids = [
            partition
            for partition, fingerprint
            in state.source_row_id_fingerprints.items()
            if (
                state.output_row_id_fingerprints.get(
                    partition
                )
                != fingerprint
            )
        ]

        state.row_ids_unchanged = (
            not changed_row_ids
        )

        if changed_row_ids:
            findings.append(
                self._critical_finding(
                    code="P7-FINGERPRINT-002",
                    category=(
                        Phase7FindingCategory
                        .ROW_INTEGRITY
                    ),
                    message=(
                        "One or more row-ID partitions changed "
                        "during Phase 7."
                    ),
                    partitions=changed_row_ids,
                )
            )

        changed_values: dict[
            PartitionName,
            list[str],
        ] = {}

        for partition_name, output in (
            output_partitions.items()
        ):
            source = source_partitions[
                partition_name
            ]

            changed_columns = [
                column
                for column
                in state.selected_feature_columns
                if (
                    column not in source.X.columns
                    or column not in output.X.columns
                    or not source.X[
                        column
                    ].equals(
                        output.X[
                            column
                        ]
                    )
                )
            ]

            if changed_columns:
                changed_values[
                    partition_name
                ] = changed_columns

        state.selected_values_unchanged = (
            not changed_values
        )

        if changed_values:
            findings.append(
                Phase7Finding(
                    code="P7-FINGERPRINT-003",
                    category=(
                        Phase7FindingCategory.FINGERPRINT
                    ),
                    severity=(
                        Phase7FindingSeverity.CRITICAL
                    ),
                    component=self.component_name,
                    message=(
                        "One or more selected feature values "
                        "changed during Phase 7."
                    ),
                    columns=sorted(
                        {
                            column
                            for columns
                            in changed_values.values()
                            for column in columns
                        }
                    ),
                    partitions=[
                        partition.value
                        for partition
                        in changed_values
                    ],
                    recommended_action=(
                        Phase7FindingAction.RESTORE_INPUT
                    ),
                    applied_action=(
                        Phase7FindingAction.REPORT
                    ),
                    blocks_readiness=True,
                )
            )

        return findings

    def configuration_fingerprint(
        self,
        configuration: object,
    ) -> str:
        """Generate a configuration fingerprint."""

        return self.object_fingerprint(
            configuration
        )

    def plan_fingerprint(
        self,
        plan: SelectionPlan,
    ) -> str:
        """Generate a selection-plan fingerprint."""

        return self.object_fingerprint(
            plan
        )

    def fitted_state_fingerprint(
        self,
        *,
        state: Phase7State,
    ) -> str:
        """Generate a fitted-state fingerprint."""

        selected_lineage = (
            state.selected_lineage.model_dump(mode="json")
            if state.selected_lineage is not None
            else None
        )
        if selected_lineage is not None:
            selected_lineage.pop("created_at", None)

        payload = {
            "source_features": (
                state.source_feature_columns
            ),
            "selected_features": (
                state.selected_feature_columns
            ),
            "removed_features": (
                state.removed_feature_columns
            ),
            "required_features": (
                state.required_features
            ),
            "protected_features": (
                state.protected_features
            ),
            "prohibited_features": (
                state.prohibited_features
            ),
            "decisions": {
                column: decision.model_dump(
                    mode="json"
                )
                for column, decision
                in sorted(
                    state.decisions.items()
                )
            },
            "filter_result": (
                state.filter_result.model_dump(
                    mode="json"
                )
                if state.filter_result is not None
                else None
            ),
            "selector_result": (
                state.selector_result.model_dump(
                    mode="json"
                )
                if state.selector_result is not None
                else None
            ),
            "stability_result": (
                state.stability_result.model_dump(
                    mode="json"
                )
                if state.stability_result is not None
                else None
            ),
            "selected_lineage": selected_lineage,
            "configuration": (
                state.configuration_fingerprint
            ),
            "plan": (
                state.selection_plan_fingerprint
            ),
        }

        return self.object_fingerprint(
            payload
        )

    def output_fingerprint(
        self,
        *,
        state: Phase7State,
    ) -> str:
        """Generate a combined output fingerprint."""

        payload = {
            "partitions": {
                partition.value: fingerprint
                for partition, fingerprint
                in sorted(
                    state
                    .output_partition_fingerprints
                    .items(),
                    key=lambda item: item[0].value,
                )
            },
            "selected_features": (
                state.selected_feature_columns
            ),
            "fitted_state": (
                state.fitted_state_fingerprint
            ),
        }

        return self.object_fingerprint(
            payload
        )

    @classmethod
    def partition_fingerprint(
        cls,
        partition: DatasetPartition,
    ) -> str:
        """Generate a deterministic partition fingerprint."""

        digest = hashlib.sha256()

        metadata = {
            "partition": partition.name.value,
            "rows": partition.row_count(),
            "features": partition.feature_count(),
            "columns": partition.feature_names(),
            "dtypes": {
                str(column): str(
                    partition.X[column].dtype
                )
                for column in partition.X.columns
            },
            "target_dtype": str(
                partition.y.dtype
            ),
            "row_id_dtype": str(
                partition.row_ids.dtype
            ),
        }

        digest.update(
            cls._json_bytes(
                metadata
            )
        )

        for value in (
            partition.X,
            partition.y,
            partition.row_ids,
        ):
            hashes = pd.util.hash_pandas_object(
                value,
                index=True,
                categorize=True,
            )

            digest.update(
                hashes.to_numpy().tobytes()
            )

        return digest.hexdigest()

    @classmethod
    def series_fingerprint(
        cls,
        series: pd.Series,
    ) -> str:
        """Generate a deterministic Series fingerprint."""

        digest = hashlib.sha256()

        digest.update(
            cls._json_bytes(
                {
                    "name": (
                        None
                        if series.name is None
                        else str(series.name)
                    ),
                    "dtype": str(series.dtype),
                    "rows": len(series),
                }
            )
        )

        hashes = pd.util.hash_pandas_object(
            series,
            index=True,
            categorize=True,
        )

        digest.update(
            hashes.to_numpy().tobytes()
        )

        return digest.hexdigest()

    @classmethod
    def object_fingerprint(
        cls,
        value: object,
    ) -> str:
        """Generate a deterministic object fingerprint."""

        return hashlib.sha256(
            cls._json_bytes(
                cls._serialize(
                    value
                )
            )
        ).hexdigest()

    @staticmethod
    def _json_bytes(
        value: object,
    ) -> bytes:
        """Serialize stable JSON bytes."""

        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")

    @classmethod
    def _serialize(
        cls,
        value: Any,
    ) -> Any:
        """Recursively serialize an object."""

        if isinstance(
            value,
            BaseModel,
        ):
            return value.model_dump(
                mode="json"
            )

        if is_dataclass(
            value
        ):
            return {
                key: cls._serialize(
                    item
                )
                for key, item
                in asdict(value).items()
            }

        if isinstance(
            value,
            dict,
        ):
            return {
                str(key): cls._serialize(
                    item
                )
                for key, item
                in value.items()
            }

        if isinstance(
            value,
            (list, tuple, set),
        ):
            return [
                cls._serialize(item)
                for item in value
            ]

        if hasattr(
            value,
            "value",
        ):
            return value.value

        return value

    def _critical_finding(
        self,
        *,
        code: str,
        category: Phase7FindingCategory,
        message: str,
        partitions: list[PartitionName],
    ) -> Phase7Finding:
        """Build one critical fingerprint finding."""

        return Phase7Finding(
            code=code,
            category=category,
            severity=(
                Phase7FindingSeverity.CRITICAL
            ),
            component=self.component_name,
            message=message,
            partitions=[
                partition.value
                for partition in partitions
            ],
            recommended_action=(
                Phase7FindingAction.RESTORE_INPUT
            ),
            applied_action=(
                Phase7FindingAction.REPORT
            ),
            blocks_readiness=True,
        )
