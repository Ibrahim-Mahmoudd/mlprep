"""Fingerprint generation and verification for Phase 6."""

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
from mlprep.phases.phase6.contract.transformation_plan import (
    TransformationPlan,
)
from mlprep.phases.phase6.output_governance.metadata import (
    Phase6Finding,
    Phase6FindingAction,
    Phase6FindingCategory,
    Phase6FindingSeverity,
)
from mlprep.phases.phase6.output_governance.state import (
    Phase6State,
)


class Phase6FingerprintManager:
    """Generate and verify Phase 6 fingerprints."""

    component_name = "fingerprint"

    def capture_inputs(
        self,
        *,
        train: DatasetPartition,
        validation: DatasetPartition,
        test: DatasetPartition,
        state: Phase6State,
    ) -> None:
        """Capture fingerprints before transformation."""

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
        state: Phase6State,
    ) -> None:
        """Capture fingerprints after transformation."""

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

    def verify_protected_data(
        self,
        *,
        state: Phase6State,
    ) -> list[object]:
        """Verify targets and row IDs remain unchanged."""

        findings: list[
            Phase6Finding
        ] = []

        changed_targets = [
            partition
            for partition, source_fingerprint
            in state
            .source_target_fingerprints
            .items()
            if (
                state
                .output_target_fingerprints
                .get(partition)
                != source_fingerprint
            )
        ]

        state.targets_unchanged = (
            not changed_targets
        )

        if changed_targets:
            findings.append(
                Phase6Finding(
                    code="P6-FINGERPRINT-001",
                    category=(
                        Phase6FindingCategory
                        .TARGET_INTEGRITY
                    ),
                    severity=(
                        Phase6FindingSeverity.CRITICAL
                    ),
                    component=self.component_name,
                    message=(
                        "One or more target partitions changed "
                        "during Phase 6."
                    ),
                    partitions=[
                        partition.value
                        for partition
                        in changed_targets
                    ],
                    recommended_action=(
                        Phase6FindingAction
                        .RESTORE_INPUT
                    ),
                    applied_action=(
                        Phase6FindingAction.REPORT
                    ),
                    blocks_readiness=True,
                )
            )

        changed_row_ids = [
            partition
            for partition, source_fingerprint
            in state
            .source_row_id_fingerprints
            .items()
            if (
                state
                .output_row_id_fingerprints
                .get(partition)
                != source_fingerprint
            )
        ]

        state.row_ids_unchanged = (
            not changed_row_ids
        )

        if changed_row_ids:
            findings.append(
                Phase6Finding(
                    code="P6-FINGERPRINT-002",
                    category=(
                        Phase6FindingCategory
                        .ROW_INTEGRITY
                    ),
                    severity=(
                        Phase6FindingSeverity.CRITICAL
                    ),
                    component=self.component_name,
                    message=(
                        "One or more row-ID partitions changed "
                        "during Phase 6."
                    ),
                    partitions=[
                        partition.value
                        for partition
                        in changed_row_ids
                    ],
                    recommended_action=(
                        Phase6FindingAction
                        .RESTORE_INPUT
                    ),
                    applied_action=(
                        Phase6FindingAction.REPORT
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
        plan: TransformationPlan,
    ) -> str:
        """Generate a transformation-plan fingerprint."""

        return self.object_fingerprint(
            plan
        )

    def fitted_state_fingerprint(
        self,
        *,
        state: Phase6State,
    ) -> str:
        """Generate a fingerprint of fitted state."""

        payload = {
            "numeric": {
                column: value.model_dump(
                    mode="json"
                )
                for column, value
                in sorted(
                    state.numeric_states.items()
                )
            },
            "categorical": {
                column: value.model_dump(
                    mode="json"
                )
                for column, value
                in sorted(
                    state.categorical_states.items()
                )
            },
            "boolean": {
                column: value.model_dump(
                    mode="json"
                )
                for column, value
                in sorted(
                    state.boolean_states.items()
                )
            },
            "datetime": {
                column: value.model_dump(
                    mode="json"
                )
                for column, value
                in sorted(
                    state.datetime_states.items()
                )
            },
            "text": {
                column: value.model_dump(
                    mode="json"
                )
                for column, value
                in sorted(
                    state.text_states.items()
                )
            },
            "timedelta": dict(
                sorted(
                    state
                    .timedelta_output_columns
                    .items()
                )
            ),
            "lineage": (
                state.lineage.model_dump(
                    mode="json"
                )
                if state.lineage is not None
                else None
            ),
            "configuration": (
                state.configuration_fingerprint
            ),
            "plan": (
                state
                .transformation_plan_fingerprint
            ),
        }

        return self.object_fingerprint(
            payload
        )

    def output_fingerprint(
        self,
        *,
        state: Phase6State,
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
            "output_columns": (
                state.output_feature_columns
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
        """Generate one partition fingerprint."""

        digest = hashlib.sha256()

        metadata = {
            "partition": partition.name.value,
            "row_count": partition.row_count(),
            "feature_count": (
                partition.feature_count()
            ),
            "columns": partition.feature_names(),
            "dtypes": {
                str(column): str(
                    partition.X[
                        column
                    ].dtype
                )
                for column
                in partition.X.columns
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
        """Generate one Series fingerprint."""

        digest = hashlib.sha256()

        metadata = {
            "name": (
                None
                if series.name is None
                else str(series.name)
            ),
            "dtype": str(series.dtype),
            "row_count": len(series),
        }

        digest.update(
            cls._json_bytes(
                metadata
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