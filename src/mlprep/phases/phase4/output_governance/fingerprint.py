"""Fingerprint generation and verification for Phase 4."""

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
from mlprep.phases.phase4.contract.imputation_plan import (
    ImputationPlan,
)
from mlprep.phases.phase4.output_governance.metadata import (
    Phase4FindingAction,
    Phase4FindingCategory,
    Phase4FindingSeverity,
)
from mlprep.phases.phase4.output_governance.result import (
    Phase4Finding,
)
from mlprep.phases.phase4.output_governance.state import (
    Phase4State,
)


class Phase4FingerprintManager:
    """Generate and validate Phase 4 fingerprints."""

    component_name = "fingerprint"

    def capture_inputs(
        self,
        *,
        train: DatasetPartition,
        validation: DatasetPartition,
        test: DatasetPartition,
        state: Phase4State,
    ) -> None:
        """Capture fingerprints before Phase 4 transformation."""

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
        state: Phase4State,
    ) -> None:
        """Capture fingerprints after Phase 4 transformation."""

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
        state: Phase4State,
    ) -> list[Phase4Finding]:
        """Verify that targets and row IDs remain unchanged."""

        findings: list[Phase4Finding] = []

        target_partitions = set(
            state.source_target_fingerprints
        ).union(
            state.output_target_fingerprints
        )

        changed_targets: list[
            PartitionName
        ] = []

        for partition in target_partitions:
            before = (
                state.source_target_fingerprints
                .get(partition)
            )

            after = (
                state.output_target_fingerprints
                .get(partition)
            )

            if before != after:
                changed_targets.append(
                    partition
                )

        state.targets_unchanged = (
            not changed_targets
        )

        if changed_targets:
            findings.append(
                Phase4Finding(
                    code="P4-FINGERPRINT-001",
                    category=(
                        Phase4FindingCategory
                        .TARGET_INTEGRITY
                    ),
                    severity=(
                        Phase4FindingSeverity.CRITICAL
                    ),
                    component=self.component_name,
                    message=(
                        "One or more target partitions changed "
                        "during Phase 4."
                    ),
                    partitions=[
                        partition.value
                        for partition
                        in changed_targets
                    ],
                    recommended_action=(
                        Phase4FindingAction
                        .RESTORE_INPUT
                    ),
                    applied_action=(
                        Phase4FindingAction.REPORT
                    ),
                    blocks_readiness=True,
                )
            )

        row_id_partitions = set(
            state.source_row_id_fingerprints
        ).union(
            state.output_row_id_fingerprints
        )

        changed_row_ids: list[
            PartitionName
        ] = []

        for partition in row_id_partitions:
            before = (
                state.source_row_id_fingerprints
                .get(partition)
            )

            after = (
                state.output_row_id_fingerprints
                .get(partition)
            )

            if before != after:
                changed_row_ids.append(
                    partition
                )

        state.row_ids_unchanged = (
            not changed_row_ids
        )

        if changed_row_ids:
            findings.append(
                Phase4Finding(
                    code="P4-FINGERPRINT-002",
                    category=(
                        Phase4FindingCategory
                        .ROW_INTEGRITY
                    ),
                    severity=(
                        Phase4FindingSeverity.CRITICAL
                    ),
                    component=self.component_name,
                    message=(
                        "One or more row-ID partitions changed "
                        "during Phase 4."
                    ),
                    partitions=[
                        partition.value
                        for partition in changed_row_ids
                    ],
                    recommended_action=(
                        Phase4FindingAction
                        .RESTORE_INPUT
                    ),
                    applied_action=(
                        Phase4FindingAction.REPORT
                    ),
                    blocks_readiness=True,
                )
            )

        return findings

    def configuration_fingerprint(
        self,
        configuration: object,
    ) -> str:
        """Generate a deterministic configuration fingerprint."""

        return self.object_fingerprint(
            configuration
        )

    def plan_fingerprint(
        self,
        plan: ImputationPlan,
    ) -> str:
        """Generate an imputation-plan fingerprint."""

        return self.object_fingerprint(
            plan
        )

    def fitted_state_fingerprint(
        self,
        *,
        state: Phase4State,
    ) -> str:
        """Generate a fingerprint of fitted imputation state."""

        payload = {
            "imputers": {
                column: imputer_state.model_dump(
                    mode="json"
                )
                for column, imputer_state
                in sorted(
                    state.imputer_states.items()
                )
            },
            "indicator_mapping": dict(
                sorted(
                    state.indicator_mapping.items()
                )
            ),
            "plan_fingerprint": (
                state.imputation_plan_fingerprint
            ),
            "configuration_fingerprint": (
                state.configuration_fingerprint
            ),
        }

        return self.object_fingerprint(
            payload
        )

    def output_fingerprint(
        self,
        *,
        state: Phase4State,
    ) -> str:
        """Generate a combined transformed-output fingerprint."""

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
            "fitted_state": (
                state.fitted_state_fingerprint
            ),
            "output_features": (
                state.output_feature_columns
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
        """Generate a fingerprint for one complete partition."""

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
                    partition.X[column].dtype
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
        """Generate a fingerprint for one Series."""

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
        """Generate a deterministic fingerprint for an object."""

        serialized = cls._serialize(
            value
        )

        return hashlib.sha256(
            cls._json_bytes(
                serialized
            )
        ).hexdigest()

    @staticmethod
    def _json_bytes(
        value: object,
    ) -> bytes:
        """Serialize a value into stable JSON bytes."""

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

        if isinstance(value, BaseModel):
            return value.model_dump(
                mode="json"
            )

        if is_dataclass(value):
            return {
                key: cls._serialize(item)
                for key, item
                in asdict(value).items()
            }

        if isinstance(value, dict):
            return {
                str(key): cls._serialize(item)
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

        if hasattr(value, "value"):
            return value.value

        return value
