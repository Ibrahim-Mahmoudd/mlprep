"""Fingerprint generation and verification for Phase 8."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, is_dataclass
from datetime import date, datetime
from typing import Any

import numpy as np
import pandas as pd
from pydantic import BaseModel

from mlprep.phases.phase2.output_governance.metadata import (
    PartitionName,
)
from mlprep.phases.phase2.output_governance.result import (
    DatasetPartition,
)
from mlprep.phases.phase8.contract import (
    BalancingPlan,
)
from mlprep.phases.phase8.output_governance.metadata import (
    Phase8Finding,
    Phase8FindingAction,
    Phase8FindingCategory,
    Phase8FindingSeverity,
)
from mlprep.phases.phase8.output_governance.state import (
    Phase8State,
)
from mlprep.phases.phase8.serialization import (
    normalize_phase8_json_value,
)


class Phase8FingerprintManager:
    """Generate and verify Phase 8 fingerprints."""

    component_name = "fingerprint"

    def capture_inputs(
        self,
        *,
        train: DatasetPartition,
        validation: DatasetPartition,
        test: DatasetPartition,
        state: Phase8State,
    ) -> None:
        """Capture input partition fingerprints."""

        partitions = {
            PartitionName.TRAIN: train,
            PartitionName.VALIDATION: validation,
            PartitionName.TEST: test,
        }

        for name, partition in partitions.items():
            fingerprint = self.partition_fingerprint(
                partition
            )

            state.input_partition_fingerprints[
                name
            ] = fingerprint

            if name in {
                PartitionName.VALIDATION,
                PartitionName.TEST,
            }:
                state.protected_input_fingerprints[
                    name
                ] = fingerprint

    def capture_outputs(
        self,
        *,
        train: DatasetPartition,
        validation: DatasetPartition,
        test: DatasetPartition,
        state: Phase8State,
    ) -> None:
        """Capture output partition fingerprints."""

        partitions = {
            PartitionName.TRAIN: train,
            PartitionName.VALIDATION: validation,
            PartitionName.TEST: test,
        }

        for name, partition in partitions.items():
            fingerprint = self.partition_fingerprint(
                partition
            )

            state.output_partition_fingerprints[
                name
            ] = fingerprint

            if name in {
                PartitionName.VALIDATION,
                PartitionName.TEST,
            }:
                state.protected_output_fingerprints[
                    name
                ] = fingerprint

    def verify_protected_partitions(
        self,
        *,
        state: Phase8State,
    ) -> list[Phase8Finding]:
        """Verify validation and test fingerprints."""

        findings: list[
            Phase8Finding
        ] = []

        for partition in (
            PartitionName.VALIDATION,
            PartitionName.TEST,
        ):
            source = (
                state
                .protected_input_fingerprints
                .get(
                    partition
                )
            )

            output = (
                state
                .protected_output_fingerprints
                .get(
                    partition
                )
            )

            if source == output and source is not None:
                continue

            findings.append(
                Phase8Finding(
                    code=(
                        "P8-FINGERPRINT-001"
                        if partition
                        == PartitionName.VALIDATION
                        else "P8-FINGERPRINT-002"
                    ),
                    category=(
                        Phase8FindingCategory
                        .PARTITION_INTEGRITY
                    ),
                    severity=(
                        Phase8FindingSeverity.CRITICAL
                    ),
                    component=self.component_name,
                    message=(
                        f"The protected {partition.value} "
                        "partition changed during Phase 8."
                    ),
                    partitions=[
                        partition.value
                    ],
                    recommended_action=(
                        Phase8FindingAction.RESTORE_INPUT
                    ),
                    applied_action=(
                        Phase8FindingAction.REPORT
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
        plan: BalancingPlan,
    ) -> str:
        """Generate a balancing-plan fingerprint."""

        return self.object_fingerprint(
            plan
        )

    def fitted_state_fingerprint(
        self,
        *,
        state: Phase8State,
    ) -> str:
        """Generate a fitted-state fingerprint."""

        payload = {
            "configured_strategy": (
                state.configured_strategy
            ),
            "executed_strategy": (
                state.executed_strategy
            ),
            "source_rows": (
                state.source_training_row_count
            ),
            "output_rows": (
                state.output_training_row_count
            ),
            "source_features": (
                state.source_feature_columns
            ),
            "distribution_before": (
                self._optional_model(
                    state.source_class_distribution
                )
            ),
            "distribution_after": (
                self._optional_model(
                    state.output_class_distribution
                )
            ),
            "metrics_before": (
                self._optional_model(
                    state.imbalance_metrics_before
                )
            ),
            "metrics_after": (
                self._optional_model(
                    state.imbalance_metrics_after
                )
            ),
            "sampler_state": (
                self._optional_model(
                    state.sampler_state
                )
            ),
            "class_weights": (
                self._optional_model(
                    state.class_weights
                )
            ),
            "sample_weights": (
                self._optional_model(
                    state.sample_weights
                )
            ),
            "row_lineage": (
                self._optional_model(
                    state.row_lineage
                )
            ),
            "configuration": (
                state.configuration_fingerprint
            ),
            "plan": (
                state.balancing_plan_fingerprint
            ),
        }

        return self.object_fingerprint(
            payload
        )

    def output_fingerprint(
        self,
        *,
        state: Phase8State,
    ) -> str:
        """Generate the combined Phase 8 output fingerprint."""

        payload = {
            "partitions": {
                partition.value: fingerprint
                for partition, fingerprint
                in sorted(
                    state
                    .output_partition_fingerprints
                    .items(),
                    key=lambda item: item[
                        0
                    ].value,
                )
            },
            "executed_strategy": (
                state.executed_strategy
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
    def object_fingerprint(
        cls,
        value: object,
    ) -> str:
        """Generate a deterministic object fingerprint."""

        return hashlib.sha256(
            cls._json_bytes(
                normalize_phase8_json_value(value)
            )
        ).hexdigest()

    @classmethod
    def serialize_model(
        cls,
        value: BaseModel,
    ) -> dict[str, Any]:
        """Return a JSON-compatible representation of a model."""

        serialized = normalize_phase8_json_value(value)
        if not isinstance(serialized, dict):
            raise TypeError("Expected a mapping while serializing model state.")
        return serialized

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
            return cls._serialize(
                value.model_dump(mode="python")
            )

        if (
            is_dataclass(value)
            and not isinstance(value, type)
        ):
            return {
                key: cls._serialize(
                    item
                )
                for key, item
                in asdict(
                    value
                ).items()
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

        if isinstance(value, pd.Series):
            return {
                "values": cls._serialize(value.tolist()),
                "index": cls._serialize(value.index.tolist()),
                "name": cls._serialize(value.name),
            }

        if isinstance(value, np.generic):
            return value.item()

        if isinstance(value, (datetime, date)):
            return value.isoformat()

        if isinstance(value, set):
            serialized_items = [
                cls._serialize(item)
                for item in value
            ]
            return sorted(
                serialized_items,
                key=lambda item: json.dumps(
                    item,
                    sort_keys=True,
                    default=str,
                ),
            )

        if isinstance(value, (list, tuple)):
            return [
                cls._serialize(
                    item
                )
                for item in value
            ]

        if hasattr(
            value,
            "value",
        ):
            return value.value

        return value

    @staticmethod
    def _optional_model(
        value: BaseModel | None,
    ) -> dict[str, Any] | None:
        """Serialize an optional Pydantic model."""

        if value is None:
            return None

        return Phase8FingerprintManager.serialize_model(value)

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
        ).encode(
            "utf-8"
        )
