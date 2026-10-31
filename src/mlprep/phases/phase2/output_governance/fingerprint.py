"""Partition fingerprint generation for Phase 2."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Literal

import pandas as pd

from mlprep.phases.phase2.output_governance.metadata import (
    PartitionName,
    SplitFindingAction,
    SplitFindingCategory,
    SplitFindingSeverity,
)
from mlprep.phases.phase2.output_governance.result import (
    DatasetPartition,
    SplitFinding,
)
from mlprep.phases.phase2.output_governance.state import (
    PartitionMembership,
    Phase2State,
)


FingerprintAlgorithm = Literal[
    "sha256",
    "sha512",
]


@dataclass(slots=True)
class Phase2FingerprintOptions:
    """Options controlling Phase 2 fingerprints."""

    enabled: bool = True
    algorithm: FingerprintAlgorithm = "sha256"

    include_feature_values: bool = True
    include_target_values: bool = True
    include_row_ids: bool = True


class Phase2FingerprintGenerator:
    """Generate fingerprints for partitions and split membership."""

    component_name = "fingerprint"

    def __init__(
        self,
        options: Phase2FingerprintOptions | None = None,
    ) -> None:
        """Initialize the fingerprint generator."""

        self.options = (
            options
            or Phase2FingerprintOptions()
        )

    def generate(
        self,
        *,
        train: DatasetPartition,
        validation: DatasetPartition,
        test: DatasetPartition,
        state: Phase2State,
    ) -> list[object]:
        """Generate all Phase 2 fingerprints."""

        if not self.options.enabled:
            return []

        partitions = {
            PartitionName.TRAIN: train,
            PartitionName.VALIDATION: validation,
            PartitionName.TEST: test,
        }

        fingerprints: dict[
            PartitionName,
            str,
        ] = {}

        for name, partition in partitions.items():
            dataset_fingerprint = (
                self._partition_fingerprint(
                    partition
                )
            )

            feature_fingerprint = (
                self._frame_fingerprint(
                    partition.X
                )
            )

            target_fingerprint = (
                self._series_fingerprint(
                    partition.y
                )
            )

            partition.metadata.dataset_fingerprint = (
                dataset_fingerprint
            )

            partition.metadata.feature_fingerprint = (
                feature_fingerprint
            )

            partition.metadata.target_fingerprint = (
                target_fingerprint
            )

            state.register_partition_metadata(
                partition.metadata
            )

            membership = PartitionMembership(
                partition=name,
                row_ids=[
                    self._safe_identifier(value)
                    for value
                    in partition.row_ids.tolist()
                ],
                row_count=len(partition.row_ids),
                fingerprint=dataset_fingerprint,
            )

            state.register_membership(
                membership
            )

            fingerprints[name] = (
                dataset_fingerprint
            )

        state.train_fingerprint = fingerprints[
            PartitionName.TRAIN
        ]

        state.validation_fingerprint = fingerprints[
            PartitionName.VALIDATION
        ]

        state.test_fingerprint = fingerprints[
            PartitionName.TEST
        ]

        state.split_fingerprint = (
            self._split_fingerprint(
                state=state
            )
        )

        state.validate_disjoint_memberships()

        return [
            SplitFinding(
                code="P2-FINGERPRINT-001",
                category=(
                    SplitFindingCategory.FINGERPRINT
                ),
                severity=SplitFindingSeverity.INFO,
                component=self.component_name,
                message=(
                    "Partition and split fingerprints were "
                    "generated successfully."
                ),
                recommended_action=(
                    SplitFindingAction.REPORT
                ),
                applied_action=(
                    SplitFindingAction.REPORT
                ),
                blocks_readiness=False,
                details={
                    "algorithm": (
                        self.options.algorithm
                    ),
                    "train_fingerprint": (
                        state.train_fingerprint
                    ),
                    "validation_fingerprint": (
                        state.validation_fingerprint
                    ),
                    "test_fingerprint": (
                        state.test_fingerprint
                    ),
                    "split_fingerprint": (
                        state.split_fingerprint
                    ),
                },
            )
        ]

    def _partition_fingerprint(
        self,
        partition: DatasetPartition,
    ) -> str:
        """Generate one complete partition fingerprint."""

        digest = self._new_digest()

        metadata = {
            "name": partition.name.value,
            "row_count": partition.row_count(),
            "feature_names": (
                partition.feature_names()
            ),
            "feature_dtypes": {
                str(column): str(
                    partition.X[column].dtype
                )
                for column in partition.X.columns
            },
            "target_dtype": str(
                partition.y.dtype
            ),
        }

        digest.update(
            self._stable_json(metadata)
        )

        if self.options.include_feature_values:
            digest.update(
                self._frame_hash_bytes(
                    partition.X
                )
            )

        if self.options.include_target_values:
            digest.update(
                self._series_hash_bytes(
                    partition.y
                )
            )

        if self.options.include_row_ids:
            digest.update(
                self._series_hash_bytes(
                    partition.row_ids
                )
            )

        return digest.hexdigest()

    def _frame_fingerprint(
        self,
        frame: pd.DataFrame,
    ) -> str:
        """Generate a feature-frame fingerprint."""

        digest = self._new_digest()

        digest.update(
            self._frame_hash_bytes(frame)
        )

        return digest.hexdigest()

    def _series_fingerprint(
        self,
        series: pd.Series,
    ) -> str:
        """Generate a Series fingerprint."""

        digest = self._new_digest()

        digest.update(
            self._series_hash_bytes(series)
        )

        return digest.hexdigest()

    def _split_fingerprint(
        self,
        *,
        state: Phase2State,
    ) -> str:
        """Generate a fingerprint for full split membership."""

        payload = {
            partition.value: (
                state.get_row_ids(partition)
            )
            for partition in (
                PartitionName.TRAIN,
                PartitionName.VALIDATION,
                PartitionName.TEST,
            )
        }

        digest = self._new_digest()

        digest.update(
            self._stable_json(payload)
        )

        return digest.hexdigest()

    @staticmethod
    def _frame_hash_bytes(
        frame: pd.DataFrame,
    ) -> bytes:
        """Return deterministic DataFrame hash bytes."""

        hashes = pd.util.hash_pandas_object(
            frame,
            index=True,
            categorize=True,
        )

        return hashes.to_numpy().tobytes()

    @staticmethod
    def _series_hash_bytes(
        series: pd.Series,
    ) -> bytes:
        """Return deterministic Series hash bytes."""

        hashes = pd.util.hash_pandas_object(
            series,
            index=True,
            categorize=True,
        )

        return hashes.to_numpy().tobytes()

    def _new_digest(
        self,
    ) -> Any:
        """Create a configured digest."""

        return hashlib.new(
            self.options.algorithm
        )

    @staticmethod
    def _stable_json(
        value: object,
    ) -> bytes:
        """Serialize a value deterministically."""

        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")

    @staticmethod
    def _safe_identifier(
        value: object,
    ) -> str | int:
        """Convert a row ID into a supported value."""

        if isinstance(value, (str, int)):
            return value

        return str(value)