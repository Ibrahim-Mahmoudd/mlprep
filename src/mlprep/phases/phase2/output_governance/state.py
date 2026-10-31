"""Persistent state produced during Phase 2."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from pydantic import Field, model_validator

from mlprep.phases.phase2.output_governance.metadata import (
    PartitionMetadata,
    PartitionName,
    Phase2Model,
    SplitBoundaries,
    SplitExecutionMetadata,
    SplitStrategyName,
)


class PartitionMembership(Phase2Model):
    """Persistent row membership for one partition."""

    partition: PartitionName

    row_ids: list[str | int] = Field(
        default_factory=list
    )

    row_count: int = Field(
        default=0,
        ge=0,
    )

    fingerprint: str | None = None

    @model_validator(mode="after")
    def validate_membership(
        self,
    ) -> PartitionMembership:
        """Validate membership consistency."""

        if self.row_count != len(self.row_ids):
            raise ValueError(
                f"Partition {self.partition.value!r} "
                f"declares {self.row_count} rows but "
                f"contains {len(self.row_ids)} row IDs."
            )

        if len(self.row_ids) != len(
            set(self.row_ids)
        ):
            raise ValueError(
                f"Partition {self.partition.value!r} "
                "contains duplicate row IDs."
            )

        return self


class SplitColumnState(Phase2Model):
    """Resolved Phase 2 column roles."""

    target_column: str

    feature_columns: list[str] = Field(
        default_factory=list
    )

    identifier_columns: list[str] = Field(
        default_factory=list
    )

    group_columns: list[str] = Field(
        default_factory=list
    )

    timestamp_column: str | None = None

    observation_key: list[str] = Field(
        default_factory=list
    )

    excluded_feature_columns: list[str] = Field(
        default_factory=list
    )

    internal_row_id_column: str = (
        "__mlprep_row_id__"
    )

    def all_required_columns(
        self,
    ) -> list[object]:
        """Return columns required to reproduce the split."""

        columns = [
            self.target_column,
            self.internal_row_id_column,
            *self.group_columns,
            *self.observation_key,
        ]

        if self.timestamp_column is not None:
            columns.append(
                self.timestamp_column
            )

        return list(
            dict.fromkeys(columns)
        )


class LeakageFirewallState(Phase2Model):
    """Persistent leakage-firewall state."""

    target_removed_from_features: bool = False

    row_overlap_checked: bool = False
    group_overlap_checked: bool = False
    duplicate_overlap_checked: bool = False
    temporal_order_checked: bool = False
    post_outcome_checked: bool = False
    preprocessing_leakage_checked: bool = False

    overlapping_row_ids: dict[
        str,
        list[str | int],
    ] = Field(default_factory=dict)

    overlapping_groups: dict[
        str,
        list[str | int],
    ] = Field(default_factory=dict)

    temporal_violations: list[
        dict[str, Any]
    ] = Field(default_factory=list)

    suspicious_features: list[str] = Field(
        default_factory=list
    )

    passed: bool = False

    metadata: dict[str, Any] = Field(
        default_factory=dict
    )


class TestLockState(Phase2Model):
    """Persistent state of the protected test partition."""

    locked: bool = False

    access_count: int = Field(
        default=0,
        ge=0,
    )

    access_reasons: list[str] = Field(
        default_factory=list
    )

    locked_at: datetime | None = None
    unlocked_at: datetime | None = None

    lock_token_hash: str | None = None

    def mark_locked(
        self,
        *,
        lock_token_hash: str | None = None,
    ) -> None:
        """Mark the test partition as locked."""

        self.locked = True
        self.locked_at = datetime.now(UTC)
        self.unlocked_at = None
        self.lock_token_hash = lock_token_hash

    def mark_unlocked(
        self,
    ) -> None:
        """Mark the test partition as unlocked."""

        self.locked = False
        self.unlocked_at = datetime.now(UTC)

    def mark_access(
        self,
        reason: str,
    ) -> None:
        """Record one authorized test access."""

        if not reason.strip():
            raise ValueError(
                "A test-access reason cannot be empty."
            )

        self.access_count += 1
        self.access_reasons.append(reason)


class Phase2State(Phase2Model):
    """Complete persistent state produced by Phase 2."""

    state_version: str = "0.1.0"

    created_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC)
    )

    strategy: SplitStrategyName

    columns: SplitColumnState

    execution_metadata: (
        SplitExecutionMetadata | None
    ) = None

    memberships: dict[
        PartitionName,
        PartitionMembership,
    ] = Field(default_factory=dict)

    partition_metadata: dict[
        PartitionName,
        PartitionMetadata,
    ] = Field(default_factory=dict)

    boundaries: SplitBoundaries = Field(
        default_factory=SplitBoundaries
    )

    leakage_firewall: LeakageFirewallState = Field(
        default_factory=LeakageFirewallState
    )

    test_lock: TestLockState = Field(
        default_factory=TestLockState
    )

    source_dataset_fingerprint: str | None = None

    train_fingerprint: str | None = None
    validation_fingerprint: str | None = None
    test_fingerprint: str | None = None

    split_fingerprint: str | None = None
    configuration_fingerprint: str | None = None

    ready_for_phase_3: bool = False

    metadata: dict[str, Any] = Field(
        default_factory=dict
    )

    def register_membership(
        self,
        membership: PartitionMembership,
    ) -> None:
        """Register membership for one partition."""

        self.memberships[
            membership.partition
        ] = membership

    def register_partition_metadata(
        self,
        metadata: PartitionMetadata,
    ) -> None:
        """Register metadata for one partition."""

        self.partition_metadata[
            metadata.name
        ] = metadata

    def get_membership(
        self,
        partition: PartitionName,
    ) -> PartitionMembership:
        """Return membership for one partition."""

        if partition not in self.memberships:
            raise KeyError(
                f"No membership is registered for "
                f"{partition.value!r}."
            )

        return self.memberships[partition]

    def get_row_ids(
        self,
        partition: PartitionName,
    ) -> list[str | int]:
        """Return row IDs assigned to one partition."""

        return list(
            self.get_membership(
                partition
            ).row_ids
        )

    def validate_disjoint_memberships(
        self,
    ) -> None:
        """Ensure partition memberships do not overlap."""

        partitions = list(
            self.memberships
        )

        for position, left_name in enumerate(
            partitions
        ):
            left_ids = set(
                self.memberships[
                    left_name
                ].row_ids
            )

            for right_name in partitions[
                position + 1:
            ]:
                right_ids = set(
                    self.memberships[
                        right_name
                    ].row_ids
                )

                overlap = left_ids.intersection(
                    right_ids
                )

                if overlap:
                    raise ValueError(
                        f"Partitions {left_name.value!r} "
                        f"and {right_name.value!r} overlap. "
                        f"Examples: {list(overlap)[:10]}"
                    )

    def all_assigned_row_ids(
        self,
    ) -> list[str | int]:
        """Return row IDs assigned to all partitions."""

        row_ids: list[str | int] = []

        for membership in self.memberships.values():
            row_ids.extend(
                membership.row_ids
            )

        return row_ids

    def get_partition_fingerprint(
        self,
        partition: PartitionName,
    ) -> str | None:
        """Return one partition fingerprint."""

        fingerprints = {
            PartitionName.TRAIN: (
                self.train_fingerprint
            ),
            PartitionName.VALIDATION: (
                self.validation_fingerprint
            ),
            PartitionName.TEST: (
                self.test_fingerprint
            ),
        }

        return fingerprints.get(partition)