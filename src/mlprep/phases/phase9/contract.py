"""Validation contract for the Phase 9 model handoff."""

from __future__ import annotations

from mlprep.phases.phase2.output_governance.metadata import PartitionName
from mlprep.phases.phase8.output_governance.fingerprint import (
    Phase8FingerprintManager,
)
from mlprep.phases.phase8.output_governance.result import Phase8Result
from mlprep.phases.phase8.output_governance.state import Phase8State


class Phase9ContractResolver:
    """Validate Phase 8 readiness and handoff structure."""

    @staticmethod
    def resolve(
        *,
        phase8_result: Phase8Result,
        phase8_state: Phase8State,
        require_disjoint_rows: bool = True,
    ) -> tuple[str, list[str], dict[str, str]]:
        """Validate the source and return target and schema declarations."""

        decision = phase8_result.readiness
        if decision is None:
            raise ValueError("Phase 8 has no readiness decision.")
        if not decision.ready:
            raise PermissionError(
                "Phase 8 is not ready for the model handoff."
            )
        if not phase8_state.ready_for_phase_9:
            raise PermissionError(
                "Phase 8 state does not authorize Phase 9."
            )
        if not phase8_state.fitted_on_training_only:
            raise ValueError(
                "Phase 8 state was not fitted on training data only."
            )
        if not phase8_state.validation_remained_protected:
            raise ValueError("The validation partition was not protected.")
        if not phase8_state.test_remained_protected:
            raise ValueError("The test partition was not protected.")
        if not phase8_state.output_fingerprint:
            raise ValueError("Phase 8 output fingerprint is missing.")
        if not phase8_state.fitted_state_fingerprint:
            raise ValueError("Phase 8 fitted-state fingerprint is missing.")

        partitions = (
            phase8_result.train,
            phase8_result.validation,
            phase8_result.test,
        )
        expected_names = (
            PartitionName.TRAIN,
            PartitionName.VALIDATION,
            PartitionName.TEST,
        )
        for partition, expected_name in zip(
            partitions, expected_names, strict=True
        ):
            if partition.name != expected_name:
                raise ValueError(
                    f"Expected the {expected_name.value} partition, "
                    f"received {partition.name.value}."
                )

        fingerprint_manager = Phase8FingerprintManager()
        for partition, partition_name in zip(
            partitions, expected_names, strict=True
        ):
            expected_fingerprint = (
                phase8_state.output_partition_fingerprints.get(
                    partition_name
                )
            )
            if expected_fingerprint is None:
                raise ValueError(
                    f"Phase 8 {partition_name.value} output fingerprint "
                    "is missing."
                )

            actual_fingerprint = (
                fingerprint_manager.partition_fingerprint(partition)
            )
            if actual_fingerprint != expected_fingerprint:
                raise ValueError(
                    f"The {partition_name.value} partition changed after "
                    "Phase 8."
                )

        features = [str(column) for column in phase8_result.train.X.columns]
        if len(features) != len(set(features)):
            raise ValueError("Training feature names are not unique.")
        if not features:
            raise ValueError("The handoff has no model features.")

        dtypes = {
            str(column): str(dtype)
            for column, dtype in phase8_result.train.X.dtypes.items()
        }
        target_names = {
            str(partition.y.name)
            for partition in partitions
            if partition.y.name is not None
        }
        if len(target_names) != 1 or any(
            partition.y.name is None for partition in partitions
        ):
            raise ValueError(
                "Train, validation, and test targets must have the same "
                "non-empty column name."
            )

        for partition in partitions[1:]:
            if [str(column) for column in partition.X.columns] != features:
                raise ValueError(
                    f"The {partition.name.value} feature schema differs "
                    "from training."
                )
            partition_dtypes = {
                str(column): str(dtype)
                for column, dtype in partition.X.dtypes.items()
            }
            if partition_dtypes != dtypes:
                raise ValueError(
                    f"The {partition.name.value} feature dtypes differ "
                    "from training."
                )

        if require_disjoint_rows:
            seen: set[object] = set()
            for partition in partitions:
                row_ids = set(partition.row_ids.tolist())
                overlap = seen.intersection(row_ids)
                if overlap:
                    raise ValueError(
                        "Partition row IDs overlap across the model "
                        "handoff."
                    )
                seen.update(row_ids)

        target_column = next(iter(target_names))
        if target_column in features:
            raise ValueError("The target is incorrectly included as a feature.")

        return target_column, features, dtypes
