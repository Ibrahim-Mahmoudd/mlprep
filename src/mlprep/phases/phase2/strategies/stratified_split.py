"""Stratified classification splitting for Phase 2."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from mlprep.phases.phase2.contract.split_contract import (
    SplitContract,
)
from mlprep.phases.phase2.output_governance.metadata import (
    PartitionName,
    SplitFindingAction,
    SplitFindingCategory,
    SplitFindingSeverity,
    SplitRatios,
    SplitStrategyName,
)
from mlprep.phases.phase2.output_governance.result import (
    SplitFinding,
)
from mlprep.phases.phase2.strategies.base import (
    BaseSplitStrategy,
    StrategyPartitions,
)


@dataclass(slots=True)
class StratifiedSplitOptions:
    """Options controlling classification-stratified splitting."""

    ratios: SplitRatios
    random_seed: int = 42

    shuffle_within_classes: bool = True

    minimum_class_count_per_partition: int = 1

    enforce_all_classes_in_train: bool = True
    enforce_all_classes_in_validation: bool = True
    enforce_all_classes_in_test: bool = True

    distribution_tolerance: float = 0.05

    def __post_init__(self) -> None:
        """Validate stratified-split options."""

        self.ratios.validate_total()

        if self.minimum_class_count_per_partition < 0:
            raise ValueError(
                "minimum_class_count_per_partition "
                "cannot be negative."
            )

        if not 0.0 <= self.distribution_tolerance <= 1.0:
            raise ValueError(
                "distribution_tolerance must be "
                "between zero and one."
            )


class StratifiedSplitStrategy(BaseSplitStrategy):
    """Preserve classification target proportions across partitions."""

    strategy_name = SplitStrategyName.STRATIFIED

    def __init__(
        self,
        options: StratifiedSplitOptions,
    ) -> None:
        """Initialize the stratified split strategy."""

        super().__init__(
            ratios=options.ratios,
            random_seed=options.random_seed,
        )

        self.options = options

    def validate_contract(
        self,
        contract: SplitContract,
    ) -> None:
        """Validate classification target requirements."""

        super().validate_contract(contract)

        target = contract.target_series()

        if target.isna().any():
            raise ValueError(
                "Stratified splitting requires a "
                "nonmissing target."
            )

        class_count = int(
            target.nunique(dropna=False)
        )

        if class_count < 2:
            raise ValueError(
                "Stratified splitting requires at least "
                "two target classes."
            )

    def create_indices(
        self,
        contract: SplitContract,
    ) -> StrategyPartitions:
        """Create class-stratified partition indices."""

        data = contract.data
        target = contract.target_series()

        train_indices: list[object] = []
        validation_indices: list[object] = []
        test_indices: list[object] = []

        random_generator = np.random.default_rng(
            self.options.random_seed
        )

        class_values = target.drop_duplicates().tolist()

        for class_value in class_values:
            class_mask = target == class_value

            class_indices = data.index[
                class_mask
            ].to_numpy(
                copy=True
            )

            if self.options.shuffle_within_classes:
                random_generator.shuffle(
                    class_indices
                )

            (
                train_count,
                validation_count,
                test_count,
            ) = self._class_partition_counts(
                class_row_count=len(
                    class_indices
                )
            )

            train_end = train_count

            validation_end = (
                train_count
                + validation_count
            )

            train_indices.extend(
                class_indices[
                    :train_end
                ].tolist()
            )

            validation_indices.extend(
                class_indices[
                    train_end:validation_end
                ].tolist()
            )

            test_indices.extend(
                class_indices[
                    validation_end:
                ].tolist()
            )

            if (
                len(
                    class_indices[
                        validation_end:
                    ]
                )
                != test_count
            ):
                raise RuntimeError(
                    "Stratified splitting produced an "
                    "unexpected class-level test count."
                )

        train_array = np.asarray(
            train_indices,
            dtype=object,
        )

        validation_array = np.asarray(
            validation_indices,
            dtype=object,
        )

        test_array = np.asarray(
            test_indices,
            dtype=object,
        )

        if self.options.shuffle_within_classes:
            random_generator.shuffle(
                train_array
            )

            random_generator.shuffle(
                validation_array
            )

            random_generator.shuffle(
                test_array
            )

        return StrategyPartitions(
            train_indices=pd.Index(
                train_array
            ),
            validation_indices=pd.Index(
                validation_array
            ),
            test_indices=pd.Index(
                test_array
            ),
        )

    def create_findings(
        self,
        *,
        contract: SplitContract,
        indices: StrategyPartitions,
    ) -> list[object]:
        """Validate class presence and distribution preservation."""

        findings: list[SplitFinding] = []

        target = contract.target_series()

        complete_distribution = (
            target.value_counts(
                normalize=True,
                dropna=False,
            )
        )

        partition_indices = {
            PartitionName.TRAIN: (
                indices.train_indices
            ),
            PartitionName.VALIDATION: (
                indices.validation_indices
            ),
            PartitionName.TEST: (
                indices.test_indices
            ),
        }

        enforcement = {
            PartitionName.TRAIN: (
                self.options
                .enforce_all_classes_in_train
            ),
            PartitionName.VALIDATION: (
                self.options
                .enforce_all_classes_in_validation
            ),
            PartitionName.TEST: (
                self.options
                .enforce_all_classes_in_test
            ),
        }

        all_classes = set(
            complete_distribution.index.tolist()
        )

        for partition_name, partition_index in (
            partition_indices.items()
        ):
            partition_target = target.loc[
                partition_index
            ]

            partition_counts = (
                partition_target.value_counts(
                    dropna=False
                )
            )

            partition_classes = set(
                partition_counts.index.tolist()
            )

            missing_classes = (
                all_classes.difference(
                    partition_classes
                )
            )

            if (
                missing_classes
                and enforcement[partition_name]
            ):
                findings.append(
                    SplitFinding(
                        code="P2-STRATIFIED-001",
                        category=(
                            SplitFindingCategory
                            .CLASS_DISTRIBUTION
                        ),
                        severity=(
                            SplitFindingSeverity.ERROR
                        ),
                        component=(
                            "stratified_split"
                        ),
                        message=(
                            f"Partition "
                            f"{partition_name.value!r} "
                            "is missing one or more "
                            "target classes."
                        ),
                        partitions=[
                            partition_name
                        ],
                        recommended_action=(
                            SplitFindingAction.RESPLIT
                        ),
                        applied_action=(
                            SplitFindingAction.REPORT
                        ),
                        blocks_readiness=True,
                        details={
                            "missing_classes": [
                                str(value)
                                for value
                                in missing_classes
                            ],
                        },
                    )
                )

            small_classes = {
                str(class_value): int(count)
                for class_value, count
                in partition_counts.items()
                if count
                < self.options
                .minimum_class_count_per_partition
            }

            if small_classes:
                findings.append(
                    SplitFinding(
                        code="P2-STRATIFIED-002",
                        category=(
                            SplitFindingCategory
                            .CLASS_DISTRIBUTION
                        ),
                        severity=(
                            SplitFindingSeverity.ERROR
                        ),
                        component=(
                            "stratified_split"
                        ),
                        message=(
                            f"Partition "
                            f"{partition_name.value!r} "
                            "contains target classes "
                            "below the minimum count."
                        ),
                        partitions=[
                            partition_name
                        ],
                        recommended_action=(
                            SplitFindingAction.RESPLIT
                        ),
                        applied_action=(
                            SplitFindingAction.REPORT
                        ),
                        blocks_readiness=True,
                        details={
                            "small_classes": (
                                small_classes
                            ),
                            "minimum_count": (
                                self.options
                                .minimum_class_count_per_partition
                            ),
                        },
                    )
                )

            partition_distribution = (
                partition_target.value_counts(
                    normalize=True,
                    dropna=False,
                )
            )

            differences: dict[str, float] = {}

            for class_value in all_classes:
                complete_ratio = float(
                    complete_distribution.get(
                        class_value,
                        0.0,
                    )
                )

                partition_ratio = float(
                    partition_distribution.get(
                        class_value,
                        0.0,
                    )
                )

                difference = abs(
                    complete_ratio
                    - partition_ratio
                )

                if (
                    difference
                    > self.options
                    .distribution_tolerance
                ):
                    differences[
                        str(class_value)
                    ] = difference

            if differences:
                findings.append(
                    SplitFinding(
                        code="P2-STRATIFIED-003",
                        category=(
                            SplitFindingCategory
                            .CLASS_DISTRIBUTION
                        ),
                        severity=(
                            SplitFindingSeverity.WARNING
                        ),
                        component=(
                            "stratified_split"
                        ),
                        message=(
                            f"Partition "
                            f"{partition_name.value!r} "
                            "differs from the full target "
                            "distribution beyond tolerance."
                        ),
                        partitions=[
                            partition_name
                        ],
                        recommended_action=(
                            SplitFindingAction.WARN
                        ),
                        applied_action=(
                            SplitFindingAction.REPORT
                        ),
                        blocks_readiness=False,
                        details={
                            "differences": differences,
                            "tolerance": (
                                self.options
                                .distribution_tolerance
                            ),
                        },
                    )
                )

        return findings

    def create_metadata(
        self,
        *,
        contract: SplitContract,
        indices: StrategyPartitions,
    ) -> dict[str, object]:
        """Create metadata for the stratified split."""

        metadata = super().create_metadata(
            contract=contract,
            indices=indices,
        )

        complete_target = (
            contract.target_series()
        )

        metadata.update(
            {
                "shuffle": (
                    self.options
                    .shuffle_within_classes
                ),
                "stratified": True,
                "grouped": False,
                "temporal": False,
                "class_distribution": {
                    str(class_value): float(
                        class_ratio
                    )
                    for class_value, class_ratio
                    in complete_target.value_counts(
                        normalize=True,
                        dropna=False,
                    ).items()
                },
            }
        )

        return metadata

    def _class_partition_counts(
        self,
        *,
        class_row_count: int,
    ) -> tuple[int, int, int]:
        """Calculate partition counts for one target class."""

        required_partition_count = (
            2
            if self.ratios.validation == 0
            else 3
        )

        if class_row_count < required_partition_count:
            raise ValueError(
                "A target class contains too few rows "
                "for the requested stratified partitions. "
                f"Received {class_row_count} rows."
            )

        test_count = max(
            1,
            int(round(
                class_row_count
                * self.ratios.test
            )),
        )

        validation_count = 0

        if self.ratios.validation > 0:
            validation_count = max(
                1,
                int(round(
                    class_row_count
                    * self.ratios.validation
                )),
            )

        train_count = (
            class_row_count
            - validation_count
            - test_count
        )

        if train_count <= 0:
            raise ValueError(
                "A target class contains too few rows "
                "for the requested split ratios."
            )

        return (
            train_count,
            validation_count,
            test_count,
        )