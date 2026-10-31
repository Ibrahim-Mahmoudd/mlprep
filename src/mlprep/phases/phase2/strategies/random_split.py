"""Random dataset splitting for Phase 2."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from mlprep.phases.phase2.contract.split_contract import (
    SplitContract,
)
from mlprep.phases.phase2.output_governance.metadata import (
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
class RandomSplitOptions:
    """Options controlling random dataset splitting."""

    ratios: SplitRatios
    random_seed: int = 42

    shuffle: bool = True

    minimum_train_rows: int = 1
    minimum_validation_rows: int = 1
    minimum_test_rows: int = 1

    def __post_init__(self) -> None:
        """Validate random-split options."""

        self.ratios.validate_total()

        minimum_values = {
            "minimum_train_rows": (
                self.minimum_train_rows
            ),
            "minimum_validation_rows": (
                self.minimum_validation_rows
            ),
            "minimum_test_rows": (
                self.minimum_test_rows
            ),
        }

        for name, value in minimum_values.items():
            if value < 0:
                raise ValueError(
                    f"{name} cannot be negative."
                )


class RandomSplitStrategy(BaseSplitStrategy):
    """Split rows randomly without target stratification."""

    strategy_name = SplitStrategyName.RANDOM

    def __init__(
        self,
        options: RandomSplitOptions,
    ) -> None:
        """Initialize the random split strategy."""

        super().__init__(
            ratios=options.ratios,
            random_seed=options.random_seed,
        )

        self.options = options

    def create_indices(
        self,
        contract: SplitContract,
    ) -> StrategyPartitions:
        """Create randomly assigned partition indices."""

        source_indices = contract.data.index.to_numpy(
            copy=True
        )

        if self.options.shuffle:
            random_generator = (
                np.random.default_rng(
                    self.options.random_seed
                )
            )

            random_generator.shuffle(
                source_indices
            )

        (
            train_count,
            validation_count,
            test_count,
        ) = self.calculate_partition_counts(
            row_count=len(source_indices),
            ratios=self.ratios,
        )

        train_end = train_count

        validation_end = (
            train_count
            + validation_count
        )

        train_indices = pd.Index(
            source_indices[
                :train_end
            ]
        )

        validation_indices = pd.Index(
            source_indices[
                train_end:validation_end
            ]
        )

        test_indices = pd.Index(
            source_indices[
                validation_end:
            ]
        )

        if len(test_indices) != test_count:
            raise RuntimeError(
                "Random split produced an unexpected "
                "test partition size."
            )

        return StrategyPartitions(
            train_indices=train_indices,
            validation_indices=validation_indices,
            test_indices=test_indices,
        )

    def create_findings(
        self,
        *,
        contract: SplitContract,
        indices: StrategyPartitions,
    ) -> list[object]:
        """Create size findings for the random split."""

        findings: list[SplitFinding] = []

        partition_sizes = {
            "train": len(
                indices.train_indices
            ),
            "validation": len(
                indices.validation_indices
            ),
            "test": len(
                indices.test_indices
            ),
        }

        minimum_sizes = {
            "train": (
                self.options.minimum_train_rows
            ),
            "validation": (
                self.options
                .minimum_validation_rows
            ),
            "test": (
                self.options.minimum_test_rows
            ),
        }

        for partition_name, minimum_size in (
            minimum_sizes.items()
        ):
            actual_size = partition_sizes[
                partition_name
            ]

            if actual_size >= minimum_size:
                continue

            findings.append(
                SplitFinding(
                    code="P2-RANDOM-001",
                    category=(
                        SplitFindingCategory
                        .SPLIT_SIZE
                    ),
                    severity=(
                        SplitFindingSeverity.ERROR
                    ),
                    component="random_split",
                    message=(
                        f"Partition {partition_name!r} "
                        f"contains {actual_size} rows, "
                        f"below the required minimum "
                        f"of {minimum_size}."
                    ),
                    recommended_action=(
                        SplitFindingAction.RESPLIT
                    ),
                    applied_action=(
                        SplitFindingAction.REPORT
                    ),
                    blocks_readiness=True,
                    details={
                        "partition": (
                            partition_name
                        ),
                        "actual_size": actual_size,
                        "minimum_size": minimum_size,
                    },
                )
            )

        if not self.options.shuffle:
            findings.append(
                SplitFinding(
                    code="P2-RANDOM-002",
                    category=(
                        SplitFindingCategory
                        .STRATEGY
                    ),
                    severity=(
                        SplitFindingSeverity.WARNING
                    ),
                    component="random_split",
                    message=(
                        "Random splitting was configured "
                        "without shuffling. Source row order "
                        "may influence partition membership."
                    ),
                    recommended_action=(
                        SplitFindingAction.WARN
                    ),
                    applied_action=(
                        SplitFindingAction.REPORT
                    ),
                    blocks_readiness=False,
                )
            )

        return findings

    def create_metadata(
        self,
        *,
        contract: SplitContract,
        indices: StrategyPartitions,
    ) -> dict[str, object]:
        """Create metadata for the random split."""

        metadata = super().create_metadata(
            contract=contract,
            indices=indices,
        )

        metadata.update(
            {
                "shuffle": (
                    self.options.shuffle
                ),
                "stratified": False,
                "grouped": False,
                "temporal": False,
            }
        )

        return metadata