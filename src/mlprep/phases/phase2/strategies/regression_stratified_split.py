"""Stratified regression splitting for Phase 2."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from mlprep.phases.phase2.contract.split_contract import (
    SplitContract,
)
from mlprep.phases.phase2.output_governance.metadata import (
    SplitRatios,
    SplitStrategyName,
)
from mlprep.phases.phase2.strategies.base import (
    StrategyPartitions,
)
from mlprep.phases.phase2.strategies.stratified_split import (
    StratifiedSplitOptions,
    StratifiedSplitStrategy,
)


@dataclass(slots=True)
class RegressionStratifiedSplitOptions:
    """Options controlling regression stratification."""

    ratios: SplitRatios

    random_seed: int = 42
    number_of_bins: int = 10

    shuffle_within_bins: bool = True

    minimum_bin_count: int = 3

    duplicate_boundary_policy: str = "drop"

    def __post_init__(self) -> None:
        """Validate regression-stratified options."""

        self.ratios.validate_total()

        if self.number_of_bins < 2:
            raise ValueError(
                "number_of_bins must be at least two."
            )

        if self.minimum_bin_count < 1:
            raise ValueError(
                "minimum_bin_count must be positive."
            )

        if self.duplicate_boundary_policy not in {
            "drop",
            "raise",
        }:
            raise ValueError(
                "duplicate_boundary_policy must be "
                "'drop' or 'raise'."
            )


class RegressionStratifiedSplitStrategy(
    StratifiedSplitStrategy
):
    """Stratify regression targets using quantile bins."""

    strategy_name = (
        SplitStrategyName.REGRESSION_STRATIFIED
    )

    def __init__(
        self,
        options: RegressionStratifiedSplitOptions,
    ) -> None:
        """Initialize the strategy."""

        self.regression_options = options

        super().__init__(
            options=StratifiedSplitOptions(
                ratios=options.ratios,
                random_seed=options.random_seed,
                shuffle_within_classes=(
                    options
                    .shuffle_within_bins
                ),
                minimum_class_count_per_partition=1,
                enforce_all_classes_in_train=True,
                enforce_all_classes_in_validation=True,
                enforce_all_classes_in_test=True,
                distribution_tolerance=0.10,
            )
        )

    def validate_contract(
        self,
        contract: SplitContract,
    ) -> None:
        """Validate a numeric regression target."""

        target = contract.target_series()

        if target.isna().any():
            raise ValueError(
                "Regression-stratified splitting requires "
                "a nonmissing target."
            )

        if not pd.api.types.is_numeric_dtype(
            target.dtype
        ):
            raise ValueError(
                "Regression-stratified splitting requires "
                "a numeric target."
            )

        if target.nunique() < 2:
            raise ValueError(
                "Regression-stratified splitting requires "
                "a nonconstant target."
            )

    def create_indices(
        self,
        contract: SplitContract,
    ) -> StrategyPartitions:
        """Create stratified indices from target quantile bins."""

        target = contract.target_series()

        target_bins = pd.qcut(
            target,
            q=self.regression_options.number_of_bins,
            labels=False,
            duplicates=(
                self.regression_options
                .duplicate_boundary_policy
            ),
        )

        bin_counts = target_bins.value_counts(
            dropna=False
        )

        insufficient_bins = bin_counts[
            bin_counts
            < self.regression_options
            .minimum_bin_count
        ]

        if not insufficient_bins.empty:
            raise ValueError(
                "One or more regression target bins contain "
                "too few rows for stratified splitting."
            )

        original_data = contract.data
        original_target_column = (
            contract.columns.target_column
        )

        temporary_target_column = (
            "__mlprep_regression_bin__"
        )

        while (
            temporary_target_column
            in original_data.columns
        ):
            temporary_target_column += "_"

        temporary_data = original_data.copy()

        temporary_data[
            temporary_target_column
        ] = target_bins

        original_contract_data = contract.data
        original_contract_target = (
            contract.columns.target_column
        )

        try:
            contract.data = temporary_data
            contract.columns.target_column = (
                temporary_target_column
            )

            return super().create_indices(
                contract
            )

        finally:
            contract.data = original_contract_data
            contract.columns.target_column = (
                original_contract_target
            )

            if (
                original_target_column
                != original_contract_target
            ):
                contract.columns.target_column = (
                    original_target_column
                )

    def create_findings(
        self,
        *,
        contract: SplitContract,
        indices: StrategyPartitions,
    ) -> list[object]:
        """Return regression-stratified findings."""

        return []

    def create_metadata(
        self,
        *,
        contract: SplitContract,
        indices: StrategyPartitions,
    ) -> dict[str, Any]:
        """Create regression-stratified metadata."""

        metadata = super().create_metadata(
            contract=contract,
            indices=indices,
        )

        metadata.update(
            {
                "strategy": (
                    self.strategy_name.value
                ),
                "regression_stratified": True,
                "number_of_bins": (
                    self.regression_options
                    .number_of_bins
                ),
                "duplicate_boundary_policy": (
                    self.regression_options
                    .duplicate_boundary_policy
                ),
            }
        )

        return metadata
