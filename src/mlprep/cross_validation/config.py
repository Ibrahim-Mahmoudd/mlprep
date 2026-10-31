"""Configuration for leakage-safe cross-validation."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class CrossValidationStrategy(StrEnum):
    """Supported fold-generation strategies."""

    K_FOLD = "k_fold"
    STRATIFIED = "stratified"
    GROUP = "group"
    TIME_SERIES = "time_series"


@dataclass(frozen=True, slots=True)
class CrossValidationConfig:
    """Settings for generating cross-validation folds."""

    strategy: CrossValidationStrategy = CrossValidationStrategy.STRATIFIED
    n_splits: int = 5
    random_seed: int = 42
    shuffle: bool | None = None

    def __post_init__(self) -> None:
        """Validate split settings."""

        if not isinstance(self.strategy, CrossValidationStrategy):
            object.__setattr__(
                self,
                "strategy",
                CrossValidationStrategy(self.strategy),
            )
        if (
            not isinstance(self.n_splits, int)
            or isinstance(self.n_splits, bool)
            or self.n_splits < 2
        ):
            raise ValueError("n_splits must be an integer of at least two.")
        if (
            not isinstance(self.random_seed, int)
            or isinstance(self.random_seed, bool)
        ):
            raise TypeError("random_seed must be an integer.")
        if self.random_seed < 0:
            raise ValueError("random_seed cannot be negative.")
        if self.shuffle is not None and not isinstance(self.shuffle, bool):
            raise TypeError("shuffle must be a Boolean or None.")
        if (
            self.strategy
            not in {
                CrossValidationStrategy.K_FOLD,
                CrossValidationStrategy.STRATIFIED,
            }
            and self.shuffle is True
        ):
            raise ValueError(
                "shuffle=True is only supported for k-fold or "
                "stratified cross-validation."
            )
