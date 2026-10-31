"""Train and validation data isolated for one fold."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(slots=True)
class FoldContext:
    """Data and fingerprints available to one fold pipeline.

    A fold context deliberately contains no test partition.
    """

    fold_index: int
    training_indices: tuple[int, ...]
    validation_indices: tuple[int, ...]
    training_features: pd.DataFrame
    training_target: pd.Series
    training_row_ids: pd.Series
    validation_features: pd.DataFrame
    validation_target: pd.Series
    validation_row_ids: pd.Series
    training_groups: pd.Series | None
    validation_groups: pd.Series | None
    training_fingerprint: str
    validation_fingerprint: str

    def __post_init__(self) -> None:
        """Validate fold alignment and partition separation."""

        if self.fold_index < 0:
            raise ValueError("fold_index cannot be negative.")
        if not self.training_indices or not self.validation_indices:
            raise ValueError("Fold train and validation must be nonempty.")
        if set(self.training_indices).intersection(self.validation_indices):
            raise ValueError("Fold train and validation indices overlap.")
        self._validate_partition(
            name="training",
            features=self.training_features,
            target=self.training_target,
            row_ids=self.training_row_ids,
            groups=self.training_groups,
        )
        self._validate_partition(
            name="validation",
            features=self.validation_features,
            target=self.validation_target,
            row_ids=self.validation_row_ids,
            groups=self.validation_groups,
        )

    @staticmethod
    def _validate_partition(
        *,
        name: str,
        features: pd.DataFrame,
        target: pd.Series,
        row_ids: pd.Series,
        groups: pd.Series | None,
    ) -> None:
        if not isinstance(features, pd.DataFrame):
            raise TypeError(f"{name} features must be a DataFrame.")
        if not isinstance(target, pd.Series):
            raise TypeError(f"{name} target must be a Series.")
        if not isinstance(row_ids, pd.Series):
            raise TypeError(f"{name} row IDs must be a Series.")
        if not (
            len(features) == len(target) == len(row_ids)
            and features.index.equals(target.index)
            and features.index.equals(row_ids.index)
        ):
            raise ValueError(f"{name} features, target, and row IDs must align.")
        if row_ids.isna().any() or not row_ids.is_unique:
            raise ValueError(f"{name} row IDs must be unique and nonmissing.")
        if groups is not None and (
            len(groups) != len(features) or not features.index.equals(groups.index)
        ):
            raise ValueError(f"{name} groups must align with its features.")
