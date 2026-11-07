"""SMOTE training sampler for Phase 8."""

from __future__ import annotations

from typing import Any

import pandas as pd

from mlprep.phases.phase8.samplers.base import (
    BaseTrainingSampler,
    ResampledRowType,
    SamplerResult,
    SamplerRowRecord,
)


class SmoteSampler(BaseTrainingSampler):
    """Generate synthetic training rows with SMOTE."""

    sampler_name = "smote_sampler"
    strategy_name = "smote"

    def __init__(
        self,
        *,
        nearest_neighbors: int = 5,
        **kwargs: Any,
    ) -> None:
        """Initialize the SMOTE sampler."""

        if nearest_neighbors < 1:
            raise ValueError(
                "nearest_neighbors must be positive."
            )

        super().__init__(
            **kwargs
        )

        self.nearest_neighbors = (
            nearest_neighbors
        )

    def requires_synthetic_authorization(
        self,
    ) -> bool:
        """Return whether synthetic authorization is required."""

        return True

    def _resample(
        self,
        *,
        features: pd.DataFrame,
        target: pd.Series,
        row_ids: pd.Series,
    ) -> SamplerResult:
        """Execute SMOTE."""

        try:
            from imblearn.over_sampling import SMOTE
        except ImportError as error:
            raise RuntimeError(
                "SMOTE sampling requires the optional imbalanced-learn package."
            ) from error

        sampler = SMOTE(
            sampling_strategy=self.sampling_strategy,
            random_state=self.random_seed,
            k_neighbors=self.nearest_neighbors,
        )

        output_values, output_target_values = (
            sampler.fit_resample(
                features,
                target,
            )
        )

        output_features = pd.DataFrame(
            output_values,
            columns=features.columns,
        )

        output_target = pd.Series(
            output_target_values,
            name=target.name,
        )

        original_count = len(
            features
        )

        output_count = len(
            output_features
        )

        synthetic_count = (
            output_count
            - original_count
        )

        original_ids = [
            str(value)
            for value in row_ids.tolist()
        ]

        existing_ids = set(
            original_ids
        )

        output_ids = list(
            original_ids
        )

        lineage = [
            SamplerRowRecord(
                output_position=position,
                output_row_id=original_ids[
                    position
                ],
                row_type=(
                    ResampledRowType.ORIGINAL
                ),
                class_value=target.iloc[
                    position
                ],
                source_row_ids=[
                    original_ids[
                        position
                    ]
                ],
                source_positions=[
                    position
                ],
                original=True,
            )
            for position in range(
                original_count
            )
        ]

        for sequence in range(
            1,
            synthetic_count + 1,
        ):
            output_position = (
                original_count
                + sequence
                - 1
            )

            synthetic_id = self._new_row_id(
                row_type=(
                    ResampledRowType.SYNTHETIC
                ),
                sequence=sequence,
                existing_ids=existing_ids,
            )

            output_ids.append(
                synthetic_id
            )

            lineage.append(
                SamplerRowRecord(
                    output_position=(
                        output_position
                    ),
                    output_row_id=(
                        synthetic_id
                    ),
                    row_type=(
                        ResampledRowType.SYNTHETIC
                    ),
                    class_value=(
                        output_target.iloc[
                            output_position
                        ]
                    ),
                    source_row_ids=[],
                    source_positions=[],
                    synthetic=True,
                    generation_sequence=(
                        sequence
                    ),
                    metadata={
                        "lineage_detail": (
                            "SMOTE does not expose selected "
                            "source-neighbor indices through "
                            "its public fit_resample interface."
                        ),
                    },
                )
            )

        output_row_ids = pd.Series(
            output_ids,
            name=row_ids.name,
            dtype="string",
        )

        return SamplerResult(
            features=output_features,
            target=output_target,
            row_ids=output_row_ids,
            row_lineage=lineage,
            removed_rows=[],
            state=self.state.model_copy(
                deep=True
            ),
            metadata={
                "nearest_neighbors": (
                    self.nearest_neighbors
                ),
                "synthetic_count": (
                    synthetic_count
                ),
            },
        )
