"""ADASYN training sampler for Phase 8."""

from __future__ import annotations

from typing import Any

import pandas as pd

from mlprep.phases.phase8.samplers.base import (
    ResampledRowType,
    SamplerResult,
    SamplerRowRecord,
)
from mlprep.phases.phase8.samplers.smote_sampler import (
    SmoteSampler,
)


class AdasynSampler(SmoteSampler):
    """Generate adaptive synthetic training rows."""

    sampler_name = "adasyn_sampler"
    strategy_name = "adasyn"

    def __init__(
        self,
        *,
        density_neighbors: int = 5,
        **kwargs: Any,
    ) -> None:
        """Initialize the ADASYN sampler."""

        if density_neighbors < 1:
            raise ValueError(
                "density_neighbors must be positive."
            )

        super().__init__(
            nearest_neighbors=density_neighbors,
            **kwargs,
        )

        self.density_neighbors = (
            density_neighbors
        )

    def _resample(
        self,
        *,
        features: pd.DataFrame,
        target: pd.Series,
        row_ids: pd.Series,
    ) -> SamplerResult:
        """Execute ADASYN."""

        try:
            from imblearn.over_sampling import ADASYN
        except ImportError as error:
            raise RuntimeError(
                "ADASYN sampling requires the optional imbalanced-learn package."
            ) from error

        sampler = ADASYN(
            sampling_strategy=self.sampling_strategy,
            random_state=self.random_seed,
            n_neighbors=self.density_neighbors,
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

        synthetic_count = (
            len(
                output_features
            )
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
                    synthetic=True,
                    generation_sequence=(
                        sequence
                    ),
                )
            )

        return SamplerResult(
            features=output_features,
            target=output_target,
            row_ids=pd.Series(
                output_ids,
                name=row_ids.name,
                dtype="string",
            ),
            row_lineage=lineage,
            removed_rows=[],
            state=self.state.model_copy(
                deep=True
            ),
            metadata={
                "density_neighbors": (
                    self.density_neighbors
                ),
                "synthetic_count": (
                    synthetic_count
                ),
            },
        )
