"""Borderline-SMOTE training sampler for Phase 8."""

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


class BorderlineSmoteSampler(SmoteSampler):
    """Generate synthetic rows near class boundaries."""

    sampler_name = "borderline_smote_sampler"
    strategy_name = "borderline_smote"

    def __init__(
        self,
        *,
        method: str = "borderline-1",
        danger_neighbors: int = 10,
        **kwargs: Any,
    ) -> None:
        """Initialize the Borderline-SMOTE sampler."""

        if method not in {
            "borderline-1",
            "borderline-2",
        }:
            raise ValueError(
                "method must be 'borderline-1' or "
                "'borderline-2'."
            )

        if danger_neighbors < 1:
            raise ValueError(
                "danger_neighbors must be positive."
            )

        super().__init__(
            **kwargs
        )

        self.method = method
        self.danger_neighbors = danger_neighbors

    def _resample(
        self,
        *,
        features: pd.DataFrame,
        target: pd.Series,
        row_ids: pd.Series,
    ) -> SamplerResult:
        """Execute Borderline-SMOTE."""

        try:
            from imblearn.over_sampling import BorderlineSMOTE
        except ImportError as error:
            raise RuntimeError(
                "Borderline-SMOTE requires the optional imbalanced-learn package."
            ) from error

        sampler = BorderlineSMOTE(
            sampling_strategy=self.sampling_strategy,
            random_state=self.random_seed,
            k_neighbors=self.nearest_neighbors,
            m_neighbors=self.danger_neighbors,
            kind=self.method,
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
                    metadata={
                        "method": self.method,
                    },
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
                "method": self.method,
                "nearest_neighbors": (
                    self.nearest_neighbors
                ),
                "danger_neighbors": (
                    self.danger_neighbors
                ),
                "synthetic_count": (
                    synthetic_count
                ),
            },
        )
