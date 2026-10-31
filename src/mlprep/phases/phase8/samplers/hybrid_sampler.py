"""Hybrid SMOTE and cleaning samplers for Phase 8."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from mlprep.phases.phase8.contract.balancing_plan import (
    HybridCleaningKind,
)
from mlprep.phases.phase8.samplers.base import (
    BaseTrainingSampler,
    RemovedRowRecord,
    ResampledRowType,
    SamplerResult,
    SamplerRowRecord,
)


class HybridSampler(BaseTrainingSampler):
    """Combine SMOTE with Tomek or ENN cleaning."""

    sampler_name = "hybrid_sampler"

    def __init__(
        self,
        *,
        cleaning: HybridCleaningKind,
        nearest_neighbors: int = 5,
        enn_neighbors: int = 3,
        **kwargs: Any,
    ) -> None:
        """Initialize the hybrid sampler."""

        super().__init__(
            **kwargs
        )

        self.cleaning = (
            cleaning
            if isinstance(
                cleaning,
                HybridCleaningKind,
            )
            else HybridCleaningKind(
                cleaning
            )
        )

        if nearest_neighbors < 1:
            raise ValueError(
                "nearest_neighbors must be positive."
            )

        if enn_neighbors < 1:
            raise ValueError(
                "enn_neighbors must be positive."
            )

        self.nearest_neighbors = (
            nearest_neighbors
        )

        self.enn_neighbors = (
            enn_neighbors
        )

        self.strategy_name = (
            "smote_tomek"
            if self.cleaning
            == HybridCleaningKind.TOMEK
            else "smote_enn"
        )

        self.state.strategy = (
            self.strategy_name
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
        """Execute hybrid oversampling and cleaning."""

        try:
            from imblearn.combine import SMOTEENN, SMOTETomek
            from imblearn.over_sampling import SMOTE
            from imblearn.under_sampling import (
                EditedNearestNeighbours,
                TomekLinks,
            )
        except ImportError as error:
            raise RuntimeError(
                "Hybrid sampling requires the optional "
                "imbalanced-learn package."
            ) from error

        smote = SMOTE(
            sampling_strategy=self.sampling_strategy,
            random_state=self.random_seed,
            k_neighbors=self.nearest_neighbors,
        )

        if self.cleaning == HybridCleaningKind.TOMEK:
            sampler = SMOTETomek(
                sampling_strategy=self.sampling_strategy,
                random_state=self.random_seed,
                smote=smote,
                tomek=TomekLinks(
                    sampling_strategy="all"
                ),
            )

        else:
            sampler = SMOTEENN(
                sampling_strategy=self.sampling_strategy,
                random_state=self.random_seed,
                smote=smote,
                enn=EditedNearestNeighbours(
                    n_neighbors=self.enn_neighbors,
                    sampling_strategy="all",
                ),
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

        (
            output_row_ids,
            lineage,
            removed_rows,
        ) = self._infer_hybrid_lineage(
            source_features=features,
            source_target=target,
            source_row_ids=row_ids,
            output_features=output_features,
            output_target=output_target,
        )

        return SamplerResult(
            features=output_features,
            target=output_target,
            row_ids=output_row_ids,
            row_lineage=lineage,
            removed_rows=removed_rows,
            state=self.state.model_copy(
                deep=True
            ),
            metadata={
                "cleaning": (
                    self.cleaning.value
                ),
                "nearest_neighbors": (
                    self.nearest_neighbors
                ),
                "enn_neighbors": (
                    self.enn_neighbors
                ),
                "lineage_inference": (
                    "Exact feature-row matching was used "
                    "because hybrid imbalanced-learn samplers "
                    "do not expose complete public row lineage."
                ),
            },
        )

    def _infer_hybrid_lineage(
        self,
        *,
        source_features: pd.DataFrame,
        source_target: pd.Series,
        source_row_ids: pd.Series,
        output_features: pd.DataFrame,
        output_target: pd.Series,
    ) -> tuple[
        pd.Series,
        list[SamplerRowRecord],
        list[RemovedRowRecord],
    ]:
        """Infer original and synthetic hybrid output lineage."""

        source_values = source_features.to_numpy(
            dtype=float
        )

        output_values = output_features.to_numpy(
            dtype=float
        )

        source_ids = [
            str(value)
            for value in source_row_ids.tolist()
        ]

        existing_ids = set(
            source_ids
        )

        matched_source_positions: set[int] = set()

        output_ids: list[str] = []
        lineage: list[
            SamplerRowRecord
        ] = []

        synthetic_sequence = 0

        for output_position, output_row in enumerate(
            output_values
        ):
            output_class = output_target.iloc[
                output_position
            ]

            matching_positions = np.flatnonzero(
                np.all(
                    np.isclose(
                        source_values,
                        output_row,
                        rtol=1e-12,
                        atol=1e-12,
                        equal_nan=True,
                    ),
                    axis=1,
                )
                & (
                    source_target.to_numpy()
                    == output_class
                )
            )

            available_matches = [
                int(position)
                for position in matching_positions
                if int(position)
                not in matched_source_positions
            ]

            if available_matches:
                source_position = available_matches[
                    0
                ]

                matched_source_positions.add(
                    source_position
                )

                output_id = source_ids[
                    source_position
                ]

                output_ids.append(
                    output_id
                )

                lineage.append(
                    SamplerRowRecord(
                        output_position=(
                            output_position
                        ),
                        output_row_id=(
                            output_id
                        ),
                        row_type=(
                            ResampledRowType
                            .RETAINED_AFTER_CLEANING
                        ),
                        class_value=(
                            output_class
                        ),
                        source_row_ids=[
                            output_id
                        ],
                        source_positions=[
                            source_position
                        ],
                        original=True,
                    )
                )

            else:
                synthetic_sequence += 1

                synthetic_id = self._new_row_id(
                    row_type=(
                        ResampledRowType.SYNTHETIC
                    ),
                    sequence=(
                        synthetic_sequence
                    ),
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
                            output_class
                        ),
                        synthetic=True,
                        generation_sequence=(
                            synthetic_sequence
                        ),
                    )
                )

        removed_rows = [
            RemovedRowRecord(
                row_id=source_ids[
                    source_position
                ],
                source_position=(
                    source_position
                ),
                class_value=(
                    source_target.iloc[
                        source_position
                    ]
                ),
                reason=(
                    "Removed by hybrid neighborhood cleaning."
                ),
                stage=self.strategy_name,
            )
            for source_position in range(
                len(
                    source_features
                )
            )
            if source_position
            not in matched_source_positions
        ]

        return (
            pd.Series(
                output_ids,
                name=source_row_ids.name,
                dtype="string",
            ),
            lineage,
            removed_rows,
        )