"""Random training undersampling for Phase 8."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from mlprep.phases.phase8.samplers.base import (
    BaseTrainingSampler,
    RemovedRowRecord,
    ResampledRowType,
    SamplerResult,
    SamplerRowRecord,
)


class RandomUnderSampler(BaseTrainingSampler):
    """Remove majority-class training rows deterministically."""

    sampler_name = "random_undersampler"
    strategy_name = "random_undersampling"

    def _resample(
        self,
        *,
        features: pd.DataFrame,
        target: pd.Series,
        row_ids: pd.Series,
    ) -> SamplerResult:
        """Execute random undersampling."""

        counts = target.value_counts(
            dropna=False
        )

        target_counts = self._target_counts(
            counts=counts
        )

        random_generator = np.random.default_rng(
            self.random_seed
        )

        retained_positions: list[int] = []

        for class_value, desired_count in (
            target_counts.items()
        ):
            class_positions = np.flatnonzero(
                target.to_numpy()
                == class_value
            )

            if desired_count > len(
                class_positions
            ):
                raise ValueError(
                    "Random undersampling cannot increase "
                    "class counts."
                )

            selected_positions = (
                random_generator.choice(
                    class_positions,
                    size=desired_count,
                    replace=False,
                )
            )

            retained_positions.extend(
                int(position)
                for position
                in selected_positions
            )

        retained_positions = sorted(
            retained_positions
        )

        retained_set = set(
            retained_positions
        )

        removed_positions = [
            position
            for position in range(
                len(
                    features
                )
            )
            if position not in retained_set
        ]

        resampled_features = (
            features.iloc[
                retained_positions
            ].copy(
                deep=True
            )
            .reset_index(
                drop=True
            )
        )

        resampled_target = (
            target.iloc[
                retained_positions
            ].copy(
                deep=True
            )
            .reset_index(
                drop=True
            )
        )

        resampled_target.name = target.name

        resampled_row_ids = pd.Series(
            [
                str(
                    row_ids.iloc[
                        position
                    ]
                )
                for position
                in retained_positions
            ],
            name=row_ids.name,
            dtype="string",
        )

        lineage = [
            SamplerRowRecord(
                output_position=output_position,
                output_row_id=str(
                    row_ids.iloc[
                        source_position
                    ]
                ),
                row_type=(
                    ResampledRowType
                    .RETAINED_AFTER_CLEANING
                ),
                class_value=target.iloc[
                    source_position
                ],
                source_row_ids=[
                    str(
                        row_ids.iloc[
                            source_position
                        ]
                    )
                ],
                source_positions=[
                    source_position
                ],
                original=True,
            )
            for output_position, source_position
            in enumerate(
                retained_positions
            )
        ]

        removed_rows = [
            RemovedRowRecord(
                row_id=str(
                    row_ids.iloc[
                        source_position
                    ]
                ),
                source_position=(
                    source_position
                ),
                class_value=target.iloc[
                    source_position
                ],
                reason=(
                    "Removed by random undersampling."
                ),
                stage=self.strategy_name,
            )
            for source_position in removed_positions
        ]

        return SamplerResult(
            features=resampled_features,
            target=resampled_target,
            row_ids=resampled_row_ids,
            row_lineage=lineage,
            removed_rows=removed_rows,
            state=self.state.model_copy(
                deep=True
            ),
            metadata={
                "target_class_counts": {
                    str(class_value): count
                    for class_value, count
                    in target_counts.items()
                },
                "retained_original_count": len(
                    retained_positions
                ),
                "removed_original_count": len(
                    removed_positions
                ),
            },
        )

    def _target_counts(
        self,
        *,
        counts: pd.Series,
    ) -> dict[Any, int]:
        """Resolve desired undersampled class counts."""

        current = {
            class_value: int(
                count
            )
            for class_value, count
            in counts.items()
        }

        minimum_count = min(
            current.values()
        )

        if isinstance(
            self.sampling_strategy,
            dict,
        ):
            targets = dict(
                current
            )

            for class_value, desired_count in (
                self.sampling_strategy.items()
            ):
                if class_value not in current:
                    raise ValueError(
                        "Sampling strategy contains an unknown "
                        f"class: {class_value!r}"
                    )

                if desired_count > current[
                    class_value
                ]:
                    raise ValueError(
                        "Random undersampling cannot increase "
                        "class counts."
                    )

                targets[
                    class_value
                ] = int(
                    desired_count
                )

            return targets

        if self.sampling_strategy in {
            "auto",
            "not_minority",
            "all",
        }:
            return {
                class_value: minimum_count
                for class_value in current
            }

        if self.sampling_strategy == "majority":
            maximum_count = max(
                current.values()
            )

            return {
                class_value: (
                    minimum_count
                    if count == maximum_count
                    else count
                )
                for class_value, count
                in current.items()
            }

        raise ValueError(
            "Unsupported random-undersampling strategy: "
            f"{self.sampling_strategy!r}"
        )