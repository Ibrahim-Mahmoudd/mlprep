"""Random training oversampling for Phase 8."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from mlprep.phases.phase8.samplers.base import (
    BaseTrainingSampler,
    ResampledRowType,
    SamplerResult,
    SamplerRowRecord,
)


class RandomOverSampler(BaseTrainingSampler):
    """Duplicate training rows to increase minority classes."""

    sampler_name = "random_oversampler"
    strategy_name = "random_oversampling"

    def _resample(
        self,
        *,
        features: pd.DataFrame,
        target: pd.Series,
        row_ids: pd.Series,
    ) -> SamplerResult:
        """Execute deterministic random oversampling."""

        target_counts = target.value_counts(
            dropna=False
        )

        target_class_counts = self._target_counts(
            counts=target_counts
        )

        random_generator = np.random.default_rng(
            self.random_seed
        )

        source_row_ids = [
            str(value)
            for value in row_ids.tolist()
        ]

        existing_ids = set(
            source_row_ids
        )

        output_features = [
            features.copy(
                deep=True
            )
        ]

        output_targets = [
            target.copy(
                deep=True
            )
        ]

        output_row_ids = list(
            source_row_ids
        )

        lineage = [
            SamplerRowRecord(
                output_position=position,
                output_row_id=source_row_ids[
                    position
                ],
                row_type=(
                    ResampledRowType.ORIGINAL
                ),
                class_value=target.iloc[
                    position
                ],
                source_row_ids=[
                    source_row_ids[
                        position
                    ]
                ],
                source_positions=[
                    position
                ],
                original=True,
            )
            for position in range(
                len(
                    features
                )
            )
        ]

        duplicate_sequence = 0

        for class_value, desired_count in (
            target_class_counts.items()
        ):
            class_positions = np.flatnonzero(
                target.to_numpy()
                == class_value
            )

            current_count = len(
                class_positions
            )

            additional_count = (
                desired_count
                - current_count
            )

            if additional_count <= 0:
                continue

            sampled_positions = (
                random_generator.choice(
                    class_positions,
                    size=additional_count,
                    replace=True,
                )
            )

            duplicated_features = (
                features.iloc[
                    sampled_positions
                ].copy(
                    deep=True
                )
            )

            duplicated_target = (
                target.iloc[
                    sampled_positions
                ].copy(
                    deep=True
                )
            )

            output_features.append(
                duplicated_features
            )

            output_targets.append(
                duplicated_target
            )

            for source_position in sampled_positions:
                duplicate_sequence += 1

                duplicate_id = self._new_row_id(
                    row_type=(
                        ResampledRowType.DUPLICATE
                    ),
                    sequence=duplicate_sequence,
                    existing_ids=existing_ids,
                )

                output_row_ids.append(
                    duplicate_id
                )

                lineage.append(
                    SamplerRowRecord(
                        output_position=(
                            len(
                                lineage
                            )
                        ),
                        output_row_id=(
                            duplicate_id
                        ),
                        row_type=(
                            ResampledRowType.DUPLICATE
                        ),
                        class_value=target.iloc[
                            int(
                                source_position
                            )
                        ],
                        source_row_ids=[
                            source_row_ids[
                                int(
                                    source_position
                                )
                            ]
                        ],
                        source_positions=[
                            int(
                                source_position
                            )
                        ],
                        duplicated=True,
                        generation_sequence=(
                            duplicate_sequence
                        ),
                    )
                )

        resampled_features = pd.concat(
            output_features,
            axis=0,
            ignore_index=True,
        )

        resampled_target = pd.concat(
            output_targets,
            axis=0,
            ignore_index=True,
        )

        resampled_target.name = target.name

        resampled_row_ids = pd.Series(
            output_row_ids,
            name=row_ids.name,
            dtype="string",
        )

        return SamplerResult(
            features=resampled_features,
            target=resampled_target,
            row_ids=resampled_row_ids,
            row_lineage=lineage,
            removed_rows=[],
            state=self.state.model_copy(
                deep=True
            ),
            metadata={
                "target_class_counts": {
                    str(class_value): count
                    for class_value, count
                    in target_class_counts.items()
                },
                "duplicate_count": (
                    duplicate_sequence
                ),
            },
        )

    def _target_counts(
        self,
        *,
        counts: pd.Series,
    ) -> dict[Any, int]:
        """Resolve desired class counts."""

        current = {
            class_value: int(
                count
            )
            for class_value, count
            in counts.items()
        }

        maximum_count = max(
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

                if desired_count < current[
                    class_value
                ]:
                    raise ValueError(
                        "Random oversampling cannot reduce "
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
            "not_majority",
            "all",
        }:
            return {
                class_value: maximum_count
                for class_value in current
            }

        if self.sampling_strategy == "minority":
            minimum_count = min(
                current.values()
            )

            return {
                class_value: (
                    maximum_count
                    if count == minimum_count
                    else count
                )
                for class_value, count
                in current.items()
            }

        raise ValueError(
            "Unsupported random-oversampling strategy: "
            f"{self.sampling_strategy!r}"
        )
