"""Treatment-strategy factory for Phase 5."""

from __future__ import annotations

from typing import Any

from mlprep.phases.phase5.contract.treatment_plan import (
    TreatmentKind,
    TreatmentPlanEntry,
)
from mlprep.phases.phase5.strategies.base import (
    BaseTreatmentStrategy,
)
from mlprep.phases.phase5.strategies.clip_strategy import (
    ClipStrategy,
)
from mlprep.phases.phase5.strategies.report_only_strategy import (
    ReportOnlyStrategy,
)
from mlprep.phases.phase5.strategies.set_missing_strategy import (
    SetMissingStrategy,
)
from mlprep.phases.phase5.strategies.winsorize_strategy import (
    WinsorizeStrategy,
    WinsorizeStrategyOptions,
)


class StrategyFactory:
    """Build treatment strategies from plan entries."""

    component_name = "strategy_factory"

    @classmethod
    def create(
        cls,
        *,
        entry: TreatmentPlanEntry,
        detector_state: dict[str, Any],
        winsorize_options: (
            WinsorizeStrategyOptions | None
        ) = None,
    ) -> BaseTreatmentStrategy:
        """Build one fitted treatment strategy."""

        if entry.status.value != "planned":
            raise ValueError(
                f"Cannot create a strategy for plan "
                f"status {entry.status.value!r}."
            )

        lower_bound = cls._optional_float(
            detector_state.get(
                "lower_bound"
            )
        )

        upper_bound = cls._optional_float(
            detector_state.get(
                "upper_bound"
            )
        )

        if entry.treatment == TreatmentKind.CLIP:
            return ClipStrategy(
                column=entry.column,
                lower_bound=lower_bound,
                upper_bound=upper_bound,
            )

        if (
            entry.treatment
            == TreatmentKind.WINSORIZE
        ):
            return WinsorizeStrategy(
                column=entry.column,
                lower_bound=lower_bound,
                upper_bound=upper_bound,
                options=winsorize_options,
            )

        if (
            entry.treatment
            == TreatmentKind.SET_MISSING
        ):
            if entry.set_missing_fallback is None:
                raise ValueError(
                    "set_missing requires a fallback value."
                )

            return SetMissingStrategy(
                column=entry.column,
                lower_bound=lower_bound,
                upper_bound=upper_bound,
                fallback_value=(
                    entry.set_missing_fallback
                ),
            )

        if (
            entry.treatment
            == TreatmentKind.REPORT_ONLY
        ):
            return ReportOnlyStrategy(
                column=entry.column,
                lower_bound=lower_bound,
                upper_bound=upper_bound,
            )

        raise ValueError(
            "Unsupported treatment method: "
            f"{entry.treatment.value!r}"
        )

    @classmethod
    def create_many(
        cls,
        *,
        entries: list[TreatmentPlanEntry],
        detector_states: dict[
            str,
            dict[str, Any],
        ],
        winsorize_options: (
            WinsorizeStrategyOptions | None
        ) = None,
    ) -> dict[
        str,
        BaseTreatmentStrategy,
    ]:
        """Build treatment strategies for multiple features."""

        strategies: dict[
            str,
            BaseTreatmentStrategy,
        ] = {}

        for entry in entries:
            if entry.column not in detector_states:
                raise KeyError(
                    "Detector state is unavailable for "
                    f"feature {entry.column!r}."
                )

            strategies[entry.column] = cls.create(
                entry=entry,
                detector_state=(
                    detector_states[
                        entry.column
                    ]
                ),
                winsorize_options=(
                    winsorize_options
                ),
            )

        return strategies

    @staticmethod
    def _optional_float(
        value: Any,
    ) -> float | None:
        """Convert an optional numeric value."""

        if value is None:
            return None

        return float(value)
