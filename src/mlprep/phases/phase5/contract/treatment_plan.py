"""Executable outlier-treatment plan for Phase 5."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    model_validator,
)

from mlprep.phases.phase5.contract.outlier_contract import (
    OutlierContract,
)


class DetectorKind(StrEnum):
    """Supported outlier-detection methods."""

    IQR = "iqr"
    QUANTILE = "quantile"
    ROBUST_ZSCORE = "robust_zscore"
    DOMAIN = "domain"


class TreatmentKind(StrEnum):
    """Supported outlier-treatment methods."""

    CLIP = "clip"
    WINSORIZE = "winsorize"
    SET_MISSING = "set_missing"
    REPORT_ONLY = "report_only"


class TreatmentPlanStatus(StrEnum):
    """Execution status of one treatment entry."""

    PLANNED = "planned"
    SKIPPED = "skipped"
    BLOCKED = "blocked"


class TreatmentPlanEntry(BaseModel):
    """Outlier-treatment plan for one numeric feature."""

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
    )

    column: str

    detector: DetectorKind
    treatment: TreatmentKind

    add_indicator: bool = False

    indicator_suffix: str = "__was_outlier"

    domain_minimum: float | None = None
    domain_maximum: float | None = None

    set_missing_fallback: float | None = None

    requires_training_fit: bool = True

    status: TreatmentPlanStatus = (
        TreatmentPlanStatus.PLANNED
    )

    priority: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
    )

    reason: str

    warnings: list[str] = Field(
        default_factory=list
    )

    metadata: dict[str, Any] = Field(
        default_factory=dict
    )

    @model_validator(mode="after")
    def validate_entry(
        self,
    ) -> TreatmentPlanEntry:
        """Validate one treatment-plan entry."""

        if not self.column.strip():
            raise ValueError(
                "column cannot be empty."
            )

        if not self.indicator_suffix:
            raise ValueError(
                "indicator_suffix cannot be empty."
            )

        if self.detector == DetectorKind.DOMAIN:
            if (
                self.domain_minimum is None
                and self.domain_maximum is None
            ):
                raise ValueError(
                    "Domain detection requires at least "
                    "one configured bound."
                )

            if (
                self.domain_minimum is not None
                and self.domain_maximum is not None
                and self.domain_minimum
                > self.domain_maximum
            ):
                raise ValueError(
                    "domain_minimum cannot exceed "
                    "domain_maximum."
                )

        if (
            self.treatment
            == TreatmentKind.SET_MISSING
            and self.set_missing_fallback is None
        ):
            raise ValueError(
                "set_missing treatment requires an explicit "
                "fallback to avoid reintroducing unresolved "
                "missing values."
            )

        return self


class TreatmentPlan(BaseModel):
    """Complete executable Phase 5 treatment plan."""

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
    )

    entries: dict[
        str,
        TreatmentPlanEntry,
    ] = Field(default_factory=dict)

    indicator_columns: dict[
        str,
        str,
    ] = Field(default_factory=dict)

    report_only_columns: list[str] = Field(
        default_factory=list
    )

    blocked_columns: list[str] = Field(
        default_factory=list
    )

    skipped_columns: list[str] = Field(
        default_factory=list
    )

    metadata: dict[str, Any] = Field(
        default_factory=dict
    )

    def entry(
        self,
        column: str,
    ) -> TreatmentPlanEntry:
        """Return one plan entry."""

        if column not in self.entries:
            raise KeyError(
                f"No treatment plan exists for {column!r}."
            )

        return self.entries[column]

    def executable_entries(
        self,
    ) -> list[TreatmentPlanEntry]:
        """Return entries eligible for execution."""

        return [
            entry
            for entry in self.entries.values()
            if entry.status
            == TreatmentPlanStatus.PLANNED
        ]

    def fitting_entries(
        self,
    ) -> list[TreatmentPlanEntry]:
        """Return entries requiring training-only fitting."""

        return [
            entry
            for entry
            in self.executable_entries()
            if entry.requires_training_fit
        ]


@dataclass(slots=True)
class TreatmentPlanOptions:
    """Options controlling treatment-plan construction."""

    enabled: bool = True

    default_detector: DetectorKind = (
        DetectorKind.IQR
    )

    default_treatment: TreatmentKind = (
        TreatmentKind.CLIP
    )

    create_entries_for_all_numeric_columns: bool = True

    add_indicators_by_default: bool = True
    indicator_suffix: str = "__was_outlier"

    allow_set_missing: bool = False
    allow_report_only: bool = True

    minimum_unique_count: int = 3

    excluded_columns: list[str] = field(
        default_factory=list
    )

    detector_overrides: dict[
        str,
        DetectorKind,
    ] = field(default_factory=dict)

    treatment_overrides: dict[
        str,
        TreatmentKind,
    ] = field(default_factory=dict)

    indicator_overrides: dict[
        str,
        bool,
    ] = field(default_factory=dict)

    domain_bounds: dict[
        str,
        tuple[
            float | None,
            float | None,
        ],
    ] = field(default_factory=dict)

    set_missing_fallbacks: dict[
        str,
        float,
    ] = field(default_factory=dict)

    priority_overrides: dict[
        str,
        float,
    ] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate treatment-plan options."""

        if self.minimum_unique_count < 2:
            raise ValueError(
                "minimum_unique_count must be at least two."
            )

        if not self.indicator_suffix:
            raise ValueError(
                "indicator_suffix cannot be empty."
            )

        self.detector_overrides = {
            str(column): (
                value
                if isinstance(
                    value,
                    DetectorKind,
                )
                else DetectorKind(value)
            )
            for column, value
            in self.detector_overrides.items()
        }

        self.treatment_overrides = {
            str(column): (
                value
                if isinstance(
                    value,
                    TreatmentKind,
                )
                else TreatmentKind(value)
            )
            for column, value
            in self.treatment_overrides.items()
        }

        for column, priority in (
            self.priority_overrides.items()
        ):
            if not 0.0 <= priority <= 1.0:
                raise ValueError(
                    f"Priority for {column!r} must be "
                    "between zero and one."
                )

        for column, bounds in (
            self.domain_bounds.items()
        ):
            if len(bounds) != 2:
                raise ValueError(
                    f"Domain bounds for {column!r} must "
                    "contain two values."
                )

            minimum, maximum = bounds

            if (
                minimum is not None
                and maximum is not None
                and minimum > maximum
            ):
                raise ValueError(
                    f"Invalid domain bounds for {column!r}."
                )


class TreatmentPlanBuilder:
    """Build an executable Phase 5 treatment plan."""

    component_name = "treatment_plan"

    def __init__(
        self,
        options: TreatmentPlanOptions | None = None,
    ) -> None:
        """Initialize the treatment-plan builder."""

        self.options = (
            options or TreatmentPlanOptions()
        )

    def build(
        self,
        *,
        contract: OutlierContract,
    ) -> TreatmentPlan:
        """Build the complete treatment plan."""

        if not self.options.enabled:
            return TreatmentPlan(
                metadata={
                    "enabled": False,
                }
            )

        excluded_columns = set(
            self.options.excluded_columns
        )

        entries: dict[
            str,
            TreatmentPlanEntry,
        ] = {}

        skipped_columns: list[str] = []
        blocked_columns: list[str] = []

        for column in contract.numeric_columns:
            if column in excluded_columns:
                skipped_columns.append(column)
                continue

            training_series = (
                contract.train.X[column]
            )

            unique_count = int(
                training_series.nunique(
                    dropna=True
                )
            )

            if (
                unique_count
                < self.options.minimum_unique_count
            ):
                skipped_columns.append(column)
                continue

            entry = self._build_entry(
                column=column,
                unique_count=unique_count,
            )

            if (
                entry.status
                == TreatmentPlanStatus.BLOCKED
            ):
                blocked_columns.append(
                    column
                )

            entries[column] = entry

        indicator_columns = (
            self._build_indicator_mapping(
                contract=contract,
                entries=entries,
            )
        )

        report_only_columns = [
            column
            for column, entry
            in entries.items()
            if entry.treatment
            == TreatmentKind.REPORT_ONLY
        ]

        return TreatmentPlan(
            entries=entries,
            indicator_columns=indicator_columns,
            report_only_columns=(
                report_only_columns
            ),
            blocked_columns=blocked_columns,
            skipped_columns=skipped_columns,
            metadata={
                "enabled": True,
                "entry_count": len(entries),
                "fitting_entry_count": len(
                    [
                        entry
                        for entry in entries.values()
                        if entry.requires_training_fit
                    ]
                ),
                "indicator_count": len(
                    indicator_columns
                ),
                "report_only_count": len(
                    report_only_columns
                ),
                "blocked_count": len(
                    blocked_columns
                ),
            },
        )

    def _build_entry(
        self,
        *,
        column: str,
        unique_count: int,
    ) -> TreatmentPlanEntry:
        """Build one feature treatment entry."""

        detector = (
            self.options.detector_overrides.get(
                column,
                self.options.default_detector,
            )
        )

        treatment = (
            self.options.treatment_overrides.get(
                column,
                self.options.default_treatment,
            )
        )

        add_indicator = (
            self.options.indicator_overrides.get(
                column,
                self.options
                .add_indicators_by_default,
            )
        )

        status = TreatmentPlanStatus.PLANNED
        warnings: list[str] = []

        if (
            treatment == TreatmentKind.SET_MISSING
            and not self.options.allow_set_missing
        ):
            status = TreatmentPlanStatus.BLOCKED

            warnings.append(
                "set_missing treatment is disabled."
            )

        if (
            treatment == TreatmentKind.REPORT_ONLY
            and not self.options.allow_report_only
        ):
            status = TreatmentPlanStatus.BLOCKED

            warnings.append(
                "report_only treatment is disabled."
            )

        domain_minimum = None
        domain_maximum = None

        if detector == DetectorKind.DOMAIN:
            bounds = self.options.domain_bounds.get(
                column
            )

            if bounds is None:
                status = TreatmentPlanStatus.BLOCKED

                warnings.append(
                    "Domain detection requires explicitly "
                    "configured feature bounds."
                )

            else:
                (
                    domain_minimum,
                    domain_maximum,
                ) = bounds

        set_missing_fallback = (
            self.options
            .set_missing_fallbacks
            .get(column)
        )

        if (
            treatment == TreatmentKind.SET_MISSING
            and set_missing_fallback is None
        ):
            status = TreatmentPlanStatus.BLOCKED

            warnings.append(
                "set_missing requires an explicit fallback."
            )

        priority = (
            self.options.priority_overrides.get(
                column,
                0.50,
            )
        )

        requires_training_fit = (
            detector
            in {
                DetectorKind.IQR,
                DetectorKind.QUANTILE,
                DetectorKind.ROBUST_ZSCORE,
            }
        )

        return TreatmentPlanEntry(
            column=column,
            detector=detector,
            treatment=treatment,
            add_indicator=add_indicator,
            indicator_suffix=(
                self.options.indicator_suffix
            ),
            domain_minimum=domain_minimum,
            domain_maximum=domain_maximum,
            set_missing_fallback=(
                set_missing_fallback
            ),
            requires_training_fit=(
                requires_training_fit
            ),
            status=status,
            priority=priority,
            reason=self._reason(
                detector=detector,
                treatment=treatment,
            ),
            warnings=warnings,
            metadata={
                "training_unique_count": (
                    unique_count
                ),
                "source": (
                    "override"
                    if (
                        column
                        in self.options
                        .detector_overrides
                        or column
                        in self.options
                        .treatment_overrides
                    )
                    else "default"
                ),
            },
        )

    @staticmethod
    def _reason(
        *,
        detector: DetectorKind,
        treatment: TreatmentKind,
    ) -> str:
        """Build a readable treatment reason."""

        detector_reasons = {
            DetectorKind.IQR: (
                "Training IQR bounds identify values "
                "outside the central distribution."
            ),
            DetectorKind.QUANTILE: (
                "Training quantile bounds identify extreme "
                "lower and upper tail values."
            ),
            DetectorKind.ROBUST_ZSCORE: (
                "Training median and MAD identify robust "
                "standardized extremes."
            ),
            DetectorKind.DOMAIN: (
                "Explicit domain bounds identify values "
                "outside permitted limits."
            ),
        }

        treatment_reasons = {
            TreatmentKind.CLIP: (
                "Detected values will be constrained to "
                "the fitted bounds."
            ),
            TreatmentKind.WINSORIZE: (
                "Detected tail values will be replaced "
                "with fitted boundary values."
            ),
            TreatmentKind.SET_MISSING: (
                "Detected values will be replaced through "
                "an explicit fallback policy."
            ),
            TreatmentKind.REPORT_ONLY: (
                "Detected values will be reported without "
                "modification."
            ),
        }

        return (
            f"{detector_reasons[detector]} "
            f"{treatment_reasons[treatment]}"
        )

    @staticmethod
    def _build_indicator_mapping(
        *,
        contract: OutlierContract,
        entries: dict[
            str,
            TreatmentPlanEntry,
        ],
    ) -> dict[str, str]:
        """Build collision-safe outlier indicator names."""

        existing_columns = set(
            contract.feature_columns
        )

        reserved_names: set[str] = set()

        mapping: dict[str, str] = {}

        for column, entry in entries.items():
            if not entry.add_indicator:
                continue

            indicator_name = (
                f"{column}"
                f"{entry.indicator_suffix}"
            )

            if (
                indicator_name in existing_columns
                or indicator_name in reserved_names
            ):
                raise ValueError(
                    "Outlier-indicator name collision for "
                    f"feature {column!r}: "
                    f"{indicator_name!r}"
                )

            mapping[column] = indicator_name
            reserved_names.add(
                indicator_name
            )

        return mapping