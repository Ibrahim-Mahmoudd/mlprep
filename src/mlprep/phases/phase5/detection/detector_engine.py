"""Unified detector engine for Phase 5."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from mlprep.phases.phase2.output_governance.metadata import (
    PartitionName,
)
from mlprep.phases.phase5.contract.treatment_plan import (
    DetectorKind,
    TreatmentPlan,
    TreatmentPlanEntry,
)
from mlprep.phases.phase5.detection.domain_detector import (
    DomainDetector,
    DomainDetectorOptions,
)
from mlprep.phases.phase5.detection.iqr_detector import (
    IQRDetector,
    IQRDetectorOptions,
)
from mlprep.phases.phase5.detection.quantile_detector import (
    QuantileDetector,
    QuantileDetectorOptions,
)
from mlprep.phases.phase5.detection.robust_zscore_detector import (
    RobustZScoreDetector,
    RobustZScoreDetectorOptions,
)


DetectorType = (
    IQRDetector
    | QuantileDetector
    | RobustZScoreDetector
    | DomainDetector
)


class ColumnDetectionSummary(BaseModel):
    """Detection summary for one feature and partition."""

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
    )

    column: str
    partition: PartitionName

    detector: DetectorKind

    row_count: int = Field(
        default=0,
        ge=0,
    )

    valid_count: int = Field(
        default=0,
        ge=0,
    )

    outlier_count: int = Field(
        default=0,
        ge=0,
    )

    outlier_percentage: float = Field(
        default=0.0,
        ge=0.0,
        le=100.0,
    )

    lower_bound: float | None = None
    upper_bound: float | None = None

    metadata: dict[str, Any] = Field(
        default_factory=dict
    )


class DetectionResult(BaseModel):
    """Complete outlier-detection output for one partition."""

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        arbitrary_types_allowed=True,
    )

    partition: PartitionName

    masks: dict[
        str,
        pd.Series,
    ] = Field(default_factory=dict)

    summaries: dict[
        str,
        ColumnDetectionSummary,
    ] = Field(default_factory=dict)

    total_outlier_count: int = Field(
        default=0,
        ge=0,
    )

    columns_with_outliers: list[str] = Field(
        default_factory=list
    )

    metadata: dict[str, Any] = Field(
        default_factory=dict
    )

    def mask(
        self,
        column: str,
    ) -> pd.Series:
        """Return the outlier mask for one feature."""

        if column not in self.masks:
            raise KeyError(
                f"No outlier mask exists for {column!r}."
            )

        return self.masks[column].copy(
            deep=True
        )


@dataclass(slots=True)
class DetectorEngineOptions:
    """Options controlling the unified detector engine."""

    iqr: IQRDetectorOptions = field(
        default_factory=IQRDetectorOptions
    )

    quantile: QuantileDetectorOptions = field(
        default_factory=QuantileDetectorOptions
    )

    robust_zscore: (
        RobustZScoreDetectorOptions
    ) = field(
        default_factory=(
            RobustZScoreDetectorOptions
        )
    )

    fail_on_missing_column: bool = True

    require_identical_index: bool = True

    def __post_init__(self) -> None:
        """Validate detector-engine options."""

        if not isinstance(
            self.fail_on_missing_column,
            bool,
        ):
            raise TypeError(
                "fail_on_missing_column must be Boolean."
            )

        if not isinstance(
            self.require_identical_index,
            bool,
        ):
            raise TypeError(
                "require_identical_index must be Boolean."
            )


class DetectorEngine:
    """Fit configured detectors and detect every partition."""

    component_name = "detector_engine"

    def __init__(
        self,
        options: DetectorEngineOptions | None = None,
    ) -> None:
        """Initialize the detector engine."""

        self.options = (
            options or DetectorEngineOptions()
        )

        self.detectors: dict[
            str,
            DetectorType,
        ] = {}

        self._fitted = False

    @property
    def is_fitted(
        self,
    ) -> bool:
        """Return whether the engine is fitted."""

        return self._fitted

    def fit(
        self,
        *,
        frame: pd.DataFrame,
        plan: TreatmentPlan,
    ) -> DetectorEngine:
        """Fit all configured detectors on training data."""

        self._validate_frame(
            frame
        )

        if self._fitted:
            raise RuntimeError(
                "The detector engine is already fitted."
            )

        executable_entries = (
            plan.executable_entries()
        )

        for entry in executable_entries:
            if entry.column not in frame.columns:
                if self.options.fail_on_missing_column:
                    raise KeyError(
                        f"Feature {entry.column!r} is "
                        "missing from training data."
                    )

                continue

            detector = self._build_detector(
                entry
            )

            detector.fit(
                frame[entry.column]
            )

            self.detectors[
                entry.column
            ] = detector

        self._fitted = True

        return self

    def detect(
        self,
        *,
        frame: pd.DataFrame,
        partition: PartitionName,
    ) -> DetectionResult:
        """Detect outliers in one partition."""

        self._validate_frame(
            frame
        )

        self._assert_fitted()

        masks: dict[
            str,
            pd.Series,
        ] = {}

        summaries: dict[
            str,
            ColumnDetectionSummary,
        ] = {}

        total_outlier_count = 0

        columns_with_outliers: list[str] = []

        for column, detector in (
            self.detectors.items()
        ):
            if column not in frame.columns:
                if self.options.fail_on_missing_column:
                    raise KeyError(
                        f"Feature {column!r} is missing "
                        f"from {partition.value!r}."
                    )

                continue

            mask = detector.detect(
                frame[column]
            )

            if (
                self.options
                .require_identical_index
                and not mask.index.equals(
                    frame.index
                )
            ):
                raise ValueError(
                    f"Detector for {column!r} changed "
                    "the row index."
                )

            mask = mask.astype(
                bool
            )

            detector_state = (
                detector.get_state()
            )

            outlier_count = int(
                mask.sum()
            )

            total_outlier_count += (
                outlier_count
            )

            if outlier_count > 0:
                columns_with_outliers.append(
                    column
                )

            valid_count = int(
                pd.to_numeric(
                    frame[column],
                    errors="coerce",
                ).notna().sum()
            )

            masks[column] = mask

            summaries[column] = (
                ColumnDetectionSummary(
                    column=column,
                    partition=partition,
                    detector=DetectorKind(
                        detector.detector_name
                    ),
                    row_count=len(frame),
                    valid_count=valid_count,
                    outlier_count=outlier_count,
                    outlier_percentage=(
                        outlier_count
                        / len(frame)
                        * 100.0
                        if len(frame) > 0
                        else 0.0
                    ),
                    lower_bound=getattr(
                        detector_state,
                        "lower_bound",
                        None,
                    ),
                    upper_bound=getattr(
                        detector_state,
                        "upper_bound",
                        None,
                    ),
                )
            )

        return DetectionResult(
            partition=partition,
            masks=masks,
            summaries=summaries,
            total_outlier_count=(
                total_outlier_count
            ),
            columns_with_outliers=(
                columns_with_outliers
            ),
            metadata={
                "fitted_on_training_only": True,
                "detector_count": len(
                    self.detectors
                ),
            },
        )

    def fit_detect(
        self,
        *,
        training_frame: pd.DataFrame,
        plan: TreatmentPlan,
    ) -> DetectionResult:
        """Fit detectors and detect training outliers."""

        self.fit(
            frame=training_frame,
            plan=plan,
        )

        return self.detect(
            frame=training_frame,
            partition=PartitionName.TRAIN,
        )

    def states(
        self,
    ) -> dict[
        str,
        dict[str, Any],
    ]:
        """Return serialized fitted detector states."""

        self._assert_fitted()

        return {
            column: detector
            .get_state()
            .model_dump(
                mode="json"
            )
            for column, detector
            in self.detectors.items()
        }

    def detector(
        self,
        column: str,
    ) -> DetectorType:
        """Return one fitted detector."""

        self._assert_fitted()

        if column not in self.detectors:
            raise KeyError(
                f"No detector exists for {column!r}."
            )

        return self.detectors[column]

    def _build_detector(
        self,
        entry: TreatmentPlanEntry,
    ) -> DetectorType:
        """Build one detector from a treatment entry."""

        if entry.detector == DetectorKind.IQR:
            return IQRDetector(
                column=entry.column,
                options=self.options.iqr,
            )

        if (
            entry.detector
            == DetectorKind.QUANTILE
        ):
            return QuantileDetector(
                column=entry.column,
                options=self.options.quantile,
            )

        if (
            entry.detector
            == DetectorKind.ROBUST_ZSCORE
        ):
            return RobustZScoreDetector(
                column=entry.column,
                options=(
                    self.options.robust_zscore
                ),
            )

        if (
            entry.detector
            == DetectorKind.DOMAIN
        ):
            return DomainDetector(
                column=entry.column,
                options=DomainDetectorOptions(
                    minimum=(
                        entry.domain_minimum
                    ),
                    maximum=(
                        entry.domain_maximum
                    ),
                ),
            )

        raise ValueError(
            f"Unsupported detector: "
            f"{entry.detector!r}"
        )

    def _assert_fitted(
        self,
    ) -> None:
        """Raise when the engine is not fitted."""

        if not self._fitted:
            raise RuntimeError(
                "The detector engine is not fitted."
            )

    @staticmethod
    def _validate_frame(
        frame: pd.DataFrame,
    ) -> None:
        """Validate an input feature frame."""

        if not isinstance(
            frame,
            pd.DataFrame,
        ):
            raise TypeError(
                "Detector-engine input must be "
                "a pandas DataFrame."
            )

        columns = [
            str(column)
            for column in frame.columns
        ]

        if len(columns) != len(
            set(columns)
        ):
            raise ValueError(
                "The feature frame contains duplicate "
                "column names."
            )