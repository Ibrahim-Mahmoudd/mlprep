"""Outlier-detection components for Phase 5."""

from mlprep.phases.phase5.detection.detector_engine import (
    ColumnDetectionSummary,
    DetectionResult,
    DetectorEngine,
    DetectorEngineOptions,
)
from mlprep.phases.phase5.detection.domain_detector import (
    DomainDetector,
    DomainDetectorOptions,
    DomainDetectorState,
)
from mlprep.phases.phase5.detection.iqr_detector import (
    IQRDetector,
    IQRDetectorOptions,
    IQRDetectorState,
)
from mlprep.phases.phase5.detection.quantile_detector import (
    QuantileDetector,
    QuantileDetectorOptions,
    QuantileDetectorState,
)
from mlprep.phases.phase5.detection.robust_zscore_detector import (
    RobustZScoreDetector,
    RobustZScoreDetectorOptions,
    RobustZScoreDetectorState,
)

ColumnDetectionResult = ColumnDetectionSummary

__all__ = [
    "ColumnDetectionSummary",
    "ColumnDetectionResult",
    "DetectionResult",
    "DetectorEngine",
    "DetectorEngineOptions",
    "DomainDetector",
    "DomainDetectorOptions",
    "DomainDetectorState",
    "IQRDetector",
    "IQRDetectorOptions",
    "IQRDetectorState",
    "QuantileDetector",
    "QuantileDetectorOptions",
    "QuantileDetectorState",
    "RobustZScoreDetector",
    "RobustZScoreDetectorOptions",
    "RobustZScoreDetectorState",
]
