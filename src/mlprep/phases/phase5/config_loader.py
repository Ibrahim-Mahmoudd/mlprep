"""Configuration loading utilities for Phase 5."""

from __future__ import annotations

import json
from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any, ClassVar, Literal

import yaml

from mlprep.phases.phase5.config import (
    Phase5Config,
    Phase5ExecutionOptions,
    Phase5ValidationOptions,
)
from mlprep.phases.phase5.contract.outlier_contract import (
    OutlierContractOptions,
)
from mlprep.phases.phase5.contract.treatment_plan import (
    DetectorKind,
    TreatmentKind,
    TreatmentPlanOptions,
)
from mlprep.phases.phase5.detection.detector_engine import (
    DetectorEngineOptions,
)
from mlprep.phases.phase5.detection.iqr_detector import (
    IQRDetectorOptions,
)
from mlprep.phases.phase5.detection.quantile_detector import (
    QuantileDetectorOptions,
)
from mlprep.phases.phase5.detection.robust_zscore_detector import (
    RobustZScoreDetectorOptions,
)
from mlprep.phases.phase5.orchestration.outlier_pipeline import (
    OutlierPipelineOptions,
)
from mlprep.phases.phase5.orchestration.partition_transformer import (
    PartitionTransformerOptions,
)
from mlprep.phases.phase5.output_governance.readiness import (
    Phase5ReadinessOptions,
)
from mlprep.phases.phase5.output_governance.report import (
    Phase5ReportOptions,
)
from mlprep.phases.phase5.strategies.winsorize_strategy import (
    WinsorizeStrategyOptions,
)
from mlprep.phases.phase5.validation.distribution_impact import (
    DistributionImpactOptions,
)
from mlprep.phases.phase5.validation.remaining_outliers import (
    RemainingOutliersOptions,
)
from mlprep.phases.phase5.validation.row_integrity import (
    RowIntegrityOptions,
)
from mlprep.phases.phase5.validation.schema_consistency import (
    SchemaConsistencyOptions,
)
from mlprep.phases.phase5.validation.target_integrity import (
    TargetIntegrityOptions,
)

ConfigFormat = Literal[
    "yaml",
    "json",
]


class Phase5ConfigLoader:
    """Load, merge, validate, and save Phase 5 configurations."""

    supported_extensions: ClassVar[frozenset[str]] = frozenset({
        ".yaml",
        ".yml",
        ".json",
    })

    allowed_sections: ClassVar[frozenset[str]] = frozenset({
        "execution",
        "contract",
        "plan",
        "detector_engine",
        "partition_transformer",
        "pipeline",
        "validation",
        "readiness",
        "reporting",
    })

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        overrides: Mapping[str, Any] | None = None,
        dotted_overrides: Mapping[
            str,
            Any,
        ] | None = None,
    ) -> Phase5Config:
        """Load Phase 5 configuration."""

        values = cls.load_raw(
            path
        )

        if overrides is not None:
            values = cls.merge(
                base=values,
                override=overrides,
            )

        if dotted_overrides is not None:
            values = cls.apply_dot_overrides(
                values=values,
                overrides=dotted_overrides,
            )

        return cls.from_dict(
            values
        )

    @classmethod
    def load_raw(
        cls,
        path: str | Path,
    ) -> dict[str, Any]:
        """Load raw YAML or JSON configuration."""

        input_path = Path(
            path
        )

        if not input_path.exists():
            raise FileNotFoundError(
                f"Configuration does not exist: "
                f"{input_path}"
            )

        if not input_path.is_file():
            raise IsADirectoryError(
                f"Configuration path is not a file: "
                f"{input_path}"
            )

        extension = input_path.suffix.casefold()

        if extension not in cls.supported_extensions:
            raise ValueError(
                "Unsupported configuration extension: "
                f"{extension!r}"
            )

        content = input_path.read_text(
            encoding="utf-8"
        )

        if extension in {
            ".yaml",
            ".yml",
        }:
            try:
                values = yaml.safe_load(
                    content
                )

            except yaml.YAMLError as error:
                raise ValueError(
                    f"Invalid YAML configuration: {error}"
                ) from error

        else:
            try:
                values = json.loads(
                    content
                )

            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON configuration: {error}"
                ) from error

        if values is None:
            raise ValueError(
                f"Configuration is empty: {input_path}"
            )

        if not isinstance(
            values,
            dict,
        ):
            raise TypeError(
                "Configuration root must be a mapping."
            )

        return values

    @classmethod
    def from_dict(
        cls,
        values: Mapping[str, Any],
    ) -> Phase5Config:
        """Create Phase 5 configuration from a mapping."""

        root = deepcopy(
            dict(values)
        )

        unknown_sections = set(
            root
        ).difference(
            cls.allowed_sections
        )

        if unknown_sections:
            raise ValueError(
                "Unknown Phase 5 configuration sections: "
                f"{sorted(unknown_sections)}"
            )

        execution = Phase5ExecutionOptions(
            **cls._mapping(
                root.get(
                    "execution",
                    {},
                ),
                section="execution",
            )
        )

        contract = OutlierContractOptions(
            **cls._mapping(
                root.get(
                    "contract",
                    {},
                ),
                section="contract",
            )
        )

        plan = cls._build_plan(
            root.get(
                "plan",
                {},
            )
        )

        detector_engine = (
            cls._build_detector_engine(
                root.get(
                    "detector_engine",
                    {},
                )
            )
        )

        partition_transformer = (
            PartitionTransformerOptions(
                **cls._mapping(
                    root.get(
                        "partition_transformer",
                        {},
                    ),
                    section=(
                        "partition_transformer"
                    ),
                )
            )
        )

        pipeline = cls._build_pipeline(
            values=root.get(
                "pipeline",
                {},
            ),
            detector_engine=detector_engine,
            partition_transformer=(
                partition_transformer
            ),
        )

        validation = cls._build_validation(
            root.get(
                "validation",
                {},
            )
        )

        readiness = Phase5ReadinessOptions(
            **cls._mapping(
                root.get(
                    "readiness",
                    {},
                ),
                section="readiness",
            )
        )

        reporting = Phase5ReportOptions(
            **cls._mapping(
                root.get(
                    "reporting",
                    {},
                ),
                section="reporting",
            )
        )

        return Phase5Config(
            execution=execution,
            contract=contract,
            plan=plan,
            detector_engine=detector_engine,
            partition_transformer=(
                partition_transformer
            ),
            pipeline=pipeline,
            validation=validation,
            readiness=readiness,
            reporting=reporting,
        )

    @classmethod
    def _build_plan(
        cls,
        values: object,
    ) -> TreatmentPlanOptions:
        """Build treatment-plan options."""

        data = cls._mapping(
            values,
            section="plan",
        )

        if "default_detector" in data:
            data["default_detector"] = (
                DetectorKind(
                    data["default_detector"]
                )
            )

        if "default_treatment" in data:
            data["default_treatment"] = (
                TreatmentKind(
                    data["default_treatment"]
                )
            )

        detector_overrides = data.get(
            "detector_overrides"
        )

        if detector_overrides is not None:
            detector_mapping = cls._mapping(
                detector_overrides,
                section=(
                    "plan.detector_overrides"
                ),
            )

            data["detector_overrides"] = {
                str(column): DetectorKind(
                    detector
                )
                for column, detector
                in detector_mapping.items()
            }

        treatment_overrides = data.get(
            "treatment_overrides"
        )

        if treatment_overrides is not None:
            treatment_mapping = cls._mapping(
                treatment_overrides,
                section=(
                    "plan.treatment_overrides"
                ),
            )

            data["treatment_overrides"] = {
                str(column): TreatmentKind(
                    treatment
                )
                for column, treatment
                in treatment_mapping.items()
            }

        raw_domain_bounds = data.get(
            "domain_bounds"
        )

        if raw_domain_bounds is not None:
            domain_mapping = cls._mapping(
                raw_domain_bounds,
                section="plan.domain_bounds",
            )

            normalized_bounds: dict[
                str,
                tuple[
                    float | None,
                    float | None,
                ],
            ] = {}

            for column, raw_bounds in (
                domain_mapping.items()
            ):
                if not isinstance(
                    raw_bounds,
                    (list, tuple),
                ):
                    raise TypeError(
                        "Each domain bound must be a "
                        "two-value list or tuple."
                    )

                if len(raw_bounds) != 2:
                    raise ValueError(
                        "Each domain bound must contain "
                        "exactly two values."
                    )

                minimum = (
                    None
                    if raw_bounds[0] is None
                    else float(raw_bounds[0])
                )

                maximum = (
                    None
                    if raw_bounds[1] is None
                    else float(raw_bounds[1])
                )

                normalized_bounds[
                    str(column)
                ] = (
                    minimum,
                    maximum,
                )

            data["domain_bounds"] = (
                normalized_bounds
            )

        return TreatmentPlanOptions(
            **data
        )

    @classmethod
    def _build_detector_engine(
        cls,
        values: object,
    ) -> DetectorEngineOptions:
        """Build detector-engine options."""

        data = cls._mapping(
            values,
            section="detector_engine",
        )

        iqr = IQRDetectorOptions(
            **cls._mapping(
                data.pop(
                    "iqr",
                    {},
                ),
                section="detector_engine.iqr",
            )
        )

        quantile = QuantileDetectorOptions(
            **cls._mapping(
                data.pop(
                    "quantile",
                    {},
                ),
                section=(
                    "detector_engine.quantile"
                ),
            )
        )

        robust_zscore = (
            RobustZScoreDetectorOptions(
                **cls._mapping(
                    data.pop(
                        "robust_zscore",
                        {},
                    ),
                    section=(
                        "detector_engine."
                        "robust_zscore"
                    ),
                )
            )
        )

        fail_on_missing_column = data.pop(
            "fail_on_missing_column",
            True,
        )
        require_identical_index = data.pop(
            "require_identical_index",
            True,
        )

        if data:
            raise ValueError(
                "Unknown detector-engine options: "
                f"{sorted(data)}"
            )

        return DetectorEngineOptions(
            iqr=iqr,
            quantile=quantile,
            robust_zscore=robust_zscore,
            fail_on_missing_column=fail_on_missing_column,
            require_identical_index=require_identical_index,
        )

    @classmethod
    def _build_pipeline(
        cls,
        *,
        values: object,
        detector_engine: DetectorEngineOptions,
        partition_transformer: (
            PartitionTransformerOptions
        ),
    ) -> OutlierPipelineOptions:
        """Build outlier-pipeline options."""

        data = cls._mapping(
            values,
            section="pipeline",
        )

        winsorization = (
            WinsorizeStrategyOptions(
                **cls._mapping(
                    data.pop(
                        "winsorization",
                        {},
                    ),
                    section=(
                        "pipeline.winsorization"
                    ),
                )
            )
        )

        if "detector_engine" in data:
            nested_detector = (
                cls._build_detector_engine(
                    data.pop(
                        "detector_engine"
                    )
                )
            )

        else:
            nested_detector = (
                detector_engine
            )

        if "partition_transformer" in data:
            nested_transformer = (
                PartitionTransformerOptions(
                    **cls._mapping(
                        data.pop(
                            "partition_transformer"
                        ),
                        section=(
                            "pipeline."
                            "partition_transformer"
                        ),
                    )
                )
            )

        else:
            nested_transformer = (
                partition_transformer
            )

        allowed_scalar_fields = {
            "require_nonempty_plan",
            "require_all_strategies_fitted",
            "test_transform_authorized",
        }

        unknown = set(
            data
        ).difference(
            allowed_scalar_fields
        )

        if unknown:
            raise ValueError(
                "Unknown pipeline options: "
                f"{sorted(unknown)}"
            )

        return OutlierPipelineOptions(
            detector_engine=(
                nested_detector
            ),
            partition_transformer=(
                nested_transformer
            ),
            winsorization=winsorization,
            **data,
        )

    @classmethod
    def _build_validation(
        cls,
        values: object,
    ) -> Phase5ValidationOptions:
        """Build validation options."""

        data = cls._mapping(
            values,
            section="validation",
        )

        allowed = {
            "remaining_outliers",
            "schema_consistency",
            "target_integrity",
            "row_integrity",
            "distribution_impact",
        }

        unknown = set(
            data
        ).difference(
            allowed
        )

        if unknown:
            raise ValueError(
                "Unknown validation options: "
                f"{sorted(unknown)}"
            )

        return Phase5ValidationOptions(
            remaining_outliers=(
                RemainingOutliersOptions(
                    **cls._mapping(
                        data.get(
                            "remaining_outliers",
                            {},
                        ),
                        section=(
                            "validation."
                            "remaining_outliers"
                        ),
                    )
                )
            ),
            schema_consistency=(
                SchemaConsistencyOptions(
                    **cls._mapping(
                        data.get(
                            "schema_consistency",
                            {},
                        ),
                        section=(
                            "validation."
                            "schema_consistency"
                        ),
                    )
                )
            ),
            target_integrity=(
                TargetIntegrityOptions(
                    **cls._mapping(
                        data.get(
                            "target_integrity",
                            {},
                        ),
                        section=(
                            "validation."
                            "target_integrity"
                        ),
                    )
                )
            ),
            row_integrity=(
                RowIntegrityOptions(
                    **cls._mapping(
                        data.get(
                            "row_integrity",
                            {},
                        ),
                        section=(
                            "validation."
                            "row_integrity"
                        ),
                    )
                )
            ),
            distribution_impact=(
                DistributionImpactOptions(
                    **cls._mapping(
                        data.get(
                            "distribution_impact",
                            {},
                        ),
                        section=(
                            "validation."
                            "distribution_impact"
                        ),
                    )
                )
            ),
        )

    @classmethod
    def save(
        cls,
        config: Phase5Config,
        path: str | Path,
        *,
        config_format: ConfigFormat | None = None,
        overwrite: bool = False,
    ) -> Path:
        """Save a resolved Phase 5 configuration."""

        if not isinstance(
            config,
            Phase5Config,
        ):
            raise TypeError(
                "config must be a Phase5Config."
            )

        output_path = Path(
            path
        )

        if (
            output_path.exists()
            and not overwrite
        ):
            raise FileExistsError(
                f"Configuration already exists: "
                f"{output_path}"
            )

        resolved_format = (
            config_format
            or cls._format_from_extension(
                output_path
            )
        )

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        values = config.to_dict()

        if resolved_format == "yaml":
            content = yaml.safe_dump(
                values,
                sort_keys=False,
                allow_unicode=True,
            )

        else:
            content = json.dumps(
                values,
                indent=2,
                ensure_ascii=False,
                default=str,
            )

        output_path.write_text(
            content,
            encoding="utf-8",
        )

        return output_path

    @classmethod
    def merge(
        cls,
        *,
        base: Mapping[str, Any],
        override: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Recursively merge configuration mappings."""

        merged = deepcopy(
            dict(base)
        )

        for key, override_value in (
            override.items()
        ):
            existing_value = merged.get(
                key
            )

            if (
                isinstance(
                    existing_value,
                    Mapping,
                )
                and isinstance(
                    override_value,
                    Mapping,
                )
            ):
                merged[key] = cls.merge(
                    base=existing_value,
                    override=override_value,
                )

            else:
                merged[key] = deepcopy(
                    override_value
                )

        return merged

    @classmethod
    def apply_dot_overrides(
        cls,
        *,
        values: Mapping[str, Any],
        overrides: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Apply dotted configuration overrides."""

        updated = deepcopy(
            dict(values)
        )

        for dotted_path, value in (
            overrides.items()
        ):
            cls._set_dotted_value(
                values=updated,
                dotted_path=dotted_path,
                value=value,
            )

        return updated

    @staticmethod
    def _mapping(
        value: object,
        *,
        section: str,
    ) -> dict[str, Any]:
        """Validate and copy one mapping."""

        if value is None:
            return {}

        if not isinstance(
            value,
            Mapping,
        ):
            raise TypeError(
                f"Configuration section {section!r} "
                "must be a mapping."
            )

        return deepcopy(
            dict(value)
        )

    @staticmethod
    def _format_from_extension(
        path: Path,
    ) -> ConfigFormat:
        """Resolve output format from the extension."""

        extension = path.suffix.casefold()

        if extension in {
            ".yaml",
            ".yml",
        }:
            return "yaml"

        if extension == ".json":
            return "json"

        raise ValueError(
            "Cannot determine configuration format "
            f"from extension {extension!r}."
        )

    @staticmethod
    def _set_dotted_value(
        *,
        values: dict[str, Any],
        dotted_path: str,
        value: Any,
    ) -> None:
        """Set a nested value using a dotted path."""

        parts = dotted_path.split(
            "."
        )

        if (
            not dotted_path.strip()
            or any(
                not part.strip()
                for part in parts
            )
        ):
            raise ValueError(
                f"Invalid dotted path: "
                f"{dotted_path!r}"
            )

        current = values

        for part in parts[:-1]:
            existing = current.get(
                part
            )

            if existing is None:
                created: dict[
                    str,
                    Any,
                ] = {}

                current[part] = created
                current = created

                continue

            if not isinstance(
                existing,
                dict,
            ):
                raise TypeError(
                    f"Cannot set {dotted_path!r}; "
                    f"{part!r} is not a mapping."
                )

            current = existing

        current[
            parts[-1]
        ] = deepcopy(
            value
        )
