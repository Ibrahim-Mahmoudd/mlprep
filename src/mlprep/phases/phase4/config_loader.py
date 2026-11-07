"""Configuration loading utilities for Phase 4."""

from __future__ import annotations

import json
from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any, ClassVar, Literal

import yaml

from mlprep.phases.phase4.config import (
    Phase4Config,
    Phase4ExecutionOptions,
    Phase4ValidationOptions,
)
from mlprep.phases.phase4.contract.imputation_contract import ImputationContractOptions
from mlprep.phases.phase4.contract.imputation_plan import (
    ImputationMethodName,
    ImputationPlanOptions,
)
from mlprep.phases.phase4.detection.invalid_value_scanner import (
    InvalidValueRule,
    InvalidValueScannerOptions,
)
from mlprep.phases.phase4.detection.missing_token_detector import (
    MissingTokenDetectorOptions,
)
from mlprep.phases.phase4.detection.missingness_scanner import MissingnessScannerOptions
from mlprep.phases.phase4.indicators.indicator_naming import (
    IndicatorCollisionPolicy,
    IndicatorNamingOptions,
)
from mlprep.phases.phase4.indicators.missing_indicator import MissingIndicatorOptions
from mlprep.phases.phase4.orchestration.imputation_pipeline import (
    ImputationPipelineOptions,
)
from mlprep.phases.phase4.orchestration.partition_transformer import (
    PartitionTransformerOptions,
)
from mlprep.phases.phase4.output_governance.readiness import Phase4ReadinessOptions
from mlprep.phases.phase4.output_governance.report import Phase4ReportOptions
from mlprep.phases.phase4.strategies.boolean_imputer import BooleanImputerOptions
from mlprep.phases.phase4.strategies.categorical_imputer import (
    CategoricalImputerOptions,
)
from mlprep.phases.phase4.strategies.datetime_imputer import DatetimeImputerOptions
from mlprep.phases.phase4.strategies.group_imputer import GroupImputerOptions
from mlprep.phases.phase4.strategies.model_imputer import ModelImputerOptions
from mlprep.phases.phase4.strategies.numeric_imputer import NumericImputerOptions
from mlprep.phases.phase4.strategies.temporal_imputer import TemporalImputerOptions
from mlprep.phases.phase4.strategies.text_imputer import TextImputerOptions
from mlprep.phases.phase4.validation.distribution_impact import (
    DistributionImpactOptions,
)
from mlprep.phases.phase4.validation.remaining_missingness import (
    RemainingMissingnessOptions,
)
from mlprep.phases.phase4.validation.row_integrity import RowIntegrityOptions
from mlprep.phases.phase4.validation.schema_consistency import SchemaConsistencyOptions
from mlprep.phases.phase4.validation.target_integrity import TargetIntegrityOptions

ConfigFormat = Literal["yaml", "json"]


class Phase4ConfigLoader:
    """Load, merge, validate, and save Phase 4 configurations."""

    supported_extensions: ClassVar[frozenset[str]] = frozenset(
        {".yaml", ".yml", ".json"}
    )
    allowed_sections: ClassVar[frozenset[str]] = frozenset({
        "execution", "contract", "plan", "missingness_scanner",
        "pipeline", "validation", "readiness", "reporting",
    })

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        overrides: Mapping[str, Any] | None = None,
        dotted_overrides: Mapping[str, Any] | None = None,
    ) -> Phase4Config:
        values = cls.load_raw(path)
        if overrides is not None:
            values = cls.merge(base=values, override=overrides)
        if dotted_overrides is not None:
            values = cls.apply_dot_overrides(values=values, overrides=dotted_overrides)
        return cls.from_dict(values)

    @classmethod
    def load_raw(cls, path: str | Path) -> dict[str, Any]:
        input_path = Path(path)
        if not input_path.exists():
            raise FileNotFoundError(f"Configuration does not exist: {input_path}")
        if not input_path.is_file():
            raise IsADirectoryError(f"Configuration is not a file: {input_path}")
        suffix = input_path.suffix.casefold()
        if suffix not in cls.supported_extensions:
            raise ValueError(f"Unsupported configuration extension: {suffix!r}")
        text = input_path.read_text(encoding="utf-8")
        values = (
            yaml.safe_load(text)
            if suffix in {".yaml", ".yml"}
            else json.loads(text)
        )
        if values is None:
            raise ValueError(f"Configuration is empty: {input_path}")
        if not isinstance(values, dict):
            raise TypeError("Configuration root must be a mapping.")
        return values

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> Phase4Config:
        root = deepcopy(dict(values))
        unknown = set(root).difference(cls.allowed_sections)
        if unknown:
            raise ValueError(f"Unknown Phase 4 sections: {sorted(unknown)}")

        execution = Phase4ExecutionOptions(
            **cls._mapping(root.get("execution", {}), "execution")
        )
        contract = ImputationContractOptions(
            **cls._mapping(root.get("contract", {}), "contract")
        )
        plan = cls._build_plan(root.get("plan", {}))
        scanner = MissingnessScannerOptions(
            **cls._mapping(
                root.get("missingness_scanner", {}),
                "missingness_scanner",
            )
        )
        pipeline = cls._build_pipeline(root.get("pipeline", {}))
        validation = cls._build_validation(root.get("validation", {}))
        readiness = Phase4ReadinessOptions(
            **cls._mapping(root.get("readiness", {}), "readiness")
        )
        reporting = Phase4ReportOptions(
            **cls._mapping(root.get("reporting", {}), "reporting")
        )
        return Phase4Config(
            execution=execution,
            contract=contract,
            plan=plan,
            missingness_scanner=scanner,
            pipeline=pipeline,
            validation=validation,
            readiness=readiness,
            reporting=reporting,
        )

    @classmethod
    def _build_plan(cls, values: object) -> ImputationPlanOptions:
        data = cls._mapping(values, "plan")
        for name in (
            "default_numeric_method", "default_categorical_method",
            "default_boolean_method", "default_datetime_method",
            "default_text_method", "default_timedelta_method",
        ):
            if name in data:
                data[name] = ImputationMethodName(data[name])
        if "method_overrides" in data:
            data["method_overrides"] = {
                str(column): ImputationMethodName(method)
                for column, method in cls._mapping(
                    data["method_overrides"],
                    "plan.method_overrides",
                ).items()
            }
        return ImputationPlanOptions(**data)

    @classmethod
    def _build_pipeline(cls, values: object) -> ImputationPipelineOptions:
        data = cls._mapping(values, "pipeline")
        invalid_data = cls._mapping(
            data.pop("invalid_values", {}),
            "pipeline.invalid_values",
        )
        raw_rules = cls._mapping(
            invalid_data.pop("rules", {}),
            "pipeline.invalid_values.rules",
        )
        rules = {
            str(column): InvalidValueRule(
                column=str(column),
                **cls._mapping(rule, f"pipeline.invalid_values.rules.{column}"),
            )
            for column, rule in raw_rules.items()
        }
        invalid = InvalidValueScannerOptions(rules=rules, **invalid_data)
        tokens = MissingTokenDetectorOptions(
            **cls._mapping(
                data.pop("missing_tokens", {}),
                "pipeline.missing_tokens",
            )
        )
        indicator_data = cls._mapping(
            data.pop("indicators", {}),
            "pipeline.indicators",
        )
        naming_data = cls._mapping(
            indicator_data.pop("naming", {}),
            "pipeline.indicators.naming",
        )
        if "collision_policy" in naming_data:
            naming_data["collision_policy"] = IndicatorCollisionPolicy(
                naming_data["collision_policy"]
            )
        indicators = MissingIndicatorOptions(
            naming=IndicatorNamingOptions(**naming_data),
            **indicator_data,
        )
        transformer = PartitionTransformerOptions(
            **cls._mapping(
                data.pop("partition_transformer", {}),
                "pipeline.partition_transformer",
            )
        )
        result = ImputationPipelineOptions(
            invalid_values=invalid,
            missing_tokens=tokens,
            indicators=indicators,
            partition_transformer=transformer,
            numeric=NumericImputerOptions(
                **cls._mapping(data.pop("numeric", {}), "pipeline.numeric")
            ),
            categorical=CategoricalImputerOptions(
                **cls._mapping(data.pop("categorical", {}), "pipeline.categorical")
            ),
            boolean=BooleanImputerOptions(
                **cls._mapping(data.pop("boolean", {}), "pipeline.boolean")
            ),
            datetime=DatetimeImputerOptions(
                **cls._mapping(data.pop("datetime", {}), "pipeline.datetime")
            ),
            text=TextImputerOptions(
                **cls._mapping(data.pop("text", {}), "pipeline.text")
            ),
            group=GroupImputerOptions(
                **cls._mapping(data.pop("group", {}), "pipeline.group")
            ),
            temporal=TemporalImputerOptions(
                **cls._mapping(data.pop("temporal", {}), "pipeline.temporal")
            ),
            model=ModelImputerOptions(
                **cls._mapping(data.pop("model", {}), "pipeline.model")
            ),
        )
        if data:
            raise ValueError(f"Unknown pipeline options: {sorted(data)}")
        return result

    @classmethod
    def _build_validation(cls, values: object) -> Phase4ValidationOptions:
        data = cls._mapping(values, "validation")
        allowed = {
            "remaining_missingness", "schema_consistency", "target_integrity",
            "row_integrity", "distribution_impact",
        }
        unknown = set(data).difference(allowed)
        if unknown:
            raise ValueError(f"Unknown validation options: {sorted(unknown)}")
        return Phase4ValidationOptions(
            remaining_missingness=RemainingMissingnessOptions(
                **cls._mapping(
                    data.get("remaining_missingness", {}),
                    "validation.remaining_missingness",
                )
            ),
            schema_consistency=SchemaConsistencyOptions(
                **cls._mapping(
                    data.get("schema_consistency", {}),
                    "validation.schema_consistency",
                )
            ),
            target_integrity=TargetIntegrityOptions(
                **cls._mapping(
                    data.get("target_integrity", {}),
                    "validation.target_integrity",
                )
            ),
            row_integrity=RowIntegrityOptions(
                **cls._mapping(
                    data.get("row_integrity", {}),
                    "validation.row_integrity",
                )
            ),
            distribution_impact=DistributionImpactOptions(
                **cls._mapping(
                    data.get("distribution_impact", {}),
                    "validation.distribution_impact",
                )
            ),
        )

    @classmethod
    def save(
        cls,
        config: Phase4Config,
        path: str | Path,
        *,
        config_format: ConfigFormat | None = None,
        overwrite: bool = False,
    ) -> Path:
        output = Path(path)
        if output.exists() and not overwrite:
            raise FileExistsError(f"Configuration already exists: {output}")
        fmt = config_format or (
            "yaml"
            if output.suffix.casefold() in {".yaml", ".yml"}
            else "json"
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        values = config.to_dict()
        content = (
            yaml.safe_dump(values, sort_keys=False, allow_unicode=True)
            if fmt == "yaml"
            else json.dumps(values, indent=2, ensure_ascii=False, default=str)
        )
        output.write_text(content, encoding="utf-8")
        return output

    @classmethod
    def merge(
        cls,
        *,
        base: Mapping[str, Any],
        override: Mapping[str, Any],
    ) -> dict[str, Any]:
        merged = deepcopy(dict(base))
        for key, value in override.items():
            if isinstance(merged.get(key), Mapping) and isinstance(value, Mapping):
                merged[key] = cls.merge(base=merged[key], override=value)
            else:
                merged[key] = deepcopy(value)
        return merged

    @classmethod
    def apply_dot_overrides(
        cls,
        *,
        values: Mapping[str, Any],
        overrides: Mapping[str, Any],
    ) -> dict[str, Any]:
        updated = deepcopy(dict(values))
        for dotted_path, value in overrides.items():
            parts = dotted_path.split(".")
            if not dotted_path.strip() or any(not part.strip() for part in parts):
                raise ValueError(f"Invalid dotted path: {dotted_path!r}")
            current = updated
            for part in parts[:-1]:
                existing = current.get(part)
                if existing is None:
                    current[part] = {}
                    existing = current[part]
                if not isinstance(existing, dict):
                    raise TypeError(f"{part!r} is not a mapping.")
                current = existing
            current[parts[-1]] = deepcopy(value)
        return updated

    @staticmethod
    def _mapping(value: object, section: str) -> dict[str, Any]:
        if value is None:
            return {}
        if not isinstance(value, Mapping):
            raise TypeError(f"Section {section!r} must be a mapping.")
        return deepcopy(dict(value))
