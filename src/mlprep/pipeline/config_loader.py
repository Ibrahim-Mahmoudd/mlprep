"""YAML and JSON configuration loading for the pipeline runner."""

from __future__ import annotations

import json
import os
from collections.abc import Callable, Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any, TypeVar

import yaml

from mlprep.cross_validation.config import (
    CrossValidationConfig,
    CrossValidationStrategy,
)
from mlprep.phases import (
    Phase1Config,
    Phase2Config,
    Phase2ConfigLoader,
    Phase3Config,
    Phase3ConfigLoader,
    Phase4Config,
    Phase5Config,
    Phase6Config,
    Phase7Config,
    Phase8Config,
    Phase9Config,
)
from mlprep.phases.phase4.config_loader import Phase4ConfigLoader
from mlprep.phases.phase5.config_loader import Phase5ConfigLoader
from mlprep.phases.phase6.config_loader import Phase6ConfigLoader
from mlprep.phases.phase7.config_loader import Phase7ConfigLoader
from mlprep.phases.phase8.config_loader import Phase8ConfigLoader
from mlprep.phases.phase9.config_loader import Phase9ConfigLoader
from mlprep.pipeline.config import (
    MLPrepConfig,
    PipelineOptions,
)

ConfigT = TypeVar("ConfigT")


class MLPrepConfigLoader:
    """Load a versioned unified configuration from YAML or JSON."""

    _allowed_keys = frozenset(
        {
            "phase1",
            "phase2",
            "phase3",
            "phase4",
            "phase5",
            "phase6",
            "phase7",
            "phase8",
            "phase9",
            "schema_version",
            "cross_validation",
            "pipeline",
            "output_directory",
            "save_intermediate_states",
        }
    )

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        dotted_overrides: Mapping[str, Any] | None = None,
        environment_prefix: str = "MLPREP_",
        environ: Mapping[str, str] | None = None,
    ) -> MLPrepConfig:
        """Read and resolve a YAML or JSON configuration.

        Overrides are applied in this order: file values, dotted overrides,
        then matching environment variables.
        """

        config_path = Path(path)
        if not config_path.exists():
            raise FileNotFoundError(
                f"Configuration does not exist: {config_path}"
            )
        if not config_path.is_file():
            raise IsADirectoryError(
                f"Configuration path is not a file: {config_path}"
            )

        content = config_path.read_text(encoding="utf-8")
        extension = config_path.suffix.casefold()
        if extension in {".yaml", ".yml"}:
            try:
                values = yaml.safe_load(content)
            except yaml.YAMLError as error:
                raise ValueError(
                    f"Invalid YAML configuration: {error}"
                ) from error
        elif extension == ".json":
            try:
                values = json.loads(content)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON configuration: {error}"
                ) from error
        else:
            raise ValueError(
                f"Unsupported configuration extension: {extension!r}"
            )

        if not isinstance(values, Mapping):
            raise ValueError("Pipeline configuration must be a mapping.")
        values = deepcopy(dict(values))
        if dotted_overrides:
            values = cls.apply_dotted_overrides(values, dotted_overrides)
        environment = os.environ if environ is None else environ
        environment_overrides = cls._environment_overrides(
            environment,
            prefix=environment_prefix,
        )
        if environment_overrides:
            values = cls.apply_dotted_overrides(
                values,
                environment_overrides,
            )
        return cls.from_dict(values)

    @classmethod
    def from_dict(
        cls,
        values: Mapping[str, Any],
    ) -> MLPrepConfig:
        """Validate a mapping and construct the resolved configuration."""

        unknown = set(values).difference(cls._allowed_keys)
        if unknown:
            raise ValueError(
                "Unknown pipeline configuration keys: "
                f"{sorted(unknown)}"
            )

        missing = {"phase1", "phase2"}.difference(values)
        if missing:
            raise ValueError(
                "Pipeline configuration requires: "
                f"{', '.join(sorted(missing))}."
            )

        phase1_value = values["phase1"]
        phase1 = (
            phase1_value
            if isinstance(phase1_value, Phase1Config)
            else Phase1Config.from_dict(
                dict(cls._mapping_for_phase(phase1_value, "phase1"))
            )
        )
        phase2_value = values["phase2"]
        phase2 = (
            phase2_value
            if isinstance(phase2_value, Phase2Config)
            else Phase2ConfigLoader.from_dict(
                cls._mapping_for_phase(phase2_value, "phase2")
            )
        )

        phase3 = cls._load_optional(
            values, "phase3", Phase3Config, Phase3ConfigLoader.from_dict
        )
        phase4 = cls._load_optional(
            values, "phase4", Phase4Config, Phase4ConfigLoader.from_dict
        )
        phase5 = cls._load_optional(
            values, "phase5", Phase5Config, Phase5ConfigLoader.from_dict
        )
        phase6 = cls._load_optional(
            values, "phase6", Phase6Config, Phase6ConfigLoader.from_dict
        )
        phase7 = cls._load_optional(
            values, "phase7", Phase7Config, Phase7ConfigLoader.from_dict
        )
        phase8 = cls._load_optional(
            values, "phase8", Phase8Config, Phase8ConfigLoader.from_dict
        )
        phase9 = cls._load_optional(
            values,
            "phase9",
            Phase9Config,
            lambda mapping: Phase9ConfigLoader.from_dict(dict(mapping)),
        )

        cross_validation = cls._cross_validation(
            values.get("cross_validation", {})
        )
        pipeline_options = cls._pipeline_options(values)
        return MLPrepConfig(
            phase1=phase1,
            phase2=phase2,
            phase3=phase3,
            phase4=phase4,
            phase5=phase5,
            phase6=phase6,
            phase7=phase7,
            phase8=phase8,
            phase9=phase9,
            schema_version=values.get("schema_version", "1"),
            cross_validation=cross_validation,
            pipeline=pipeline_options,
        )

    @classmethod
    def apply_dotted_overrides(
        cls,
        values: Mapping[str, Any],
        overrides: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Apply dotted-key overrides to a copy of the raw configuration."""

        resolved = deepcopy(dict(values))
        for dotted_path, value in overrides.items():
            path_parts = dotted_path.split(".")
            if not dotted_path or any(not part for part in path_parts):
                raise ValueError(
                    f"Invalid dotted override path: {dotted_path!r}."
                )
            parent = resolved
            for part in path_parts[:-1]:
                child = parent.get(part)
                if child is None:
                    child = {}
                    parent[part] = child
                if not isinstance(child, dict):
                    raise ValueError(
                        f"Dotted override path {dotted_path!r} "
                        "crosses a non-mapping value."
                    )
                parent = child
            parent[path_parts[-1]] = deepcopy(value)
        return resolved

    @classmethod
    def _environment_overrides(
        cls,
        environment: Mapping[str, str],
        *,
        prefix: str,
    ) -> dict[str, Any]:
        overrides: dict[str, Any] = {}
        normalized_prefix = prefix.upper()
        for variable_name, raw_value in environment.items():
            if not variable_name.upper().startswith(normalized_prefix):
                continue
            suffix = variable_name[len(prefix) :]
            segments = suffix.split("__")
            if not suffix or any(not segment for segment in segments):
                raise ValueError(
                    f"Invalid MLPrep environment override name "
                    f"{variable_name!r}."
                )
            dotted_path = ".".join(
                segment.casefold() for segment in segments
            )
            try:
                overrides[dotted_path] = yaml.safe_load(raw_value)
            except yaml.YAMLError as error:
                raise ValueError(
                    "Invalid YAML value in environment override "
                    f"{variable_name!r}."
                ) from error
        return overrides

    @staticmethod
    def _cross_validation(value: Any) -> CrossValidationConfig:
        if isinstance(value, CrossValidationConfig):
            return value
        if not isinstance(value, Mapping):
            raise TypeError("cross_validation must be a mapping.")
        allowed = {"strategy", "n_splits", "random_seed", "shuffle"}
        unknown = set(value).difference(allowed)
        if unknown:
            raise ValueError(
                "Unknown cross_validation configuration keys: "
                f"{sorted(unknown)}"
            )
        resolved = dict(value)
        if "strategy" in resolved:
            resolved["strategy"] = CrossValidationStrategy(
                resolved["strategy"]
            )
        return CrossValidationConfig(**resolved)

    @staticmethod
    def _pipeline_options(values: Mapping[str, Any]) -> PipelineOptions:
        pipeline_value = values.get("pipeline", {})
        if isinstance(pipeline_value, PipelineOptions):
            pipeline_values = {
                "output_directory": pipeline_value.output_directory,
                "save_intermediate_states": (
                    pipeline_value.save_intermediate_states
                ),
            }
        elif isinstance(pipeline_value, Mapping):
            pipeline_values = dict(pipeline_value)
        else:
            raise TypeError("pipeline must be a mapping.")

        allowed = {"output_directory", "save_intermediate_states"}
        unknown = set(pipeline_values).difference(allowed)
        if unknown:
            raise ValueError(
                "Unknown pipeline configuration keys: "
                f"{sorted(unknown)}"
            )

        for key in allowed:
            if key in values:
                legacy_value = values[key]
                if (
                    key in pipeline_values
                    and pipeline_values[key] != legacy_value
                ):
                    raise ValueError(
                        f"Top-level {key} conflicts with pipeline.{key}."
                    )
                pipeline_values[key] = legacy_value
        return PipelineOptions(**pipeline_values)

    @staticmethod
    def _mapping_for_phase(
        value: Any,
        phase_name: str,
    ) -> Mapping[str, Any]:
        if not isinstance(value, Mapping):
            raise TypeError(
                f"{phase_name} must be a configuration mapping."
            )
        return value

    @classmethod
    def _load_optional(
        cls,
        values: Mapping[str, Any],
        name: str,
        config_type: type[ConfigT],
        from_dict: Callable[[Mapping[str, Any]], ConfigT],
    ) -> ConfigT:
        value = values.get(name)
        if value is None:
            return config_type()
        if isinstance(value, config_type):
            return value
        if not isinstance(value, Mapping):
            raise TypeError(
                f"{name} must be a configuration mapping."
            )
        return from_dict(value)


class PipelineConfigLoader(MLPrepConfigLoader):
    """Backward-compatible name for the unified configuration loader."""
