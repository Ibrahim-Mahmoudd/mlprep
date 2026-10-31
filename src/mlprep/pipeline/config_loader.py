"""YAML and JSON configuration loading for the pipeline runner."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, TypeVar

import yaml

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
from mlprep.pipeline.config import MLPrepPipelineConfig

ConfigT = TypeVar("ConfigT")


class PipelineConfigLoader:
    """Load resolved phase configurations from one YAML or JSON file."""

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
            "output_directory",
            "save_intermediate_states",
        }
    )

    @classmethod
    def load(cls, path: str | Path) -> MLPrepPipelineConfig:
        """Read a YAML or JSON pipeline configuration."""

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
        return cls.from_dict(values)

    @classmethod
    def from_dict(
        cls,
        values: Mapping[str, Any],
    ) -> MLPrepPipelineConfig:
        """Validate a mapping and construct all nine phase configurations."""

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

        return MLPrepPipelineConfig(
            phase1=phase1,
            phase2=phase2,
            phase3=phase3,
            phase4=phase4,
            phase5=phase5,
            phase6=phase6,
            phase7=phase7,
            phase8=phase8,
            phase9=phase9,
            output_directory=values.get("output_directory"),
            save_intermediate_states=values.get(
                "save_intermediate_states", True
            ),
        )

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
