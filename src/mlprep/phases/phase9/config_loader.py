"""Configuration loading for Phase 9."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from mlprep.phases.phase9.config import Phase9Config


class Phase9ConfigLoader:
    """Load and validate a Phase 9 YAML or JSON configuration."""

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> Phase9Config:
        """Construct configuration from a mapping."""

        allowed = {
            "require_disjoint_rows",
            "copy_partitions",
        }
        unknown = set(values) - allowed
        if unknown:
            raise ValueError(
                f"Unknown Phase 9 configuration keys: {sorted(unknown)}"
            )
        return Phase9Config(**values)

    @classmethod
    def load(cls, path: str | Path) -> Phase9Config:
        """Load a Phase 9 configuration from a YAML or JSON file."""

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

        if not isinstance(values, dict):
            raise ValueError("Phase 9 configuration must be a mapping.")
        return cls.from_dict(values)
