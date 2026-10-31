"""YAML and JSON configuration loading for Phase 7."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import fields, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Union, get_args, get_origin, get_type_hints
from types import UnionType

import yaml
from pydantic import BaseModel

from mlprep.phases.phase7.config import Phase7Config


class Phase7ConfigLoader:
    """Load, validate, merge, override, and save Phase 7 config."""

    allowed_sections = {
        "execution",
        "contract",
        "plan",
        "pipeline",
        "validation",
        "readiness",
        "reporting",
    }

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        overrides: Mapping[str, Any] | None = None,
        dotted_overrides: Mapping[str, Any] | None = None,
    ) -> Phase7Config:
        """Load a configuration from YAML or JSON."""

        values = cls.load_raw(path)
        if overrides is not None:
            values = cls.merge(values, overrides)
        if dotted_overrides is not None:
            values = cls.apply_dot_overrides(values, dotted_overrides)
        return cls.from_dict(values)

    @staticmethod
    def load_raw(path: str | Path) -> dict[str, Any]:
        """Read a YAML or JSON mapping from disk."""

        input_path = Path(path)
        if not input_path.exists():
            raise FileNotFoundError(f"Configuration does not exist: {input_path}")
        if not input_path.is_file():
            raise IsADirectoryError(f"Configuration is not a file: {input_path}")
        suffix = input_path.suffix.casefold()
        text = input_path.read_text(encoding="utf-8")
        if suffix in {".yaml", ".yml"}:
            try:
                values = yaml.safe_load(text)
            except yaml.YAMLError as error:
                raise ValueError(f"Invalid YAML configuration: {error}") from error
        elif suffix == ".json":
            try:
                values = json.loads(text)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON configuration: {error}") from error
        else:
            raise ValueError(f"Unsupported configuration extension: {suffix!r}")
        if not isinstance(values, dict):
            raise TypeError("Configuration root must be a mapping.")
        return values

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> Phase7Config:
        """Construct strict nested configuration from a mapping."""

        unknown = set(values).difference(cls.allowed_sections)
        if unknown:
            raise ValueError(f"Unknown Phase 7 sections: {sorted(unknown)}")
        return cls._construct(Phase7Config, values)

    @staticmethod
    def to_dict(config: Phase7Config) -> dict[str, Any]:
        """Convert configuration to a JSON/YAML-safe mapping."""

        return Phase7ConfigLoader._serialize(config)

    @classmethod
    def save(
        cls,
        config: Phase7Config,
        path: str | Path,
    ) -> Path:
        """Save a Phase 7 configuration as YAML or JSON."""

        output = Path(path)
        suffix = output.suffix.casefold()
        values = cls.to_dict(config)
        if suffix in {".yaml", ".yml"}:
            text = yaml.safe_dump(values, sort_keys=True, allow_unicode=True)
        elif suffix == ".json":
            text = json.dumps(values, indent=2, sort_keys=True, ensure_ascii=False)
        else:
            raise ValueError(f"Unsupported configuration extension: {suffix!r}")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text + "\n", encoding="utf-8")
        return output

    @classmethod
    def merge(
        cls,
        base: Mapping[str, Any],
        override: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Recursively merge an override mapping."""

        merged = dict(base)
        for key, value in override.items():
            if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
                merged[key] = cls.merge(merged[key], value)
            else:
                merged[key] = value
        return merged

    @staticmethod
    def apply_dot_overrides(
        values: Mapping[str, Any],
        overrides: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Apply validated dotted-path overrides."""

        result = json.loads(json.dumps(values))
        for path, value in overrides.items():
            parts = path.split(".")
            if any(not part for part in parts):
                raise ValueError(f"Invalid dotted override path: {path!r}")
            target: dict[str, Any] = result
            for part in parts[:-1]:
                child = target.get(part)
                if not isinstance(child, dict):
                    raise KeyError(f"Unknown dotted override path: {path!r}")
                target = child
            if parts[-1] not in target:
                raise KeyError(f"Unknown dotted override path: {path!r}")
            target[parts[-1]] = value
        return result

    @classmethod
    def _construct(cls, model_type: type[Any], value: Any) -> Any:
        if not isinstance(value, Mapping):
            raise TypeError(f"Expected a mapping for {model_type.__name__}.")
        if issubclass(model_type, BaseModel):
            return model_type.model_validate(value)
        model_fields = {item.name for item in fields(model_type)}
        unknown = set(value).difference(model_fields)
        if unknown:
            raise ValueError(
                f"Unknown {model_type.__name__} fields: {sorted(unknown)}"
            )
        hints = get_type_hints(model_type)
        converted = {
            key: cls._convert(hints[key], item)
            for key, item in value.items()
        }
        return model_type(**converted)

    @classmethod
    def _convert(cls, annotation: Any, value: Any) -> Any:
        if value is None:
            return None
        origin = get_origin(annotation)
        args = get_args(annotation)
        if origin in {Union, UnionType}:
            last_error: TypeError | ValueError | None = None
            for option in args:
                if option is type(None):
                    continue
                try:
                    return cls._convert(option, value)
                except (TypeError, ValueError) as error:
                    last_error = error
            if last_error is not None:
                raise last_error
            return value
        if origin is list and args:
            if not isinstance(value, list):
                raise TypeError("Expected a list in configuration.")
            return [cls._convert(args[0], item) for item in value]
        if origin is dict and len(args) == 2:
            if not isinstance(value, Mapping):
                raise TypeError("Expected a mapping in configuration.")
            return {
                cls._convert(args[0], key): cls._convert(args[1], item)
                for key, item in value.items()
            }
        if isinstance(annotation, type) and issubclass(annotation, BaseModel):
            return annotation.model_validate(value)
        if isinstance(annotation, type) and is_dataclass(annotation):
            return cls._construct(annotation, value)
        if isinstance(annotation, type) and issubclass(annotation, Enum):
            return annotation(value)
        return value

    @classmethod
    def _serialize(cls, value: Any) -> Any:
        if is_dataclass(value):
            return {
                item.name: cls._serialize(getattr(value, item.name))
                for item in fields(value)
            }
        if isinstance(value, BaseModel):
            return cls._serialize(value.model_dump(mode="python"))
        if isinstance(value, Enum):
            return value.value
        if isinstance(value, dict):
            return {str(key): cls._serialize(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [cls._serialize(item) for item in value]
        if isinstance(value, Path):
            return str(value)
        return value