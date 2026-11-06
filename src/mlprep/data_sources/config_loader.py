"""Strict YAML and JSON loading for data-source configurations."""

import json
from dataclasses import fields
from pathlib import Path
from typing import Any, ClassVar

import yaml

from mlprep.data_sources.config import (
    APISourceConfig,
    CSVSourceConfig,
    DataFrameSourceConfig,
    JSONSourceConfig,
    ParquetSourceConfig,
    SQLSourceConfig,
)
from mlprep.exceptions import ConfigurationError


class DataSourceConfigLoader:
    """Load one typed data-source configuration and reject unknown keys."""

    _CONFIG_TYPES: ClassVar[dict[str, type]] = {
        "dataframe": DataFrameSourceConfig,
        "csv": CSVSourceConfig,
        "json": JSONSourceConfig,
        "jsonl": JSONSourceConfig,
        "parquet": ParquetSourceConfig,
        "sql": SQLSourceConfig,
        "api": APISourceConfig,
    }

    @classmethod
    def from_mapping(cls, values: dict[str, Any]) -> tuple[str, object]:
        """Create a source config from ``type`` plus a nested ``config`` map."""
        if not isinstance(values, dict):
            raise ConfigurationError("Data-source configuration must be a mapping.")
        kind = values.get("type")
        if not isinstance(kind, str) or kind not in cls._CONFIG_TYPES:
            raise ConfigurationError(f"Unsupported data source type: {kind!r}")
        nested = values.get("config", values)
        if nested is values:
            nested = {key: value for key, value in values.items() if key != "type"}
        else:
            unknown_top_level = sorted(set(values) - {"type", "config"})
            if unknown_top_level:
                raise ConfigurationError(
                    f"Unknown data-source keys: {unknown_top_level}"
                )
        if not isinstance(nested, dict):
            raise ConfigurationError("Data-source config must be a mapping.")
        config_type = cls._CONFIG_TYPES[kind]
        allowed = {field.name for field in fields(config_type)}
        unknown = sorted(set(nested) - allowed)
        if unknown:
            raise ConfigurationError(
                f"Unknown {kind} data-source configuration keys: {unknown}"
            )
        if kind == "jsonl":
            nested = {**nested, "lines": True}
            config_type = JSONSourceConfig
        normalized = _normalize_values(nested, config_type)
        return kind, config_type(**normalized)

    @classmethod
    def load(cls, path: str | Path) -> tuple[str, object]:
        """Load YAML or JSON from a file without evaluating arbitrary content."""
        source_path = Path(path)
        try:
            if source_path.suffix.lower() == ".json":
                values = json.loads(source_path.read_text(encoding="utf-8"))
            elif source_path.suffix.lower() in {".yaml", ".yml"}:
                values = yaml.safe_load(source_path.read_text(encoding="utf-8"))
            else:
                raise ConfigurationError("Configuration must be JSON or YAML.")
        except (OSError, UnicodeError, json.JSONDecodeError, yaml.YAMLError) as exc:
            raise ConfigurationError(
                f"Unable to load data-source configuration: {source_path}"
            ) from exc
        return cls.from_mapping(values)


def _normalize_values(values: dict[str, Any], config_type: type) -> dict[str, Any]:
    normalized = dict(values)
    if config_type in {
        CSVSourceConfig,
        JSONSourceConfig,
        ParquetSourceConfig,
        SQLSourceConfig,
    } and isinstance(normalized.get("path"), str):
        normalized["path"] = normalized["path"]
    for key in ("parse_dates", "missing_tokens", "columns", "allowed_hosts"):
        if key in normalized and isinstance(normalized[key], list):
            normalized[key] = tuple(normalized[key])
    return normalized
