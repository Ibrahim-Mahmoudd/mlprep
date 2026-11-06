"""Configuration models for local data sources."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class DataFrameSourceConfig:
    """Configuration for an in-memory DataFrame."""

    source_name: str = "dataframe"


@dataclass(frozen=True, slots=True)
class CSVSourceConfig:
    """Configuration for a CSV file."""

    path: str | Path = ""
    encoding: str = "utf-8"
    delimiter: str = ","
    parse_dates: tuple[str, ...] = ()
    dtype: dict[str, str] = field(default_factory=dict)
    missing_tokens: tuple[str, ...] = ()
    maximum_file_size_mb: float = 500.0
    allowed_root: str | Path | None = None
    expected_schema: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class JSONSourceConfig:
    """Configuration for JSON or JSON Lines files."""

    path: str | Path = ""
    lines: bool = False
    encoding: str = "utf-8"
    record_path: str | None = None
    maximum_file_size_mb: float = 500.0
    allowed_root: str | Path | None = None
    expected_schema: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ParquetSourceConfig:
    """Configuration for a Parquet file."""

    path: str | Path = ""
    columns: tuple[str, ...] | None = None
    filters: list[list[tuple[str, str, Any]]] | None = None
    maximum_file_size_mb: float = 500.0
    allowed_root: str | Path | None = None
    expected_schema: dict[str, str] = field(default_factory=dict)
