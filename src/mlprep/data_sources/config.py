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


@dataclass(frozen=True, slots=True)
class SQLSourceConfig:
    """Configuration for a parameterized SQL query."""

    connection_url_env: str = ""
    query_file: str | Path = ""
    query_parameters: dict[str, Any] = field(default_factory=dict)
    timeout_seconds: float = 60.0
    chunksize: int | None = None
    read_only: bool = True
    allowed_query_root: str | Path | None = None
    expected_schema: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class APISourceConfig:
    """Configuration for a JSON REST endpoint."""

    base_url: str = ""
    endpoint: str = ""
    token_env: str | None = None
    timeout_seconds: float = 30.0
    max_response_size_mb: float = 50.0
    max_pages: int = 1
    page_parameter: str | None = None
    page_size_parameter: str | None = None
    page_size: int | None = None
    next_url_field: str | None = None
    record_path: str | None = None
    allowed_hosts: tuple[str, ...] = ()
    expected_schema: dict[str, str] = field(default_factory=dict)
