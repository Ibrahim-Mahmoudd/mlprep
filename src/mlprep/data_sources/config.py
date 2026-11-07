"""Configuration models for local data sources."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class DataFrameSourceConfig:
    """Configuration for an in-memory DataFrame."""

    source_name: str = "dataframe"

    def __post_init__(self) -> None:
        if not self.source_name.strip():
            raise ValueError("source_name must not be empty.")


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

    def __post_init__(self) -> None:
        _validate_local_options(self.maximum_file_size_mb)
        if len(self.delimiter) != 1:
            raise ValueError("delimiter must be exactly one character.")


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

    def __post_init__(self) -> None:
        _validate_local_options(self.maximum_file_size_mb)
        if self.record_path is not None and not self.record_path.strip():
            raise ValueError("record_path must not be empty.")


@dataclass(frozen=True, slots=True)
class ParquetSourceConfig:
    """Configuration for a Parquet file."""

    path: str | Path = ""
    columns: tuple[str, ...] | None = None
    filters: list[list[tuple[str, str, Any]]] | None = None
    maximum_file_size_mb: float = 500.0
    allowed_root: str | Path | None = None
    expected_schema: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _validate_local_options(self.maximum_file_size_mb)


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

    def __post_init__(self) -> None:
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive.")
        if self.chunksize is not None and self.chunksize <= 0:
            raise ValueError("chunksize must be positive when configured.")


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

    def __post_init__(self) -> None:
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive.")
        if self.max_response_size_mb <= 0:
            raise ValueError("max_response_size_mb must be positive.")
        if self.max_pages <= 0:
            raise ValueError("max_pages must be positive.")
        if self.page_size is not None and self.page_size <= 0:
            raise ValueError("page_size must be positive when configured.")


def _validate_local_options(maximum_file_size_mb: float) -> None:
    if maximum_file_size_mb <= 0:
        raise ValueError("maximum_file_size_mb must be positive.")


@dataclass(frozen=True, slots=True)
class DataSourceConfiguration:
    """Resolved source type and its validated, secret-safe options."""

    source_type: str
    options: (
        DataFrameSourceConfig
        | CSVSourceConfig
        | JSONSourceConfig
        | ParquetSourceConfig
        | SQLSourceConfig
        | APISourceConfig
    )

    def to_dict(self) -> dict[str, Any]:
        """Serialize configuration without sensitive parameter values."""
        values = {
            "type": self.source_type,
            "config": _serialize_dataclass(self.options),
        }
        return values


def _serialize_dataclass(value: Any) -> Any:
    if hasattr(value, "__dataclass_fields__"):
        return {
            name: _serialize_dataclass(getattr(value, name))
            for name in value.__dataclass_fields__
        }
    if isinstance(value, dict):
        return {
            str(key): (
                "<redacted>"
                if any(
                    marker in str(key).casefold()
                    for marker in ("password", "secret", "token", "credential")
                )
                else _serialize_dataclass(item)
            )
            for key, item in value.items()
        }
    if isinstance(value, (tuple, list)):
        return [_serialize_dataclass(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value
