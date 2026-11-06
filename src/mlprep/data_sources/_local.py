"""Shared validation for local file sources."""

import csv
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from mlprep.exceptions import DataSourceLoadError, DataSourceSchemaError


def resolve_file(
    raw_path: str | Path,
    *,
    allowed_root: str | Path | None,
    maximum_file_size_mb: float,
) -> Path:
    """Resolve and validate a local input path."""
    if not raw_path:
        raise DataSourceLoadError("A source path is required.")
    if maximum_file_size_mb <= 0:
        raise DataSourceLoadError("maximum_file_size_mb must be positive.")
    path = Path(raw_path).expanduser()
    root = Path(allowed_root).expanduser().resolve() if allowed_root else None
    resolved = path.resolve()
    if root is not None:
        try:
            resolved.relative_to(root)
        except ValueError as exc:
            raise DataSourceLoadError(
                "The source path must remain inside allowed_root."
            ) from exc
    if not resolved.is_file():
        raise DataSourceLoadError(f"Source file does not exist: {resolved}")
    maximum_bytes = int(maximum_file_size_mb * 1024 * 1024)
    size = resolved.stat().st_size
    if size > maximum_bytes:
        raise DataSourceLoadError(
            f"Source file exceeds maximum_file_size_mb: {resolved}"
        )
    return resolved


def validate_columns(frame: pd.DataFrame) -> None:
    """Reject duplicate column names before data enters the pipeline."""
    if frame.columns.duplicated().any():
        duplicates = sorted(
            {str(column) for column in frame.columns[frame.columns.duplicated()]}
        )
        raise DataSourceSchemaError(
            f"Duplicate column names are not allowed: {duplicates}"
        )


def validate_csv_header(path: Path, *, encoding: str, delimiter: str) -> None:
    """Reject duplicate CSV headers before pandas disambiguates them."""
    with path.open("r", encoding=encoding, newline="") as handle:
        header = next(csv.reader(handle, delimiter=delimiter), [])
    if len(header) != len(set(header)):
        duplicates = sorted({column for column in header if header.count(column) > 1})
        raise DataSourceSchemaError(
            f"Duplicate column names are not allowed: {duplicates}"
        )


def validate_schema(frame: pd.DataFrame, expected: dict[str, str]) -> None:
    """Validate required columns and their declared pandas dtype names."""
    missing = sorted(set(expected) - set(frame.columns))
    if missing:
        raise DataSourceSchemaError(f"Missing expected columns: {missing}")
    mismatched = {
        column: (str(frame[column].dtype), expected_type)
        for column, expected_type in expected.items()
        if not _dtype_matches(frame[column], expected_type)
    }
    if mismatched:
        raise DataSourceSchemaError(f"Schema dtype mismatch: {mismatched}")


def _dtype_matches(series: pd.Series, expected: str) -> bool:
    actual = str(series.dtype)
    if expected in {"string", "str"}:
        return actual in {"object", "string", "str"}
    if expected in {"integer", "int"}:
        return pd.api.types.is_integer_dtype(series)
    if expected in {"number", "float"}:
        return pd.api.types.is_numeric_dtype(series)
    if expected in {"datetime", "datetime64[ns]"}:
        return pd.api.types.is_datetime64_any_dtype(series)
    if expected in {"boolean", "bool"}:
        return pd.api.types.is_bool_dtype(series)
    return actual == expected


def build_file_metadata(
    *,
    source_type: str,
    source_name: str,
    path: Path,
    frame: pd.DataFrame,
    warnings: tuple[str, ...] = (),
) -> dict[str, Any]:
    """Build common metadata fields for a local file result."""
    stat = path.stat()
    modified = datetime.fromtimestamp(stat.st_mtime).astimezone()
    return {
        "source_type": source_type,
        "source_name": source_name,
        "extraction_time": datetime.now().astimezone(),
        "row_count": len(frame),
        "column_count": len(frame.columns),
        "schema": {str(column): str(dtype) for column, dtype in frame.dtypes.items()},
        "file_path": str(path),
        "file_size_bytes": stat.st_size,
        "file_modified_time": modified,
        "warnings": warnings,
    }
