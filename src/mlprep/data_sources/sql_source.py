"""Optional SQLAlchemy-backed data source."""

import hashlib
import os
from pathlib import Path

import pandas as pd

from mlprep.data_sources._local import validate_columns, validate_schema
from mlprep.data_sources.config import SQLSourceConfig
from mlprep.data_sources.metadata import DataSourceMetadata
from mlprep.data_sources.result import DataSourceResult
from mlprep.exceptions import DataSourceLoadError


class SQLSource:
    """Execute a parameterized, file-backed query without storing credentials."""

    def __init__(self, config: SQLSourceConfig) -> None:
        self._config = config

    def load(self) -> DataSourceResult:
        """Load SQL results through the optional SQLAlchemy dependency."""
        if not self._config.connection_url_env:
            raise DataSourceLoadError("connection_url_env is required.")
        connection_url = os.environ.get(self._config.connection_url_env)
        if not connection_url:
            raise DataSourceLoadError(
                f"Database URL environment variable is not set: "
                f"{self._config.connection_url_env}"
            )
        query_path = Path(self._config.query_file).expanduser().resolve()
        if self._config.allowed_query_root is not None:
            root = Path(self._config.allowed_query_root).expanduser().resolve()
            try:
                query_path.relative_to(root)
            except ValueError as exc:
                raise DataSourceLoadError(
                    "The query file must remain inside allowed_query_root."
                ) from exc
        if not query_path.is_file():
            raise DataSourceLoadError(f"Query file does not exist: {query_path}")
        query = query_path.read_text(encoding="utf-8")
        if not query.strip():
            raise DataSourceLoadError("The SQL query must not be empty.")
        try:
            from sqlalchemy import create_engine, text
        except ImportError as exc:
            raise DataSourceLoadError(
                "SQL support requires the optional sqlalchemy dependency."
            ) from exc
        try:
            engine = create_engine(
                connection_url,
                connect_args={"connect_timeout": self._config.timeout_seconds},
            )
            with engine.connect() as connection:
                frame = pd.read_sql_query(
                    text(query),
                    connection,
                    params=self._config.query_parameters,
                    chunksize=self._config.chunksize,
                )
            if not isinstance(frame, pd.DataFrame):
                frame = pd.concat(list(frame), ignore_index=True)
        except (OSError, ValueError, RuntimeError) as exc:
            raise DataSourceLoadError("Unable to execute SQL source.") from exc
        validate_columns(frame)
        validate_schema(frame, self._config.expected_schema)
        query_fingerprint = hashlib.sha256(query.encode("utf-8")).hexdigest()
        schema = {str(column): str(dtype) for column, dtype in frame.dtypes.items()}
        metadata = DataSourceMetadata(
            source_type="sql",
            source_name=query_path.name,
            extraction_time=pd.Timestamp.now(tz="UTC").to_pydatetime(),
            row_count=len(frame),
            column_count=len(frame.columns),
            schema=schema,
            query_fingerprint=query_fingerprint,
        )
        return DataSourceResult(
            frame=frame,
            metadata=metadata,
            source_fingerprint=query_fingerprint,
            schema=schema,
            provenance={"source_type": "sql", "query_file": query_path.name},
        )
