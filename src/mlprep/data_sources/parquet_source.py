"""Parquet data source with an optional engine dependency."""

import pandas as pd

from mlprep.data_sources._local import (
    build_file_metadata,
    resolve_file,
    validate_columns,
    validate_schema,
)
from mlprep.data_sources.config import ParquetSourceConfig
from mlprep.data_sources.fingerprint import file_fingerprint
from mlprep.data_sources.metadata import DataSourceMetadata
from mlprep.data_sources.result import DataSourceResult
from mlprep.exceptions import DataSourceLoadError


class ParquetSource:
    """Read Parquet using pandas and report a clear optional-dependency error."""

    def __init__(self, config: ParquetSourceConfig) -> None:
        self._config = config

    def load(self) -> DataSourceResult:
        """Load, validate, and fingerprint Parquet data."""
        path = resolve_file(
            self._config.path,
            allowed_root=self._config.allowed_root,
            maximum_file_size_mb=self._config.maximum_file_size_mb,
        )
        try:
            frame = pd.read_parquet(
                path,
                columns=list(self._config.columns)
                if self._config.columns is not None
                else None,
                filters=self._config.filters,
            )
        except ImportError as exc:
            raise DataSourceLoadError(
                "Parquet support requires an optional engine such as pyarrow."
            ) from exc
        except (OSError, ValueError) as exc:
            raise DataSourceLoadError(f"Unable to read Parquet source: {path}") from exc
        validate_columns(frame)
        validate_schema(frame, self._config.expected_schema)
        metadata = DataSourceMetadata(
            **build_file_metadata(
                source_type="parquet",
                source_name=path.name,
                path=path,
                frame=frame,
            )
        )
        return DataSourceResult(
            frame=frame,
            metadata=metadata,
            source_fingerprint=file_fingerprint(path),
            schema=metadata.schema,
            provenance={"source_type": "parquet", "path": str(path)},
        )
