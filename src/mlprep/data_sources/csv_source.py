"""CSV data source."""

import pandas as pd

from mlprep.data_sources._local import (
    build_file_metadata,
    resolve_file,
    validate_columns,
    validate_csv_header,
    validate_schema,
)
from mlprep.data_sources.config import CSVSourceConfig
from mlprep.data_sources.fingerprint import file_fingerprint
from mlprep.data_sources.metadata import DataSourceMetadata
from mlprep.data_sources.result import DataSourceResult
from mlprep.exceptions import DataSourceLoadError


class CSVSource:
    """Read a governed CSV file with bounded local-file access."""

    def __init__(self, config: CSVSourceConfig) -> None:
        self._config = config

    def load(self) -> DataSourceResult:
        """Load, validate, and fingerprint the CSV."""
        path = resolve_file(
            self._config.path,
            allowed_root=self._config.allowed_root,
            maximum_file_size_mb=self._config.maximum_file_size_mb,
        )
        try:
            validate_csv_header(
                path,
                encoding=self._config.encoding,
                delimiter=self._config.delimiter,
            )
            frame = pd.read_csv(
                path,
                encoding=self._config.encoding,
                sep=self._config.delimiter,
                parse_dates=list(self._config.parse_dates),
                dtype=self._config.dtype or None,
                na_values=list(self._config.missing_tokens) or None,
                keep_default_na=True,
            )
        except (OSError, UnicodeError, ValueError, pd.errors.ParserError) as exc:
            raise DataSourceLoadError(f"Unable to read CSV source: {path}") from exc
        validate_columns(frame)
        validate_schema(frame, self._config.expected_schema)
        metadata_values = build_file_metadata(
            source_type="csv",
            source_name=path.name,
            path=path,
            frame=frame,
        )
        metadata = DataSourceMetadata(**metadata_values)
        return DataSourceResult(
            frame=frame,
            metadata=metadata,
            source_fingerprint=file_fingerprint(path),
            schema=metadata.schema,
            provenance={"source_type": "csv", "path": str(path)},
        )
