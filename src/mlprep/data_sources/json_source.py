"""JSON and JSON Lines data source."""

import pandas as pd

from mlprep.data_sources._local import (
    build_file_metadata,
    resolve_file,
    validate_columns,
    validate_schema,
)
from mlprep.data_sources.config import JSONSourceConfig
from mlprep.data_sources.fingerprint import file_fingerprint
from mlprep.data_sources.metadata import DataSourceMetadata
from mlprep.data_sources.result import DataSourceResult
from mlprep.exceptions import DataSourceLoadError


class JSONSource:
    """Read JSON records or JSON Lines with a bounded file policy."""

    def __init__(self, config: JSONSourceConfig) -> None:
        self._config = config

    def load(self) -> DataSourceResult:
        """Load, validate, and fingerprint JSON data."""
        path = resolve_file(
            self._config.path,
            allowed_root=self._config.allowed_root,
            maximum_file_size_mb=self._config.maximum_file_size_mb,
        )
        try:
            if self._config.record_path is not None:
                frame = pd.json_normalize(
                    pd.read_json(path, lines=self._config.lines).to_dict(
                        orient="records"
                    ),
                    record_path=self._config.record_path,
                )
            else:
                frame = pd.read_json(
                    path,
                    lines=self._config.lines,
                    encoding=self._config.encoding,
                )
        except (OSError, UnicodeError, ValueError, TypeError) as exc:
            raise DataSourceLoadError(f"Unable to read JSON source: {path}") from exc
        validate_columns(frame)
        validate_schema(frame, self._config.expected_schema)
        metadata = DataSourceMetadata(
            **build_file_metadata(
                source_type="jsonl" if self._config.lines else "json",
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
            provenance={"source_type": metadata.source_type, "path": str(path)},
        )
