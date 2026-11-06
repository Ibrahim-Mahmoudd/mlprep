"""Safe metadata emitted by data sources."""

from dataclasses import dataclass
from datetime import datetime
from typing import Any


@dataclass(frozen=True, slots=True)
class DataSourceMetadata:
    """Non-secret provenance metadata for one extraction."""

    source_type: str
    source_name: str
    extraction_time: datetime
    row_count: int
    column_count: int
    schema: dict[str, str]
    source_snapshot_id: str | None = None
    cutoff_timestamp: datetime | None = None
    query_fingerprint: str | None = None
    file_path: str | None = None
    file_size_bytes: int | None = None
    file_modified_time: datetime | None = None
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """Return serializable metadata without credentials or raw payloads."""
        return {
            "source_type": self.source_type,
            "source_name": self.source_name,
            "extraction_time": self.extraction_time.isoformat(),
            "row_count": self.row_count,
            "column_count": self.column_count,
            "schema": dict(self.schema),
            "source_snapshot_id": self.source_snapshot_id,
            "cutoff_timestamp": (
                self.cutoff_timestamp.isoformat()
                if self.cutoff_timestamp is not None
                else None
            ),
            "query_fingerprint": self.query_fingerprint,
            "file_path": self.file_path,
            "file_size_bytes": self.file_size_bytes,
            "file_modified_time": (
                self.file_modified_time.isoformat()
                if self.file_modified_time is not None
                else None
            ),
            "warnings": list(self.warnings),
        }
