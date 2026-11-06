"""Result contract returned by data sources."""

from dataclasses import dataclass
from typing import Any

import pandas as pd

from mlprep.data_sources.metadata import DataSourceMetadata


@dataclass(frozen=True, slots=True)
class DataSourceResult:
    """Loaded data plus provenance and deterministic identity."""

    frame: pd.DataFrame
    metadata: DataSourceMetadata
    source_fingerprint: str
    schema: dict[str, str]
    provenance: dict[str, Any]
    warnings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Prevent callers from mutating the governed result through the frame."""
        object.__setattr__(self, "frame", self.frame.copy(deep=True))

    def to_dict(self) -> dict[str, Any]:
        """Return metadata suitable for state or reporting."""
        return {
            "metadata": self.metadata.to_dict(),
            "source_fingerprint": self.source_fingerprint,
            "schema": dict(self.schema),
            "provenance": dict(self.provenance),
            "warnings": list(self.warnings),
        }
