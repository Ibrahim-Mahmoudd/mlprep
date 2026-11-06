"""In-memory DataFrame source."""

from datetime import datetime

import pandas as pd

from mlprep.data_sources.config import DataFrameSourceConfig
from mlprep.data_sources.fingerprint import frame_fingerprint
from mlprep.data_sources.metadata import DataSourceMetadata
from mlprep.data_sources.result import DataSourceResult


class DataFrameSource:
    """Load an existing DataFrame without mutating its caller-owned object."""

    def __init__(
        self,
        frame: pd.DataFrame,
        config: DataFrameSourceConfig | None = None,
    ) -> None:
        self._frame = frame
        self._config = config or DataFrameSourceConfig()

    def load(self) -> DataSourceResult:
        """Return a deep copy and deterministic provenance."""
        frame = self._frame.copy(deep=True)
        schema = {str(column): str(dtype) for column, dtype in frame.dtypes.items()}
        fingerprint = frame_fingerprint(frame)
        metadata = DataSourceMetadata(
            source_type="dataframe",
            source_name=self._config.source_name,
            extraction_time=datetime.now().astimezone(),
            row_count=len(frame),
            column_count=len(frame.columns),
            schema=schema,
        )
        return DataSourceResult(
            frame=frame,
            metadata=metadata,
            source_fingerprint=fingerprint,
            schema=schema,
            provenance={"source_type": "dataframe"},
        )
