"""Factory for configured data sources."""

from dataclasses import dataclass
from typing import Any

from mlprep.data_sources.base import DataSource
from mlprep.data_sources.config import (
    CSVSourceConfig,
    DataFrameSourceConfig,
    JSONSourceConfig,
    ParquetSourceConfig,
)
from mlprep.data_sources.csv_source import CSVSource
from mlprep.data_sources.dataframe_source import DataFrameSource
from mlprep.data_sources.json_source import JSONSource
from mlprep.data_sources.parquet_source import ParquetSource
from mlprep.exceptions import ConfigurationError


@dataclass(frozen=True, slots=True)
class DataSourceFactory:
    """Construct supported local sources from typed configuration."""

    @staticmethod
    def create(kind: str, config: Any, *, frame: Any = None) -> DataSource:
        """Create a source and reject unsupported or mismatched configuration."""
        if kind == "dataframe" and isinstance(config, DataFrameSourceConfig):
            if frame is None:
                raise ConfigurationError(
                    "A DataFrame is required for dataframe source."
                )
            return DataFrameSource(frame, config)
        if kind == "csv" and isinstance(config, CSVSourceConfig):
            return CSVSource(config)
        if kind == "json" and isinstance(config, JSONSourceConfig):
            return JSONSource(config)
        if kind == "parquet" and isinstance(config, ParquetSourceConfig):
            return ParquetSource(config)
        raise ConfigurationError(f"Unsupported data source or configuration: {kind}")
