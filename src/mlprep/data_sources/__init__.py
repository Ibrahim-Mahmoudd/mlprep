"""Unified data-source contracts and local file adapters."""

from mlprep.data_sources.base import DataSource
from mlprep.data_sources.config import (
    CSVSourceConfig,
    DataFrameSourceConfig,
    JSONSourceConfig,
    ParquetSourceConfig,
)
from mlprep.data_sources.csv_source import CSVSource
from mlprep.data_sources.dataframe_source import DataFrameSource
from mlprep.data_sources.factory import DataSourceFactory
from mlprep.data_sources.json_source import JSONSource
from mlprep.data_sources.metadata import DataSourceMetadata
from mlprep.data_sources.parquet_source import ParquetSource
from mlprep.data_sources.result import DataSourceResult

__all__ = [
    "CSVSource",
    "CSVSourceConfig",
    "DataFrameSource",
    "DataFrameSourceConfig",
    "DataSource",
    "DataSourceFactory",
    "DataSourceMetadata",
    "DataSourceResult",
    "JSONSource",
    "JSONSourceConfig",
    "ParquetSource",
    "ParquetSourceConfig",
]
