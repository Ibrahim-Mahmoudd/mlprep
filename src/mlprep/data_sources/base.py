"""Protocols shared by all data sources."""

from typing import Protocol

from mlprep.data_sources.result import DataSourceResult


class DataSource(Protocol):
    """A loadable tabular data source."""

    def load(self) -> DataSourceResult:
        """Load the source and return a governed tabular result."""
        ...
