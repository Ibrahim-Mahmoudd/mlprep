import json
from typing import ClassVar

import pandas as pd
import pytest

from mlprep.data_sources import (
    APISource,
    APISourceConfig,
    CSVSource,
    CSVSourceConfig,
    DataFrameSource,
    DataSourceConfiguration,
    JSONSource,
    JSONSourceConfig,
    ParquetSource,
    ParquetSourceConfig,
    SQLSource,
    SQLSourceConfig,
    api_source,
)
from mlprep.data_sources.config_loader import DataSourceConfigLoader
from mlprep.exceptions import (
    ConfigurationError,
    DataSourceLoadError,
    DataSourceSchemaError,
)


def test_dataframe_source_copies_input_and_returns_stable_metadata() -> None:
    frame = pd.DataFrame({"value": [1, 2]})
    result = DataFrameSource(frame).load()

    result.frame.loc[0, "value"] = 99
    assert frame.loc[0, "value"] == 1
    assert result.metadata.source_type == "dataframe"
    assert result.metadata.row_count == 2
    assert result.source_fingerprint


def test_csv_source_supports_options_schema_and_missing_tokens(tmp_path) -> None:
    path = tmp_path / "customers.csv"
    path.write_text("id,value\n1,NA\n2,ok\n", encoding="utf-8")

    result = CSVSource(
        CSVSourceConfig(
            path=path,
            missing_tokens=("NA",),
            expected_schema={"id": "integer", "value": "string"},
        )
    ).load()

    assert result.frame["value"].isna().iloc[0]
    assert result.metadata.file_path == str(path.resolve())
    assert result.provenance["source_type"] == "csv"


def test_local_source_rejects_path_outside_allowed_root(tmp_path) -> None:
    outside = tmp_path.parent / "outside.csv"
    outside.write_text("value\n1\n", encoding="utf-8")

    with pytest.raises(DataSourceLoadError, match="inside allowed_root"):
        CSVSource(
            CSVSourceConfig(
                path=outside,
                allowed_root=tmp_path,
            )
        ).load()


def test_csv_source_rejects_duplicate_columns(tmp_path) -> None:
    path = tmp_path / "duplicate.csv"
    path.write_text("value,value\n1,2\n", encoding="utf-8")

    with pytest.raises(DataSourceSchemaError, match="Duplicate column"):
        CSVSource(CSVSourceConfig(path=path)).load()


def test_json_source_supports_json_lines(tmp_path) -> None:
    path = tmp_path / "events.jsonl"
    path.write_text(
        json.dumps({"id": 1, "kind": "a"}) + "\n"
        + json.dumps({"id": 2, "kind": "b"}) + "\n",
        encoding="utf-8",
    )

    result = JSONSource(JSONSourceConfig(path=path, lines=True)).load()

    assert result.metadata.source_type == "jsonl"
    assert result.frame.to_dict(orient="records") == [
        {"id": 1, "kind": "a"},
        {"id": 2, "kind": "b"},
    ]


def test_parquet_source_reports_optional_engine_error(tmp_path, monkeypatch) -> None:
    path = tmp_path / "data.parquet"
    path.write_bytes(b"not parquet")

    def missing_engine(*args, **kwargs):
        raise ImportError("missing engine")

    monkeypatch.setattr(pd, "read_parquet", missing_engine)
    with pytest.raises(DataSourceLoadError, match="optional engine"):
        ParquetSource(ParquetSourceConfig(path=path)).load()


def test_api_source_requires_explicit_host_allowlist() -> None:
    with pytest.raises(DataSourceLoadError, match="allowed_hosts"):
        APISource(
            APISourceConfig(
                base_url="https://example.test",
                endpoint="/records",
            )
        ).load()


def test_sql_source_requires_environment_url(tmp_path) -> None:
    query = tmp_path / "query.sql"
    query.write_text("SELECT 1", encoding="utf-8")

    with pytest.raises(DataSourceLoadError, match="environment variable"):
        SQLSource(
            SQLSourceConfig(
                connection_url_env="MLPREP_TEST_DATABASE_URL",
                query_file=query,
            )
        ).load()


def test_config_loader_rejects_unknown_keys_and_loads_yaml(tmp_path) -> None:
    path = tmp_path / "source.yaml"
    path.write_text(
        "type: csv\nconfig:\n  path: data.csv\n  unknown: true\n",
        encoding="utf-8",
    )
    with pytest.raises(ConfigurationError, match="Unknown"):
        DataSourceConfigLoader.load(path)


def test_config_loader_returns_secret_safe_typed_configuration(tmp_path) -> None:
    path = tmp_path / "source.json"
    path.write_text(
        json.dumps(
            {
                "type": "sql",
                "config": {
                    "connection_url_env": "MLPREP_DATABASE_URL",
                    "query_file": "query.sql",
                    "query_parameters": {"customer_token": "secret-value"},
                },
            }
        ),
        encoding="utf-8",
    )

    resolved = DataSourceConfigLoader.load(path)

    assert isinstance(resolved, DataSourceConfiguration)
    assert resolved.source_type == "sql"
    assert "secret-value" not in repr(resolved.to_dict())


def test_api_source_uses_token_only_in_authorization_header(monkeypatch) -> None:
    captured: dict[str, str] = {}

    class Response:
        headers: ClassVar[dict[str, str]] = {"Content-Type": "application/json"}

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self, size: int) -> bytes:
            return b'[{"id": 1}]'

    def fake_urlopen(request, timeout):
        captured["authorization"] = request.headers["Authorization"]
        return Response()

    monkeypatch.setattr(api_source, "urlopen", fake_urlopen)
    monkeypatch.setenv("MLPREP_API_TOKEN", "secret-value")
    result = APISource(
        APISourceConfig(
            base_url="https://example.test",
            endpoint="/records",
            token_env="MLPREP_API_TOKEN",
            allowed_hosts=("example.test",),
        )
    ).load()

    assert result.frame["id"].tolist() == [1]
    assert captured["authorization"] == "Bearer secret-value"
    assert "secret-value" not in repr(result.to_dict())
