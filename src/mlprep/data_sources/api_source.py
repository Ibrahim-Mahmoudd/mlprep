"""Standard-library REST data source with bounded retries and SSRF policy."""

import json
import os
import time
from datetime import datetime
from hashlib import sha256
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qsl, urlencode, urljoin, urlparse, urlunparse
from urllib.request import Request, urlopen

import pandas as pd

from mlprep.data_sources._local import validate_columns, validate_schema
from mlprep.data_sources.config import APISourceConfig
from mlprep.data_sources.metadata import DataSourceMetadata
from mlprep.data_sources.result import DataSourceResult
from mlprep.exceptions import DataSourceLoadError


class APISource:
    """Load JSON API records without exposing authorization material."""

    def __init__(self, config: APISourceConfig) -> None:
        self._config = config

    def load(self) -> DataSourceResult:
        """Fetch bounded pages and normalize records into a DataFrame."""
        url = self._validated_url()
        token = (
            os.environ.get(self._config.token_env)
            if self._config.token_env
            else None
        )
        records: list[object] = []
        first_payload: object = None
        for page in range(1, self._config.max_pages + 1):
            page_url = self._page_url(url, page)
            payload_bytes = self._fetch(page_url, token)
            try:
                payload = json.loads(payload_bytes)
            except json.JSONDecodeError as exc:
                raise DataSourceLoadError("API response is not valid JSON.") from exc
            if first_payload is None:
                first_payload = payload
            page_records = self._records(payload)
            records.extend(page_records)
            if not self._config.next_url_field or not isinstance(payload, dict):
                break
            next_url = payload.get(self._config.next_url_field)
            if not next_url:
                break
            url = self._validated_url(urljoin(url, str(next_url)))
        frame = pd.json_normalize(records)
        validate_columns(frame)
        validate_schema(frame, self._config.expected_schema)
        fingerprint = sha256(
            json.dumps(first_payload, sort_keys=True, default=str).encode("utf-8")
        ).hexdigest()
        schema = {str(column): str(dtype) for column, dtype in frame.dtypes.items()}
        metadata = DataSourceMetadata(
            source_type="api",
            source_name=self._config.endpoint or self._config.base_url,
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
            provenance={"source_type": "api", "endpoint": self._config.endpoint},
        )

    def _validated_url(self) -> str:
        url = urljoin(self._config.base_url.rstrip("/") + "/", self._config.endpoint)
        parsed = urlparse(url)
        if parsed.scheme not in {"https", "http"} or not parsed.hostname:
            raise DataSourceLoadError("API URL must include a valid HTTP(S) host.")
        if parsed.hostname not in self._config.allowed_hosts:
            raise DataSourceLoadError("API host is not present in allowed_hosts.")
        return url

    def _page_url(self, url: str, page: int) -> str:
        if self._config.page_parameter is None:
            return url
        parsed = urlparse(url)
        query = dict(parse_qsl(parsed.query, keep_blank_values=True))
        query[self._config.page_parameter] = str(page)
        if self._config.page_size_parameter and self._config.page_size is not None:
            query[self._config.page_size_parameter] = str(self._config.page_size)
        return urlunparse(parsed._replace(query=urlencode(query)))

    def _fetch(self, url: str, token: str | None) -> bytes:
        headers = {"Accept": "application/json"}
        if token:
            headers["Authorization"] = f"Bearer {token}"
        request = Request(url, headers=headers, method="GET")
        for attempt in range(3):
            try:
                with urlopen(request, timeout=self._config.timeout_seconds) as response:
                    content_type = response.headers.get("Content-Type", "")
                    if "json" not in content_type.lower():
                        raise DataSourceLoadError(
                            "API response content type is not JSON."
                        )
                    body = response.read(
                        int(self._config.max_response_size_mb * 1024 * 1024) + 1
                    )
                    if len(body) > self._config.max_response_size_mb * 1024 * 1024:
                        raise DataSourceLoadError(
                            "API response exceeds configured size."
                        )
                    return body
            except HTTPError as exc:
                if exc.code not in {429, 500, 502, 503, 504} or attempt == 2:
                    raise DataSourceLoadError("API request failed.") from exc
            except URLError as exc:
                if attempt == 2:
                    raise DataSourceLoadError("API connection failed.") from exc
            time.sleep(2**attempt)
        raise DataSourceLoadError("API request failed after retries.")

    def _records(self, payload: object) -> list[object]:
        if self._config.record_path is None:
            if isinstance(payload, list):
                return payload
            return [payload]
        if not isinstance(payload, dict):
            raise DataSourceLoadError("record_path requires an object response.")
        records = payload.get(self._config.record_path)
        if not isinstance(records, list):
            raise DataSourceLoadError("API record_path does not contain a list.")
        return records
