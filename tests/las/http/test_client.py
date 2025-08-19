from __future__ import annotations

import asyncio
import time
from unittest.mock import MagicMock
from urllib.parse import urlparse

import httpx
import pytest

from daft.las.infra.http.auth import VolcOpenApiAuthProvider
from daft.las.infra.http.client import (
    DEFAULT_RETRYABLE_EXCEPTIONS,
    AsyncHttpClient,
    HttpClient,
    _default_retry_condition,
)
from daft.las.infra.http.retry import MaxRetriesExceeded, RetryPolicy
from daft.las.io.tos import TOSConfig


@pytest.mark.asyncio
async def test_tos_http_client(object_store_test_dir):
    config = TOSConfig.from_env()
    parsed = urlparse(object_store_test_dir)
    bucket = parsed.netloc
    key_path = parsed.path.rstrip("/") + "/test_tos_http_client.txt"

    client_args = {
        "base_url": config.virtual_host_endpoint(bucket),
        "auth_provider": VolcOpenApiAuthProvider(
            service="tos",
            region=config.region,
            access_key=config.access_key,
            secret_key=config.secret_key,
            signing_algorithm="TOS4-HMAC-SHA256",
            date_key="x-tos-date",
            content_hash_key="x-tos-content-sha256",
        ),
    }
    with HttpClient(**client_args) as client:
        data = "hello world"
        res = client.request(
            method="PUT",
            path=key_path,
            content=data,
        )
        etag = res.headers["ETag"]

        resp = client.request(
            method="HEAD",
            path=key_path,
        )
        assert etag == resp.headers["ETag"]

    async with AsyncHttpClient(**client_args) as client:
        data = "hello world"
        res = await client.request(
            method="PUT",
            path=key_path,
            content=data,
        )
        etag = res.headers["ETag"]

        resp = await client.request(
            method="HEAD",
            path=key_path,
        )
        assert etag == resp.headers["ETag"]


@pytest.fixture
def mock_http_server(monkeypatch):
    class MockServer:
        def __init__(self):
            self.responses = []
            self.requests = []
            self.delays = []
            self.active_connections = 0
            self.max_connections = 0

        def reset(self):
            self.responses = []
            self.requests = []
            self.delays = []

        def add_response(self, status_code=200, content=None, delay=0, exception=None):
            self.responses.append((status_code, content, delay, exception))
            return self

        def get_mock_client(self):
            client = MagicMock(spec=httpx.Client)
            client.request.side_effect = self._sync_request_handler
            return client

        def get_mock_async_client(self):
            client = MagicMock(spec=httpx.AsyncClient)
            client.request.side_effect = self._async_request_handler
            return client

        def _sync_request_handler(self, *args, **kwargs):
            try:
                self.active_connections += 1
                self.max_connections = max(self.max_connections, self.active_connections)

                self.requests.append((time.time(), args, kwargs))

                if not self.responses:
                    raise httpx.RequestError("No responses configured")

                status_code, content, delay, exception = self.responses.pop(0)

                if exception is not None:
                    raise exception

                if delay > 0:
                    time.sleep(delay)

                response = MagicMock(spec=httpx.Response)
                response.status_code = status_code
                if content:
                    response.text = content
                    response.content = content.encode()
                return response
            finally:
                self.active_connections -= 1

        async def _async_request_handler(self, *args, **kwargs):
            try:
                self.active_connections += 1
                self.max_connections = max(self.max_connections, self.active_connections)

                self.requests.append((time.time(), args, kwargs))

                if not self.responses:
                    raise httpx.RequestError("No responses configured")

                status_code, content, delay, exception = self.responses.pop(0)

                if exception is not None:
                    raise exception

                if delay > 0:
                    await asyncio.sleep(delay)

                response = MagicMock(spec=httpx.Response)
                response.status_code = status_code
                if content:
                    response.text = content
                    response.content = content.encode()
                return response
            finally:
                self.active_connections -= 1

    return MockServer()


@pytest.fixture
def mock_sleep(monkeypatch):
    def mock_sleep_imp(duration):
        mock_sleep_imp.total_delay += duration

    mock_sleep_imp.total_delay = 0
    monkeypatch.setattr(time, "sleep", mock_sleep_imp)
    monkeypatch.setattr(asyncio, "sleep", mock_sleep_imp)
    return mock_sleep_imp


def mock_client(retry_config: RetryPolicy = None) -> HttpClient:
    return HttpClient(base_url="https://example.com", retry_config=retry_config)


def mock_async_client(retry_config: RetryPolicy = None, max_connections: int = 50) -> AsyncHttpClient:
    return AsyncHttpClient(base_url="https://example.com", retry_config=retry_config, max_connections=max_connections)


def test_retryable_client(mock_http_server):
    policy = RetryPolicy(
        max_retires=1,
        max_wait=1,
        retry_on_exceptions=DEFAULT_RETRYABLE_EXCEPTIONS,
        retry_condition=_default_retry_condition,
    )
    client = mock_client(policy)
    client.client = mock_http_server.get_mock_client()

    mock_http_server.add_response(status_code=500)
    mock_http_server.add_response(status_code=200)
    response = client.request("GET", "/test")
    assert response.status_code == 200
    assert len(mock_http_server.requests) == 2

    mock_http_server.reset()
    mock_http_server.add_response(status_code=500)
    mock_http_server.add_response(status_code=429)
    mock_http_server.add_response(status_code=200)
    response = client.request("GET", "/test")
    assert response.status_code == 429
    assert len(mock_http_server.requests) == 2

    # overwrite the default max retry times
    mock_http_server.reset()
    mock_http_server.add_response(status_code=500)
    mock_http_server.add_response(status_code=429)
    mock_http_server.add_response(status_code=200)
    response = client.request("GET", "/test", max_retries=2)
    assert response.status_code == 200
    assert len(mock_http_server.requests) == 3

    # overwrite the default max retry duration
    mock_http_server.reset()
    mock_http_server.add_response(status_code=500, delay=1)
    mock_http_server.add_response(status_code=200)
    with pytest.raises(MaxRetriesExceeded):
        client.request("GET", "/test", max_retry_duration=0.5)
    assert len(mock_http_server.requests) == 1

    # should not retry the request for 4xx error
    mock_http_server.reset()
    mock_http_server.add_response(status_code=400)
    response = client.request("GET", "/test")
    assert response.status_code == 400
    assert len(mock_http_server.requests) == 1

    # Assume 403 is a retryable error because it can be solved in background
    def retry_condition(r, e):
        return r.status_code == 403

    mock_http_server.reset()
    mock_http_server.add_response(status_code=403)
    mock_http_server.add_response(status_code=200)
    response = client.request("GET", "/test", retry_condition=retry_condition)
    assert response.status_code == 200
    assert len(mock_http_server.requests) == 2

    # Retry the exceptions
    mock_http_server.reset()
    mock_http_server.add_response(exception=httpx.ConnectTimeout("error"))
    mock_http_server.add_response(exception=httpx.ReadTimeout("error"))
    mock_http_server.add_response(exception=httpx.WriteTimeout("error"))
    mock_http_server.add_response(exception=httpx.PoolTimeout("error"))
    mock_http_server.add_response(status_code=200)
    response = client.request("GET", "/test", max_retries=5)
    assert response.status_code == 200
    assert len(mock_http_server.requests) == 5

    mock_http_server.reset()
    mock_http_server.add_response(exception=httpx.DecodingError("error"))
    with pytest.raises(httpx.DecodingError):
        client.request("GET", "/test")
    assert len(mock_http_server.requests) == 1


@pytest.mark.asyncio
async def test_concurrency(mock_http_server):
    for _ in range(10):
        mock_http_server.add_response(status_code=200, delay=0.5)

    client = mock_async_client(RetryPolicy(max_retires=1), max_connections=5)
    client.client = mock_http_server.get_mock_async_client()

    tasks = [asyncio.create_task(client.request("GET", "/test")) for _ in range(10)]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    assert all(r.status_code == 200 for r in results)
    assert len(results) == 10
    assert mock_http_server.max_connections == 5
