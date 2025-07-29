# Copyright (c) Beijing Volcano Engine Technology Ltd.

from __future__ import annotations

import asyncio
import logging
import threading
from datetime import datetime
from typing import TYPE_CHECKING, Any, Callable

from httpx import AsyncClient, Client, Limits, NetworkError, Response, Timeout, TimeoutException, TooManyRedirects

from daft.las.infra.http.auth import AnonymousProvider, AuthProvider, FunctionAuth
from daft.las.infra.http.retry import RetryPolicy

if TYPE_CHECKING:
    from collections.abc import AsyncIterable, Iterable

logger = logging.getLogger(__name__)


def default_retry_condition(resp: Response) -> bool:
    return 500 <= resp.status_code or resp.status_code == 429


DEFAULT_RETRYABLE_EXCEPTIONS = [TimeoutException, NetworkError, TooManyRedirects]
DEFAULT_RETRY_POLICY = RetryPolicy(
    retry_on_exceptions=DEFAULT_RETRYABLE_EXCEPTIONS,
    retry_condition=default_retry_condition,
)


class BaseClient:
    def __init__(
        self,
        base_url: str,
        connect_timeout: float | None = 5,
        read_timeout: float | None = None,
        write_timeout: float | None = None,
        pool_timeout: float | None = 5,
        retry_config: RetryPolicy = DEFAULT_RETRY_POLICY,
        auth_provider: AuthProvider = AnonymousProvider(),
    ):
        """Create a new HTTP client.

        Args:
            base_url (str): The base URL for the client.
            connect_timeout (float, optional): The maximum amount of time to wait until a socket connection to the
                                                requested host is established. Defaults to 5 seconds. A ConnectTimeout
                                                exception is raised if unable to connect with this time frame.
            read_timeout (float, optional):  The maximum duration to wait for a chunk of data to be received. Defaults to None.
                                                A ReadTimeout exception is raised if unable to receive data from this time frame.
            write_timeout (float, optional): The maximum duration to wait for a chunk of data to be sent. Defaults to None.
                                                A WriteTimeout exception is raised if unable to send data from this time frame.
            pool_timeout (float, optional): The maximum duration to wait for acquiring a connection from the connection pool.
                                                Defaults to 5. A PoolTimeout exception is raised if unable to acquire a connection
                                                from the connection pool.
        """
        base_url = base_url.rstrip("/")
        self.base_url = base_url if base_url.startswith("http") else "https://" + base_url
        self.timeout = Timeout(connect=connect_timeout, read=read_timeout, write=write_timeout, pool=pool_timeout)
        self.auth = FunctionAuth(auth_provider)
        self.retry_policy = retry_config


class HttpClient(BaseClient):
    def __init__(
        self,
        base_url: str,
        max_connections: int = 50,
        max_keepalive_connections: int = 20,
        connect_timeout: float | None = 5,
        read_timeout: float | None = None,
        write_timeout: float | None = None,
        pool_timeout: float | None = 5,
        http2: bool = True,
        retry_config: RetryPolicy = RetryPolicy(),
        auth_provider: AuthProvider = AnonymousProvider(),
    ) -> None:
        super().__init__(
            base_url, connect_timeout, read_timeout, write_timeout, pool_timeout, retry_config, auth_provider
        )
        self.client = Client(
            base_url=self.base_url,
            timeout=self.timeout,
            http2=http2,
            limits=Limits(
                max_connections=max_connections,
                max_keepalive_connections=min(max_connections, max_keepalive_connections),
            ),
            follow_redirects=True,
        )
        self.semaphore = threading.Semaphore(max_connections)

    def __enter__(self) -> HttpClient:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        if self.client:
            self.client.close()

    def request(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, str] | None = None,
        headers: dict[str, str] | None = None,
        content: str | bytes | Iterable[bytes] | AsyncIterable[bytes] | None = None,
        json: Any | None = None,
        max_retries: int | None = None,
        max_retry_duration: float | None = None,
        retry_on_exceptions: Iterable[type[Exception]] | None = None,
        retry_condition: Callable[[Response], bool] | None = None,
        timeout: float | Timeout | None = None,
    ) -> Response:
        return self.retry_policy.retry_on(
            func=lambda attempt: self._send_request(
                method, path, attempt, params=params, headers=headers, content=content, json=json, timeout=timeout
            ),
            max_retries=max_retries,
            max_retry_duration=max_retry_duration,
            retry_on_exceptions=retry_on_exceptions,
            retry_condition=retry_condition,
        )

    def _send_request(
        self,
        method: str,
        url: str,
        attempt: int,
        *,
        params: dict[str, str] | None = None,
        headers: dict[str, str] | None = None,
        content: str | bytes | Iterable[bytes] | AsyncIterable[bytes] | None = None,
        json: Any | None = None,
        timeout: float | Timeout | None = None,
    ) -> Response:
        with self.semaphore:
            start = datetime.now()
            resp = self.client.request(
                method=method,
                url=url,
                params=params,
                headers=headers,
                content=content,
                json=json,
                timeout=timeout or self.timeout,
                auth=self.auth,
            )
            elapsed = (datetime.now() - start).total_seconds()
            logger.info("Attempt: %d - %s %s -> %d (%.2fms)", attempt, method, url, resp.status_code, elapsed * 1000)

            return resp


class AsyncHttpClient(BaseClient):
    def __init__(
        self,
        base_url: str,
        max_connections: int = 50,
        max_keepalive_connections: int = 20,
        connect_timeout: float | None = 5,
        read_timeout: float | None = None,
        write_timeout: float | None = None,
        pool_timeout: float | None = 5,
        http2: bool = True,
        retry_config: RetryPolicy = RetryPolicy(),
        auth_provider: AuthProvider = AnonymousProvider(),
    ):
        super().__init__(
            base_url, connect_timeout, read_timeout, write_timeout, pool_timeout, retry_config, auth_provider
        )
        self.client = AsyncClient(
            base_url=self.base_url,
            timeout=self.timeout,
            http2=http2,
            limits=Limits(
                max_connections=max_connections,
                max_keepalive_connections=min(max_connections, max_keepalive_connections),
            ),
            follow_redirects=True,
        )
        self.semaphore = asyncio.Semaphore(max_connections)

    async def __aenter__(self) -> AsyncHttpClient:
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        if self.client:
            await self.client.aclose()

    async def request(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, str] | None = None,
        headers: dict[str, str] | None = None,
        content: str | bytes | Iterable[bytes] | AsyncIterable[bytes] | None = None,
        json: Any | None = None,
        max_retries: int | None = None,
        max_retry_duration: float | None = None,
        retry_on_exceptions: Iterable[type[Exception]] | None = None,
        retry_condition: Callable[[Response], bool] | None = None,
        timeout: float | Timeout | None = None,
    ) -> Response:
        return await self.retry_policy.async_retry_on(
            func=lambda attempt: self._send_request(
                method, path, attempt, params=params, headers=headers, content=content, json=json, timeout=timeout
            ),
            max_retries=max_retries,
            max_retry_duration=max_retry_duration,
            retry_on_exceptions=retry_on_exceptions,
            retry_condition=retry_condition,
        )

    async def _send_request(
        self,
        method: str,
        url: str,
        attempt: int,
        *,
        params: dict[str, str] | None = None,
        headers: dict[str, str] | None = None,
        content: str | bytes | Iterable[bytes] | AsyncIterable[bytes] | None = None,
        json: Any | None = None,
        timeout: float | Timeout | None = None,
    ) -> Response:
        async with self.semaphore:
            start = datetime.now()
            resp = await self.client.request(
                method=method,
                url=url,
                params=params,
                headers=headers,
                content=content,
                json=json,
                timeout=timeout or self.timeout,
                auth=self.auth,
            )
            elapsed = (datetime.now() - start).total_seconds()
            logger.info("Attempt: %d - %s %s -> %d (%.2fms)", attempt, method, url, resp.status_code, elapsed * 1000)

            return resp
