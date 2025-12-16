# Copyright (c) Beijing Volcano Engine Technology Ltd.

from __future__ import annotations

import asyncio
import logging
import os
import uuid
from typing import TYPE_CHECKING, Any

from dotenv import load_dotenv
from pydantic import ValidationError

from daft.las.infra.http.auth import ApiKeyAuthProvider
from daft.las.infra.http.client import (
    DEFAULT_RETRY_POLICY,
    AsyncHttpClient,
)

if TYPE_CHECKING:
    from httpx import Response

DEFAULT_REQUEST_TIMEOUT = 1200
DEFAULT_MAX_CONCURRENCY = 100
DEFAULT_MAX_CONNECTIONS = 100
DEFAULT_MAX_KEEPALIVE_CONNECTIONS = 20
DEFAULT_MAX_RETRIES = 10
DEFAULT_INFERENCE_TYPE = "batch"
DEFAULT_LAS_BASE_URL = "http://sd1fm9vn6rmou3g0tfj6g.apigateway-cn-beijing.volceapi.com/"

ENDPOINT_MAP = {
    "online": os.environ.get("LAS_ONLINE_CHAT_ENDPOINT", "/api/v1/online/chat"),
    "batch": os.environ.get("LAS_BATCH_CHAT_ENDPOINT", "/api/v1/batch/chat"),
    "embedding_multimodal": os.environ.get("LAS_EMBEDDING_MULTIMODAL_ENDPOINT", "/api/v1/embedding/multimodal"),
    "embedding": os.environ.get("LAS_EMBEDDING_TEXT_ENDPOINT", "/api/v1/embedding"),
    "bots_chat": os.environ.get("LAS_BOT_CHAT_ENDPOINT", "/api/v1/bots/chat"),
}

logger = logging.getLogger(__name__)


class LasArkConfig:
    """Configs for the las ark service."""

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        request_timeout: int = DEFAULT_REQUEST_TIMEOUT,
        max_connections: int = DEFAULT_MAX_CONNECTIONS,
        max_keepalive_connections: int = DEFAULT_MAX_KEEPALIVE_CONNECTIONS,
        max_concurrency: int = DEFAULT_MAX_CONCURRENCY,
        inference_type: str = DEFAULT_INFERENCE_TYPE,
    ):
        self.base_url = base_url
        self.api_key = api_key
        self.request_timeout = request_timeout
        self.max_connections = max_connections
        self.max_keepalive_connections = max_keepalive_connections
        self.max_concurrency = max_concurrency
        self.inference_type = inference_type

    @staticmethod
    def from_env() -> LasArkConfig:
        load_dotenv()

        return LasArkConfig(
            base_url=os.environ.get("LAS_BASE_URL", DEFAULT_LAS_BASE_URL),
            api_key=os.environ.get("LAS_API_KEY") or os.environ.get("API_KEY"),
            request_timeout=int(os.environ.get("LAS_REQUEST_TIMEOUT", DEFAULT_REQUEST_TIMEOUT)),
            max_connections=int(os.environ.get("LAS_MAX_CONNECTIONS", DEFAULT_MAX_CONNECTIONS)),
            max_keepalive_connections=(
                int(os.environ.get("LAS_MAX_KEEPALIVE_CONNECTIONS", DEFAULT_MAX_KEEPALIVE_CONNECTIONS))
            ),
            max_concurrency=int(os.environ.get("LAS_MAX_CONCURRENCY", DEFAULT_MAX_CONCURRENCY)),
            inference_type=os.environ.get("LAS_INFERENCE_TYPE", DEFAULT_INFERENCE_TYPE).lower(),
        )


class LasArkClient:
    """Client for LAS routed ARK Service."""

    def __init__(self, config: LasArkConfig):
        assert config.base_url is not None
        assert config.inference_type.lower() in ENDPOINT_MAP
        assert config.api_key is not None

        self.api_key = config.api_key
        self.chat_endpoint = ENDPOINT_MAP[config.inference_type.lower()]
        self.max_retries = int(os.environ.get("LAS_MAX_RETRIES", DEFAULT_MAX_RETRIES))

        self.semaphore = asyncio.Semaphore(config.max_concurrency)
        self.client = AsyncHttpClient(
            base_url=config.base_url,
            auth_provider=ApiKeyAuthProvider(config.api_key),
            connect_timeout=config.request_timeout,
            read_timeout=config.request_timeout,
            write_timeout=config.request_timeout,
            max_connections=config.max_connections,
            max_keepalive_connections=config.max_keepalive_connections,
            http2=True,
            retry_config=DEFAULT_RETRY_POLICY.with_max_retries(self.max_retries).with_max_wait(600),
        )

    @classmethod
    def _retry_condition(cls, resp: Response | None, ex: Exception | None) -> bool:
        if resp is not None:
            try:
                result = resp.json()
            except Exception:
                result = None

            if isinstance(result, dict):
                code = result.get("code")
                if code in [429, 500, 502, 503, 504, 499]:
                    logger.warning("Request - Retryable API code: %s", code)
                    return True

            logger.warning("Request - Response status code: %s", resp.status_code)
            return 500 <= resp.status_code or resp.status_code in [429, 499]

        if ex is not None:
            message = str(ex)
            return "ConnectionTerminated" in message or "All connection attempts failed" in message

        return False

    async def _send_request(self, payload: dict[str, Any] | None) -> dict[str, Any] | None:
        if payload is None:
            return None

        request_id = str(uuid.uuid4())
        try:
            async with self.semaphore:
                resp = await self.client.request(
                    method="POST",
                    path=self.chat_endpoint,
                    json=payload,
                    headers={
                        "X-Request-ID": request_id,
                    },
                    retry_condition=self._retry_condition,
                    max_retry_duration=172800,
                )
            resp.raise_for_status()
            result = resp.json()

            code = result.get("code", 200)
            if code == 200:
                return result.get("data")

            logger.error(
                "Request %s - API error code: %s, message: %s",
                request_id,
                code,
                result.get("message"),
            )
            return {"error": result.get("message") or f"API error code: {code}"}

        except ValidationError as e:
            logger.exception("Request %s - Validation error: %s", request_id, e)
            return {"error": str(e)}
        except Exception as e:
            logger.exception("Request %s - Error: %s", request_id, e)
            raise

    async def batch_process(self, requests: list[dict[str, Any] | None]) -> list[dict[str, Any] | None]:
        """Process batch requests."""
        tasks = [self._send_request(req) for req in requests]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        return [
            result
            if not isinstance(result, Exception)  # type: ignore[misc]
            else {"error": str(result)}
            for result in results
        ]
