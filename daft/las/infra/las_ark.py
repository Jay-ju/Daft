# Copyright (c) Beijing Volcano Engine Technology Ltd.

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any

import httpx
from dotenv import load_dotenv
from pydantic import ValidationError
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

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

        self.client = httpx.AsyncClient(
            base_url=config.base_url,
            timeout=httpx.Timeout(config.request_timeout),
            limits=httpx.Limits(
                max_connections=config.max_connections, max_keepalive_connections=config.max_keepalive_connections
            ),
            http2=True,
        )
        self.semaphore = asyncio.Semaphore(config.max_concurrency)
        self.chat_endpoint = ENDPOINT_MAP[config.inference_type.lower()]

    @retry(  # type: ignore[misc]
        wait=wait_exponential(multiplier=1, min=2, max=5),
        stop=stop_after_attempt(os.environ.get("LAS_MAX_RETRIES", DEFAULT_MAX_RETRIES)),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        retry=retry_if_exception_type((httpx.HTTPError, Exception)),
    )
    async def _send_request(self, payload: dict[str, Any] | None) -> dict[str, Any] | None:
        if payload is None:
            return None

        async with self.semaphore:
            try:
                response = await self.client.post(
                    self.chat_endpoint,
                    json=payload,
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                    },
                )
                response.raise_for_status()
                result = response.json()

                status_code = result.get("code")
                if status_code in [429, 500, 502, 503, 504]:
                    logger.warning("Retryable error code: %s", status_code)
                    raise Exception(f"Retryable error: {result.get('message')}")
                elif status_code != 200:
                    logger.error("API request failed: %s", result.get("message"))
                    return {"error": result.get("message")}
                return result["data"]
            except ValidationError as e:
                err_msg = f"Parameter validation failed: {e}"
                logger.exception(err_msg)
                return {"error": err_msg}
            except Exception:
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
