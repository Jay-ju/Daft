# Copyright (c) Beijing Volcano Engine Technology Ltd.

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Any

from httpx import HTTPStatusError

from daft.las.infra.http.auth import VolcOpenApiAuthProvider
from daft.las.infra.http.client import HttpClient
from daft.las.infra.http.retry import RetryPolicy

if TYPE_CHECKING:
    from httpx import Response

logger = logging.getLogger(__name__)

DEFAULT_MAX_RETRIES = 10
DEFAULT_TIMEOUT = 30

HOST = os.environ.get("VOLC_OPENAPI_HOST", "open.volcengineapi.com")
PATH = "/"
VERSION = os.environ.get("VOLC_OPENAPI_VERSION", "2024-06-30")


class OpenAPIClient:
    """Volc OpenAPI Client."""

    def __init__(
        self,
        service: str,
        region: str,
        access_key: str,
        secret_key: str,
        session_token: str | None = None,
        host: str = HOST,
        path: str = PATH,
        version: str = VERSION,
    ):
        self.host = host
        self.path = path
        self.version = version

        self.client = HttpClient(
            base_url=self.host,
            auth_provider=VolcOpenApiAuthProvider(
                service=service,
                region=region,
                access_key=access_key,
                secret_key=secret_key,
                session_token=session_token,
            ),
            read_timeout=DEFAULT_TIMEOUT,
            write_timeout=DEFAULT_TIMEOUT,
            retry_config=RetryPolicy(max_retires=int(os.environ.get("LAS_MAX_RETRIES", DEFAULT_MAX_RETRIES))),
        )

    def call_api(
        self, method: str, params: dict[str, Any], headers: dict[str, Any], action: str, body: dict[str, Any]
    ) -> Response:
        response = self.client.request(
            method=method,
            path=self.path,
            params={"Action": action, "Version": self.version, **params},
            headers=headers,
            json=body,
        )

        if response.status_code >= 400:
            response_meta = response.json().get("ResponseMetadata")
            raise HTTPStatusError(message=str(response_meta), request=response.request, response=response)

        return response
