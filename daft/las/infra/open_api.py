# Copyright (c) Beijing Volcano Engine Technology Ltd.

from __future__ import annotations

import logging
import os
from typing import Any

import requests
from tenacity import before_sleep_log, retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from daft.las.infra.top.volcauth import VolcAuth

logger = logging.getLogger(__name__)

DEFAULT_MAX_RETRIES = 10
DEFAULT_TIMEOUT = 30

HOST = "open.volcengineapi.com"
PATH = "/"
VERSION = "2024-06-30"


class RetryableError(Exception):
    pass


class OpenAPIClient:
    """Volc OpenAPI Client."""

    def __init__(
        self,
        service: str,
        region: str,
        access_key: str,
        secret_key: str,
        session_token: str | None = None,
    ):
        self.auth = VolcAuth(access_key, secret_key, region, service, session_token=session_token)

    @retry(  # type: ignore[misc]
        wait=wait_exponential(multiplier=1, min=2, max=5),
        stop=stop_after_attempt(os.environ.get("LAS_MAX_RETRIES", DEFAULT_MAX_RETRIES)),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        retry=retry_if_exception_type(RetryableError),
    )
    def call_api(
        self, method: str, params: dict[str, Any], headers: dict[str, Any], action: str, body: dict[str, Any]
    ) -> requests.Response:
        try:
            response = requests.request(
                method=method,
                url=f"https://{HOST}{PATH}",
                headers=headers,
                params={"Action": action, "Version": VERSION, **params},
                json=body,
                auth=self.auth,
                timeout=DEFAULT_TIMEOUT,
            )
            response.raise_for_status()
            return response

        except requests.HTTPError as e:
            logger.error(
                "HTTP error, request_id: %s, status code: %s, info: %s",
                e.response.json()["ResponseMetadata"]["RequestId"],
                e.response.status_code,
                e.response.text,
            )
            if e.response.status_code in [429, 500, 502, 503, 504]:
                logger.error("Status code: %s, now retry", e.response.status_code)
                raise RetryableError
            raise e
        except (requests.ConnectionError, requests.Timeout) as e:
            logger.error("Network failure: %s. Now retry", e)
            raise RetryableError
        except requests.RequestException as e:
            logger.error("Request error: %s", e)
            raise e
        except Exception as e:
            logger.error("Unknown error: %s", e)
            raise e
