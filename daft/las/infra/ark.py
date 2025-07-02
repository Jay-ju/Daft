# Copyright (c) Beijing Volcano Engine Technology Ltd.

from __future__ import annotations

import os
import re
from typing import Any
from urllib.parse import urlparse

from volcenginesdkarkruntime import Ark, AsyncArk

from daft.las.utils import get_ak_sk, is_static_credential, not_blank


class ArkConfig:
    """The configuration for ark service."""

    def __init__(
        self,
        base_url: str | None = None,
        region: str | None = None,
        access_key: str | None = None,
        secret_key: str | None = None,
        api_key: str | None = None,
        max_retries: int = 3,
        timeout: int = 600,
    ):
        self.base_url, self.region = self._parse_base_url(base_url, region)
        self.access_key = access_key
        self.secret_key = secret_key
        self.api_key = api_key
        self.max_retries = max_retries
        self.timeout = timeout

        self._check_credential_info()

    def _check_credential_info(self) -> None:
        if not is_static_credential(self.access_key, self.secret_key) and not not_blank(self.api_key):
            raise ValueError("Cannot found credentials or credential provider.")

    def _parse_base_url(self, base_url: str | None, region: str | None = None) -> tuple[str, str]:
        if base_url is None:
            raise ValueError("base_url must be specified")

        if region is None:
            try:
                parsed = urlparse(base_url)
            except Exception:
                raise ValueError(f"invalid base_url format, check your base_url: {base_url}")
            region = self._extract_region_regex(parsed.hostname)
            if region is None:
                raise ValueError("region must be specified")

        return base_url, region

    @classmethod
    def _extract_region_regex(cls, hostname: str | None) -> str | None:
        if hostname is None:
            return None
        pattern = r"ark\.([a-z0-9-]+)\.(?:i)?volces\.com"
        match = re.fullmatch(pattern, hostname)
        if match:
            return match.group(1)
        else:
            return None

    @staticmethod
    def from_env() -> ArkConfig:
        access_key, secret_key = get_ak_sk("ark")
        return ArkConfig(
            base_url=os.getenv("ARK_BASE_URL"),
            region=os.getenv("ARK_REGION"),
            access_key=access_key,
            secret_key=secret_key,
            api_key=os.getenv("ARK_API_KEY"),
            max_retries=int(os.getenv("ARK_MAX_RETRIES", 3)),
            timeout=int(os.getenv("ARK_TIMEOUT", 600)),
        )


def get_ark_client(config: ArkConfig) -> Ark:
    """Get the ark client."""
    return Ark(**_argconfig_to_options(config))


def get_async_ark_client(config: ArkConfig) -> AsyncArk:
    """Get the ark async client."""
    return AsyncArk(**_argconfig_to_options(config))


def _argconfig_to_options(config: ArkConfig) -> dict[str, Any]:
    options = {
        "base_url": config.base_url,
        "ak": config.access_key,
        "sk": config.secret_key,
        "api_key": config.api_key,
        "region": config.region,
        "timeout": config.timeout,
        "max_retries": config.max_retries,
    }
    return {k: v for k, v in options.items() if v is not None}
