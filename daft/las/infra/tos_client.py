# Copyright (c) Beijing Volcano Engine Technology Ltd.

from __future__ import annotations

import logging
import os
import threading
from typing import Any
from urllib.parse import urlparse

import tos
from tos import HttpMethodType
from tosfs.certification import NoLockUrlCredentialsProvider

from daft.las.io.tos import TOSConfig
from daft.las.utils import (
    is_static_credential,
)

logger = logging.getLogger(__name__)


class TosClient:
    _instance = None
    _initialized = False
    _lock = threading.Lock()

    def __new__(cls, *args: Any, **kwargs: Any) -> TosClient:
        with cls._lock:
            if not cls._instance:
                cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, config: TOSConfig | None = None) -> None:
        """Initialize TosClient.

        Args:
            config: TOS config. Defaults to None.
        """
        with self._lock:
            if self._initialized:
                return

        if config is None:
            try:
                config = TOSConfig.from_env()
                logger.info("Fetch config from env.")
            except ValueError:
                raise ValueError("Cannot found TOS config.")

        self.tos_client = self._create_tos_client(config)
        self._initialized = True

    @staticmethod
    def _create_tos_client(config: TOSConfig | None) -> tos.TosClientV2:
        if config is None:
            raise ValueError("TOS config is not provided")

        if is_static_credential(config.access_key, config.secret_key):
            return tos.TosClientV2(
                endpoint=config.endpoint,
                region=config.region,
                ak=config.access_key,
                sk=config.secret_key,
                security_token=config.session_token,
            )

        if config.credentials_provider_url:
            if config.credentials_provider_url.isspace():
                raise ValueError("credentials_provider_url cannot be empty")

            return tos.TosClientV2(
                endpoint=config.endpoint,
                region=config.region,
                credentials_provider=NoLockUrlCredentialsProvider(config.credentials_provider_url),
            )

        if config.credentials_provider:
            return tos.TosClientV2(
                endpoint=config.endpoint,
                region=config.region,
                credentials_provider=lambda: config.credentials_provider().to_tosfs_credentials(),
            )

        raise ValueError("Cannot found credentials or credential provider.")

    def pre_sign_url(self, url: str, expires: int | None = None) -> str:
        """Pre-signs a URL for TOS.

        Args:
            url (str): The URL to pre-sign.
            expires (int, optional): Expiration time in seconds. Defaults to 3600 * 24. Set TOS_PRE_SIGN_URL_EXPIRES to override.

        Returns:
            str: The pre-signed URL.
        """
        expires = expires or int(os.getenv("TOS_PRE_SIGN_URL_EXPIRES", 3600 * 24))
        if expires > 2592000:
            logger.warning("Expires is too large, set to 2592000")
            expires = 2592000
        if expires < 1:
            logger.warning("Expires is too small, set to 1")
            expires = 1

        parsed_url = urlparse(url)
        bucket_name = parsed_url.netloc
        object_name = parsed_url.path.lstrip("/")
        return self.tos_client.pre_signed_url(
            HttpMethodType.Http_Method_Get,
            bucket=bucket_name,
            key=object_name,
            expires=expires,
        ).signed_url
