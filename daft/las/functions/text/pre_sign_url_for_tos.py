# Copyright (c) Beijing Volcano Engine Technology Ltd.

from __future__ import annotations

import logging
from typing import Any
from urllib.parse import urlparse

import tos
from tos import HttpMethodType

from daft.dependencies import pa
from daft.las.functions.types import Operator
from daft.las.io.tos import TOSConfig

logger = logging.getLogger(__name__)


class PreSignUrlForTos(Operator):
    """生成 TOS 文件路径签名.

    通过该算子对 TOS 文件路径生成带签名的 URL，您可直接用该 URL 发起 HTTP 请求，也可以将该 URL 共享给第三方实现访问授权。
    """

    def __init__(
        self,
        expires: int = 3600,
        **kwargs: Any,
    ):
        """对 TOS 文件路径生成带签名的 URL.

        Args:
            expires: URL过期时间
                签名 URL 的过期时间，单位秒. 默认值 3600.
        """
        super().__init__(**kwargs)
        self.expires = expires or 3600

        tosconfig = TOSConfig.from_env()
        self.tos_client = tos.TosClientV2(
            tosconfig.access_key, tosconfig.secret_key, tosconfig.endpoint, tosconfig.region
        )

    def _get_pre_signed_policy_url(self, url: str) -> Any:
        parsed_url = urlparse(url)
        schema_name = parsed_url.scheme

        if schema_name == "http" or schema_name == "https":
            logger.warning("TOS URL schema is http or https, will not sign url: %s", url)
            return url

        if schema_name != "tos" and schema_name != "s3":
            logger.error("Invalid TOS URL schema: %s, expect tos or s3, got %s", schema_name, url)
            return None

        bucket_name = parsed_url.netloc
        object_name = parsed_url.path.lstrip("/")
        if not bucket_name or not object_name:
            logger.error("Invalid TOS URL: %s", url)
            return None
        pre_signed_policy_url = self.tos_client.pre_signed_url(
            HttpMethodType.Http_Method_Get,
            bucket=bucket_name,
            key=object_name,
            expires=self.expires,
        )
        return pre_signed_policy_url.signed_url

    def transform(self, urls: pa.Array) -> pa.Array:
        """生成 TOS 文件路径签名 URL。当路径 schema 是 http 或 https 时，直接返回路径；若是 tos 或 s3，则对路径进行签名，其他情况返回 None.

        Args:
            urls: 输入的 TOS 文件路径.

        Returns:
            生成的签名 URL.
        """
        processed = [self._get_pre_signed_policy_url(url.as_py()) for url in urls]
        return pa.array(processed, type=pa.string())

    @staticmethod
    def __return_column_type__() -> pa.DataType:
        return pa.string()

    def __del__(self) -> None:
        """Close TOS client."""
        try:
            if hasattr(self, "tos_client"):
                self.tos_client.close()
                logger.debug("TOS client closed successfully")
        except Exception as e:
            logger.warning("Error closing TOS client: %s", e)
