# Copyright (c) Beijing Volcano Engine Technology Ltd.

from __future__ import annotations

import logging
from typing import Any

from daft.dependencies import pa
from daft.las.functions.types import EventLooper, Operator
from daft.las.functions.utils.common_utils import tracking_usage
from daft.las.infra.las_ark import (
    DEFAULT_MAX_CONCURRENCY,
    DEFAULT_REQUEST_TIMEOUT,
    LasArkClient,
    LasArkConfig,
)

logger = logging.getLogger(__name__)


class DoubaoEmbeddingText(Operator):
    """**文本向量化生成处理器**

    **核心功能：**
    - 文本向量化支持：支持文本的向量化生成，实现文本检索能力，参考文档：https://www.volcengine.com/docs/82379/1521766

    **输入输出规范：**
    - 输入格式：
        - 文本数据：string类型
    - 输出格式：
        - 默认模式：float数组类型的向量表示
    """  # noqa: D415

    def __init__(
        self,
        model: str = "doubao-embedding-large",
        version: str | None = None,
        api_key: str | None = None,
        encoding_format: str | None = None,
        request_timeout: int = DEFAULT_REQUEST_TIMEOUT,
        max_concurrency: int = DEFAULT_MAX_CONCURRENCY,
        **kwargs: dict[str, Any],
    ) -> None:
        """使用火山方舟模型服务的文本向量化服务，将文本转化为向量.

        Args:
            model: 模型名称，示例：doubao-embedding、doubao-embedding-large
            version: 模型版本
                输入模型对应的版本信息。示例 text-250515
            encoding_format: embedding的编码格式
                支持的编码格式有: float、base64
            request_timeout: 超时时间
                单次请求的超时时间（秒）
            max_concurrency: 并发数
                每个进程的最大并发数.
        """
        super().__init__(**kwargs)

        self.encoding_format = encoding_format

        ark_config = LasArkConfig.from_env()
        ark_config.request_timeout = request_timeout
        ark_config.max_concurrency = max_concurrency
        ark_config.max_connections = max_concurrency
        ark_config.max_keepalive_connections = max(int(max_concurrency / 2), 1)
        ark_config.inference_type = "embedding"
        ark_config.api_key = api_key or ark_config.api_key

        options_tmp = {
            "model_name": model,
            "version": version,
            "encoding_format": encoding_format,
        }
        self.options = {k: v for k, v in options_tmp.items() if v is not None}

        self.client = LasArkClient(config=ark_config)
        self.event_loop = EventLooper()

        tracking_usage(op=self.__class__.__name__, model_service_or_lib=model)

    @staticmethod
    def __return_column_type__() -> pa.DataType:
        return pa.list_(pa.float32())

    def transform(self, text_contents: pa.Array) -> pa.Array:
        """批量使用大模型进行文本向量化推理.

        该方法使用文本向量化模型对输入的文本数组进行批量推理，生成对应的模型输出结果。

        Args:
            text_contents: 传入待处理的文本数据。

        Returns:
            返回模型处理后的向量化数组。类型为list[float]
        """
        model_messages: list[list[str] | None] = [self._build_text_message(text) for text in text_contents.to_pylist()]
        return self.process(model_messages)

    def process(self, messages: list[list[str] | None]) -> pa.Array:
        try:
            requests = [{"input": msg, **self.options} if msg and len(msg) > 0 else None for msg in messages]
            results = self.event_loop.run(self._async_requests(requests))
            return self._update_array_with_results(results)
        except Exception:
            logger.exception("Error in transform.")
            return pa.nulls(len(messages), type=self.__return_column_type__())

    async def _async_requests(self, requests: list[dict[str, Any] | None]) -> pa.Array:
        return await self.client.batch_process(requests)

    def _update_array_with_results(self, results: list[dict[str, Any] | None]) -> pa.Array:
        # init output_data and finish_reason_data
        output_data: list[list[float] | None] = []
        for i, result in enumerate(results):
            if result is None:
                output_data.append(None)
                continue
            output_data.append(result.get("data", [])[0].get("embedding"))

        return pa.array(output_data, type=pa.list_(pa.float32()))

    def _build_text_message(self, text_content: str | None = None) -> list[str]:
        input = []
        if text_content:
            input.append(text_content)
        return input
