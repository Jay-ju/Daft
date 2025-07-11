# Copyright (c) Beijing Volcano Engine Technology Ltd.

from __future__ import annotations

import asyncio
from typing import Any

from daft.dependencies import pa
from daft.las.functions.ark_llm.llm_generate_utils import gen_media_data
from daft.las.functions.types import Operator
from daft.las.infra.las_ark import (
    DEFAULT_MAX_CONCURRENCY,
    DEFAULT_REQUEST_TIMEOUT,
    LasArkClient,
    LasArkConfig,
)


class DoubaoEmbeddingVision(Operator):
    """**多模态向量生成处理器**

    **核心功能：**
    - 多模态向量化支持：支持图像/视频与文本的联合向量生成，实现跨模态检索能力，参考文档：https://www.volcengine.com/docs/82379/1523520
    - 输入格式自适应：
        - 原生支持图像/视频的base64编码、二进制数据、URL等输入格式
        - 自动处理媒体格式转换（JPEG/PNG/MP4/AVI等）
    - 模型名称：doubao-embedding-vision

    **输入输出规范：**
    - 输入格式：
        - 图片/视频数据：string类型，支持base64编码/url地址
        - 文本数据（可选）：string类型
    - 输出格式：
        - 默认模式：float数组类型的向量表示

    **支持模型版本示例：**
        - 250615
        - 250328
    """  # noqa: D415

    def __init__(
        self,
        model: str = "doubao-embedding-vision",
        version: str = "250615",
        access_key: str | None = None,
        account_id: str | None = None,
        multimodal_type: str = "image",
        image_format: str = "jpeg",
        video_format: str = "mp4",
        source_type: str = "url",
        encoding_format: str | None = None,
        dimensions: int = 2048,
        request_timeout: int = DEFAULT_REQUEST_TIMEOUT,
        max_concurrency: int = DEFAULT_MAX_CONCURRENCY,
        **kwargs: dict[str, Any],
    ) -> None:
        """使用火山方舟模型服务的图像向量化服务，将视频、图像和文本转化为向量.

        Args:
            version: 模型版本
                输入模型对应的版本信息。示例 250115
            multimodal_type: 媒体内容类型
                指定处理的是图像还是视频，默认是 image。可选值:
                - image: 图片
                - video: 视频
                - text: 文本
            image_format: 图片编码格式
                支持格式有：JPEG、PNG、WEBP、BMP、TIFF、ICO、DIB、ICNS、SGI、JPEG2000。其中，TIFF、 SGI、ICNS、JPEG2000 格式图片。
            video_format: 视频编码格式
                配置视频格式，默认是mp4。支持的视频格式：MP4、AVI、MOV。
                可以参考https://www.volcengine.com/docs/82379/1362931#%E8%A7%86%E9%A2%91%E7%90%86%E8%A7%A3给定的格式。
            source_type: 数据来源类型
                指定媒体数据的来源格式，默认 url。可选值:
                - binary: 原始二进制数据
                - base64: Base64编码数据
                - url: 网络资源地址（支持 http/https/tos）
            encoding_format: embedding的编码格式
                支持的编码格式有: float、base64
            dimensions: embedding的维度
                用于指定输出的向量维度。此参数仅doubao-embedding-vision-250615及后续版本支持，历史版本可以参见向量降维.
                取值范围： 1024 或 2048。默认值 2048
            request_timeout: 超时时间
                单次请求的超时时间（秒）
            max_concurrency: 并发数
                每个进程的最大并发数.
        """
        self.source_type = source_type.lower() if source_type else "url"
        assert self.source_type in ["binary", "base64", "url"], "source_type must be binary, base64 or url"
        self.multimodal_type = multimodal_type.lower() if multimodal_type else "image"
        assert self.multimodal_type in ["image", "video", "text"], "multimodal_type must be image, video, text"

        super().__init__(**kwargs)

        self.image_format = image_format.lower() if image_format else "jpeg"
        self.video_format = video_format.lower() if video_format else "mp4"
        self.encoding_format = encoding_format
        self.dimensions = dimensions

        ark_config = LasArkConfig.from_env()
        ark_config.request_timeout = request_timeout
        ark_config.max_concurrency = max_concurrency
        ark_config.max_connections = max_concurrency
        ark_config.max_keepalive_connections = max(int(max_concurrency / 2), 1)
        ark_config.inference_type = "embedding_multimodal"

        self.access_key = access_key or ark_config.access_key
        self.account_id = account_id or ark_config.account_id
        assert self.access_key is not None, "access_key is required"
        assert self.account_id is not None, "account_id is required"

        options_tmp = {
            "model_name": model,
            "version": version,
            "access_key": self.access_key,
            "account_id": self.account_id,
            "encoding_format": encoding_format,
            "dimensions": dimensions,
        }
        self.options = {k: v for k, v in options_tmp.items() if v is not None}

        self.client = LasArkClient(config=ark_config)

    @staticmethod
    def __return_column_type__() -> pa.DataType:
        return pa.list_(pa.float32())

    def transform(self, media_datas: pa.Array, text_contents: pa.Array | None = None) -> pa.Array:
        """批量使用大模型进行文本数组推理.

        该方法使用预加载的大模型对输入的文本数组进行批量推理，生成对应的模型输出结果。

        Args:
            media_datas: 传入待处理的图片或视频数据、文本数据。图片或视频数据支持传入base64编码或url、bytes；文本数据支持传入文本数据。
            text_contents: 图文向量化场景下，通过media_datas字段传入图片或者视频，通过text_contents字段传入文本数据。
                输入给模型的文本内容，需要满足一下条件
                单条文本以 utf-8 编码，长度不超过 100,000 字节。
                单条文本不超过模型的最大输入 token 数为 8k。

        Returns:
            返回模型处理后的向量化数组。类型为list[float]
        """
        message_generator = {
            "image": self._build_image_message,
            "video": self._build_video_message,
            "text": self._build_text_message,
        }[self.multimodal_type]

        media_list = media_datas.to_pylist()
        text_list = text_contents.to_pylist() if text_contents else [None] * len(media_datas)

        model_messages: list[list[dict[str, Any]]] = [
            message_generator(media_data=media, text_content=text) for media, text in zip(media_list, text_list)
        ]
        return self.process(model_messages)

    def process(self, messages: list[list[dict[Any, Any]]]) -> pa.Array:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        requests = [{"input": msg, **self.options} for msg in messages]
        results = loop.run_until_complete(self._async_requests(requests))

        return self._update_array_with_results(results)

    async def _async_requests(self, requests: list[dict[Any, Any]]) -> pa.Array:
        return await self.client.batch_process(requests)

    def _update_array_with_results(self, results: list[dict[str, Any]]) -> pa.Array:
        # init output_data and finish_reason_data
        output_data: list[list[float]] = []

        for i, result in enumerate(results):
            output_data.append(result.get("data", {}).get("embedding", []))

        return pa.array(output_data, type=pa.list_(pa.float32()))

    def _build_image_message(self, media_data: Any, text_content: str | None = None) -> list[dict[str, Any]]:
        media_url_or_data = gen_media_data("image", media_data, self.image_format, self.source_type)
        image_info = {"type": "image_url", "imageUrl": media_url_or_data}
        return self._assemble_message(image_content=image_info, text_content=text_content)

    def _build_video_message(self, media_data: Any, text_content: str | None = None) -> list[dict[str, Any]]:
        media_url_or_data = gen_media_data("video", media_data, self.video_format, self.source_type)
        video_info = {"type": "video_url", "videoUrl": media_url_or_data}
        return self._assemble_message(video_content=video_info, text_content=text_content)

    def _build_text_message(self, media_data: Any, text_content: str | None = None) -> list[dict[str, Any]]:
        return self._assemble_message(text_content=media_data)

    def _assemble_message(
        self,
        *,
        text_content: str | None = None,
        image_content: dict[str, Any] | None = None,
        video_content: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        input = []
        if text_content:
            input.append({"type": "text", "text": text_content})

        if image_content:
            input.append(image_content)
        if video_content:
            input.append(video_content)

        return input
