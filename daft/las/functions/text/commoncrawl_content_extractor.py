# Copyright (c) Beijing Volcano Engine Technology Ltd.

from __future__ import annotations

import logging
import os
from io import BytesIO
from typing import Any, BinaryIO

from daft.dependencies import pa
from daft.las.functions.types import Operator
from daft.las.functions.utils.common_utils import run_on_local_path, tracking_usage

logger = logging.getLogger(__name__)


class CommonCrawlContentExtractor(Operator):
    """**CommonCrawl网页内容提取器，支持多种解析策略**

    **核心功能：**
    - 多解析器支持：trafilatura/justext/goose3
    - 支持本地和远程WARC文件
    - 支持二进制数据和base64编码
    - 批量处理WARC文件，提取网页正文
    - 智能内容提取，过滤广告和导航元素

    **解析器对比：**
    - **justext**：速度最快，占用资源少，过滤比较严格
    - **trafilatura**：速度适中，内置规则+轻量NLP模型，提取质量高
    - **goose3**：速度较慢，启用多层逻辑提取，主要针对英文新闻
    """  # noqa: D415

    def __init__(
        self,
        warc_src_type: str,
        extractor_type: str = "trafilatura",
        max_records: int | None = None,
        **kwargs: Any,
    ) -> None:
        """初始化CommonCrawl内容提取器

        Args:
            warc_src_type: WARC数据来源类型
                支持的WARC格式类型，包含：
                - warc_binary: 原始二进制数据
                - warc_base64: Base64编码数据
                - warc_url: 文件路径或TOS存储链接
                可选值：["warc_binary", "warc_url", "warc_base64"]
            extractor_type: 选择使用的网页内容提取器类型
                可选值：["trafilatura", "justext", "goose3"]
                默认值："trafilatura"
            max_records: 限制处理的WARC记录数量
                默认值：None（无限制）
            **kwargs: 其他参数
        """  # noqa: D415
        super().__init__(**kwargs)
        self.warc_src_type = warc_src_type
        self.extractor_type = extractor_type
        self.max_records = max_records
        self._loaded_extractors: dict[str, Any] = {}
        self._load_extractor(extractor_type)
        logger.info(
            "Initialized CommonCrawl content extractor with warc_src_type: %s, extractor_type: %s",
            warc_src_type,
            extractor_type,
        )

        tracking_usage(op=self.__class__.__name__, model_service_or_lib=self.extractor_type)

    def _load_extractor(self, extractor_type: str) -> None:
        if extractor_type == "trafilatura":
            try:
                import trafilatura

                self._loaded_extractors["trafilatura"] = trafilatura
            except ImportError:
                raise RuntimeError("Please install trafilatura: pip install trafilatura")

        elif extractor_type == "justext":
            try:
                import justext

                self._loaded_extractors["justext"] = justext
            except ImportError:
                raise RuntimeError("Please install justext: pip install justext")

        elif extractor_type == "goose3":
            try:
                from goose3 import Goose

                self._loaded_extractors["goose3"] = Goose()
            except ImportError:
                raise RuntimeError("Please install goose3: pip install goose3")

        else:
            raise ValueError(
                f"Unsupported extractor type: {extractor_type!r}. " "Supported types: trafilatura, justext, goose3"
            )

    def _extract_with_trafilatura(self, html: str) -> str:
        try:
            trafilatura = self._loaded_extractors["trafilatura"]
            return trafilatura.extract(html, include_formatting=True) or ""
        except Exception as e:
            logger.error("Trafilatura extraction failed: %s", e)
            return ""

    def _extract_with_justext(self, html: str) -> str:
        try:
            justext = self._loaded_extractors["justext"]
            paragraphs = justext.justext(html)
            return "\n\n".join(p.text for p in paragraphs if not p.is_boilerplate)
        except Exception as e:
            logger.error("Justext extraction failed: %s", e)
            return ""

    def _extract_with_goose3(self, html: str) -> str:
        try:
            goose = self._loaded_extractors["goose3"]
            article = goose.extract(raw_html=html)
            return article.cleaned_text or ""
        except Exception as e:
            logger.error("Goose3 extraction failed: %s", e)
            return ""

    def _process_warc_stream(self, stream: BinaryIO, source: str) -> list[dict[str, Any]]:
        from warcio import ArchiveIterator
        from warcio.recordloader import ArchiveLoadFailed

        results: list[dict[str, Any]] = []
        processed_count = 0

        for record in ArchiveIterator(stream):
            processed_count += 1

            if record.rec_type != "response":
                continue

            content_type = record.http_headers.get_header("content-type", "")
            if not content_type.startswith("text/html"):
                continue

            try:
                html = record.content_stream().read().decode("utf-8", errors="ignore")

                if self.extractor_type == "trafilatura":
                    content = self._extract_with_trafilatura(html)
                elif self.extractor_type == "justext":
                    content = self._extract_with_justext(html)
                elif self.extractor_type == "goose3":
                    content = self._extract_with_goose3(html)
                else:
                    content = ""

                if content.strip():
                    results.append(
                        {
                            "url": record.rec_headers.get_header("WARC-Target-URI", ""),
                            "content": content,
                            "warc_file": source,
                            "extractor": self.extractor_type,
                        }
                    )

                    if self.max_records and len(results) >= self.max_records:
                        break

            except (ArchiveLoadFailed, UnicodeDecodeError) as e:
                logger.debug("Failed to process record in %s: %s", source, str(e))
                continue

        logger.info("Processed %s: total=%d, extracted=%d", source, processed_count, len(results))
        return results

    def transform(self, warc_files: pa.Array) -> pa.Array:
        """批量处理WARC数据，提取网页正文

        Args:
            warc_files: 包含WARC数据的列，支持以下格式：
                - warc_base64: base64编码的WARC字符串
                - warc_url: WARC文件路径或TOS链接
                - warc_binary: 原始WARC二进制数据

        Returns:
            pyarrow.Array: 提取结果列表，每个元素包含以下字段：
                - url: 网页URL
                - content: 提取的正文内容
                - warc_file: 源WARC文件标识
                - extractor: 使用的提取器名称

        Raises:
            ValueError: 当warc_src_type不支持时抛出
            Exception: 处理过程中出现未捕获的异常时抛出
        """  # noqa: D415
        logger.info("Processing WARC source type: %s", self.warc_src_type)

        results = []
        warc_list = warc_files.to_pylist()

        for warc_input in warc_list:
            try:
                if self.warc_src_type == "warc_base64":
                    from daft.las.functions.utils.common_utils import base64_to_byte

                    warc_binary = base64_to_byte(warc_input)
                    source_name = "[base64_input]"
                    extracted_content = self._process_warc_stream(BytesIO(warc_binary), source_name)
                elif self.warc_src_type == "warc_url":
                    source_name = os.path.basename(warc_input) if isinstance(warc_input, str) else "[url_input]"

                    def process_file_directly(local_path: str) -> list[dict[str, Any]]:
                        with open(local_path, "rb") as file:
                            return self._process_warc_stream(file, source_name)

                    extracted_content = run_on_local_path(warc_input, process_file_directly)
                elif self.warc_src_type == "warc_binary":
                    warc_binary = warc_input
                    source_name = "[binary_input]"
                    extracted_content = self._process_warc_stream(BytesIO(warc_binary), source_name)
                else:
                    raise ValueError(f"Unsupported warc_src_type: {self.warc_src_type}")
                results.append(extracted_content)
            except Exception as e:
                logger.error("Failed to process WARC data: %s", str(e))
                results.append([])

        return pa.array(results, type=self.__return_column_type__())

    @staticmethod
    def __return_column_type__() -> pa.DataType:
        fields = [
            pa.field("url", pa.string()),
            pa.field("content", pa.string()),
            pa.field("warc_file", pa.string()),
            pa.field("extractor", pa.string()),
        ]
        return pa.list_(pa.struct(fields))
