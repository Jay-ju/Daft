# Copyright (c) Beijing Volcano Engine Technology Ltd.

from __future__ import annotations

import base64
import logging
import os
from io import BytesIO
from typing import Any, BinaryIO

from daft.dependencies import pa
from daft.las.functions.types import Operator
from daft.las.functions.utils.common_utils import log_op_call, run_on_local_path

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
        extractor_type: str = "trafilatura",
        max_records: int | None = None,
        **kwargs: Any,
    ) -> None:
        """初始化CommonCrawl内容提取器

        Args:
            extractor_type: 选择使用的网页内容提取器类型
                可选值：["trafilatura", "justext", "goose3"]
                默认值："trafilatura"
            max_records: 限制处理的WARC记录数量
                默认值：None（无限制）
            **kwargs: 其他参数
        """  # noqa: D415
        super().__init__(**kwargs)
        self.extractor_type = extractor_type
        self.max_records = max_records
        self._loaded_extractors: dict[str, Any] = {}
        self._load_extractor(extractor_type)
        logger.info("Initialized CommonCrawl content extractor with extractor_type: %s", extractor_type)

        log_op_call(logger=logger, op=self.__class__.__name__, model_service_or_lib=self.extractor_type)

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

    def _process_warc_file(self, warc_path: str) -> list[dict[str, Any]]:
        def inner(local_path: str) -> list[dict[str, Any]]:
            with open(local_path, "rb") as f:
                return self._process_warc_stream(f, os.path.basename(warc_path))

        return run_on_local_path(warc_path, inner)

    def _process_warc_bytes(self, warc_bytes: bytes) -> list[dict[str, Any]]:
        return self._process_warc_stream(BytesIO(warc_bytes), "[binary_input]")

    def _process_warc_data(self, warc_input: str | bytes) -> list[dict[str, Any]]:
        logger.info("Processing input of type: %s", type(warc_input))
        if isinstance(warc_input, bytes):
            return self._process_warc_bytes(warc_input)
        elif isinstance(warc_input, str):
            if warc_input.startswith("data:application/octet-stream;base64,"):
                base64_data = warc_input.split(",", 1)[1]
                binary_data = base64.b64decode(base64_data)
                return self._process_warc_bytes(binary_data)
            else:
                return self._process_warc_file(warc_input)
        else:
            raise ValueError(f"Unsupported input type: {type(warc_input)}")

    def transform(self, warc_files: pa.Array) -> pa.Array:
        """批量处理WARC数据，提取网页正文

        Args:
            warc_files: 包含WARC数据的列，支持文件路径、TOS路径、二进制数据和base64编码

        Returns:
            pyarrow.Array: 处理后的列，包含提取的文本内容列表
                每个元素是一个字典列表，包含url、content、warc_file、extractor字段
        """  # noqa: D415
        logger.debug("Processing %s WARC files using %s", len(warc_files), self.extractor_type)

        warc_list = warc_files.to_pylist()
        results: list[list[dict[str, Any]] | None] = []

        for path in warc_list:
            try:
                if not path:
                    results.append(None)
                else:
                    results.append(self._process_warc_data(path))
            except Exception:
                logger.exception("Error processing WARC file: %s", path)
                results.append(None)

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
