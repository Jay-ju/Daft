# Copyright (c) Beijing Volcano Engine Technology Ltd.

from __future__ import annotations

import logging
import re
from typing import Any

from daft.dependencies import pa
from daft.las.functions.types import Operator
from daft.las.functions.utils.text_utils import strip_markdown_images

logger = logging.getLogger(__name__)


class ChunkTextSentenceSplitter(Operator):
    """多格式文本分块处理器，支持结构化解析与智能切分.

    核心功能：
    - 三格式支持：
        • 纯文本：基于段落/标点的语义分块
        • Markdown：保留文档结构，过滤图片链接
        • HTML：提取正文内容，保留章节结构
    - 智能分块策略：结合语义与语法规则
    - 重叠优化：保持上下文连贯性

    技术实现：
    ▸ 解析引擎：BeautifulSoup（HTML）/llama-index（Markdown）
    ▸ 分块算法：递归式语义分割（sentence splitter）
    ▸ 文本归一化：冗余字符过滤与空白标准化
    """

    def __init__(
        self,
        content_type: str = "text",
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        **kwargs: Any,
    ) -> None:
        """文本分句切分器初始化方法.

        Args:
            content_type: 文本类型
                描述：被切分的文本类型
                可选值：["text", "md", "html"]
                默认值："text"
            chunk_size: chunk长度
                描述：chunk的长度，单位为字符
                默认值：500
            chunk_overlap: chunk重叠长度
                描述：切分文本时chunk之间重叠的最大长度
                默认值：50
        """
        super().__init__(**kwargs)

        supported_types = ["text", "md", "html"]
        self.content_type = content_type
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        if self.content_type not in supported_types:
            raise ValueError(
                f"Unsupported content type: {self.content_type!r}. " f"Supported types: {', '.join(supported_types)}"
            )

        from llama_index.core.node_parser import SentenceSplitter
        from llama_index.readers.file import HTMLTagReader, MarkdownReader

        self._sentence_splitter = SentenceSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separator=" ",
            paragraph_separator="\n\n",
            secondary_chunking_regex="[^,，.;；。？?！!]+[,，.;；。？?！!]?",
        )
        logger.info("The chunk size is %d, the chunk overlap is %d", self.chunk_size, self.chunk_overlap)
        logger.info("The content type is %s", self.content_type)

        self._md_parser = MarkdownReader()
        self._html_parser = HTMLTagReader("section")

    def transform(self, texts: pa.Array) -> pa.Array:
        """批量处理文本数组生成分句chunks.

        Args:
            texts: 待处理的文本数组，要求元素类型为字符串。

        Returns:
            pyarrow.Array: List[str]类型字段，存储切分后的文本块

        """
        from bs4 import BeautifulSoup
        from llama_index.core import Document

        processed_chunks = []

        for content_index, raw_text in enumerate(texts):
            raw_text = raw_text.as_py()

            current_chunks = []
            try:
                if self.content_type == "text":
                    text_sequence = [raw_text] if isinstance(raw_text, str) else raw_text
                    documents = [Document(text=t) for t in text_sequence]

                elif self.content_type == "md":
                    cleaned_markdown = strip_markdown_images(raw_text)
                    cleaned_markdown = self._md_parser.remove_hyperlinks(cleaned_markdown)
                    cleaned_markdown = re.sub(r"\n{2,}", "\n\n", cleaned_markdown)
                    header_content_pairs = self._md_parser.markdown_to_tups(cleaned_markdown)
                    documents = []
                    for header, value in header_content_pairs:
                        if header is None:
                            documents.append(Document(text=value, metadata={}))
                        else:
                            documents.append(Document(text=f"\n\n{header}\n{value}", metadata={}))

                elif self.content_type == "html":
                    soup = BeautifulSoup(raw_text, "html.parser")
                    tags = soup.find_all("section")
                    documents = []
                    for tag in tags:
                        tag_id = tag.get("id")
                        tag_text = self._html_parser._extract_text_from_tag(tag)
                        tag_text = re.sub(r"\n{2,}", "\n\n", tag_text)
                        metadata = {"tag": "section", "tag_id": tag_id}
                        doc = Document(
                            text=tag_text,
                            metadata=metadata,
                        )
                        documents.append(doc)

                text_segments = self._sentence_splitter.get_nodes_from_documents(documents)
                for segment in text_segments:
                    segment_text = segment.text
                    if segment_text:
                        current_chunks.append(segment_text)

            except Exception:
                logger.exception("Error processing content at index %d", content_index)
                current_chunks = []

            finally:
                processed_chunks.append(current_chunks)

        return pa.array(processed_chunks, type=self.__return_column_type__())

    @staticmethod
    def __return_column_type__() -> pa.DataType:
        return pa.list_(pa.string())
