# Copyright (c) Beijing Volcano Engine Technology Ltd.

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from daft.dependencies import pa
from daft.las.functions.types import Operator
from daft.las.functions.utils.common_utils import tracking_usage

logger = logging.getLogger(__name__)


class ChunkTextSentenceSimilarity(Operator):
    """**多格式文本分块处理器 - 结构化解析与智能切分解决方案**

    **核心功能**

    - **支持格式**
      - `纯文本`：基于段落/标点的语义分块
      - `Markdown`：保留文档结构，过滤图片链接
      - `HTML`：提取正文内容，保留章节结构
    - **智能分块策略**
      - 结合语义与语法规则
      - 重叠优化保持上下文连贯性

    **技术实现**
    - **解析引擎**
      - HTML：`BeautifulSoup`
      - Markdown：`llama-index`
    - **分块算法**
      - 递归式语义分割（sentence splitter）
    """  # noqa: D415

    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        model_path: str = "/opt/las/models",
        embedding_model_name: str = "BAAI/bge-m3",
        breakpoint_percentile_threshold: int = 80,
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

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        from llama_index.core.node_parser import SemanticSplitterNodeParser, SentenceSplitter
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding

        model_dir = Path(model_path) / embedding_model_name
        self.embedding_model = HuggingFaceEmbedding(model_name=str(model_dir))

        self._sentence_splitter = SentenceSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separator=" ",
            paragraph_separator="\n\n",
            secondary_chunking_regex="[^,，.;；。？?！!]+[,，.;；。？?！!]?",
        )

        self.semantic_splitter = SemanticSplitterNodeParser(
            buffer_size=1,
            embed_model=self.embedding_model,
            sentence_splitter=lambda text: self._sentence_splitter.split_text(text),
            include_metadata=False,
            include_prev_next_rel=True,
            breakpoint_percentile_threshold=breakpoint_percentile_threshold,
        )

        logger.info("The chunk size is %d, the chunk overlap is %d", self.chunk_size, self.chunk_overlap)

        tracking_usage(op=self.__class__.__name__, model_service_or_lib="LlamaIndex")

    def transform(self, texts: pa.Array) -> pa.Array:
        """批量处理文本数组生成分句chunks.

        Args:
            texts: 待处理的文本数组，要求元素类型为字符串。

        Returns:
            pyarrow.Array: 切分后的文本块，元素为List[str]类型。

        """
        from llama_index.core import Document

        processed_chunks = []

        for content_index, raw_text in enumerate(texts):
            raw_text = raw_text.as_py()

            current_chunks = []
            try:
                text_list = [raw_text]
                documents = [Document(text=t) for t in text_list]
                semantic_split_nodes = self.semantic_splitter.get_nodes_from_documents(documents)

                for node in semantic_split_nodes:
                    node_text = node.text
                    if len(node_text) == 0:
                        continue
                    current_chunks.append(node_text)

            except Exception:
                logger.exception("Error processing content at index %d", content_index)
                current_chunks = []

            finally:
                processed_chunks.append(current_chunks)

        return pa.array(processed_chunks, type=self.__return_column_type__())

    @staticmethod
    def __return_column_type__() -> pa.DataType:
        return pa.list_(pa.string())
