# Copyright (c) Beijing Volcano Engine Technology Ltd.

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

from daft.dependencies import pa
from daft.las.functions.types import Operator
from daft.las.functions.utils.common_utils import tracking_usage

logger = logging.getLogger(__name__)

MODEL_PROMPT_MAPPING = {
    "BAAI/bge-m3": "",
    "BAAI/bge-large-zh-v1.5": "为这个句子生成表示以用于检索相关文章：",
    "BAAI/bge-large-en-v1.5": "Represent this sentence for searching relevant passages:",
    "BAAI/bge-multilingual-gemma2": "<instruct>Given a web search query, retrieve relevant "
    "passages that answer the query.\n<query>",
}


class BgeEmbedding(Operator):
    """**基于 BGE 系列的文本嵌入模型，支持稠密向量生成**

    **核心功能**
    - **多模型支持：**
        - 支持 BGE-M3、BGE-Large-zh-v1.5、BGE-Large-en-v1.5、BGE-Multilingual-Gemma2 等模型
        - 每个模型有不同的输入提示，用户可以根据需要选择不同的模型

    - **硬件加速**：支持 `FP16` 量化与 GPU 并行计算
    """  # noqa: D415

    def __init__(
        self,
        dtype: str = "float32",
        batch_size: int = 512,
        model_path: str = "/opt/las/models",
        model_name: str = "BAAI/bge-m3",
        rank: int | None = None,
        **kwargs: Any,
    ) -> None:
        """初始化BGE稠密嵌入模型.

        Args:
            dtype: 模型精度，支持 float32 和 float16
                可选值：["float32", "float16"]
                默认值："float32"
            batch_size: 模型推理时的批处理大小
                默认值：512
            model_path: 模型文件所在的路径
                默认值："/opt/las/models"
            model_name: 模型名称
                可选值：["BAAI/bge-m3", "BAAI/bge-large-zh-v1.5", "BAAI/bge-large-en-v1.5", "BAAI/bge-multilingual-gemma2"]
                默认值："BAAI/bge-m3"
            rank: GPU 编号
                默认值：None
        """
        super().__init__(**kwargs)
        self.dtype = dtype
        self.batch_size = batch_size
        self.model_path = model_path
        self.model_name = model_name
        self.rank = rank
        self.prompt = MODEL_PROMPT_MAPPING.get(self.model_name, "")

        import torch

        use_gpu = self.use_gpu and torch.cuda.is_available()

        model_dir = str(Path(self.model_path) / self.model_name)
        if self.rank is None:
            device = "cuda" if use_gpu else "cpu"
        else:
            device = f"cuda:{self.rank % self.cuda_device_count}" if use_gpu else "cpu"

        logger.info(
            "Initializing BGE model with:\n" "- model_name: %s\n" "- model_path: %s\n" "- dtype: %s\n" "- device: %s",
            self.model_name,
            self.model_path,
            self.dtype,
            device,
        )

        from sentence_transformers import SentenceTransformer

        logger.info("Loading sentence transformer model...")
        try:
            self._embedding_model = SentenceTransformer(model_dir, device=device, trust_remote_code=True)
            if self.dtype == "float16":
                self._embedding_model = self._embedding_model.to(dtype=torch.float16)

        except Exception as e:
            raise RuntimeError(f"Failed to load model from {model_dir}") from e

        tracking_usage(op=self.__class__.__name__, model_service_or_lib=self.model_name)

    def _generate_embeddings(self, texts: list[str]) -> list[Any]:
        logger.debug("Start model inference with batch_size=%s", self.batch_size)

        start_time = time.perf_counter()

        try:
            output = self._embedding_model.encode(
                [self.prompt + x for x in texts],
                normalize_embeddings=True,
                batch_size=self.batch_size,
            )
        finally:
            elapsed = time.perf_counter() - start_time
            logger.debug("Inference completed in %.2fs", elapsed)

        return output

    def transform(self, texts: pa.Array) -> pa.Array:
        """批量处理文本数组生成嵌入向量.

        该方法使用预加载的嵌入模型对输入的文本数组进行批量编码，生成对应的稠密嵌入向量。

        Args:
            texts: 包含待处理文本的数组，元素类型为str。
        Returns:
            处理后的数组，包含每个文本对应的稠密嵌入向量

        Raises:
            ValueError: 当输入数据格式不符合要求时抛出
        """
        logger.debug("Processing batch with %s texts", len(texts))
        text_cnt = len(texts)
        texts = texts.to_pylist()

        try:
            output = self._generate_embeddings(texts)
            batch_dense_embeddings = [embedding.tolist() for embedding in output]

            logger.info(
                "Generated embeddings:\n" "- Dense vectors: %d entries\n" "- Dense vector dimension: %d",
                len(batch_dense_embeddings),
                len(batch_dense_embeddings[0]),
            )

        except Exception:
            logger.exception(
                "Embedding generation failed!",
                extra={"batch_size": text_cnt},
            )
            batch_dense_embeddings = [None] * text_cnt
        return pa.array(batch_dense_embeddings, type=self.__return_column_type__())

    @staticmethod
    def __return_column_type__() -> pa.DataType:
        return pa.list_(pa.float32())
