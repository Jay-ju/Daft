# Copyright (c) Beijing Volcano Engine Technology Ltd.

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

from daft.dependencies import pa
from daft.las.functions.types import Operator

logger = logging.getLogger(__name__)


class BgeSparseDenseEmbedding(Operator):
    """**基于 BGE-M3 的文本嵌入模型，支持稀疏/稠密/token 三级向量生成**

    **核心功能**
    - **多粒度嵌入输出：**
        - 稀疏向量：词项权重表示，适合关键词检索
        - 稠密向量：1024维语义表示，适合语义相似度计算
        - Token向量：细粒度上下文表征
    - **硬件加速**：支持 `FP16` 量化与 GPU 并行计算
    """  # noqa: D415

    def __init__(
        self,
        is_output_token_vec: bool = False,
        dtype: str = "float32",
        batch_size: int = 512,
        model_path: str = "./models",
        model_name: str = "BAAI/bge-m3",
        rank: int | None = None,
        **kwargs: Any,
    ) -> None:
        """初始化BGE稀疏稠密嵌入模型.

        Args:
            is_output_token_vec: 是否输出 token 向量
                默认值：False
            dtype: 模型精度，支持 float32 和 float16
                可选值：["float32", "float16"]
                默认值："float32"
            batch_size: 模型推理时的批处理大小
                默认值：512
            model_path: 模型文件所在的路径
                默认值："./models"
            model_name: 模型名称
                可选值：["BAAI/bge-m3"]
                默认值："BAAI/bge-m3"
            rank: GPU 编号
                默认值：0
        """
        super().__init__(**kwargs)
        self.is_output_token_vec = is_output_token_vec
        self.dtype = dtype
        self.batch_size = batch_size
        self.model_path = model_path
        self.model_name = model_name
        self.rank = rank

        import torch

        use_gpu = self.use_gpu and torch.cuda.is_available()

        model_dir = str(Path(self.model_path) / self.model_name)
        use_fp16 = self.dtype == "float16"
        if self.rank is None:
            device = "cuda" if use_gpu else "cpu"
        else:
            device = f"cuda:{self.rank % self.cuda_device_count}" if use_gpu else "cpu"

        logger.info(
            "Initializing BGE model with:\n"
            "- model_name: %s\n"
            "- model_path: %s\n"
            "- dtype: %s (use_fp16: %s)\n"
            "- device: %s",
            self.model_name,
            self.model_path,
            self.dtype,
            use_fp16,
            device,
        )

        from FlagEmbedding import BGEM3FlagModel

        self.embedding_model = BGEM3FlagModel(model_dir, use_fp16=use_fp16, device=device)

    def _generate_embeddings(self, texts: list[str]) -> dict[str, Any]:
        logger.debug("Start model inference with batch_size=%s", self.batch_size)

        start_time = time.perf_counter()

        try:
            output = self.embedding_model.encode(
                texts,
                return_dense=True,
                return_sparse=True,
                return_colbert_vecs=self.is_output_token_vec,
                batch_size=self.batch_size,
            )
        finally:
            elapsed = time.perf_counter() - start_time
            logger.debug("Inference completed in %.2fs", elapsed)

        return output

    def transform(self, texts: pa.Array) -> pa.Array:
        """批量处理文本数组生成嵌入向量.

        该方法使用预加载的嵌入模型对输入的文本数组进行批量编码，生成对应的稠密/稀疏嵌入向量。

        Args:
            texts: 包含待处理文本的数组，元素类型为str。

        Returns:
            pyarrow.Array: 处理后的数组，包含以下字段：
                - dense_embedding: 稠密嵌入向量
                - sparse_embedding: 稀疏嵌入向量
                - token_embedding: 可选的token级嵌向量

        Raises:
            ValueError: 当输入数据格式不符合要求时抛出
        """
        logger.debug("Processing batch with %s texts", len(texts))
        text_cnt = len(texts)
        texts = texts.to_pylist()

        try:
            output = self._generate_embeddings(texts)
            batch_dense_embeddings = [embedding.tolist() for embedding in output["dense_vecs"]]

            sparse_vectors = self.embedding_model.convert_id_to_token(output["lexical_weights"])
            if isinstance(sparse_vectors, dict):
                sparse_vectors = [sparse_vectors]
            batch_sparse_embeddings = [{str(key): float(value) for key, value in vec.items()} for vec in sparse_vectors]

            logger.info(
                "Generated embeddings:\n"
                "- Dense vectors: %d entries\n"
                "- Sparse vectors: %d tokens\n"
                "- Dense vector dimension: %d",
                len(batch_dense_embeddings),
                sum(len(v) for v in batch_sparse_embeddings),
                len(batch_dense_embeddings[0]),
            )

            if self.is_output_token_vec:
                batch_token_embeddings = [embedding.tolist() for embedding in output["colbert_vecs"]]
                logger.info("Token embeddings generated: %s sequences", len(batch_token_embeddings))

        except Exception:
            logger.exception(
                "Embedding generation failed!",
                extra={"batch_size": text_cnt},
            )
            batch_dense_embeddings = [None] * text_cnt
            batch_sparse_embeddings = [{} for _ in range(text_cnt)]
            batch_token_embeddings = [None] * text_cnt

        results = []
        for i in range(text_cnt):
            record = {
                "dense_embedding": batch_dense_embeddings[i],
                "sparse_embedding": batch_sparse_embeddings[i],
            }
            if self.is_output_token_vec:
                record["token_embedding"] = batch_token_embeddings[i]
            else:
                record["token_embedding"] = None
            results.append(record)

        return pa.array(results, type=self.__return_column_type__())

    @staticmethod
    def __return_column_type__() -> pa.DataType:
        fields = [
            pa.field("dense_embedding", pa.list_(pa.float32())),
            pa.field("sparse_embedding", pa.map_(pa.string(), pa.float32())),
            pa.field("token_embedding", pa.list_(pa.list_(pa.float32()))),
        ]
        return pa.struct(fields)
