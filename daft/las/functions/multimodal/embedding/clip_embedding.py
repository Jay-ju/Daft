# Copyright (c) Beijing Volcano Engine Technology Ltd.

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

from daft.dependencies import pa
from daft.las.functions.types import Operator
from daft.las.functions.utils.common_utils import log_op_call
from daft.las.functions.utils.image_utils import decode_image

logger = logging.getLogger(__name__)


class ClipEmbedding(Operator):
    """**CLIP（Contrastive Language-Image Pretraining）跨模态嵌入生成器，实现基于CLIP模型的图文联合嵌入空间映射**

    **核心功能**

    - **多模态统一编码**
      - 文本编码：中文文本 → 512/768/1024维语义向量
      - 图像编码：图像 → 512/768/1024维视觉特征向量

    - **跨模态相似度计算**
      - 支持余弦相似度/内积计算图文嵌入向量的关联度

    **典型应用场景**
    - ✅ 电商场景 - 商品图文互搜
    - ✅ 内容审核 - 图文一致性校验
    - ✅ 推荐系统 - 多模态特征融合
    """  # noqa: D415

    def __init__(
        self,
        content_type: str,
        batch_size: int = 16,
        model_path: str = "/opt/las/models",
        model_name: str = "iic/multi-modal_clip-vit-base-patch16_zh",
        model_version: str = "v1.0.1",
        rank: int = 0,
        **kwargs: Any,
    ) -> None:
        """多模态向量模型参数初始化.

        Args:
            content_type: 输入图像的格式类型，支持：
                - 文本(text)
                - tos/http 地址(image_url)
                - base64 编码(image_base64)
                - 二进制流(image_binary)
                可选值：["text", "image_url", "image_base64", "image_binary"]
                默认值："image_url"
            model_path: 模型存储路径，默认：'/opt/las/models'（内部参数）
            model_name: 模型名称，可选：
                - 'iic/multi-modal_clip-vit-base-patch16_zh'
                - 'iic/multi-modal_clip-vit-huge-patch14_zh'
                - 'iic/multi-modal_clip-vit-large-patch14_zh'
                - 'iic/multi-modal_clip-vit-large-patch14_336_zh'
                默认：'iic/multi-modal_clip-vit-base-patch16_zh'
            model_version: 模型版本，当前仅支持'v1.0.1'
            batch_size: 批量计算数据量，默认：16
            rank: 指定GPU设备编号（多卡环境有效），默认：0（内部参数）

        Raises:
            ValueError: 当content_type不在允许值范围内时抛出
            RuntimeError: 模型加载失败时抛出
        """
        super().__init__(**kwargs)

        self.content_type = content_type
        if self.content_type not in ["text", "image_url", "image_base64", "image_binary"]:
            raise ValueError(f"Invalid content_type: {self.content_type}")

        self.model_path = model_path
        self.model_name = model_name
        self.model_version = model_version
        self.batch_size = batch_size
        self.rank = rank

        # These packages are heavy, so we import them lazily.
        import torch

        model_dir = Path(self.model_path) / self.model_name
        if not model_dir.exists():
            raise FileNotFoundError(f"Model directory not found: {model_dir}")

        use_gpu = self.use_gpu and torch.cuda.is_available()
        self.model_device: str = "cpu"
        if self.rank is None:
            self.model_device = "cuda" if use_gpu else "cpu"
        else:
            self.model_device = f"cuda:{self.rank % self.cuda_device_count}" if use_gpu else "cpu"
        logger.info("Model will be loaded on device: %s", self.model_device)

        from modelscope.pipelines import pipeline
        from modelscope.utils.constant import Tasks

        self._embedding_model = pipeline(
            task=Tasks.multi_modal_embedding,
            model=str(model_dir),
            model_revision=self.model_version,
            device=self.model_device,
        )

        logger.info(
            "Model initialization configuration:\n" "- Model Name: %s\n" "- Storage Path: %s\n" "- Device: %s\n",
            self.model_name,
            model_dir,
            self.model_device,
        )

        log_op_call(logger=logger, op=self.__class__.__name__, model_service_or_lib=self.model_name)

    def transform(self, content: pa.Array) -> pa.Array:
        """批量生成文本或图像的CLIP嵌入向量

        Args:
            content: 包含输入数据的数组，支持以下元素类型：
                - 文本模式: UTF-8字符串
                - 图像模式: Base64字符串/二进制数据/图像URL

        Returns:
            pa.Array: 包含浮点数嵌入向量的数组，每个元素为List[float]。

        Raises:
            ValueError: 当输入数据类型与content_type不匹配时
            RuntimeError: 模型推理失败或硬件资源不足时
        """  # noqa: D415
        start_time = time.monotonic()
        logger.info("Starting batch processing, input type: %s", self.content_type)

        total_embeddings = []
        total_content = len(content)
        total_batches = (total_content + self.batch_size - 1) // self.batch_size
        for batch_idx in range(0, total_content, self.batch_size):
            sub_content = content.slice(batch_idx, self.batch_size)
            current_batch = sub_content.to_pylist()
            logger.debug("Processing batch %.2f with %d contents", (batch_idx + 1) / total_batches, len(current_batch))

            if not current_batch:
                break

            try:
                if self.content_type == "text":
                    batch_embeddings = self._embedding_model.forward({"text": current_batch})["text_embedding"]
                else:
                    batch_images = [decode_image(img, self.content_type) for img in current_batch]
                    batch_embeddings = self._embedding_model.forward({"img": batch_images})["img_embedding"]

                batch_embeddings = [embedding.tolist() for embedding in batch_embeddings]
            except RuntimeError:
                logger.exception("Model inference failed (possibly OOM)!")
                logger.info("Current batch size: %d, consider reducing batch_size", self.batch_size)
                batch_embeddings = [None] * len(current_batch)
            except Exception:
                logger.exception("Inference error!")
                batch_embeddings = [None] * len(current_batch)

            total_embeddings.extend(batch_embeddings)

        processing_time = time.monotonic() - start_time

        logger.info(
            "Completed %d images | Total time: %.2fs | Throughput: %.2f content/s",
            total_content,
            processing_time,
            total_content / processing_time,
        )

        return pa.array(total_embeddings, type=self.__return_column_type__())

    @staticmethod
    def __return_column_type__() -> pa.DataType:
        return pa.list_(pa.float32())
