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


class ImageViTEmbedding(Operator):
    """**ViT 图像语义嵌入处理器，适用于图像相似性搜索、内容检索等场景。**

    **核心功能**
    - 多模型支持：
        - Google 官方 ViT 模型
        - Meta DINOv2 视觉模型
    - 特征提取模式：
        - CLS Token 嵌入向量
        - 全局平均池化
    - 输入格式兼容：
        - URL
        - Base64 编码
        - 二进制流
    - 性能优化：
        - FP16 推理加速
        - 多 GPU 并行计算

    **容错机制**
    - 提取失败时返回全零向量

    **推荐模型**
    - `google/vit-base-patch16-224-in21k`（768维）
    - `google/vit-large-patch16-224-in21k`（1024维）
    - `facebook/dinov2-base`（768维）
    - `facebook/dinov2-large`（1024维）
    """  # noqa: D415

    def __init__(
        self,
        image_src_type: str = "image_url",
        dtype: str = "float32",
        batch_size: int = 32,
        model_path: str = "/opt/las/models",
        model_name: str = "facebook/dinov2-large",
        use_cls_token_embedding: bool = True,
        rank: int = 0,
        **kwargs: Any,
    ) -> None:
        """ViT图像语义嵌入处理器初始化方法.

        Args:
            image_src_type: 输入图像的格式类型，支持：
                - tos/http 地址(image_url)
                - base64 编码(image_base64)
                - 二进制流(image_binary)
                可选值：["image_url", "image_base64", "image_binary"]
                默认值："image_url"
            dtype: 模型推理精度选择：
                - bfloat16: 平衡精度与速度（TPU上更快）
                - float16: 更快的推理速度
                - float32: 最高精度但显存消耗最大
                可选值：["bfloat16", "float16", "float32"]
                默认值："float16"
            batch_size: 批处理大小
                默认值: 32
            model_path: 模型文件存储路径
                默认值: "/opt/las/models"
            model_name: 使用的图像向量模型名称
                可选值: [
                    "google/vit-base-patch16-224-in21k",
                    "google/vit-large-patch16-224-in21k",
                    "facebook/dinov2-base",
                    "facebook/dinov2-large"
                ]
                默认值: "facebook/dinov2-large"
            use_cls_token_embedding: 是否使用CLS Token特征
                默认值: True
            rank: 指定使用的GPU设备编号（多卡环境有效）。例如：0表示第一张GPU，1表示第二张GPU
                默认值：0
        """
        super().__init__(**kwargs)
        self.image_src_type = image_src_type
        self.dtype = dtype
        self.batch_size = batch_size
        self.model_path = model_path
        self.model_name = model_name
        self.use_cls_token_embedding = use_cls_token_embedding
        self.rank = rank

        # These packages are heavy, so we import them lazily.
        import torch
        from transformers import AutoImageProcessor, AutoModel

        if self.image_src_type not in ["image_url", "image_binary", "image_base64"]:
            logger.error("Invalid image source type: %s", self.image_src_type)
            raise ValueError(f"Unsupported image source type: {self.image_src_type}")

        dtype_mapping = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        self.torch_dtype = dtype_mapping.get(self.dtype)

        model_dir = str(Path(self.model_path) / self.model_name)

        try:
            logger.info("Loading ViT model: name= %s, path=%s", self.model_name, model_dir)
            self._preprocessor = AutoImageProcessor.from_pretrained(model_dir, use_fast=True)
            self._model = AutoModel.from_pretrained(model_dir)
            # logger.debug("Model config: %s", {self._model.config})
        except Exception:
            logger.exception("Model loading failed!")
            raise

        if self.torch_dtype in [torch.float16, torch.bfloat16]:
            logger.info("Converting model to %s precision", self.dtype)
            self._model.to(dtype=self.torch_dtype)

        if self.use_gpu:
            cuda_devices = self.cuda_device_count
            rank = self.rank % cuda_devices if self.rank is not None else 0
            logger.info(
                "Moving model to GPU: %s, cuda: %s (Available devices: %s)", self.model_name, rank, cuda_devices
            )
            self._model.to(f"cuda:{rank}")

        log_op_call(logger=logger, op=self.__class__.__name__, model_service_or_lib=self.model_name)

    def _generate_embedding(self, current_batch: list[Any]) -> list[str | None]:
        import torch

        with torch.no_grad():
            try:
                batch_images = [decode_image(img, self.image_src_type) for img in current_batch]
                logger.info("Image conversion successful for %d images.", len(batch_images))

                logger.debug("Performing image preprocessing")
                preprocess_start = time.time()
                inputs = self._preprocessor(images=batch_images, return_tensors="pt").to(self._model.device)

                if self.torch_dtype in [torch.float16, torch.bfloat16]:
                    logger.info("Converting input to %s precision", self.dtype)
                    inputs = inputs.to(dtype=self.torch_dtype)

                logger.debug("Input tensor shape: %s", inputs.pixel_values.shape)
                preprocess_time = time.time() - preprocess_start

                inference_start = time.time()
                outputs = self._model(**inputs)
                last_hidden_state = outputs.last_hidden_state
                inference_time = time.time() - inference_start

                logger.info(
                    "Model inference completed | " "Preprocess: %.2fs | " "Inference: %.2fs",
                    preprocess_time,
                    inference_time,
                )

                if self.use_cls_token_embedding:
                    features = last_hidden_state[:, 0, :]
                else:
                    features = torch.mean(last_hidden_state, dim=1)

                l2_norm = torch.norm(features, p=2, dim=1, keepdim=True)
                normalized_embeddings = features / l2_norm
                normalized_embeddings = [emb.tolist() for emb in normalized_embeddings]

            except Exception:
                logger.exception("Feature processing error!")
                normalized_embeddings = [None] * len(current_batch)

        return normalized_embeddings

    def transform(self, images: pa.Array) -> pa.Array:
        """批量处理图像数据并生成特征向量嵌入.

        Args:
            images: 包含图像数据的数组，元素类型可为图像URL、
                Base64编码或二进制数据

        Returns:
            pyarrow.Array: 包含特征向量的数组，每个元素为float类型的嵌套数组，
                数组维度由模型输出决定
        """
        all_embeddings = []
        total_images = len(images)
        logger.info("Starting batch processing of %d images", total_images)

        total_batches = (total_images + self.batch_size - 1) // self.batch_size
        for batch_idx in range(0, total_images, self.batch_size):
            sub_images = images.slice(batch_idx, self.batch_size)
            current_batch = sub_images.to_pylist()
            logger.debug("Processing batch %.2f with %d contents", (batch_idx + 1) / total_batches, len(current_batch))

            if not current_batch:
                break
            logger.debug("Processing batch %d", batch_idx + 1)
            batch_embedding = self._generate_embedding(current_batch)
            all_embeddings.extend(batch_embedding)

        logger.info(
            "Batch processing completed | " "Total embeddings: %d | " "Embedding dimension: %d",
            len(all_embeddings),
            len(all_embeddings[0]) if all_embeddings[0] else 0,
        )

        return pa.array(all_embeddings, type=self.__return_column_type__())

    @staticmethod
    def __return_column_type__() -> pa.DataType:
        return pa.list_(pa.float32())
