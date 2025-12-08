# Copyright (c) Beijing Volcano Engine Technology Ltd.

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np  # noqa: TID253
import torch

from daft.dependencies import pa
from daft.las.functions.types import Operator
from daft.las.functions.utils.common_utils import tracking_usage
from daft.las.functions.utils.image_utils import decode_image

if TYPE_CHECKING:
    from PIL import Image


logger = logging.getLogger(__name__)


class ImageNsfwDetect(Operator):
    """**图像安全性(NSFW)检测器，支持多源输入与批量推理。**

    **核心功能**
    - 使用预训练的图像分类模型进行 NSFW 概率检测，输出每张图片的 NSFW 置信度分数。
    - 支持多种输入来源：
        - URL 地址（image_url）
        - Base64 编码（image_base64）
        - 二进制流（image_binary）
    - 批量处理：
        - 通过配置 batch_size 进行批量推理以提升吞吐性能。

    **适用场景**
    - 内容安全审核：电商、社交平台的图片审核流程。
    - 生产管线中的预过滤：在图像处理/生成任务之前进行安全性筛查。

    **注意事项**
    - 仅输出数值型 NSFW 置信度（0~1），不保存图像；如需持久化请在上游/下游管线中实现。
    - 请确保模型路径(model_path)与模型名称(model_name)在本地可用，或已正确配置权重下载策略。
    """  # noqa: D415

    def __init__(
        self,
        image_src_type: str = "image_url",
        model_path: str = "/opt/las/models",
        model_name: str = "Falconsai/nsfw_image_detection",
        dtype: str = "float16",
        batch_size: int = 16,
        rank: int = 0,
        **kwargs: Any,
    ) -> None:
        """图像安全性检测器初始化方法.

        Args:
            image_src_type: 输入图像的格式类型，支持：
                - url地址(image_url)
                - base64编码(image_base64)
                - 二进制流(image_binary)
                描述: 输入图像的格式类型。
                可选值: ["image_url", "image_base64", "image_binary"]
                默认值: "image_url"
            model_path: 模型基础路径。
                描述: 预训练模型在本地的根目录路径。
                默认值: "/home/ray/workdir/models"
            model_name: 模型名称/子目录。
                描述: 在 model_path 下的具体模型目录名称。
                可选值: ["Falconsai/nsfw_image_detection"]
                默认值: "Falconsai/nsfw_image_detection"
            dtype: 模型推理精度选择：
                - float16: 更快的推理速度
                - float32: 最高精度但显存消耗最大
                可选值：["float16", "float32"]
                默认值："float16"
            batch_size: 批量推理的图片数量。
                描述: 每次送入模型进行推理的图片数量，批量越大吞吐越高，但显存占用也更高。
                默认值: 16
            rank: GPU 编号。
                描述: 推理所使用的 GPU 编号，CPU 推理时可保持为 0。
                默认值: 0
        """
        super().__init__(**kwargs)
        self.image_src_type = image_src_type
        self.batch_size = batch_size
        self.model_path = model_path
        self.model_name = model_name
        self.rank = rank

        import torch
        from transformers import AutoModelForImageClassification, ViTImageProcessor

        dtype_mapping = {
            "float16": torch.float16,
            "float32": torch.float32,
        }
        self.torch_dtype = dtype_mapping.get(dtype)
        if not self.torch_dtype:
            raise ValueError(f"Unsupported precision type: {dtype}")

        if self.image_src_type not in ["image_url", "image_base64", "image_binary"]:
            logger.error("Unsupported image source type: %s", self.image_src_type)
            raise ValueError(f"Invalid image source type: {self.image_src_type}")

        # Device initialization
        use_gpu = self.use_gpu and torch.cuda.is_available()
        if self.rank is None:
            self.device: str = "cuda" if use_gpu else "cpu"
        else:
            self.device = f"cuda:{self.rank % self.cuda_device_count}" if use_gpu else "cpu"
        logger.info("Model will be loaded on device: %s", self.device)

        model_dir = str(Path(self.model_path) / self.model_name)
        self.model = AutoModelForImageClassification.from_pretrained(
            model_dir,
            torch_dtype=self.torch_dtype,
            device_map=self.device,
            trust_remote_code=True,
        )
        self.processor = ViTImageProcessor.from_pretrained(model_dir)

        logger.info("Model path is %s", self.model_path)
        logger.info("Model name is %s", self.model_name)
        logger.info("Image source type is %s", self.image_src_type)

        tracking_usage(op=self.__class__.__name__, model_service_or_lib="Falconsai/nsfw_image_detection")

    def _prepare_pil_images(self, batch_raw: list[Any]) -> list[Image.Image]:
        """将原始输入转为 PIL.Image 列表，并统一为 RGB."""
        pil_images: list[Image.Image] = []
        for data in batch_raw:
            try:
                img = decode_image(data, self.image_src_type)
                pil_images.append(img.convert("RGB"))
            except Exception:
                logger.exception("Failed to decode image from %s", self.image_src_type)
                pil_images.append(None)
        return pil_images

    def transform(self, images: pa.Array) -> pa.Array:
        """批量执行图像 NSFW 检测，支持多种输入格式与批处理配置.

        Args:
            images: 包含输入图像的数组，支持 URL/base64/数组 格式。

        Returns:
            包含检测结果的数组，每个元素为NSFW 置信度分数(float)，若检测失败则为 None。
        """
        raw_images = [image.as_py() for image in images]
        images_cnt = len(raw_images)
        scores_results: list[float | None] = [None] * images_cnt

        # 批次推理，保持与原实现一致的批量行为
        with torch.no_grad():
            for start in range(0, images_cnt, self.batch_size):
                end = min(images_cnt, start + self.batch_size)
                batch_raw = raw_images[start:end]
                try:
                    pil_image_list = self._prepare_pil_images(batch_raw)
                    # 若全部失败，跳过该批次
                    if not any(img is not None for img in pil_image_list):
                        raise ValueError("Empty valid images in batch")
                    # 过滤 None，保留索引用于回填得分
                    valid_pairs = [(idx, img) for idx, img in enumerate(pil_image_list) if img is not None]
                    valid_indices = [idx for idx, _ in valid_pairs]
                    valid_images = [img for _, img in valid_pairs]

                    inputs = self.processor(images=valid_images, return_tensors="pt").to(self.model.device)
                    outputs = self.model(**inputs)
                    logits = outputs.logits
                    nsfw_probs_batch = torch.softmax(logits, dim=-1)
                    nsfw_scores_batch = [float(scores[1]) for scores in nsfw_probs_batch]
                    nsfw_scores_batch = np.round(np.array(nsfw_scores_batch), decimals=6).tolist()

                    # 回填到对应位置
                    for local_i, score in zip(valid_indices, nsfw_scores_batch):
                        global_i = start + local_i
                        scores_results[global_i] = score
                except Exception:
                    logger.exception("Image NSFW detection failed [batch: %d-%d]", start, end - 1)
                    # 失败批次的项保持 None
                    continue

        logger.info("Batch NSFW detection completed. Total processed: %d images", images_cnt)
        return pa.array(scores_results, type=self.__return_column_type__())

    @staticmethod
    def __return_column_type__() -> pa.DataType:
        return pa.float64()
