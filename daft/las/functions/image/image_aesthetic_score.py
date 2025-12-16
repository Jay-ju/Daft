# Copyright (c) Beijing Volcano Engine Technology Ltd.

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from transformers import CLIPModel, CLIPProcessor

from daft.dependencies import np, pa
from daft.las.functions.types import Operator
from daft.las.functions.utils.common_utils import FastWriteCounter, get_logger, tracking_usage
from daft.las.functions.utils.image_utils import decode_image_pil


class MLP(nn.Module):  # type: ignore[misc]
    """A simple multilayer perceptron model for aesthetic scoring."""

    def __init__(self, input_dim: int = 768) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.Dropout(0.2),
            nn.Linear(1024, 128),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.Dropout(0.1),
            nn.Linear(64, 16),
            nn.Linear(16, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.layers(x)
        out = out / 10.0
        return torch.clamp(out, 0.0, 1.0)


class ImageAestheticScore(Operator):
    """**图像美学评分处理器，智能评估图像的审美质量和构图效果**

    **核心功能：**
    - 美学评分：对输入图像进行专业美学质量评估
    - 构图分析：基于视觉感知理论分析图像构图效果
    - 标准化输出：评分范围归一化至0-1，便于后续处理
    - 批量处理：支持高效的批量图像评分
    - 多格式支持：兼容多种图像输入格式

    **格式支持：**
    - 输入：支持图像URL、TOS地址、二进制流等多种格式
    - 输出：浮点数美学评分(0.0-1.0)，数值越高表示美学质量越好
    - 图像格式：JPG、PNG、WebP等主流图像格式
    - 分辨率：自动适配不同分辨率的图像输入

    **评分原理：**
    - 视觉特征提取：使用先进的视觉模型提取图像特征
    - 美学建模：基于大规模美学数据集训练的评分模型
    - 多维度评估：综合考虑色彩、构图、对比度等多个美学要素
    - 感知对齐：评分结果与人类美学感知高度一致
    """  # noqa: D415

    def __init__(
        self,
        batch_size: int = 32,
        model_path: str = "/data00/tiger/las/models",
        clip_model_name: str = "openai/clip-vit-large-patch14",
        mlp_model_name: str = "laion_aesthetic_v2/sac+logos+ava1-l14-linearMSE.pth",
        device: str = "cpu",  # "cpu" / "cuda" / "cuda:0"
        **kwargs: Any,
    ) -> None:
        """初始化图像美学评分算子

        Args:
            batch_size: 批处理大小，控制单次推理处理的图像数量
                默认值：32
            model_path: 模型文件存储路径
                默认值：'/data00/tiger/las/models'
            clip_model_name: CLIP视觉模型名称
                默认值：'openai/clip-vit-large-patch14'
            mlp_model_name: MLP评分模型名称
                默认值：'laion_aesthetic_v2/sac+logos+ava1-l14-linearMSE.pth'
            device: 设备类型，支持CPU和GPU设备
                默认值："cpu"
                可选值："cpu", "cuda", "cuda:0", "cuda:1"等
        """  # noqa: D415
        super().__init__(**kwargs)
        self.batch_size = batch_size
        self.model_path = model_path
        self.clip_model_name = clip_model_name
        self.mlp_model_name = mlp_model_name
        self.device_param = device

        self.submit_counter = FastWriteCounter()
        self.success_counter = FastWriteCounter()
        self.failed_counter = FastWriteCounter()

        self.logger = get_logger(f"ImageAestheticScore-{id(self)}")

        self.logger.info(
            "ImageAestheticScore initialized with batch_size=%s, "
            "model_path=%s, clip_model_name=%s, mlp_model_name=%s",
            batch_size,
            model_path,
            clip_model_name,
            mlp_model_name,
        )

        # Initialize models
        self._initialize_models()

        tracking_usage(op=self.__class__.__name__, model_service_or_lib="clip_aesthetic")

    def _initialize_models(self) -> None:
        """Initialize CLIP and MLP models."""
        try:
            # Load CLIP model and processor
            clip_model_path = str(Path(self.model_path) / self.clip_model_name)
            self.clip_model = CLIPModel.from_pretrained(clip_model_path)
            self.clip_processor = CLIPProcessor.from_pretrained(clip_model_path)

            # Set up device
            if self.device_param.startswith("cuda") and torch.cuda.is_available():
                self.device = self.device_param
                self.clip_model = self.clip_model.to(self.device)
                self.clip_model = self.clip_model.half()  # Use FP16 for efficiency
                self.logger.info("Using GPU device: %s", self.device)
            else:
                self.device = "cpu"
                self.logger.info("Using CPU device")

            self.clip_model.eval()

            # Load MLP model
            mlp_model_path = str(Path(self.model_path) / self.mlp_model_name)
            self.mlp_model = MLP()
            self._load_mlp_model(mlp_model_path)

            self.logger.info("Models initialized successfully")

        except Exception as e:
            self.logger.error("Failed to initialize models: %s", e)
            raise

    def _load_mlp_model(self, model_path: str) -> None:
        """Load the trained MLP weights from a file."""
        model_path_obj = Path(model_path)
        if not model_path_obj.is_file():
            raise FileNotFoundError(f"MLP model checkpoint not found: {model_path_obj}")

        state_dict = torch.load(model_path, map_location=self.device)
        self.mlp_model.load_state_dict(state_dict)
        self.mlp_model.to(self.device)
        self.mlp_model.eval()
        self.logger.info("MLP model loaded from: %s", model_path)

    def _normalize(self, arr: np.ndarray, axis: int = -1, order: int = 2) -> np.ndarray:
        """Normalize array along specified axis."""
        norm = np.atleast_1d(np.linalg.norm(arr, order, axis))
        norm[norm == 0] = 1
        return arr / np.expand_dims(norm, axis)

    def log_progress(self) -> None:
        submitted = self.submit_counter.value
        succeed = self.success_counter.value
        failed = self.failed_counter.value
        finished = succeed + failed
        running = submitted - finished
        self.logger.info(
            "%s/%s running, finished/succeed/failed: %s/%s/%s", running, submitted, finished, succeed, failed
        )

    def _process_batch(self, image_inputs: list[Any]) -> list[float | None]:
        """Process a batch of images and return aesthetic scores or None for failed cases."""
        images: list[Any | None] = []
        for img_input in image_inputs:
            try:
                image = decode_image_pil(img_input, mode="RGB")
                images.append(image)
            except Exception as e:
                self.logger.warning("Failed to decode image: %s", e)
                images.append(None)
                self.failed_counter.increment()

        valid_indices = [i for i, img in enumerate(images) if img is not None]
        valid_images = [images[i] for i in valid_indices]

        if not valid_images:
            return [None] * len(image_inputs)

        try:
            inputs = self.clip_processor(images=valid_images, return_tensors="pt", padding=True)
            if self.device != "cpu":
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                image_features = self.clip_model.get_image_features(**inputs)

            embs = self._normalize(image_features.cpu().numpy())

            with torch.no_grad():
                scores = self.mlp_model(torch.from_numpy(embs).to(self.device).float())
                batch_scores = scores.squeeze().cpu().tolist()

            if not isinstance(batch_scores, list):
                batch_scores = [batch_scores]

            final_scores: list[float | None] = [None] * len(image_inputs)
            for idx, score in zip(valid_indices, batch_scores):
                final_scores[idx] = float(score)

            for _ in range(len(valid_images)):
                self.success_counter.increment()

            return final_scores

        except Exception as e:
            self.logger.error("Inference failed for batch: %s", e)
            for _ in range(len(valid_images)):
                self.failed_counter.increment()
            return [None] * len(image_inputs)

    def transform(self, image_inputs: pa.Array) -> pa.Array:
        """批量计算图像美学评分，对每张图像进行审美质量评估。

        Args:
            image_inputs: 包含输入图像的数组

        Returns:
            pyarrow.Array: 包含美学评分的浮点数组，评分范围0.0-1.0，
                无法解码或处理失败的图像返回null值
        """  # noqa: D415
        input_list = image_inputs.to_pylist()
        results: list[float | None] = []

        for i in range(0, len(input_list), self.batch_size):
            batch = input_list[i : i + self.batch_size]

            for _ in range(len(batch)):
                self.submit_counter.increment()

            batch_results = self._process_batch(batch)
            results.extend(batch_results)

            self.log_progress()

        return pa.array(results, type=self.__return_column_type__())

    @staticmethod
    def __return_column_type__() -> pa.DataType:
        return pa.float64()
