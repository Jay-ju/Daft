# Copyright (c) Beijing Volcano Engine Technology Ltd.

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch

from daft.dependencies import pa
from daft.las.functions.types import Operator
from daft.las.functions.utils.common_utils import run_on_local_path, tracking_usage

logger = logging.getLogger(__name__)


class MultilingualTextQualityScorer(Operator):
    """**多语言文本质量评分算子 - 基于E5模型的多语言文本质量评估**

    **核心功能**
    - **多语言支持**：使用multilingual-e5模型支持多种语言的文本质量评分
    - **深度学习评估**：基于Transformer架构的E5模型进行质量评估
    - **GPU加速**：支持GPU推理加速，提高处理效率
    - **批量处理**：支持批量处理文本，优化推理性能

    **评分标准**
    - 输出范围：0-1之间的浮点数
    - 分数越高表示文本质量越好
    - 一般来讲，分数超过0.5，则表示文本质量较好

    **支持语言**
    - 英文、中文、日文、韩文、法文、德文、西班牙文等多种语言
    - 基于multilingual-e5-small-aligned-quality模型
    """  # noqa: D415

    def __init__(
        self,
        model_path: str = "/opt/las/models",
        model_name: str = "multilingual-e5-small-aligned-quality",
        dtype: str = "float32",
        batch_size: int = 100,
        rank: int = 0,
        **kwargs: Any,
    ) -> None:
        """初始化多语言文本质量评分算子

        Args:
            model_path: 模型文件所在的基础路径
                默认值："/opt/las/models"
            model_name: 模型名称
                默认值："multilingual-e5-small-aligned-quality"
            dtype: 模型精度，支持 bfloat16、float16 和 float32
                默认值："float32"
            batch_size: 批处理大小
                描述：模型推理时的批处理大小
                默认值：100
            rank: GPU编号
                描述：指定使用的GPU设备编号
                默认值：0
        """  # noqa: D415
        super().__init__(**kwargs)

        if dtype not in ["bfloat16", "float16", "float32"]:
            raise ValueError(f"Unsupported dtype: {dtype}. Supported: bfloat16, float16, float32")

        self.model_path = model_path
        self.model_name = model_name
        self.dtype = dtype
        self.batch_size = batch_size
        self.rank = rank

        model_dir = str(Path(self.model_path) / self.model_name)

        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        use_gpu = self.use_gpu and torch.cuda.is_available()
        if self.rank is None:
            device = "cuda" if use_gpu else "cpu"
        else:
            device = f"cuda:{self.rank % self.cuda_device_count}" if use_gpu else "cpu"

        if use_gpu:
            device_props = torch.cuda.get_device_properties(device)
            logger.info(
                "GPU acceleration enabled | Device: %s | VRAM: %.1fGB | Compute Capability: %d.%d",
                device,
                device_props.total_memory / 1024**3,
                device_props.major,
                device_props.minor,
            )
        else:
            logger.info("Using CPU for inference")

        def load_model_components(local_path: str) -> tuple[Any, Any]:
            tokenizer = AutoTokenizer.from_pretrained(local_path)

            dtype_mapping = {
                "bfloat16": torch.bfloat16,
                "float16": torch.float16,
                "float32": torch.float32,
            }
            torch_dtype = dtype_mapping[self.dtype]

            model = AutoModelForSequenceClassification.from_pretrained(
                local_path,
                torch_dtype=torch_dtype,
                device_map=device,
                trust_remote_code=True,
            )
            return tokenizer, model

        try:
            self.tokenizer, self.model = run_on_local_path(model_dir, load_model_components)

            param_count = sum(p.numel() for p in self.model.parameters())
            logger.info(
                "Model loaded: %s | Device: %s | Parameters: %s | Model dtype: %s",
                model_dir,
                device.upper(),
                f"{param_count:,}",
                self.dtype,
            )
        except Exception as e:
            logger.error(
                "Model loading failed: %s | Model: %s | Device: %s",
                str(e),
                self.model_name,
                device,
            )
            raise RuntimeError(f"Failed to load model {self.model_name}") from e

        self.device = device

        tracking_usage(op=self.__class__.__name__, model_service_or_lib="multilingual-e5")

    def _predict_quality_scores(self, texts: list[str]) -> list[float]:
        if not texts:
            return []

        quality_scores = []
        total_batches = (len(texts) + self.batch_size - 1) // self.batch_size

        with torch.no_grad():
            for batch_idx in range(total_batches):
                batch_start = batch_idx * self.batch_size
                batch_end = (batch_idx + 1) * self.batch_size
                batch_texts = texts[batch_start:batch_end]

                progress = (batch_idx + 1) / total_batches * 100
                logger.info(
                    "Processing: %.1f%% | Batch: %d/%d | Samples: %d",
                    progress,
                    batch_idx + 1,
                    total_batches,
                    len(batch_texts),
                )

                if not batch_texts:
                    continue

                try:
                    inputs = self.tokenizer(
                        batch_texts,
                        return_tensors="pt",
                        truncation=True,
                        padding=True,
                        max_length=512,
                    ).to(self.device)

                    outputs = self.model(**inputs)
                    batch_scores = torch.sigmoid(outputs.logits.squeeze()).cpu().tolist()

                    if isinstance(batch_scores, float):
                        batch_scores = [batch_scores]

                    quality_scores.extend(batch_scores)
                    logger.debug("Batch %d scores: %s", batch_idx + 1, batch_scores)

                except ValueError as ve:
                    logger.error("Input processing error: %s", str(ve))
                    quality_scores.extend([None] * len(batch_texts))
                except RuntimeError as re:
                    logger.error("Inference failure: %s", str(re))
                    quality_scores.extend([None] * len(batch_texts))
                except Exception as e:
                    logger.error("Unexpected error: %s", str(e))
                    quality_scores.extend([None] * len(batch_texts))

        return quality_scores

    def transform(self, texts: pa.Array) -> pa.Array:
        """批量计算多语言文本质量分数

        Args:
            texts: 包含待处理文本的列，元素类型为字符串

        Returns:
            pyarrow.Array: 包含文本质量分数的列，元素类型为float32
        """  # noqa: D415
        logger.debug("Processing batch with %s texts", len(texts))
        input_texts: list[str | None] = []

        for text in texts:
            text_value = text.as_py()
            if text_value is None or (isinstance(text_value, str) and not text_value.strip()):
                input_texts.append(None)
            else:
                input_texts.append(text_value)

        valid_texts = [text for text in input_texts if text is not None]

        if not valid_texts:
            return pa.array([None] * len(input_texts), type=self.__return_column_type__())

        try:
            quality_scores = self._predict_quality_scores(valid_texts)

            result_scores: list[float | None] = []
            valid_idx = 0
            for text in input_texts:
                if text is None:
                    result_scores.append(None)
                else:
                    if valid_idx < len(quality_scores):
                        result_scores.append(quality_scores[valid_idx])
                    else:
                        result_scores.append(None)
                    valid_idx += 1

            logger.debug("Generated quality scores for %d texts", len(result_scores))
            return pa.array(result_scores, type=self.__return_column_type__())

        except Exception as e:
            logger.error("Quality scoring failed: %s", str(e))
            return pa.array([None] * len(input_texts), type=self.__return_column_type__())

    @staticmethod
    def __return_column_type__() -> pa.DataType:
        return pa.float32()
