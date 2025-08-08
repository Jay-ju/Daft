# Copyright (c) Beijing Volcano Engine Technology Ltd.

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

import fasttext

from daft.dependencies import pa
from daft.las.functions.types import Operator
from daft.las.functions.utils.common_utils import tracking_usage

logger = logging.getLogger(__name__)


class EnTextQualityScorer(Operator):
    """**英文文本质量评分算子 - 基于FastText的文本质量评估**

    **核心功能**
    - **质量评分**：使用FastText模型对英文文本质量进行评分，偏好于科学知识，只支持CPU环境。
    - **批量处理**：支持批量处理文本，提高处理效率

    **评分标准**
    - 0: 低质量 (Low)
    - 1: 中等质量 (Mid)
    - 2: 高质量 (High)
    - 最终得分为0-2之间的浮点数，分数越高表示质量越好
    - 一般来讲，分数超过0.5，则表示文本质量较好。
    """  # noqa: D415

    def __init__(
        self,
        model_path: str = "/opt/las/models",
        model_name: str = "llm-data-textbook-quality-fasttext-classifier-v2/model_quantized.bin",
        **kwargs: Any,
    ) -> None:
        """初始化英文文本质量评分算子

        Args:
            model_path: 模型文件所在的基础路径
                默认值："/opt/las/models"
            model_name: 模型文件名
                默认值："llm-data-textbook-quality-fasttext-classifier-v2/model_quantized.bin"
        """  # noqa: D415
        super().__init__(**kwargs)
        self.model_path = model_path
        self.model_name = model_name

        self._score_dict = {
            "__label__": 0,
            "__label__Low": 0,
            "__label__Mid": 1,
            "__label__High": 2,
        }

        model_dir = str(Path(self.model_path) / self.model_name)

        try:
            self.model = fasttext.load_model(model_dir)
            logger.info(
                "Text quality model loaded successfully: %s",
                model_dir,
            )
        except Exception as e:
            logger.error("Failed to load text quality model: %s", e)
            raise RuntimeError(f"Failed to load model {model_dir}") from e

        tracking_usage(op=self.__class__.__name__, model_service_or_lib="fasttext")

    def _predict_quality_scores(self, text_list: list[str]) -> list[float | None]:
        cleaned_texts = [re.sub(r"\n+", " ", text) if text is not None else None for text in text_list]
        valid_texts = [text for text in cleaned_texts if text is not None]

        if not valid_texts:
            return [None] * len(text_list)

        pred = self.model.predict(valid_texts, k=-1)
        pred_iter = iter(zip(pred[0], pred[1]))

        score_list: list[float | None] = []
        for text in cleaned_texts:
            if text is None:
                score_list.append(None)
            else:
                labels, scores = next(pred_iter)
                score = 0.0
                for label, label_score in zip(labels, scores):
                    score += self._score_dict.get(label, 0) * label_score
                score_list.append(score)

        return score_list

    def transform(self, texts: pa.Array) -> pa.Array:
        """批量计算英文文本质量分数

        Args:
            texts: 包含待处理文本的列，元素类型为字符串。

        Returns:
            pyarrow.Array: 包含文本质量分数的列，元素类型为float64。
        """  # noqa: D415
        logger.debug("Processing batch with %s texts", len(texts))
        input_texts = texts.to_pylist()

        try:
            quality_scores = self._predict_quality_scores(input_texts)
            logger.debug("Generated quality scores for %d texts", len(quality_scores))
        except Exception as e:
            logger.exception("Quality scoring failed: %s", e)
            quality_scores = [None] * len(input_texts)

        return pa.array(quality_scores, type=self.__return_column_type__())

    @staticmethod
    def __return_column_type__() -> pa.DataType:
        return pa.float64()
