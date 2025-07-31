# Copyright (c) Beijing Volcano Engine Technology Ltd.

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from daft.dependencies import pa
from daft.las.functions.types import Operator
from daft.las.functions.utils.common_utils import log_op_call

logger = logging.getLogger(__name__)


class AlphanumericRatioCalculator(Operator):
    """**字符占比计算器 - 基于字母和数字字符占比的文本特征提取**

    **核心功能**
    - 字符占比计算：精确统计字母和数字字符在文本中的占比
    - 分词模式支持：可选择基于分词或字符级别的占比计算
    - 多语言支持：支持英文、中文、日文、韩文等多种语言的字符识别

    **应用场景**
    - 多语言文本质量评估
    - 数据清洗和预处理
    - 文本分类特征提取
    - 内容安全检测
    - 多语言文本分析

    **技术特性**
    - 支持两种计算模式：
        - 字符模式：字母和数字字符占总字符数的比例
        - 分词模式：字母字符占分词总数的比例
    - 支持多语言Unicode字符识别，包括但不限于：
        - 英文字母 (a-z, A-Z)
        - 中文字符 (汉字)
        - 日文字符 (平假名、片假名、汉字)
        - 韩文字符 (谚文)
        - 数字字符 (0-9)
    """  # noqa: D415

    def __init__(
        self,
        tokenization: bool = False,
        model_path: str = "/opt/las/models",
        model_name: str = "pythia-6.9b-deduped",
        **kwargs: Any,
    ) -> None:
        """字符占比计算器初始化方法

        Args:
            tokenization: 是否分词
                描述：是否使用分词模式计算占比
                默认值：False
            model_path: 模型文件所在的路径
                默认值："/opt/las/models"
            model_name: 模型名称
                默认值："pythia-6.9b-deduped"
        """  # noqa: D415
        super().__init__(**kwargs)

        self.tokenization = tokenization
        self.model_path = model_path
        self.model_name = model_name

        if self.tokenization:
            try:
                from transformers import AutoTokenizer

                model_dir = str(Path(self.model_path) / self.model_name)
                self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
                logger.info("Tokenizer loaded successfully for tokenization mode from %s", model_dir)
            except Exception as e:
                logger.error("Failed to load tokenizer from %s: %s, falling back to character mode", model_dir, str(e))
                self.tokenization = False
                self.tokenizer = None
        else:
            self.tokenizer = None

        logger.info("Alphanumeric ratio calculator initialized with tokenization=%s", self.tokenization)

        log_op_call(logger=logger, op=self.__class__.__name__, model_service_or_lib=self.model_name)

    def _calculate_ratio(self, text: str) -> float:
        if self.tokenization and self.tokenizer:
            alpha_count = sum(1 if char.isalpha() else 0 for char in text)
            tokens = self.tokenizer.tokenize(text)
            token_count = len(tokens) if tokens else 1
            return alpha_count / token_count
        else:
            alnum_count = sum(1 if char.isalnum() else 0 for char in text)
            return alnum_count / len(text)

    def transform(self, texts: pa.Array) -> pa.Array:
        """批量处理文本列，计算字符占比

        Args:
            texts: 待处理的文本列，要求元素类型为字符串

        Returns:
            pa.Array: 占比结果列，元素为浮点数，表示字母数字字符的占比
        """  # noqa: D415
        results: list[float | None] = []

        for text in texts:
            text_value = text.as_py()
            if text_value is None or (isinstance(text_value, str) and not text_value.strip()):
                results.append(None)
                continue

            try:
                ratio = self._calculate_ratio(text_value)
                results.append(ratio)

                logger.debug("Text: %s, Ratio: %f", text_value[:50], ratio)

            except Exception as e:
                logger.error("Error processing text: %s, Error: %s", text_value[:50], str(e))
                results.append(None)

        return pa.array(results, type=pa.float64())

    @staticmethod
    def __return_column_type__() -> pa.DataType:
        return pa.float64()
