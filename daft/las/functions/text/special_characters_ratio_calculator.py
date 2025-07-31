# Copyright (c) Beijing Volcano Engine Technology Ltd.

from __future__ import annotations

import logging
import string
from typing import Any

from daft.dependencies import pa
from daft.las.functions.types import Operator
from daft.las.functions.utils.common_utils import log_op_call
from daft.las.functions.utils.special_characters import SPECIAL_CHARACTERS

logger = logging.getLogger(__name__)


class SpecialCharactersRatioCalculator(Operator):
    """**特殊字符占比计算器 - 基于特殊字符占比的文本特征提取**

    **核心功能**
    - 特殊字符占比计算：精确统计特殊字符在文本中的占比
    - 多粒度支持：可选择不同类型的特殊字符进行计算
    - 灵活配置：支持计算所有特殊字符或特定类型字符的占比

    **应用场景**
    - 文本质量评估
    - 数据清洗和预处理
    - 文本分类特征提取
    - 内容安全检测
    - 多语言文本分析

    **技术特性**
    - 支持多种字符类型：
        - all: 所有特殊字符（默认）
        - whitespace: 空白字符
        - punctuation: 标点符号
        - digits: 数字字符
        - emoji: 表情符号
    - 支持多语言Unicode字符识别
    """  # noqa: D415

    def __init__(
        self,
        character_type: str = "all",
        **kwargs: Any,
    ) -> None:
        """特殊字符占比计算器初始化方法

        Args:
            character_type: 字符类型
                描述：选择要计算的特殊字符类型
                可选值：all, whitespace, punctuation, digits, emoji
                默认值：all
        """  # noqa: D415
        super().__init__(**kwargs)

        self.character_type = character_type
        self._setup_target_chars()

        logger.info("Special characters ratio calculator initialized with character_type=%s", self.character_type)

        log_op_call(logger=logger, op=self.__class__.__name__)

    def _setup_target_chars(self) -> None:
        if self.character_type == "whitespace":
            self.target_chars = set(string.whitespace)
        elif self.character_type == "punctuation":
            self.target_chars = set(string.punctuation)
        elif self.character_type == "digits":
            self.target_chars = set(string.digits)
        elif self.character_type == "emoji":
            import emoji

            self.target_chars = set(emoji.EMOJI_DATA.keys())
        elif self.character_type == "all":
            self.target_chars = SPECIAL_CHARACTERS
        else:
            raise ValueError(
                f"Unsupported character_type: {self.character_type}. "
                f"Supported: all, whitespace, punctuation, digits, emoji"
            )

    def _calculate_ratio(self, text: str) -> float:
        special_char_count = sum(1 if char in self.target_chars else 0 for char in text)
        return special_char_count / len(text)

    def transform(self, texts: pa.Array) -> pa.Array:
        """批量处理文本列，计算特殊字符占比

        Args:
            texts: 待处理的文本列，要求元素类型为字符串

        Returns:
            pa.Array: 占比结果列，元素为浮点数，表示特殊字符的占比
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
