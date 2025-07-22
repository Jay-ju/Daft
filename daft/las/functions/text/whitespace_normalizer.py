# Copyright (c) Beijing Volcano Engine Technology Ltd.

from __future__ import annotations

import logging
import re
from typing import Any

from daft.dependencies import pa
from daft.las.functions.types import Operator
from daft.las.functions.utils.special_characters import VARIOUS_WHITESPACES

logger = logging.getLogger(__name__)


class WhitespaceNormalizer(Operator):
    """**空白字符标准化器 - 将文本中不同种类的空白符号替换成标准空格**

    **核心功能**
    - 空白字符识别：自动识别各种Unicode空白字符
    - 标准化处理：将所有空白字符替换为标准空格

    **应用场景**
    - 文本预处理
    - 数据清洗和标准化
    - 格式统一化
    - 文本规范化

    **技术特性**
    - 支持多种空白字符：空格、制表符、换行符、全角空格等
    - 智能处理：保留文本内容，只替换空白字符
    - 前后清理：自动去除文本前后的空白字符
    - 保持结构：不改变文本的基本结构和内容
    """  # noqa: D415

    def __init__(self, **kwargs: Any) -> None:
        """空白字符标准化器初始化方法"""  # noqa: D415
        super().__init__(**kwargs)

        logger.info("Whitespace normalizer initialized successfully")

    def _normalize_whitespace(self, text: str) -> str:
        text = text.strip()

        normalized_text = "".join([char if char not in VARIOUS_WHITESPACES else " " for char in text])

        normalized_text = re.sub(r"\s+", " ", normalized_text)

        return normalized_text

    def transform(self, texts: pa.Array) -> pa.Array:
        """批量处理文本列，标准化空白字符

        Args:
            texts: 待处理的文本列，要求元素类型为字符串

        Returns:
            pa.Array: 标准化后的文本列
        """  # noqa: D415
        results: list[str | None] = []

        for text in texts:
            text_value = text.as_py()
            if text_value is None or (isinstance(text_value, str) and not text_value.strip()):
                results.append(None)
            else:
                try:
                    normalized_text = self._normalize_whitespace(text_value)
                    results.append(normalized_text)
                    logger.debug("Text: %s, Normalized: %s", text_value[:50], normalized_text[:50])
                except Exception as e:
                    logger.error("Error processing text: %s, Error: %s", text_value[:50], str(e))
                    results.append(None)

        return pa.array(results, type=pa.string())

    @staticmethod
    def __return_column_type__() -> pa.DataType:
        return pa.string()
