# Copyright (c) Beijing Volcano Engine Technology Ltd.

from __future__ import annotations

import logging
from typing import Any

from daft.dependencies import pa
from daft.las.functions.types import Operator
from daft.las.functions.utils.common_utils import tracking_usage

logger = logging.getLogger(__name__)


class TextLengthCalculator(Operator):
    """**文本长度计算器 - 计算文本的字符长度**

    **核心功能**
    - 文本长度计算：计算输入文本的字符数量
    - 批量处理：支持批量文本长度计算
    - 数值输出：返回整数类型的长度值

    **应用场景**
    - 文本长度统计分析
    - 数据质量检查
    - 文本预处理和筛选

    **技术特性**
    - 精确计算：使用Python内置len()函数计算字符数
    - 类型安全：确保输入为字符串类型
    """  # noqa: D415

    def __init__(self, **kwargs: Any) -> None:
        """文本长度计算器初始化方法

        初始化计算器用于文本长度计算
        """  # noqa: D415
        super().__init__(**kwargs)
        logger.info("Text length calculator initialized successfully")

        tracking_usage(op=self.__class__.__name__)

    def transform(self, texts: pa.Array) -> pa.Array:
        """批量计算文本列的长度

        Args:
            texts: 待处理的文本列，要求元素类型为字符串

        Returns:
            pa.Array: 文本长度列，元素类型为整数
        """  # noqa: D415
        results: list[int | None] = []

        for text in texts:
            text_value = text.as_py()
            if text_value is None or (isinstance(text_value, str) and not text_value.strip()):
                results.append(None)
            else:
                length = len(text_value)
                results.append(length)

            logger.debug("Text: %s, Length: %d", str(text_value)[:50], length if text_value is not None else None)

        return pa.array(results, type=pa.int64())

    @staticmethod
    def __return_column_type__() -> pa.DataType:
        return pa.int64()
