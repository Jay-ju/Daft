# Copyright (c) Beijing Volcano Engine Technology Ltd.

from __future__ import annotations

import logging
from typing import Any

import regex as re

from daft.dependencies import pa
from daft.las.functions.types import Operator
from daft.las.functions.utils.common_utils import log_op_call

logger = logging.getLogger(__name__)


class MaximumWordLengthCalculator(Operator):
    """**最大英文单词长度计算器 - 统计文本中英文单词的最大长度**

    **核心功能**
    - 英文单词识别：使用正则表达式识别文本中的英文单词
    - 最大长度计算：计算所有英文单词中的最大长度
    - 批量处理：支持批量文本的最大英文单词长度计算

    **应用场景**
    - 文本质量检查
    - 数据预处理和筛选

    **技术特性**
    - 仅识别英文单词：使用正则表达式 `[A-Za-z]+` 匹配英文单词
    - 智能处理：自动忽略非英文单词，只计算英文单词长度
    """  # noqa: D415

    def __init__(self, **kwargs: Any) -> None:
        """最大单词长度计算器初始化方法

        初始化计算器用于英文单词最大长度计算
        """  # noqa: D415
        super().__init__(**kwargs)
        logger.info("Maximum word length calculator initialized successfully")

        log_op_call(logger=logger, op=self.__class__.__name__, model_service_or_lib="regex")

    def transform(self, texts: pa.Array) -> pa.Array:
        """批量计算文本列中英文单词的最大长度

        Args:
            texts: 待处理的文本列，要求元素类型为字符串

        Returns:
            pa.Array: 最大单词长度列，元素类型为整数
        """  # noqa: D415
        results: list[int | None] = []

        for text in texts:
            text_value = text.as_py()
            if text_value is None or (isinstance(text_value, str) and not text_value.strip()):
                results.append(None)
            else:
                words = re.findall(r"[A-Za-z]+", text_value)
                max_len = max((len(w) for w in words), default=0)
                results.append(max_len)

            logger.debug("Text: %s, Max word length: %d", str(text_value)[:50], max_len)

        return pa.array(results, type=pa.int64())

    @staticmethod
    def __return_column_type__() -> pa.DataType:
        return pa.int64()
