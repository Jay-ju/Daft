# Copyright (c) Beijing Volcano Engine Technology Ltd.

from __future__ import annotations

import logging
from typing import Any

from daft.dependencies import pa
from daft.las.functions.types import Operator
from daft.las.functions.utils.common_utils import log_op_call

logger = logging.getLogger(__name__)


class RepeatedLinesCalculator(Operator):
    """**重复行计算器 - 计算文本中重复行的比例**

    **核心功能**
    - 重复行检测：自动识别文本中重复的行内容
    - 比例计算：计算重复行数与原始行数的比值
    - 质量评估：评估文本的重复程度和质量

    **应用场景**
    - 文本质量评估
    - 数据清洗和预处理
    - 内容重复检测
    - 文档质量检查

    **技术特性**
    - 精确匹配：完全相同的内容才算重复
    - 空行过滤：自动过滤空白行，不影响计算
    - 结果范围：0-1，越接近1表示重复越多
    """  # noqa: D415

    def __init__(self, **kwargs: Any) -> None:
        """重复行计算器初始化方法

        初始化重复行计算器
        """  # noqa: D415
        super().__init__(**kwargs)

        logger.info("Repeated lines calculator initialized successfully")

        log_op_call(logger=logger, op=self.__class__.__name__)

    def _calculate_repeated_ratio(self, text: str) -> float:
        lines = [line for line in text.splitlines() if line.strip()]
        if not lines:
            return 0.0

        unique_lines = len(set(lines))
        total_lines = len(lines)
        return (total_lines - unique_lines) / total_lines

    def transform(self, texts: pa.Array) -> pa.Array:
        """批量处理文本列，计算重复行比例

        Args:
            texts: 待处理的文本列，要求元素类型为字符串

        Returns:
            pa.Array: 重复行比例列，元素为浮点数，表示重复行比例
        """  # noqa: D415
        results: list[float | None] = []

        for text in texts:
            text_value = text.as_py()
            if text_value is None or (isinstance(text_value, str) and not text_value.strip()):
                results.append(None)
            else:
                ratio = self._calculate_repeated_ratio(text_value)
                results.append(ratio)
                logger.debug("Text: %s, Repeated ratio: %f", text_value[:50], ratio)

        return pa.array(results, type=pa.float64())

    @staticmethod
    def __return_column_type__() -> pa.DataType:
        return pa.float64()
