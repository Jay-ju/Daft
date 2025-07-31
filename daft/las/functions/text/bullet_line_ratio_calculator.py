# Copyright (c) Beijing Volcano Engine Technology Ltd.

from __future__ import annotations

import logging
from typing import Any

from daft.dependencies import pa
from daft.las.functions.types import Operator
from daft.las.functions.utils.common_utils import log_op_call
from daft.las.functions.utils.special_characters import BULLET_POINTS

logger = logging.getLogger(__name__)


class BulletLineRatioCalculator(Operator):
    """**项目符号行占比计算器 - 计算文本中项目符号行的比例**

    **核心功能**
    - 项目符号检测：自动识别文本中以项目符号开头的行
    - 比例计算：计算项目符号行数与总行数的比值

    **应用场景**
    - 文本质量评估
    - 数据清洗和预处理

    **技术特性**
    - 支持多种项目符号：•、-、·、●、▪、—、*等
    - 精确识别：只统计以项目符号开头的行
    - 结果范围：0-1，越接近1表示项目符号行越多
    - 智能过滤：自动过滤空白行，不影响计算
    """  # noqa: D415

    def __init__(self, **kwargs: Any) -> None:
        """项目符号行占比计算器初始化方法"""  # noqa: D415
        super().__init__(**kwargs)

        logger.info("Bullet line ratio calculator initialized successfully")

        log_op_call(logger=logger, op=self.__class__.__name__)

    def _calculate_bullet_ratio(self, text: str) -> float:
        lines = [line for line in text.splitlines() if line.strip()]
        if not lines:
            return 0.0

        bullet_count = 0
        for line in lines:
            if any(line.lstrip().startswith(bullet) for bullet in BULLET_POINTS):
                bullet_count += 1

        return bullet_count / len(lines)

    def transform(self, texts: pa.Array) -> pa.Array:
        """批量处理文本列，计算项目符号行比例

        Args:
            texts: 待处理的文本列，要求元素类型为字符串

        Returns:
            pa.Array: 项目符号行比例列，元素为浮点数，表示项目符号行比例
        """  # noqa: D415
        results: list[float | None] = []

        for text in texts:
            text_value = text.as_py()
            if text_value is None or (isinstance(text_value, str) and not text_value.strip()):
                results.append(None)
                continue

            try:
                ratio = self._calculate_bullet_ratio(text_value)
                results.append(ratio)

                logger.debug("Text: %s, Bullet line ratio: %f", text_value[:50], ratio)

            except Exception as e:
                logger.error("Error processing text: %s, Error: %s", text_value[:50], str(e))
                results.append(None)

        return pa.array(results, type=pa.float64())

    @staticmethod
    def __return_column_type__() -> pa.DataType:
        return pa.float64()
