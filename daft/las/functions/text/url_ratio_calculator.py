# Copyright (c) Beijing Volcano Engine Technology Ltd.

from __future__ import annotations

import logging
import re
from typing import Any

from daft.dependencies import pa
from daft.las.functions.types import Operator

logger = logging.getLogger(__name__)


class UrlRatioCalculator(Operator):
    """**URL占比计算器 - 基于URL字符占比的文本特征提取**

    **核心功能**
    - URL占比计算：精确统计URL字符在文本中的占比
    - 多协议支持：支持HTTP、HTTPS等多种URL协议
    - 智能识别：使用正则表达式精确识别URL格式

    **应用场景**
    - 文本质量评估
    - 数据清洗和预处理
    - 文本分类特征提取
    - 内容安全检测
    - 网页内容分析

    **技术特性**
    - URL字符占比：URL字符占总字符数的比例
    - 支持多种URL格式，包括但不限于：
        - HTTP/HTTPS链接
        - 带查询参数的URL
        - 带锚点的URL
        - 带端口的URL
    """  # noqa: D415

    def __init__(
        self,
        **kwargs: Any,
    ) -> None:
        """URL占比计算器初始化方法"""  # noqa: D415
        super().__init__(**kwargs)

        self.regex_url = re.compile(
            r"http[s]?://(?:[a-zA-Z0-9$-_@.&+#%=~:/?]+|(?:%[0-9a-fA-F][0-9a-fA-F]))+", re.IGNORECASE
        )

        logger.info("URL ratio calculator initialized")

    def _calculate_ratio(self, text: str) -> float:
        all_urls = self.regex_url.findall(text)
        url_chars = sum(len(url) for url in all_urls)
        return url_chars / len(text)

    def transform(self, texts: pa.Array) -> pa.Array:
        """批量处理文本列，计算URL占比

        Args:
            texts: 待处理的文本列，要求元素类型为字符串

        Returns:
            pa.Array: 占比结果列，元素为浮点数，表示URL字符的占比
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

                logger.debug("Text: %s, URL Ratio: %f", text_value[:50], ratio)

            except Exception as e:
                logger.error("Error processing text: %s, Error: %s", text_value[:50], str(e))
                results.append(None)

        return pa.array(results, type=pa.float64())

    @staticmethod
    def __return_column_type__() -> pa.DataType:
        return pa.float64()
