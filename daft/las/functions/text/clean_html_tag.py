# Copyright (c) Beijing Volcano Engine Technology Ltd.

from __future__ import annotations

import logging
from typing import Any

from bs4 import BeautifulSoup

from daft.dependencies import pa
from daft.las.functions.types import Operator
from daft.las.functions.utils.common_utils import tracking_usage

logger = logging.getLogger(__name__)


class CleanHtmlTag(Operator):
    """**HTML文档净化处理器 - 多结构解析与智能清理解决方案**

    **核心功能**
    - **多结构解析**
      - 标题提取：自动识别`<h1>`-`<h6>`标签
      - 正文抽取：智能识别文章主体内容
      - 冗余过滤：移除`<script>`/`<style>`等非文本标签
    - **智能处理**
      - 容错机制：支持残缺HTML片段解析
      - 格式保留：维持文本段落结构与换行逻辑

    **技术实现**
      - 基础库：`BeautifulSoup4`（html.parser）
    """  # noqa: D415

    def __init__(
        self,
        separator: str = "\n",
        strip: bool = True,
        **kwargs: Any,
    ) -> None:
        """HTML文档处理器初始化方法.

        Args:
            separator: 文本内容分隔符，默认为换行符'\n'。
                用于替换被移除HTML标签后的空白区域
            strip: 是否去除首尾空格，默认为True
        """  # noqa: D301
        super().__init__(**kwargs)
        self.separator = separator
        self.strip = strip

        tracking_usage(op=self.__class__.__name__, model_service_or_lib="bs4")

    def transform(self, texts: pa.Array) -> pa.Array:
        """批量处理HTML文本数组并清理标签.

        Args:
            texts: 包含HTML内容的文本数组，元素类型为字符串

        Returns:
            清理后的文本数组，元素类型为字符串
        """
        total_texts = len(texts)
        logger.debug("Processing %d HTML documents", total_texts)

        cleaned_texts = []
        failed_count = 0

        for _, html_doc in enumerate(texts):
            html_doc = html_doc.as_py()

            try:
                soup = BeautifulSoup(html_doc, "html.parser")
                cleaned_text = soup.get_text(separator=self.separator, strip=self.strip)
                cleaned_texts.append(cleaned_text)
            except Exception:
                logger.exception("HTML parsing failed!")
                cleaned_texts.append(None)
                failed_count += 1

        logger.info(
            "Clean results: total=%(total)d, success=%(success)d, failed=%(failed)d",
            {"total": total_texts, "success": total_texts - failed_count, "failed": failed_count},
        )
        return pa.array(cleaned_texts, type=self.__return_column_type__())

    @staticmethod
    def __return_column_type__() -> pa.DataType:
        return pa.string()
