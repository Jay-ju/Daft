# Copyright (c) Beijing Volcano Engine Technology Ltd.

from __future__ import annotations

import logging
from typing import Any

import regex as re

from daft.dependencies import pa
from daft.las.functions.types import Operator

logger = logging.getLogger(__name__)


class CopyrightCleaner(Operator):
    """**版权声明移除器 - 移除文本中跟版权声明相关的文本**

    **核心功能**
    - 版权声明检测：自动识别代码中的版权声明注释
    - 智能清理：根据版权声明特征进行精确的内容移除
    - 多格式支持：支持块注释和行注释格式

    **应用场景**
    - 代码版权声明清理
    - 文本预处理和标准化
    - 代码质量检查和优化

    **技术特性**
    - 支持两种版权声明格式：
        - 块注释格式：/* ... */ 中包含copyright关键词的注释块
        - 行注释格式：以//、#、--开头的行注释
    """  # noqa: D415

    def __init__(self, **kwargs: Any) -> None:
        """版权声明移除器初始化方法

        初始化正则表达式模式用于版权声明检测和清理
        """  # noqa: D415
        super().__init__(**kwargs)

        self.block_comment_pattern = re.compile("/\\*[^*]*\\*+(?:[^/*][^*]*\\*+)*/")
        self.copyright_pattern = re.compile("copyright", re.IGNORECASE)

        logger.info("Copyright cleaner initialized successfully")

    def _clean_copyright_content(self, content: str) -> str:
        block_match = self.block_comment_pattern.search(content)
        if block_match:
            start, end = block_match.span()
            comment_block = content[start:end]
            if self.copyright_pattern.search(comment_block):
                content = content[:start] + content[end:]
                logger.debug("Removed copyright block comment from position %d to %d", start, end)
                return content

        lines = content.split("\n")
        skip_lines = 0

        for line in lines:
            stripped_line = line.strip()
            if (
                stripped_line.startswith("//")
                or stripped_line.startswith("#")
                or stripped_line.startswith("--")
                or not stripped_line
            ):
                skip_lines += 1
            else:
                break

        if skip_lines > 0:
            content = "\n".join(lines[skip_lines:])
            logger.debug("Removed %d leading comment lines", skip_lines)

        return content

    def transform(self, texts: pa.Array) -> pa.Array:
        """批量处理文本列，移除版权声明内容

        Args:
            texts: 待处理的文本列，要求元素类型为字符串

        Returns:
            pa.Array: 清理后的文本列，移除版权声明相关内容
        """  # noqa: D415
        results: list[str | None] = []

        for text in texts:
            text_value = text.as_py()
            if text_value is None or (isinstance(text_value, str) and not text_value.strip()):
                results.append(None)
            else:
                cleaned_text = self._clean_copyright_content(text_value)
                results.append(cleaned_text)

            logger.debug("Original text: %s, Cleaned text: %s", text_value[:50], cleaned_text[:50])

        return pa.array(results, type=pa.string())

    @staticmethod
    def __return_column_type__() -> pa.DataType:
        return pa.string()
