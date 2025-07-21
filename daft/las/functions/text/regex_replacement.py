# Copyright (c) Beijing Volcano Engine Technology Ltd.

from __future__ import annotations

import logging
from typing import Any

import regex as re
from regex import Pattern

from daft.dependencies import pa
from daft.las.functions.types import Operator

logger = logging.getLogger(__name__)


class RegexReplacer(Operator):
    """**正则表达式批量替换处理器**

    **核心功能**
    - 双模式替换机制：
        - 精确字符串匹配替换
        - 正则表达式模式匹配替换
    - 批量处理能力：
        - 支持多组 `pattern-replacement` 对并行处理
    - 容错机制：
        - 异常模式跳过并记录详细日志
    """  # noqa: D415

    def __init__(self, patterns: list[str], replacements: list[str], **kwargs: Any) -> None:
        """正则表达式替换处理器初始化方法.

        Args:
            patterns: 需要替换的特定内容列表
                * 支持正则表达式和特定字符串
                * 多元素列表会逐个替换文本内容
                * 示例: [r"\d+", "http://"]
            replacements: 被替换成的内容列表
                * 若仅一个元素，则以该元素替换所有patterns内容
                * 多元素时与patterns一一对应替换
                * 示例: ["NUM", "URL"]
        """  # noqa: D301
        super().__init__(**kwargs)
        self.patterns = patterns
        self.replacements = replacements

        self.compiled_patterns = []
        for p in self.patterns:
            self.compiled_patterns.append(self._prepare_pattern(p))

        logger.debug("Configured replacement patterns: %s", self.patterns)
        logger.debug("Configured replacement replacements: %s", self.replacements)

    def _prepare_pattern(self, pattern: str) -> Pattern[str]:
        if (pattern is not None and len(pattern) > 2) and (
            (pattern.startswith("r'") and pattern.endswith("'")) or (pattern.startswith('r"') and pattern.endswith('"'))
        ):
            pattern = pattern[2:-1]
        return re.compile(pattern, flags=re.DOTALL)

    def _get_replacement(self, index: int) -> str:
        if len(self.replacements) == 1:
            return self.replacements[0]
        if len(self.replacements) == 0:
            return ""
        if index < len(self.replacements):
            return self.replacements[index]
        raise ValueError(
            f"Patterns数量 ({len(self.patterns)}) 必须小于等于Replacements数量 "
            f"({len(self.replacements)}) 或者Replacements数量为0或1"
        )

    def transform(self, texts: pa.Array) -> pa.Array:
        """批量处理文本数组并进行正则表达式替换.

        Args:
            texts: 包含原始文本内容的数组，元素类型为字符串

        Returns:
            pyarrow.Array: 替换后的文本内容数组，元素类型为字符串
        """
        processed_contents = []
        error_reports = []

        for content_idx, orig_text in enumerate(texts):
            current_text = orig_text.as_py()
            content_errors = []

            try:
                for pattern_idx, pattern in enumerate(self.compiled_patterns):
                    replacement = self._get_replacement(pattern_idx)
                    current_text = pattern.sub(replacement, current_text)
            except Exception as e:
                error_msg = (
                    f"Content[{content_idx}]:{orig_text} failed at pattern[{pattern_idx}]: "
                    f"({self.patterns[pattern_idx]} -> {replacement}) - {e!s}"
                )
                content_errors.append(error_msg)
                logger.exception(error_msg)
                current_text = None

            if content_errors:
                error_reports.extend(content_errors)
            processed_contents.append(current_text)

        if error_reports:
            logger.warning(
                "Completed with %d content errors\nSample errors:\n%s", len(error_reports), "\n".join(error_reports[:3])
            )

        return pa.array(processed_contents, type=self.__return_column_type__())

    @staticmethod
    def __return_column_type__() -> pa.DataType:
        return pa.string()
