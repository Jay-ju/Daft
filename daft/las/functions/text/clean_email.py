# Copyright (c) Beijing Volcano Engine Technology Ltd.

from __future__ import annotations

import logging
from typing import Any

import regex as re

from daft.dependencies import pa
from daft.las.functions.types import Operator
from daft.las.functions.utils.common_utils import tracking_usage

logger = logging.getLogger(__name__)


class CleanEmail(Operator):
    """基于正则的 Email 地址清理算子.

    使用预设或自定义的正则表达式扫描文本中的 email 地址，并将其替换为指定的字符串，可用于敏感信息脱敏或格式标准化。

    **核心功能**
        - 多场景支持：内置通用匹配模式，同时允许注入自定义正则表达式。
        - 可控替换：可配置替换串，实现脱敏或占位填充。
        - 批量兼容：支持数组批量处理
    """

    def __init__(self, pattern: str | None = "", repl: str = "", **kwargs: Any) -> None:
        r"""初始化 Email 清理算子.

        Args:
            pattern: 定位 email 的正则表达式，如果以 `r'...'` 或 `r"..."` 形式传入，算子会去除前缀，支持自定义。
                默认值：r"[A-Za-z0-9.\-+_]+@[a-z0-9.\-+_]+\.[a-z]+"
            repl: 用于替换匹配到 email 地址的字符串。
                默认值：""
        """
        super().__init__(**kwargs)

        self.pattern = pattern or r"[A-Za-z0-9.\-+_]+@[a-z0-9.\-+_]+\.[a-z]+"
        self.repl = repl

        # 如果 pattern 包含 r'...' 或 r"..." 的字面量标记，则去掉外层标识
        if (len(self.pattern) > 2) and (
            (self.pattern.startswith("r'") and self.pattern.endswith("'"))
            or (self.pattern.startswith('r"') and self.pattern.endswith('"'))
        ):
            self.pattern = self.pattern[2:-1]

        tracking_usage(op=self.__class__.__name__, model_service_or_lib="regex")

    def _clean_text(self, text: str) -> str | None:
        """对单个字符串进行 email 替换处理，返回替换后的字符串或 None（若输入为 None）."""
        try:
            if not re.search(self.pattern, text, flags=re.DOTALL):
                return text
            return re.sub(pattern=self.pattern, repl=self.repl, string=text, flags=re.DOTALL)
        except Exception:
            logger.exception("Email 移除失败！")
            return None

    def transform(self, texts: pa.Array) -> pa.Array:
        """批量处理文本数组，返回清洗后的字符串数组.

        Args:
            texts: 字符串数组，每个元素为待处理的文本（允许 None）。

        Returns:
            字符串数组，包含替换/清洗后的文本，若处理失败则为 None。
        """
        total_texts = len(texts)
        logger.debug("Processing %d texts for email cleaning", total_texts)

        cleaned: list[str | None] = []
        failed = 0

        for _, item in enumerate(texts):
            text = item.as_py()
            result = self._clean_text(text)
            if result is None and text is not None:
                failed += 1
            cleaned.append(result)

        logger.info(
            "Email clean results: total=%(total)d, success=%(success)d, failed=%(failed)d",
            {"total": total_texts, "success": total_texts - failed, "failed": failed},
        )

        return pa.array(cleaned, type=self.__return_column_type__())

    @staticmethod
    def __return_column_type__() -> pa.DataType:
        return pa.string()
