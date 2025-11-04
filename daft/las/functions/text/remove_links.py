# Copyright (c) Beijing Volcano Engine Technology Ltd.
from __future__ import annotations

import logging
import re
from typing import Any

from daft.dependencies import pa
from daft.las.functions.types import Operator
from daft.las.functions.utils.common_utils import tracking_usage

logger = logging.getLogger(__name__)


class RemoveLinks(Operator):
    """**超链接移除算子 - 文本链接正则替换**

    **核心功能**
    - 识别协议链接、www 链接、域名 + 路径等通用形式的超链接
    - 将命中的超链接替换为指定字符串（`repl`），默认替换为""
    - 批量处理字符串，异常项返回 `None`

    **技术实现**
    - 正则匹配：默认内置表达式与 LAS 版本语义等价
    - 兼容传入字符串的 `r'...'` / `r"..."` 包裹形式（去除前缀 `r` 与引号）
    - 预编译正则并启用 `DOTALL` 语义，覆盖跨行文本
    - 日志记录：输出默认/自定义正则与处理统计，便于排查
    """  # noqa: D415

    def __init__(self, pattern: str = "", repl: str = "", **kwargs: Any) -> None:
        """初始化超链接移除算子.

        Args:
            pattern: 超链接定位正则。如果为空则使用算子内置表达式；兼容传入 `r'...'` 或 `r"..."` 的包裹形式。
            repl: 替换命中超链接的字符串，默认空字符串。
        """
        super().__init__(**kwargs)

        self.repl = repl

        if pattern is None or len(pattern) == 0:
            self.pattern = r"""(?i)\b((?:[a-z][\w-]+:(?:\/{1,3}|[a-z0-9%])|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}\/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'\".,<>?«»“”‘’]))"""
            logger.info("CleanLinksOperator using default pattern")
        else:
            if (len(pattern) > 2) and (
                (pattern.startswith("r'") and pattern.endswith("'"))
                or (pattern.startswith('r"') and pattern.endswith('"'))
            ):
                pattern = pattern[2:-1]
            self.pattern = pattern
            logger.info("CleanLinksOperator using custom pattern: %s", self.pattern)

        # 预编译正则，提高性能；启用 DOTALL 以覆盖跨行文本（与原 flags=re.DOTALL 等价）
        try:
            self._compiled = re.compile(self.pattern, flags=re.DOTALL)
        except Exception:
            logger.exception("Invalid regex pattern, fallback to a no-op pattern: %s", self.pattern)
            self._compiled = re.compile(r"$^", flags=re.DOTALL)

        tracking_usage(op=self.__class__.__name__, model_service_or_lib="regex")

    def transform(self, texts: pa.Array) -> pa.Array:
        """批量处理字符串数组，移除/替换其中的超链接.

        Args:
            texts: 字符串数组，每个元素为文本内容

        Returns:
            处理后的字符串数组；异常项返回 `None`
        """
        total_texts = len(texts)
        logger.debug("CleanLinksOperator: processing %d texts", total_texts)

        cleaned_texts: list[str | None] = []
        failed_count: int = 0

        for idx, item in enumerate(texts):
            try:
                text = item.as_py()
                if text is None:
                    cleaned_texts.append(None)
                    continue

                if not self._compiled.search(text):
                    cleaned_texts.append(text)
                    continue

                replaced = self._compiled.sub(self.repl, text)
                cleaned_texts.append(replaced)
            except Exception:
                logger.exception("Link removal failed at index %d", idx)
                cleaned_texts.append(None)
                failed_count += 1

        logger.info(
            "Remove links results: total=%(total)d, success=%(success)d, failed=%(failed)d",
            {"total": total_texts, "success": total_texts - failed_count, "failed": failed_count},
        )

        return pa.array(cleaned_texts, type=self.__return_column_type__())

    @staticmethod
    def __return_column_type__() -> pa.DataType:
        return pa.string()
