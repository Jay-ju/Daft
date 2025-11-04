# Copyright (c) Beijing Volcano Engine Technology Ltd.

from __future__ import annotations

import logging
from typing import Any

from daft.dependencies import pa
from daft.las.functions.types import Operator
from daft.las.functions.utils.common_utils import tracking_usage

logger = logging.getLogger(__name__)


class ChineseTextConverter(Operator):
    """**基于 OpenCC 的中文简繁体转换算子**

    **核心功能**
    - **多方向转换**：支持简繁体、台湾正体、香港繁体等多种转换方向
    - **混合文本处理**：正确处理中英文混杂内容，仅转换中文部分
    - **高效批处理**：支持大批量文本的快速转换处理

    **技术实现**
    - **转换引擎**：使用 OpenCC 进行转换
    """  # noqa: D415

    def __init__(
        self,
        direction: str = "t2s",
        **kwargs: Any,
    ) -> None:
        """初始化中文简繁体转换算子

        Args:
            direction: 转换方向
                可选值：["t2s", "s2t", "t2tw", "s2tw", "t2hk", "s2hk", "tw2s", "hk2s"]
                默认值："t2s" (繁体转简体)
        """  # noqa: D415
        super().__init__(**kwargs)
        self.direction = direction

        valid_directions = {"t2s", "s2t", "t2tw", "s2tw", "t2hk", "s2hk", "tw2s", "hk2s"}
        if self.direction not in valid_directions:
            raise ValueError(
                f"Unsupported conversion direction: {self.direction}. "
                f"Supported directions: {', '.join(sorted(valid_directions))}"
            )

        import opencc

        self._converter = opencc.OpenCC(self.direction)
        logger.info(
            "Initializing OpenCC converter with direction: %s",
            self.direction,
        )

        tracking_usage(op=self.__class__.__name__, model_service_or_lib="opencc")

    def transform(self, texts: pa.Array) -> pa.Array:
        """批量处理文本列进行中文简繁体转换

        Args:
            texts: 包含待转换文本的列，元素类型为字符串。

        Returns:
            转换后的文本列，元素类型为字符串。

        """  # noqa: D415
        if not self._converter:
            raise RuntimeError("OpenCC converter not properly initialized")

        logger.debug("Processing batch with %s texts", len(texts))
        results: list[str | None] = []

        for text in texts.to_pylist():
            if text is None:
                results.append(None)
                continue

            if not isinstance(text, str):
                logger.warning("Skipping non-string input: %s", type(text))
                results.append(text)
                continue

            try:
                converted_text = self._converter.convert(text)
                results.append(converted_text)
            except Exception as e:
                logger.error("Text conversion failed: %s, original text: %s...", e, text[:50])
                results.append(text)

        return pa.array(results, type=pa.string())

    @staticmethod
    def __return_column_type__() -> pa.DataType:
        return pa.string()
