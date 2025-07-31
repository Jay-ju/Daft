# Copyright (c) Beijing Volcano Engine Technology Ltd.

from __future__ import annotations

import hashlib
import logging
from typing import Any

from daft.dependencies import pa
from daft.las.functions.types import Operator
from daft.las.functions.utils.common_utils import log_op_call

logger = logging.getLogger(__name__)


class Md5Calculator(Operator):
    """**MD5哈希值计算器 - 计算文本的MD5指纹**

    **核心功能**
    - 针对每条文本数据生成对应的MD5哈希值
    - 输出固定长度（32位小写十六进制）指纹
    - 支持批量处理

    **应用场景**
    - 文本唯一标识生成
    - 重复检测、数据去重
    - 数据一致性验证
    """  # noqa: D415

    def __init__(self, **kwargs: Any) -> None:
        """MD5哈希值计算器初始化方法"""  # noqa: D415
        super().__init__(**kwargs)
        logger.info("Md5Calculator initialized successfully")

        log_op_call(logger=logger, op=self.__class__.__name__)

    def transform(self, texts: pa.Array) -> pa.Array:
        """批量计算文本的MD5哈希值

        Args:
            texts: 待处理的文本列，要求元素类型为字符串

        Returns:
            pa.Array: MD5哈希值列，元素类型为字符串
        """  # noqa: D415
        results: list[str | None] = []

        for idx, text in enumerate(texts):
            text_value = text.as_py()
            if text_value is None or (isinstance(text_value, str) and not text_value.strip()):
                results.append(None)
                continue
            try:
                md5_hash = hashlib.md5(text_value.encode("utf-8")).hexdigest()
                results.append(md5_hash)
            except Exception:
                logger.exception("Error computing MD5 at index %d", idx)
                results.append(None)

        return pa.array(results, type=self.__return_column_type__())

    @staticmethod
    def __return_column_type__() -> pa.DataType:
        return pa.string()
