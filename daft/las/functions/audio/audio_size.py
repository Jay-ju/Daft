# Copyright (c) Beijing Volcano Engine Technology Ltd.

from __future__ import annotations

import logging
from typing import Any

from daft.dependencies import pa
from daft.las.functions.types import Operator
from daft.las.io import file_size

logger = logging.getLogger(__name__)


class AudioSize(Operator):
    """**音频文件元数据分析处理器，精确计算文件大小**

    **核心功能**
    - 精确计算音频文件字节大小
    - 支持本地文件与TOS存储
    - 轻量高效，适合批量处理

    **格式支持**
    - MP3 (.mp3)
    - WAV (.wav)
    - FLAC (.flac)
    - OGG (.ogg)
    - AAC (.aac)
    - M4A (.m4a)
    """  # noqa: D415

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

    @staticmethod
    def __return_column_type__() -> pa.DataType:
        return pa.float32()

    @staticmethod
    def _calculate_size(audio_path: str) -> float:
        """计算音频文件大小

        Args:
            audio_path: 音频文件路径(支持本地或TOS路径)

        Returns:
            文件大小(字节), 如果计算失败返回np.nan
        """  # noqa: D415
        try:
            size_bytes = float(file_size(audio_path))
        except FileNotFoundError:
            logger.warning("File not found: %s", audio_path)
            return float("nan")
        except Exception:
            logger.exception("Failed to calculate size for %s", audio_path)
            return float("nan")
        else:
            logger.debug("Calculated size for %s: %s bytes", audio_path, size_bytes)
            return size_bytes

    def transform(self, audio_paths: pa.Array) -> pa.Array:
        """计算音频文件大小

        Args:
            audio_paths: 存放音频路径的列

        Returns:
            存放音频大小的列
        """  # noqa: D415
        result = [self._calculate_size(audio_path.as_py()) for audio_path in audio_paths]
        return pa.array(result, type=self.__return_column_type__())
