# Copyright (c) Beijing Volcano Engine Technology Ltd.

from __future__ import annotations

import logging
from typing import Any

from daft.dependencies import pa
from daft.las.functions.types import Operator
from daft.las.functions.utils.audio_utils import get_duration
from daft.las.functions.utils.common_utils import run_on_local_path

logger = logging.getLogger(__name__)


class AudioDuration(Operator):
    """**音频时长分析处理器，精确计算音频内容时长**

    **核心功能**
    - 精确计算音频时长(秒级精度)
    - 支持本地文件、TOS存储与二进制
    - 基于librosa专业音频处理库

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

    def _calculate_duration(self, audio: str | bytes | bytearray) -> float:
        """计算音频文件时长

        Args:
            audio: 音频文件路径(支持本地或TOS路径) 或 bytes

        Returns:
            音频时长(秒), 如果计算失败返回np.nan
        """  # noqa: D415
        import tempfile

        try:
            if isinstance(audio, (bytes, bytearray)):
                with tempfile.NamedTemporaryFile(suffix=".tmp", delete=True) as tmp:
                    tmp.write(audio)
                    tmp.flush()
                    return float(run_on_local_path(tmp.name, get_duration))
            elif isinstance(audio, str):
                return float(run_on_local_path(audio, get_duration))
            else:
                logger.warning("Unsupported audio input type: %s", type(audio))
                return float("nan")
        except Exception:
            logger.exception("Failed to calculate duration for input: %s", audio)
            return float("nan")

    def transform(self, audio_inputs: pa.Array) -> pa.Array:
        """计算音频文件时长

        Args:
            audio_inputs: 存放音频路径或二进制的列

        Returns:
            存放音频时长的列
        """  # noqa: D415
        result = [self._calculate_duration(x.as_py()) for x in audio_inputs]
        return pa.array(result, type=self.__return_column_type__())
