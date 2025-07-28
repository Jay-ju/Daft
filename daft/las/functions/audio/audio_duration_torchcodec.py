# Copyright (c) Beijing Volcano Engine Technology Ltd.

from __future__ import annotations

import asyncio
import logging
from typing import Any

from daft.dependencies import pa
from daft.las.functions.types import Operator
from daft.las.functions.utils.audio_utils import decode_audio

logger = logging.getLogger(__name__)


class AudioDurationTorchcodec(Operator):
    """**音频时长获取模块 - 支持视频或音频输入，返回时长（秒）**

    **核心功能**
    - 支持解码任意音频/视频文件中的音轨
    - 返回音频的总时长（秒）
    """  # noqa: D415

    def __init__(
        self,
        num_coroutines: int = 1,
        **kwargs: Any,
    ) -> None:
        """初始化音频时长获取算子。

        参数：
            num_coroutines (int): 异步并发处理数量
        """  # noqa: D415
        super().__init__(**kwargs)
        self.num_coroutines = num_coroutines

    async def process(self, audio: bytes) -> float | None:
        if not audio:
            return None

        try:
            decoder = decode_audio(audio)

            # 优先尝试从 metadata 获取 duration
            duration_sec = decoder.metadata.duration_seconds_from_header

            # fallback：若 metadata 无法提供，则从 samples 获取
            if duration_sec is None:
                try:
                    duration_sec = decoder.get_all_samples().duration_seconds
                except Exception as fallback_e:
                    logger.error("Fallback duration calculation failed: %s", fallback_e)
                    return None

            return float(duration_sec)

        except Exception as e:
            logger.error("Audio duration extract failed: %s", e)
            return None

    async def async_run(self, audio_list: list[bytes]) -> list[float | None]:
        semaphore = asyncio.Semaphore(self.num_coroutines)

        async def bounded(audio: bytes) -> float | None:
            async with semaphore:
                return await self.process(audio)

        return await asyncio.gather(*[bounded(a) for a in audio_list])

    def transform(self, audio_col: pa.Array) -> pa.Array:
        """批量提取音频时长。

        Args:
            audio_col: 输入的音频/视频内容

        Returns:
            pa.float64()：音频时长（秒）
        """  # noqa: D415
        loop = asyncio.get_event_loop()
        results = loop.run_until_complete(self.async_run(audio_col.to_pylist()))
        return pa.array(results, type=AudioDurationTorchcodec.__return_column_type__())

    @staticmethod
    def __return_column_type__() -> pa.DataType:
        return pa.float64()
