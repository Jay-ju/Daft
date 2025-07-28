# Copyright (c) Beijing Volcano Engine Technology Ltd.

from __future__ import annotations

import asyncio
import logging
from typing import Any

from daft.dependencies import pa
from daft.las.functions.types import Operator
from daft.las.functions.utils.audio_utils import decode_audio, encode_audio

logger = logging.getLogger(__name__)


class AudioExtractAndSplit(Operator):
    """**音频提取与分段模块 - 支持视频或音频输入，按时间切分为多个片段**

    **核心功能**
    - 支持解码任意音频/视频文件中的音轨
    - 可指定分片时长（默认 2 小时），将长音频按时间切分
    """  # noqa: D415

    def __init__(
        self,
        split_duration: int = 7200,  # 单位：秒
        sample_rate: int | None = None,
        num_channels: int | None = None,
        file_format: str = "WAV",
        num_coroutines: int = 1,
        **kwargs: Any,
    ) -> None:
        """初始化音频提取与切分算子。

        参数：
            split_duration (int): 每段切片的时长（秒），默认7200（2小时）
            sample_rate (int): 解码/重采样目标采样率，默认16kHz
            num_channels (int): 声道数，默认2（立体声）
            num_coroutines (int): 异步并发处理数量
        """  # noqa: D415
        super().__init__(**kwargs)
        self.split_duration = split_duration
        self.sample_rate = sample_rate
        self.num_channels = num_channels
        self.file_format = file_format
        self.num_coroutines = num_coroutines

    async def process(self, audio: bytes) -> list[bytes] | None:
        if not audio:
            return None

        try:
            decoder = decode_audio(
                audio,
                sample_rate=self.sample_rate,
                num_channels=self.num_channels,
            )
            samples = decoder.get_all_samples()
            duration_sec = samples.duration_seconds

            if duration_sec == 0:
                logger.warning("Zero duration audio")
                return None

            # 基于 samples Tensor 切分（按时间）
            chunk_sec = self.split_duration
            sr = samples.sample_rate
            total_samples = samples.data.shape[1]
            chunk_size = int(sr * chunk_sec)

            results = []
            for start_idx in range(0, total_samples, chunk_size):
                end_idx = min(start_idx + chunk_size, total_samples)
                chunk = samples.data[:, start_idx:end_idx]
                encoded = encode_audio({"samples": chunk, "sample_rate": sr}, file_format=self.file_format)
                results.append(encoded)
            return results  # type: ignore

        except Exception as e:
            logger.warning("Audio extraction and split failed: %s", e)
            return None

    async def async_run(self, audio_list: list[bytes]) -> list[list[bytes] | None]:
        semaphore = asyncio.Semaphore(self.num_coroutines)

        async def bounded(audio: bytes) -> list[bytes] | None:
            async with semaphore:
                return await self.process(audio)

        return await asyncio.gather(*[bounded(a) for a in audio_list])

    def transform(self, audio_col: pa.Array) -> pa.Array:
        """批量提取音频并按时间切分。

        Args:
            audio_col: 输入的音频/视频内容（pa.binary 类型）

        Returns:
            pa.list(pa.binary())：切分后的多个音频片段（bytes）列表
        """  # noqa: D415
        loop = asyncio.get_event_loop()
        results = loop.run_until_complete(self.async_run(audio_col.to_pylist()))
        return pa.array(results, type=AudioExtractAndSplit.__return_column_type__())

    @staticmethod
    def __return_column_type__() -> pa.DataType:
        return pa.list_(pa.binary())
