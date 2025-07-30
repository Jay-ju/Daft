# Copyright (c) Beijing Volcano Engine Technology Ltd.

from __future__ import annotations

import asyncio
import logging
from typing import Any

from daft.dependencies import np, pa
from daft.las.functions.types import Operator
from daft.las.functions.utils.audio_utils import decode_audio, encode_audio
from daft.las.functions.utils.common_utils import tracking_usage
from daft.las.functions.utils.audio_utils import decode_audio_torchaudio, encode_audio

logger = logging.getLogger(__name__)


class AudioStandardization(Operator):
    """**音频标准化模块 - 将音频统一为指定格式（采样率、声道、响度等）**

    **核心功能**
    - 支持采样率重采样
    - 支持声道统一（如转为单声道）
    - 支持响度归一化（目标 dBFS，带限制增益范围）
    - 默认输入输出音频为字节（bytes）格式

    **使用场景**
    - 多来源音频数据统一处理
    - 为下游 ASR/TTS 等模型做预处理
    - 多模态内容处理中的音频标准化环节
    """  # noqa: D415

    def __init__(
        self,
        target_sr: int | None = None,
        target_channels: int | None = None,
        target_dbfs: float | None = None,
        target_gain_range: list[float] = [-3, 3],
        num_coroutines: int = 1,
        **kwargs: Any,
    ) -> None:
        """初始化 AudioStandardization 算子

        Args:
            target_sr: 目标采样率（单位 Hz），如 16000
            target_channels: 目标声道数，如 1 表示单声道
            target_dbfs: 目标响度（单位 dBFS）
            target_gain_range: 归一化时允许的增益范围，如 [-3, 3]
            num_coroutines: 并发处理数量限制
            **kwargs: 其他传入 Operator 的参数
        """  # noqa: D415
        super().__init__(**kwargs)
        self.target_sr = target_sr
        self.target_channels = target_channels
        self.target_dbfs = target_dbfs
        self.target_gain_range = target_gain_range or [-3, 3]
        self.num_coroutines = num_coroutines

        tracking_usage(op=self.__class__.__name__, model_service_or_lib="torchcodec")

    async def process(self, audio: Any) -> bytes | None:
        if not audio:
            return None
        try:
            waveform, _ = decode_audio_torchaudio(
                audio,
                sample_rate=self.target_sr,
                num_channels=self.target_channels,
            )

            waveform = waveform.numpy()
            # 响度归一化
            if self.target_dbfs is not None:
                rms = np.sqrt(np.mean(waveform**2))
                current_dbfs = 20 * np.log10(rms + 1e-9)
                gain = np.clip(self.target_dbfs - current_dbfs, self.target_gain_range[0], self.target_gain_range[1])
                waveform *= 10 ** (gain / 20)

            # [-1, 1] 归一化
            max_amp = np.max(np.abs(waveform))  # type: ignore
            if max_amp > 0:
                waveform /= max_amp

            # 保存为 bytes
            return encode_audio({"samples": waveform, "sample_rate": self.target_sr})  # type: ignore

        except Exception as e:
            logger.warning("Audio standardization failed: %s", e)
            return None

    async def async_run(self, audio_list: list[bytes]) -> list[bytes | None]:
        semaphore = asyncio.Semaphore(self.num_coroutines)

        async def bounded(audio: bytes) -> bytes | None:
            async with semaphore:
                return await self.process(audio)

        return await asyncio.gather(*[bounded(a) for a in audio_list])

    def transform(self, audio_col: pa.Array) -> pa.Array:
        """批量处理音频字节数组

        Args:
            audio_col: 输入音频的 PyArrow 字节数组

        Returns:
            处理后的音频结果；失败则返回 None。
        """  # noqa: D415
        loop = asyncio.get_event_loop()
        results = loop.run_until_complete(self.async_run(audio_col.to_pylist()))
        return pa.array(results, type=AudioStandardization.__return_column_type__())

    @staticmethod
    def __return_column_type__() -> pa.DataType:
        return pa.binary()
