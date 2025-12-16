# Copyright (c) Beijing Volcano Engine Technology Ltd.

from __future__ import annotations

from typing import Any, cast

from daft.dependencies import np, pa
from daft.las.functions.types import Operator
from daft.las.functions.utils.audio_utils import decode_audio_torchaudio, encode_audio
from daft.las.functions.utils.common_utils import FastWriteCounter, get_logger, tracking_usage


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
        **kwargs: Any,
    ) -> None:
        """初始化 AudioStandardization 算子

        Args:
            target_sr: 目标采样率（单位 Hz），如 16000，为 None 时保留原始采样率
            target_channels: 目标声道数，如 1 表示单声道，为 None 时保留原始声道数
            target_dbfs: 目标响度（单位 dBFS），为 None 时不进行响度归一化
            target_gain_range: 归一化时允许的增益范围，如 [-3, 3]
            **kwargs: 其他传入 Operator 的参数
        """  # noqa: D415
        super().__init__(**kwargs)
        self.target_sr = target_sr
        self.target_channels = target_channels
        self.target_dbfs = target_dbfs
        self.target_gain_range = target_gain_range or [-3, 3]

        self.submit_counter = FastWriteCounter()
        self.success_counter = FastWriteCounter()
        self.failed_counter = FastWriteCounter()

        self.logger = get_logger(f"AudioStandardization-{id(self)}")

        tracking_usage(op=self.__class__.__name__, model_service_or_lib="torchcodec")

    def log_progress(self) -> None:
        submitted = self.submit_counter.value
        succeed = self.success_counter.value
        failed = self.failed_counter.value
        finished = succeed + failed
        running = submitted - finished
        self.logger.info(
            "%s/%s running, finished/succeed/failed: %s/%s/%s", running, submitted, finished, succeed, failed
        )

    def process(self, audio: Any) -> bytes | None:
        """Process a single audio file for standardization.

        Args:
            audio: Input audio (bytes or path)

        Returns:
            bytes | None: Output audio bytes on success, None on failure
        """
        self.submit_counter.increment()

        if not audio:
            self.failed_counter.increment()
            self.log_progress()
            return None

        try:
            waveform, actual_sr = decode_audio_torchaudio(
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
            max_amp = np.max(np.abs(waveform))
            if max_amp > 0:
                waveform /= max_amp

            # 保存为 bytes，使用实际的采样率
            result = cast("bytes", encode_audio({"samples": waveform, "sample_rate": actual_sr}))

            self.success_counter.increment()
            self.log_progress()
            self.logger.info("Finished standardization")
            return result

        except Exception as e:
            self.failed_counter.increment()
            self.log_progress()
            self.logger.error("Audio standardization failed: %s", e)
            return None

    def transform(self, audio_col: pa.Array) -> pa.Array:
        """批量处理音频字节数组

        Args:
            audio_col: 输入音频的 PyArrow 字节数组

        Returns:
            处理后的音频结果；失败则返回 None。
        """  # noqa: D415
        results = []
        for audio in audio_col.to_pylist():
            try:
                result = self.process(audio)
            except Exception as e:
                self.logger.error("[transform] Failed to process audio: %s", e)
                result = None
            results.append(result)

        return pa.array(results, type=AudioStandardization.__return_column_type__())

    @staticmethod
    def __return_column_type__() -> pa.DataType:
        return pa.binary()
