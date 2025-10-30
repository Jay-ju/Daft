# Copyright (c) Beijing Volcano Engine Technology Ltd.

from __future__ import annotations

import logging
from typing import Any

import librosa
from sklearn.decomposition import NMF

from daft.dependencies import pa
from daft.las.functions.types import Operator
from daft.las.functions.utils.common_utils import run_on_local_path, tracking_usage

logger = logging.getLogger(__name__)


class AudioSNR(Operator):
    """音频信噪比（SNR）计算器，基于非负矩阵分解（NMF）进行信号-噪声分离

    **核心功能：**
    - 对音频进行 STFT 频谱分析；
    - 使用 NMF 将频谱分解为信号与噪声分量；
    - 基于重建的时域信号与噪声估计能量比值，计算 SNR（单位 dB）。

    **格式支持：**
    - 常见音频格式：mp3, wav, flac, ogg, aac, m4a
    - 支持本地路径与对象存储路径（tos:// 或 s3://）
    """  # noqa: D415

    def __init__(self, n_components: int = 2, max_iter: int = 200, **kwargs: Any) -> None:
        """初始化音频 SNR 计算配置

        Args:
            n_components: NMF 分解的组件数量
                默认值：2（通常认为第一个分量为信号、其余分量为噪声）
            max_iter: NMF 最大迭代次数
                默认值：200
        """  # noqa: D415
        super().__init__(**kwargs)
        if n_components < 2:
            raise ValueError("n_components 必须 >= 2，至少包含一个信号分量与一个噪声分量。")
        if max_iter <= 0:
            raise ValueError("max_iter 必须为正整数。")

        self.n_components = int(n_components)
        self.max_iter = int(max_iter)

        tracking_usage(op=self.__class__.__name__, model_service_or_lib="librosa+sklearn.NMF")

    @staticmethod
    def __return_column_type__() -> pa.DataType:
        return pa.float64()

    def _calculate_snr_on_local_file(self, local_path: str) -> float:
        """Compute SNR on a local audio file (returns dB).

        Args:
            local_path: local audio file path

        Returns:
            float: SNR in dB. Returns NaN on failure. Returns +inf if estimated noise power is ~0.
        """
        try:
            import numpy as np

            # Load waveform at native sampling rate
            y, _ = librosa.load(local_path, sr=None)

            # STFT and magnitude/phase
            D = librosa.stft(y)
            magnitude = np.abs(D)
            angle = np.angle(D)

            # NMF decomposition on magnitude
            model = NMF(
                n_components=self.n_components,
                max_iter=self.max_iter,
                init="random",
                random_state=0,
            )
            W = model.fit_transform(magnitude)
            H = model.components_

            # Signal component: first component
            signal_mag = np.outer(W[:, 0], H[0, :])
            # Noise: sum of remaining components
            if self.n_components > 1:
                noise_mag = np.sum([np.outer(W[:, i], H[i, :]) for i in range(1, self.n_components)], axis=0)
            else:
                noise_mag = np.zeros_like(signal_mag)

            # Reconstruct complex spectrograms using original phase
            signal_complex = signal_mag * np.exp(1j * angle)
            noise_complex = noise_mag * np.exp(1j * angle)

            # Inverse STFT to time-domain
            y_signal = librosa.istft(signal_complex)
            y_noise = librosa.istft(noise_complex)

            # Compute mean-squared energy
            signal_power = float(np.mean(y_signal**2))
            noise_power = float(np.mean(y_noise**2))

            if noise_power < 1e-12:
                return float("inf")

            snr_db = 10.0 * np.log10(signal_power / noise_power)
            logger.debug("Calculated SNR for %s: %.4f dB", local_path, snr_db)
            return float(snr_db)
        except Exception:
            logger.exception("Failed to calculate SNR on local file: %s", local_path)
            return float("nan")

    def transform(self, audio_paths: pa.Array | None = None) -> pa.Array:
        """计算音频的信噪比（SNR，单位 dB）

        Args:
            audio_paths: 音频路径数组（支持本地与对象存储路径）
                - 本地路径：绝对路径或相对路径
                - 对象存储：以 "tos://" 或 "s3://" 开头的路径

        Returns:
            pa.Array: 浮点数组（float64），每行为对应音频的 SNR 值（单位 dB）。
        """  # noqa: D415
        if audio_paths is None or len(audio_paths) == 0:
            return pa.array([], type=self.__return_column_type__())

        paths_list = audio_paths.to_pylist()
        results: list[float] = []

        for p in paths_list:
            if not p or not isinstance(p, str):
                results.append(float("nan"))
                continue

            try:

                def process(local_path: str) -> float:
                    return self._calculate_snr_on_local_file(local_path)

                snr = run_on_local_path(p, process)
                results.append(float(snr))
            except Exception:
                logger.exception("Processing failed for path: %s", p)
                results.append(float("nan"))

        return pa.array(results, type=self.__return_column_type__())
