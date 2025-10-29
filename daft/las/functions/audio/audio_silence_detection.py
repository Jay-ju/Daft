# Copyright (c) Beijing Volcano Engine Technology Ltd.

from __future__ import annotations

import itertools
import logging
import os
import re
import subprocess
import tempfile
import threading
from typing import Any

from daft.dependencies import pa
from daft.las.functions.types import Operator
from daft.las.functions.utils.common_utils import is_local_path, pre_sign_url_for_tos, tracking_usage


class FastWriteCounter:
    def __init__(self, init: int = 0, step: int = 1) -> None:
        self._number_of_read = 0
        self._step = step
        self._counter = itertools.count(init, step)
        self._lock = threading.Lock()

    def increment(self) -> None:
        next(self._counter)

    @property
    def value(self) -> int:
        with self._lock:
            value = next(self._counter) - self._number_of_read
            self._number_of_read += self._step
        return value


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    handler.setFormatter(formatter)
    if not logger.handlers:
        logger.addHandler(handler)
    return logger


class AudioSilenceDetection(Operator):
    """**音频静音检测处理器，智能识别音频是否为完全静音**

    **核心功能：**
    - 静音检测：分析整个音频文件，判断是否为静音
    - 可配置阈值：支持自定义静音检测的敏感度阈值(dB)
    - 高效处理：优化的音量分析算法，处理速度快
    - 精确分析：通过专业音量检测技术判断静音状态
    - 支持本地文件、HTTP/HTTPS URL和TOS/S3存储

    **格式支持：**
    - 输入：MP3、WAV、FLAC、AAC、M4A、OGG等主流音频格式
    - 检测算法：基于专业音量分析技术
    - 采样率：自动适配各种采样率
    - 声道：支持单声道和多声道音频

    **检测原理：**
    - 音量阈值检测：分析音频的最大音量和平均音量
    - 静音判断：当最大音量低于设定阈值时，认为音频为静音
    - 无损分析：采用专业音频处理技术，保证分析精度
    """  # noqa: D415

    def __init__(
        self,
        silence_threshold_db: float = -60.0,
        timeout: int | None = None,
        **kwargs: Any,
    ) -> None:
        """初始化音频静音检测算子

        Args:
            silence_threshold_db: 静音音量阈值(dB)，当音频最大音量低于此值时认为是静音
                默认值：-60.0 (dB)
            timeout: 单个音频处理超时时间（秒），为None时不限制
                默认值：None
        """  # noqa: D415
        super().__init__(**kwargs)
        self.silence_threshold_db = silence_threshold_db
        self.timeout = timeout

        self.submit_counter = FastWriteCounter()
        self.success_counter = FastWriteCounter()
        self.failed_counter = FastWriteCounter()

        self.logger = get_logger(f"AudioSilenceDetection-{id(self)}")

        self.logger.info(
            "AudioSilenceDetection initialized with silence_threshold_db=%s, timeout=%s",
            silence_threshold_db,
            timeout,
        )

        tracking_usage(op=self.__class__.__name__, model_service_or_lib="audio_analysis")

    def log_progress(self) -> None:
        submitted = self.submit_counter.value
        succeed = self.success_counter.value
        failed = self.failed_counter.value
        finished = succeed + failed
        running = submitted - finished
        self.logger.info(
            "%s/%s running, finished/succeed/failed: %s/%s/%s", running, submitted, finished, succeed, failed
        )

    def get_input_path_for_ffmpeg(self, input_path: str) -> str:
        """Get the appropriate path for audio processing based on input type.

        Args:
            input_path: Input file path (local, HTTP/HTTPS URL, or TOS/S3 URL)

        Returns:
            str: Path that audio processor can use directly
        """
        if is_local_path(input_path):
            # Local file path - use as is
            return input_path
        elif input_path.startswith(("http://", "https://")):
            # HTTP/HTTPS URL - use as is
            return input_path
        elif input_path.startswith(("tos://", "s3://")):
            # TOS/S3 URL - need to pre-sign
            return pre_sign_url_for_tos(input_path, expires=360000)
        else:
            # Assume it's a remote path that needs pre-signing
            self.logger.warning("Unknown path type for %s, treating as TOS/S3", input_path)
            return pre_sign_url_for_tos(input_path, expires=360000)

    @staticmethod
    def __return_column_type__() -> pa.DataType:
        return pa.bool_()

    def _detect_silence(self, audio: str | bytes | bytearray) -> bool:
        """检测音频是否为静音.

        Args:
            audio: 音频文件路径(支持本地或TOS路径) 或 bytes

        Returns:
            bool: True表示音频为静音，False表示音频包含有效声音
        """
        self.submit_counter.increment()

        tmp_file = None
        try:
            if isinstance(audio, (bytes, bytearray)):
                # 将bytes写入临时文件供音频处理器处理
                tmp_file = tempfile.NamedTemporaryFile(suffix=".audio", delete=False)
                tmp_file.write(audio)
                tmp_file.close()
                result = self._analyze_audio_file(tmp_file.name)
            elif isinstance(audio, str):
                result = self._analyze_audio_file(audio)
            else:
                self.logger.warning("Unsupported audio input type: %s", type(audio))
                self.failed_counter.increment()
                self.log_progress()
                return False

            self.success_counter.increment()
            self.log_progress()
            return result

        except Exception as e:
            self.failed_counter.increment()
            self.log_progress()
            self.logger.error("[AudioSilenceDetection] Failed to detect silence for input %s: %s", audio, e)
            return False
        finally:
            # 清理临时文件
            if tmp_file and os.path.exists(tmp_file.name):
                try:
                    os.remove(tmp_file.name)
                except Exception as remove_error:
                    self.logger.warning("Failed to remove temp file %s: %s", tmp_file.name, remove_error)

    def _analyze_audio_file(self, file_path: str) -> bool:
        """分析音频文件的静音状态.

        Args:
            file_path: 音频文件路径

        Returns:
            bool: True表示静音，False表示非静音
        """
        try:
            url_path = self.get_input_path_for_ffmpeg(file_path)

            # 使用音量检测滤镜分析音量
            cmd = ["ffmpeg", "-i", url_path, "-af", "volumedetect", "-f", "null", "-"]

            self.logger.info("[AudioSilenceDetection] Running audio volume analysis: %s", " ".join(cmd))
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=self.timeout)

            # 从stderr中解析音量分析的输出
            stderr_output = result.stderr

            # 解析max_volume值
            max_volume = None
            mean_volume = None

            for line in stderr_output.split("\n"):
                if "max_volume:" in line:
                    match = re.search(r"max_volume:\s*([\-\d\.]+)\s*dB", line)
                    if match:
                        max_volume = float(match.group(1))
                elif "mean_volume:" in line:
                    match = re.search(r"mean_volume:\s*([\-\d\.]+)\s*dB", line)
                    if match:
                        mean_volume = float(match.group(1))

            # 判断是否为静音
            if max_volume is None:
                self.logger.warning("Could not extract max_volume from audio analysis, treating as non-silent")
                return False

            is_silence = max_volume <= self.silence_threshold_db

            self.logger.debug(
                "Audio volume analysis: max_volume=%.2f dB, mean_volume=%s dB, " "threshold=%.2f dB, is_silence=%s",
                max_volume,
                mean_volume if mean_volume is not None else "N/A",
                self.silence_threshold_db,
                is_silence,
            )

            return is_silence

        except subprocess.TimeoutExpired:
            self.logger.error("Audio volume analysis timed out for file: %s", file_path)
            return False
        except Exception:
            self.logger.exception("Failed to analyze audio file volume: %s", file_path)
            return False

    def transform(self, audio_inputs: pa.Array) -> pa.Array:
        """检测音频文件是否为静音

        Args:
            audio_inputs: 存放音频路径或二进制的列

        Returns:
            pa.Array: 存放静音检测结果的布尔值列，True表示静音，False表示非静音
        """  # noqa: D415
        results = []
        for audio_input in audio_inputs:
            try:
                is_silence = self._detect_silence(audio_input.as_py())
                results.append(is_silence)
            except Exception:
                self.logger.exception("Failed to process audio input: %s", audio_input)
                results.append(False)  # 出错时默认认为非静音

        return pa.array(results, type=self.__return_column_type__())
