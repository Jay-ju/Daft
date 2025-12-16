# Copyright (c) Beijing Volcano Engine Technology Ltd.

from __future__ import annotations

import os
import re
import subprocess
import tempfile
from typing import Any

from daft.dependencies import pa
from daft.las.functions.types import Operator
from daft.las.functions.utils.common_utils import FastWriteCounter, get_logger, tracking_usage
from daft.las.io import download_file


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

    @staticmethod
    def __return_column_type__() -> pa.DataType:
        return pa.bool_()

    def _detect_silence(self, audio: str | bytes | bytearray) -> bool | None:
        """检测音频是否为静音.

        Args:
            audio: 音频文件路径(支持本地或TOS路径) 或 bytes

        Returns:
            bool | None: True表示音频为静音，False表示音频包含有效声音，None表示无声道或处理失败
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
                return None

            if result is None:
                self.failed_counter.increment()
            else:
                self.success_counter.increment()
            self.log_progress()
            return result

        except Exception as e:
            self.failed_counter.increment()
            self.log_progress()
            self.logger.error("[AudioSilenceDetection] Failed to detect silence for input %s: %s", audio, e)
            return None
        finally:
            # 清理临时文件
            if tmp_file and os.path.exists(tmp_file.name):
                try:
                    os.remove(tmp_file.name)
                except Exception as remove_error:
                    self.logger.warning("Failed to remove temp file %s: %s", tmp_file.name, remove_error)

    def _analyze_audio_file(self, file_path: str) -> bool | None:
        """分析音频文件的静音状态.

        Args:
            file_path: 音频文件路径(远程)

        Returns:
            bool | None: True表示静音，False表示非静音，None表示处理失败或没有声道
        """
        tmp_in = None
        try:
            # 下载远程文件到本地临时文件
            tmp_in = tempfile.NamedTemporaryFile(suffix=os.path.splitext(file_path)[1] or ".tmp", delete=False)
            tmp_in.close()
            self.logger.info("Downloading remote file: %s -> %s", file_path, tmp_in.name)
            download_file(file_path, tmp_in.name)

            # 使用音量检测滤镜分析音量
            cmd = ["ffmpeg", "-i", tmp_in.name, "-af", "volumedetect", "-f", "null", "-"]

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
                self.logger.warning(
                    "Could not extract max_volume from audio analysis (no audio stream or processing failed)"
                )
                return None

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
            return None
        except Exception:
            self.logger.exception("Failed to analyze audio file volume: %s", file_path)
            return None
        finally:
            if tmp_in and os.path.exists(tmp_in.name):
                os.remove(tmp_in.name)

    def transform(self, audio_inputs: pa.Array) -> pa.Array:
        """检测音频文件是否为静音

        Args:
            audio_inputs: 存放音频路径或二进制的列

        Returns:
            pa.Array: 存放静音检测结果的布尔值列，True表示静音，False表示非静音，None表示处理失败或没有声道
        """  # noqa: D415
        results = []
        for audio_input in audio_inputs:
            try:
                is_silence = self._detect_silence(audio_input.as_py())
                results.append(is_silence)
            except Exception:
                self.logger.exception("Failed to process audio input: %s", audio_input)
                results.append(None)  # 出错时返回None

        return pa.array(results, type=self.__return_column_type__())
