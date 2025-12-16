# Copyright (c) Beijing Volcano Engine Technology Ltd.

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from daft.dependencies import pa
from daft.las.functions.types import Operator
from daft.las.functions.utils.common_utils import FastWriteCounter, get_logger, tracking_usage
from daft.las.io import download_file, upload_file
from daft.las.utils import not_blank


class AudioConcat(Operator):
    """**音频拼接处理器，支持将多个音频文件拼接成一段音频**

    **核心功能**
    - 支持拼接多个音频文件为一个音频
    - 支持多输入格式：
       - 本地文件路径
       - TOS/S3存储路径
       - HTTP/HTTPS路径
    - 支持输出到指定路径
    - 自动处理不同采样率和声道的音频
    - 自动根据输出格式选择合适的编码器

    **格式支持**
    - WAV (.wav) - 使用 pcm_s16le 编码器
    - MP3 (.mp3) - 使用 libmp3lame 编码器
    - FLAC (.flac) - 使用 flac 编码器
    """  # noqa: D415

    def __init__(
        self,
        output_format: str = "mp3",
        sample_rate: int = 16000,
        timeout: int | None = None,
        extra_params: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        """
        初始化音频拼接处理器参数

        Args:
            output_format: 输出音频格式，仅支持 "wav", "mp3", "flac"，默认为 "mp3"
            sample_rate: 输出音频采样率，默认为 16000
                注意：由于使用 concat filter 进行拼接时必须重编码，所以采样率是必选参数
                常用采样率：8000, 16000, 22050, 44100, 48000
            timeout: ffmpeg执行超时时间（秒），默认为None（无超时）
            extra_params: 额外的ffmpeg参数列表，直接拼接到命令中
                例如：
                - 比特率: ["-b:a", "192k"]  # 适用于 MP3
                - 压缩级别: ["-compression_level", "8"]  # 适用于 FLAC
                默认值：None
            **kwargs: 其他参数

        Raises:
            ValueError: 如果 output_format 不是 "wav", "mp3" 或 "flac"
        """  # noqa: D212, D415
        super().__init__(**kwargs)

        # 支持的输出格式
        self.supported_formats = ("wav", "mp3", "flac")
        self.output_format = output_format.lstrip(".").lower()

        if self.output_format not in self.supported_formats:
            raise ValueError(f"output_format must be one of {self.supported_formats}, got '{output_format}'")

        # 根据输出格式自动选择编码器
        self.audio_codec = self._get_codec_for_format(self.output_format)
        self.sample_rate = sample_rate
        self.timeout = timeout
        self.extra_params = extra_params or []

        self.submit_counter = FastWriteCounter()
        self.success_counter = FastWriteCounter()
        self.failed_counter = FastWriteCounter()

        self.logger = get_logger(f"AudioConcat-{id(self)}")

        self.logger.info(
            "AudioConcat initialized with output_format=%s, codec=%s, sample_rate=%s, extra_params=%s",
            self.output_format,
            self.audio_codec,
            self.sample_rate,
            self.extra_params,
        )

        tracking_usage(op=self.__class__.__name__, model_service_or_lib="ffmpeg")

    def _get_codec_for_format(self, format: str) -> str:
        """根据输出格式选择合适的编码器."""
        format_codec_map = {
            "wav": "pcm_s16le",
            "mp3": "libmp3lame",
            "flac": "flac",
        }
        return format_codec_map[format]

    def log_progress(self) -> None:
        submitted = self.submit_counter.value
        succeed = self.success_counter.value
        failed = self.failed_counter.value
        finished = succeed + failed
        running = submitted - finished
        self.logger.info(
            "%s/%s running, finished/succeed/failed: %s/%s/%s", running, submitted, finished, succeed, failed
        )

    def _concat_audio_files(
        self,
        input_paths: list[str],
        output_path: str,
    ) -> bool:
        """Concatenate multiple audio files into one."""
        try:
            # 使用 ffmpeg concat filter 拼接音频
            # 构建 ffmpeg 命令
            cmd = ["ffmpeg", "-y"]

            # 添加所有输入文件
            for input_path in input_paths:
                cmd.extend(["-i", input_path])

            # 构建 filter_complex
            # 格式：[0:a][1:a][2:a]concat=n=3:v=0:a=1[out]
            n = len(input_paths)
            filter_inputs = "".join([f"[{i}:a]" for i in range(n)])
            filter_complex = f"{filter_inputs}concat=n={n}:v=0:a=1[out]"

            cmd.extend(["-filter_complex", filter_complex])
            cmd.extend(["-map", "[out]"])

            # 设置音频编码器
            cmd.extend(["-c:a", self.audio_codec])

            # 设置采样率（concat filter 重编码时必须）
            cmd.extend(["-ar", str(self.sample_rate)])

            # 用户自定义参数（如比特率、压缩级别等）
            cmd.extend(self.extra_params)

            # 输出文件
            cmd.extend(["-loglevel", "error", output_path])

            self.logger.info("[AudioConcat] Running ffmpeg command to concat %d audio files", n)
            subprocess.run(
                cmd,
                check=True,
                timeout=self.timeout,
                capture_output=True,
            )
            return True

        except subprocess.TimeoutExpired:
            self.logger.exception("FFmpeg timeout while concatenating audio files")
            return False
        except subprocess.CalledProcessError as e:
            self.logger.exception("FFmpeg error while concatenating audio files: %s", e.stderr)
            return False
        except Exception:
            self.logger.exception("Failed to concatenate audio files")
            return False

    def _process_audio_list(
        self,
        audio_paths: list[str],
        output_path: str,
    ) -> str | None:
        """Process a list of audio files and concatenate them.

        Args:
            audio_paths: List of audio file paths to concatenate
            output_path: Output file path

        Returns:
            Output path on success, None on failure
        """
        self.submit_counter.increment()

        if not audio_paths or len(audio_paths) == 0:
            self.logger.warning("No audio paths provided for concatenation")
            self.failed_counter.increment()
            self.log_progress()
            return None

        tmp_dir = None

        try:
            # 创建临时目录
            tmp_dir = tempfile.mkdtemp(dir="/tmp")

            # 下载所有远程文件到本地
            local_paths: list[str] = []
            for i, audio_path in enumerate(audio_paths):
                if not not_blank(audio_path):
                    self.logger.warning("Empty audio path at index %d, skipping", i)
                    continue

                # 下载文件
                suffix = Path(audio_path).suffix or ".tmp"
                tmp_file = os.path.join(tmp_dir, f"input_{i}{suffix}")

                self.logger.info("Downloading audio file %d: %s -> %s", i, audio_path, tmp_file)
                download_file(audio_path, tmp_file)
                local_paths.append(tmp_file)

            if len(local_paths) == 0:
                self.logger.warning("No valid audio files to concatenate")
                self.failed_counter.increment()
                self.log_progress()
                return None

            # 创建临时输出文件
            tmp_output = os.path.join(tmp_dir, f"output.{self.output_format}")

            # 拼接音频
            success = self._concat_audio_files(local_paths, tmp_output)
            if not success:
                self.logger.error("Failed to concatenate audio files")
                self.failed_counter.increment()
                self.log_progress()
                return None

            # 上传到指定路径
            upload_file(tmp_output, output_path)
            self.logger.info("Uploaded concatenated audio to: %s", output_path)

            self.success_counter.increment()
            self.log_progress()
            return output_path

        except Exception:
            self.logger.exception("Failed to process audio concatenation")
            self.failed_counter.increment()
            self.log_progress()
            return None
        finally:
            # 清理临时目录
            if tmp_dir:
                try:
                    shutil.rmtree(tmp_dir)
                except Exception:
                    self.logger.exception("Failed to clean up temporary directory: %s", tmp_dir)

    def transform(
        self,
        audio_paths_list: pa.Array,
        output_col: pa.Array,
    ) -> pa.Array:
        """批量拼接音频文件

        Args:
            audio_paths_list: 包含音频文件路径列表的列，每个元素是一个字符串列表
            output_col: 包含输出音频文件路径的数组

        Returns:
            包含拼接后音频文件路径的数组，成功返回输出路径，失败返回None
        """  # noqa: D415
        paths_lists = audio_paths_list.to_pylist()
        output_paths = output_col.to_pylist()

        results = []

        for paths, output_path in zip(paths_lists, output_paths):
            result = self._process_audio_list(paths, output_path)
            results.append(result)

        return pa.array(results, type=self.__return_column_type__())

    @staticmethod
    def __return_column_type__() -> pa.DataType:
        return pa.large_string()
