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


class AudioConcatFast(Operator):
    """**音频快速拼接处理器（同源音频）**

    **核心功能**
    - 使用 concat demuxer 快速拼接同源音频（无需重编码）
    - 适用于格式、编码、采样率完全相同的音频文件
    - 速度快，无质量损失
    - 支持多输入格式：
       - 本地文件路径
       - TOS/S3存储路径
       - HTTP/HTTPS路径
    - 支持输出到指定路径

    **使用场景**
    - 拼接同一录音分段
    - 拼接相同格式的音频片段
    - 需要快速拼接且无需格式转换

    **格式支持**
    - 支持所有音频格式（mp3, wav, flac, aac, m4a等）
    - 要求：所有输入音频必须具有相同的格式、编码、采样率、声道数

    **与 AudioConcat 的区别**
    - AudioConcat: 使用 concat filter，支持不同格式音频，需要重编码
    - AudioConcatFast: 使用 concat demuxer，仅支持同源音频，无需重编码，速度更快
    """  # noqa: D415

    def __init__(
        self,
        output_format: str = "mp3",
        timeout: int | None = None,
        **kwargs: Any,
    ) -> None:
        """
        初始化音频快速拼接处理器参数

        Args:
            output_format: 输出音频格式（仅用于确定文件扩展名），默认为 "mp3"
                注意：由于使用 concat demuxer，输出格式应与输入音频格式一致
            timeout: ffmpeg执行超时时间（秒），默认为None（无超时）
            **kwargs: 其他参数
        """  # noqa: D212, D415
        super().__init__(**kwargs)

        self.output_format = output_format.lstrip(".").lower()
        self.timeout = timeout

        self.submit_counter = FastWriteCounter()
        self.success_counter = FastWriteCounter()
        self.failed_counter = FastWriteCounter()

        self.logger = get_logger(f"AudioConcatFast-{id(self)}")

        self.logger.info(
            "AudioConcatFast initialized with output_format=%s",
            self.output_format,
        )

        tracking_usage(op=self.__class__.__name__, model_service_or_lib="ffmpeg")

    def log_progress(self) -> None:
        submitted = self.submit_counter.value
        succeed = self.success_counter.value
        failed = self.failed_counter.value
        finished = succeed + failed
        running = submitted - finished
        self.logger.info(
            "%s/%s running, finished/succeed/failed: %s/%s/%s", running, submitted, finished, succeed, failed
        )

    def _concat_audio_files_fast(
        self,
        input_paths: list[str],
        output_path: str,
    ) -> bool:
        """Concatenate audio files using concat demuxer (no re-encoding)."""
        list_file = None
        try:
            # 创建临时文件列表
            list_file = tempfile.NamedTemporaryFile(
                mode="w", suffix=".txt", delete=False, dir=os.path.dirname(input_paths[0])
            )

            # 写入文件列表（concat demuxer 格式）
            for input_path in input_paths:
                # 使用绝对路径，并转义单引号
                escaped_path = input_path.replace("'", "'\\''")
                list_file.write(f"file '{escaped_path}'\n")
            list_file.close()

            # 使用 concat demuxer 拼接音频
            # -f concat: 使用 concat demuxer
            # -safe 0: 允许使用任意路径
            # -i list.txt: 输入文件列表
            # -c copy: 不重编码，直接复制流
            cmd = [
                "ffmpeg",
                "-y",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                list_file.name,
                "-c",
                "copy",
                "-loglevel",
                "error",
                output_path,
            ]

            self.logger.info(
                "[AudioConcatFast] Running ffmpeg concat demuxer to concat %d audio files", len(input_paths)
            )
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
        finally:
            # 清理临时文件列表
            if list_file and os.path.exists(list_file.name):
                try:
                    os.remove(list_file.name)
                except Exception:
                    self.logger.warning("Failed to remove temp file list: %s", list_file.name)

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

            # 快速拼接音频（使用 concat demuxer）
            success = self._concat_audio_files_fast(local_paths, tmp_output)
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
        """批量快速拼接音频文件（同源音频）

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
