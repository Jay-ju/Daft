from __future__ import annotations

import os
import subprocess
import tempfile
from typing import Any

from daft.dependencies import pa
from daft.las.functions.types import Operator
from daft.las.functions.utils.common_utils import FastWriteCounter, get_logger, tracking_usage
from daft.las.io import download_file, upload_file


class AudioConvertToMp3(Operator):
    """**音频格式转换处理器，将各种音频格式转换为MP3**

    **核心功能：**
    - 支持多种音频格式转换为MP3
    - 音频质量和编码参数自定义
    - 支持音频采样率、比特率精细控制
    - 自动选择第一个音轨
    - 支持本地文件、HTTP/HTTPS URL和TOS/S3存储

    **格式支持：**
    - 输入：WAV、FLAC、AAC、M4A、OGG、WMA、APE等主流音频格式
    - 输出：MP3 (.mp3)
    - 音频编解码器：LAME MP3编码器
    - 采样率：8kHz-96kHz
    - 比特率：32kbps-320kbps
    """  # noqa: D415

    def __init__(
        self,
        bitrate: str = "192k",
        sample_rate: int | None = None,
        quality: int = 2,
        extra_params: list[str] | None = None,
        timeout: int | None = None,
        **kwargs: Any,
    ) -> None:
        """初始化音频转换为MP3算子

        Args:
            bitrate: 音频比特率，如"128k"、"192k"、"256k"、"320k"
                默认值："192k"
            sample_rate: 音频采样率，如22050、44100、48000，为None时保持原始采样率
                默认值：None
            quality: MP3编码质量，取值范围0-9，0最高质量最慢，9最低质量最快
                默认值：2
            extra_params: 额外的ffmpeg参数列表，如["-ac", "1"]
                默认值：None
            timeout: 单个音频处理超时时间（秒），为None时不限制
                默认值：None
        """  # noqa: D415
        super().__init__(**kwargs)
        self.bitrate = bitrate
        self.sample_rate = sample_rate
        self.quality = quality
        self.extra_params = extra_params or []
        self.timeout = timeout

        self.submit_counter = FastWriteCounter()
        self.success_counter = FastWriteCounter()
        self.failed_counter = FastWriteCounter()

        self.logger = get_logger(f"AudioConvertToMp3-{id(self)}")

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

    def process(self, input_path: str, output_path: str) -> str | None:
        """Process a single audio file for MP3 conversion.

        Args:
            input_path: Input audio file path (local path, HTTP/HTTPS URL, or TOS/S3 URL)
            output_path: Output audio file path

        Returns:
            str | None: Output path on success, None on failure
        """
        self.submit_counter.increment()

        tmp_input = None
        tmp_out = None
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                # Download input file to local temporary file
                tmp_input = tempfile.NamedTemporaryFile(
                    suffix=os.path.splitext(input_path)[1] or ".tmp", delete=False, dir=tmpdir
                )
                tmp_input.close()
                download_file(input_path, tmp_input.name)

                # Create temporary output file
                tmp_out = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False, dir=tmpdir)
                tmp_out.close()

                cmd = ["ffmpeg", "-y", "-i", tmp_input.name]

                # Select first audio track by default
                cmd += ["-map", "0:a:0"]

                # Audio encoder - use libmp3lame
                cmd += ["-c:a", "libmp3lame"]

                # Bitrate setting
                cmd += ["-b:a", self.bitrate]

                # MP3 encoding quality
                cmd += ["-q:a", str(self.quality)]

                # Sample rate setting
                if self.sample_rate is not None:
                    cmd += ["-ar", str(self.sample_rate)]

                # Extra parameters
                cmd += self.extra_params

                # Output settings
                cmd += ["-loglevel", "error", tmp_out.name]

                self.logger.info("Running command: %s", " ".join(cmd))
                subprocess.run(cmd, check=True, timeout=self.timeout)

                upload_file(tmp_out.name, output_path)

            self.success_counter.increment()
            self.log_progress()
            self.logger.info("Finished conversion %s → %s", input_path, output_path)
            return output_path

        except Exception as e:
            self.failed_counter.increment()
            self.log_progress()
            self.logger.error("Failed on %s: %s", input_path, e)
            return None
        finally:
            # Clean up temporary files if they exist outside tmpdir
            if tmp_input and os.path.exists(tmp_input.name):
                try:
                    os.remove(tmp_input.name)
                except Exception:
                    pass
            if tmp_out and os.path.exists(tmp_out.name):
                try:
                    os.remove(tmp_out.name)
                except Exception:
                    pass

    def transform(self, input_col: pa.Array, output_col: pa.Array) -> pa.Array:
        """将音频文件转换为MP3格式

        Args:
            input_col: 包含输入音频路径的数组（支持本地路径、HTTP/HTTPS URL、TOS/S3 URL）
            output_col: 包含输出MP3文件路径的数组

        Returns:
            pa.Array: 包含转换结果路径的数组，成功返回输出路径，失败返回None
        """  # noqa: D415
        results = []
        for input_path, output_path in zip(input_col.to_pylist(), output_col.to_pylist()):
            try:
                result = self.process(input_path, output_path)
            except Exception as e:
                self.logger.error("[transform] Failed to process %s: %s", input_path, e)
                result = None
            results.append(result)

        return pa.array(results, type=self.__return_column_type__())

    @staticmethod
    def __return_column_type__() -> pa.DataType:
        return pa.large_string()
