from __future__ import annotations

import os
import subprocess
import tempfile
from typing import Any

from daft.dependencies import pa
from daft.las.functions.types import Operator
from daft.las.functions.utils.common_utils import FastWriteCounter, get_logger, tracking_usage
from daft.las.io import download_file, upload_file


class VideoConvertToMp4(Operator):
    """**视频格式转换处理器，将各种视频格式转换为MP4**

    **核心功能：**
    - 支持多种视频格式转换为MP4
    - 自动选择第一个音轨
    - 视频质量和编码参数自定义
    - 支持视频高度限制和缩放
    - 音频编码参数精细控制

    **格式支持：**
    - 输入：AVI、MOV、MKV、FLV、WMV、3GP等主流视频格式
    - 输出：MP4 (.mp4)
    - 视频编解码器：H.264/H.265等
    - 音频编解码器：AAC、MP3等
    """  # noqa: D415

    def __init__(
        self,
        video_codec: str = "libx264",
        crf: int = 23,
        preset: str = "medium",
        max_height: int | None = None,
        audio_codec: str = "aac",
        audio_bitrate: str = "192k",
        audio_sample_rate: int | None = None,
        extra_params: list[str] | None = None,
        timeout: int | None = None,
        **kwargs: Any,
    ) -> None:
        """初始化视频转换为MP4算子

        Args:
            video_codec: 视频编码器，支持libx264、libx265等
                默认值："libx264"
            crf: 视频质量控制，取值范围0-51，越小质量越好
                默认值：23
            preset: 编码速度预设，支持ultrafast、superfast、veryfast、faster、fast、medium、slow、slower、veryslow
                默认值："medium"
            max_height: 视频最大高度限制，超过时自动缩放，为None时不限制
                默认值：None
            audio_codec: 音频编码器，支持aac等
                默认值："aac"
            audio_bitrate: 音频码率，如"192k"、"128k"
                默认值："192k"
            audio_sample_rate: 音频采样率，如44100。48000，为None时保持原始采样率
                默认值：None
            extra_params: 额外的ffmpeg参数列表，如["-movflags", "+faststart"]
                默认值：None
            timeout: 单个视频处理超时时间（秒），为None时不限制
                默认值：None
        """  # noqa: D415
        super().__init__(**kwargs)
        self.video_codec = video_codec
        self.crf = crf
        self.preset = preset
        self.max_height = max_height
        self.audio_codec = audio_codec
        self.audio_bitrate = audio_bitrate
        self.audio_sample_rate = audio_sample_rate
        self.extra_params = extra_params or []
        self.timeout = timeout

        self.submit_counter = FastWriteCounter()
        self.success_counter = FastWriteCounter()
        self.failed_counter = FastWriteCounter()

        self.logger = get_logger(f"VideoConvertToMp4-{id(self)}")

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
        """Process a single video file for MP4 conversion.

        Args:
            input_path: Input video file path (local path, HTTP/HTTPS URL, or TOS/S3 URL)
            output_path: Output video file path

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
                tmp_out = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False, dir=tmpdir)
                tmp_out.close()

                cmd = ["ffmpeg", "-y", "-i", tmp_input.name, "-map", "0:v:0"]

                # Video parameters
                cmd += ["-c:v", self.video_codec]
                if self.crf is not None:
                    cmd += ["-crf", str(self.crf)]
                if self.preset is not None:
                    cmd += ["-preset", self.preset]
                if self.max_height is not None and self.max_height > 0:
                    cmd += ["-vf", f"scale=-2:min(ih\\,{self.max_height})"]

                # Audio parameters
                cmd += ["-c:a", self.audio_codec]
                if self.audio_bitrate is not None:
                    cmd += ["-b:a", self.audio_bitrate]
                if self.audio_sample_rate is not None:
                    cmd += ["-ar", str(self.audio_sample_rate)]

                # Select first audio track by default
                cmd += ["-map", "0:a:0"]

                # Extra parameters
                cmd += self.extra_params

                # Output and optimization
                cmd += ["-movflags", "+faststart", "-loglevel", "error", tmp_out.name]

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
        """将视频文件转换为MP4格式

        Args:
            input_col: 包含输入视频路径的数组（支持本地路径、HTTP/HTTPS URL、TOS/S3 URL）
            output_col: 包含输出MP4文件路径的数组

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
