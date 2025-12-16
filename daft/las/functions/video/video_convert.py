from __future__ import annotations

import os
import subprocess
import tempfile
from typing import Any

from daft.dependencies import pa
from daft.las.functions.types import Operator
from daft.las.functions.utils.common_utils import FastWriteCounter, get_logger, tracking_usage
from daft.las.io import download_file, upload_file


class VideoConvert(Operator):
    """**通用视频格式转换处理器**

    **核心功能：**
    - 支持多种视频格式之间的转换
    - 自动选择合适的编码器
    - 通过extra_params支持自定义ffmpeg参数

    **格式支持：**
    - 输入：AVI、MOV、MKV、FLV、WMV、3GP、MP4等主流视频格式
    - 输出：MP4、AVI、MOV、MKV、FLV、WEBM等
    - 视频编解码器：H.264、H.265、VP8、VP9等
    - 音频编解码器：AAC、MP3、Opus等
    """  # noqa: D415

    def __init__(
        self,
        output_format: str,
        timeout: int | None = None,
        extra_params: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        """初始化通用视频转换算子

        Args:
            output_format: 输出视频格式，支持 "mp4", "avi", "mov", "mkv", "flv", "webm"
            timeout: ffmpeg 执行超时时间（秒），默认为 None（无超时）
            extra_params: 额外的 ffmpeg 参数列表，直接拼接到命令中
                例如：
                - 视频质量: ["-crf", "23"]
                - 视频码率: ["-b:v", "2M"]
                - 编码预设: ["-preset", "medium"]
                - 视频缩放: ["-vf", "scale=-2:720"]
                - 音频码率: ["-b:a", "192k"]
                - 音频采样率: ["-ar", "48000"]
                - 特定编码器: ["-c:v", "libx265", "-c:a", "aac"]
            **kwargs: 其他参数

        Raises:
            ValueError: 如果 output_format 不在支持的格式列表中
        """  # noqa: D415
        super().__init__(**kwargs)

        # 支持的输出格式
        self.supported_formats = ("mp4", "avi", "mov", "mkv", "flv", "webm")
        self.output_format = output_format.lstrip(".").lower()

        if self.output_format not in self.supported_formats:
            raise ValueError(f"output_format must be one of {self.supported_formats}, got '{output_format}'")

        # 根据输出格式自动选择编码器
        self.video_codec, self.audio_codec = self._get_codecs_for_format(self.output_format)
        self.timeout = timeout
        self.extra_params = extra_params or []

        self.submit_counter = FastWriteCounter()
        self.success_counter = FastWriteCounter()
        self.failed_counter = FastWriteCounter()

        self.logger = get_logger(f"VideoConvert-{id(self)}")

        tracking_usage(op=self.__class__.__name__, model_service_or_lib="ffmpeg")

    def _get_codecs_for_format(self, format: str) -> tuple[str, str]:
        """根据输出格式选择合适的视频和音频编码器.

        Returns:
            tuple[str, str]: (video_codec, audio_codec)
        """
        format_codec_map = {
            "mp4": ("libx264", "aac"),
            "avi": ("libx264", "mp3"),
            "mov": ("libx264", "aac"),
            "mkv": ("libx264", "aac"),
            "flv": ("libx264", "aac"),
            "webm": ("libvpx-vp9", "libopus"),
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

    def process(self, input_path: str, output_path: str) -> str | None:
        """Process a single video file for format conversion.

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
                tmp_out = tempfile.NamedTemporaryFile(suffix=f".{self.output_format}", delete=False, dir=tmpdir)
                tmp_out.close()

                cmd = ["ffmpeg", "-y", "-i", tmp_input.name]

                # Select first video and audio track by default
                cmd += ["-map", "0:v:0", "-map", "0:a:0"]

                # Set video and audio codec
                cmd += ["-c:v", self.video_codec, "-c:a", self.audio_codec]

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
        """将视频文件转换为指定格式

        Args:
            input_col: 包含输入视频路径的数组（支持本地路径、HTTP/HTTPS URL、TOS/S3 URL）
            output_col: 包含输出视频文件路径的数组

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
