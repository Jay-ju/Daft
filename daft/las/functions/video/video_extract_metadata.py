from __future__ import annotations

import json
import subprocess
from typing import Any
from urllib.parse import urlparse

from daft.dependencies import pa
from daft.las.functions.text.pre_sign_url_for_tos import PreSignUrlForTos
from daft.las.functions.types import Operator
from daft.las.functions.utils.common_utils import FastWriteCounter, get_logger, tracking_usage


class VideoExtractMetadata(Operator):
    """视频元数据提取算子.

    功能:
    1. 支持视频文件格式
    2. 提取完整的视频元数据信息,包括:
       - 基础信息: 时长、格式、比特率
       - 视频信息: 分辨率、帧率、编码器
       - 音频信息: 采样率、声道数、编码器
    3. 使用 ffprobe 工具进行元数据提取
    4. 支持本地文件、远程文件(TOS/S3)、HTTP/HTTPS链接

    参数:
        timeout (int, optional): ffprobe 命令执行超时时间(秒),默认为 None (无超时)

    返回:
        PyArrow Struct 包含以下字段:
        - duration (float): 视频时长(秒)
        - format_name (string): 格式名称
        - bit_rate (int64): 比特率
        - has_video (bool): 是否包含视频流
        - has_audio (bool): 是否包含音频流
        - video_codec (string): 视频编码器名称
        - video_width (int32): 视频宽度
        - video_height (int32): 视频高度
        - video_fps (float): 视频帧率
        - audio_codec (string): 音频编码器名称
        - audio_sample_rate (int32): 音频采样率
        - audio_channels (int32): 音频声道数
    """

    def __init__(self, timeout: int | None = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.timeout = timeout

        # 初始化预签名 URL 生成器 (用于 TOS 路径)
        self.pre_signer = None
        try:
            self.pre_signer = PreSignUrlForTos(expires=3600)
        except Exception as e:
            self.logger = get_logger(f"VideoExtractMetadata-{id(self)}")
            self.logger.warning("Failed to initialize PreSignUrlForTos, will fallback to download: %s", e)

        self.submit_counter = FastWriteCounter()
        self.success_counter = FastWriteCounter()
        self.failed_counter = FastWriteCounter()

        self.logger = get_logger(f"VideoExtractMetadata-{id(self)}")
        self.logger.info("VideoExtractMetadata initialized with timeout=%s", self.timeout)
        tracking_usage(op=self.__class__.__name__, model_service_or_lib="ffprobe")

    def log_progress(self) -> None:
        """记录处理进度."""
        submitted = self.submit_counter.value
        succeed = self.success_counter.value
        failed = self.failed_counter.value
        finished = succeed + failed
        running = submitted - finished
        self.logger.info(
            "%s/%s running, finished/succeed/failed: %s/%s/%s",
            running,
            submitted,
            finished,
            succeed,
            failed,
        )

    def _get_processable_url(self, input_path: str) -> str | None:
        """获取可直接处理的 URL.

        对于 TOS 路径,生成预签名 URL 以避免下载整个文件
        对于 HTTP/HTTPS 路径,直接返回
        对于本地路径,直接返回

        Args:
            input_path: 输入文件路径

        Returns:
            可处理的 URL 或路径,如果失败则返回 None
        """
        parsed_url = urlparse(input_path)
        schema = parsed_url.scheme

        # TOS/S3 路径需要转换为预签名 URL
        if schema in ("tos", "s3"):
            if self.pre_signer is None:
                self.logger.error("PreSignUrlForTos not available for TOS path: %s", input_path)
                return None

            try:
                signed_url = self.pre_signer._get_pre_signed_policy_url(input_path)
                if signed_url:
                    self.logger.debug("Generated pre-signed URL for: %s", input_path)
                    return signed_url
                else:
                    self.logger.error("Failed to generate pre-signed URL for: %s", input_path)
                    return None
            except Exception as e:
                self.logger.error("Error generating pre-signed URL for %s: %s", input_path, e)
                return None

        # HTTP/HTTPS 或本地路径直接返回
        return input_path

    def extract_metadata(self, input_path: str) -> dict[str, Any] | None:
        """使用 ffprobe 提取单个视频文件的元数据.

        对于 TOS 路径,会先生成预签名 URL,然后直接传给 ffprobe 处理,
        避免下载整个文件,只读取元数据信息

        Args:
            input_path: 输入文件路径(支持本地路径、TOS 路径、HTTP/HTTPS 链接)

        Returns:
            包含元数据的字典,如果失败则返回 None
        """
        self.submit_counter.increment()

        try:
            # 获取可处理的 URL
            processable_url = self._get_processable_url(input_path)
            if processable_url is None:
                self.failed_counter.increment()
                self.log_progress()
                return None

            # 使用 ffprobe 直接从 URL 提取元数据,输出为 JSON 格式
            cmd = [
                "ffprobe",
                "-v",
                "error",
                "-show_format",
                "-show_streams",
                "-of",
                "json",
                processable_url,
            ]

            result = subprocess.run(
                cmd,
                check=True,
                timeout=self.timeout,
                capture_output=True,
                text=True,
            )

            # 解析 JSON 输出
            probe_data = json.loads(result.stdout)

            # 提取格式信息
            format_info = probe_data.get("format", {})
            duration = float(format_info.get("duration", 0.0))
            format_name = format_info.get("format_name", "")
            bit_rate = int(format_info.get("bit_rate", 0))

            # 提取视频流信息
            video_stream = None
            audio_stream = None
            for stream in probe_data.get("streams", []):
                if stream.get("codec_type") == "video" and video_stream is None:
                    video_stream = stream
                elif stream.get("codec_type") == "audio" and audio_stream is None:
                    audio_stream = stream

            # 构建返回的元数据字典
            metadata = {
                "duration": duration,
                "format_name": format_name,
                "bit_rate": bit_rate,
                "has_video": video_stream is not None,
                "has_audio": audio_stream is not None,
            }

            # 视频信息
            if video_stream:
                metadata["video_codec"] = video_stream.get("codec_name", "")
                metadata["video_width"] = int(video_stream.get("width", 0))
                metadata["video_height"] = int(video_stream.get("height", 0))

                # 计算帧率
                fps_str = video_stream.get("r_frame_rate", "0/1")
                try:
                    num, denom = fps_str.split("/")
                    fps = float(num) / float(denom) if float(denom) != 0 else 0.0
                except (ValueError, ZeroDivisionError):
                    fps = 0.0
                metadata["video_fps"] = fps
            else:
                metadata["video_codec"] = None
                metadata["video_width"] = None
                metadata["video_height"] = None
                metadata["video_fps"] = None

            # 音频信息
            if audio_stream:
                metadata["audio_codec"] = audio_stream.get("codec_name", "")
                metadata["audio_sample_rate"] = int(audio_stream.get("sample_rate", 0))
                metadata["audio_channels"] = int(audio_stream.get("channels", 0))
            else:
                metadata["audio_codec"] = None
                metadata["audio_sample_rate"] = None
                metadata["audio_channels"] = None

            self.success_counter.increment()
            self.log_progress()
            return metadata

        except subprocess.TimeoutExpired:
            self.failed_counter.increment()
            self.log_progress()
            self.logger.error("Timeout on %s after %s seconds", input_path, self.timeout)
            return None
        except Exception as e:
            self.failed_counter.increment()
            self.log_progress()
            self.logger.error("Failed to extract metadata from %s: %s", input_path, e)
            return None

    def transform(self, input_col: pa.Array) -> pa.Array:
        """批量处理多个视频文件.

        Args:
            input_col: 输入文件路径的 PyArrow Array

        Returns:
            包含元数据的 PyArrow Struct Array
        """
        results = []
        for input_path in input_col.to_pylist():
            try:
                metadata = self.extract_metadata(input_path)
            except Exception as e:
                self.logger.error("[transform] Failed on %s: %s", input_path, e)
                metadata = None

            results.append(metadata)

        return pa.array(results, type=self.__return_column_type__())

    @staticmethod
    def __return_column_type__() -> pa.DataType:
        """返回输出列的 PyArrow 数据类型."""
        return pa.struct(
            [
                pa.field("duration", pa.float64()),
                pa.field("format_name", pa.string()),
                pa.field("bit_rate", pa.int64()),
                pa.field("has_video", pa.bool_()),
                pa.field("has_audio", pa.bool_()),
                pa.field("video_codec", pa.string()),
                pa.field("video_width", pa.int32()),
                pa.field("video_height", pa.int32()),
                pa.field("video_fps", pa.float64()),
                pa.field("audio_codec", pa.string()),
                pa.field("audio_sample_rate", pa.int32()),
                pa.field("audio_channels", pa.int32()),
            ]
        )
