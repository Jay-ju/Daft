# Copyright (c) Beijing Volcano Engine Technology Ltd.

from __future__ import annotations

import logging
import math
import tempfile
import uuid
from pathlib import Path
from typing import Any

import ffmpeg

from daft.dependencies import pa
from daft.las.functions.types import Operator
from daft.las.functions.utils.common_utils import run_on_local_path, tracking_usage
from daft.las.functions.utils.video_utils import decode_video
from daft.las.io import mkdirs, upload_file
from daft.las.utils import not_blank

logger = logging.getLogger(__name__)


class VideoResizeResolution(Operator):
    """**视频分辨率调整**

    **核心功能：**
    - 智能分辨率调整到指定范围内
    - 支持多种宽高比保持策略
    - 可控制视频质量和编码参数
    - 保持音频流不受影响
    - 支持路径输入、二进制输入和TOS输出

    **格式支持：**
    - MP4 (.mp4)
    - AVI (.avi)
    - MOV (.mov)
    - MKV (.mkv)
    - 其他常见视频格式

    **编码模式：**
    - **CPU编码 (libx264)**：质量优先，压缩效率高，速度较慢
    - **GPU编码 (h264_nvenc)**：速度快，适合批量处理，质量略低于libx264（相同码率下），压缩效率稍差，文件可能更大

    """  # noqa: D415

    def __init__(
        self,
        output_tos_dir: str = "",
        min_width: int = 1280,
        max_width: int = 2560,
        min_height: int = 1280,
        max_height: int = 2560,
        force_original_aspect_ratio_type: str = "disable",
        force_divisible_by: int = 2,
        crf: float = 23.0,
        preset: str = "medium",
        cq: float = 0,
        rc: str = "vbr",
        rank: int | None = None,
        **kwargs: Any,
    ) -> None:
        """初始化视频分辨率调整算子。

        Args:
            output_tos_dir: 将分辨率调整后的视频保存到该TOS目录中，如果为空则不保存。
                格式："tos://bucket/path/"
                默认值：""
            min_width: 视频最小宽度，小于该值时将被调整。
                单位：像素
                默认值：1280
            max_width: 视频最大宽度，大于该值时将被调整。
                单位：像素
                默认值：2560
            min_height: 视频最小高度，小于该值时将被调整。
                单位：像素
                默认值：1280
            max_height: 视频最大高度，大于该值时将被调整。
                单位：像素
                默认值：2560
            force_original_aspect_ratio_type: 宽高比保持策略。
                disable: 不强制保持原始宽高比，可能会拉伸变形
                increase: 保持宽高比，调整到大于等于目标尺寸
                decrease: 保持宽高比，调整到小于等于目标尺寸
                可选值：["disable", "increase", "decrease"]
                默认值："disable"
            force_divisible_by: 像素对齐步长，确保宽高能被该值整除。
                默认值：2
            crf: libx264编码器的恒定质量因子。
                适用：仅在CPU编码(libx264)时使用
                范围：0.0-51.0，数值越小质量越高文件越大
                推荐：18(高质量) 23(平衡) 28(压缩优先)
                默认值：23.0
            preset: libx264编码器的编码速度预设。
                适用：仅在CPU编码(libx264)时使用
                权衡：速度 ↔ 压缩效率
                可选值：["ultrafast", "superfast", "veryfast", "faster", "fast", "medium", "slow", "slower", "veryslow"]
                推荐："medium"(平衡) "fast"(速度优先) "slow"(质量优先)
                默认值："medium"
            cq: NVENC编码器的质量控制参数。
                适用：仅在GPU编码(h264_nvenc)时使用
                范围：0-51，0表示自动质量控制
                推荐：0(自动) 或 18-28(手动控制)
                默认值：0
            rc: NVENC编码器的码率控制模式。
                适用：仅在GPU编码(h264_nvenc)时使用
                constqp: 恒定量化参数，质量稳定
                vbr: 变码率，平衡质量和文件大小
                cbr: 恒定码率，适合流媒体传输
                推荐："vbr"(通用) "cbr"(直播)
                默认值："vbr"
            rank: 指定使用的GPU设备编号（多卡环境有效）。
                说明：0表示第一张GPU，1表示第二张GPU，None表示自动选择
                适用：仅在GPU编码时有效
                默认值：None
            **kwargs: 其他参数，透传给父类。
        """  # noqa: D415
        super().__init__(**kwargs)

        if output_tos_dir:
            if not (isinstance(output_tos_dir, str) and output_tos_dir.startswith("tos://")):
                raise ValueError(
                    f"Invalid output_tos_dir: {output_tos_dir!r}, it should be a valid tos path starting with 'tos://'"
                )
            self.output_tos_dir = output_tos_dir.rstrip("/")
            mkdirs(self.output_tos_dir)
        else:
            self.output_tos_dir = ""

        self.min_width = min_width
        self.max_width = max_width
        self.min_height = min_height
        self.max_height = max_height
        self.force_original_aspect_ratio_type = force_original_aspect_ratio_type
        self.force_divisible_by = force_divisible_by
        self.crf = crf
        self.preset = preset
        self.cq = cq
        self.rc = rc
        self.rank = rank
        self.device_id: int | None = None

        try:
            import torch

            gpu_available = torch.cuda.is_available()
        except ImportError:
            gpu_available = False

        if self.use_gpu and gpu_available:
            if self.rank is None:
                self.device_id = 0
            else:
                try:
                    import torch

                    cuda_count = torch.cuda.device_count()
                    self.device_id = self.rank % cuda_count
                except ImportError:
                    self.device_id = 0

            self.video_codec = "h264_nvenc"
            logger.info("GPU encoding enabled: using %s on device %d", self.video_codec, self.device_id)
        else:
            self.video_codec = "libx264"
            if self.use_gpu:
                logger.warning("GPU requested but not available, falling back to CPU encoding")
            else:
                logger.info("CPU encoding enabled: using %s", self.video_codec)

        tracking_usage(op=self.__class__.__name__, model_service_or_lib="ffmpeg")

    def _resize_video(self, video_path: str, resized_video_path: str) -> str:
        container = decode_video(video_path)
        video = container.streams.video[0]
        width = video.codec_context.width
        height = video.codec_context.height
        logger.info("Origin video width: %d, height: %d", width, height)
        origin_ratio = width / height
        container.close()

        need_resize = not (self.min_width <= width <= self.max_width and self.min_height <= height <= self.max_height)

        if need_resize:
            if width < self.min_width:
                height = self.min_width / origin_ratio
                width = self.min_width
            if width > self.max_width:
                height = self.max_width / origin_ratio
                width = self.max_width
            if height < self.min_height:
                width = self.min_height * origin_ratio
                height = self.min_height
            if height > self.max_height:
                width = self.max_height * origin_ratio
                height = self.max_height

            if self.force_original_aspect_ratio_type == "disable":
                force_divisible_by = 2
            else:
                force_divisible_by = self.force_divisible_by

            width = int(max(width, self.min_width))
            width = math.ceil(width / force_divisible_by) * force_divisible_by
            width = int(min(width, self.max_width))
            width = int(width / force_divisible_by) * force_divisible_by
            height = int(max(height, self.min_height))
            height = math.ceil(height / force_divisible_by) * force_divisible_by
            height = int(min(height, self.max_height))
            height = int(height / force_divisible_by) * force_divisible_by

            if self.force_original_aspect_ratio_type == "increase":
                if width / height < origin_ratio:
                    width = height * origin_ratio
                elif width / height > origin_ratio:
                    height = width / origin_ratio
            elif self.force_original_aspect_ratio_type == "decrease":
                if width / height < origin_ratio:
                    height = width / origin_ratio
                elif width / height > origin_ratio:
                    width = height * origin_ratio
            width = round(width / force_divisible_by) * force_divisible_by
            height = round(height / force_divisible_by) * force_divisible_by

        args = ["-nostdin", "-v", "quiet", "-y"]
        try:
            input_stream = ffmpeg.input(video_path)

            if need_resize:
                video_stream = input_stream.video.filter("scale", width=width, height=height)
                logger.info("Resizing video: %s -> %s (%dx%d)", video_path, resized_video_path, width, height)
            else:
                video_stream = input_stream.video
                logger.info("Standardizing video without resize: %s -> %s", video_path, resized_video_path)

            audio_stream = input_stream.audio

            output_params: dict[str, Any] = {
                "vcodec": self.video_codec,
                "acodec": "aac",
                "strict": "-2",
            }

            if self.video_codec == "h264_nvenc":
                output_params.update(
                    {
                        "cq": self.cq,
                        "gpu": self.device_id,
                        "rc": self.rc,
                    }
                )
            else:
                output_params.update(
                    {
                        "crf": self.crf,
                        "preset": self.preset,
                    }
                )

            stream = (
                ffmpeg.output(video_stream, audio_stream, resized_video_path, **output_params)
                .global_args(*args)
                .overwrite_output()
            )

            _, stderr = ffmpeg.run(stream, capture_stdout=True, capture_stderr=True)
            if stderr:
                logger.error("FFmpeg stderr output: %s", stderr.decode("utf-8"))

        except ffmpeg.Error as e:
            logger.exception("FFmpeg failed with command: %s", e)
            raise

        logger.info("Resized video width: %d, height: %d", width, height)
        return resized_video_path

    def _process_video(
        self,
        video_path: str | None,
        video_binary: bytes | None,
        video_format: str | None,
        output_basename: str | None = None,
    ) -> str:
        is_valid_video_path = not_blank(video_path)
        if not is_valid_video_path and video_binary is None:
            return ""

        if not_blank(output_basename):
            video_name_base = output_basename
        elif is_valid_video_path:
            video_name_base = Path(str(video_path)).stem
        else:
            video_name_base = f"binary_{uuid.uuid4().hex}"

        try:
            if is_valid_video_path and video_path is not None:
                video_name = ".".join(Path(video_path).name.split(".")[:-1])
                video_type = Path(video_path).name.split(".")[-1]
                resized_video_name = f"{video_name}_resized.{video_type}"

                def process_with_path(local_path: str) -> str:
                    local_dir = Path(local_path).parent
                    resized_video_tmp_file = local_dir / resized_video_name
                    resized_video_path = self._resize_video(local_path, str(resized_video_tmp_file))
                    logger.info("Resized video file tmp path: %s", resized_video_path)

                    if self.output_tos_dir:
                        tos_path = f"{self.output_tos_dir}/{resized_video_name}"
                        upload_file(resized_video_path, tos_path)
                        return tos_path
                    else:
                        return resized_video_path

                return run_on_local_path(str(video_path), process_with_path)
            elif video_binary is not None:
                with tempfile.TemporaryDirectory(dir="/tmp") as temp_sub_dir:
                    temp_dir = temp_sub_dir.rstrip("/")
                    ext = f".{video_format.lower()}" if video_format else ".mp4"
                    temp_filename = f"{video_name_base}{ext}"
                    temp_filepath = Path(temp_dir) / temp_filename
                    with temp_filepath.open("wb") as tmp:
                        tmp.write(video_binary)

                    resized_video_name = f"{video_name_base}_resized{ext}"
                    resized_video_tmp_file = Path(temp_dir) / resized_video_name
                    resized_video_path = self._resize_video(str(temp_filepath), str(resized_video_tmp_file))
                    logger.info("Resized video file tmp path: %s", resized_video_path)

                    if self.output_tos_dir:
                        tos_path = f"{self.output_tos_dir}/{resized_video_name}"
                        upload_file(resized_video_path, tos_path)
                        return tos_path
                    else:
                        return resized_video_path
            else:
                raise ValueError("Neither path nor binary is valid")

        except Exception:
            logger.exception("Failed to resize video: %s or video_binary", video_path)
            return ""

    def transform(
        self,
        video_paths: pa.Array | None = None,
        video_binaries: pa.Array | None = None,
        video_formats: pa.Array | None = None,
        output_basenames: pa.Array | None = None,
    ) -> pa.Array:
        """调整视频分辨率到指定范围内，支持路径和二进制输入

        Args:
            video_paths: 视频文件路径列（本地、TOS、HTTP等），与video_binaries二选一
            video_binaries: 视频二进制数据列，与video_paths二选一
            video_formats: 视频格式字符串列，配合video_binaries使用
            output_basenames: 输出文件基础名称列（不含扩展名）

        Returns:
            调整后的视频路径列
        """  # noqa: D415
        n = 0
        if video_paths is not None:
            n = len(video_paths)
        elif video_binaries is not None:
            n = len(video_binaries)
        else:
            return pa.array([], type=self.__return_column_type__())

        paths_list = video_paths.to_pylist() if video_paths is not None else [None] * n
        binaries_list = video_binaries.to_pylist() if video_binaries is not None else [None] * n
        formats_list = video_formats.to_pylist() if video_formats is not None else [None] * n
        basenames_list = output_basenames.to_pylist() if output_basenames is not None else [None] * n

        resized_videos_list = []

        for video_path, binary, fmt, basename in zip(paths_list, binaries_list, formats_list, basenames_list):
            resized_path = self._process_video(video_path, binary, fmt, basename)
            resized_videos_list.append(resized_path)

        logger.info("Video resize finished.")
        return pa.array(resized_videos_list)

    @staticmethod
    def __return_column_type__() -> pa.DataType:
        return pa.string()
