# Copyright (c) Beijing Volcano Engine Technology Ltd.
from __future__ import annotations

import logging
import tempfile
import uuid
from pathlib import Path
from typing import Any

import ffmpeg

from daft.dependencies import pa
from daft.las.functions.types import Operator
from daft.las.functions.utils.common_utils import run_on_local_path, tracking_usage
from daft.las.io import mkdirs, upload_file

logger = logging.getLogger(__name__)


class VideoSplitByDuration(Operator):
    """**视频片段切分处理器，按固定时长智能分割**

    **核心功能：**
    - 按固定秒长切分视频
    - 支持剩余片段最小时长过滤
    - 支持片段二进制输出或TOS存储
    - 提供格式自动推断与自定义

    **格式支持：**
    - MP4 (.mp4)
    - AVI (.avi)
    - MOV (.mov)
    - MKV (.mkv)
    - 其他常见视频格式
    """  # noqa: D415

    def __init__(
        self,
        segment_duration: float = 5.0,
        min_segment_duration: float = 0.0,
        output_tos_dir: str = "",
        output_segments_binary: bool = False,
        output_video_format: str | None = None,
        **kwargs: Any,
    ) -> None:
        """初始化视频按时长切分算子

        Args:
            segment_duration: 每段时长（秒），必须 > 0
                默认值：5.0
            min_segment_duration: 剩余片段小于该值将被丢弃（秒）
                默认值：0.0
            output_tos_dir: 保存视频片段的TOS路径，若为空字符串则不上传
                默认值：""
            output_segments_binary: 是否输出视频片段的二进制数据
                默认值：False
            output_video_format: 全局指定输出视频格式（如"mp4"、"avi"等），优先级高于输入文件后缀和video_format列
        """  # noqa: D415
        super().__init__(**kwargs)
        if segment_duration <= 0:
            raise ValueError(f"segment_duration must be positive, got {segment_duration}")
        self.segment_duration = segment_duration
        self.min_segment_duration = min_segment_duration
        self.output_tos_dir = output_tos_dir.strip("/") if output_tos_dir else ""
        self.output_segments_binary = output_segments_binary
        self.output_video_format = output_video_format.lower() if output_video_format else None

        logger.info("Segment duration: %s seconds", self.segment_duration)
        logger.info("Min segment duration: %s seconds", self.min_segment_duration)
        logger.info("Output video format: %s", self.output_video_format)

        tracking_usage(op=self.__class__.__name__, model_service_or_lib="ffmpeg")

    def _get_output_extension(self, video_path: str | None, video_format: str | None) -> str:
        """Determine the output file extension."""
        if self.output_video_format:
            return f".{self.output_video_format}"
        if video_format:
            return f".{video_format.lower()}"
        if video_path:
            return Path(video_path).suffix.lower()
        logger.warning("No video format specified, defaulting to .mp4 for segment")
        return ".mp4"

    def _get_video_duration(self, video_path: str) -> float:
        """Get video duration in seconds."""
        try:
            probe = ffmpeg.probe(video_path)
            return float(probe["format"]["duration"])
        except (ffmpeg.Error, KeyError, ValueError):
            logger.exception("Failed to get duration for %s", video_path)
            return 0.0

    def _process_and_upload_segment(
        self,
        video_path: str,
        start_time: float,
        end_time: float,
        segment_path: str,
        tos_output_dir: str | None,
    ) -> str | None:
        """Cut segment and upload to TOS if needed."""
        try:
            output_kwargs: dict[str, Any] = {"strict": "experimental"}
            if not self.output_video_format:  # copy stream
                output_kwargs["vcodec"] = "copy"
                output_kwargs["acodec"] = "copy"
            (
                ffmpeg.input(video_path, ss=start_time, to=end_time)
                .output(segment_path, **output_kwargs)
                .overwrite_output()
                .run(quiet=True)
            )
        except ffmpeg.Error:
            logger.exception("FFmpeg error while cutting segment %s-%s", start_time, end_time)
            return None

        if tos_output_dir:
            tos_path = f"{tos_output_dir}/{Path(segment_path).name}"
            upload_file(segment_path, tos_path)
            return tos_path
        return segment_path

    def _split_video_by_duration(
        self,
        video_path: str,
        local_output_path: str,
        tos_output_dir: str | None,
        video_format: str | None,
    ) -> tuple[list[str], list[bytes]]:
        """Core logic: split video by fixed duration."""
        duration = self._get_video_duration(video_path)
        if duration <= 0:
            return [], []

        output_ext = self._get_output_extension(video_path, video_format).lstrip(".")
        segments: list[str] = []
        binaries: list[bytes] = []

        start_time = 0.0
        idx = 1
        while start_time < duration:
            end_time = min(start_time + self.segment_duration, duration)
            if end_time - start_time < self.min_segment_duration:
                break

            segment_path = f"{local_output_path}/segment_{idx}.{output_ext}"
            idx += 1

            segment_result = self._process_and_upload_segment(
                video_path, start_time, end_time, segment_path, tos_output_dir
            )
            if not segment_result:
                start_time = end_time
                continue

            segments.append(segment_result)

            if self.output_segments_binary:
                with Path(segment_path).open("rb") as f:
                    binaries.append(f.read())

            start_time = end_time

        return segments, binaries

    def _process_video(
        self,
        video_path: str | None,
        video_binary: bytes | None,
        video_format: str | None,
        output_basename: str | None = None,
    ) -> tuple[list[str], list[bytes]]:
        """Process one video_path and return (segments, binaries)."""
        from daft.las.utils import not_blank

        if not_blank(output_basename):
            video_sub_dir = str(output_basename)
        elif video_path:
            video_sub_dir = str(Path(video_path).stem)
        else:
            video_sub_dir = f"binary_{uuid.uuid4().hex}"

        if self.output_tos_dir:
            tos_output_dir = f"{self.output_tos_dir}/{video_sub_dir}"
            logger.info("Video segments tos output dir: %s", tos_output_dir)
            mkdirs(tos_output_dir)
        else:
            tos_output_dir = None

        try:
            if video_path is None and video_binary is not None:
                with tempfile.TemporaryDirectory(dir="/tmp") as temp_sub_dir:
                    temp_dir = temp_sub_dir.rstrip("/")
                    ext = self._get_output_extension(None, video_format)
                    temp_filename = f"{video_sub_dir}{ext}"
                    temp_filepath = Path(temp_dir) / temp_filename
                    with temp_filepath.open("wb") as tmp:
                        tmp.write(video_binary)

                    local_output_dir = Path(temp_dir) / str(video_sub_dir)
                    local_output_dir.mkdir(exist_ok=True)

                    return self._split_video_by_duration(
                        str(temp_filepath), str(local_output_dir), tos_output_dir, video_format
                    )
            elif video_path:

                def process(local_path: str) -> tuple[list[str], list[bytes]]:
                    local_output_dir = Path(local_path).parent / str(video_sub_dir)
                    local_output_dir.mkdir(exist_ok=True)
                    return self._split_video_by_duration(
                        local_path, str(local_output_dir), tos_output_dir, video_format
                    )

                return run_on_local_path(video_path, process)
            else:
                return [], []

        except (OSError, ffmpeg.Error):
            logger.exception("Failed to split video %s by duration", video_path)
            return [], []

    def transform(
        self,
        video_paths: pa.Array | None = None,
        video_binaries: pa.Array | None = None,
        video_formats: pa.Array | None = None,
        output_basenames: pa.Array | None = None,
    ) -> pa.Array:
        """将视频按固定时长切分为片段

        注意：`video_paths` 和 `video_binaries` 至少需要指定一个，否则返回空结果

        Args:
            video_paths: 包含输入视频路径的数组
                默认值：None
            video_binaries: 包含视频二进制数据的数组
                默认值：None
            video_formats: 包含输入视频格式（如 'mp4'、'avi' 等）的数组，指定 video_binaries 时可以提供格式信息
                默认值：None
            output_basenames: 可选，输出子目录名（文件名）数组

        Returns:
            pa.Array: 处理后的结构体字段包括：
                - segments: 切分后视频片段的路径列表
                - segments_binary: 切分后视频片段的二进制数据列表
                - video_format: 切分后视频片段的格式列表
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

        results = []
        for video_path, video_binary, video_format, basename in zip(
            paths_list, binaries_list, formats_list, basenames_list
        ):
            segments, binaries = self._process_video(video_path, video_binary, video_format, basename)
            results.append(
                {
                    "segments": segments,
                    "segments_binary": binaries if self.output_segments_binary else [],
                }
            )

        return pa.array(results, type=self.__return_column_type__())

    @staticmethod
    def __return_column_type__() -> pa.DataType:
        fields = [
            pa.field("segments", pa.list_(pa.string())),
            pa.field("segments_binary", pa.list_(pa.binary())),
        ]
        return pa.struct(fields)
