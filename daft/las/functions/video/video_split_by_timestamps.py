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


class VideoSplitByTimestamps(Operator):
    """**视频时间戳切分处理器，支持按指定时间范围分割**

    **核心功能：**
    - 按给定的时间戳区间切分视频
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
        output_tos_dir: str = "",
        output_segments_binary: bool = False,
        output_video_format: str | None = None,
        **kwargs: Any,
    ) -> None:
        """初始化视频按时间戳切分算子

        Args:
            output_tos_dir: 保存视频片段的TOS路径，若为空字符串则不上传
                默认值：""
            output_segments_binary: 是否输出视频片段的二进制数据
                默认值：False
            output_video_format: 全局指定输出视频格式（如"mp4"、"avi"等），优先级高于输入文件后缀和video_format列
            **kwargs: 其他参数，透传给父类
        """  # noqa: D415
        super().__init__(**kwargs)
        self.output_tos_dir = output_tos_dir.strip("/") if output_tos_dir else ""
        self.output_segments_binary = output_segments_binary
        self.output_video_format = output_video_format.lower() if output_video_format else None

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

    def _normalize_timestamp_ranges(self, timestamp_ranges: Any) -> list[tuple[float, float]]:
        """Normalize timestamp_ranges input to a list of (start, end) float tuples.

        Supported input formats:
        - List of tuple/list: [(start1, end1), (start2, end2)] or [[start1, end1], [start2, end2]]
        - List of struct/dict (Arrow to_pylist): [{"_0": start1, "_1": end1}, {"_0": start2, "_1": end2}]
          Also supports keys "start"/"end" or 0/1.
        - Single range: (start, end) or [start, end] or {"_0": start, "_1": end}
        """
        if timestamp_ranges is None:
            return []

        def _is_number(x: Any) -> bool:
            return isinstance(x, (int, float))

        def _to_pair(t: Any) -> tuple[float, float]:
            if isinstance(t, (tuple, list)) and len(t) == 2:
                return float(t[0]), float(t[1])
            raise ValueError("Invalid range element, expected sequence of length 2")

        def _from_mapping(m: dict[Any, Any]) -> tuple[float, float]:
            # Accept Arrow-style {"_0": start, "_1": end}, or {"start": ..., "end": ...}, or {0: ..., 1: ...}
            if "_0" in m and "_1" in m:
                return float(m["_0"]), float(m["_1"])
            if "start" in m and "end" in m:
                return float(m["start"]), float(m["end"])
            if 0 in m and 1 in m:
                return float(m[0]), float(m[1])
            raise ValueError("Invalid mapping keys for range; expected ('_0','_1') or ('start','end') or (0,1)")

        # List inputs
        if isinstance(timestamp_ranges, list):
            if not timestamp_ranges:
                return []
            first = timestamp_ranges[0]

            # Case: list of tuple/list
            if isinstance(first, (tuple, list)):
                return [_to_pair(item) for item in timestamp_ranges]

            # Case: list of dicts (Arrow struct to_pylist produces dicts)
            if isinstance(first, dict):
                return [_from_mapping(item) for item in timestamp_ranges]

            # Case: flat list of two numbers -> interpret as a single range
            if len(timestamp_ranges) == 2 and all(_is_number(x) for x in timestamp_ranges):
                return [(float(timestamp_ranges[0]), float(timestamp_ranges[1]))]

        # Single tuple/list
        if isinstance(timestamp_ranges, tuple) and len(timestamp_ranges) == 2:
            return [(float(timestamp_ranges[0]), float(timestamp_ranges[1]))]

        # Single dict
        if isinstance(timestamp_ranges, dict):
            return [_from_mapping(timestamp_ranges)]

        raise ValueError(
            "timestamp_ranges 必须为以下格式之一：\n"
            "- 单个范围：(start,end) 或 [start,end] 或 {'_0':start,'_1':end}\n"
            "- 多个范围：[(start,end), (start,end)] 或 [[start,end], [start,end]] 或 "
            "[{'_0':start,'_1':end}, {'_0':start,'_1':end}]"
        )

    def _process_and_upload_segment(
        self,
        video_path: str,
        start_time: float,
        end_time: float,
        segment_path: str,
        tos_output_dir: str | None,
        dir_created_flag: dict[str, bool] | None = None,
    ) -> str | None:
        try:
            output_kwargs: dict[str, Any] = {"strict": "experimental"}
            input_ext = Path(video_path).suffix.lower().lstrip(".")
            if not self.output_video_format:
                output_kwargs["vcodec"] = "copy"
                output_kwargs["acodec"] = "copy"
            elif self.output_video_format and input_ext == self.output_video_format:
                output_kwargs["vcodec"] = "copy"
                output_kwargs["acodec"] = "copy"
            (
                ffmpeg.input(video_path, ss=start_time, to=end_time)
                .output(segment_path, **output_kwargs)
                .overwrite_output()
                .run(quiet=True)
            )
        except ffmpeg.Error:
            logger.exception("FFmpeg error while splitting segment %s-%s", start_time, end_time)
            return None

        if tos_output_dir:
            if dir_created_flag is not None and not dir_created_flag.get("created", False):
                mkdirs(tos_output_dir)
                dir_created_flag["created"] = True
            tos_path = f"{tos_output_dir}/{Path(segment_path).name}"
            upload_file(segment_path, tos_path)
            return tos_path
        return segment_path

    def _split_video_by_timestamps(
        self,
        video_path: str,
        local_output_path: str,
        tos_output_dir: str | None,
        timestamp_ranges: list[tuple[float, float]],
        video_format: str | None,
    ) -> tuple[list[str], list[bytes]]:
        output_ext = self._get_output_extension(video_path, video_format).lstrip(".")
        segments: list[str] = []
        binaries: list[bytes] = []

        dir_created_flag = {"created": False}
        for start_time, end_time in timestamp_ranges:
            start_str = f"{start_time}".replace(".", "_")
            end_str = f"{end_time}".replace(".", "_")
            segment_path = f"{local_output_path}/segment_{start_str}-{end_str}.{output_ext}"

            segment_result = self._process_and_upload_segment(
                video_path, start_time, end_time, segment_path, tos_output_dir, dir_created_flag
            )
            if not segment_result:
                continue

            segments.append(segment_result)
            if self.output_segments_binary:
                with Path(segment_path).open("rb") as f:
                    binaries.append(f.read())

        return segments, binaries

    def _process_video(
        self,
        video_path: str | None,
        video_binary: bytes | None,
        video_format: str | None,
        timestamp_ranges: list[tuple[float, float]],
        output_basename: str | None = None,
    ) -> tuple[list[str], list[bytes]]:
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
        else:
            tos_output_dir = None

        try:
            if video_path is None and video_binary is not None:
                if len(video_binary) < 128:
                    logger.warning("Empty or too small binary input, skip.")
                    # 切分失败时返回空字符串列表，保证 explode 后每行至少有一个元素
                    return [""], [b""]
                with tempfile.TemporaryDirectory(dir="/tmp") as temp_sub_dir:
                    temp_dir = temp_sub_dir.rstrip("/")
                    ext = self._get_output_extension(None, video_format)
                    temp_filename = f"{video_sub_dir}{ext}"
                    temp_filepath = Path(temp_dir) / temp_filename
                    with temp_filepath.open("wb") as tmp:
                        tmp.write(video_binary)

                    local_output_dir = Path(temp_dir) / str(video_sub_dir)
                    local_output_dir.mkdir(exist_ok=True)

                    return self._split_video_by_timestamps(
                        str(temp_filepath), str(local_output_dir), tos_output_dir, timestamp_ranges, video_format
                    )
            elif video_path:

                def process(local_path: str) -> tuple[list[str], list[bytes]]:
                    local_output_dir = Path(local_path).parent / str(video_sub_dir)
                    local_output_dir.mkdir(exist_ok=True)
                    return self._split_video_by_timestamps(
                        local_path, str(local_output_dir), tos_output_dir, timestamp_ranges, video_format
                    )

                return run_on_local_path(video_path, process)
            else:
                # 切分失败时返回空字符串列表，保证 explode 后每行至少有一个元素
                return [""], [b""]

        except (OSError, ffmpeg.Error):
            logger.exception("Failed to split video %s by timestamps", video_path)
            # 切分失败时返回空字符串列表，保证 explode 后每行至少有一个元素
            return [""], [b""]

    def transform(
        self,
        video_paths: pa.Array | None = None,
        video_binaries: pa.Array | None = None,
        video_formats: pa.Array | None = None,
        timestamp_ranges: pa.Array | None = None,
        output_basenames: pa.Array | None = None,
    ) -> pa.Array:
        """按时间戳切分视频为片段

        注意：`video_paths` 和 `video_binaries` 至少需要指定一个，否则返回空结果

        Args:
            video_paths: 包含输入视频路径的数组
                默认值：None
            video_binaries: 包含视频二进制数据的数组
                默认值：None
            video_formats: 包含输入视频格式（如 'mp4'、'avi' 等）的数组，指定 video_binaries 时可以提供格式信息
                默认值：None
            timestamp_ranges: 每行对应的时间戳范围列表，支持格式：
                - [(start,end), (start,end)] 或 [[start,end], [start,end]]
                - 单个 (start,end) 或 [start,end]
            output_basenames: 可选，输出子目录名（文件名）数组

        Returns:
            pa.Array: 处理后的结构体字段包括：
                - segments: list[str]，切分后视频片段的路径列表
                - segments_binary: list[bytes]，切分后视频片段的二进制数据列表
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
        ranges_list_raw = timestamp_ranges.to_pylist() if timestamp_ranges is not None else [[] for _ in range(n)]

        results = []
        for video_path, video_binary, video_format, ranges_raw, basename in zip(
            paths_list, binaries_list, formats_list, ranges_list_raw, basenames_list
        ):
            try:
                ranges = self._normalize_timestamp_ranges(ranges_raw)
            except ValueError:
                ranges = []

            segments, binaries = self._process_video(video_path, video_binary, video_format, ranges, basename)

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
