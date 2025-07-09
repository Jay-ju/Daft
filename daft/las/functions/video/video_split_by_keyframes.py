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
from daft.las.functions.utils.common_utils import run_on_local_path
from daft.las.functions.video.video_keyframes import VideoKeyframes
from daft.las.io import mkdirs, upload_file

logger = logging.getLogger(__name__)


class VideoSplitByKeyframes(Operator):
    """**视频关键帧切分处理器，支持智能片段分割。**

    **核心功能：**
    - 多算法关键帧检测：
        - I_frame: 基于I帧检测(推荐)
        - difference: 像素差异检测
        - histogram: 直方图差异检测
    - 支持片段二进制输出或TOS存储
    - 提供时间戳定位功能

    **格式支持：**
    - MP4 (.mp4)
    - AVI (.avi)
    - MOV (.mov)
    - MKV (.mkv)
    - 其他常见视频格式
    """  # noqa: D415

    def __init__(
        self,
        method: str = "I_frame",
        threshold: float = 0,
        keyframes_cnt: int = 10,
        seconds_per_frame: int = -1,
        output_tos_dir: str = "",
        output_segments_binary: bool = False,
        **kwargs: Any,
    ) -> None:
        """初始化视频按关键帧切分算子。

        Args:
            method: 抽取关键帧的方法，支持 "difference"（像素差分法）、"histogram"（直方图法）、"I_frame"（I型关键帧标识）。
                注意: 非I-frame方法可能需要较长时间处理视频。
                可选值：["difference", "histogram", "I_frame"]
                默认值："I_frame"
            threshold: 用于判断关键帧的阈值。difference 推荐 2000000，histogram 推荐 0.01。
                默认值：0
            keyframes_cnt: 用于指定抽取视频关键帧的数量。-1表示使用所有检测到的关键帧；0表示无效参数（不会切分视频）；
                如果设置的数量大于实际关键帧数，则使用所有关键帧；否则会从所有关键帧中均匀地选取指定数量的关键帧，
                避免关键帧集中分布在某个时间段内。
                默认值：10
            seconds_per_frame: 抽帧间隔，单位为秒，-1 表示不指定间隔。
                默认值：-1
            output_tos_dir: 保存视频片段的TOS路径，若为空字符串则不上传。
                默认值：""
            output_segments_binary: 是否输出视频片段的二进制数据。
                默认值：False
            **kwargs: 其他参数，透传给父类。
        """  # noqa: D415
        super().__init__(**kwargs)
        self.method = method
        self.threshold = threshold
        self.keyframes_cnt = keyframes_cnt
        self.seconds_per_frame = seconds_per_frame
        self.output_tos_dir = output_tos_dir.strip("/") if output_tos_dir else ""
        self.output_segments_binary = output_segments_binary

        logger.info("Keyframe extraction method: %s", self.method)
        logger.info("Keyframe threshold: %s", self.threshold)
        logger.info("Max keyframes count: %s", self.keyframes_cnt)
        logger.info("Seconds per frame: %s", self.seconds_per_frame)

    def _get_output_extension(self, video_path: str | None, video_format: str | None) -> str:
        """Determine the output file extension."""
        if video_format:
            return f".{video_format.lower()}"
        if video_path:
            return Path(video_path).suffix.lower()
        logger.warning("No video format specified, defaulting to .mp4 for segment")
        return ".mp4"

    def _prepare_timestamps(self, timestamps: list[float], video_path: str) -> list[float]:
        """Prepare timestamps list by ensuring it starts with 0.0 and ends with video duration."""
        if not timestamps:
            return []

        try:
            probe = ffmpeg.probe(video_path)
            duration = float(probe["format"]["duration"])
        except (ffmpeg.Error, KeyError, ValueError):
            logger.exception("Failed to get video duration for %s", video_path)
            return []

        prepared_timestamps = [0.0]
        prepared_timestamps.extend(timestamps)
        prepared_timestamps.append(duration)

        return sorted(list(set(prepared_timestamps)))

    def _get_segment_output_path(
        self, i: int, local_output_path: str, video_path: str | None, video_format: str | None
    ) -> str:
        """Generate output path for a segment."""
        output_ext = self._get_output_extension(video_path, video_format)
        return f"{local_output_path}/segment_{i+1}{output_ext}"

    def _process_and_upload_segment(
        self, video_path: str, start_time: float, end_time: float, segment_path: str, tos_output_dir: str | None
    ) -> str | None:
        """Process a segment and upload to TOS if needed."""
        try:
            (
                ffmpeg.input(video_path, ss=start_time, to=end_time)
                .output(segment_path, vcodec="copy", acodec="copy", strict="experimental")
                .overwrite_output()
                .run(quiet=True)
            )
        except ffmpeg.Error:
            logger.exception("FFmpeg error while splitting segment %s-%s", start_time, end_time)
            return None

        if tos_output_dir:
            tos_path = f"{tos_output_dir}/{Path(segment_path).name}"
            upload_file(segment_path, tos_path)
            return tos_path
        return segment_path

    def _split_video_by_keyframes(
        self,
        video_path: str,
        local_output_path: str,
        tos_output_dir: str | None,
        timestamps: list[float],
        video_format: str | None,
    ) -> tuple[list[str], list[bytes], list[str]]:
        timestamps = self._prepare_timestamps(timestamps, video_path)
        if not timestamps or len(timestamps) < 2:
            logger.warning("Not enough keyframes found in video %s to split", video_path)
            return [], [], []

        segments = []
        binaries = []
        output_exts = []

        for i in range(len(timestamps) - 1):
            start_time = timestamps[i]
            end_time = timestamps[i + 1]

            segment_path = self._get_segment_output_path(i, local_output_path, video_path, video_format)
            output_ext = Path(segment_path).suffix.lstrip(".")

            segment_result = self._process_and_upload_segment(
                video_path, start_time, end_time, segment_path, tos_output_dir
            )
            if not segment_result:
                continue

            segments.append(segment_result)
            output_exts.append(output_ext)

            if self.output_segments_binary:
                with Path(segment_path).open("rb") as f:
                    binaries.append(f.read())

        return segments, binaries, output_exts

    def _process_video(
        self, video: str | None, video_binary: bytes | None, video_format: str | None, timestamps: list[float]
    ) -> tuple[list[str], list[bytes], list[str]]:
        """Process a single video file and return tuple of (segments, binaries, formats)."""
        if self.output_tos_dir:
            video_sub_dir = Path(video).stem if video else f"binary_{uuid.uuid4().hex}"
            tos_output_dir = f"{self.output_tos_dir}/{video_sub_dir}"
            logger.info("Video segments tos output dir: %s", tos_output_dir)
            mkdirs(tos_output_dir)
        else:
            tos_output_dir = None

        try:
            if video is None and video_binary is not None:
                with tempfile.TemporaryDirectory(dir="/tmp") as temp_sub_dir:
                    temp_dir = temp_sub_dir.rstrip("/")
                    ext = self._get_output_extension(None, video_format)
                    temp_filename = f"binary_input_{uuid.uuid4().hex}{ext}"
                    temp_filepath = Path(temp_dir) / temp_filename
                    with temp_filepath.open("wb") as tmp:
                        tmp.write(video_binary)

                    local_output_dir = Path(temp_dir) / "segments"
                    local_output_dir.mkdir(exist_ok=True)

                    return self._split_video_by_keyframes(
                        str(temp_filepath), str(local_output_dir), tos_output_dir, timestamps, video_format
                    )
            elif video:

                def process_video(local_path: str) -> tuple[list[str], list[bytes], list[str]]:
                    video_name = Path(local_path).stem
                    local_output_dir = Path(local_path).parent / video_name
                    Path(local_output_dir).mkdir(exist_ok=True)
                    return self._split_video_by_keyframes(
                        str(local_path), str(local_output_dir), tos_output_dir, timestamps, video_format
                    )

                return run_on_local_path(video, process_video)
            else:
                return [], [], []

        except (OSError, ffmpeg.Error):
            logger.exception("Failed to split video %s by keyframes", video)
            return [], [], []

    def _select_keyframes(self, timestamps: list[float], target_count: int) -> list[float]:
        """Select keyframes based on target count."""
        if not timestamps or target_count == 0:
            return []

        timestamps_cnt = len(timestamps)
        if target_count < 0 or timestamps_cnt <= target_count:
            return timestamps

        # Uniformly select keyframes
        indices = [int(i * (timestamps_cnt - 1) / (target_count - 1)) for i in range(target_count)]
        return [timestamps[i] for i in sorted(list(set(indices)))]

    def _extract_timestamps(
        self,
        video_paths: pa.Array | None,
        video_binaries: pa.Array | None,
        video_formats: pa.Array | None,
    ) -> list[list[float]]:
        """Extract keyframe timestamps for all videos."""
        keyframe_extractor = VideoKeyframes(
            method=self.method,
            threshold=self.threshold,
            keyframes_cnt=-1,  # Get all keyframes first
            seconds_per_frame=self.seconds_per_frame,
            output_tos_dir="",  # We don't need to save keyframes
        )
        keyframes_results = keyframe_extractor.transform(video_paths, video_binaries, video_formats)

        all_timestamps: list[list[float]] = []
        for result in keyframes_results.to_pylist():
            if result is None:
                all_timestamps.append([])
                continue
            timestamps = result.get("timestamps", [])
            selected = self._select_keyframes(timestamps, self.keyframes_cnt)
            all_timestamps.append(selected)

        return all_timestamps

    def transform(
        self,
        video_paths: pa.Array | None = None,
        video_binaries: pa.Array | None = None,
        video_formats: pa.Array | None = None,
    ) -> pa.Array:
        """将视频按关键帧切分为片段

        注意：`video_paths` 和 `video_binaries` 至少需要指定一个，否则返回空结果

        Args:
            video_paths: 包含输入视频路径的数组
                默认值：None
            video_binaries: 包含视频二进制数据的数组
                默认值：None
            video_formats: 包含输入视频格式（如 'mp4'、'avi' 等）的数组，指定 video_binaries 时可以提供格式信息
                默认值：None

        Returns:
            pa.Array: 处理后的结构体字段包括：
                - segments: list[str]，切分后视频片段的路径列表
                - segments_binary: list[bytes]，切分后视频片段的二进制数据列表
                - video_format: list[str]，切分后视频片段的格式列表
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

        all_timestamps = self._extract_timestamps(video_paths, video_binaries, video_formats)

        results = []
        for video, video_binary, video_format, timestamps in zip(
            paths_list, binaries_list, formats_list, all_timestamps
        ):
            segments, binaries, formats = self._process_video(video, video_binary, video_format, timestamps)

            result_dict = {
                "segments": segments,
                "segments_binary": binaries if self.output_segments_binary else [],
                "video_format": formats,
            }
            results.append(result_dict)

        return pa.array(results, type=self.__return_column_type__())

    @staticmethod
    def __return_column_type__() -> pa.DataType:
        fields = [
            pa.field("segments", pa.list_(pa.string())),
            pa.field("segments_binary", pa.list_(pa.binary())),
            pa.field("video_format", pa.list_(pa.string())),
        ]
        return pa.struct(fields)
