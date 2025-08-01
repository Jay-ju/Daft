# Copyright (c) Beijing Volcano Engine Technology Ltd.

from __future__ import annotations

import logging
import tempfile
import uuid
from pathlib import Path
from typing import Any

import ffmpeg
import numpy as np  # noqa: TID253

from daft.dependencies import pa
from daft.las.functions.types import Operator
from daft.las.functions.utils.common_utils import log_op_call, run_on_local_path
from daft.las.io import mkdirs, upload_file
from daft.las.utils import not_blank

logger = logging.getLogger(__name__)


class AudioSplitByTimestamps(Operator):
    """**音频时间戳切分处理器，支持精准片段提取。**

    **核心功能**
    - 基于时间戳精确切分音频
    - 支持多输入格式：
       - 本地文件路径
       - TOS存储路径
       - 二进制数据流
    - 支持片段二进制输出或TOS存储

    **格式支持**
    - MP3 (.mp3)
    - WAV (.wav)
    - FLAC (.flac)
    - OGG (.ogg)
    - AAC (.aac)
    - M4A (.m4a)
    """  # noqa: D415

    def __init__(
        self,
        output_tos_dir: str = "",
        output_segments_binary: bool = False,
        output_format: str | None = None,
        **kwargs: Any,
    ) -> None:
        """初始化音频时间戳切分处理器参数。

        Args:
            output_tos_dir: 切分后的音频片段保存到 TOS 的路径。
            output_segments_binary: 是否返回切分后音频片段的二进制数据。
            output_format: 全局指定输出音频格式（如"mp3"、"wav"等），优先级高于输入文件后缀和 audio_format 列。
        """  # noqa: D415
        super().__init__(**kwargs)

        self.output_tos_dir = output_tos_dir.strip("/") if output_tos_dir else ""
        self.output_segments_binary = output_segments_binary
        self.output_format = output_format.lower() if output_format else None

        logger.info("Output to TOS dir: %s", self.output_tos_dir)
        logger.info("Output binary data: %s", self.output_segments_binary)
        logger.info("Output format: %s", self.output_format)

        log_op_call(logger=logger, op=self.__class__.__name__, model_service_or_lib="ffmpeg")

    def _process_segment(
        self,
        audio_path: str,
        segment_filename: str,
        start_time: float,
        end_time: float,
    ) -> bool:
        """Process and save a single audio segment."""
        try:
            output_kwargs = {"strict": "experimental"}
            # 如果指定了output_format，则不指定acodec，让ffmpeg自动选择编码器
            if not self.output_format:
                output_kwargs["acodec"] = "copy"
            (
                ffmpeg.input(audio_path, ss=start_time, to=end_time)
                .output(segment_filename, **output_kwargs)
                .overwrite_output()
                .run(quiet=True)
            )
        except ffmpeg.Error:
            logger.exception("FFmpeg error while splitting segment %s-%s", start_time, end_time)
            return False
        else:
            return True

    def _get_output_extension(
        self,
        audio_path: str | None,
        audio_binary: bytes | None,
        audio_format: str | None,
    ) -> str:
        """Determine the output file extension."""
        if self.output_format:
            return f".{self.output_format}"
        if audio_binary is not None and audio_format:
            return f".{audio_format.lower()}"
        if audio_path:
            return Path(audio_path).suffix.lower()
        logger.warning("No audio format specified, defaulting to mp3 for segments")
        return ".mp3"

    def _process_segment_file(
        self,
        segment_filename: str,
        tos_output_dir: str | None,
    ) -> tuple[str, bytes | None]:
        """Process a single segment file - read binary and/or upload to TOS."""
        binary: bytes | None = None
        if self.output_segments_binary:
            with Path(segment_filename).open("rb") as f:
                binary = f.read()

        if tos_output_dir:
            tos_path = f"{tos_output_dir}/{Path(segment_filename).name}"
            upload_file(segment_filename, tos_path)
            return tos_path, binary
        return str(segment_filename), binary

    def _prepare_result(
        self,
        segments: list[str],
        binaries: list[bytes],
    ) -> tuple[list[str], list[bytes]]:
        """Prepare the final result tuple based on requested outputs."""
        if not segments:
            return [], []

        if not self.output_segments_binary:
            binaries = []

        return segments, binaries

    def _normalize_timestamp_ranges(
        self,
        timestamp_ranges: Any,
    ) -> list[tuple[float, float]]:
        """Normalize timestamp_ranges input to a list of (start, end) tuples."""
        if not timestamp_ranges:
            return []

        arr = timestamp_ranges
        if isinstance(arr, np.ndarray):
            arr = arr.tolist()

        # Single interval: [start, end] or (start, end)
        if isinstance(arr, (list, tuple)) and len(arr) == 2 and all(isinstance(x, (int, float)) for x in arr):
            return [(float(arr[0]), float(arr[1]))]

        # Multiple intervals: [[start, end], ...] or [(start, end), ...]
        if isinstance(arr, (list, tuple)):
            valid_items: list[tuple[float, float]] = []
            for item in arr:
                if (
                    isinstance(item, (list, tuple))
                    and len(item) == 2
                    and all(isinstance(x, (int, float)) for x in item)
                ):
                    valid_items.append((float(item[0]), float(item[1])))
            if len(valid_items) == len(arr):
                return valid_items

        raise ValueError(
            "timestamp_ranges must be either:\n"
            "- A single (start, end) tuple or [start, end] list\n"
            "- A list of (start, end) tuples or [start, end] lists\n"
            "Example valid formats:\n"
            "  [(0.0, 5.0), (5.0, 10.0)]\n"
            "  [[0.0, 5.0], [5.0, 10.0]]\n"
            "  (0.0, 5.0)\n"
            "  [0.0, 5.0]"
        )

    def split_audio_by_timestamps(
        self,
        audio_path: str,
        local_output_path: str | Path,
        tos_output_dir: str | None,
        timestamp_ranges: Any,
        audio_binary: bytes | None = None,
        audio_format: str | None = None,
    ) -> tuple[list[str], list[bytes]]:
        """Split audio into segments based on timestamp ranges."""
        try:
            normalized_ranges = self._normalize_timestamp_ranges(timestamp_ranges)
        except Exception:
            logger.exception("Invalid timestamp_ranges format for audio %s", audio_path)
            return [], []

        if not normalized_ranges:
            logger.warning("No timestamp ranges provided for audio %s", audio_path)
            return [], []

        try:
            segments: list[str] = []
            binaries: list[bytes] = []

            output_ext = self._get_output_extension(audio_path, audio_binary, audio_format)

            for start_time, end_time in normalized_ranges:
                start_str = f"{start_time}".replace(".", "_")
                end_str = f"{end_time}".replace(".", "_")

                segment_basename = f"segment_{start_str}-{end_str}{output_ext}"
                segment_filename = f"{local_output_path}/{segment_basename}"
                if not self._process_segment(audio_path, segment_filename, start_time, end_time):
                    continue

                segment_path, binary = self._process_segment_file(segment_filename, tos_output_dir)
                segments.append(segment_path)
                if binary is not None:
                    binaries.append(binary)

            return self._prepare_result(segments, binaries)

        except Exception:
            logger.exception("Failed to split audio %s by timestamps", audio_path)
            return [], []

    def _process_audio(
        self,
        audio_path: str | None,
        audio_binary: bytes | None,
        audio_format: str | None,
        timestamp_ranges: Any,
        output_basename: str | None = None,
    ) -> tuple[list[str], list[bytes]]:
        """Process a single audio file."""
        is_valid_audio_path = not_blank(audio_path)

        # 选择子目录名
        if not_blank(output_basename):
            audio_sub_dir = output_basename
        elif is_valid_audio_path:
            audio_sub_dir = Path(audio_path).stem  # type: ignore[arg-type]
        else:
            audio_sub_dir = f"binary_{uuid.uuid4().hex}"

        if self.output_tos_dir:
            tos_output_dir = f"{self.output_tos_dir}/{audio_sub_dir}"
            logger.info("Audio segments tos output dir: %s", tos_output_dir)
            mkdirs(tos_output_dir)
        else:
            tos_output_dir = None

        try:
            if is_valid_audio_path:
                # If audio_path is valid, always use it (prefer path over binary)
                def process_with_path(local_path: str) -> tuple[list[str], list[bytes]]:
                    local_output_dir = Path(local_path).parent / audio_sub_dir  # type: ignore[operator]
                    local_output_dir.mkdir(exist_ok=True)
                    return self.split_audio_by_timestamps(
                        local_path,
                        local_output_dir,
                        tos_output_dir,
                        timestamp_ranges,
                        audio_binary,
                        audio_format,
                    )

                result = run_on_local_path(audio_path, process_with_path)  # type: ignore[arg-type]
            elif audio_binary is not None:
                with tempfile.TemporaryDirectory(dir="/tmp") as temp_sub_dir:
                    temp_dir = temp_sub_dir.rstrip("/")
                    ext = self._get_output_extension(None, audio_binary, audio_format)
                    temp_filename = f"{audio_sub_dir}{ext}"
                    temp_filepath = Path(temp_dir) / temp_filename
                    with temp_filepath.open("wb") as tmp:
                        tmp.write(audio_binary)

                    local_output_dir = Path(temp_dir) / audio_sub_dir  # type: ignore[operator]
                    local_output_dir.mkdir(exist_ok=True)

                    result = self.split_audio_by_timestamps(
                        str(temp_filepath),
                        local_output_dir,
                        tos_output_dir,
                        timestamp_ranges,
                        audio_binary,
                        audio_format,
                    )
            else:
                raise ValueError("Both audio_path and audio_binary are empty.")

        except Exception:
            logger.exception("Failed to split audio %s by timestamps", audio_path)
            return [], []
        else:
            return result

    def transform(
        self,
        timestamp_ranges: pa.Array,
        audio_paths: pa.Array | None = None,
        audio_binaries: pa.Array | None = None,
        audio_formats: pa.Array | None = None,
        output_basenames: pa.Array | None = None,
    ) -> pa.Array:
        """根据给定的时间戳区间批量切分音频

        注意：`audio_paths` 和 `audio_binaries` 至少需要指定一个，否则返回空结果

        Args:
            timestamp_ranges: 必需，包含切分用的时间戳区间的列
            audio_paths: 可选，包含音频文件路径的列
            audio_binaries: 可选，包含音频二进制数据的列
            audio_formats: 可选，包含音频格式字符串的列，指定 audio_binaries 时必须指定该列
            output_basenames: 可选，包含指定文件名的列

        Returns:
            一个结构体，包含音频切分结果：
                - segments: 片段路径列表
                - binaries: 片段二进制数据列表（可选）
        """  # noqa: D415
        # Convert to Python lists
        ranges_list = timestamp_ranges.to_pylist()
        paths_list = audio_paths.to_pylist() if audio_paths is not None else [None] * len(ranges_list)
        binaries_list = audio_binaries.to_pylist() if audio_binaries is not None else [None] * len(ranges_list)
        formats_list = audio_formats.to_pylist() if audio_formats is not None else [None] * len(ranges_list)
        basenames_list = output_basenames.to_pylist() if output_basenames is not None else [None] * len(ranges_list)

        results = []

        for path, binary, fmt, ranges, basename in zip(
            paths_list, binaries_list, formats_list, ranges_list, basenames_list
        ):
            segments, binaries = self._process_audio(path, binary, fmt, ranges, basename)

            # Build result struct
            result: dict[str, Any] = {
                "segments": segments,
                "binaries": binaries,
            }

            results.append(result)

        return pa.array(results, type=self.__return_column_type__())

    @staticmethod
    def __return_column_type__() -> pa.DataType:
        fields = []
        fields.append(pa.field("segments", pa.list_(pa.string())))
        fields.append(pa.field("binaries", pa.list_(pa.binary())))
        return pa.struct(fields)
