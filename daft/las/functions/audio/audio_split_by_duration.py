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
from daft.las.functions.utils.audio_utils import get_duration
from daft.las.functions.utils.common_utils import run_on_local_path
from daft.las.io import mkdirs, upload_file
from daft.las.utils import not_blank

logger = logging.getLogger(__name__)


class AudioSplitByDuration(Operator):
    """**音频按时长切分，支持按固定时长分割音频片段**

    **核心功能**
    - 按指定时长切分音频为多个片段
    - 支持多输入格式：
       - 本地文件路径
       - TOS/S3存储路径
       - 二进制数据流
    - 支持片段二进制输出或TOS存储
    - 支持最小片段时长过滤

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
        segment_duration: float = 5.0,
        min_segment_duration: float = 0.0,
        **kwargs: Any,
    ) -> None:
        """
        初始化音频时长切分处理器参数

        Args:
            output_tos_dir (str): 切分后的音频片段保存到 TOS 的路径
                - 如果指定该参数，所有切分后的音频片段将自动上传到该 TOS 路径下的子目录
                - 支持以 "tos://" 或 "s3://" 开头的远程路径，也支持本地路径
                - 为空时仅在本地生成片段，不上传
            output_segments_binary (bool): 是否返回切分后音频片段的二进制数据
                - True：返回每个片段的二进制内容（适合直接用于后续处理）
                - False：不返回二进制，仅返回片段路径
            output_audio_format (bool): 是否返回音频格式信息
                - True：返回每个片段的音频格式（如 "mp3"、"wav"）
                - False：不返回格式信息
            segment_duration (float): 每个片段的时长（单位：秒）
                - 必须为正数，决定每个音频片段的最大时长
                - 例如 5.0 表示每 5 秒切分一个片段
            min_segment_duration (float): 最小片段时长（单位：秒）
                - 小于该时长的片段不会被单独保存
                - 例如设置为 1.0，则最后不足 1 秒的片段会被丢弃
        """  # noqa: D212, D415
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
        self.output_segments_binary = output_segments_binary
        self.segment_duration = segment_duration
        self.min_segment_duration = min_segment_duration

        logger.info("Output to TOS dir: %s", self.output_tos_dir)
        logger.info("Output binary data: %s", self.output_segments_binary)
        logger.info("Segment duration: %s", self.segment_duration)
        logger.info("Minimum segment duration: %s", self.min_segment_duration)

    def _get_output_extension(
        self,
        audio_path: str | None,
        audio_binary: bytes | None,
        audio_format: str | None,
    ) -> str:
        """Determine the output file extension."""
        if audio_binary is not None and audio_format:
            return f".{audio_format.lower()}"
        if audio_path:
            return Path(audio_path).suffix.lower()
        logger.warning("No audio format specified, defaulting to mp3 for segments")
        return ".mp3"

    def _process_single_segment(
        self,
        audio_path: str,
        start_time: float,
        end_time: float,
        segment_filename: str,
    ) -> bool:
        """Process and save a single audio segment."""
        try:
            (
                ffmpeg.input(audio_path, ss=start_time, to=end_time)
                .output(segment_filename, acodec="copy", strict="experimental")
                .overwrite_output()
                .run(quiet=True)
            )
        except ffmpeg.Error:
            logger.exception("FFmpeg error while splitting segment %s-%s", start_time, end_time)
            return False
        else:
            return True

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

    def split_audio_by_duration(
        self,
        audio_path: str,
        local_output_path: str | Path,
        tos_output_dir: str | None,
        audio_binary: bytes | None = None,
        audio_format: str | None = None,
    ) -> tuple[list[str], list[bytes]]:
        """Split audio into segments by duration."""
        try:
            output_ext = self._get_output_extension(audio_path, audio_binary, audio_format)
            duration = get_duration(audio_path)
            if duration <= 0:
                logger.warning("Audio duration is 0 for %s", audio_path)
                return [], []

            segments: list[str] = []
            binaries: list[bytes] = []
            start_time = 0.0
            segment_idx = 1

            while start_time < duration:
                end_time = min(start_time + self.segment_duration, duration)
                if (end_time - start_time) >= self.min_segment_duration:
                    segment_filename = f"{local_output_path}/segment_{segment_idx}{output_ext}"
                    segment_idx += 1
                    if self._process_single_segment(audio_path, start_time, end_time, segment_filename):
                        segment_path, binary = self._process_segment_file(segment_filename, tos_output_dir)
                        segments.append(segment_path)
                        if binary is not None:
                            binaries.append(binary)
                start_time = end_time

            return self._prepare_result(segments, binaries)

        except Exception:
            logger.exception("Failed to split audio %s by duration", audio_path)
            return [], []

    def _process_audio(
        self,
        audio_path: str | None,
        audio_binary: bytes | None,
        audio_format: str | None,
    ) -> tuple[list[str], list[bytes]]:
        """Process a single audio file."""
        is_valid_audio_path = not_blank(audio_path)

        if self.output_tos_dir:
            if is_valid_audio_path:
                audio_sub_dir = Path(audio_path).stem  # type: ignore[arg-type]
            else:
                audio_sub_dir = f"binary_{uuid.uuid4().hex}"
            tos_output_dir = f"{self.output_tos_dir}/{audio_sub_dir}"
            logger.info("Audio segments tos output dir: %s", tos_output_dir)
            mkdirs(tos_output_dir)
        else:
            tos_output_dir = None

        try:
            if is_valid_audio_path:
                # If audio_path is valid, always use it (prefer path over binary)
                def process_with_path(local_path: str) -> tuple[list[str], list[bytes]]:
                    audio_name = ".".join(Path(audio_path).name.split(".")[:-1])  # type: ignore[arg-type]
                    local_output_dir = Path(local_path).parent / Path(audio_name)
                    local_output_dir.mkdir(exist_ok=True)
                    return self.split_audio_by_duration(
                        local_path,
                        local_output_dir,
                        tos_output_dir,
                        audio_binary,
                        audio_format,
                    )

                result = run_on_local_path(audio_path, process_with_path)  # type: ignore[arg-type]
            # If no valid path, but binary is provided, use binary
            elif audio_binary is not None:
                with tempfile.TemporaryDirectory(dir="/tmp") as temp_sub_dir:
                    temp_dir = temp_sub_dir.rstrip("/")
                    ext = self._get_output_extension(None, audio_binary, audio_format)
                    temp_filename = f"binary_input_{uuid.uuid4().hex}{ext}"
                    temp_filepath = Path(temp_dir) / temp_filename
                    with temp_filepath.open("wb") as tmp:
                        tmp.write(audio_binary)

                    local_output_dir = Path(temp_dir) / "segments"
                    local_output_dir.mkdir(exist_ok=True)

                    result = self.split_audio_by_duration(
                        str(temp_filepath),
                        local_output_dir,
                        tos_output_dir,
                        audio_binary,
                        audio_format,
                    )
            else:
                raise ValueError("Both audio_path and audio_binary are empty.")

        except Exception:
            logger.exception("Failed to split audio %s by duration", audio_path)
            return [], []
        else:
            return result

    def __call__(
        self,
        audio_paths: pa.Array | None = None,
        audio_binaries: pa.Array | None = None,
        audio_formats: pa.Array | None = None,
    ) -> pa.Array:
        """根据给定的音频输入批量按时长切分音频

        注意：`audio_paths` 和 `audio_binaries` 至少需要指定一个，否则返回空结果

        Args:
            audio_paths: 可选，包含音频文件路径的列
            audio_binaries: 可选，包含音频二进制数据的列
            audio_formats: 可选，包含音频格式字符串的列
        Returns:
            结构体数组，包含：
                - segments: 片段路径列表
                - binaries: 片段二进制数据列表（可选）
        """  # noqa: D415
        # Determine batch size
        batch_size = 0
        if audio_paths is not None:
            batch_size = len(audio_paths)
        elif audio_binaries is not None:
            batch_size = len(audio_binaries)
        elif audio_formats is not None:
            batch_size = len(audio_formats)
        else:
            return pa.array([], type=self.__return_column_type__())

        paths_list = audio_paths.to_pylist() if audio_paths is not None else [None] * batch_size
        binaries_list = audio_binaries.to_pylist() if audio_binaries is not None else [None] * batch_size
        formats_list = audio_formats.to_pylist() if audio_formats is not None else [None] * batch_size

        results = []

        for path, binary, fmt in zip(paths_list, binaries_list, formats_list):
            segments, binaries = self._process_audio(path, binary, fmt)

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
