# Copyright (c) Beijing Volcano Engine Technology Ltd.

from __future__ import annotations

import logging
import tempfile
import uuid
from pathlib import Path
from typing import Any

import av
import numpy as np  # noqa: TID253

from daft.dependencies import pa
from daft.las.functions.types import Operator
from daft.las.functions.utils.common_utils import run_on_local_path, tracking_usage
from daft.las.io import mkdirs, upload_file
from daft.las.utils import not_blank

logger = logging.getLogger(__name__)


class VideoExtractAudio(Operator):
    """**视频音频抽取处理器，支持多流分离**

    **核心功能：**
    - 支持从本地、TOS、HTTP等多种路径或二进制输入的视频中抽取音频流
    - 支持多音频流选择、只取第一个流或全部流
    - 支持输出音频到TOS、返回二进制、采样率等
    - 支持抽取指定时间区间（start_second, end_second）
    - **所有输出音频均为 mp3 格式**
    """  # noqa: D415

    def __init__(
        self,
        output_tos_dir: str = "",
        output_audio_binary: bool = False,
        output_audio_array: bool = False,
        stream_indexes: list[int] | None = None,
        return_first_stream: bool = True,
        start_second: float | None = None,
        end_second: float | None = None,
        output_format: str = "mp3",
        output_sample_rate: int = 48000,
        **kwargs: Any,
    ) -> None:
        """
        初始化视频音频抽取处理器参数

        Args:
            output_tos_dir: 将从视频中抽取出的音频保存到该 TOS 目录中，如果为空，则不保存音频
            output_audio_binary: 是否返回音频的二进制内容，默认为 False
            output_audio_array: 是否返回音频的 numpy array 格式，默认为 False
            stream_indexes: 指定抽取哪些音频流，默认全部。超出范围的索引会被自动忽略
            return_first_stream: 如果为 True，则只返回第一个音频流，否则返回所有流的列表，默认为 True
            start_second: 从视频的第几秒开始抽取音频，None 表示从头开始，单位为秒
            end_second: 到视频的第几秒结束抽取音频，None 表示到结尾，单位为秒
            output_format: 输出音频格式，默认 "mp3"
            output_sample_rate: 输出音频采样率，默认 48000
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
        self.output_audio_binary = output_audio_binary
        self.output_audio_array = output_audio_array
        self.stream_indexes = stream_indexes
        self.return_first_stream = return_first_stream
        self.start_second = start_second
        self.end_second = end_second
        self.output_format = output_format.lower()
        self.output_sample_rate = output_sample_rate

        tracking_usage(op=self.__class__.__name__, model_service_or_lib="ffmpeg")

    def _extract_audio_from_video(
        self,
        video_path: str,
        local_output_dir: str | Path,
        tos_output_dir: str | None,
        stream_indexes: list[int] | None = None,
        return_first_stream: bool = True,
        start_second: float | None = None,
        end_second: float | None = None,
    ) -> tuple[list[str], list[bytes], list[float], list[np.ndarray]]:
        """Extract audio streams from a video file.

        Returns:
            audio_paths: List of audio TOS paths.
            binaries: List of audio binary data.
            sampling_rates: List of audio sampling rates.
            audio_arrays: List of audio array (numpy array, or empty if not enabled).
        """
        try:
            input_container = av.open(video_path, "r")
        except Exception:
            logger.exception("Failed to open video: %s", video_path)
            return [], [], [], []

        num_audio_streams = len(input_container.streams.audio)
        if num_audio_streams == 0:
            input_container.close()
            return [], [], [], []

        if stream_indexes is None:
            valid_stream_indexes = list(range(num_audio_streams))
        else:
            valid_stream_indexes = [idx for idx in stream_indexes if idx < num_audio_streams]

        if not valid_stream_indexes:
            input_container.close()
            return [], [], [], []

        if return_first_stream:
            valid_stream_indexes = valid_stream_indexes[:1]

        audio_paths: list[str] = []
        binaries: list[bytes] = []
        original_audio_sampling_rates: list[float] = []
        audio_arrays: list[np.ndarray] = []

        ext = f".{self.output_format}"
        codec_name = self.output_format
        sample_rate = self.output_sample_rate

        for idx in valid_stream_indexes:
            audio_stream = input_container.streams.audio[idx]
            original_sampling_rate = float(1 / audio_stream.time_base)
            original_audio_sampling_rates.append(original_sampling_rate)
            out_basename = f"audio_stream_{idx}{ext}"
            out_path = str(Path(local_output_dir) / out_basename)

            audio_data = []
            try:
                output_container = av.open(out_path, "w")
                output_stream = output_container.add_stream(codec_name)
                output_stream.codec_context.sample_rate = sample_rate

                # Calculate start and end frame
                start_pts = int(start_second / audio_stream.time_base) if start_second is not None else None
                end_pts = int(end_second / audio_stream.time_base) if end_second is not None else None

                for frame in input_container.decode(audio_stream):
                    if frame.pts is None:
                        continue
                    if start_pts is not None and frame.pts < start_pts:
                        continue
                    if end_pts is not None and frame.pts > end_pts:
                        break
                    # collect audio array if needed
                    if self.output_audio_array:
                        arr = frame.to_ndarray()
                        if arr.ndim == 2:
                            arr = arr[0]
                        audio_data.append(arr)
                    for packet in output_stream.encode(frame):
                        output_container.mux(packet)
                for packet in output_stream.encode(None):
                    output_container.mux(packet)
                output_container.close()
            except Exception:
                logger.exception("Failed to extract audio stream %d from %s", idx, video_path)
                if self.output_audio_array:
                    audio_arrays.append(np.array([], dtype=np.float32))
                continue

            # Upload to TOS
            if tos_output_dir:
                tos_path = f"{tos_output_dir}/{Path(out_path).name}"
                upload_file(out_path, tos_path)
                audio_paths.append(tos_path)
            else:
                audio_paths.append(str(out_path))

            # Read binary
            if self.output_audio_binary:
                with Path(out_path).open("rb") as f:
                    binaries.append(f.read())
            else:
                binaries.append(b"")

            # Save audio array as numpy array (float32)
            if self.output_audio_array:
                if audio_data:
                    arr = np.concatenate(audio_data).astype(np.float32)
                    audio_arrays.append(arr)
                else:
                    audio_arrays.append(np.array([], dtype=np.float32))
        input_container.close()
        return audio_paths, binaries, original_audio_sampling_rates, audio_arrays

    def _process_video(
        self,
        video_path: str | None,
        video_binary: bytes | None,
        video_format: str | None,
        output_basename: str | None = None,
    ) -> tuple[list[str], list[bytes], list[float], list[np.ndarray]]:
        """Process a single video input, supporting path or binary input.

        Args:
            video_path: Path to the video file (can be local, TOS, HTTP, etc.)
            video_binary: Video file content in bytes (optional)
            video_format: Video format string (optional)
            output_basename: Optional output subdirectory name

        Returns:
            Tuple of (audio_paths, binaries, original_audio_sampling_rates, audio_arrays)
        """
        is_valid_video_path = not_blank(video_path)
        if not is_valid_video_path and video_binary is None:
            return [], [], [], []

        # 选择子目录名
        if not_blank(output_basename):
            video_sub_dir = output_basename
        elif is_valid_video_path:
            video_sub_dir = Path(str(video_path)).stem
        else:
            video_sub_dir = f"binary_{uuid.uuid4().hex}"

        if self.output_tos_dir:
            tos_output_dir = f"{self.output_tos_dir}/{video_sub_dir}"
            mkdirs(tos_output_dir)
        else:
            tos_output_dir = None

        try:
            # If video_path is valid, always use it (prefer path over binary)
            if is_valid_video_path and video_path is not None:

                def process_with_path(local_path: str) -> tuple[list[str], list[bytes], list[float], list[np.ndarray]]:
                    local_output_dir = Path(local_path).parent / video_sub_dir  # type: ignore[operator]
                    local_output_dir.mkdir(exist_ok=True)
                    return self._extract_audio_from_video(
                        local_path,
                        local_output_dir,
                        tos_output_dir,
                        self.stream_indexes,
                        self.return_first_stream,
                        self.start_second,
                        self.end_second,
                    )

                result = run_on_local_path(str(video_path), process_with_path)
            # If no valid path, but binary is provided, use binary
            elif video_binary is not None:
                with tempfile.TemporaryDirectory(dir="/tmp") as temp_sub_dir:
                    temp_dir = temp_sub_dir.rstrip("/")
                    ext = f".{video_format.lower()}" if video_format else ".mp4"
                    temp_filename = f"{video_sub_dir}{ext}"
                    temp_filepath = Path(temp_dir) / temp_filename
                    with temp_filepath.open("wb") as tmp:
                        tmp.write(video_binary)

                    local_output_dir = Path(temp_dir) / video_sub_dir  # type: ignore[operator]
                    local_output_dir.mkdir(exist_ok=True)

                    return self._extract_audio_from_video(
                        str(temp_filepath),
                        local_output_dir,
                        tos_output_dir,
                        self.stream_indexes,
                        self.return_first_stream,
                        self.start_second,
                        self.end_second,
                    )
            # Neither path nor binary is valid, raise error
            else:
                raise ValueError("Neither path nor binary is valid")

        except Exception:
            logger.exception("Failed to extract audio from video: %s or video_binary", video_path)
            return [], [], [], []
        else:
            return result

    def transform(
        self,
        video_paths: pa.Array | None = None,
        video_binaries: pa.Array | None = None,
        video_formats: pa.Array | None = None,
        output_basenames: pa.Array | None = None,
    ) -> pa.Array:
        """从视频中抽取音频流，支持多种输入格式，输出结构体字段，所有输出音频均为用户指定格式（默认 mp3）

        Args:
            video_paths: 视频文件路径数组（本地、TOS、HTTP等）
            video_binaries: 视频二进制数据数组（可选）
            video_formats: 视频格式字符串数组（可选）
            output_basenames: 可选，输出子目录名（文件名）数组

        Returns:
            结构体数组，包含：
                - audio_paths: 包含音频 TOS 路径的列
                - audio_arrays: 包含音频 array 数据的列（可选）
                - binaries: 包含音频二进制数据的列（可选）
                - original_audio_sampling_rates: 始终返回，包含每个音频流的原始采样率，和audio_paths/binaries一一对应

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

        for video_path, binary, fmt, basename in zip(paths_list, binaries_list, formats_list, basenames_list):
            audio_paths, binaries, original_audio_sampling_rates, audio_arrays = self._process_video(
                video_path, binary, fmt, basename
            )
            # audio_arrays: list[np.ndarray] or empty
            if self.output_audio_array:
                # convert each np.ndarray to list[float] for Arrow compatibility
                audio_arrays_out = [arr.tolist() if arr is not None else [] for arr in audio_arrays]
            else:
                audio_arrays_out = []
            result: dict[str, Any] = {
                "audio_paths": audio_paths,
                "audio_arrays": audio_arrays_out,
                "binaries": binaries,
                "original_audio_sampling_rates": original_audio_sampling_rates,
            }
            results.append(result)

        return pa.array(results, type=self.__return_column_type__())

    @staticmethod
    def __return_column_type__() -> pa.DataType:
        fields = [
            pa.field("audio_paths", pa.list_(pa.string())),
            pa.field("audio_arrays", pa.list_(pa.list_(pa.float32()))),
            pa.field("binaries", pa.list_(pa.binary())),
            pa.field("original_audio_sampling_rates", pa.list_(pa.float64())),
        ]
        return pa.struct(fields)
