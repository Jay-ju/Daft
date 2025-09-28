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
from daft.las.functions.utils.video_utils import decode_video
from daft.las.io import mkdirs, upload_file
from daft.las.utils import not_blank

logger = logging.getLogger(__name__)


class VideoAdaptiveCompress(Operator):
    """**视频自适应压缩**

    **核心功能：**
    - 根据目标文件大小自适应压缩视频
    - 多级压缩策略：帧率调整 -> 分辨率调整 -> 码率控制
    - 保持视频质量的前提下尽可能压缩文件大小
    - 支持CPU和GPU编码
    - 支持路径输入、二进制输入和TOS输出

    **压缩策略：**
    1. 检查原始文件大小，小于目标大小直接返回
    2. 调整帧率到target_fps，默认5fps，检查大小是否满足（原fps小于或接近目标2fps内则跳过）
    3. 按比例调整分辨率到最小分辨率，默认360p，检查大小是否满足（原分辨率小于或接近目标分辨率1.2倍内则跳过）
    4. 使用两遍码率控制强制压缩到目标大小

    **格式支持：**
    - MP4 (.mp4)
    - AVI (.avi)
    - MOV (.mov)
    - MKV (.mkv)
    - 其他常见视频格式

    **编码模式：**
    - **CPU编码 (libx264)**：质量优先，压缩效率高，速度较慢，成本低
    - **GPU编码 (h264_nvenc)**：速度快，成本相对高
    """  # noqa: D415

    def __init__(
        self,
        output_tos_dir: str = "",
        max_output_size_mb: float = 50.0,
        target_fps: float = 5.0,
        min_resolution_height: int = 360,
        allowed_formats: list[str] = ["mp4", "avi", "mov"],
        rank: int | None = None,
        **kwargs: Any,
    ) -> None:
        """初始化视频自适应压缩算子。

        Args:
            output_tos_dir: 将压缩后的视频保存到该TOS目录中，如果为空则不保存。
                格式："tos://bucket/path/"
                默认值：""
            max_output_size_mb: 最大输出文件大小（MB）。
                默认值：50.0
            target_fps: 帧率调整的目标值。
                默认值：5.0
            min_resolution_height: 最小分辨率高度（像素），保持宽高比。
                默认值：360
            allowed_formats: 当视频大小满足时，允许直接输出的视频格式列表。
                默认值：["mp4", "avi", "mov"]
            rank: 指定使用的GPU设备编号（多卡环境有效）。
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

        self.max_output_size_mb = max_output_size_mb
        self.target_fps = target_fps
        self.min_resolution_height = min_resolution_height
        self.rank = rank
        self.device_id: int | None = None
        self.preset = "medium"
        self.allowed_formats = [fmt.lower() for fmt in allowed_formats]

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

    def _get_file_size_mb(self, source: str | bytes) -> float:
        try:
            if isinstance(source, bytes):
                return len(source) / (1024 * 1024)
            else:
                return Path(source).stat().st_size / (1024 * 1024)
        except Exception as e:
            logger.error("Failed to get file size: %s", e)
            return 0.0

    def _get_video_info(self, video_path: str) -> dict[str, float]:
        container = decode_video(video_path)
        try:
            video_stream = container.streams.video[0]
            return {
                "width": video_stream.codec_context.width,
                "height": video_stream.codec_context.height,
                "fps": float(video_stream.average_rate),
                "duration": float(container.duration / 1000000),
            }
        finally:
            container.close()

    def _calculate_target_resolution(self, original_width: int, original_height: int) -> tuple[int, int]:
        if original_height <= self.min_resolution_height:
            return original_width, original_height

        scale_factor = self.min_resolution_height / original_height
        new_width = int(original_width * scale_factor)
        new_height = self.min_resolution_height

        new_width = (new_width // 2) * 2
        new_height = (new_height // 2) * 2

        return new_width, new_height

    def _calculate_target_bitrate(self, duration_seconds: float, target_size_mb: float) -> int:
        target_size_bits = target_size_mb * 1024 * 1024 * 8
        audio_bits = 32 * 1000 * duration_seconds
        container_overhead = target_size_bits * 0.08

        available_bits = target_size_bits - audio_bits - container_overhead
        video_bitrate_bps = available_bits / duration_seconds

        safety_factor = 0.9
        video_bitrate_bps = video_bitrate_bps * safety_factor

        return int(video_bitrate_bps / 1000)

    def _compress_video_fps(self, input_path: str, output_path: str) -> str:
        logger.info("Compressing video by reducing FPS to %s: %s -> %s", self.target_fps, input_path, output_path)

        args = ["-nostdin", "-v", "quiet", "-y"]

        try:
            input_stream = ffmpeg.input(input_path)
            video_stream = input_stream.video.filter("fps", fps=self.target_fps)
            audio_stream = input_stream.audio

            output_params: dict[str, str | int | None] = {
                "vcodec": self.video_codec,
                "acodec": "aac",
                "strict": "-2",
            }

            if self.video_codec == "h264_nvenc":
                output_params.update(
                    {
                        "gpu": self.device_id,
                        "cq": 28,
                    }
                )
            else:
                output_params.update(
                    {
                        "crf": 28,
                    }
                )

            stream = (
                ffmpeg.output(video_stream, audio_stream, output_path, **output_params)
                .global_args(*args)
                .overwrite_output()
            )

            logger.info("FFmpeg command: %s", " ".join(ffmpeg.get_args(stream)))

            ffmpeg.run(stream, capture_stdout=True, capture_stderr=True)

        except ffmpeg.Error as e:
            stderr_text = e.stderr.decode("utf-8") if hasattr(e, "stderr") else "No stderr available"
            logger.error("FFmpeg failed with error: %s\nDetailed stderr: %s", e, stderr_text)
            raise

        return output_path

    def _compress_video_resolution(
        self, input_path: str, output_path: str, target_width: int, target_height: int
    ) -> str:
        logger.info(
            "Compressing video by reducing resolution to %dx%d: %s -> %s",
            target_width,
            target_height,
            input_path,
            output_path,
        )

        args = ["-nostdin", "-v", "quiet", "-y"]

        try:
            input_stream = ffmpeg.input(input_path)
            video_stream = input_stream.video.filter("scale", width=target_width, height=target_height)
            audio_stream = input_stream.audio

            output_params: dict[str, str | int | None] = {
                "vcodec": self.video_codec,
                "acodec": "aac",
                "strict": "-2",
            }

            if self.video_codec == "h264_nvenc":
                output_params.update(
                    {
                        "gpu": self.device_id,
                        "cq": 28,
                    }
                )
            else:
                output_params.update(
                    {
                        "crf": 28,
                    }
                )

            stream = (
                ffmpeg.output(video_stream, audio_stream, output_path, **output_params)
                .global_args(*args)
                .overwrite_output()
            )

            _, stderr = ffmpeg.run(stream, capture_stdout=True, capture_stderr=True)
            if stderr:
                logger.warning("FFmpeg stderr: %s", stderr.decode("utf-8"))

        except ffmpeg.Error as e:
            logger.exception("FFmpeg failed: %s", e)
            raise

        return output_path

    def _compress_video_two_pass(self, input_path: str, output_path: str, target_bitrate_kbps: int) -> str:
        logger.info(
            "Compressing video using 2-pass encoding with bitrate %dkbps: %s -> %s",
            target_bitrate_kbps,
            input_path,
            output_path,
        )

        audio_bitrate_stage3 = 32
        logger.info("Using reduced audio bitrate %dkbps for final compression", audio_bitrate_stage3)

        args = ["-nostdin", "-v", "quiet", "-y"]

        with tempfile.TemporaryDirectory() as temp_log_dir:
            pass_log_prefix = Path(temp_log_dir) / "ffmpeg2pass"

            try:
                if self.video_codec == "libx264":
                    pass1_params = {
                        "c:v": "libx264",
                        "b:v": f"{target_bitrate_kbps}k",
                        "pass": 1,
                        "passlogfile": str(pass_log_prefix),
                        "preset": self.preset,
                        "an": None,
                        "f": "mp4",
                    }

                    pass2_params = {
                        "c:v": "libx264",
                        "b:v": f"{target_bitrate_kbps}k",
                        "pass": 2,
                        "passlogfile": str(pass_log_prefix),
                        "preset": self.preset,
                        "c:a": "aac",
                        "b:a": f"{audio_bitrate_stage3}k",
                    }
                else:
                    pass1_params = {
                        "c:v": "h264_nvenc",
                        "b:v": f"{target_bitrate_kbps}k",
                        "pass": 1,
                        "passlogfile": str(pass_log_prefix),
                        "gpu": self.device_id,
                        "an": None,
                        "f": "mp4",
                    }

                    pass2_params = {
                        "c:v": "h264_nvenc",
                        "b:v": f"{target_bitrate_kbps}k",
                        "pass": 2,
                        "passlogfile": str(pass_log_prefix),
                        "gpu": self.device_id,
                        "c:a": "aac",
                        "b:a": f"{audio_bitrate_stage3}k",
                    }

                input_stream = ffmpeg.input(input_path)

                pass1_stream = (
                    ffmpeg.output(input_stream, "/dev/null", **pass1_params).global_args(*args).overwrite_output()
                )

                logger.info("Running first pass...")
                _, stderr = ffmpeg.run(pass1_stream, capture_stdout=True, capture_stderr=True)
                if stderr:
                    logger.warning("Pass 1 stderr: %s", stderr.decode("utf-8"))

                pass2_stream = (
                    ffmpeg.output(input_stream, output_path, **pass2_params).global_args(*args).overwrite_output()
                )

                logger.info("Running second pass...")
                _, stderr = ffmpeg.run(pass2_stream, capture_stdout=True, capture_stderr=True)
                if stderr:
                    logger.warning("Pass 2 stderr: %s", stderr.decode("utf-8"))

            except ffmpeg.Error as e:
                logger.exception("2-pass encoding failed: %s", e)
                raise

            return output_path

    def _convert_format_only(self, input_path: str, output_path: str) -> str:
        logger.info("Converting video format for compatibility: %s -> %s", input_path, output_path)

        args = ["-nostdin", "-v", "quiet", "-y"]

        try:
            input_stream = ffmpeg.input(input_path)
            video_stream = input_stream.video
            audio_stream = input_stream.audio

            output_params: dict[str, str | int | None] = {
                "vcodec": self.video_codec,
                "acodec": "aac",
                "strict": "-2",
            }

            if self.video_codec == "h264_nvenc":
                output_params.update(
                    {
                        "gpu": self.device_id,
                        "cq": 23,
                    }
                )
            else:
                output_params.update(
                    {
                        "crf": 23,
                    }
                )

            stream = (
                ffmpeg.output(video_stream, audio_stream, output_path, **output_params)
                .global_args(*args)
                .overwrite_output()
            )

            ffmpeg.run(stream, capture_stdout=True, capture_stderr=True)

        except ffmpeg.Error as e:
            stderr_text = e.stderr.decode("utf-8") if hasattr(e, "stderr") else "No stderr available"
            logger.error("Format conversion failed with error: %s\nDetailed stderr: %s", e, stderr_text)
            logger.warning("Format conversion failed, using original file")
            return input_path

        return output_path

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
                video_ext = Path(video_path).name.split(".")[-1]

                def process_with_path(local_path: str) -> str:
                    working_path = local_path
                    original_size_mb = self._get_file_size_mb(local_path)

                    if video_ext.lower() not in self.allowed_formats and original_size_mb <= self.max_output_size_mb:
                        logger.info("Non-compatible format detected for small file, converting to MP4")
                        format_converted_path = str(Path(local_path).parent / f"{video_name}_format_converted.mp4")
                        working_path = self._convert_format_only(local_path, format_converted_path)
                        video_ext_final = "mp4"
                        video_name_final = video_name
                    else:
                        video_ext_final = video_ext
                        video_name_final = video_name

                    current_size_mb = self._get_file_size_mb(working_path)
                    if current_size_mb <= self.max_output_size_mb:
                        compressed_video_name = f"{video_name_final}_compressed.{video_ext_final}"
                    else:
                        compressed_video_name = f"{video_name_final}_compressed.mp4"

                    compressed_local_path = self._compress_video_local(working_path, compressed_video_name)

                    if self.output_tos_dir:
                        tos_path = f"{self.output_tos_dir}/{compressed_video_name}"
                        upload_file(compressed_local_path, tos_path)
                        return tos_path
                    else:
                        return compressed_local_path

                return run_on_local_path(str(video_path), process_with_path)
            elif video_binary is not None:
                with tempfile.TemporaryDirectory(dir="/tmp") as temp_sub_dir:
                    temp_dir = temp_sub_dir.rstrip("/")
                    ext = f".{video_format.lower()}" if video_format else ".mp4"
                    temp_filename = f"{video_name_base}{ext}"
                    temp_filepath = Path(temp_dir) / temp_filename
                    with temp_filepath.open("wb") as tmp:
                        tmp.write(video_binary)

                    working_path = str(temp_filepath)
                    original_size_mb = self._get_file_size_mb(working_path)

                    if (
                        video_format
                        and video_format.lower() not in self.allowed_formats
                        and original_size_mb <= self.max_output_size_mb
                    ):
                        logger.info("Non-compatible format detected for small file, converting to MP4")
                        format_converted_path = str(Path(temp_dir) / f"{video_name_base}_format_converted.mp4")
                        working_path = self._convert_format_only(working_path, format_converted_path)
                        ext_final = ".mp4"
                    else:
                        ext_final = ext

                    current_size_mb = self._get_file_size_mb(working_path)
                    if current_size_mb <= self.max_output_size_mb:
                        compressed_video_name = f"{video_name_base}_compressed{ext_final}"
                    else:
                        compressed_video_name = f"{video_name_base}_compressed.mp4"

                    compressed_local_path = self._compress_video_local(working_path, compressed_video_name)

                    if self.output_tos_dir:
                        tos_path = f"{self.output_tos_dir}/{compressed_video_name}"
                        upload_file(compressed_local_path, tos_path)
                        return tos_path
                    else:
                        return compressed_local_path
            else:
                raise ValueError("Neither path nor binary is valid")

        except Exception:
            logger.exception("Failed to compress video: %s or video_binary", video_path)
            return ""

    def _compress_video_local(self, working_path: str, output_filename: str) -> str:
        local_dir = Path(working_path).parent

        original_size_mb = self._get_file_size_mb(working_path)
        logger.info("Original video size: %.2f MB, target: %.2f MB", original_size_mb, self.max_output_size_mb)

        if original_size_mb <= self.max_output_size_mb:
            logger.info("Video already meets size requirement, no compression needed")
            return working_path

        video_info = self._get_video_info(working_path)
        current_path = working_path

        original_fps = video_info["fps"]
        if original_fps <= self.target_fps + 2.0:
            logger.info(
                "Original FPS %.1f is close to target %.1f, skipping FPS reduction", original_fps, self.target_fps
            )
            current_size_mb = self._get_file_size_mb(current_path)
        else:
            step1_path = local_dir / f"step1_{output_filename}"
            current_path = self._compress_video_fps(current_path, str(step1_path))
            current_size_mb = self._get_file_size_mb(current_path)
            logger.info(
                "After FPS reduction from %.1f to %.1f: %.2f MB", original_fps, self.target_fps, current_size_mb
            )

        if current_size_mb <= self.max_output_size_mb:
            final_path = current_path
        else:
            original_height = int(video_info["height"])
            if original_height <= self.min_resolution_height * 1.2:
                logger.info(
                    "Original height %d is close to target %d, skipping resolution reduction",
                    original_height,
                    self.min_resolution_height,
                )
                current_size_mb = self._get_file_size_mb(current_path)
            else:
                target_width, target_height = self._calculate_target_resolution(
                    int(video_info["width"]), int(video_info["height"])
                )

                step2_path = local_dir / f"step2_{output_filename}"
                current_path = self._compress_video_resolution(
                    current_path, str(step2_path), target_width, target_height
                )
                current_size_mb = self._get_file_size_mb(current_path)
                logger.info(
                    "After resolution reduction from %dx%d to %dx%d: %.2f MB",
                    int(video_info["width"]),
                    int(video_info["height"]),
                    target_width,
                    target_height,
                    current_size_mb,
                )

            if current_size_mb <= self.max_output_size_mb:
                final_path = current_path
            else:
                target_bitrate = self._calculate_target_bitrate(video_info["duration"], self.max_output_size_mb)

                step3_path = local_dir / f"step3_{output_filename}"
                final_path = self._compress_video_two_pass(current_path, str(step3_path), target_bitrate)
                final_size_mb = self._get_file_size_mb(final_path)
                logger.info("After 2-pass encoding: %.2f MB", final_size_mb)

        return final_path

    def transform(
        self,
        video_paths: pa.Array | None = None,
        video_binaries: pa.Array | None = None,
        video_formats: pa.Array | None = None,
        output_basenames: pa.Array | None = None,
    ) -> pa.Array:
        """自适应压缩视频到指定大小，支持路径和二进制输入

        Args:
            video_paths: 视频文件路径列（本地、TOS、HTTP等），与video_binaries二选一
            video_binaries: 视频二进制数据列，与video_paths二选一
            video_formats: 视频格式字符串列，配合video_binaries使用
            output_basenames: 输出文件基础名称列（不含扩展名）

        Returns:
            压缩后的视频路径列
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

        compressed_videos_list = []

        for video_path, binary, fmt, basename in zip(paths_list, binaries_list, formats_list, basenames_list):
            compressed_path = self._process_video(video_path, binary, fmt, basename)
            compressed_videos_list.append(compressed_path)

        logger.info("Video adaptive compression finished.")
        return pa.array(compressed_videos_list)

    @staticmethod
    def __return_column_type__() -> pa.DataType:
        return pa.string()
