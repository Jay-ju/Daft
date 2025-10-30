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


class AudioFFMPEGWrapped(Operator):
    """**音频滤镜处理器，基于 FFmpeg 的灵活音频效果应用。**

    **核心功能：**
    - 通过 FFmpeg 应用常见音频滤镜（volume、highpass、lowpass、bass、treble、aecho 等）
    - 支持本地路径与 TOS/S3 远程路径自动下载与处理
    - 支持将处理结果上传到 TOS，或返回二进制结果

    **格式支持：**
    - MP3 (.mp3)
    - WAV (.wav)
    - FLAC (.flac)
    - OGG (.ogg)
    - AAC (.aac)
    - M4A (.m4a)

    参考文档: https://ffmpeg.org/ffmpeg-filters.html#Audio-Filters
    """  # noqa: D415

    def __init__(
        self,
        filter_name: str,
        filter_kwargs: dict[str, Any] | None = None,
        global_args: list[str] | None = None,
        output_tos_dir: str = "",
        output_audio_binary: bool = False,
        output_audio_format: str | None = None,
        **kwargs: Any,
    ) -> None:
        """初始化音频滤镜算子。

        Args:
            filter_name: FFmpeg 音频滤镜名称。
                常见滤镜:
                - volume: 音量调整
                - highpass: 高通滤波（去除低频噪声）
                - lowpass: 低通滤波（去除高频噪声）
                - bass/lowshelf: 低音增强
                - treble/highshelf: 高音增强
                - aecho: 回声效果
                更多滤镜详见官方文档。
            filter_kwargs: 滤镜参数字典（因滤镜而异）。
                示例：
                - volume: {"volume": 1.5}
                - highpass: {"f": 300, "width_type": "h", "width": 0.5}
                - lowpass: {"f": 3000, "width_type": "h", "width": 0.5}
            global_args: FFmpeg 全局参数列表。
                常用组合：
                - 静默模式(仅错误): ["-loglevel", "error", "-hide_banner"]
                - 调试模式: ["-loglevel", "debug", "-stats"]
                - 强制覆盖输出: ["-y"]
                - 性能优化: ["-threads", "4"]
            output_tos_dir: 输出结果的 TOS 目录（为空则不上传）。
            output_audio_binary: 是否返回处理后的音频二进制。
                默认值：False
            output_audio_format: 指定输出音频格式（如 "mp3"、"wav" 等）。
                若为空则沿用输入文件后缀；对于二进制输入则默认 "wav"。
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
        self.filter_name = filter_name
        self.filter_kwargs = filter_kwargs or {}
        self.global_args = global_args or []
        self.output_audio_binary = output_audio_binary
        self.output_audio_format = output_audio_format.lower() if output_audio_format else None

        logger.info("Audio filter: %s", self.filter_name)
        logger.info("Audio filter kwargs: %s", self.filter_kwargs)
        logger.info("FFmpeg global args: %s", self.global_args)
        logger.info("Output audio format: %s", self.output_audio_format)

        tracking_usage(op=self.__class__.__name__, model_service_or_lib="ffmpeg")

    def _get_output_extension(self, audio_path: str | None, input_format: str | None) -> str:
        if self.output_audio_format:
            return f".{self.output_audio_format}"
        if input_format:
            return f".{input_format.lower()}"
        if audio_path:
            return Path(audio_path).suffix.lower()
        logger.warning("No audio format specified, defaulting to .wav for output")
        return ".wav"

    def _apply_filter(self, src_path: str, dst_path: str) -> None:
        try:
            input_stream = ffmpeg.input(src_path)
            audio_stream = input_stream.audio.filter(self.filter_name, **self.filter_kwargs)
            (ffmpeg.output(audio_stream, dst_path).global_args(*self.global_args).overwrite_output().run(quiet=True))
        except ffmpeg.Error:
            logger.exception("FFmpeg error while processing audio: %s -> %s", src_path, dst_path)
            raise

    def _process_and_upload(
        self, local_src: str, local_dir: str, basename: str, input_format: str | None, tos_output_dir: str | None
    ) -> tuple[str, bytes | None]:
        ext = self._get_output_extension(local_src, input_format).lstrip(".")
        processed_name = f"{basename}_processed.{ext}"
        processed_path = str(Path(local_dir) / processed_name)

        self._apply_filter(local_src, processed_path)

        if tos_output_dir:
            mkdirs(tos_output_dir)
            tos_path = f"{tos_output_dir}/{processed_name}"
            upload_file(processed_path, tos_path)
            final_path = tos_path
        else:
            final_path = processed_path

        binary: bytes | None = None
        if self.output_audio_binary:
            with Path(processed_path).open("rb") as f:
                binary = f.read()

        return final_path, binary

    def _process_audio(
        self,
        audio_path: str | None,
        audio_binary: bytes | None,
        audio_format: str | None,
        output_basename: str | None = None,
    ) -> tuple[str, bytes | None]:
        from daft.las.utils import not_blank

        if not_blank(output_basename):
            base = str(output_basename)
        elif audio_path:
            base = Path(audio_path).stem
        else:
            base = f"binary_{uuid.uuid4().hex}"

        tos_output_dir = self.output_tos_dir or None

        try:
            if audio_path is None and audio_binary is not None:
                if len(audio_binary) < 16:
                    logger.warning("Binary audio too small, skip.")
                    return "", None

                with tempfile.TemporaryDirectory(dir="/tmp") as temp_sub_dir:
                    temp_dir = temp_sub_dir.rstrip("/")
                    ext = self._get_output_extension(None, audio_format)
                    temp_filename = f"{base}{ext}"
                    temp_src = str(Path(temp_dir) / temp_filename)
                    with Path(temp_src).open("wb") as tmp:
                        tmp.write(audio_binary)

                    local_out_dir = temp_dir
                    return self._process_and_upload(temp_src, local_out_dir, base, audio_format, tos_output_dir)

            elif audio_path:

                def process(local_src: str) -> tuple[str, bytes | None]:
                    local_dir = str(Path(local_src).parent)
                    return self._process_and_upload(local_src, local_dir, base, audio_format, tos_output_dir)

                return run_on_local_path(audio_path, process)
            else:
                return "", None

        except (OSError, ffmpeg.Error):
            logger.exception("Failed to process audio: %s", audio_path)
            return "", None

    def transform(
        self,
        audio_paths: pa.Array | None = None,
        audio_binaries: pa.Array | None = None,
        audio_formats: pa.Array | None = None,
        output_basenames: pa.Array | None = None,
    ) -> pa.Array:
        """对输入音频应用 FFmpeg 滤镜。

        注意：`audio_paths` 和 `audio_binaries` 至少需要指定一个，否则返回空结果

        Args:
            audio_paths: 包含输入音频路径的数组
                默认值：None
            audio_binaries: 包含音频二进制数据的数组
                默认值：None
            audio_formats: 包含输入音频格式（如 'mp3'、'wav' 等）的数组，指定 audio_binaries 时可以提供格式信息
                默认值：None
            output_basenames: 可选，输出文件的基础名数组（不含扩展名），用于自定义输出文件名
                默认值：None

        Returns:
            pa.Array: 处理后的结构体字段包括：
                - processed_audio_path: str，处理后音频的路径（本地或 TOS）
                - processed_audio_binary: bytes，处理后音频的二进制内容（当 output_audio_binary=True 时）
        """  # noqa: D415
        n = 0
        if audio_paths is not None:
            n = len(audio_paths)
        elif audio_binaries is not None:
            n = len(audio_binaries)
        else:
            return pa.array([], type=self.__return_column_type__())

        paths_list = audio_paths.to_pylist() if audio_paths is not None else [None] * n
        binaries_list = audio_binaries.to_pylist() if audio_binaries is not None else [None] * n
        formats_list = audio_formats.to_pylist() if audio_formats is not None else [None] * n
        basenames_list = output_basenames.to_pylist() if output_basenames is not None else [None] * n

        results: list[dict[str, Any]] = []
        for audio_path, audio_binary, audio_format, basename in zip(
            paths_list, binaries_list, formats_list, basenames_list
        ):
            processed_path, processed_binary = self._process_audio(audio_path, audio_binary, audio_format, basename)
            results.append(
                {
                    "processed_audio_path": processed_path,
                    "processed_audio_binary": processed_binary if self.output_audio_binary else None,
                }
            )

        return pa.array(results, type=self.__return_column_type__())

    @staticmethod
    def __return_column_type__() -> pa.DataType:
        fields = [
            pa.field("processed_audio_path", pa.string()),
            pa.field("processed_audio_binary", pa.binary()),
        ]
        return pa.struct(fields)
