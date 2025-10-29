from __future__ import annotations

import itertools
import json
import logging
import os
import subprocess
import tempfile
import threading
from typing import Any

from daft.dependencies import pa
from daft.las.functions.types import Operator
from daft.las.functions.utils.common_utils import is_local_path, pre_sign_url_for_tos
from daft.las.io import upload_file


class FastWriteCounter:
    def __init__(self, init: int = 0, step: int = 1) -> None:
        self._number_of_read = 0
        self._step = step
        self._counter = itertools.count(init, step)
        self._lock = threading.Lock()

    def increment(self) -> None:
        next(self._counter)

    @property
    def value(self) -> int:
        with self._lock:
            value = next(self._counter) - self._number_of_read
            self._number_of_read += self._step
        return value


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    handler.setFormatter(formatter)
    if not logger.handlers:
        logger.addHandler(handler)
    return logger


class VideoConvertToMp4(Operator):
    """**视频格式转换处理器，将各种视频格式转换为MP4**

    **核心功能：**
    - 支持多种视频格式转换为MP4
    - 智能音轨选择(全部/自动/首个)
    - 视频质量和编码参数自定义
    - 支持视频高度限制和缩放
    - 音频编码参数精细控制
    - 支持本地文件、HTTP/HTTPS URL和TOS/S3存储

    **格式支持：**
    - 输入：AVI、MOV、MKV、FLV、WMV、3GP等主流视频格式
    - 输出：MP4 (.mp4)
    - 视频编解码器：H.264/H.265等
    - 音频编解码器：AAC、MP3等
    """  # noqa: D415

    def __init__(
        self,
        video_codec: str = "libx264",
        crf: int = 23,
        preset: str = "medium",
        max_height: int | None = None,
        audio_codec: str = "aac",
        audio_bitrate: str = "192k",
        audio_sample_rate: int | None = None,
        select_audio: str = "all",
        extra_params: list[str] | None = None,
        timeout: int | None = None,
        **kwargs: Any,
    ) -> None:
        """初始化视频转换为MP4算子

        Args:
            video_codec: 视频编码器，支持libx264、libx265等
                默认值："libx264"
            crf: 视频质量控制，取值范围0-51，越小质量越好
                默认值：23
            preset: 编码速度预设，支持ultrafast、superfast、veryfast、faster、fast、medium、slow、slower、veryslow
                默认值："medium"
            max_height: 视频最大高度限制，超过时自动缩放，为None时不限制
                默认值：None
            audio_codec: 音频编码器，支持aac等
                默认值："aac"
            audio_bitrate: 音频码率，如"192k"、"128k"
                默认值："192k"
            audio_sample_rate: 音频采样率，如44100。48000，为None时保持原始采样率
                默认值：None
            select_audio: 音轨选择策略："all"（全部音轨）、"auto"（智能选择默认音轨）、"first"（首个音轨）
                默认值："all"
            extra_params: 额外的ffmpeg参数列表，如["-movflags", "+faststart"]
                默认值：None
            timeout: 单个视频处理超时时间（秒），为None时不限制
                默认值：None
        """  # noqa: D415
        super().__init__(**kwargs)
        self.video_codec = video_codec
        self.crf = crf
        self.preset = preset
        self.max_height = max_height
        self.audio_codec = audio_codec
        self.audio_bitrate = audio_bitrate
        self.audio_sample_rate = audio_sample_rate
        self.select_audio = select_audio
        self.extra_params = extra_params or []
        self.timeout = timeout

        self.submit_counter = FastWriteCounter()
        self.success_counter = FastWriteCounter()
        self.failed_counter = FastWriteCounter()

        self.logger = get_logger(f"VideoConvertToMp4-{id(self)}")

        self.logger.info(
            "VideoConvertToMp4 initialized with video_codec=%s, crf=%s, "
            "preset=%s, max_height=%s, audio_codec=%s, "
            "audio_bitrate=%s, audio_sample_rate=%s, "
            "select_audio=%s, extra_params=%s",
            video_codec,
            crf,
            preset,
            max_height,
            audio_codec,
            audio_bitrate,
            audio_sample_rate,
            select_audio,
            self.extra_params,
        )

    def log_progress(self) -> None:
        submitted = self.submit_counter.value
        succeed = self.success_counter.value
        failed = self.failed_counter.value
        finished = succeed + failed
        running = submitted - finished
        self.logger.info(
            "%s/%s running, finished/succeed/failed: %s/%s/%s", running, submitted, finished, succeed, failed
        )

    def get_input_path_for_ffmpeg(self, input_path: str) -> str:
        """Get the appropriate path for ffmpeg based on input type.

        Args:
            input_path: Input file path (local, HTTP/HTTPS URL, or TOS/S3 URL)

        Returns:
            str: Path that ffmpeg can use directly
        """
        if is_local_path(input_path):
            # Local file path - use as is
            return input_path
        elif input_path.startswith(("http://", "https://")):
            # HTTP/HTTPS URL - use as is
            return input_path
        elif input_path.startswith(("tos://", "s3://")):
            # TOS/S3 URL - need to pre-sign
            return pre_sign_url_for_tos(input_path, expires=360000)
        else:
            # Assume it's a remote path that needs pre-signing
            self.logger.warning("Unknown path type for %s, treating as TOS/S3", input_path)
            return pre_sign_url_for_tos(input_path, expires=360000)

    def pick_audio_map(self, url_path: str) -> list[str]:
        cmd_probe = [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "stream=index:stream_tags=default",
            "-select_streams",
            "a",
            "-of",
            "json",
            url_path,
        ]
        result = subprocess.run(cmd_probe, capture_output=True, text=True)

        # ffprobe 失败 → 兜底用第一条音频
        if result.returncode != 0:
            self.logger.warning(
                "[pick_audio_map] ffprobe failed on %s: %s, fallback to first audio stream",
                url_path,
                result.stderr.strip(),
            )
            return ["-map", "0:a:0"]

        streams = json.loads(result.stdout).get("streams", [])
        if not streams:
            self.logger.info("[pick_audio_map] No audio streams found in %s", url_path)
            return []

        # 遍历 audio 流，优先 default
        for rel_idx, s in enumerate(streams):
            default_tag = str(s.get("tags", {}).get("default", "")).lower()
            if default_tag in {"1", "yes", "true"}:
                self.logger.info(
                    "[pick_audio_map] Selected default audio stream rel_index=%s, global_index=%s from %s",
                    rel_idx,
                    s["index"],
                    url_path,
                )
                return ["-map", f"0:a:{rel_idx}"]

        # fallback 第一个音频流
        self.logger.info(
            "[pick_audio_map] No default audio, fallback to first audio stream rel_index=0, global_index=%s from %s",
            streams[0]["index"],
            url_path,
        )
        return ["-map", "0:a:0"]

    def process(self, input_path: str, output_path: str) -> str | None:
        self.submit_counter.increment()
        url_path = self.get_input_path_for_ffmpeg(input_path)

        tmp_out = None
        try:
            tmp_out = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
            tmp_out.close()

            cmd = ["ffmpeg", "-y", "-i", url_path, "-map", "0:v:0"]

            # 视频参数
            cmd += ["-c:v", self.video_codec]
            if self.crf is not None:
                cmd += ["-crf", str(self.crf)]
            if self.preset is not None:
                cmd += ["-preset", self.preset]
            if self.max_height is not None and self.max_height > 0:
                cmd += ["-vf", f"scale=-2:min(ih\\,{self.max_height})"]

            # 音频参数
            cmd += ["-c:a", self.audio_codec]
            if self.audio_bitrate is not None:
                cmd += ["-b:a", self.audio_bitrate]
            if self.audio_sample_rate is not None:
                cmd += ["-ar", str(self.audio_sample_rate)]

            # 音轨选择
            if self.select_audio == "all":
                cmd += ["-map", "0:a"]
            elif self.select_audio == "auto":
                cmd += self.pick_audio_map(url_path)
            elif self.select_audio == "first":
                cmd += ["-map", "0:a:0"]

            # 其它参数
            cmd += self.extra_params

            # 输出及优化
            cmd += ["-movflags", "+faststart", "-loglevel", "error", tmp_out.name]

            self.logger.info("Running command: %s", " ".join(cmd))
            subprocess.run(cmd, check=True, timeout=self.timeout)

            upload_file(tmp_out.name, output_path)
            self.success_counter.increment()
            self.log_progress()
            self.logger.info("Finished conversion %s → %s", input_path, output_path)
            return output_path

        except Exception as e:
            self.failed_counter.increment()
            self.log_progress()
            self.logger.error("Failed: %s", e)
            return None
        finally:
            if tmp_out and os.path.exists(tmp_out.name):
                try:
                    os.remove(tmp_out.name)
                except Exception as remove_error:
                    self.logger.warning("Failed to remove temp file %s: %s", tmp_out.name, remove_error)

    def transform(self, input_col: pa.Array, output_col: pa.Array) -> pa.Array:
        """将视频文件转换为MP4格式

        Args:
            input_col: 包含输入视频路径的数组（支持本地路径、HTTP/HTTPS URL、TOS/S3 URL）
            output_col: 包含输出MP4文件路径的数组

        Returns:
            pa.Array: 包含转换结果路径的数组，成功返回输出路径，失败返回None
        """  # noqa: D415
        results = []
        for input_path, output_path in zip(input_col.to_pylist(), output_col.to_pylist()):
            try:
                result = self.process(input_path, output_path)
            except Exception as e:
                self.logger.error("[transform] Failed to process %s: %s", input_path, e)
                result = None
            results.append(result)

        return pa.array(results, type=self.__return_column_type__())

    @staticmethod
    def __return_column_type__() -> pa.DataType:
        return pa.large_string()
