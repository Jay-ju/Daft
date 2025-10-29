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


class AudioConvertToMp3(Operator):
    """**音频格式转换处理器，将各种音频格式转换为MP3**

    **核心功能：**
    - 支持多种音频格式转换为MP3
    - 音频质量和编码参数自定义
    - 支持音频采样率、比特率精细控制
    - 智能音轨选择（全部/自动/首个）
    - 支持本地文件、HTTP/HTTPS URL和TOS/S3存储

    **格式支持：**
    - 输入：WAV、FLAC、AAC、M4A、OGG、WMA、APE等主流音频格式
    - 输出：MP3 (.mp3)
    - 音频编解码器：LAME MP3编码器
    - 采样率：8kHz-96kHz
    - 比特率：32kbps-320kbps
    """  # noqa: D415

    def __init__(
        self,
        bitrate: str = "192k",
        sample_rate: int | None = None,
        audio_map: str = "all",
        quality: int = 2,
        extra_params: list[str] | None = None,
        timeout: int | None = None,
        **kwargs: Any,
    ) -> None:
        """初始化音频转换为MP3算子

        Args:
            bitrate: 音频比特率，如"128k"、"192k"、"256k"、"320k"
                默认值："192k"
            sample_rate: 音频采样率，如22050、44100、48000，为None时保持原始采样率
                默认值：None
            audio_map: 音轨选择策略：
                - "all"：选择所有音轨（默认）
                - "auto"：智能选择默认音轨（自动检测default标记）
                - "first"：仅选择第一个音轨
                默认值："all"
            quality: MP3编码质量，取值范围0-9，0最高质量最慢，9最低质量最快
                默认值：2
            extra_params: 额外的ffmpeg参数列表，如["-ac", "1"]
                默认值：None
            timeout: 单个音频处理超时时间（秒），为None时不限制
                默认值：None
        """  # noqa: D415
        super().__init__(**kwargs)
        self.bitrate = bitrate
        self.sample_rate = sample_rate
        self.audio_map = audio_map
        self.quality = quality
        self.extra_params = extra_params or []
        self.timeout = timeout

        self.submit_counter = FastWriteCounter()
        self.success_counter = FastWriteCounter()
        self.failed_counter = FastWriteCounter()

        self.logger = get_logger(f"AudioConvertToMp3-{id(self)}")

        self.logger.info(
            "AudioConvertToMp3 initialized with bitrate=%s, sample_rate=%s, "
            "audio_map=%s, quality=%s, extra_params=%s",
            bitrate,
            sample_rate,
            audio_map,
            quality,
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
        """Intelligently select audio stream based on default tag.

        Uses ffprobe to detect audio streams and selects the one marked as default.
        Falls back to first audio stream if no default is found or ffprobe fails.

        Args:
            url_path: Pre-signed URL or local path for ffprobe

        Returns:
            list[str]: ffmpeg map arguments, e.g., ["-map", "0:a:0"]
        """
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

        # ffprobe failed → fallback to first audio stream
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

        # Iterate audio streams, prioritize default
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

        # Fallback to first audio stream
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
            tmp_out = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
            tmp_out.close()

            cmd = ["ffmpeg", "-y", "-i", url_path]

            # Audio track selection
            if self.audio_map == "all":
                cmd += ["-map", "0:a"]
            elif self.audio_map == "auto":
                cmd += self.pick_audio_map(url_path)
            elif self.audio_map == "first":
                cmd += ["-map", "0:a:0"]

            # 音频编码器 - 使用libmp3lame
            cmd += ["-c:a", "libmp3lame"]

            # 比特率设置
            cmd += ["-b:a", self.bitrate]

            # MP3编码质量
            cmd += ["-q:a", str(self.quality)]

            # 采样率设置
            if self.sample_rate is not None:
                cmd += ["-ar", str(self.sample_rate)]

            # 其它参数
            cmd += self.extra_params

            # 输出设置
            cmd += ["-loglevel", "error", tmp_out.name]

            self.logger.info("[AudioConvertToMp3] Running command: %s", " ".join(cmd))
            subprocess.run(cmd, check=True, timeout=self.timeout)

            upload_file(tmp_out.name, output_path)
            self.success_counter.increment()
            self.log_progress()
            self.logger.info("Finished conversion %s → %s", input_path, output_path)
            return output_path

        except Exception as e:
            self.failed_counter.increment()
            self.log_progress()
            self.logger.error("[AudioConvertToMp3] Failed: %s", e)
            return None
        finally:
            if tmp_out and os.path.exists(tmp_out.name):
                try:
                    os.remove(tmp_out.name)
                except Exception as remove_error:
                    self.logger.warning("Failed to remove temp file %s: %s", tmp_out.name, remove_error)

    def transform(self, input_col: pa.Array, output_col: pa.Array) -> pa.Array:
        """将音频文件转换为MP3格式

        Args:
            input_col: 包含输入音频路径的数组（支持本地路径、HTTP/HTTPS URL、TOS/S3 URL）
            output_col: 包含输出MP3文件路径的数组

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
