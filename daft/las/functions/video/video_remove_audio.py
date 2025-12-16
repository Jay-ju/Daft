from __future__ import annotations

import os
import subprocess
import tempfile
from typing import Any

from daft.dependencies import pa
from daft.las.functions.types import Operator
from daft.las.functions.utils.common_utils import FastWriteCounter, get_logger, tracking_usage
from daft.las.io import download_file, upload_file


class VideoRemoveAudio(Operator):
    """视频音轨移除处理器

    特点：
    - 使用 ffmpeg subprocess 方式移除视频音轨
    - 保留原始视频编码和质量（无需重编码）
    - 自动下载远程文件 / 上传结果
    - 支持超时控制
    - 支持多种视频格式
    """  # noqa: D415

    def __init__(
        self,
        output_format: str | None = None,
        extra_params: list[str] | None = None,
        timeout: int | None = None,
        **kwargs: Any,
    ) -> None:
        """初始化视频音轨移除处理器

        Args:
            output_format: 输出视频格式，如"mp4"、"avi"、"mkv"，为None时保持原格式
                默认值：None
            extra_params: 额外的ffmpeg参数列表，如["-preset", "fast"]
                默认值：None
            timeout: ffmpeg执行超时时间（秒），默认为None（无超时）
            **kwargs: 其他参数
        """  # noqa: D415
        super().__init__(**kwargs)
        self.output_format = output_format
        self.extra_params = extra_params or []
        self.timeout = timeout

        self.submit_counter = FastWriteCounter()
        self.success_counter = FastWriteCounter()
        self.failed_counter = FastWriteCounter()

        self.logger = get_logger(f"VideoRemoveAudio-{id(self)}")

        tracking_usage(op=self.__class__.__name__, model_service_or_lib="ffmpeg")

    def log_progress(self) -> None:
        """记录处理进度."""
        submitted = self.submit_counter.value
        succeed = self.success_counter.value
        failed = self.failed_counter.value
        finished = succeed + failed
        running = submitted - finished
        self.logger.info(
            "%s/%s running, finished/succeed/failed: %s/%s/%s", running, submitted, finished, succeed, failed
        )

    def process(self, input_path: str, output_path: str) -> str | None:
        """处理单个视频移除音轨任务

        Args:
            input_path: 输入视频路径（远程）
            output_path: 输出视频路径（远程）

        Returns:
            成功返回输出路径，失败返回 None
        """  # noqa: D415
        self.submit_counter.increment()
        tmp_in = None
        tmp_out = None

        try:
            # 下载远程文件到本地临时文件
            tmp_in = tempfile.NamedTemporaryFile(suffix=os.path.splitext(input_path)[1] or ".tmp", delete=False)
            tmp_in.close()
            self.logger.info("Downloading remote file: %s -> %s", input_path, tmp_in.name)
            download_file(input_path, tmp_in.name)

            # Determine output file extension
            if self.output_format:
                suffix = f".{self.output_format}"
            else:
                # Extract extension from output_path
                _, ext = os.path.splitext(output_path)
                suffix = ext if ext else ".mp4"

            tmp_out = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
            tmp_out.close()

            cmd = [
                "ffmpeg",
                "-y",
                "-i",
                tmp_in.name,
                "-map",
                "0:v",
                "-c:v",
                "copy",
            ]

            # Add any extra parameters
            if self.extra_params:
                cmd.extend(self.extra_params)

            # Output settings
            cmd.extend(["-loglevel", "error", tmp_out.name])

            self.logger.info("[VideoRemoveAudio] Running ffmpeg command")
            subprocess.run(cmd, check=True, timeout=self.timeout, capture_output=True)

            upload_file(tmp_out.name, output_path)
            self.success_counter.increment()
            self.log_progress()
            self.logger.info("Finished removing audio %s → %s", input_path, output_path)
            return output_path

        except Exception as e:
            self.failed_counter.increment()
            self.log_progress()
            self.logger.error("[VideoRemoveAudio] Failed processing %s: %s", input_path, e)
            return None
        finally:
            if tmp_in and os.path.exists(tmp_in.name):
                os.remove(tmp_in.name)
            if tmp_out and os.path.exists(tmp_out.name):
                os.remove(tmp_out.name)

    def transform(self, input_col: pa.Array, output_col: pa.Array) -> pa.Array:
        """批量处理视频移除音轨

        Args:
            input_col: 输入视频路径列
            output_col: 输出视频路径列

        Returns:
            输出路径的 PyArrow Array
        """  # noqa: D415
        results = []
        for input_path, output_path in zip(input_col.to_pylist(), output_col.to_pylist()):
            results.append(self.process(input_path, output_path))
        return pa.array(results, type=self.__return_column_type__())

    @staticmethod
    def __return_column_type__() -> pa.DataType:
        return pa.large_string()
