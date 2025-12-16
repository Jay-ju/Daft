from __future__ import annotations

import os
import subprocess
import tempfile
from typing import Any

from daft.dependencies import pa
from daft.las.functions.types import Operator
from daft.las.functions.utils.common_utils import FastWriteCounter, get_logger, tracking_usage
from daft.las.io import download_file


class VideoDetectAudio(Operator):
    """视频音频检测处理器

    特点：
    - 使用 ffprobe 检测视频中是否存在音频流
    - 自动下载远程文件
    - 支持超时控制
    - 支持多种视频格式
    - 返回布尔值表示是否存在音频
    """  # noqa: D415

    def __init__(
        self,
        timeout: int | None = None,
        **kwargs: Any,
    ) -> None:
        """初始化视频音频检测处理器.

        Args:
            timeout: ffprobe执行超时时间（秒），默认为None（无超时）
            **kwargs: 其他参数
        """
        super().__init__(**kwargs)
        self.timeout = timeout

        self.submit_counter = FastWriteCounter()
        self.success_counter = FastWriteCounter()
        self.failed_counter = FastWriteCounter()

        self.logger = get_logger(f"VideoDetectAudio-{id(self)}")

        tracking_usage(op=self.__class__.__name__, model_service_or_lib="ffprobe")

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

    def process(self, input_path: str) -> bool | None:
        """检测单个视频中是否存在音频

        Args:
            input_path: 输入视频路径（远程）

        Returns:
            True 表示存在音频，False 表示不存在音频，None 表示检测失败
        """  # noqa: D415
        self.submit_counter.increment()
        tmp_in = None

        try:
            # 下载远程文件到本地临时文件
            tmp_in = tempfile.NamedTemporaryFile(suffix=os.path.splitext(input_path)[1] or ".tmp", delete=False)
            tmp_in.close()
            # self.logger.info("Downloading remote file: %s -> %s", input_path, tmp_in.name)
            download_file(input_path, tmp_in.name)

            # 使用 ffprobe 检测音频流
            # 命令说明：
            # -v error: 只显示错误信息
            # -select_streams a: 只选择音频流
            # -show_entries stream=codec_type: 只显示流类型
            # -of csv=p=0: 输出格式为csv，不显示键名
            cmd = [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "a",
                "-show_entries",
                "stream=codec_type",
                "-of",
                "csv=p=0",
                tmp_in.name,
            ]

            result = subprocess.run(cmd, check=False, timeout=self.timeout, capture_output=True, text=True)

            # 如果输出包含 "audio"，说明存在音频流
            has_audio = "audio" in result.stdout.lower()

            self.success_counter.increment()
            self.log_progress()
            self.logger.info("Finished detecting audio for %s, has_audio=%s", input_path, has_audio)
            return has_audio

        except Exception as e:
            self.failed_counter.increment()
            self.log_progress()
            self.logger.error("[VideoDetectAudio] Failed processing %s: %s", input_path, e)
            return None
        finally:
            if tmp_in and os.path.exists(tmp_in.name):
                os.remove(tmp_in.name)

    def transform(self, input_col: pa.Array) -> pa.Array:
        """批量检测视频中是否存在音频

        Args:
            input_col: 输入视频路径列

        Returns:
            布尔值的 PyArrow Array，True 表示存在音频，False 表示不存在，None 表示检测失败
        """  # noqa: D415
        results = []
        for input_path in input_col.to_pylist():
            results.append(self.process(input_path))
        return pa.array(results, type=self.__return_column_type__())

    @staticmethod
    def __return_column_type__() -> pa.DataType:
        return pa.bool_()
