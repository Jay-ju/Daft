# Copyright (c) Beijing Volcano Engine Technology Ltd.

from __future__ import annotations

import logging
import random
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np  # noqa: TID253

from daft.dependencies import pa
from daft.las.functions.types import Operator
from daft.las.functions.utils.common_utils import save_file_to_local

logger = logging.getLogger(__name__)


class AudioVadFsmn(Operator):
    """**语音端点检测模块 - 基于 FSMN VAD 的高效音频分割解决方案**

    **核心功能**
    - 语音端点检测：自动识别音频中的语音片段起止时间，实现语音与静音的精准分割
    - 批量处理：支持大批量音频数据的高效端点检测
    - 多格式输入：兼容原始二进制、Base64 编码、TOS/HTTP 链接等多种音频输入方式
    - GPU 加速：支持 GPU 环境下的高性能推理

    **推荐实践**
    - 建议输入 16k 采样率、单声道的 WAV 格式音频，提升检测准确率
    - 长音频建议分段处理，单次处理时长建议不超过 1 小时
    - 适用于语音活动检测、语音切分等场景

    **支持模型**
    - `iic/speech_fsmn_vad_zh-cn-16k-common-pytorch`（中文通用 FSMN VAD）

    **输出说明**
    - 每条音频输出二维浮点数列表，表示所有语音片段的起止时间戳（单位：秒），如：[[0.0, 4.34], [5.50, 7.12]]
    - 处理失败时返回 None
    """  # noqa: D415

    def __init__(
        self,
        audio_src_type: str,
        batch_size_s: int = 3600,
        model_path: str = "/opt/las/models",
        model_name: str = "iic/speech_fsmn_vad_zh-cn-16k-common-pytorch",
        model_revision: str = "v2.0.4",
        rank: int = 0,
        **kwargs: Any,
    ) -> None:
        """语音端点识别参数初始化.

        Args:
            audio_src_type: 音频格式类型
                支持的音频格式类型，包含：
                - tos/http 地址(audio_url)
                - base64 编码(audio_base64)
                - 二进制流(audio_binary)
                可选值：["audio_binary", "audio_url", "audio_base64"]
            model_path: 模型路径
                本地模型存储路径。
                默认值："/opt/las/models"
            model_name: 模型名称
                使用的模型名称，包含：
                - iic/speech_fsmn_vad_zh-cn-16k-common-pytorch
                可选值：["iic/speech_fsmn_vad_zh-cn-16k-common-pytorch"]
                默认值："iic/speech_fsmn_vad_zh-cn-16k-common-pytorch"
            model_revision: 模型版本
                指定模型的版本号，包含：
                - v2.0.4
                可选值：["v2.0.4"]
                默认值："v2.0.4"
            batch_size_s: 批量计算的秒数
                每批次处理的音频时长（秒），仅在使用GPU时生效。
                默认值：3600
            rank: 指定使用的GPU设备编号（多卡环境有效）。例如：0表示第一张GPU，1表示第二张GPU
                默认值：None
        """
        super().__init__(**kwargs)
        self.audio_src_type = audio_src_type
        self.model_name = model_name
        self.model_revision = model_revision
        self.batch_size_s = batch_size_s
        self.rank = rank

        import torch

        self.use_gpu = self.use_gpu and torch.cuda.is_available()
        if self.use_gpu:
            rank = 0 if self.rank is None else self.rank
            self.rank = rank % self.cuda_device_count
            self.device = f"cuda:{self.rank}"
        else:
            self.device = "cpu"
        logger.info("Model will be loaded on device: %s", self.device)

        from funasr import AutoModel

        model_dir = str(Path(model_path) / self.model_name)
        self.model = AutoModel(
            model=model_dir,
            model_revision=self.model_revision,
            disable_update=True,
            device=self.device,
        )

        logger.info(
            "Model initialization configuration:\n"
            "- Model Name: %s\n"
            "- Storage Path: %s\n"
            "- Device: %s\n"
            "- GPU number: %d\n"
            "- Audio type: %s\n",
            self.model_name,
            model_dir,
            self.device,
            self.cuda_device_count if self.use_gpu else 0,
            self.audio_src_type,
        )

    def transform(self, videos: pa.Array) -> pa.Array:
        """批量处理音频数组，提取语音端点时间戳.

        Args:
            videos: 包含音频数据的列，支持以下格式：
                - audio_base64: base64 编码的音频字符串
                - audio_url: 音频文件的 URL 路径
                - audio_binary: 原始音频字节数据
        Returns:
            包含语音端点时间戳的列，每个元素为二维浮点数列表，表示音频中各语音片段的起止时间戳（单位：秒），
            例如：[[0.0, 4.34], [5.50, 7.12]]。若处理失败则为 None。
        """
        current_videos = videos.to_pylist()
        audio_timestamps_total: list[list[list[Any]] | None] = []

        with tempfile.TemporaryDirectory(dir="/tmp") as tmp_dir:
            for audio in current_videos:
                try:
                    audio_file_name = f"{int(time.time())}_{random.randint(1, 1000000)}.wav"
                    tmp_file_name = save_file_to_local(audio, self.audio_src_type, tmp_dir, audio_file_name)
                    logger.info("Downloading audio to %s", tmp_file_name)
                    if not Path(tmp_file_name).exists():
                        raise FileNotFoundError(tmp_file_name)

                    res = self.model.generate(input=tmp_file_name, batch_size_s=self.batch_size_s)
                    timestamps = res[0]["value"]
                    timestamps_second = [list(x) for x in np.array(timestamps) / 1000]
                    audio_timestamps_total.append(timestamps_second)
                except Exception as e:
                    logger.error("Error when processing audio %s: %s", audio, e)
                    audio_timestamps_total.append(None)

            return pa.array(audio_timestamps_total, type=self.__return_column_type__())

    @staticmethod
    def __return_column_type__() -> pa.DataType:
        return pa.list_(pa.list_(pa.float32()))
