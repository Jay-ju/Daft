# Copyright (c) Beijing Volcano Engine Technology Ltd.

from __future__ import annotations

import logging
import random
import tempfile
import time
from pathlib import Path
from typing import Any

from daft.dependencies import pa
from daft.las.functions.types import Operator
from daft.las.functions.utils.common_utils import save_file_to_local, tracking_usage

logger = logging.getLogger(__name__)


class AudioVadSilero(Operator):
    """**语音端点检测模块 - 基于 Silero VAD 的高效音频分割解决方案**

    **核心功能**
    - 语音端点检测：自动识别音频中的语音片段起止时间，实现语音与静音的精准分割
    - 批量处理：支持大批量音频数据的高效端点检测
    - 多格式输入：兼容原始二进制、Base64 编码、TOS/HTTP 链接等多种音频输入方式
    - GPU 加速：支持 GPU 环境下的高性能推理
    - 通用性强：在处理不同领域、存在各种背景噪声和质量水平的音频时表现优异

    **推荐实践**
    - 建议输入 8k/16k 采样率、单声道的 WAV 格式音频，提升检测准确率
    - GPU提速不明显，在不要求极致性能的场景中，可以使用CPU推理onnx模型
    - 适用于语音活动检测、语音切分等场景

    **支持模型**
    - silero_vad.onnx (对应 onnx_model_revision=16)
    - silero_vad_16k_op15.onnx (对应 onnx_model_revision=15)
    - silero_vad.jit

    **输出说明**
    - 每条音频输出二维浮点数列表，表示所有语音片段的起止时间戳（单位：秒），如：[[0.0, 4.34], [5.50, 7.12]]
    - 处理失败时返回 None
    """  # noqa: D415

    def __init__(
        self,
        audio_src_type: str,
        model_path: str = "/opt/las/models",
        model_name: str = "silero-vad",
        use_onnx_model: bool = True,
        onnx_model_revision: int = 16,
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
                - silero-vad
                可选值：["silero-vad"]
                默认值："silero-vad"
            use_onnx_model: 是否使用onnx模型
                默认值：True
            onnx_model_revision: onnx模型版本
                可选值：[16, 15]
                默认值：16
        """
        super().__init__(**kwargs)
        self.audio_src_type = audio_src_type
        self.model_name = model_name
        self.use_onnx_model = use_onnx_model
        self.onnx_model_revision = onnx_model_revision

        import torch

        self.use_gpu = self.use_gpu and torch.cuda.is_available()
        if self.use_gpu:
            self.device = "cuda"
        else:
            self.device = "cpu"
        logger.info("Model will be loaded on device: %s", self.device)

        import torch

        model_dir = str(Path(model_path) / self.model_name)
        if self.use_onnx_model:
            self.model, utils = torch.hub.load(
                repo_or_dir=model_dir,
                model="silero_vad",
                source="local",
                force_reload=True,
                onnx=True,
                opset_version=self.onnx_model_revision,
                force_onnx_cpu=True if self.device == "cpu" else False,
            )
        else:
            self.model, utils = torch.hub.load(
                repo_or_dir=model_dir, model="silero_vad", source="local", force_reload=True
            )
            self.model = self.model.to(self.device)

        (self.get_speech_timestamps, _, self.read_audio, _, _) = utils

        logger.info(
            "Model initialization configuration:\n"
            "- Model Name: %s\n"
            "- Use ONNX Model: %s\n"
            "- onnx version: %s\n"
            "- Storage Path: %s\n"
            "- Device: %s\n"
            "- Audio type: %s\n",
            self.model_name,
            self.use_onnx_model,
            self.onnx_model_revision,
            model_dir,
            self.device,
            self.audio_src_type,
        )

        tracking_usage(op=self.__class__.__name__, model_service_or_lib=self.model_name)

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
                    audio = self.read_audio(tmp_file_name)

                    if not self.use_onnx_model:
                        audio = audio.to(self.device)

                    res = self.get_speech_timestamps(
                        audio,
                        self.model,
                        return_seconds=True,
                    )
                    timestamps_second = [[x["start"], x["end"]] for x in res]
                    audio_timestamps_total.append(timestamps_second)
                except Exception as e:
                    logger.error("Error when processing audio %s: %s", audio, e)
                    audio_timestamps_total.append(None)

            return pa.array(audio_timestamps_total, type=self.__return_column_type__())

    @staticmethod
    def __return_column_type__() -> pa.DataType:
        return pa.list_(pa.list_(pa.float32()))
