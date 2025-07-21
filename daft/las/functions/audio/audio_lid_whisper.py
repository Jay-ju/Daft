# Copyright (c) Beijing Volcano Engine Technology Ltd.

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

from daft.dependencies import pa
from daft.las.functions.types import Operator
from daft.las.functions.utils.audio_utils import decode_audio, encode_audio

logger = logging.getLogger(__name__)


class AudioLidWhisper(Operator):
    """**AudioLidWhisper 音频语言识别处理器**

    **核心功能**
    - 音频解码：支持多种格式音频 → 16kHz 波形数据
    - 语言检测：基于Whisper-large模型识别几十种语言代码
    - 全称映射：自动转换语言代码为完整语言名称

    **技术特性**
    - 基于ModelScope框架加载模型
    - 16kHz采样率音频预处理流水线

    **典型应用场景**
    - ✅ 语音内容分析 - 多语种音频分类
    - ✅ 流媒体处理 - 实时语言识别
    - ✅ 语音数据集 - 自动化语言标注

    **建议**
    - 先将上游音频统一标准化成 wav 格式，再进行语种识别，有助于提升准确性。
    """  # noqa: D415

    def __init__(
        self,
        model_path: str = "/opt/las/models",
        model_name: str = "iic/speech_whisper-large_lid_multilingual_pytorch",
        model_version: str = "v2.0.4",
        rank: int = 0,
        **kwargs: Any,
    ) -> None:
        """初始化Whisper语言识别模型.

        Args:
            model_path: 模型存储根路径
                默认值："/opt/las/models"
            model_name: 预训练模型名称
                默认值："iic/speech_whisper-large_lid_multilingual_pytorch"
            model_version: 模型版本标识
                默认值："v2.0.4"
            rank: GPU设备标识
                默认值：0
        """
        super().__init__(**kwargs)

        self.model_path = model_path
        self.model_name = model_name
        self.model_version = model_version
        self.rank = rank

        # These packages are heavy, so we import them lazily.
        import torch

        model_dir = Path(self.model_path) / self.model_name
        if not model_dir.exists():
            raise FileNotFoundError(f"Model directory not found: {model_dir}")

        use_gpu = self.use_gpu and torch.cuda.is_available()
        self.model_device: str = "cpu"
        if self.rank is None:
            self.model_device = "cuda" if use_gpu else "cpu"
        else:
            self.model_device = f"cuda:{self.rank % self.cuda_device_count}" if use_gpu else "cpu"
        logger.info("Model will be loaded on device: %s", self.model_device)

        from modelscope.pipelines import pipeline
        from modelscope.utils.constant import Tasks

        self._inference_pipeline = pipeline(
            task=Tasks.auto_speech_recognition,
            model=str(model_dir),
            model_revision=self.model_version,
            device=self.model_device,
            disable_update=True,
        )

        logger.info(
            "Model initialization configuration:\n" "- Model Name: %s\n" "- Storage Path: %s\n" "- Device: %s\n",
            self.model_name,
            model_dir,
            self.model_device,
        )

        with open(str(model_dir / "whisper_language_map.json")) as f:
            self.language_map = json.load(f)[0]
        logger.info("Language map: %s", self.language_map)

    def transform(self, audios: pa.Array) -> pa.Array:
        """批量处理音频数组生成语言识别结果.

        该方法使用预训练的Whisper-large模型对输入音频进行语言检测，
        返回包含语言代码和完整语言名称的结构化数据

        Args:
            audios: 包含音频数据的数组
                支持格式：
                - 原始音频字节数据
                - 音频文件路径，比如：TOS url, http url, 本地文件路径

        Returns:
            pyarrow.Array: 结构化数组，每个元素包含：
                - language_code: 语言代码
                - language_code_full_name: 语言完整英文名称

        Raises:
            Exception: 音频处理异常时记录错误日志
        """
        start_time = time.monotonic()

        total_audios_lid = []
        total_audios_lid_full_name = []
        total_audios = len(audios)
        for current_audio in audios:
            try:
                current_audio = current_audio.as_py()
                current_audio = decode_audio(current_audio, sample_rate=16000)
                current_audio = encode_audio(current_audio)
                audio_lid = self._inference_pipeline(input=current_audio)[0]["lid"].strip("<").strip(">")
                if audio_lid in self.language_map:
                    audio_lid_full_name = self.language_map[audio_lid]
                else:
                    audio_lid_full_name = "unknown"
                total_audios_lid.append(audio_lid)
                total_audios_lid_full_name.append(audio_lid_full_name)
            except Exception:
                logger.exception("Inference error!")
                total_audios_lid.append(None)
                total_audios_lid_full_name.append(None)

        processing_time = time.monotonic() - start_time

        logger.info(
            "Completed %d images | Total time: %.2fs | Throughput: %.2f content/s",
            total_audios,
            processing_time,
            total_audios / processing_time,
        )

        total_result_details = []
        for i in range(total_audios):
            record = {"language_code": total_audios_lid[i], "language_code_full_name": total_audios_lid_full_name[i]}
            total_result_details.append(record)

        return pa.array(total_result_details, type=self.__return_column_type__())

    @staticmethod
    def __return_column_type__() -> pa.DataType:
        fields = [pa.field("language_code", pa.string()), pa.field("language_code_full_name", pa.string())]
        return pa.struct(fields)
