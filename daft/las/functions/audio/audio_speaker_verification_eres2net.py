# Copyright (c) Beijing Volcano Engine Technology Ltd.

from __future__ import annotations

import logging
import random
import tempfile
import time
from pathlib import Path
from typing import Any

import librosa

from daft.dependencies import pa
from daft.las.functions.types import Operator
from daft.las.functions.utils.common_utils import log_op_call, save_file_to_local

logger = logging.getLogger(__name__)


class AudioSpeakerVerificationEres2net(Operator):
    """**AudioSpeakerVerificationEres2net 音频说话人验证处理器**

    **核心功能**
    - 说话人验证：判断两段音频是否为同一说话人，输出相似度分数
    - 多源音频支持：支持本地文件、URL、对象存储、base64、binary等多种音频输入类型
    - 批量处理：支持批量音频对的高效验证与异常处理

    **技术特性**
    - 基于 ModelScope 框架加载 ERes2Net 说话人验证模型
    - 支持 GPU 多卡推理，自动选择或指定设备
    - 16kHz 采样率音频预处理，兼容多种音频格式

    **典型应用场景**
    - ✅ 语音身份认证 - 说话人一致性验证
    - ✅ 智能客服 - 多通道语音归一化与核查
    - ✅ 语音数据清洗 - 自动化说话人去重

    **建议**
    - 推荐将输入音频统一采样率为16kHz，提升验证准确性
    - 多卡环境下可通过 rank 参数灵活指定 GPU 设备
    """  # noqa: D415

    def __init__(
        self,
        audio_src_type: str,
        model_path: str = "/opt/las/models",
        model_name: str = "iic/speech_eres2net_sv_zh-cn_16k-common",
        rank: int | None = None,
        **kwargs: Any,
    ) -> None:
        """初始化 ERes2Net 说话人验证模型.

        Args:
            audio_src_type: 音频格式类型
                支持的音频格式类型，包含：
                    - tos/http 地址(audio_url)
                    - base64 编码(audio_base64)
                    - 二进制流(audio_binary)
                可选值：["audio_binary", "audio_url", "audio_base64"]
            model_path: 模型存储路径
                默认值："/opt/las/models"
            model_name: 模型名称
                默认值："iic/speech_eres2net_sv_zh-cn_16k-common"
            rank: 指定使用的GPU设备编号（多卡环境有效）
                例如：0表示第一张GPU，1表示第二张GPU。默认值：None（自动选择可用设备）
                默认值：None
        """
        super().__init__(**kwargs)

        self.audio_src_type = audio_src_type
        self.model_path = model_path
        self.model_name = model_name
        self.rank = rank

        model_dir = Path(self.model_path) / self.model_name
        if not model_dir.exists():
            raise FileNotFoundError(f"Model directory not found: {model_dir}")

        # These packages are heavy, so we import them lazily.
        import torch
        from modelscope.pipelines import pipeline

        # Device initialization
        use_gpu = self.use_gpu and torch.cuda.is_available()
        if self.rank is None:
            self.device = "cuda" if use_gpu else "cpu"
        else:
            self.device = f"cuda:{self.rank % self.cuda_device_count}" if use_gpu else "cpu"
        logger.info("Model will be loaded on device: %s", self.device)

        self.sv_pipeline = pipeline(
            task="speaker-verification",
            model=str(model_dir),
            model_revision="v1.0.5",
            device=self.device,
        )

        logger.info(
            "Model initialization configuration:\n"
            "- Model Name: %s\n"
            "- Storage Path: %s\n"
            "- Device: %s\n"
            "- GPU number: %d\n",
            self.model_name,
            model_dir,
            self.device,
            self.cuda_device_count if use_gpu else 0,
        )

        log_op_call(logger=logger, op=self.__class__.__name__, model_service_or_lib=self.model_name)

    def transform(self, speaker_a_audios: pa.Array, speaker_b_audios: pa.Array) -> pa.Array:
        """批量处理音频对并生成说话人验证分数.

        该方法对输入的两组音频（A/B）进行一一配对，使用预训练的 ERes2Net 说话人验证模型判断每对音频是否为同一说话人，
        返回每对音频的相似度分数。支持多种音频输入类型（如 URL、本地路径、二进制数据），自动处理音频下载、采样率转换等预处理流程。
        处理过程中自动捕获异常并记录日志，异常样本返回 None。

        Args:
            speaker_a_audios: 说话人A的音频
                支持格式：
                - 音频文件路径（如 TOS url, S3 url, http(s) url, 本地路径）
                - 音频二进制数据
                - 音频base64编码
            speaker_b_audios: 说话人B的音频
                支持格式同上

        Returns:
            每对音频的说话人相似度分数（float），异常样本为 None

        Raises:
            FileNotFoundError: 音频文件不存在时记录错误日志
            RuntimeError: 模型推理失败（如显存不足）时记录错误日志
            Exception: 其他异常时记录错误日志
        """
        start_time = time.monotonic()
        logger.info("Starting batch processing, input type: %s", self.audio_src_type)

        all_scores = []
        total_audios = len(speaker_a_audios)
        speaker_a_audios = speaker_a_audios.to_pylist()
        speaker_b_audios = speaker_b_audios.to_pylist()

        with tempfile.TemporaryDirectory(dir="/tmp") as tmp_dir:
            for audio_a, audio_b in zip(speaker_a_audios, speaker_b_audios):
                try:
                    if self.audio_src_type == "audio_url":
                        logger.info("speaker a url is %s, speaker b url is %s.", audio_a, audio_b)
                        if audio_a and audio_a.startswith(("tos://", "s3://", "/")):
                            file_name_a = audio_a.split("/")[-1]
                            file_name_b = audio_b.split("/")[-1]
                        elif audio_a and audio_a.startswith(("https://", "http://")):
                            file_name_a = f"{int(time.time())!s}_{random.randint(0, 1000000)}_a.wav"
                            file_name_b = f"{int(time.time())!s}_{random.randint(0, 1000000)}_b.wav"
                        else:
                            raise ValueError("audio_url must be startswith https:// | http:// | / | tos:// | s3://")
                    else:
                        file_name_a = "video_binary_a"
                        file_name_b = "video_binary_b"

                    tmp_file_name_a = save_file_to_local(audio_a, self.audio_src_type, tmp_dir, file_name_a)
                    tmp_file_name_b = save_file_to_local(audio_b, self.audio_src_type, tmp_dir, file_name_b)

                    logger.info("Downloading speaker a audio to %s", tmp_file_name_a)
                    logger.info("Downloading speaker b audio to %s", tmp_file_name_b)

                    if not Path(tmp_file_name_a).exists() or not Path(tmp_file_name_b).exists():
                        raise FileNotFoundError(tmp_file_name_a + " or " + tmp_file_name_b)

                    tmp_file_a_numpy, _ = librosa.load(tmp_file_name_a, sr=16000, mono=True)
                    tmp_file_b_numpy, _ = librosa.load(tmp_file_name_b, sr=16000, mono=True)
                    verification_result = self.sv_pipeline([tmp_file_a_numpy, tmp_file_b_numpy])

                    score = verification_result["score"]
                    all_scores.append(score)
                except FileNotFoundError:
                    logger.exception("File not Found!")
                    all_scores.append(None)
                except RuntimeError:
                    logger.exception("Model inference failed (possibly OOM)!")
                    all_scores.append(None)
                except Exception:
                    logger.exception("Inference error!")
                    all_scores.append(None)

        processing_time = time.monotonic() - start_time

        logger.info(
            "Completed %d audios | Total time: %.2fs | Throughput: %.2f audio/s",
            total_audios,
            processing_time,
            total_audios / processing_time,
        )
        return pa.array(all_scores, type=self.__return_column_type__())

    @staticmethod
    def __return_column_type__() -> pa.DataType:
        return pa.float32()
