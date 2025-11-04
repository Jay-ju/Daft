# Copyright (c) Beijing Volcano Engine Technology Ltd.

from __future__ import annotations

import logging
import random
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

from daft.dependencies import pa
from daft.las.functions.types import Operator
from daft.las.functions.utils.common_utils import generate_filename_base_input, save_file_to_local, tracking_usage

logger = logging.getLogger(__name__)


class AudioAsrFireRed(Operator):
    """**语音识别模块 - 基于 FireRed ASR 模型的多语言语音转文字解决方案**

    **核心功能**
    - 多语言识别：支持中英文、以及中文方言
    - 多模型选择：支持多种模型，包括AED模型和LLM模型
    - 音频类型： 支持单声道、16K采样率的wav音频文件

    **推荐实践**
    - 为保证识别效果，对于AED模型，建议音频长度不超过60s；对于LLM模型，建议音频长度不超过30s
    - 使用LLM进行批量推理时，建议确保音频长度差异不大，如差异较大，建议设置batch_size=1
    - 使用之前建议对音频进行标准化处理，转化成该算子支持的音频文件

    **支持模型**
    - `FireRedASR-AED-L`
    - `FireRedASR-LLM-L`
    """  # noqa: D415

    def __init__(
        self,
        audio_src_type: str,
        model_path: str = "/opt/las/models",
        model_name: str = "FireRedAsr/FireRedASR-AED-L",
        batch_size: int = 1,
        beam_size: int = 3,
        decode_min_len: int = 0,
        decode_max_len: int = 0,
        # aed model
        nbest: int = 1,
        softmax_smoothing: float = 1.25,
        aed_length_penalty: float = 0.6,
        eos_penalty: float = 1.0,
        # llm model
        repetition_penalty: float = 3.0,
        llm_length_penalty: float = 1.0,
        temperature: float = 1.0,
        use_fp16: bool = False,
        **kwargs: Any,
    ) -> None:
        """初始化 Fire Red ASR语音识别模型

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
                支持的FireRed系列模型：
                - FireRedAsr/FireRedASR-AED-L: 小模型
                - FireRedAsr/FireRedASR-LLM-L: 大模型
                可选值：[
                    "FireRedAsr/FireRedASR-AED-L",
                    "FireRedAsr/FireRedASR-LLM-L"
                ]
                默认值："FireRedAsr/FireRedASR-AED-L"
            batch_size: 单次处理的音频样本数量
                默认值：1
            beam_size: 解码时的beam宽度
                控制解码时的搜索空间大小，数值越大，识别准确率可能提升但速度变慢
                默认值：3
            decode_min_len: 最小解码长度
                限制输出文本的最小长度，0表示不限制
                默认值：0
            decode_max_len: 最大解码长度
                限制输出文本的最大长度，0表示不限制
                默认值：0
            nbest: AED模型输出候选数
                控制输出的候选文本数量，仅AED模型有效
                默认值：1
            softmax_smoothing: AED模型softmax平滑系数
                调整softmax分布的平滑程度，仅AED模型有效
                默认值：1.25
            aed_length_penalty: AED模型长度惩罚
                控制输出文本长度的惩罚系数，仅AED模型有效
                默认值：0.6
            eos_penalty: AED模型终止符惩罚
                控制终止符的惩罚系数，仅AED模型有效
                默认值：1.0
            repetition_penalty: LLM模型重复惩罚
                控制生成文本时重复内容的惩罚系数，仅LLM模型有效
                默认值：3.0
            llm_length_penalty: LLM模型长度惩罚
                控制生成文本长度的惩罚系数，仅LLM模型有效
                默认值：1.0
            temperature: 温度系数
                控制生成文本的随机性（0.0-1.0）
                较高值适合创造性场景，较低值适合确定性场景
                默认值：1.0
            use_fp16: 是否使用FP16推理
                是否启用半精度浮点数加速推理，节省显存
                默认值：False
        """  # noqa: D415
        super().__init__(**kwargs)

        self.audio_src_type = audio_src_type
        self.model_path = model_path
        self.model_name = model_name
        self.batch_size = batch_size
        self.beam_size = beam_size
        self.decode_min_len = decode_min_len
        self.decode_max_len = decode_max_len
        self.nbest = nbest
        self.softmax_smoothing = softmax_smoothing
        self.aed_length_penalty = aed_length_penalty
        self.eos_penalty = eos_penalty
        self.repetition_penalty = repetition_penalty
        self.llm_length_penalty = llm_length_penalty
        self.temperature = temperature
        self.use_fp16 = use_fp16

        supported_models_name = ["FireRedAsr/FireRedASR-AED-L", "FireRedAsr/FireRedASR-LLM-L"]
        if self.model_name not in supported_models_name:
            raise ValueError(
                f"model_name {self.model_name} is not supported, supported model names are {supported_models_name}"
            )

        model_dir = str(Path(self.model_path) / self.model_name)
        asr_infer_path = Path(self.model_path) / "FireRedAsr"
        if asr_infer_path.exists():
            if str(asr_infer_path) not in sys.path:
                sys.path.insert(0, str(asr_infer_path))
                logger.info("Added ASR infer path to sys.path: %s", asr_infer_path)
            else:
                logger.debug("ASR infer path already in sys.path: %s", asr_infer_path)
        else:
            raise FileNotFoundError(f"ASR infer path not found: {asr_infer_path}")

        try:
            logger.debug("Loading ASR model from: %s", model_dir)
            from fireredasr.models.fireredasr import FireRedAsr

            model_type = "llm" if "LLM" in model_name else "aed"
            if model_type == "aed":
                self.model = FireRedAsr.from_pretrained(model_type, model_dir)
                self.generate_kwargs = {
                    "use_gpu": 1 if self.use_gpu else 0,
                    "beam_size": self.beam_size,
                    "nbest": self.nbest,
                    "decode_max_len": self.decode_max_len,
                    "softmax_smoothing": self.softmax_smoothing,
                    "aed_length_penalty": self.aed_length_penalty,
                    "eos_penalty": self.eos_penalty,
                }
            else:
                self.use_flash_attn = self.use_fp16
                self.model = FireRedAsr.from_pretrained(model_type, model_dir, self.use_fp16, self.use_flash_attn)
                self.generate_kwargs = {
                    "use_gpu": 1 if self.use_gpu else 0,
                    "beam_size": self.beam_size,
                    "decode_max_len": self.decode_max_len,
                    "decode_min_len": self.decode_min_len,
                    "repetition_penalty": self.repetition_penalty,
                    "llm_length_penalty": self.llm_length_penalty,
                    "temperature": self.temperature,
                }

            logger.info("ASR model loaded successfully on device: %s", "GPU" if self.use_gpu else "CPU")
        except Exception as e:
            logger.exception("Model loading failed - Path: %s", model_dir)
            raise RuntimeError(f"Model loading failed, please check model path: {model_dir}") from e

        tracking_usage(op=self.__class__.__name__, model_service_or_lib=self.model_name)

    def transform(self, contents: pa.Array, contents_ids: pa.Array | None = None) -> pa.Array:
        """对输入的音频内容数组进行批量处理，生成包含音频理解结果的文本描述.

        Args:
            contents: 包含音频数据的数组，元素类型为字符串（文件路径或URL）。
                支持本地文件路径以及tos://、s3://、http://、https://开头的远程文件链接
            contents_ids: 包含音频数据的数组，元素类型为字符串，用于唯一标识音频
                默认值：None

        Returns:
            处理后的数组，元素为每个音频的理解结果文本。对于处理失败的音频，返回空字符串。

        Raises:
            RuntimeError: 模型推理过程中发生错误时抛出（如内存不足）
            Exception: 其他推理相关错误时抛出
        """
        start_time = time.monotonic()

        all_asr_results = []
        total_content = len(contents)
        if contents_ids is None:
            contents_ids = [f"{int(time.time())!s}_{random.randint(1,100000)}" for _ in range(total_content)]

        total_batches = (total_content + self.batch_size - 1) // self.batch_size
        with tempfile.TemporaryDirectory() as temp_dir:
            for batch_idx in range(0, total_content, self.batch_size):
                sub_contents = contents.slice(batch_idx, self.batch_size)
                sub_contents_id = contents_ids[batch_idx * self.batch_size : (batch_idx + 1) * self.batch_size]
                current_batch = sub_contents.to_pylist()
                logger.debug(
                    "Processing batch %.2f with %d audio files", (batch_idx + 1) / total_batches, len(current_batch)
                )

                if not current_batch:
                    break

                try:
                    inputs = []
                    for idx, audio in enumerate(current_batch):
                        file_name = generate_filename_base_input(audio, self.audio_src_type, "wav", batch_idx, idx)
                        tmp_file_name = save_file_to_local(audio, self.audio_src_type, temp_dir, file_name)
                        logger.info("Downloading audio to %s", tmp_file_name)
                        if not Path(tmp_file_name).exists():
                            raise FileNotFoundError(tmp_file_name)

                        inputs.append(tmp_file_name)

                    results = self.model.transcribe(sub_contents_id, inputs, self.generate_kwargs)
                    batch_asr_results = [result["text"] for result in results]
                    all_asr_results.extend(batch_asr_results)

                except RuntimeError:
                    logger.exception(
                        "Model inference failed (possibly OOM)! Current batch size: %d, consider reducing batch_size.",
                        self.batch_size,
                    )
                    all_asr_results.extend([""] * len(sub_contents))
                except Exception:
                    logger.exception("Inference error!")
                    all_asr_results.extend([""] * len(sub_contents))

        processing_time = time.monotonic() - start_time

        logger.info(
            "Completed %d contents | Total time: %.2fs | Throughput: %.2f contents/s",
            total_content,
            processing_time,
            total_content / processing_time,
        )

        return pa.array(all_asr_results, type=self.__return_column_type__())

    @staticmethod
    def __return_column_type__() -> pa.DataType:
        return pa.string()
