# Copyright (c) Beijing Volcano Engine Technology Ltd.

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from daft.dependencies import pa
from daft.las.functions.types import Operator
from daft.las.functions.utils.common_utils import (
    base64_to_byte,
    path_to_byte,
    run_on_local_path,
)

logger = logging.getLogger(__name__)


class AudioAsrWhisper(Operator):
    """**语音识别模块 - 基于Whisper模型的多语言语音转文字解决方案**

    **核心功能**

    - **多语言识别**：支持中英文等主流语言
    - **语音翻译**：可将识别结果翻译为英文

    **推荐实践**
    - 优先处理30秒内的音频片段
    - 英文场景识别准确率最高

    **支持模型**
    - `openai/whisper-large-v3-turbo`
    - `openai/whisper-large-v3`
    - `openai/whisper-medium`（中文支持一般）
    - `openai/whisper-small`（中文输出繁体字）

    **语种支持**
    完整语种列表请参考：
    https://github.com/ggml-org/whisper.cpp/blob/d682e150908e10caa4c15883c633d7902d385237/src/whisper.cpp#L248
    """  # noqa: D415

    def __init__(
        self,
        audio_src_type: str,
        model_path: str = "/opt/las/models",
        model_name: str = "openai/whisper-large-v3",
        batch_size: int = 10,
        source_language: str = "chinese",
        translate_to_english: bool = False,
        condition_on_prev_tokens: bool = True,
        compression_ratio_threshold: float = 1.35,
        temperature: float = 0.5,
        logprob_threshold: float = -1.0,
        dtype: str = "bfloat16",
        rank: int = 0,
        **kwargs: Any,
    ) -> None:
        """初始化Whisper语音识别模型.

        Args:
            audio_src_type: 音频格式类型
                支持的音频格式类型，包含：
                - audio_binary: 原始二进制数据
                - audio_base64: Base64编码数据
                - audio_url: TOS存储链接
                可选值：["audio_binary", "audio_url", "audio_base64"]
            model_path: 模型存储路径
                默认值："/opt/las/models"
            model_name: 模型名称
                支持的Whisper系列模型：
                - whisper-small: 小模型
                - whisper-medium: 中等模型
                - whisper-large-v3: 最新大模型
                - whisper-large-v3-turbo: 优化版大模型
                可选值：[
                    "openai/whisper-small",
                    "openai/whisper-medium",
                    "openai/whisper-large-v3-turbo",
                    "openai/whisper-large-v3"
                ]
                默认值："openai/whisper-large-v3"
            batch_size: 单次处理的音频样本数量
                默认值：10
            source_language: 音频源语言
                支持：chinese/english/japanese/korean等
                默认值："chinese"
            translate_to_english: 英文翻译模式
                是否将识别结果翻译为英文
                启用后输出文本将为英文翻译结果
                默认值：False
            condition_on_prev_tokens: 历史依赖模式
                是否基于历史token进行预测
                关闭后会降低结果连贯性但提升处理速度
                默认值：True
            compression_ratio_threshold: 文本压缩阈值
                控制生成文本的压缩程度（建议范围1.2-2.0）
                值越大保留的重复内容越多
                默认值：1.35
            temperature: 温度系数
                控制生成文本的随机性（0.0-1.0）
                较高值适合创造性场景，较低值适合确定性场景
                默认值：0.5
            logprob_threshold: 对数概率阈值
                对数概率阈值，过滤置信度过低的词。若词的对数概率低于此值，可能被拒绝。
                默认为-1.0，不启用过滤，保留所有词。
            dtype: 计算精度类型
                模型推理使用的数值精度：
                - bfloat16: 平衡精度与速度（默认）
                - float16: 更快的推理速度
                - float32: 最高精度
                可选值：["bfloat16", "float16", "float32"]
                默认值："bfloat16"
            rank: GPU设备编号
                指定使用的GPU设备ID（多卡环境生效）
                默认使用首张显卡（ID=0）
        """
        super().__init__(**kwargs)

        self.audio_src_type = audio_src_type
        self.model_path = model_path
        self.model_name = model_name
        self.batch_size = batch_size
        self.source_language = source_language
        self.translate_to_english = translate_to_english
        self.condition_on_prev_tokens = condition_on_prev_tokens
        self.compression_ratio_threshold = compression_ratio_threshold
        self.temperature = temperature
        self.logprob_threshold = logprob_threshold
        self.dtype = dtype
        self.rank = rank

        from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

        model_dir = str(Path(self.model_path) / self.model_name)

        if self.use_gpu:
            rank = 0 if self.rank is None else self.rank
            self.rank = rank % self.cuda_device_count
            self.device = f"cuda:{self.rank}"
        else:
            self.device = "cpu"
        logger.info("ASR model will be loaded on device: %s", self.device)

        try:
            logger.debug("Loading ASR model from: %s", model_dir)
            model = AutoModelForSpeechSeq2Seq.from_pretrained(
                model_dir, torch_dtype=self.dtype, low_cpu_mem_usage=True, use_safetensors=True
            )
            model.to(self.device)
            logger.info("ASR model loaded successfully on device: %s", self.device)
            processor = AutoProcessor.from_pretrained(model_dir)
            logger.debug("Audio processor initialized successfully")
        except Exception as e:
            logger.exception("Model loading failed - Path: %s", model_dir)
            raise RuntimeError(f"Model loading failed, please check model path: {model_dir}") from e

        # Generation parameters logging
        generate_kwargs = {
            "condition_on_prev_tokens": self.condition_on_prev_tokens,
            "compression_ratio_threshold": self.compression_ratio_threshold,
            "temperature": self.temperature,
            "logprob_threshold": self.logprob_threshold,
            "return_timestamps": True,
            "language": self.source_language,
        }
        if self.translate_to_english:
            generate_kwargs["task"] = "translate"

        logger.debug("Generation parameters:\n %s", json.dumps(generate_kwargs, indent=2))

        try:
            self.pipe = pipeline(
                "automatic-speech-recognition",
                model=model,
                tokenizer=processor.tokenizer,
                feature_extractor=processor.feature_extractor,
                torch_dtype=self.dtype,
                device=self.device,
                generate_kwargs=generate_kwargs,
            )
            logger.info("ASR pipeline initialized successfully")
        except RuntimeError as e:
            logger.critical("Pipeline creation failed!")
            raise RuntimeError("ASR pipeline initialization failed") from e

    @staticmethod
    def _update_timestamps(timestamps: list[list[float]]) -> list[list[float]]:
        timestamps_list = [[x, y] for x, y in timestamps]
        logger.info("Original timestamp sequence received: %d segments", len(timestamps_list))

        time_interval: float = 0.0
        timestamps_update: list[list[float]] = []
        for i in range(len(timestamps_list)):
            timestamp = timestamps_list[i]
            timestamp[0] = timestamp[0] if timestamp[0] is not None else 0
            timestamp[1] = timestamp[1] if timestamp[1] is not None else 0

            if i == len(timestamps_list) - 1 and i > 0:
                logger.debug("Processing final timestamp segment")
                if (timestamp[1] == 0 and timestamp[0] > 29) or (timestamp[0] == 0 and timestamp[1] > 29):
                    timestamp_update = (
                        time_interval + timestamps_list[i - 1][-1],
                        time_interval + timestamps_list[i - 1][-1],
                    )
                elif timestamp[1] == 0 and timestamp[0] != 0:
                    timestamp_update = (
                        timestamps_update[i - 1][-1],
                        timestamps_update[i - 1][-1] + timestamp[0],
                    )
                elif timestamp[0] == 0:
                    timestamp_update = (
                        timestamp[0] + time_interval + timestamps_list[i - 1][-1],
                        timestamp[1] + time_interval + timestamps_list[i - 1][-1],
                    )
                else:
                    timestamp_update = (timestamp[0] + time_interval, timestamp[1] + time_interval)
            elif 0 < i < len(timestamps_list) - 1:
                logger.debug("Processing middle segment %.2f", i / len(timestamps_list))
                if timestamp[1] < timestamp[0]:
                    timestamp_update = (
                        timestamp[0] + time_interval,
                        timestamp[0] + timestamp[1] + time_interval,
                    )
                    time_interval = timestamp[0] + time_interval
                elif timestamp[0] < timestamps_list[i - 1][1]:
                    if timestamps_list[i - 1][0] == 0:
                        time_interval = 30 + time_interval
                    else:
                        time_interval = timestamps_list[i - 1][1] + time_interval
                    timestamp_update = (time_interval, timestamp[1] + time_interval)
                else:
                    timestamp_update = (timestamp[0] + time_interval, timestamp[1] + time_interval)
            else:
                logger.debug("Processing first timestamp segment")
                timestamp_update = (timestamp[0], timestamp[1])
            timestamps_update.append([round(x, 3) for x in timestamp_update])

        logger.info(
            "Completed timestamp adjustment\nOriginal: %s\n \
            Adjusted: %s",
            str(timestamps_list),
            str(timestamps_update),
        )
        return timestamps_update

    def transform(self, audios: pa.Array) -> pa.Array:
        """批量处理文本数组生成嵌入向量.

        该方法使用预加载的嵌入模型对输入的文本数组进行批量编码，生成对应的稠密/稀疏嵌入向量。

        Args:
            texts: 包含待处理文本的数组，元素类型为str。

        Returns:
            pyarrow.Array: 处理后的数组，元素包含以下字段：
                - dense_embedding: 稠密嵌入向量
                - sparse_embedding: 稀疏嵌入向量
                - token_embedding: 可选的token级嵌向量

        Raises:
            ValueError: 当输入数据格式不符合要求时抛出
        """
        logger.info("Processing audio source type: %s", self.audio_src_type)
        try:
            results, timestamps, segments = [], [], []
            total_audios = len(audios)

            for batch_idx in range(0, total_audios, self.batch_size):
                sub_audios = audios.slice(batch_idx, self.batch_size)
                batch_audio = sub_audios.to_pylist()
                results_batch, timestamps_batch, segments_batch = [], [], []

                try:
                    if not batch_audio:
                        logger.debug("Skipping empty batch %d", batch_idx)
                        break

                    processed_batch = []
                    for audio_ref in batch_audio:
                        if self.audio_src_type == "audio_base64":
                            audio_binary = base64_to_byte(audio_ref)
                        elif self.audio_src_type == "audio_url":
                            audio_binary = run_on_local_path(audio_ref, lambda path: path_to_byte(path))
                        elif self.audio_src_type == "audio_binary":
                            audio_binary = audio_ref
                        else:
                            ValueError(f"Unsupported audio format: {self.audio_src_type}")
                        processed_batch.append(audio_binary)
                        logger.debug("Audio processing for item %s", audio_ref[:15])

                    batch_result = self.pipe(
                        processed_batch,
                        return_timestamps=True,
                        batch_size=len(processed_batch),
                    )
                    for item in batch_result:
                        results_batch.append(item["text"])
                        chunk_data = item["chunks"]
                        ts_data = [c["timestamp"] for c in chunk_data]
                        text_data = [c["text"] for c in chunk_data]
                        timestamps_batch.append(ts_data)
                        segments_batch.append(text_data)
                except Exception:
                    logger.exception("Error processing audio batch!")
                    results_batch = [None] * len(batch_audio)
                    timestamps_batch = [[None]] * len(batch_audio)
                    segments_batch = [[None]] * len(batch_audio)

                results.extend(results_batch)
                timestamps.extend(timestamps_batch)
                segments.extend(segments_batch)

            # Post-process timestamps
            clean_timestamps, clean_segments = [], []
            for raw_ts, raw_txt in zip(timestamps, segments):
                if raw_ts is None:
                    clean_timestamps.append([None])
                    clean_segments.append([None])
                    continue
                try:
                    adjusted_ts = self._update_timestamps(raw_ts)
                    valid_ts, valid_txt = [], []
                    for ts, txt in zip(adjusted_ts, raw_txt):
                        if ts[0] != ts[1]:
                            valid_ts.append(ts)
                            valid_txt.append(txt)
                    clean_timestamps.append(valid_ts)
                    clean_segments.append(valid_txt)
                except Exception:
                    logger.exception("Timestamp adjustment failed!")
                    clean_timestamps.append([])
                    clean_segments.append([])

        except Exception:
            logger.exception("Batch processing failed!")
            results = [None] * total_audios
            clean_timestamps = [[]] * total_audios
            clean_segments = [[]] * total_audios

        total_result_details = []
        for i in range(total_audios):
            record = {"asr_result": results[i], "timestamps": clean_timestamps[i], "segments": clean_segments[i]}
            total_result_details.append(record)

        return pa.array(total_result_details, type=self.__return_column_type__())

    @staticmethod
    def __return_column_type__() -> pa.DataType:
        fields = [
            pa.field("asr_result", pa.string()),
            pa.field("timestamps", pa.list_(pa.list_(pa.float32()))),
            pa.field("segments", pa.list_(pa.string())),
        ]
        return pa.struct(fields)
