# Copyright (c) Beijing Volcano Engine Technology Ltd.

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch
import torchaudio

from daft.dependencies import pa
from daft.las.functions.types import Operator
from daft.las.functions.utils.audio_utils import decode_audio_torchaudio
from daft.las.functions.utils.common_utils import (
    run_on_local_path,
    tracking_usage,
)

logger = logging.getLogger(__name__)


class ASRPipeline:
    def __init__(
        self,
        model_path: str,
        punc_model_id_or_path: str | None = "iic/punc_ct-transformer_cn-en-common-vocab471067-large",
        return_language_only: bool = False,
        device: str = "cpu",
        progress_bar: bool = False,
    ):
        from transformers.models.whisper.feature_extraction_whisper import WhisperFeatureExtractor
        from transformers.models.whisper.modeling_whisper import WhisperForConditionalGeneration
        from transformers.models.whisper.tokenization_whisper_fast import WhisperTokenizerFast

        self.tokenizer: WhisperTokenizerFast = WhisperTokenizerFast.from_pretrained(model_path)
        self.feature_extractor: WhisperFeatureExtractor = WhisperFeatureExtractor.from_pretrained(model_path)
        model: WhisperForConditionalGeneration = WhisperForConditionalGeneration.from_pretrained(
            model_path,
            attn_implementation="sdpa",
            torch_dtype=torch.bfloat16,
        ).to(device)
        model.eval()
        model = torch.compile(model, mode="max-autotune")
        self.model = model
        self.return_language_only = return_language_only

        if punc_model_id_or_path and not return_language_only:
            try:
                from funasr import AutoModel as FunasrModel

                self.punc_model = FunasrModel(
                    model_class="CTTransformer",
                    model=punc_model_id_or_path,
                    device=device,
                    bf16=True,
                    disable_update=True,
                )
            except ImportError:
                logger.exception("Funasr is not installed, punctuation model will not be used.")
                self.punc_model = None
        else:
            self.punc_model = None

        self.device = device
        self.progress_bar = progress_bar

    def transcribe_segments_in_batch(
        self,
        waveforms: list[torch.Tensor],
        sample_rate: int,
        batch_size: int = 16,
    ) -> list[dict[str, object]]:
        if not waveforms:
            return []

        if sample_rate != 16000:
            waveforms = [
                torchaudio.transforms.Resample(sample_rate, 16000).to(self.device)(waveform.to(self.device))
                for waveform in waveforms
            ]

        results = []
        segment_audio_list: list[torch.Tensor] = []

        for waveform in waveforms:
            if waveform.dim() == 2:
                waveform = waveform.mean(dim=0).squeeze(0)
            segment_audio_list.append(waveform.cpu())

        for i in range(0, len(segment_audio_list), batch_size):
            batch_audio = segment_audio_list[i : i + batch_size]
            inputs = self.feature_extractor(
                raw_speech=[audio.numpy() for audio in batch_audio],
                sampling_rate=16000,
                return_attention_mask=True,
                return_tensors="pt",
            )
            input_features = inputs.input_features.to(self.device, dtype=torch.bfloat16)
            attention_mask = inputs.attention_mask.to(self.device, dtype=torch.long)

            with torch.no_grad():
                output = self.model.generate(
                    input_features=input_features,
                    attention_mask=attention_mask,
                    do_sample=False,
                    return_dict_in_generate=True,
                    language=None,
                )
                generated_ids = output.sequences

                # (1) Language prediction
                language_tokens = generated_ids[:, 1].view(-1)
                languages = self.tokenizer.batch_decode(
                    language_tokens,
                )
                languages = [language[2:-2] for language in languages]

                # (2) Transcription
                if not self.return_language_only:
                    transcriptions = self.tokenizer.batch_decode(
                        generated_ids,
                        skip_special_tokens=True,
                        normalize=False,
                    )
                else:
                    transcriptions = [""] * len(languages)

                for idx in range(len(transcriptions)):
                    text = transcriptions[idx].strip()
                    # (3) Punctuation, 只支持中英文
                    if self.punc_model is not None and text and (languages[idx] in ["en", "zh"]):
                        punc_result = self.punc_model.generate(input=[text], disable_pbar=True)[0]
                        text_with_punc = punc_result.get("text", "").strip()
                    else:
                        text_with_punc = None
                    results.append(
                        {
                            "language": languages[idx],
                            "text": transcriptions[idx],
                            "text_with_punc": text_with_punc,
                        }
                    )

        return results


class AudioAsrLidWhisper(Operator):
    """**语种识别 + 语言识别模块 - 基于Whisper模型的多语言 LID + ASR 解决方案**

    **核心功能**
    - 多语言识别：支持中英文等百种语言
    - 语言识别（LID）：在识别文本的同时输出语言标签（例如 `en`、`zh`）
    - 标点符号恢复：可选的中英文标点恢复功能，提升文本可读性

    **推荐实践**
    - 支持多种音频输入格式（URL、二进制等）
    - 批量处理30秒内的音频片段
    - 英文场景识别准确率通常最高
    - 中英文场景可选配合标点恢复模型提升文本可读性
    - 支持只返回语言识别结果
    - 支持GPU加速推理，推荐使用CUDA设备，显存在4G以上

    **支持模型**
    - Whisper系列模型（LID + ASR）
        - `openai/whisper-large-v3-turbo`
        - `openai/whisper-large-v3`
        - `openai/whisper-medium`（中文支持一般）
        - `openai/whisper-small`（中文输出可能为繁体）
    - 中英文标点恢复模型
        - `iic/punc_ct-transformer_cn-en-common-vocab471067-large`

    **语种支持**
    完整语种列表请参考：
    https://github.com/ggml-org/whisper.cpp/blob/d682e150908e10caa4c15883c633d7902d385237/src/whisper.cpp#L248
    """  # noqa: D415

    def __init__(
        self,
        audio_src_type: str,
        model_path: str = "/opt/las/models",
        model_name: str = "openai/whisper-large-v3",
        punc_model_name: str | None = None,
        return_language_only: bool = False,
        batch_size: int = 10,
        device: str = "cpu",
        **kwargs: Any,
    ) -> None:
        """初始化带语言识别（LID）能力的 Whisper ASR.

        Args:
            audio_src_type: 输入音频的来源类型，支持值：
                - ``audio_url``: 音频文件的 URL 或 TOS 对象存储路径。
                - ``audio_binary``: 原始音频二进制数据。
                请确保该值与传入的 ``audios`` 数据格式一致。
            model_path: 模型根目录路径，通常包含若干模型子目录，
                默认 ``"/opt/las/models"``。
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
            punc_model_name: 可选的标点恢复模型名称
                支持使用 ``iic/punc_ct-transformer_cn-en-common-vocab471067-large`` 进行中英文标点恢复
                可选值：[
                    "iic/punc_ct-transformer_cn-en-common-vocab471067-large"
                ]
                默认值：``None``。
                若提供该参数，Operator会对识别出的纯文本进行标点化处理并通过 ``asr_result_with_punc`` 字段返回。
                若未提供或加载失败，则该字段为 ``None``。
            return_language_only: 是否仅返回语言识别结果而不进行语音转文本。
                若设置为 ``True``，则 ``asr_result`` 和 ``asr_result_with_punc`` 字段均为 ``None``。
                默认值：``False``。
            batch_size: 每次批处理的音频数量，值越大吞吐越高但显存/内存占用也越大，默认 ``10``。
            device: 推理设备标识，例如 ``"cpu"``, ``"cuda"``, ``"cuda:0"``，
                默认使用 ``"cpu"``。

        Notes:
            - 初始化过程中会从 ``Path(model_path)/model_name`` 加载 Whisper 模型、tokenizer 与 feature_extractor；
            - 若提供 ``punc_model_name``，会尝试加载用于中文/英文标点恢复的模型（加载失败时会回退为无标点输出）；
            - 建议优先处理短音频片段（<30s）以获得更稳定且准确的识别；
            - 在多卡环境下请显式指定 ``device``（如 ``cuda:0``）以控制显卡分配。
        """
        super().__init__(**kwargs)

        self.audio_src_type = audio_src_type
        self.model_path = model_path
        self.model_name = model_name
        self.batch_size = batch_size
        self.punc_model_name = punc_model_name

        model_dir = str(Path(self.model_path) / self.model_name)
        self.device = device

        if self.punc_model_name is not None and not return_language_only:
            punc_model_dir = str(Path(self.model_path) / self.punc_model_name)
        else:
            punc_model_dir = None

        try:
            logger.debug("Loading ASR model from: %s", model_dir)
            self.asr_pipeline = ASRPipeline(
                model_path=model_dir,
                punc_model_id_or_path=punc_model_dir,
                return_language_only=return_language_only,
                device=self.device,
            )
            logger.info("ASR model loaded successfully on device: %s", self.device)

        except Exception as e:
            logger.exception("Model loading failed - Path: %s", model_dir)
            raise RuntimeError(f"Model loading failed, please check model path: {model_dir}") from e

        tracking_usage(op=self.__class__.__name__, model_service_or_lib=self.model_name)

    def transform(self, audios: pa.Array) -> pa.Array:
        """批量处理音频数组生成语音识别（ASR）和语言识别（LID）结果.

        该方法使用预加载的 Whisper 模型对输入的音频数据进行批量处理，返回包含识别文本、
        语言标签以及（可选）带标点化识别结果的结构化记录。

        Args:
            audios: 包含音频数据的数组，支持以下格式：
                - `audio_url`: 音频文件的 URL 或 TOS 对象存储路径（将会下载到本地后解码）
                - `audio_binary`: 原始音频字节数据（已解码或原始音频二进制）

        Returns:
            处理后的结构化数组，每个元素为一个 struct，包含字段：
                - `asr_result`: 语音识别得到的文本结果；
                - `language`: 识别出的语言代码（例如 `en`、`zh`）；
                - `asr_result_with_punc`: 可选的带标点化文本（当初始化时加载了标点模型且可用时返回），否则为 `None`。

        Raises:
            ValueError: 当传入的 `audio_src_type` 非受支持类型时抛出。
            Exception: 在批处理过程中出现未捕获的异常时抛出，调用方可根据返回值中 `None` 判断单条失败。

        实践建议：
            - 优先处理短片段（推荐小于30秒）以获得最稳定的识别效果；
            - 若需要更好中文标点化效果，可在初始化时提供 `punc_model_name` 并保证相关依赖可用。
        """
        logger.info("Processing audio source type: %s", self.audio_src_type)
        try:
            results, languages, punc_results = [], [], []
            total_audios = len(audios)

            for batch_idx in range(0, total_audios, self.batch_size):
                sub_audios = audios.slice(batch_idx, self.batch_size)
                batch_audio = sub_audios.to_pylist()
                results_batch = []
                languages_batch = []
                punc_results_batch = []

                try:
                    if not batch_audio:
                        logger.debug("Skipping empty batch %d", batch_idx)
                        break

                    processed_batch = []
                    for audio_ref in batch_audio:
                        if self.audio_src_type == "audio_url":
                            audio_tensor, _ = run_on_local_path(
                                audio_ref, lambda path: decode_audio_torchaudio(path, 16000, 1)
                            )
                        elif self.audio_src_type == "audio_binary":
                            audio_tensor, _ = decode_audio_torchaudio(audio_ref, 16000, 1)
                        else:
                            raise ValueError(f"Unsupported audio format: {self.audio_src_type}")

                        processed_batch.append(audio_tensor)
                        logger.debug(
                            "Audio processing for item %s", audio_ref[:15] if isinstance(audio_ref, str) else "binary"
                        )

                    batch_result = self.asr_pipeline.transcribe_segments_in_batch(
                        processed_batch, 16000, batch_size=self.batch_size
                    )

                    for result in batch_result:
                        results_batch.append(result["text"])
                        languages_batch.append(result["language"])
                        punc_results_batch.append(result.get("text_with_punc"))
                except Exception:
                    logger.exception("Error processing audio batch!")
                    results_batch = [None] * len(batch_audio)
                    languages_batch = [None] * len(batch_audio)
                    punc_results_batch = [None] * len(batch_audio)

                results.extend(results_batch)
                languages.extend(languages_batch)
                punc_results.extend(punc_results_batch)

        except Exception:
            logger.exception("Batch processing failed!")
            results = [None] * total_audios
            languages = [None] * total_audios
            punc_results = [None] * total_audios

        total_result_details = []
        for i in range(total_audios):
            record = {"asr_result": results[i], "language": languages[i], "asr_result_with_punc": punc_results[i]}
            total_result_details.append(record)

        return pa.array(total_result_details, type=self.__return_column_type__())

    @staticmethod
    def __return_column_type__() -> pa.DataType:
        fields = [
            pa.field("asr_result", pa.string()),
            pa.field("language", pa.string()),
            pa.field("asr_result_with_punc", pa.string()),
        ]
        return pa.struct(fields)
