# Copyright (c) Beijing Volcano Engine Technology Ltd.

from __future__ import annotations

import logging
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import torch

from daft.dependencies import pa
from daft.las.functions.types import Operator
from daft.las.functions.utils.common_utils import save_file_to_local, tracking_usage

logger = logging.getLogger(__name__)


class KimiAudioUnderstanding(Operator):
    """**Kimi-Audio 多模态音频理解模型 - 音频语义解析与自然语言描述生成**

    **核心功能**

    - 多模态音频处理
      - 支持 `URL`/`Base64编码`/`二进制流` 三种音频格式
    - 音频-语言联合建模
      - 实现音频内容到语义空间的精准映射
    - 对话式提示支持
      - 通过 `prompt` 参数引导生成方向
    - 资源使用
      - 推荐使用48G及以上显存的GPU

    **场景优化**
    - 中英文混合场景优化：特别针对中文语义增强
    - 支持多种音频理解任务：ASR、音频描述、情感识别、内容分析
    """  # noqa: D415

    def __init__(
        self,
        audio_src_type: str = "audio_url",
        model_path: str = "/opt/las/models",
        model_name: str = "moonshotai/Kimi-Audio-7B-Instruct",
        prompt: str = "请分析这段音频的内容。",
        batch_size: int = 4,
        text_temperature: float = 0.0,
        text_top_k: int = 5,
        text_repetition_penalty: float = 1.0,
        text_repetition_window_size: int = 16,
        rank: int | None = None,
        **kwargs: Any,
    ) -> None:
        """音频理解参数初始化.

        Args:
            audio_src_type: 音频数据源类型，支持三种格式
                可选值：["audio_url", "audio_base64", "audio_binary"]
                默认值："audio_url"
            model_path: 本地模型文件存储的绝对路径，默认为容器内预置路径。当使用自定义模型时需修改此路径
                默认值："/opt/las/models"
            model_name: 支持的音频语言模型版本，当前仅支持 Kimi-Audio系列模型
                可选值：["moonshotai/Kimi-Audio-7B-Instruct"]
                默认值："moonshotai/Kimi-Audio-7B-Instruct"
            prompt: 用户理解音频内容的提示词，模型会根据提示词来生成音频的分析结果。
                默认值："请分析这段音频的内容。"
            batch_size: 分组处理的音频数量。较大值可能增加显存消耗。
                默认值：4
            text_temperature: 文本生成的温度参数，控制生成文本的随机性。0.0表示确定性生成
                默认值：0.0
            text_top_k: 文本生成时考虑的top-k候选词数量
                默认值：5
            text_repetition_penalty: 文本重复惩罚系数，避免生成重复内容
                默认值：1.0
            text_repetition_window_size: 文本重复检测的窗口大小
                默认值：16
            rank: 指定使用的GPU设备编号（多卡环境有效）。例如：0表示第一张GPU，1表示第二张GPU
                默认值：None
        """
        super().__init__(**kwargs)

        self.audio_src_type = audio_src_type
        self.model_path = model_path
        self.model_name = model_name

        model_dir_path = Path(self.model_path) / self.model_name
        kimia_infer_path = model_dir_path / "kimia_infer"
        if kimia_infer_path.exists():
            if str(kimia_infer_path) not in sys.path:
                # Kimi音频模型需要加载kimia_infer目录下的自定义推理脚本
                # 这些脚本包含模型特定的推理逻辑，不是标准的模型权重文件
                sys.path.insert(0, str(kimia_infer_path))
                logger.info("Added kimia_infer path to sys.path: %s", kimia_infer_path)
            else:
                logger.debug("kimia_infer path already in sys.path: %s", kimia_infer_path)
        else:
            logger.warning("kimia_infer path not found: %s", kimia_infer_path)
        self.prompt = prompt
        self.batch_size = batch_size
        self.text_temperature = text_temperature
        self.text_top_k = text_top_k
        self.text_repetition_penalty = text_repetition_penalty
        self.text_repetition_window_size = text_repetition_window_size
        self.rank = rank

        try:
            from models.tokenizer.glm4_tokenizer import Glm4Tokenizer
            from models.tokenizer.whisper_Lv3.whisper import WhisperEncoder

            from utils.data import KimiAContent
            from utils.sampler import KimiASampler
            from utils.special_tokens import instantiate_extra_tokens

            logger.info("Successfully imported kimia_infer components")
        except ImportError as e:
            raise RuntimeError(f"Failed to import kimia_infer components, please check model directory structure: {e}")

        if not model_dir_path.exists():
            final_model_path = self.model_name
        else:
            final_model_path = str(model_dir_path)

        use_gpu = self.use_gpu and torch.cuda.is_available()
        if self.rank is None:
            self.device = "cuda" if use_gpu else "cpu"
        else:
            device_id = self.rank % self.cuda_device_count
            self.device = f"cuda:{device_id}" if use_gpu else "cpu"
            if use_gpu:
                torch.cuda.set_device(device_id)
        self.model = self._KimiAudio(
            model_path=final_model_path,
            load_detokenizer=False,
            device=self.device,
            kimi_audio_sampler=KimiASampler,
            instantiate_extra_tokens=instantiate_extra_tokens,
            kimi_audio_content=KimiAContent,
            glm4_tokenizer=Glm4Tokenizer,
            whisper_encoder=WhisperEncoder,
        )
        self.sampling_params = {
            "audio_temperature": 0.8,
            "audio_top_k": 10,
            "audio_repetition_penalty": 1.0,
            "audio_repetition_window_size": 64,
            "text_temperature": self.text_temperature,
            "text_top_k": self.text_top_k,
            "text_repetition_penalty": self.text_repetition_penalty,
            "text_repetition_window_size": self.text_repetition_window_size,
        }

        logger.info(
            "Model initialization configuration:\n"
            "- Model Name: %s\n"
            "- Storage Path: %s\n"
            "- Device: %s\n"
            "- Text Temperature: %s\n"
            "- Prompt Template: %s",
            self.model_name,
            final_model_path,
            self.device,
            self.text_temperature,
            self.prompt,
        )

        tracking_usage(op=self.__class__.__name__, model_service_or_lib=self.model_name)

    def _build_message_template(self, tmp_file_name: str) -> list[dict[str, Any]]:
        message: list[dict[str, Any]] = [
            {"role": "user", "message_type": "text", "content": self.prompt},
            {"role": "user", "message_type": "audio", "content": tmp_file_name},
        ]
        return message

    def transform(self, audios: pa.Array) -> pa.Array:
        """对输入的音频列进行批量处理，生成包含音频理解结果的文本描述。

        Args:
            audios: 包含音频数据的列，元素类型为字符串或者二进制。

        Returns:
            pa.Array: 处理后的列，元素为每个音频的理解结果。

        Raises:
            ValueError: 当输入数据格式不符合要求时抛出
            RuntimeError: 模型推理过程中发生错误时抛出
        """  # noqa: D415
        start_time = time.monotonic()
        logger.info("Starting batch processing, input type: %s", self.audio_src_type)

        all_results = []
        total_audios = len(audios)
        total_batches = (total_audios + self.batch_size - 1) // self.batch_size
        for batch_idx in range(0, total_audios, self.batch_size):
            sub_audios = audios.slice(batch_idx, self.batch_size)
            current_batch = sub_audios.to_pylist()
            logger.debug(
                "Processing batch %d/%d with %d audios",
                (batch_idx // self.batch_size) + 1,
                total_batches,
                len(current_batch),
            )

            if not current_batch:
                break

            try:
                with tempfile.TemporaryDirectory(dir="/tmp") as tmp_dir:
                    for idx, audio_data in enumerate(current_batch):
                        try:
                            if self.audio_src_type == "audio_url":
                                logger.info("audio url is %s", audio_data)
                                if audio_data and audio_data.startswith(("tos://", "s3://")):
                                    file_name = f"audio_{batch_idx}_{idx}.{audio_data.split('.')[-1]}"
                                elif audio_data and audio_data.startswith(("https://", "http://")):
                                    url_parts = audio_data.split(".")
                                    if len(url_parts) > 1 and url_parts[-1].lower() in [
                                        "mp3",
                                        "wav",
                                        "flac",
                                        "m4a",
                                        "aac",
                                        "ogg",
                                        "wma",
                                    ]:
                                        file_extension = url_parts[-1].lower()
                                    else:
                                        file_extension = "wav"
                                    file_name = f"{int(time.time())!s}_{idx}.{file_extension}"
                                else:
                                    raise ValueError(f"Unsupported audio URL format: {audio_data}")
                            else:
                                file_name = "audio_binary"

                            tmp_file_name = save_file_to_local(audio_data, self.audio_src_type, tmp_dir, file_name)
                            if not Path(tmp_file_name).exists():
                                raise FileNotFoundError(tmp_file_name)

                            message = self._build_message_template(tmp_file_name)

                            _, text_output = self.model.generate(message, **self.sampling_params, output_type="text")
                            logger.info("Model output for audio %d: %s", idx, text_output)
                            all_results.append(text_output)
                        except Exception as e:
                            logger.error("Processing failed for audio %d: %s", idx, str(e))
                            all_results.append("")

            except Exception as e:
                logger.exception("Processing error: %s", str(e))
                all_results.extend([""] * len(current_batch))

        processing_time = time.monotonic() - start_time

        logger.info(
            "Completed %d audios | Total time: %.2fs | Throughput: %.2f audio/s",
            total_audios,
            processing_time,
            total_audios / processing_time,
        )

        return pa.array(all_results, type=self.__return_column_type__())

    @staticmethod
    def __return_column_type__() -> pa.DataType:
        return pa.string()

    class _KimiAudio:
        def __init__(self, model_path: str, load_detokenizer: bool = True, device: str = "cuda", **deps: Any) -> None:
            import torch
            from huggingface_hub import snapshot_download
            from transformers import AutoModelForCausalLM

            self.device = device
            self.kimi_audio_sampler = deps["kimi_audio_sampler"]
            self.instantiate_extra_tokens = deps["instantiate_extra_tokens"]
            self.kimi_audio_content = deps["kimi_audio_content"]
            self.glm4_tokenizer = deps["glm4_tokenizer"]
            self.whisper_encoder = deps["whisper_encoder"]

            if os.path.exists(model_path):
                cache_path = model_path
            else:
                cache_path = snapshot_download(model_path)

            self.alm = AutoModelForCausalLM.from_pretrained(
                cache_path, torch_dtype=torch.bfloat16, trust_remote_code=True
            )
            self.alm = self.alm.to(self.device)

            model_config = self.alm.config
            self.kimia_text_audiodelaytokens = model_config.kimia_mimo_audiodelaytokens
            self.kimia_token_offset = model_config.kimia_token_offset

            if load_detokenizer:
                try:
                    from models.detokenizer import get_audio_detokenizer

                    self.detokenizer = get_audio_detokenizer(cache_path)
                except ImportError:
                    logger.warning("Audio detokenizer loading failed, only text output supported")
                    self.detokenizer = None
            else:
                self.detokenizer = None

            self.prompt_manager = KimiAudioUnderstanding._KimiAPromptManager(
                model_path=cache_path,
                kimia_token_offset=self.kimia_token_offset,
                kimia_text_audiodelaytokens=self.kimia_text_audiodelaytokens,
                device=self.device,
                **deps,
            )

            self.extra_tokens = self.prompt_manager.extra_tokens
            self.eod_ids = [self.extra_tokens.msg_end, self.extra_tokens.media_end]

        def generate(self, message: list[dict[str, Any]], output_type: str = "text", **kwargs: Any) -> tuple[Any, str]:
            import torch

            with torch.inference_mode():
                history = self.prompt_manager.get_prompt(message, output_type=output_type)
                audio_input_ids, text_input_ids, is_continuous_mask, _, _ = history.to_tensor()
                audio_features = history.continuous_feature

                max_new_tokens = kwargs.get("max_new_tokens", 7500 - audio_input_ids.shape[1])
                audio_input_ids = audio_input_ids.to(self.device)
                text_input_ids = text_input_ids.to(self.device)
                is_continuous_mask = is_continuous_mask.to(self.device)
                audio_features = [f.to(self.device) for f in audio_features]

                _, generated_text_tokens = self._generate_loop(
                    audio_input_ids=audio_input_ids,
                    text_input_ids=text_input_ids,
                    max_new_tokens=max_new_tokens,
                    is_continuous_mask=is_continuous_mask,
                    continous_feature=audio_features,
                    output_type=output_type,
                    **{k: v for k, v in kwargs.items() if k != "max_new_tokens"},
                )

                generated_text_tokens = [t for t in generated_text_tokens if t < self.kimia_token_offset]
                generated_text = self.detokenize_text(generated_text_tokens)

                return None, generated_text

        def _generate_loop(
            self,
            audio_input_ids: Any,
            text_input_ids: Any,
            max_new_tokens: int,
            is_continuous_mask: Any,
            continous_feature: Any,
            output_type: str = "text",
            **kwargs: Any,
        ) -> tuple[Any, Any]:
            import torch
            import tqdm

            sampler = self.kimi_audio_sampler(
                audio_top_k=kwargs.get("audio_top_k", 10),
                audio_temperature=kwargs.get("audio_temperature", 0.8),
                audio_repetition_penalty=kwargs.get("audio_repetition_penalty", 1.0),
                audio_repetition_window_size=kwargs.get("audio_repetition_window_size", 64),
                text_top_k=kwargs.get("text_top_k", 5),
                text_temperature=kwargs.get("text_temperature", 0.0),
                text_repetition_penalty=kwargs.get("text_repetition_penalty", 1.0),
                text_repetition_window_size=kwargs.get("text_repetition_window_size", 16),
            )

            text_stream_is_finished = False
            previous_audio_tokens = torch.zeros((4096,), dtype=torch.int, device=self.device)
            text_previous_tokens = torch.zeros((4096,), dtype=torch.int, device=self.device)

            decoder_input_audio_ids = audio_input_ids.clone()
            decoder_input_text_ids = text_input_ids.clone()
            decoder_position_ids = (
                torch.arange(0, decoder_input_audio_ids.shape[1], device=self.device).unsqueeze(0).long()
            )
            decoder_input_whisper_feature = continous_feature
            decoder_is_continuous_mask = is_continuous_mask
            past_key_values = None

            last_position_id = decoder_input_audio_ids.shape[1] - 1
            valid_text_length = 0
            valid_audio_length = 0

            for i in tqdm.tqdm(range(max_new_tokens), desc="Generating tokens", disable=False):
                audio_logits, text_logits, past_key_values = self.alm.forward(
                    input_ids=decoder_input_audio_ids,
                    text_input_ids=decoder_input_text_ids,
                    whisper_input_feature=decoder_input_whisper_feature,
                    is_continuous_mask=decoder_is_continuous_mask,
                    position_ids=decoder_position_ids,
                    past_key_values=past_key_values,
                    return_dict=False,
                )

                next_token_text = sampler.sample_text_logits(
                    text_logits, recent_tokens=text_previous_tokens[:i] if i > 0 else None
                )

                next_audio_token = sampler.sample_audio_logits(
                    audio_logits, recent_tokens=previous_audio_tokens[:i] if i > 0 else None
                )

                if text_stream_is_finished:
                    next_token_text.fill_(self.extra_tokens.kimia_text_blank)
                elif next_token_text.item() == self.extra_tokens.kimia_text_eos:
                    text_stream_is_finished = True
                else:
                    valid_text_length += 1

                text_previous_tokens[i : i + 1] = next_token_text

                if i < self.kimia_text_audiodelaytokens:
                    next_audio_token.fill_(self.extra_tokens.kimia_text_blank)
                else:
                    if output_type == "text":
                        next_audio_token.fill_(self.extra_tokens.kimia_text_blank)
                    else:
                        valid_audio_length += 1

                previous_audio_tokens[i : i + 1] = next_audio_token
                audio_stream_is_finished = next_audio_token.item() in self.eod_ids

                if (output_type == "text" and text_stream_is_finished) or (
                    output_type == "both" and audio_stream_is_finished
                ):
                    return_text_tokens = text_previous_tokens[:valid_text_length].detach().cpu().numpy().tolist()
                    return_audio_tokens = (
                        previous_audio_tokens[
                            self.kimia_text_audiodelaytokens : valid_audio_length + self.kimia_text_audiodelaytokens
                        ]
                        .detach()
                        .cpu()
                        .numpy()
                        .tolist()
                    )
                    return return_audio_tokens, return_text_tokens
                else:
                    decoder_input_audio_ids = next_audio_token.unsqueeze(1)
                    decoder_input_text_ids = next_token_text.unsqueeze(1)
                    decoder_position_ids = (
                        torch.zeros(1, 1, device=self.device).fill_(last_position_id + 1).long().view(1, 1)
                    )
                    last_position_id += 1
                    decoder_input_whisper_feature = None
                    decoder_is_continuous_mask = None

            return_text_tokens = text_previous_tokens[:valid_text_length].detach().cpu().numpy().tolist()
            return_audio_tokens = (
                previous_audio_tokens[
                    self.kimia_text_audiodelaytokens : valid_audio_length + self.kimia_text_audiodelaytokens
                ]
                .detach()
                .cpu()
                .numpy()
                .tolist()
            )
            return return_audio_tokens, return_text_tokens

        def detokenize_text(self, text_tokens: list[int]) -> str:
            valid_text_ids = []
            for x in text_tokens:
                if x == self.extra_tokens.kimia_text_eos:
                    break
                valid_text_ids.append(x)
            return self.prompt_manager.text_tokenizer.decode(valid_text_ids)

    class _KimiAPromptManager:
        def __init__(
            self,
            model_path: str,
            kimia_token_offset: int,
            kimia_text_audiodelaytokens: int,
            device: str = "cuda",
            **deps: Any,
        ) -> None:
            import torch
            from transformers import AutoTokenizer

            self.device = device
            self.glm4_tokenizer = deps["glm4_tokenizer"]
            self.whisper_encoder = deps["whisper_encoder"]
            self.instantiate_extra_tokens = deps["instantiate_extra_tokens"]
            self.kimi_audio_content = deps["kimi_audio_content"]

            local_tokenizer_path = Path(model_path) / "glm-4-voice-tokenizer"
            if local_tokenizer_path.exists():
                tokenizer_path = str(local_tokenizer_path)
            else:
                tokenizer_path = "THUDM/glm-4-voice-tokenizer"

            self.audio_tokenizer = self.glm4_tokenizer(tokenizer_path)
            self.audio_tokenizer = self.audio_tokenizer.to(self.device).to(torch.bfloat16)

            whisper_path = Path(model_path) / "whisper-large-v3"
            if whisper_path.exists():
                whisper_encoder_instance = self.whisper_encoder(str(whisper_path))
            else:
                whisper_encoder_instance = self.whisper_encoder("openai/whisper-large-v3")
            self.whisper_encoder = whisper_encoder_instance.to(self.device).to(torch.bfloat16)

            self.text_tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

            self.extra_tokens = self.instantiate_extra_tokens(self.text_tokenizer)

            self.kimia_token_offset = kimia_token_offset
            self.kimia_text_audiodelaytokens = kimia_text_audiodelaytokens

        def _tokenize_text(self, text: str | None) -> list[int] | None:
            if text is None:
                return None
            token_ids = self.text_tokenizer.encode(text, bos=False, eos=False)
            return token_ids

        def _tokenize_audio(self, wav_path: str) -> list[int]:
            wav_tokens = self.audio_tokenizer.tokenize(audio_path=wav_path)
            wav_tokens = wav_tokens + self.kimia_token_offset
            wav_tokens_list = wav_tokens.squeeze(0).cpu().numpy().tolist()
            return wav_tokens_list

        def extract_whisper_feat(self, wav: str | Any) -> Any:
            import librosa
            import torch

            if isinstance(wav, str):
                wav = librosa.load(wav, sr=16000)[0]
                wav_tensor = torch.tensor(wav).unsqueeze(0)
            elif isinstance(wav, torch.Tensor):
                wav_tensor = wav
            else:
                raise ValueError(f"Invalid wav type: {type(wav)}")

            wav_tensor = wav_tensor.to(self.device)
            continous_feature = self.whisper_encoder.tokenize_waveform(wav_tensor)
            continous_feature = continous_feature.reshape(
                continous_feature.shape[0],
                int(continous_feature.shape[1] // 4),
                continous_feature.shape[2] * 4,
            )
            return continous_feature

        def tokenize_message(
            self,
            message: dict[str, Any],
            tokenize_role: bool = True,
            has_ct_token: bool = False,
            has_msg_end_token: bool = False,
            extract_whisper_feature: bool = False,
            output_type: str = "text",
        ) -> Any:
            kimia_content_msg = self.kimi_audio_content()
            role = message["role"]
            has_loss = role == "assistant"

            if tokenize_role:
                if role == "user":
                    kimia_content_msg.audio_append(self.extra_tokens.kimia_user_msg_start)
                    kimia_content_msg.text_append(self.extra_tokens.kimia_text_blank)
                elif role == "assistant":
                    kimia_content_msg.audio_append(self.extra_tokens.kimia_assistant_msg_start)
                    kimia_content_msg.text_append(self.extra_tokens.kimia_text_blank)

            if message.get("message_type") == "text":
                text = message["content"]
                text_tokens = self._tokenize_text(text)
                if text_tokens is not None:
                    kimia_content_msg.text_extend(text_tokens, has_loss)
                    kimia_content_msg.audio_extend([self.extra_tokens.kimia_text_blank] * len(text_tokens))

                if role == "assistant":
                    kimia_content_msg.text_append(self.extra_tokens.kimia_text_eos, has_loss)
                    kimia_content_msg.audio_append(self.extra_tokens.kimia_text_blank, audio_token_loss_mask=False)

            elif message.get("message_type") == "audio":
                if "audio_tokens" in message:
                    speech_tokens = message["audio_tokens"]
                else:
                    speech_tokens = self._tokenize_audio(message["content"])

                kimia_content_msg.audio_extend(speech_tokens, is_continuous=False)
                kimia_content_msg.text_extend([self.extra_tokens.kimia_text_blank] * len(speech_tokens))

                if extract_whisper_feature:
                    continous_feature = self.extract_whisper_feat(message["content"])
                    kimia_content_msg.continuous_feature.append(continous_feature)

            if has_ct_token:
                if output_type == "text":
                    kimia_content_msg.audio_append(self.extra_tokens.kimia_speech_ct_id)
                    kimia_content_msg.text_append(self.extra_tokens.kimia_text_blank)
                elif output_type == "both":
                    kimia_content_msg.audio_append(self.extra_tokens.kimia_speech_ctd_id)
                    kimia_content_msg.text_append(self.extra_tokens.kimia_text_blank)

            if has_msg_end_token:
                kimia_content_msg.audio_append(self.extra_tokens.msg_end)
                kimia_content_msg.text_append(self.extra_tokens.kimia_text_blank)

            return kimia_content_msg

        def get_prompt(
            self, messages: list[dict[str, Any]], output_type: str = "text", add_assistant_start_msg: bool = True
        ) -> Any:
            msgs = []
            tokenize_role = True
            has_ct_token = False
            has_msg_end_token = False
            previous_role = None

            for msg_idx, message in enumerate(messages):
                if previous_role is None:
                    tokenize_role = True
                else:
                    tokenize_role = message["role"] != previous_role

                if msg_idx == len(messages) - 1:
                    has_ct_token = True
                    has_msg_end_token = True
                else:
                    if messages[msg_idx + 1]["role"] != message["role"]:
                        has_ct_token = True
                        has_msg_end_token = True
                    else:
                        has_ct_token = False
                        has_msg_end_token = False

                previous_role = message["role"]

                msg = self.tokenize_message(
                    message=message,
                    tokenize_role=tokenize_role,
                    has_ct_token=has_ct_token,
                    has_msg_end_token=has_msg_end_token,
                    extract_whisper_feature=True,
                    output_type=output_type,
                )
                msgs.append(msg)

            if add_assistant_start_msg:
                assistant_start_msg = self.tokenize_message(
                    message={"role": "assistant", "message_type": None},
                    tokenize_role=True,
                    has_ct_token=False,
                    has_msg_end_token=False,
                )
                msgs.append(assistant_start_msg)

            ret_msg = msgs[0]
            for msg in msgs[1:]:
                ret_msg.merge(msg)

            return ret_msg
