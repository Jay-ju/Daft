# Copyright (c) Beijing Volcano Engine Technology Ltd.

from __future__ import annotations

import logging
import tempfile
import time
from pathlib import Path
from typing import Any

from daft.dependencies import pa
from daft.las.functions.types import Operator
from daft.las.functions.utils.common_utils import tracking_usage
from daft.las.io import download_file

logger = logging.getLogger(__name__)


class QwenOmniAudioUnderstanding(Operator):
    """**Qwen2.5-Omni 多模态音频理解模型 - 音频内容解析与自然语言描述生成 核心功能**

    **核心功能**
    - 智能内容理解与描述生成
      - 基于音频信号自动生成详细准确的自然语言描述，支持通过 prompt 参数自定义提示词
      - 支持mp3、acc、m4a等音频格式
    - 高效模型加载与推理优化
      - 支持多种计算精度选择（bfloat16、float16、float32），适配不同性能需求
      - 集成FlashAttention2加速技术，显著提升推理效率
      - 支持自动或手动设备分配，完美适配单卡/多卡环境

    **场景优化**
    - 广泛适用于音频内容分析、智能音频描述、多媒体内容理解等应用场景
    - 内置批处理机制，高效支持大规模音频数据并行推理
    - 灵活配置返回音频数据选项，满足多样化业务需求
    """  # noqa: D415

    def __init__(
        self,
        model_path: str = "/opt/las/models",
        model_name: str = "Qwen/Qwen2.5-Omni-7B",
        prompt: str = "请给出这个音频的详细描述。",
        batch_size: int = 4,
        dtype: str = "bfloat16",
        use_flash_attention_2: bool = True,
        max_caption_length: int = 256,
        rank: int | None = None,
        **kwargs: Any,
    ) -> None:
        """多模态音频理解参数初始化.

        Args:
            model_path: 本地模型文件存储的绝对路径，默认为容器内预置路径。当使用自定义模型时需修改此路径
                默认值："/opt/las/models"
            model_name: 支持的多模态模型版本，当前支持Qwen2.5-Omni系列模型
                可选值：["Qwen/Qwen2.5-Omni-7B"]
                默认值："Qwen/Qwen2.5-Omni-7B"
            prompt: 用户理解音频内容的提示词，模型会根据提示词来生成音频的描述
                默认值："请给出这个音频的详细描述。"
            batch_size: 单次推理处理的样本数量。较大的batch_size可提升吞吐但增加显存消耗，建议根据GPU显存调整
                默认值：4
            dtype: 模型推理精度选择：
                - bfloat16: 平衡精度与速度
                - float16: 更快的推理速度
                - float32: 最高精度但显存消耗最大
                可选值：["bfloat16", "float16", "float32"]
                默认值："bfloat16"
            use_flash_attention_2: 是否使用Flash Attention 2优化注意力计算（需CUDA兼容且dtype为16位浮点时生效）
                默认值：True
            max_caption_length: 模型生成描述的最大token数。较长的生成可能包含更多细节但增加计算时间
                默认值：256
            rank: 指定使用的GPU设备编号（多卡环境有效）。例如：0表示第一张GPU，1表示第二张GPU
                默认值：None
        """
        super().__init__(**kwargs)

        self.model_path = model_path
        self.model_name = model_name
        self.prompt = prompt
        self.batch_size = batch_size
        self.dtype = dtype
        self.use_flash_attention_2 = use_flash_attention_2
        self.max_caption_length = max_caption_length
        self.rank = rank

        # These packages are heavy, so we import them lazily.
        import torch
        from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor

        model_dir = Path(self.model_path) / self.model_name
        if not model_dir.exists():
            raise FileNotFoundError(f"Model directory not found: {model_dir}")

        use_gpu = self.use_gpu and torch.cuda.is_available()
        if self.rank is None:
            self.model_device = "auto"
            self.data_device = "cuda" if use_gpu else "cpu"
        else:
            self.model_device = f"cuda:{self.rank % self.cuda_device_count}" if use_gpu else "cpu"
            self.data_device = self.model_device
        logger.info("Model will be loaded on device: %s", self.model_device)

        # Type conversion mapping
        dtype_mapping = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        self.torch_dtype = dtype_mapping.get(self.dtype)
        if not self.torch_dtype:
            raise ValueError(f"Unsupported precision type: {self.dtype}")

        self.model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
            str(model_dir),
            device_map=self.model_device,
            torch_dtype=self.torch_dtype,
            attn_implementation="flash_attention_2" if self.use_flash_attention_2 else None,
        )

        logger.info(
            "Model initialization configuration:\n"
            "- Model Name: %s\n"
            "- Storage Path: %s\n"
            "- Computation Precision: %s\n"
            "- Acceleration: %s\n"
            "- Device: %s\n",
            self.model_name,
            model_dir,
            self.dtype,
            "FlashAttention2" if self.use_flash_attention_2 else "Native attention",
            self.model_device,
        )
        logger.debug("Prompt template: %s", self.prompt)

        self.processor = Qwen2_5OmniProcessor.from_pretrained(str(model_dir), trust_remote_code=True)
        tracking_usage(op=self.__class__.__name__, model_service_or_lib=self.model_name)

    def _build_message_template(self, content: str) -> list[dict[str, Any]]:
        message: list[dict[str, Any]] = [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech.",
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "audio", "audio": content},
                    {"type": "text", "text": self.prompt},
                ],
            },
        ]
        return message

    def _prepare_model_inputs(self, batch_messages: list[list[dict[str, Any]]]) -> dict[str, Any]:
        batch_texts = self.processor.apply_chat_template(batch_messages, tokenize=False, add_generation_prompt=True)
        from qwen_omni_utils import process_mm_info

        batch_audio_inputs, batch_image_inputs, batch_video_inputs = process_mm_info(
            batch_messages, use_audio_in_video=False
        )

        inputs = (
            self.processor(
                text=batch_texts,
                audio=batch_audio_inputs,
                images=batch_image_inputs,
                videos=batch_video_inputs,
                return_tensors="pt",
                padding=True,
                use_audio_in_video=False,
            )
            .to(self.model.device)
            .to(self.model.dtype)
        )

        return inputs

    def transform(self, contents: pa.Array) -> pa.Array:
        """对输入的音频内容数组进行批量处理，生成包含音频理解结果的文本描述.

        Args:
            contents: 包含音频数据的数组，元素类型为字符串（文件路径或URL）。
            支持本地文件路径以及tos://、s3://、http://、https://开头的远程文件链接

        Returns:
            处理后的数组，元素为每个音频的理解结果文本。对于处理失败的音频，返回空字符串。

        Raises:
            RuntimeError: 模型推理过程中发生错误时抛出（如内存不足）
            Exception: 其他推理相关错误时抛出
        """
        start_time = time.monotonic()

        all_captions = []
        total_content = len(contents)
        total_batches = (total_content + self.batch_size - 1) // self.batch_size
        with tempfile.TemporaryDirectory() as temp_dir:
            for batch_idx in range(0, total_content, self.batch_size):
                sub_contents = contents.slice(batch_idx, self.batch_size)
                current_batch = sub_contents.to_pylist()
                logger.debug(
                    "Processing batch %.2f with %d audio files", (batch_idx + 1) / total_batches, len(current_batch)
                )

                if not current_batch:
                    break
                batch_messages = []

                try:
                    for path in current_batch:
                        if path.startswith(("tos://", "s3://", "http://", "https://")):
                            temp_file_path = str(Path(temp_dir, Path(path).name))
                            download_file(path, temp_file_path)

                        else:
                            temp_file_path = path
                        message = self._build_message_template(temp_file_path)
                        batch_messages.append(message)
                    inputs = self._prepare_model_inputs(batch_messages)

                    text_ids = self.model.generate(
                        **inputs, max_new_tokens=self.max_caption_length, use_audio_in_video=False, return_audio=False
                    )

                    batch_captions = self.processor.batch_decode(
                        text_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
                    )
                    batch_captions = [caption.split("\nassistant\n")[-1] for caption in batch_captions]
                    all_captions.extend(batch_captions)

                except RuntimeError:
                    logger.exception("Model inference failed (possibly OOM)!")
                    logger.info("Current batch size: %d, consider reducing batch_size", self.batch_size)
                    all_captions.extend([""] * len(sub_contents))
                except Exception:
                    logger.exception("Inference error!")
                    all_captions.extend([""] * len(sub_contents))

        processing_time = time.monotonic() - start_time

        logger.info(
            "Completed %d contents | Total time: %.2fs | Throughput: %.2f contents/s",
            total_content,
            processing_time,
            total_content / processing_time,
        )

        return pa.array(all_captions, type=self.__return_column_type__())

    @staticmethod
    def __return_column_type__() -> pa.DataType:
        return pa.string()
