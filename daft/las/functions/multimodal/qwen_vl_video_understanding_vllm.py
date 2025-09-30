# Copyright (c) Beijing Volcano Engine Technology Ltd.

from __future__ import annotations

import logging
import tempfile
import time
from pathlib import Path
from typing import Any

from qwen_vl_utils import process_vision_info

from daft.dependencies import pa
from daft.las.functions.types import Operator
from daft.las.functions.utils.common_utils import generate_filename_base_input, save_file_to_local, tracking_usage

logger = logging.getLogger(__name__)


class QwenVLVideoUnderstandingVLLM(Operator):
    """**Qwen2.5-VL 多模态视频理解模型，支持时序语义解析与自然语言描述生成，适用于多种视频分析场景。**

    **主要功能**
        - 多模态时序建模：支持三种视频输入格式（URL、Base64编码、二进制流），灵活适配不同数据源。
        - 时空联合建模：自动捕捉视频的时空特征与语义关联，实现更精准的视频内容理解。
        - 对话式提示支持：通过 `prompt` 参数自定义生成方向，满足多样化业务需求。
        - 高效推理：集成VLLM推理引擎，支持 `bfloat16`、`float16`、`float32` 三种精度，充分利用GPU算力。

    **适用场景**
        - 视频内容理解与摘要
        - 智能监控与事件分析
        - 多模态对话与交互
        - 其他需要视频语义解析的AI应用

    **注意事项**
        - 仅支持GPU环境，建议根据显存和业务需求合理设置参数。
        - 推荐在推理前统一视频尺寸和帧率，以获得最佳效果。
    """  # noqa: D415

    def __init__(
        self,
        video_src_type: str = "video_url",
        model_path: str = "/opt/las/models",
        model_name: str = "Qwen/Qwen2.5-VL-7B-Instruct",
        prompt: str = "请给出这段视频的详细描述。",
        batch_size: int = 4,
        dtype: str = "bfloat16",
        max_model_len: int = 128000,
        max_num_seqs: int = 128,
        tensor_parallel_size: int = 1,
        enable_prefix_caching: bool = True,
        gpu_memory_utilization: float = 0.9,
        enforce_eager: bool = False,
        min_pixels: int | None = None,
        max_pixels: int | None = None,
        fps: float | None = None,
        temperature: float = 1.0,
        top_p: float = 0.2,
        repetition_penalty: float = 1.05,
        max_tokens: int = 8192,
        stop_token_ids: list[int] = [],
        seed: int = 42,
        **kwargs: Any,
    ) -> None:
        """初始化Qwen2.5-VL多模态视频理解模型参数.

        Args:
            video_src_type: 输入视频的格式类型。支持：
                - "video_url": tos/http地址
                - "video_base64": base64编码
                - "video_binary": 二进制流
                可选值：["video_url", "video_base64", "video_binary"]
                默认值："video_url"
            model_path: 本地模型文件存储的绝对路径。默认为容器内预置路径，当使用自定义模型时需修改此路径。
                默认值："/opt/las/models"
            model_name: 支持的视觉语言模型版本。当前仅支持Qwen2.5-VL系列模型。
                可选值：["Qwen/Qwen2.5-VL-7B-Instruct-AWQ", "Qwen/Qwen2.5-VL-7B-Instruct", "Qwen/Qwen2.5-VL-32B-Instruct-AWQ", "Qwen/Qwen2.5-VL-32B-Instruct", "Qwen/Qwen2.5-VL-72B-Instruct"]
                默认值："Qwen/Qwen2.5-VL-7B-Instruct"
            prompt: 用户理解视频内容的提示词，模型会根据提示词来生成视频的描述。设置为空时，建议针对每条数据传入特定的prompt。
                默认值："请给出这段视频的详细描述。"
            batch_size: 单次推理处理的样本数量。较大的batch_size可提升吞吐但增加显存消耗，建议根据GPU显存调整。
                默认值：4
            dtype: 模型推理精度选择。
                - "bfloat16": 平衡精度与速度
                - "float16": 更快的推理速度
                - "float32": 最高精度但显存消耗最大
                可选值：["bfloat16", "float16", "float32"]
                默认值："bfloat16"
            max_model_len: 支持的最大模型输入长度（token数），影响可处理视频描述的长度，不能超过128000。
                默认值：128000
            max_num_seqs: 单批次最大序列数，影响并发推理能力。
                默认值：128
            tensor_parallel_size: 张量并行的GPU数量，提升推理速度。
                默认值：1
            enable_prefix_caching: 是否启用前缀缓存以加速多轮推理。
                默认值：True
            gpu_memory_utilization: 单卡GPU显存利用率上限，范围0~1。
                默认值：0.9
            enforce_eager: 是否强制使用eager模式推理，调试或特殊场景可用。
                默认值：False
            min_pixels: 视频最小像素。不设置时，默认使用视频的原像素。视频像素越大，GPU显存占用越高。
                默认值：None
            max_pixels: 视频最大像素。不设置时，默认使用视频的原像素。视频像素越大，GPU显存占用越高。
                默认值：None
            fps: 视频帧率。不设置时，默认使用视频的原帧率。视频帧率越高，GPU显存占用越高。建议设置特定的帧率值。
                默认值：None
            temperature: 采样温度，控制生成内容的多样性。值越高生成越随机。
                默认值：1.0
            top_p: nucleus采样的概率阈值，控制生成内容的多样性。值越小生成越保守。
                默认值：0.2
            repetition_penalty: 重复惩罚系数，防止生成重复内容。值越大重复内容越少。
                默认值：1.05
            max_tokens: 单次生成的最大token数，影响描述长度。
                默认值：8192
            stop_token_ids: 生成时遇到这些token id则停止。用于自定义生成终止条件。
                默认值：[]
            seed: 随机种子，保证推理结果可复现。
                默认值：42
        """
        super().__init__(**kwargs)

        self.video_src_type = video_src_type
        self.model_path = model_path
        self.model_name = model_name
        self.prompt = prompt
        self.batch_size = batch_size
        self.dtype = dtype
        self.max_model_len = max_model_len
        self.max_num_seqs = max_num_seqs
        self.tensor_parallel_size = tensor_parallel_size
        self.enable_prefix_caching = enable_prefix_caching
        self.gpu_memory_utilization = gpu_memory_utilization
        self.enforce_eager = enforce_eager
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.fps = fps
        self.temperature = temperature
        self.top_p = top_p
        self.repetition_penalty = repetition_penalty
        self.max_tokens = max_tokens
        self.stop_token_ids = stop_token_ids
        self.seed = seed

        # These packages are heavy, so we import them lazily.
        from transformers import AutoProcessor
        from vllm import LLM, SamplingParams

        model_dir = Path(self.model_path) / self.model_name
        if not model_dir.exists():
            raise FileNotFoundError(f"Model directory not found: {model_dir}")

        self.model = LLM(
            model=str(model_dir),
            dtype=self.dtype,
            max_model_len=self.max_model_len,
            max_num_seqs=self.max_num_seqs,
            seed=self.seed,
            trust_remote_code=True,
            tensor_parallel_size=self.tensor_parallel_size,
            enable_prefix_caching=self.enable_prefix_caching,
            gpu_memory_utilization=self.gpu_memory_utilization,
            enforce_eager=self.enforce_eager,
            limit_mm_per_prompt={"image": 1, "video": 1},
        )

        logger.info(
            "Model initialization configuration:\n"
            "- Model Name: %s\n"
            "- Storage Path: %s\n"
            "- Computation Precision: %s\n"
            "- Device: %s\n"
            "- GPU number: %d\n"
            "- Video Size: %s"
            "- Video fps: %s\n",
            self.model_name,
            model_dir,
            self.dtype,
            "cuda",
            self.tensor_parallel_size,
            f"{self.min_pixels} - {self.max_pixels}",
            self.fps,
        )
        logger.debug("Prompt template: %s", self.prompt)

        self.sampling_params = SamplingParams(
            temperature=self.temperature,
            top_p=self.top_p,
            repetition_penalty=self.repetition_penalty,
            max_tokens=self.max_tokens,
            stop_token_ids=self.stop_token_ids,
        )
        self.processor = AutoProcessor.from_pretrained(str(model_dir), trust_remote_code=True)
        tracking_usage(op=self.__class__.__name__, model_service_or_lib=self.model_name)

    def _build_message_template(self, tmp_file_name: str, prompt: str) -> list[dict[str, Any]]:
        message: list[dict[str, Any]] = [
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": tmp_file_name},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        if self.min_pixels is not None:
            message[0]["content"][0]["min_pixels"] = self.min_pixels
        if self.max_pixels is not None:
            message[0]["content"][0]["max_pixels"] = self.max_pixels
        if self.fps is not None:
            message[0]["content"][0]["fps"] = self.fps
        return message

    def _decode_generated_text(self, inputs: Any, generated_ids: list[list[int]]) -> list[str]:
        trimmed_ids = [out[len(inp) :] for inp, out in zip(inputs.input_ids, generated_ids)]
        return self.processor.batch_decode(trimmed_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)

    def transform(self, videos: pa.Array, user_prompts: pa.Array | None = None) -> pa.Array:
        """对输入的视频数组进行批量处理，生成包含视觉理解结果的文本描述

        Args:
            videos: 包含视频数据的数组，元素类型为 字符串 或者 二进制。

        Returns:
            pa.Array: 处理后的数组，元素为每个视频的视觉理解结果。

        Raises:
            ValueError: 当输入数据格式不符合要求时抛出
            RuntimeError: 模型推理过程中发生错误时抛出
        """  # noqa: D415
        start_time = time.monotonic()
        logger.info("Starting batch processing, input type: %s", self.video_src_type)

        all_captions = []
        total_videos = len(videos)
        prompts_list = user_prompts.to_pylist() if user_prompts else [self.prompt] * total_videos

        total_batches = (total_videos + self.batch_size - 1) // self.batch_size
        for batch_idx in range(0, total_videos, self.batch_size):
            sub_videos = videos.slice(batch_idx, self.batch_size)
            current_batch = sub_videos.to_pylist()
            sub_prompts = prompts_list[batch_idx * self.batch_size : (batch_idx + 1) * self.batch_size]
            logger.debug("Processing batch %.2f with %d videos", (batch_idx + 1) / total_batches, len(current_batch))

            if not current_batch:
                break

            try:
                with tempfile.TemporaryDirectory(dir="/tmp") as tmp_dir:
                    inputs = []
                    for idx, video in enumerate(current_batch):
                        file_name = generate_filename_base_input(video, self.video_src_type, "mp4", batch_idx, idx)
                        tmp_file_name = save_file_to_local(video, self.video_src_type, tmp_dir, file_name)
                        logger.info("Downloading video to %s", tmp_file_name)

                        if not Path(tmp_file_name).exists():
                            raise FileNotFoundError(tmp_file_name)

                        prompt = sub_prompts[idx]
                        message = self._build_message_template(tmp_file_name, prompt)
                        text = self.processor.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
                        _, video_inputs = process_vision_info(message)
                        mm_data = {"video": video_inputs}
                        inputs.append(
                            {
                                "prompt": text,
                                "multi_modal_data": mm_data,
                            }
                        )

                    outputs = self.model.generate(inputs, sampling_params=self.sampling_params)
                    batch_captions = [output.outputs[0].text for output in outputs]
                    all_captions.extend(batch_captions)

            except FileNotFoundError:
                logger.exception("File not Found!")
                all_captions.extend([""] * len(sub_videos))
            except RuntimeError:
                logger.exception("Model inference failed (possibly OOM)!")
                logger.info("Current batch size: {self.batch_size}, consider reducing batch_size")
                all_captions.extend([""] * len(sub_videos))
            except Exception:
                logger.exception("Inference error!")
                all_captions.extend([""] * len(sub_videos))

        processing_time = time.monotonic() - start_time

        logger.info(
            "Completed %d videos | Total time: %.2fs | Throughput: %.2f video/s",
            total_videos,
            processing_time,
            total_videos / processing_time,
        )

        return pa.array(all_captions, type=self.__return_column_type__())

    @staticmethod
    def __return_column_type__() -> pa.DataType:
        return pa.string()
