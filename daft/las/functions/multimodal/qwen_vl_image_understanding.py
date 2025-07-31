# Copyright (c) Beijing Volcano Engine Technology Ltd.

from __future__ import annotations

import base64
import logging
import time
from pathlib import Path
from typing import Any

from daft.dependencies import pa
from daft.las.functions.types import Operator
from daft.las.functions.utils.common_utils import log_op_call, path_to_base64, run_on_local_path

logger = logging.getLogger(__name__)


class QwenVLImageUnderstanding(Operator):
    """**Qwen2.5-VL 多模态图像理解模型 - 视觉语义解析与自然语言描述生成**

    **核心功能**

    - 多模态输入处理
      - 支持 `URL`/`Base64编码`/`二进制流` 三种图像格式
    - 视觉-语言联合建模
      - 实现图像内容到语义空间的精准映射
    - 对话式提示支持
      - 通过 `prompt` 参数引导生成方向

    **场景优化**
    - 中英文混合场景优化：特别针对中文语义增强
    - 支持先将图像尺寸修正为一致，建议按照图像和GPU情况设置图像尺寸
    """  # noqa: D415

    def __init__(
        self,
        image_src_type: str = "image_url",
        model_path: str = "/opt/las/models",
        model_name: str = "Qwen/Qwen2.5-VL-7B-Instruct",
        prompt: str = "请给出这张图片的详细描述。",
        batch_size: int = 4,
        dtype: str = "bfloat16",
        use_flash_attention_2: bool = True,
        max_caption_length: int = 256,
        resized_height: int | None = None,
        resized_width: int | None = None,
        rank: int | None = None,
        **kwargs: Any,
    ) -> None:
        """图像理解参数初始化.

        Args:
            image_src_type: 输入图像的格式类型，支持：
                - tos/http 地址(image_url)
                - base64 编码(image_base64)
                - 二进制流(image_binary)
                可选值：["image_url", "image_base64", "image_binary"]
                默认值："image_url"
            model_path: 本地模型文件存储的绝对路径，默认为容器内预置路径。当使用自定义模型时需修改此路径
                默认值："/opt/las/models"
            model_name: 支持的视觉语言模型版本，当前仅支持 Qwen2.5-VL系列模型
                可选值：["Qwen/Qwen2.5-VL-7B-Instruct"]
                默认值："Qwen/Qwen2.5-VL-7B-Instruct"
            prompt: 用户理解图片内容的提示词，模型会根据提示词来生成图像的描述。
                默认值："请给出这张图片的详细描述。"
            batch_size: 单次推理处理的样本数量。较大的batch_size可提升吞吐但增加显存消耗，建议根据GPU显存调整。
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
            resized_height: 预处理时统一缩放图像的高度（像素单位），空值保留原始尺寸。增大尺寸可保留细节但显著增加显存占用
                默认值：None
            resized_width: 预处理时统一缩放图像的宽度（像素单位），空值保留原始尺寸。建议与resized_height配合使用
                默认值：None
            rank: 指定使用的GPU设备编号（多卡环境有效）。例如：0表示第一张GPU，1表示第二张GPU
                默认值：None
        """
        super().__init__(**kwargs)

        self.image_src_type = image_src_type
        self.model_path = model_path
        self.model_name = model_name
        self.prompt = prompt
        self.batch_size = batch_size
        self.dtype = dtype
        self.use_flash_attention_2 = use_flash_attention_2
        self.max_caption_length = max_caption_length
        self.resized_height = resized_height
        self.resized_width = resized_width
        self.rank = rank

        # These packages are heavy, so we import them lazily.
        import torch
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        model_dir = Path(self.model_path) / self.model_name
        if not model_dir.exists():
            raise FileNotFoundError(f"Model directory not found: {model_dir}")

        # Device initialization
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

        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            str(model_dir),
            torch_dtype=self.torch_dtype,
            device_map=self.model_device,
            attn_implementation="flash_attention_2" if self.use_flash_attention_2 else None,
        )

        logger.info(
            "Model initialization configuration:\n"
            "- Model Name: %s\n"
            "- Storage Path: %s\n"
            "- Computation Precision: %s\n"
            "- Acceleration: %s\n"
            "- Device: %s\n"
            "- Image Size: %s x %s",
            self.model_name,
            model_dir,
            self.dtype,
            "FlashAttention2" if self.use_flash_attention_2 else "Native attention",
            self.model_device,
            self.resized_height,
            self.resized_width,
        )
        logger.debug("Prompt template: %s", self.prompt)

        self.processor = AutoProcessor.from_pretrained(str(model_dir), trust_remote_code=True)

        log_op_call(logger=logger, op=self.__class__.__name__, model_service_or_lib=self.model_name)

    def _process_image_data(self, img_data: Any) -> str:
        if self.image_src_type == "image_base64":
            return f"data:image;base64,{img_data}"
        if self.image_src_type == "image_binary":
            return f"data:image;base64,{base64.b64encode(img_data).decode('utf-8')}"
        if self.image_src_type == "image_url":
            return f"data:image;base64, \
                    {run_on_local_path(img_data, lambda path: path_to_base64(path))}"
        raise ValueError(f"Unsupported image type: {self.image_src_type}")

    def _build_message_template(self, image_base64: str) -> list[dict[str, Any]]:
        message: list[dict[str, Any]] = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_base64},
                    {"type": "text", "text": self.prompt},
                ],
            }
        ]
        if self.resized_height:
            message[0]["content"][0]["resized_height"] = self.resized_height
        if self.resized_width:
            message[0]["content"][0]["resized_width"] = self.resized_width
        return message

    def _prepare_model_inputs(self, batch_messages: list[list[dict[str, Any]]]) -> dict[str, Any]:
        batch_texts = self.processor.apply_chat_template(batch_messages, tokenize=False, add_generation_prompt=True)
        from qwen_vl_utils import process_vision_info

        batch_image_inputs, batch_video_inputs = process_vision_info(batch_messages)

        return self.processor(
            text=batch_texts,
            images=batch_image_inputs,
            videos=batch_video_inputs,
            padding=True,
            return_tensors="pt",
            padding_side="left",
        ).to(self.data_device)

    def _decode_generated_text(self, inputs: Any, generated_ids: list[list[int]]) -> list[str]:
        trimmed_ids = [out[len(inp) :] for inp, out in zip(inputs.input_ids, generated_ids)]
        return self.processor.batch_decode(trimmed_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)

    def transform(self, images: pa.Array) -> pa.Array:
        """对输入的图像数组进行批量处理，生成包含视觉理解结果的文本描述。

        Args:
            images: 包含图像数据的数组，元素类型为 字符串 或者 二进制。

        Returns:
            pa.Array: 处理后的数组，元素为每个图片的视觉理解结果。

        Raises:
            ValueError: 当输入数据格式不符合要求时抛出
            RuntimeError: 模型推理过程中发生错误时抛出
        """  # noqa: D415
        start_time = time.monotonic()
        logger.info("Starting batch processing, input type: %s", self.image_src_type)

        all_captions = []
        total_images = len(images)
        total_batches = (total_images + self.batch_size - 1) // self.batch_size
        for batch_idx in range(0, total_images, self.batch_size):
            sub_images = images.slice(batch_idx, self.batch_size)
            current_batch = sub_images.to_pylist()
            logger.debug("Processing batch %.2f with %d images", (batch_idx + 1) / total_batches, len(current_batch))

            if not current_batch:
                break
            batch_messages = []

            try:
                for img_data in current_batch:
                    image_base64 = self._process_image_data(img_data)
                    message = self._build_message_template(image_base64)
                    batch_messages.append(message)
                inputs = self._prepare_model_inputs(batch_messages)
                generated_ids = self.model.generate(**inputs, max_new_tokens=self.max_caption_length)
                batch_captions = self._decode_generated_text(inputs, generated_ids)
                all_captions.extend(batch_captions)
            except RuntimeError:
                logger.exception("Model inference failed (possibly OOM)!")
                logger.info("Current batch size: %d, consider reducing batch_size", self.batch_size)
                all_captions.extend([""] * len(sub_images))
            except Exception:
                logger.exception("Inference error!")
                all_captions.extend([""] * len(sub_images))

        processing_time = time.monotonic() - start_time

        logger.info(
            "Completed %d images | Total time: %.2fs | Throughput: %.2f img/s",
            total_images,
            processing_time,
            total_images / processing_time,
        )

        return pa.array(all_captions, type=self.__return_column_type__())

    @staticmethod
    def __return_column_type__() -> pa.DataType:
        return pa.string()
