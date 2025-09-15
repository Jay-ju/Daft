# Copyright (c) Beijing Volcano Engine Technology Ltd.

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

from daft.dependencies import pa
from daft.las.functions.types import Operator
from daft.las.functions.utils.common_utils import tracking_usage

logger = logging.getLogger(__name__)


class MultilingualTextTranslate(Operator):
    """**Seed-X-Instruct-7B 多语言文本翻译模型 - 跨语言文本翻译 核心功能**

    **核心功能**
    - 多语言智能翻译
      - 支持多种语言间的文本转换，可通过source_language和target_language参数自定义源语言和目标语言
      - 基于Seed-X-Instruct-7B模型，提供高质量翻译结果
      - 支持语种请参考: https://huggingface.co/ByteDance-Seed/Seed-X-Instruct-7B
    - 灵活配置与优化
      - 支持多种计算精度选择（bfloat16等），适配不同性能需求
      - 集成张量并行处理和前缀缓存技术，显著提升推理效率
      - 支持自动或手动设备分配，完美适配单卡/多卡环境

    **场景优化**
    - 广泛适用于跨语言内容转换、多语言文档处理、国际化应用开发等场景
    - 内置批处理机制，高效支持大规模文本数据并行翻译
    - 支持精确控制最大生成 tokens 数量，满足多样化业务需求
    """  # noqa: D415

    def __init__(
        self,
        model_path: str = "/opt/las/models",
        model_name: str = "Seed-X-Instruct-7B",
        dtype: str = "bfloat16",
        max_model_len: int = 32768,
        max_num_seqs: int = 128,
        tensor_parallel_size: int = 1,
        enable_prefix_caching: bool = True,
        gpu_memory_utilization: float = 0.9,
        use_cot: bool = False,
        source_language: str = "Chinese",
        target_language: str = "English",
        max_tokens: int = 1024,
        batch_size: int = 4,
        seed: int = 42,
        **kwargs: Any,
    ) -> None:
        """多语言文本翻译模型参数初始化.

        Args:
            model_path: 本地模型文件存储的绝对路径，默认为容器内预置路径。当使用自定义模型时需修改此路径
                默认值："/opt/las/models"
            model_name: 支持的多语言模型名称，当前支持Seed-X-Instruct系列模型
                默认值："Seed-X-Instruct-7B"
            dtype: 模型推理精度选择
                默认值："bfloat16"
            max_model_len: 模型支持的最大序列长度
                默认值：32768
            max_num_seqs: 模型同时处理的最大序列数量
                默认值：128
            tensor_parallel_size: 张量并行计算的设备数量，用于多GPU并行推理
                默认值：1
            enable_prefix_caching: 是否启用前缀缓存机制，可提升重复前缀的推理效率
                默认值：True
            gpu_memory_utilization: GPU内存使用比例，范围0-1
                默认值：0.9
            use_cot: 是否使用思维链(Chain-of-Thought)模式进行翻译
                默认值：False
            source_language: 源语言名称，支持的语言请参考模型文档
                默认值："Chinese"
            target_language: 目标语言名称，支持的语言请参考模型文档
                默认值："English"
            max_tokens: 模型生成翻译结果的最大token数
                默认值：1024
            batch_size: 单次推理处理的文本样本数量
                默认值：4
            seed: 随机数种子，用于结果复现
                默认值：42
        """
        super().__init__(**kwargs)

        self.model_path = model_path
        self.model_name = model_name
        self.batch_size = batch_size
        self.dtype = dtype
        self.max_model_len = max_model_len
        self.max_num_seqs = max_num_seqs
        self.max_tokens = max_tokens
        self.seed = seed
        self.tensor_parallel_size = tensor_parallel_size
        self.enable_prefix_caching = enable_prefix_caching
        self.gpu_memory_utilization = gpu_memory_utilization
        self.use_cot = use_cot
        self.source_language = source_language
        self.target_language = target_language

        model_dir = Path(self.model_path) / self.model_name
        if not model_dir.exists():
            raise FileNotFoundError(f"Model directory not found: {model_dir}")

        # 获取语言 mapping 信息
        language_file_path = str(model_dir / "multilingual_language_map.json")
        with open(language_file_path) as f:
            language_mapping = json.load(f)[0]

        if self.source_language not in language_mapping:
            raise ValueError(f"Source language {self.source_language} not found in language mapping")
        if self.target_language not in language_mapping:
            raise ValueError(f"Target language {self.target_language} not found in language mapping")

        self.target_language_code = language_mapping[self.target_language]

        # These packages are heavy, so we import them lazily.
        from vllm import LLM, SamplingParams

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
        )

        self.decoding_params = SamplingParams(temperature=0, max_tokens=self.max_tokens, skip_special_tokens=True)

        logger.info(
            "Model initialization configuration:\n"
            "- Model Name: %s\n"
            "- Storage Path: %s\n"
            "- Computation Precision: %s\n",
            self.model_name,
            model_dir,
            self.dtype,
        )

        tracking_usage(op=self.__class__.__name__, model_service_or_lib=self.model_name)

    def _build_message_template(self, content: str) -> str:
        if self.use_cot:
            message = f"Translate the following {self.source_language} sentence into {self.target_language} and explain it in detail:\n{content} <{self.target_language_code}>"
        else:
            message = f"Translate the following {self.source_language} sentence into {self.target_language}:\n{content} <{self.target_language_code}>"
        return message

    def transform(self, contents: pa.Array) -> pa.Array:
        """对输入的文本内容数组进行批量处理，生成多语言翻译结果.

        Args:
            contents: 包含待翻译文本的数组，元素类型为字符串。

        Returns:
            处理后的数组，元素为每个文本的翻译结果。对于处理失败的文本，返回空字符串。

        Raises:
            RuntimeError: 模型推理过程中发生错误时抛出（如内存不足）
            Exception: 其他推理相关错误时抛出
        """
        start_time = time.monotonic()

        all_translations = []
        total_content = len(contents)
        total_batches = (total_content + self.batch_size - 1) // self.batch_size
        for batch_idx in range(0, total_content, self.batch_size):
            sub_contents = contents.slice(batch_idx, self.batch_size)
            current_batch = sub_contents.to_pylist()
            logger.debug(
                "Processing batch %.2f with %d text files", (batch_idx + 1) / total_batches, len(current_batch)
            )

            if not current_batch:
                break
            batch_messages = []

            try:
                for text in current_batch:
                    message = self._build_message_template(text)
                    batch_messages.append(message)

                results = self.model.generate(batch_messages, self.decoding_params)
                batch_translations = [res.outputs[0].text.strip() for res in results]
                all_translations.extend(batch_translations)

            except RuntimeError:
                logger.exception("Model inference failed (possibly OOM)!")
                logger.info("Current batch size: %d, consider reducing batch_size", self.batch_size)
                all_translations.extend([""] * len(sub_contents))
            except Exception:
                logger.exception("Inference error!")
                all_translations.extend([""] * len(sub_contents))

        processing_time = time.monotonic() - start_time

        logger.info(
            "Completed %d contents | Total time: %.2fs | Throughput: %.2f contents/s",
            total_content,
            processing_time,
            total_content / processing_time,
        )

        return pa.array(all_translations, type=self.__return_column_type__())

    @staticmethod
    def __return_column_type__() -> pa.DataType:
        return pa.string()
