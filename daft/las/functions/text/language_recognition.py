# Copyright (c) Beijing Volcano Engine Technology Ltd.

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

from daft.dependencies import pa
from daft.las.functions.types import Operator
from daft.las.functions.utils.common_utils import run_on_local_path

logger = logging.getLogger(__name__)


class LanguageRecognitionOperator(Operator):
    """**文本语种识别算子 - 基于FastText模型提供多语言识别能力**

    **核心功能**
    - 支持识别176种语言（基于fasttext/lid.176系列模型）
    - 批量推理优化，适合处理大规模文本数据
    - 支持同时输出语种标签及置信度分数
    - 支持本地路径和TOS路径的模型文件

    **应用场景**
    - 多语言文本分类
    - 语种过滤和筛选
    - 多语言数据集构建

    **建议**
    - 输入文本长度建议不少于10个字符以提高识别准确率
    - 需要至少4GB内存以进行模型批式推理
    """  # noqa: D415

    def __init__(
        self,
        model_path: str = "/opt/las/models",
        model_name: str = "fasttext/lid.176.bin",
        batch_size: int = 1000,
        **kwargs: Any,
    ) -> None:
        """语言识别算子初始化方法

        Args:
            model_path: 模型文件所在的路径
                默认值："/opt/las/models"
            model_name: 模型文件名，支持 "fasttext/lid.176.bin" 或 "fasttext/lid.176.ftz"
            batch_size: 批量处理大小，较大的batch_size可提升吐吐但增加内存消耗
        """  # noqa: D415
        super().__init__(**kwargs)

        if model_name not in ["fasttext/lid.176.bin", "fasttext/lid.176.ftz"]:
            raise ValueError(
                f"Unsupported model_name: {model_name}. Supported: fasttext/lid.176.bin, fasttext/lid.176.ftz"
            )

        self.model_path = model_path
        self.model_name = model_name
        self.batch_size = batch_size

        self._load_model()
        logger.info("Initialized LanguageRecognitionOperator")

    def _load_model(self) -> None:
        try:
            import fasttext
        except ImportError:
            raise ImportError("fasttext library is required. Please install with: pip install fasttext-wheel")

        try:
            model_dir = str(Path(self.model_path) / self.model_name)

            def load_model_from_path(path: str) -> Any:
                return fasttext.load_model(path)

            self._ft_model = run_on_local_path(model_dir, load_model_from_path)

            logger.info("Successfully loaded language model")
        except FileNotFoundError as e:
            logger.exception("Model file not found, model_path: %s", model_dir)
            raise RuntimeError(f"Model file not found at {model_dir}") from e
        except Exception as e:
            logger.exception("Unexpected error loading model, model_path: %s", model_dir)
            raise RuntimeError(f"Failed to load model from {model_dir}: {e!s}") from e

    def transform(self, texts: pa.Array) -> pa.Array:
        """批量计算文本的语种识别结果和置信度分数

        Args:
            texts: 原始字符串列，要求元素类型为字符串

        Returns:
            pyarrow.Array: 包含语种标签和置信度分数的结构体列
                每个元素包含 language 和 confidence 字段
        """  # noqa: D415
        start_time = time.monotonic()
        all_texts = [text.as_py().lower().replace("\n", " ") for text in texts]

        all_rec_lang_results = []
        all_rec_score_results = []
        total_batches = (len(all_texts) + self.batch_size - 1) // self.batch_size

        for batch_idx in range(total_batches):
            batch_start = batch_idx * self.batch_size
            batch_end = (batch_idx + 1) * self.batch_size
            current_batch = all_texts[batch_start:batch_end]

            logger.debug("Processing batch %d/%d with %d texts", batch_idx + 1, total_batches, len(current_batch))

            if not current_batch:
                break

            try:
                all_labels, all_probs = self._ft_model.predict(current_batch)
                lang_ids = [label[0].replace("__label__", "") for label in all_labels]
                lang_scores = [float(prob[0]) for prob in all_probs]
                all_rec_lang_results.extend(lang_ids)
                all_rec_score_results.extend(lang_scores)
            except Exception as e:
                logger.error("Inference error in batch %d: %s", batch_idx + 1, e)
                all_rec_lang_results.extend(["unknown"] * len(current_batch))
                all_rec_score_results.extend([0.0] * len(current_batch))

        processing_time = time.monotonic() - start_time
        logger.info(
            "Completed %d texts | Total time: %.2fs | Throughput: %.2f text/s",
            len(all_texts),
            processing_time,
            len(all_texts) / processing_time,
        )

        results = []
        for lang, score in zip(all_rec_lang_results, all_rec_score_results):
            results.append({"language": lang, "confidence": score})

        return pa.array(results, type=self.__return_column_type__())

    @staticmethod
    def __return_column_type__() -> pa.DataType:
        return pa.struct(
            [
                pa.field("language", pa.string()),
                pa.field("confidence", pa.float64()),
            ]
        )
