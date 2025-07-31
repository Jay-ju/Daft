# Copyright (c) Beijing Volcano Engine Technology Ltd.
# Some code here has been modified from:
# https://huggingface.co/spaces/huggingface/text-data-filtering
# --------------------------------------------------------

from __future__ import annotations

import logging
from typing import Any

from daft.dependencies import pa
from daft.las.functions.types import Operator
from daft.las.functions.utils.common_utils import log_op_call

logger = logging.getLogger(__name__)


class PerplexityCalculator(Operator):
    """**困惑度计算算子 - 基于语言模型的文本质量评估解决方案**

    **核心功能**

    - **语言模型评估**
      - 基于 KenLM 语言模型计算文本困惑度
      - 支持中英文文本质量评估
      - 提供文本可读性指标
    - **质量评估**
      - 困惑度越低，文本质量越高
      - 适用于文本质量筛选和评估

    **技术实现**
    - **模型核心**
      - KenLM: 高效的语言模型推理
      - SentencePiece: 中英文分词处理
    - **计算优化**
      - 批量处理提升效率
      - 内存友好的模型加载
    """  # noqa: D415

    def __init__(
        self,
        lang: str = "zh",
        model_path: str = "/opt/las/models",
        model_name: str = "kenlm/wikipedia",
        rank: int = 0,
        **kwargs: Any,
    ) -> None:
        """困惑度计算算子初始化方法

        Args:
            lang: 语种
                描述：需要计算的文本的语种
                可选值：["en", "zh"]
                默认值："zh"
            model_path: 模型文件所在的路径
                默认值："/opt/las/models"
            model_name: 模型名称
                默认值："kenlm/wikipedia"
            rank: GPU 编号
                描述：GPU 编号，用于模型加载
                默认值：0
        """  # noqa: D415
        super().__init__(**kwargs)

        self.lang = lang
        self.model_path = model_path
        self.model_name = model_name
        self.rank = rank

        if self.lang not in ["zh", "en"]:
            raise ValueError(f"Unsupported language: {self.lang}. Supported: zh, en")

        self._initialize_models()

        logger.info("Perplexity calculator initialized for language: %s", self.lang)
        logger.info("Model path: %s", self.model_path)
        logger.info("Model name: %s", self.model_name)

        log_op_call(logger=logger, op=self.__class__.__name__, model_service_or_lib=self.model_name)

    def _initialize_models(self) -> None:
        try:
            from pathlib import Path

            import kenlm
            import sentencepiece as spm

            from daft.las.functions.utils.common_utils import run_on_local_path
        except ImportError as e:
            logger.error("Required dependencies not available: %s", e)
            logger.error("Please install: pip install sentencepiece kenlm")
            self._tokenizer = None
            self._kenlm_model = None
            return

        try:

            def load_tokenizer(local_path: str) -> spm.SentencePieceProcessor:
                tokenizer = spm.SentencePieceProcessor()
                tokenizer.load(local_path)
                return tokenizer

            model_dir = str(Path(self.model_path) / self.model_name)
            sp_model_path = str(Path(model_dir) / f"{self.lang}.sp.model")

            self._tokenizer = run_on_local_path(sp_model_path, load_tokenizer)
            logger.info("SentencePiece model loaded successfully")

            def load_kenlm_model(local_path: str) -> kenlm.Model:
                return kenlm.Model(local_path)

            kl_model_path = str(Path(model_dir) / f"{self.lang}.arpa.bin")

            self._kenlm_model = run_on_local_path(kl_model_path, load_kenlm_model)
            logger.info("KenLM model loaded successfully")

        except Exception as e:
            logger.error("Failed to load models: %s", e)
            self._tokenizer = None
            self._kenlm_model = None

    def _compute_perplexity(self, text: str) -> float:
        if self._kenlm_model is None:
            logger.error("KenLM model not available, returning 0.0")
            return 0.0

        if self._tokenizer is None:
            logger.error("Tokenizer not available, returning 0.0")
            return 0.0

        try:
            words = self._tokenizer.encode_as_pieces(text)
            text = " ".join(words)

            logits, length = 0, 0
            for line in text.splitlines():
                if line.strip():
                    logits += self._kenlm_model.score(line)
                    length += len(line.split()) + 1

            if length == 0:
                return 0.0

            ppl = 10.0 ** (-logits / length)
            return round(ppl, 1)

        except Exception as e:
            logger.error("Error computing perplexity for text: %s", e)
            return 0.0

    def transform(self, texts: pa.Array) -> pa.Array:
        """批量计算文本困惑度.

        Args:
            texts: 待处理的文本列，要求元素类型为字符串。

        Returns:
            pyarrow.Array: 困惑度值列，元素为浮点数类型。

        """
        perplexities: list[float | None] = []

        for text in texts:
            text = text.as_py()
            if text is None or (isinstance(text, str) and not text.strip()):
                perplexities.append(None)
            elif isinstance(text, str):
                perplexity = self._compute_perplexity(text)
                perplexities.append(perplexity)
            else:
                perplexities.append(None)

        logger.info("Perplexity calculation completed for %d texts", len(texts))

        return pa.array(perplexities, type=self.__return_column_type__())

    @staticmethod
    def __return_column_type__() -> pa.DataType:
        return pa.float64()
