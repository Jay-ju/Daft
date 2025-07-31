# Copyright (c) Beijing Volcano Engine Technology Ltd.
# Some code here has been modified from:
# https://huggingface.co/spaces/huggingface/text-data-filtering

from __future__ import annotations

import logging
from typing import Any

from daft.dependencies import pa
from daft.las.functions.types import Operator
from daft.las.functions.utils.common_utils import log_op_call
from daft.las.functions.utils.special_characters import SPECIAL_CHARACTERS

logger = logging.getLogger(__name__)


class WordRepetitionCalculator(Operator):
    """**词重复比例计算器 - 基于N-gram词组重复比例的文本特征提取**

    **核心功能**
    - 词重复比例计算：精确统计文本中重复词组的比例
    - 双语言支持：支持中文、英文的分词处理
    - 灵活配置：支持不同长度的N-gram词组计算

    **应用场景**
    - 文本质量评估
    - 数据清洗和预处理
    - 内容重复检测

    **技术特性**
    - 支持两种分词模式：
        - 基于空格分词：适用于英文等空格分隔的语言
        - 基于模型分词：使用sentencepiece模型，中文须使用此模式
    - 可配置N-gram长度：支持1-N个词的组合
    - 智能过滤：自动过滤特殊字符，提高计算准确性
    """  # noqa: D415

    def __init__(
        self,
        repetition: int = 5,
        lang: str = "zh",
        tokenization: bool = True,
        model_path: str = "/opt/las/models",
        model_name: str = "kenlm/wikipedia",
        **kwargs: Any,
    ) -> None:
        """词重复比例计算器初始化方法

        Args:
            repetition: 词组长度
                描述：用于拼接成词组的单词数量
                默认值：5
            lang: 语种
                描述：文本的语种，支持zh(中文)、en(英文)
                默认值："zh"
            tokenization: 是否使用分词器
                描述：是否使用sentencepiece模型进行分词，中文须开启
                默认值：True
            model_path: 模型文件所在的路径
                默认值："/opt/las/models"
            model_name: 模型名称
                默认值："kenlm/wikipedia"
        """  # noqa: D415
        super().__init__(**kwargs)

        self.repetition = repetition
        self.lang = lang
        self.tokenization = tokenization
        self.model_path = model_path
        self.model_name = model_name

        if self.lang not in ["zh", "en"]:
            raise ValueError(f"Unsupported language: {self.lang}. Supported: zh, en")

        if self.repetition < 1:
            raise ValueError(f"Repetition must be >= 1, got {self.repetition}")

        if self.tokenization:
            try:
                from pathlib import Path

                import sentencepiece as spm

                from daft.las.functions.utils.common_utils import run_on_local_path

                def load_tokenizer(local_path: str) -> spm.SentencePieceProcessor:
                    tokenizer = spm.SentencePieceProcessor()
                    tokenizer.load(local_path)
                    return tokenizer

                model_dir = str(Path(self.model_path) / self.model_name)
                full_model_path = str(Path(model_dir) / f"{self.lang}.sp.model")

                self._tokenizer = run_on_local_path(full_model_path, load_tokenizer)
                logger.info("SentencePiece tokenizer loaded successfully")
            except Exception as e:
                logger.error("Failed to load tokenizer: %s, falling back to space-based tokenization", str(e))
                self._tokenizer = None
        else:
            self._tokenizer = None

        logger.info("Word repetition calculator initialized with repetition=%d, lang=%s", self.repetition, self.lang)

        log_op_call(logger=logger, op=self.__class__.__name__, model_service_or_lib=self.model_name)

    def _tokenize_text(self, text: str) -> list[str]:
        if self._tokenizer:
            pieces = self._tokenizer.encode_as_pieces(text)
            return [piece for piece in pieces if piece.strip()]
        else:
            return text.split()

    def _refine_words(self, words: list[str]) -> list[str]:
        refined_words = []
        for word in words:
            word = word.lower()
            for char in SPECIAL_CHARACTERS:
                word = word.replace(char, "")
            if word.strip():
                refined_words.append(word.strip())
        return refined_words

    def _calculate_repetition_ratio(self, text: str) -> float:
        words = self._tokenize_text(text)
        words = self._refine_words(words)

        if len(words) < self.repetition:
            return 0.0

        word_ngrams = [" ".join(words[i : i + self.repetition]) for i in range(len(words) - self.repetition + 1)]

        if not word_ngrams:
            return 0.0

        freq_word_ngrams: dict[str, int] = {}
        for word_ngram in word_ngrams:
            freq_word_ngrams[word_ngram] = freq_word_ngrams.get(word_ngram, 0) + 1

        freq_values = list(freq_word_ngrams.values())
        total_occurrences = sum(freq_values)
        repeated_occurrences = sum(freq for freq in freq_values if freq > 1)

        ratio = repeated_occurrences / total_occurrences if total_occurrences > 0 else 0.0

        return ratio

    def transform(self, texts: pa.Array) -> pa.Array:
        """批量处理文本列，计算词重复比例

        Args:
            texts: 待处理的文本列，要求元素类型为字符串

        Returns:
            pa.Array: 词重复比例列，元素为浮点数，表示重复词组比例
        """  # noqa: D415
        results: list[float | None] = []

        for text in texts:
            text_value = text.as_py()
            if text_value is None or (isinstance(text_value, str) and not text_value.strip()):
                results.append(None)
                continue

            try:
                ratio = self._calculate_repetition_ratio(text_value)
                results.append(ratio)

                logger.debug("Text: %s, Word repetition ratio: %f", text_value[:50], ratio)

            except Exception as e:
                logger.error("Error processing text: %s, Error: %s", text_value[:50], str(e))
                results.append(None)

        return pa.array(results, type=pa.float64())

    @staticmethod
    def __return_column_type__() -> pa.DataType:
        return pa.float64()
