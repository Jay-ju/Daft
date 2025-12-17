# Copyright (c) Beijing Volcano Engine Technology Ltd.
from __future__ import annotations

import json
import logging
import os
import re
import time
import unicodedata
from abc import ABC, abstractmethod
from typing import Any

import pypinyin
import torch
import uroman as ur
from pypinyin import Style
from torchaudio.pipelines import MMS_FA as bundle

from daft.dependencies import pa
from daft.las.functions.types import Operator
from daft.las.functions.utils.audio_utils import decode_audio_torchaudio

logger = logging.getLogger(__name__)


class Normalizer(ABC):
    def __init__(self, allowed_chars: set[str]):
        self.allowed_chars = allowed_chars

    @abstractmethod
    def __call__(self, text: str) -> tuple[str, list[str], list[str]]:
        pass


class EnglishNormalizer(Normalizer):
    def __init__(self, allowed_chars: set[str]):
        super().__init__(allowed_chars)

        self.digit_to_word = {
            "0": "zero",
            "1": "one",
            "2": "two",
            "3": "three",
            "4": "four",
            "5": "five",
            "6": "six",
            "7": "seven",
            "8": "eight",
            "9": "nine",
        }

        self.word_to_digit = {
            "zero": "0",
            "one": "1",
            "two": "2",
            "three": "3",
            "four": "4",
            "five": "5",
            "six": "6",
            "seven": "7",
            "eight": "8",
            "nine": "9",
        }

    def __call__(self, text: str) -> tuple[str, list[str], list[str]]:
        # Unicode normalization and lowercase
        text = unicodedata.normalize("NFKC", text).lower()

        # Always allow spaces for word segmentation
        def filter_to_allowed(s: str) -> str:
            return "".join(ch if (ch in self.allowed_chars or ch == " ") else " " for ch in s)

        def collapse_spaces(s: str) -> str:
            return re.sub(r"\s+", " ", s).strip()

        def expand_digits_en(m: re.Match) -> str:  # type: ignore[type-arg]
            digits = m.group(0)
            return " ".join(self.digit_to_word.get(d, d) for d in digits)

        text = re.sub(r"\d+", expand_digits_en, text)

        # Keep allowed characters: a-z, ', - (if included in the dictionary), replace the rest with spaces
        text = filter_to_allowed(text)
        text = collapse_spaces(text)
        words = text.split()
        return text, words, [self.word_to_digit.get(word, word) for word in words]


class ChineseNormalizer(Normalizer):
    def __init__(self, allowed_chars: set[str]):
        super().__init__(allowed_chars)

        self.zh_digit_to_pinyin = {
            "零": "ling",
            "一": "yi",
            "二": "er",
            "三": "san",
            "四": "si",
            "五": "wu",
            "六": "liu",
            "七": "qi",
            "八": "ba",
            "九": "jiu",
            "十": "shi",
            "百": "bai",
            "千": "qian",
        }

        self.zh_pinyin_to_digit = {
            "ling": "零",
            "yi": "一",
            "er": "二",
            "san": "三",
            "si": "四",
            "wu": "五",
            "liu": "六",
            "qi": "七",
            "ba": "八",
            "jiu": "九",
            "shi": "十",
            "bai": "百",
            "qian": "千",
        }

        self.digit_to_pinyin = {
            "0": "ling",
            "1": "yi",
            "2": "er",
            "3": "san",
            "4": "si",
            "5": "wu",
            "6": "liu",
            "7": "qi",
            "8": "ba",
            "9": "jiu",
        }

        self.pinyin_to_digit = {
            "ling": "0",
            "yi": "1",
            "er": "2",
            "san": "3",
            "si": "4",
            "wu": "5",
            "liu": "6",
            "qi": "7",
            "ba": "8",
            "jiu": "9",
        }

        self.uroman = ur.Uroman()

    def __call__(self, text: str) -> tuple[str, list[str], list[str]]:
        # Unicode normalization and lowercase
        text = unicodedata.normalize("NFKC", text).lower()

        # Always allow spaces for word segmentation
        def filter_to_allowed(s: str) -> str:
            return "".join(ch if (ch in self.allowed_chars or ch == " ") else " " for ch in s)

        def collapse_spaces(s: str) -> str:
            return re.sub(r"\s+", " ", s).strip()

        # Determine Chinese characters (CJK Unified Ideographs range)
        def is_han_char(ch: str) -> bool:
            try:
                codepoint = ord(ch)
                return (
                    0x4E00 <= codepoint <= 0x9FFF
                    or 0x3400 <= codepoint <= 0x4DBF
                    or 0x20000 <= codepoint <= 0x2A6DF
                    or 0x2A700 <= codepoint <= 0x2B73F
                    or 0x2B740 <= codepoint <= 0x2B81F
                    or 0x2B820 <= codepoint <= 0x2CEAF
                    or 0xF900 <= codepoint <= 0xFAFF
                    or 0x2F800 <= codepoint <= 0x2FA1F
                )
            except Exception:
                return False

        def to_pinyin(text: str, style: Style = Style.NORMAL, separator: str = " ") -> str:
            pinyin_list = pypinyin.lazy_pinyin(text, style=style)
            return separator.join(pinyin_list)

        # Convert the numbers to Chinese digits digit by digit and do word segmentation (separated by spaces).
        # Directly replace the Chinese digits with pinyin English words to avoid uroman from converting them into Arabic numerals
        spaced_tokens: list[str] = []
        for ch in text:
            if ch.isdigit():
                # Replace with the corresponding pinyin word and add it directly as ASCII
                spaced_tokens.append(self.digit_to_pinyin.get(ch, ch))
            elif is_han_char(ch):
                spaced_tokens.append(self.zh_digit_to_pinyin.get(ch, ch))
            else:
                # Non-Chinese characters are directly used as a token (unified filtering will be performed later), and non-whitespace
                # delimiters are treated as independent tokens to avoid adhesion
                spaced_tokens.append((ch if ch in self.allowed_chars else " ") if ch.strip() else " ")

        spaced_input = " ".join(spaced_tokens)

        # Eliminate characters other than Chinese characters that are not in allowed and replace pinyin with numbers
        processed_tokens = []
        for token in spaced_input.split():
            if token.lower() in self.zh_pinyin_to_digit:
                processed_tokens.append(self.zh_pinyin_to_digit[token.lower()])
            elif is_han_char(token):
                processed_tokens.append(token)
            else:
                # Eliminate characters other than Chinese characters that are not in allowed
                filtered_token = "".join(ch for ch in token if ch in self.allowed_chars)
                if filtered_token:
                    processed_tokens.append(filtered_token)
        raw_words = (" ".join(processed_tokens)).split()

        try:
            # Translate to ASCII using uroman
            romanized_text = self.uroman.romanize_string(to_pinyin(text=spaced_input, separator=""))
        except Exception as e:
            # romanized_text = spaced_input
            raise AssertionError(f"Failed to romanize string '{spaced_input}': {e}")

        # Filter to MMS_FA dictionary allowing characters and collapsing spaces
        romanized_text = filter_to_allowed(romanized_text)
        romanized_text = collapse_spaces(romanized_text)
        nor_words = romanized_text.split()

        if len(nor_words) != len(raw_words):
            raise AssertionError(
                f"normalized word size {len(nor_words)} != {len(raw_words)},\n"
                f"text: '{text}',\n"
                f"nor_words: {nor_words},\n"
                f"raw_words: {raw_words}"
            )

        return romanized_text, nor_words, raw_words


class AudioCTCAligner(Operator):
    """音频 CTC 对齐算子，用于将音频与文本按时间戳实施对齐，目前支持中文和英文。

    Args:
        model_path: 模型文件所在路径，你可以从 https://dl.fbaipublicfiles.com/mms/torchaudio/ctc_alignment_mling_uroman/model.pt
                    下载模型，并从本地离线加载
        model_name: 模型文件名，默认值为 MMS/ctc_alignment_mling_uroman_model.pt

    Returns:
        返回结果为 JSON 格式字符串（如下所示），每个元素包含 word（单词）、score（置信度）、start（开始时间戳）和 end（结束时间戳），其中时间戳以毫秒为单位。

        [
        {"word": "i", "score": 1.0, "start": 644, "end": 664},
        {"word": "had", "score": 0.98, "start": 704, "end": 845},
        {"word": "that", "score": 1.0, "start": 885, "end": 1026},
        {"word": "curiosity", "score": 1.0, "start": 1086, "end": 1790},
        {"word": "beside", "score": 0.97, "start": 1871, "end": 2314},
        {"word": "me", "score": 1.0, "start": 2334, "end": 2414},
        {"word": "at", "score": 1.0, "start": 2495, "end": 2575},
        {"word": "this", "score": 1.0, "start": 2595, "end": 2756},
        {"word": "moment", "score": 1.0, "start": 2837, "end": 3138},
    ]
    """  # noqa: D415

    def __init__(
        self,
        model_path: str = "/opt/las/models",
        model_name: str = "MMS/ctc_alignment_mling_uroman_model.pt",
        **kwargs: Any,
    ):
        super().__init__(**kwargs)

        start_time = time.time()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.allowed_chars = {ch for token in bundle.get_dict().keys() for ch in token if ch != "-" and ch != "*"}
        self.en_normalizer = EnglishNormalizer(self.allowed_chars)
        self.zh_normalizer = ChineseNormalizer(self.allowed_chars)

        self.model_path = f"{model_path}/{model_name}"
        self.model = bundle.get_model(
            with_star=False,
            dl_kwargs={
                "model_dir": os.path.dirname(self.model_path),
                "file_name": os.path.basename(self.model_path),
            },
        )
        self.model.to(self.device)
        self.model.eval()

        self.tokenizer = bundle.get_tokenizer()
        self.aligner = bundle.get_aligner()

        logger.info(
            "Finish initializing audio ctc aligner, model path: %s, device: %s, elapsed: %ss",
            self.model_path,
            self.device,
            round(time.time() - start_time, 2),
        )

    def transform(self, audios: pa.Array, texts: pa.Array, langs: pa.Array) -> pa.Array:
        results = [
            self._align(audio.as_py(), text.as_py(), lang.as_py()) for audio, text, lang in zip(audios, texts, langs)
        ]
        return pa.array(obj=results, type=self.__return_column_type__())

    @staticmethod
    def __return_column_type__() -> pa.DataType:
        return pa.large_string()

    def _align(self, audio: str | bytes, text: str, lang: str) -> str:
        if text is None or not isinstance(text, str) or text.strip() == "":
            raise ValueError(f"Input text can't be None or empty, but got {text}")

        # Normalize & Split
        (_, nor_words, raw_words) = self._normalize(text, lang)
        if len(nor_words) == 0:
            logger.warning("No valid word found after normalization, input text [%s]", text)
            return json.dumps([])

        # Load audio
        waveform, sample_rate = decode_audio_torchaudio(source=audio, sample_rate=16000, num_channels=1)

        # CTC FA
        with torch.inference_mode():
            emission, _ = self.model(waveform.to(self.device))
            token_spans = self.aligner(emission[0], self.tokenizer(nor_words))

        if len(nor_words) != len(token_spans):
            raise RuntimeError(f"Expected {len(nor_words)} token spans, but got {len(token_spans)}")

        return json.dumps(self._build_result(waveform, token_spans, emission.size(1), raw_words, sample_rate))

    def _normalize(self, text: str, lang: str) -> tuple[str, list[str], list[str]]:
        if lang == "en":
            return self.en_normalizer(text)
        elif lang == "zh":
            return self.zh_normalizer(text)
        else:
            raise NotImplementedError(f"Input lang must be 'en' or 'zh', but got {lang}")

    @staticmethod
    def _build_result(waveform, token_spans, num_frames, words, sample_rate) -> list[dict[str, Any]]:  # type: ignore[no-untyped-def]
        def cal_score(spans):  # type: ignore[no-untyped-def]
            return sum(s.score * len(s) for s in spans) / sum(len(s) for s in spans)

        result = []

        ratio = waveform.size(1) / num_frames
        for i in range(len(words)):
            span = token_spans[i]
            score = round(cal_score(span), 2)  # type: ignore[no-untyped-call]
            x0 = int(ratio * span[0].start)
            x1 = int(ratio * span[-1].end)
            start = int(round(x0 / sample_rate, 3) * 1000)
            end = int(round(x1 / sample_rate, 3) * 1000)
            result.append(
                {
                    "word": words[i],
                    "score": score,
                    "start": start,
                    "end": end,
                }
            )

        return result
