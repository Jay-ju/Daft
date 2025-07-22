# Copyright (c) Beijing Volcano Engine Technology Ltd.

from __future__ import annotations

import pandas as pd

import daft
from daft import col
from daft.las.functions.text.word_repetition_calculator import WordRepetitionCalculator
from daft.las.functions.udf import las_udf


def test_word_repetition_calculator_chinese(local_models_dir):
    chinese_samples = {
        "text": [
            "人工智能技术正在快速发展，人工智能技术已经广泛应用于各个领域。人工智能技术的进步为人类社会带来了巨大变化，人工智能技术的未来充满无限可能。",
            "这是一个独特的文本，没有任何重复的内容，每个词组都只出现一次。",
            "",
            None,
        ]
    }

    chinese_df = pd.DataFrame(chinese_samples)
    ds = daft.from_pandas(chinese_df)
    ds = ds.with_column(
        "repetition_ratio",
        las_udf(
            WordRepetitionCalculator,
            construct_args={
                "repetition": 2,
                "lang": "zh",
                "tokenization": True,
                "model_path": local_models_dir,
                "model_name": "kenlm/wikipedia",
            },
        )(col("text")),
    )

    actual_df = ds.to_pandas()

    assert len(actual_df) == 4
    assert actual_df["repetition_ratio"][0] > 0.1
    assert actual_df["repetition_ratio"][1] == 0.0
    assert pd.isna(actual_df["repetition_ratio"][2])
    assert pd.isna(actual_df["repetition_ratio"][3])


def test_word_repetition_calculator_english(local_models_dir):
    english_samples = {
        "text": [
            "The machine learning algorithm is very powerful. The machine learning algorithm can process large datasets efficiently. The machine learning algorithm provides accurate predictions.",
            "This is a unique text with no repeated content, each phrase appears only once.",
            "",
            None,
        ]
    }

    english_df = pd.DataFrame(english_samples)
    ds = daft.from_pandas(english_df)
    ds = ds.with_column(
        "repetition_ratio",
        las_udf(
            WordRepetitionCalculator,
            construct_args={
                "repetition": 2,
                "lang": "en",
                "tokenization": True,
                "model_path": local_models_dir,
                "model_name": "kenlm/wikipedia",
            },
        )(col("text")),
    )

    actual_df = ds.to_pandas()

    assert len(actual_df) == 4
    assert actual_df["repetition_ratio"][0] > 0.2
    assert actual_df["repetition_ratio"][1] < 0.1
    assert pd.isna(actual_df["repetition_ratio"][2])
    assert pd.isna(actual_df["repetition_ratio"][3])


def test_word_repetition_calculator_different_lengths(local_models_dir):
    chinese_samples = {
        "text": [
            "人工智能技术正在快速发展，人工智能技术已经广泛应用于各个领域。人工智能技术的进步为人类社会带来了巨大变化，人工智能技术的未来充满无限可能。",
        ]
    }

    chinese_df = pd.DataFrame(chinese_samples)
    ds = daft.from_pandas(chinese_df)

    ds = ds.with_column(
        "ratio_1",
        las_udf(
            WordRepetitionCalculator,
            construct_args={
                "repetition": 1,
                "lang": "zh",
                "tokenization": True,
                "model_path": local_models_dir,
                "model_name": "kenlm/wikipedia",
            },
        )(col("text")),
    )

    ds = ds.with_column(
        "ratio_5",
        las_udf(
            WordRepetitionCalculator,
            construct_args={
                "repetition": 5,
                "lang": "zh",
                "tokenization": True,
                "model_path": local_models_dir,
                "model_name": "kenlm/wikipedia",
            },
        )(col("text")),
    )

    actual_df = ds.to_pandas()

    assert len(actual_df) == 1
    assert actual_df["ratio_1"][0] > actual_df["ratio_5"][0]


def test_word_repetition_calculator_large_text(local_models_dir):
    large_text = "人工智能 " * 100 + "机器学习 " * 50 + "深度学习 " * 30

    large_samples = {"text": [large_text]}
    large_df = pd.DataFrame(large_samples)

    ds = daft.from_pandas(large_df)
    ds = ds.with_column(
        "repetition_ratio",
        las_udf(
            WordRepetitionCalculator,
            construct_args={
                "repetition": 2,
                "lang": "zh",
                "tokenization": True,
                "model_path": local_models_dir,
                "model_name": "kenlm/wikipedia",
            },
        )(col("text")),
    )

    actual_df = ds.to_pandas()

    assert len(actual_df) == 1
    assert actual_df["repetition_ratio"][0] > 0.5
