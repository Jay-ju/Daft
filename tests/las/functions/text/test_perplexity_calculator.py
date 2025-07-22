# Copyright (c) Beijing Volcano Engine Technology Ltd.

from __future__ import annotations

import pandas as pd

import daft
from daft import col
from daft.las.functions.text.perplexity_calculator import PerplexityCalculator
from daft.las.functions.udf import las_udf


def test_perplexity_calculator_chinese(local_models_dir):
    samples = {
        "text": [
            "人工智能技术正在快速发展，人工智能技术已经广泛应用于各个领域。",
            "这是一个测试句子。",
            "这是一个正常的句子。",
            "这是一个正常的句子。这是一个正常的句子。这是一个正常的句子。",
            "乱码文本 12345 !@#$%",
        ]
    }
    df = pd.DataFrame(samples)
    ds = daft.from_pandas(df)
    ds = ds.with_column(
        "perplexity",
        las_udf(
            PerplexityCalculator,
            construct_args={
                "lang": "zh",
                "model_path": local_models_dir,
                "model_name": "kenlm/wikipedia",
            },
            num_gpus=0,
            batch_size=1,
            concurrency=1,
        )(col("text")),
    )
    actual_df = ds.to_pandas()
    print(f"Chinese test perplexities: {actual_df['perplexity'].tolist()}")
    assert len(actual_df) == 5
    assert actual_df["perplexity"][0] > 0
    assert actual_df["perplexity"][1] > 0
    assert actual_df["perplexity"][2] > 0
    assert actual_df["perplexity"][3] > 0
    assert actual_df["perplexity"][4] > 0
    assert actual_df["perplexity"][2] < actual_df["perplexity"][3]
    assert actual_df["perplexity"][3] < actual_df["perplexity"][4]


def test_perplexity_calculator_english(local_models_dir):
    samples = {
        "text": [
            "Machine learning is a subset of artificial intelligence that enables computers to learn and make decisions without being explicitly programmed.",
            "The development of renewable energy sources is crucial for addressing climate change and ensuring sustainable future.",
        ]
    }
    df = pd.DataFrame(samples)
    ds = daft.from_pandas(df)
    ds = ds.with_column(
        "perplexity",
        las_udf(
            PerplexityCalculator,
            construct_args={
                "lang": "en",
                "model_path": local_models_dir,
                "model_name": "kenlm/wikipedia",
            },
            num_gpus=0,
            batch_size=1,
            concurrency=1,
        )(col("text")),
    )
    actual_df = ds.to_pandas()
    print(f"English test perplexities: {actual_df['perplexity'].tolist()}")
    assert len(actual_df) == 2
    assert actual_df["perplexity"][0] > 0
    assert actual_df["perplexity"][1] > 0
