# Copyright (c) Beijing Volcano Engine Technology Ltd.

from __future__ import annotations

import pandas as pd

import daft
from daft import col
from daft.las.functions.text.text_safety_scorer import TextSafetyScorer
from daft.las.functions.udf import las_udf

samples_zh = {
    "text": [
        "今天天气很好。",
        "我想要你的银行卡密码和个人信息。",
    ]
}

samples_en = {
    "text": [
        "The weather is nice today.",
        "I want to hack into your computer and steal your data.",
    ]
}

input_df_zh = pd.DataFrame(samples_zh)
input_df_en = pd.DataFrame(samples_en)

batch_size = 2
model_name = "thu-coai/ShieldLM-6B-chatglm3"
rank = 0
num_gpus = 0
concurrency = 1


def test_text_safety_scorer_chinese(local_models_dir):
    ds = daft.from_pandas(input_df_zh)
    ds = ds.with_column(
        "safety_scores",
        las_udf(
            TextSafetyScorer,
            construct_args={
                "lang": "zh",
                "model_path": local_models_dir,
                "model_name": model_name,
                "batch_size": batch_size,
                "rank": rank,
            },
            num_gpus=num_gpus,
            batch_size=1,
            concurrency=concurrency,
        )(col("text")),
    )

    actual_df = ds.to_pandas()

    safe_result = actual_df["safety_scores"][0]
    assert isinstance(safe_result, dict)
    assert "safe" in safe_result
    assert "unsafe" in safe_result
    assert "controversial" in safe_result
    assert safe_result["safe"] > safe_result["unsafe"]
    assert safe_result["safe"] > safe_result["controversial"]

    unsafe_result = actual_df["safety_scores"][1]
    assert isinstance(unsafe_result, dict)
    assert unsafe_result["unsafe"] > unsafe_result["safe"]


def test_text_safety_scorer_english(local_models_dir):
    ds = daft.from_pandas(input_df_en)
    ds = ds.with_column(
        "safety_scores",
        las_udf(
            TextSafetyScorer,
            construct_args={
                "lang": "en",
                "model_path": local_models_dir,
                "model_name": model_name,
                "batch_size": batch_size,
                "rank": rank,
            },
            num_gpus=num_gpus,
            batch_size=1,
            concurrency=concurrency,
        )(col("text")),
    )

    actual_df = ds.to_pandas()

    safe_result = actual_df["safety_scores"][0]
    assert isinstance(safe_result, dict)
    assert "safe" in safe_result
    assert "unsafe" in safe_result
    assert "controversial" in safe_result
    assert safe_result["safe"] > safe_result["unsafe"]
    assert safe_result["safe"] > safe_result["controversial"]

    unsafe_result = actual_df["safety_scores"][1]
    assert isinstance(unsafe_result, dict)
    assert unsafe_result["unsafe"] > unsafe_result["safe"]
