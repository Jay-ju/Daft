# Copyright (c) Beijing Volcano Engine Technology Ltd.

from __future__ import annotations

import daft
from daft import col
from daft.las.functions.text.alphanumeric_ratio_calculator import AlphanumericRatioCalculator
from daft.las.functions.udf import las_udf


def test_alphanumeric_ratio_calculator_basic():
    test_texts = [
        "HelloWorld123",
        "Hello, world!",
        "Test 123! Is it working?",
        "!!!@@@###$$$",
        "abc123!!!@@@",
        "你好Hello123",
        "这是中文文本",
        "这是中文文本!!!",
        "",
        "     ",
        "9999999999abc",
        "a*b&c@1#2$",
        "1234567890",
        "Hello 😊😊123",
        "Hello\u2003World",
    ]

    df = daft.DataFrame._from_pylist([{"text": text} for text in test_texts])

    df = df.with_column(
        "alphanumeric_ratio",
        las_udf(
            AlphanumericRatioCalculator,
            construct_args={
                "tokenization": False,
            },
            num_gpus=0,
            batch_size=1,
            concurrency=1,
        )(col("text")),
    )

    result_df = df.select("text", "alphanumeric_ratio")
    result_list = result_df.to_pydict()

    assert len(result_list["alphanumeric_ratio"]) == 15

    assert result_list["alphanumeric_ratio"][0] == 1.0
    assert 0.76 <= result_list["alphanumeric_ratio"][1] <= 0.77
    assert 0.74 <= result_list["alphanumeric_ratio"][2] <= 0.76
    assert result_list["alphanumeric_ratio"][3] == 0.0
    assert result_list["alphanumeric_ratio"][4] == 0.5
    assert result_list["alphanumeric_ratio"][5] == 1.0
    assert result_list["alphanumeric_ratio"][6] == 1.0
    assert 0.66 <= result_list["alphanumeric_ratio"][7] <= 0.67
    assert result_list["alphanumeric_ratio"][8] is None
    assert result_list["alphanumeric_ratio"][9] is None
    assert result_list["alphanumeric_ratio"][10] == 1.0
    assert result_list["alphanumeric_ratio"][11] == 0.5
    assert result_list["alphanumeric_ratio"][12] == 1.0
    assert 0.72 <= result_list["alphanumeric_ratio"][13] <= 0.73
    assert 0.90 <= result_list["alphanumeric_ratio"][14] <= 0.91


def test_alphanumeric_ratio_calculator_tokenization(local_models_dir):
    test_texts = [
        "Hello world",
        "Hello, world!",
        "你好世界",
        "Hello 你好",
        "Test 123",
        "!!!@@@",
    ]

    df = daft.DataFrame._from_pylist([{"text": text} for text in test_texts])

    df = df.with_column(
        "alphanumeric_ratio",
        las_udf(
            AlphanumericRatioCalculator,
            construct_args={
                "tokenization": True,
                "model_path": local_models_dir,
                "model_name": "pythia-6.9b-deduped",
            },
            num_gpus=0,
            batch_size=1,
            concurrency=1,
        )(col("text")),
    )

    result_df = df.select("text", "alphanumeric_ratio")
    result_list = result_df.to_pydict()

    assert len(result_list["alphanumeric_ratio"]) == 6
    assert all(ratio >= 0.0 for ratio in result_list["alphanumeric_ratio"])
