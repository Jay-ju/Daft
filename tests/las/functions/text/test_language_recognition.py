# Copyright (c) Beijing Volcano Engine Technology Ltd.

from __future__ import annotations

import daft
from daft import col
from daft.las.functions.text.language_recognition import LanguageRecognitionOperator
from daft.las.functions.udf import las_udf


def test_language_recognition_basic(local_models_dir):
    test_texts = [
        "这是一行测试内容。",
        "This is a test content.",
        "This is a test content.这是一行测试内容。",
        "こんにちは",
        "안녕하세요",
    ]

    df = daft.DataFrame._from_pylist([{"text": text} for text in test_texts])

    df = df.with_column(
        "language_result",
        las_udf(
            LanguageRecognitionOperator,
            construct_args={
                "model_path": local_models_dir,
                "model_name": "fasttext/lid.176.bin",
                "batch_size": 1000,
            },
            num_gpus=0,
            batch_size=1,
            concurrency=1,
        )(col("text")),
    )

    df = df.with_column("language", df["language_result"]["language"])
    df = df.with_column("confidence", df["language_result"]["confidence"])

    result_df = df.select("text", "language", "confidence")
    result_list = result_df.to_pydict()

    assert len(result_list["language"]) == 5

    assert result_list["language"][0] == "zh"
    assert result_list["language"][1] == "en"
    assert result_list["language"][2] == "zh"
    assert result_list["language"][3] == "ja"
    assert result_list["language"][4] == "ko"

    assert result_list["confidence"][0] > 0.5
    assert result_list["confidence"][1] > 0.5
    assert result_list["confidence"][2] > 0.5
    assert result_list["confidence"][3] > 0.5
    assert result_list["confidence"][4] > 0.5
