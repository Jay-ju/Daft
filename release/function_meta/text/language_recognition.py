from __future__ import annotations

import os

import daft
from daft import col
from daft.las.functions.text.language_recognition import LanguageRecognitionOperator
from daft.las.functions.udf import las_udf

if __name__ == "__main__":
    samples = {
        "text": [
            "这是一行测试内容。",
            "This is a test content.",
            "This is a test content.这是一行测试内容。",
            "こんにちは",
            "안녕하세요",
        ]
    }

    ds = daft.from_pydict(samples)
    ds = ds.with_column(
        "language_result",
        las_udf(
            LanguageRecognitionOperator,
            construct_args={
                "model_path": os.getenv("MODEL_PATH", "./models"),
                "model_name": "fasttext/lid.176.bin",
                "batch_size": 1000,
            },
        )(col("text")),
    )

    ds.show()
    # ╭──────────────────────────────────────┬─────────────────────────────────────────────╮
    # │ text                                 ┆ language_result                             │
    # │ ---                                  ┆ ---                                         │
    # │ Utf8                                 ┆ Struct[language: Utf8, confidence: Float64] │
    # ╞══════════════════════════════════════╪═════════════════════════════════════════════╡
    # │ 这是一行测试内容。                   ┆ {language: zh, confidence: 1.000048279762268} │
    # │ This is a test content.              ┆ {language: en, confidence: 0.9209088683128357} │
    # │ This is a test content.这是一行测试… ┆ {language: zh, confidence: 0.6892356276512146} │
    # │ こんにちは                           ┆ {language: ja, confidence: 1.0000269412994385} │
    # │ 안녕하세요                           ┆ {language: ko, confidence: 0.9996028542518616} │
    # ╰──────────────────────────────────────┴─────────────────────────────────────────────╯
