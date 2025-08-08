# Copyright (c) Beijing Volcano Engine Technology Ltd.

from __future__ import annotations

import pandas as pd

import daft
from daft import col
from daft.las.functions.text.chinese_text_converter import ChineseTextConverter
from daft.las.functions.udf import las_udf


def test_chinese_text_converter_basic():
    samples = {
        "text": [
            "這是一個繁體中文的測試文本，包含了一些專業術語和技術名詞。",
            "Hello 世界！這裡有中英文混雜的內容，測試OpenCC是否能正確處理。",
            "简体中文文本，应该保持不变或根据转换方向进行处理。",
            "Mixed content: 繁體字轉換測試 with English words and numbers 123。",
            "",
            None,
        ]
    }

    df = pd.DataFrame(samples)
    ds = daft.from_pandas(df)

    ds = ds.with_column(
        "converted_text",
        las_udf(
            ChineseTextConverter,
            construct_args={
                "direction": "t2s",
            },
            num_gpus=0,
            batch_size=1,
            concurrency=1,
        )(col("text")),
    )

    actual_df = ds.to_pandas()
    assert len(actual_df) == 6
    assert isinstance(actual_df["converted_text"][0], str)
    assert isinstance(actual_df["converted_text"][1], str)
    assert isinstance(actual_df["converted_text"][2], str)
    assert isinstance(actual_df["converted_text"][3], str)
    assert actual_df["converted_text"][4] == ""
    assert pd.isna(actual_df["converted_text"][5])

    assert "这是一个繁体中文的测试文本" in actual_df["converted_text"][0]
    assert "Hello 世界" in actual_df["converted_text"][1]
    assert "这里有中英文混杂的内容" in actual_df["converted_text"][1]
    assert "繁体字转换测试" in actual_df["converted_text"][3]
    assert "with English words and numbers 123" in actual_df["converted_text"][3]
