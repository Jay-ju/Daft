# Copyright (c) Beijing Volcano Engine Technology Ltd.

from __future__ import annotations

import pandas as pd

import daft
from daft import col
from daft.las.functions.text.maximum_word_length_calculator import MaximumWordLengthCalculator
from daft.las.functions.udf import las_udf

samples = {
    "text": [
        "Hello world 你好世界",
        "Python编程 is fun",
        "这是一个中文句子",
        "The quick brown fox jumps over the lazy dog",
        "supercalifragilisticexpialidocious is a very long word",
        "",
        None,
    ]
}
txt_input_df = pd.DataFrame(samples)


def test_maximum_word_length_calculator():
    ds = daft.from_pandas(txt_input_df)
    ds = ds.with_column(
        "max_word_length",
        las_udf(
            MaximumWordLengthCalculator,
            construct_args={},
        )(col("text")),
    )

    actual_df = ds.to_pandas()
    assert actual_df["max_word_length"][0] == 5
    assert actual_df["max_word_length"][1] == 6
    assert actual_df["max_word_length"][2] == 0
    assert actual_df["max_word_length"][3] == 5
    assert actual_df["max_word_length"][4] == 34
    assert pd.isna(actual_df["max_word_length"][5])
    assert pd.isna(actual_df["max_word_length"][6])
