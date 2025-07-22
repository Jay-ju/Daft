# Copyright (c) Beijing Volcano Engine Technology Ltd.

from __future__ import annotations

import pandas as pd

import daft
from daft import col
from daft.las.functions.text.text_length_calculator import TextLengthCalculator
from daft.las.functions.udf import las_udf

samples = {
    "text": [
        "Hello World",
        "你好世界",
        "Python编程 is fun",
        "",
        None,
    ]
}
txt_input_df = pd.DataFrame(samples)


def test_text_length_calculator():
    ds = daft.from_pandas(txt_input_df)
    ds = ds.with_column(
        "length",
        las_udf(
            TextLengthCalculator,
            construct_args={},
        )(col("text")),
    )

    actual_df = ds.to_pandas()
    assert actual_df["length"][0] == 11
    assert actual_df["length"][1] == 4
    assert actual_df["length"][2] == 15
    assert pd.isna(actual_df["length"][3])
    assert pd.isna(actual_df["length"][4])
