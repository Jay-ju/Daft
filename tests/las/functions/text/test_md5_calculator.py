# Copyright (c) Beijing Volcano Engine Technology Ltd.

from __future__ import annotations

import pandas as pd

import daft
from daft import col
from daft.las.functions.text.md5_calculator import Md5Calculator
from daft.las.functions.udf import las_udf

samples = {
    "text": [
        "Hello World!",
        "The quick brown fox jumps over the lazy dog",
        "大数据处理与分布式计算",
    ]
}

expected_md5_values = [
    "ed076287532e86365e841e92bfc50d8c",
    "9e107d9d372bb6826bd81d3542a419d6",
    "3b70b38a60b19394e90e8a9b3df2bbdb",
]

input_df = pd.DataFrame(samples)


def test_compute_md5_hash():
    ds = daft.from_pandas(input_df)
    ds = ds.with_column(
        "md5",
        las_udf(
            Md5Calculator,
        )(col("text")),
    )

    actual_df = ds.to_pandas()

    for i, (original, md5_val) in enumerate(zip(input_df["text"], actual_df["md5"])):
        expected_md5 = expected_md5_values[i]
        assert md5_val == expected_md5, f"MD5 mismatch for '{original}': expected {expected_md5}, got {md5_val}"
