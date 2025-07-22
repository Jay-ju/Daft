# Copyright (c) Beijing Volcano Engine Technology Ltd.

from __future__ import annotations

import daft
from daft import col
from daft.las.functions.text.repeated_lines_calculator import RepeatedLinesCalculator
from daft.las.functions.udf import las_udf


def test_repeated_lines_calculator_basic():
    test_texts = [
        "第一行内容\n第二行内容\n第三行内容",
        "第一行内容\n第二行内容\n第一行内容\n第三行内容",
        "第一行内容\n第一行内容\n第一行内容",
        "第一行内容\n\n第二行内容\n   \n第三行内容",
        "第一行内容\r\n第二行内容\r\n第一行内容",
    ]

    df = daft.DataFrame._from_pylist([{"text": text} for text in test_texts])

    df = df.with_column(
        "repeated_ratio",
        las_udf(
            RepeatedLinesCalculator,
            construct_args={},
            num_gpus=0,
            batch_size=1,
            concurrency=1,
        )(col("text")),
    )

    result_df = df.select("text", "repeated_ratio")
    result_list = result_df.to_pydict()

    assert len(result_list["repeated_ratio"]) == 5

    assert result_list["repeated_ratio"][0] == 0.0
    assert result_list["repeated_ratio"][1] == 0.25
    assert result_list["repeated_ratio"][2] == 0.6666666666666666
    assert result_list["repeated_ratio"][3] == 0.0
    assert result_list["repeated_ratio"][4] == 0.3333333333333333
