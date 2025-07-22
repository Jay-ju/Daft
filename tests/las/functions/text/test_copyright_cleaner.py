# Copyright (c) Beijing Volcano Engine Technology Ltd.

from __future__ import annotations

import daft
from daft import col
from daft.las.functions.text.copyright_cleaner import CopyrightCleaner
from daft.las.functions.udf import las_udf


def test_copyright_cleaner_basic():
    test_texts = [
        "/* \n * Copyright (c) 2023 Jane Smith\n * 本代码依据 Apache License 2.0 授权，详见随附的 LICENSE 文件。\n */ 你好",
        "# 版权 (c) 2023 Jane Smith\n# 本代码依据 Apache License 2.0 授权，详见随附的 LICENSE 文件。\n 你好",
    ]

    df = daft.DataFrame._from_pylist([{"text": text} for text in test_texts])

    df = df.with_column(
        "cleaned_text",
        las_udf(
            CopyrightCleaner,
            construct_args={},
            num_gpus=0,
            batch_size=1,
            concurrency=1,
        )(col("text")),
    )

    result_df = df.select("text", "cleaned_text")
    result_list = result_df.to_pydict()

    assert len(result_list["cleaned_text"]) == 2

    assert result_list["cleaned_text"][0] == " 你好"
    assert result_list["cleaned_text"][1] == " 你好"
