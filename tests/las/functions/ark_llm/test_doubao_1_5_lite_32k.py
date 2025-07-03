# Copyright (c) Beijing Volcano Engine Technology Ltd.

from __future__ import annotations

import pandas as pd
import pytest

import daft
from daft import col
from daft.las.functions.ark_llm.doubao_1_5_lite_32k import Doubao15Lite32k
from daft.las.functions.udf import las_udf

OUTPUT_COLUMN_NAME = "llm_result"


@pytest.mark.skip
def test_doubao_1_5_lite_32k():
    data = pd.DataFrame(
        {
            "query": [
                "java 进程和线程的区别",
                "中国的首都在哪里",
            ]
        }
    )

    ds = daft.from_pandas(data)
    ds = ds.with_column(
        OUTPUT_COLUMN_NAME,
        las_udf(Doubao15Lite32k, construct_args={"version": "250115", "inference_type": "batch"})(col("query")),
    )

    result_df = ds.to_pandas()
    print(result_df)
    assert "llm_result" in result_df.columns
    assert len(result_df) == 2
    assert "线程" in result_df["llm_result"][0]
    assert "北京" in result_df["llm_result"][1]
