# Copyright (c) Beijing Volcano Engine Technology Ltd.

from __future__ import annotations

import pandas as pd

import daft
from daft import col
from daft.las.functions.image import ImageAestheticScore
from daft.las.functions.udf import las_udf
from tests.las.functions import assert_dataframe_result


def generate_test_data(tos_test_data_dir, local_test_data_dir, http_test_data_dir):
    samples = {
        "input_path": [
            f"{tos_test_data_dir}/image/non-exist.png",  # 不存在的文件
            f"{tos_test_data_dir}/image/forest.jpg",  # 高美学质量图像
            f"{local_test_data_dir}/image/forest.jpg",
            f"{http_test_data_dir}/image/forest.jpg",
        ],
    }
    input_df = pd.DataFrame(samples)

    expected_results = [
        None,
        0.56,
        0.56,
        0.56,
    ]

    expected_df = pd.DataFrame(
        {
            "input_path": samples["input_path"],
            "aesthetic_score": expected_results,
        }
    )

    return input_df, expected_df


def test_image_aesthetic_score_basic(tos_test_data_dir, local_test_data_dir, http_test_data_dir):
    """Test ImageAestheticScore operator with default parameters."""
    input_df, _ = generate_test_data(tos_test_data_dir, local_test_data_dir, http_test_data_dir)
    constructor_kwargs = {
        "batch_size": 2,
        "device": "cpu",  # Use CPU device for testing
    }

    df = daft.from_pandas(input_df)
    result_df = df.with_column(
        "aesthetic_score",
        las_udf(
            ImageAestheticScore,
            construct_args=constructor_kwargs,
            num_gpus=0,
            batch_size=1,
            concurrency=1,
        )(col("input_path")),
    ).to_pandas()

    # Basic assertions using the assert_dataframe_result helper
    assert_dataframe_result(
        result_df,
        expect_columns=["input_path", "aesthetic_score"],
        expect_row_num=4,
    )
