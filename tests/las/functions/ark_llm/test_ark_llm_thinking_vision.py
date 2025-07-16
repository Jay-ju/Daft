# Copyright (c) Beijing Volcano Engine Technology Ltd.

from __future__ import annotations

import pandas as pd
import pytest

import daft
from daft import col
from daft.las.functions.ark_llm.ark_llm_thinking_vision import ArkLLMThinkingVision
from daft.las.functions.udf import las_udf
from tests.las.functions import assert_dataframe_result

INPUT_COLUMN_NAME = "image_path"
OUTPUT_COLUMN_NAME = "vision_result"


def generate_test_data(tos_test_data_dir, local_test_data_dir, http_test_data_dir):
    return pd.DataFrame(
        {
            INPUT_COLUMN_NAME: [
                f"{local_test_data_dir}/image/cat.png",
                f"{tos_test_data_dir}/image/cat.png",
                f"{http_test_data_dir}/image/invalid.png",
                f"{http_test_data_dir}/image/cat.png",
            ],
            "text": ["猫", "猫", "无效图片", "样本图片"],
        }
    )


@pytest.mark.ark_llm
def test_doubao_thinking_vision(tos_test_data_dir, local_test_data_dir, http_test_data_dir):
    ArkLLMThinkingVision._finish_reason_check = True

    input_pd_df = generate_test_data(tos_test_data_dir, local_test_data_dir, http_test_data_dir)
    df = daft.from_pandas(input_pd_df)

    df = df.with_column(
        OUTPUT_COLUMN_NAME,
        las_udf(
            ArkLLMThinkingVision,
            construct_args={
                "model": "doubao-1.5-thinking-vision-pro",
                "version": "250428",
                "image_format": "png",
                "inference_type": "online",
            },
        )(col(INPUT_COLUMN_NAME), col("text")),
    )

    df = df.with_column("llm_result", col(OUTPUT_COLUMN_NAME).struct.get("llm_result"))
    df = df.with_column("reasoning_content", col(OUTPUT_COLUMN_NAME).struct.get("reasoning_content"))
    df = df.with_column("finish_reason", col(OUTPUT_COLUMN_NAME).struct.get("finish_reason"))
    output_pd_df = df.to_pandas()

    assert len(output_pd_df["reasoning_content"][1]) > 0
    assert len(output_pd_df["llm_result"][1]) > 0

    expect_columns = [INPUT_COLUMN_NAME, "text", OUTPUT_COLUMN_NAME, "llm_result", "reasoning_content", "finish_reason"]
    expect_row_num = len(input_pd_df)

    assert_dataframe_result(
        actual_df=output_pd_df,
        expect_columns=expect_columns,
        expect_row_num=expect_row_num,
    )
