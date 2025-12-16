# Copyright (c) Beijing Volcano Engine Technology Ltd.

from __future__ import annotations

import pandas as pd

import daft
from daft import col
from daft.las.functions.udf import las_udf
from daft.las.functions.video import VideoConvert
from tests.las.functions import assert_dataframe_result


def generate_test_data(tos_test_data_dir, local_test_data_dir, http_test_data_dir):
    samples = {
        "input_path": [
            f"{tos_test_data_dir}/video/non-exist.avi",
            f"{tos_test_data_dir}/video/music_sample.mov",
        ],
        "target_output_path": [
            f"{tos_test_data_dir}/video/video_convert/non-exist.mp4",
            f"{tos_test_data_dir}/video/video_convert/music_sample.mp4",
        ],
    }
    input_df = pd.DataFrame(samples)

    expected_output_paths = [
        None,
        f"{tos_test_data_dir}/video/video_convert/music_sample.mp4",
    ]
    expected_df = pd.DataFrame(
        {
            "input_path": samples["input_path"],
            "target_output_path": samples["target_output_path"],
            "result": expected_output_paths,
        }
    )

    return input_df, expected_df


def test_video_convert_basic(tos_test_data_dir, local_test_data_dir, http_test_data_dir):
    """Test VideoConvert operator with basic conversion parameters."""
    input_df, expected_df = generate_test_data(tos_test_data_dir, local_test_data_dir, http_test_data_dir)
    constructor_kwargs = {
        "output_format": "mp4",
        "extra_params": ["-crf", "23", "-preset", "medium"],
    }

    df = daft.from_pandas(input_df)
    result_df = df.with_column(
        "result",
        las_udf(
            VideoConvert,
            construct_args=constructor_kwargs,
            num_gpus=0,
            batch_size=1,
            concurrency=1,
        )(col("input_path"), col("target_output_path")),
    ).to_pandas()

    assert_dataframe_result(result_df, expected_df)
