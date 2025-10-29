# Copyright (c) Beijing Volcano Engine Technology Ltd.

from __future__ import annotations

import pandas as pd

import daft
from daft import col
from daft.las.functions.udf import las_udf
from daft.las.functions.video import VideoConvertToMp4
from tests.las.functions import assert_dataframe_result


def generate_test_data(tos_test_data_dir, local_test_data_dir, http_test_data_dir):
    samples = {
        "input_path": [
            f"{tos_test_data_dir}/video/non-exist.mp4",
            f"{tos_test_data_dir}/video/sample.mp4",
            f"{local_test_data_dir}/video/sample.mp4",
            f"{http_test_data_dir}/video/sample.mp4",
        ],
        "target_output_path": [
            f"{tos_test_data_dir}/video/outputs/video_convert_to_mp4/non-exist.mp4",
            f"{tos_test_data_dir}/video/outputs/video_convert_to_mp4/sample_converted.mp4",
            f"{tos_test_data_dir}/video/outputs/video_convert_to_mp4/sample_converted.mp4",
            f"{tos_test_data_dir}/video/outputs/video_convert_to_mp4/sample_converted.mp4",
        ],
    }
    input_df = pd.DataFrame(samples)

    expected_output_paths = [
        None,
        f"{tos_test_data_dir}/video/outputs/video_convert_to_mp4/sample_converted.mp4",
        f"{tos_test_data_dir}/video/outputs/video_convert_to_mp4/sample_converted.mp4",
        f"{tos_test_data_dir}/video/outputs/video_convert_to_mp4/sample_converted.mp4",
    ]
    expected_df = pd.DataFrame(
        {
            "input_path": samples["input_path"],
            "target_output_path": samples["target_output_path"],
            "result": expected_output_paths,
        }
    )

    return input_df, expected_df


def test_video_convert_to_mp4_basic(tos_test_data_dir, local_test_data_dir, http_test_data_dir):
    """Test VideoConvertToMp4 operator with basic conversion parameters."""
    input_df, expected_df = generate_test_data(tos_test_data_dir, local_test_data_dir, http_test_data_dir)
    constructor_kwargs = {
        "video_codec": "libx264",
        "crf": 23,
        "preset": "medium",
        "audio_codec": "aac",
        "audio_bitrate": "192k",
    }

    df = daft.from_pandas(input_df)
    result_df = df.with_column(
        "result",
        las_udf(
            VideoConvertToMp4,
            construct_args=constructor_kwargs,
            num_gpus=0,
            batch_size=1,
            concurrency=1,
        )(col("input_path"), col("target_output_path")),
    ).to_pandas()

    assert_dataframe_result(result_df, expected_df)
