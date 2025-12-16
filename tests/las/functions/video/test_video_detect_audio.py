# Copyright (c) Beijing Volcano Engine Technology Ltd.

from __future__ import annotations

import pandas as pd

import daft
from daft import col
from daft.las.functions.udf import las_udf
from daft.las.functions.video import VideoDetectAudio
from tests.las.functions import assert_dataframe_result


def generate_test_data(tos_test_data_dir, local_test_data_dir, http_test_data_dir):
    samples = {
        "input_path": [
            f"{tos_test_data_dir}/video/non-exist.mp4",
            f"{tos_test_data_dir}/video/music_sample.mp4",
            f"{tos_test_data_dir}/video/music_sample_no_audio.mp4",
        ],
    }
    input_df = pd.DataFrame(samples)

    expected_results = [
        False,
        True,
        False,
    ]
    expected_df = pd.DataFrame(
        {
            "input_path": samples["input_path"],
            "result": expected_results,
        }
    )

    return input_df, expected_df


def test_video_detect_audio_basic(tos_test_data_dir, local_test_data_dir, http_test_data_dir):
    """Test VideoDetectAudio operator with basic parameters."""
    input_df, expected_df = generate_test_data(tos_test_data_dir, local_test_data_dir, http_test_data_dir)
    constructor_kwargs = {
        "timeout": None,
    }

    df = daft.from_pandas(input_df)
    result_df = df.with_column(
        "result",
        las_udf(
            VideoDetectAudio,
            construct_args=constructor_kwargs,
            num_gpus=0,
            batch_size=1,
            concurrency=1,
        )(col("input_path")),
    ).to_pandas()

    assert_dataframe_result(result_df, expected_df)
