# Copyright (c) Beijing Volcano Engine Technology Ltd.

from __future__ import annotations

import pandas as pd

import daft
from daft import col
from daft.las.functions.udf import las_udf
from daft.las.functions.video import VideoAdaptiveCompress
from tests.las.functions import assert_dataframe_result


def test_video_adaptive_compress(tos_test_data_dir, local_test_data_dir):
    samples = {
        "videos": [
            "",
            f"{tos_test_data_dir}/video/video_adaptive_compress/sample.mp4",
        ],
    }
    input_df = pd.DataFrame(samples)

    expected_output_paths = [
        "",
        f"{tos_test_data_dir}/video/video_adaptive_compress/sample_compressed.mp4",
    ]

    expected_df = pd.DataFrame(
        {
            "videos": samples["videos"],
            "compressed_videos": expected_output_paths,
        }
    )

    df = daft.from_pandas(input_df)
    output_tos_dir = f"{tos_test_data_dir}/video/video_adaptive_compress"
    constructor_kwargs = {
        "output_tos_dir": output_tos_dir,
        "max_output_size_mb": 50.0,
        "target_fps": 5.0,
        "min_resolution_height": 360,
    }

    df = df.with_column(
        "compressed_videos",
        las_udf(
            VideoAdaptiveCompress,
            construct_args=constructor_kwargs,
            num_gpus=1,
            batch_size=1,
            concurrency=1,
        )(col("videos")),
    )
    actual_df = df.select("videos", "compressed_videos").to_pandas()

    assert_dataframe_result(actual_df, expected_df)
