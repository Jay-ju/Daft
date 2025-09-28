# Copyright (c) Beijing Volcano Engine Technology Ltd.

from __future__ import annotations

import pandas as pd

import daft
from daft import col
from daft.las.functions.udf import las_udf
from daft.las.functions.video import VideoResizeResolution
from tests.las.functions import assert_dataframe_result


def generate_test_data(tos_test_data_dir, local_test_data_dir):
    samples = {
        "videos": [
            "",
            f"{local_test_data_dir}/video/video_resize_resolution/non-exist.mp4",
            f"{tos_test_data_dir}/video/video_resize_resolution/sample.mp4",
        ],
    }
    input_df = pd.DataFrame(samples)

    expected_output_paths = [
        "",
        "",
        f"{tos_test_data_dir}/video/video_resize_resolution/sample_resized.mp4",
    ]

    expected_df = pd.DataFrame(
        {
            "videos": samples["videos"],
            "resized_videos": expected_output_paths,
        }
    )

    return input_df, expected_df


def test_video_resize_resolution_cpu(tos_test_data_dir, local_test_data_dir):
    input_df, expected_df = generate_test_data(tos_test_data_dir, local_test_data_dir)

    df = daft.from_pandas(input_df)
    output_tos_dir = f"{tos_test_data_dir}/video/video_resize_resolution"
    constructor_kwargs = {
        "output_tos_dir": output_tos_dir,
        "min_width": 1280,
        "max_width": 2560,
        "min_height": 720,
        "max_height": 1440,
        "force_original_aspect_ratio_type": "decrease",
        "crf": 23.0,
        "preset": "medium",
    }

    df = df.with_column(
        "resized_videos",
        las_udf(
            VideoResizeResolution,
            construct_args=constructor_kwargs,
            num_gpus=0,
            batch_size=1,
            concurrency=1,
        )(col("videos")),
    )
    actual_df = df.select("videos", "resized_videos").to_pandas()

    assert_dataframe_result(actual_df, expected_df)


def test_video_resize_resolution_gpu(tos_test_data_dir, local_test_data_dir):
    input_df, expected_df = generate_test_data(tos_test_data_dir, local_test_data_dir)

    df = daft.from_pandas(input_df)
    output_tos_dir = f"{tos_test_data_dir}/video/video_resize_resolution"
    constructor_kwargs = {
        "output_tos_dir": output_tos_dir,
        "min_width": 1280,
        "max_width": 2560,
        "min_height": 720,
        "max_height": 1440,
        "force_original_aspect_ratio_type": "decrease",
        "cq": 0,
        "rc": "vbr",
        "rank": None,
    }

    df = df.with_column(
        "resized_videos",
        las_udf(
            VideoResizeResolution,
            construct_args=constructor_kwargs,
            num_gpus=1,
            batch_size=1,
            concurrency=1,
        )(col("videos")),
    )
    actual_df = df.select("videos", "resized_videos").to_pandas()

    assert_dataframe_result(actual_df, expected_df)


def test_video_resize_resolution_no_output_dir(tos_test_data_dir, local_test_data_dir):
    samples = {
        "videos": [
            f"{tos_test_data_dir}/video/video_resize_resolution/sample.mp4",
        ],
    }
    input_df = pd.DataFrame(samples)

    df = daft.from_pandas(input_df)
    constructor_kwargs = {
        "output_tos_dir": "",
        "min_width": 1280,
        "max_width": 2560,
        "min_height": 720,
        "max_height": 1440,
        "force_original_aspect_ratio_type": "decrease",
        "crf": 23.0,
        "preset": "medium",
    }

    df = df.with_column(
        "resized_videos",
        las_udf(
            VideoResizeResolution,
            construct_args=constructor_kwargs,
            num_gpus=0,
            batch_size=1,
            concurrency=1,
        )(col("videos")),
    )
    actual_df = df.select("videos", "resized_videos").to_pandas()

    assert actual_df["resized_videos"][0].endswith("sample_resized.mp4")


def generate_test_data_binary(tos_test_data_dir):
    from daft.las.functions.utils.common_utils import load_file

    sample_video_path = f"{tos_test_data_dir}/video/video_resize_resolution/sample.mp4"
    video_binary = load_file(sample_video_path)
    output_basename = "binary_test_video"

    samples = {
        "videos": [None],
        "video_binaries": [video_binary],
        "video_formats": ["mp4"],
        "output_basenames": [output_basename],
    }

    input_df = pd.DataFrame(samples)

    expected_output_paths = [
        f"{tos_test_data_dir}/video/video_resize_resolution/{output_basename}_resized.mp4",
    ]
    expected_df = pd.DataFrame(
        {
            "videos": samples["videos"],
            "video_binaries": samples["video_binaries"],
            "video_formats": samples["video_formats"],
            "output_basenames": samples["output_basenames"],
            "resized_videos": expected_output_paths,
        }
    )
    return input_df, expected_df


def test_video_resize_resolution_binary_and_basename(tos_test_data_dir):
    input_df, expected_df = generate_test_data_binary(tos_test_data_dir)

    df = daft.from_pandas(input_df)
    output_tos_dir = f"{tos_test_data_dir}/video/video_resize_resolution"
    constructor_kwargs = {
        "output_tos_dir": output_tos_dir,
        "min_width": 1280,
        "max_width": 2560,
        "min_height": 720,
        "max_height": 1440,
        "force_original_aspect_ratio_type": "decrease",
        "crf": 23.0,
        "preset": "medium",
    }

    df = df.with_column(
        "resized_videos",
        las_udf(
            VideoResizeResolution,
            construct_args=constructor_kwargs,
            num_gpus=0,
            batch_size=1,
            concurrency=1,
        )(
            col("videos"),
            col("video_binaries"),
            col("video_formats"),
            col("output_basenames"),
        ),
    )
    actual_df = df.select("videos", "video_binaries", "video_formats", "output_basenames", "resized_videos").to_pandas()

    assert_dataframe_result(actual_df, expected_df)
