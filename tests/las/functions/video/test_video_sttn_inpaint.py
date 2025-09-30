# Copyright (c) Beijing Volcano Engine Technology Ltd.

from __future__ import annotations

import pandas as pd

import daft
from daft import col
from daft.las.functions.udf import las_udf
from daft.las.functions.video import VideoSttnInpaint
from tests.las.functions import assert_dataframe_result


def generate_test_data(tos_test_data_dir, local_test_data_dir):
    samples = {
        "videos": [
            "",
            f"{local_test_data_dir}/video/video_sttn_inpaint/non-exist.mp4",
            f"{tos_test_data_dir}/video/video_sttn_inpaint/sample.mp4",
        ],
    }
    input_df = pd.DataFrame(samples)

    expected_output_paths = [
        None,
        None,
        f"{tos_test_data_dir}/video/video_sttn_inpaint/sample_inpainted.mp4",
    ]

    expected_df = pd.DataFrame(
        {
            "videos": samples["videos"],
            "output_path": expected_output_paths,
        }
    )

    return input_df, expected_df


def test_video_sttn_inpaint_basic(tos_test_data_dir, local_test_data_dir, local_models_dir):
    input_df, expected_df = generate_test_data(tos_test_data_dir, local_test_data_dir)

    df = daft.from_pandas(input_df)
    output_tos_dir = f"{tos_test_data_dir}/video/video_sttn_inpaint"
    constructor_kwargs = {
        "output_tos_dir": output_tos_dir,
        "model_path": local_models_dir,
        "model_name": "researchmm/STTN",
        "neighbor_stride": 5,
        "reference_length": 10,
        "max_load_num": 50,
    }

    df = df.with_column(
        "results",
        las_udf(
            VideoSttnInpaint,
            construct_args=constructor_kwargs,
            num_gpus=1,
            batch_size=1,
            concurrency=1,
        )(col("videos")),
    )
    df = df.with_column("output_path", col("results").struct.get("output_path"))
    actual_df = df.select("videos", "output_path").to_pandas()

    assert_dataframe_result(actual_df, expected_df)


def generate_test_data_binary(tos_test_data_dir):
    from daft.las.functions.utils.common_utils import load_file

    sample_video_path = f"{tos_test_data_dir}/video/video_sttn_inpaint/sample.mp4"
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
        f"{tos_test_data_dir}/video/video_sttn_inpaint/{output_basename}_inpainted.mp4",
    ]
    expected_df = pd.DataFrame(
        {
            "videos": samples["videos"],
            "video_binaries": samples["video_binaries"],
            "video_formats": samples["video_formats"],
            "output_basenames": samples["output_basenames"],
            "output_path": expected_output_paths,
        }
    )
    return input_df, expected_df


def test_video_sttn_inpaint_binary_and_basename(tos_test_data_dir, local_models_dir):
    input_df, expected_df = generate_test_data_binary(tos_test_data_dir)

    df = daft.from_pandas(input_df)
    output_tos_dir = f"{tos_test_data_dir}/video/video_sttn_inpaint"
    constructor_kwargs = {
        "output_tos_dir": output_tos_dir,
        "model_path": local_models_dir,
        "model_name": "researchmm/STTN",
        "neighbor_stride": 5,
        "reference_length": 10,
        "max_load_num": 50,
    }

    df = df.with_column(
        "results",
        las_udf(
            VideoSttnInpaint,
            construct_args=constructor_kwargs,
            num_gpus=1,
            batch_size=1,
            concurrency=1,
        )(
            col("videos"),
            col("video_binaries"),
            col("video_formats"),
            None,
            col("output_basenames"),
        ),
    )
    df = df.with_column("output_path", col("results").struct.get("output_path"))
    actual_df = df.select("videos", "video_binaries", "video_formats", "output_basenames", "output_path").to_pandas()

    assert_dataframe_result(actual_df, expected_df)
