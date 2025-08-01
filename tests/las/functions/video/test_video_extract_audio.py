# Copyright (c) Beijing Volcano Engine Technology Ltd.

from __future__ import annotations

import pandas as pd

import daft
from daft import col
from daft.las.functions.udf import las_udf
from daft.las.functions.video import VideoExtractAudio
from tests.las.functions import assert_dataframe_result


def generate_test_data(tos_test_data_dir, local_test_data_dir):
    samples = {
        "videos": [
            "",
            f"{local_test_data_dir}/video/non-exist.mp4",
            f"{tos_test_data_dir}/video/sample.mp4",
        ],
    }
    input_df = pd.DataFrame(samples)

    expected_audio_paths = [
        [],
        [],
        [
            f"{tos_test_data_dir}/video/video_extract_audio/sample/audio_stream_0.mp3",
        ],
    ]
    expected_df = pd.DataFrame(
        {
            "videos": samples["videos"],
            "results.audio_paths": expected_audio_paths,
        }
    )

    return input_df, expected_df


def generate_test_data_binary(tos_test_data_dir):
    from daft.las.functions.utils.common_utils import load_file

    sample_video_path = f"{tos_test_data_dir}/video/sample.mp4"
    video_binary = load_file(sample_video_path)
    output_basename = "my_test_video_202408"

    samples = {
        "videos": [None],
        "video_binaries": [video_binary],
        "video_formats": ["mp4"],
        "output_basenames": [output_basename],
    }

    input_df = pd.DataFrame(samples)

    expected_audio_paths = [
        [
            f"{tos_test_data_dir}/video/video_extract_audio/{output_basename}/audio_stream_0.mp3",
        ]
    ]
    expected_df = pd.DataFrame(
        {
            "videos": samples["videos"],
            "video_binaries": samples["video_binaries"],
            "video_formats": samples["video_formats"],
            "output_basenames": samples["output_basenames"],
            "results.audio_paths": expected_audio_paths,
        }
    )
    return input_df, expected_df


def test_video_extract_audio(tos_test_data_dir, local_test_data_dir):
    input_df, expected_df = generate_test_data(tos_test_data_dir, local_test_data_dir)

    df = daft.from_pandas(input_df)

    output_tos_dir = f"{tos_test_data_dir}/video/video_extract_audio"
    constructor_kwargs = {
        "output_tos_dir": output_tos_dir,
        "output_audio_binary": True,
        "output_sampling_rate": 16000,
        "output_audio_format": "mp3",
    }

    df = df.with_column(
        "results",
        las_udf(VideoExtractAudio, construct_args=constructor_kwargs)(col("videos")),
    )
    df = df.with_column("results.audio_paths", col("results").struct.get("audio_paths"))
    actual_df = df.select("videos", "results.audio_paths").to_pandas()

    assert_dataframe_result(actual_df, expected_df)


def test_video_extract_audio_binary_and_basename(tos_test_data_dir):
    input_df, expected_df = generate_test_data_binary(tos_test_data_dir)

    df = daft.from_pandas(input_df)

    output_tos_dir = f"{tos_test_data_dir}/video/video_extract_audio"
    constructor_kwargs = {
        "output_tos_dir": output_tos_dir,
        "output_audio_binary": True,
        "output_sampling_rate": 16000,
        "output_audio_format": "mp3",
    }

    df = df.with_column(
        "results",
        las_udf(VideoExtractAudio, construct_args=constructor_kwargs)(
            col("videos"),
            col("video_binaries"),
            col("video_formats"),
            col("output_basenames"),
        ),
    )
    df = df.with_column("results.audio_paths", col("results").struct.get("audio_paths"))
    actual_df = df.select(
        "videos", "video_binaries", "video_formats", "output_basenames", "results.audio_paths"
    ).to_pandas()

    assert_dataframe_result(actual_df, expected_df)
