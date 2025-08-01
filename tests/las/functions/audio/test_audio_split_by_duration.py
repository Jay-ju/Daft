# Copyright (c) Beijing Volcano Engine Technology Ltd.

from __future__ import annotations

import pandas as pd

import daft
from daft import col
from daft.las.functions.audio import AudioSplitByDuration
from daft.las.functions.udf import las_udf
from tests.las.functions import assert_dataframe_result


def generate_test_data(tos_test_data_dir, local_test_data_dir):
    samples = {
        "audios": [
            "",
            f"{local_test_data_dir}/audio/non-exist.mp3",
            f"{tos_test_data_dir}/audio/耙耙柑大叔.aac",
            f"{tos_test_data_dir}/audio/sample.mp3",
        ],
    }
    input_df = pd.DataFrame(samples)

    expected_segments = [
        [],
        [],
        [
            f"{tos_test_data_dir}/audio/audio_split_by_duration/耙耙柑大叔/segment_1.flac",
            f"{tos_test_data_dir}/audio/audio_split_by_duration/耙耙柑大叔/segment_2.flac",
        ],
        [
            f"{tos_test_data_dir}/audio/audio_split_by_duration/sample/segment_1.flac",
            f"{tos_test_data_dir}/audio/audio_split_by_duration/sample/segment_2.flac",
            f"{tos_test_data_dir}/audio/audio_split_by_duration/sample/segment_3.flac",
            f"{tos_test_data_dir}/audio/audio_split_by_duration/sample/segment_4.flac",
            f"{tos_test_data_dir}/audio/audio_split_by_duration/sample/segment_5.flac",
        ],
    ]
    expected_df = pd.DataFrame(
        {
            "audios": samples["audios"],
            "results.segments": expected_segments,
        }
    )

    return input_df, expected_df


def generate_test_data_binary(tos_test_data_dir):
    from daft.las.functions.utils.common_utils import load_file

    sample_audio_path = f"{tos_test_data_dir}/audio/sample.mp3"
    audio_binary = load_file(sample_audio_path)

    output_basename = "my_test_audio_duration_202408"

    samples = {
        "audios": [None],
        "audio_binaries": [audio_binary],
        "audio_formats": ["mp3"],
        "output_basenames": [output_basename],
    }

    input_df = pd.DataFrame(samples)

    expected_segments = [
        [
            f"{tos_test_data_dir}/audio/audio_split_by_duration/{output_basename}/segment_1.mp3",
            f"{tos_test_data_dir}/audio/audio_split_by_duration/{output_basename}/segment_2.mp3",
            f"{tos_test_data_dir}/audio/audio_split_by_duration/{output_basename}/segment_3.mp3",
            f"{tos_test_data_dir}/audio/audio_split_by_duration/{output_basename}/segment_4.mp3",
            f"{tos_test_data_dir}/audio/audio_split_by_duration/{output_basename}/segment_5.mp3",
        ]
    ]
    expected_df = pd.DataFrame(
        {
            "audios": samples["audios"],
            "audio_binaries": samples["audio_binaries"],
            "audio_formats": samples["audio_formats"],
            "output_basenames": samples["output_basenames"],
            "results.segments": expected_segments,
        }
    )
    return input_df, expected_df


def test_audio_split_by_duration(tos_test_data_dir, local_test_data_dir):
    input_df, expected_df = generate_test_data(tos_test_data_dir, local_test_data_dir)

    df = daft.from_pandas(input_df)

    output_tos_dir = f"{tos_test_data_dir}/audio/audio_split_by_duration"
    # 新增：测试 output_format 参数
    constructor_kwargs = {
        "output_tos_dir": output_tos_dir,
        "segment_duration": 10.0,
        "min_segment_duration": 1.0,
        "output_format": "flac",
    }

    df = df.with_column(
        "results",
        las_udf(AudioSplitByDuration, construct_args=constructor_kwargs)(col("audios")),
    )
    df = df.with_column("results.segments", col("results").struct.get("segments"))
    actual_df = df.select("audios", "results.segments").to_pandas()

    # 检查所有输出路径后缀为 .flac（如果有结果）
    for segs in actual_df["results.segments"]:
        for seg in segs:
            assert seg.endswith(".flac")

    assert_dataframe_result(actual_df, expected_df)


def test_audio_split_by_duration_binary_and_basename(tos_test_data_dir):
    input_df, expected_df = generate_test_data_binary(tos_test_data_dir)

    df = daft.from_pandas(input_df)

    output_tos_dir = f"{tos_test_data_dir}/audio/audio_split_by_duration"
    constructor_kwargs = {
        "output_tos_dir": output_tos_dir,
        "segment_duration": 10.0,
        "min_segment_duration": 1.0,
    }

    df = df.with_column(
        "results",
        las_udf(AudioSplitByDuration, construct_args=constructor_kwargs)(
            col("audios"),
            col("audio_binaries"),
            col("audio_formats"),
            col("output_basenames"),
        ),
    )
    df = df.with_column("results.segments", col("results").struct.get("segments"))
    actual_df = df.select(
        "audios", "audio_binaries", "audio_formats", "output_basenames", "results.segments"
    ).to_pandas()

    assert_dataframe_result(actual_df, expected_df)
