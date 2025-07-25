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
            f"{tos_test_data_dir}/audio/audio_split_by_duration/耙耙柑大叔/segment_1.aac",
            f"{tos_test_data_dir}/audio/audio_split_by_duration/耙耙柑大叔/segment_2.aac",
        ],
        [
            f"{tos_test_data_dir}/audio/audio_split_by_duration/sample/segment_1.mp3",
            f"{tos_test_data_dir}/audio/audio_split_by_duration/sample/segment_2.mp3",
            f"{tos_test_data_dir}/audio/audio_split_by_duration/sample/segment_3.mp3",
            f"{tos_test_data_dir}/audio/audio_split_by_duration/sample/segment_4.mp3",
            f"{tos_test_data_dir}/audio/audio_split_by_duration/sample/segment_5.mp3",
        ],
    ]
    expected_df = pd.DataFrame(
        {
            "audios": samples["audios"],
            "results.segments": expected_segments,
        }
    )

    return input_df, expected_df


def test_audio_split_by_duration(tos_test_data_dir, local_test_data_dir):
    input_df, expected_df = generate_test_data(tos_test_data_dir, local_test_data_dir)

    df = daft.from_pandas(input_df)

    output_tos_dir = f"{tos_test_data_dir}/audio/audio_split_by_duration"
    constructor_kwargs = {
        "output_tos_dir": output_tos_dir,
        "segment_duration": 10.0,
        "min_segment_duration": 1.0,
    }

    df = df.with_column(
        "results",
        las_udf(AudioSplitByDuration, construct_args=constructor_kwargs)(col("audios")),
    )
    df = df.with_column("results.segments", col("results").struct.get("segments"))
    actual_df = df.select("audios", "results.segments").to_pandas()

    assert_dataframe_result(actual_df, expected_df)
