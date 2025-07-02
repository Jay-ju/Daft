# Copyright (c) Beijing Volcano Engine Technology Ltd.

from __future__ import annotations

import pandas as pd

import daft
from daft import col
from daft.las.functions.audio import AudioSplitByTimestamps
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
        "timestamps": [
            [],
            [(0, 5)],
            [(0, 5), (5, 10)],
            [(0, 5), (5, 10)],
        ],
    }
    input_df = pd.DataFrame(samples)

    expected_segments = [
        [],
        [],
        [
            f"{tos_test_data_dir}/audio/audio_split_by_timestamps/耙耙柑大叔/segment_0-5.aac",
            f"{tos_test_data_dir}/audio/audio_split_by_timestamps/耙耙柑大叔/segment_5-10.aac",
        ],
        [
            f"{tos_test_data_dir}/audio/audio_split_by_timestamps/sample/segment_0-5.mp3",
            f"{tos_test_data_dir}/audio/audio_split_by_timestamps/sample/segment_5-10.mp3",
        ],
    ]
    expected_df = pd.DataFrame(
        {
            "audios": samples["audios"],
            "timestamps": samples["timestamps"],
            "results.segments": expected_segments,
        }
    )

    return input_df, expected_df


def test_audio_split_by_timestamps(tos_test_data_dir, local_test_data_dir):
    input_df, expected_df = generate_test_data(tos_test_data_dir, local_test_data_dir)

    df = daft.from_pandas(input_df)

    output_tos_dir = f"{tos_test_data_dir}/audio/audio_split_by_timestamps"
    constructor_kwargs = {
        "output_tos_dir": output_tos_dir,
    }

    df = df.with_column(
        "results",
        las_udf(AudioSplitByTimestamps, construct_args=constructor_kwargs)(col("timestamps"), col("audios")),
    )
    df = df.with_column("results.segments", col("results").struct.get("segments"))
    actual_df = df.select("audios", "timestamps", "results.segments").to_pandas()

    assert_dataframe_result(actual_df, expected_df)
