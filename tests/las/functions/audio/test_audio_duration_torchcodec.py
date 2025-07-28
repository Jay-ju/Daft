# Copyright (c) Beijing Volcano Engine Technology Ltd.

from __future__ import annotations

import pandas as pd

import daft
from daft import col
from daft.las.functions.audio import AudioDurationTorchcodec
from daft.las.functions.udf import las_udf
from tests.las.functions import assert_dataframe_result


def generate_test_data(tos_test_data_dir: str, local_test_data_dir: str):
    paths = [
        "",  # 空路径
        f"{local_test_data_dir}/audio/non-exist.wav",  # 本地不存在
        f"{tos_test_data_dir}/audio/test_music.m4a",  # 有效 TOS 文件
    ]

    expected_durations = [
        None,
        None,
        9.92,
    ]

    input_df = pd.DataFrame({"path": paths})
    expected_df = pd.DataFrame({"path": paths, "audio_duration_seconds": expected_durations})

    return input_df, expected_df


def test_audio_duration_torchcodec(local_test_data_dir: str, tos_test_data_dir: str):
    input_df, expected_df = generate_test_data(tos_test_data_dir, local_test_data_dir)

    df = daft.from_pandas(input_df)
    df = df.with_column(
        "audio_duration_seconds",
        las_udf(
            AudioDurationTorchcodec,
            batch_size=1,
            concurrency=1,
        )(col("path")),
    )
    actual_df = df.to_pandas()

    assert_dataframe_result(actual_df, expected_df)
