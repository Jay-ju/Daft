# Copyright (c) Beijing Volcano Engine Technology Ltd.

from __future__ import annotations

import pandas as pd

import daft
from daft import col
from daft.las.functions.audio.audio_duration import AudioDuration
from daft.las.functions.udf import las_udf
from daft.las.functions.utils.audio_utils import get_duration
from tests.las.functions import assert_dataframe_result


def generate_test_data(tos_test_data_dir, local_test_data_dir):
    paths = [
        "",
        f"{local_test_data_dir}/audio/non-exist.mp3",
        f"{tos_test_data_dir}/audio/耙耙柑大叔.aac",
        f"{tos_test_data_dir}/audio/sample.mp3",
    ]
    # Sample binary audio data with known durations
    binary_data = [
        b"",  # empty
        b"RIFF\x00\x00\x00\x00WAVEfmt \x00\x00\x00\x00",  # small WAV header
        b"\xff\xfb\x90\x00\x00\x00\x00" * 1000,  # sample MP3 data
    ]
    # 读取真实 wav 文件内容
    wav_path = f"{local_test_data_dir}/audio/sample.mp3"
    try:
        with open(wav_path, "rb") as f:
            real_wav_binary = f.read()
        real_wav_duration = get_duration(wav_path)
    except Exception:
        real_wav_binary = b""
        real_wav_duration = pd.NA

    # binary_data + [None] -> binary_data + [real_wav_binary]
    binary_data_full = binary_data + [real_wav_binary]

    input_df = pd.DataFrame(
        {
            "audio_path": paths,
            "audio_binary": binary_data_full,
        }
    )
    expected_df = pd.DataFrame(
        {
            "audio_path": paths,
            "audio_binary": binary_data_full,
            "path_duration": [pd.NA, pd.NA, 17.552572, 49.711],
            "binary_duration": [pd.NA, pd.NA, pd.NA, real_wav_duration],
        }
    )
    return input_df, expected_df


def test_audio_duration_path(tos_test_data_dir, local_test_data_dir):
    input_df, expected_df = generate_test_data(tos_test_data_dir, local_test_data_dir)

    ds = daft.from_pandas(input_df)
    ds = ds.with_column("path_duration", las_udf(AudioDuration)(col("audio_path")))
    actual = ds.to_pandas()
    assert_dataframe_result(actual, expected_df[["audio_path", "audio_binary", "path_duration"]])


def test_audio_duration_binary(tos_test_data_dir, local_test_data_dir):
    input_df, expected_df = generate_test_data(tos_test_data_dir, local_test_data_dir)

    ds = daft.from_pandas(input_df)
    ds = ds.with_column("binary_duration", las_udf(AudioDuration)(col("audio_binary")))
    actual = ds.to_pandas()
    assert_dataframe_result(actual, expected_df[["audio_path", "audio_binary", "binary_duration"]])
