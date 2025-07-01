# Copyright (c) Beijing Volcano Engine Technology Ltd.

from __future__ import annotations

import pandas as pd

import daft
from daft import col
from daft.las.functions.audio.audio_size import AudioSize
from daft.las.functions.udf import las_udf
from tests.las.functions import assert_dataframe_result


def generate_test_data(tos_test_data_dir, local_test_data_dir):
    paths = [
        "",
        f"{local_test_data_dir}/audio/non-exist.mp3",
        f"{tos_test_data_dir}/audio/耙耙柑大叔.aac",
        f"{tos_test_data_dir}/audio/sample.mp3",
    ]
    input_df = pd.DataFrame({"audio_path": paths})
    expected_df = pd.DataFrame(
        {
            "audio_path": paths,
            "size_result": [pd.NA, pd.NA, 569785.0, 795426.0],
        }
    )
    return input_df, expected_df


def test_audio_size(tos_test_data_dir, local_test_data_dir):
    input_df, expected_df = generate_test_data(tos_test_data_dir, local_test_data_dir)

    ds = daft.from_pandas(input_df)
    ds = ds.with_column("size_result", las_udf(AudioSize)(col("audio_path")))
    actual = ds.to_pandas()
    assert_dataframe_result(actual, expected_df)
