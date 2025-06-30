from __future__ import annotations

import pandas as pd

import daft
from daft import col
from daft.las.functions.audio.audio_size import AudioSize
from daft.las.functions.udf import las_udf


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
    assert_dataframe_result(ds.to_pandas(), expected_df)


def assert_dataframe_result(
    actual_df: pd.DataFrame,
    expect_df: pd.DataFrame | None = None,
    expect_columns: list | None = None,
    expect_row_num: int | None = None,
) -> None:
    if expect_df is not None:
        pd.testing.assert_frame_equal(actual_df, expect_df, check_dtype=False)

    if expect_columns:
        assert sorted(actual_df.columns.tolist()) == sorted(expect_columns)

    if expect_row_num:
        assert actual_df.shape[0] == expect_row_num
