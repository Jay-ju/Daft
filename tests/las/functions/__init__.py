# Copyright (c) Beijing Volcano Engine Technology Ltd.

import pandas as pd


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
