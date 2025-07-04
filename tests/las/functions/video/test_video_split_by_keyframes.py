# Copyright (c) Beijing Volcano Engine Technology Ltd.

from __future__ import annotations

import pytest

import daft
from daft import col
from daft.las.functions.udf import las_udf
from daft.las.functions.video import VideoSplitByKeyframes
from tests.las.functions import assert_dataframe_result


def generate_test_data(tos_test_data_dir, local_test_data_dir):
    paths = [
        "",
        f"{local_test_data_dir}/video/non-exist.mp4",
        f"{tos_test_data_dir}/video/sample.mp4",
    ]
    return {"video_path": paths}


def test_video_split_by_keyframes(tos_test_data_dir, local_test_data_dir):
    input_dict = generate_test_data(tos_test_data_dir, local_test_data_dir)
    df = daft.from_pydict(input_dict)

    splitter = las_udf(
        VideoSplitByKeyframes,
        construct_args={
            "method": "I_frame",
            "keyframes_cnt": 2,
            "output_tos_dir": f"{tos_test_data_dir}/video/video_split_by_keyframes",
            "output_segments_binary": True,
        },
    )

    df = df.with_column("results", splitter(col("video_path")))
    df = df.select(
        "video_path",
        col("results").struct.get("segments").alias("segments"),
        col("results").struct.get("segments_binary").alias("segments_binary"),
    )
    pd_df = df.to_pandas()

    # Verify dataframe structure
    assert_dataframe_result(
        pd_df,
        expect_columns=["video_path", "segments", "segments_binary"],
        expect_row_num=3,
    )

    # Verify empty and non-existent videos result in empty lists
    assert len(pd_df.iloc[0]["segments"]) == 0
    assert len(pd_df.iloc[1]["segments"]) == 0

    # Verify TOS video processing
    assert len(pd_df.iloc[2]["segments"]) == 2
    assert len(pd_df.iloc[2]["segments_binary"][0]) == pytest.approx(24178388, abs=10)
