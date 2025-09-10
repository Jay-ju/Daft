# Copyright (c) Beijing Volcano Engine Technology Ltd.

from __future__ import annotations

import pytest

import daft
from daft import col
from daft.las.functions.udf import las_udf
from daft.las.functions.video import VideoSplitByDuration
from tests.las.functions import assert_dataframe_result


def generate_test_data(tos_test_data_dir, local_test_data_dir):
    paths = [
        "",
        f"{local_test_data_dir}/video/non-exist.mp4",
        f"{tos_test_data_dir}/video/sample.mp4",
    ]
    return {"video_path": paths}


def generate_test_data_binary(tos_test_data_dir):
    from daft.las.functions.utils.common_utils import load_file

    sample_video_path = f"{tos_test_data_dir}/video/sample.mp4"
    video_binary = load_file(sample_video_path)
    output_basename = "my_test"

    samples = {
        "video_path": [None],
        "video_binary": [video_binary],
        "video_format": ["mp4"],
        "output_basename": [output_basename],
    }
    return samples


def test_video_split_by_duration(tos_test_data_dir, local_test_data_dir):
    input_dict = generate_test_data(tos_test_data_dir, local_test_data_dir)
    df = daft.from_pydict(input_dict)

    splitter = las_udf(
        VideoSplitByDuration,
        construct_args={
            "segment_duration": 60.0,
            "min_segment_duration": 1.0,
            "output_tos_dir": f"{tos_test_data_dir}/video/video_split_by_duration",
            "output_segments_binary": True,
            "output_video_format": "avi",
        },
    )

    df = df.with_column("results", splitter(col("video_path")))
    df = df.select(
        "video_path",
        col("results").struct.get("segments").alias("segments"),
        col("results").struct.get("segments_binary").alias("segments_binary"),
    )
    pd_df = df.to_pandas()

    for segs in pd_df["segments"]:
        for seg in segs:
            assert seg.endswith(".avi")

    # Verify dataframe structure
    assert_dataframe_result(
        pd_df,
        expect_columns=["video_path", "segments", "segments_binary"],
        expect_row_num=3,
    )

    # Verify empty and non-existent videos result in empty lists
    assert len(pd_df.iloc[0]["segments"]) == 0
    assert len(pd_df.iloc[1]["segments"]) == 0

    assert len(pd_df.iloc[2]["segments"]) == 4
    assert pd_df.iloc[2]["segments_binary"][0][0] == pytest.approx(82, abs=10)


def test_video_split_by_duration_binary_and_basename(tos_test_data_dir):
    samples = generate_test_data_binary(tos_test_data_dir)
    df = daft.from_pydict(samples)

    splitter = las_udf(
        VideoSplitByDuration,
        construct_args={
            "segment_duration": 60.0,
            "min_segment_duration": 1.0,
            "output_tos_dir": f"{tos_test_data_dir}/video/video_split_by_duration",
            "output_segments_binary": True,
        },
    )

    df = df.with_column(
        "results",
        splitter(
            col("video_path"),
            col("video_binary"),
            col("video_format"),
            col("output_basename"),
        ),
    )
    df = df.select(
        "video_path",
        "video_binary",
        "video_format",
        "output_basename",
        col("results").struct.get("segments").alias("segments"),
        col("results").struct.get("segments_binary").alias("segments_binary"),
    )
    pd_df = df.to_pandas()

    assert_dataframe_result(
        pd_df,
        expect_columns=["video_path", "video_binary", "video_format", "output_basename", "segments", "segments_binary"],
        expect_row_num=1,
    )
    assert len(pd_df.iloc[0]["segments"]) == 4
    assert all(isinstance(b, (bytes, bytearray)) for b in pd_df.iloc[0]["segments_binary"])
