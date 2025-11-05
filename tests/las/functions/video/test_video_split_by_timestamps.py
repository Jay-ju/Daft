# Copyright (c) Beijing Volcano Engine Technology Ltd.

from __future__ import annotations

import daft
from daft import col
from daft.las.functions.udf import las_udf
from daft.las.functions.video import VideoSplitByTimestamps
from tests.las.functions import assert_dataframe_result


def generate_test_data(tos_test_data_dir, local_test_data_dir):
    # 多格式视频样例
    video_files = [
        "sample.mp4",
        "file_example_AVI_640_800kB.avi",
        "file_example_MOV_480_700kB.mov",
        "file_example_MP4_480_1_5MG.mp4",
        "file_example_WEBM_480_900KB.webm",
        "file_example_WMV_480_1_2MB.wmv",
    ]
    paths = [
        "",
        f"{local_test_data_dir}/video/non-exist.mp4",
    ]
    # TOS 路径全部加进来
    paths += [f"{tos_test_data_dir}/video/{fname}" for fname in video_files]
    # 时间戳范围：每个视频都切两段
    ranges = [
        [(0.0, 2.0)],  # 空路径 → 不处理
        [(0.0, 2.0)],  # 不存在的本地文件 → 不处理
    ]
    ranges += [[(0.0, 2.0), (2.0, 4.0)] for _ in video_files]
    return {"video_path": paths, "timestamp_ranges": ranges}


def generate_test_data_binary(tos_test_data_dir):
    from daft.las.functions.utils.common_utils import load_file

    sample_video_path = f"{tos_test_data_dir}/video/sample.mp4"
    video_binary = load_file(sample_video_path)
    output_basename = "my_test_video_202408"

    samples = {
        "video_path": [None],
        "video_binary": [video_binary],
        "video_format": ["mp4"],
        "timestamp_ranges": [[(0.0, 2.0), (2.0, 4.0)]],
        "output_basename": [output_basename],
    }
    return samples


def test_video_split_by_timestamps(tos_test_data_dir, local_test_data_dir):
    input_dict = generate_test_data(tos_test_data_dir, local_test_data_dir)
    df = daft.from_pydict(input_dict)

    splitter = las_udf(
        VideoSplitByTimestamps,
        construct_args={
            "output_tos_dir": f"{tos_test_data_dir}/video/video_split_by_timestamps",
            "output_segments_binary": True,
            "output_video_format": "mp4",
        },
    )

    df = df.with_column("results", splitter(col("video_path"), None, None, col("timestamp_ranges")))
    df = df.select(
        "video_path",
        "timestamp_ranges",
        col("results").struct.get("segments").alias("segments"),
        col("results").struct.get("segments_binary").alias("segments_binary"),
    )
    pd_df = df.to_pandas()

    # 所有返回的路径扩展名均为 .mp4（非空字符串）
    for segs in pd_df["segments"]:
        for seg in segs:
            if seg:  # 只检查非空字符串
                assert seg.endswith(".mp4")

    # 结构校验
    expect_row_num = len(pd_df)
    assert_dataframe_result(
        pd_df,
        expect_columns=["video_path", "timestamp_ranges", "segments", "segments_binary"],
        expect_row_num=expect_row_num,
    )

    # 空路径与不存在文件 → 返回 [""] 而不是空列表
    assert len(pd_df.iloc[0]["segments"]) == 1
    assert pd_df.iloc[0]["segments"][0] == ""
    assert len(pd_df.iloc[1]["segments"]) == 1
    assert pd_df.iloc[1]["segments"][0] == ""

    # 各种格式视频都能切出 2 段
    for i in range(2, expect_row_num):
        segs = pd_df.iloc[i]["segments"]
        bins = pd_df.iloc[i]["segments_binary"]
        assert len(segs) == 2
        assert all(isinstance(b, (bytes, bytearray)) for b in bins)
        # 检查扩展名
        assert segs[0].split(".")[-1].lower() == "mp4"


def test_video_split_by_timestamps_binary_and_basename(tos_test_data_dir):
    samples = generate_test_data_binary(tos_test_data_dir)
    df = daft.from_pydict(samples)

    splitter = las_udf(
        VideoSplitByTimestamps,
        construct_args={
            "output_tos_dir": f"{tos_test_data_dir}/video/video_split_by_timestamps",
            "output_segments_binary": True,
        },
    )

    df = df.with_column(
        "results",
        splitter(
            col("video_path"),
            col("video_binary"),
            col("video_format"),
            col("timestamp_ranges"),
            col("output_basename"),
        ),
    )
    df = df.select(
        "video_path",
        "video_binary",
        "video_format",
        "timestamp_ranges",
        "output_basename",
        col("results").struct.get("segments").alias("segments"),
        col("results").struct.get("segments_binary").alias("segments_binary"),
    )
    pd_df = df.to_pandas()

    assert_dataframe_result(
        pd_df,
        expect_columns=[
            "video_path",
            "video_binary",
            "video_format",
            "timestamp_ranges",
            "output_basename",
            "segments",
            "segments_binary",
        ],
        expect_row_num=1,
    )
    assert len(pd_df.iloc[0]["segments"]) == 2
    assert all(isinstance(b, (bytes, bytearray)) for b in pd_df.iloc[0]["segments_binary"])
