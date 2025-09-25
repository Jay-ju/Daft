# Copyright (c) Beijing Volcano Engine Technology Ltd.

from __future__ import annotations

import pytest

import daft
from daft import col
from daft.las.functions.udf import las_udf
from daft.las.functions.video.video_frame_sampler import VideoFrameSampler
from tests.las.functions import assert_dataframe_result


def generate_test_data(tos_test_data_dir, local_test_data_dir):
    file_formats = [
        "file_example_AVI_640_800kB.avi",
        "file_example_MOV_480_700kB.mov",
        "file_example_MP4_480_1_5MG.mp4",
        "file_example_WEBM_480_900KB.webm",
        "file_example_WMV_480_1_2MB.wmv",
    ]
    file_paths = [f"{tos_test_data_dir}/video/{fname}" for fname in file_formats]
    paths = [
        "",
        f"{local_test_data_dir}/video/non-exist.mp4",
        f"{tos_test_data_dir}/video/sample.mp4",
        *file_paths,
    ]
    # Use 3.0 seconds for all paths
    return {"video_path": paths, "video_duration": [3.0] * len(paths)}


def generate_test_data_binary(tos_test_data_dir):
    from daft.las.functions.utils.common_utils import load_file

    sample_video_path = f"{tos_test_data_dir}/video/sample.mp4"
    video_binary = load_file(sample_video_path)
    samples = {
        "video_path": [None],
        "video_binary": [video_binary],
        "video_format": ["mp4"],
        "video_duration": [3.0],  # Use 3.0 seconds consistently
    }
    return samples


def test_video_frame_sampler_paths(tos_test_data_dir, local_test_data_dir):
    input_dict = generate_test_data(tos_test_data_dir, local_test_data_dir)
    df = daft.from_pydict(input_dict)

    sampler = las_udf(
        VideoFrameSampler,
        construct_args={
            "sample_mode": "by_count_uniform",
            "count_k": 3,
            "output_tos_dir": f"{tos_test_data_dir}/video/video_frame_sampler",
            "img_type": ".jpg",
        },
    )

    # Pass in video_duration
    df = df.with_column("results", sampler(col("video_path"), col("video_duration")))
    df = df.select(
        "video_path",
        col("results").struct.get("frames").alias("frames"),
        col("results").struct.get("base64").alias("base64"),
        col("results").struct.get("timestamps").alias("timestamps"),
        col("results").struct.get("frame_indices").alias("frame_indices"),
        col("results").struct.get("tos_paths").alias("tos_paths"),
    )

    pd_df = df.to_pandas()

    # Validate columns and row count
    assert_dataframe_result(
        pd_df,
        expect_columns=["video_path", "frames", "base64", "timestamps", "frame_indices", "tos_paths"],
        expect_row_num=8,
    )

    # First two rows (empty/non-existent paths) should return empty results
    assert len(pd_df.iloc[0]["frames"]) == 0
    assert len(pd_df.iloc[1]["frames"]) == 0

    # Valid sample.mp4: at least one frame, and field lengths are consistent
    frames = pd_df.iloc[2]["frames"]
    base64s = pd_df.iloc[2]["base64"]
    ts = pd_df.iloc[2]["timestamps"]
    idxs = pd_df.iloc[2]["frame_indices"]
    tos_paths = pd_df.iloc[2]["tos_paths"]

    assert len(frames) == len(base64s) == len(ts) == len(idxs)
    assert len(frames) >= 1

    # If TOS output is configured, number of paths should match frames and suffix should be correct
    if len(tos_paths) > 0:
        assert len(tos_paths) == len(frames)
        assert all(p.endswith(".jpg") for p in tos_paths)

    for i, ext in enumerate(["avi", "mov", "mp4", "webm", "wmv"], start=3):
        frames = pd_df.iloc[i]["frames"]
        base64s = pd_df.iloc[i]["base64"]
        ts = pd_df.iloc[i]["timestamps"]
        idxs = pd_df.iloc[i]["frame_indices"]
        assert len(frames) == len(base64s) == len(ts) == len(idxs)
        assert len(frames) >= 1


def test_video_frame_sampler_binary_and_duration(tos_test_data_dir):
    samples = generate_test_data_binary(tos_test_data_dir)
    df = daft.from_pydict(samples)

    sampler = las_udf(
        VideoFrameSampler,
        construct_args={
            "sample_mode": "by_count_uniform",
            "count_k": 2,
            "img_type": ".jpg",
        },
    )

    # Pass in path, binary, format, and duration columns
    df = df.with_column(
        "results",
        sampler(
            col("video_path"),
            col("video_binary"),
            col("video_format"),
            col("video_duration"),
        ),
    )
    df = df.select(
        "video_path",
        "video_binary",
        "video_format",
        "video_duration",
        col("results").struct.get("frames").alias("frames"),
        col("results").struct.get("base64").alias("base64"),
        col("results").struct.get("timestamps").alias("timestamps"),
        col("results").struct.get("frame_indices").alias("frame_indices"),
    )

    pd_df = df.to_pandas()

    assert_dataframe_result(
        pd_df,
        expect_columns=[
            "video_path",
            "video_binary",
            "video_format",
            "video_duration",
            "frames",
            "base64",
            "timestamps",
            "frame_indices",
        ],
        expect_row_num=1,
    )

    # Binary mode should produce results, and field lengths should be consistent
    frames = pd_df.iloc[0]["frames"]
    base64s = pd_df.iloc[0]["base64"]
    ts = pd_df.iloc[0]["timestamps"]
    idxs = pd_df.iloc[0]["frame_indices"]

    assert len(frames) == len(base64s) == len(ts) == len(idxs)
    assert len(frames) >= 1


@pytest.mark.skip()
def test_video_frame_sampler_interval_frames_paths(tos_test_data_dir):
    # Use frame-interval sampling to avoid unstable timestamp seeks
    df = daft.from_pydict(
        {
            "video_path": [f"{tos_test_data_dir}/video/sample.mp4"],
            "video_duration": [3.0],
        }
    )
    sampler = las_udf(
        VideoFrameSampler,
        construct_args={
            "sample_mode": "by_interval_frames",
            "interval_frames": 60,
            "img_type": ".jpg",
        },
    )

    df = df.with_column("results", sampler(col("video_path"), col("video_duration")))
    df = df.select(
        "video_path",
        col("results").struct.get("frames").alias("frames"),
        col("results").struct.get("base64").alias("base64"),
        col("results").struct.get("timestamps").alias("timestamps"),
        col("results").struct.get("frame_indices").alias("frame_indices"),
    )

    pd_df = df.to_pandas()
    assert_dataframe_result(
        pd_df,
        expect_columns=["video_path", "frames", "base64", "timestamps", "frame_indices"],
        expect_row_num=1,
    )
    frames = pd_df.iloc[0]["frames"]
    base64s = pd_df.iloc[0]["base64"]
    ts = pd_df.iloc[0]["timestamps"]
    idxs = pd_df.iloc[0]["frame_indices"]
    assert len(frames) == len(base64s) == len(ts) == len(idxs)
    assert len(frames) >= 1


@pytest.mark.skip()
def test_video_frame_sampler_interval_time_paths(tos_test_data_dir):
    # Sample by time interval
    df = daft.from_pydict(
        {
            "video_path": [f"{tos_test_data_dir}/video/sample.mp4"],
            "video_duration": [3.0],
        }
    )
    sampler = las_udf(
        VideoFrameSampler,
        construct_args={
            "sample_mode": "by_interval_time",
            "interval_sec": 0.5,
            "img_type": ".jpg",
        },
    )

    df = df.with_column("results", sampler(col("video_path"), col("video_duration")))
    df = df.select(
        "video_path",
        col("results").struct.get("frames").alias("frames"),
        col("results").struct.get("base64").alias("base64"),
        col("results").struct.get("timestamps").alias("timestamps"),
        col("results").struct.get("frame_indices").alias("frame_indices"),
    )

    pd_df = df.to_pandas()
    assert_dataframe_result(
        pd_df,
        expect_columns=["video_path", "frames", "base64", "timestamps", "frame_indices"],
        expect_row_num=1,
    )
    frames = pd_df.iloc[0]["frames"]
    base64s = pd_df.iloc[0]["base64"]
    ts = pd_df.iloc[0]["timestamps"]
    idxs = pd_df.iloc[0]["frame_indices"]
    assert len(frames) == len(base64s) == len(ts) == len(idxs)
    assert len(frames) >= 1


@pytest.mark.skip()
def test_video_frame_sampler_fps_paths(tos_test_data_dir):
    # Sample at target FPS
    df = daft.from_pydict(
        {
            "video_path": [f"{tos_test_data_dir}/video/sample.mp4"],
            "video_duration": [3.0],
        }
    )
    sampler = las_udf(
        VideoFrameSampler,
        construct_args={
            "sample_mode": "by_fps",
            "target_fps": 2.0,
            "img_type": ".jpg",
        },
    )

    df = df.with_column("results", sampler(col("video_path"), col("video_duration")))
    df = df.select(
        "video_path",
        col("results").struct.get("frames").alias("frames"),
        col("results").struct.get("base64").alias("base64"),
        col("results").struct.get("timestamps").alias("timestamps"),
        col("results").struct.get("frame_indices").alias("frame_indices"),
    )

    pd_df = df.to_pandas()
    assert_dataframe_result(
        pd_df,
        expect_columns=["video_path", "frames", "base64", "timestamps", "frame_indices"],
        expect_row_num=1,
    )
    frames = pd_df.iloc[0]["frames"]
    base64s = pd_df.iloc[0]["base64"]
    ts = pd_df.iloc[0]["timestamps"]
    idxs = pd_df.iloc[0]["frame_indices"]
    assert len(frames) == len(base64s) == len(ts) == len(idxs)
    assert len(frames) >= 1


def test_video_frame_sampler_timestamps_paths(tos_test_data_dir):
    # Sample by specified timestamps
    df = daft.from_pydict(
        {
            "video_path": [f"{tos_test_data_dir}/video/sample.mp4"],
            "video_duration": [3.0],
        }
    )

    sampler = las_udf(
        VideoFrameSampler,
        construct_args={
            "sample_mode": "by_timestamps",
            "timestamps_sec": [0.2, 0.8, 1.4],
            "img_type": ".jpg",
            "output_frames": False,
        },
    )

    df = df.with_column("results", sampler(col("video_path"), col("video_duration")))
    df = df.select(
        "video_path",
        col("results").struct.get("frames").alias("frames"),
        col("results").struct.get("base64").alias("base64"),
        col("results").struct.get("timestamps").alias("timestamps"),
        col("results").struct.get("frame_indices").alias("frame_indices"),
    )

    pd_df = df.to_pandas()
    assert_dataframe_result(
        pd_df,
        expect_columns=["video_path", "frames", "base64", "timestamps", "frame_indices"],
        expect_row_num=1,
    )
    frames = pd_df.iloc[0]["frames"]
    base64s = pd_df.iloc[0]["base64"]
    ts = pd_df.iloc[0]["timestamps"]
    idxs = pd_df.iloc[0]["frame_indices"]
    assert len(frames) == 0
    assert len(base64s) == len(ts) == len(idxs)
    assert len(base64s) >= 1
