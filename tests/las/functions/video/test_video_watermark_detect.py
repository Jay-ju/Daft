# Copyright (c) Beijing Volcano Engine Technology Ltd.

from __future__ import annotations

import pandas as pd
import pytest

import daft
from daft import col
from daft.las.functions.udf import las_udf
from daft.las.functions.video import VideoWatermarkDetect
from tests.las.functions import assert_dataframe_result

pytestmark = pytest.mark.skip(reason="skip this file that tensorflow is not compatitable with paddlepaddle-gpu")


def generate_test_data(tos_test_data_dir, local_test_data_dir):
    samples = {
        "videos": [
            "",
            f"{local_test_data_dir}/video/video_watermark_detect/non-exist.mp4",
            f"{tos_test_data_dir}/video/video_watermark_detect/sample.mp4",
        ],
    }
    input_df = pd.DataFrame(samples)
    return input_df


def test_video_watermark_detect_basic(tos_test_data_dir, local_test_data_dir, local_models_dir):
    input_df = generate_test_data(tos_test_data_dir, local_test_data_dir)

    df = daft.from_pandas(input_df)
    constructor_kwargs = {
        "model_path": local_models_dir,
        "model_name": "PP-OCRv4/ch_det",
        "sample_count": 15,
        "consistency_threshold": 0.8,
        "position_tolerance": 15,
    }

    df = df.with_column(
        "results",
        las_udf(
            VideoWatermarkDetect,
            construct_args=constructor_kwargs,
            num_gpus=0,
            batch_size=1,
            concurrency=1,
        )(col("videos")),
    )
    df = df.with_column("watermark_regions", col("results").struct.get("watermark_regions"))
    actual_df = df.select("videos", "watermark_regions").to_pandas()

    assert len(actual_df) == 3

    regions_list = actual_df["watermark_regions"].tolist()

    assert pd.isna(regions_list[0])
    assert pd.isna(regions_list[1])

    watermark_regions = regions_list[2]
    assert watermark_regions is not None
    assert len(watermark_regions) > 0

    region = watermark_regions[0]
    assert "ymin" in region
    assert "ymax" in region
    assert "xmin" in region
    assert "xmax" in region
    assert "confidence" in region


def generate_test_data_binary(tos_test_data_dir):
    from daft.las.functions.utils.common_utils import load_file

    sample_video_path = f"{tos_test_data_dir}/video/video_watermark_detect/sample.mp4"
    video_binary = load_file(sample_video_path)

    samples = {
        "videos": [None],
        "video_binaries": [video_binary],
        "video_formats": ["mp4"],
    }

    input_df = pd.DataFrame(samples)
    return input_df


def test_video_watermark_detect_binary_and_basename(tos_test_data_dir, local_models_dir):
    input_df = generate_test_data_binary(tos_test_data_dir)

    df = daft.from_pandas(input_df)
    constructor_kwargs = {
        "model_path": local_models_dir,
        "model_name": "PP-OCRv4/ch_det",
        "sample_count": 15,
        "consistency_threshold": 0.8,
        "position_tolerance": 15,
    }

    df = df.with_column(
        "results",
        las_udf(
            VideoWatermarkDetect,
            construct_args=constructor_kwargs,
            num_gpus=0,
            batch_size=1,
            concurrency=1,
        )(
            col("videos"),
            col("video_binaries"),
            col("video_formats"),
        ),
    )
    df = df.with_column("watermark_regions", col("results").struct.get("watermark_regions"))
    actual_df = df.select("videos", "video_binaries", "video_formats", "watermark_regions").to_pandas()

    assert_dataframe_result(
        actual_df,
        expect_columns=["videos", "video_binaries", "video_formats", "watermark_regions"],
        expect_row_num=1,
    )

    regions_list = actual_df["watermark_regions"].tolist()
    watermark_regions = regions_list[0]
    assert watermark_regions is not None
    assert len(watermark_regions) > 0

    region = watermark_regions[0]
    assert "ymin" in region
    assert "ymax" in region
    assert "xmin" in region
    assert "xmax" in region
    assert "confidence" in region
