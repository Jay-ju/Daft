# Copyright (c) Beijing Volcano Engine Technology Ltd.

from __future__ import annotations

import os

import pandas as pd
import pytest

import daft
from daft import col
from daft.las.functions.multimodal.qwen_vl_video_understanding import QwenVLVideoUnderstanding
from daft.las.functions.udf import las_udf

video_src_type = "video_url"
model_name = "Qwen/Qwen2.5-VL-3B-Instruct"
dtype = "float16"
use_flash_attention_2 = True
prompt = "请给出该视频的详细描述。"
max_caption_length = 256
min_pixels = 320 * 160
max_pixels = 320 * 160
fps = 1
batch_size = 1
rank = None
num_gpus = int(os.getenv("NUM_GPUS", 1))
num_gpus = 1


def generate_test_data(tos_test_data_dir, local_test_data_dir, http_test_data_dir):
    paths = [
        "",
        f"{tos_test_data_dir}/video/non-exist.mp4",
        f"{local_test_data_dir}/video/singer.mp4",
        f"{tos_test_data_dir}/video/singer.mp4",
        f"{http_test_data_dir}/video/singer.mp4",
    ]
    return pd.DataFrame({"video_path": paths})


@pytest.mark.gpu
def test_qwen_vl_video_understanding(local_models_dir, tos_test_data_dir, local_test_data_dir, http_test_data_dir):
    input_df = generate_test_data(tos_test_data_dir, local_test_data_dir, http_test_data_dir)

    ds = daft.from_pandas(input_df)
    ds = ds.with_column(
        "caption",
        las_udf(
            QwenVLVideoUnderstanding,
            construct_args={
                "video_src_type": video_src_type,
                "model_path": local_models_dir,
                "model_name": model_name,
                "dtype": dtype,
                "use_flash_attention_2": use_flash_attention_2,
                "prompt": prompt,
                "max_caption_length": max_caption_length,
                "min_pixels": min_pixels,
                "max_pixels": max_pixels,
                "fps": fps,
                "batch_size": batch_size,
                "rank": rank,
            },
            num_gpus=num_gpus,
            batch_size=1,
            concurrency=1,
        )(col("video_path")),
    )

    actual_df = ds.to_pandas()
    assert actual_df["caption"][0] is None or len(actual_df["caption"][0]) == 0
    assert actual_df["caption"][1] is None or len(actual_df["caption"][1]) == 0
    assert "手风琴" in actual_df["caption"][2]
