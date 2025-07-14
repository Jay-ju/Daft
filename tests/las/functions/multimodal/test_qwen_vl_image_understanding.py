# Copyright (c) Beijing Volcano Engine Technology Ltd.

from __future__ import annotations

import os

import pandas as pd

import daft
from daft import col
from daft.las.functions.multimodal.qwen_vl_image_understanding import QwenVLImageUnderstanding
from daft.las.functions.udf import las_udf

image_src_type = "image_url"
model_name = "Qwen/Qwen2.5-VL-7B-Instruct"
dtype = "float16"
use_flash_attention_2 = False
prompt = "请给出这张图片的详细描述。"
max_caption_length = 256
resized_height = None
resized_width = None
batch_size = 2
rank = None
num_gpus = int(os.getenv("NUM_GPUS", 1))


def generate_test_data(tos_test_data_dir, local_test_data_dir, http_test_data_dir):
    paths = [
        "",
        f"{local_test_data_dir}/image/non-exist.png",
        f"{local_test_data_dir}/image/cat.png",
        f"{tos_test_data_dir}/image/cat.png",
        f"{tos_test_data_dir}/image/bird.jpeg",
        f"{http_test_data_dir}/image/cat.png",
    ]
    return pd.DataFrame({"image_path": paths})


def test_qwen_vl_image_understanding(local_models_dir, tos_test_data_dir, local_test_data_dir, http_test_data_dir):
    input_df = generate_test_data(tos_test_data_dir, local_test_data_dir, http_test_data_dir)

    ds = daft.from_pandas(input_df)
    ds = ds.with_column(
        "caption",
        las_udf(
            QwenVLImageUnderstanding,
            construct_args={
                "image_src_type": image_src_type,
                "model_path": local_models_dir,
                "model_name": model_name,
                "dtype": dtype,
                "use_flash_attention_2": use_flash_attention_2,
                "prompt": prompt,
                "max_caption_length": max_caption_length,
                "resized_height": resized_height,
                "resized_width": resized_width,
                "batch_size": batch_size,
                "rank": rank,
            },
            num_gpus=num_gpus,
            batch_size=1,
            concurrency=1,
        )(col("image_path")),
    )

    actual_df = ds.to_pandas()
    assert actual_df["caption"][0] is None or len(actual_df["caption"][0]) == 0
    assert actual_df["caption"][1] is None or len(actual_df["caption"][1]) == 0
    assert "猫" in actual_df["caption"][2]
