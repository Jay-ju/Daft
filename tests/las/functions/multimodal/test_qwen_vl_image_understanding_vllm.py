# Copyright (c) Beijing Volcano Engine Technology Ltd.

from __future__ import annotations

import os

import pandas as pd
import pytest

import daft
from daft import col
from daft.las.functions.multimodal.qwen_vl_image_understanding_vllm import QwenVLImageUnderstandingVLLM
from daft.las.functions.udf import las_udf

image_src_type = "image_url"
model_name = "Qwen/Qwen2.5-VL-3B-Instruct"

dtype = "bfloat16"
prompt = "请给出该图片的详细描述。"
max_caption_length = 256
resized_height = 320
resized_width = 320
batch_size = 1
seed = 42
max_model_len = 128000
max_num_seqs = 128
tensor_parallel_size = int(os.getenv("NUM_GPUS", 8))
enable_prefix_caching = True
gpu_memory_utilization = 0.95
enforce_eager = True


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


@pytest.mark.skip(reason="""T4 GPU not support Flash Attention 2.""")
def test_qwen_vl_video_understanding(local_models_dir, tos_test_data_dir, local_test_data_dir, http_test_data_dir):
    input_df = generate_test_data(tos_test_data_dir, local_test_data_dir, http_test_data_dir)
    ds = daft.from_pandas(input_df)
    ds = ds.with_column(
        "caption",
        las_udf(
            QwenVLImageUnderstandingVLLM,
            construct_args={
                "image_src_type": image_src_type,
                "model_path": local_models_dir,
                "model_name": model_name,
                "dtype": dtype,
                "prompt": prompt,
                "max_caption_length": max_caption_length,
                "resized_height": resized_height,
                "resized_width": resized_width,
                "batch_size": batch_size,
                "seed": seed,
                "max_model_len": max_model_len,
                "max_num_seqs": max_num_seqs,
                "tensor_parallel_size": tensor_parallel_size,
                "enable_prefix_caching": enable_prefix_caching,
                "gpu_memory_utilization": gpu_memory_utilization,
                "enforce_eager": enforce_eager,
            },
            num_gpus=tensor_parallel_size,
            batch_size=1,
            concurrency=1,
        )(col("image_path")),
    )

    actual_df = ds.to_pandas()
    assert actual_df["caption"][0] is None or len(actual_df["caption"][0]) == 0
    assert actual_df["caption"][1] is None or len(actual_df["caption"][1]) == 0
    assert "猫" in actual_df["caption"][2]
