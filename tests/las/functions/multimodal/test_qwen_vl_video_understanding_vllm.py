# Copyright (c) Beijing Volcano Engine Technology Ltd.

from __future__ import annotations

import os

import pandas as pd
import pytest
import torch

import daft
from daft import col
from daft.las.functions.multimodal.qwen_vl_video_understanding_vllm import QwenVLVideoUnderstandingVLLM
from daft.las.functions.udf import las_udf

daft.set_execution_config(actor_udf_ready_timeout=600)

video_src_type = "video_url"
model_name = "Qwen/Qwen2.5-VL-3B-Instruct"
dtype = "bfloat16"
prompt = "请给出该视频的详细描述。"
max_caption_length = 256
min_pixels = 320 * 160
max_pixels = 320 * 160
fps = 2
batch_size = 1
seed = 42
max_model_len = 128000
max_num_seqs = 128
tensor_parallel_size = 1
enable_prefix_caching = True
gpu_memory_utilization = 0.9
enforce_eager = True

num_gpus = torch.cuda.device_count()
if num_gpus == 1:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
else:
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"
enable_prefix_caching = False
gpu_memory_utilization = 0.7


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
@pytest.mark.skip(reason="GPU is required for this test.")
def test_qwen_vl_video_understanding(local_models_dir, tos_test_data_dir, local_test_data_dir, http_test_data_dir):
    input_df = generate_test_data(tos_test_data_dir, local_test_data_dir, http_test_data_dir)

    ds = daft.from_pandas(input_df)
    ds = ds.with_column(
        "caption",
        las_udf(
            QwenVLVideoUnderstandingVLLM,
            construct_args={
                "video_src_type": video_src_type,
                "model_path": local_models_dir,
                "model_name": model_name,
                "dtype": dtype,
                "prompt": prompt,
                "max_caption_length": max_caption_length,
                "min_pixels": min_pixels,
                "max_pixels": max_pixels,
                "fps": fps,
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
            batch_size=5,
            concurrency=1,
        )(col("video_path")),
    )

    actual_df = ds.to_pandas()
    assert actual_df["caption"][0] is None or len(actual_df["caption"][0]) == 0
    assert actual_df["caption"][1] is None or len(actual_df["caption"][1]) == 0
    assert "手风琴" in actual_df["caption"][2]
