# Copyright (c) Beijing Volcano Engine Technology Ltd.

from __future__ import annotations

import os

import pandas as pd
import pytest

import daft
from daft import col
from daft.las.functions.multimodal.llava_one_vision_image_understanding import LlavaOneVisionImageUnderstanding
from daft.las.functions.udf import las_udf
from daft.las.functions.utils.image_utils import decode_image, image_to_base64, image_to_binary

image_src_type = "image_url"
model_name = "LLaVA-OneVision-1.5-4B-Instruct"
dtype = "float16"
use_flash_attention_2 = False
prompt = "请给出这张图片的详细描述。"
max_caption_length = 256
resized_height = None
resized_width = None
batch_size = 1
rank = None
num_gpus = int(os.getenv("NUM_GPUS", 1))
num_gpus = 1


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


def generate_test_data_base64(tos_test_data_dir, local_test_data_dir, http_test_data_dir):
    paths = [
        f"{local_test_data_dir}/image/打印体.jpg",
        f"{tos_test_data_dir}/image/打印体.jpg",
        f"{http_test_data_dir}/image/打印体.jpg",
    ]
    images = [decode_image(image, "image_url") for image in paths]
    image_base64 = [image_to_base64(image) for image in images]
    return pd.DataFrame({"image_base64": image_base64})


def generate_test_data_binary(tos_test_data_dir, local_test_data_dir, http_test_data_dir):
    paths = [
        f"{local_test_data_dir}/image/打印体.jpg",
        f"{tos_test_data_dir}/image/打印体.jpg",
        f"{http_test_data_dir}/image/打印体.jpg",
    ]
    images = [decode_image(image, "image_url") for image in paths]
    image_binary = [image_to_binary(image) for image in images]
    return pd.DataFrame({"image_binary": image_binary})


@pytest.mark.gpu
def test_llava_one_vision_image_understanding(
    local_models_dir, tos_test_data_dir, local_test_data_dir, http_test_data_dir
):
    input_df = generate_test_data(tos_test_data_dir, local_test_data_dir, http_test_data_dir)

    ds = daft.from_pandas(input_df)
    ds = ds.with_column(
        "caption",
        las_udf(
            LlavaOneVisionImageUnderstanding,
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
    assert "猫" in actual_df["caption"][2] or "动物" in actual_df["caption"][2]


@pytest.mark.skip(reason="Repeated Unit Test.")
def test_llava_one_vision_image_understanding_base64(
    local_models_dir, tos_test_data_dir, local_test_data_dir, http_test_data_dir
):
    input_df = generate_test_data_base64(tos_test_data_dir, local_test_data_dir, http_test_data_dir)

    ds = daft.from_pandas(input_df)
    ds = ds.with_column(
        "caption",
        las_udf(
            LlavaOneVisionImageUnderstanding,
            construct_args={
                "image_src_type": "image_base64",
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
        )(col("image_base64")),
    )

    actual_df = ds.to_pandas()
    assert "文字" in actual_df["caption"][0]


@pytest.mark.skip(reason="Repeated Unit Test.")
def test_llava_one_vision_image_understanding_binary(
    local_models_dir, tos_test_data_dir, local_test_data_dir, http_test_data_dir
):
    input_df = generate_test_data_binary(tos_test_data_dir, local_test_data_dir, http_test_data_dir)

    ds = daft.from_pandas(input_df)
    ds = ds.with_column(
        "caption",
        las_udf(
            LlavaOneVisionImageUnderstanding,
            construct_args={
                "image_src_type": "image_binary",
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
        )(col("image_binary")),
    )

    actual_df = ds.to_pandas()
    assert "文字" in actual_df["caption"][0]
