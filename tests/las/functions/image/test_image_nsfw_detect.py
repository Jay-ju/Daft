# Copyright (c) Beijing Volcano Engine Technology Ltd.

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import daft
from daft import col
from daft.las.functions.image.image_nsfw_detect import ImageNsfwDetect
from daft.las.functions.udf import las_udf
from daft.las.functions.utils.image_utils import decode_image, image_to_base64, image_to_binary

image_src_type_url = "image_url"
num_gpus = 1
batch_size = 1


def generate_test_data_url(tos_test_data_dir, local_test_data_dir, http_test_data_dir):
    paths = [
        "",  # 空路径，应返回 None
        f"{local_test_data_dir}/image/non-exist.png",  # 不存在的本地图片，应返回 None
        f"{local_test_data_dir}/image/cat.png",  # 本地有效图片
        f"{tos_test_data_dir}/image/cat.png",  # TOS 有效图片
        f"{http_test_data_dir}/image/cat.png",  # HTTP 有效图片
    ]
    return pd.DataFrame({"image_path": paths})


def generate_test_data_base64(tos_test_data_dir, local_test_data_dir, http_test_data_dir):
    paths = [
        f"{local_test_data_dir}/image/cat.png",
        f"{tos_test_data_dir}/image/cat.png",
        f"{http_test_data_dir}/image/cat.png",
    ]
    images = [decode_image(image, "image_url") for image in paths]
    image_base64 = [image_to_base64(image) for image in images]
    return pd.DataFrame({"image_base64": image_base64})


def generate_test_data_binary(
    tos_test_data_dir: str, local_test_data_dir: str, http_test_data_dir: str
) -> pd.DataFrame:
    paths = [
        f"{local_test_data_dir}/image/cat.png",
        f"{tos_test_data_dir}/image/cat.png",
        f"{http_test_data_dir}/image/cat.png",
    ]
    images = [decode_image(image, "image_url") for image in paths]
    image_binary = [image_to_binary(image) for image in images]
    return pd.DataFrame({"image_binary": image_binary})


@pytest.mark.gpu
def test_nsfw_detect_url(tos_test_data_dir, local_test_data_dir, http_test_data_dir, local_models_dir):
    input_df = generate_test_data_url(tos_test_data_dir, local_test_data_dir, http_test_data_dir)

    ds = daft.from_pandas(input_df)
    ds = ds.with_column(
        "nsfw",
        las_udf(
            ImageNsfwDetect,
            construct_args={
                "model_path": local_models_dir,
                "image_src_type": image_src_type_url,
                "batch_size": batch_size,
            },
            num_gpus=num_gpus,
            batch_size=1,
            concurrency=1,
        )(col("image_path")),
    )

    actual_df = ds.to_pandas()
    assert pd.isna(actual_df["nsfw"][0])
    assert pd.isna(actual_df["nsfw"][1])
    assert np.allclose(actual_df["nsfw"][2], 0.000111, atol=0.1)


@pytest.mark.gpu
def test_nsfw_detect_base64(tos_test_data_dir, local_test_data_dir, http_test_data_dir, local_models_dir):
    input_df = generate_test_data_base64(tos_test_data_dir, local_test_data_dir, http_test_data_dir)
    ds = daft.from_pandas(input_df)
    ds = ds.with_column(
        "nsfw",
        las_udf(
            ImageNsfwDetect,
            construct_args={"model_path": local_models_dir, "image_src_type": "image_base64", "batch_size": batch_size},
            num_gpus=num_gpus,
            batch_size=1,
            concurrency=1,
        )(col("image_base64")),
    )

    actual_df = ds.to_pandas()
    assert np.allclose(actual_df["nsfw"][2], 0.000111, atol=0.1)


@pytest.mark.gpu
def test_nsfw_detect_binary(tos_test_data_dir, local_test_data_dir, http_test_data_dir, local_models_dir):
    input_df = generate_test_data_binary(tos_test_data_dir, local_test_data_dir, http_test_data_dir)
    ds = daft.from_pandas(input_df)
    ds = ds.with_column(
        "nsfw",
        las_udf(
            ImageNsfwDetect,
            construct_args={"model_path": local_models_dir, "image_src_type": "image_binary", "batch_size": batch_size},
            num_gpus=num_gpus,
            batch_size=1,
            concurrency=1,
        )(col("image_binary")),
    )
    actual_df = ds.to_pandas()
    assert np.allclose(actual_df["nsfw"][2], 0.000111, atol=0.1)
