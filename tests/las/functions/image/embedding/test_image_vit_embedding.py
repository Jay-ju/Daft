# Copyright (c) Beijing Volcano Engine Technology Ltd.

from __future__ import annotations

import math

import pandas as pd
import pytest

import daft
from daft import col
from daft.las.functions.image.embedding.image_vit_embedding import ImageViTEmbedding
from daft.las.functions.udf import las_udf
from daft.las.functions.utils.image_utils import decode_image, image_to_base64, image_to_binary

image_src_type = "image_url"
batch_size = 64
model_name = "facebook/dinov2-base"
dtype = "float16"
use_cls_token_embedding = True
rank = 0
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
        f"{local_test_data_dir}/image/cat.png",
        f"{tos_test_data_dir}/image/cat.png",
        f"{http_test_data_dir}/image/cat.png",
    ]
    images = [decode_image(image, "image_url") for image in paths]
    image_base64 = [image_to_base64(image) for image in images]
    return pd.DataFrame({"image_base64": image_base64})


def generate_test_data_binary(tos_test_data_dir, local_test_data_dir, http_test_data_dir):
    paths = [
        f"{local_test_data_dir}/image/cat.png",
        f"{tos_test_data_dir}/image/cat.png",
        f"{http_test_data_dir}/image/cat.png",
    ]
    images = [decode_image(image, "image_url") for image in paths]
    image_binary = [image_to_binary(image) for image in images]
    return pd.DataFrame({"image_binary": image_binary})


@pytest.mark.gpu
@pytest.mark.parametrize(
    "model_name",
    [
        "google/vit-base-patch16-224-in21k",
        "google/vit-large-patch16-224-in21k",
        "facebook/dinov2-base",
        "facebook/dinov2-large",
    ],
)
def test_vit_embedding(local_models_dir, tos_test_data_dir, local_test_data_dir, http_test_data_dir, model_name):
    input_df = generate_test_data(tos_test_data_dir, local_test_data_dir, http_test_data_dir)

    ds = daft.from_pandas(input_df)
    ds = ds.with_column(
        "embedding",
        las_udf(
            ImageViTEmbedding,
            construct_args={
                "image_src_type": image_src_type,
                "batch_size": batch_size,
                "model_path": local_models_dir,
                "model_name": model_name,
                "dtype": dtype,
                "use_cls_token_embedding": use_cls_token_embedding,
                "rank": rank,
            },
            num_gpus=num_gpus,
            batch_size=1,
        )(col("image_path")),
    )
    actual_df = ds.to_pandas()

    assert actual_df["embedding"][0] is None
    assert len(actual_df["embedding"][2]) == 1024 or len(actual_df["embedding"][2]) == 768
    if model_name == "facebook/dinov2-large":
        assert math.fabs(actual_df["embedding"][2][0] - 0.006334126) < 0.001


@pytest.mark.gpu
@pytest.mark.parametrize("dtype", ["bfloat16", "float16", "float32"])
def test_vit_embedding_img_base64(local_models_dir, tos_test_data_dir, local_test_data_dir, http_test_data_dir, dtype):
    input_df = generate_test_data_base64(tos_test_data_dir, local_test_data_dir, http_test_data_dir)

    ds = daft.from_pandas(input_df)
    ds = ds.with_column(
        "embedding",
        las_udf(
            ImageViTEmbedding,
            construct_args={
                "image_src_type": "image_base64",
                "batch_size": batch_size,
                "model_path": local_models_dir,
                "model_name": model_name,
                "dtype": dtype,
                "use_cls_token_embedding": use_cls_token_embedding,
                "rank": rank,
            },
            num_gpus=num_gpus,
            batch_size=1,
        )(col("image_base64")),
    )

    actual_df = ds.to_pandas()
    assert len(actual_df["embedding"][2]) == 768
    assert math.fabs(actual_df["embedding"][0][0] - 0.0079315146) < 0.001


@pytest.mark.gpu
def test_vit_embedding_img_binary(local_models_dir, tos_test_data_dir, local_test_data_dir, http_test_data_dir):
    input_df = generate_test_data_binary(tos_test_data_dir, local_test_data_dir, http_test_data_dir)

    ds = daft.from_pandas(input_df)
    ds = ds.with_column(
        "embedding",
        las_udf(
            ImageViTEmbedding,
            construct_args={
                "image_src_type": "image_binary",
                "batch_size": batch_size,
                "model_path": local_models_dir,
                "model_name": model_name,
                "dtype": dtype,
                "use_cls_token_embedding": use_cls_token_embedding,
                "rank": rank,
            },
            num_gpus=num_gpus,
            batch_size=1,
        )(col("image_binary")),
    )

    actual_df = ds.to_pandas()
    assert len(actual_df["embedding"][2]) == 768
    assert math.fabs(actual_df["embedding"][0][0] - 0.0079315146) < 0.001
