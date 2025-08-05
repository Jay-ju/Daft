# Copyright (c) Beijing Volcano Engine Technology Ltd.

from __future__ import annotations

import math

import pandas as pd
import pytest

import daft
from daft import col
from daft.las.functions.multimodal.embedding.clip_embedding import ClipEmbedding
from daft.las.functions.udf import las_udf
from daft.las.functions.utils.image_utils import decode_image, image_to_base64, image_to_binary

samples = {"text": ["皮卡丘", "Pikachu", "小狗", "小猫", None]}
content_type = "text"
input_df = pd.DataFrame(samples)

model_name = "iic/multi-modal_clip-vit-base-patch16_zh"
model_version = "v1.0.1"
embedding_col_name = "embedding"
batch_size = 2
rank = None
num_gpus = 0


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
def test_clip_embedding_txt(local_models_dir):
    model_list = [
        "iic/multi-modal_clip-vit-large-patch14_zh",
        "iic/multi-modal_clip-vit-large-patch14_336_zh",
        "iic/multi-modal_clip-vit-huge-patch14_zh",
        "iic/multi-modal_clip-vit-base-patch16_zh",
    ]
    ds = daft.from_pandas(input_df)
    for model_name in model_list:
        ds = ds.with_column(
            "embedding",
            las_udf(
                ClipEmbedding,
                construct_args={
                    "content_type": content_type,
                    "model_path": local_models_dir,
                    "model_name": model_name,
                    "model_version": model_version,
                    "batch_size": batch_size,
                    "rank": rank,
                },
                num_gpus=num_gpus,
                batch_size=1,
                concurrency=1,
            )(col("text")),
        )

        actual_df = ds.to_pandas()
    assert len(actual_df["embedding"][0]) == 512
    assert math.fabs(actual_df["embedding"][0][0] - 0.12005615) < 0.001
    assert actual_df["embedding"][4] is None


@pytest.mark.gpu
def test_clip_embedding_img(local_models_dir, tos_test_data_dir, local_test_data_dir, http_test_data_dir):
    input_df = generate_test_data(tos_test_data_dir, local_test_data_dir, http_test_data_dir)

    ds = daft.from_pandas(input_df)
    ds = ds.with_column(
        "embedding",
        las_udf(
            ClipEmbedding,
            construct_args={
                "content_type": "image_url",
                "model_path": local_models_dir,
                "model_name": model_name,
                "model_version": model_version,
                "batch_size": batch_size,
                "rank": rank,
            },
            num_gpus=num_gpus,
            batch_size=1,
            concurrency=1,
        )(col("image_path")),
    )

    actual_df = ds.to_pandas()
    assert actual_df["embedding"][0] is None
    assert len(actual_df["embedding"][2]) == 512
    assert math.fabs(actual_df["embedding"][2][0] - 0.0477461) < 0.001


@pytest.mark.gpu
def test_clip_embedding_img_base64(local_models_dir, tos_test_data_dir, local_test_data_dir, http_test_data_dir):
    input_df = generate_test_data_base64(tos_test_data_dir, local_test_data_dir, http_test_data_dir)

    ds = daft.from_pandas(input_df)
    ds = ds.with_column(
        "embedding",
        las_udf(
            ClipEmbedding,
            construct_args={
                "content_type": "image_base64",
                "model_path": local_models_dir,
                "model_name": model_name,
                "model_version": model_version,
                "batch_size": batch_size,
                "rank": rank,
            },
            num_gpus=num_gpus,
            batch_size=1,
            concurrency=1,
        )(col("image_base64")),
    )

    actual_df = ds.to_pandas()
    assert len(actual_df["embedding"][2]) == 512
    assert math.fabs(actual_df["embedding"][0][0] - 0.0477461) < 0.001


@pytest.mark.gpu
def test_clip_embedding_img_binary(local_models_dir, tos_test_data_dir, local_test_data_dir, http_test_data_dir):
    input_df = generate_test_data_binary(tos_test_data_dir, local_test_data_dir, http_test_data_dir)

    ds = daft.from_pandas(input_df)
    ds = ds.with_column(
        "embedding",
        las_udf(
            ClipEmbedding,
            construct_args={
                "content_type": "image_binary",
                "model_path": local_models_dir,
                "model_name": model_name,
                "model_version": model_version,
                "batch_size": batch_size,
                "rank": rank,
            },
            num_gpus=num_gpus,
            batch_size=1,
            concurrency=1,
        )(col("image_binary")),
    )

    actual_df = ds.to_pandas()
    assert len(actual_df["embedding"][2]) == 512
    assert math.fabs(actual_df["embedding"][0][0] - 0.0477461) < 0.001
