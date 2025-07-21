# Copyright (c) Beijing Volcano Engine Technology Ltd.

from __future__ import annotations

import pandas as pd
import pytest

import daft
from daft import col
from daft.las.functions.image.image_easyocr import ImageEasyOcr
from daft.las.functions.udf import las_udf
from daft.las.functions.utils.image_utils import decode_image, image_to_base64, image_to_binary

image_src_type = "image_url"
model_name = "EasyOCR"
quantize = True
lang_list = ["en", "ch_sim"]
batch_size = 16
num_gpus = 1


def generate_test_data(tos_test_data_dir, local_test_data_dir, http_test_data_dir):
    paths = [
        "",
        f"{local_test_data_dir}/image/non-exist.png",
        f"{local_test_data_dir}/image/打印体.jpg",
        f"{tos_test_data_dir}/image/cat.png",
        f"{tos_test_data_dir}/image/打印体.jpg",
        f"{http_test_data_dir}/image/打印体.jpg",
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


@pytest.mark.parametrize("quantize", [True, False])
def test_easyocr(local_models_dir, tos_test_data_dir, local_test_data_dir, http_test_data_dir, quantize):
    input_df = generate_test_data(tos_test_data_dir, local_test_data_dir, http_test_data_dir)

    ds = daft.from_pandas(input_df)
    ds = ds.with_column(
        "ocr_result",
        las_udf(
            ImageEasyOcr,
            construct_args={
                "image_src_type": image_src_type,
                "model_path": local_models_dir,
                "model_name": model_name,
                "quantize": quantize,
                "lang_list": lang_list,
                "batch_size": batch_size,
            },
            num_gpus=num_gpus,
            batch_size=1,
        )(col("image_path")),
    )
    actual_df = ds.to_pandas()
    assert actual_df["ocr_result"][0] is None
    assert "人生四然" in actual_df["ocr_result"][2]


def test_easyocr_base64(local_models_dir, tos_test_data_dir, local_test_data_dir, http_test_data_dir):
    input_df = generate_test_data_base64(tos_test_data_dir, local_test_data_dir, http_test_data_dir)

    ds = daft.from_pandas(input_df)
    ds = ds.with_column(
        "ocr_result",
        las_udf(
            ImageEasyOcr,
            construct_args={
                "image_src_type": "image_base64",
                "model_path": local_models_dir,
                "model_name": model_name,
                "quantize": quantize,
                "lang_list": lang_list,
                "batch_size": batch_size,
            },
            num_gpus=num_gpus,
            batch_size=1,
        )(col("image_base64")),
    )

    actual_df = ds.to_pandas()
    assert "人生四然" in actual_df["ocr_result"][2]


def test_easyocr_binary(local_models_dir, tos_test_data_dir, local_test_data_dir, http_test_data_dir):
    input_df = generate_test_data_binary(tos_test_data_dir, local_test_data_dir, http_test_data_dir)

    ds = daft.from_pandas(input_df)
    ds = ds.with_column(
        "ocr_result",
        las_udf(
            ImageEasyOcr,
            construct_args={
                "image_src_type": "image_binary",
                "model_path": local_models_dir,
                "model_name": model_name,
                "quantize": quantize,
                "lang_list": lang_list,
                "batch_size": batch_size,
            },
            num_gpus=num_gpus,
            batch_size=1,
        )(col("image_binary")),
    )

    actual_df = ds.to_pandas()
    assert "人生四然" in actual_df["ocr_result"][2]
