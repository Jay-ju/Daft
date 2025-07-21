# Copyright (c) Beijing Volcano Engine Technology Ltd.

from __future__ import annotations

import pandas as pd
import pytest

import daft
from daft import col
from daft.las.functions.image.image_resample import ImageResample
from daft.las.functions.udf import las_udf
from daft.las.functions.utils.image_utils import decode_image, image_to_base64, image_to_binary

image_suffix = ".jpg"
image_src_type = "image_url"
target_size = (200, 200)
# todo
target_dpi = (72, 72)
target_dpi = (120, 120)
method = "lanczos"
local_output = ""
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
    image_name = ["cat_test.png"] * 3
    return pd.DataFrame({"image_binary": image_binary, "image_name": image_name})


@pytest.mark.parametrize("method", ["nearest", "bilinear", "bicubic", "lanczos"])
def test_image_resample(tos_test_data_dir, local_test_data_dir, http_test_data_dir, method):
    input_df = generate_test_data(tos_test_data_dir, local_test_data_dir, http_test_data_dir)

    ds = daft.from_pandas(input_df)
    ds = ds.with_column(
        "image_resample",
        las_udf(
            ImageResample,
            construct_args={
                "image_suffix": image_suffix,
                "tos_dir": f"{tos_test_data_dir}/image_resample/",
                "local_output": local_output,
                "image_src_type": image_src_type,
                "target_size": target_size,
                "target_dpi": target_dpi,
                "method": method,
            },
            num_gpus=num_gpus,
            batch_size=1,
        )(col("image_path")),
    )
    actual_df = ds.to_pandas()
    assert actual_df["image_resample"][0]["image_path"] is None
    assert "iVBORw0KGgoAAAA" in actual_df["image_resample"][2]["base64"]


def test_image_resample_base64(tos_test_data_dir, local_test_data_dir, http_test_data_dir):
    input_df = generate_test_data_base64(tos_test_data_dir, local_test_data_dir, http_test_data_dir)

    ds = daft.from_pandas(input_df)
    ds = ds.with_column(
        "image_resample",
        las_udf(
            ImageResample,
            construct_args={
                "image_suffix": image_suffix,
                "tos_dir": f"{tos_test_data_dir}/image_resample/",
                "local_output": local_output,
                "image_src_type": "image_base64",
                "target_size": target_size,
                "target_dpi": target_dpi,
                "method": method,
            },
            num_gpus=num_gpus,
            batch_size=1,
        )(col("image_base64")),
    )
    actual_df = ds.to_pandas()
    assert "iVBORw0KGgoAAAA" in actual_df["image_resample"][2]["base64"]


def test_image_resample_binary(tos_test_data_dir, local_test_data_dir, http_test_data_dir):
    input_df = generate_test_data_binary(tos_test_data_dir, local_test_data_dir, http_test_data_dir)

    ds = daft.from_pandas(input_df)
    ds = ds.with_column(
        "image_resample",
        las_udf(
            ImageResample,
            construct_args={
                "image_suffix": image_suffix,
                "tos_dir": f"{tos_test_data_dir}/image_resample/",
                "local_output": local_output,
                "image_src_type": "image_binary",
                "target_size": target_size,
                "target_dpi": target_dpi,
                "method": method,
            },
            num_gpus=num_gpus,
            batch_size=1,
        )(col("image_binary"), col("image_name")),
    )
    actual_df = ds.to_pandas()
    assert "iVBORw0KGgoAAAA" in actual_df["image_resample"][2]["base64"]
