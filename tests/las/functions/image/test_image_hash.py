# Copyright (c) Beijing Volcano Engine Technology Ltd.

from __future__ import annotations

import pandas as pd
import pytest

import daft
from daft import col
from daft.las.functions.image.image_hash import ImageHash
from daft.las.functions.udf import las_udf
from daft.las.functions.utils.image_utils import decode_image, image_to_base64, image_to_binary


def generate_test_data(tos_test_data_dir: str, local_test_data_dir: str, http_test_data_dir: str) -> pd.DataFrame:
    paths = [
        "",
        f"{local_test_data_dir}/image/non-exist.png",
        f"{local_test_data_dir}/image/cat.png",
        f"{tos_test_data_dir}/image/cat.png",
        f"{tos_test_data_dir}/image/bird.jpeg",
        f"{http_test_data_dir}/image/cat.png",
    ]
    return pd.DataFrame({"image_path": paths})


def generate_test_data_base64(
    tos_test_data_dir: str, local_test_data_dir: str, http_test_data_dir: str
) -> pd.DataFrame:
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


# 校验函数：十六进制(16位)与二进制(64位)
def is_hex16(s: str) -> bool:
    if not isinstance(s, str) or len(s) != 16:
        return False
    s_lower = s.lower()
    return all(ch in "0123456789abcdef" for ch in s_lower)


def is_bin64(s: str) -> bool:
    if not isinstance(s, str) or len(s) != 64:
        return False
    return set(s) <= {"0", "1"}


@pytest.mark.parametrize("method", ["ahash", "dhash", "phash", "whash", "md5"])
def test_image_hash_url(tos_test_data_dir, local_test_data_dir, http_test_data_dir, method):
    input_df = generate_test_data(tos_test_data_dir, local_test_data_dir, http_test_data_dir)

    ds = daft.from_pandas(input_df)
    ds = ds.with_column(
        "image_hash",
        las_udf(
            ImageHash,
            construct_args={
                "image_src_type": "image_url",
                "method": method,
            },
            batch_size=1,
            concurrency=1,
        )(col("image_path")),
    )

    actual_df = ds.to_pandas()
    # 无效输入返回空字符串
    assert actual_df["image_hash"][0]["hash_hex"] == ""
    assert actual_df["image_hash"][0]["hash_bin"] == ""

    # 有效输入为合法 hex/bin 格式
    assert is_hex16(actual_df["image_hash"][2]["hash_hex"]) is True
    assert is_bin64(actual_df["image_hash"][2]["hash_bin"]) is True


def test_image_hash_base64(tos_test_data_dir, local_test_data_dir, http_test_data_dir):
    input_df = generate_test_data_base64(tos_test_data_dir, local_test_data_dir, http_test_data_dir)

    ds = daft.from_pandas(input_df)
    ds = ds.with_column(
        "image_hash",
        las_udf(
            ImageHash,
            construct_args={
                "image_src_type": "image_base64",
                "method": "phash",
            },
            batch_size=1,
            concurrency=1,
        )(col("image_base64")),
    )

    actual_df = ds.to_pandas()
    assert is_hex16(actual_df["image_hash"][2]["hash_hex"]) is True
    assert is_bin64(actual_df["image_hash"][2]["hash_bin"]) is True


def test_image_hash_binary(tos_test_data_dir, local_test_data_dir, http_test_data_dir):
    input_df = generate_test_data_binary(tos_test_data_dir, local_test_data_dir, http_test_data_dir)

    ds = daft.from_pandas(input_df)
    ds = ds.with_column(
        "image_hash",
        las_udf(
            ImageHash,
            construct_args={
                "image_src_type": "image_binary",
                "method": "phash",
            },
            batch_size=1,
            concurrency=1,
        )(col("image_binary")),
    )

    actual_df = ds.to_pandas()
    assert is_hex16(actual_df["image_hash"][2]["hash_hex"]) is True
    assert is_bin64(actual_df["image_hash"][2]["hash_bin"]) is True
