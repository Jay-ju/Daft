from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import daft
from daft import col
from daft.las.functions.udf import las_udf
from daft.las.functions.utils.common_utils import load_file
from daft.las.functions.video.video_nsfw_detect import VideoNsfwDetect

num_gpus = 1


def generate_test_data_url(tos_test_data_dir: str, local_test_data_dir: str, http_test_data_dir: str) -> pd.DataFrame:
    paths = [
        "",  # 空路径，应返回 NaN
        f"{local_test_data_dir}/video/non-exist.mp4",  # 不存在的本地视频，应返回 NaN
        f"{local_test_data_dir}/video/music_sample.mp4",  # 本地有效视频
        f"{tos_test_data_dir}/video/music_sample.mp4",  # TOS 有效视频（通常为 HTTP 可访问）
        f"{http_test_data_dir}/video/music_sample.mp4",  # HTTP 有效视频
    ]
    return pd.DataFrame({"video_path": paths})


def generate_test_data_base64(local_test_data_dir: str) -> pd.DataFrame:
    paths = [
        f"{local_test_data_dir}/video/music_sample.mp4",
    ]
    b64_list = [load_file(p, as_base64=True) for p in paths]
    return pd.DataFrame({"video_base64": b64_list})


def generate_test_data_binary(local_test_data_dir: str) -> pd.DataFrame:
    paths = [
        f"{local_test_data_dir}/video/music_sample.mp4",
    ]
    binaries = [load_file(p, as_base64=False) for p in paths]
    return pd.DataFrame({"video_binary": binaries})


def _is_prob(x: float) -> bool:
    try:
        return np.isfinite(x) and 0.0 <= float(x) <= 1.0
    except Exception:
        return False


@pytest.mark.gpu
def test_nsfw_detect_url(tos_test_data_dir, local_test_data_dir, http_test_data_dir, local_models_dir):
    input_df = generate_test_data_url(tos_test_data_dir, local_test_data_dir, http_test_data_dir)

    ds = daft.from_pandas(input_df)
    ds = ds.with_column(
        "nsfw",
        las_udf(
            VideoNsfwDetect,
            construct_args={
                "model_path": local_models_dir,
                "video_src_type": "video_url",
                "sample_mode": "by_count_uniform",
                "count_k": 3,
            },
            num_gpus=num_gpus,
            batch_size=1,
            concurrency=1,
        )(col("video_path")),
    )

    actual_df = ds.to_pandas()
    # 前两项应为 NaN
    assert pd.isna(actual_df["nsfw"][0])
    assert pd.isna(actual_df["nsfw"][1])
    # 后三项若可访问则应返回 [0,1] 内概率
    for i in [2, 3, 4]:
        val = actual_df["nsfw"][i]
        if pd.isna(val):
            continue
        assert _is_prob(val)
    assert np.allclose(actual_df["nsfw"][2], 0.000428, atol=0.1)


@pytest.mark.gpu
def test_nsfw_detect_base64(tos_test_data_dir, local_test_data_dir, http_test_data_dir, local_models_dir):
    input_df = generate_test_data_base64(local_test_data_dir)

    ds = daft.from_pandas(input_df)
    ds = ds.with_column(
        "nsfw",
        las_udf(
            VideoNsfwDetect,
            construct_args={
                "model_path": local_models_dir,
                "video_src_type": "video_base64",
                "sample_mode": "by_interval_time",
                "interval_sec": 1,
            },
            num_gpus=num_gpus,
            batch_size=1,
            concurrency=1,
            num_cpus=4,
        )(col("video_base64")),
    )
    actual_df = ds.to_pandas()
    assert np.allclose(actual_df["nsfw"][0], 0.000428, atol=0.1)


@pytest.mark.gpu
def test_nsfw_detect_binary(tos_test_data_dir, local_test_data_dir, http_test_data_dir, local_models_dir):
    input_df = generate_test_data_binary(local_test_data_dir)

    ds = daft.from_pandas(input_df)
    ds = ds.with_column(
        "nsfw",
        las_udf(
            VideoNsfwDetect,
            construct_args={
                "model_path": local_models_dir,
                "video_src_type": "video_binary",
                "sample_mode": "by_fps",
                "target_fps": 0.2,
                "video_format": "mp4",
            },
            num_gpus=num_gpus,
            batch_size=1,
            concurrency=1,
        )(col("video_binary")),
    )
    actual_df = ds.to_pandas()
    assert np.allclose(actual_df["nsfw"][0], 0.000215, atol=0.1)
