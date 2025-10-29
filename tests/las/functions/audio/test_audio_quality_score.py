# Copyright (c) Beijing Volcano Engine Technology Ltd.

from __future__ import annotations

import pandas as pd

import daft
from daft import col
from daft.las.functions.audio import AudioQualityScore
from daft.las.functions.udf import las_udf


def generate_test_data(tos_test_data_dir, local_test_data_dir, http_test_data_dir):
    samples = {
        "input_path": [
            f"{tos_test_data_dir}/audio/non-exist.wav",
            f"{tos_test_data_dir}/audio/sample.wav",  # 正常音频
            f"{local_test_data_dir}/audio/sample.wav",
            f"{http_test_data_dir}/audio/sample.wav",
        ],
    }
    input_df = pd.DataFrame(samples)

    # 期望结果：不存在的文件返回None，正常音频返回质量评分结构（包含ovrl, sig, bak三个字段，范围1.0-5.0）
    expected_results = [
        {"ovrl": None, "sig": None, "bak": None},  # 不存在的文件
        {"ovrl": 1.747, "sig": 2.418, "bak": 1.744},
        {"ovrl": 1.747, "sig": 2.418, "bak": 1.744},
        {"ovrl": 1.747, "sig": 2.418, "bak": 1.744},
    ]
    expected_df = pd.DataFrame(
        {
            "input_path": samples["input_path"],
            "quality_score": expected_results,
        }
    )

    return input_df, expected_df


def test_audio_quality_score_basic(tos_test_data_dir, local_test_data_dir, http_test_data_dir, local_models_dir):
    """Test AudioQualityScore operator with default parameters."""
    input_df, expected_df = generate_test_data(tos_test_data_dir, local_test_data_dir, http_test_data_dir)
    constructor_kwargs = {
        "model_path": local_models_dir,
        "device": "cpu",
        "is_personalized_mos": False,
    }

    df = daft.from_pandas(input_df)
    result_df = df.with_column(
        "quality_score",
        las_udf(
            AudioQualityScore,
            construct_args=constructor_kwargs,
            num_gpus=0,
            batch_size=1,
            concurrency=1,
        )(col("input_path")),
    ).to_pandas()

    import math

    for i, (res, exp) in enumerate(zip(result_df["quality_score"], expected_df["quality_score"])):
        for key in res:
            assert (res[key] is None and exp[key] is None) or math.isclose(
                res[key], exp[key], rel_tol=1e-3, abs_tol=1e-3
            ), f"Row {i}, key {key} differ: {res[key]} vs {exp[key]}"
