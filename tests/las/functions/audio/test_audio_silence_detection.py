# Copyright (c) Beijing Volcano Engine Technology Ltd.

from __future__ import annotations

import pandas as pd

import daft
from daft import col
from daft.las.functions.audio import AudioSilenceDetection
from daft.las.functions.udf import las_udf
from tests.las.functions import assert_dataframe_result


def generate_test_data(tos_test_data_dir, local_test_data_dir, http_test_data_dir):
    samples = {
        "input_path": [
            f"{tos_test_data_dir}/audio/non-exist.wav",
            f"{tos_test_data_dir}/audio/silence.wav",  # 静音音频
            f"{tos_test_data_dir}/audio/sample.wav",  # 正常音频
            f"{local_test_data_dir}/audio/silence.wav",
            f"{local_test_data_dir}/audio/sample.wav",
            f"{http_test_data_dir}/audio/silence.wav",
            f"{http_test_data_dir}/audio/sample.wav",
        ],
    }
    input_df = pd.DataFrame(samples)

    # 期望结果：不存在的文件返回False，silence.wav返回True，sample.wav返回False
    expected_results = [
        False,  # 不存在的文件
        True,  # 静音音频
        False,  # 正常音频
        True,  # 静音音频
        False,  # 正常音频
        True,  # 静音音频
        False,  # 正常音频
    ]
    expected_df = pd.DataFrame(
        {
            "input_path": samples["input_path"],
            "is_silence": expected_results,
        }
    )

    return input_df, expected_df


def test_audio_silence_detection_basic(tos_test_data_dir, local_test_data_dir, http_test_data_dir):
    """Test AudioSilenceDetection operator with default parameters."""
    input_df, expected_df = generate_test_data(tos_test_data_dir, local_test_data_dir, http_test_data_dir)
    constructor_kwargs = {
        "silence_threshold_db": -60.0,
    }

    df = daft.from_pandas(input_df)
    result_df = df.with_column(
        "is_silence",
        las_udf(
            AudioSilenceDetection,
            construct_args=constructor_kwargs,
            num_gpus=0,
            batch_size=1,
            concurrency=1,
        )(col("input_path")),
    ).to_pandas()

    assert_dataframe_result(result_df, expected_df)
