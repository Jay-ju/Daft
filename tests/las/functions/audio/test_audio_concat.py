# Copyright (c) Beijing Volcano Engine Technology Ltd.

from __future__ import annotations

import pandas as pd

import daft
from daft import col
from daft.las.functions.audio import AudioConcat
from daft.las.functions.udf import las_udf
from tests.las.functions import assert_dataframe_result


def generate_test_data(tos_test_data_dir, local_test_data_dir, http_test_data_dir):
    samples = {
        "audio_paths": [
            [
                f"{tos_test_data_dir}/audio/speaker1_a_cn_16k.wav",
                f"{tos_test_data_dir}/audio/speaker1_b_cn_16k.wav",
            ],
            [
                f"{tos_test_data_dir}/audio/non-exist.wav",
            ],
            [
                f"{tos_test_data_dir}/audio/speaker1_a_cn_16k.wav",
            ],
        ],
        "output_path": [
            f"{tos_test_data_dir}/audio/audio_concat/concatenated_audio.mp3",
            f"{tos_test_data_dir}/audio/audio_concat/non_exist_concat.mp3",
            f"{tos_test_data_dir}/audio/audio_concat/single_audio.mp3",
        ],
    }
    input_df = pd.DataFrame(samples)

    # 预期结果：
    # - 第一个：成功拼接两个音频文件
    # - 第二个：文件不存在，返回 None
    # - 第三个：只有一个音频文件，也应该成功处理
    expected_output_paths = [
        f"{tos_test_data_dir}/audio/audio_concat/concatenated_audio.mp3",
        None,
        f"{tos_test_data_dir}/audio/audio_concat/single_audio.mp3",
    ]
    expected_df = pd.DataFrame(
        {
            "audio_paths": samples["audio_paths"],
            "output_path": expected_output_paths,
        }
    )

    return input_df, expected_df


def test_audio_concat_basic(tos_test_data_dir, local_test_data_dir, http_test_data_dir):
    """Test AudioConcat operator with basic parameters."""
    input_df, expected_df = generate_test_data(tos_test_data_dir, local_test_data_dir, http_test_data_dir)
    constructor_kwargs = {
        "output_format": "mp3",
        "sample_rate": 16000,
        "extra_params": ["-b:a", "128k"],
    }

    df = daft.from_pandas(input_df)
    result_df = df.with_column(
        "result_path",
        las_udf(
            AudioConcat,
            construct_args=constructor_kwargs,
            num_gpus=0,
            batch_size=1,
            concurrency=1,
        )(col("audio_paths"), col("output_path")),
    )

    result_df = result_df.with_column("output_path", col("result_path")).select(col("audio_paths"), col("output_path"))

    result_pandas = result_df.to_pandas()
    assert_dataframe_result(result_pandas, expected_df)
