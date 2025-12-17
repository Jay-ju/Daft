# Copyright (c) Beijing Volcano Engine Technology Ltd.

from __future__ import annotations

import pandas as pd
import pytest

import daft
from daft import col
from daft.las.functions.audio.audio_denoise import AudioDenoise
from daft.las.functions.audio.audio_duration import AudioDuration
from daft.las.functions.udf import las_udf
from tests.las.functions import assert_dataframe_result


def generate_test_data(tos_test_data_dir, local_test_data_dir, http_test_data_dir):
    paths = [
        f"{tos_test_data_dir}/audio/non-exist.wav",
        f"{tos_test_data_dir}/audio/sample.wav",  # 正常音频
        f"{local_test_data_dir}/audio/黑神话悟空对话.mp3",
        f"{http_test_data_dir}/audio/街边叫卖声.wav",
    ]
    output_paths = [
        f"{tos_test_data_dir}/audio/outputs/audio_denoised/non-exist.wav",
        f"{tos_test_data_dir}/audio/outputs/audio_denoised/sample.wav",  # 正常音频
        f"{tos_test_data_dir}/audio/outputs/audio_denoised/黑神话悟空对话.mp3",  # 44.1kHz mp3
        f"{tos_test_data_dir}/audio/outputs/audio_denoised/街边叫卖声.wav",  # http do not support upload, use local path
    ]
    result_paths = [
        None,
        f"{tos_test_data_dir}/audio/outputs/audio_denoised/sample.wav",
        f"{tos_test_data_dir}/audio/outputs/audio_denoised/黑神话悟空对话.mp3",
        f"{tos_test_data_dir}/audio/outputs/audio_denoised/街边叫卖声.wav",
    ]
    durations = [None, "49.71", "49.75", "16.31"]

    input_df = pd.DataFrame(
        {
            "audio_path": paths,
            "output_path": output_paths,
        }
    )
    expected_df = pd.DataFrame(
        {
            "result_path": result_paths,
            "audio_duration": durations,
        }
    )
    return input_df, expected_df


@pytest.mark.gpu
def test_audio_denoise(local_models_dir, tos_test_data_dir, local_test_data_dir, http_test_data_dir):
    input_df, expected_df = generate_test_data(tos_test_data_dir, local_test_data_dir, http_test_data_dir)

    df = daft.from_pandas(input_df)
    df = df.with_column(
        "result_path",
        las_udf(
            AudioDenoise,
            construct_args={"model_path": local_models_dir},
            num_gpus=1,
            concurrency=1,
        )(col("audio_path"), col("output_path")),
    )

    # 计算 denoise 后的音频时长
    df = df.with_column("audio_duration", las_udf(AudioDuration)(col("result_path")))
    df = df.with_column("audio_duration", daft.col("audio_duration").round(2))

    actual = df.select("result_path", "audio_duration").to_pandas()
    assert_dataframe_result(actual[["result_path"]], expected_df[["result_path"]])
    assert abs(float(actual["audio_duration"][2]) - float(expected_df["audio_duration"][2])) < 0.01
