# Copyright (c) Beijing Volcano Engine Technology Ltd.

from __future__ import annotations

import pandas as pd

import daft
from daft import col
from daft.las.functions.audio import AudioConvert
from daft.las.functions.udf import las_udf
from tests.las.functions import assert_dataframe_result


def generate_test_data(tos_test_data_dir, local_test_data_dir, http_test_data_dir):
    samples = {
        "input_path": [
            f"{tos_test_data_dir}/audio/non-exist.wav",
            f"{tos_test_data_dir}/audio/sample.wav",
            f"{local_test_data_dir}/audio/sample.wav",
            f"{http_test_data_dir}/audio/sample.wav",
        ],
        "target_output_path": [
            f"{tos_test_data_dir}/audio/outputs/audio_convert/non-exist.mp3",
            f"{tos_test_data_dir}/audio/outputs/audio_convert/sample_converted.mp3",
            f"{tos_test_data_dir}/audio/outputs/audio_convert/sample_converted.mp3",
            f"{tos_test_data_dir}/audio/outputs/audio_convert/sample_converted.mp3",
        ],
    }
    input_df = pd.DataFrame(samples)

    expected_output_paths = [
        None,
        f"{tos_test_data_dir}/audio/outputs/audio_convert/sample_converted.mp3",
        f"{tos_test_data_dir}/audio/outputs/audio_convert/sample_converted.mp3",
        f"{tos_test_data_dir}/audio/outputs/audio_convert/sample_converted.mp3",
    ]
    expected_df = pd.DataFrame(
        {
            "input_path": samples["input_path"],
            "target_output_path": samples["target_output_path"],
            "result": expected_output_paths,
        }
    )

    return input_df, expected_df


def test_audio_convert_basic(tos_test_data_dir, local_test_data_dir, http_test_data_dir):
    """Test AudioConvert operator with basic conversion parameters."""
    input_df, expected_df = generate_test_data(tos_test_data_dir, local_test_data_dir, http_test_data_dir)
    constructor_kwargs = {
        "output_format": "mp3",
        "extra_params": ["-c:a", "libmp3lame", "-b:a", "192k"],
        "bitrate": "192k",
        "sample_rate": 44100,
        "quality": 2,
    }

    df = daft.from_pandas(input_df)
    result_df = df.with_column(
        "result",
        las_udf(
            AudioConvert,
            construct_args=constructor_kwargs,
            num_gpus=0,
            batch_size=1,
            concurrency=1,
        )(col("input_path"), col("target_output_path")),
    ).to_pandas()

    assert_dataframe_result(result_df, expected_df)
