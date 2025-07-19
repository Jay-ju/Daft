# Copyright (c) Beijing Volcano Engine Technology Ltd.

from __future__ import annotations

import pandas as pd

import daft
from daft import col
from daft.las.functions.audio import AudioSourceSeparation
from daft.las.functions.udf import las_udf


def generate_test_data(tos_test_data_dir: str) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "audio_path": [
                f"tos://{tos_test_data_dir}/audio/test_music.m4a",  # 正常音频
                f"tos://{tos_test_data_dir}/audio/non_exist.m4a",  # 不存在音频
                "",  # 空路径
            ]
        }
    )


def test_audio_source_separation(tos_test_data_dir: str, local_models_dir: str):
    input_df = generate_test_data(tos_test_data_dir)

    df = daft.from_pandas(input_df)
    df = df.with_column(
        "audio_vocal",
        las_udf(
            AudioSourceSeparation,
            construct_args={"model_path": local_models_dir},
            num_gpus=1,
            batch_size=1,
            concurrency=2,
        )(col("audio_path")),
    )

    result_df = df.to_pandas()

    # Case 1: 正常音频
    result = result_df.iloc[0]["audio_vocal"]
    assert isinstance(result, bytes) and len(result) > 0, "Valid audio should produce non-empty bytes"

    # Case 2: 不存在的路径
    assert result_df.iloc[1]["audio_vocal"] is None

    # Case 3: 空字符串
    assert result_df.iloc[2]["audio_vocal"] is None
