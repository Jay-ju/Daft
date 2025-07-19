# Copyright (c) Beijing Volcano Engine Technology Ltd.

from __future__ import annotations

import pandas as pd

import daft
from daft import col
from daft.las.functions.audio.audio_standardization import AudioStandardization
from daft.las.functions.udf import las_udf


def generate_test_data(tos_test_data_dir: str) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "audio_bytes": [
                f"{tos_test_data_dir}/audio/non-exist.wav",
                f"{tos_test_data_dir}/audio/耙耙柑大叔.aac",
                "",
            ]
        }
    )


def test_audio_standardization(tos_test_data_dir: str):
    input_df = generate_test_data(tos_test_data_dir)

    ds = daft.from_pandas(input_df)
    ds = ds.with_column(
        "standardized_audio",
        las_udf(
            AudioStandardization,
            construct_args={
                "target_sr": 16000,
                "target_channels": 1,
                "target_dbfs": -20.0,
                "target_gain_range": [-3.0, 3.0],
                "num_coroutines": 2,
            },
            num_gpus=0,
            batch_size=1,
            concurrency=2,
        )(col("audio_bytes")),
    )

    df = ds.to_pandas()

    assert df.iloc[0]["standardized_audio"] is None

    result_audio = df.iloc[1]["standardized_audio"]
    assert isinstance(result_audio, bytes) and len(result_audio) > 0

    assert df.iloc[2]["standardized_audio"] is None
