# Copyright (c) Beijing Volcano Engine Technology Ltd.

from __future__ import annotations

import os

import pandas as pd

import daft
from daft import col
from daft.las.functions.audio.audio_tts_doubao import AudioTtsDoubao
from daft.las.functions.udf import las_udf


def generate_test_data() -> pd.DataFrame:
    texts = [
        "",
        "今天天气真好，适合出去走走。",
    ]
    return pd.DataFrame({"text_input": texts})


def test_audio_tts_doubao():
    appid = os.getenv("OPENSPEECH_APPID")
    token = os.getenv("OPENSPEECH_TOKEN")

    input_df = generate_test_data()

    ds = daft.from_pandas(input_df)
    ds = ds.with_column(
        "tts_audio",
        las_udf(
            AudioTtsDoubao,
            construct_args={
                "appid": appid,
                "token": token,
                "uid": "test",
                "timeout": 60,
                "num_coroutines": 1,
            },
            batch_size=1,
            concurrency=1,
        )(col("text_input")),
    )

    actual_df = ds.to_pandas()

    assert actual_df.iloc[0]["tts_audio"] is None
    assert actual_df.iloc[1]["tts_audio"] is not None
