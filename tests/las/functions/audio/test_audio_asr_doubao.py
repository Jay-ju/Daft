# Copyright (c) Beijing Volcano Engine Technology Ltd.

from __future__ import annotations

import os

import pandas as pd

import daft
from daft import col
from daft.las.functions.audio.audio_asr_doubao import AudioAsrDoubao
from daft.las.functions.udf import las_udf


def generate_test_data(http_test_data_dir):
    paths = [
        f"{http_test_data_dir}/audio/non-exist.wav",
        f"{http_test_data_dir}/audio/全剧大部分都是在中国取景拍摄，地点位于浙江省温州市。.wav",
    ]
    return pd.DataFrame({"audio_path": paths})


def test_audio_asr_doubao(http_test_data_dir):
    appid = os.getenv("OPENSPEECH_APPID")
    token = os.getenv("OPENSPEECH_TOKEN")
    input_df = generate_test_data(http_test_data_dir)

    ds = daft.from_pandas(input_df)
    ds = ds.with_column(
        "asr_result",
        las_udf(
            AudioAsrDoubao,
            construct_args={"appid": appid, "token": token, "concurrency": 1, "poll_interval": 15},
            batch_size=1,
            concurrency=1,
        )(col("audio_path")),
    )

    ds = ds.with_column("asr_result_raw", col("asr_result").struct.get("asr_result_raw"))
    ds = ds.with_column("asr_result_simple", col("asr_result").struct.get("asr_result_simple"))

    actual_df = ds.to_pandas()
    assert actual_df.iloc[0]["asr_result_simple"] is None
    assert "说话人" in actual_df.iloc[1]["asr_result_simple"]
