# Copyright (c) Beijing Volcano Engine Technology Ltd.

from __future__ import annotations

import os
import random

import pandas as pd

import daft
from daft import col
from daft.las.functions.audio.audio_lid_whisper import AudioLidWhisper
from daft.las.functions.udf import las_udf

model_name = "iic/speech_whisper-large_lid_multilingual_pytorch"
model_version = "v2.0.4"
num_gpus = int(os.getenv("NUM_GPUS", 1))
rank = random.randint(0, num_gpus - 1)


def generate_test_data(tos_test_data_dir, local_test_data_dir, http_test_data_dir):
    paths = [
        "",
        f"{local_test_data_dir}/audio/non-exist.wav",
        f"{local_test_data_dir}/audio/全剧大部分都是在中国取景拍摄，地点位于浙江省温州市。.wav",
        f"{tos_test_data_dir}/audio/街边叫卖声.wav",
        f"{tos_test_data_dir}/audio/全剧大部分都是在中国取景拍摄，地点位于浙江省温州市。.wav",
        f"{tos_test_data_dir}/audio/Philippines_english_Haduken_20250701_19817.wav",
        f"{http_test_data_dir}/audio/全剧大部分都是在中国取景拍摄，地点位于浙江省温州市。.wav",
    ]
    return pd.DataFrame({"audio_path": paths})


def test_audio_lid_whisper(local_models_dir, tos_test_data_dir, local_test_data_dir, http_test_data_dir):
    input_df = generate_test_data(tos_test_data_dir, local_test_data_dir, http_test_data_dir)

    ds = daft.from_pandas(input_df)
    ds = ds.with_column(
        "lid_result",
        las_udf(
            AudioLidWhisper,
            construct_args={
                "model_path": local_models_dir,
                "model_name": model_name,
                "model_version": model_version,
                "rank": rank,
            },
            num_gpus=num_gpus,
            batch_size=1,
            concurrency=1,
        )(col("audio_path")),
    )

    actual_df = ds.to_pandas()
    print(actual_df)

    assert actual_df["lid_result"][0]["language_code"] is None or len(actual_df["lid_result"][0]["language_code"]) == 0
    assert actual_df["lid_result"][2]["language_code"] == "zh"
    assert actual_df["lid_result"][5]["language_code"] == "en"
