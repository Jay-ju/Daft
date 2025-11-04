# Copyright (c) Beijing Volcano Engine Technology Ltd.

from __future__ import annotations

import os

import numpy as np
import pandas as pd

import daft
from daft import col
from daft.las.functions.audio.audio_vad_silero import AudioVadSilero
from daft.las.functions.udf import las_udf

audio_src_type = "audio_url"
batch_size_s = 3600
model_name = "silero-vad"
use_onnx_model = True
onnx_model_revision = 16
num_gpus = int(os.getenv("NUM_GPUS", 1))
num_gpus = 1


def generate_test_data(tos_test_data_dir, local_test_data_dir, http_test_data_dir):
    paths = [
        "",
        f"{local_test_data_dir}/audio/non-exist.wav",
        f"{local_test_data_dir}/audio/全剧大部分都是在中国取景拍摄，地点位于浙江省温州市。.wav",
        f"{local_test_data_dir}/audio/耙耙柑大叔.aac",
        f"{tos_test_data_dir}/audio/全剧大部分都是在中国取景拍摄，地点位于浙江省温州市。.wav",
        f"{http_test_data_dir}/audio/全剧大部分都是在中国取景拍摄，地点位于浙江省温州市。.wav",
    ]
    return pd.DataFrame({"audio_path": paths})


def test_audio_vad_silero(local_models_dir, tos_test_data_dir, local_test_data_dir, http_test_data_dir):
    input_df = generate_test_data(tos_test_data_dir, local_test_data_dir, http_test_data_dir)

    ds = daft.from_pandas(input_df)
    ds = ds.with_column(
        "audio_vad_result",
        las_udf(
            AudioVadSilero,
            construct_args={
                "audio_src_type": audio_src_type,
                "model_path": local_models_dir,
                "model_name": model_name,
                "use_onnx_model": use_onnx_model,
                "onnx_model_revision": onnx_model_revision,
            },
            num_gpus=1,
            batch_size=1,
            concurrency=1,
        )(col("audio_path")),
    )

    actual_df = ds.to_pandas()
    assert actual_df["audio_vad_result"][0] is None or len(actual_df["audio_vad_result"][0]) == 0
    assert actual_df["audio_vad_result"][1] is None or len(actual_df["audio_vad_result"][1]) == 0
    assert np.allclose(actual_df["audio_vad_result"][2][0], [0.8, 4.4], atol=0.1)
