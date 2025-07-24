# Copyright (c) Beijing Volcano Engine Technology Ltd.

from __future__ import annotations

import os
import random

import numpy as np
import pandas as pd

import daft
from daft import col
from daft.las.functions.audio.audio_vad_fsmn import AudioVadFsmn
from daft.las.functions.udf import las_udf

audio_src_type = "audio_url"
batch_size_s = 3600
model_name = "iic/speech_fsmn_vad_zh-cn-16k-common-pytorch"
model_revision = "v2.0.4"
num_gpus = int(os.getenv("NUM_GPUS", 1))
rank = random.randint(0, num_gpus - 1)


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


def test_audio_vad_fsmn(local_models_dir, tos_test_data_dir, local_test_data_dir, http_test_data_dir):
    input_df = generate_test_data(tos_test_data_dir, local_test_data_dir, http_test_data_dir)

    ds = daft.from_pandas(input_df)
    ds = ds.with_column(
        "audio_vad_result",
        las_udf(
            AudioVadFsmn,
            construct_args={
                "audio_src_type": audio_src_type,
                "model_path": local_models_dir,
                "model_name": model_name,
                "model_revision": model_revision,
                "batch_size_s": batch_size_s,
                "rank": rank,
            },
            num_gpus=num_gpus,
            batch_size=1,
            concurrency=1,
        )(col("audio_path")),
    )

    actual_df = ds.to_pandas()
    assert actual_df["audio_vad_result"][0] is None or len(actual_df["audio_vad_result"][0]) == 0
    assert actual_df["audio_vad_result"][1] is None or len(actual_df["audio_vad_result"][1]) == 0
    assert np.allclose(actual_df["audio_vad_result"][2][0], [0.54, 7.45], atol=0.1)
