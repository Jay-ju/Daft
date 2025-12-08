# Copyright (c) Beijing Volcano Engine Technology Ltd.

from __future__ import annotations

import pandas as pd
import pytest

import daft
from daft import col
from daft.las.functions.audio.audio_asr_lid_whisper import AudioAsrLidWhisper
from daft.las.functions.udf import las_udf

audio_src_type = "audio_url"
dtype = "bfloat16"
model_name = "openai/whisper-large-v3"
# model_name = "openai/whisper-large-v3-turbo"
# model_name = "openai/whisper-medium"
# model_name = "openai/whisper-small"
punc_model_name = "iic/punc_ct-transformer_cn-en-common-vocab471067-large"
batch_size = 1
num_gpus = 1
device = "cuda" if num_gpus > 0 else "cpu"


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


@pytest.mark.gpu
def test_audio_asr_lid_whisper(local_models_dir, tos_test_data_dir, local_test_data_dir, http_test_data_dir):
    input_df = generate_test_data(tos_test_data_dir, local_test_data_dir, http_test_data_dir)

    ds = daft.from_pandas(input_df)
    ds = ds.with_column(
        "asr_result_detail",
        las_udf(
            AudioAsrLidWhisper,
            construct_args={
                "audio_src_type": audio_src_type,
                "model_path": local_models_dir,
                "model_name": model_name,
                "punc_model_name": punc_model_name,
                "batch_size": batch_size,
                "device": device,
            },
            num_gpus=num_gpus,
            batch_size=1,
            num_cpus=4,
            concurrency=1,
        )(col("audio_path")),
    )

    actual_df = ds.to_pandas()
    assert (
        actual_df["asr_result_detail"][0]["asr_result"] is None
        or len(actual_df["asr_result_detail"][0]["asr_result"]) == 0
    )
    assert (
        actual_df["asr_result_detail"][1]["asr_result"] is None
        or len(actual_df["asr_result_detail"][1]["asr_result"]) == 0
    )
    assert "浙江省" in actual_df["asr_result_detail"][2]["asr_result"]
    assert "zh" in actual_df["asr_result_detail"][2]["language"]
    assert "浙江省" in actual_df["asr_result_detail"][2]["asr_result_with_punc"]
